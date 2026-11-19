"""Two requested A6/H2 combinations, differing only in the main loss.

Historical trainers/components and results stay immutable. The paired screening
loop is reused unchanged; no additional architecture or standalone training.
"""

from copy import deepcopy
from pathlib import Path
import shutil
import time

import torch

from .common import ROOT, digest_file, digest_json, write_json
from .cp2_attention import attach_attention
from .cp2_data import verify_cp2_manifest
from .cp2_heads import build_head_model, morphological_boundary
from .cp2_losses import PhaseLObjective
from .cp2_phase_c import pairing, sources as original_sources
from .cp2_phase_l import validate_epoch
from .cp2_phase_n import read, utc
from .cp2_screening import fit
from .cp2_train import export_evaluation, make_cp2_loader, tensor_fingerprint, training_fingerprints
from .engine import setup


FRESH = ("B1A6H2", "L3A6H2")
CACHED = ("B1", "A6", "H2", "L3")


class CombinationObjective(torch.nn.Module):
    """Keep H2 auxiliary B1 fixed; apply L3 exclusively to the main mask."""
    def __init__(self, variant, settings):
        super().__init__()
        if variant not in FRESH:
            raise ValueError("Only the two registered combinations")
        self.variant = variant
        self.main = PhaseLObjective("L3" if variant == "L3A6H2" else "L0", settings)
        if settings["boundary"]["weight"] != .01:
            raise ValueError("L3 coefficient must remain 0.01")

    def components(self, output, truth):
        if output.auxiliary_logits is None:
            raise ValueError("Actual H2 auxiliary logits required")
        main = self.main.components(output.main_logits, truth)
        auxiliary_dice, auxiliary_focal = self.main.baseline.components(
            output.auxiliary_logits, morphological_boundary(truth))
        auxiliary = auxiliary_dice + auxiliary_focal
        weighted = .1 * auxiliary
        return {**main, "main_total": main["total"], "auxiliary_dice": auxiliary_dice,
                "auxiliary_focal": auxiliary_focal, "auxiliary": auxiliary,
                "weighted_auxiliary": weighted, "total": main["total"] + weighted}

    def forward(self, output, truth):
        return self.components(output, truth)["total"]


def sources():
    result = original_sources()
    for name in ("src/foot_ulcer_segmentation/cp2_combinations.py",
                 "scripts/run_cp2_combinations.py", "scripts/report_cp2_combinations.py"):
        result[name] = digest_file(ROOT / name)
    return result


def run_config(config, name):
    if name not in FRESH:
        raise ValueError("Unregistered training variant")
    result = deepcopy(config)
    result["active_combination"] = name
    return result


def build(config, name, initial):
    if name not in FRESH:
        raise ValueError("Only two fresh combinations")
    c = config["combinations"]
    model, head = build_head_model(config, "H2", initial)
    attach_attention(model.base, "A6", config)
    state = model.base.state_dict()
    for key, value in initial["state_dict"].items():
        if not torch.equal(value, state[key]):
            raise ValueError(f"Shared source tensor changed: {key}")
    additional = {k: v for k, v in state.items() if k not in initial["state_dict"]}
    if tensor_fingerprint(additional) != c["a6_additional_tensor_sha256"]:
        raise ValueError("CBAM initial tensors differ from historical A6")
    if tensor_fingerprint(state) != c["a6_base_tensor_sha256"]:
        raise ValueError("A6 base initialization differs")
    if head["head_tensor_sha256"] != c["h2_head_tensor_sha256"]:
        raise ValueError("Auxiliary initialization differs from H2")
    proof = {"all_original_named_tensors_equal": True,
             "source_tensor_sha256": head["source_tensor_sha256"],
             "base_tensor_sha256": tensor_fingerprint(state),
             "cbam_tensor_sha256": tensor_fingerprint(additional),
             "head_tensor_sha256": head["head_tensor_sha256"], "head_parameters": 33,
             "complete_initial_tensor_sha256": tensor_fingerprint(model.state_dict()),
             "CBAM_is_exact_historical_A6_including_gain3": True,
             "final_decoder_stage_unchanged": True,
             "boundary_logits_do_not_guide_attention": True}
    return model, CombinationObjective(name, config["phase_l"]), proof


def directory(root, config, name):
    if name in FRESH:
        return root / name
    if name in CACHED:
        return ROOT / config["combinations"]["controls"][name]["directory"]
    raise ValueError("Unknown condition")


def histories(root, config, fresh=()):
    return {name: read(directory(root, config, name) / "history.json") for name in (*CACHED, *fresh)}


def prepare(root, config):
    c = config["combinations"]
    control = ROOT / c["control_root"]
    old = read(control / "protocol.json")
    for key in ("seed", "split_seed", "cpu_threads", "data", "model", "training", "loss"):
        if config[key] != old[key]:
            raise ValueError(f"Fixed CP2 protocol changed: {key}")
    for key, value in old["evaluation"].items():
        if key != "test_policy" and config["evaluation"][key] != value:
            raise ValueError(f"Fixed evaluator changed: {key}")
    for phase in ("a", "h", "l"):
        if config[f"phase_{phase}"] != read(ROOT / f"configs/cp2/phase_{phase}.json")[f"phase_{phase}"]:
            raise ValueError("Historical component definition changed")
    if (c["order"] != list(FRESH) or c["new_training_runs"] != 2 or c["standalone_smoke_training"]
            or config["seed"] != 42 or c["auxiliary_weight"] != .1 or c["boundary_loss_weight"] != .01):
        raise ValueError("Unregistered matrix, seed or loss weights")
    manifest = verify_cp2_manifest(read(control / "data/manifest.json"))
    if manifest["fingerprint"] != c["manifest_fingerprint"]:
        raise ValueError("Clean membership changed")
    if read(control / "B1/status.json")["training_sources"] != training_fingerprints():
        raise ValueError("Original CP2 utilities changed")
    cached = {}
    for name in CACHED:
        run = directory(root, config, name)
        status = read(run / "status.json")
        if status["status"] != "completed" or status["diagnostic_smoke_only"]:
            raise ValueError("Completed real cached controls required")
        if status["split_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Control membership differs")
        if digest_file(run / "best.pth") != c["controls"][name]["checkpoint_sha256"]:
            raise ValueError("Control checkpoint changed")
        files = ("best.pth", "status.json", "history.json", "initialization_verified.json",
                 "evaluation-validation.json", "evaluation-test.json")
        cached[name] = {"source": str(run.relative_to(ROOT)), "mode": "CACHED, never retrained",
                        "validation": read(run / "evaluation-validation.json"),
                        "artifact_sha256": {n: digest_file(run / n) for n in files}}
    if digest_file(control / "initial_weights.pth") != c["initial_weights_sha256"]:
        raise ValueError("Shared initial weights changed")
    if (root / "protocol.json").exists():
        if read(root / "protocol.json") != config or read(root / "control_reference.json") != cached:
            raise ValueError("Existing preparation differs")
    else:
        if not (root / "protected_before.json").exists():
            raise ValueError("Pre-edit preservation inventory required")
        write_json(root / "protocol.json", config)
        write_json(root / "data/manifest.json", manifest)
        write_json(root / "control_reference.json", cached)
        shutil.copyfile(control / "initial_weights.pth", root / "initial_weights.pth")
        write_json(root / "cached_input_pairing.json", pairing(histories(root, config)))
    if digest_file(root / "initial_weights.pth") != c["initial_weights_sha256"]:
        raise ValueError("Local initializer differs")
    return manifest


def train(root, config, manifest, name, device, resume=False):
    setup(config["seed"], config["cpu_threads"])
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model, objective, proof = build(config, name, initial)
    del initial
    checked = {"encoder": next(iter(dict(model.named_parameters()))),
               "boundary_head_weight": "auxiliary_head.weight", "boundary_head_bias": "auxiliary_head.bias"}
    for stage in (2, 3):
        checked[f"cbam_s{stage}_channel_gate_bias"] = f"base.decoder.blocks.{stage}.cbam.mlp.2.bias"
        checked[f"cbam_s{stage}_spatial_kernel"] = f"base.decoder.blocks.{stage}.cbam.spatial.0.weight"
    return fit(run_config(config, name), manifest, root / name, model, objective, proof,
               sources(), histories(root, config), checked, device, resume)


def evaluate(root, config, manifest, name, split, device_name):
    if name not in FRESH or split not in ("validation", "test"):
        raise ValueError("Only registered fresh conditions and source splits")
    run = root / name
    status = read(run / "status.json")
    if status["status"] != "completed":
        raise ValueError("Completed full training required")
    if split == "test" and not (root / "selection.json").exists():
        raise ValueError("Lock joint source-validation selection before test access")
    destination = run / f"evaluation-{split}.json"
    if destination.exists():
        old = read(destination)
        if old["checkpoint_sha256"] != digest_file(run / "best.pth"):
            raise ValueError("Completed evaluation checkpoint changed")
        return old
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model, objective, _ = build(config, name, initial)
    del initial
    checkpoint = torch.load(run / "best.pth", map_location="cpu")
    if checkpoint["identity"] != status["identity"]:
        raise ValueError("Checkpoint identity differs")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    device = torch.device(device_name)
    model.to(device)
    loader, rows = make_cp2_loader(config, manifest, split)
    begin = time.monotonic()
    evaluator, components = validate_epoch(model, objective, loader, rows, config, device,
                                          run / "predictions" / split)
    return export_evaluation(destination, evaluator, {
        "variant": name, "split": split, "selected_epoch": status["selected_epoch"],
        "checkpoint_sha256": digest_file(run / "best.pth"), "split_fingerprint": manifest["fingerprint"],
        "components": components, "seconds": time.monotonic() - begin,
        "diagnostic_smoke_only": False, "test_caveat": "historically exposed source test; descriptive only"})


def select(reports, profiles):
    if set(reports) != set((*CACHED, *FRESH)) or any(r["split"] != "validation" for r in reports.values()):
        raise ValueError("All six SOURCE VALIDATION conditions required")
    base = reports["B1"]
    decisions = {}
    for name in FRESH:
        candidate = reports[name]
        relevant = ("A6", "H2", "L3") if name == "L3A6H2" else ("A6", "H2")
        guards = {
            "higher_overall_dice": candidate["all"]["image_based"]["dice"] > base["all"]["image_based"]["dice"],
            "small_not_lower": candidate["subgroups"]["small"]["image_based"]["dice"] >= base["subgroups"]["small"]["image_based"]["dice"],
            "boundary_not_lower": candidate["all"]["image_based"]["boundary_iou"] >= base["all"]["image_based"]["boundary_iou"],
            "no_extra_total_miss": candidate["gt_positive"]["total_foreground_misses"] <= base["gt_positive"]["total_foreground_misses"],
            "no_extra_empty_fp": candidate["gt_empty"]["images_with_prediction"] <= base["gt_empty"]["images_with_prediction"],
            "capacity_budget": profiles[name]["active_forward_parameters"] <= 1.01 * profiles["B1"]["active_forward_parameters"],
            "higher_than_relevant_standalone": candidate["all"]["image_based"]["dice"] > max(reports[n]["all"]["image_based"]["dice"] for n in relevant),
        }
        decisions[name] = {"eligible": all(guards.values()), "guards": guards,
                           "validation_image_dice": candidate["all"]["image_based"]["dice"],
                           "checkpoint_sha256": candidate["checkpoint_sha256"]}
    eligible = [n for n in FRESH if decisions[n]["eligible"]]
    chosen = max(eligible, key=lambda n: decisions[n]["validation_image_dice"]) if eligible else "B1"
    return {"validation_only": True, "selected_candidate": chosen, "decisions": decisions,
            "tie_rule": "earlier B1A6H2 (simpler main loss)", "historical_selections_unchanged": True,
            "automatic_expansion": False, "single_seed_exploratory": True}


def contrasts(reports):
    if set(reports) != set((*CACHED, *FRESH)) or len({r["split"] for r in reports.values()}) != 1:
        raise ValueError("All six conditions must use the same split")
    extractors = {
        "image_dice_pp": lambda r: 100 * r["all"]["image_based"]["dice"],
        "data_dice_pp": lambda r: 100 * r["all"]["data_based"]["dice"],
        "small_dice_pp": lambda r: 100 * r["subgroups"]["small"]["image_based"]["dice"],
        "boundary_iou_pp": lambda r: 100 * r["all"]["image_based"]["boundary_iou"],
        "positive_recall_pp": lambda r: 100 * r["gt_positive"]["data_based"]["recall"],
        "gt_positive_dice_pp": lambda r: 100 * r["gt_positive"]["image_based"]["dice"],
        "empty_fp_images": lambda r: r["gt_empty"]["images_with_prediction"],
        "empty_fp_pixels": lambda r: r["gt_empty"]["counts"]["fp"],
        "foreground_misses": lambda r: r["gt_positive"]["total_foreground_misses"],
    }
    result = {}
    for key, value in extractors.items():
        v = {n: value(r) for n, r in reports.items()}
        result[key] = {"B1A6H2_minus_B1": v["B1A6H2"] - v["B1"],
                       "B1A6H2_minus_A6": v["B1A6H2"] - v["A6"],
                       "B1A6H2_minus_H2": v["B1A6H2"] - v["H2"],
                       "A6_H2_descriptive_interaction": v["B1A6H2"] - v["A6"] - v["H2"] + v["B1"],
                       "L3A6H2_minus_B1A6H2": v["L3A6H2"] - v["B1A6H2"],
                       "L3A6H2_minus_L3": v["L3A6H2"] - v["L3"],
                       "L3A6H2_minus_B1": v["L3A6H2"] - v["B1"]}
    return result
