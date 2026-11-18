"""Two-factor A1/H2 interaction study; exactly one new joint-model training."""

from itertools import combinations
from pathlib import Path
import shutil
import time

import torch

from .common import ROOT, digest_file, write_json
from .cp2_attention import attach_attention
from .cp2_data import verify_cp2_manifest
from .cp2_heads import HeadObjective, build_head_model
from .cp2_phase_h import phase_h_sources
from .cp2_phase_l import validate_epoch
from .cp2_phase_n import read, utc
from .cp2_screening import fit
from .cp2_train import export_evaluation, make_cp2_loader, tensor_fingerprint, training_fingerprints
from .engine import setup


MODELS = ("B1", "A1", "H2", "A1H2")


def sources():
    result = phase_h_sources()
    for path in ("src/foot_ulcer_segmentation/cp2_attention.py",
                 "src/foot_ulcer_segmentation/cp2_phase_n.py",
                 "src/foot_ulcer_segmentation/cp2_presence.py",
                 "src/foot_ulcer_segmentation/cp2_screening.py",
                 "src/foot_ulcer_segmentation/cp2_phase_c.py",
                 "scripts/run_cp2_phase_c.py", "scripts/report_cp2_phase_c.py",
                 "scripts/validate_cp2_phase_a.py", "scripts/validate_cp2_phase_l.py"):
        result[path] = digest_file(ROOT / path)
    return result


def directory(config, name, root):
    if name not in MODELS:
        raise ValueError("Only registered B1/A1/H2/A1H2")
    return root / name if name == "A1H2" else ROOT / config["phase_c"]["controls"][name]["directory"]


def build(config, name, initial):
    if name not in MODELS:
        raise ValueError("No matrix expansion")
    settings = config["phase_c"]
    model, head_proof = build_head_model(config, "H2" if "H2" in name else "H0", initial)
    if "A1" in name:
        attach_attention(model.base, "A1", config)
    state = model.base.state_dict()
    for key, tensor in initial["state_dict"].items():
        if not torch.equal(state[key], tensor):
            raise ValueError(f"Pairable base initialization changed: {key}")
    additional = {k: v for k, v in state.items() if k not in initial["state_dict"]}
    local = {k: v for k, v in additional.items() if ".local_attention." in k}
    base_fingerprint = tensor_fingerprint(state)
    if "A1" in name:
        if (tensor_fingerprint(additional) != settings["a1_additional_tensor_sha256"]
                or tensor_fingerprint(local) != settings["a1_local_tensor_sha256"]
                or base_fingerprint != settings["a1_base_tensor_sha256"]):
            raise ValueError("Local branches/prior do not match historical A1 exactly")
    elif base_fingerprint != settings["initial_tensor_sha256"]:
        raise ValueError("Original B1 base initialization changed")
    if "H2" in name and head_proof["head_tensor_sha256"] != settings["h2_head_tensor_sha256"]:
        raise ValueError("Auxiliary head does not match historical H2")
    proof = {"all_original_named_tensors_equal": True,
             "source_tensor_sha256": head_proof["source_tensor_sha256"],
             "base_tensor_sha256": base_fingerprint,
             "local_tensor_sha256": tensor_fingerprint(local) if local else None,
             "additional_tensor_sha256": tensor_fingerprint(additional) if additional else None,
             "head_tensor_sha256": head_proof["head_tensor_sha256"],
             "head_parameters": head_proof["head_parameters"],
             "complete_initial_tensor_sha256": tensor_fingerprint(model.state_dict()),
             "boundary_is_not_attention_input": True, "final_decoder_stage_unchanged": True}
    return model, HeadObjective("H2" if "H2" in name else "H0"), proof


def histories(config, root, include_joint=False):
    names = MODELS if include_joint else MODELS[:-1]
    return {name: read(directory(config, name, root) / "history.json") for name in names}


def pairing(histories_by_model):
    proofs = []
    for first, second in combinations(histories_by_model, 2):
        a, b = histories_by_model[first], histories_by_model[second]
        for left, right in zip(a, b):
            if any(left[key] != right[key] for key in ("training_order_sha256", "augmented_input_sha256")):
                raise ValueError(f"Actual input pairing differs: {first}/{second} epoch{left['epoch']}")
        proofs.append({"first": first, "second": second, "common_epochs": min(len(a), len(b))})
    return {"status": "passed", "actual_order_and_augmented_bytes_equal": True, "pairs": proofs}


def prepare(root, config):
    settings = config["phase_c"]
    control = ROOT / settings["control_root"]
    original = read(control / "protocol.json")
    for key in ("seed", "split_seed", "cpu_threads", "data", "model", "training", "loss"):
        if config[key] != original[key]:
            raise ValueError(f"Frozen CP2 protocol changed: {key}")
    for key, value in original["evaluation"].items():
        if key != "test_policy" and config["evaluation"][key] != value:
            raise ValueError(f"Evaluator changed: {key}")
    for phase in ("a", "h"):
        if config[f"phase_{phase}"] != read(ROOT / f"configs/cp2/phase_{phase}.json")[f"phase_{phase}"]:
            raise ValueError("Historical component definitions changed")
    if settings["fresh_model"] != "A1H2" or settings["new_training_runs"] != 1 or settings["standalone_smoke_training"]:
        raise ValueError("Exactly one new A1H2 full training, no separate smoke")
    if config["seed"] != 42 or settings["head_weight"] != .1:
        raise ValueError("Unregistered seed or head weight")
    manifest = verify_cp2_manifest(read(control / "data/manifest.json"))
    if manifest["fingerprint"] != settings["manifest_fingerprint"]:
        raise ValueError("Membership changed")
    if read(control / "B1/status.json")["training_sources"] != training_fingerprints():
        raise ValueError("Frozen B1 executable utilities changed")
    cached = {}
    for name in MODELS[:-1]:
        run = directory(config, name, root)
        status = read(run / "status.json")
        if status["status"] != "completed" or status["diagnostic_smoke_only"]:
            raise ValueError("Completed real cached controls required")
        if status["split_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Control evaluation population differs")
        if digest_file(run / "best.pth") != settings["controls"][name]["checkpoint_sha256"]:
            raise ValueError("Historical checkpoint changed")
        artifacts = ["best.pth", "status.json", "history.json", "initialization_verified.json",
                     "evaluation-validation.json", "evaluation-test.json"]
        cached[name] = {"source": str(run.relative_to(ROOT)), "mode": "CACHED; not retrained",
                        "validation": read(run / "evaluation-validation.json"),
                        "artifact_sha256": {file: digest_file(run / file) for file in artifacts}}
    if digest_file(control / "initial_weights.pth") != settings["initial_weights_sha256"]:
        raise ValueError("Shared source initialization changed")
    if (root / "protocol.json").exists():
        if read(root / "protocol.json") != config or read(root / "control_reference.json") != cached:
            raise ValueError("Prepared config/cached controls changed")
        if digest_file(root / "initial_weights.pth") != settings["initial_weights_sha256"]:
            raise ValueError("Local initial copy changed")
        return manifest
    if not (root / "protected_before.json").exists():
        raise ValueError("Save pre-edit inventory first")
    write_json(root / "protocol.json", config)
    write_json(root / "data/manifest.json", manifest)
    write_json(root / "control_reference.json", cached)
    temporary = root / "initial_weights.pth.tmp"
    shutil.copyfile(control / "initial_weights.pth", temporary)
    temporary.replace(root / "initial_weights.pth")
    write_json(root / "cached_input_pairing.json", pairing(histories(config, root)))
    write_json(root / "dataset_summary.json", {
        "manifest_fingerprint": manifest["fingerprint"], "patient_ids_available": False,
        "duplicate_rgb_overlap": 0, "splits": {
            split: {"images": len(rows), "positive": sum(r["gt_pixels"] > 0 for r in rows),
                    "empty": sum(r["gt_pixels"] == 0 for r in rows)} for split, rows in manifest["splits"].items()}})
    return manifest


def train(root, config, manifest, device, resume=False):
    setup(config["seed"], config["cpu_threads"])
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model, objective, proof = build(config, "A1H2", initial)
    del initial
    names = {"encoder": next(iter(dict(model.named_parameters()))),
             "boundary_head_weight": "auxiliary_head.weight", "boundary_head_bias": "auxiliary_head.bias"}
    for stage in (2, 3):
        for part in ("depthwise", "pointwise"):
            for parameter in ("weight", "bias"):
                names[f"local_s{stage}_{part}_{parameter}"] = f"base.decoder.blocks.{stage}.local_attention.{part}.{parameter}"
    return fit(config, manifest, root / "A1H2", model, objective, proof, sources(), histories(config, root),
               names, device, resume)


def evaluate(root, config, manifest, split, device_name):
    run = root / "A1H2"
    status = read(run / "status.json")
    if status["status"] != "completed":
        raise ValueError("Completed full training required")
    if split == "test" and not (root / "selection.json").exists():
        raise ValueError("Lock source-validation selection before any new source-test predictions")
    destination = run / f"evaluation-{split}.json"
    if destination.exists():
        previous = read(destination)
        if previous["checkpoint_sha256"] != digest_file(run / "best.pth"):
            raise ValueError("Completed evaluation checkpoint differs")
        return previous
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model, objective, _ = build(config, "A1H2", initial)
    del initial
    state = torch.load(run / "best.pth", map_location="cpu")
    if state["identity"] != status["identity"]:
        raise ValueError("Checkpoint identity differs")
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    device = torch.device(device_name)
    model.to(device)
    loader, rows = make_cp2_loader(config, manifest, split)
    begin = time.monotonic()
    evaluator, parts = validate_epoch(model, objective, loader, rows, config, device,
                                     run / "predictions" / split)
    return export_evaluation(destination, evaluator, {
        "variant": "A1H2", "split": split, "selected_epoch": status["selected_epoch"],
        "checkpoint_sha256": digest_file(run / "best.pth"), "split_fingerprint": manifest["fingerprint"],
        "components": parts, "seconds": time.monotonic() - begin, "diagnostic_smoke_only": False,
        "test_caveat": "source-test already exposed; descriptive only"})


def select(reports, profiles):
    if set(reports) != set(MODELS) or any(report["split"] != "validation" for report in reports.values()):
        raise ValueError("Selection requires all four SOURCE VALIDATION reports")
    candidate, baseline = reports["A1H2"], reports["B1"]
    checks = {
        "higher_overall_dice": candidate["all"]["image_based"]["dice"] > baseline["all"]["image_based"]["dice"],
        "small_not_lower": candidate["subgroups"]["small"]["image_based"]["dice"] >= baseline["subgroups"]["small"]["image_based"]["dice"],
        "boundary_not_lower": candidate["all"]["image_based"]["boundary_iou"] >= baseline["all"]["image_based"]["boundary_iou"],
        "no_extra_total_miss": candidate["gt_positive"]["total_foreground_misses"] <= baseline["gt_positive"]["total_foreground_misses"],
        "no_extra_empty_fp": candidate["gt_empty"]["images_with_prediction"] <= baseline["gt_empty"]["images_with_prediction"],
        "capacity_budget": profiles["A1H2"]["active_forward_parameters"] <= 1.01 * profiles["B1"]["active_forward_parameters"],
    }
    best_single = max(reports[name]["all"]["image_based"]["dice"] for name in ("A1", "H2"))
    better_than_singles = candidate["all"]["image_based"]["dice"] > best_single
    eligible = all(checks.values()) and better_than_singles
    return {"validation_only": True, "selected_candidate": "A1H2" if eligible else "B1",
            "eligible": eligible, "original_cp2_guards": checks,
            "higher_overall_than_both_individuals": better_than_singles,
            "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
            "individual_historical_decisions_unchanged": True,
            "automatic_expansion": False, "single_seed_exploratory": True}


def interaction(reports):
    """Descriptive difference of recipe effects, without claiming causal synergy."""
    if set(reports) != set(MODELS) or len({r["split"] for r in reports.values()}) != 1:
        raise ValueError("All four conditions must use the same split")
    keys = {
        "overall_image_dice_pp": lambda r: r["all"]["image_based"]["dice"] * 100,
        "gt_positive_image_dice_pp": lambda r: r["gt_positive"]["image_based"]["dice"] * 100,
        "small_image_dice_pp": lambda r: r["subgroups"]["small"]["image_based"]["dice"] * 100,
        "boundary_iou_pp": lambda r: r["all"]["image_based"]["boundary_iou"] * 100,
        "data_dice_pp": lambda r: r["all"]["data_based"]["dice"] * 100,
        "positive_pixel_recall_pp": lambda r: r["gt_positive"]["data_based"]["recall"] * 100,
        "empty_fp_images": lambda r: r["gt_empty"]["images_with_prediction"],
        "empty_fp_pixels": lambda r: r["gt_empty"]["counts"]["fp"],
        "total_foreground_misses": lambda r: r["gt_positive"]["total_foreground_misses"],
    }
    return {key: {"A1H2_minus_B1": extract(reports["A1H2"]) - extract(reports["B1"]),
                  "A1H2_minus_A1": extract(reports["A1H2"]) - extract(reports["A1"]),
                  "A1H2_minus_H2": extract(reports["A1H2"]) - extract(reports["H2"]),
                  "interaction": extract(reports["A1H2"]) - extract(reports["A1"]) - extract(reports["H2"]) + extract(reports["B1"])}
            for key, extract in keys.items()}
