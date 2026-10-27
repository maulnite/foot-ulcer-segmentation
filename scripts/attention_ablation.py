"""Controlled branch ablations around the unchanged pinned training engine."""

from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from foot_ulcer_segmentation import engine, reference
from foot_ulcer_segmentation.common import ROOT, digest_file, digest_json, project_path, write_json
from foot_ulcer_segmentation.data import prepare_splits

BASELINE = ROOT / "outputs/runs/cp1-baseline-01"
PROTOCOL_PATH = ROOT / "configs/ablation/cp1-attention-01.json"
VERIFICATION_PATH = ROOT / "outputs/ablations/cp1-attention-01/verification.json"
VARIANTS = {"additive_only": "attention2", "maxout_only": "attention1"}
RUNTIME_FILES = ["scripts/attention_ablation.py", "scripts/run_attention_ablation.py"]


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_config(config):
    baseline = read_json(BASELINE / "config.json")
    for key in ["reference_revision", "seed", "cpu_threads", "data", "model", "training", "evaluation"]:
        if config[key] != baseline[key]:
            raise ValueError(f"Ablation unexpectedly changes baseline {key}")
    settings = config["ablation"]
    if (settings["variant"] not in VARIANTS or settings["modified_stages"] != [0, 1, 2, 3]
            or settings["final_stage_policy"] != "unchanged_author_shorted"):
        raise ValueError("Unsupported branch ablation scope")
    protocol = read_json(PROTOCOL_PATH)
    if settings["expected_split_fingerprint"] != protocol["split_fingerprint"]:
        raise ValueError("Ablation and protocol split identities differ")
    declared = next(row for row in protocol["runs"] if row["variant"] == settings["variant"])
    if config["name"] != declared["name"]:
        raise ValueError("Ablation configuration differs from predeclared run")
    return settings


def check_frozen_baseline():
    record = read_json(BASELINE / "environment.json")
    for name, expected in record["packages"].items():
        if version(name) != expected:
            raise ValueError(f"Training dependency changed: {name}")
    for name, expected in record["pipeline_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"Frozen baseline source changed: {name}")
    reference.verify_reference()
    return record


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _single_branch_forward(block, x, skip=None, strategy="add"):
    if skip is None or strategy != "add":
        raise ValueError("Ablated regular stages require a skip and additive outer aggregation")
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = torch.cat([x, skip], dim=1)
    branch = getattr(block, block._ablation_branch)
    return block.conv1(branch(x))


def apply_branch_ablation(model, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    if len(model.decoder.blocks) != 5:
        raise ValueError("Expected the original five-stage decoder")
    for block in model.decoder.blocks:
        if block.attention_type != "pscse":
            raise ValueError("Construct the complete original P-scSE model first")
    # Keep both branches registered: construction and RNG consumption stay identical.
    # Only the four regular stages change. The last source forward is untouched.
    for block in model.decoder.blocks[:4]:
        block._ablation_branch = VARIANTS[variant]
        block.forward = types.MethodType(_single_branch_forward, block)
    return model


def probe_model(model, inputs):
    used, calls, handles = set(), {}, []

    def hook(name):
        def record(module, args, output):
            calls[name] = calls.get(name, 0) + 1
            used.update(id(p) for p in module.parameters(recurse=False))
        return record

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(hook(name)))
    try:
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return {"registered_parameters": sum(p.numel() for p in model.parameters()),
            "forward_called_parameters": sum(p.numel() for p in model.parameters() if id(p) in used),
            "output_shape": list(prediction.shape),
            "decoder_calls": {name: calls.get(name, 0)
                              for i in range(5) for name in
                              [f"decoder.blocks.{i}.{part}" for part in ["conv1", "conv2", "attention1", "attention2"]]}}, prediction


def probe_variant(variant):
    """One model per fresh process, matching the actual CLI import lifecycle."""
    baseline = read_json(BASELINE / "config.json")
    torch.hub.set_dir(str(ROOT / "outputs/cache/torch/hub"))
    engine.setup(baseline["seed"], baseline["cpu_threads"])
    model = reference.build_model(baseline, pretrained=True)
    initial_hash = state_digest(model)
    before_rng = torch.get_rng_state().clone()
    if variant != "full":
        apply_branch_ablation(model, variant)
    if not torch.equal(before_rng, torch.get_rng_state()) or state_digest(model) != initial_hash:
        raise ValueError("Applying an ablation changed initialization or consumed RNG")
    inputs = torch.linspace(-1, 1, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
    record, prediction = probe_model(model, inputs)
    if record["output_shape"] != [2, 1, 224, 224] or not torch.isfinite(prediction).all():
        raise ValueError("Ablation has incorrect or nonfinite outputs")
    for i in range(5):
        expected = {"conv1": 1, "conv2": 0,
                    "attention1": int(i < 4 and variant != "additive_only"),
                    "attention2": int(i == 4 or variant != "maxout_only")}
        for part, count in expected.items():
            if record["decoder_calls"][f"decoder.blocks.{i}.{part}"] != count:
                raise ValueError(f"Unexpected active graph: {variant}, stage {i}, {part}")
    record["initial_state_sha256"] = initial_hash
    record["probe_output_sha256"] = hashlib.sha256(prediction.cpu().numpy().tobytes()).hexdigest()
    return record


def verify_models():
    check_frozen_baseline()
    baseline = read_json(BASELINE / "config.json")
    manifest = prepare_splits(baseline)
    protocol = read_json(PROTOCOL_PATH)
    if manifest["fingerprint"] != protocol["split_fingerprint"]:
        raise ValueError("Ablation must use the exact original split")
    torch.hub.set_dir(str(ROOT / "outputs/cache/torch/hub"))
    cache = ROOT / "outputs/cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth"
    expected_weight = read_json(BASELINE / "status.json")["pretrained_weight_sha256"][cache.name]
    if digest_file(cache) != expected_weight:
        raise ValueError("ImageNet cache differs from the original run")
    records, expected_hash = {}, None
    probe_dir = VERIFICATION_PATH.parent / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for variant in ["full", *VARIANTS]:
        destination = probe_dir / f"{variant}.json"
        with (probe_dir / f"{variant}.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/run_attention_ablation.py"), "probe",
                            "--variant", variant, "--probe-output", str(destination)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = read_json(destination)
        initial_hash = record["initial_state_sha256"]
        if expected_hash is None:
            expected_hash = initial_hash
        if initial_hash != expected_hash:
            raise ValueError("Initial model tensors differ despite the matched construction")
        if variant != "full" and record["probe_output_sha256"] == records["full"]["probe_output_sha256"]:
            raise ValueError("Ablation did not change model outputs")
        records[variant] = record
    result = {"status": "passed", "protocol_fingerprint": digest_json(protocol),
              "runtime_sha256": {name: digest_file(ROOT / name) for name in RUNTIME_FILES},
              "full_initial_state_sha256": expected_hash, "split_fingerprint": manifest["fingerprint"],
              "pretrained_weight_sha256": expected_weight,
              "frozen_baseline_source_and_packages_match": True, "variants": records}
    write_json(VERIFICATION_PATH, result)
    return result


@contextmanager
def adapted_engine(config, run, split, *, verify_initial=True):
    settings = validate_config(config)
    proof = read_json(VERIFICATION_PATH)
    if proof["status"] != "passed" or proof["protocol_fingerprint"] != digest_json(read_json(PROTOCOL_PATH)):
        raise ValueError("Run model verification for the frozen protocol first")
    for name, expected in proof["runtime_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"Ablation runtime changed after verification: {name}")
    original_build, original_environment, original_evaluator = engine.build_model, engine.environment_record, engine.SegmentationEvaluator
    output = ROOT / "outputs/analysis" / run.name
    captures = []

    def build(cfg, pretrained=True):
        model = reference.build_model(cfg, pretrained=pretrained)
        if pretrained and verify_initial:
            initial_hash = state_digest(model)
            if initial_hash != proof["full_initial_state_sha256"]:
                raise ValueError("Training initialization differs from the reconstructed baseline")
            write_json(run / "model-ablation.json", {**proof["variants"][settings["variant"]],
                       "variant": settings["variant"], "initial_state_sha256": initial_hash,
                       "last_stage_unchanged": True, "dormant_branches_retained": True})
        return apply_branch_ablation(model, settings["variant"])

    def environment(cfg, manifest, device):
        record = original_environment(cfg, manifest, device)
        record["pipeline_sha256"].update({name: digest_file(ROOT / name) for name in RUNTIME_FILES})
        record["pipeline_sha256"][str(PROTOCOL_PATH.relative_to(ROOT))] = digest_file(PROTOCOL_PATH)
        record["ablation"] = settings
        record["initialization_verification"] = proof["full_initial_state_sha256"]
        return record

    class CachedEvaluator(original_evaluator):
        def __init__(self, threshold=.5):
            super().__init__(threshold)
            self.prediction_records = []
            captures.append(self)

        def add(self, name, truth, probability):
            row = super().add(name, truth, probability)
            if Path(name).name != name:
                raise ValueError("Expected a plain sample filename")
            path = output / "predictions" / split / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((np.asarray(probability) >= self.threshold).astype(np.uint8) * 255).save(path)
            self.prediction_records.append({"split": split, "name": name,
                                            "path": str(path.relative_to(ROOT)), "sha256": digest_file(path)})
            return row

    engine.build_model, engine.environment_record, engine.SegmentationEvaluator = build, environment, CachedEvaluator
    try:
        yield captures
    finally:
        engine.build_model, engine.environment_record, engine.SegmentationEvaluator = original_build, original_environment, original_evaluator


def save_cache_records(run, split, captures):
    if len(captures) != 1:
        raise ValueError("Expected one final offline evaluation")
    write_json(run / f"mask-cache-{split}.json", captures[0].prediction_records)


def finalize_predictions(run, device):
    config = read_json(run / "config.json")
    status = read_json(run / "status.json")
    manifest = read_json(run / "split-manifest.json")
    records = []
    for split in ["validation", "test"]:
        batch = read_json(run / f"mask-cache-{split}.json")
        if {row["name"] for row in batch} != {row["name"] for row in manifest["splits"][split]}:
            raise ValueError("Mask cache inventory differs from the split")
        records.extend(batch)
    identity = {"checkpoint_sha256": digest_file(run / "best.pth"),
                "config_fingerprint": digest_json(config), "split_fingerprint": manifest["fingerprint"],
                "threshold": config["evaluation"]["prediction_threshold"],
                "evaluation_sha256": {name: digest_file(run / name) for name in ["validation.csv", "evaluation-test.csv"]}}
    write_json(ROOT / "outputs/analysis" / run.name / "prediction-manifest.json",
               {"identity": identity, "device": device, "selected_epoch": status["selected_epoch"],
                "training_performed": False, "all_counts_match_final_evaluation": True, "predictions": records,
                "export_method": "Masks saved during the same final evaluations; no second inference pass"})
