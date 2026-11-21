"""Targeted CP1 training-policy control on the immutable CP2 cohort.

Historical training modules are imported, never edited. This is an exploratory
policy-bundle comparison, not an exact CP1 reproduction or a causal isolation of
checkpointing alone. The original B0 focal-input mismatch is fixed in all arms,
as explicitly requested by the user; no corrected-logit claim is made.
"""

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, Subset

from .checkpoint import AuthorCheckpointPolicy, link_best, random_state, restore_random_state, save_checkpoint
from .common import ROOT, digest_file, digest_json, write_json
from .cp2_attention import build_attention_model, model_profile
from .cp2_data import verify_cp2_manifest
from .cp2_heads import AuxiliarySegmentationModel, HeadObjective, build_head_model
from .cp2_phase_h import select_candidate
from .cp2_phase_l import _verify_precision, phase_l_sources, train_epoch, validate_epoch
from .cp2_protocol import BaselineObjective
from .cp2_train import export_evaluation, gpu_snapshot, tensor_fingerprint
from .data import make_dataset
from .engine import environment_record, setup
from .reference import build_loss_and_metrics, build_model, load_reference, verify_reference


EXPERIMENTS = ("R0", "RH2", "RA6")


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    value = phase_l_sources()
    for name in ("cp2_recipe.py", "cp2_attention.py", "cp2_heads.py", "cp2_phase_h.py"):
        path = "src/foot_ulcer_segmentation/" + name
        value[path] = digest_file(ROOT / path)
    for path in ("scripts/run_cp2_cp1_recipe.py", "configs/cp2/cp1_recipe.json"):
        value[path] = digest_file(ROOT / path)
    return value


def validate_config(config):
    """Fail closed if the bounded preregistered comparison is accidentally expanded."""
    reference = read(ROOT / "configs/baseline.json")
    control = read(ROOT / config["recipe"]["control_root"] / "protocol.json")
    if tuple(config["recipe"]["experiment_order"]) != EXPERIMENTS:
        raise ValueError("Exactly R0, RH2, RA6 are registered")
    if config["recipe"]["main_loss"] != "B0":
        raise ValueError("This registered experiment preserves the explicitly requested CP1 mismatch")
    for key in ("seed", "cpu_threads", "data", "model", "loss"):
        if config[key] != control[key]:
            raise ValueError(f"Frozen CP2 control changed: {key}")
    training = config["training"]
    for key in ("batch_size", "num_workers", "max_epochs", "learning_rate", "weight_decay",
                "early_stopping_patience", "checkpoint_policy"):
        if training[key] != reference["training"][key]:
            raise ValueError(f"CP1 training recipe changed: {key}")
    expected = {"type": "ReduceLROnPlateau", "monitor": "main_validation_regional_loss",
                "mode": "min", "threshold": .0001, "threshold_mode": "rel", "cooldown": 0,
                "eps": 1e-8, **reference["training"]["scheduler"]}
    if training["scheduler"] != expected:
        raise ValueError("Scheduler must reproduce CP1 defaults and main-loss monitoring")
    for key in ("precision", "amp", "matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
                "augmentation", "finite_gradient_check_every_step"):
        if training[key] != control["training"][key]:
            raise ValueError(f"Unregistered training change: {key}")
    for key, value in control["evaluation"].items():
        if key != "test_policy" and config["evaluation"][key] != value:
            raise ValueError(f"Evaluator changed: {key}")
    if config["phase_a"] != read(ROOT / "configs/cp2/phase_a.json")["phase_a"]:
        raise ValueError("Historical A6 architecture/configuration changed")
    if config["phase_h"] != read(ROOT / "configs/cp2/phase_h.json")["phase_h"]:
        raise ValueError("Historical H2 architecture/configuration changed")


class RecordingBatchSampler:
    """Observe original sampler output without changing global RNG consumption."""
    def __init__(self, sampler):
        self.sampler = sampler
        self.indices = []

    def __iter__(self):
        self.indices = []
        for batch in self.sampler:
            self.indices.extend(batch)
            yield batch

    def __len__(self):
        return len(self.sampler)


def make_loader(config, manifest, split, smoke=False):
    dataset, rows = make_dataset(config, manifest, split, augment=split == "train")
    if smoke:
        # Three real batches include empty/positive masks and the final singleton.
        indices = ([i for i, row in enumerate(rows) if not row["gt_pixels"]][:1]
                   + [i for i, row in enumerate(rows) if row["gt_pixels"]][:4])
        dataset, rows = Subset(dataset, indices), [rows[i] for i in indices]
    sampler = RandomSampler(dataset) if split == "train" else SequentialSampler(dataset)
    batches = RecordingBatchSampler(BatchSampler(sampler, config["training"]["batch_size"], False))
    loader = DataLoader(dataset, batch_sampler=batches,
                        num_workers=0 if smoke else config["training"]["num_workers"])
    return loader, rows, batches


class RecipeObjective(HeadObjective):
    """Record exact source batch-IoU alongside source B0 main/auxiliary primitives."""
    def __init__(self, experiment):
        if experiment not in EXPERIMENTS:
            raise ValueError("Unknown recipe experiment")
        super().__init__("H2" if experiment == "RH2" else "H0")
        self.baseline = BaselineObjective("B0")
        _, metrics = build_loss_and_metrics()
        self.source_iou = metrics[0]

    def components(self, outputs, truth):
        parts = super().components(outputs, truth)
        parts["source_iou"] = self.source_iou(outputs.main_logits.detach().sigmoid(), truth)
        return parts


def build_recipe_model(config, experiment, initial):
    if experiment == "RA6":
        base, proof = build_attention_model(config, "A6", initial)
        model = AuxiliarySegmentationModel(base)
    elif experiment in ("R0", "RH2"):
        model, proof = build_head_model(config, "H2" if experiment == "RH2" else "H0", initial)
    else:
        raise ValueError("Unknown recipe experiment")
    for name, tensor in initial["state_dict"].items():
        if not torch.equal(model.base.state_dict()[name], tensor):
            raise ValueError(f"Pairable original tensor changed: {name}")
    return model, {**proof, "all_original_named_tensors_equal": True,
                   "source_tensor_sha256": tensor_fingerprint(initial["state_dict"])}


def prepare(config, root):
    validate_config(config)
    verify_reference()
    root = Path(root)
    control = ROOT / config["recipe"]["control_root"]
    manifest = verify_cp2_manifest(read(control / "data/manifest.json"))
    if manifest["fingerprint"] != config["recipe"]["control_manifest_fingerprint"]:
        raise ValueError("CP2 split changed")
    if digest_file(control / "initial_weights.pth") != config["recipe"]["control_initial_weights_sha256"]:
        raise ValueError("Historical initialization changed")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        if read(root / "protocol.json") != config or read(root / "data/manifest.json") != manifest:
            raise ValueError("Protocol changed; use a new run")
    else:
        write_json(root / "protocol.json", config)
        write_json(root / "data/manifest.json", manifest)
    return manifest


def initial_state(config):
    """Capture one common post-construction RNG state, independently of added modules."""
    # First imports may consume RNG. Complete lazy imports BEFORE the common seed,
    # otherwise the first arm differs from later arms despite the same seed.
    load_reference()
    setup(config["seed"], config["cpu_threads"])
    if torch.cuda.is_available():
        torch.cuda.get_rng_state_all()  # Materialize seeded CUDA generators before capturing.
    base = build_model(config, pretrained=False)
    rng = random_state()
    del base
    return rng


def run_experiment(config, manifest, root, experiment, device_name, smoke=False, resume=False):
    root, device = Path(root), torch.device(device_name)
    run = root / (("smoke-" if smoke else "") + experiment)
    code = sources()
    identity = digest_json({"config": config, "sources": code, "manifest": manifest["fingerprint"],
                            "experiment": experiment, "smoke": smoke})
    if (run / "status.json").exists():
        old = read(run / "status.json")
        if old["identity"] != identity:
            raise ValueError("Existing run source/config mismatch")
        if old["status"] in ("completed", "passed"):
            if digest_file(run / "best.pth") != old["best_checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
            return old
        if not resume:
            raise FileExistsError("Interrupted run; inspect before explicit resume")
    elif resume:
        resume = False
    if not resume:
        run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    setup(config["seed"], config["cpu_threads"])
    _verify_precision(config)
    environment = environment_record(config, manifest, device)
    if resume and read(run / "environment.json")["packages"] != environment["packages"]:
        raise ValueError("Packages changed; cannot resume")
    if not resume:
        write_json(run / "environment.json", environment)
        write_json(run / "config.json", {**config, "experiment": experiment})
        write_json(run / "split-manifest.json", manifest)
    common_rng = initial_state(config)
    initial = torch.load(ROOT / config["recipe"]["control_root"] / "initial_weights.pth", map_location="cpu")
    model, proof = build_recipe_model(config, experiment, initial)
    del initial
    write_json(run / "initialization_verified.json", proof)
    model.to(device)
    objective = RecipeObjective(experiment)
    settings = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **{k: v for k, v in settings["scheduler"].items() if k not in ("type", "monitor")})
    policy, history = AuthorCheckpointPolicy(), []
    restore_random_state(common_rng)
    if resume:
        state = torch.load(run / "last.pth", map_location=device)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        policy = AuthorCheckpointPolicy(**state["policy"])
        history = state["history"]
        restore_random_state(state["rng"])
        del state
    train_loader, training_rows, sampler = make_loader(config, manifest, "train", smoke)
    valid_loader, validation_rows, _ = make_loader(config, manifest, "validation", smoke)
    profile_rng = random_state()
    profile = model_profile(model, torch.zeros(1, 3, 224, 224, device=device))
    restore_random_state(profile_rng)
    write_json(run / "profile.json", profile)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    limit = 1 if smoke else settings["max_epochs"]
    for epoch in range(len(history), limit):
        if policy.patience_count >= settings["early_stopping_patience"]:
            break
        epoch_started = time.monotonic()
        trained = train_epoch(model, objective, optimizer, train_loader, device)
        evaluator, validated = validate_epoch(model, objective, valid_loader, validation_rows, config, device)
        selected = policy.update(validated["regional"], validated["source_iou"])
        scheduler.step(validated["regional"])
        order = [training_rows[i]["sample_id"] for i in sampler.indices]
        if len(order) != len(training_rows) or len(set(order)) != len(order):
            raise ValueError("Training membership/count mismatch")
        entry = {"epoch": epoch + 1, "train": trained["components"], "validation": validated,
                 "validation_image_dice": evaluator.summary()["all"]["image_based"]["dice"],
                 "selected_checkpoint": selected, "policy": policy.state_dict(),
                 "learning_rate": optimizer.param_groups[0]["lr"],
                 "training_order_sha256": digest_json(order),
                 "augmented_input_sha256": trained["augmented_input_sha256"],
                 "training_steps": trained["steps"], "step_seconds_mean": trained["step_seconds_mean"],
                 "step_seconds_p95": trained["step_seconds_p95"],
                 "seconds": time.monotonic() - epoch_started, "gpu": gpu_snapshot()}
        history.append(entry)
        state = {"identity": identity, "completed_epochs": epoch + 1, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "policy": policy.state_dict(), "history": history, "rng": random_state()}
        save_checkpoint(run / "last.pth", state)
        if selected:
            link_best(run / "last.pth", run / "best.pth")
        del state
        write_json(run / "history.json", history)
        write_json(run / "status.json", {"status": "running", "identity": identity,
                   "diagnostic_smoke_only": smoke, "completed_epochs": len(history),
                   "updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
                   "training_sources": code, "last_epoch": entry})
        print(json.dumps({"experiment": experiment, **entry}), flush=True)
    if not history:
        raise ValueError("No training executed")
    state = torch.load(run / "best.pth", map_location=device)
    model.load_state_dict(state["state_dict"])
    # Compare the same selected weights before and after reload, then re-export.
    replay_rng = random_state()
    images, _ = next(iter(valid_loader))
    model.eval()
    with torch.no_grad():
        expected = model(images.to(device)).sigmoid().cpu()
        next(model.parameters()).add_(1)
    model.load_state_dict(state["state_dict"])
    with torch.no_grad():
        actual = model(images.to(device)).sigmoid().cpu()
    if not torch.equal(expected, actual):
        raise ValueError("Checkpoint reload prediction mismatch")
    restore_random_state(replay_rng)
    selected_epoch = state["completed_epochs"]
    del state
    export = run / "predictions/validation"
    if export.exists():
        export.rename(run / ("incomplete_validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    evaluator, parts = validate_epoch(model, objective, valid_loader, validation_rows, config, device, export)
    report = export_evaluation(run / "evaluation-validation.json", evaluator, {
        "split": "validation", "experiment": experiment, "checkpoint_sha256": digest_file(run / "best.pth"),
        "selected_epoch": selected_epoch, "diagnostic_smoke_only": smoke, "components": parts})
    outcome = {"status": "passed" if smoke else "completed", "identity": identity,
               "diagnostic_smoke_only": smoke, "completed_epochs": len(history), "selected_epoch": selected_epoch,
               "training_sources": code, "best_checkpoint_sha256": report["checkpoint_sha256"],
               "checkpoint_reload_passed": True, "seconds": time.monotonic() - started,
               "updated_utc": datetime.now(timezone.utc).isoformat(), "test_images_evaluated": 0,
               "training_steps": sum(e["training_steps"] for e in history),
               "active_forward_parameters": profile["active_forward_parameters"],
               "registered_parameters": profile["registered_parameters"],
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    write_json(run / "status.json", outcome)
    del model, optimizer, scheduler
    gc.collect()
    return outcome


def verify_paired_inputs(root, smoke=False):
    histories = {n: read(Path(root) / (("smoke-" if smoke else "") + n) / "history.json") for n in EXPERIMENTS}
    comparisons = []
    for name in EXPERIMENTS[1:]:
        epochs = min(len(histories["R0"]), len(histories[name]))
        for a, b in zip(histories["R0"], histories[name]):
            if any(a[k] != b[k] for k in ("training_order_sha256", "augmented_input_sha256")):
                raise ValueError(f"Actual common-epoch training inputs differ: {name}, epoch {a['epoch']}")
        comparisons.append({"candidate": name, "common_epochs": epochs})
    return {"status": "passed", "actual_order_and_augmented_bytes_equal": True, "comparisons": comparisons}


def select_recipe(root):
    """Reuse historical validation guards against the NEW recipe control, never test."""
    root = Path(root)
    reports = {n: read(root / n / "evaluation-validation.json") for n in EXPERIMENTS}
    profiles = {n: read(root / n / "profile.json") for n in EXPERIMENTS}
    decision = select_candidate({"H1": reports["RA6"], "H2": reports["RH2"]}, reports["R0"],
                                {"H0": profiles["R0"], "H1": profiles["RA6"], "H2": profiles["RH2"]})
    names = {"H0": "R0", "H1": "RA6", "H2": "RH2"}
    decision["selected_candidate"] = names[decision["selected_candidate"]]
    decision["decisions"] = {names[k]: v for k, v in decision["decisions"].items()}
    decision["unexecuted"] = []
    decision["tie_order"] = ["RA6", "RH2"]
    return decision


def evaluate_test(config, manifest, root, experiment, device_name):
    root, run = Path(root), Path(root) / experiment
    if not (root / "selection.json").exists():
        raise ValueError("Persist validation-only selection before test")
    status = read(run / "status.json")
    if status["status"] != "completed" or status["diagnostic_smoke_only"]:
        raise ValueError("Completed full training required")
    checksum = digest_file(run / "best.pth")
    if checksum != status["best_checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    if (run / "evaluation-test.json").exists():
        report = read(run / "evaluation-test.json")
        if report["checkpoint_sha256"] != checksum:
            raise ValueError("Cached test checkpoint changed")
        return report
    setup(config["seed"], config["cpu_threads"])
    initial = torch.load(ROOT / config["recipe"]["control_root"] / "initial_weights.pth", map_location="cpu")
    model, _ = build_recipe_model(config, experiment, initial)
    del initial
    state = torch.load(run / "best.pth", map_location="cpu")
    if state["identity"] != status["identity"]:
        raise ValueError("Checkpoint identity mismatch")
    model.load_state_dict(state["state_dict"])
    del state
    device = torch.device(device_name)
    model.to(device)
    loader, rows, _ = make_loader(config, manifest, "test")
    export = run / "predictions/test"
    if export.exists():
        export.rename(run / ("incomplete_test_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    evaluator, parts = validate_epoch(model, RecipeObjective(experiment), loader, rows, config, device, export)
    return export_evaluation(run / "evaluation-test.json", evaluator, {
        "split": "test", "experiment": experiment, "checkpoint_sha256": checksum,
        "selected_epoch": status["selected_epoch"], "diagnostic_smoke_only": False,
        "components": parts, "interpretation": "Previously exposed source-test; exploratory only"})
