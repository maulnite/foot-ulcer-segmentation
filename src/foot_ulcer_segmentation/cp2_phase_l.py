"""Isolated phase-L screening using frozen phase-P data and training utilities.

The old trainer remains byte-identical. Its data loaders, model, checkpoint I/O,
evaluator and RNG helpers are reused; only the objective-aware loop is new.
"""

from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .checkpoint import link_best, random_state, restore_random_state, save_checkpoint
from .common import ROOT, digest_file, digest_json, write_json
from .cp2_data import verify_cp2_manifest
from .cp2_losses import PhaseLObjective
from .cp2_protocol import CP2Evaluator, ImageDiceCheckpoint
from .cp2_train import (
    export_evaluation, gpu_snapshot, make_cp2_loader, sync, tensor_fingerprint,
    training_fingerprints,
)
from .engine import environment_record, setup
from .reference import build_model, verify_reference


EXPERIMENTS = ("L1", "L2", "L3", "L4")


def read(path):
    return json.loads(Path(path).read_text())


def phase_l_sources():
    result = training_fingerprints()
    for name in ("src/foot_ulcer_segmentation/cp2_losses.py",
                 "src/foot_ulcer_segmentation/cp2_phase_l.py",
                 "scripts/run_cp2_phase_l.py"):
        result[name] = digest_file(ROOT / name)
    return result


def validate_protocol(config):
    """Verify the original comparison controls rather than constructing a new split."""
    phase = config["phase_l"]
    control = ROOT / phase["control_root"]
    old = read(control / "protocol.json")
    for key in ("seed", "split_seed", "cpu_threads", "data", "model", "training", "loss"):
        if config[key] != old[key]:
            raise ValueError(f"Frozen phase-P control changed: {key}")
    for key, value in old["evaluation"].items():
        if key != "test_policy" and config["evaluation"][key] != value:
            raise ValueError(f"Evaluator changed: {key}")
    if tuple(phase["experiment_order"]) != EXPERIMENTS:
        raise ValueError("This runner executes exactly the L1–L4 screening")
    baseline = read(control / "B1/status.json")
    if baseline["status"] != "completed" or baseline["diagnostic_smoke_only"]:
        raise ValueError("A completed real B1 control is required")
    if digest_file(control / "B1/best.pth") != phase["control_B1_checkpoint_sha256"]:
        raise ValueError("Frozen B1 checkpoint changed")
    if digest_file(control / "initial_weights.pth") != phase["control_initial_weights_sha256"]:
        raise ValueError("Shared initial weights changed")
    if baseline["training_sources"] != training_fingerprints():
        raise ValueError("Frozen phase-P source code changed")
    manifest = verify_cp2_manifest(read(control / "data/manifest.json"))
    if manifest["fingerprint"] != phase["control_manifest_fingerprint"]:
        raise ValueError("Frozen duplicate-group membership changed")
    return manifest


def prepare(config, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = validate_protocol(config)
    protocol = root / "protocol.json"
    if protocol.exists():
        if read(protocol) != config:
            raise ValueError("Protocol changed; use a new append-only run directory")
        if read(root / "data/manifest.json") != manifest:
            raise ValueError("Cached membership differs from phase P")
    else:
        write_json(protocol, config)
        write_json(root / "data/manifest.json", manifest)
        source = ROOT / config["phase_l"]["control_root"]
        temporary = root / "initial_weights.pth.tmp"
        shutil.copyfile(source / "initial_weights.pth", temporary)
        temporary.replace(root / "initial_weights.pth")
        write_json(root / "initialization.json", read(source / "initialization.json"))
        write_json(root / "control_reference.json", {
            "status": "CACHED — L0 is unchanged phase-P B1; not retrained",
            "source_root": config["phase_l"]["control_root"],
            "status_record": read(source / "B1/status.json"),
            "validation": read(source / "B1/evaluation-validation.json"),
            "history_sha256": digest_file(source / "B1/history.json"),
        })
    if digest_file(root / "initial_weights.pth") != config["phase_l"]["control_initial_weights_sha256"]:
        raise ValueError("Local shared initialization copy changed")
    return manifest


def _component_means(values):
    return {key: float(np.mean([row[key] for row in values])) for key in values[0]}


def train_epoch(model, objective, optimizer, loader, device):
    """One epoch over original training examples, including the final singleton."""
    model.train()
    input_hash = hashlib.sha256()
    components, timings = [], []
    for images, truth in loader:
        input_hash.update(images.numpy().tobytes())
        input_hash.update(truth.numpy().tobytes())
        sync(device)
        started = time.monotonic()
        images, truth = images.to(device), truth.to(device)
        optimizer.zero_grad()
        parts = objective.components(model(images), truth)
        loss = parts["total"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss; no configuration fallback")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not torch.stack([torch.isfinite(value).all() for value in gradients]).all():
            raise ValueError("Missing/nonfinite gradients; no configuration fallback")
        optimizer.step()
        sync(device)
        timings.append(time.monotonic() - started)
        components.append({key: float(value.detach()) for key, value in parts.items()})
    return {"components": _component_means(components), "steps": len(timings),
            "step_seconds_mean": float(np.mean(timings)),
            "step_seconds_p95": float(np.percentile(timings, 95)),
            "augmented_input_sha256": input_hash.hexdigest()}


def validate_epoch(model, objective, loader, rows, config, device, export=None):
    evaluator = CP2Evaluator(config["evaluation"], boundaries=export is not None)
    if export is not None:
        export = Path(export)
        export.mkdir(parents=True, exist_ok=False)
    model.eval()
    values, exported, offset = [], [], 0
    with torch.no_grad():
        for images, truth in loader:
            logits = model(images.to(device))
            parts = objective.components(logits, truth.to(device))
            if not all(torch.isfinite(value) for value in parts.values()):
                raise ValueError("Nonfinite validation objective component")
            values.append({key: float(value) for key, value in parts.items()})
            for probability, mask in zip(logits.sigmoid().cpu().numpy(), truth.numpy()):
                row = rows[offset]
                evaluator.add(row, mask[0], probability[0])
                if export is not None:
                    file = export / (Path(row["name"]).stem + ".npz")
                    temporary = file.with_suffix(".npz.tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, probability=probability[0],
                                            prediction=(probability[0] >= .5).astype(np.uint8),
                                            truth=mask[0].astype(np.uint8))
                    temporary.replace(file)
                    exported.append({"sample_id": row["sample_id"], "file": file.name,
                                     "sha256": digest_file(file)})
                offset += 1
    if offset != len(rows):
        raise ValueError("Prediction count differs from original image count")
    if export is not None:
        write_json(export / "manifest.json", exported)
    return evaluator, _component_means(values)


def _verify_precision(config):
    settings = config["training"]
    if torch.get_default_dtype() != torch.float32 or settings["amp"] or settings["precision"] != "float32":
        raise ValueError("Preserve phase-P FP32; automatic AMP is prohibited")
    if (torch.backends.cuda.matmul.allow_tf32 != settings["matmul_allow_tf32"]
            or torch.backends.cudnn.allow_tf32 != settings["cudnn_allow_tf32"]
            or torch.backends.cudnn.benchmark != settings["cudnn_benchmark"]):
        raise ValueError("Backend precision flags differ from the frozen baseline")


def run_experiment(config, manifest, root, experiment, device_name, smoke=False, resume=False):
    if experiment not in EXPERIMENTS:
        raise ValueError("L0 is cached, never retrained")
    root = Path(root)
    run = root / (("smoke-" if smoke else "") + experiment)
    sources = phase_l_sources()
    identity = digest_json({"config": config, "objective": experiment,
                            "manifest": manifest["fingerprint"], "sources": sources})
    if (run / "status.json").exists():
        previous = read(run / "status.json")
        if previous["status"] in ("completed", "passed"):
            if previous["identity"] != identity or digest_file(run / "best.pth") != previous["best_checkpoint_sha256"]:
                raise ValueError("Completed stage/checkpoint compatibility failed")
            print(f"CACHED {experiment}; completed stage preserved", flush=True)
            return previous
        if not resume:
            raise FileExistsError("Interrupted run exists; inspect failure before --resume")
    elif run.exists():
        if not resume or not (run / "last.pth").exists():
            raise FileExistsError("Partial initialization is preserved; no valid checkpoint to resume")
    else:
        resume = False
        run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = torch.device(device_name)
    settings = config["training"]
    verify_cp2_manifest(manifest)
    verify_reference()
    setup(config["seed"], config["cpu_threads"])
    _verify_precision(config)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    environment = environment_record(config, manifest, device)
    if resume:
        if read(run / "identity.json")["fingerprint"] != identity:
            raise ValueError("Cannot resume after config/code/input changes")
        if read(run / "environment.json")["packages"] != environment["packages"]:
            raise ValueError("Cannot resume after environment changes")
    else:
        write_json(run / "identity.json", {"fingerprint": identity})
        write_json(run / "config.json", {"objective_id": experiment, **config})
        write_json(run / "environment.json", environment)
        write_json(run / "split-manifest.json", manifest)
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model = build_model(config, pretrained=False)
    if sum(p.numel() for p in model.parameters()) != 70_968_939:
        raise ValueError("Source model parameter count changed")
    model.load_state_dict(initial["state_dict"])
    fingerprint = tensor_fingerprint(model.state_dict())
    if fingerprint != config["phase_l"]["control_initial_tensor_sha256"]:
        raise ValueError("Actual initial tensors/buffers differ from L0/B1")
    initialization = {"shared_initialization_tensor_sha256": fingerprint,
                      "all_named_tensors_equal": True,
                      "initial_weights_file_sha256": digest_file(root / "initial_weights.pth"),
                      "split_fingerprint": manifest["fingerprint"],
                      "parameter_count": sum(p.numel() for p in model.parameters())}
    if (run / "initialization_verified.json").exists():
        if read(run / "initialization_verified.json") != initialization:
            raise ValueError("Resume initialization differs")
    else:
        write_json(run / "initialization_verified.json", initialization)
    del initial
    model = model.to(device)
    objective = PhaseLObjective(experiment, config["phase_l"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{
        key: value for key, value in settings["scheduler"].items() if key not in ("type", "monitor")})
    policy, history, first_epoch = ImageDiceCheckpoint(), [], 0
    if resume:
        state = torch.load(run / "last.pth", map_location=device)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity differs")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        policy = ImageDiceCheckpoint(**state["policy"])
        history, first_epoch = state["history"], state["completed_epochs"]
        restore_random_state(state["rng"])
        del state
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    finite_steps = 0
    write_json(run / "status.json", {"status": "running", "objective_id": experiment,
                                     "completed_epochs": first_epoch, "pid": os.getpid()})
    for epoch in range(first_epoch, 1 if smoke else settings["max_epochs"]):
        if policy.bad_epochs >= settings["early_stopping_patience"]:
            break
        setup(config["seed"] + 10007 * epoch, config["cpu_threads"])
        train_loader, train_rows = make_cp2_loader(config, manifest, "train", epoch, smoke)
        valid_loader, valid_rows = make_cp2_loader(config, manifest, "validation", epoch, smoke)
        epoch_start = time.monotonic()
        before = next(model.parameters()).detach().clone() if smoke else None
        used_lr = optimizer.param_groups[0]["lr"]
        training = train_epoch(model, objective, optimizer, train_loader, device)
        finite_steps += training["steps"]
        if smoke and torch.equal(before, next(model.parameters()).detach()):
            raise ValueError("Smoke optimizer did not update encoder weights")
        evaluator, validation_parts = validate_epoch(model, objective, valid_loader, valid_rows, config, device)
        metric = evaluator.summary()["all"]["image_based"]["dice"]
        selected = policy.update(metric, epoch + 1)
        scheduler.step(metric)
        entry = {"epoch": epoch + 1, "train_loss": training["components"]["total"],
                 "validation_loss": validation_parts["total"],
                 "train_components": training["components"], "validation_components": validation_parts,
                 "validation_image_dice": metric, "learning_rate_used": used_lr,
                 "learning_rate_next": optimizer.param_groups[0]["lr"], "selected": selected,
                 "bad_epochs": policy.bad_epochs, "epoch_compute_seconds": time.monotonic() - epoch_start,
                 "step_seconds_mean": training["step_seconds_mean"],
                 "step_seconds_p95": training["step_seconds_p95"], "steps": training["steps"],
                 "training_order_sha256": digest_json([r["sample_id"] for r in train_rows]),
                 "augmented_input_sha256": training["augmented_input_sha256"],
                 "gpu": gpu_snapshot() if device.type == "cuda" else None}
        history.append(entry)
        save_checkpoint(run / "last.pth", {"identity": identity, "objective_id": experiment,
                        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng": random_state(),
                        "policy": policy.state_dict(), "history": history, "completed_epochs": epoch + 1})
        if selected:
            link_best(run / "last.pth", run / "best.pth")
        entry["epoch_with_checkpoint_seconds"] = time.monotonic() - epoch_start
        write_json(run / "history.json", history)
        write_json(run / "status.json", {"status": "running", "objective_id": experiment,
                   "completed_epochs": epoch + 1, "selected_epoch": policy.epoch,
                   "latest": entry, "pid": os.getpid()})
        print(json.dumps({"objective_id": experiment, **entry}), flush=True)
    chosen = torch.load(run / "best.pth", map_location="cpu")
    if chosen["identity"] != identity:
        raise ValueError("Selected checkpoint identity changed")
    model.load_state_dict(chosen["state_dict"])
    loader, _ = make_cp2_loader(config, manifest, "validation", smoke=smoke)
    images, _ = next(iter(loader))
    model.eval()
    with torch.no_grad():
        expected = model(images.to(device)).sigmoid().cpu()
        next(model.parameters()).add_(1)
    model.load_state_dict(chosen["state_dict"])
    with torch.no_grad():
        replay = model(images.to(device)).sigmoid().cpu()
    if not torch.equal(expected, replay):
        raise ValueError("Checkpoint reload prediction mismatch")
    result = {"status": "passed" if smoke else "completed", "objective_id": experiment,
              "diagnostic_smoke_only": smoke, "pid": os.getpid(),
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "completed_epochs": len(history), "selected_epoch": chosen["completed_epochs"],
              "selected_validation_image_dice": policy.best,
              "stop_reason": "early_stopping_30_nonimproving_epochs" if policy.bad_epochs >= 30 else "epoch_budget",
              "seconds_this_invocation": time.monotonic() - started,
              "finite_gradient_steps_this_invocation": finite_steps, "checkpoint_replay_exact": True,
              "encoder_update_verified_in_smoke": smoke, "best_checkpoint_sha256": digest_file(run / "best.pth"),
              "training_sources": sources, "identity": identity,
              "shared_initialization_tensor_sha256": fingerprint,
              "split_fingerprint": manifest["fingerprint"],
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
    write_json(run / "status.json", result)
    return result


def verify_paired_inputs(config, root, smoke=False):
    root = Path(root)
    control = ROOT / config["phase_l"]["control_root"] / ("smoke-B1" if smoke else "B1")
    reference = read(control / "history.json")
    prefix = "smoke-" if smoke else ""
    records, histories = [], []
    initialization = read(control / "initialization_verified.json")
    for name in EXPERIMENTS:
        run = root / (prefix + name)
        if read(run / "initialization_verified.json") != initialization:
            raise ValueError("L0/L1–L4 initialization or membership differs")
        history = read(run / "history.json")
        for first, second in zip(reference, history):
            for key in ("training_order_sha256", "augmented_input_sha256"):
                if first[key] != second[key]:
                    raise ValueError(f"{name} inputs differ from L0 at epoch {second['epoch']}")
        records.append({"objective_id": name, "epochs_compared_with_L0": min(len(reference), len(history)),
                        "initialization_identical": True, "order_and_augmented_inputs_identical": True})
        histories.append(history)
    for epoch_rows in zip(*histories):
        if any(len({r[key] for r in epoch_rows}) != 1 for key in ("training_order_sha256", "augmented_input_sha256")):
            raise ValueError("New loss candidates have different common-epoch input batches")
    return {"status": "passed", "comparisons": records,
            "common_epochs_all_four": min(map(len, histories))}


def evaluate_candidate(config, manifest, root, experiment, split, device_name):
    root = Path(root)
    if split not in ("validation", "test"):
        raise ValueError("Expected validation or test")
    if split == "test" and not (root / "selection.json").exists():
        raise ValueError("Persist validation-only selection before any new test predictions")
    for name in EXPERIMENTS:
        status = read(root / name / "status.json")
        if status["status"] != "completed" or status["diagnostic_smoke_only"]:
            raise ValueError("All four real-data training runs must finish before evaluation")
    run = root / experiment
    destination = run / f"evaluation-{split}.json"
    if destination.exists():
        return read(destination)
    config_status = read(run / "status.json")
    if digest_file(run / "best.pth") != config_status["best_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    setup(config["seed"], config["cpu_threads"])
    device = torch.device(device_name)
    model = build_model(config, pretrained=False).to(device)
    checkpoint = torch.load(run / "best.pth", map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    export = run / "predictions" / split
    if export.exists():
        # A failed export is retained; a completed evaluation above is never rewritten.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = run / "interrupted_exports" / f"{split}-{stamp}"
        archived.parent.mkdir(exist_ok=True)
        export.rename(archived)
    loader, rows = make_cp2_loader(config, manifest, split)
    started = time.monotonic()
    evaluator, parts = validate_epoch(model, PhaseLObjective(experiment, config["phase_l"]),
                                     loader, rows, config, device, export)
    return export_evaluation(destination, evaluator, {
        "objective_id": experiment, "split": split, "diagnostic_smoke_only": False,
        "selected_epoch": checkpoint["completed_epochs"],
        "checkpoint_sha256": config_status["best_checkpoint_sha256"],
        "split_fingerprint": manifest["fingerprint"], "objective_components": parts,
        "split_loss": parts["total"], "seconds": time.monotonic() - started,
        "test_caveat": "CP1/phase-P-exposed source test; descriptive, not untouched confirmatory evaluation"})


def select_candidate(validation, control):
    """Conservative, preregistered validation-only screen; never consumes test data."""
    if any(report.get("split") != "validation" for report in [control, *validation.values()]):
        raise ValueError("Candidate selection accepts validation metrics only")
    baseline = control["all"]["image_based"]["dice"]
    decisions, eligible = {}, []
    for name in EXPERIMENTS:
        report = validation[name]
        checks = {
            "higher_overall_image_dice": report["all"]["image_based"]["dice"] > baseline,
            "no_small_wound_image_dice_decline": report["subgroups"]["small"]["image_based"]["dice"] >= control["subgroups"]["small"]["image_based"]["dice"],
            "no_more_total_foreground_misses": report["gt_positive"]["total_foreground_misses"] <= control["gt_positive"]["total_foreground_misses"],
            "no_more_empty_gt_images_with_fp": report["gt_empty"]["images_with_prediction"] <= control["gt_empty"]["images_with_prediction"],
        }
        decisions[name] = {"eligible": all(checks.values()), "checks": checks,
                           "validation_image_dice": report["all"]["image_based"]["dice"],
                           "checkpoint_sha256": report["checkpoint_sha256"]}
        if all(checks.values()):
            eligible.append(name)
    chosen = max(eligible, key=lambda n: decisions[n]["validation_image_dice"]) if eligible else "L0"
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "validation_only": True,
            "selected_screening_candidate": chosen, "decisions": decisions,
            "baseline_validation_image_dice": baseline,
            "status": "exploratory one-seed screen; not a clinical acceptance decision",
            "further_phases": "No automatic loss-development matrix, architecture or external evaluation"}
