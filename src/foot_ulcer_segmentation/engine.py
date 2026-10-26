"""Train using the original package runners, with explicit run provenance."""

import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import (AuthorCheckpointPolicy, link_best, random_state,
                         restore_random_state, save_checkpoint)
from .common import ROOT, digest_file, digest_json, project_path, write_json
from .data import make_dataset, prepare_splits, verify_manifest
from .evaluation import SegmentationEvaluator
from .reference import build_loss_and_metrics, build_model, load_reference, verify_reference


def setup(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    import cv2
    cv2.setNumThreads(1)
    # Do not claim bitwise GPU determinism; versions and seed are recorded.


def device_from_name(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    return device


def make_loaders(config, manifest, smoke=False):
    settings = config["training"]
    limit = settings["batch_size"] if smoke else None
    train, _ = make_dataset(config, manifest, "train", augment=True, limit=limit)
    validation, rows = make_dataset(config, manifest, "validation", limit=limit)
    workers = 0 if smoke else settings["num_workers"]
    return (DataLoader(train, batch_size=settings["batch_size"], shuffle=True, num_workers=workers),
            DataLoader(validation, batch_size=settings["batch_size"], shuffle=False, num_workers=workers),
            rows)


def environment_record(config, manifest, device):
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    paths = list((ROOT / "src/foot_ulcer_segmentation").glob("*.py"))
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "interpreter": sys.executable, "packages": packages,
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "git_revision": git.stdout.strip() if git.returncode == 0 else None,
            "git_dirty": bool(status.stdout.strip()),
            "pipeline_sha256": {str(p.relative_to(ROOT)): digest_file(p) for p in paths},
            "config_fingerprint": digest_json(config), "split_fingerprint": manifest["fingerprint"],
            "reference_revision": config["reference_revision"],
            "limits": config["provenance"]}


def evaluate_model(model, loader, rows, device, threshold):
    evaluator = SegmentationEvaluator(threshold)
    model.eval()
    offset = 0
    with torch.no_grad():
        for images, masks in loader:
            predictions = model(images.to(device)).cpu().numpy()
            for prediction, mask in zip(predictions, masks.numpy()):
                evaluator.add(rows[offset]["name"], mask[0], prediction[0])
                offset += 1
    if offset != len(rows):
        raise RuntimeError("Evaluation rows do not match loader samples")
    return evaluator


def write_evaluation(path, evaluator, split, smoke=False):
    result = evaluator.summary()
    result.update(split=split, diagnostic_smoke_only=smoke)
    write_json(path, result)
    with Path(path).with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluator.rows[0]))
        writer.writeheader()
        writer.writerows(evaluator.rows)
    return result


def _check_state(state, config, manifest):
    if state["config_fingerprint"] != digest_json(config):
        raise ValueError("Checkpoint configuration does not match this run")
    if state["split_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint split does not match this run")


def run_training(config, run_directory, device_name="cpu", smoke=False, resume=False):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    device = device_from_name(device_name)
    setup(config["seed"], config["cpu_threads"])
    verify_reference()
    manifest = prepare_splits(config)
    run = project_path(run_directory)
    if resume:
        saved_config = json.loads((run / "config.json").read_text())
        if saved_config != config or smoke:
            raise ValueError("Resume requires the original full-run configuration")
        previous_manifest = verify_manifest(json.loads((run / "split-manifest.json").read_text()))
        if previous_manifest["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("The original run used a different split")
        previous_environment = json.loads((run / "environment.json").read_text())
        current_environment = environment_record(config, manifest, device)
        if previous_environment["pipeline_sha256"] != current_environment["pipeline_sha256"]:
            raise ValueError("Pipeline source changed; preserve the original code when resuming")
        if previous_environment["packages"] != current_environment["packages"]:
            raise ValueError("Environment packages changed; restore the recorded environment")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", config)
        write_json(run / "split-manifest.json", manifest)
        write_json(run / "environment.json", environment_record(config, manifest, device))
    write_json(run / "status.json", {"status": "running", "smoke_only": smoke})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Keep downloaded ImageNet weights in an explicitly ignored project cache.
    torch.hub.set_dir(str(ROOT / "outputs/cache/torch/hub"))
    model = build_model(config, pretrained=not resume).to(device)
    if sum(p.numel() for p in model.parameters()) != 70_968_939:
        raise RuntimeError("Unexpected reference architecture parameter count")
    settings = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", **settings["scheduler"])
    loss, metrics = build_loss_and_metrics()
    _, utils, _ = load_reference()
    train_epoch = utils.train.TrainEpoch(model, loss, metrics, optimizer,
                                         device=str(device), verbose=not smoke)
    valid_epoch = utils.train.ValidEpoch(model, loss, metrics, device=str(device), verbose=not smoke)
    train_loader, valid_loader, validation_rows = make_loaders(config, manifest, smoke=smoke)
    policy, start_epoch = AuthorCheckpointPolicy(), 0
    history = []
    if resume:
        state = torch.load(run / "last.pth", map_location=device)
        _check_state(state, config, manifest)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        policy = AuthorCheckpointPolicy(**state["checkpoint_policy"])
        start_epoch = state["completed_epochs"]
        history = state["history"]
        restore_random_state(state["random_state"])
    max_epochs = 1 if smoke else settings["max_epochs"]
    smoke_checks = {}
    for epoch in range(start_epoch, max_epochs):
        if policy.patience_count >= settings["early_stopping_patience"]:
            break
        epoch_started = time.monotonic()
        before_update = next(model.parameters()).detach().clone() if smoke else None
        train_logs = {k: float(v) for k, v in train_epoch.run(train_loader).items()}
        if smoke:
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError("Smoke backward pass produced missing/nonfinite gradients")
            if torch.equal(before_update, next(model.parameters()).detach()):
                raise RuntimeError("Smoke optimizer step did not update encoder weights")
            smoke_checks = {"finite_gradients_passed": True, "encoder_weights_updated": True}
        valid_logs = {k: float(v) for k, v in valid_epoch.run(valid_loader).items()}
        if not all(np.isfinite(value) for value in [*train_logs.values(), *valid_logs.values()]):
            raise ValueError("Nonfinite training/validation logs")
        loss_key = loss.__name__
        selected = policy.update(valid_logs[loss_key], valid_logs["iou_score"])
        scheduler.step(valid_logs[sorted(valid_logs)[0]])
        entry = {"epoch": epoch + 1, "train": train_logs, "validation": valid_logs,
                 "selected_checkpoint": selected, "learning_rate": optimizer.param_groups[0]["lr"],
                 "seconds": time.monotonic() - epoch_started}
        history.append(entry)
        state = {"completed_epochs": epoch + 1, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "checkpoint_policy": policy.state_dict(), "random_state": random_state(),
                 "config_fingerprint": digest_json(config), "split_fingerprint": manifest["fingerprint"],
                 "history": history}
        save_checkpoint(run / "last.pth", state)
        if selected:
            link_best(run / "last.pth", run / "best.pth")
        write_json(run / "history.json", history)
        write_json(run / "status.json", {
            "status": "running", "smoke_only": smoke, "device": str(device),
            "started_at_utc": started_at, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_epochs": epoch + 1, "max_epochs": max_epochs,
            "validation_loss": valid_logs[loss_key], "validation_iou": valid_logs["iou_score"],
            "early_stopping_count": policy.patience_count,
            "elapsed_seconds": time.monotonic() - started,
        })
        print(json.dumps({"epoch": epoch + 1, "validation_loss": valid_logs[loss_key],
                          "validation_iou": valid_logs["iou_score"], "selected": selected}), flush=True)
        # Release references to the old checkpoint object before another epoch.
        del state
    checkpoint = torch.load(run / "best.pth", map_location=device)
    _check_state(checkpoint, config, manifest)
    model.load_state_dict(checkpoint["state_dict"])
    evaluator = evaluate_model(model, valid_loader, validation_rows, device,
                               config["evaluation"]["prediction_threshold"])
    result = write_evaluation(run / "validation.json", evaluator, "validation", smoke)
    outcome = {"status": "passed" if smoke else "completed", "smoke_only": smoke,
               "device": str(device), "started_at_utc": started_at,
               "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "completed_epochs": len(history), "selected_epoch": checkpoint["completed_epochs"],
               "seconds": time.monotonic() - started, "split_counts": {s: len(v) for s, v in manifest["splits"].items()},
               "evaluated_validation_images": len(validation_rows),
               "test_images_evaluated": 0, "checkpoint_reload_passed": True,
               "metrics_are_reproduction_results": not smoke,
               "config_fingerprint": digest_json(config), "split_fingerprint": manifest["fingerprint"]}
    outcome.update(smoke_checks)
    if device.type == "cuda":
        outcome["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        outcome["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    if smoke:
        # Confirm exact prediction restoration after an intentional parameter change.
        images, _ = next(iter(valid_loader))
        model.eval()
        with torch.no_grad():
            expected_prediction = model(images.to(device)).cpu()
            next(model.parameters()).add_(1.0)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        with torch.no_grad():
            actual_prediction = model(images.to(device)).cpu()
        if not torch.equal(expected_prediction, actual_prediction):
            raise RuntimeError("Predictions changed after checkpoint reload")
        outcome["checkpoint_prediction_equality_passed"] = True
        # Smoke checkpoints are not useful trained models; keep small diagnostics.
        (run / "last.pth").unlink()
        (run / "best.pth").unlink()
        outcome["temporary_checkpoint_removed"] = True
    weights = ROOT / "outputs/cache/torch/hub/checkpoints"
    outcome["pretrained_weight_sha256"] = {p.name: digest_file(p) for p in weights.glob("*.pth")}
    write_json(run / "status.json", outcome)
    return outcome, result


def evaluate_run(run_directory, split="validation", checkpoint_name="best.pth", device_name="cpu"):
    run = project_path(run_directory)
    config = json.loads((run / "config.json").read_text())
    setup(config["seed"], config["cpu_threads"])
    manifest = verify_manifest(json.loads((run / "split-manifest.json").read_text()))
    device = device_from_name(device_name)
    model = build_model(config, pretrained=False).to(device)
    state = torch.load(run / checkpoint_name, map_location=device)
    _check_state(state, config, manifest)
    model.load_state_dict(state["state_dict"])
    dataset, rows = make_dataset(config, manifest, split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    evaluator = evaluate_model(model, loader, rows, device, config["evaluation"]["prediction_threshold"])
    return write_evaluation(run / f"evaluation-{split}.json", evaluator, split)
