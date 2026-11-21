"""Isolated phase-P training; CP1 source and artifacts remain untouched."""

import csv
import gc
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .checkpoint import link_best, random_state, restore_random_state, save_checkpoint
from .common import ROOT, digest_file, digest_json, write_json
from .cp2_data import verify_cp2_manifest
from .cp2_protocol import BaselineObjective, CP2Evaluator, ImageDiceCheckpoint
from .data import make_dataset
from .engine import environment_record, setup
from .reference import build_model, verify_reference


def training_fingerprints():
    names = ["cp2_train.py", "cp2_data.py", "cp2_protocol.py", "reference.py", "data.py",
             "checkpoint.py", "evaluation.py", "engine.py", "common.py"]
    paths = [ROOT / "src/foot_ulcer_segmentation" / n for n in names]
    paths += [ROOT / "scripts/pfom_metrics.py", ROOT / "requirements-training.lock.txt"]
    return {str(p.relative_to(ROOT)): digest_file(p) for p in paths}


def tensor_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def shared_initialization(config, root):
    root = Path(root)
    path = root / "initial_weights.pth"
    if path.exists():
        record = json.loads((root / "initialization.json").read_text())
        if digest_file(path) != record["file_sha256"]:
            raise ValueError("Shared initialization file changed")
        return record
    setup(config["seed"], config["cpu_threads"])
    torch.hub.set_dir(str(ROOT / "outputs/cache/torch/hub"))
    weights = list((ROOT / "outputs/cache/torch/hub/checkpoints").glob("*.pth"))
    if not weights:
        raise FileNotFoundError("Pretrained weights cache absent; no automatic download in phase P")
    model = build_model(config, pretrained=True)
    state = model.state_dict()
    fingerprint = tensor_fingerprint(state)
    save_checkpoint(path, {"state_dict": state, "seed": config["seed"],
                           "tensor_fingerprint": fingerprint})
    record = {"file_sha256": digest_file(path), "tensor_fingerprint": fingerprint,
              "parameters_total": sum(p.numel() for p in model.parameters()),
              "pretrained_weights": {p.name: digest_file(p) for p in weights},
              "method": "One shared state_dict; exact tensor fingerprint verified independently in each run"}
    write_json(root / "initialization.json", record)
    return record


def seed_worker(_):
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)
    import cv2
    cv2.setNumThreads(1)


def make_cp2_loader(config, manifest, split, epoch=0, smoke=False):
    dataset, rows = make_dataset(config, manifest, split, augment=split == "train")
    seed = config["seed"] + 10007 * epoch
    generator = torch.Generator().manual_seed(seed)
    indices = list(range(len(rows)))
    if smoke:
        empty = [i for i in indices if rows[i]["gt_pixels"] == 0]
        positive = [i for i in indices if rows[i]["gt_pixels"] > 0]
        indices = (empty[:1] + positive[:5])[:6]
    if split == "train":
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[i] for i in order]
    selected = [rows[i] for i in indices]
    loader = DataLoader(Subset(dataset, indices), batch_size=config["training"]["batch_size"],
                        shuffle=False, drop_last=False, num_workers=0 if smoke else config["training"]["num_workers"],
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + 1))
    return loader, selected


def gpu_snapshot():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return {"utc": datetime.now(timezone.utc).isoformat(), "gpu_csv": result.stdout.strip(),
            "columns": "timestamp,name,memory_used_MiB,memory_free_MiB,gpu_util_percent,power_W,temp_C"}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_epoch(model, objective, loader, rows, config, device, export=None):
    evaluator = CP2Evaluator(config["evaluation"], boundaries=export is not None)
    model.eval()
    losses, offset, exported = [], 0, []
    if export is not None:
        export = Path(export)
        export.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        for images, truth in loader:
            logits = model(images.to(device))
            loss = objective(logits, truth.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            losses.append(float(loss))
            probabilities = logits.sigmoid().cpu().numpy()
            for probability, mask in zip(probabilities, truth.numpy()):
                row = rows[offset]
                evaluator.add(row, mask[0], probability[0])
                if export is not None:
                    file = export / (Path(row["name"]).stem + ".npz")
                    np.savez_compressed(file, probability=probability[0],
                                        prediction=(probability[0] >= .5).astype(np.uint8),
                                        truth=mask[0].astype(np.uint8))
                    exported.append({"sample_id": row["sample_id"], "file": file.name,
                                     "sha256": digest_file(file)})
                offset += 1
    if offset != len(rows):
        raise ValueError("Validation prediction count mismatch")
    if export is not None:
        write_json(export / "manifest.json", exported)
    return evaluator, float(np.mean(losses))


def export_evaluation(path, evaluator, metadata):
    result = {**metadata, **evaluator.summary()}
    write_json(path, result)
    with Path(path).with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluator.rows[0]))
        writer.writeheader()
        writer.writerows(evaluator.rows)
    return result


def run_baseline(config, manifest, root, baseline, device_name, smoke=False, resume=False):
    root = Path(root)
    run = root / (("smoke-" if smoke else "") + baseline)
    if (run / "status.json").exists():
        old = json.loads((run / "status.json").read_text())
        if old["status"] in ("completed", "passed"):
            if old["training_sources"] != training_fingerprints():
                raise ValueError("Completed run uses different training sources")
            if digest_file(run / "best.pth") != old["best_checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
            print(f"CACHED {baseline}; completed stage not retrained", flush=True)
            return old
        if not resume:
            raise FileExistsError("Interrupted run exists; inspect then explicitly resume")
    elif resume:
        resume = False
    if not resume:
        run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = torch.device(device_name)
    settings = config["training"]
    verify_cp2_manifest(manifest)
    verify_reference()
    setup(config["seed"], config["cpu_threads"])
    if torch.get_default_dtype() != torch.float32 or settings["amp"] or settings["precision"] != "float32":
        raise ValueError("Phase P must retain FP32 without AMP")
    if (torch.backends.cuda.matmul.allow_tf32 != settings["matmul_allow_tf32"]
            or torch.backends.cudnn.allow_tf32 != settings["cudnn_allow_tf32"]
            or torch.backends.cudnn.benchmark != settings["cudnn_benchmark"]):
        raise ValueError("Backend precision/performance flags differ from the locked environment")
    code = training_fingerprints()
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()  # Only this process's unused allocator cache.
    environment = environment_record(config, manifest, device)
    run_identity = digest_json({"config": config, "baseline": baseline,
                                "manifest": manifest["fingerprint"], "sources": code})
    if resume:
        if json.loads((run / "identity.json").read_text())["fingerprint"] != run_identity:
            raise ValueError("Config/code/input changed; cannot resume this run")
        if json.loads((run / "environment.json").read_text())["packages"] != environment["packages"]:
            raise ValueError("Environment changed; cannot resume")
    else:
        write_json(run / "identity.json", {"fingerprint": run_identity})
        write_json(run / "config.json", {"baseline": baseline, **config})
        write_json(run / "environment.json", environment)
        write_json(run / "split-manifest.json", manifest)
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model = build_model(config, pretrained=False)
    if sum(p.numel() for p in model.parameters()) != 70_968_939:
        raise ValueError("Unexpected source architecture parameter count")
    model.load_state_dict(initial["state_dict"])
    actual = tensor_fingerprint(model.state_dict())
    if actual != initial["tensor_fingerprint"]:
        raise ValueError("Initial parameters differ from the shared state")
    write_json(run / "initialization_verified.json", {
        "shared_initialization_tensor_sha256": actual, "all_named_tensors_equal": True,
        "initial_weights_file_sha256": digest_file(root / "initial_weights.pth"),
        "split_fingerprint": manifest["fingerprint"],
        "parameter_count": sum(p.numel() for p in model.parameters())})
    del initial
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler_settings = {k: v for k, v in settings["scheduler"].items() if k not in ("type", "monitor")}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_settings)
    objective = BaselineObjective(baseline)
    policy, history, first_epoch = ImageDiceCheckpoint(), [], 0
    if resume:
        state = torch.load(run / "last.pth", map_location=device)
        if state["identity"] != run_identity:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        policy = ImageDiceCheckpoint(**state["policy"])
        history, first_epoch = state["history"], state["completed_epochs"]
        restore_random_state(state["rng"])
        del state
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch_limit = 1 if smoke else settings["max_epochs"]
    finite_steps = 0
    for epoch in range(first_epoch, epoch_limit):
        if policy.bad_epochs >= settings["early_stopping_patience"]:
            break
        setup(config["seed"] + 10007 * epoch, config["cpu_threads"])
        train_loader, training_rows = make_cp2_loader(config, manifest, "train", epoch, smoke)
        valid_loader, valid_rows = make_cp2_loader(config, manifest, "validation", epoch, smoke)
        order_hash = digest_json([r["sample_id"] for r in training_rows])
        input_hash = hashlib.sha256()
        model.train()
        losses, timings = [], []
        epoch_start = time.monotonic()
        previous_encoder = next(model.parameters()).detach().clone() if smoke else None
        used_lr = optimizer.param_groups[0]["lr"]
        for images, truth in train_loader:
            input_hash.update(images.numpy().tobytes())
            input_hash.update(truth.numpy().tobytes())
            sync(device)
            step_start = time.monotonic()
            images, truth = images.to(device), truth.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = objective(logits, truth)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
                raise ValueError("Missing/nonfinite gradients; configuration unchanged")
            optimizer.step()
            finite_steps += 1
            sync(device)
            timings.append(time.monotonic() - step_start)
            losses.append(float(loss.detach()))
        if smoke and torch.equal(previous_encoder, next(model.parameters()).detach()):
            raise ValueError("Smoke optimizer did not update encoder weights")
        evaluator, val_loss = validate_epoch(model, objective, valid_loader, valid_rows, config, device)
        val_dice = evaluator.summary()["all"]["image_based"]["dice"]
        selected = policy.update(val_dice, epoch + 1)
        scheduler.step(val_dice)
        entry = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                 "validation_loss": val_loss, "validation_image_dice": val_dice,
                 "learning_rate_used": used_lr, "learning_rate_next": optimizer.param_groups[0]["lr"],
                 "selected": selected, "bad_epochs": policy.bad_epochs,
                 "epoch_compute_seconds": time.monotonic() - epoch_start,
                 "step_seconds_mean": float(np.mean(timings)),
                 "step_seconds_p95": float(np.percentile(timings, 95)), "steps": len(timings),
                 "training_order_sha256": order_hash, "augmented_input_sha256": input_hash.hexdigest(),
                 "gpu": gpu_snapshot() if device.type == "cuda" else None}
        history.append(entry)
        save_checkpoint(run / "last.pth", {
            "identity": run_identity, "baseline": baseline,
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": random_state(),
            "policy": policy.state_dict(), "history": history, "completed_epochs": epoch + 1})
        if selected:
            link_best(run / "last.pth", run / "best.pth")
        entry["epoch_with_checkpoint_seconds"] = time.monotonic() - epoch_start
        write_json(run / "history.json", history)
        write_json(run / "status.json", {"status": "running", "baseline": baseline,
                                          "completed_epochs": epoch + 1, "selected_epoch": policy.epoch,
                                          "latest": entry, "pid": os.getpid()})
        print(json.dumps({"baseline": baseline, **entry}), flush=True)
    selected = torch.load(run / "best.pth", map_location="cpu")
    if selected["identity"] != run_identity:
        raise ValueError("Selected checkpoint identity mismatch")
    model.load_state_dict(selected["state_dict"])
    valid_loader, valid_rows = make_cp2_loader(config, manifest, "validation", smoke=smoke)
    first_images, _ = next(iter(valid_loader))
    model.eval()
    with torch.no_grad():
        expected = model(first_images.to(device)).sigmoid().cpu()
        next(model.parameters()).add_(1)
    model.load_state_dict(selected["state_dict"])
    with torch.no_grad():
        replay = model(first_images.to(device)).sigmoid().cpu()
    if not torch.equal(expected, replay):
        raise ValueError("Checkpoint reload does not reproduce predictions")
    result = {"status": "passed" if smoke else "completed", "baseline": baseline,
              "diagnostic_smoke_only": smoke, "pid": os.getpid(),
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "completed_epochs": len(history), "selected_epoch": selected["completed_epochs"],
              "selected_validation_image_dice": policy.best,
              "stop_reason": "early_stopping_30_nonimproving_epochs" if policy.bad_epochs >= 30 else "epoch_budget",
              "seconds_this_invocation": time.monotonic() - started,
              "finite_gradient_steps_this_invocation": finite_steps,
              "checkpoint_replay_exact": True, "encoder_update_verified_in_smoke": smoke,
              "best_checkpoint_sha256": digest_file(run / "best.pth"),
              "training_sources": code, "identity": run_identity,
              "shared_initialization_tensor_sha256": actual, "split_fingerprint": manifest["fingerprint"],
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
    write_json(run / "status.json", result)
    return result


def evaluate_locked(config, manifest, root, baseline, device_name):
    run = Path(root) / baseline
    if (run / "final_evaluation.json").exists():
        return json.loads((run / "final_evaluation.json").read_text())
    for name in ("B0", "B1"):
        if json.loads((Path(root) / name / "status.json").read_text())["status"] != "completed":
            raise ValueError("Both baseline training runs must complete before final test evaluation")
    device = torch.device(device_name)
    setup(config["seed"], config["cpu_threads"])
    model = build_model(config, pretrained=False).to(device)
    state = torch.load(run / "best.pth", map_location="cpu")
    model.load_state_dict(state["state_dict"])
    results = {}
    for split in ("validation", "test"):
        destination = run / f"evaluation-{split}.json"
        if destination.exists():
            results[split] = json.loads(destination.read_text())
            continue
        loader, rows = make_cp2_loader(config, manifest, split)
        start = time.monotonic()
        evaluator, loss = validate_epoch(model, BaselineObjective(baseline), loader, rows,
                                         config, device, run / "predictions" / split)
        results[split] = export_evaluation(destination, evaluator, {
            "baseline": baseline, "split": split, "selected_epoch": state["completed_epochs"],
            "checkpoint_sha256": digest_file(run / "best.pth"), "split_fingerprint": manifest["fingerprint"],
            "validation_loss": loss, "seconds": time.monotonic() - start,
            "diagnostic_smoke_only": False,
            "test_caveat": "CP1-exposed source test, cleaned by annotation/RGB audit; not an untouched confirmatory test"})
    write_json(run / "final_evaluation.json", results)
    return results
