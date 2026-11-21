"""Isolated one-seed presence experiment, reusing frozen CP2 training utilities."""

from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import shutil
import time

import torch

from .checkpoint import link_best, random_state, restore_random_state, save_checkpoint
from .common import ROOT, digest_file, digest_json, write_json
from .cp2_data import verify_cp2_manifest
from .cp2_phase_l import _verify_precision, phase_l_sources, train_epoch, validate_epoch
from .cp2_presence import FirstStepAdam, PresenceObjective, build_presence_model
from .cp2_protocol import ImageDiceCheckpoint
from .cp2_train import export_evaluation, gpu_snapshot, make_cp2_loader, training_fingerprints
from .engine import environment_record, setup
from .reference import verify_reference


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def sources():
    result = phase_l_sources()
    for name in ("src/foot_ulcer_segmentation/cp2_presence.py",
                 "src/foot_ulcer_segmentation/cp2_phase_n.py",
                 "src/foot_ulcer_segmentation/cp2_attention.py",
                 "scripts/run_cp2_phase_n.py", "scripts/report_cp2_phase_n.py",
                 "scripts/validate_cp2_phase_a.py", "scripts/validate_cp2_phase_l.py"):
        result[name] = digest_file(ROOT / name)
    return result


def preservation(root):
    inventory = read(root / "protected_before.json")
    changed = [name for name, expected in inventory.items()
               if not (ROOT / name).is_file() or digest_file(ROOT / name) != expected]
    if changed:
        raise ValueError(f"Protected historical/user assets changed: {changed}")
    return {"files": len(inventory), "unchanged": True}


def prepare(root, config):
    settings = config["phase_n"]
    control = ROOT / settings["control_root"]
    old = read(control / "protocol.json")
    for key in ("seed", "split_seed", "cpu_threads", "data", "model", "training", "loss"):
        if config[key] != old[key]:
            raise ValueError(f"Fixed B1 protocol changed: {key}")
    for key, value in old["evaluation"].items():
        if key != "test_policy" and config["evaluation"][key] != value:
            raise ValueError(f"Fixed evaluator changed: {key}")
    if settings["experiment_order"] != ["N1"] or settings["standalone_smoke_training"]:
        raise ValueError("Only one N1 full training; no extra smoke/combinations")
    if config["seed"] != 42 or settings["auxiliary_weight"] != .1:
        raise ValueError("Unregistered seed or objective weight")
    manifest = verify_cp2_manifest(read(control / "data/manifest.json"))
    if manifest["fingerprint"] != settings["control_manifest_fingerprint"]:
        raise ValueError("Changed membership")
    status = read(control / "B1/status.json")
    if status["status"] != "completed" or status["diagnostic_smoke_only"]:
        raise ValueError("Completed real B1 control required")
    if status["training_sources"] != training_fingerprints():
        raise ValueError("Frozen B1 training utilities changed")
    for file, expected in (("initial_weights.pth", settings["control_initial_weights_sha256"]),
                           ("B1/best.pth", settings["control_B1_checkpoint_sha256"])):
        if digest_file(control / file) != expected:
            raise ValueError("Historical initialization/checkpoint changed")
    if (root / "protocol.json").exists():
        if read(root / "protocol.json") != config or read(root / "data/manifest.json") != manifest:
            raise ValueError("Preserve existing run; use a new root")
        if digest_file(root / "initial_weights.pth") != settings["control_initial_weights_sha256"]:
            raise ValueError("Local initial weights changed")
        return manifest
    if not (root / "protected_before.json").exists():
        raise ValueError("Save pre-edit preservation inventory before preparing")
    write_json(root / "protocol.json", config)
    write_json(root / "data/manifest.json", manifest)
    temporary = root / "initial_weights.pth.tmp"
    shutil.copyfile(control / "initial_weights.pth", temporary)
    temporary.replace(root / "initial_weights.pth")
    write_json(root / "control_reference.json", {
        "status": "CACHED N0=B1 seed42; never retrained",
        "source_root": settings["control_root"], "status_record": status,
        "validation": read(control / "B1/evaluation-validation.json"),
        "history_sha256": digest_file(control / "B1/history.json"),
    })
    counts = {split: {"images": len(rows), "positive": sum(r["gt_pixels"] > 0 for r in rows),
                      "empty": sum(r["gt_pixels"] == 0 for r in rows)}
              for split, rows in manifest["splits"].items()}
    write_json(root / "dataset_summary.json", {
        "splits": counts, "manifest_fingerprint": manifest["fingerprint"],
        "participant_ids_available": False, "duplicate_group_overlap": 0,
        "training_target": "presence recomputed from actual augmented mask; not static filename metadata",
    })
    return manifest


def check_pairing(root, config):
    baseline = read(ROOT / config["phase_n"]["control_root"] / "B1/history.json")
    history = read(root / "N1/history.json")
    for candidate, control in zip(history, baseline):
        for key in ("training_order_sha256", "augmented_input_sha256"):
            if candidate[key] != control[key]:
                raise ValueError(f"Actual B1/N1 training inputs differ: epoch {candidate['epoch']} {key}")
    return {"status": "passed", "common_epochs": min(len(history), len(baseline)),
            "actual_order_and_augmented_bytes_equal": True}


def checkpoint_replay(model, checkpoint, images, device):
    """Strict reload check of both outputs; no optimizer step or new observations."""
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        expected = model(images.to(device))
        expected = tuple(value.detach().cpu().clone() for value in expected)
        next(model.parameters()).add_(1)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        actual = model(images.to(device))
    if not all(torch.equal(a, b.detach().cpu()) for a, b in zip(expected, actual)):
        raise ValueError("Checkpoint reload prediction differs")
    return {"primary_and_presence_bit_exact": True}


def train(root, config, manifest, device_name, resume=False):
    run = root / "N1"
    code = sources()
    identity = digest_json({"config": config, "manifest": manifest["fingerprint"], "sources": code})
    if (run / "status.json").exists():
        previous = read(run / "status.json")
        if previous["status"] == "completed":
            if previous["identity"] != identity or digest_file(run / "best.pth") != previous["best_checkpoint_sha256"]:
                raise ValueError("Completed checkpoint/source differs")
            return previous
        if not resume:
            raise FileExistsError("Interrupted training requires inspection and explicit --resume")
    elif run.exists():
        raise FileExistsError("Partial initialization preserved; no silent overwrite")
    else:
        resume = False
        run.mkdir()
    started = time.monotonic()
    device = torch.device(device_name)
    settings = config["training"]
    setup(config["seed"], config["cpu_threads"])
    _verify_precision(config)
    verify_reference()
    verify_cp2_manifest(manifest)
    environment = environment_record(config, manifest, device)
    if resume:
        if read(run / "identity.json")["fingerprint"] != identity:
            raise ValueError("Config/code/input differs from interrupted training")
        if read(run / "environment.json")["packages"] != environment["packages"]:
            raise ValueError("Environment changed")
    else:
        write_json(run / "identity.json", {"fingerprint": identity})
        write_json(run / "config.json", config)
        write_json(run / "environment.json", environment)
        write_json(run / "split-manifest.json", manifest)
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model, proof = build_presence_model(config, initial)
    del initial
    if resume:
        if read(run / "initialization_verified.json") != proof:
            raise ValueError("Initialization proof differs")
    else:
        write_json(run / "initialization_verified.json", proof)
    model.to(device)
    objective = PresenceObjective(config["phase_n"]["auxiliary_weight"])
    optimizer = FirstStepAdam(model, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{
        key: value for key, value in settings["scheduler"].items() if key not in ("type", "monitor")})
    policy, history, first_epoch = ImageDiceCheckpoint(), [], 0
    if resume:
        state = torch.load(run / "last.pth", map_location=device)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity differs")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        optimizer.first_step_checks = state["first_step_checks"]
        scheduler.load_state_dict(state["scheduler"])
        policy = ImageDiceCheckpoint(**state["policy"])
        history, first_epoch = state["history"], state["completed_epochs"]
        restore_random_state(state["rng"])
        del state
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline_history = read(ROOT / config["phase_n"]["control_root"] / "B1/history.json")
    write_json(run / "status.json", {"status": "running", "variant": "N1", "pid": os.getpid(),
                                     "completed_epochs": first_epoch, "started_utc": utc()})
    for epoch in range(first_epoch, settings["max_epochs"]):
        if policy.bad_epochs >= settings["early_stopping_patience"]:
            break
        setup(config["seed"] + 10007 * epoch, config["cpu_threads"])
        loader, rows = make_cp2_loader(config, manifest, "train", epoch)
        valid_loader, valid_rows = make_cp2_loader(config, manifest, "validation", epoch)
        epoch_start = time.monotonic()
        learning_rate = optimizer.param_groups[0]["lr"]
        training = train_epoch(model, objective, optimizer, loader, device)
        if not (run / "first_step_checks.json").exists():
            write_json(run / "first_step_checks.json", optimizer.first_step_checks)
        evaluator, parts = validate_epoch(model, objective, valid_loader, valid_rows, config, device)
        metric = evaluator.summary()["all"]["image_based"]["dice"]
        selected = policy.update(metric, epoch + 1)
        scheduler.step(metric)
        entry = {"epoch": epoch + 1, "train_loss": training["components"]["total"],
                 "validation_loss": parts["total"], "train_components": training["components"],
                 "validation_components": parts, "validation_image_dice": metric,
                 "learning_rate_used": learning_rate, "learning_rate_next": optimizer.param_groups[0]["lr"],
                 "selected": selected, "bad_epochs": policy.bad_epochs,
                 "steps": training["steps"], "step_seconds_mean": training["step_seconds_mean"],
                 "step_seconds_p95": training["step_seconds_p95"],
                 "training_order_sha256": digest_json([r["sample_id"] for r in rows]),
                 "augmented_input_sha256": training["augmented_input_sha256"],
                 "epoch_compute_seconds": time.monotonic() - epoch_start,
                 "gpu": gpu_snapshot() if device.type == "cuda" else None}
        if epoch < len(baseline_history):
            if any(entry[key] != baseline_history[epoch][key]
                   for key in ("training_order_sha256", "augmented_input_sha256")):
                raise ValueError("Actual input/order changed relative to B1; stop instead of changing settings")
        history.append(entry)
        save_checkpoint(run / "last.pth", {
            "identity": identity, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": random_state(), "policy": policy.state_dict(),
            "history": history, "completed_epochs": epoch + 1, "first_step_checks": optimizer.first_step_checks})
        if selected:
            link_best(run / "last.pth", run / "best.pth")
        entry["epoch_with_checkpoint_seconds"] = time.monotonic() - epoch_start
        write_json(run / "history.json", history)
        write_json(run / "status.json", {"status": "running", "variant": "N1", "pid": os.getpid(),
                   "completed_epochs": epoch + 1, "selected_epoch": policy.epoch, "latest": entry})
        print(json.dumps({"variant": "N1", **entry}), flush=True)
    chosen = torch.load(run / "best.pth", map_location="cpu")
    if chosen["identity"] != identity:
        raise ValueError("Best checkpoint identity differs")
    loader, _ = make_cp2_loader(config, manifest, "validation")
    replay = checkpoint_replay(model, chosen, next(iter(loader))[0], device)
    result = {"status": "completed", "variant": "N1", "seed": 42, "pid": os.getpid(),
              "diagnostic_smoke_only": False, "standalone_smoke_training": False,
              "completed_epochs": len(history), "selected_epoch": chosen["completed_epochs"],
              "selected_validation_image_dice": policy.best, "finished_utc": utc(),
              "seconds_this_invocation": time.monotonic() - started,
              "stop_reason": "early_stopping_30_nonimproving_epochs" if policy.bad_epochs >= 30 else "epoch_budget",
              "identity": identity, "training_sources": code, "split_fingerprint": manifest["fingerprint"],
              "best_checkpoint_sha256": digest_file(run / "best.pth"), "checkpoint_replay": replay,
              "first_step_checks": optimizer.first_step_checks,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
    write_json(run / "status.json", result)
    return result


def evaluate(root, config, manifest, split, device_name):
    run = root / "N1"
    status = read(run / "status.json")
    if status["status"] != "completed":
        raise ValueError("Completed real full training required")
    if split == "test" and not (root / "selection.json").exists():
        raise ValueError("Persist validation-only selection before new test predictions")
    output = run / f"evaluation-{split}.json"
    if output.exists():
        previous = read(output)
        if previous["checkpoint_sha256"] != digest_file(run / "best.pth"):
            raise ValueError("Completed evaluation checkpoint differs")
        return previous
    setup(config["seed"], config["cpu_threads"])
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model, _ = build_presence_model(config, initial)
    del initial
    checkpoint = torch.load(run / "best.pth", map_location="cpu")
    if checkpoint["identity"] != status["identity"]:
        raise ValueError("Evaluation identity differs")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    device = torch.device(device_name)
    model.to(device)
    captured = []

    def capture(module, inputs, output):
        captured.extend(output.detach().sigmoid().cpu().flatten().tolist())

    handle = model.presence_head.register_forward_hook(capture)
    loader, rows = make_cp2_loader(config, manifest, split)
    started = time.monotonic()
    try:
        evaluator, parts = validate_epoch(model, PresenceObjective(), loader, rows, config, device,
                                         run / "predictions" / split)
    finally:
        handle.remove()
    if len(captured) != len(rows):
        raise ValueError("Presence/image count mismatch")
    records = [{"sample_id": row["sample_id"], "ground_truth_presence": int(row["gt_pixels"] > 0),
                "presence_probability": probability, "predicted_presence": int(probability >= .5),
                "diagnostic_only_not_mask_gate": True} for row, probability in zip(rows, captured)]
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for row in records:
        key = ("tp" if row["predicted_presence"] else "fn") if row["ground_truth_presence"] else (
            "fp" if row["predicted_presence"] else "tn")
        counts[key] += 1
    with (run / f"presence-{split}.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    positive_recall = counts["tp"] / (counts["tp"] + counts["fn"])
    negative_recall = counts["tn"] / (counts["tn"] + counts["fp"])
    write_json(run / f"presence-{split}.json", {
        "threshold": .5, "counts": counts, "images": len(rows),
        "positive_recall": positive_recall, "negative_recall": negative_recall,
        "balanced_accuracy": (positive_recall + negative_recall) / 2,
        "no_mask_modulation": True, "clinical_diagnosis": False})
    return export_evaluation(output, evaluator, {
        "variant": "N1", "split": split, "selected_epoch": status["selected_epoch"],
        "checkpoint_sha256": digest_file(run / "best.pth"), "split_fingerprint": manifest["fingerprint"],
        "components": parts, "seconds": time.monotonic() - started,
        "diagnostic_smoke_only": False, "test_caveat": "previously exposed source-test; descriptive only"})
