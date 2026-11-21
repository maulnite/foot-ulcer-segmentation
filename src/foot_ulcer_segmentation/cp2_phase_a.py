"""Controlled attention screening around frozen phase-P/L training utilities.

The isolated checkpoint loop follows phase L so historical executable sources
stay byte-identical. Training steps, validation, data, loss and persistence reuse
their existing implementations; only model construction and staged gates differ.
"""

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch

from .checkpoint import link_best, random_state, restore_random_state, save_checkpoint
from .common import ROOT, digest_file, digest_json, write_json
from .cp2_attention import VARIANTS, build_attention_model
from .cp2_data import verify_cp2_manifest
from .cp2_phase_l import _verify_precision, phase_l_sources, train_epoch, validate_epoch
from .cp2_protocol import BaselineObjective, ImageDiceCheckpoint
from .cp2_train import export_evaluation, gpu_snapshot, make_cp2_loader, training_fingerprints
from .engine import environment_record, setup
from .reference import verify_reference


EXPERIMENTS = VARIANTS


def read(path):
    return json.loads(Path(path).read_text())


def phase_a_sources():
    result = phase_l_sources()
    for path in ("src/foot_ulcer_segmentation/cp2_attention.py",
                 "src/foot_ulcer_segmentation/cp2_phase_a.py", "scripts/run_cp2_phase_a.py"):
        result[path] = digest_file(ROOT / path)
    return result


class FixedB1Objective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.baseline = BaselineObjective("B1")

    def components(self, logits, truth):
        dice, focal = self.baseline.components(logits, truth)
        total = dice + focal
        return {"dice": dice, "focal": focal, "regional": total, "total": total}

    def forward(self, logits, truth):
        return self.components(logits, truth)["total"]


def prepare(config, directory):
    root = Path(directory)
    settings = config["phase_a"]
    control = ROOT / settings["control_root"]
    old = read(control / "protocol.json")
    for key in ("seed", "split_seed", "cpu_threads", "data", "model", "training", "loss"):
        if config[key] != old[key]:
            raise ValueError(f"Frozen B1 control changed: {key}")
    for key, value in old["evaluation"].items():
        if key != "test_policy" and config["evaluation"][key] != value:
            raise ValueError(f"Evaluator changed: {key}")
    if tuple(settings["experiment_order"]) != EXPERIMENTS:
        raise ValueError("Unregistered architecture/matrix expansion")
    manifest = verify_cp2_manifest(read(control / "data/manifest.json"))
    if manifest["fingerprint"] != settings["control_manifest_fingerprint"]:
        raise ValueError("Membership differs from fixed B1")
    for file, expected in (("initial_weights.pth", settings["control_initial_weights_sha256"]),
                           ("B1/best.pth", settings["control_B1_checkpoint_sha256"])):
        if digest_file(control / file) != expected:
            raise ValueError("Historical checkpoint/initialization changed")
    status = read(control / "B1/status.json")
    if status["status"] != "completed" or status["diagnostic_smoke_only"]:
        raise ValueError("Completed real B1 required")
    if status["training_sources"] != training_fingerprints():
        raise ValueError("B1 training utilities changed")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        if read(root / "protocol.json") != config or read(root / "data/manifest.json") != manifest:
            raise ValueError("Existing protocol/input changed; use a new run")
    else:
        write_json(root / "protocol.json", config)
        write_json(root / "data/manifest.json", manifest)
        temporary = root / "initial_weights.pth.tmp"
        shutil.copyfile(control / "initial_weights.pth", temporary)
        temporary.replace(root / "initial_weights.pth")
        write_json(root / "control_reference.json", {
            "status": "CACHED A0=L0=B1; not retrained", "source_root": settings["control_root"],
            "validation": read(control / "B1/evaluation-validation.json"), "status_record": status})
    if digest_file(root / "initial_weights.pth") != settings["control_initial_weights_sha256"]:
        raise ValueError("Local initial weights changed")
    return manifest


def expansion_decision(candidate, baseline, config):
    if candidate["split"] != "validation" or baseline["split"] != "validation":
        raise ValueError("Expansion cannot inspect source-test")
    rules = config["phase_a"]["expansion_gate"]
    checks = {
        "overall_floor": candidate["all"]["image_based"]["dice"] >= baseline["all"]["image_based"]["dice"] + rules["minimum_overall_dice_vs_B1"],
        "small_floor": candidate["subgroups"]["small"]["image_based"]["dice"] >= baseline["subgroups"]["small"]["image_based"]["dice"] + rules["minimum_small_dice_vs_B1"],
        "miss_limit": candidate["gt_positive"]["total_foreground_misses"] <= baseline["gt_positive"]["total_foreground_misses"] + rules["maximum_extra_total_foreground_misses"],
        "empty_fp_limit": candidate["gt_empty"]["images_with_prediction"] <= baseline["gt_empty"]["images_with_prediction"] + rules["maximum_extra_empty_fp_images"]}
    return {"validation_only": True, "checks": checks, "expand": all(checks.values()),
            "remaining_models": list(EXPERIMENTS[1:]) if all(checks.values()) else [],
            "criterion": "feasibility, not improvement acceptance", "created_utc": datetime.now(timezone.utc).isoformat()}


def select_candidate(reports, baseline, profiles):
    if baseline["split"] != "validation" or any(r["split"] != "validation" for r in reports.values()):
        raise ValueError("Model selection must use validation only")
    decisions = {}
    for name in EXPERIMENTS:
        if name not in reports:
            continue
        report = reports[name]
        checks = {
            "higher_overall_dice": report["all"]["image_based"]["dice"] > baseline["all"]["image_based"]["dice"],
            "small_not_lower": report["subgroups"]["small"]["image_based"]["dice"] >= baseline["subgroups"]["small"]["image_based"]["dice"],
            "boundary_not_lower": report["all"]["image_based"]["boundary_iou"] >= baseline["all"]["image_based"]["boundary_iou"],
            "no_extra_total_miss": report["gt_positive"]["total_foreground_misses"] <= baseline["gt_positive"]["total_foreground_misses"],
            "no_extra_empty_fp": report["gt_empty"]["images_with_prediction"] <= baseline["gt_empty"]["images_with_prediction"],
            "capacity_budget": profiles[name]["active_forward_parameters"] <= 1.01 * profiles["A0"]["active_forward_parameters"]}
        decisions[name] = {"eligible": all(checks.values()), "checks": checks,
                           "validation_image_dice": report["all"]["image_based"]["dice"],
                           "checkpoint_sha256": report["checkpoint_sha256"]}
    eligible = [n for n in EXPERIMENTS if n in decisions and decisions[n]["eligible"]]
    selected = max(eligible, key=lambda n: decisions[n]["validation_image_dice"]) if eligible else "A0"
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "validation_only": True,
            "selected_candidate": selected, "decisions": decisions,
            "unexecuted": [n for n in EXPERIMENTS if n not in reports],
            "caveat": "single-seed exploratory screen, not clinical or global novelty evidence"}


def verify_inputs(config, root, names, smoke=False):
    control = ROOT / config["phase_a"]["control_root"] / ("smoke-B1" if smoke else "B1")
    baseline = read(control / "history.json")
    histories = {name: read(root / (("smoke-" if smoke else "") + name) / "history.json") for name in names}
    proof = []
    for name, history in histories.items():
        for a, b in zip(history, baseline):
            if any(a[k] != b[k] for k in ("training_order_sha256", "augmented_input_sha256")):
                raise ValueError("Actual augmented input/order differs from B1")
        proof.append({"variant": name, "common_B1_epochs": min(len(history), len(baseline))})
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            for a, b in zip(histories[first], histories[second]):
                if any(a[k] != b[k] for k in ("training_order_sha256", "augmented_input_sha256")):
                    raise ValueError("Candidate common-epoch inputs differ")
    return {"status": "passed", "baseline_comparisons": proof,
            "all_candidate_pairwise_common_epoch_inputs_equal": True}


def evaluate_candidate(config, manifest, root, variant, split, device_name):
    run = root / variant
    status = read(run / "status.json")
    if status["status"] != "completed" or status["diagnostic_smoke_only"]:
        raise ValueError("Completed real-data training required")
    if split == "test" and not (root / "selection.json").exists():
        raise ValueError("Persist validation-only selection before source-test")
    destination = run / f"evaluation-{split}.json"
    if destination.exists():
        old = read(destination)
        if old["checkpoint_sha256"] != digest_file(run / "best.pth"):
            raise ValueError("Cached evaluation checkpoint differs")
        return old
    export = run / "predictions" / split
    if export.exists():
        archive = run / "incomplete_exports" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.parent.mkdir(exist_ok=True)
        export.rename(archive)
    setup(config["seed"], config["cpu_threads"])
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model, _ = build_attention_model(config, variant, initial)
    del initial
    state = torch.load(run / "best.pth", map_location="cpu")
    if state["identity"] != status["identity"]:
        raise ValueError("Checkpoint identity differs")
    model.load_state_dict(state["state_dict"])
    del state
    device = torch.device(device_name)
    model.to(device)
    loader, rows = make_cp2_loader(config, manifest, split)
    started = time.monotonic()
    evaluator, parts = validate_epoch(model, FixedB1Objective(), loader, rows, config, device, export)
    return export_evaluation(destination, evaluator, {
        "variant": variant, "split": split, "selected_epoch": status["selected_epoch"],
        "checkpoint_sha256": digest_file(run / "best.pth"), "split_fingerprint": manifest["fingerprint"],
        "components": parts, "seconds": time.monotonic() - started, "diagnostic_smoke_only": False,
        "test_caveat": "CP1/P/L-exposed source-test; descriptive only"})


def run_experiment(config, manifest, root, experiment, device_name, smoke=False, resume=False):
    if experiment not in EXPERIMENTS:
        raise ValueError("A0/B1 is cached, never retrained")
    root = Path(root)
    run = root / (("smoke-" if smoke else "") + experiment)
    sources = phase_a_sources()
    identity = digest_json({"config": config, "architecture": experiment,
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
        write_json(run / "config.json", {"variant": experiment, **config})
        write_json(run / "environment.json", environment)
        write_json(run / "split-manifest.json", manifest)
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    model, initialization = build_attention_model(config, experiment, initial)
    initialization.update({"initial_weights_file_sha256": digest_file(root / "initial_weights.pth"),
                           "split_fingerprint": manifest["fingerprint"]})
    fingerprint = initialization["complete_initial_tensor_sha256"]
    new_before = {name: value.detach().clone() for name, value in model.named_parameters()
                  if name not in initial["state_dict"]} if smoke else {}
    if (run / "initialization_verified.json").exists():
        if read(run / "initialization_verified.json") != initialization:
            raise ValueError("Resume initialization differs")
    else:
        write_json(run / "initialization_verified.json", initialization)
    del initial
    if smoke:
        new_before = {name: value.to(device) for name, value in new_before.items()}
    model = model.to(device)
    objective = FixedB1Objective()
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
    write_json(run / "status.json", {"status": "running", "variant": experiment,
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
        save_checkpoint(run / "last.pth", {"identity": identity, "variant": experiment,
                        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng": random_state(),
                        "policy": policy.state_dict(), "history": history, "completed_epochs": epoch + 1})
        if selected:
            link_best(run / "last.pth", run / "best.pth")
        entry["epoch_with_checkpoint_seconds"] = time.monotonic() - epoch_start
        write_json(run / "history.json", history)
        write_json(run / "status.json", {"status": "running", "variant": experiment,
                   "completed_epochs": epoch + 1, "selected_epoch": policy.epoch,
                   "latest": entry, "pid": os.getpid()})
        print(json.dumps({"variant": experiment, **entry}), flush=True)
    changed_new = [name for name, value in model.named_parameters()
                   if name in new_before and not torch.equal(value.detach(), new_before[name])] if smoke else []
    if smoke:
        required = {"A1": ["local_attention"], "A2": ["local_attention", "fusion_logits"],
                    "A3": ["local_attention", "image_gate"], "A5": ["capacity_block"], "A6": ["cbam"]}[experiment]
        if any(not any(key in name for name in changed_new) for key in required):
            raise ValueError("Smoke did not update each newly introduced component")
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
    result = {"status": "passed" if smoke else "completed", "variant": experiment,
              "diagnostic_smoke_only": smoke, "pid": os.getpid(),
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "completed_epochs": len(history), "selected_epoch": chosen["completed_epochs"],
              "selected_validation_image_dice": policy.best,
              "stop_reason": "early_stopping_30_nonimproving_epochs" if policy.bad_epochs >= 30 else "epoch_budget",
              "seconds_this_invocation": time.monotonic() - started,
              "finite_gradient_steps_this_invocation": finite_steps, "checkpoint_replay_exact": True,
              "encoder_update_verified_in_smoke": smoke, "best_checkpoint_sha256": digest_file(run / "best.pth"),
              "training_sources": sources, "identity": identity,
              "new_parameters_updated_in_smoke": changed_new,
              "source_initial_tensor_sha256": initialization["source_tensor_sha256"],
              "complete_initial_tensor_sha256": fingerprint,
              "split_fingerprint": manifest["fingerprint"],
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
    write_json(run / "status.json", result)
    return result
