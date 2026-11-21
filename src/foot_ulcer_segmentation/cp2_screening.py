"""Paired CP2 screening loop for newly registered experiments only.

Frozen historical trainers remain unchanged. Data, step implementations,
checkpoint persistence, precision and evaluator are reused from prior phases.
"""

import json
import os
import time

import torch

from .checkpoint import link_best, random_state, restore_random_state, save_checkpoint
from .common import digest_file, digest_json, write_json
from .cp2_data import verify_cp2_manifest
from .cp2_phase_l import _verify_precision, train_epoch, validate_epoch
from .cp2_phase_n import checkpoint_replay, read, utc
from .cp2_protocol import ImageDiceCheckpoint
from .cp2_train import gpu_snapshot, make_cp2_loader
from .engine import environment_record, setup
from .reference import verify_reference


class CheckedAdam(torch.optim.Adam):
    """The normal Adam update, with one-time actual-step diagnostics."""
    def __init__(self, model, checked_parameters, evidence_path, **kwargs):
        super().__init__(model.parameters(), **kwargs)
        self.checked_parameters = checked_parameters
        self.evidence_path = evidence_path
        self.first_step_checks = None

    def step(self, closure=None):
        if self.first_step_checks is not None:
            return super().step(closure)
        before = {name: p.detach().clone() for name, p in self.checked_parameters.items()}
        for name, parameter in self.checked_parameters.items():
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
                raise ValueError(f"Missing, nonfinite or zero actual first-step gradient: {name}")
        result = super().step(closure)
        updates = {name: not torch.equal(before[name], p.detach())
                   for name, p in self.checked_parameters.items()}
        if not all(updates.values()):
            raise ValueError(f"First real optimizer step did not update components: {updates}")
        self.first_step_checks = {"real_training_step": 1, "finite_nonzero_gradients": True,
                                  "updated_parameters": updates, "separate_smoke_training": False}
        if self.evidence_path is not None:
            write_json(self.evidence_path, self.first_step_checks)
        return result


def fit(config, manifest, directory, model, objective, proof, code, controls,
        checked_names, device_name, resume=False):
    """One full fit, paired against cached control input bytes on common epochs."""
    run = directory
    identity = digest_json({"config": config, "manifest": manifest["fingerprint"], "sources": code})
    if (run / "status.json").exists():
        previous = read(run / "status.json")
        if previous["status"] == "completed":
            if previous["identity"] != identity or digest_file(run / "best.pth") != previous["best_checkpoint_sha256"]:
                raise ValueError("Completed checkpoint/source differs")
            return previous
        if not resume:
            raise FileExistsError("Inspect interrupted training before explicit --resume")
    elif run.exists():
        raise FileExistsError("Partial initialization preserved; never overwrite")
    else:
        run.mkdir()
        resume = False
    started = time.monotonic()
    device = torch.device(device_name)
    settings = config["training"]
    setup(config["seed"], config["cpu_threads"])
    _verify_precision(config)
    verify_cp2_manifest(manifest)
    verify_reference()
    environment = environment_record(config, manifest, device)
    if resume:
        if read(run / "identity.json")["fingerprint"] != identity:
            raise ValueError("Changed code/config/input")
        if read(run / "environment.json")["packages"] != environment["packages"]:
            raise ValueError("Changed environment")
        if read(run / "initialization_verified.json") != proof:
            raise ValueError("Initial source/new tensors changed")
    else:
        write_json(run / "identity.json", {"fingerprint": identity})
        write_json(run / "config.json", config)
        write_json(run / "environment.json", environment)
        write_json(run / "split-manifest.json", manifest)
        write_json(run / "initialization_verified.json", proof)
    model.to(device)
    parameters = dict(model.named_parameters())
    checked_parameters = {label: parameters[name] for label, name in checked_names.items()}
    optimizer = CheckedAdam(model, checked_parameters, run / "first_step_checks.json",
                            lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{
        k: v for k, v in settings["scheduler"].items() if k not in ("type", "monitor")})
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
        persisted_history = read(run / "history.json")
        if len(persisted_history) == first_epoch:
            # The history file adds post-save timing. Restore only verified rows;
            # scientific values must agree with the authoritative checkpoint.
            if any(any(saved[key] != complete[key] for key in saved)
                   for saved, complete in zip(history, persisted_history)):
                raise ValueError("History file differs from checkpoint scientific history")
            history = persisted_history
        restore_random_state(state["rng"])
        del state
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    write_json(run / "status.json", {"status": "running", "pid": os.getpid(),
               "variant": run.name, "completed_epochs": first_epoch, "started_utc": utc()})
    for epoch in range(first_epoch, settings["max_epochs"]):
        if policy.bad_epochs >= settings["early_stopping_patience"]:
            break
        setup(config["seed"] + 10007 * epoch, config["cpu_threads"])
        loader, rows = make_cp2_loader(config, manifest, "train", epoch)
        valid_loader, valid_rows = make_cp2_loader(config, manifest, "validation", epoch)
        begin = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        training = train_epoch(model, objective, optimizer, loader, device)
        evaluator, parts = validate_epoch(model, objective, valid_loader, valid_rows, config, device)
        metric = evaluator.summary()["all"]["image_based"]["dice"]
        selected = policy.update(metric, epoch + 1)
        scheduler.step(metric)
        entry = {"epoch": epoch + 1, "train_loss": training["components"]["total"],
                 "validation_loss": parts["total"], "train_components": training["components"],
                 "validation_components": parts, "validation_image_dice": metric,
                 "learning_rate_used": lr, "learning_rate_next": optimizer.param_groups[0]["lr"],
                 "selected": selected, "bad_epochs": policy.bad_epochs, "steps": training["steps"],
                 "step_seconds_mean": training["step_seconds_mean"], "step_seconds_p95": training["step_seconds_p95"],
                 "training_order_sha256": digest_json([r["sample_id"] for r in rows]),
                 "augmented_input_sha256": training["augmented_input_sha256"],
                 "epoch_compute_seconds": time.monotonic() - begin,
                 "gpu": gpu_snapshot() if device.type == "cuda" else None}
        for name, control in controls.items():
            if epoch < len(control) and any(entry[k] != control[epoch][k]
                                           for k in ("training_order_sha256", "augmented_input_sha256")):
                raise ValueError(f"Actual sample order/augmented bytes differ from {name}")
        history.append(entry)
        save_checkpoint(run / "last.pth", {
            "identity": identity, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": random_state(), "policy": policy.state_dict(),
            "history": history, "completed_epochs": epoch + 1, "first_step_checks": optimizer.first_step_checks})
        if selected:
            link_best(run / "last.pth", run / "best.pth")
        entry["epoch_with_checkpoint_seconds"] = time.monotonic() - begin
        write_json(run / "history.json", history)
        write_json(run / "status.json", {"status": "running", "pid": os.getpid(), "variant": run.name,
                   "completed_epochs": epoch + 1, "selected_epoch": policy.epoch, "latest": entry})
        print(json.dumps({"variant": run.name, **entry}), flush=True)
    chosen = torch.load(run / "best.pth", map_location="cpu")
    if chosen["identity"] != identity:
        raise ValueError("Best checkpoint identity differs")
    loader, _ = make_cp2_loader(config, manifest, "validation")
    images = next(iter(loader))[0]
    replay = checkpoint_replay(model, chosen, images, device)
    replay = {"primary_and_auxiliary_bit_exact": replay["primary_and_presence_bit_exact"]}
    with torch.no_grad():
        with_head = model(images.to(device)).main_logits
        without_head = model(images.to(device), auxiliary=False).main_logits
    if not torch.equal(with_head, without_head):
        raise ValueError("Removing trained boundary head changes primary output")
    replay["head_removal_primary_bit_exact"] = True
    result = {"status": "completed", "variant": run.name, "seed": config["seed"], "pid": os.getpid(),
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
