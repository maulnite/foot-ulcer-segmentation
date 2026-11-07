"""Bounded CP2 phase P: prepare, smoke, sequential B0/B1, evaluate and report."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foot_ulcer_segmentation.common import digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_data import prepare_cp2_manifest  # noqa: E402
from foot_ulcer_segmentation.cp2_train import (  # noqa: E402
    evaluate_locked, gpu_snapshot, run_baseline, shared_initialization, training_fingerprints,
)


def validate_config(config):
    inherited = json.loads((ROOT / "configs/baseline.json").read_text())
    for key in ("batch_size", "num_workers", "max_epochs", "learning_rate", "weight_decay"):
        if config["training"][key] != inherited["training"][key]:
            raise ValueError(f"Phase-P inherited control changed: {key}")
    expected_model = {**inherited["model"], "activation": None}
    if config["model"] != expected_model:
        raise ValueError("Architecture changed beyond exposing pre-sigmoid logits")
    expected = {"dice_epsilon": 1.0, "dice_beta": 1.0, "focal_alpha": None,
                "focal_gamma": 2.0, "focal_epsilon": 1e-6, "focal_reduction": "mean",
                "focal_normalized": False, "focal_reduced_threshold": None,
                "component_weights": [1.0, 1.0]}
    if any(config["loss"][k] != v for k, v in expected.items()):
        raise ValueError("Loss control changed beyond focal input pairing")
    if config["evaluation"]["prediction_threshold"] != .5 or config["training"]["amp"]:
        raise ValueError("Threshold >=0.5 and no AMP are required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cp2/phase_p.json"))
    parser.add_argument("--stage", choices=["prepare", "smoke", "train", "report"], default="prepare")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / ".phase_p.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads(args.config.read_text())
    validate_config(config)
    frozen = args.root / "protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != config:
            raise ValueError("Protocol changed; choose a new CP2 output directory")
    else:
        write_json(frozen, config)
    manifest = prepare_cp2_manifest(config, args.root / "data")
    print(json.dumps({"stage": args.stage, "manifest": manifest["fingerprint"],
                      "audit": manifest["audit"],
                      "near_duplicate_candidates": len(manifest["near_duplicate_screen"]["candidates"])}), flush=True)
    if args.stage == "prepare":
        return
    if args.stage == "report":
        subprocess.run([str(ROOT / ".venv-analysis/bin/python"), "scripts/report_cp2_phase_p.py",
                        "--root", str(args.root)], check=True)
        return
    shared_initialization(config, args.root)
    if args.stage == "train":
        checks = json.loads((args.root / "pretraining_checks.json").read_text())
        if checks["status"] != "passed" or checks["protocol_fingerprint"] != digest_json(config):
            raise ValueError("Tests/inspection gate not passed for this locked protocol")
        if (checks["manifest_fingerprint"] != manifest["fingerprint"]
                or checks["training_sources"] != training_fingerprints()):
            raise ValueError("Training source or membership changed since pretraining checks")
        for baseline in ("B0", "B1"):
            smoke = json.loads((args.root / f"smoke-{baseline}/status.json").read_text())
            if smoke["status"] != "passed" or not smoke["checkpoint_replay_exact"]:
                raise ValueError("Both real-data smoke tests must pass before full training")
            if smoke["training_sources"] != training_fingerprints():
                raise ValueError("Training sources changed after smoke verification")
    status = args.root / (args.stage + "_workflow.json")
    if status.exists() and json.loads(status.read_text())["status"] == "completed":
        print("CACHED completed workflow; no artifact or checkpoint overwritten", flush=True)
        return
    stop_monitor = threading.Event()

    def monitor_gpu():
        with (args.root / f"{args.stage}_gpu_samples.jsonl").open("a", buffering=1) as stream:
            while not stop_monitor.is_set():
                stream.write(json.dumps(gpu_snapshot()) + "\n")
                stop_monitor.wait(5)

    write_json(status, {"status": "running", "pid": os.getpid(), "stage": args.stage,
                        "started_utc": datetime.now(timezone.utc).isoformat(), "device": args.device})
    try:
        if args.device.startswith("cuda"):
            import torch
            free, _ = torch.cuda.mem_get_info()
            if free < 6 * 1024**3:
                raise RuntimeError("Insufficient currently free VRAM for bounded FP32 baseline; configuration unchanged")
            write_json(args.root / f"{args.stage}_gpu_start.json", gpu_snapshot())
            threading.Thread(target=monitor_gpu, daemon=True).start()
        for baseline in ("B0", "B1"):
            run_baseline(config, manifest, args.root, baseline, args.device,
                         smoke=args.stage == "smoke", resume=args.resume)
        first = "smoke-B0" if args.stage == "smoke" else "B0"
        second = "smoke-B1" if args.stage == "smoke" else "B1"
        controls = []
        for name in (first, second):
            controls.append(json.loads((args.root / name / "initialization_verified.json").read_text()))
        if controls[0] != controls[1]:
            raise ValueError("Baseline initial weights, parameter count or membership differ")
        histories = [json.loads((args.root / name / "history.json").read_text()) for name in (first, second)]
        for a, b in zip(*histories):
            for key in ("training_order_sha256", "augmented_input_sha256"):
                if a[key] != b[key]:
                    raise ValueError("Baseline training order/augmentation differs in a common epoch")
        write_json(args.root / f"{args.stage}_paired_controls.json", {
            "initialization_exact": True, "manifest_exact": True,
            "common_epochs_with_identical_inputs": min(map(len, histories)),
            "all_common_epoch_order_and_augmented_input_hashes_equal": True})
        if args.stage == "train":
            for baseline in ("B0", "B1"):
                evaluate_locked(config, manifest, args.root, baseline, args.device)
            subprocess.run([str(ROOT / ".venv-analysis/bin/python"), "scripts/report_cp2_phase_p.py",
                            "--root", str(args.root)], check=True)
        write_json(status, {"status": "completed", "pid": os.getpid(), "stage": args.stage,
                            "finished_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        write_json(status, {"status": "failed", "stage": args.stage, "pid": os.getpid(),
                            "error_type": type(error).__name__, "message": str(error),
                            "traceback": traceback.format_exc(),
                            "configuration_changed_automatically": False})
        raise
    finally:
        stop_monitor.set()


if __name__ == "__main__":
    main()
