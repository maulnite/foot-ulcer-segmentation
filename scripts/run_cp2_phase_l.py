"""Sequential CP2 L1–L4 screen with cached B1, immutable controls and test gate."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_l import (  # noqa: E402
    EXPERIMENTS, evaluate_candidate, phase_l_sources, prepare, read, run_experiment,
    select_candidate, verify_paired_inputs,
)
from foot_ulcer_segmentation.cp2_train import gpu_snapshot  # noqa: E402


def require_pretraining_checks(config, root):
    gate = read(root / "pretraining_checks.json")
    if (gate["status"] != "passed" or gate["protocol_fingerprint"] != digest_json(config)
            or gate["training_sources"] != phase_l_sources()):
        raise ValueError("Tests and protocol/source preflight must pass before real training")
    for name in EXPERIMENTS:
        smoke = read(root / f"smoke-{name}/status.json")
        if (smoke["status"] != "passed" or not smoke["checkpoint_replay_exact"]
                or smoke["training_sources"] != phase_l_sources()):
            raise ValueError(f"Actual real-data {name} smoke gate not passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cp2/phase_l.json"))
    parser.add_argument("--stage", choices=["prepare", "smoke", "train", "report", "status"], default="prepare")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    root = args.root.resolve()
    if args.stage == "status":
        result = {"root": str(args.root), "stages": {}, "experiments": {}}
        for stage in ("smoke", "train"):
            path = root / f"{stage}_workflow.json"
            result["stages"][stage] = read(path) if path.exists() else {"status": "not_started"}
        for name in EXPERIMENTS:
            path = root / name / "status.json"
            result["experiments"][name] = read(path) if path.exists() else {"status": "not_started"}
        print(json.dumps(result, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".phase_l.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(args.config)
    manifest = prepare(config, root)
    if args.stage == "prepare":
        print(json.dumps({"status": "prepared", "manifest": manifest["fingerprint"],
                          "counts": {s: len(rows) for s, rows in manifest["splits"].items()},
                          "control": "CACHED L0 / B1; no retraining"}))
        return
    if args.stage == "report":
        subprocess.run([str(ROOT / ".venv-analysis/bin/python"), "scripts/report_cp2_phase_l.py",
                        "--root", str(root)], check=True)
        return
    if args.stage == "train":
        require_pretraining_checks(config, root)
    status = root / f"{args.stage}_workflow.json"
    if status.exists() and read(status)["status"] == "completed":
        print("CACHED completed workflow; no completed artifact overwritten")
        return
    snapshot = root / f"{args.stage}_code_snapshot"
    if not snapshot.exists():
        snapshot.mkdir()
        for relative in phase_l_sources():
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, path)
    stop = threading.Event()

    def monitor_gpu():
        with (root / f"{args.stage}_gpu_samples.jsonl").open("a", buffering=1) as stream:
            while not stop.is_set():
                stream.write(json.dumps(gpu_snapshot()) + "\n")
                stop.wait(5)

    write_json(status, {"status": "running", "stage": args.stage, "pid": os.getpid(),
                        "started_utc": datetime.now(timezone.utc).isoformat(), "device": args.device})
    try:
        if args.device.startswith("cuda"):
            import torch
            free, _ = torch.cuda.mem_get_info()
            if free < 6 * 1024**3:
                raise RuntimeError("Insufficient currently free VRAM; no batch/precision fallback")
            write_json(root / f"{args.stage}_gpu_start.json", gpu_snapshot())
            threading.Thread(target=monitor_gpu, daemon=True).start()
        for name in EXPERIMENTS:
            run_experiment(config, manifest, root, name, args.device,
                           smoke=args.stage == "smoke", resume=args.resume)
        controls = verify_paired_inputs(config, root, smoke=args.stage == "smoke")
        write_json(root / f"{args.stage}_paired_controls.json", controls)
        if args.stage == "train":
            validation = {name: evaluate_candidate(config, manifest, root, name, "validation", args.device)
                          for name in EXPERIMENTS}
            selection_path = root / "selection.json"
            control = read(root / "control_reference.json")["validation"]
            if not selection_path.exists():
                write_json(selection_path, select_candidate(validation, control))
            elif read(selection_path) != {**select_candidate(validation, control),
                                           "created_utc": read(selection_path)["created_utc"]}:
                raise ValueError("Persisted validation-only selection changed")
            for name in EXPERIMENTS:
                evaluate_candidate(config, manifest, root, name, "test", args.device)
                write_json(root / name / "final_evaluation.json", {
                    split: read(root / name / f"evaluation-{split}.json") for split in ("validation", "test")})
        write_json(status, {"status": "completed", "stage": args.stage, "pid": os.getpid(),
                            "finished_utc": datetime.now(timezone.utc).isoformat(),
                            "note": "Report and final QA are separate CPU stages; training is genuinely complete"})
    except Exception:
        failure = {"status": "failed", "stage": args.stage, "pid": os.getpid(),
                   "utc": datetime.now(timezone.utc).isoformat(), "traceback": traceback.format_exc()}
        write_json(status, failure)
        print(json.dumps(failure), flush=True)
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
