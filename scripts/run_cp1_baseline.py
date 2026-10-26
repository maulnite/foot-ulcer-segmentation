#!/usr/bin/env python3
"""Supervise one baseline training run and its final held-out test evaluation.

Designed to be launched in a detached session. Never retune against test results.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--run-dir", default="outputs/runs/cp1-baseline-01")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = ROOT / args.run_dir
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}")
    run.parent.mkdir(parents=True, exist_ok=True)
    metadata = run.with_name(run.name + ".launch.json")
    log_path = run.with_name(run.name + ".console.log")
    if metadata.exists():
        raise FileExistsError(f"Launch record already exists: {metadata}")
    status = {"supervisor_pid": os.getpid(), "device": args.device,
              "run_directory": args.run_dir, "config": args.config,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "log": str(log_path.relative_to(ROOT)), "status": "starting",
              "runtime_environment": {key: os.environ.get(key) for key in
                                      ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}}

    def save_status():
        temporary = metadata.with_name(metadata.name + ".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(metadata)

    save_status()
    with log_path.open("x") as log:
        for stage, command in [
            ("training", [sys.executable, "-u", "-m", "foot_ulcer_segmentation", "train",
                          "--config", args.config, "--device", args.device, "--run-dir", args.run_dir]),
            ("test_evaluation", [sys.executable, "-u", "-m", "foot_ulcer_segmentation", "evaluate",
                                 "--run-dir", args.run_dir, "--split", "test", "--device", args.device]),
        ]:
            status.update(status=stage, command=command)
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = process.pid
            save_status()
            return_code = process.wait()
            status[stage + "_exit_code"] = return_code
            if return_code:
                status.update(status="failed", failed_stage=stage,
                              finished_at_utc=datetime.now(timezone.utc).isoformat())
                save_status()
                raise SystemExit(return_code)
    status.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
    save_status()


if __name__ == "__main__":
    main()
