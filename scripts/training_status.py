#!/usr/bin/env python3
"""Print local progress for the supervised CP1 baseline without importing torch."""

import argparse
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/runs/cp1-baseline-01")
    args = parser.parse_args()
    run = ROOT / args.run_dir
    launch_path = run.with_name(run.name + ".launch.json")
    launch = json.loads(launch_path.read_text())
    result = {"workflow_status": launch["status"], "run_directory": args.run_dir}
    if (run / "status.json").exists():
        result["training"] = json.loads((run / "status.json").read_text())
    if (run / "history.json").exists():
        history = json.loads((run / "history.json").read_text())
        result["completed_epochs"] = len(history)
        if history:
            result["latest_epoch"] = history[-1]
            result["median_epoch_compute_seconds"] = statistics.median(row["seconds"] for row in history)
    for key in ["supervisor_pid", "child_pid"]:
        try:
            os.kill(launch[key], 0)
            alive = True
            proc = Path(f"/proc/{launch[key]}/stat")
            if proc.exists():
                state = proc.read_text().rsplit(")", 1)[1].split()[0]
                alive = state not in {"Z", "X"}
        except (ProcessLookupError, FileNotFoundError):
            alive = False
        result[key] = {"pid": launch[key], "alive": alive}
    for split in ["validation", "test"]:
        path = run / ("validation.json" if split == "validation" else "evaluation-test.json")
        if path.exists():
            report = json.loads(path.read_text())
            result[split] = {"images": report["images"], "offline_hard_line": report["offline_hard_line"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
