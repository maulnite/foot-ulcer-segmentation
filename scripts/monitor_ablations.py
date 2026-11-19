#!/usr/bin/env python3
"""Record read-only ablation health checks every 30 minutes until completion."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/ablations/cp1-attention-01"
TERMINAL = {"completed", "failed"}


def read(path):
    # Training status can be replaced while it is being read.
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.05)


def process_alive(pid, script=None):
    if not pid:
        return False
    directory = Path("/proc") / str(pid)
    try:
        state = (directory / "stat").read_text().rsplit(")", 1)[1].split()[0]
        command = (directory / "cmdline").read_bytes().split(b"\0")
        return state != "Z" and (script is None or any(Path(arg.decode()).name == script for arg in command if arg))
    except (FileNotFoundError, ProcessLookupError):
        return False


def check(interval):
    now = datetime.now(timezone.utc)
    workflow = read(BATCH / "workflow.json")
    supervisor_alive = process_alive(workflow.get("supervisor_pid"), "run_ablation_batch.py")
    if not supervisor_alive:
        workflow = read(BATCH / "workflow.json")
    record = {"checked_at_utc": now.isoformat(), "interval_seconds": interval,
              "workflow_status": workflow["status"], "current_variant": workflow.get("current_variant"),
              "publication_status": workflow.get("publication_status"),
              "supervisor_alive": supervisor_alive,
              "child_alive": process_alive(workflow.get("child_pid")), "runs": [], "alerts": []}
    if workflow.get("error"):
        record["alerts"].append(workflow["error"])
    if workflow["status"] not in TERMINAL and not supervisor_alive:
        record["alerts"].append("Workflow is unfinished but its supervisor is not running")
    for declaration in workflow["runs"]:
        run = ROOT / declaration["run_dir"]
        item = {"variant": declaration["variant"], "status": "queued", "completed_epochs": 0}
        if (run / "status.json").exists():
            status = read(run / "status.json")
            for key in ["status", "completed_epochs", "max_epochs", "selected_epoch", "validation_loss",
                        "validation_iou", "early_stopping_count", "updated_at_utc", "seconds"]:
                if key in status:
                    item[key] = status[key]
            if (run / "history.json").exists():
                history = read(run / "history.json")
                item["completed_epochs"] = len(history)
                if history:
                    item["latest_validation"] = history[-1]["validation"]
            updated = status.get("updated_at_utc", status.get("started_at_utc"))
            if updated and status["status"] == "running" and workflow["status"] == "training":
                age = (now - datetime.fromisoformat(updated)).total_seconds()
                if age > max(interval, 1800):
                    record["alerts"].append(f"{declaration['variant']}: no epoch status update for {age/60:.1f} minutes")
        if (run / "failure.json").exists():
            record["alerts"].append(f"{declaration['variant']}: {read(run / 'failure.json').get('error')}")
        log = run.with_name(run.name + ".console.log")
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 32768))
                if b"Traceback (most recent call last)" in stream.read():
                    record["alerts"].append(f"{declaration['variant']}: traceback found in the recent log")
        record["runs"].append(item)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=True)
        record["gpu_index_memory_mib_utilization_percent"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        record["gpu_check_error"] = str(error)
    record["stop_monitor"] = workflow["status"] in TERMINAL or not supervisor_alive
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--output-dir", default="outputs/ablations/cp1-attention-01/monitor")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 1:
        parser.error("interval must be positive")
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic()
    while True:
        try:
            record = check(args.interval_seconds)
        except (OSError, KeyError, ValueError) as error:
            record = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                      "interval_seconds": args.interval_seconds,
                      "alerts": [f"Monitor read failed: {type(error).__name__}: {error}"], "stop_monitor": False}
        record["next_check_at_utc"] = (datetime.fromtimestamp(time.time() + max(0, deadline + args.interval_seconds - time.monotonic()), timezone.utc).isoformat()
                                       if not record["stop_monitor"] and not args.once else None)
        temporary = output / "latest.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output / "latest.json")
        with (output / "checks.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.once or record["stop_monitor"]:
            return
        deadline += args.interval_seconds
        time.sleep(max(0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()
