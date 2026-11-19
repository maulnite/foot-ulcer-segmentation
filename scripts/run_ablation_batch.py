#!/usr/bin/env python3
"""Detached sequential training, evaluation, analysis, and result publication."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/ablation/cp1-attention-01.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def gpu_idle(device):
    if not device.startswith("cuda:"):
        raise ValueError("Full ablation batch requires an explicit CUDA GPU index")
    index = device.split(":", 1)[1]
    processes = subprocess.run(["nvidia-smi", f"--id={index}", "--query-compute-apps=pid", "--format=csv,noheader"],
                               check=True, capture_output=True, text=True).stdout.strip()
    used, utilization = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout.strip().split(",")
    return not processes and int(used) < 1024 and int(utilization) < 10


def publish_results(batch, status):
    manifest = json.loads((batch / "publication-manifest.json").read_text())
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if dirty or branch != "main":
        status.update(publication_status="deferred", publication_reason="Working tree changed or current branch is not main")
        return
    destinations = []
    for row in manifest["files"]:
        relative = Path(row["destination"])
        if (relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"reports", "notebooks"}
                or relative.suffix not in {".md", ".csv", ".json", ".ipynb"}):
            raise ValueError("Unexpected publication target")
        source, destination = batch / row["source"], ROOT / relative
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            status.update(publication_status="deferred", publication_reason=f"Existing file differs: {relative}")
            return
        destinations.append((source, destination, str(relative)))
    for source, destination, _ in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    names = [name for _, _, name in destinations]
    with (batch / "publication.log").open("a") as log:
        subprocess.run(["git", "add", "--", *names], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode
        if staged:
            subprocess.run(["git", "commit", "-m", "Report completed CP1 attention branch ablations"],
                           cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = subprocess.run(["git", "-c", "credential.helper=", "-c", "credential.helper=!gh-orie auth git-credential",
                                 "push", "origin", "main"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    status["publication_status"] = "pushed" if result.returncode == 0 else "push_failed"
    status["result_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                             capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-dir", default="outputs/ablations/cp1-attention-01")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    batch = ROOT / args.batch_dir
    batch.mkdir(parents=True, exist_ok=True)
    lock = (batch / "workflow.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    destination = batch / "workflow.json"
    if destination.exists() and not args.resume:
        raise FileExistsError("Workflow already exists; use --resume only after inspecting its status")
    protocol = json.loads(PROTOCOL.read_text())
    state = {"id": protocol["id"], "supervisor_pid": os.getpid(), "device": args.device,
             "started_at_utc": utc(), "status": "starting", "runs": protocol["runs"],
             "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
             "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "publication_status": "pending"}

    def save():
        state["updated_at_utc"] = utc()
        write(destination, state)

    def execute(command, log_path, stage, launch_path=None, launch=None):
        state.update(status=stage, command=command)
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = process.pid
            if launch is not None:
                launch.update(status=stage, child_pid=process.pid)
                write(launch_path, launch)
            save()
            code = process.wait()
        state.pop("child_pid", None)
        if launch is not None:
            launch[stage + "_exit_code"] = code
            write(launch_path, launch)
        if code:
            raise RuntimeError(f"{stage} exited with code {code}; see {log_path.relative_to(ROOT)}")

    try:
        save()
        for row in protocol["runs"]:
            state["current_variant"] = row["variant"]
            run = ROOT / row["run_dir"]
            log = run.with_name(run.name + ".console.log")
            launch_path = run.with_name(run.name + ".launch.json")
            launch = {"supervisor_pid": os.getpid(), "device": args.device, "run_directory": row["run_dir"],
                      "config": row["config"], "started_at_utc": utc(), "log": str(log.relative_to(ROOT)), "status": "queued"}
            run.parent.mkdir(parents=True, exist_ok=True)
            write(launch_path, launch)
            for stage in ["training", "test_evaluation"]:
                completed_training = ((run / "status.json").exists()
                                      and json.loads((run / "status.json").read_text()).get("status") == "completed")
                if stage == "training" and completed_training:
                    continue
                if stage == "test_evaluation" and (ROOT / "outputs/analysis" / run.name / "prediction-manifest.json").exists():
                    continue
                while not gpu_idle(args.device):
                    state["status"] = "waiting_for_gpu"
                    launch["status"] = "waiting_for_gpu"
                    write(launch_path, launch)
                    save()
                    time.sleep(30)
                command = [sys.executable, "-u", "scripts/run_attention_ablation.py",
                           "train" if stage == "training" else "evaluate", "--run-dir", row["run_dir"], "--device", args.device]
                if stage == "training":
                    command += ["--config", row["config"]]
                    if run.exists():
                        if not args.resume or not (run / "last.pth").exists():
                            raise FileExistsError(f"Inspect existing run before restarting: {run}")
                        command.append("--resume")
                execute(command, log, stage, launch_path, launch)
            launch.update(status="completed", finished_at_utc=utc())
            write(launch_path, launch)
        state["current_variant"] = "all"
        execute([str(ROOT / ".venv-analysis/bin/python"), "scripts/build_ablation_analysis.py", "--batch-dir", args.batch_dir],
                batch / "analysis.log", "analysis")
        try:
            publish_results(batch, state)
        except Exception as error:
            state.update(publication_status="failed", publication_error=f"{type(error).__name__}: {error}")
        state.update(status="completed", finished_at_utc=utc())
        save()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=utc())
        save()
        raise


if __name__ == "__main__":
    main()
