"""CPU watcher; run post-training diagnostics/report/QA sequentially, then stop."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / ".completion.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "completion_checks.json").exists():
        print("CACHED post-training checks; manual review evidence remains separate")
        return
    try:
        while True:
            state = read(root / "train_workflow.json")
            if state["status"] == "failed":
                raise RuntimeError("Training workflow failed; inspect persisted trace; no silent config fallback")
            if state["status"] == "completed":
                break
            if not args.wait:
                raise RuntimeError("Training has not completed")
            progress = {}
            for name in ("H1", "H2"):
                path = root / name / "status.json"
                if path.exists():
                    item = read(path)
                    progress[name] = {k:item.get(k) for k in ("status", "completed_epochs", "selected_epoch")}
            pid = read(root / "launch.json")["pid"]
            process = Path(f"/proc/{pid}/stat")
            if not process.exists() or process.read_text().split()[2] == "Z":
                raise RuntimeError("Trainer exited without completing/finalizing its workflow")
            print(json.dumps({"utc":datetime.now(timezone.utc).isoformat(), "training":progress}),flush=True)
            time.sleep(30)
        pid = read(root / "launch.json")["pid"]
        while Path(f"/proc/{pid}/stat").exists() and Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z":
            time.sleep(2)
        training = str(ROOT / ".venv-training/bin/python")
        analysis = str(ROOT / ".venv-analysis/bin/python")
        index_path = root / "analysis_index.json"
        if index_path.exists():
            index = read(index_path)
            if digest_file(ROOT / index["directory"] / "summary.json") != index["summary_sha256"]:
                raise ValueError("Completed diagnostic artifact changed")
        else:
            subprocess.run([training,"-u","scripts/analyze_cp2_phase_h.py","--root",str(root),"--device",args.device],cwd=ROOT,check=True)
        subprocess.run([analysis,"scripts/report_cp2_phase_h.py","--root",str(root)],cwd=ROOT,check=True)
        if not (root / "final_validation.json").exists():
            subprocess.run([training,"-u","scripts/validate_cp2_phase_h.py","--root",str(root)],cwd=ROOT,check=True)
        elif read(root / "final_validation.json")["status"] != "passed":
            raise ValueError("Previous final validation failed; inspect before retry")
        write_json(root / "completion_checks.json", {
            "status":"COMPLETED — REAL TRAINING, DIAGNOSTICS AND ARTIFACT QA",
            "finished_utc":datetime.now(timezone.utc).isoformat(),"visual_review":"pending direct inspection",
            "report":"reports/cp2-phase-h.md","no_automatic_H3_or_combination":True})
        print("Completed real training and posthoc QA; direct visual review still required",flush=True)
    except Exception:
        record = {"status":"NOT COMPLETED — inspect failure; completed models preserved",
                  "utc":datetime.now(timezone.utc).isoformat(),"traceback":traceback.format_exc()}
        directory = root / "posthoc_failures"
        write_json(directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json"),record)
        raise


if __name__=="__main__":
    main()
