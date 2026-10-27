#!/usr/bin/env python3
"""Read-only ablation progress; does not import torch or start any training."""

import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    batch = ROOT / "outputs/ablations/cp1-attention-01"
    if not (batch / "workflow.json").exists():
        print(json.dumps({"status": "not_launched"}, indent=2))
        return
    workflow = json.loads((batch / "workflow.json").read_text())
    result = {key: workflow.get(key) for key in ["status", "current_variant", "device", "updated_at_utc",
                                                "publication_status", "error", "result_commit"] if workflow.get(key) is not None}
    result["runs"] = []
    for row in workflow["runs"]:
        run = ROOT / row["run_dir"]
        item = {"variant": row["variant"], "run_directory": row["run_dir"], "status": "queued"}
        if (run / "status.json").exists():
            item.update(json.loads((run / "status.json").read_text()))
        if (run / "history.json").exists():
            history = json.loads((run / "history.json").read_text())
            item["completed_epochs"] = len(history)
            if history:
                item["latest_validation"] = history[-1]["validation"]
                item["median_epoch_seconds"] = statistics.median(r["seconds"] for r in history)
        if (run / "evaluation-test.json").exists():
            item["test"] = json.loads((run / "evaluation-test.json").read_text())["offline_hard_line"]
        result["runs"].append(item)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
