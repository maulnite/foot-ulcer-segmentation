"""Read CP2 phase-P status without importing a model or allocating GPU memory."""

import argparse
import json
from pathlib import Path


def status(root):
    root = Path(root)
    result = {"root": str(root), "baselines": {}}
    for stage in ("smoke", "train"):
        path = root / f"{stage}_workflow.json"
        if path.exists():
            result[stage] = json.loads(path.read_text())
    for name in ("B0", "B1"):
        path = root / name / "status.json"
        result["baselines"][name] = json.loads(path.read_text()) if path.exists() else {"status": "not_started_or_initializing"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root
    if root is None:
        root = json.loads(Path("outputs/cp2/latest_phase_p.json").read_text())["root"]
    print(json.dumps(status(root), indent=2))
