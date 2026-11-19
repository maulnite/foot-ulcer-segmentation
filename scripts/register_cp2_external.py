"""Create a new append-only evaluation run from the public frozen model registry."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cp2/external.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    records = config["models"]
    if len(records) != 26 or len({record["id"] for record in records}) != 26:
        raise ValueError("Preserve all 26 preregistered unique checkpoints")
    for record in records:
        for kind in ("checkpoint", "config"):
            if digest_file(ROOT / record[kind]) != record[kind + "_sha256"]:
                raise ValueError(f"Missing or changed frozen {kind}: {record['id']}")
    if args.run.exists():
        raise FileExistsError("Never overwrite a previous evaluation run")
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"],
                               cwd=ROOT, capture_output=True, check=True).stdout
    paths = {Path(item.decode()) for item in (tracked + untracked).split(b"\0") if item}
    for record in records:
        paths.update([Path(record["checkpoint"]), Path(record["config"]), Path(record["run"]) / "status.json"])
    preserved = {str(path): digest_file(ROOT / path) for path in sorted(paths) if (ROOT / path).is_file()}
    args.run.mkdir(parents=True)
    write_json(args.run / "models.json", records)
    protocol = dict(config["protocol"])
    protocol["created_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["frozen_models_manifest_sha256"] = digest_file(args.run / "models.json")
    write_json(args.run / "preregistration.json", protocol)
    write_json(args.run / "protected_before.json", preserved)
    metadata = ROOT / "data/metadata/co2wounds_v2_mendeley_v1_acquisition.json"
    if metadata.exists():
        acquisition = json.loads(metadata.read_text())
        if acquisition["archive_sha256"] != config["archive_sha256"]:
            raise ValueError("Cached dataset is not the registered version")
        write_json(args.run / "acquisition.json", acquisition)
    print(f"Registered {len(records)} checkpoints at {args.run}")


if __name__ == "__main__":
    main()
