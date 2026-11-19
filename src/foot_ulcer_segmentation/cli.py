import argparse
from datetime import datetime, timezone
import json

from .common import project_path, read_config, write_json


def main():
    parser = argparse.ArgumentParser(description="FUSegNet author-code reproduction pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("prepare-splits", help="Capture or verify the local author-algorithm split")
    split.add_argument("--config", default="configs/baseline.json")
    for name in ["smoke", "train"]:
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/baseline.json")
        command.add_argument("--device", default="cpu")
        command.add_argument("--run-dir")
        if name == "train":
            command.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run-dir", required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluation.add_argument("--checkpoint", default="best.pth")
    evaluation.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "prepare-splits":
        from .data import prepare_splits
        manifest = prepare_splits(read_config(args.config))
        print(json.dumps({"counts": {s: len(rows) for s, rows in manifest["splits"].items()},
                          "fingerprint": manifest["fingerprint"]}, indent=2))
        return
    if args.command == "evaluate":
        from .engine import evaluate_run
        result = evaluate_run(args.run_dir, args.split, args.checkpoint, args.device)
    else:
        from .engine import run_training
        if getattr(args, "resume", False) and not args.run_dir:
            parser.error("--resume requires --run-dir")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = args.run_dir or f"outputs/{args.command}/{stamp}"
        owns_directory = getattr(args, "resume", False) or not project_path(directory).exists()
        try:
            result, _ = run_training(read_config(args.config), directory, args.device,
                                     smoke=args.command == "smoke", resume=getattr(args, "resume", False))
        except Exception as error:
            status_path = project_path(directory) / "status.json"
            if owns_directory and status_path.exists():
                write_json(status_path, {"status": "failed", "command": args.command,
                                         "error": f"{type(error).__name__}: {error}"})
            raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
