#!/usr/bin/env python3
"""Train or evaluate x-FUSegNet using the pinned author's implementation."""

import argparse
import json

from foot_ulcer_segmentation.common import ROOT, project_path, write_json
from xfuseg_data import load_config, prepare
from xfuseg_train import train, verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "train", "infer"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config()
    run = project_path(args.run_dir or config["run_dir"])
    if args.command == "prepare":
        verify_source(config)
        manifest = prepare(config)
        result = {"status": "verified", "fingerprint": manifest["fingerprint"], "labeled": 1010, "test": 200,
                  "folds": [{"fold": f["fold"], "train": len(f["train"]), "validation": len(f["validation"]),
                             "validation_identical_rgb_to_train": f["validation_images_identical_rgb_to_train"]} for f in manifest["folds"]]}
    elif args.command == "infer":
        if args.run_dir:
            parser.error("inference uses the declared full run only")
        from xfuseg_inference import infer
        result = infer(config, args.device)
    else:
        smoke = args.command == "smoke"
        if smoke and (not args.run_dir or run == ROOT / config["run_dir"]):
            parser.error("smoke requires a separate --run-dir")
        if not smoke and run != ROOT / config["run_dir"]:
            parser.error("full training uses the declared run directory")
        try:
            result = train(config, run, args.device, smoke=smoke, resume=args.resume)
        except Exception as error:
            if run.exists():
                write_json(run / "failure.json", {"error": f"{type(error).__name__}: {error}"})
            raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
