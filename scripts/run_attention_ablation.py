#!/usr/bin/env python3
"""Run controlled attention ablations through the frozen original engine."""

import argparse
import json

from attention_ablation import (BASELINE, adapted_engine, check_frozen_baseline,
                                finalize_predictions, read_json, save_cache_records,
                                validate_config, verify_models, probe_variant)
from foot_ulcer_segmentation import engine
from foot_ulcer_segmentation.common import project_path, read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "probe", "smoke", "train", "evaluate"])
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variant", choices=["full", "additive_only", "maxout_only"])
    parser.add_argument("--probe-output")
    args = parser.parse_args()
    if args.command == "probe":
        if not args.variant or not args.probe_output:
            parser.error("probe requires --variant and --probe-output")
        write_json(args.probe_output, probe_variant(args.variant))
        return
    if args.command == "verify":
        print(json.dumps(verify_models(), indent=2))
        return
    if not args.run_dir:
        parser.error("--run-dir is required")
    run = project_path(args.run_dir)
    if run.resolve() == BASELINE.resolve():
        raise ValueError("The completed baseline is immutable")
    config = read_json(run / "config.json") if args.command == "evaluate" else read_config(args.config)
    validate_config(config)
    check_frozen_baseline()
    split = "test" if args.command == "evaluate" else "validation"
    try:
        with adapted_engine(config, run, split) as captures:
            if args.command == "evaluate":
                result = engine.evaluate_run(run, split="test", device_name=args.device)
            else:
                result, _ = engine.run_training(config, run, args.device,
                                                smoke=args.command == "smoke", resume=args.resume)
                result.update(experiment_type="attention_branch_ablation", variant=config["ablation"]["variant"],
                              metrics_are_reproduction_results=False)
                write_json(run / "status.json", result)
        save_cache_records(run, split, captures)
        if args.command == "evaluate":
            finalize_predictions(run, args.device)
    except Exception as error:
        if (run / "status.json").exists() and args.command != "evaluate":
            write_json(run / "failure.json", {"error": f"{type(error).__name__}: {error}", "command": args.command})
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
