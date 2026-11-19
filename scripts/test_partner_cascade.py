"""Replay frozen Ganta checkpoints and compare batch-1 coarse/cascade CPU latency."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.evaluation import SegmentationEvaluator  # noqa: E402
from foot_ulcer_segmentation.partner_cascade import (  # noqa: E402
    CONSTANTS, latency_summary, load_model, load_partner_namespace, verify_inventory,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partner-root", type=Path, required=True)
    parser.add_argument("--source", default="notebook/13_cascade_crop_refine_chronic_wound.py")
    parser.add_argument("--run-name", default="cascade_chronic_wound_efficientnet-b7_2026-11-14_00-59-20")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timing-images", type=int, default=16)
    parser.add_argument("--timing-repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Engineering subset only; omit for all 278 images")
    args = parser.parse_args()
    if min(args.threads, args.timing_images, args.timing_repeats, args.warmup) <= 0:
        parser.error("Thread, timing, and warmup counts must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("Subset size must be positive")
    partner = args.partner_root.resolve()
    if args.output.resolve().is_relative_to(partner):
        parser.error("Outputs must stay outside the partner directory")
    return args


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,utilization.gpu", "--format=csv"],
        capture_output=True, text=True, check=False)
    return result.stdout.strip()


def execute(args):
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    cv2.setNumThreads(1)
    source = args.partner_root / args.source
    base = args.partner_root / "output_chronic_wound"
    raw = args.partner_root / "data/raw/chronic_wound/test"
    checkpoints = base / "checkpoints" / args.run_name
    expected_predictions = base / "predictions" / args.run_name
    historical = {}
    for mode in ("coarse", "cascade"):
        path = base / "results" / f"{args.run_name}_per_image_{mode}.csv"
        historical[mode] = {row["Name"]: row for row in csv.DictReader(path.open())}
    names = sorted(path.name for path in (raw / "images").glob("*.png"))
    if len(names) != len(set(names)) or {Path(n).stem for n in names} != set(historical["cascade"]):
        raise ValueError("Test images and historical prediction identities differ")
    if set(historical["coarse"]) != set(historical["cascade"]):
        raise ValueError("Historical evaluation populations differ")
    if args.limit:
        names = names[:args.limit]
    # Hash all actually used input files before loading checkpoints or inference.
    protected = [source, checkpoints / "stage1_coarse.pth", checkpoints / "stage2_refiner.pth"]
    protected += [base / "results" / f"{args.run_name}_per_image_{m}.csv" for m in historical]
    for name in names:
        protected += [raw / "images" / name, raw / "labels" / name, expected_predictions / name]
    inventory = {str(path.resolve()): digest_file(path) for path in protected}
    write_json(args.output / "input_inventory.json", inventory)
    namespace = load_partner_namespace(source, device="cpu")
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "mode": "checkpoint_replay_no_training",
        "run_name": args.run_name, "device": "cpu", "precision": "float32", "batch_size": 1,
        "cpu_threads": args.threads, "cv2_threads": 1, "seed": 42,
        "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
        "machine": platform.platform(), "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
             if line.startswith("model name")), "unknown"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "gpu_before": gpu_snapshot(),
        "source_sha256": digest_file(source), "runner_sha256": digest_file(__file__),
        "helper_sha256": digest_file(ROOT / "src/foot_ulcer_segmentation/partner_cascade.py"),
        "source_configuration": {key: namespace[key] for key in sorted(CONSTANTS)},
        "images": len(names), "full_historical_population": len(names) == len(historical["cascade"]),
        "timing": {"images": min(args.timing_images, len(names)), "repeats": args.timing_repeats,
                   "warmup_per_pipeline": args.warmup, "selection": "evenly spaced sorted test IDs",
                   "order": "alternate coarse-first/cascade-first; no result-based sample selection",
                   "scope": "preloaded normalized tensor to host probability maps; excludes image loading, metrics and PNG export",
                   "shared_cpu": True, "exclusive_gpu_latency": "NOT RUN — GPU BUSY"},
        "limitations": ["No retraining or CP2 split experiment", "Historical test is exploratory",
                        "CPU replay differs from original torch 2.7.1 CUDA environment",
                        "No hardware-matched comparison with paper RTX 2080 Ti timing"],
    }
    write_json(args.output / "protocol.json", protocol)
    coarse, coarse_metadata = load_model(namespace, checkpoints / "stage1_coarse.pth")
    refiner, refiner_metadata = load_model(namespace, checkpoints / "stage2_refiner.pth")
    dataset = namespace["Dataset"](names, str(raw / "images"), str(raw / "labels"))
    models = {"coarse": coarse_metadata, "refiner": refiner_metadata}
    print(json.dumps({"strict_load": models, "images": len(names), "device": "cpu"}), flush=True)
    evaluators = {mode: SegmentationEvaluator(namespace["THRESHOLD"]) for mode in historical}
    replay_rows = []
    candidate_counts = []
    inference_seconds = []
    with torch.no_grad():
        for index, name in enumerate(names):
            image, truth, _ = dataset[index]
            inputs = torch.from_numpy(np.ascontiguousarray(image[None]))
            before = time.perf_counter()
            first, final, candidates = namespace["cascade_predict"](coarse, refiner, inputs)
            inference_seconds.append(time.perf_counter() - before)
            if not np.isfinite(first).all() or not np.isfinite(final).all():
                raise ValueError(f"Nonfinite probabilities: {name}")
            candidate_counts.append(candidates)
            saved = cv2.imread(str(expected_predictions / name), cv2.IMREAD_GRAYSCALE)
            if saved is None or saved.shape != final.shape:
                raise ValueError(f"Missing or invalid saved prediction: {name}")
            replay = {"name": name, "candidates": candidates,
                      "cascade_changed_pixels": int(np.count_nonzero(
                          (final >= namespace["THRESHOLD"]) != (saved >= 128)))}
            for mode, probability in (("coarse", first), ("cascade", final)):
                row = evaluators[mode].add(name, truth[0] > .5, probability)
                old = historical[mode][Path(name).stem]
                replay[f"{mode}_counts_match"] = all(row[key] == int(old[key]) for key in ("tp", "tn", "fp", "fn"))
                replay[f"{mode}_dice_delta_pp"] = row["dice"] * 100 - float(old["Dice"])
            replay_rows.append(replay)
            if (index + 1) % 20 == 0 or index == len(names) - 1:
                write_json(args.output / "status.json", {"status": "running", "stage": "checkpoint_replay",
                           "images_done": index + 1, "images_total": len(names),
                           "elapsed_seconds": time.perf_counter() - started})
                print(f"Replay {index + 1}/{len(names)} | {time.perf_counter() - started:.1f}s", flush=True)
    for mode, evaluator in evaluators.items():
        write_csv(args.output / f"{mode}_per_image.csv", evaluator.rows)
    write_csv(args.output / "replay_comparison.csv", replay_rows)
    # Paired benchmark: same preloaded actual tensors, same CPU/precision, alternating order.
    indices = np.linspace(0, len(names) - 1, min(args.timing_images, len(names)), dtype=int)
    examples = [(int(i), torch.from_numpy(np.ascontiguousarray(dataset[int(i)][0][None]))) for i in indices]
    pipelines = {"coarse": lambda x: coarse(x).sigmoid()[0, 0].cpu().numpy(),
                 "cascade": lambda x: namespace["cascade_predict"](coarse, refiner, x)}
    timing_rows = []
    with torch.no_grad():
        for pipeline in pipelines.values():
            for _ in range(args.warmup):
                pipeline(examples[0][1])
        for repeat in range(args.timing_repeats):
            for position, (index, inputs) in enumerate(examples):
                order = ("coarse", "cascade") if (repeat + position) % 2 == 0 else ("cascade", "coarse")
                for mode in order:
                    before = time.perf_counter()
                    pipelines[mode](inputs)
                    timing_rows.append({"name": names[index], "mode": mode, "repeat": repeat,
                                        "candidates": candidate_counts[index],
                                        "seconds": time.perf_counter() - before})
            print(f"Paired timing repeat {repeat + 1}/{args.timing_repeats}", flush=True)
    write_csv(args.output / "paired_timing.csv", timing_rows)
    timing = {mode: latency_summary([row["seconds"] for row in timing_rows if row["mode"] == mode])
              for mode in pipelines}
    preservation = verify_inventory(inventory)
    result = {
        "status": "completed", "training_performed": False, "models": models,
        "metrics": {mode: evaluator.summary() for mode, evaluator in evaluators.items()},
        "replay": {"images": len(names),
                   "bitwise_equal_cascade_masks": sum(row["cascade_changed_pixels"] == 0 for row in replay_rows),
                   "total_changed_pixels": sum(row["cascade_changed_pixels"] for row in replay_rows),
                   "coarse_confusion_matches": sum(row["coarse_counts_match"] for row in replay_rows),
                   "cascade_confusion_matches": sum(row["cascade_counts_match"] for row in replay_rows)},
        "candidates": {"histogram": dict(sorted(Counter(candidate_counts).items())),
                       "mean": float(np.mean(candidate_counts)), "max": max(candidate_counts),
                       "refined_images": sum(value > 0 for value in candidate_counts)},
        "cpu_paired_latency": timing,
        "cpu_cascade_mean_time_ratio": timing["cascade"]["mean_ms"] / timing["coarse"]["mean_ms"],
        "full_replay_latency_including_first_call": latency_summary(inference_seconds),
        "preservation": preservation, "gpu_after": gpu_snapshot(),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(args.output / "result.json", result)
    write_json(args.output / "status.json", {"status": "completed", "result": "result.json"})
    print(json.dumps(result, indent=2), flush=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        execute(args)
    except Exception as exc:
        write_json(args.output / "status.json", {"status": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
