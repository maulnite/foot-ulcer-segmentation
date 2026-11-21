"""Verify real phase-L outputs and immutable CP1/P assets; optional GPU inference timing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_data import verify_cp2_manifest  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_l import (  # noqa: E402
    EXPERIMENTS, phase_l_sources, read, select_candidate, verify_paired_inputs,
)
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator  # noqa: E402
from foot_ulcer_segmentation.cp2_train import gpu_snapshot  # noqa: E402
from foot_ulcer_segmentation.data import make_dataset  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402
from foot_ulcer_segmentation.reference import build_model  # noqa: E402


def compare(actual, expected):
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Metric schema differs")
        for key in actual:
            compare(actual[key], expected[key])
    elif isinstance(actual, (float, int)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    elif actual != expected:
        raise ValueError("Metric value differs")


def compare_csv(saved, computed):
    by_name = {r["name"]: r for r in saved}
    if len(saved) != len(computed) or len(by_name) != len(saved):
        raise ValueError("Duplicate/missing CSV predictions")
    for row in computed:
        stored = by_name[row["name"]]
        if stored.keys() != row.keys():
            raise ValueError("CSV schema differs")
        for key, value in row.items():
            if isinstance(value, bool):
                assert stored[key] == str(value)
            elif isinstance(value, (int, float)):
                np.testing.assert_allclose(float(stored[key]), value, rtol=0, atol=1e-12)
            elif stored[key] != str(value):
                raise ValueError("CSV identity differs")


def benchmark(root, config, manifest, device_name):
    """Only after training completes; one same validation example for every model."""
    path = root / "batch1_inference.json"
    if path.exists():
        return read(path)
    device = torch.device(device_name)
    setup(config["seed"], config["cpu_threads"])
    dataset, rows = make_dataset(config, manifest, "validation")
    image, _ = dataset[0]
    results = {}
    for name in EXPERIMENTS:
        checkpoint = root / name / "best.pth"
        if digest_file(checkpoint) != read(root / name / "status.json")["best_checkpoint_sha256"]:
            raise ValueError("Inference checkpoint checksum changed")
        model = build_model(config, pretrained=False)
        saved = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(saved["state_dict"])
        del saved
        model = model.to(device).eval()
        inputs = torch.from_numpy(image[None]).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        with torch.no_grad():
            for _ in range(5):
                model(inputs).sigmoid()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(30):
                started = time.perf_counter()
                probability = model(inputs).sigmoid()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append(time.perf_counter() - started)
        assert torch.isfinite(probability).all()
        results[name] = {"sample_id": rows[0]["sample_id"], "input_shape": list(inputs.shape),
                         "warmup_steps": 5, "timed_steps": 30,
                         "latency_median_ms": float(np.median(times) * 1000),
                         "latency_p95_ms": float(np.percentile(times, 95) * 1000),
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                         "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                         "gpu_overall": gpu_snapshot() if device.type == "cuda" else None,
                         "caveat": "Shared-GPU forward+sigmoid only, same image, excludes data loading/GT loss/export"}
        del model, inputs, probability
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(path, results)
    return results


def validate(directory, output=None, inference_device=None):
    root = Path(directory).resolve()
    output = Path(output) if output else root / "final_validation.json"
    if output.exists():
        raise FileExistsError("Preserve completed QA; choose a new --output")
    if read(root / "train_workflow.json")["status"] != "completed":
        raise ValueError("All four actual training/evaluation stages must complete")
    config = read(root / "protocol.json")
    manifest = verify_cp2_manifest(read(root / "data/manifest.json"))
    gate = read(root / "pretraining_checks.json")
    if gate["protocol_fingerprint"] != digest_json(config):
        raise ValueError("Frozen protocol changed after pretraining checks")
    if gate["training_sources"] != phase_l_sources():
        raise ValueError("Executed training code differs from frozen preflight")
    protected = read(root / "protected_prior_sha256.json")

    def unchanged(item):
        name, expected = item
        return name, (ROOT / name).is_file() and digest_file(ROOT / name) == expected

    with ThreadPoolExecutor(max_workers=6) as pool:
        changed = [name for name, ok in pool.map(unchanged, protected.items()) if not ok]
    if changed:
        raise ValueError(f"Historical CP1/P or user work changed: {changed}")
    if manifest["fingerprint"] != config["phase_l"]["control_manifest_fingerprint"]:
        raise ValueError("Evaluation cohort differs from frozen baseline")
    if digest_file(root / "initial_weights.pth") != config["phase_l"]["control_initial_weights_sha256"]:
        raise ValueError("Shared initial weights changed")
    controls = verify_paired_inputs(config, root)
    if controls != read(root / "train_paired_controls.json"):
        raise ValueError("Common-epoch paired input verification differs")
    pairwise = []
    histories = {name: read(root / name / "history.json") for name in EXPERIMENTS}
    for index, first in enumerate(EXPERIMENTS):
        for second in EXPERIMENTS[index + 1:]:
            for a, b in zip(histories[first], histories[second]):
                if any(a[key] != b[key] for key in ("training_order_sha256", "augmented_input_sha256")):
                    raise ValueError("Loss candidates differ in a shared epoch beyond the cached baseline's duration")
            pairwise.append({"first": first, "second": second,
                             "common_epochs": min(len(histories[first]), len(histories[second])),
                             "all_order_and_augmented_input_hashes_equal": True})
    selection = read(root / "selection.json")
    selected_at = datetime.fromisoformat(selection["created_utc"]).timestamp()
    validation_reports, verified = {}, []
    public = read(ROOT / "reports/cp2-phase-l-results.json")
    for name in EXPERIMENTS:
        run = root / name
        state = read(run / "status.json")
        if state["status"] != "completed" or state["diagnostic_smoke_only"]:
            raise ValueError("Actual completed training required")
        if state["training_sources"] != phase_l_sources() or read(run / "split-manifest.json") != manifest:
            raise ValueError("Training code or population mismatch")
        expected_identity = digest_json({"config": config, "objective": name,
                                        "manifest": manifest["fingerprint"], "sources": phase_l_sources()})
        if state["identity"] != expected_identity or read(run / "config.json") != {"objective_id": name, **config}:
            raise ValueError("Executed objective/config identity differs from the frozen protocol")
        history = read(run / "history.json")
        best = max(history, key=lambda row: row["validation_image_dice"])
        if best["epoch"] != state["selected_epoch"] or len(history) != state["completed_epochs"]:
            raise ValueError("Selected epoch is not earliest maximum validation Dice")
        if digest_file(run / "best.pth") != state["best_checkpoint_sha256"]:
            raise ValueError("Best checkpoint checksum changed")
        for filename, expected_epoch in [("best.pth", state["selected_epoch"]), ("last.pth", state["completed_epochs"])]:
            checkpoint = torch.load(run / filename, map_location="cpu")
            if checkpoint["identity"] != state["identity"] or checkpoint["completed_epochs"] != expected_epoch:
                raise ValueError("Checkpoint identity/epoch metadata differs")
            del checkpoint
        for split in ("validation", "test"):
            export = run / "predictions" / split
            index = read(export / "manifest.json")
            rows = {r["sample_id"]: r for r in manifest["splits"][split]}
            if len(index) != len(rows) or {r["sample_id"] for r in index} != rows.keys():
                raise ValueError("Missing/duplicate original-image predictions")
            evaluator = CP2Evaluator(config["evaluation"], boundaries=True)
            for item in index:
                path = export / item["file"]
                if split == "test" and path.stat().st_mtime < selected_at:
                    raise ValueError("Test prediction predates validation-only selection")
                if digest_file(path) != item["sha256"]:
                    raise ValueError("Exported probability/mask checksum differs")
                with np.load(path, allow_pickle=False) as arrays:
                    row = rows[item["sample_id"]]
                    with Image.open(ROOT / row["mask_path"]) as image:
                        truth = np.asarray(image.convert("L")) / 255
                    if arrays["probability"].dtype != np.float32:
                        raise ValueError("Prediction precision changed")
                    np.testing.assert_array_equal(arrays["truth"], truth)
                    np.testing.assert_array_equal(arrays["prediction"], arrays["probability"] >= .5)
                    evaluator.add(row, truth, arrays["probability"])
            report = read(run / f"evaluation-{split}.json")
            recomputed = evaluator.summary()
            compare(recomputed, {key: report[key] for key in recomputed})
            compare(report, read(run / "final_evaluation.json")[split])
            compare(report, public["metrics"][name][split])
            if split == "validation":
                assert abs(report["all"]["image_based"]["dice"] - state["selected_validation_image_dice"]) < 1e-12
                validation_reports[name] = report
            with (run / f"evaluation-{split}.csv").open(newline="") as stream:
                saved = list(csv.DictReader(stream))
            compare_csv(saved, evaluator.rows)
            with (ROOT / f"reports/tables/cp2-phase-l-{name.lower()}-{split}.csv").open(newline="") as stream:
                if list(csv.DictReader(stream)) != saved:
                    raise ValueError("Published CSV differs from scientific run")
            verified.append({"objective_id": name, "split": split, "predictions": len(rows),
                             "GT_probability_threshold_checksums": "passed",
                             "all_region_boundary_subgroup_metrics_recomputed": True,
                             "per_image_csv_verified": True})
    expected_selection = select_candidate(validation_reports, read(root / "control_reference.json")["validation"])
    compare({k: v for k, v in selection.items() if k != "created_utc"},
            {k: v for k, v in expected_selection.items() if k != "created_utc"})
    if digest_file(ROOT / config["phase_l"]["control_root"] / "B1/best.pth") != config["phase_l"]["control_B1_checkpoint_sha256"]:
        raise ValueError("Frozen L0/B1 checkpoint changed")
    for figure in public["figures"]:
        if digest_file(ROOT / figure["path"]) != figure["sha256"]:
            raise ValueError("Presentation figure checksum differs")
    inference = benchmark(root, config, manifest, inference_device) if inference_device else None
    result = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "historical_and_user_files_unchanged": len(protected), "changed": changed,
              "manifest_fingerprint": manifest["fingerprint"], "duplicate_rgb_overlap": 0,
              "patient_independence_established": False, "paired_inputs": controls,
              "all_six_candidate_pairwise_input_checks": pairwise,
              "selection_recomputed_from_validation_only": True,
              "selection_precedes_every_new_test_probability": True,
              "new_predictions_verified": sum(r["predictions"] for r in verified), "evaluations": verified,
              "training_sources_unchanged": True, "checkpoint_epochs_checksums_and_replay": "passed",
              "preflight": gate, "boundary_oracle": read(root / "boundary_loss_oracle.json"),
              "inference": inference,
              "resource_interruptions": read(root / "resource_interruptions.json") if (root / "resource_interruptions.json").exists() else None,
              "initial_launcher_failure": read(root / "launch_failure.json") if (root / "launch_failure.json").exists() else None}
    write_json(output, result)
    public_result = json.loads(json.dumps(result).replace(str(ROOT), "."))
    write_json(ROOT / "reports/cp2-phase-l-validation.json", public_result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inference-device", help="Optional post-training sequential batch-1 timing, e.g. cuda:0")
    args = parser.parse_args()
    result = validate(args.root, args.output, args.inference_device)
    print(json.dumps({k: result[k] for k in ("status", "historical_and_user_files_unchanged", "new_predictions_verified")}, indent=2))
