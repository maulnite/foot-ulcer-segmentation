"""Post-selection gate/RMS, registered interventions and same-image inference cost."""

import argparse
import csv
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_attention import build_attention_model, capture_attention  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_a import FixedB1Objective, phase_a_sources, read  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_l import validate_epoch  # noqa: E402
from foot_ulcer_segmentation.cp2_train import export_evaluation, gpu_snapshot, make_cp2_loader, sync  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402


def capture_rows(model, loader, identities, variant, device, cached_run=None):
    """Read-only model diagnostics; GT never enters a module or a gate."""
    capture_attention(model, True)
    baseline, handles = {}, []
    if variant == "A0":
        for stage in (2, 3):
            baseline[stage] = {"input": [], "fused": []}

            def hook(stage_id, kind):
                def record(module, args):
                    values = args[0].detach().square().mean(dim=(1, 2, 3)).sqrt().cpu().tolist()
                    baseline[stage_id][kind].append(values)
                return record

            block = model.decoder.blocks[stage]
            handles.append(block.attention2.register_forward_pre_hook(hook(stage, "input")))
            handles.append(block.conv1.register_forward_pre_hook(hook(stage, "fused")))
    offset, bit_exact, max_difference = 0, 0, 0.
    try:
        model.eval()
        with torch.no_grad():
            for images, _ in loader:
                probabilities = model(images.to(device)).sigmoid().cpu().numpy()
                for probability in probabilities:
                    row = identities[offset]
                    if cached_run is not None:
                        path = cached_run / "predictions/validation" / (Path(row["name"]).stem + ".npz")
                        with np.load(path, allow_pickle=False) as cached:
                            difference = float(np.max(np.abs(probability[0] - cached["probability"])))
                            max_difference = max(max_difference, difference)
                            bit_exact += int(difference == 0)
                            np.testing.assert_allclose(probability[0], cached["probability"], rtol=0, atol=1e-6)
                            np.testing.assert_array_equal(probability[0] >= .5, cached["prediction"])
                    offset += 1
    finally:
        for handle in handles:
            handle.remove()
    if offset != len(identities):
        raise ValueError("Diagnostic sample count mismatch")
    records = []
    for stage in (2, 3):
        if variant == "A0":
            batches = [{"input_rms": a, "fused_rms": b} for a, b in
                       zip(baseline[stage]["input"], baseline[stage]["fused"])]
        else:
            batches = model.decoder.blocks[stage].phase_a_captures
        index = 0
        for batch in batches:
            for j, (first, last) in enumerate(zip(batch["input_rms"], batch["fused_rms"])):
                row = identities[index]
                fraction = row["gt_pixels"] / row["canvas_pixels"]
                group = "empty" if not row["gt_pixels"] else "small" if fraction <= .01 else "medium" if fraction <= .05 else "large"
                value = {"sample_id": row["sample_id"], "variant": variant, "stage": stage,
                         "size_group_evaluation_only": group, "input_rms": first, "fused_rms": last,
                         "fused_over_input_rms": last / first if first else None,
                         "w_additive_half": None, "w_maxout": None, "w_local": None,
                         "normalized_gate_entropy": None, "max_gate_weight": None,
                         "local_map_mean": None, "local_map_std": None}
                if not np.isfinite([first, last]).all():
                    raise ValueError("Nonfinite measured activation RMS")
                if variant != "A5" and first and last / first > 3 + 1e-5:
                    raise ValueError("Attention exceeds its preregistered magnitude bound")
                if "weights" in batch:
                    weights = np.asarray(batch["weights"][j])
                    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
                        raise ValueError("Invalid normalized branch weights")
                    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-6)
                    for key, weight in zip(("w_additive_half", "w_maxout", "w_local"), weights):
                        value[key] = float(weight)
                    value.update(normalized_gate_entropy=float(-(weights * np.log(weights.clip(1e-12))).sum() / np.log(3)),
                                 max_gate_weight=float(weights.max()), local_map_mean=batch["local_map_mean"][j],
                                 local_map_std=batch["local_map_std"][j])
                records.append(value)
                index += 1
        if index != len(identities):
            raise ValueError("Missing/duplicate stage captures")
    capture_attention(model, False)
    return records, {"images": offset, "bit_exact_probability_images": bit_exact if cached_run else None,
                     "maximum_absolute_probability_difference": max_difference if cached_run else None,
                     "hard_masks_equal": True if cached_run else None,
                     "probability_tolerance": 1e-6 if cached_run else None}


def grouped_statistics(records):
    result = {}
    for stage in (2, 3):
        result[str(stage)] = {}
        for group in ("all", "empty", "small", "medium", "large"):
            rows = [r for r in records if r["stage"] == stage and
                    (group == "all" or r["size_group_evaluation_only"] == group)]
            values = {"images": len(rows)}
            if rows:
                for key in ("fused_over_input_rms", "w_additive_half", "w_maxout", "w_local", "normalized_gate_entropy", "local_map_mean"):
                    finite = [r[key] for r in rows if r[key] is not None]
                    values[key] = {"mean": float(np.mean(finite)), "std": float(np.std(finite)),
                                   "minimum": min(finite), "maximum": max(finite)} if finite else None
                weights = [r["max_gate_weight"] for r in rows if r["max_gate_weight"] is not None]
                values["fraction_max_weight_over_0_95"] = float(np.mean(np.asarray(weights) > .95)) if weights else None
            result[str(stage)][group] = values
    return result


def inference_cost(model, image, device):
    model.eval()
    inputs = image[None].to(device)
    times = []
    with torch.no_grad():
        for _ in range(5):
            model(inputs).sigmoid()
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(30):
            started = time.perf_counter()
            model(inputs).sigmoid()
            sync(device)
            times.append(time.perf_counter() - started)
    return {"input_shape": list(inputs.shape), "warmup": 5, "timed_steps": 30,
            "median_ms": float(np.median(times) * 1000), "p95_ms": float(np.percentile(times, 95) * 1000),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "gpu_overall": gpu_snapshot() if device.type == "cuda" else None,
            "scope": "same-image sequential forward+sigmoid; excludes data/GT/loss/export; shared load, not exclusive hardware benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-name", default="analysis")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in (".", ".."):
        raise ValueError("Use a fresh single-component analysis directory")
    root, device = args.root.resolve(), torch.device(args.device)
    if read(root / "train_workflow.json")["status"] != "completed":
        raise ValueError("Full executed screening and primary selection must finish first")
    gate = read(root / "pretraining_checks.json")
    if gate["training_sources"] != phase_a_sources():
        raise ValueError("Executed training sources changed")
    config, manifest, selection = read(root / "protocol.json"), read(root / "data/manifest.json"), read(root / "selection.json")
    output = root / args.output_name
    output.mkdir(exist_ok=False)
    names = ["A0", *selection["decisions"]]
    control = ROOT / config["phase_a"]["control_root"] / "B1"
    setup(config["seed"], config["cpu_threads"])
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(root.relative_to(ROOT)),
              "selection_sha256": digest_file(root / "selection.json"),
              "posthoc_source_sha256": digest_file(__file__), "primary_selection_changed": False,
              "initial_two_validation_images": {}, "trained_validation": {}, "inference": {}, "interventions": {}}
    all_records = []
    for name in names:
        model, _ = build_attention_model(config, name, initial)
        model.to(device)
        loader, rows = make_cp2_loader(config, manifest, "validation")
        first_images, first_truth = next(iter(loader))
        initial_records, _ = capture_rows(model, [(first_images, first_truth)], rows[:len(first_images)], name, device)
        result["initial_two_validation_images"][name] = initial_records
        run = control if name == "A0" else root / name
        status = read(run / "status.json")
        if digest_file(run / "best.pth") != status["best_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        checkpoint = torch.load(run / "best.pth", map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        del checkpoint
        records, replay = capture_rows(model, loader, rows, name, device, run)
        all_records.extend(records)
        result["trained_validation"][name] = {"statistics": grouped_statistics(records), "checkpoint_validation_replay": replay}
        result["inference"][name] = inference_cost(model, first_images[0], device)
        if name in ("A1", "A3"):
            interventions = ["drop_local"] + (["fixed_prior_gate"] if name == "A3" else [])
            for intervention in interventions:
                destination = output / "interventions" / f"{name}-{intervention}"
                capture_attention(model, False, intervention)
                evaluator, loss = validate_epoch(model, FixedB1Objective(), loader, rows, config, device,
                                                  destination / "predictions")
                metrics = export_evaluation(destination / "metrics.json", evaluator, {
                    "variant": name, "intervention": intervention, "split": "validation",
                    "selection_sha256": result["selection_sha256"], "components": loss,
                    "checkpoint_sha256": status["best_checkpoint_sha256"], "training": "NOT RETRAINED — inference intervention only"})
                result["interventions"][f"{name}-{intervention}"] = metrics
            capture_attention(model, False)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Completed actual posthoc diagnostics {name}", flush=True)
    csv_path = output / "attention-validation.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_records)
    result.update(status="COMPLETED — REAL POST-SELECTION DIAGNOSTICS", rows=len(all_records),
                  numeric_csv_sha256=digest_file(csv_path), numeric_csv=str(csv_path.relative_to(ROOT)),
                  caveats=["GT subgroup labels used only for reporting; never a model/gate input",
                           "Normalized branch weights are not causal or physiological importance",
                           "RMS after training also reflects learned upstream features; it is not an isolated scaling experiment",
                           "Initial two-image RMS is descriptive only; no gain was fitted to these values",
                           "Inference interventions do not train missing-branch robustness"])
    if digest_file(root / "selection.json") != result["selection_sha256"]:
        raise ValueError("Primary selection changed during posthoc analysis")
    write_json(output / "summary.json", result)
    write_json(root / "analysis_index.json", {"directory": str(output.relative_to(ROOT)),
                                               "summary_sha256": digest_file(output / "summary.json")})
    write_json(ROOT / "reports/cp2-phase-a-analysis.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
