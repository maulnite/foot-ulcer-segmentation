"""Recompute CP2 A metrics and verify frozen controls, chronology and provenance."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_data import verify_cp2_manifest  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_a import (  # noqa: E402
    expansion_decision, phase_a_sources, read, select_candidate, verify_inputs,
)
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator  # noqa: E402
from validate_cp2_phase_l import compare, compare_csv  # noqa: E402


def validate_exports(export, rows, expected, settings):
    index = read(export / "manifest.json")
    by_id = {row["sample_id"]: row for row in rows}
    if len(index) != len(rows) or {r["sample_id"] for r in index} != by_id.keys():
        raise ValueError("Prediction population/count differs")
    evaluator = CP2Evaluator(settings, boundaries=True)
    for item in index:
        path = export / item["file"]
        if digest_file(path) != item["sha256"]:
            raise ValueError("Prediction checksum differs")
        row = by_id[item["sample_id"]]
        with Image.open(ROOT / row["mask_path"]) as im:
            truth = np.asarray(im.convert("L")) / 255
        with np.load(path, allow_pickle=False) as data:
            assert data["probability"].dtype == np.float32
            np.testing.assert_array_equal(data["truth"], truth)
            np.testing.assert_array_equal(data["prediction"], data["probability"] >= .5)
            evaluator.add(row, truth, data["probability"])
    computed = evaluator.summary()
    compare(computed, {key: expected[key] for key in computed})
    return evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "final_validation.json"
    if output.exists():
        raise FileExistsError("Preserve completed QA; use a new output filename")
    config, gate = read(root / "protocol.json"), read(root / "pretraining_checks.json")
    if gate["training_sources"] != phase_a_sources() or gate["protocol_fingerprint"] != digest_json(config):
        raise ValueError("Executed code/config changed")
    assert read(root / "train_workflow.json")["status"] == "completed"
    manifest = verify_cp2_manifest(read(root / "data/manifest.json"))
    assert manifest["fingerprint"] == config["phase_a"]["control_manifest_fingerprint"]
    assert digest_file(root / "initial_weights.pth") == config["phase_a"]["control_initial_weights_sha256"]
    prior = read(root / "protected_prior_sha256.json")
    with ThreadPoolExecutor(max_workers=6) as pool:
        changed = [name for name, ok in pool.map(lambda n: (n, (ROOT / n).is_file() and digest_file(ROOT / n) == prior[n]), prior) if not ok]
    if changed:
        raise ValueError(f"Historical CP1/P/L or user assets changed: {changed}")
    selection, expansion = read(root / "selection.json"), read(root / "expansion_decision.json")
    names = list(selection["decisions"])
    assert names == ["A1", *expansion["remaining_models"]]
    assert expansion["validation_only"] and selection["validation_only"]
    control = ROOT / config["phase_a"]["control_root"] / "B1"
    assert digest_file(control / "best.pth") == config["phase_a"]["control_B1_checkpoint_sha256"]
    paired = verify_inputs(config, root, names)
    compare(paired, read(root / "train_paired_controls.json"))
    public = read(ROOT / "reports/cp2-phase-a-results.json")
    profiles = read(root / "architecture_preflight.json")
    compare(public["profiles"], profiles["profiles"])
    assert profiles["attention_source_sha256"] == digest_file(ROOT / "src/foot_ulcer_segmentation/cp2_attention.py")
    assert profiles["A1_A2_A3_initial_forward_exactly_equal"]
    assert profiles["A5_initial_forward_equals_A0"]
    selected_at = datetime.fromisoformat(selection["created_utc"]).timestamp()
    expanded_at = datetime.fromisoformat(expansion["created_utc"]).timestamp()
    registered_at = datetime.fromisoformat(read(root / "preregistration.json")["registered_before_full_training_utc"])
    evaluations, validation_reports, local_proofs = [], {}, set()
    for name in names:
        run, state = root / name, read(root / name / "status.json")
        compare(public["statuses"][name], state)
        assert state["status"] == "completed" and not state["diagnostic_smoke_only"]
        identity = digest_json({"config": config, "architecture": name, "manifest": manifest["fingerprint"], "sources": phase_a_sources()})
        assert state["identity"] == identity and state["training_sources"] == phase_a_sources()
        assert read(run / "split-manifest.json") == manifest
        environment = read(run / "environment.json")
        started = datetime.fromisoformat(environment["created_at_utc"])
        assert registered_at < started
        if name != "A1":
            assert started.timestamp() >= expanded_at
        history = read(run / "history.json")
        cost = public["resources"]["fresh_models"][name]
        assert cost["completed_epochs"] == len(history)
        assert cost["sum_completed_epoch_compute_seconds"] == sum(r["epoch_compute_seconds"] for r in history)
        assert cost["peak_cuda_allocated_bytes_last_invocation"] == state["peak_cuda_allocated_bytes"]
        assert cost["peak_cuda_reserved_bytes_last_invocation"] == state["peak_cuda_reserved_bytes"]
        best = max(history, key=lambda r: r["validation_image_dice"])
        assert best["epoch"] == state["selected_epoch"] and len(history) == state["completed_epochs"]
        assert best["validation_image_dice"] == state["selected_validation_image_dice"]
        assert state["checkpoint_replay_exact"] and digest_file(run / "best.pth") == state["best_checkpoint_sha256"]
        initialization = read(run / "initialization_verified.json")
        assert initialization["source_tensor_sha256"] == config["phase_a"]["control_initial_tensor_sha256"]
        assert initialization["all_original_named_tensors_equal"]
        if name in ("A1", "A2", "A3"):
            local_proofs.add(initialization["local_tensor_sha256"])
        for checkpoint_name, epoch in (("best.pth", state["selected_epoch"]), ("last.pth", state["completed_epochs"])):
            checkpoint = torch.load(run / checkpoint_name, map_location="cpu")
            assert checkpoint["identity"] == identity and checkpoint["completed_epochs"] == epoch
            del checkpoint
        for split in ("validation", "test"):
            expected = read(run / f"evaluation-{split}.json")
            evaluator = validate_exports(run / "predictions" / split, manifest["splits"][split], expected, config["evaluation"])
            compare(expected, read(run / "final_evaluation.json")[split])
            compare(expected, public["metrics"][name][split])
            with (run / f"evaluation-{split}.csv").open(newline="") as stream:
                compare_csv(list(csv.DictReader(stream)), evaluator.rows)
            with (ROOT / f"reports/tables/cp2-phase-a-{name.lower()}-{split}.csv").open(newline="") as stream:
                compare_csv(list(csv.DictReader(stream)), evaluator.rows)
            if split == "test":
                for item in read(run / "predictions/test/manifest.json"):
                    assert (run / "predictions/test" / item["file"]).stat().st_mtime >= selected_at
            else:
                assert expected["all"]["image_based"]["dice"] == state["selected_validation_image_dice"]
                validation_reports[name] = expected
            evaluations.append({"variant": name, "split": split, "predictions": len(evaluator.rows),
                                "checksums_GT_threshold_region_boundary_subgroup_and_CSV": "passed"})
    assert len(local_proofs) == 1
    expected_expansion = expansion_decision(validation_reports["A1"], read(control / "evaluation-validation.json"), config)
    compare({k:v for k,v in expansion.items() if k != "created_utc"}, {k:v for k,v in expected_expansion.items() if k != "created_utc"})
    expected_selection = select_candidate(validation_reports, read(control / "evaluation-validation.json"), profiles["profiles"])
    compare({k:v for k,v in selection.items() if k != "created_utc"}, {k:v for k,v in expected_selection.items() if k != "created_utc"})
    analysis_index = read(root / "analysis_index.json")
    analysis_root = ROOT / analysis_index["directory"]
    assert digest_file(analysis_root / "summary.json") == analysis_index["summary_sha256"]
    analysis = read(analysis_root / "summary.json")
    assert analysis["selection_sha256"] == digest_file(root / "selection.json")
    assert digest_file(ROOT / analysis["numeric_csv"]) == analysis["numeric_csv_sha256"]
    interventions = []
    for label, metrics in analysis["interventions"].items():
        folder = analysis_root / "interventions" / label
        e = validate_exports(folder / "predictions", manifest["splits"]["validation"], metrics, config["evaluation"])
        with (folder / "metrics.csv").open(newline="") as stream:
            compare_csv(list(csv.DictReader(stream)), e.rows)
        interventions.append({"label": label, "validation_images": len(e.rows), "metrics_recomputed": True})
    for figure in public["figures"]:
        assert digest_file(ROOT / figure["path"]) == figure["sha256"]
    result = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "run_root": str(root.relative_to(ROOT)), "historical_and_user_files_unchanged": len(prior),
              "changed": changed, "training_sources_and_protocol_unchanged": True,
              "preregistered_before_full_training": True, "all_original_initial_tensors_match_B1": True,
              "paired_local_initialization_A1_A2_A3": True, "A1_A2_A3_initial_forward_equal": True,
              "paired_inputs": paired, "duplicate_rgb_overlap": 0, "patient_independence_established": False,
              "expansion_recomputed_from_A1_validation_only": True,
              "expansion_saved_before_expanded_training": True,
              "selection_recomputed_from_validation_only": True, "selection_precedes_all_new_test_probabilities": True,
              "checkpoint_hashes_epochs_and_metadata_verified": True,
              "primary_predictions_verified": sum(v["predictions"] for v in evaluations), "evaluations": evaluations,
              "validation_intervention_predictions_verified": sum(v["validation_images"] for v in interventions),
              "interventions": interventions, "per_image_gate_RMS_and_entropy_rows": analysis["rows"],
              "preflight": gate, "limitations": "one seed; RGB group split, not patient; CP1/P/L-exposed test descriptive; no external data"}
    write_json(output, result)
    write_json(ROOT / "reports/cp2-phase-a-validation.json", result)
    print(json.dumps({k:result[k] for k in ("status", "historical_and_user_files_unchanged", "primary_predictions_verified")}), flush=True)


if __name__ == "__main__":
    main()
