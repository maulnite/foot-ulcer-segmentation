"""Read-only validation of completed CP2 baselines, exported arrays and CP1 preservation."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_data import verify_cp2_manifest  # noqa: E402
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator  # noqa: E402
from foot_ulcer_segmentation.cp2_train import training_fingerprints  # noqa: E402


def compare(actual, expected):
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Metric schema changed")
        for key in actual:
            compare(actual[key], expected[key])
    elif actual is None:
        if expected is not None:
            raise ValueError("Undefined metric changed")
    elif isinstance(actual, (float, int)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    elif actual != expected:
        raise ValueError("Metric value changed")


def validate(root, output=None):
    root = Path(root).resolve()
    output = Path(output) if output else root / "final_validation.json"
    if output.exists():
        raise FileExistsError("Preserve completed QA; use --output for a new validation record")
    manifest = verify_cp2_manifest(json.loads((root / "data/manifest.json").read_text()))
    config = json.loads((root / "protocol.json").read_text())
    preflight = json.loads((root / "pretraining_checks.json").read_text())
    if (manifest["fingerprint"] != preflight["manifest_fingerprint"]
            or digest_file(root / "data/manifest.json") != preflight["manifest_file_sha256"]):
        raise ValueError("Manifest changed after pretraining approval gate")
    initial = json.loads((root / "initial_audit.json").read_text())
    if digest_file(ROOT / "Pengembangan_Novelty_dan_Eksperimen_CP2_FUSegNet.md") != initial["roadmap_sha256"]:
        raise ValueError("User-uploaded roadmap changed")
    protected = json.loads((root / "protected_cp1_sha256.json").read_text())
    changed = [name for name, expected in protected.items()
               if not Path(name).is_file() or digest_file(name) != expected]
    if changed:
        raise ValueError(f"CP1/user work changed: {changed}")
    verified, initialization, histories = [], [], []
    code = training_fingerprints()
    for name in ("B0", "B1"):
        run = root / name
        status = json.loads((run / "status.json").read_text())
        run_manifest = json.loads((run / "split-manifest.json").read_text())
        if run_manifest != manifest or status["split_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Baseline membership differs from locked master manifest")
        if status["status"] != "completed" or status["diagnostic_smoke_only"]:
            raise ValueError("Full completed real-data baseline required")
        if status["training_sources"] != code:
            raise ValueError("Executed training sources differ from current implementation")
        if digest_file(run / "best.pth") != status["best_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        history = json.loads((run / "history.json").read_text())
        histories.append(history)
        selected = max(history, key=lambda r: r["validation_image_dice"])
        if selected["epoch"] != status["selected_epoch"]:
            raise ValueError("Checkpoint was not earliest highest validation Image Dice")
        checkpoint = torch.load(run / "best.pth", map_location="cpu")
        if checkpoint["completed_epochs"] != status["selected_epoch"]:
            raise ValueError("Checkpoint metadata epoch differs")
        del checkpoint
        initialization.append(json.loads((run / "initialization_verified.json").read_text()))
        for split in ("validation", "test"):
            export = run / "predictions" / split
            exported = json.loads((export / "manifest.json").read_text())
            rows = {r["sample_id"]: r for r in manifest["splits"][split]}
            if len(exported) != len(rows) or {r["sample_id"] for r in exported} != rows.keys():
                raise ValueError("Prediction identities/counts differ from locked split")
            evaluator = CP2Evaluator(config["evaluation"], boundaries=True)
            for item in exported:
                path = export / item["file"]
                if digest_file(path) != item["sha256"]:
                    raise ValueError("Exported probability/mask changed")
                arrays = np.load(path, allow_pickle=False)
                row = rows[item["sample_id"]]
                with Image.open(ROOT / row["mask_path"]) as image:
                    original = np.asarray(image.convert("L")) / 255
                np.testing.assert_array_equal(arrays["truth"], original)
                np.testing.assert_array_equal(arrays["prediction"], arrays["probability"] >= .5)
                evaluator.add(row, original, arrays["probability"])
            report = json.loads((run / f"evaluation-{split}.json").read_text())
            compare(evaluator.summary(), {k: report[k] for k in evaluator.summary()})
            final = json.loads((run / "final_evaluation.json").read_text())[split]
            compare(report, final)
            with (run / f"evaluation-{split}.csv").open() as stream:
                saved = list(csv.DictReader(stream))
            saved_by_name = {r["name"]: r for r in saved}
            if len(saved) != len(rows) or len(saved_by_name) != len(saved):
                raise ValueError("Per-image CSV prediction count differs")
            for computed in evaluator.rows:
                stored = saved_by_name[computed["name"]]
                if stored.keys() != computed.keys():
                    raise ValueError("Per-image CSV schema differs")
                for key, value in computed.items():
                    if isinstance(value, bool):
                        if stored[key] != str(value):
                            raise ValueError("Per-image CSV boolean differs")
                    elif isinstance(value, (float, int)):
                        np.testing.assert_allclose(float(stored[key]), value, rtol=0, atol=1e-12)
                    elif stored[key] != str(value):
                        raise ValueError("Per-image CSV identity/value differs")
            public_csv = ROOT / f"reports/tables/cp2-phase-p-{name.lower()}-{split}.csv"
            # Git normalizes CSV line endings; compare scientific content, not CRLF/LF.
            with public_csv.open(newline="") as stream:
                public_rows = list(csv.DictReader(stream))
            if public_rows != saved:
                raise ValueError("Published per-image table differs from run artifact")
            verified.append({"baseline": name, "split": split, "predictions": len(rows),
                             "all_probability_and_ground_truth_checksums": "passed",
                             "per_image_csv_and_public_table_verified": True,
                             "region_boundary_subgroup_metrics_recomputed": True})
    if initialization[0] != initialization[1]:
        raise ValueError("Initial weights or split membership differ")
    if digest_file(root / "initial_weights.pth") != initialization[0]["initial_weights_file_sha256"]:
        raise ValueError("Shared initial weights changed after training")
    for a, b in zip(*histories):
        for key in ("training_order_sha256", "augmented_input_sha256"):
            if a[key] != b[key]:
                raise ValueError("Common-epoch input/augmentation order differs")
    result = {"status": "passed", "manifest_fingerprint": manifest["fingerprint"],
              "original_cp1_and_user_files_verified": len(protected),
              "unexpected_changes": changed, "baseline_initialization_identical": True,
              "duplicate_rgb_group_overlap": 0, "patient_independence_established": False,
              "common_epochs_with_identical_training_inputs": min(map(len, histories)),
              "selected_checkpoint_policy_verified": True,
              "evaluations": verified,
              "pretraining_checks": json.loads((root / "pretraining_checks.json").read_text())}
    result["pixel_decoder_parity"] = json.loads((root / "decoder_parity.json").read_text())
    result["boundary_metric_oracle"] = json.loads((root / "boundary_oracle.json").read_text())
    write_json(output, result)
    write_json(ROOT / "reports/cp2-phase-p-validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New local QA evidence path; completed records are preserved")
    args = parser.parse_args()
    result = validate(args.root, args.output)
    print(json.dumps(result, indent=2))
