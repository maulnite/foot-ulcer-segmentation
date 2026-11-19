#!/usr/bin/env python3
"""Export frozen-checkpoint masks for analysis and verify final CSV parity.

Inference only. Uses the unchanged training environment, model, and threshold.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from foot_ulcer_segmentation.common import ROOT, digest_file, digest_json, project_path, write_json
from foot_ulcer_segmentation.data import make_dataset, verify_manifest
from foot_ulcer_segmentation.engine import setup
from foot_ulcer_segmentation.evaluation import confusion_counts
from foot_ulcer_segmentation.reference import build_model, verify_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/runs/cp1-baseline-01")
    parser.add_argument("--output-dir", default="outputs/analysis/cp1-baseline-01")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run, output = project_path(args.run_dir), project_path(args.output_dir)
    config = json.loads((run / "config.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Analysis requires a completed training run")
    manifest = verify_manifest(json.loads((run / "split-manifest.json").read_text()))
    verify_reference()
    checkpoint_path = run / "best.pth"
    identity = {"checkpoint_sha256": digest_file(checkpoint_path),
                "config_fingerprint": digest_json(config), "split_fingerprint": manifest["fingerprint"],
                "threshold": config["evaluation"]["prediction_threshold"],
                "evaluation_sha256": {name: digest_file(run / name) for name in ["validation.csv", "evaluation-test.csv"]}}
    destination = output / "prediction-manifest.json"
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached["identity"] != identity:
            raise ValueError("Cached predictions belong to a different checkpoint/configuration")
        for record in cached["predictions"]:
            if digest_file(ROOT / record["path"]) != record["sha256"]:
                raise ValueError(f"Cached prediction changed: {record['path']}")
        print(f"Verified cached masks: {len(cached['predictions'])}; no inference performed.")
        return
    setup(config["seed"], config["cpu_threads"])
    device = torch.device(args.device)
    model = build_model(config, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["split_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint does not match the run split")
    if checkpoint["config_fingerprint"] != digest_json(config):
        raise ValueError("Checkpoint does not match the run configuration")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    del checkpoint
    records = []
    for split, csv_name in [("validation", "validation.csv"), ("test", "evaluation-test.csv")]:
        with (run / csv_name).open() as stream:
            expected = {row["name"]: row for row in csv.DictReader(stream)}
        dataset, rows = make_dataset(config, manifest, split)
        batch_size = config["training"]["batch_size"] if split == "validation" else 1
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        offset = 0
        with torch.no_grad():
            for images, masks in loader:
                probabilities = model(images.to(device)).cpu().numpy()
                for probability, mask in zip(probabilities, masks.numpy()):
                    name = rows[offset]["name"]
                    prediction = probability[0] >= identity["threshold"]
                    counts = confusion_counts(mask[0], prediction)
                    for key in ["tp", "tn", "fp", "fn"]:
                        if counts[key] != int(expected[name][key]):
                            raise ValueError(f"Prediction differs from final evaluation: {split}/{name}, {key}")
                    path = output / "predictions" / split / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(path)
                    records.append({"split": split, "name": name,
                                    "path": str(path.relative_to(ROOT)), "sha256": digest_file(path)})
                    offset += 1
        if offset != len(expected) or offset != len(rows):
            raise ValueError("Exported sample inventory differs from final evaluation")
        print(f"{split}: {offset} masks exported; all confusion counts match final CSV.", flush=True)
    write_json(destination, {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                             "identity": identity, "device": str(device),
                             "selected_epoch": status["selected_epoch"],
                             "training_performed": False, "all_counts_match_final_evaluation": True,
                             "predictions": records})
    print(f"Prediction manifest: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
