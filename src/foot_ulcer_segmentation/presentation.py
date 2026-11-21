"""Read-only presentation evidence from completed, checksum-verified caches."""

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ROOT, digest_file
from .cp2_protocol import boundary_iou
from .evaluation import confusion_counts, scores


def read(path):
    return json.loads(Path(path).read_text())


def read_table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def binary_image(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        if not np.all(array == array[..., :1]):
            raise ValueError("Mask channels disagree")
        array = array[..., 0]
    if array.ndim != 2 or not np.isin(array, [0, 255]).all():
        raise ValueError("Expected original binary 0/255 mask")
    return array > 0


def measure(truth, prediction):
    """Recompute hard-mask metrics, including the locked per-image empty rule."""
    counts = confusion_counts(truth, prediction)
    values = scores(counts)
    if not np.any(truth):
        values = {key: float(not np.any(prediction)) for key in values}
    return {**counts, **values, "gt_pixels": int(np.count_nonzero(truth)),
            "gt_fraction": float(np.count_nonzero(truth) / truth.size),
            "prediction_pixels": int(np.count_nonzero(prediction)),
            "boundary_iou": boundary_iou(truth, prediction)}


def assert_metrics(measured, recorded):
    for key in ["tp", "tn", "fp", "fn"]:
        if measured[key] != int(recorded[key]):
            raise ValueError(f"Cached confusion count changed: {key}")
    for key in ["dice", "boundary_iou"]:
        if key in recorded and not np.isclose(measured[key], float(recorded[key]), rtol=0, atol=1e-12):
            raise ValueError(f"Cached metric changed: {key}")


def aggregate(rows):
    small = [row for row in rows if 0 < row["gt_fraction"] <= .01]
    return {"images": len(rows), "image_dice": float(np.mean([row["dice"] for row in rows])),
            "small_dice": float(np.mean([row["dice"] for row in small])) if small else None,
            "boundary_iou": float(np.mean([row["boundary_iou"] for row in rows])),
            "misses": sum(row["gt_pixels"] > 0 and row["tp"] == 0 for row in rows),
            "empty_images": sum(row["gt_pixels"] == 0 for row in rows),
            "empty_fp_images": sum(row["gt_pixels"] == 0 and row["fp"] > 0 for row in rows)}


def choose(rows, predicate, key, excluded=()):
    """Minimum key, then stable sample ID; never choose a training checkpoint."""
    candidates = [row for row in rows if row["name"] not in excluded and predicate(row)]
    if not candidates:
        raise ValueError("No real case satisfies the declared illustration rule")
    return min(candidates, key=lambda row: (key(row), row["name"]))["name"]


def display_bounds(rgb, truth, predictions, margin=4):
    """Display-only zoom retaining all RGB content, GT and every model FP/FN."""
    if rgb.shape[:2] != truth.shape or any(np.shape(p) != truth.shape for p in predictions):
        raise ValueError("Display grids differ")
    support = np.any(rgb != 0, axis=-1) | truth.astype(bool)
    for prediction in predictions:
        support |= np.asarray(prediction, dtype=bool)
    y, x = np.where(support)
    height, width = truth.shape
    if not len(x):
        return (0, 0, width, height)
    return (max(0, int(x.min())-margin), max(0, int(y.min())-margin),
            min(width, int(x.max())+margin+1), min(height, int(y.max())+margin+1))


def overlay(rgb, truth, prediction):
    """TP=green, FP=vermillion, FN=blue; all other pixels retain the photograph."""
    confusion_counts(truth, prediction)
    output = rgb.astype(np.float32).copy()
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    for selected, color in [(truth & prediction, [0, 158, 115]),
                            (~truth & prediction, [213, 94, 0]),
                            (truth & ~prediction, [0, 114, 178])]:
        output[selected] = .30*output[selected] + .70*np.asarray(color)
    return np.clip(output, 0, 255).astype(np.uint8)


@dataclass
class CompletedModel:
    name: str
    family: str
    root: Path
    status: dict
    reported: dict
    guards: dict
    checkpoint_sha256: str


class Evidence:
    def __init__(self, settings):
        self.settings = settings
        self.manifest = read(ROOT/settings["manifest"])
        self.rows = {split: {row["sample_id"]: row for row in values}
                     for split, values in self.manifest["splits"].items()}
        self.models, self.metrics, self.tables, self.indices = {}, {}, {}, {}
        self.raw, self.checks = {}, []
        for family in settings["families"]:
            report = read(ROOT/family["report"])
            run = ROOT/report.get("run_root", report.get("run"))
            for name in family["models"]:
                directory = run/name
                status = read(directory/"status.json")
                if status["status"] != "completed":
                    raise ValueError(f"Model not completed: {name}")
                checksum = digest_file(directory/"best.pth")
                if checksum != status["best_checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint changed: {name}")
                reported = report["metrics"][name]
                if any(reported[s]["split_fingerprint"] != self.manifest["fingerprint"] for s in ["validation", "test"]):
                    raise ValueError("Noncomparable cohorts cannot enter the CP2 scoreboard")
                if family["id"] in ["L", "A", "H"]:
                    guards = report["selection"]["decisions"][name]["checks"]
                elif family["id"] == "N":
                    guards = report["selection"]["checks"]
                elif family["id"] == "C":
                    guards = report["selection"]["original_cp2_guards"]
                else:
                    guards = {}
                self.models[name] = CompletedModel(name, family["id"], directory, status, reported, guards, checksum)

    def source(self, split, sample):
        key = (split, sample)
        if key not in self.raw:
            row = self.rows[split][sample]
            for kind in ["image", "mask"]:
                if digest_file(ROOT/row[f"{kind}_path"]) != row[f"{kind}_sha256"]:
                    raise ValueError("Original data bytes changed")
            with Image.open(ROOT/row["image_path"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            truth = binary_image(ROOT/row["mask_path"])
            if truth.shape != (224, 224) or rgb.shape != (224, 224, 3):
                raise ValueError("Source image/GT grids do not match the evaluation canvas")
            self.raw[key] = (rgb, truth)
        return self.raw[key]

    def case(self, model, split, sample):
        key = (model, split)
        directory = self.models[model].root/"predictions"/split
        if key not in self.indices:
            records = read(directory/"manifest.json")
            if [r["sample_id"] for r in records] != list(self.rows[split]):
                raise ValueError("Prediction identities/order differ from the canonical split")
            self.indices[key] = {r["sample_id"]: r for r in records}
        record = self.indices[key][sample]
        file = directory/record["file"]
        if digest_file(file) != record["sha256"]:
            raise ValueError("Prediction cache checksum changed")
        rgb, truth = self.source(split, sample)
        with np.load(file, allow_pickle=False) as data:
            probability, prediction, cached_truth = (data[k] for k in ["probability", "prediction", "truth"])
            if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError("Invalid actual model probability")
            np.testing.assert_array_equal(cached_truth, truth)
            np.testing.assert_array_equal(prediction, probability >= .5)
            prediction = prediction.astype(bool)
        return rgb, truth, prediction, record

    def verify(self):
        for name, model in self.models.items():
            self.metrics[name], self.tables[name] = {}, {}
            for split in ["validation", "test"]:
                expected = {r["name"]: r for r in read_table(model.root/f"evaluation-{split}.csv")}
                measured = []
                for sample in self.rows[split]:
                    _, truth, prediction, _ = self.case(name, split, sample)
                    row = {"name": sample, **measure(truth, prediction)}
                    assert_metrics(row, expected[sample])
                    measured.append(row)
                summary = aggregate(measured)
                original = model.reported[split]
                for key, value in [("image_dice", original["all"]["image_based"]["dice"]),
                                   ("small_dice", original["subgroups"]["small"]["image_based"]["dice"]),
                                   ("boundary_iou", original["all"]["image_based"]["boundary_iou"]),
                                   ("misses", original["gt_positive"]["total_foreground_misses"]),
                                   ("empty_fp_images", original["gt_empty"]["images_with_prediction"])]:
                    if not np.isclose(summary[key], value, rtol=0, atol=1e-12):
                        raise ValueError(f"Aggregate mismatch: {name} {split} {key}")
                self.metrics[name][split] = summary
                self.tables[name][split] = {r["name"]: r for r in measured}
                self.checks.append({"model": name, "split": split, "verified_predictions": len(measured),
                                    "raw_GT_hashes_cache_hashes_threshold_counts_printed_metrics": "passed"})
            print(f"Verified completed {name}: 118 validation + 261 test caches", flush=True)

    def provenance(self, model, split, sample):
        item = self.models[model]
        rgb, truth, prediction, cache = self.case(model, split, sample)
        return {"model": model, "family": item.family, "seed": 42,
                "sample_id": sample, "split": split, "dataset": "Chronic Wound clean CP2",
                "image_sha256": self.rows[split][sample]["image_sha256"],
                "mask_sha256": self.rows[split][sample]["mask_sha256"],
                "selected_epoch": item.status["selected_epoch"], "checkpoint_sha256": item.checkpoint_sha256,
                "cache_sha256": cache["sha256"], "source": "REAL / CACHED",
                "metrics_recomputed": measure(truth, prediction),
                "manifest_fingerprint": self.manifest["fingerprint"]}
