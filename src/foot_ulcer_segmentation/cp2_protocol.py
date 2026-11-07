"""Locked CP2 objective, checkpoint rule and source-grid evaluation."""

from dataclasses import dataclass, asdict
from functools import lru_cache
import math
import runpy

import cv2
import numpy as np
from scipy.ndimage import label
import torch

from .common import ROOT
from .evaluation import SegmentationEvaluator, scores
from .reference import load_reference


class BaselineObjective(torch.nn.Module):
    """Only focal's argument differs. The pinned reference supplies both primitives."""

    def __init__(self, baseline):
        super().__init__()
        if baseline not in ("B0", "B1"):
            raise ValueError("Phase P supports exactly B0 and B1")
        self.baseline = baseline
        _, utils, _ = load_reference()
        self.dice = utils.losses.DiceLoss(eps=1.0, beta=1.0)
        self.focal = utils.losses.FocalLoss(alpha=None, gamma=2.0, reduction="mean",
                                           normalized=False, reduced_threshold=None)

    def components(self, logits, truth):
        probability = logits.sigmoid()
        return self.dice(probability, truth), self.focal(
            probability if self.baseline == "B0" else logits, truth)

    def forward(self, logits, truth):
        dice, focal = self.components(logits, truth)
        return dice + focal


@dataclass
class ImageDiceCheckpoint:
    best: float = -1.0
    epoch: int = 0
    bad_epochs: int = 0

    def update(self, value, epoch):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Checkpoint metric must be finite Image Dice in [0,1]")
        if value > self.best:
            self.best, self.epoch, self.bad_epochs = value, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    def state_dict(self):
        return asdict(self)


def boundary_iou(truth, prediction, diagonal_fraction=0.01):
    """Inner erosion band with padded zero exterior; radius round(1% diagonal), min 1."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("Boundary IoU requires matched 2D masks")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Boundary IoU requires binary masks")
    if not truth.any() or not prediction.any():
        return float(not truth.any() and not prediction.any())
    radius = max(1, round(diagonal_fraction * math.hypot(*truth.shape)))

    def band(mask):
        padded = np.pad(mask.astype(np.uint8), 1)
        eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=radius)[1:-1, 1:-1]
        return mask.astype(bool) & ~eroded.astype(bool)

    first, second = band(truth), band(prediction)
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 1.0


@lru_cache(maxsize=1)
def _pfom():
    # Reuse the tested CP1 implementation without editing the frozen module.
    return runpy.run_path(str(ROOT / "scripts/pfom_metrics.py"))["score_masks"]


class CP2Evaluator:
    def __init__(self, settings, boundaries=False):
        self.settings, self.boundaries = settings, boundaries
        self.region = SegmentationEvaluator(settings["prediction_threshold"])

    @property
    def rows(self):
        return self.region.rows

    def add(self, row, truth, probability):
        truth, probability = np.asarray(truth), np.asarray(probability)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Nonfinite/out-of-range predicted probabilities")
        if truth.ndim != 2 or truth.shape != probability.shape:
            raise ValueError("Expected matched 2D GT/probability grids")
        value = self.region.add(row["sample_id"], truth, probability)
        gt_pixels = int(np.count_nonzero(truth))
        fraction = gt_pixels / truth.size
        prediction = probability >= self.settings["prediction_threshold"]
        group = ("empty" if gt_pixels == 0 else "small" if fraction <= .01 else
                 "medium" if fraction <= .05 else "large")
        value.update(rgb_group=row["rgb_group"], gt_pixels=gt_pixels,
                     gt_fraction=fraction, size_group=group,
                     prediction_pixels=int(prediction.sum()),
                     total_foreground_miss=bool(gt_pixels and value["tp"] == 0),
                     gt_components_8_connected=int(label(truth, np.ones((3, 3)))[1]))
        if self.boundaries:
            value["boundary_iou"] = boundary_iou(
                truth, prediction, self.settings["boundary_radius_diagonal_fraction"])
            value["pfom"] = _pfom()(truth, prediction)["pfom"]
        return value

    def summary(self):
        metric_names = ["dice", "iou", "precision", "recall"]
        if self.boundaries:
            metric_names += ["boundary_iou", "pfom"]

        def summarize(rows):
            if not rows:
                return {"images": 0, "image_based": None, "data_based": None}
            counts = {k: sum(r[k] for r in rows) for k in ("tp", "tn", "fp", "fn")}
            return {"images": len(rows), "counts": counts, "data_based": scores(counts),
                    "image_based": {k: float(np.mean([r[k] for r in rows])) for k in metric_names},
                    "total_foreground_misses": sum(r["total_foreground_miss"] for r in rows),
                    "images_with_prediction": sum(r["prediction_pixels"] > 0 for r in rows)}

        return {"units": "fraction", "threshold_rule": "probability >= 0.5",
                "evaluator": "ordinary confusion counts; source offline_hard_line empty rules",
                "all": summarize(self.rows),
                "gt_positive": summarize([r for r in self.rows if r["gt_pixels"] > 0]),
                "gt_empty": summarize([r for r in self.rows if r["gt_pixels"] == 0]),
                "subgroups": {g: summarize([r for r in self.rows if r["size_group"] == g])
                              for g in ("small", "medium", "large")}}
