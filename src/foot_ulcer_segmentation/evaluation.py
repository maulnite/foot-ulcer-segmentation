"""Separately labeled offline and legacy-inline binary segmentation metrics."""

import numpy as np


def confusion_counts(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape:
        raise ValueError("Prediction and ground truth shapes differ")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Evaluation requires binary masks; select a mask policy explicitly")
    truth, prediction = truth.astype(bool), prediction.astype(bool)
    return {"tp": int(np.count_nonzero(truth & prediction)),
            "tn": int(np.count_nonzero(~truth & ~prediction)),
            "fp": int(np.count_nonzero(~truth & prediction)),
            "fn": int(np.count_nonzero(truth & ~prediction))}


def scores(counts):
    tp, fp, fn = (counts[k] for k in ["tp", "fp", "fn"])
    denominator = 2 * tp + fp + fn
    return {"dice": 2 * tp / denominator if denominator else None,
            "iou": tp / (tp + fp + fn + 1e-6),
            "precision": tp / (tp + fp + 1e-6),
            "recall": tp / (tp + fn + 1e-6)}


class SegmentationEvaluator:
    """Offline HARD_LINE=True as primary, with inline-source counts for comparison.

    Values are fractions, not percentages. Undefined aggregate Dice is null.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.rows = []
        self.offline = {k: 0 for k in ["tp", "tn", "fp", "fn"]}
        self.inline = dict(self.offline)

    def add(self, name, truth, probability):
        prediction = np.asarray(probability) >= self.threshold
        counts = confusion_counts(truth, prediction)
        inline_counts = ({"tp": prediction.size, "tn": 0, "fp": 0, "fn": 0}
                         if np.array_equal(truth, prediction) else counts)
        per_image = scores(counts)
        if not np.count_nonzero(truth):
            value = 0.0 if np.count_nonzero(prediction) else 1.0
            per_image = {k: value for k in per_image}
        for key in self.offline:
            self.offline[key] += counts[key]
            self.inline[key] += inline_counts[key]
        row = {"name": name, **counts, **per_image}
        self.rows.append(row)
        return row

    def summary(self):
        if not self.rows:
            raise ValueError("Cannot evaluate an empty collection")
        means = {key: float(np.mean([row[key] for row in self.rows]))
                 for key in ["dice", "iou", "precision", "recall"]}
        return {"images": len(self.rows), "units": "fraction",
                "prediction_rule": f"probability >= {self.threshold}",
                "offline_hard_line": {"counts": self.offline, "data_based": scores(self.offline),
                                      "image_based": means},
                "author_inline_legacy": {"counts": self.inline, "data_based": scores(self.inline)},
                "undefined_aggregate_dice": "null when the entire collection has no positive GT or prediction"}
