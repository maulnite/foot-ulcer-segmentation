"""Metric, alignment and nonconcealing display checks for read-only presentation."""

import numpy as np
from PIL import Image
import pytest

from foot_ulcer_segmentation.presentation import (
    Evidence, aggregate, assert_metrics, binary_image, choose, display_bounds, measure, overlay,
)
from foot_ulcer_segmentation.presentation_replay import proposal_boxes


def test_locked_empty_rule_and_counts():
    truth = np.zeros((8, 8), bool)
    prediction = truth.copy()
    assert measure(truth, prediction)["dice"] == 1
    prediction[0, 0] = True
    row = measure(truth, prediction)
    assert row["dice"] == 0 and row["fp"] == 1 and row["fn"] == 0
    summary = aggregate([row])
    assert summary["empty_fp_images"] == 1 and summary["misses"] == 0


def test_positive_counts_and_metric_mismatch_rejected():
    truth = np.array([[1, 1], [0, 0]], bool)
    prediction = np.array([[1, 0], [1, 0]], bool)
    row = measure(truth, prediction)
    assert [row[k] for k in ["tp", "tn", "fp", "fn", "dice"]] == [1, 1, 1, 1, .5]
    assert_metrics(row, row)
    with pytest.raises(ValueError, match="count changed"):
        assert_metrics(row, {**row, "fp": 3})
    with pytest.raises(ValueError, match="metric changed"):
        assert_metrics(row, {**row, "dice": .6})


def test_zoom_retains_distant_false_positive_from_every_comparator():
    rgb = np.zeros((224, 224, 3), np.uint8)
    rgb[100:125, 100:125] = 100
    truth = np.zeros((224, 224), bool)
    truth[110, 110] = True
    a, b = truth.copy(), truth.copy()
    a[0, 0], b[-1, -1] = True, True
    assert display_bounds(rgb, truth, [a, b]) == (0, 0, 224, 224)
    with pytest.raises(ValueError, match="grids differ"):
        display_bounds(rgb, truth, [a[:4]])


def test_actual_overlay_confusion_colors_and_background():
    rgb = np.zeros((2, 2, 3), np.uint8)
    truth = np.array([[1, 1], [0, 0]], bool)
    prediction = np.array([[1, 0], [1, 0]], bool)
    output = overlay(rgb, truth, prediction)
    np.testing.assert_array_equal(output[0, 0], np.array([0, 158, 115])*.7 // 1)
    np.testing.assert_array_equal(output[0, 1], np.array([0, 114, 178])*.7 // 1)
    np.testing.assert_array_equal(output[1, 0], np.array([213, 94, 0])*.7 // 1)
    np.testing.assert_array_equal(output[1, 1], [0, 0, 0])


def test_deterministic_ties_exclusions_and_empty_selection():
    rows = [{"name": "b", "score": 1}, {"name": "a", "score": 1}]
    assert choose(rows, lambda r: True, lambda r: r["score"]) == "a"
    assert choose(rows, lambda r: True, lambda r: r["score"], excluded=["a"]) == "b"
    with pytest.raises(ValueError, match="No real case"):
        choose(rows, lambda r: False, lambda r: r["score"])


def test_nonbinary_or_disagreeing_mask_channels_rejected(tmp_path):
    path = tmp_path/"mask.png"
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(path)
    assert not binary_image(path).any()
    Image.fromarray(np.ones((2, 2), np.uint8)*128).save(path)
    with pytest.raises(ValueError, match="binary"):
        binary_image(path)
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0, 0] = 255
    Image.fromarray(rgb).save(path)
    with pytest.raises(ValueError, match="channels disagree"):
        binary_image(path)


def test_proposals_use_probability_only_and_preserve_source_order():
    namespace = {"PROP_THR": .3, "MIN_COMPONENT_PX": 4, "MAX_COMPONENTS": 5,
                 "CROP_MARGIN_EVAL": .5, "MIN_BOX_FRAC": .25,
                 "expand_box": lambda x0, y0, x1, y1, *args: (x0, y0, x1, y1)}
    p = np.zeros((20, 20), np.float32)
    p[2:4, 2:4] = .4
    p[10:13, 10:13] = .5
    p[18, 18] = 1  # below source component-area threshold
    assert proposal_boxes(namespace, p) == [(10, 10, 13, 13), (2, 2, 4, 4)]
    assert proposal_boxes(namespace, np.zeros_like(p)) == []


@pytest.mark.parametrize("status,fingerprint,message", [
    ("running", "locked", "not completed"),
    ("completed", "different", "Noncomparable cohorts"),
])
def test_incomplete_models_and_different_cohorts_cannot_enter_scoreboard(monkeypatch, status, fingerprint, message):
    from foot_ulcer_segmentation import presentation
    manifest = {"fingerprint": "locked", "splits": {"validation": [], "test": []}}
    report = {"run": "fake", "metrics": {"B1": {
        split: {"split_fingerprint": fingerprint} for split in ["validation", "test"]}}}
    state = {"status": status, "best_checkpoint_sha256": "checksum"}
    def fake_read(path):
        if path.name == "manifest.json":
            return manifest
        return state if path.name == "status.json" else report
    monkeypatch.setattr(presentation, "read", fake_read)
    monkeypatch.setattr(presentation, "digest_file", lambda path: "checksum")
    with pytest.raises(ValueError, match=message):
        Evidence({"manifest": "manifest.json", "families": [{"id": "P", "report": "report.json", "models": ["B1"]}]})
