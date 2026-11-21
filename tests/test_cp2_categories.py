"""Analytic category fixtures; these are software tests, not research scores."""

import numpy as np
import pytest

from foot_ulcer_segmentation.cp2_categories import (
    author_cohort, author_membership, grouped_metrics,
)
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator


def categories():
    return {str(i): [] for i in range(10)}


def test_membership_refuses_duplicate_missing_keys_and_paths():
    mapping = categories()
    mapping["0"] = ["empty.png"]
    mapping["1"] = ["empty.png"]
    with pytest.raises(ValueError, match="Duplicate"):
        author_membership(mapping)
    with pytest.raises(ValueError, match="keys"):
        author_membership({"0": []})
    mapping["1"] = ["../other.png"]
    with pytest.raises(ValueError, match="filenames"):
        author_membership(mapping)


def test_cohort_accounts_for_exclusion_without_inventing_members():
    def row(name, pixels):
        return {"sample_id": "test/" + name, "name": name, "source_split": "test",
                "gt_pixels": pixels, "canvas_pixels": 100, "reason": "audit_conflict"}
    manifest = {"splits": {"test": [row("empty.png", 0), row("one.png", 1)]},
                "excluded": [row("two.png", 2)]}
    result = author_cohort(manifest, {"empty.png": 0, "one.png": 1, "two.png": 1})
    assert result[0]["eligible_images"] == 1
    assert result[1]["original_images"] == 2 and result[1]["excluded_images"] == 1
    assert result[1]["eligible_gt_fraction_min"] == .01
    assert result[2]["eligible_images"] == 0 and result[2]["eligible_gt_fraction_min"] is None
    with pytest.raises(ValueError, match="population"):
        author_cohort(manifest, {"empty.png": 0, "one.png": 1})


def evaluated_rows():
    evaluator = CP2Evaluator({"prediction_threshold": .5,
                             "boundary_radius_diagonal_fraction": .01}, boundaries=True)
    for name, truth, probability in [
        ("empty_good.png", [[0, 0, 0]], [[0., 0., 0.]]),
        ("empty_fp.png", [[0, 0, 0]], [[.5, 0., 0.]]),
        ("one.png", [[1, 0, 0]], [[.5, 0., 0.]]),
        ("three.png", [[1, 1, 1]], [[0., 0., 0.]]),
    ]:
        evaluator.add({"sample_id": "test/" + name, "rgb_group": name},
                      np.asarray(truth), np.asarray(probability))
    return evaluator.rows


def test_category_reduction_preserves_image_vs_data_metrics_and_empty_rules():
    rows = evaluated_rows()
    membership = {"empty_good.png": 0, "empty_fp.png": 0, "one.png": 1, "three.png": 1}
    result = grouped_metrics(rows, membership)
    assert result["C1"]["image_based"]["dice"] == .5
    assert result["C1"]["data_based"]["dice"] == .4
    assert result["C1"]["total_foreground_misses"] == 1
    assert result["C0"]["image_based"]["dice"] == .5
    assert result["C0"]["counts"]["fp"] == 1
    assert result["C0"]["images_with_prediction"] == 1
    assert result["C0"]["data_based"]["dice"] == 0
    assert result["C9"] == {"images": 0, "image_based": None, "data_based": None}
    assert grouped_metrics(rows[:1], membership)["C0"]["data_based"]["dice"] is None
    assert grouped_metrics(list(reversed(rows)), membership) == result


@pytest.mark.parametrize("failure", ["unmapped", "duplicate", "validation", "wrong_empty"])
def test_category_reduction_rejects_identity_and_label_errors(failure):
    rows = evaluated_rows()[:1]
    membership = {"empty_good.png": 0}
    if failure == "unmapped":
        membership = {}
    elif failure == "duplicate":
        rows = rows * 2
    elif failure == "validation":
        rows[0]["name"] = "validation/empty_good.png"
    else:
        membership["empty_good.png"] = 1
    with pytest.raises(ValueError):
        grouped_metrics(rows, membership)
