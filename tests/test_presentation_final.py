"""Final presentation checks: genuine pairing, stitch and deterministic selection."""

import numpy as np
import pytest

from foot_ulcer_segmentation.presentation import measure
from foot_ulcer_segmentation.presentation_final import (
    FinalEvidence, compare_values, miss_transitions, stitched_probability, summary,
)


def test_refiner_max_overlap_preserves_coarse_outside_roi():
    coarse = np.full((8, 8), .9, np.float32)
    # A lower refiner value must replace coarse INSIDE its ROI; this is not
    # maximum(coarse,refiner). Only overlaps BETWEEN refined crops take max.
    first, second = [np.full((224, 224), v, np.float32) for v in (.2, .7)]
    actual = stitched_probability(coarse, [[1, 1, 5, 5], [3, 3, 7, 7]], [first, second])
    assert actual[0, 0] == coarse[0, 0]
    assert actual[1, 1] == np.float32(.2)
    assert actual[3, 3] == np.float32(.7)
    assert actual[6, 6] == np.float32(.7)
    assert actual[7, 7] == coarse[7, 7]
    np.testing.assert_array_equal(stitched_probability(coarse, [], []), coarse)


def test_stitch_rejects_unpaired_crops_bad_geometry_or_nonfinite_output():
    coarse = np.zeros((8, 8), np.float32)
    crop = np.zeros((224, 224), np.float32)
    with pytest.raises(ValueError, match="association"):
        stitched_probability(coarse, [[0, 0, 3, 3]], [])
    with pytest.raises(ValueError, match="outside"):
        stitched_probability(coarse, [[-1, 0, 3, 3]], [crop])
    crop[0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        stitched_probability(coarse, [[0, 0, 3, 3]], [crop])


def test_miss_transitions_use_tp_zero_not_prediction_empty():
    def row(tp):
        return {"gt_pixels": 10, "tp": tp, "fp": 100}
    a = {"a": row(0), "b": row(1), "c": row(0), "d": row(3)}
    b = {"a": row(1), "b": row(0), "c": row(0), "d": row(3)}
    assert miss_transitions(a, b) == {"both_missed": 1, "recovered": 1, "newly_missed": 1, "overlap_both": 1}
    with pytest.raises(ValueError, match="populations"):
        miss_transitions(a, {"e": row(1)})
    b["a"]["gt_pixels"] = 11
    with pytest.raises(ValueError, match="aligned"):
        miss_transitions(a, b)


def test_exact_metric_comparison_rejects_changed_counts_and_subgroups():
    expected = {"all": {"dice": .8, "count": 12}, "empty": None}
    compare_values(expected, expected)
    with pytest.raises(ValueError, match="mismatch"):
        compare_values(expected, {"all": {"dice": .8, "count": 13}, "empty": None})
    with pytest.raises(ValueError, match="keys"):
        compare_values(expected, {"all": {"dice": .8, "count": 12}})


def test_summary_retains_empty_rule_and_pooled_data_dice():
    a = np.zeros((224, 224), bool)
    b = a.copy()
    b[0, 0] = True
    values = summary([measure(a, a), measure(b, a)])
    assert values["image_dice"] == .5
    assert values["data_dice"] == 0
    assert values["small_images"] == 1
    assert values["empty_images"] == 1
    assert values["misses"] == 1


def test_four_case_rules_have_unique_ids_and_stable_ties():
    evidence = FinalEvidence.__new__(FinalEvidence)
    rows = {name: {"name": name, "dice": score, "gt_fraction": fraction,
                   "gt_pixels": 1 if fraction else 0, "fp": fp}
            for name, score, fraction, fp in [("b", .2, .005, 0), ("a", .2, .005, 0),
                 ("standalone", .5, .02, 0), ("empty", 0, 0, 10), ("bad", .9, .02, 0)]}
    final = {k: {**v, "dice": 1 if k in ("a", "b") else .2 if k == "bad" else v["dice"]}
             for k, v in rows.items()}
    evidence.cascade_tables = {"validation": {"coarse": rows, "cascade": final}}
    evidence.tables = {n: {"validation": {k: {**v, "dice": .95 if k == "standalone" and n == "A6" else v["dice"]}
                       for k, v in rows.items()}} for n in ("B1", "A6", "H2")}
    selected = [sample for sample, _ in evidence.select_cases()]
    assert selected == ["a", "standalone", "empty", "bad"]
    assert len(set(selected)) == 4
    final["bad"]["dice"] = rows["bad"]["dice"]
    with pytest.raises(ValueError, match="No real case"):
        evidence.select_cases()
