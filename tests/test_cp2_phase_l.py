"""Engineering checks for validation-only selection and unchanged phase-P controls."""

import copy

import pytest

from foot_ulcer_segmentation.cp2_phase_l import select_candidate


def report(overall=.84, small=.77, misses=0, empty_fp=4, split="validation"):
    return {"split": split, "checkpoint_sha256": "synthetic-engineering-fixture",
            "all": {"image_based": {"dice": overall}},
            "subgroups": {"small": {"image_based": {"dice": small}}},
            "gt_positive": {"total_foreground_misses": misses},
            "gt_empty": {"images_with_prediction": empty_fp}}


def test_selection_rejects_metric_only_wins_with_predeclared_tradeoffs():
    control = report()
    candidates = {"L1": report(.85, .78), "L2": report(.95, .76),
                  "L3": report(.96, .80, misses=1), "L4": report(.86, .79)}
    result = select_candidate(candidates, control)
    assert result["validation_only"]
    assert result["selected_screening_candidate"] == "L4"
    assert not result["decisions"]["L2"]["eligible"]
    assert not result["decisions"]["L3"]["eligible"]


def test_selection_uses_cached_control_when_nothing_is_eligible():
    candidates = {name: report(.84, .77) for name in ("L1", "L2", "L3", "L4")}
    assert select_candidate(candidates, report())["selected_screening_candidate"] == "L0"


def test_selection_tie_is_predeclared_and_rejects_test_data():
    candidates = {name: report(.85, .78) for name in ("L1", "L2", "L3", "L4")}
    assert select_candidate(candidates, report())["selected_screening_candidate"] == "L1"
    contaminated = copy.deepcopy(candidates)
    contaminated["L1"]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_candidate(contaminated, report())
    with pytest.raises(ValueError, match="validation"):
        select_candidate(candidates, report(split="test"))
