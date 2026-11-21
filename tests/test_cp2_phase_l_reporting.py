"""Engineering fixtures for paired reporting and deterministic failure galleries."""

import copy
import importlib.util
from pathlib import Path

import pytest

from foot_ulcer_segmentation.cp2_phase_l_report import dice_contributions, format_metric


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engineering_rows():
    """Tiny invented rows for software tests only, never experiment outputs."""
    rows = {}
    for index in range(6):
        rows[f"fixture-{index}"] = {
            name: {"gt_pixels": "0" if index == 0 else str(index),
                   "dice": str(index / 10), "tp": "0" if index == 1 else "1",
                   "fp": "1" if index == 0 else "0",
                   "gt_components_8_connected": "2" if index == 3 else "1"}
            for name in ("L0", "L1", "L2", "L3", "L4")}
    return rows


def test_case_selection_is_deterministic_symmetric_and_includes_failures():
    select = load_script("cp2_phase_l_cases").select_cases
    rows = engineering_rows()
    expected = select(rows)
    assert select(dict(reversed(list(rows.items())))) == expected
    assert expected[2][1] == "fixture-1"  # Existing miss, not a best-case image.
    assert expected[3][1] == "fixture-0"  # Actual empty-GT false positive.
    assert expected[4][1] == "fixture-3"
    assert expected[5][1] == "fixture-1"
    permuted = {key: dict(zip(row, reversed(list(row.values()))))
                for key, row in copy.deepcopy(rows).items()}
    assert select(permuted) == expected  # No privileged candidate in median rule.


def test_missing_failure_case_is_explicit_not_fabricated():
    select = load_script("cp2_phase_l_cases").select_cases
    rows = engineering_rows()
    for row in rows.values():
        for values in row.values():
            values["tp"] = "1"
    assert select(rows)[2][1] is None


def test_csv_metric_validation_rejects_duplicates_and_changed_values():
    compare = load_script("validate_cp2_phase_l").compare_csv
    actual = [{"name": "fixture", "dice": .25, "tp": 2, "gt_empty": False}]
    saved = [{"name": "fixture", "dice": "0.25", "tp": "2", "gt_empty": "False"}]
    compare(saved, actual)
    with pytest.raises(ValueError, match="Duplicate/missing"):
        compare(saved * 2, actual * 2)
    changed = copy.deepcopy(saved)
    changed[0]["dice"] = "0.26"
    with pytest.raises(AssertionError):
        compare(changed, actual)


def test_undefined_aggregate_metrics_are_not_reported_as_zero():
    assert format_metric(None) == "undefined"
    assert format_metric(0) == "0.0000"


def grouped_report(positive, empty):
    return {"all": {"images": 3, "image_based": {"dice": (2 * positive + empty) / 3}},
            "gt_positive": {"images": 2, "image_based": {"dice": positive}},
            "gt_empty": {"images": 1, "image_based": {"dice": empty}}}


def test_positive_gain_can_be_outweighed_by_empty_gt_false_positive():
    parts = dice_contributions(grouped_report(.85, 0), grouped_report(.7, 1))
    assert parts["gt_positive"] == pytest.approx(.1)
    assert parts["gt_empty"] == pytest.approx(-1 / 3)
    assert parts["total"] == pytest.approx(.1 - 1 / 3)


def test_decomposition_rejects_changed_cohort():
    changed = grouped_report(.85, 0)
    changed["all"]["images"] = 4
    with pytest.raises(ValueError, match="same evaluation population"):
        dice_contributions(changed, grouped_report(.7, 1))


def test_decomposition_checks_against_canonical_overall_metric():
    changed = grouped_report(.85, 0)
    changed["all"]["image_based"]["dice"] = .9
    with pytest.raises(AssertionError):
        dice_contributions(changed, grouped_report(.7, 1))
