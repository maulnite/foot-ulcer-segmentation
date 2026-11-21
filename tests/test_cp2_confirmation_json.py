"""Regression: persisted JSON decisions must compare without changing metrics."""

import math
from pathlib import Path
import runpy
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
helpers = runpy.run_path(str(ROOT / 'scripts/finish_cp2_confirmation.py'))
json_value = helpers['json_value']
json_safe_validator = helpers['json_safe_validator']


def test_nested_integer_seed_keys_survive_persistence_comparison():
    decision = {'decisions': {'H2': {'per_seed_guards': {42: {'ok': False}, 43: {'ok': True}},
                                   'deltas': {42: .006, 43: -.005}}}}
    persisted = json_value(decision)
    assert decision != persisted
    assert json_value(decision) == persisted
    assert persisted['decisions']['H2']['deltas']['43'] == -.005


def test_normalization_cannot_hide_actual_metric_or_selection_changes():
    original = {'model': 'B1', 'means': {'B1': .844}, 'seeds': {42: True}}
    tampered = {'model': 'H2', 'means': {'B1': .845}, 'seeds': {'42': True}}
    assert json_value(original) != tampered
    with pytest.raises(ValueError):
        json_value({'score': math.nan})


def test_validator_code_and_original_globals_remain_unchanged():
    def frozen_check():
        return select() == {'seeds': {'42': True}}  # noqa: F821
    selector = lambda: {'seeds': {42: True}}
    original = types.FunctionType(frozen_check.__code__, {'select': selector})
    corrected = json_safe_validator(original)
    assert not original()
    assert corrected()
    assert corrected.__code__ is original.__code__
    assert original.__globals__['select'] is selector
