"""Engineering fixtures only; synthetic arrays are never research results."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from foot_ulcer_segmentation.cp2_confirmation import (
    RegionalObjective, check_pairing, fp_geometry, guard_checks, seed_config, select,
)
from foot_ulcer_segmentation.cp2_protocol import BaselineObjective

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('empty', [True, False])
def test_regional_dictionary_interface_preserves_b1_loss_and_gradients(empty):
    logits = torch.linspace(-2, 2, 128).reshape(2, 1, 8, 8).requires_grad_()
    truth = torch.zeros_like(logits)
    if not empty:
        truth[:, :, 2:5, 1:7] = 1
    loss = RegionalObjective()(logits, truth)
    expected = BaselineObjective('B1')(logits, truth)
    assert torch.equal(loss, expected)
    actual_grad, = torch.autograd.grad(loss, logits, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, logits)
    assert torch.equal(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0


def test_fp_geometry_threshold_components_and_empty_rule():
    probability = np.zeros((8, 8), dtype=np.float32)
    probability[0, 0] = .5
    probability[4:6, 4:6] = .7
    result = fp_geometry(probability >= .5, probability)
    assert result['fp_pixels'] == 5
    assert result['components_8_connected'] == 2
    assert result['largest_component_pixels'] == 4
    assert result['border_3px_fp_pixels'] == 4
    assert fp_geometry(np.zeros((8, 8)), np.zeros((8, 8)))['centroid_x_fraction'] is None
    with pytest.raises(ValueError, match='threshold'):
        fp_geometry(probability > .5, probability)


def report(dice, empty=4, split='validation'):
    return {'split': split, 'all': {'image_based': {'dice': dice, 'boundary_iou': .6, 'pfom': .7},
                                  'data_based': {'dice': .9}},
            'subgroups': {'small': {'image_based': {'dice': .8}}},
            'gt_positive': {'total_foreground_misses': 0},
            'gt_empty': {'images_with_prediction': empty, 'counts': {'fp': 500}}}


def selection_fixture():
    protocol = json.loads((ROOT / 'configs/cp2/confirmation.json').read_text())
    profiles = {n: {'active_forward_parameters': 100} for n in ('B1', 'H2', 'A6')}
    validation = {s: {'B1': report(.8), 'H2': report(.82), 'A6': report(.81)} for s in (42, 43, 44)}
    return protocol, profiles, validation


def test_selection_never_reads_test_or_selects_a_favorable_seed_subset():
    protocol, profiles, validation = selection_fixture()
    assert select(validation, profiles, protocol)['selected_candidate'] == 'H2'
    del validation[44]
    with pytest.raises(ValueError, match='All registered seeds'):
        select(validation, profiles, protocol)
    validation[44] = {n: report(.9, split='test') for n in profiles}
    with pytest.raises(ValueError, match='never read test'):
        select(validation, profiles, protocol)


def test_seed42_failure_not_hidden_and_two_full_guard_passes_required():
    protocol, profiles, validation = selection_fixture()
    validation[42]['H2'] = report(.82, empty=5)
    validation[43]['H2'] = report(.82, empty=3)
    decision = select(validation, profiles, protocol)
    assert not decision['decisions']['H2']['per_seed_guards'][42]['no_extra_empty_fp']
    assert decision['decisions']['H2']['full_guard_seed_passes'] == 2
    assert decision['decisions']['H2']['eligible']
    validation[44]['H2'] = report(.82, empty=5)
    decision = select(validation, profiles, protocol)
    assert not decision['decisions']['H2']['eligible']
    assert not decision['decisions']['H2']['mean_guards']['no_extra_empty_fp']


def test_large_gain_cannot_override_empty_false_positive_guard():
    baseline = {'image_dice': .8, 'small_dice': .7, 'boundary_iou': .5, 'misses': 0, 'empty_fp_images': 4}
    candidate = {**baseline, 'image_dice': .99, 'empty_fp_images': 5}
    checks = guard_checks(candidate, baseline, 1)
    assert checks['higher_overall_dice'] and not checks['no_extra_empty_fp']
    assert not all(checks.values())


def test_seed_config_preserves_split_and_every_numerical_training_setting(tmp_path):
    protocol = json.loads((ROOT / 'configs/cp2/confirmation.json').read_text())
    (tmp_path / 'initialization.json').write_text(json.dumps({'tensor_fingerprint': 'fixture'}))
    config = seed_config(protocol, 43, tmp_path)
    original = json.loads((ROOT / protocol['base_config']).read_text())
    assert config['seed'] == 43 and config['split_seed'] == 42
    assert config['data'] == original['data'] and config['loss'] == original['loss']
    for key, value in original['training'].items():
        if key != 'epoch_rng_seed_rule':
            assert config['training'][key] == value
    assert config['phase_h']['head_seed'] == 30046
    assert config['phase_a']['modified_stages'] == [2, 3]
    assert config['phase_a']['control_initial_tensor_sha256'] == 'fixture'


def test_actual_inputs_must_match_not_only_seed_and_parameter_names(tmp_path):
    protocol, _, _ = selection_fixture()
    for name in ('B1', 'H2', 'A6'):
        directory = tmp_path / 'seed-43' / name
        directory.mkdir(parents=True)
        (directory / 'history.json').write_text(json.dumps([{'training_order_sha256': 'same', 'augmented_input_sha256': 'same'}]))
        (directory / 'initialization_verified.json').write_text(json.dumps({'source_tensor_sha256': 'same'}))
    assert check_pairing(tmp_path, protocol, 43)['status'] == 'passed'
    file = tmp_path / 'seed-43/A6/history.json'
    data = json.loads(file.read_text())
    data[0]['augmented_input_sha256'] = 'different'
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='augmented bytes'):
        check_pairing(tmp_path, protocol, 43)
