"""Protocol and RNG tests use engineering fixtures, never scientific scores."""

import copy
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, BatchSampler

from foot_ulcer_segmentation.checkpoint import AuthorCheckpointPolicy, random_state, restore_random_state
from foot_ulcer_segmentation.common import ROOT, write_json
from foot_ulcer_segmentation.cp2_heads import HeadObjective, HeadOutputs
from foot_ulcer_segmentation.cp2_recipe import (
    EXPERIMENTS, RecipeObjective, RecordingBatchSampler, validate_config, verify_paired_inputs,
)
from foot_ulcer_segmentation.engine import setup as initialize_rng


def test_configuration_is_bounded_and_preserves_the_registered_controls():
    config = json.loads((ROOT / "configs/cp2/cp1_recipe.json").read_text())
    validate_config(config)
    for keys, value in [(('recipe', 'main_loss'), 'B0'), (('recipe', 'experiment_order'), ['R0']),
                        (('training', 'batch_size'), 4), (('training', 'amp'), True),
                        (('training', 'checkpoint_policy'), 'highest_dice')]:
        changed = copy.deepcopy(config)
        changed[keys[0]][keys[1]] = value
        with pytest.raises(ValueError):
            validate_config(changed)
    changed = copy.deepcopy(config)
    changed['training']['scheduler']['mode'] = 'max'
    with pytest.raises(ValueError):
        validate_config(changed)


def test_recording_batch_sampler_matches_cp1_order_and_global_rng_exactly():
    dataset = TensorDataset(torch.arange(19))
    initialize_rng(42, 2)
    original = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    initial = random_state()
    expected = [torch.cat([x[0] for x in original]) for _ in range(3)]
    expected_rng = torch.get_rng_state()
    restore_random_state(initial)
    sampler = RecordingBatchSampler(BatchSampler(RandomSampler(dataset), 2, False))
    observed = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    for order in expected:
        assert torch.equal(torch.cat([x[0] for x in observed]), order)
        assert sampler.indices == order.tolist()
        assert len(sampler) == 10  # Last singleton retained.
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize('name', EXPERIMENTS)
@pytest.mark.parametrize('empty', [False, True])
def test_main_loss_remains_b1_and_validation_iou_uses_source_strict_threshold(name, empty):
    logits = torch.linspace(-1, 1, 42).reshape(2, 1, 3, 7).requires_grad_()
    truth = torch.zeros_like(logits)
    if not empty:
        truth[:, :, 1, 2:5] = 1
    auxiliary = torch.zeros_like(logits, requires_grad=True)
    outputs = HeadOutputs(logits, auxiliary if name == 'RH2' else None)
    objective = RecipeObjective(name)
    parts = objective.components(outputs, truth)
    expected = HeadObjective('H2' if name == 'RH2' else 'H0').components(outputs, truth)
    assert all(torch.equal(parts[k], v) for k, v in expected.items())
    prediction = (logits.detach().sigmoid() > .5).float()
    intersection = (prediction * truth).sum()
    union = prediction.sum() + truth.sum() - intersection
    assert torch.equal(parts['source_iou'], (intersection + 1e-7) / (union + 1e-7))
    parts['total'].backward()
    assert torch.isfinite(logits.grad).all()
    if name == 'RH2':
        assert torch.isfinite(auxiliary.grad).all()
        assert parts['total'] > parts['regional']


def test_source_checkpoint_rule_does_not_silently_become_highest_image_dice():
    policy = AuthorCheckpointPolicy()
    assert policy.update(1, .9)
    assert policy.best_iou == 0  # Preserve the original if/elif behavior.
    assert policy.update(1.1, .8)
    assert not policy.update(1.2, .7)
    assert policy.patience_count == 1


def test_actual_data_hash_verification_detects_common_epoch_mismatch(tmp_path):
    history = [{'epoch': 1, 'training_order_sha256': 'order', 'augmented_input_sha256': 'bytes'}]
    for name in EXPERIMENTS:
        write_json(tmp_path / name / 'history.json', history)
    assert verify_paired_inputs(tmp_path)['actual_order_and_augmented_bytes_equal']
    write_json(tmp_path / 'RH2/history.json', [{**history[0], 'augmented_input_sha256': 'different'}])
    with pytest.raises(ValueError, match='Actual common-epoch'):
        verify_paired_inputs(tmp_path)


def test_epoch_boundary_global_rng_resume_preserves_next_loader_order():
    dataset = TensorDataset(torch.arange(19))
    initialize_rng(42, 2)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    list(loader)
    saved = random_state()
    expected = torch.cat([x[0] for x in loader])
    torch.rand(100)
    restore_random_state(saved)
    restarted = DataLoader(dataset, batch_size=2, shuffle=True)
    actual = torch.cat([x[0] for x in restarted])
    assert torch.equal(actual, expected)
