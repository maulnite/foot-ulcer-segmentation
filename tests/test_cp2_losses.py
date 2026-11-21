"""Synthetic engineering fixtures, never presented as real segmentation scores."""

import copy
import json

import numpy as np
import pytest
import torch

from foot_ulcer_segmentation.common import ROOT
from foot_ulcer_segmentation.cp2_losses import (
    PhaseLObjective, distance_batch, signed_distance, tversky_loss,
)
from foot_ulcer_segmentation.cp2_protocol import BaselineObjective


@pytest.fixture
def settings():
    return json.loads((ROOT / "configs/cp2/phase_l.json").read_text())["phase_l"]


def test_cached_control_objective_is_bit_exact_b1(settings):
    logits = torch.linspace(-6, 6, 32).reshape(2, 1, 4, 4).requires_grad_()
    truth = (torch.arange(32).reshape_as(logits) % 3 == 0).float()
    actual = PhaseLObjective("L0", settings)(logits, truth)
    expected = BaselineObjective("B1")(logits, truth)
    assert torch.equal(actual, expected)
    first = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    second = torch.autograd.grad(expected, logits)[0]
    assert torch.equal(first, second)


@pytest.mark.parametrize("q", [1., .75])
def test_soft_tversky_matches_analytical_batch_counts(q):
    probability = torch.tensor([.8, .4, .1, .7], dtype=torch.float64).reshape(2, 1, 1, 2)
    truth = torch.tensor([1., 0., 0., 1.], dtype=torch.float64).reshape_as(probability)
    tp, fp, fn = 1.5, .5, .5
    error = 1 - (tp + 1) / (tp + .3 * fp + .7 * fn + 1)
    expected = (error + 1e-6) ** q - (1e-6) ** q
    actual = tversky_loss(probability, truth, alpha=.3, beta=.7, epsilon=1, q=q, delta=1e-6)
    assert float(actual) == pytest.approx(expected, abs=1e-12)


def test_greater_fn_weight_penalizes_equal_false_negative_more():
    truth = torch.tensor([1., 0.]).reshape(1, 1, 1, 2)
    fp = torch.tensor([1., .5]).reshape_as(truth)
    fn = torch.tensor([.5, 0.]).reshape_as(truth)
    kwargs = dict(alpha=.3, beta=.7, epsilon=1, q=1, delta=1e-6)
    assert tversky_loss(fn, truth, **kwargs) > tversky_loss(fp, truth, **kwargs)


@pytest.mark.parametrize("experiment", ["L1", "L2", "L3", "L4"])
@pytest.mark.parametrize("target", [0., 1.])
def test_empty_full_and_saturated_gradients_are_finite(settings, experiment, target):
    logits = torch.tensor([-80., -3., 0., 3., 80.]).reshape(1, 1, 1, 5).requires_grad_()
    truth = torch.full_like(logits, target)
    objective = PhaseLObjective(experiment, settings)
    loss = objective(logits, truth)
    gradient = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    if target == 0:
        assert gradient[0, 0, 0, 3] > 0


def test_focal_power_has_no_infinite_derivative_at_exact_perfection():
    probability = torch.ones(1, 1, 3, 3, requires_grad=True)
    truth = torch.ones_like(probability)
    loss = tversky_loss(probability, truth, alpha=.3, beta=.7, epsilon=1, q=.75, delta=1e-6)
    assert float(loss) == pytest.approx(0, abs=1e-10)
    assert torch.isfinite(torch.autograd.grad(loss, probability)[0]).all()


def test_signed_distance_convention_and_empty_policy():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    distance = signed_distance(mask)
    assert distance[3, 3] == -1
    assert distance[2, 3] == 0
    assert distance[1, 3] == 1
    assert distance[0, 0] == pytest.approx(np.sqrt(8))
    assert distance.dtype == np.float32
    assert not signed_distance(np.zeros_like(mask)).any()


def test_distance_target_is_computed_from_augmented_geometry():
    truth = torch.zeros(1, 1, 9, 9)
    truth[:, :, 1:6, 1:6] = 1
    augmented = torch.flip(truth, [-1])
    actual = distance_batch(augmented)
    np.testing.assert_array_equal(actual[0, 0].numpy(), signed_distance(augmented[0, 0].numpy()))
    assert actual.shape == truth.shape
    assert actual.dtype == truth.dtype and not actual.requires_grad
    assert not torch.equal(actual, distance_batch(truth))


def test_boundary_addition_only_and_signed_negative_allowed(settings):
    logits = torch.full((1, 1, 7, 7), -3.)
    truth = torch.zeros_like(logits)
    truth[:, :, 1:6, 1:6] = 1
    logits[:, :, 1:6, 1:6] = 3.
    parts = PhaseLObjective("L3", settings).components(logits, truth)
    assert torch.equal(parts["regional"], BaselineObjective("B1")(logits, truth))
    assert float(parts["boundary_raw"]) < 0
    assert torch.equal(parts["total"], parts["regional"] + .01 * parts["boundary_raw"])
    no_boundary = copy.deepcopy(settings)
    no_boundary["boundary"]["weight"] = 0
    assert torch.equal(PhaseLObjective("L3", no_boundary)(logits, truth), parts["regional"])


def test_invalid_distance_input_is_rejected():
    with pytest.raises(ValueError):
        signed_distance(np.full((3, 3), .5))
    with pytest.raises(ValueError):
        distance_batch(torch.zeros(2, 3, 4, 4))
