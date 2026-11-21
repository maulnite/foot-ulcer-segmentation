"""Engineering tests for attention math, matching, gates and isolated scope."""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from foot_ulcer_segmentation.cp2_attention import (
    CapacityBlock, ImageGate, LocalSpatial, attach_attention, fusion_weights,
)
from foot_ulcer_segmentation.cp2_phase_a import FixedB1Objective, expansion_decision, select_candidate
from foot_ulcer_segmentation.cp2_protocol import BaselineObjective


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/cp2/phase_a.json").read_text())


class DummyBlock(nn.Module):
    """Tiny structural fixture, never used for scientific training."""
    def __init__(self, channels):
        super().__init__()
        self.attention_type = "pscse"
        self.conv1 = nn.Sequential(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x


def structural_model():
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.blocks = nn.ModuleList([DummyBlock(c) for c in (864, 336, 176, 128, 32)])
    return model


@pytest.mark.parametrize("channels", [16, 128, 176])
def test_local_map_has_one_channel_and_finite_gradients(channels):
    layer = LocalSpatial(channels)
    x = torch.randn(2, channels, 7, 9, requires_grad=True)
    mapping = layer(x)
    assert mapping.shape == (2, 1, 7, 9)
    assert ((mapping >= 0) & (mapping <= 1)).all()
    (x * mapping).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_new_modules_leave_shared_rng_source_tensors_and_last_stage_untouched():
    model = structural_model()
    before = {n: v.clone() for n, v in model.state_dict().items()}
    last = model.decoder.blocks[4].forward.__func__
    rng = torch.get_rng_state().clone()
    attach_attention(model, "A3", config())
    assert torch.equal(rng, torch.get_rng_state())
    for n, value in before.items():
        assert torch.equal(model.state_dict()[n], value)
    assert model.decoder.blocks[4].forward.__func__ is last
    for stage in (0, 1, 4):
        assert not hasattr(model.decoder.blocks[stage], "phase_a_variant")
    with pytest.raises(ValueError, match="untouched"):
        attach_attention(model, "A3", config())


def test_all_three_fusions_have_identical_initial_weights_and_local_tensors():
    weights, local = [], []
    x = torch.randn(2, 176, 7, 9)
    for variant in ("A1", "A2", "A3"):
        m = attach_attention(structural_model(), variant, config())
        b = m.decoder.blocks[2]
        weights.append(fusion_weights(b, x))
        local.append(b.local_attention.state_dict())
    assert torch.equal(weights[0], weights[1]) and torch.equal(weights[0], weights[2])
    for name in local[0]:
        assert torch.equal(local[0][name], local[1][name]) and torch.equal(local[0][name], local[2][name])
    torch.testing.assert_close(weights[0].sum(1), torch.ones(2))
    torch.testing.assert_close(weights[0][0], torch.tensor([4/9, 2/9, 1/3]))


def test_gate_can_change_with_input_after_learning_and_drop_local_is_normalized():
    m = attach_attention(structural_model(), "A3", config())
    b = m.decoder.blocks[2]
    with torch.no_grad():
        b.image_gate.hidden.weight.fill_(.01)
        b.image_gate.hidden.bias.zero_()
        b.image_gate.output.weight[0].fill_(.1)
    x = torch.stack([torch.zeros(176, 4, 4), torch.ones(176, 4, 4)])
    weights = fusion_weights(b, x)
    assert not torch.equal(weights[0], weights[1])
    b.phase_a_intervention = "drop_local"
    weights = fusion_weights(b, x)
    assert torch.equal(weights[:, 2], torch.zeros(2))
    torch.testing.assert_close(weights.sum(1), torch.ones(2))
    b.phase_a_intervention = "fixed_prior_gate"
    assert torch.equal(fusion_weights(b, x)[0], fusion_weights(b, x)[1])


def test_capacity_correction_starts_at_zero_without_a_multiplicative_attention_map():
    block = CapacityBlock(176)
    output = block(torch.randn(2, 176, 7, 9))
    assert torch.equal(output, torch.zeros_like(output))
    assert not any(isinstance(m, nn.Sigmoid) for m in block.modules())


@pytest.mark.parametrize("empty", [True, False])
def test_phase_a_loss_and_gradients_match_fixed_b1(empty):
    x = torch.randn(2, 1, 7, 9, requires_grad=True)
    y = torch.zeros_like(x) if empty else (torch.rand_like(x) > .8).float()
    actual = FixedB1Objective()(x, y)
    expected = BaselineObjective("B1")(x, y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    first, = torch.autograd.grad(actual, x, retain_graph=True)
    second, = torch.autograd.grad(expected, x)
    assert torch.equal(first, second) and torch.isfinite(first).all()


def report(dice=.843, small=.773, boundary=.540, misses=0, empty=4, split="validation"):
    return {"split": split, "checkpoint_sha256": "SYNTHETIC-engineering-fixture",
            "all": {"image_based": {"dice": dice, "boundary_iou": boundary}},
            "subgroups": {"small": {"image_based": {"dice": small}}},
            "gt_positive": {"total_foreground_misses": misses}, "gt_empty": {"images_with_prediction": empty}}


def test_expansion_gate_distinguishes_feasibility_from_improvement_and_rejects_test():
    assert expansion_decision(report(.83, .76, empty=5), report(), config())["expand"]
    assert not expansion_decision(report(.80), report(), config())["expand"]
    assert not expansion_decision(report(empty=7), report(), config())["expand"]
    with pytest.raises(ValueError, match="source-test"):
        expansion_decision(report(split="test"), report(), config())


def test_selection_preserves_baseline_unless_all_predeclared_guards_pass():
    profiles = {n: {"active_forward_parameters": 100} for n in ("A0", "A1", "A2", "A3", "A5", "A6")}
    candidates = {"A1": report(.85, boundary=.53), "A2": report(.86, empty=5),
                  "A3": report(.85), "A5": report(.9)}
    profiles["A5"]["active_forward_parameters"] = 102
    selection = select_candidate(candidates, report(), profiles)
    assert selection["selected_candidate"] == "A3"
    assert not selection["decisions"]["A1"]["eligible"]
    assert not selection["decisions"]["A5"]["eligible"]
    assert select_candidate({"A1": report()}, report(), profiles)["selected_candidate"] == "A0"
    assert select_candidate({"A1": report(.85), "A2": report(.85)}, report(), profiles)["selected_candidate"] == "A1"
    with pytest.raises(ValueError, match="validation"):
        select_candidate({"A1": report(split="test")}, report(), profiles)


def test_stage_placement_is_not_expanded_silently():
    changed = copy.deepcopy(config())
    changed["phase_a"]["modified_stages"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Stage-placement"):
        attach_attention(structural_model(), "A1", changed)


def test_gain_configuration_cannot_silently_disagree_with_implementation():
    changed = copy.deepcopy(config())
    changed["phase_a"]["fusion"]["scale"] = 6
    with pytest.raises(ValueError, match="Fusion gain"):
        attach_attention(structural_model(), "A1", changed)
