"""Engineering fixtures only; these tests do not produce research scores."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from foot_ulcer_segmentation.cp2_presence import (
    FirstStepAdam, PresenceModel, PresenceObjective, PresenceOutputs,
    presence_targets, select_presence,
)
from foot_ulcer_segmentation.cp2_protocol import BaselineObjective
from foot_ulcer_segmentation.cp2_phase_n import checkpoint_replay


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU())
        self.decoder = nn.Module()
        self.decoder.blocks = nn.ModuleList([nn.Identity() for _ in range(5)])
        self.head = nn.Conv2d(32, 1, 1)
        self.fail = False

    def forward(self, images):
        features = self.encoder(images)
        for block in self.decoder.blocks:
            features = block(features)
            if self.fail:
                raise RuntimeError("deliberate fixture failure")
        return F.interpolate(self.head(features), size=images.shape[-2:], mode="bilinear", align_corners=False)


def model():
    return PresenceModel(ToyBase(), nn.Linear(32, 1))


def test_target_follows_actual_augmented_mask_including_single_pixel():
    masks = torch.zeros(3, 1, 7, 9)
    masks[1, 0, 3, 5] = 1
    masks[2] = 1
    torch.testing.assert_close(presence_targets(masks), torch.tensor([[0.], [1.], [1.]]))
    cropped = masks[:, :, :2, :2]
    torch.testing.assert_close(presence_targets(cropped), torch.tensor([[0.], [0.], [1.]]))


@pytest.mark.parametrize("truth", [torch.zeros(2, 7, 9), torch.full((1, 1, 7, 9), .5),
                                  torch.full((1, 1, 7, 9), float("nan"))])
def test_invalid_presence_targets_are_rejected(truth):
    with pytest.raises(ValueError):
        presence_targets(truth)


def test_primary_forward_unchanged_with_head_enabled_removed_or_extreme():
    network = model().eval()
    inputs = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = network.base(inputs)
        output = network(inputs)
        assert output.presence_logits.shape == (2, 1)
        assert output.shape == expected.shape
        assert torch.equal(output.sigmoid(), expected.sigmoid())
        assert torch.equal(network(inputs, auxiliary=False).main_logits, expected)
        network.presence_head.weight.zero_()
        network.presence_head.bias.fill_(-100)
        assert torch.equal(network(inputs).main_logits, expected)
    assert not network.base.decoder.blocks[3]._forward_hooks


def test_temporary_feature_hook_removed_after_forward_failure():
    network = model()
    network.base.fail = True
    with pytest.raises(RuntimeError, match="deliberate"):
        network(torch.zeros(1, 3, 224, 224))
    assert not network.base.decoder.blocks[3]._forward_hooks


def test_objective_is_exact_b1_plus_unweighted_presence_bce():
    logits = torch.randn(3, 1, 7, 9)
    presence = torch.tensor([[-1.], [0.], [1.]])
    truth = torch.zeros_like(logits)
    truth[1, :, 0, 0] = 1
    truth[2] = 1
    parts = PresenceObjective().components(PresenceOutputs(logits, presence), truth)
    regional = BaselineObjective("B1")(logits, truth)
    bce = F.binary_cross_entropy_with_logits(presence, torch.tensor([[0.], [1.], [1.]]))
    assert torch.equal(parts["regional"], regional)
    assert torch.equal(parts["presence_bce"], bce)
    assert torch.equal(parts["total"], regional + .1 * bce)
    with pytest.raises(ValueError):
        PresenceObjective(.2)
    with pytest.raises(ValueError):
        PresenceObjective()(PresenceOutputs(logits, None), truth)


@pytest.mark.parametrize("empty", [False, True])
def test_actual_optimizer_step_finite_reaches_encoder_and_presence_head(empty):
    torch.manual_seed(42)
    network = model()
    optimizer = FirstStepAdam(network, lr=1e-4, weight_decay=1e-5)
    truth = torch.zeros(2, 1, 224, 224)
    if not empty:
        truth[:, :, 80:100, 50:65] = 1
    loss = PresenceObjective()(network(torch.randn(2, 3, 224, 224)), truth)
    loss.backward()
    for parameter in (next(network.base.parameters()), network.presence_head.weight, network.presence_head.bias):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    optimizer.step()
    assert optimizer.first_step_checks["finite_gradients"]
    assert all(optimizer.first_step_checks["updated_parameters"].values())


def test_diagnostic_optimizer_is_identical_to_standard_adam():
    torch.manual_seed(17)
    first = model()
    second = deepcopy(first)
    a = FirstStepAdam(first, lr=1e-4, weight_decay=1e-5)
    b = torch.optim.Adam(second.parameters(), lr=1e-4, weight_decay=1e-5)
    # Same arbitrary finite gradients isolate the optimizer update itself.
    for pa, pb in zip(first.parameters(), second.parameters()):
        pa.grad = torch.randn_like(pa)
        pb.grad = pa.grad.clone()
    a.step()
    b.step()
    assert all(torch.equal(pa, pb) for pa, pb in zip(first.parameters(), second.parameters()))


def test_checkpoint_replay_restores_both_predictions_and_preserves_mask_path(tmp_path):
    network = model().eval()
    filename = tmp_path / "N1.pth"
    torch.save({"state_dict": network.state_dict()}, filename)
    state = torch.load(filename, map_location="cpu")
    result = checkpoint_replay(network, state, torch.randn(2, 3, 224, 224), torch.device("cpu"))
    assert result["primary_and_presence_bit_exact"]
    assert not network.base.decoder.blocks[3]._forward_hooks


def report(dice=.85, small=.80, boundary=.55, recall=.92, empty=3, pixels=400, misses=0, split="validation"):
    return {"split": split, "checkpoint_sha256": "SYNTHETIC TEST FIXTURE",
            "all": {"image_based": {"dice": dice, "boundary_iou": boundary}},
            "subgroups": {"small": {"image_based": {"dice": small}}},
            "gt_positive": {"data_based": {"recall": recall}, "total_foreground_misses": misses},
            "gt_empty": {"images_with_prediction": empty, "counts": {"fp": pixels}}}


def test_selection_requires_validation_and_fp_improvement_with_old_guards():
    profile = {"active_forward_parameters": 100}
    baseline = report(dice=.84, small=.79, boundary=.54, recall=.91, empty=4, pixels=526)
    selected = select_presence(report(), baseline, profile, profile)
    assert selected["eligible"] and selected["selected_candidate"] == "N1"
    assert not selected["automatic_combinations"]
    for change in ({"recall": .90}, {"small": .78}, {"boundary": .53}, {"misses": 1},
                   {"empty": 5}, {"pixels": 527}, {"dice": .83}, {"empty": 4, "pixels": 526}):
        assert not select_presence(report(**change), baseline, profile, profile)["eligible"]
    assert not select_presence(report(), baseline, {"active_forward_parameters": 102}, profile)["eligible"]
    with pytest.raises(ValueError, match="validation"):
        select_presence(report(split="test"), baseline, profile, profile)


def test_frozen_protocol_and_no_extra_smoke_or_matrix_expansion():
    root = Path(__file__).resolve().parents[1]
    current = json.loads((root / "configs/cp2/phase_n.json").read_text())
    baseline = json.loads((root / "configs/cp2/phase_p.json").read_text())
    for key in ("seed", "split_seed", "cpu_threads", "data", "model", "training", "loss"):
        assert current[key] == baseline[key]
    assert current["phase_n"]["experiment_order"] == ["N1"]
    assert not current["phase_n"]["standalone_smoke_training"]
