"""Synthetic engineering fixtures only; no new research predictions or scores."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from foot_ulcer_segmentation.cp2_attention import LocalSpatial
from foot_ulcer_segmentation.cp2_heads import AuxiliarySegmentationModel, HeadObjective
from foot_ulcer_segmentation.cp2_phase_c import MODELS, interaction, pairing, select
from foot_ulcer_segmentation.cp2_screening import CheckedAdam


class ToyLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_attention = LocalSpatial(32)

    def forward(self, x):
        # Simple fixture only; actual A1 fixed fusion is checked with the real model preflight.
        return x * (1 + self.local_attention(x))


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.decoder = nn.Module()
        self.decoder.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), ToyLocal(), ToyLocal(), nn.Identity()])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, images):
        x = self.encoder(images)
        for block in self.decoder.blocks:
            x = block(x)
        return F.interpolate(self.head(x), size=images.shape[-2:], mode="bilinear", align_corners=False)


def toy():
    return AuxiliarySegmentationModel(ToyBase(), nn.Conv2d(32, 1, 1))


@pytest.mark.parametrize("empty", [False, True])
def test_joint_gradients_update_both_local_branches_boundary_head_and_encoder(empty, tmp_path):
    torch.manual_seed(42)
    model = toy()
    checked = {n: p for n, p in model.named_parameters()
               if ".local_attention." in n or n.startswith("auxiliary_head.") or n == "base.encoder.weight"}
    optimizer = CheckedAdam(model, checked, tmp_path / "first.json", lr=1e-4, weight_decay=1e-5)
    truth = torch.zeros(2, 1, 224, 224)
    if not empty:
        truth[:, :, 70:95, 80:110] = 1
    output = model(torch.randn(2, 3, 224, 224))
    assert output.main_logits.shape == output.auxiliary_logits.shape == truth.shape
    HeadObjective("H2")(output, truth).backward()
    optimizer.step()
    assert optimizer.first_step_checks["finite_nonzero_gradients"]
    assert all(optimizer.first_step_checks["updated_parameters"].values())
    assert json.loads((tmp_path / "first.json").read_text()) == optimizer.first_step_checks
    assert not model.base.decoder.blocks[3]._forward_hooks


def test_diagnostic_adam_update_matches_standard_adam_exactly():
    torch.manual_seed(42)
    first, second = toy(), None
    second = deepcopy(first)
    a = CheckedAdam(first, {"encoder": first.base.encoder.weight}, None, lr=1e-4, weight_decay=1e-5)
    b = torch.optim.Adam(second.parameters(), lr=1e-4, weight_decay=1e-5)
    for p, q in zip(first.parameters(), second.parameters()):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    a.step()
    b.step()
    assert all(torch.equal(p, q) for p, q in zip(first.parameters(), second.parameters()))


@pytest.mark.parametrize("kind", ["missing", "zero", "nonfinite"])
def test_invalid_first_step_gradient_fails_before_update(kind):
    model = toy()
    parameter = model.base.encoder.weight
    optimizer = CheckedAdam(model, {"encoder": parameter}, None, lr=1e-4)
    if kind == "zero":
        parameter.grad = torch.zeros_like(parameter)
    if kind == "nonfinite":
        parameter.grad = torch.full_like(parameter, float("nan"))
    before = parameter.clone()
    with pytest.raises(ValueError, match="gradient"):
        optimizer.step()
    assert torch.equal(parameter, before)


def report(dice=.84, small=.77, boundary=.54, empty=4, misses=0, split="validation"):
    return {"split": split, "checkpoint_sha256": "SYNTHETIC FIXTURE",
            "all": {"image_based": {"dice": dice, "boundary_iou": boundary}, "data_based": {"dice": .91}},
            "gt_positive": {"image_based": {"dice": .87}, "data_based": {"recall": .91}, "total_foreground_misses": misses},
            "gt_empty": {"images_with_prediction": empty, "counts": {"fp": 526}},
            "subgroups": {"small": {"image_based": {"dice": small}}}}


def reports():
    return {"B1": report(), "A1": report(dice=.845, small=.80),
            "H2": report(dice=.848, small=.80), "A1H2": report(dice=.85, small=.81, boundary=.55)}


def test_original_guards_and_higher_than_individuals_are_both_required():
    r = reports()
    profiles = {n: {"active_forward_parameters": 100} for n in MODELS}
    decision = select(r, profiles)
    assert decision["eligible"] and decision["selected_candidate"] == "A1H2"
    assert not decision["automatic_expansion"] and decision["individual_historical_decisions_unchanged"]
    for change in ({"dice": .847}, {"small": .76}, {"boundary": .53}, {"empty": 5}, {"misses": 1}):
        candidate = {**r, "A1H2": report(dice=.85, **change)} if "dice" not in change else {**r, "A1H2": report(**change)}
        assert not select(candidate, profiles)["eligible"]
    profiles["A1H2"]["active_forward_parameters"] = 102
    assert not select(r, profiles)["eligible"]
    with pytest.raises(ValueError, match="VALIDATION"):
        select({**r, "A1H2": report(split="test")}, profiles)


def test_interaction_is_difference_of_effects_and_not_same_as_beating_individuals():
    r = reports()
    result = interaction(r)["overall_image_dice_pp"]
    assert result["A1H2_minus_A1"] > 0 and result["A1H2_minus_H2"] > 0
    assert result["interaction"] == pytest.approx(-.3)
    with pytest.raises(ValueError):
        interaction({**r, "A1H2": report(split="test")})


def test_actual_input_pairing_checks_all_condition_pairs():
    h = {n: [{"epoch": 1, "training_order_sha256": "same", "augmented_input_sha256": "same"}] for n in MODELS}
    assert len(pairing(h)["pairs"]) == 6
    h["A1H2"][0]["augmented_input_sha256"] = "different"
    with pytest.raises(ValueError, match="pairing"):
        pairing(h)


def test_config_preserves_main_training_loss_split_and_component_definitions():
    root = Path(__file__).resolve().parents[1]
    read = lambda p: json.loads((root / p).read_text())
    c, p = read("configs/cp2/phase_c.json"), read("configs/cp2/phase_p.json")
    for key in ("seed", "split_seed", "cpu_threads", "data", "model", "training", "loss"):
        assert c[key] == p[key]
    for phase in ("a", "h"):
        assert c[f"phase_{phase}"] == read(f"configs/cp2/phase_{phase}.json")[f"phase_{phase}"]
    assert c["phase_c"]["fresh_model"] == "A1H2" and c["phase_c"]["new_training_runs"] == 1
    assert not c["phase_c"]["standalone_smoke_training"]
