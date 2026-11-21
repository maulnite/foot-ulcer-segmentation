"""Synthetic engineering fixtures; never scientific combination scores."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from foot_ulcer_segmentation.common import digest_json
from foot_ulcer_segmentation.cp2_attention import CBAM
from foot_ulcer_segmentation.cp2_combinations import (
    CACHED, FRESH, CombinationObjective, contrasts, run_config, select,
)
from foot_ulcer_segmentation.cp2_heads import AuxiliarySegmentationModel, HeadObjective, HeadOutputs
from foot_ulcer_segmentation.cp2_losses import PhaseLObjective, distance_batch
from foot_ulcer_segmentation.cp2_screening import CheckedAdam


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/cp2/phase_c_combinations.json").read_text())


@pytest.mark.parametrize("empty", [False, True])
def test_B1_combination_objective_and_gradients_match_historical_H2(empty, config):
    torch.manual_seed(42)
    main, aux = torch.randn(2, 1, 12, 12, requires_grad=True), torch.randn(2, 1, 12, 12, requires_grad=True)
    truth = torch.zeros_like(main)
    if not empty:
        truth[:, :, 3:6, 4:8] = 1
    outputs = HeadOutputs(main, aux)
    old = HeadObjective("H2")(outputs, truth)
    new = CombinationObjective("B1A6H2", config["phase_l"])(outputs, truth)
    assert torch.equal(old, new)
    left = torch.autograd.grad(old, (main, aux), retain_graph=True)
    right = torch.autograd.grad(new, (main, aux))
    assert all(torch.equal(a, b) for a, b in zip(left, right))


@pytest.mark.parametrize("empty", [False, True])
def test_L3_changes_only_main_signed_distance_term_and_keeps_auxiliary_identical(empty, config):
    torch.manual_seed(42)
    outputs = HeadOutputs(torch.randn(2, 1, 12, 12, requires_grad=True), torch.randn(2, 1, 12, 12, requires_grad=True))
    truth = torch.zeros_like(outputs.main_logits)
    if not empty:
        truth[:, :, 3:6, 4:8] = 1
    a = CombinationObjective("B1A6H2", config["phase_l"]).components(outputs, truth)
    b = CombinationObjective("L3A6H2", config["phase_l"]).components(outputs, truth)
    expected = PhaseLObjective("L3", config["phase_l"]).components(outputs.main_logits, truth)
    assert torch.equal(b["main_total"], expected["total"])
    for key in ("dice", "focal", "regional", "auxiliary_dice", "auxiliary_focal", "weighted_auxiliary"):
        assert torch.equal(a[key], b[key])
    boundary = .01 * (outputs.main_logits.sigmoid() * distance_batch(truth)).mean()
    torch.testing.assert_close(b["total"] - a["total"], boundary, atol=2e-7, rtol=0)
    ga = torch.autograd.grad(a["total"], outputs.auxiliary_logits, retain_graph=True)[0]
    gb = torch.autograd.grad(b["total"], outputs.auxiliary_logits)[0]
    assert torch.equal(ga, gb)
    if empty:
        assert torch.equal(a["total"], b["total"])


def test_missing_head_or_modified_boundary_weight_rejected(config):
    objective = CombinationObjective(FRESH[0], config["phase_l"])
    truth = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="auxiliary"):
        objective(HeadOutputs(truth, None), truth)
    altered = deepcopy(config["phase_l"])
    altered["boundary"]["weight"] = .02
    with pytest.raises(ValueError, match="0.01"):
        CombinationObjective(FRESH[1], altered)


class ToyCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbam = CBAM(32)

    def forward(self, x):
        return 3 * self.cbam(x)


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.decoder = nn.Module()
        self.decoder.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), ToyCBAM(), ToyCBAM(), nn.Identity()])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        shape = x.shape[-2:]
        x = self.encoder(x)
        for block in self.decoder.blocks:
            x = block(x)
        return F.interpolate(self.head(x), size=shape, mode="bilinear", align_corners=False)


@pytest.mark.parametrize("variant", FRESH)
@pytest.mark.parametrize("empty", [False, True])
def test_actual_CBAM_H2_modules_receive_finite_gradients_and_updates(variant, empty, config, tmp_path):
    torch.manual_seed(42)
    model = AuxiliarySegmentationModel(ToyBase(), nn.Conv2d(32, 1, 1))
    checked = {n: p for n, p in model.named_parameters() if n.startswith("auxiliary_head.") or n == "base.encoder.weight"
               or n.endswith("cbam.mlp.2.bias") or n.endswith("cbam.spatial.0.weight")}
    optimizer = CheckedAdam(model, checked, tmp_path / "first.json", lr=1e-4, weight_decay=1e-5)
    truth = torch.zeros(2, 1, 224, 224)
    if not empty:
        truth[:, :, 60:90, 100:135] = 1
    outputs = model(torch.randn(2, 3, 224, 224))
    assert outputs.main_logits.shape == outputs.auxiliary_logits.shape == truth.shape
    loss = CombinationObjective(variant, config["phase_l"])(outputs, truth)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert optimizer.first_step_checks["finite_nonzero_gradients"]
    assert all(optimizer.first_step_checks["updated_parameters"].values())
    assert not model.base.decoder.blocks[3]._forward_hooks


def report(dice=.84, small=.77, boundary=.54, empty=4, split="validation"):
    return {"split": split, "checkpoint_sha256": "SYNTHETIC FIXTURE",
            "all": {"image_based": {"dice": dice, "boundary_iou": boundary}, "data_based": {"dice": .91}},
            "gt_positive": {"image_based": {"dice": .87}, "data_based": {"recall": .91}, "total_foreground_misses": 0},
            "gt_empty": {"images_with_prediction": empty, "counts": {"fp": 526}},
            "subgroups": {"small": {"image_based": {"dice": small}}}}


def reports():
    return {"B1": report(), "A6": report(dice=.85), "H2": report(dice=.848), "L3": report(dice=.836),
            "B1A6H2": report(dice=.86, small=.8, boundary=.55),
            "L3A6H2": report(dice=.865, small=.81, boundary=.56)}


def test_validation_only_selection_guards_and_predeclared_tie_order():
    r = reports()
    profiles = {n: {"active_forward_parameters": 100} for n in (*CACHED, *FRESH)}
    assert select(r, profiles)["selected_candidate"] == "L3A6H2"
    r["L3A6H2"] = deepcopy(r["B1A6H2"])
    assert select(r, profiles)["selected_candidate"] == "B1A6H2"
    for n in FRESH:
        r[n]["gt_empty"]["images_with_prediction"] = 5
    assert select(r, profiles)["selected_candidate"] == "B1"
    with pytest.raises(ValueError, match="VALIDATION"):
        select({**r, "B1A6H2": report(split="test")}, profiles)


def test_contrast_includes_paired_loss_effect_but_not_unidentified_three_way_synergy():
    r = reports()
    c = contrasts(r)["image_dice_pp"]
    assert c["L3A6H2_minus_B1A6H2"] == pytest.approx(.5)
    assert c["A6_H2_descriptive_interaction"] == pytest.approx(.2)
    assert "three_way_interaction" not in c


def test_distinct_run_identities_share_every_training_control(config):
    a, b = (run_config(config, n) for n in FRESH)
    assert digest_json(a) != digest_json(b)
    a.pop("active_combination")
    b.pop("active_combination")
    assert a == b == config
    assert config["combinations"]["order"] == list(FRESH)
    assert config["combinations"]["new_training_runs"] == 2
    assert not config["combinations"]["standalone_smoke_training"]
    assert not config["combinations"]["external_inference"]
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "configs/cp2/phase_p.json").read_text())
    for key in ("seed", "split_seed", "cpu_threads", "training", "model", "data", "loss"):
        assert config[key] == old[key]
