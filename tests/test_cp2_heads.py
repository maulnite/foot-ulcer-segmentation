"""Synthetic engineering fixtures only; no scientific scores are generated here."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from foot_ulcer_segmentation.cp2_heads import (
    AuxiliarySegmentationModel, HeadObjective, HeadOutputs, morphological_boundary,
)
from foot_ulcer_segmentation.cp2_phase_h import expansion_decision, select_candidate
from foot_ulcer_segmentation.cp2_protocol import BaselineObjective


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.decoder = nn.Module()
        self.decoder.blocks = nn.ModuleList([nn.Identity() for _ in range(5)])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, images):
        features = self.encoder(images)
        for block in self.decoder.blocks:
            features = block(features)
        return F.interpolate(self.head(features), size=images.shape[-2:], mode="bilinear", align_corners=False)


def test_auxiliary_forward_preserves_primary_prediction_and_removes_temporary_hook():
    base = ToyBase()
    model = AuxiliarySegmentationModel(base, nn.Conv2d(32, 1, 1))
    inputs = torch.randn(2, 3, 224, 224)
    expected = base(inputs)
    actual = model(inputs)
    assert torch.equal(actual.main_logits, expected)
    assert torch.equal(actual.sigmoid(), expected.sigmoid())
    assert actual.auxiliary_logits.shape == (2, 1, 224, 224)
    assert not base.decoder.blocks[3]._forward_hooks
    inference = model(inputs, auxiliary=False)
    assert inference.auxiliary_logits is None
    assert torch.equal(inference.main_logits, expected)


@pytest.mark.parametrize("variant", ["H1", "H2"])
@pytest.mark.parametrize("empty", [False, True])
def test_finite_gradients_reach_head_and_encoder_for_empty_and_positive_masks(variant, empty):
    model = AuxiliarySegmentationModel(ToyBase(), nn.Conv2d(32, 1, 1))
    truth = torch.zeros(2, 1, 224, 224)
    if not empty:
        truth[:, :, 50:70, 90:120] = 1
    loss = HeadObjective(variant)(model(torch.randn(2, 3, 224, 224)), truth)
    loss.backward()
    for parameter in (model.auxiliary_head.weight, model.auxiliary_head.bias, model.base.encoder.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_boundary_target_matches_independent_cv2_oracle_including_canvas_edges():
    rng = np.random.default_rng(42)
    masks = [np.zeros((7, 9), np.uint8), np.ones((7, 9), np.uint8)]
    masks += [(rng.random((7, 9)) > .8).astype(np.uint8) for _ in range(30)]
    for mask in masks:
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        erosion = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        actual = morphological_boundary(torch.from_numpy(mask.astype(np.float32))[None, None])
        np.testing.assert_array_equal(actual[0, 0].numpy(), dilation - erosion)
    assert not morphological_boundary(torch.zeros(1, 1, 7, 9)).any()


@pytest.mark.parametrize("truth", [torch.full((1, 1, 7, 9), .5), torch.full((1, 1, 7, 9), float("nan")), torch.zeros(1, 7, 9)])
def test_invalid_targets_are_rejected_instead_of_inventing_annotations(truth):
    with pytest.raises(ValueError):
        morphological_boundary(truth)


def test_main_objective_exactly_matches_b1_and_auxiliary_target_is_explicit():
    logits = torch.randn(2, 1, 7, 9, requires_grad=True)
    auxiliary = torch.randn_like(logits, requires_grad=True)
    truth = (torch.rand_like(logits) > .8).float()
    original = BaselineObjective("B1")(logits, truth)
    outputs = HeadOutputs(logits, auxiliary)
    for variant in ("H0", "H1", "H2"):
        parts = HeadObjective(variant).components(outputs, truth)
        assert torch.equal(parts["regional"], original)
        torch.testing.assert_close(parts["total"], original + .1 * parts["auxiliary"], rtol=0, atol=0)
        if variant != "H0":
            target = truth if variant == "H1" else morphological_boundary(truth)
            assert torch.equal(parts["auxiliary"], BaselineObjective("B1")(auxiliary, target))
    with pytest.raises(ValueError, match="requires actual"):
        HeadObjective("H2")(HeadOutputs(logits, None), truth)


def report(dice=.843, small=.773, boundary=.540, misses=0, empty=4, split="validation"):
    return {"split": split, "checkpoint_sha256": "synthetic-fixture-only",
            "all": {"image_based": {"dice": dice, "boundary_iou": boundary}},
            "subgroups": {"small": {"image_based": {"dice": small}}},
            "gt_positive": {"total_foreground_misses": misses}, "gt_empty": {"images_with_prediction": empty}}


def test_expansion_is_validation_only_and_does_not_relax_final_acceptance():
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/cp2/phase_h.json").read_text())
    assert expansion_decision(report(empty=5), report(), config)["expand"]
    assert not expansion_decision(report(dice=.80), report(), config)["expand"]
    profiles = {n: {"active_forward_parameters": 100} for n in ("H0", "H1", "H2")}
    assert select_candidate({"H1": report(dice=.85, empty=5)}, report(), profiles)["selected_candidate"] == "H0"
    assert select_candidate({"H1": report(dice=.85), "H2": report(dice=.85)}, report(), profiles)["selected_candidate"] == "H1"
    with pytest.raises(ValueError, match="validation"):
        select_candidate({"H1": report(split="test")}, report(), profiles)
    with pytest.raises(ValueError, match="validation"):
        expansion_decision(report(split="test"), report(), config)
