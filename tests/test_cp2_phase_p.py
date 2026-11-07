"""Scientific CP2 guards; tiny arrays here are explicitly synthetic engineering fixtures."""

import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from foot_ulcer_segmentation.common import digest_json
from foot_ulcer_segmentation.cp2_data import grouped_membership, verify_cp2_manifest
from foot_ulcer_segmentation.cp2_protocol import (
    BaselineObjective, CP2Evaluator, ImageDiceCheckpoint, boundary_iou,
)
from foot_ulcer_segmentation.cp2_train import export_evaluation, tensor_fingerprint
from foot_ulcer_segmentation.reference import build_loss_and_metrics


@pytest.mark.parametrize("target", [torch.zeros(2, 1, 3, 4), torch.ones(2, 1, 3, 4),
                                    torch.arange(24).reshape(2, 1, 3, 4).float() % 2])
def test_b0_matches_pinned_source_values_and_gradients(target):
    logits = torch.linspace(-8, 8, 24).reshape_as(target).requires_grad_()
    source, _ = build_loss_and_metrics()
    expected = source(logits.sigmoid(), target)
    actual = BaselineObjective("B0")(logits, target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_b1_corrects_only_focal_pair_and_keeps_empty_gradients_finite():
    logits = torch.tensor([-80., -3., 0., 3., 80.], requires_grad=True)
    truth = torch.zeros_like(logits)
    b0, b1 = BaselineObjective("B0"), BaselineObjective("B1")
    dice0, focal0 = b0.components(logits, truth)
    dice1, focal1 = b1.components(logits, truth)
    assert torch.equal(dice0, dice1)
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    torch.testing.assert_close(focal1, ((1 - torch.exp(-bce)) ** 2 * bce).mean())
    assert focal0 != focal1
    for objective in (b0, b1):
        gradient = torch.autograd.grad(objective(logits, truth), logits)[0]
        assert torch.isfinite(gradient).all()
        assert gradient[-2] > 0  # nonsaturated false positive on empty GT is penalized
        if objective.baseline == "B1":
            assert gradient[-1] > 0
        else:
            # Source B0 saturates through sigmoid(80); preserve rather than conceal it.
            assert gradient[-1] == 0


def test_checkpoint_uses_image_dice_and_earliest_tie():
    policy = ImageDiceCheckpoint()
    assert policy.update(.8, 1)
    assert not policy.update(.8, 2)
    assert not policy.update(.7, 3)
    assert policy.epoch == 1 and policy.bad_epochs == 2
    assert policy.update(.81, 4)
    assert policy.epoch == 4 and policy.bad_epochs == 0
    with pytest.raises(ValueError):
        policy.update(float("nan"), 5)


def fixture_rows():
    rows = []
    for i in range(12):
        rows.append({"sample_id": f"train/{i}", "name": str(i), "source_split": "train",
                     "rgb_group": str(i), "mask_pixels_sha256": f"mask-{i}"})
    rows += [{**rows[0], "sample_id": "train/duplicate"},
             {**rows[1], "sample_id": "train/conflict", "mask_pixels_sha256": "different"},
             {**rows[2], "sample_id": "test/overlap", "source_split": "test"}]
    return rows


def test_group_membership_quarantines_conflict_and_source_test_overlap():
    rows = fixture_rows()
    splits, excluded, audit = grouped_membership(rows, .2, 42)
    assert audit["conflicting_annotation_groups"] == 1
    assert len(excluded) == 4
    assert all(r["rgb_group"] != "1" for v in splits.values() for r in v)
    assert [r["rgb_group"] for r in splits["test"]] == ["2"]
    roles = [{r["rgb_group"] for r in splits[s]} for s in ("train", "validation", "test")]
    assert not any(a & b for i, a in enumerate(roles) for b in roles[i + 1:])
    shuffled = list(reversed(rows))
    assert grouped_membership(shuffled, .2, 42) == (splits, excluded, audit)


def test_manifest_rejects_tampering_and_repeated_groups():
    splits, _, _ = grouped_membership(fixture_rows(), .2, 42)
    manifest = {"splits": splits}
    manifest["fingerprint"] = digest_json(manifest)
    verify_cp2_manifest(manifest, files=False)
    altered = copy.deepcopy(manifest)
    altered["splits"]["validation"].append(altered["splits"]["train"][0])
    with pytest.raises(ValueError, match="hash"):
        verify_cp2_manifest(altered, files=False)
    altered["fingerprint"] = digest_json({k: v for k, v in altered.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="Duplicate"):
        verify_cp2_manifest(altered, files=False)


def settings():
    return {"prediction_threshold": .5, "boundary_radius_diagonal_fraction": .01}


@pytest.mark.parametrize("gt,pred,expected", [(0, 0, 1.), (0, 1, 0.), (1, 0, 0.), (1, 1, 1.)])
def test_empty_rules_threshold_and_boundary(gt, pred, expected):
    truth = np.zeros((4, 4), dtype=np.uint8)
    probability = np.zeros((4, 4), dtype=np.float32)
    truth[1, 1] = gt
    probability[1, 1] = .5 if pred else 0  # threshold must be inclusive
    evaluator = CP2Evaluator(settings(), boundaries=True)
    row = evaluator.add({"sample_id": "synthetic", "rgb_group": "fixture"}, truth, probability)
    assert row["dice"] == expected
    assert row["boundary_iou"] == expected
    assert row["tp"] == (gt & pred) and row["fp"] == ((not gt) & pred)


def test_boundary_iou_analytical_geometry_and_full_canvas():
    truth = np.zeros((10, 10), dtype=np.uint8)
    truth[2:5, 2:5] = 1
    prediction = np.zeros_like(truth)
    prediction[2:5, 3:6] = 1
    # Radius 1; 8-pixel outlines intersect at 4 pixels, union 12.
    assert boundary_iou(truth, prediction) == pytest.approx(4 / 12)
    assert boundary_iou(np.ones_like(truth), np.ones_like(truth)) == 1


def test_subgroups_and_export_preserve_ordinary_counts(tmp_path):
    evaluator = CP2Evaluator(settings())
    empty = np.zeros((20, 20), dtype=np.uint8)
    small = empty.copy()
    small[0, 0] = 1
    evaluator.add({"sample_id": "empty", "rgb_group": "zero"}, empty, empty)
    evaluator.add({"sample_id": "small", "rgb_group": "one"}, small, empty)
    result = export_evaluation(tmp_path / "metrics.json", evaluator, {"diagnostic_smoke_only": True})
    assert result["all"]["image_based"]["dice"] == .5
    assert result["all"]["counts"]["tp"] == 0
    assert result["gt_empty"]["images"] == 1 and result["gt_positive"]["images"] == 1
    assert result["subgroups"]["small"]["total_foreground_misses"] == 1
    assert json.loads((tmp_path / "metrics.json").read_text()) == result
    assert (tmp_path / "metrics.csv").read_text().count("\n") == 3
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluator.add({"sample_id": "nan", "rgb_group": "bad"}, empty, empty.astype(float) + np.nan)


def test_state_hash_checks_actual_parameters_and_buffers():
    state = {"weight": torch.zeros(2, 3), "counter": torch.tensor(2)}
    reordered = dict(reversed(list(state.items())))
    assert tensor_fingerprint(state) == tensor_fingerprint(reordered)
    changed = copy.deepcopy(state)
    changed["weight"][0, 0] = 1
    assert tensor_fingerprint(state) != tensor_fingerprint(changed)
