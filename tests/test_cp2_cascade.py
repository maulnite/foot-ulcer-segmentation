"""Structural fixtures only; no synthetic values are reported as research scores."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from foot_ulcer_segmentation.cp2_cascade import (
    CONSTANTS, DEFINITIONS, CropTraining, decision, namespace, proposals, refine, source_key,
)
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator


def expand_box(x0, y0, x1, y1, margin, height, width, min_frac):
    # Deliberately simple test-only geometry; real geometry gets differential QA.
    return x0, y0, x1, y1


@pytest.fixture
def ns():
    return {"PROP_THR": .3, "MIN_COMPONENT_PX": 4, "MAX_COMPONENTS": 5,
            "CROP_MARGIN_EVAL": .5, "MIN_BOX_FRAC": .25, "CROP_SIZE": 224,
            "expand_box": expand_box}


def test_active_mapping_respects_source_attention_branch_semantics():
    assert source_key("encoder.net._conv_stem.weight") == "encoder._conv_stem.weight"
    assert source_key("decoder.blocks.2.att.add_branch.cse.1.weight") == "decoder.blocks.2.attention2.cSE.1.weight"
    assert source_key("decoder.blocks.2.att.max_branch.sse.0.bias") == "decoder.blocks.2.attention1.sSE.0.bias"
    assert source_key("decoder.blocks.2.conv.1.running_mean") == "decoder.blocks.2.conv1.1.running_mean"
    assert source_key("head.weight") == "segmentation_head.0.weight"


def test_namespace_excludes_operational_statements(tmp_path):
    marker = tmp_path / "must_not_exist"
    path = tmp_path / "partner.py"
    text = f"open({str(marker)!r}, 'w').write('unsafe top-level work')\n"
    text += "\n".join(f"{name} = 1" for name in sorted(CONSTANTS))
    text += "\n" + "\n".join(f"def {name}(*args, **kwargs): return 7" for name in sorted(DEFINITIONS))
    path.write_text(text)
    result = namespace(path)
    assert result["cascade_predict"]() == 7
    assert not marker.exists()


def test_namespace_fails_closed_on_duplicate_definitions(tmp_path):
    path = tmp_path / "invalid.py"
    path.write_text("def FUSegNet(): pass\ndef FUSegNet(): pass")
    with pytest.raises(ValueError):
        namespace(path)


@pytest.mark.parametrize("value", [np.full((3, 3), np.nan), np.ones((3, 3)) * 1.1, np.zeros((1, 3, 3))])
def test_invalid_probabilities_rejected(value, ns):
    with pytest.raises(ValueError):
        proposals(value, ns)


def test_proposals_threshold_area_top_five_and_order(ns):
    image = np.zeros((224, 224), dtype=np.float32)
    image[1, 1] = .9  # Too small.
    for i in range(7):
        image[20 + i * 20:22 + i * 20, 10:13 + i] = .3
    boxes = proposals(image, ns)
    assert len(boxes) == 5
    assert [b["component_pixels"] for b in boxes] == [18, 16, 14, 12, 10]


class ConstantRefiner(torch.nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.logit = float(torch.logit(torch.tensor(probability)))
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return torch.full((1, 1, 224, 224), self.logit, device=image.device)


def test_no_proposal_preserves_coarse_and_skips_refiner(ns):
    coarse = np.full((224, 224), .1, dtype=np.float32)
    model = ConstantRefiner(.9)
    actual, boxes = refine(model, torch.zeros(1, 3, 224, 224), coarse, ns, "cpu")
    np.testing.assert_array_equal(actual, coarse)
    assert boxes == [] and model.calls == 0


def test_refinement_replaces_inside_preserves_outside_and_never_reads_gt(ns):
    coarse = np.zeros((224, 224), dtype=np.float32)
    coarse[40:50, 20:30] = .8
    model = ConstantRefiner(.2)
    actual, boxes = refine(model, torch.zeros(1, 3, 224, 224), coarse, ns, "cpu")
    assert model.calls == len(boxes) == 1
    np.testing.assert_allclose(actual[40:50, 20:30], .2, atol=1e-7)
    assert actual.sum() == pytest.approx(20.)


def test_overlapping_boxes_use_max_probability(ns):
    coarse = np.zeros((224, 224), dtype=np.float32)
    coarse[20:22, 20:22] = .6
    coarse[50:52, 50:52] = .6
    ns["expand_box"] = lambda *args: (0, 0, 100, 100)

    class Alternating(ConstantRefiner):
        def forward(self, image):
            self.calls += 1
            return torch.full((1, 1, 224, 224), float(torch.logit(torch.tensor(.2 if self.calls == 1 else .8))))
    output, boxes = refine(Alternating(.2), torch.zeros(1, 3, 224, 224), coarse, ns, "cpu")
    assert len(boxes) == 2
    np.testing.assert_allclose(output[:100, :100], .8, atol=1e-7)
    assert not output[100:, :].any()


def test_crop_training_rejects_source_test_rows():
    with pytest.raises(ValueError, match="Only original train"):
        CropTraining([{"source_split": "test"}], {})


def clinical_metrics():
    return {"all": {"image_based": {"dice": .8, "boundary_iou": .5}},
            "subgroups": {"small": {"image_based": {"dice": .7}}},
            "gt_positive": {"total_foreground_misses": 0},
            "gt_empty": {"images_with_prediction": 2}}


def test_selection_requires_all_guards_and_uses_no_test_scores():
    coarse = clinical_metrics()
    cascade = copy.deepcopy(coarse)
    assert decision(coarse, cascade)["selected"] == "coarse_B1"  # Tie favors baseline.
    cascade["all"]["image_based"]["dice"] = .81
    assert decision(coarse, cascade)["selected"] == "cascade"
    cascade["gt_empty"]["images_with_prediction"] += 1
    assert decision(coarse, cascade)["selected"] == "coarse_B1"


def test_empty_and_exact_half_probability_evaluator():
    evaluator = CP2Evaluator({"prediction_threshold": .5})
    truth = np.zeros((2, 2), dtype=np.uint8)
    for i, value in enumerate((0., .5)):
        evaluator.add({"sample_id": str(i), "rgb_group": str(i)}, truth, np.full((2, 2), value))
    assert evaluator.rows[0]["dice"] == 1.
    assert evaluator.rows[1]["dice"] == 0.
    assert evaluator.summary()["gt_empty"]["images_with_prediction"] == 1


def test_locked_config_contains_one_new_refiner_no_external_selection():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/cp2/cascade_clean.json").read_text())
    c = config["cascade"]
    assert c["trainable_stages"] == ["refiner"]
    assert not c["external_selection"] and not c["standalone_smoke_training"]
    assert not config["training"]["amp"] and config["training"]["batch_size"] == 2
    assert config["training"]["precision"] == "float32"
    original = json.loads((root / "configs/cp2/phase_p.json").read_text())
    for key in ("seed", "split_seed", "data", "training", "loss", "evaluation", "model"):
        assert config[key] == original[key]
