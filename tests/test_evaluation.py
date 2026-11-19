import ast
from contextlib import redirect_stdout
import io

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from foot_ulcer_segmentation.evaluation import SegmentationEvaluator
from foot_ulcer_segmentation.reference import REFERENCE, verify_reference


@pytest.mark.parametrize("truth,prediction", [
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([0, 0, 0, 0], [0, 1, 0, 0]),
    ([0, 1, 0, 1], [0, 1, 0, 1]),
    ([0, 1, 0, 1], [1, 1, 0, 0]),
    ([0, 1, 0, 1], [0, 0, 0, 0]),
])
def test_metrics_match_original_offline_evaluator(truth, prediction):
    verify_reference()
    tree = ast.parse((REFERENCE / "utils/eval.py").read_text())
    loop = next(node for node in tree.body if isinstance(node, ast.For))
    conditional = next(node for node in loop.body if isinstance(node, ast.If))
    truth, prediction = np.array(truth), np.array(prediction)
    scope = {"np": np, "confusion_matrix": confusion_matrix, "flat_mask": truth,
             "flat_pred": prediction, "unq_mask_val": np.unique(truth),
             "HARD_LINE": True, "ep": 1e-6, "i": 0, "name": "synthetic"}
    with redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=[conditional], type_ignores=[]), "reference_eval", "exec"), scope)
    evaluator = SegmentationEvaluator()
    actual = evaluator.add("synthetic", truth, prediction)
    for key in ["tp", "tn", "fp", "fn"]:
        assert actual[key] == scope[key]
    for key, source_key in [("dice", "dice"), ("iou", "iou"), ("precision", "p"), ("recall", "r")]:
        assert actual[key] == pytest.approx(scope[source_key] / 100, abs=1e-12)


def test_aggregation_and_legacy_scores_are_distinct():
    evaluator = SegmentationEvaluator()
    evaluator.add("missed", [1, 0], [0.0, 0.0])
    evaluator.add("perfect", [1, 1, 1, 0], [1.0, 1.0, 1.0, 0.0])
    report = evaluator.summary()
    assert report["offline_hard_line"]["data_based"]["dice"] == pytest.approx(6 / 7)
    assert report["offline_hard_line"]["image_based"]["dice"] == pytest.approx(0.5)
    assert report["author_inline_legacy"]["data_based"]["dice"] == pytest.approx(8 / 9)


def test_threshold_and_empty_collection_semantics():
    evaluator = SegmentationEvaluator()
    evaluator.add("threshold", [0, 1], [0.0, 0.5])
    assert evaluator.summary()["offline_hard_line"]["data_based"]["dice"] == 1.0
    empty = SegmentationEvaluator()
    empty.add("empty", [0, 0], [0.0, 0.0])
    assert empty.summary()["offline_hard_line"]["data_based"]["dice"] is None
    assert empty.summary()["offline_hard_line"]["image_based"]["dice"] == 1.0


def test_fractional_ground_truth_requires_an_explicit_policy():
    with pytest.raises(ValueError, match="binary"):
        SegmentationEvaluator().add("fractional", [0.0, 32 / 255], [0.0, 1.0])
