import numpy as np
import pytest
import torch

from foot_ulcer_segmentation.cp2_external import (
    binary_mask, buffers_fingerprint, image_index, letterbox, predict_primary, restore_probability,
)


def test_dataset_mixed_image_extensions_and_ambiguous_pairing(tmp_path):
    for name in ["IMG1.jpg", "IMG2.jpeg", "IMG3.png"]:
        (tmp_path / name).touch()
    assert {key: path.name for key, path in image_index(tmp_path).items()} == {
        "IMG1": "IMG1.jpg", "IMG2": "IMG2.jpeg", "IMG3": "IMG3.png"}
    (tmp_path / "IMG2.jpg").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        image_index(tmp_path)


def test_native_zero_one_masks_are_not_erased_by_a_255_threshold():
    truth = np.array([[0, 1], [1, 0]], np.uint8)
    assert np.array_equal(binary_mask(truth), truth)
    assert np.array_equal(binary_mask(truth * 255), truth)
    with pytest.raises(ValueError):
        binary_mask(np.array([[0, 32]], np.uint8))


def test_rgb_mask_channels_must_agree():
    mask = np.zeros((3, 4, 3), np.uint8)
    mask[0, 0, 1] = 1
    with pytest.raises(ValueError):
        binary_mask(mask)


def test_letterbox_is_image_only_preserves_aspect_and_restores_native_grid():
    rgb = np.full((100, 200, 3), 123, np.uint8)
    boxed, geometry = letterbox(rgb)
    assert geometry["resized_h"] == 112 and geometry["resized_w"] == 224
    assert geometry["top"] == 56 and geometry["left"] == 0
    assert (boxed[:56] == 0).all() and (boxed[56:168] == 123).all()
    probability = np.zeros((224, 224), np.float32)
    probability[56:168] = .8
    restored = restore_probability(probability, geometry)
    assert restored.shape == (100, 200)
    assert np.allclose(restored, .8)


@pytest.mark.parametrize("shape", [(101, 201, 3), (201, 101, 3), (224, 224, 3), (1, 500, 3)])
def test_geometry_handles_odd_sizes_and_thin_images(shape):
    _, geometry = letterbox(np.zeros(shape, np.uint8))
    probability = np.full((224, 224), .5, np.float32)
    restored = restore_probability(probability, geometry)
    assert restored.shape == shape[:2] and (restored >= .5).all()


class ConstantOutput(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, inputs):
        self.bn(inputs)
        return inputs.new_full((len(inputs), 1, 224, 224), self.value)


def test_sigmoid_is_applied_once_and_batchnorm_does_not_update():
    inputs = torch.zeros(1, 3, 224, 224)
    logits = ConstantOutput(0.).eval()
    before = buffers_fingerprint(logits)
    assert (predict_primary(logits, inputs, False) == .5).all()
    assert buffers_fingerprint(logits) == before
    probabilities = ConstantOutput(.8).eval()
    assert np.allclose(predict_primary(probabilities, inputs, True), .8)
    probabilities.train()
    with pytest.raises(ValueError, match="freeze BatchNorm"):
        predict_primary(probabilities, inputs, True)


def test_nonfinite_or_out_of_range_predictions_fail():
    inputs = torch.zeros(1, 3, 224, 224)
    for value in [float("nan"), float("inf"), -1., 2.]:
        with pytest.raises(ValueError):
            predict_primary(ConstantOutput(value).eval(), inputs, True)
