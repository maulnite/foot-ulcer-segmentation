import numpy as np
import pytest
import torch

from foot_ulcer_segmentation.common import digest_file
from foot_ulcer_segmentation.partner_cascade import (
    extract_definitions, latency_summary, load_model, verify_inventory,
)


def test_definition_extraction_never_executes_top_level_training(tmp_path):
    marker = tmp_path / "must_not_exist"
    source = f"open({str(marker)!r}, 'w').write('training')\n\ndef inference(x):\n    return x + 1\n"
    namespace = extract_definitions(source, {"inference"}, {})
    assert namespace["inference"](4) == 5
    assert not marker.exists()


def test_missing_or_duplicate_definitions_fail_closed():
    with pytest.raises(ValueError):
        extract_definitions("def other(): pass", {"inference"}, {})
    with pytest.raises(ValueError):
        extract_definitions("def inference(): pass\ndef inference(): pass", {"inference"}, {})


def test_preservation_check_detects_a_changed_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"historical weights")
    inventory = {str(checkpoint): digest_file(checkpoint)}
    assert verify_inventory(inventory) == {"files_checked": 1, "all_unchanged": True}
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed during replay"):
        verify_inventory(inventory)


class TinyModel(torch.nn.Linear):
    def __init__(self, *args):
        super().__init__(2, 1)


def tiny_namespace():
    return {"FUSegNet": TinyModel, "ENCODER": "dummy", "DECODER_CHANNELS": (),
            "DEVICE": torch.device("cpu")}


def test_checkpoint_replay_loads_exact_weights_without_training(tmp_path):
    expected = TinyModel()
    checkpoint = tmp_path / "model.pth"
    torch.save({"epoch": 7, "state_dict": expected.state_dict()}, checkpoint)
    actual, metadata = load_model(tiny_namespace(), checkpoint)
    assert metadata["strict_load"] and metadata["epoch"] == 7
    assert metadata["checkpoint_sha256"] == digest_file(checkpoint)
    assert not actual.training
    assert all(not parameter.requires_grad for parameter in actual.parameters())
    inputs = torch.tensor([[1., 2.]])
    assert torch.equal(actual(inputs), expected(inputs))


def test_checkpoint_replay_rejects_missing_weights(tmp_path):
    checkpoint = tmp_path / "incomplete.pth"
    torch.save({"epoch": 7, "state_dict": {"weight": TinyModel().weight}}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(tiny_namespace(), checkpoint)


def test_timing_units_and_throughput():
    result = latency_summary([.1, .2, .3])
    assert result["mean_ms"] == pytest.approx(200)
    assert result["median_ms"] == pytest.approx(200)
    assert result["p95_ms"] == pytest.approx(290)
    assert result["throughput_images_per_second"] == pytest.approx(5)


@pytest.mark.parametrize("values", [[], [0], [-1], [np.nan], [np.inf], [[.1]]])
def test_timing_rejects_invalid_measurements(values):
    with pytest.raises(ValueError):
        latency_summary(values)
