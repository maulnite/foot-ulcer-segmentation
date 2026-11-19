"""Branch isolation, unchanged initialization, and untouched shorted stage."""

import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from attention_ablation import apply_branch_ablation, state_digest
from foot_ulcer_segmentation.reference import load_reference


def fixture_model():
    load_reference()
    from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.blocks = torch.nn.ModuleList([
        DecoderBlock(32, 16 if i < 4 else 0, 16, attention_type="pscse") for i in range(5)])
    return model


@pytest.mark.parametrize("variant", ["additive_only", "maxout_only"])
def test_patch_preserves_initialization_rng_and_last_stage(variant):
    torch.set_num_threads(2)
    model = fixture_model()
    original = copy.deepcopy(model).eval()
    before, rng = state_digest(model), torch.get_rng_state().clone()
    final_forward = model.decoder.blocks[-1].forward.__func__
    apply_branch_ablation(model, variant)
    assert state_digest(model) == before
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.decoder.blocks[-1].forward.__func__ is final_forward
    model.eval()
    x = torch.randn(2, 32, 4, 4)
    with torch.no_grad():
        assert torch.equal(model.decoder.blocks[-1](x), original.decoder.blocks[-1](x))


@pytest.mark.parametrize("variant,active,inactive", [
    ("additive_only", "attention2", "attention1"),
    ("maxout_only", "attention1", "attention2"),
])
def test_backward_reaches_only_the_selected_regular_branch(variant, active, inactive):
    torch.set_num_threads(2)
    model = apply_branch_ablation(fixture_model(), variant)
    block = model.decoder.blocks[0]
    x, skip = torch.randn(2, 32, 4, 4), torch.randn(2, 16, 8, 8)
    block(x, skip).square().mean().backward()
    assert any(p.grad is not None for p in getattr(block, active).parameters())
    assert all(p.grad is None for p in getattr(block, inactive).parameters())
    assert all(p.grad is None for p in block.conv2.parameters())
    assert any(p.grad is not None for p in block.conv1.parameters())


def test_invalid_variant_does_not_mutate_the_model():
    model = fixture_model()
    before = state_digest(model)
    with pytest.raises(ValueError, match="Unknown"):
        apply_branch_ablation(model, "scse")
    assert state_digest(model) == before
