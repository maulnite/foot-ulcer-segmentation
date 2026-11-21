"""Explicit auxiliary outputs around the unchanged source segmentation network.

Only a temporary read-only hook taps decoder stage 3. Source forward operations,
including the final decoder stage, are never replaced. No GT enters the model.
"""

from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .cp2_protocol import BaselineObjective
from .cp2_train import tensor_fingerprint
from .reference import build_model


class HeadOutputs(NamedTuple):
    main_logits: torch.Tensor
    auxiliary_logits: Optional[torch.Tensor]

    def sigmoid(self):
        """Frozen validation utilities evaluate the PRIMARY mask only."""
        return self.main_logits.sigmoid()

    @property
    def shape(self):
        return self.main_logits.shape


class AuxiliarySegmentationModel(nn.Module):
    def __init__(self, base, auxiliary_head=None):
        super().__init__()
        self.base = base
        self.auxiliary_head = auxiliary_head

    def forward(self, images, auxiliary=True):
        if self.auxiliary_head is None or not auxiliary:
            return HeadOutputs(self.base(images), None)
        features = []

        def capture(module, inputs, output):
            features.append(output)

        handle = self.base.decoder.blocks[3].register_forward_hook(capture)
        try:
            logits = self.base(images)
        finally:
            handle.remove()
        if len(features) != 1 or features[0].shape[1:] != (32, 112, 112):
            raise ValueError("Expected one source stage-3 feature [B,32,112,112]")
        auxiliary_logits = F.interpolate(
            self.auxiliary_head(features[0]), size=logits.shape[-2:],
            mode="bilinear", align_corners=False)
        return HeadOutputs(logits, auxiliary_logits)


def morphological_boundary(truth):
    """3x3 dilation minus erosion, with explicit zero exterior on BOTH sides.

    Called on the actual geometrically augmented mask, not on the original file.
    This target band differs from the evaluator's inner Boundary IoU band.
    """
    if truth.ndim != 4 or truth.shape[1] != 1:
        raise ValueError("Boundary target requires [B,1,H,W]")
    if not torch.isfinite(truth).all() or not ((truth == 0) | (truth == 1)).all():
        raise ValueError("Use binary finite masks, without silently relabeling")
    padded = F.pad(truth, (1, 1, 1, 1), value=0)
    dilation = F.max_pool2d(padded, 3, stride=1)
    erosion = -F.max_pool2d(-padded, 3, stride=1)
    return dilation - erosion


class HeadObjective(nn.Module):
    """Identical main B1 and auxiliary B1 primitives; only auxiliary TARGET differs."""
    def __init__(self, variant, weight=.1):
        super().__init__()
        if variant not in ("H0", "H1", "H2") or weight != .1:
            raise ValueError("Only the preregistered H0/H1/H2 and weight 0.1")
        self.variant, self.weight = variant, weight
        self.baseline = BaselineObjective("B1")

    def components(self, outputs, truth):
        dice, focal = self.baseline.components(outputs.main_logits, truth)
        zero = dice.new_zeros(())
        auxiliary_dice = auxiliary_focal = zero
        if self.variant != "H0":
            if outputs.auxiliary_logits is None:
                raise ValueError("Training/validation objective requires actual auxiliary logits")
            target = truth if self.variant == "H1" else morphological_boundary(truth)
            auxiliary_dice, auxiliary_focal = self.baseline.components(outputs.auxiliary_logits, target)
        regional = dice + focal
        auxiliary = auxiliary_dice + auxiliary_focal
        weighted = self.weight * auxiliary
        return {"dice": dice, "focal": focal, "regional": regional,
                "auxiliary_dice": auxiliary_dice, "auxiliary_focal": auxiliary_focal,
                "auxiliary": auxiliary, "weighted_auxiliary": weighted,
                "total": regional + weighted}

    def forward(self, outputs, truth):
        return self.components(outputs, truth)["total"]


def build_head_model(config, variant, initial):
    if variant not in ("H0", "H1", "H2"):
        raise ValueError("H3 and combinations are deferred")
    settings = config["phase_h"]
    if settings["stage"] != 3 or settings["feature_channels"] != 32:
        raise ValueError("Head placement is frozen to source stage 3")
    base = build_model(config, pretrained=False)
    base.load_state_dict(initial["state_dict"])
    fingerprint = tensor_fingerprint(base.state_dict())
    if fingerprint != settings["control_initial_tensor_sha256"]:
        raise ValueError("Pairable initial tensors differ from B1")
    head = None
    if variant != "H0":
        # Use a CPU Generator without changing CPU/CUDA global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(settings["head_seed"])
            head = nn.Conv2d(32, 1, 1, bias=True)
    model = AuxiliarySegmentationModel(base, head)
    for name, value in initial["state_dict"].items():
        if not torch.equal(model.base.state_dict()[name], value):
            raise ValueError("Adding an auxiliary head modified a source tensor")
    return model, {"source_tensor_sha256": fingerprint,
                   "all_original_named_tensors_equal": True,
                   "head_tensor_sha256": tensor_fingerprint(head.state_dict()) if head is not None else None,
                   "head_parameters": sum(p.numel() for p in head.parameters()) if head is not None else 0,
                   "source_forward_unchanged": True, "final_decoder_stage_unchanged": True,
                   "complete_initial_tensor_sha256": tensor_fingerprint(model.state_dict())}
