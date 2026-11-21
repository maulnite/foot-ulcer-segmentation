"""Auxiliary foreground-presence supervision without changing mask inference.

Presence is defined by the actual augmented annotation, not patient diagnosis.
The original decoder and segmentation forward remain untouched.
"""

from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .cp2_protocol import BaselineObjective
from .cp2_train import tensor_fingerprint
from .reference import build_model


class PresenceOutputs(NamedTuple):
    main_logits: torch.Tensor
    presence_logits: Optional[torch.Tensor]

    def sigmoid(self):
        """Existing evaluator consumes only primary segmentation probabilities."""
        return self.main_logits.sigmoid()

    @property
    def shape(self):
        return self.main_logits.shape


def presence_targets(truth):
    """One target per ACTUAL augmented mask, including a one-pixel wound."""
    if truth.ndim != 4 or truth.shape[1] != 1:
        raise ValueError("Presence requires binary masks [B,1,H,W]")
    if not torch.isfinite(truth).all() or not ((truth == 0) | (truth == 1)).all():
        raise ValueError("Presence requires finite binary masks; do not relabel")
    return truth.flatten(1).any(dim=1, keepdim=True).to(dtype=truth.dtype)


class PresenceModel(nn.Module):
    def __init__(self, base, presence_head=None):
        super().__init__()
        self.base = base
        self.presence_head = presence_head

    def forward(self, images, auxiliary=True):
        if self.presence_head is None or not auxiliary:
            return PresenceOutputs(self.base(images), None)
        features = []

        def capture(module, inputs, output):
            features.append(output)

        handle = self.base.decoder.blocks[3].register_forward_hook(capture)
        try:
            logits = self.base(images)
        finally:
            handle.remove()
        if len(features) != 1 or features[0].shape[1:] != (32, 112, 112):
            raise ValueError("Expected stage-3 feature [B,32,112,112]")
        pooled = features[0].mean(dim=(2, 3))
        return PresenceOutputs(logits, self.presence_head(pooled))


class PresenceObjective(nn.Module):
    def __init__(self, weight=.1):
        super().__init__()
        if weight != .1:
            raise ValueError("Only preregistered presence weight 0.1")
        self.weight = weight
        self.baseline = BaselineObjective("B1")

    def components(self, outputs, truth):
        if outputs.presence_logits is None:
            raise ValueError("Auxiliary objective requires actual presence logits")
        target = presence_targets(truth)
        if outputs.presence_logits.shape != target.shape:
            raise ValueError("One presence logit per original image required")
        dice, focal = self.baseline.components(outputs.main_logits, truth)
        auxiliary = F.binary_cross_entropy_with_logits(
            outputs.presence_logits, target, reduction="mean")
        regional = dice + focal
        weighted = self.weight * auxiliary
        return {"dice": dice, "focal": focal, "regional": regional,
                "presence_bce": auxiliary, "weighted_presence": weighted,
                "total": regional + weighted}

    def forward(self, outputs, truth):
        return self.components(outputs, truth)["total"]


def build_presence_model(config, initial, auxiliary=True):
    settings = config["phase_n"]
    if settings["stage"] != 3 or settings["feature_channels"] != 32:
        raise ValueError("Presence head placement is frozen")
    base = build_model(config, pretrained=False)
    base.load_state_dict(initial["state_dict"], strict=True)
    fingerprint = tensor_fingerprint(base.state_dict())
    if fingerprint != settings["control_initial_tensor_sha256"]:
        raise ValueError("Source initial tensors differ from B1")
    head = None
    if auxiliary:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(settings["head_seed"])
            head = nn.Linear(32, 1, bias=True)
    model = PresenceModel(base, head)
    if any(not torch.equal(base.state_dict()[name], value)
           for name, value in initial["state_dict"].items()):
        raise ValueError("Adding presence head changed a base tensor")
    return model, {
        "all_original_named_tensors_equal": True,
        "source_tensor_sha256": fingerprint,
        "head_parameters": 33 if auxiliary else 0,
        "head_tensor_sha256": tensor_fingerprint(head.state_dict()) if auxiliary else None,
        "complete_initial_tensor_sha256": tensor_fingerprint(model.state_dict()),
        "main_forward_unchanged": True, "head_controls_mask_at_inference": False,
    }


class FirstStepAdam(torch.optim.Adam):
    """Unchanged Adam update with a one-time diagnostic on the REAL first step."""
    def __init__(self, model, **kwargs):
        super().__init__(model.parameters(), **kwargs)
        self.checked_parameters = {
            "encoder": next(model.base.parameters()),
            "presence_weight": model.presence_head.weight,
            "presence_bias": model.presence_head.bias,
        }
        self.first_step_checks = None

    def step(self, closure=None):
        if self.first_step_checks is not None:
            return super().step(closure)
        before = {name: p.detach().clone() for name, p in self.checked_parameters.items()}
        for name, parameter in self.checked_parameters.items():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(f"Missing/nonfinite first-step gradient: {name}")
        result = super().step(closure)
        changed = {name: not torch.equal(before[name], p.detach())
                   for name, p in self.checked_parameters.items()}
        if not all(changed.values()):
            raise ValueError(f"Real first optimizer step failed to update: {changed}")
        self.first_step_checks = {"real_training_step": 1, "finite_gradients": True,
                                  "updated_parameters": changed,
                                  "separate_smoke_training": False}
        return result


def select_presence(candidate, baseline, profile, baseline_profile):
    if candidate["split"] != "validation" or baseline["split"] != "validation":
        raise ValueError("Presence selection requires source validation only")
    checks = {
        "higher_overall_dice": candidate["all"]["image_based"]["dice"] > baseline["all"]["image_based"]["dice"],
        "small_not_lower": candidate["subgroups"]["small"]["image_based"]["dice"] >= baseline["subgroups"]["small"]["image_based"]["dice"],
        "boundary_not_lower": candidate["all"]["image_based"]["boundary_iou"] >= baseline["all"]["image_based"]["boundary_iou"],
        "positive_pixel_recall_not_lower": candidate["gt_positive"]["data_based"]["recall"] >= baseline["gt_positive"]["data_based"]["recall"],
        "no_extra_total_miss": candidate["gt_positive"]["total_foreground_misses"] <= baseline["gt_positive"]["total_foreground_misses"],
        "no_extra_empty_fp_images": candidate["gt_empty"]["images_with_prediction"] <= baseline["gt_empty"]["images_with_prediction"],
        "no_extra_empty_fp_pixels": candidate["gt_empty"]["counts"]["fp"] <= baseline["gt_empty"]["counts"]["fp"],
        "empty_fp_improved": (candidate["gt_empty"]["images_with_prediction"] < baseline["gt_empty"]["images_with_prediction"]
                              or candidate["gt_empty"]["counts"]["fp"] < baseline["gt_empty"]["counts"]["fp"]),
        "capacity_budget": profile["active_forward_parameters"] <= 1.01 * baseline_profile["active_forward_parameters"],
    }
    eligible = all(checks.values())
    return {"validation_only": True, "selected_candidate": "N1" if eligible else "N0/B1",
            "eligible": eligible, "checks": checks,
            "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
            "validation_image_dice": candidate["all"]["image_based"]["dice"],
            "next_route_for_review": "REVIEW_REPLICATION_BEFORE_COMBINATION" if eligible else "HOLD_AND_DIAGNOSE",
            "automatic_combinations": False, "single_seed_exploratory": True}
