"""Phase-L regional losses and foreground signed-distance supervision.

Distance convention adapted from LIVIAETS/boundary-loss, revision
171c32d88a4ce59af8be46fb88b96d3637b9515b (MIT; licenses/boundary-loss-MIT.txt).
This is a binary foreground adaptation, not a reproduction of its training study.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

from .cp2_protocol import BaselineObjective


def signed_distance(mask):
    """Native pixel units; empty foreground returns zero, as in the reference."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Distance target requires a binary 2D augmented mask")
    foreground = mask.astype(bool)
    if not foreground.any():
        return np.zeros(mask.shape, dtype=np.float32)
    background = ~foreground
    outside = distance_transform_edt(background) * background
    inside = (distance_transform_edt(foreground) - 1) * foreground
    return np.asarray(outside - inside, dtype=np.float32)


def distance_batch(truth):
    """Compute on current augmented GT; never transform a precomputed distance map."""
    if truth.ndim != 4 or truth.shape[1] != 1:
        raise ValueError("Expected binary segmentation targets [B, 1, H, W]")
    masks = truth.detach().cpu().numpy()
    distances = np.stack([signed_distance(mask[0]) for mask in masks])[:, None]
    return torch.from_numpy(distances).to(device=truth.device, dtype=truth.dtype)


def tversky_loss(probability, truth, *, alpha, beta, epsilon, q, delta):
    """Batch-global soft counts; stabilized power has finite gradients at TI=1."""
    if probability.shape != truth.shape or probability.ndim != 4:
        raise ValueError("Tversky requires matched [B, 1, H, W] tensors")
    if alpha <= 0 or beta <= 0 or epsilon <= 0 or q <= 0 or delta <= 0:
        raise ValueError("Tversky coefficients and stability constants must be positive")
    true_positive = (probability * truth).sum()
    false_positive = (probability * (1 - truth)).sum()
    false_negative = ((1 - probability) * truth).sum()
    index = (true_positive + epsilon) / (
        true_positive + alpha * false_positive + beta * false_negative + epsilon)
    error = (1 - index).clamp_min(0)
    return (error + delta).pow(q) - delta ** q


class PhaseLObjective(torch.nn.Module):
    """The predeclared L1–L4 screen; L0 exists only for B1 parity tests."""

    def __init__(self, experiment, settings):
        super().__init__()
        if experiment not in ("L0", "L1", "L2", "L3", "L4"):
            raise ValueError("Unknown phase-L screening objective")
        self.experiment = experiment
        self.settings = settings
        self.baseline = BaselineObjective("B1")

    def components(self, logits, truth):
        if logits.shape != truth.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Expected matched [B, 1, H, W] logits and targets")
        probability = logits.sigmoid()
        zero = logits.new_zeros(())
        parts = {"dice": zero, "focal": zero, "tversky": zero,
                 "boundary_raw": zero, "boundary_weighted": zero}
        if self.experiment in ("L0", "L3"):
            parts["dice"], parts["focal"] = self.baseline.components(logits, truth)
            regional = parts["dice"] + parts["focal"]
        else:
            parameters = self.settings["tversky"]
            q = parameters["q_L1"] if self.experiment == "L1" else parameters["q_L2"]
            parts["tversky"] = tversky_loss(
                probability, truth, alpha=parameters["alpha_fp"],
                beta=parameters["beta_fn"], epsilon=parameters["epsilon"], q=q,
                delta=parameters["power_stability_delta"])
            regional = parts["tversky"]
        if self.experiment in ("L3", "L4"):
            coefficient = self.settings["boundary"]["weight"]
            if coefficient < 0:
                raise ValueError("Boundary coefficient cannot be negative")
            parts["boundary_raw"] = (probability * distance_batch(truth)).mean()
            parts["boundary_weighted"] = coefficient * parts["boundary_raw"]
        parts["regional"] = regional
        parts["total"] = regional + parts["boundary_weighted"]
        return parts

    def forward(self, logits, truth):
        return self.components(logits, truth)["total"]
