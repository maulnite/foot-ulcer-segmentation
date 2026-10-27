"""PFOM Equation 13 with explicit empty-edge policies; CPU analysis only.

Reference: FUSegNet, arXiv:2305.02961v2, sections 3.2.6 and 4.2.
The sentinel-2 variant is an interpretation of ambiguous paper prose, not
verified parity with an unpublished author implementation.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import canny


def _binary_image(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a nonempty 2D image")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must be binary 0/1; threshold/normalize explicitly")
    return array.astype(bool)


def pratt_from_edges(truth_edges, predicted_edges, *, beta=1 / 9):
    """Sum 1/(1+beta*d**2) over predicted edges, divided by max counts.

    Each distance is to the nearest ground-truth edge, in pixel units. This
    directed score need not be symmetric when exchanging GT and prediction.
    Empty-edge handling is an explicit extension, separated from Equation 13.
    """
    truth = _binary_image(truth_edges, "GT edges")
    prediction = _binary_image(predicted_edges, "Prediction edges")
    if truth.shape != prediction.shape:
        raise ValueError("GT and prediction shapes differ")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    n_gt, n_pred = int(truth.sum()), int(prediction.sum())
    if n_gt and n_pred:
        distances = distance_transform_edt(~truth)[prediction]
        value = float(np.sum(1.0 / (1.0 + beta * distances ** 2)) / max(n_gt, n_pred))
        case, proxy = "both_nonempty", value
    else:
        case = "both_empty" if not n_gt and not n_pred else ("gt_empty" if not n_gt else "prediction_empty")
        value = 1.0 if case == "both_empty" else 0.0
        proxy = 2.0
    return {"gt_edge_pixels": n_gt, "pred_edge_pixels": n_pred, "edge_case": case,
            "pfom": value, "pfom_paper_sentinel2_proxy": proxy}


def canny_edges(mask, *, sigma=.1, low_threshold=.1, high_threshold=.2,
                use_quantiles=False, mode="constant", cval=0.0):
    mask = _binary_image(mask, "Mask")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if not (np.isfinite(low_threshold) and np.isfinite(high_threshold)
            and 0 <= low_threshold <= high_threshold):
        raise ValueError("Canny thresholds must be finite and ordered")
    return canny(mask.astype(np.float64), sigma=sigma,
                 low_threshold=low_threshold, high_threshold=high_threshold,
                 use_quantiles=use_quantiles, mode=mode, cval=cval)


def score_masks(truth, prediction, *, beta=1 / 9, **canny_parameters):
    truth, prediction = _binary_image(truth, "GT mask"), _binary_image(prediction, "Prediction mask")
    if truth.shape != prediction.shape:
        raise ValueError("GT and prediction shapes differ")
    return pratt_from_edges(canny_edges(truth, **canny_parameters),
                            canny_edges(prediction, **canny_parameters), beta=beta)
