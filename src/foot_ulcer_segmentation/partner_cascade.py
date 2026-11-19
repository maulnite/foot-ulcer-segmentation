"""Read-only replay of a partner's audited cascade definitions and checkpoints.

Only explicitly selected definitions and literal configuration values are loaded.
The source script's downloads, directory creation, and training are never run.
"""

import ast
import os
from pathlib import Path

import cv2
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset

from .common import digest_file


DEFINITIONS = {
    "Dataset", "ConvBNReLU", "EfficientNetEncoder", "SCSE", "LocalAttention",
    "PscSE", "CrossLevelFusion", "RawDetailPath", "FrequencyDetailPath",
    "GuidedOffsetUpsample", "DecoderBlock", "FUSegNetDecoder", "FUSegNet",
    "expand_box", "cascade_predict",
}
CONSTANTS = {
    "ENCODER", "DECODER_CHANNELS", "SE_REDUCTION", "LG_LOCAL_KERNEL", "LG_FUSION",
    "CL_PROJ_CH", "RDS_PROJ_CH", "RDS_INJECT", "FREQ_PROJ_CH", "FREQ_INJECT",
    "UP_MAX_OFFSET", "CROP_SIZE", "CROP_MARGIN_EVAL", "MIN_BOX_FRAC", "PROP_THR",
    "MAX_COMPONENTS", "MIN_COMPONENT_PX", "THRESHOLD",
}


def extract_definitions(text, names, namespace, filename="<partner-source>"):
    """Execute named definitions, excluding all top-level operational statements."""
    tree = ast.parse(text)
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    found = {node.name for node in nodes}
    if found != set(names) or len(nodes) != len(names):
        raise ValueError(f"Missing or duplicate source definitions: {set(names) - found}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def load_partner_namespace(source, device="cpu"):
    source = Path(source)
    text = source.read_text()
    namespace = {"__name__": __name__, "os": os, "cv2": cv2, "np": np,
                 "torch": torch, "nn": nn, "F": F, "EfficientNet": EfficientNet,
                 "BaseDataset": BaseDataset, "DEVICE": torch.device(device)}
    for node in ast.parse(text).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name in CONSTANTS:
            namespace[name] = ast.literal_eval(node.value)
        if name in {"IMAGENET_MEAN", "IMAGENET_STD"}:
            # Audited numpy-array literals only; do not execute arbitrary assignments.
            call = node.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "np" and call.func.attr == "array"):
                raise ValueError(f"Unexpected normalization declaration: {name}")
            namespace[name] = np.array(ast.literal_eval(call.args[0]), dtype=np.float32)
    missing = CONSTANTS - namespace.keys()
    if missing or not {"IMAGENET_MEAN", "IMAGENET_STD"} <= namespace.keys():
        raise ValueError(f"Missing audited configuration: {missing}")
    return extract_definitions(text, DEFINITIONS, namespace, str(source))


def load_model(namespace, checkpoint):
    """Instantiate without downloads and require an exact state-dict match."""
    model = namespace["FUSegNet"](
        namespace["ENCODER"], False, namespace["DECODER_CHANNELS"], False, False)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)
    model.requires_grad_(False)
    metadata = {"epoch": int(state["epoch"]),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "checkpoint_sha256": digest_file(checkpoint), "strict_load": True}
    return model.to(namespace["DEVICE"]).eval(), metadata


def verify_inventory(inventory):
    changed = [str(path) for path, digest in inventory.items() if digest_file(path) != digest]
    if changed:
        raise RuntimeError(f"Read-only inputs changed during replay: {changed[:5]}")
    return {"files_checked": len(inventory), "all_unchanged": True}


def latency_summary(seconds):
    values = np.asarray(seconds, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Timing requires finite positive durations")
    return {"samples": len(values), "mean_ms": float(values.mean() * 1000),
            "median_ms": float(np.median(values) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "throughput_images_per_second": float(1 / values.mean())}
