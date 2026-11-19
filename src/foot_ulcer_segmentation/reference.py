"""Load the complete, hash-verified author package without changing its files."""

import ast
import importlib
import json
import os
from pathlib import Path
import sys

from .common import ROOT, digest_file

REVISION = "60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc"
REFERENCE = ROOT / "source/upstream/FUSegNet" / REVISION
_CACHE = None


def verify_reference():
    metadata = REFERENCE / "SNAPSHOT.json"
    if not metadata.exists():
        raise FileNotFoundError("Fetch reference first: python scripts/fetch_reference.py")
    snapshot = json.loads(metadata.read_text())
    if snapshot["commit"] != REVISION or not snapshot.get("sha256"):
        raise ValueError("Missing pinned reference checksums")
    for name, expected in snapshot["sha256"].items():
        if digest_file(REFERENCE / name) != expected:
            raise ValueError(f"Reference source was modified: {name}")
    return snapshot


def load_reference():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    verify_reference()
    # The authors' utils/losses.py imports _functional by an unqualified name.
    # Make that exact file importable without depending on the launch directory.
    sys.path.insert(0, str(REFERENCE / "segmentation_models_pytorch/losses"))
    sys.path.insert(0, str(REFERENCE))
    smp = importlib.import_module("segmentation_models_pytorch")
    if REFERENCE not in Path(smp.__file__).resolve().parents:
        raise RuntimeError("A different segmentation_models_pytorch was already imported")
    utils = importlib.import_module("segmentation_models_pytorch.utils")

    import albumentations as A
    import cv2
    import numpy as np
    from torch.utils.data import Dataset

    names = {"Dataset", "get_training_augmentation", "get_validation_augmentation",
             "get_preprocessing", "to_tensor"}
    source = REFERENCE / "fusegnet_train.py"
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = {"__name__": __name__, "A": A, "cv2": cv2, "np": np,
                 "os": os, "BaseDataset": Dataset}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    # Give Dataset a stable importable module identity for worker compatibility.
    globals()["Dataset"] = namespace["Dataset"]
    _CACHE = smp, utils, namespace
    return _CACHE


def build_model(config, pretrained=True):
    smp, _, _ = load_reference()
    settings = dict(config["model"])
    if not pretrained:
        settings["encoder_weights"] = None
    return smp.Unet(**settings)


def build_loss_and_metrics():
    _, utils, _ = load_reference()
    loss = utils.base.SumOfLosses(utils.losses.DiceLoss(), utils.losses.FocalLoss())
    metrics = [utils.metrics.IoU(threshold=0.5), utils.metrics.Fscore(threshold=0.5)]
    return loss, metrics
