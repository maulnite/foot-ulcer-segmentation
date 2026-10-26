#!/usr/bin/env python3
"""Audit pinned FUSegNet source using small CPU probes; never start training.

Run: .venv/bin/python scripts/audit_reference.py --model-check
The optional model check requires efficientnet-pytorch==0.7.1.
The source snapshot is local/ignored; the JSON report is written under outputs/.
"""

import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
from types import SimpleNamespace
from typing import List, Optional, Union


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "mrinal054/FUSegNet"
REVISION = "60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc"
REFERENCE = ROOT / "source/upstream/FUSegNet" / REVISION


def snapshot():
    metadata = REFERENCE / "SNAPSHOT.json"
    if not metadata.exists():
        archive = subprocess.check_output([
            "gh-orie", "api", f"repos/{REPOSITORY}/tarball/{REVISION}"
        ])
        tree = json.loads(subprocess.check_output([
            "gh-orie", "api", f"repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
        ]))
        if tree.get("truncated"):
            raise ValueError("Reference Git tree is truncated")
        blobs = {v["path"]: v["sha"] for v in tree["tree"] if v["type"] == "blob"}
        hashes = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            for member in bundle:
                if not member.isfile():
                    continue
                relative = member.name.partition("/")[2]
                if not (relative.endswith((".py", ".json", ".md", ".txt")) or relative == "LICENSE"):
                    continue
                path = PurePosixPath(relative)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe reference path: {relative}")
                payload = bundle.extractfile(member).read()
                blob_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                if blob_hash != blobs[relative]:
                    raise ValueError(f"Reference Git blob mismatch: {relative}")
                target = REFERENCE / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError(f"Changed reference file: {target}")
                target.write_bytes(payload)
                hashes[relative] = hashlib.sha256(payload).hexdigest()
        metadata.write_text(json.dumps({
            "repository": REPOSITORY, "commit": REVISION,
            "verified_files": sorted(hashes), "sha256": hashes,
        }, indent=2) + "\n")
    record = json.loads(metadata.read_text())
    if record["commit"] != REVISION or not record.get("sha256"):
        raise ValueError("Missing pinned snapshot hashes")
    for path, digest in record["sha256"].items():
        if hashlib.sha256((REFERENCE / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reference snapshot was modified: {path}")
    return record


def definitions(relative, namespace, names=None):
    """Load audited definitions only; do not execute upstream top-level scripts."""
    path = REFERENCE / relative
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and (names is None or node.name in names)]
    scope = {"__name__": "reference_audit", **namespace}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def cpu_probes(check_model):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as torch_functional

    torch.set_num_threads(2)
    torch.manual_seed(0)  # Audit-only seed, not a recovered training seed.
    modules = definitions("segmentation_models_pytorch/base/modules.py", {
        "torch": torch, "nn": nn, "InPlaceABN": None,
    })
    functional = definitions("segmentation_models_pytorch/utils/functional.py", {"torch": torch})
    base = definitions("segmentation_models_pytorch/utils/base.py", {"nn": nn, "re": re})
    focal = definitions("segmentation_models_pytorch/losses/_functional.py", {
        "torch": torch, "F": torch_functional, "Optional": Optional,
    }, {"focal_loss_with_logits"})["focal_loss_with_logits"]
    losses = definitions("segmentation_models_pytorch/utils/losses.py", {
        "torch": torch, "Tensor": torch.Tensor, "Optional": Optional,
        "base": SimpleNamespace(**base), "F": SimpleNamespace(**functional),
        "Activation": modules["Activation"], "_Loss": nn.modules.loss._Loss,
        "partial": partial, "focal_loss_with_logits": focal,
        "BINARY_MODE": "binary", "MULTICLASS_MODE": "multiclass", "MULTILABEL_MODE": "multilabel",
    }, {"DiceLoss", "FocalLoss"})
    probability = torch.tensor([0.01, 0.99])
    target = torch.tensor([0.0, 1.0])
    observed = losses["FocalLoss"]()(probability, target)
    expected = focal(probability, target, alpha=None, gamma=2.0)
    corrected = focal(torch.logit(probability), target, alpha=None, gamma=2.0)
    assert torch.allclose(observed, expected)
    assert not torch.allclose(observed, corrected)
    boundary = functional["_threshold"](torch.tensor([0.5]), threshold=0.5).item()
    assert boundary == 0

    # Execute the original checkpoint conditional on two synthetic validation logs.
    source = ast.parse((REFERENCE / "fusegnet_train.py").read_text())
    checkpoint_node = next(node for node in source.body if isinstance(node, ast.For))
    checkpoint_if = next(node for node in checkpoint_node.body
                         if isinstance(node, ast.If) and "best_vloss" in ast.unparse(node.test))
    checkpoint_code = compile(ast.Module(body=[checkpoint_if], type_ignores=[]), "checkpoint_probe", "exec")
    state = {"best_vloss": 1_000_000.0, "best_viou": 0.0, "cnt_patience": 0,
             "val_loss_key": "dice_loss + focal_loss", "save_model": False}
    events = []
    with redirect_stdout(io.StringIO()):
        for epoch, (loss, iou) in enumerate([(1.0, 0.9), (1.1, 0.8)]):
            state.update(epoch=epoch, save_model=False,
                         valid_logs={state["val_loss_key"]: loss, "iou_score": iou})
            exec(checkpoint_code, state)
            events.append({"epoch": epoch, "validation_loss": loss, "validation_iou": iou,
                           "saved": state["save_model"], "tracked_best_iou": state["best_viou"]})
    assert events[0]["tracked_best_iou"] == 0.0 and events[1]["saved"]

    # Execute original inference/evaluation conditionals on synthetic masks.
    from sklearn.metrics import confusion_matrix
    inline_tree = ast.parse((REFERENCE / "fusegnet_test.py").read_text())
    inline_if = next(node for node in ast.walk(inline_tree) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "np.array_equal(flat_mask, flat_pred)")
    offline_tree = ast.parse((REFERENCE / "utils/eval.py").read_text())
    offline_loop = next(node for node in offline_tree.body if isinstance(node, ast.For))
    offline_if = next(node for node in offline_loop.body if isinstance(node, ast.If))
    def eval_counts(node, truth, prediction):
        scope = {"np": np, "confusion_matrix": confusion_matrix,
                 "flat_mask": truth, "flat_pred": prediction,
                 "unq_mask_val": np.unique(truth), "HARD_LINE": True,
                 "ep": 1e-6, "i": 0, "name": "synthetic"}
        code = compile(ast.Module(body=[node], type_ignores=[]), "original_eval_probe", "exec")
        with redirect_stdout(io.StringIO()):
            exec(code, scope)
        return [int(scope[k]) for k in ["tn", "fp", "fn", "tp"]]
    empty = np.zeros(4, dtype=np.uint8)
    mixed = np.array([0, 0, 1, 1], dtype=np.uint8)
    inline_counts = eval_counts(inline_if, empty, empty)
    offline_counts = eval_counts(offline_if, empty, empty)
    assert inline_counts == [0, 0, 0, 4] and offline_counts == [4, 0, 0, 0]
    mixed_inline = eval_counts(inline_if, mixed, mixed)
    mixed_offline = eval_counts(offline_if, mixed, mixed)
    assert mixed_inline == [0, 0, 0, 4] and mixed_offline == [2, 0, 0, 2]
    result = {
        "device": "cpu", "threads": 2, "full_training_started": False,
        "focal_loss_on_probability_input": float(observed),
        "focal_loss_with_probability_converted_to_logits": float(corrected),
        "focal_probability_logits_mismatch_confirmed": True,
        "training_prediction_at_exact_threshold_0_5": boundary,
        "inference_prediction_at_exact_threshold_0_5": int(0.5 >= 0.5),
        "checkpoint_branch_probe": events,
        "empty_image_inline_counts_tn_fp_fn_tp": inline_counts,
        "empty_image_offline_counts_tn_fp_fn_tp": offline_counts,
        "perfect_mixed_image_inline_counts_tn_fp_fn_tp": mixed_inline,
        "perfect_mixed_image_offline_counts_tn_fp_fn_tp": mixed_offline,
    }
    if not check_model:
        return result

    from efficientnet_pytorch import EfficientNet
    from efficientnet_pytorch.utils import get_model_params, url_map, url_map_advprop

    mixin = definitions("segmentation_models_pytorch/encoders/_base.py", {
        "torch": torch, "nn": nn, "List": List,
    })["EncoderMixin"]
    encoder_scope = definitions("segmentation_models_pytorch/encoders/efficientnet.py", {
        "nn": nn, "EfficientNet": EfficientNet, "EncoderMixin": mixin,
        "get_model_params": get_model_params, "url_map": url_map, "url_map_advprop": url_map_advprop,
    })
    encoder = encoder_scope["EfficientNetEncoder"]
    encoder_entry = {
        "encoder": encoder,
        "params": {"stage_idxs": (11, 18, 38, 55), "out_channels": (3, 64, 48, 80, 224, 640),
                   "model_name": "efficientnet-b7"},
    }
    get_encoder = definitions("segmentation_models_pytorch/encoders/__init__.py", {
        "encoders": {"efficientnet-b7": encoder_entry},
    }, {"get_encoder"})["get_encoder"]
    decoder = definitions("segmentation_models_pytorch/decoders/unet/decoder.py", {
        "torch": torch, "nn": nn, "F": torch_functional, "md": SimpleNamespace(**modules),
    })
    initialization = definitions("segmentation_models_pytorch/base/initialization.py", {"nn": nn})
    model_base = definitions("segmentation_models_pytorch/base/model.py", {
        "torch": torch, "init": SimpleNamespace(**initialization),
    })
    heads = definitions("segmentation_models_pytorch/base/heads.py", {
        "nn": nn, "Activation": modules["Activation"],
    })
    unet = definitions("segmentation_models_pytorch/decoders/unet/model.py", {
        "Optional": Optional, "Union": Union, "List": List, "get_encoder": get_encoder,
        "SegmentationModel": model_base["SegmentationModel"],
        "SegmentationHead": heads["SegmentationHead"], "ClassificationHead": heads["ClassificationHead"],
        "UnetDecoder": decoder["UnetDecoder"],
    })["Unet"]
    model = unet(encoder_name="efficientnet-b7", encoder_weights=None,
                 classes=1, activation="sigmoid", decoder_attention_type="pscse").eval()
    calls = Counter()
    hooks = []
    for name, module in model.named_modules():
        def capture(module, args, output, name=name):
            calls[name] += 1
        hooks.append(module.register_forward_hook(capture))
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    for hook in hooks:
        hook.remove()
    assert output.shape == (1, 1, 224, 224)
    assert torch.isfinite(output).all()
    assert all(calls[f"decoder.blocks.{i}.conv2"] == 0 for i in range(5))
    assert calls["decoder.blocks.4.attention1"] == 0
    assert calls["decoder.blocks.4.attention2"] == 1
    inactive = {name: sum(p.numel() for p in module.parameters(recurse=False))
                for name, module in model.named_modules()
                if not calls[name] and sum(p.numel() for p in module.parameters(recurse=False))}
    result["model"] = {
        "input_shape": [1, 3, 224, 224], "output_shape": list(output.shape),
        "encoder_weights": None, "purpose": "shape and architecture audit only; no accuracy evaluation",
        "registered_parameters": sum(p.numel() for p in model.parameters()),
        "parameters_in_modules_called_by_forward": sum(p.numel() for p in model.parameters()) - sum(inactive.values()),
        "parameters_in_modules_not_called_by_forward": sum(inactive.values()),
        "decoder_conv2_forward_calls": [calls[f"decoder.blocks.{i}.conv2"] for i in range(5)],
        "last_decoder_maxout_forward_calls": calls["decoder.blocks.4.attention1"],
        "unused_parameter_modules": inactive,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-check", action="store_true")
    args = parser.parse_args()
    record = snapshot()
    categories = json.loads((REFERENCE / "categorized_oldDfu.json").read_text())
    category_names = {name for names in categories.values() for name in names}
    local_test_names = {p.name for p in (ROOT / "data/raw/chronic_wound/test/images").glob("*.png")}
    versions = {}
    for name in ["torch", "torchvision", "numpy", "Pillow", "pandas", "scikit-learn", "efficientnet-pytorch"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {"repository": REPOSITORY, "commit": REVISION,
                      "verified_files": len(record["verified_files"])},
        "versions": versions,
        "test_category_file": {"names": len(category_names), "local_test_names": len(local_test_names),
                               "exact_name_set_match": category_names == local_test_names,
                               "category_counts": {k: len(v) for k, v in categories.items()}},
        "cpu_probes": cpu_probes(args.model_check),
    }
    destination = ROOT / "outputs/reference-audit/report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(destination.relative_to(ROOT)),
                      "reference_files_verified": report["reference"]["verified_files"],
                      "category_names_match": report["test_category_file"]["exact_name_set_match"],
                      "model_parameters": report["cpu_probes"].get("model", {}).get("registered_parameters"),
                      "focal_input_mismatch": report["cpu_probes"]["focal_probability_logits_mismatch_confirmed"],
                      "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
