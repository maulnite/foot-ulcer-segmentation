"""Frozen 1010/200 FUSeg inventory and the exact author shuffle/KFold recipe."""

from collections import defaultdict
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold

from foot_ulcer_segmentation.common import ROOT, digest_file, digest_json, project_path, write_json


def load_config():
    return json.loads((ROOT / "configs/xfusegnet.json").read_text())


def verify_manifest(manifest):
    if digest_json({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("FUSeg split fingerprint changed")
    for row in manifest["labeled"] + manifest["test"]:
        for kind in ["image", "mask"] if row.get("mask_path") else ["image"]:
            if digest_file(ROOT / row[f"{kind}_path"]) != row[f"{kind}_sha256"]:
                raise ValueError(f"Dataset file changed: {row['name']}")
    folds = manifest["folds"]
    validations = [name for fold in folds for name in fold["validation"]]
    expected = {r["name"] for r in manifest["labeled"]}
    if len(validations) != 1010 or set(validations) != expected or len(set(validations)) != 1010:
        raise ValueError("Each labeled sample must be validation exactly once")
    for fold in folds:
        if (len(fold["train"]) != 808 or len(fold["validation"]) != 202
                or set(fold["train"]) & set(fold["validation"])
                or set(fold["train"]) | set(fold["validation"]) != expected):
            raise ValueError("Invalid five-fold partition")
    if expected & {r["name"] for r in manifest["test"]}:
        raise ValueError("Test filenames overlap labeled pool")
    return manifest


def prepare(config):
    destination = project_path(config["data"]["manifest"])
    if destination.exists():
        manifest = verify_manifest(json.loads(destination.read_text()))
        if manifest["data_settings"] != config["data"]:
            raise ValueError("Existing split settings differ")
        return manifest
    root, staging = project_path(config["data"]["root"]), project_path(config["data"]["staging"])
    labeled, test = [], []
    for split, expected in config["data"]["source_counts"].items():
        names = [n for n in os.listdir(root / split / "images") if (root / split / "images" / n).is_file()]
        if len(names) != expected or any(Path(n).suffix.lower() != ".png" for n in names):
            raise ValueError(f"Unexpected {split} inventory")
        if split != "test" and {p.name for p in (root / split / "labels").iterdir() if p.is_file()} != set(names):
            raise ValueError("Unpaired labeled images")
        for name in names:
            image = root / split / "images" / name
            with Image.open(image) as opened:
                rgb = np.asarray(opened.convert("RGB"))
            if rgb.shape != (512, 512, 3):
                raise ValueError("FUSeg must retain native 512x512 images")
            import hashlib
            row = {"name": name, "provided_split": split, "image_path": str(image.relative_to(ROOT)),
                   "image_sha256": digest_file(image), "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
            if split != "test":
                mask = root / split / "labels" / name
                with Image.open(mask) as opened:
                    values = np.asarray(opened.convert("L"))
                if values.shape != (512, 512):
                    raise ValueError("Unexpected mask size")
                row.update(mask_path=str(mask.relative_to(ROOT)), mask_sha256=digest_file(mask),
                           mask_values=np.unique(values).tolist(), gt_pixels=int((values >= 127.5).sum()))
                labeled.append(row)
            else:
                row["mask_path"] = None
                test.append(row)
    if len({r["name"] for r in labeled}) != 1010:
        raise ValueError("Labeled filenames must be unique before merging provided splits")
    for row in labeled:
        for kind, folder in [("image", "images"), ("mask", "labels")]:
            target = staging / "train" / folder / row["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            source = ROOT / row[f"{kind}_path"]
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError("Staging link differs")
            elif target.exists():
                raise ValueError("Refusing to overwrite staging files")
            else:
                target.symlink_to(source)
    initial = os.listdir(staging / "train/images")
    if set(initial) != {r["name"] for r in labeled}:
        raise ValueError("Staging contains unexpected entries")
    shuffled = initial.copy()
    random.Random(42).shuffle(shuffled)
    by_name = {r["name"]: r for r in labeled}
    folds = []
    for index, (train, valid) in enumerate(KFold(n_splits=5, shuffle=False).split(shuffled), 1):
        train_names, valid_names = [shuffled[i] for i in train], [shuffled[i] for i in valid]
        train_hashes = {by_name[n]["rgb_sha256"] for n in train_names}
        folds.append({"fold": index, "train": train_names, "validation": valid_names,
                      "validation_images_identical_rgb_to_train": sum(by_name[n]["rgb_sha256"] in train_hashes for n in valid_names)})
    groups = defaultdict(list)
    for row in labeled + test:
        groups[row["rgb_sha256"]].append({"name": row["name"], "provided_split": row["provided_split"]})
    manifest = {"data_settings": config["data"], "initial_directory_order": initial, "shuffled_order": shuffled,
                "labeled": labeled, "test": test, "folds": folds,
                "identical_rgb_groups": [rows for rows in groups.values() if len(rows) > 1],
                "oof_examples": [fold["validation"][0] for fold in folds],
                "test_examples": [row["name"] for row in sorted(test, key=lambda row: row["name"])[::40]],
                "test_masks_available": False, "original_author_directory_order_available": False}
    manifest["fingerprint"] = digest_json(manifest)
    write_json(destination, verify_manifest(manifest))
    return manifest
