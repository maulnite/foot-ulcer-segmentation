"""Capture the authors' split algorithm and preserve a verifiable local split."""

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

from .common import ROOT, digest_file, digest_json, project_path, write_json


def author_split(names, validation_fraction, seed):
    names = list(names)
    random.Random(seed).shuffle(names)
    n_validation = int(len(names) * validation_fraction)
    if n_validation < 1 or n_validation >= len(names):
        raise ValueError("Split must contain both training and validation samples")
    return names[:-n_validation], names[-n_validation:]


def _file_record(root, source_split, name):
    image = root / source_split / "images" / name
    mask = root / source_split / "labels" / name
    if not image.is_file() or not mask.is_file():
        raise FileNotFoundError(f"Missing image/mask pair: {source_split}/{name}")
    return {"name": name, "image_path": str(image.relative_to(ROOT)),
            "mask_path": str(mask.relative_to(ROOT)),
            "image_sha256": digest_file(image), "mask_sha256": digest_file(mask)}


def verify_manifest(manifest):
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if digest_json(payload) != manifest.get("fingerprint"):
        raise ValueError("Split manifest fingerprint changed")
    for split in ["train", "validation", "test"]:
        for row in manifest["splits"][split]:
            for field in ["image", "mask"]:
                path = project_path(row[f"{field}_path"])
                if digest_file(path) != row[f"{field}_sha256"]:
                    raise ValueError(f"Dataset file changed: {path}")
    return manifest


def prepare_splits(config):
    destination = project_path(config["data"]["split_manifest"])
    if destination.exists():
        manifest = verify_manifest(json.loads(destination.read_text()))
        expected = {"validation_fraction": config["data"]["validation_fraction"],
                    "shuffle_seed": config["data"]["shuffle_seed"],
                    "data_root": config["data"]["root"]}
        if manifest["settings"] != expected:
            raise ValueError("Existing split settings differ; choose a new manifest path")
        return manifest
    root = project_path(config["data"]["root"])
    # Preserve actual os.listdir order, as the authors do; then persist it.
    initial_entries = {split: os.listdir(root / split / "images") for split in ["train", "test"]}
    ignored = {split: [name for name in names if (root / split / "images" / name).is_dir()]
               for split, names in initial_entries.items()}
    order = [name for name in initial_entries["train"] if name not in ignored["train"]]
    test_order = [name for name in initial_entries["test"] if name not in ignored["test"]]
    if any(Path(name).suffix.lower() != ".png" for name in order + test_order):
        raise ValueError("Unexpected non-PNG file; reference os.listdir would include it")
    expected = config["data"]["source_counts"]
    if len(order) != expected["train"] or len(test_order) != expected["test"]:
        raise ValueError("Dataset inventory differs from the declared source version")
    for source_split, names in [("train", order), ("test", test_order)]:
        labels = {p.name for p in (root / source_split / "labels").iterdir() if p.is_file()}
        if labels != set(names):
            raise ValueError(f"Unpaired labels in {source_split}")
    training, validation = author_split(order, config["data"]["validation_fraction"],
                                        config["data"]["shuffle_seed"])
    splits = {"train": [_file_record(root, "train", n) for n in training],
              "validation": [_file_record(root, "train", n) for n in validation],
              "test": [_file_record(root, "test", n) for n in test_order]}
    groups = defaultdict(list)
    for split, rows in splits.items():
        for row in rows:
            groups[row["image_sha256"]].append(split)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"validation_fraction": config["data"]["validation_fraction"],
                     "shuffle_seed": config["data"]["shuffle_seed"],
                     "data_root": config["data"]["root"]},
        "initial_train_file_order": order, "initial_test_file_order": test_order,
        "ignored_nonimage_directories": ignored,
        "splits": splits,
        "cross_split_identical_file_groups": sum(len(set(v)) > 1 for v in groups.values()),
        "limitations": ["Author's original directory order is unavailable; this captures the local order.",
                        "Source duplicates retained; patient independence is not established.",
                        "Source sample counts differ from those stated in the paper."]}
    manifest["fingerprint"] = digest_json(manifest)
    write_json(destination, manifest)
    return manifest


def make_dataset(config, manifest, split, augment=False, limit=None):
    from .reference import load_reference

    smp, _, functions = load_reference()
    rows = manifest["splits"][split]
    if limit is not None:
        rows = rows[:limit]
    source_split = "test" if split == "test" else "train"
    root = project_path(config["data"]["root"])
    preprocessing = smp.encoders.get_preprocessing_fn(
        config["model"]["encoder_name"], config["model"]["encoder_weights"])
    augmentation = functions["get_training_augmentation" if augment else "get_validation_augmentation"]()
    return functions["Dataset"](
        [row["name"] for row in rows], str(root / source_split / "images"),
        str(root / source_split / "labels"), augmentation=augmentation,
        preprocessing=functions["get_preprocessing"](preprocessing)), rows
