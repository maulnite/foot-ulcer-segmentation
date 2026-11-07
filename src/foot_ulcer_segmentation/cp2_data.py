"""CP2-only decoded-pixel audit and immutable duplicate-group manifests."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import random

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .common import ROOT, digest_file, digest_json, project_path, write_json


def audit_pairs(root, expected_shape=(224, 224)):
    root = project_path(root)
    rows, thumbnails = [], {}
    for split in ("train", "test"):
        pictures = {p.name: p for p in (root / split / "images").glob("*.png")}
        masks = {p.name: p for p in (root / split / "labels").glob("*.png")}
        if not pictures or pictures.keys() != masks.keys():
            raise ValueError(f"Unpaired or empty source split: {split}")
        for name in sorted(pictures):
            with Image.open(pictures[name]) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(masks[name]) as image:
                mask = np.asarray(image)
            if mask.ndim == 3:
                if not np.all(mask == mask[..., :1]):
                    raise ValueError(f"Mask channels disagree: {split}/{name}")
                mask = mask[..., 0]
            if rgb.shape[:2] != expected_shape or mask.shape != expected_shape:
                raise ValueError(f"Unexpected image/mask grid: {split}/{name}")
            if not np.isin(mask, [0, 255]).all():
                raise ValueError(f"Nonbinary Chronic Wound annotation: {split}/{name}")
            group = hashlib.sha256(rgb.tobytes()).hexdigest()
            # Source images have substantial black canvas. Screen the RGB-only
            # content crop, otherwise padding dominates perceptual similarity.
            visible = np.any(rgb > 8, axis=2)
            ys, xs = np.where(visible)
            content = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(xs) else rgb
            thumb = cv2.resize(content, (32, 32), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            coefficients = cv2.dct(gray)[:8, :8].ravel()[1:]
            phash = sum(int(v > np.median(coefficients)) << i for i, v in enumerate(coefficients))
            thumbnails[group] = (phash, thumb)
            rows.append({
                "sample_id": f"{split}/{name}", "name": name, "source_split": split,
                "image_path": str(pictures[name].relative_to(ROOT)),
                "mask_path": str(masks[name].relative_to(ROOT)),
                "image_sha256": digest_file(pictures[name]),
                "mask_sha256": digest_file(masks[name]),
                "rgb_group": group, "mask_pixels_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "gt_pixels": int(np.count_nonzero(mask)), "canvas_pixels": int(mask.size),
            })
    near = []
    groups = sorted(thumbnails)
    for i, first in enumerate(groups):
        phash, thumb = thumbnails[first]
        for second in groups[i + 1:]:
            other_hash, other_thumb = thumbnails[second]
            distance = (phash ^ other_hash).bit_count()
            if distance <= 2:
                mae = float(np.abs(thumb.astype(float) - other_thumb).mean())
                if mae <= 3.0:
                    similarity = float(structural_similarity(thumb, other_thumb, channel_axis=-1, data_range=255))
                    if similarity >= .995:
                        near.append({"first_rgb_group": first, "second_rgb_group": second,
                                     "phash_hamming": distance, "thumbnail_rgb_mae": mae,
                                     "rgb_content_ssim": similarity})
    return rows, near


def grouped_membership(rows, validation_fraction, seed):
    """Quarantine conflicting masks; hold out source-test RGB before seeded splitting."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["rgb_group"]].append(dict(row))
    conflicts = {k for k, v in grouped.items() if len({r["mask_pixels_sha256"] for r in v}) > 1}
    cross = {k for k, v in grouped.items() if len({r["source_split"] for r in v}) > 1}
    eligible = sorted(k for k, v in grouped.items()
                      if k not in conflicts | cross and v[0]["source_split"] == "train")
    random.Random(seed).shuffle(eligible)
    n_valid = round(len(eligible) * validation_fraction)
    if not 0 < n_valid < len(eligible):
        raise ValueError("Not enough eligible groups for train/validation")
    validation = set(eligible[-n_valid:])
    splits = {"train": [], "validation": [], "test": []}
    excluded = []
    for group, members in sorted(grouped.items()):
        members.sort(key=lambda r: r["sample_id"])
        if group in conflicts:
            excluded.extend({**r, "reason": "conflicting_masks_for_identical_rgb"} for r in members)
            continue
        if group in cross:
            excluded.extend({**r, "reason": "source_train_rgb_present_in_source_test"}
                            for r in members if r["source_split"] == "train")
            members = [r for r in members if r["source_split"] == "test"]
        chosen, *duplicates = members
        excluded.extend({**r, "reason": "redundant_identical_rgb_and_mask"} for r in duplicates)
        split = "test" if chosen["source_split"] == "test" else (
            "validation" if group in validation else "train")
        splits[split].append(chosen)
    for members in splits.values():
        members.sort(key=lambda r: r["sample_id"])
    return splits, excluded, {
        "source_images": dict(Counter(r["source_split"] for r in rows)),
        "exact_rgb_groups": len(grouped), "duplicate_rgb_groups": sum(len(v) > 1 for v in grouped.values()),
        "conflicting_annotation_groups": len(conflicts), "cross_source_split_groups": len(cross),
        "excluded_by_reason": dict(Counter(r["reason"] for r in excluded)),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "original_files_modified_or_deleted": 0,
    }


def verify_cp2_manifest(manifest, files=True):
    if digest_json({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("CP2 manifest hash changed")
    seen_groups, seen_samples = set(), set()
    for rows in manifest["splits"].values():
        for row in rows:
            if row["rgb_group"] in seen_groups or row["sample_id"] in seen_samples:
                raise ValueError("Duplicate group/sample occurs more than once across CP2 partitions")
            seen_groups.add(row["rgb_group"])
            seen_samples.add(row["sample_id"])
            if files:
                for kind in ("image", "mask"):
                    if digest_file(project_path(row[f"{kind}_path"])) != row[f"{kind}_sha256"]:
                        raise ValueError("Source image/mask changed")
    return manifest


def prepare_cp2_manifest(config, directory):
    destination = project_path(directory) / "manifest.json"
    if destination.exists():
        manifest = verify_cp2_manifest(json.loads(destination.read_text()))
        if manifest["data_settings"] != config["data"] or manifest["split_seed"] != config["split_seed"]:
            raise ValueError("CP2 split configuration changed")
        return manifest
    rows, near = audit_pairs(config["data"]["root"], tuple(config["data"]["image_size"]))
    splits, excluded, audit = grouped_membership(rows, config["data"]["validation_fraction"], config["split_seed"])
    roles = {r["rgb_group"]: s for s, members in splits.items() for r in members}
    for candidate in near:
        candidate["roles"] = [roles.get(candidate[k], "quarantined")
                              for k in ("first_rgb_group", "second_rgb_group")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "data_settings": config["data"], "split_seed": config["split_seed"],
                "splits": splits, "excluded": excluded, "audit": audit,
                "near_duplicate_screen": {"method": "RGB-only nonblack content crop; 32px pHash63 <=2, RGB MAE <=3, SSIM >=0.995",
                                          "policy": "screening only; not patient identification; no automatic merging",
                                          "candidates": near},
                "limitations": ["Duplicate-group split, NOT patient-level.",
                                "CP1 source test was previously analyzed; not pristine confirmatory evaluation.",
                                "Quarantining ambiguous annotations changes the evaluation population.",
                                "External datasets not accessed or used for configuration selection."]}
    manifest["fingerprint"] = digest_json(manifest)
    verify_cp2_manifest(manifest)
    write_json(destination, manifest)
    return manifest
