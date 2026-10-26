#!/usr/bin/env python3
"""Fetch pinned upstream data and audit it without altering original samples.

Run from any directory: python scripts/prepare_datasets.py
Requires gh-orie, Pillow, and NumPy. Does not install packages or create splits.
"""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPO = "uwm-bigdata/wound-segmentation"
REVISION = "42a272dfe0679f20675e826385925cb7562934b6"
ARCHIVE_SHA256 = "6a6977dce562430dff415cbe481fa30f71c7f49b45e018d87f9cb464795c0968"
ZIP_NAME = "azh_wound_care_center_dataset_patches.zip"
CHALLENGE = "data/Foot Ulcer Segmentation Challenge/"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def put(path, payload):
    """Keep identical existing files; refuse to replace changed originals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing file differs from source: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return Path(*path.parts)


def fetch_and_extract():
    downloads = DATA / "downloads"
    metadata = DATA / "metadata"
    downloads.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"wound-segmentation-{REVISION}.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".gz.part")
        with temporary.open("wb") as stream:
            subprocess.run(
                ["gh-orie", "api", f"repos/{REPO}/tarball/{REVISION}"],
                stdout=stream, check=True,
            )
        temporary.rename(archive)
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("Repository archive checksum mismatch; no extraction performed")
    tree_path = metadata / "upstream-tree.json"
    if not tree_path.exists():
        payload = subprocess.check_output([
            "gh-orie", "api", f"repos/{REPO}/git/trees/{REVISION}?recursive=1"
        ])
        put(tree_path, payload)
    tree = json.loads(tree_path.read_text())
    if tree.get("truncated"):
        raise ValueError("Upstream tree is truncated")
    blobs = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
    verified = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            _, _, relative = member.name.partition("/")
            if relative.startswith(CHALLENGE):
                target = DATA / "raw/fuseg_2021" / safe_relative(relative[len(CHALLENGE):])
            elif relative == f"data/wound_dataset/{ZIP_NAME}":
                target = downloads / ZIP_NAME
            elif relative == "README.md":
                target = metadata / "upstream-README.md"
            else:
                continue
            payload = bundle.extractfile(member).read()
            git_hash = hashlib.sha1(
                f"blob {len(payload)}\0".encode() + payload
            ).hexdigest()
            if git_hash != blobs[relative]["sha"]:
                raise ValueError(f"Upstream Git blob mismatch: {relative}")
            put(target, payload)
            verified += 1
    with zipfile.ZipFile(downloads / ZIP_NAME) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            relative = safe_relative(member.filename)
            if relative.parts[0] not in {"train", "test"}:
                raise ValueError(f"Unexpected ZIP entry: {member.filename}")
            # ZipFile.read verifies the member CRC as well.
            put(DATA / "raw/chronic_wound" / relative, bundle.read(member))
    print(f"Verified {verified} upstream blobs; extracted both datasets.", flush=True)
    return {
        "repository": f"https://github.com/{REPO}",
        "commit": REVISION,
        "archive": str(archive.relative_to(ROOT)),
        "archive_sha256": ARCHIVE_SHA256,
        "chronic_wound_zip_sha256": sha256_file(downloads / ZIP_NAME),
        "verified_git_blobs": verified,
        "license_status": "No standalone license file found at upstream repository root; redistribution terms need review.",
    }


def read_image(path):
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        mode = image.mode
        size = image.size
        rgb = np.asarray(image.convert("RGB"))
    return hashlib.sha256(payload).hexdigest(), hashlib.sha256(rgb.tobytes()).hexdigest(), size, mode


def audit_dataset(name, splits, expected_size):
    base = DATA / "raw" / name
    rows, summary, nonbinary_masks = [], {}, []
    pixel_groups, source_groups = defaultdict(list), defaultdict(list)
    for split in splits:
        images = sorted((base / split / "images").glob("*.png"))
        labels = sorted((base / split / "labels").glob("*.png"))
        needs_labels = not (name == "fuseg_2021" and split == "test")
        image_ids = {p.stem for p in images}
        label_ids = {p.stem for p in labels}
        if not images or len(image_ids) != len(images):
            raise ValueError(f"Missing images or duplicate IDs: {name}/{split}")
        if (needs_labels and image_ids != label_ids) or (not needs_labels and labels):
            raise ValueError(f"Image/mask pairing mismatch: {name}/{split}")
        mask_values, image_modes, mask_modes = Counter(), Counter(), Counter()
        empty_masks = 0
        for path in images:
            file_hash, pixel_hash, size, mode = read_image(path)
            if size != expected_size:
                raise ValueError(f"Unexpected dimensions {size}: {path}")
            image_modes[mode] += 1
            sample = {"split": split, "id": path.stem}
            pixel_groups[pixel_hash].append(sample)
            if name == "chronic_wound":
                # A filename stem groups patches; it does NOT establish patient identity.
                source_groups[path.stem.rsplit("_", 1)[0]].append(sample)
            row = {
                "dataset": name, "split": split, "sample_id": path.stem,
                "image_path": str(path.relative_to(ROOT)), "mask_path": "",
                "width": size[0], "height": size[1], "image_mode": mode,
                "mask_mode": "", "image_sha256": file_hash,
                "image_rgb_pixels_sha256": pixel_hash,
                "mask_sha256": "", "mask_nonzero_pixels": "", "mask_nonzero_fraction": "",
            }
            if needs_labels:
                mask_path = base / split / "labels" / path.name
                with Image.open(mask_path) as mask:
                    mask.load()
                    if mask.size != size:
                        raise ValueError(f"Mask dimensions differ: {mask_path}")
                    mask_mode = mask.mode
                    array = np.asarray(mask)
                    if array.ndim == 3 and array.shape[2] == 3:
                        if not np.all(array == array[..., :1]):
                            raise ValueError(f"RGB mask channels disagree: {mask_path}")
                        array = array[..., 0]
                    elif array.ndim != 2:
                        raise ValueError(f"Unsupported mask format: {mask_path}")
                    values = np.unique(array)
                    if not set(values.tolist()).issubset({0, 1, 255}):
                        unexpected = ~np.isin(array, [0, 1, 255])
                        nonbinary_masks.append({
                            "split": split, "sample_id": path.stem,
                            "values": values.tolist(),
                            "nonbinary_pixels": int(np.count_nonzero(unexpected)),
                        })
                    foreground = int(np.count_nonzero(array))
                mask_modes[mask_mode] += 1
                mask_values.update(int(value) for value in values)
                empty_masks += foreground == 0
                row.update({
                    "mask_path": str(mask_path.relative_to(ROOT)), "mask_mode": mask_mode,
                    "mask_sha256": sha256_file(mask_path), "mask_nonzero_pixels": foreground,
                    "mask_nonzero_fraction": foreground / (size[0] * size[1]),
                })
            rows.append(row)
        summary[split] = {
            "images": len(images), "masks": len(labels), "size": list(expected_size),
            "image_modes": dict(image_modes), "mask_modes": dict(mask_modes),
            "mask_values": sorted(mask_values), "empty_masks": empty_masks if needs_labels else None,
        }
        print(f"{name}/{split}: {len(images)} images, {len(labels)} masks; pairing and decoding verified.", flush=True)
    duplicates = [v for v in pixel_groups.values() if len(v) > 1]
    cross_split_duplicates = [v for v in duplicates if len({s['split'] for s in v}) > 1]
    source_overlap = [v for v in source_groups.values() if len({s['split'] for s in v}) > 1]
    manifest = DATA / "manifests" / f"{name}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "splits": summary, "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest),
        "identical_rgb_pixel_groups": duplicates,
        "cross_split_identical_rgb_pixel_groups": cross_split_duplicates,
        "cross_split_filename_source_groups": source_overlap,
        "nonbinary_mask_samples": nonbinary_masks,
        "patient_independence": "Not established; filenames are not verified patient identifiers.",
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true",
                        help="Audit extracted files without downloading compressed archives")
    args = parser.parse_args()
    if args.audit_only:
        previous = DATA / "metadata/preparation-report.json"
        if not previous.exists():
            raise ValueError("Audit-only requires existing raw data and its preparation report")
        provenance = json.loads(previous.read_text())["provenance"]
    else:
        provenance = fetch_and_extract()
    chronic, chronic_rows = audit_dataset("chronic_wound", ["train", "test"], (224, 224))
    fuseg, fuseg_rows = audit_dataset("fuseg_2021", ["train", "validation", "test"], (512, 512))
    counts = chronic["splits"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "datasets": {"chronic_wound": chronic, "fuseg_2021": fuseg},
        "chronic_wound_matches_paper_counts": counts["train"]["images"] == 810 and counts["test"]["images"] == 200,
        "cross_dataset_overlap": "Paper reports overlap; crop-to-full-image correspondence has not been resolved. Do not combine datasets as independent samples.",
        "transformations": "None. Original bytes and provided splits preserved; no new validation splits created.",
        "files_audited": len(chronic_rows) + len(fuseg_rows),
    }
    destination = DATA / "metadata/preparation-report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {destination.relative_to(ROOT)}", flush=True)
    if not report["chronic_wound_matches_paper_counts"]:
        print("NOTE: Chronic Wound source counts differ from the paper; original split preserved.", flush=True)
    for name, result in report["datasets"].items():
        print(f"{name}: {len(result['cross_split_identical_rgb_pixel_groups'])} cross-split exact pixel duplicate groups; "
              f"{len(result['cross_split_filename_source_groups'])} cross-split filename source groups.", flush=True)


if __name__ == "__main__":
    main()
