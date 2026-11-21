"""Frozen-checkpoint external evaluation; target data never enter model fitting."""

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import ROOT, digest_file, digest_json, write_json
from .cp2_attention import attach_attention
from .cp2_heads import AuxiliarySegmentationModel
from .reference import build_model, load_reference


def read(path):
    return json.loads(Path(path).read_text())


def rgb_digest(rgb):
    value = np.ascontiguousarray(rgb)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def image_index(directory):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if path.stem in result:
            raise ValueError("Ambiguous duplicate image stem")
        result[path.stem] = path
    return result


def binary_mask(mask):
    """CO2's native masks use 0/1, rather than the source dataset's 0/255."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if not all(np.array_equal(mask[..., 0], mask[..., channel]) for channel in range(mask.shape[-1])):
            raise ValueError("RGB mask channels disagree")
        mask = mask[..., 0]
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("Expected documented binary mask encoding")
    return (mask > 0).astype(np.uint8)


def letterbox(rgb, size=224):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    resized_h, resized_w = max(1, round(height * scale)), max(1, round(width * scale))
    top, left = (size - resized_h) // 2, (size - resized_w) // 2
    resized = cv2.resize(rgb, (resized_w, resized_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    boxed = np.zeros((size, size, 3), dtype=np.uint8)
    boxed[top:top + resized_h, left:left + resized_w] = resized
    return boxed, {"height": height, "width": width, "resized_h": resized_h,
                   "resized_w": resized_w, "top": top, "left": left, "size": size}


def restore_probability(probability, geometry):
    probability = np.asarray(probability)
    size = geometry["size"]
    if probability.shape != (size, size) or not np.isfinite(probability).all():
        raise ValueError("Invalid probability grid")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Probabilities must be in [0,1]")
    top, left = geometry["top"], geometry["left"]
    cropped = probability[top:top + geometry["resized_h"], left:left + geometry["resized_w"]]
    return cv2.resize(cropped, (geometry["width"], geometry["height"]),
                      interpolation=cv2.INTER_LINEAR).astype(np.float32)


def image_descriptor(rgb):
    # RGB-only nonblack crop; no target annotation determines preprocessing/overlap.
    ys, xs = np.where(np.any(rgb > 0, axis=2))
    content = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else rgb
    gray = cv2.cvtColor(cv2.resize(content, (32, 32)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    low = cv2.dct(gray)[:8, :8].ravel()
    bits = low > np.median(low[1:])
    code = sum(int(bit) << index for index, bit in enumerate(bits))
    return code, cv2.resize(content, (32, 32)).astype(np.float32)


def prepare_dataset(run):
    run = Path(run)
    if (run / "dataset_manifest.json").exists():
        manifest = read(run / "dataset_manifest.json")
        verify_dataset(manifest)
        if not (run / "input_cache.json").is_file():
            raise FileExistsError("Partial input preparation requires explicit inspection")
        cached = read(run / "input_cache.json")
        if {row["sample_id"] for row in cached} != {row["sample_id"] for row in manifest["eligible"]}:
            raise ValueError("Cached input population changed")
        for row in cached:
            if digest_file(run / row["file"]) != row["sha256"]:
                raise ValueError("Cached external input changed")
        return manifest
    raw = ROOT / read(run / "acquisition.json")["dataset_root"]
    roots = [p for p in raw.iterdir() if p.is_dir() and (p / "imgs").is_dir()]
    if len(roots) != 1:
        raise ValueError("Expected one official dataset root")
    data = roots[0]
    annotations = read(data / "annotations/annotations.json")
    annotated = {image["file_name"]: image for image in annotations["images"]}
    if not annotations["categories"] or any(c["name"] != "wound" for c in annotations["categories"]):
        raise ValueError("Unexpected annotation target")
    provider = {}
    for split in ("train", "val", "test"):
        for file in image_index(data / "split" / split).values():
            if file.stem in provider:
                raise ValueError("Provider splits overlap")
            provider[file.stem] = split
    source_images = []
    for file in sorted((ROOT / "data/raw/chronic_wound").glob("*/images/*.png")):
        rgb = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
        code, thumb = image_descriptor(rgb)
        source_images.append((str(file.relative_to(ROOT)), rgb_digest(rgb), code, thumb))
    source_hashes = {value[1] for value in source_images}
    images = image_index(data / "imgs")
    samples, target_groups, near = [], defaultdict(list), []
    for mask_path in sorted((data / "masks").glob("*.png")):
        if mask_path.stem not in images:
            raise ValueError(f"Mask has no source image: {mask_path.stem}")
        image_path = images[mask_path.stem]
        bgr = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask is None:
            raise ValueError(f"Undecodable pair: {mask_path.stem}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        truth = binary_mask(mask)
        if truth.shape != rgb.shape[:2] or image_path.name not in annotated:
            raise ValueError("Image/mask/COCO identities differ")
        meta = annotated[image_path.name]
        if (meta["height"], meta["width"]) != truth.shape:
            raise ValueError("COCO dimensions differ from decoded image")
        if provider[mask_path.stem] not in ("train", "val"):
            raise ValueError("Unlabeled provider-test file entered quantitative evaluation")
        group = rgb_digest(rgb)
        code, thumb = image_descriptor(rgb)
        for name, _, source_code, source_thumb in source_images:
            distance = (code ^ source_code).bit_count()
            if distance <= 4:
                mae = float(np.abs(thumb - source_thumb).mean())
                if mae <= 3:
                    near.append({"target": image_path.name, "source": name,
                                 "phash_hamming": distance, "thumbnail_rgb_mae": mae})
        row = {"sample_id": mask_path.stem, "name": image_path.name,
               "image_path": str(image_path.relative_to(ROOT)),
               "mask_path": str(mask_path.relative_to(ROOT)),
               "image_sha256": digest_file(image_path), "mask_sha256": digest_file(mask_path),
               "rgb_group": group, "binary_mask_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
               "height": truth.shape[0], "width": truth.shape[1],
               "gt_pixels": int(truth.sum()), "gt_fraction": float(truth.mean()),
               "provider_split": provider[mask_path.stem], "patient_id": None,
               "source_exact_overlap": group in source_hashes}
        samples.append(row)
        target_groups[group].append(row)
    if len(samples) != 607 or sum(v == "test" for v in provider.values()) != 157:
        raise ValueError("Official version1 labeled/unlabeled counts differ")
    if near:
        write_json(run / "near_overlap_requires_review.json", near)
        raise ValueError("Potential source-target near duplicates require review before inference")
    eligible, excluded = [], []
    for rows in target_groups.values():
        conflicts = len({row["binary_mask_sha256"] for row in rows}) > 1
        for index, row in enumerate(rows):
            reason = ("source_exact_overlap" if row["source_exact_overlap"] else
                      "conflicting_target_annotations" if conflicts else
                      "duplicate_rgb_and_mask_representative_only" if index else None)
            if reason:
                excluded.append({**row, "exclusion_reason": reason})
            else:
                eligible.append(row)
    eligible.sort(key=lambda row: row["sample_id"])
    manifest = {"dataset": "CO2Wounds-V2 Mendeley v1", "doi": "10.17632/s2w7rjwz49.1",
                "provider_labeled": len(samples), "provider_unlabeled_excluded": 157,
                "patient_ids_available": False, "published_patient_count": 96,
                "mask_encoding": "0 background / 1 wound; accepts validated 0/255 encoding",
                "annotation_licenses": annotations.get("licenses"),
                "source_images_audited": len(source_images),
                "source_target_exact_overlap": sum(r["source_exact_overlap"] for r in samples),
                "near_duplicates_flagged": near,
                "target_duplicate_groups": sum(len(rows) > 1 for rows in target_groups.values()),
                "eligible": eligible, "excluded": excluded,
                "provider_distribution": dict(Counter(row["provider_split"] for row in eligible)),
                "empty_masks": sum(row["gt_pixels"] == 0 for row in eligible)}
    manifest["fingerprint"] = digest_json(manifest)
    write_json(run / "dataset_manifest.json", manifest)
    # Cache only source-normalized 224 input tensors, never fit on target statistics.
    smp, _, _ = load_reference()
    preprocess = smp.encoders.get_preprocessing_fn("efficientnet-b7", "imagenet")
    cache = run / "inputs"
    cache.mkdir(exist_ok=False)
    cache_rows = []
    for row in eligible:
        rgb = cv2.cvtColor(cv2.imread(str(ROOT / row["image_path"])), cv2.COLOR_BGR2RGB)
        boxed, geometry = letterbox(rgb)
        tensor = np.ascontiguousarray(preprocess(boxed).transpose(2, 0, 1).astype(np.float32))
        path = cache / (row["sample_id"] + ".npy")
        np.save(path, tensor, allow_pickle=False)
        cache_rows.append({"sample_id": row["sample_id"], "geometry": geometry,
                           "file": str(path.relative_to(run)), "sha256": digest_file(path)})
    write_json(run / "input_cache.json", cache_rows)
    return manifest


def verify_dataset(manifest):
    check = copy.deepcopy(manifest)
    expected = check.pop("fingerprint")
    if digest_json(check) != expected:
        raise ValueError("Dataset manifest fingerprint changed")
    for row in manifest["eligible"]:
        for kind in ("image", "mask"):
            if digest_file(ROOT / row[kind + "_path"]) != row[kind + "_sha256"]:
                raise ValueError("External input changed")
    return True


def load_frozen_model(record):
    """Strict full-state load, followed by primary-mask-only inference."""
    if digest_file(ROOT / record["checkpoint"]) != record["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    if digest_file(ROOT / record["config"]) != record["config_sha256"]:
        raise ValueError("Frozen config changed")
    config = read(ROOT / record["config"])
    base = build_model(config, pretrained=False)
    if "attention" in record["kind"]:
        architecture = config if "phase_a" in config else read(ROOT / "configs/cp2/phase_a.json")
        attach_attention(base, record["variant"], architecture)
    checkpoint = torch.load(ROOT / record["checkpoint"], map_location="cpu")
    state = checkpoint["state_dict"]
    if record["kind"].startswith("wrapped"):
        auxiliary = torch.nn.Conv2d(32, 1, 1) if "auxiliary_head.weight" in state else None
        full = AuxiliarySegmentationModel(base, auxiliary)
    else:
        full = base
    full.load_state_dict(state, strict=True)
    del state, checkpoint
    full.eval().requires_grad_(False)
    base.eval()
    return base, {"strict_load": True, "full_registered_parameters": sum(p.numel() for p in full.parameters()),
                  "primary_registered_parameters": sum(p.numel() for p in base.parameters()),
                  "outputs_probabilities": record["outputs_probabilities"]}


def predict_primary(model, inputs, outputs_probabilities):
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("External evaluation must freeze BatchNorm/dropout")
    with torch.no_grad():
        output = model(inputs)
        probability = output if outputs_probabilities else output.sigmoid()
    if probability.shape != (len(inputs), 1, 224, 224) or not torch.isfinite(probability).all():
        raise ValueError("Invalid external prediction tensor")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Invalid probability range")
    return probability.detach().cpu().numpy()[:, 0]


def buffers_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.named_buffers():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
