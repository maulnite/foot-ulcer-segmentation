"""Ganta crop/refine design on the frozen clean CP2 split.

Only the new refiner is fitted. The coarse model is the existing clean B1.
Partner definitions are AST-loaded from an audited private snapshot; no partner
top-level training, downloads, SSL changes or writes are executed.
"""

import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import cv2
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .checkpoint import link_best, random_state, restore_random_state, save_checkpoint
from .common import ROOT, digest_file, digest_json, write_json
from .cp2_data import verify_cp2_manifest
from .cp2_phase_l import _verify_precision
from .cp2_protocol import BaselineObjective, CP2Evaluator, ImageDiceCheckpoint
from .cp2_screening import CheckedAdam
from .cp2_train import gpu_snapshot, make_cp2_loader, seed_worker, sync, tensor_fingerprint
from .engine import environment_record, setup
from .reference import build_model, load_reference, verify_reference


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


DEFINITIONS = {
    "ConvBNReLU", "EfficientNetEncoder", "SCSE", "LocalAttention", "PscSE",
    "CrossLevelFusion", "RawDetailPath", "FrequencyDetailPath", "GuidedOffsetUpsample",
    "DecoderBlock", "FUSegNetDecoder", "FUSegNet", "expand_box", "CropDataset",
    "cascade_predict",
}
CONSTANTS = {
    "ENCODER", "DECODER_CHANNELS", "SE_REDUCTION", "LG_LOCAL_KERNEL", "LG_FUSION",
    "CL_PROJ_CH", "RDS_PROJ_CH", "RDS_INJECT", "FREQ_PROJ_CH", "FREQ_INJECT",
    "UP_MAX_OFFSET", "CROP_SIZE", "CROP_MARGIN", "CROP_MARGIN_EVAL", "MIN_BOX_FRAC",
    "P_RANDOM_CROP", "PROP_THR", "MAX_COMPONENTS", "MIN_COMPONENT_PX", "THRESHOLD",
}


def namespace(source, device="cpu"):
    """Execute definitions and literal config only, never operational statements."""
    tree = ast.parse(Path(source).read_text())
    result = {"__name__": __name__, "os": os, "random": random, "cv2": cv2, "np": np,
              "torch": torch, "nn": nn, "F": F, "EfficientNet": EfficientNet,
              "BaseDataset": Dataset, "DEVICE": torch.device(device)}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in CONSTANTS:
                result[name] = ast.literal_eval(node.value)
    if not CONSTANTS <= result.keys():
        raise ValueError("Missing audited partner constants")
    result.update(IMAGENET_MEAN=np.array([.485, .456, .406], dtype=np.float32),
                  IMAGENET_STD=np.array([.229, .224, .225], dtype=np.float32))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in DEFINITIONS]
    if len(nodes) != len(DEFINITIONS) or {n.name for n in nodes} != DEFINITIONS:
        raise ValueError("Missing or duplicate audited definitions")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), result)
    return result


def source_key(key):
    """Map active partner tensors to the same operations in the pinned source."""
    if key.startswith("encoder.net."):
        return key.replace("encoder.net.", "encoder.", 1)
    if key.startswith("head."):
        return key.replace("head.", "segmentation_head.0.", 1)
    return (key.replace(".att.add_branch.cse.", ".attention2.cSE.")
            .replace(".att.add_branch.sse.", ".attention2.sSE.")
            .replace(".att.max_branch.cse.", ".attention1.cSE.")
            .replace(".att.max_branch.sse.", ".attention1.sSE.")
            .replace(".conv.", ".conv1."))


def mapped_model(ns, state):
    model = ns["FUSegNet"](ns["ENCODER"], False, ns["DECODER_CHANNELS"], False, False)
    mapped = {}
    for name, value in model.state_dict().items():
        key = source_key(name)
        if key not in state or state[key].shape != value.shape or state[key].dtype != value.dtype:
            raise ValueError(f"Unmatched active partner tensor: {name}")
        mapped[name] = state[key]
    model.load_state_dict(mapped, strict=True)
    if any(not torch.equal(value, state[source_key(name)]) for name, value in model.state_dict().items()):
        raise ValueError("Initial active tensors differ")
    return model, {"mapped_named_tensors": len(mapped), "all_mapped_tensors_equal": True,
                   "source_tensor_sha256": tensor_fingerprint(state),
                   "mapped_tensor_sha256": tensor_fingerprint(model.state_dict()),
                   "parameters": sum(p.numel() for p in model.parameters())}


def sources():
    from .cp2_train import training_fingerprints
    result = training_fingerprints()
    for name in ("src/foot_ulcer_segmentation/cp2_cascade.py",
                 "src/foot_ulcer_segmentation/cp2_screening.py",
                 "src/foot_ulcer_segmentation/cp2_phase_l.py",
                 "scripts/run_cp2_cascade.py", "scripts/report_cp2_cascade.py"):
        result[name] = digest_file(ROOT / name)
    return result


def check_run(root):
    registration = read(root / "preregistration.json")
    config = read(root / "protocol.json")
    if registration["sources"] != sources() or registration["config_sha256"] != digest_json(config):
        raise ValueError("Registered code/config changed")
    c = config["cascade"]
    if digest_file(root / "partner_source.py") != c["partner_source_sha256"]:
        raise ValueError("Partner snapshot changed")
    if digest_file(ROOT / c["control_root"] / "B1/best.pth") != c["coarse_checkpoint_sha256"]:
        raise ValueError("Frozen coarse checkpoint changed")
    if digest_file(ROOT / c["control_root"] / "initial_weights.pth") != c["initial_weights_sha256"]:
        raise ValueError("Shared original initializer changed")
    manifest = verify_cp2_manifest(read(root / "manifest.json"))
    if manifest["fingerprint"] != c["manifest_fingerprint"]:
        raise ValueError("Clean membership changed")
    for name, sha in registration["inputs"].items():
        if digest_file(ROOT / name) != sha:
            raise ValueError(f"Registered input changed: {name}")
    return config, manifest


def preserve(root):
    inventory = read(root / "protected_before.json")
    changed = [name for name, sha in inventory.items() if not (ROOT / name).is_file() or digest_file(ROOT / name) != sha]
    if changed:
        raise ValueError(f"Protected work changed: {changed[:5]}")
    return {"files": len(inventory), "all_unchanged": True}


def prepare(root, config):
    """Reuse immutable inputs; never rebuild the group split or fit on target."""
    if (root / "protocol.json").exists():
        if read(root / "protocol.json") != config:
            raise ValueError("Existing preparation differs")
        return verify_cp2_manifest(read(root / "manifest.json"))
    c = config["cascade"]
    old = ROOT / c["control_root"]
    original = read(old / "protocol.json")
    for key in ("seed", "split_seed", "cpu_threads", "model", "data", "training", "evaluation", "loss"):
        if config[key] != original[key]:
            raise ValueError(f"B1 protocol differs: {key}")
    if digest_file(ROOT / c["partner_source"]) != c["partner_source_sha256"]:
        raise ValueError("Partner source changed before snapshot")
    if read(old / "B1/status.json")["status"] != "completed":
        raise ValueError("Completed clean B1 is required")
    manifest = verify_cp2_manifest(read(old / "data/manifest.json"))
    if manifest["fingerprint"] != c["manifest_fingerprint"]:
        raise ValueError("Wrong clean manifest")
    if {s: len(r) for s, r in manifest["splits"].items()} != {"train": 671, "validation": 118, "test": 261}:
        raise ValueError("Unexpected clean population")
    for path, sha in ((old / "B1/best.pth", c["coarse_checkpoint_sha256"]),
                      (old / "initial_weights.pth", c["initial_weights_sha256"])):
        if digest_file(path) != sha:
            raise ValueError("Frozen B1 input checksum differs")
    shutil.copyfile(ROOT / c["partner_source"], root / "partner_source.py")
    write_json(root / "protocol.json", config)
    write_json(root / "manifest.json", manifest)
    write_json(root / "input_inventory.json", {str(p.relative_to(ROOT)): digest_file(p) for p in
               (old / "B1/best.pth", old / "B1/status.json", old / "B1/history.json",
                old / "initial_weights.pth", old / "data/manifest.json")})
    return manifest


def proposals(probability, ns):
    probability = np.asarray(probability)
    if probability.ndim != 2 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Proposal input must be finite 2D probabilities")
    height, width = probability.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((probability >= ns["PROP_THR"]).astype(np.uint8), connectivity=8)
    components = sorted([(int(stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, n)
                         if stats[k, cv2.CC_STAT_AREA] >= ns["MIN_COMPONENT_PX"]], reverse=True)[:ns["MAX_COMPONENTS"]]
    boxes = []
    for area, k in components:
        x, y, w, h = [int(v) for v in stats[k, :4]]
        box = ns["expand_box"](x, y, x + w, y + h, ns["CROP_MARGIN_EVAL"], height, width, ns["MIN_BOX_FRAC"])
        boxes.append({"xyxy": list(box), "component_pixels": area})
    return boxes


def refine(model, image, coarse, ns, device):
    """Exact Ganta proposal, normalized RGB crop, max-overlap merge and fallback."""
    if image.shape != (1, 3, 224, 224) or image.dtype != torch.float32:
        raise ValueError("Expected one normalized RGB float32 input")
    boxes = proposals(coarse, ns)
    rgb = np.ascontiguousarray(image[0].cpu().numpy().transpose(1, 2, 0))
    covered, output = np.zeros(coarse.shape, dtype=bool), np.zeros_like(coarse)
    with torch.no_grad():
        for row in boxes:
            x0, y0, x1, y1 = row["xyxy"]
            crop = cv2.resize(np.ascontiguousarray(rgb[y0:y1, x0:x1]), (ns["CROP_SIZE"],) * 2, interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).unsqueeze(0).to(device)
            probability = model(tensor).sigmoid()[0, 0].cpu().numpy()
            if not np.isfinite(probability).all():
                raise ValueError("Nonfinite refiner probabilities")
            probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            output[y0:y1, x0:x1] = np.maximum(output[y0:y1, x0:x1], probability)
            covered[y0:y1, x0:x1] = True
    return np.where(covered, output, coarse).astype(np.float32), boxes


class CropTraining(Dataset):
    """Only clean training rows can enter the partner's original crop sampler."""
    def __init__(self, rows, ns):
        if not rows or any(r["source_split"] != "train" for r in rows):
            raise ValueError("Only original train rows are eligible for refiner training")
        images = {str((ROOT / r["image_path"]).parent) for r in rows}
        masks = {str((ROOT / r["mask_path"]).parent) for r in rows}
        if len(images) != 1 or len(masks) != 1:
            raise ValueError("Mixed source directories")
        self.rows = rows
        augmentation = load_reference()[2]["get_training_augmentation"]()
        self.original = ns["CropDataset"]([r["name"] for r in rows], images.pop(), masks.pop(), augmentation, True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, mask, _ = self.original[index]
        if image.shape != (3, 224, 224) or mask.shape != (1, 224, 224) or not np.isfinite(image).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError("Invalid actual training crop")
        return image, mask


def crop_loader(config, manifest, ns, epoch):
    rows = manifest["splits"]["train"]
    seed = config["seed"] + 10007 * epoch
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    loader = DataLoader(Subset(CropTraining(rows, ns), order), batch_size=config["training"]["batch_size"],
                        num_workers=config["training"]["num_workers"], shuffle=False,
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + 1))
    return loader, [rows[i] for i in order]


def cache_validation(root, config, manifest):
    """Predicted validation proposals can be prepared on CPU while Phase C trains."""
    destination = root / "validation_coarse"
    if (destination / "manifest.json").exists():
        return load_cache(root, manifest)
    destination.mkdir(exist_ok=False)
    model = build_model(config, False)
    state = torch.load(ROOT / config["cascade"]["control_root"] / "B1/best.pth", map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    model.eval().requires_grad_(False)
    loader, rows = make_cp2_loader(config, manifest, "validation")
    ns = namespace(root / "partner_source.py")
    offset, records = 0, []
    # Batch1 matches later inference; avoid batch-size-dependent GPU/CPU kernels.
    with torch.no_grad():
        for images, truth in loader:
            for image, mask in zip(images, truth):
                row = rows[offset]
                probability = model(image[None]).sigmoid()[0, 0].numpy()
                file = destination / f"{offset:03d}.npz"
                with file.open("xb") as stream:
                    np.savez_compressed(stream, image=image.numpy(), truth=mask[0].numpy(), probability=probability)
                records.append({"sample_id": row["sample_id"], "file": str(file.relative_to(root)),
                                "sha256": digest_file(file), "proposal_count": len(proposals(probability, ns))})
                offset += 1
    if offset != 118:
        raise ValueError("Incomplete full validation cache")
    write_json(destination / "manifest.json", records)
    return load_cache(root, manifest)


def load_cache(root, manifest):
    records = read(root / "validation_coarse/manifest.json")
    if [r["sample_id"] for r in records] != [r["sample_id"] for r in manifest["splits"]["validation"]]:
        raise ValueError("Validation cache membership/order differs")
    result = []
    for row in records:
        if digest_file(root / row["file"]) != row["sha256"]:
            raise ValueError("Validation cache changed")
        with np.load(root / row["file"], allow_pickle=False) as data:
            result.append({key: data[key].copy() for key in ("image", "truth", "probability")})
    return result


def full_validation(model, cached, rows, config, ns, device, boundaries=False):
    if len(cached) != len(rows) or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Validation prediction count/identities differ")
    evaluator = CP2Evaluator(config["evaluation"], boundaries=boundaries)
    model.eval()
    for row, data in zip(rows, cached):
        output, _ = refine(model, torch.from_numpy(data["image"])[None], data["probability"], ns, device)
        evaluator.add(row, data["truth"], output)
    return evaluator


def train(root, device_name, resume=False):
    config, manifest = check_run(root)
    run = root / "refiner"
    identity = digest_json({"registration": read(root / "preregistration.json")})
    if (run / "status.json").exists():
        previous = read(run / "status.json")
        if previous["status"] == "completed":
            if previous["identity"] != identity or digest_file(run / "best.pth") != previous["best_checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
            return previous
        if not resume:
            raise FileExistsError("Explicit inspected --resume required")
    else:
        run.mkdir(exist_ok=False)
        resume = False
    setup(config["seed"], config["cpu_threads"])
    _verify_precision(config)
    verify_reference()
    device = torch.device(device_name)
    ns = namespace(root / "partner_source.py", device_name)
    initial = torch.load(ROOT / config["cascade"]["control_root"] / "initial_weights.pth", map_location="cpu")
    model, proof = mapped_model(ns, initial["state_dict"])
    del initial
    model.to(device)
    settings = config["training"]
    parameters = dict(model.named_parameters())
    optimizer = CheckedAdam(model, {k: parameters[k] for k in ("encoder.net._conv_stem.weight", "head.weight")},
                            run / "first_step_checks.json", lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{
        k: v for k, v in settings["scheduler"].items() if k not in ("type", "monitor")})
    policy, history, first = ImageDiceCheckpoint(), [], 0
    objective = BaselineObjective("B1")
    environment = environment_record(config, manifest, device)
    if resume:
        if read(run / "environment.json")["packages"] != environment["packages"]:
            raise ValueError("Environment changed")
        state = torch.load(run / "last.pth", map_location=device)
        if state["identity"] != identity:
            raise ValueError("Resume config/input/code mismatch")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        optimizer.first_step_checks = state["first_step_checks"]
        scheduler.load_state_dict(state["scheduler"])
        policy, history, first = ImageDiceCheckpoint(**state["policy"]), state["history"], state["completed_epochs"]
        restore_random_state(state["rng"])
        del state
    else:
        write_json(run / "environment.json", environment)
        write_json(run / "initialization_verified.json", proof)
    cached = load_cache(root, manifest)
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    write_json(run / "status.json", {"status": "running", "pid": os.getpid(), "completed_epochs": first, "started_utc": utc()})
    for epoch in range(first, settings["max_epochs"]):
        if policy.bad_epochs >= settings["early_stopping_patience"]:
            break
        setup(config["seed"] + 10007 * epoch, config["cpu_threads"])
        loader, rows = crop_loader(config, manifest, ns, epoch)
        model.train()
        losses, times, inputs = [], [], hashlib.sha256()
        begin = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        for image, truth in loader:
            inputs.update(image.numpy().tobytes())
            inputs.update(truth.numpy().tobytes())
            sync(device)
            step = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(image.to(device)), truth.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite refiner training loss")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError("Nonfinite or missing gradients")
            optimizer.step()
            sync(device)
            times.append(time.monotonic() - step)
            losses.append(float(loss.detach()))
        validation = full_validation(model, cached, manifest["splits"]["validation"], config, ns, device)
        score = validation.summary()["all"]["image_based"]["dice"]
        selected = policy.update(score, epoch + 1)
        scheduler.step(score)
        row = {"epoch": epoch + 1, "train_crop_loss": float(np.mean(losses)),
               "validation_full_cascade_image_dice": score, "selected": selected, "bad_epochs": policy.bad_epochs,
               "learning_rate_used": lr, "learning_rate_next": optimizer.param_groups[0]["lr"],
               "steps": len(losses), "step_seconds_mean": float(np.mean(times)),
               "epoch_compute_seconds": time.monotonic() - begin,
               "training_order_sha256": digest_json([r["sample_id"] for r in rows]),
               "augmented_crop_sha256": inputs.hexdigest(), "gpu": gpu_snapshot()}
        history.append(row)
        save_checkpoint(run / "last.pth", {"identity": identity, "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "rng": random_state(), "policy": policy.state_dict(), "history": history,
                        "completed_epochs": epoch + 1, "first_step_checks": optimizer.first_step_checks})
        if selected:
            link_best(run / "last.pth", run / "best.pth")
        write_json(run / "history.json", history)
        write_json(run / "status.json", {"status": "running", "pid": os.getpid(),
                   "completed_epochs": epoch + 1, "selected_epoch": policy.epoch, "latest": row})
        print(json.dumps(row), flush=True)
    state = torch.load(run / "best.pth", map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)
    fixture = torch.from_numpy(cached[0]["image"])[None].to(device)
    model.eval()
    with torch.no_grad():
        expected = model(fixture).clone()
        next(model.parameters()).add_(1)
        model.load_state_dict(state["state_dict"], strict=True)
        replay = model(fixture)
    if not torch.equal(expected, replay):
        raise ValueError("Refiner checkpoint replay differs")
    score = full_validation(model, cached, manifest["splits"]["validation"], config, ns, device).summary()["all"]["image_based"]["dice"]
    if score != policy.best:
        raise ValueError("Selected full-validation cascade score not reproducible")
    result = {"status": "completed", "identity": identity, "completed_epochs": len(history),
              "selected_epoch": state["completed_epochs"], "selected_validation_image_dice": score,
              "finished_utc": utc(), "seconds_this_invocation": time.monotonic() - started,
              "first_step_checks": optimizer.first_step_checks, "checkpoint_reload_bit_exact": True,
              "best_checkpoint_sha256": digest_file(run / "best.pth"),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
    write_json(run / "status.json", result)
    return result


def decision(coarse, cascade):
    guards = {
        "overall_image_dice_higher": cascade["all"]["image_based"]["dice"] > coarse["all"]["image_based"]["dice"],
        "small_image_dice_not_lower": cascade["subgroups"]["small"]["image_based"]["dice"] >= coarse["subgroups"]["small"]["image_based"]["dice"],
        "boundary_iou_not_lower": cascade["all"]["image_based"]["boundary_iou"] >= coarse["all"]["image_based"]["boundary_iou"],
        "no_additional_foreground_misses": cascade["gt_positive"]["total_foreground_misses"] <= coarse["gt_positive"]["total_foreground_misses"],
        "no_additional_empty_fp_images": cascade["gt_empty"]["images_with_prediction"] <= coarse["gt_empty"]["images_with_prediction"],
    }
    return {"validation_only": True, "selected": "cascade" if all(guards.values()) else "coarse_B1", "guards": guards,
            "capacity_guard": "not applied to explicitly two-model cascade; extra cost reported"}


def save_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate(root, split):
    """Frozen CPU evaluation; source selection must precede test/external access."""
    from .cp2_external import binary_mask, buffers_fingerprint, restore_probability, verify_dataset
    config, manifest = check_run(root)
    if split not in ("validation", "test", "CO2Wounds"):
        raise ValueError("Unknown evaluation population")
    if split != "validation" and not (root / "selection.json").is_file():
        raise ValueError("Lock source-validation selection before test/external inference")
    output = root / "evaluation" / split
    if (output / "result.json").exists():
        return read(output / "result.json")
    output.mkdir(parents=True, exist_ok=False)
    ns = namespace(root / "partner_source.py")
    refiner_state = read(root / "refiner/status.json")
    if refiner_state["status"] != "completed" or digest_file(root / "refiner/best.pth") != refiner_state["best_checkpoint_sha256"]:
        raise ValueError("Completed immutable refiner required")
    refiner = ns["FUSegNet"](ns["ENCODER"], False, ns["DECODER_CHANNELS"], False, False)
    state = torch.load(root / "refiner/best.pth", map_location="cpu")
    refiner.load_state_dict(state["state_dict"], strict=True)
    del state
    refiner.eval().requires_grad_(False)
    coarse_model = None
    external = None
    if split == "validation":
        rows = manifest["splits"][split]
        cached = load_cache(root, manifest)
        iterator = ((row, torch.from_numpy(data["image"])[None], data["truth"], data["probability"], None)
                    for row, data in zip(rows, cached))
    else:
        coarse_model = build_model(config, False).eval().requires_grad_(False)
        state = torch.load(ROOT / config["cascade"]["control_root"] / "B1/best.pth", map_location="cpu")
        coarse_model.load_state_dict(state["state_dict"], strict=True)
        del state
        if split == "test":
            loader, rows = make_cp2_loader(config, manifest, "test")

            def source_iter():
                offset = 0
                for images, truth in loader:
                    for image, mask in zip(images, truth):
                        yield rows[offset], image[None], mask[0].numpy(), None, None
                        offset += 1
            iterator = source_iter()
        else:
            external = ROOT / config["cascade"]["external_root"]
            target = read(external / "dataset_manifest.json")
            if target["fingerprint"] != config["cascade"]["external_fingerprint"]:
                raise ValueError("External population differs from prior frozen comparison")
            verify_dataset(target)
            rows = target["eligible"]
            cache = {r["sample_id"]: r for r in read(external / "input_cache.json")}

            def external_iter():
                for row in rows:
                    record = cache[row["sample_id"]]
                    if digest_file(external / record["file"]) != record["sha256"]:
                        raise ValueError("External normalized input changed")
                    image = np.load(external / record["file"], allow_pickle=False)
                    truth = binary_mask(cv2.imread(str(ROOT / row["mask_path"]), cv2.IMREAD_UNCHANGED))
                    yield row, torch.from_numpy(image)[None], truth, None, record["geometry"]
            iterator = external_iter()
    before = {"refiner": buffers_fingerprint(refiner)}
    if coarse_model is not None:
        before["coarse"] = buffers_fingerprint(coarse_model)
    evaluators = {name: CP2Evaluator(config["evaluation"], boundaries=True) for name in ("coarse", "cascade")}
    for name in evaluators:
        (output / name).mkdir()
    exported, counts, timings = [], [], []
    started = time.monotonic()
    with torch.no_grad():
        for i, (row, image, truth, coarse, geometry) in enumerate(iterator):
            begin = time.monotonic()
            if coarse is None:
                coarse = coarse_model(image).sigmoid()[0, 0].numpy()
            cascade, boxes = refine(refiner, image, coarse, ns, torch.device("cpu"))
            timings.append(time.monotonic() - begin)
            record = {"sample_id": row["sample_id"], "proposals": len(boxes), "boxes": boxes}
            for name, probability in (("coarse", coarse), ("cascade", cascade)):
                native = restore_probability(probability, geometry) if geometry is not None else probability
                evaluators[name].add(row, truth, native)
                path = output / name / f"{i:04d}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(stream, probability=probability,
                                        prediction=(native >= .5).astype(np.uint8), truth=truth.astype(np.uint8))
                record[name] = {"file": str(path.relative_to(output)), "sha256": digest_file(path)}
            exported.append(record)
            counts.append(len(boxes))
    if len(exported) != len(rows) or len({r["sample_id"] for r in exported}) != len(rows):
        raise ValueError("Evaluation prediction count mismatch")
    if buffers_fingerprint(refiner) != before["refiner"] or (coarse_model is not None and buffers_fingerprint(coarse_model) != before["coarse"]):
        raise ValueError("Inference changed BatchNorm buffers")
    for name, evaluator in evaluators.items():
        save_csv(output / f"{name}.csv", evaluator.rows)
    write_json(output / "prediction_manifest.json", exported)
    result = {"status": "completed", "split": split, "images": len(rows), "finished_utc": utc(),
              "metrics": {name: e.summary() for name, e in evaluators.items()},
              "source_manifest_fingerprint": manifest["fingerprint"],
              "external_manifest_fingerprint": target["fingerprint"] if external is not None else None,
              "coarse_checkpoint_sha256": config["cascade"]["coarse_checkpoint_sha256"],
              "refiner_checkpoint_sha256": refiner_state["best_checkpoint_sha256"],
              "proposal_distribution": dict(Counter(counts)), "mean_proposals": float(np.mean(counts)),
              "seconds": time.monotonic() - started, "mean_pipeline_seconds_shared_cpu": float(np.mean(timings)),
              "timing_note": "coarse validation cache reused; test/external include coarse forward; not a dedicated latency benchmark",
              "device": "cpu", "precision": "float32", "buffers_unchanged": True,
              "training_performed": False, "selection_before_evaluation": split != "validation"}
    write_json(output / "result.json", result)
    return result
