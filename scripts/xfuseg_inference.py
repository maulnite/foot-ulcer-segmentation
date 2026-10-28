"""Held-out-fold validation and the author's five-probability test ensemble."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from foot_ulcer_segmentation.common import ROOT, digest_file, digest_json, project_path, write_json
from foot_ulcer_segmentation.evaluation import SegmentationEvaluator
from xfuseg_data import prepare
from xfuseg_train import load_definitions, verify_source


def mean_probability(probabilities):
    maps = np.asarray(probabilities)
    if maps.shape[0] != 5 or maps.dtype != np.float32 or not np.isfinite(maps).all():
        raise ValueError("Ensemble requires five finite float32 sigmoid probability maps")
    if maps.min() < 0 or maps.max() > 1:
        raise ValueError("Invalid probabilities")
    # Exactly the numpy array / mean(axis=0) used in xfusegnet_test.py.
    return np.mean(maps, axis=0)


def evaluate_oof(evaluator, name, soft_truth, probability):
    # Only evaluation is binarized; original training receives mask/255 unchanged.
    return evaluator.add(name, np.asarray(soft_truth) >= .5, probability)


class TestImages(Dataset):
    """Same image decoding/preprocessing, omitting the unavailable/unused GT read."""
    def __init__(self, rows, functions, preprocessing):
        self.rows = rows
        self.augmentation = functions["get_validation_augmentation"]()
        self.preprocessing = functions["get_preprocessing"](preprocessing)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(str(ROOT / self.rows[index]["image_path"])), cv2.COLOR_BGR2RGB)
        image = self.augmentation(image=image)["image"]
        return self.preprocessing(image=image)["image"]


def store_probability(path, array):
    if array.shape != (512, 512) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Expected a finite native-resolution float32 probability map")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest_file(path)}


def store_mask(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array >= .5).astype(np.uint8) * 255).save(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest_file(path)}


def write_evaluation(path, evaluator):
    write_json(path, evaluator.summary())
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluator.rows[0]))
        writer.writeheader(); writer.writerows(evaluator.rows)


def infer(config, device="cuda:0"):
    verify_source(config)
    manifest = prepare(config)
    run, output = project_path(config["run_dir"]), project_path(config["workflow_dir"])
    status = json.loads((run / "status.json").read_text())
    if status["status"] != "completed" or status["smoke_only"]:
        raise ValueError("All five full training folds must finish before inference")
    smp, _, functions = load_definitions()
    torch.hub.set_dir(str(ROOT / "outputs/cache/torch/hub"))
    preprocessing = smp.encoders.get_preprocessing_fn("efficientnet-b7", "imagenet")
    staging = project_path(config["data"]["staging"])
    all_oof, all_test, checkpoints = [], [], []
    checkpoint_dir = run / "author-checkpoints"
    for fold in manifest["folds"]:
        number = fold["fold"]
        checkpoint_path = checkpoint_dir / f"best_model_{number}.pth"
        identity = {"fold": number, "checkpoint_sha256": digest_file(checkpoint_path),
                    "split_fingerprint": manifest["fingerprint"], "config_fingerprint": digest_json(config)}
        journal = output / f"inference-fold-{number}.json"
        if journal.exists():
            completed = json.loads(journal.read_text())
            if completed["identity"] != identity:
                raise ValueError("Inference checkpoint identity changed")
            for record in completed["oof"] + completed["test"]:
                for kind in ["probability", "mask"] if "mask" in record else ["probability"]:
                    if digest_file(ROOT / record[kind]["path"]) != record[kind]["sha256"]:
                        raise ValueError("Completed inference artifact changed")
            all_oof.extend(completed["oof"]); all_test.extend(completed["test"])
            checkpoints.append(identity)
            continue
        write_json(output / "inference-status.json", {"status": "running", "fold": number, "stage": "oof"})
        model = smp.Unet(**config["model"]).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        del checkpoint
        model.eval()
        oof_dataset = functions["Dataset"](fold["validation"], str(staging / "train/images"), str(staging / "train/labels"),
                   augmentation=functions["get_validation_augmentation"](), preprocessing=functions["get_preprocessing"](preprocessing))
        oof_loader = DataLoader(oof_dataset, batch_size=1, shuffle=False, num_workers=2)
        evaluator, oof_records, test_records = SegmentationEvaluator(.5), [], []
        for index, (images, truths) in enumerate(oof_loader):
            name = fold["validation"][index]
            probability = model.predict(images.to(device)).squeeze().cpu().numpy()
            metric = evaluate_oof(evaluator, name, truths[0, 0].numpy(), probability)
            base = output / "predictions/oof"
            oof_records.append({"fold": number, "name": name,
                    "probability": store_probability(base / "probabilities" / (name + ".npy"), probability),
                    "mask": store_mask(base / "masks" / name, probability), "evaluation": metric})
        write_evaluation(run / f"fold-{number}/oof-validation.json", evaluator)
        test_loader = DataLoader(TestImages(manifest["test"], functions, preprocessing), batch_size=1, shuffle=False, num_workers=2)
        write_json(output / "inference-status.json", {"status": "running", "fold": number, "stage": "test"})
        for index, images in enumerate(test_loader):
            name = manifest["test"][index]["name"]
            probability = model.predict(images.to(device)).squeeze().cpu().numpy()
            test_records.append({"fold": number, "name": name,
                "probability": store_probability(output / f"predictions/test/fold-{number}" / (name + ".npy"), probability)})
        write_json(journal, {"identity": identity, "oof": oof_records, "test": test_records})
        all_oof.extend(oof_records); all_test.extend(test_records); checkpoints.append(identity)
        del model, oof_loader, test_loader
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    oof_by_name = {row["name"]: row for row in all_oof}
    if len(all_oof) != 1010 or len(oof_by_name) != 1010 or len(all_test) != 1000:
        raise ValueError("Inference inventory mismatch")
    evaluator = SegmentationEvaluator(.5)
    for sample in manifest["labeled"]:
        prediction = oof_by_name[sample["name"]]
        if sample["name"] not in manifest["folds"][prediction["fold"] - 1]["validation"]:
            raise ValueError("OOF image was not held out from its predicting model")
        truth = cv2.imread(str(ROOT / sample["mask_path"]), 0) / 255.
        probability = np.load(ROOT / prediction["probability"]["path"], allow_pickle=False)
        evaluate_oof(evaluator, sample["name"], truth, probability)
    write_evaluation(run / "oof-validation.json", evaluator)
    indexed = {(r["fold"], r["name"]): r for r in all_test}
    ensemble_records = []
    for sample in manifest["test"]:
        name = sample["name"]
        mean = mean_probability([np.load(ROOT / indexed[(fold, name)]["probability"]["path"], allow_pickle=False) for fold in range(1, 6)])
        ensemble_records.append({"name": name,
            "probability": store_probability(output / "predictions/test/mean" / (name + ".npy"), mean),
            "mask": store_mask(output / "predictions/test/masks" / name, mean)})
    result = {"status": "completed", "split_fingerprint": manifest["fingerprint"],
              "config_fingerprint": digest_json(config), "checkpoints": checkpoints,
              "oof": all_oof, "test_per_fold": all_test, "test_ensemble": ensemble_records,
              "official_test_metrics_available": False, "official_test_metrics": None,
              "oof_is_ensemble_evaluation": False, "test_threshold": .5,
              "training_mask_policy": config["implementation"]["training_masks"],
              "evaluation_mask_policy": config["implementation"]["evaluation_masks"],
              "inference_source_sha256": digest_file(Path(__file__))}
    write_json(output / "prediction-manifest.json", result)
    write_json(output / "inference-status.json", {"status": "completed", "oof_images": 1010, "test_images": 200})
    return {"oof": evaluator.summary(), "test_ensemble_images": 200, "official_test_metrics": None}
