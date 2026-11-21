"""Read-only historical CP1 caches and instrumented original Cascade inference."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import torch

from .common import ROOT, digest_file
from .partner_cascade import load_model, load_partner_namespace
from .presentation import aggregate, assert_metrics, binary_image, choose, measure, read, read_table


def cp1_cases(settings):
    run = ROOT/settings["cp1_run"]
    cache = read(ROOT/settings["cp1_cache"])
    split = read(run/"split-manifest.json")
    identity = cache["identity"]
    if (identity["checkpoint_sha256"] != digest_file(run/"best.pth")
            or identity["split_fingerprint"] != split["fingerprint"]
            or identity["threshold"] != .5):
        raise ValueError("CP1 cache/checkpoint/split identity changed")
    if identity["evaluation_sha256"]["evaluation-test.csv"] != digest_file(run/"evaluation-test.csv"):
        raise ValueError("Historical CP1 evaluation changed")
    records = {r["name"]: r for r in cache["predictions"] if r["split"] == "test"}
    expected = {r["name"]: r for r in read_table(run/"evaluation-test.csv")}
    sources = {r["name"]: r for r in split["splits"]["test"]}
    if set(records) != set(expected) or set(records) != set(sources):
        raise ValueError("CP1 test membership differs")
    cases, metrics = {}, []
    for name, record in records.items():
        source = sources[name]
        for kind in ["image", "mask"]:
            if digest_file(ROOT/source[f"{kind}_path"]) != source[f"{kind}_sha256"]:
                raise ValueError("CP1 source checksum changed")
        if digest_file(ROOT/record["path"]) != record["sha256"]:
            raise ValueError("CP1 prediction checksum changed")
        rgb = np.asarray(Image.open(ROOT/source["image_path"]).convert("RGB"))
        truth = binary_image(ROOT/source["mask_path"])
        prediction = binary_image(ROOT/record["path"])
        row = {"name": name, **measure(truth, prediction)}
        assert_metrics(row, expected[name])
        metrics.append(row)
        cases[name] = {"rgb": rgb, "truth": truth, "predictions": {"CP1": prediction},
                       "metrics": {"CP1": row}, "epochs": {"CP1": cache["selected_epoch"]},
                       "sample_id": name, "provenance": [{
                           "model": "CP1 original", "split": "historical test n=278", "seed": 42,
                           "sample_id": name, "selected_epoch": cache["selected_epoch"],
                           "checkpoint_sha256": identity["checkpoint_sha256"],
                           "cache_sha256": record["sha256"], "manifest_fingerprint": split["fingerprint"],
                           "image_sha256": source["image_sha256"], "mask_sha256": source["mask_sha256"],
                           "metrics_recomputed": row, "source": "REAL / CACHED"}]}
    miss = choose(metrics, lambda r: r["gt_pixels"] > 0 and r["tp"] == 0, lambda r: r["gt_pixels"])
    empty = choose(metrics, lambda r: r["gt_pixels"] == 0 and r["fp"] > 0, lambda r: -r["fp"])
    under = choose(metrics, lambda r: r["tp"] > 0 and r["fn"] > r["fp"],
                   lambda r: r["recall"], excluded=[miss, empty])
    selected = [cases[name] for name in [miss, empty, under]]
    for case, label in zip(selected, ["Luka kecil terlewat", "GT kosong · FP terbesar", "Undersegmentation"]):
        case["label"] = label
    return selected, {"verified_test_masks": len(metrics), "summary": aggregate(metrics),
                      "selection": "minimum GT pixels among TP=0; maximum FP on GT-empty; minimum recall with TP>0 and FN>FP; ties lexicographic"}


def proposal_boxes(namespace, probability):
    """Replay the source proposal rule, not a ground-truth crop."""
    height, width = probability.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (probability >= namespace["PROP_THR"]).astype(np.uint8), connectivity=8)
    components = [(int(stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, len(stats))
                  if stats[k, cv2.CC_STAT_AREA] >= namespace["MIN_COMPONENT_PX"]]
    boxes = []
    for _, k in sorted(components, reverse=True)[:namespace["MAX_COMPONENTS"]]:
        x, y, w, h = [int(stats[k, field]) for field in
                       [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
        boxes.append(namespace["expand_box"](x, y, x+w, y+h, namespace["CROP_MARGIN_EVAL"],
                                             height, width, namespace["MIN_BOX_FRAC"]))
    return boxes


def original_cascade_cases(settings, trace_directory):
    partner = (ROOT/settings["partner_root"]).resolve()
    source = partner/settings["partner_source"]
    replay = ROOT/settings["original_ganta_replay"]
    protocol, result = read(replay/"protocol.json"), read(replay/"result.json")
    if digest_file(source) != protocol["source_sha256"]:
        raise ValueError("Original partner source differs from the completed replay")
    namespace = load_partner_namespace(source, device="cpu")
    checkpoint_dir = partner/"output_chronic_wound/checkpoints"/protocol["run_name"]
    coarse, coarse_info = load_model(namespace, checkpoint_dir/"stage1_coarse.pth")
    refiner, refiner_info = load_model(namespace, checkpoint_dir/"stage2_refiner.pth")
    if result["models"] != {"coarse": coarse_info, "refiner": refiner_info}:
        raise ValueError("Original checkpoint metadata changed")
    tables = {stage: {r["name"]: r for r in read_table(replay/f"{stage}_per_image.csv")}
              for stage in ["coarse", "cascade"]}
    if set(tables["coarse"]) != set(tables["cascade"]) or len(tables["coarse"]) != 278:
        raise ValueError("Original Ganta replay cohort changed")
    rows = [{"name": name, **{k: int(row[k]) for k in ["tp", "tn", "fp", "fn"]},
             "gain": float(row["dice"])-float(tables["coarse"][name]["dice"])}
            for name, row in tables["cascade"].items()]
    gain = choose(rows, lambda r: r["tp"]+r["fn"] > 0 and r["gain"] > 0, lambda r: -r["gain"])
    failure = choose(rows, lambda r: r["tp"]+r["fn"] == 0 and r["fp"] > 0, lambda r: -r["fp"])
    names = [gain, failure]
    raw = partner/"data/raw/chronic_wound/test"
    dataset = namespace["Dataset"](names, str(raw/"images"), str(raw/"labels"))
    inventory = read(replay/"input_inventory.json")
    cases, evidence = [], []
    for index, name in enumerate(names):
        captures = []
        def capture(module, inputs, output):
            captures.append((inputs[0].detach().cpu().clone(), output.detach().cpu().clone()))
        handle = refiner.register_forward_hook(capture)
        x, y, _ = dataset[index]
        tensor = torch.from_numpy(x)[None]
        started = perf_counter()
        try:
            with torch.no_grad():
                probability, final, count = namespace["cascade_predict"](coarse, refiner, tensor)
        finally:
            handle.remove()
        elapsed = perf_counter()-started
        boxes = proposal_boxes(namespace, probability)
        if count != len(boxes) or count != len(captures):
            raise ValueError("Trace does not match original forward calls")
        reconstructed, covered = np.zeros_like(probability), np.zeros(probability.shape, bool)
        crops, crop_predictions, private = [], [], {"coarse": probability, "final": final}
        for crop_index, (box, (crop_tensor, logits)) in enumerate(zip(boxes, captures)):
            x0, y0, x1, y1 = box
            actual = crop_tensor[0].numpy().transpose(1, 2, 0)
            expected = cv2.resize(np.ascontiguousarray(x.transpose(1, 2, 0)[y0:y1, x0:x1]),
                                  (namespace["CROP_SIZE"], namespace["CROP_SIZE"]), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_array_equal(actual, expected)
            crop_probability = torch.sigmoid(logits)[0, 0].numpy()
            resized = cv2.resize(crop_probability, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
            reconstructed[y0:y1, x0:x1] = np.maximum(reconstructed[y0:y1, x0:x1], resized)
            covered[y0:y1, x0:x1] = True
            crops.append(np.rint(np.clip(actual*namespace["IMAGENET_STD"]+namespace["IMAGENET_MEAN"], 0, 1)*255).astype(np.uint8))
            crop_predictions.append(crop_probability >= .5)
            private[f"crop_input_{crop_index}"] = actual
            private[f"crop_probability_{crop_index}"] = crop_probability
        np.testing.assert_array_equal(final, np.where(covered, reconstructed, probability))
        paths = {kind: raw/folder/name for kind, folder in [("image", "images"), ("mask", "labels")]}
        hashes = {kind: digest_file(path) for kind, path in paths.items()}
        if any(inventory[str(path)] != hashes[kind] for kind, path in paths.items()):
            raise ValueError("Original Ganta input changed")
        rgb = np.asarray(Image.open(paths["image"]).convert("RGB"))
        truth = y[0] > .5
        np.testing.assert_array_equal(truth, binary_image(paths["mask"]))
        metrics = {"coarse": measure(truth, probability >= .5), "cascade": measure(truth, final >= .5)}
        for stage in metrics:
            assert_metrics(metrics[stage], tables[stage][name])
        trace = Path(trace_directory)/(Path(name).stem+".npz")
        if trace.exists():
            raise FileExistsError(trace)
        np.savez_compressed(trace, **private)
        record = {"sample_id": name, "source": "REAL / FRESH CPU", "split": "Ganta historical test n=278",
                  "seed": 42, "source_sha256": protocol["source_sha256"], "models": result["models"],
                  "image_sha256": hashes["image"], "mask_sha256": hashes["mask"], "metrics": metrics,
                  "boxes_xyxy": boxes, "forward_calls": 1+count, "wall_seconds_with_trace": elapsed,
                  "trace_sha256": digest_file(trace), "crop_input_and_stitch_exactly_verified": True}
        evidence.append(record)
        cases.append({"sample_id": name, "label": "Gain Dice terbesar" if index == 0 else "GT kosong · FP terbesar",
                      "rgb": rgb, "truth": truth, "coarse": probability, "final": final,
                      "boxes": boxes, "crops": crops, "crop_predictions": crop_predictions,
                      "coarse_metrics": metrics["coarse"], "final_metrics": metrics["cascade"]})
        print(f"Original Cascade verified {name}: {count} actual crops, {elapsed:.2f}s CPU", flush=True)
    return cases, evidence
