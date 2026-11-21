"""Final CP2 figures: immutable historical caches and exact coarse/refiner replay.

Only the selected, completed refiner is executed, in eval/no-grad mode on CPU.
Ground truth is used for metric verification and posthoc illustration selection,
never for inference proposals. No functions in this module fit or select models.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import ROOT, digest_file, write_json
from .cp2_cascade import check_run, decision, namespace, proposals, refine
from .cp2_external import binary_mask, restore_probability
from .cp2_protocol import CP2Evaluator
from .data import make_dataset
from .evaluation import scores
from .presentation import (
    CompletedModel, Evidence, aggregate, choose, measure, read, read_table,
)


def compare_values(expected, actual):
    """Compare entire metric trees, including subgroup sizes and empty rules."""
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ValueError("Metric tree keys differ")
        for key in expected:
            compare_values(expected[key], actual[key])
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(expected, actual, rtol=0, atol=1e-12):
            raise ValueError(f"Metric mismatch: {expected} versus {actual}")
    elif expected != actual:
        raise ValueError("Metric metadata differs")


def summary(rows):
    result = aggregate(rows)
    counts = {key: sum(row[key] for row in rows) for key in ("tp", "tn", "fp", "fn")}
    return {**result, "data_dice": scores(counts)["dice"], "small_images": sum(
        0 < row["gt_fraction"] <= .01 for row in rows)}


def stitched_probability(coarse, boxes, crop_probabilities):
    """Independent check of exact ROI max merge; no ground truth argument."""
    if len(boxes) != len(crop_probabilities):
        raise ValueError("Crop/box association differs")
    output, covered = np.zeros_like(coarse), np.zeros(coarse.shape, bool)
    for box, probability in zip(boxes, crop_probabilities):
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= coarse.shape[1] and 0 <= y0 < y1 <= coarse.shape[0]):
            raise ValueError("ROI outside the actual image")
        if probability.shape != (224, 224) or not np.isfinite(probability).all():
            raise ValueError("Invalid refiner output")
        resized = cv2.resize(probability, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
        output[y0:y1, x0:x1] = np.maximum(output[y0:y1, x0:x1], resized)
        covered[y0:y1, x0:x1] = True
    return np.where(covered, output, coarse).astype(np.float32)


def miss_transitions(coarse_rows, cascade_rows):
    """Unit is one original positive image, not a pixel or crop."""
    if set(coarse_rows) != set(cascade_rows):
        raise ValueError("Unpaired evaluation populations")
    result = dict(both_missed=0, recovered=0, newly_missed=0, overlap_both=0)
    for sample, coarse in coarse_rows.items():
        final = cascade_rows[sample]
        if coarse["gt_pixels"] != final["gt_pixels"] or coarse["gt_pixels"] <= 0:
            raise ValueError("Miss transition requires aligned positive GT")
        a, b = coarse["tp"] == 0, final["tp"] == 0
        key = "both_missed" if a and b else "recovered" if a else "newly_missed" if b else "overlap_both"
        result[key] += 1
    return result


class FinalEvidence(Evidence):
    """Reuse the old 15-model verifier; add the two completed combinations."""

    def __init__(self, settings):
        self.final_settings = settings
        super().__init__(read(ROOT/settings["historical_settings"]))
        report = read(ROOT/settings["combinations_report"])
        for name in ("B1A6H2", "L3A6H2"):
            directory = ROOT/report["run"]/name
            state = read(directory/"status.json")
            checksum = digest_file(directory/"best.pth")
            if state["status"] != "completed" or checksum != state["best_checkpoint_sha256"]:
                raise ValueError("Combination checkpoint unavailable or changed")
            for split in ("validation", "test"):
                if report["metrics"][name][split]["split_fingerprint"] != self.manifest["fingerprint"]:
                    raise ValueError("Combination cohort differs")
            self.models[name] = CompletedModel(
                name, "C+", directory, state, report["metrics"][name],
                report["selection"]["decisions"][name]["guards"], checksum)
        self.cascade_report = read(ROOT/settings["cascade_report"])
        self.cascade_root = ROOT/self.cascade_report["run"]
        self.config, cascade_manifest = check_run(self.cascade_root)
        compare_values(self.cascade_report["config"], self.config)
        if cascade_manifest != self.manifest:
            raise ValueError("Cascade source membership differs")
        if {s: len(r) for s, r in self.manifest["splits"].items()} != settings["source_cohort"]:
            raise ValueError("Unexpected source cohort")
        self.refiner_state = read(self.cascade_root/"refiner/status.json")
        state = self.refiner_state
        if state["status"] != "completed" or state["selected_epoch"] != settings["selected_refiner_epoch"]:
            raise ValueError("Only the final completed refiner checkpoint is allowed")
        if digest_file(self.cascade_root/"refiner/best.pth") != state["best_checkpoint_sha256"]:
            raise ValueError("Refiner checkpoint changed")
        history = read(self.cascade_root/"refiner/history.json")
        if max(history, key=lambda r: r["validation_full_cascade_image_dice"])["epoch"] != state["selected_epoch"]:
            raise ValueError("Selected epoch is not the earliest best")
        self.ns = namespace(self.cascade_root/"partner_source.py")
        expected = {"PROP_THR": .3, "MIN_COMPONENT_PX": 4, "MAX_COMPONENTS": 5,
                    "CROP_MARGIN_EVAL": .5, "MIN_BOX_FRAC": .25, "CROP_SIZE": 224, "THRESHOLD": .5}
        for key, value in expected.items():
            if self.ns[key] != value:
                raise ValueError("Source Cascade proposal rule differs")
        self.cascade_indices, self.cascade_tables, self.cascade_metrics = {}, {}, {}
        self.trace_checks = []

    def verify_cascade(self):
        """Recompute every source/native external metric with the locked evaluator."""
        external = ROOT/self.config["cascade"]["external_root"]
        target = read(external/"dataset_manifest.json")
        if target["fingerprint"] != self.config["cascade"]["external_fingerprint"]:
            raise ValueError("External cohort changed")
        if len(target["eligible"]) != self.final_settings["external_images"]:
            raise ValueError("External population differs")
        cache = {r["sample_id"]: r for r in read(external/"input_cache.json")}
        selected = read(self.cascade_root/"selection.json")
        compare_values(self.cascade_report["selection"], selected)
        locked = datetime.fromisoformat(selected["locked_utc"]).timestamp()
        for split in ("validation", "test", "CO2Wounds"):
            directory = self.cascade_root/"evaluation"/split
            result = read(directory/"result.json")
            compare_values(self.cascade_report["results"][split], result)
            rows = self.manifest["splits"][split] if split != "CO2Wounds" else target["eligible"]
            index = read(directory/"prediction_manifest.json")
            if [r["sample_id"] for r in rows] != [r["sample_id"] for r in index]:
                raise ValueError("Cascade prediction order or identity differs")
            if not result["buffers_unchanged"] or result["training_performed"]:
                raise ValueError("Expected frozen inference outputs")
            evaluators = {name: CP2Evaluator(self.config["evaluation"], boundaries=True)
                          for name in ("coarse", "cascade")}
            for row_number, (row, item) in enumerate(zip(rows, index), start=1):
                if split == "CO2Wounds":
                    for kind in ("image", "mask"):
                        if digest_file(ROOT/row[f"{kind}_path"]) != row[f"{kind}_sha256"]:
                            raise ValueError("External raw source changed")
                    truth = binary_mask(cv2.imread(str(ROOT/row["mask_path"]), cv2.IMREAD_UNCHANGED))
                else:
                    _, truth = self.source(split, row["sample_id"])
                coarse_probability = None
                for name, evaluator in evaluators.items():
                    path = directory/item[name]["file"]
                    if digest_file(path) != item[name]["sha256"]:
                        raise ValueError("Cascade cache changed")
                    if split != "validation" and path.stat().st_mtime < locked:
                        raise ValueError("Test/external cache predates validation lock")
                    with np.load(path, allow_pickle=False) as data:
                        probability = data["probability"]
                        if probability.dtype != np.float32 or probability.shape != (224, 224):
                            raise ValueError("Unexpected model output grid")
                        native = restore_probability(probability, cache[row["sample_id"]]["geometry"]) if split == "CO2Wounds" else probability
                        np.testing.assert_array_equal(data["truth"], truth)
                        np.testing.assert_array_equal(data["prediction"], native >= .5)
                        evaluator.add(row, truth, native)
                        if name == "coarse":
                            coarse_probability = probability.copy()
                boxes = proposals(coarse_probability, self.ns)
                if boxes != item["boxes"] or len(boxes) != item["proposals"]:
                    raise ValueError("Saved proposals are not derived from coarse prediction")
                if split == "CO2Wounds" and row_number % 100 == 0:
                    print(f"Native external metric checks: {row_number}/{len(rows)} images", flush=True)
            self.cascade_tables[split], self.cascade_metrics[split] = {}, {}
            for name, evaluator in evaluators.items():
                compare_values(result["metrics"][name], evaluator.summary())
                saved = read_table(directory/f"{name}.csv")
                if len(saved) != len(evaluator.rows):
                    raise ValueError("CSV population differs")
                for old, fresh in zip(saved, evaluator.rows):
                    for key, value in fresh.items():
                        if isinstance(value, (float, int)) and not isinstance(value, bool):
                            if abs(float(old[key])-value) > 1e-12:
                                raise ValueError("Per-image CSV differs")
                        elif old[key] != str(value):
                            raise ValueError("CSV identity differs")
                measured = {r["name"]: r for r in evaluator.rows}
                self.cascade_tables[split][name] = measured
                self.cascade_metrics[split][name] = summary(list(measured.values()))
                self.checks.append({"model": name, "split": split, "verified_predictions": len(rows),
                                    "all_region_boundary_PFOM_CSV_metrics": "passed"})
            self.cascade_indices[split] = {r["sample_id"]: r for r in index}
            print(f"Verified Cascade + paired coarse: {split}, {len(rows)} images", flush=True)
        compare_values(decision(*[self.cascade_report["results"]["validation"]["metrics"][n]
                                  for n in ("coarse", "cascade")]),
                       {k: selected[k] for k in ("validation_only", "selected", "guards", "capacity_guard")})
        self.metrics["Cascade"] = {s: self.cascade_metrics[s]["cascade"] for s in ("validation", "test")}

    def cascade_case(self, split, sample, name="cascade"):
        item = self.cascade_indices[split][sample]
        path = self.cascade_root/"evaluation"/split/item[name]["file"]
        with np.load(path, allow_pickle=False) as data:
            return data["probability"].copy(), data["prediction"].astype(bool)

    def select_cases(self):
        """Declared extreme examples; unique IDs, no selection of model weights."""
        base, final = (self.cascade_tables["validation"][n] for n in ("coarse", "cascade"))
        rows = list(final.values())
        delta = lambda r: r["dice"]-base[r["name"]]["dice"]
        small = choose(rows, lambda r: 0 < r["gt_fraction"] <= .01 and delta(r) > 0, lambda r: -delta(r))
        standalone = lambda r: max(self.tables[n]["validation"][r["name"]]["dice"] for n in ("A6", "H2"))-self.tables["B1"]["validation"][r["name"]]["dice"]
        single = choose(rows, lambda r: r["gt_pixels"] > 0 and standalone(r) > 0,
                        lambda r: -standalone(r), excluded=[small])
        max_fp = lambda r: max(r["fp"], base[r["name"]]["fp"], *[
            self.tables[n]["validation"][r["name"]]["fp"] for n in ("A6", "H2")])
        empty = choose(rows, lambda r: r["gt_pixels"] == 0 and max_fp(r) > 0,
                       lambda r: -max_fp(r), excluded=[small, single])
        bad = choose(rows, lambda r: r["gt_pixels"] > 0 and delta(r) < 0,
                     delta, excluded=[small, single, empty])
        return [(small, "Cascade · gain luka kecil"), (single, "Standalone · gain A6/H2"),
                (empty, "GT kosong · false positive"), (bad, "Cascade · regresi")]

    def gallery_case(self, sample, label):
        rgb, truth = self.source("validation", sample)
        predictions = {"B1": self.cascade_case("validation", sample, "coarse")[1],
                       **{n: self.case(n, "validation", sample)[2] for n in ("A6", "H2")},
                       "Cascade": self.cascade_case("validation", sample)[1]}
        return {"sample_id": sample, "label": label, "rgb": rgb, "truth": truth,
                "predictions": predictions, "metrics": {n: measure(truth, p) for n, p in predictions.items()}}

    def trace(self, samples, destination):
        """Capture actual input/output of every refiner forward and replay max stitch."""
        ns = self.ns
        model = ns["FUSegNet"](ns["ENCODER"], False, ns["DECODER_CHANNELS"], False, False)
        checkpoint = torch.load(self.cascade_root/"refiner/best.pth", map_location="cpu")
        if checkpoint["completed_epochs"] != self.refiner_state["selected_epoch"]:
            raise ValueError("Checkpoint epoch differs")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        model.eval().requires_grad_(False)
        params = sum(p.numel() for p in model.parameters())
        self.parameter_count = params
        before = {n: p.detach().clone() for n, p in model.named_buffers()}
        index = {r["sample_id"]: r for r in read(self.cascade_root/"validation_coarse/manifest.json")}
        dataset, source_rows = make_dataset(self.config, self.manifest, "validation", augment=False)
        positions = {r["sample_id"]: i for i, r in enumerate(source_rows)}
        traces = []
        for sample, label in samples:
            record = index[sample]
            path = self.cascade_root/record["file"]
            if digest_file(path) != record["sha256"]:
                raise ValueError("Actual cached input tensor changed")
            rgb, truth = self.source("validation", sample)
            with np.load(path, allow_pickle=False) as data:
                image, coarse = data["image"], data["probability"]
                np.testing.assert_array_equal(data["truth"], truth)
            # Invoke the exact source preprocessing, including its float64
            # intermediate before to_tensor(float32). A hand-written float32
            # approximation is close numerically, but not byte-identical.
            expected_image, expected_truth = dataset[positions[sample]]
            np.testing.assert_array_equal(image, expected_image)
            np.testing.assert_array_equal(truth, expected_truth[0])
            np.testing.assert_array_equal(coarse, self.cascade_case("validation", sample, "coarse")[0])
            actual_crops, actual_probabilities = [], []

            def capture(module, inputs, output):
                actual_crops.append(inputs[0].detach().cpu().numpy()[0].copy())
                actual_probabilities.append(output.detach().sigmoid()[0, 0].cpu().numpy().copy())

            handle = model.register_forward_hook(capture)
            try:
                final, boxes = refine(model, torch.from_numpy(image).unsqueeze(0), coarse, ns, torch.device("cpu"))
            finally:
                handle.remove()
            if boxes != self.cascade_indices["validation"][sample]["boxes"]:
                raise ValueError("Fresh replay ROI changed")
            coords = [b["xyxy"] for b in boxes]
            np.testing.assert_array_equal(final, stitched_probability(coarse, coords, actual_probabilities))
            cached = self.cascade_case("validation", sample)[0]
            np.testing.assert_array_equal(final, cached)
            case = self.gallery_case(sample, label)
            display_crops = [np.clip((c.transpose(1, 2, 0)*ns["IMAGENET_STD"]+ns["IMAGENET_MEAN"])*255, 0, 255).round().astype(np.uint8) for c in actual_crops]
            traces.append({**case, "boxes": coords, "coarse": coarse, "final": final,
                           "crops": display_crops, "crop_probabilities": actual_probabilities})
            file = Path(destination)/(Path(sample).stem+".npz")
            np.savez_compressed(file, image=image, coarse=coarse, final=final,
                                boxes=np.asarray(coords), crops=np.asarray(actual_crops),
                                crop_probabilities=np.asarray(actual_probabilities))
            self.trace_checks.append({"sample_id": sample, "boxes_xyxy": coords,
                "trace_sha256": digest_file(file), "source_input_npz_sha256": record["sha256"],
                "forward_count_refiner": len(actual_crops), "full_image_evaluation": case["metrics"],
                "normalized_input_matches_raw_RGB": True, "cached_final_probability_bit_exact": True,
                "independent_max_stitch_bit_exact": True, "GT_used_for_proposals": False})
            print(f"Replayed actual refiner: {sample}, {len(coords)} crop(s), bit-exact", flush=True)
        for n, p in model.named_buffers():
            if not torch.equal(before[n], p):
                raise ValueError("Eval replay changed BatchNorm buffers")
        write_json(Path(destination)/"validation.json", {"checks": self.trace_checks, "parameters": params,
                   "buffers_unchanged": True, "training_performed": False, "device": "cpu"})
        return traces

    def model_metadata(self):
        result = {n: {"family": m.family, "checkpoint_sha256": m.checkpoint_sha256,
                  "epoch": m.status["selected_epoch"], "seed": 42, "guards": m.guards,
                  "checkpoint": str((m.root/"best.pth").relative_to(ROOT))}
                  for n, m in self.models.items()}
        result["Cascade"] = {"family": "Cascade", "epoch": self.refiner_state["selected_epoch"],
            "checkpoint_sha256": self.refiner_state["best_checkpoint_sha256"], "seed": 42,
            "checkpoint": str((self.cascade_root/"refiner/best.pth").relative_to(ROOT)),
            "coarse_checkpoint_sha256": self.models["B1"].checkpoint_sha256,
            "coarse_epoch": self.models["B1"].status["selected_epoch"],
            "guards": self.cascade_report["selection"]["guards"],
            "capacity_guard": "NOT APPLICABLE: two-model Cascade, not the 1.01x single-model budget"}
        return result

    def case_metadata(self, sample):
        row = self.rows["validation"][sample]
        return {"sample_id": sample, "source_split": "validation", "image_sha256": row["image_sha256"],
            "GT_sha256": row["mask_sha256"], "predictions": {
                **{n: self.provenance(n, "validation", sample) for n in ("A6", "H2")},
                **{n: {"cache_sha256": self.cascade_indices["validation"][sample][key]["sha256"],
                       "metrics": self.cascade_tables["validation"][key][sample]}
                   for n, key in (("B1", "coarse"), ("Cascade", "cascade"))}},
            "proposal_boxes": self.cascade_indices["validation"][sample]["boxes"],
            "B1_version": "paired CPU coarse; historical GPU B1 retained separately in scoreboard"}
