"""Validate all external masks and report every frozen checkpoint together."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_external import (  # noqa: E402
    binary_mask, letterbox, read, restore_probability, verify_dataset,
)
from foot_ulcer_segmentation.cp2_protocol import boundary_iou, _pfom  # noqa: E402
from foot_ulcer_segmentation.evaluation import confusion_counts, scores  # noqa: E402
from foot_ulcer_segmentation.reference import load_reference  # noqa: E402


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save_csv(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def verify_input_alignment(run):
    manifest = read(run / "dataset_manifest.json")
    cache = {row["sample_id"]: row for row in read(run / "input_cache.json")}
    smp, _, _ = load_reference()
    preprocessing = smp.encoders.get_preprocessing_fn("efficientnet-b7", "imagenet")
    for sample in manifest["eligible"]:
        rgb = cv2.cvtColor(cv2.imread(str(ROOT / sample["image_path"])), cv2.COLOR_BGR2RGB)
        boxed, geometry = letterbox(rgb)
        record = cache[sample["sample_id"]]
        actual = np.load(run / record["file"], allow_pickle=False)
        expected = preprocessing(boxed).transpose(2, 0, 1).astype(np.float32)
        if geometry != record["geometry"] or not np.array_equal(actual, expected):
            raise ValueError("Input tensor cannot be reproduced from raw RGB with source preprocessing")
    return len(manifest["eligible"])


def validate_model(arguments):
    run_name, record = arguments
    run = Path(run_name)
    cv2.setNumThreads(1)
    output = run / "models" / record["id"]
    result = read(output / "result.json")
    manifest = read(run / "dataset_manifest.json")
    truth_rows = manifest["eligible"]
    cache = {item["sample_id"]: item for item in read(run / "input_cache.json")}
    predictions = read(output / "prediction_manifest.json")
    per_image = rows(output / "per_image.csv")
    expected = [row["sample_id"] for row in truth_rows]
    if [row["sample_id"] for row in predictions] != expected or [row["name"] for row in per_image] != expected:
        raise ValueError("Prediction identities/order differ from frozen cohort")
    if result["status"] != "completed" or result["images"] != len(expected):
        raise ValueError("Incomplete checkpoint evaluation")
    if not result["buffers_unchanged"] or not result["parameters_frozen"] or result["training_performed"]:
        raise ValueError("External inference was not frozen")
    if digest_file(ROOT / record["checkpoint"]) != record["checkpoint_sha256"]:
        raise ValueError("Historical checkpoint changed")
    selected_boundary = set(np.linspace(0, len(expected) - 1, 3, dtype=int))
    totals = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    for index, (sample, prediction, saved) in enumerate(zip(truth_rows, predictions, per_image)):
        file = output / prediction["file"]
        if digest_file(file) != prediction["sha256"]:
            raise ValueError("External probability artifact changed")
        with np.load(file, allow_pickle=False) as archive:
            probability = archive["probability"]
            if probability.dtype != np.float32:
                raise ValueError("Probability precision changed")
            native = restore_probability(probability, cache[sample["sample_id"]]["geometry"])
        truth = binary_mask(cv2.imread(str(ROOT / sample["mask_path"]), cv2.IMREAD_UNCHANGED))
        counts = confusion_counts(truth, native >= .5)
        for key in totals:
            if counts[key] != int(saved[key]):
                raise ValueError(f"Native confusion counts differ: {record['id']}/{sample['sample_id']}")
            totals[key] += counts[key]
        metrics = scores(counts)
        if not truth.any():
            metrics = {key: float(not (native >= .5).any()) for key in metrics}
        if any(abs(value - float(saved[key])) > 1e-12 for key, value in metrics.items()):
            raise ValueError("Per-image region metric differs")
        if index in selected_boundary:
            boundary = boundary_iou(truth, native >= .5, .01)
            pfom = _pfom()(truth, native >= .5)["pfom"]
            if abs(boundary - float(saved["boundary_iou"])) > 1e-12 or abs(pfom - float(saved["pfom"])) > 1e-12:
                raise ValueError("Boundary/PFOM replay differs")
    aggregate = result["metrics"]["all"]
    if totals != aggregate["counts"]:
        raise ValueError("Aggregate confusion counts differ")
    for key, value in scores(totals).items():
        if abs(value - aggregate["data_based"][key]) > 1e-12:
            raise ValueError("Aggregate pixel metric differs")
    for key, value in aggregate["image_based"].items():
        if abs(np.mean([float(row[key]) for row in per_image]) - value) > 1e-12:
            raise ValueError("Aggregate image metric differs")
    return {"model": record["id"], "predictions_verified": len(expected),
            "all_probability_checksums": True, "all_native_region_metrics_recomputed": True,
            "boundary_pfom_independently_recomputed": len(selected_boundary)}


def source_scores(record):
    registry = ROOT / "configs/cp2/external.json"
    if registry.exists():
        historical = read(registry).get("historical_source_scores", {})
        if record["id"] in historical:
            return historical[record["id"]]
    name = record["id"].split("_s")[0]
    if record["id"] == "CP1_original":
        return {split: read(ROOT / record["run"] / file)["offline_hard_line"]["image_based"]["dice"]
                for split, file in [("validation", "validation.json"), ("test", "evaluation-test.json")]}
    if record["seed"] != 42:
        metrics = read(ROOT / "reports/cp2-confirmation-results.json")["metrics"][str(record["seed"])][name]
        return {split: metrics[split]["image_dice"] for split in ("validation", "test")}
    report = {"CP2_P": "cp2-phase-p-results", "CP2_L": "cp2-phase-l-results",
              "CP2_A": "cp2-phase-a-results", "CP2_H": "cp2-phase-h-results",
              "CP1_recipe_CP2_split": "cp2-cp1-recipe-results", "CP1_matched": "cp1-matched-results"}[record["family"]]
    metrics = read(ROOT / "reports" / (report + ".json"))["metrics"][name]
    return {split: metrics[split]["all"]["image_based"]["dice"] for split in ("validation", "test")}


def comparison_table(run, records):
    table = []
    for record in records:
        result = read(run / "models" / record["id"] / "result.json")
        metrics = result["metrics"]
        all_scores = metrics["all"]
        original = source_scores(record)
        row = {"model": record["id"], "family": record["family"], "seed": record["seed"],
               "selected_epoch": record["selected_epoch"], "external_n": result["images"],
               "source_validation_image_dice_percent": original["validation"] * 100,
               "source_test_image_dice_percent": original["test"] * 100,
               "external_image_dice_percent": all_scores["image_based"]["dice"] * 100,
               "external_data_dice_percent": all_scores["data_based"]["dice"] * 100,
               "external_image_iou_percent": all_scores["image_based"]["iou"] * 100,
               "external_data_iou_percent": all_scores["data_based"]["iou"] * 100,
               "external_pixel_precision_percent": all_scores["data_based"]["precision"] * 100,
               "external_pixel_recall_percent": all_scores["data_based"]["recall"] * 100,
               "external_boundary_iou_percent": all_scores["image_based"]["boundary_iou"] * 100,
               "external_pfom": all_scores["image_based"]["pfom"],
               "total_foreground_misses": all_scores["total_foreground_misses"],
               "gt_empty_n": metrics["gt_empty"]["images"],
               "full_registered_parameters": result["model_proof"]["full_registered_parameters"],
               "evaluation_seconds_shared_cpu": result["seconds"]}
        for group in ("small", "medium", "large"):
            summary = metrics["subgroups"][group]
            row[group + "_n"] = summary["images"]
            row[group + "_image_dice_percent"] = (summary["image_based"]["dice"] * 100
                                                  if summary["image_based"] is not None else None)
        table.append(row)
    return table


def paired_effects(run, records):
    by_id = {record["id"]: rows(run / "models" / record["id"] / "per_image.csv") for record in records}
    contrasts = [("CP1_original", "B0_s42"), ("CP1_original", "B1_s42"),
                 ("B0_s42", "B1_s42")]
    contrasts += [("B1_s42", name) for name in ["L1_s42", "L2_s42", "L3_s42", "L4_s42",
                  "A1_s42", "A2_s42", "A3_s42", "A5_s42", "A6_s42", "H1_s42", "H2_s42"]]
    contrasts += [("R0_s42", name) for name in ("RH2_s42", "RA6_s42")]
    contrasts += [("C0_s42", name) for name in ("CH2_s42", "CA6_s42")]
    result = []
    for reference, candidate in contrasts:
        first, second = by_id[reference], by_id[candidate]
        if [row["name"] for row in first] != [row["name"] for row in second]:
            raise ValueError("Paired comparison populations differ")
        delta = np.array([float(b["dice"]) - float(a["dice"]) for a, b in zip(first, second)]) * 100
        result.append({"reference": reference, "candidate": candidate,
                       "mean_image_dice_delta_pp": float(delta.mean()),
                       "improved_images": int((delta > 1e-10).sum()),
                       "worse_images": int((delta < -1e-10).sum()),
                       "tied_images": int((np.abs(delta) <= 1e-10).sum())})
    return result


def replication_summary(table):
    summaries = []
    for model in ("B1", "H2", "A6"):
        selected = [row for row in table if row["model"] in {model + f"_s{seed}" for seed in (42, 43, 44)}]
        if len(selected) != 3:
            raise ValueError("Do not omit an unfavorable seed")
        row = {"model": model, "seeds": "42,43,44"}
        for key in ("external_image_dice_percent", "external_data_dice_percent", "external_boundary_iou_percent"):
            values = [value[key] for value in selected]
            row[key + "_mean"] = float(np.mean(values))
            row[key + "_sd"] = float(np.std(values, ddof=1))
        summaries.append(row)
    return summaries


def figures(run, table, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    colors = ["#2563eb" if row["model"] == "CP1_original" else
              "#64748b" if row["model"].split("_s")[0] in {"B0", "B1", "R0", "C0"} else "#0f766e"
              for row in table]
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 12), sharey=True, layout="constrained")
    y = np.arange(len(table))
    for ax, key, title in zip(axes, ["external_image_dice_percent", "external_data_dice_percent", "external_boundary_iou_percent"],
                              ["Image Dice", "Data Dice", "Boundary IoU"]):
        values = [row[key] for row in table]
        ax.barh(y, values, color=colors, height=.7)
        ax.set(xlim=(0, 100), xlabel=title + " (%)", title=title)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        for position, value in enumerate(values):
            ax.text(value + 1, position, f"{value:.2f}", va="center", fontsize=9)
    axes[0].set(yticks=y, yticklabels=[row["model"] for row in table])
    axes[0].invert_yaxis()
    fig.suptitle("CO2Wounds-V2 · 558 gambar · 26 checkpoint dibekukan\nREAL / FRESH — tanpa retraining atau fine-tuning", fontsize=16, fontweight="bold")
    fig.legend(handles=[Patch(color="#2563eb", label="CP1 original"), Patch(color="#64748b", label="Baseline tiap protokol"),
                        Patch(color="#0f766e", label="Kandidat pengembangan")], loc="outside lower center", ncol=3)
    fig.savefig(destination / "all_models.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    labels, palette = ["B1", "H2", "A6"], ["#64748b", "#0f766e", "#7c3aed"]
    for ax, key, title in zip(axes, ["external_image_dice_percent", "external_boundary_iou_percent"], ["Image Dice", "Boundary IoU"]):
        for index, (label, color) in enumerate(zip(labels, palette)):
            values = [next(row[key] for row in table if row["model"] == label + f"_s{seed}") for seed in (42, 43, 44)]
            ax.bar(index, np.mean(values), color=color, alpha=.6, width=.6)
            ax.errorbar(index, np.mean(values), yerr=np.std(values, ddof=1), color="#1e293b", capsize=5)
            for offset, (seed, value) in enumerate(zip((42, 43, 44), values)):
                ax.scatter(index + (offset - 1) * .14, value, marker=("o", "s", "^")[offset],
                           color="#1e293b", zorder=3, label=f"Seed {seed}" if index == 0 else None)
        ax.set(xticks=range(3), xticklabels=labels, ylim=(0, 100), ylabel=title + " (%)", title=title + " — mean ± SD tiga seed")
        ax.grid(axis="y", alpha=.15)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Transfer B1 / H2 / CBAM pada cohort eksternal yang sama", fontsize=14, fontweight="bold")
    fig.savefig(destination / "three_seeds.png", dpi=160)
    plt.close(fig)

    first = rows(run / "models/B0_s42/per_image.csv")
    second = rows(run / "models/B1_s42/per_image.csv")
    delta = np.array([float(b["dice"]) - float(a["dice"]) for a, b in zip(first, second)]) * 100
    area = [float(row["gt_fraction"]) * 100 for row in first]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    axes[0].scatter(area, delta, s=16, color="#0f766e", alpha=.5)
    axes[0].axhline(0, color="#64748b", linestyle="--")
    axes[0].set(xlabel="Luas GT (% kanvas native)", ylabel="Image Dice B1 − B0 (poin persentase)", ylim=(-100, 100), title="Efek koreksi focal per gambar")
    axes[1].hist(delta, bins=np.linspace(-100, 100, 31), color="#0f766e")
    axes[1].axvline(0, color="#64748b", linestyle="--")
    axes[1].axvline(delta.mean(), color="#7c3aed", label=f"Mean Δ {delta.mean():+.2f} poin")
    axes[1].set(xlabel="Image Dice B1 − B0 (poin persentase)", ylabel="Jumlah gambar", title="Seluruh 558 pasangan, tanpa seleksi kasus")
    axes[1].legend()
    fig.suptitle("B0 vs B1 — protokol CP2 sama, pasangan input focal berbeda", fontsize=14, fontweight="bold")
    fig.savefig(destination / "focal_paired_effects.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run = args.run.resolve()
    workflow = read(run / "models_workflow.json")
    if workflow["status"] != "completed" or workflow["failed"]:
        raise ValueError("Finish every registered checkpoint before opening comparison")
    records = read(run / "models.json")
    if set(workflow["finished"]) != {record["id"] for record in records}:
        raise ValueError("Do not omit models from final comparison")
    if (run / "validation.json").exists():
        raise FileExistsError("Preserve completed validation/report; use its saved results")
    for name, expected in read(run / "execution_sources.json").items():
        if digest_file(ROOT / name) != expected:
            raise ValueError("Frozen inference source changed")
    for name, expected in read(run / "execution_inputs.json").items():
        if digest_file(run / name) != expected:
            raise ValueError("Frozen dataset/geometry/protocol changed")
    verify_dataset(read(run / "dataset_manifest.json"))
    aligned_inputs = verify_input_alignment(run)
    with ProcessPoolExecutor(max_workers=min(4, args.workers), mp_context=multiprocessing.get_context("spawn")) as pool:
        checks = list(pool.map(validate_model, [(str(run), record) for record in records]))
    protected = read(run / "protected_before.json")
    changed = [name for name, expected in protected.items() if digest_file(ROOT / name) != expected]
    if changed:
        raise ValueError(f"Historical/user files changed: {changed[:5]}")
    validation = {"status": "passed", "models": len(records),
                  "raw_rgb_to_input_tensors_verified": aligned_inputs,
                  "predictions_verified": sum(item["predictions_verified"] for item in checks),
                  "boundary_pfom_checks": sum(item["boundary_pfom_independently_recomputed"] for item in checks),
                  "model_checks": checks, "protected_files": len(protected), "historical_files_unchanged": True}
    write_json(run / "validation.json", validation)
    table = comparison_table(run, records)
    paired = paired_effects(run, records)
    replication = replication_summary(table)
    save_csv(run / "comparison.csv", table)
    save_csv(run / "paired_effects.csv", paired)
    save_csv(run / "three_seed_summary.csv", replication)
    write_json(run / "comparison.json", {"models": table, "paired_effects": paired, "replication": replication,
                                         "validation": validation})
    figures(run, table, run / "figures")
    print(json.dumps({"validation": {key: value for key, value in validation.items() if key != "model_checks"},
                      "comparison": table, "replication": replication}, indent=2))


if __name__ == "__main__":
    main()
