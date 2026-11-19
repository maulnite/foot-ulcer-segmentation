"""Read-only CP1 analysis and plotting helpers; no torch or training imports."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, digest_json
from foot_ulcer_segmentation.evaluation import confusion_counts, scores

NAVY, TEAL, CORAL, BLUE = "#18324B", "#0F8B8D", "#E76F51", "#457B9D"
COUNTS = ["tp", "tn", "fp", "fn"]
METRICS = ["dice", "iou", "precision", "recall"]


def configure_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelcolor": NAVY,
        "text.color": NAVY, "axes.edgecolor": "#CCD5DF", "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "grid.color": "#E7EDF2", "grid.linewidth": .7,
        "savefig.facecolor": "white", "svg.fonttype": "none", "figure.dpi": 110,
    })


def read_json(path):
    return json.loads(Path(path).read_text())


def binary_mask(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        if not np.all(array == array[..., :1]):
            raise ValueError(f"Mask channels disagree: {path}")
        array = array[..., 0]
    if array.ndim != 2 or not np.isin(array, [0, 255]).all():
        raise ValueError(f"Expected binary 0/255 mask: {path}")
    return array > 0


def summarize(frame):
    counts = {key: int(frame[key].sum()) for key in COUNTS}
    return {"n": len(frame), "counts": counts, "data_based": scores(counts),
            "image_based": {key: float(frame[key].mean()) for key in METRICS}}


def load_analysis(run_id="cp1-baseline-01"):
    """Verify immutable artifacts, every cached mask, and final metric parity."""
    run = ROOT / "outputs/runs" / run_id
    output = ROOT / "outputs/analysis" / run_id
    export_path = output / "prediction-manifest.json"
    if not export_path.exists():
        raise FileNotFoundError(
            "Prediksi belum diekspor. Jalankan .venv-training/bin/python "
            "scripts/export_analysis_predictions.py --device cpu (inference saja).")
    context = {"run_id": run_id, "run": run, "output": output,
               "figures": output / "figures", "tables": output / "tables"}
    for key in ["config", "status", "environment", "history"]:
        context[key] = read_json(run / f"{key}.json")
    context["manifest"] = manifest = read_json(run / "split-manifest.json")
    context["export"] = export = read_json(export_path)
    config, status = context["config"], context["status"]
    assert status["status"] == "completed" and not status["smoke_only"]
    assert digest_json(config) == export["identity"]["config_fingerprint"] == status["config_fingerprint"]
    assert manifest["fingerprint"] == export["identity"]["split_fingerprint"] == status["split_fingerprint"]
    manifest_payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    assert digest_json(manifest_payload) == manifest["fingerprint"]
    assert digest_file(run / "best.pth") == export["identity"]["checkpoint_sha256"]
    assert export["selected_epoch"] == status["selected_epoch"] == 52
    assert export["identity"]["threshold"] == config["evaluation"]["prediction_threshold"] == .5
    assert export["training_performed"] is False and export["all_counts_match_final_evaluation"]
    for path, expected in context["environment"]["pipeline_sha256"].items():
        assert digest_file(ROOT / path) == expected, f"Training source changed: {path}"

    inventory = pd.read_csv(ROOT / "data/manifests/chronic_wound.csv").set_index("image_path")
    split_frames = {}
    for split, rows in manifest["splits"].items():
        frame = pd.DataFrame(rows)
        hashes = []
        for row in rows:
            for kind in ["image", "mask"]:
                assert digest_file(ROOT / row[f"{kind}_path"]) == row[f"{kind}_sha256"]
            with Image.open(ROOT / row["image_path"]) as image:
                pixel_hash = hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
            assert pixel_hash == inventory.loc[row["image_path"], "image_rgb_pixels_sha256"]
            hashes.append(pixel_hash)
        frame["pixel_hash"] = hashes
        split_frames[split] = frame
    train_hashes = set(split_frames["train"]["pixel_hash"])
    prediction_map = {(r["split"], r["name"]): r for r in export["predictions"]}
    assert len(prediction_map) == len(export["predictions"])
    context["frames"], context["summaries"], context["evaluations"] = {}, {}, {}
    for split, stem in [("validation", "validation"), ("test", "evaluation-test")]:
        assert digest_file(run / f"{stem}.csv") == export["identity"]["evaluation_sha256"][f"{stem}.csv"]
        csv = pd.read_csv(run / f"{stem}.csv")
        assert csv["name"].is_unique
        frame = split_frames[split].merge(csv, on="name", validate="one_to_one", how="outer", indicator=True)
        assert frame["_merge"].eq("both").all()
        frame = frame.drop(columns="_merge")
        prediction_paths = []
        for row in frame.to_dict("records"):
            record = prediction_map[(split, row["name"])]
            path = ROOT / record["path"]
            assert digest_file(path) == record["sha256"]
            truth, pred = binary_mask(ROOT / row["mask_path"]), binary_mask(path)
            counts = confusion_counts(truth, pred)
            assert all(counts[key] == row[key] for key in COUNTS)
            measured = scores(counts)
            if not truth.any():
                measured = {key: float(not pred.any()) for key in METRICS}
            assert all(np.isclose(measured[key], row[key], rtol=0, atol=1e-12) for key in METRICS)
            prediction_paths.append(record["path"])
        frame["prediction_path"] = prediction_paths
        frame["gt_pixels"] = frame.tp + frame.fn
        frame["pred_pixels"] = frame.tp + frame.fp
        frame["area_pct"] = frame.gt_pixels / frame[COUNTS].sum(axis=1) * 100
        frame["overlap_train"] = frame.pixel_hash.isin(train_hashes)
        frame["split"] = split
        summary, evaluation = summarize(frame), read_json(run / f"{stem}.json")
        primary = evaluation["offline_hard_line"]
        assert summary["n"] == evaluation["images"]
        assert summary["counts"] == primary["counts"]
        for family in ["data_based", "image_based"]:
            for key in METRICS:
                assert np.isclose(summary[family][key], primary[family][key], rtol=0, atol=1e-12)
        context["frames"][split], context["summaries"][split] = frame, summary
        context["evaluations"][split] = evaluation
    assert len(prediction_map) == sum(len(f) for f in context["frames"].values())
    revision = config["reference_revision"]
    categories = read_json(ROOT / "source/upstream/FUSegNet" / revision / "categorized_oldDfu.json")
    membership = {name: int(category) for category, names in categories.items() for name in names}
    assert len(membership) == sum(map(len, categories.values())) == len(context["frames"]["test"])
    assert set(membership) == set(context["frames"]["test"].name)
    context["frames"]["test"]["category"] = context["frames"]["test"].name.map(membership)
    for folder in [context["figures"], context["tables"]]:
        folder.mkdir(parents=True, exist_ok=True)
    context["verified_masks"] = len(prediction_map)
    return context


def save_figure(context, fig, name, clinical=False):
    fig.savefig(context["figures"] / f"{name}.png", dpi=200 if clinical else 300, bbox_inches="tight")
    if not clinical:
        fig.savefig(context["figures"] / f"{name}.svg", bbox_inches="tight")
    return fig


def compute_pfom(context):
    """Score all verified cached masks without changing the frozen evaluator."""
    from importlib.metadata import version
    from pfom_metrics import score_masks

    protocol = read_json(ROOT / "configs/pfom.json")
    if (protocol["empty_edges_bounded"] != {"both_empty": 1.0, "gt_empty": 0.0, "prediction_empty": 0.0}
            or protocol["empty_edges_paper_sentinel2_proxy"] != {"both_empty": 2.0, "gt_empty": 2.0, "prediction_empty": 2.0}):
        raise ValueError("PFOM configuration disagrees with implemented empty-edge policies")
    summaries = {}
    columns = ["split", "name", "category", "gt_pixels", "pred_pixels", "dice",
               "gt_edge_pixels", "pred_edge_pixels", "edge_case", "pfom", "pfom_paper_sentinel2_proxy"]
    rows = []
    for split, frame in context["frames"].items():
        measured = []
        for row in frame.to_dict("records"):
            result = score_masks(binary_mask(ROOT / row["mask_path"]),
                                 binary_mask(ROOT / row["prediction_path"]),
                                 beta=protocol["beta"], **protocol["canny"])
            measured.append(result)
        for key in measured[0]:
            frame[key] = [record[key] for record in measured]
        valid = frame.edge_case.eq("both_nonempty")
        summaries[split] = {
            "n": len(frame), "mean": float(frame.pfom.mean()), "median": float(frame.pfom.median()),
            "mean_positive_gt": float(frame.loc[frame.gt_pixels.gt(0), "pfom"].mean()),
            "n_both_edges_nonempty": int(valid.sum()),
            "mean_both_edges_nonempty": float(frame.loc[valid, "pfom"].mean()) if valid.any() else None,
            "edge_cases": {case: int(frame.edge_case.eq(case).sum())
                           for case in ["both_nonempty", "both_empty", "gt_empty", "prediction_empty"]},
            "mean_paper_sentinel2_proxy": float(frame.pfom_paper_sentinel2_proxy.mean()),
        }
        rows.append(frame.reindex(columns=columns))
    per_image = pd.concat(rows, ignore_index=True)
    if not per_image.pfom.between(0, 1).all():
        raise ValueError("Bounded PFOM must lie between zero and one")
    per_image.to_csv(context["tables"] / "pfom_per_image.csv", index=False, float_format="%.15g")
    context["pfom"] = {
        "protocol": protocol, "protocol_fingerprint": digest_json(protocol),
        "implementation_sha256": digest_file(ROOT / "scripts/pfom_metrics.py"),
        "libraries": {name: version(name) for name in ["numpy", "scipy", "scikit-image"]},
        "prediction_identity": context["export"]["identity"], "summaries": summaries,
        "per_image_csv_sha256": digest_file(context["tables"] / "pfom_per_image.csv"),
        "training_performed": False, "inference_performed": False,
    }
    (context["output"] / "pfom-metadata.json").write_text(json.dumps(context["pfom"], indent=2, allow_nan=False) + "\n")
    return per_image


def pfom_summary_table(context):
    return pd.DataFrame([
        {"Split": split.title(), "n": summary["n"], "Mean PFOM": summary["mean"],
         "Median PFOM": summary["median"], "Mean GT positif": summary["mean_positive_gt"],
         "n kedua edge ada": summary["n_both_edges_nonempty"],
         "Mean kedua edge ada": summary["mean_both_edges_nonempty"]}
        for split, summary in context["pfom"]["summaries"].items()])


def pfom_empty_table(context):
    return pd.DataFrame([
        {"Split": split.title(), "Kedua edge kosong": summary["edge_cases"]["both_empty"],
         "Hanya edge GT kosong": summary["edge_cases"]["gt_empty"],
         "Hanya edge prediksi kosong": summary["edge_cases"]["prediction_empty"],
         "Mean proxy sentinel 2": summary["mean_paper_sentinel2_proxy"]}
        for split, summary in context["pfom"]["summaries"].items()])


def pfom_category_table(context):
    rows = []
    for category, group in context["frames"]["test"].groupby("category", sort=True):
        valid = group.edge_case.eq("both_nonempty")
        rows.append({"Kategori": f"C{category}", "n": len(group),
                     "Mean PFOM": group.pfom.mean(), "Median PFOM": group.pfom.median(),
                     "n kedua edge ada": int(valid.sum()),
                     "Mean kedua edge ada": group.loc[valid, "pfom"].mean(),
                     "Mean proxy sentinel 2": group.pfom_paper_sentinel2_proxy.mean()})
    return pd.DataFrame(rows)


def pfom_plot(context):
    test = context["frames"]["test"]
    positive = test[test.gt_pixels.gt(0)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for both, color, marker, label in [(True, TEAL, "o", "Kedua edge ada"),
                                        (False, CORAL, "x", "Salah satu edge kosong")]:
        subset = positive[positive.edge_case.eq("both_nonempty").eq(both)]
        axes[0].scatter(subset.dice * 100, subset.pfom, s=25, c=color, marker=marker, alpha=.7, label=label)
    axes[0].set(title=f"Test GT positif · n={len(positive)}", xlabel="Dice per gambar (%)",
                ylabel="PFOM · 0–1, lebih tinggi lebih baik", xlim=(-3, 103), ylim=(-.04, 1.04))
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    values = [test.loc[test.category.eq(i), "pfom"] for i in range(10)]
    boxes = axes[1].boxplot(values, labels=[f"C{i}\nn={len(v)}" for i, v in enumerate(values)],
                            patch_artist=True, medianprops={"color": NAVY},
                            flierprops={"markersize": 3, "alpha": .5})
    for patch in boxes["boxes"]:
        patch.set(facecolor=TEAL, alpha=.25)
    axes[1].scatter(np.arange(1, 11), [v.mean() for v in values], marker="D", color=CORAL, s=22, label="Mean")
    axes[1].set(title="PFOM per kategori penulis", ylabel="PFOM dengan kebijakan kosong 1/0", ylim=(-.04, 1.04))
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    for ax in axes:
        ax.grid(alpha=.8); ax.set_axisbelow(True)
    fig.suptitle("07  /  PFOM melengkapi pembacaan ketepatan batas", fontsize=16, fontweight="bold")
    return save_figure(context, fig, "07_pfom_boundaries")


def metric_table(context):
    rows = []
    for family, label in [("data_based", "Data-based"), ("image_based", "Image-based")]:
        for key in METRICS:
            rows.append({"Agregasi": label, "Metrik": key.title(),
                         **{split.title(): context["summaries"][split][family][key] * 100
                            for split in ["validation", "test"]}})
    return pd.DataFrame(rows)


def paper_comparison(context):
    return pd.DataFrame([
        {"Dice": "Data-based", "Paper (%)": 92.70,
         "Run ini (%)": context["summaries"]["test"]["data_based"]["dice"] * 100},
        {"Dice": "Image-based", "Paper (%)": 86.05,
         "Run ini (%)": context["summaries"]["test"]["image_based"]["dice"] * 100},
    ]).assign(**{"Selisih (pp)": lambda frame: frame["Run ini (%)"] - frame["Paper (%)"]})


def category_table(context):
    rows = []
    for category, group in context["frames"]["test"].groupby("category", sort=True):
        rows.append({"Kategori": f"C{category}", "n": len(group),
                     "Luas min (%)": group.area_pct.min(), "Luas max (%)": group.area_pct.max(),
                     "Mean Dice (%)": group.dice.mean() * 100,
                     "Median Dice (%)": group.dice.median() * 100,
                     "Data Dice (%)": summarize(group)["data_based"]["dice"] * 100})
    return pd.DataFrame(rows)


def empty_table(context):
    rows = []
    for split, frame in context["frames"].items():
        empty = frame[frame.gt_pixels.eq(0)]
        rows.append({"Split": split.title(), "GT kosong": len(empty),
                     "Benar kosong": int(empty.pred_pixels.eq(0).sum()),
                     "Ada false positive": int(empty.pred_pixels.gt(0).sum()),
                     "FP terbesar (piksel)": int(empty.fp.max())})
    return pd.DataFrame(rows)


def overlap_table(context):
    rows = []
    for split, frame in context["frames"].items():
        for overlap in [True, False]:
            group = frame[frame.overlap_train.eq(overlap)]
            summary = summarize(group)
            rows.append({"Split": split.title(), "Piksel RGB sama dengan train": "Ya" if overlap else "Tidak",
                         "n": len(group), "Data Dice (%)": summary["data_based"]["dice"] * 100,
                         "Image Dice (%)": summary["image_based"]["dice"] * 100})
    return pd.DataFrame(rows)


def learning_curves(context):
    history = context["history"]
    epochs = [row["epoch"] for row in history]
    selected = context["status"]["selected_epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout="constrained")
    for split, color, label in [("train", TEAL, "Train"), ("validation", CORAL, "Validation")]:
        axes[0].plot(epochs, [r[split]["dice_loss + focal_loss"] for r in history], color=color, label=label)
        axes[1].plot(epochs, [r[split]["fscore"] * 100 for r in history], color=color, label=label)
    axes[0].set(title="Loss turun, lalu melandai", ylabel="Dice loss + focal loss")
    axes[1].set(title="Dice selama training", ylabel="F-score / Dice batch (%)")
    axes[1].yaxis.set_major_formatter(PercentFormatter(100))
    axes[2].step(epochs, [r["learning_rate"] for r in history], where="post", color=BLUE)
    axes[2].set(title="Learning rate", ylabel="Learning rate", yscale="log", ylim=(6e-6, 1.7e-4))
    axes[2].set_yticks([1e-5, 1e-4], ["0.00001", "0.0001"])
    for ax in axes:
        ax.axvline(selected, color=NAVY, linestyle="--", linewidth=1, label=f"Checkpoint {selected}")
        ax.set(xlabel="Epoch", xlim=(1, max(epochs)))
        ax.grid(alpha=.8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("01  /  Perjalanan baseline · 82 epoch, checkpoint terpilih 52", fontsize=16, fontweight="bold")
    return save_figure(context, fig, "01_learning_curves")


def final_metrics_plot(context):
    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    labels = ["Data Dice", "Data IoU", "Image Dice", "Image IoU"]
    for split, offset, color in [("validation", -.18, CORAL), ("test", .18, TEAL)]:
        summary = context["summaries"][split]
        values = [summary[family][key] * 100 for family, key in
                  [("data_based", "dice"), ("data_based", "iou"), ("image_based", "dice"), ("image_based", "iou")]]
        bars = ax.bar(np.arange(4) + offset, values, .34, color=color, label=split.title())
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10)
    ax.set(xticks=np.arange(4), xticklabels=labels, ylim=(0, 105), ylabel="Skor (%)",
           title="02  /  Evaluasi akhir · checkpoint 52, threshold 0.5")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(axis="y", alpha=.8); ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.12), frameon=False, ncol=2)
    return save_figure(context, fig, "02_final_metrics")


def distribution_plot(context):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    positive = []
    for split, color in [("validation", CORAL), ("test", TEAL)]:
        frame = context["frames"][split]
        values = frame.loc[frame.gt_pixels.gt(0), "dice"] * 100
        positive.append(values)
        weights = np.ones(len(values)) / len(values) * 100
        axes[0].hist(values, bins=np.arange(0, 101, 10), weights=weights, histtype="step", linewidth=2,
                     color=color, label=f"{split.title()} · n={len(values)}")
    axes[0].set(title="Distribusi pada GT berisi luka", xlabel="Dice per gambar (%)", ylabel="Proporsi gambar (%)", xlim=(0, 100))
    axes[0].legend(frameon=False, fontsize=9)
    boxes = axes[1].boxplot(positive, labels=["Validation", "Test"], patch_artist=True,
                            medianprops={"color": NAVY}, flierprops={"markersize": 4, "alpha": .5})
    for patch, color in zip(boxes["boxes"], [CORAL, TEAL]):
        patch.set(facecolor=color, alpha=.35)
    for i, values in enumerate(positive, 1):
        axes[1].text(i + .18, values.median(), f"Median {values.median():.2f}%", fontsize=10, va="center")
    axes[1].set(title="Sebaran dan median", ylabel="Dice per gambar (%)", ylim=(-3, 103), xlim=(.5, 2.9))
    for ax in axes:
        ax.grid(axis="y", alpha=.8); ax.set_axisbelow(True)
    fig.suptitle("03  /  Rata-rata belum menceritakan setiap gambar", fontsize=16, fontweight="bold")
    return save_figure(context, fig, "03_dice_distribution")


def size_plot(context):
    test = context["frames"]["test"]
    positive = test[test.gt_pixels.gt(0)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    colors = np.where(positive.overlap_train, CORAL, TEAL)
    axes[0].scatter(positive.area_pct, positive.dice * 100, c=colors, s=23, alpha=.7, edgecolors="white", linewidths=.3)
    axes[0].set(xscale="log", xlabel="Luas GT / luas gambar (%) · skala log", ylabel="Dice per gambar (%)",
                title=f"Test positif · {len(positive)} gambar", ylim=(-4, 104))
    axes[0].legend(handles=[Patch(color=TEAL, label="Tidak ada RGB identik di train"),
                            Patch(color=CORAL, label="Ada RGB identik di train")], frameon=False, fontsize=8,
                   loc="upper left", bbox_to_anchor=(0, -.23), ncol=2)
    groups = [test.loc[test.category.eq(i), "dice"] * 100 for i in range(1, 10)]
    boxes = axes[1].boxplot(groups, labels=[f"C{i}\nn={len(g)}" for i, g in enumerate(groups, 1)],
                            patch_artist=True, medianprops={"color": NAVY},
                            flierprops={"markersize": 3, "alpha": .6})
    for patch in boxes["boxes"]:
        patch.set(facecolor=TEAL, alpha=.25)
    axes[1].set(title="Kelompok ukuran dari JSON penulis", ylabel="Dice per gambar (%)", ylim=(-4, 104))
    axes[1].tick_params(axis="x", labelsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.8); ax.set_axisbelow(True)
    fig.suptitle("04  /  Error lebih besar pada sebagian luka kecil", fontsize=16, fontweight="bold")
    return save_figure(context, fig, "04_wound_size")


def empty_plot(context):
    test = context["frames"]["test"]
    empty = test[test.gt_pixels.eq(0)].sort_values(["fp", "name"], ascending=[False, True])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained", gridspec_kw={"width_ratios": [1, 2]})
    for i, (split, frame) in enumerate(context["frames"].items()):
        subset = frame[frame.gt_pixels.eq(0)]
        correct, wrong = int(subset.fp.eq(0).sum()), int(subset.fp.gt(0).sum())
        axes[0].barh(i, correct, color=TEAL)
        axes[0].barh(i, wrong, left=correct, color=CORAL)
        axes[0].text(correct / 2, i, str(correct), color="white", ha="center", va="center", fontweight="bold")
        axes[0].text(correct + wrong / 2, i, str(wrong), color="white", ha="center", va="center", fontweight="bold")
    axes[0].set(yticks=[0, 1], yticklabels=["Validation", "Test"], xlabel="Jumlah gambar", title="GT tanpa luka")
    axes[0].legend(handles=[Patch(color=TEAL, label="Prediksi kosong"), Patch(color=CORAL, label="Ada FP")],
                   frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -.17))
    bars = axes[1].bar(np.arange(len(empty)), empty.fp, color=[CORAL if n else "#CCD5DF" for n in empty.fp])
    axes[1].bar_label(bars, labels=[str(n) if n else "" for n in empty.fp], padding=3, fontsize=9)
    axes[1].set(xlabel="17 gambar test tanpa luka · diurutkan berdasarkan FP", ylabel="Piksel false positive",
                title="Besarnya false positive bervariasi", ylim=(0, empty.fp.max() * 1.2), xticks=[])
    axes[1].grid(axis="y", alpha=.8); axes[1].set_axisbelow(True)
    fig.suptitle("05  /  Kasus GT kosong perlu dibaca terpisah", fontsize=16, fontweight="bold")
    return save_figure(context, fig, "05_empty_masks")


def select_examples(context):
    """Freeze transparent selection; prefer distinct RGB photos across galleries."""
    test = context["frames"]["test"].copy()
    positive = test[test.gt_pixels.gt(0)].sort_values("name").drop_duplicates("pixel_hash")
    selected, used = {}, set()

    def take(label, candidates, count=3):
        available = candidates[~candidates.pixel_hash.isin(used)].drop_duplicates("pixel_hash")
        chosen = available.head(count).copy()
        assert len(chosen) == count
        chosen["selection"] = label
        used.update(chosen.pixel_hash)
        selected[label] = chosen

    # Draw first so the random gallery is not conditioned on extreme-case selection.
    take("random", positive.sample(frac=1, random_state=42))
    take("best", test[test.gt_pixels.gt(0)].sort_values(["dice", "name"], ascending=[False, True]))
    median = test.loc[test.gt_pixels.gt(0), "dice"].median()
    near_median = test[test.gt_pixels.gt(0)].assign(distance=lambda f: (f.dice - median).abs())
    take("median", near_median.sort_values(["distance", "name"]))
    take("worst", test[test.gt_pixels.gt(0)].sort_values(["dice", "gt_pixels", "name"], ascending=[True, False, True]))
    empty = test[test.gt_pixels.eq(0)]
    take("empty_correct", empty[empty.fp.eq(0)].sort_values("name"), 1)
    take("empty_fp", empty[empty.fp.gt(0)].sort_values(["fp", "name"], ascending=[False, True]), 2)
    selected["empty"] = pd.concat([selected.pop("empty_correct"), selected.pop("empty_fp")])
    order = ["best", "median", "worst", "random", "empty"]
    frame = pd.concat([selected[label] for label in order], ignore_index=True)
    frame.to_csv(context["tables"] / "selected_examples.csv", index=False)
    context["selected_examples"] = {label: selected[label] for label in order}
    return frame


def prediction_gallery(context, group):
    choices = context["selected_examples"][group]
    titles = {"best": "Dice tertinggi", "median": "Dekat median Dice", "worst": "Dice terendah",
              "random": "Sampel acak · seed 42", "empty": "GT kosong · benar kosong dan false positive"}
    panels, heights = [], []
    for row in choices.to_dict("records"):
        with Image.open(ROOT / row["image_path"]) as source:
            image = np.asarray(source.convert("RGB"))
        truth, pred = binary_mask(ROOT / row["mask_path"]), binary_mask(ROOT / row["prediction_path"])
        # Remove only empty black margins from the view. Include all GT/predicted
        # positives, even on padding, and keep the exact same view for all columns.
        active = np.any(image != 0, axis=-1) | truth | pred
        ys, xs = np.where(active)
        if len(xs):
            bounds = (max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4),
                      min(image.shape[1], int(xs.max()) + 5), min(image.shape[0], int(ys.max()) + 5))
        else:
            bounds = (0, 0, image.shape[1], image.shape[0])
        left, top, right, bottom = bounds
        heights.append(float(np.clip((bottom - top) / (right - left), .55, 1.3)))
        panels.append((row, image, truth, pred, bounds))
    figure_height = 2.7 * sum(heights) + 1.8
    fig, axes = plt.subplots(len(choices), 4, figsize=(12, figure_height), gridspec_kw={"height_ratios": heights})
    fig.subplots_adjust(left=.04, right=.98, top=1 - .95 / figure_height,
                        bottom=.65 / figure_height, hspace=.5, wspace=.07)
    colors = {"tp": "#2EC27E", "fp": "#F15B5D", "fn": "#5EA5FF"}
    for row_index, (row, image, truth, pred, bounds) in enumerate(panels):
        overlay = image.astype(float) / 255
        for name, pixels in [("tp", truth & pred), ("fp", ~truth & pred), ("fn", truth & ~pred)]:
            overlay[pixels] = .4 * overlay[pixels] + .6 * np.array(mpl.colors.to_rgb(colors[name]))
        axes[row_index, 0].imshow(image)
        axes[row_index, 1].imshow(truth, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row_index, 2].imshow(pred, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row_index, 3].imshow(overlay)
        left, top, right, bottom = bounds
        for col, title in enumerate(["Gambar · zoom", "Ground truth", "Prediksi", "Overlay error"]):
            axes[row_index, col].set_xlim(left - .5, right - .5)
            axes[row_index, col].set_ylim(bottom - .5, top - .5)
            axes[row_index, col].set_axis_off()
            if row_index == 0:
                axes[row_index, col].set_title(title, fontsize=11, pad=10)
        name = row["name"]
        axes[row_index, 0].text(.5, -.04, f"{name[:16]}\n{name[16:]}", ha="center", va="top",
                                transform=axes[row_index, 0].transAxes, fontsize=8, family="monospace")
        axes[row_index, 1].text(.5, -.04, f"Luas GT {row['area_pct']:.2f}%", ha="center", va="top",
                                transform=axes[row_index, 1].transAxes, fontsize=10)
        axes[row_index, 2].text(.5, -.04, f"Dice {row['dice'] * 100:.2f}%", ha="center", va="top",
                                transform=axes[row_index, 2].transAxes, fontsize=10)
        axes[row_index, 3].text(.5, -.04, f"FP {row['fp']:,} px · FN {row['fn']:,} px", ha="center", va="top",
                                transform=axes[row_index, 3].transAxes, fontsize=9)
    fig.suptitle(f"{titles[group]}\nTest · checkpoint 52 · threshold ≥ 0.5 · crop tampilan", fontsize=14, fontweight="bold", y=.98)
    fig.legend(handles=[Patch(color=colors["tp"], label="TP · luka terdeteksi"),
                        Patch(color=colors["fp"], label="FP · prediksi berlebih"),
                        Patch(color=colors["fn"], label="FN · luka terlewat")],
               loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(.5, .005))
    return save_figure(context, fig, f"06_predictions_{group}", clinical=True)


def export_tables(context):
    tables = {"final_metrics": metric_table(context), "paper_comparison": paper_comparison(context),
              "categories": category_table(context), "empty_masks": empty_table(context),
              "exact_pixel_overlap": overlap_table(context)}
    if "pfom" in context:
        tables.update({"pfom_summary": pfom_summary_table(context),
                       "pfom_empty_edges": pfom_empty_table(context),
                       "pfom_categories": pfom_category_table(context)})
    for name, frame in tables.items():
        frame.to_csv(context["tables"] / f"{name}.csv", index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": context["run_id"],
        "training_performed": False, "checkpoint_epoch": context["status"]["selected_epoch"],
        "threshold": context["config"]["evaluation"]["prediction_threshold"],
        "verified_prediction_masks": context["verified_masks"],
        "identity": context["export"]["identity"], "metrics": context["summaries"],
        "pfom": context.get("pfom"),
        "selection_seed": 42,
        "selection_rules": {
            "random": "3 sampled first from filename-sorted unique RGB positive-GT test photos; seed 42",
            "best": "3 highest positive-GT Dice; filename tie-break; exclude photos already selected",
            "median": "3 nearest to median positive-GT test Dice; filename tie-break; exclude selected photos",
            "worst": "3 lowest positive-GT Dice; larger GT then filename tie-break; exclude selected photos",
            "empty": "1 empty/empty sorted by filename, 2 largest-FP empty-GT; exclude selected photos",
        },
        "limitations": ["One seed/run", "831/278 source versus 810/200 reported in paper",
                        "Original train/validation split and paper-table evaluator unknown",
                        "Exact RGB cross-split duplicates retained; patient independence unverified",
                        "Test error analysis is descriptive; do not tune against the test set"],
        "figures": [str(p.relative_to(ROOT)) for p in sorted(context["figures"].glob("*.png"))],
    }
    (context["output"] / "analysis-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return tables
