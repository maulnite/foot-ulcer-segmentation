#!/usr/bin/env python3
"""Verify completed ablation masks, evaluate PFOM, execute a comparison notebook."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from PIL import Image

from cp1_analysis import ROOT, binary_mask, configure_style, summarize, TEAL, CORAL, NAVY
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json
from foot_ulcer_segmentation.data import verify_manifest
from foot_ulcer_segmentation.evaluation import SegmentationEvaluator
from pfom_metrics import score_masks

LABELS = {"full": "P-scSE penuh", "additive_only": "Additive-only", "maxout_only": "Maxout-only"}
COLORS = {"full": NAVY, "additive_only": TEAL, "maxout_only": CORAL}
PROTOCOL = ROOT / "configs/ablation/cp1-attention-01.json"


def read(path):
    return json.loads(Path(path).read_text())


def prepare_comparison(batch, diagnostic=False):
    protocol, proof = read(PROTOCOL), read(ROOT / "outputs/ablations/cp1-attention-01/verification.json")
    baseline = read(ROOT / protocol["baseline_run"] / "config.json")
    categories = read(ROOT / "source/upstream/FUSegNet" / baseline["reference_revision"] / "categorized_oldDfu.json")
    membership = {name: int(k) for k, names in categories.items() for name in names}
    pfom = read(ROOT / "configs/pfom.json")
    if digest_json(pfom) != read(ROOT / "reports/tables/cp1-pfom-metadata.json")["protocol_fingerprint"]:
        raise ValueError("PFOM settings differ from the published baseline")
    declarations = [{"variant": "full", "run_dir": protocol["baseline_run"]}, *protocol["runs"]]
    summaries, frames, category_rows, provenance = [], [], [], []
    for declaration in declarations:
        variant = declaration["variant"]
        run = ROOT / (protocol["baseline_run"] if diagnostic else declaration["run_dir"])
        status, config = read(run / "status.json"), read(run / "config.json")
        if status["status"] != "completed" or status["smoke_only"]:
            raise ValueError(f"Full training has not completed: {run.name}")
        for key in ["data", "model", "training", "evaluation", "seed"]:
            if config[key] != baseline[key]:
                raise ValueError(f"Uncontrolled configuration change in {run.name}: {key}")
        if variant != "full" and not diagnostic:
            if config["ablation"]["variant"] != variant:
                raise ValueError("Unexpected ablation variant")
            if read(run / "model-ablation.json")["initial_state_sha256"] != proof["full_initial_state_sha256"]:
                raise ValueError("Initial model hash differs from the matched baseline reconstruction")
        manifest = verify_manifest(read(run / "split-manifest.json"))
        if manifest["fingerprint"] != protocol["split_fingerprint"]:
            raise ValueError("Split differs from the declared baseline split")
        environment = read(run / "environment.json")
        for path, expected in environment["pipeline_sha256"].items():
            if digest_file(ROOT / path) != expected:
                raise ValueError(f"Training runtime source changed: {path}")
        predictions = read(ROOT / "outputs/analysis" / run.name / "prediction-manifest.json")
        identity = predictions["identity"]
        if (digest_file(run / "best.pth") != identity["checkpoint_sha256"]
                or digest_json(config) != identity["config_fingerprint"]
                or identity["split_fingerprint"] != manifest["fingerprint"] or identity["threshold"] != .5):
            raise ValueError("Prediction/checkpoint identity differs")
        cache = {(r["split"], r["name"]): r for r in predictions["predictions"]}
        if len(cache) != 402:
            raise ValueError("Expected all 402 validation/test predictions")
        provenance.append({"variant": variant, "run_directory": str(run.relative_to(ROOT)),
                           "identity": identity, "environment_git_revision": environment["git_revision"],
                           "model": proof["variants"][variant]})
        for split, filename in [("validation", "validation"), ("test", "evaluation-test")]:
            if digest_file(run / f"{filename}.csv") != identity["evaluation_sha256"][f"{filename}.csv"]:
                raise ValueError("Per-image evaluation CSV changed")
            metrics = pd.read_csv(run / f"{filename}.csv").set_index("name")
            evaluator, records = SegmentationEvaluator(.5), []
            for sample in manifest["splits"][split]:
                name = sample["name"]
                record = cache[(split, name)]
                if digest_file(ROOT / record["path"]) != record["sha256"]:
                    raise ValueError("Cached mask changed")
                truth, pred = binary_mask(ROOT / sample["mask_path"]), binary_mask(ROOT / record["path"])
                measured = evaluator.add(name, truth, pred.astype(float))
                for key in ["tp", "tn", "fp", "fn", "dice", "iou", "precision", "recall"]:
                    if not np.isclose(measured[key], metrics.loc[name, key], rtol=0, atol=1e-12):
                        raise ValueError(f"Mask/CSV mismatch: {variant}, {split}, {name}, {key}")
                records.append({**sample, **measured, **score_masks(truth, pred, beta=pfom["beta"], **pfom["canny"]),
                                "prediction_path": record["path"], "variant": variant, "model": LABELS[variant],
                                "split": split, "category": membership[name] if split == "test" else np.nan,
                                "gt_pixels": int(truth.sum()), "pred_pixels": int(pred.sum())})
            frame = pd.DataFrame(records)
            expected = read(run / f"{filename}.json")["offline_hard_line"]
            actual = evaluator.summary()["offline_hard_line"]
            if actual["counts"] != expected["counts"]:
                raise ValueError("Aggregate counts differ from final evaluation")
            for family in ["data_based", "image_based"]:
                for key in ["dice", "iou", "precision", "recall"]:
                    if not np.isclose(actual[family][key], expected[family][key], rtol=0, atol=1e-12):
                        raise ValueError("Aggregate scores differ from final evaluation")
            empty, positive = frame[frame.gt_pixels.eq(0)], frame[frame.gt_pixels.gt(0)]
            summaries.append({"variant": variant, "model": LABELS[variant], "split": split, "n": len(frame),
                              **{f"{family}_{key}": value for family in ["data_based", "image_based"]
                                 for key, value in actual[family].items()},
                              "pfom_mean": frame.pfom.mean(), "pfom_median": frame.pfom.median(),
                              "pfom_positive_gt": positive.pfom.mean(),
                              "pfom_sentinel2_proxy": frame.pfom_paper_sentinel2_proxy.mean(),
                              "missed_wounds": int(positive.tp.eq(0).sum()),
                              "empty_gt_images": len(empty), "empty_gt_fp_images": int(empty.fp.gt(0).sum()),
                              "empty_gt_fp_pixels": int(empty.fp.sum()),
                              "epochs": status["completed_epochs"], "selected_epoch": status["selected_epoch"],
                              "training_minutes": status["seconds"] / 60,
                              "peak_cuda_gib": status["peak_cuda_allocated_bytes"] / 1024 ** 3,
                              "registered_parameters": proof["variants"][variant]["registered_parameters"],
                              "forward_called_parameters": proof["variants"][variant]["forward_called_parameters"]})
            if split == "test":
                for category, group in frame.groupby("category"):
                    aggregate = summarize(group)
                    category_rows.append({"variant": variant, "model": LABELS[variant], "category": int(category),
                                          "n": len(group), "image_dice": group.dice.mean(),
                                          "data_dice": aggregate["data_based"]["dice"], "pfom_mean": group.pfom.mean(),
                                          "pfom_sentinel2_proxy": group.pfom_paper_sentinel2_proxy.mean()})
            frames.append(frame)
    batch.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(batch / "comparison.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(batch / "per-image.csv", index=False, float_format="%.15g")
    pd.DataFrame(category_rows).to_csv(batch / "categories.csv", index=False)
    selected = pd.read_csv(ROOT / "outputs/analysis/cp1-baseline-01/tables/selected_examples.csv")
    names = [name for selection in ["worst", "median", "empty_fp"]
             for name in selected.loc[selected.selection.eq(selection), "name"].head(2)]
    if len(names) != 6:
        raise ValueError("Expected six predeclared baseline-selected examples")
    write_json(batch / "comparison.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "diagnostic_fixture": diagnostic, "protocol": protocol, "protocol_fingerprint": digest_json(protocol),
               "model_verification": proof, "pfom_protocol": pfom, "runs": provenance, "example_names": names,
               "analysis_source_sha256": digest_file(Path(__file__)),
               "units": "Region metrics fractions; PFOM 0-1; sentinel-2 proxy separate",
               "training_performed_by_analysis": False, "all_1206_mask_pairs_verified": True})


def load_comparison(batch):
    batch = Path(batch)
    return {"batch": batch, "metadata": read(batch / "comparison.json"),
            "summary": pd.read_csv(batch / "comparison.csv"), "per_image": pd.read_csv(batch / "per-image.csv"),
            "categories": pd.read_csv(batch / "categories.csv")}


def display_table(context, split):
    frame = context["summary"].query("split == @split").copy()
    return pd.DataFrame({"Model": frame.model, "n": frame.n, "Epoch selesai": frame.epochs,
                         "Checkpoint": frame.selected_epoch, "Image Dice (%)": frame.image_based_dice * 100,
                         "Data Dice (%)": frame.data_based_dice * 100, "PFOM (1/0)": frame.pfom_mean})


def save_plot(context, fig, name, clinical=False):
    folder = context["batch"] / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=200 if clinical else 300, bbox_inches="tight")
    if not clinical:
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
    return fig


def plot_curves(context):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for run in context["metadata"]["runs"]:
        variant = run["variant"]
        history = read(ROOT / run["run_directory"] / "history.json")
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["validation"]["dice_loss + focal_loss"] for r in history],
                     color=COLORS[variant], label=LABELS[variant])
        axes[1].plot(epochs, [r["validation"]["fscore"] * 100 for r in history], color=COLORS[variant])
        selected = context["summary"].query("variant == @variant and split == 'validation'").iloc[0].selected_epoch
        for ax in axes:
            ax.axvline(selected, color=COLORS[variant], linestyle=":", alpha=.6)
    axes[0].set(title="Validation loss", ylabel="Dice loss + focal loss")
    axes[1].set(title="Validation F-score dalam log", ylabel="Batch F-score (%)")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=.7)
    axes[0].legend(frameon=False)
    fig.suptitle("Ablasi attention · checkpoint dipilih oleh aturan loss/IoU penulis", fontweight="bold")
    return save_plot(context, fig, "01_learning_curves")


def plot_categories(context):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for variant in LABELS:
        rows = context["categories"].query("variant == @variant and category > 0").sort_values("category")
        axes[0].plot(rows.category, rows.image_dice * 100, "o-", color=COLORS[variant], label=LABELS[variant])
        axes[1].plot(rows.category, rows.pfom_mean, "o-", color=COLORS[variant])
    axes[0].set(title="Dice per kelompok GT positif", ylabel="Mean image Dice (%)", ylim=(0, 103))
    axes[1].set(title="PFOM per kelompok GT positif", ylabel="Mean PFOM · kosong 1/0", ylim=(0, 1.03))
    for ax in axes:
        ax.set(xticks=range(1, 10), xticklabels=[f"C{i}" for i in range(1, 10)])
        ax.grid(alpha=.7)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Test · semua varian yang ditetapkan sebelum eksperimen", fontweight="bold")
    return save_plot(context, fig, "02_categories")


def plot_errors(context):
    rows = context["summary"].query("split == 'test'")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key, title in [(axes[0], "missed_wounds", "GT positif dengan TP=0"),
                           (axes[1], "empty_gt_fp_images", "GT kosong dengan false positive")]:
        bars = ax.bar(rows.model, rows[key], color=[COLORS[v] for v in rows.variant])
        ax.bar_label(bars, padding=3)
        ax.set(title=title, ylabel="Jumlah gambar", ylim=(0, max(1, rows[key].max()) * 1.3))
        ax.grid(axis="y", alpha=.7); ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9)
    return save_plot(context, fig, "03_error_counts")


def plot_examples(context, offset=0):
    names = context["metadata"]["example_names"][offset:offset + 3]
    frame = context["per_image"].query("split == 'test'")
    fig, axes = plt.subplots(len(names), 5, figsize=(14, 3.4 * len(names)))
    fig.subplots_adjust(left=.03, right=.99, top=.90, bottom=.07, hspace=.4, wspace=.08)
    for i, name in enumerate(names):
        group = frame[frame.name.eq(name)].set_index("variant")
        row = group.loc["full"]
        with Image.open(ROOT / row.image_path) as image:
            image = np.asarray(image.convert("RGB"))
        truth = binary_mask(ROOT / row.mask_path)
        predictions = {variant: binary_mask(ROOT / group.loc[variant, "prediction_path"]) for variant in LABELS}
        active = np.any(image != 0, axis=-1) | truth
        for pred in predictions.values():
            active |= pred
        ys, xs = np.where(active)
        left, top = max(0, xs.min() - 4), max(0, ys.min() - 4)
        right, bottom = min(224, xs.max() + 5), min(224, ys.max() + 5)
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        for col, variant in enumerate(LABELS, 2):
            pred, overlay = predictions[variant], image.astype(float) / 255
            for pixels, color in [(truth & pred, "#2EC27E"), (~truth & pred, "#F15B5D"), (truth & ~pred, "#5EA5FF")]:
                overlay[pixels] = .4 * overlay[pixels] + .6 * np.array(mpl.colors.to_rgb(color))
            axes[i, col].imshow(overlay)
            value = group.loc[variant]
            axes[i, col].text(.5, -.04, f"Dice {value.dice * 100:.2f}%\nPFOM {value.pfom:.4f}",
                              ha="center", va="top", transform=axes[i, col].transAxes, fontsize=9)
        axes[i, 0].text(.5, -.04, f"{name[:16]}\n{name[16:]}", ha="center", va="top",
                        transform=axes[i, 0].transAxes, fontsize=8)
        for col, title in enumerate(["Gambar · zoom", "Ground truth", *LABELS.values()]):
            axes[i, col].set(xlim=(left - .5, right - .5), ylim=(bottom - .5, top - .5))
            axes[i, col].set_axis_off()
            if i == 0:
                axes[i, col].set_title(title, fontsize=11)
    fig.suptitle("Contoh dipilih dari baseline sebelum training ablasi\nCrop hanya tampilan; metrik pada kanvas penuh", fontweight="bold")
    fig.legend(handles=[Patch(color=c, label=t) for c, t in [("#2EC27E", "TP"), ("#F15B5D", "FP"), ("#5EA5FF", "FN")]],
               loc="lower center", ncol=3, frameon=False)
    return save_plot(context, fig, f"04_examples_{offset}", clinical=True)


def make_notebook(batch):
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))
    context = load_comparison(batch)
    diagnostic = context["metadata"]["diagnostic_fixture"]
    md("# CP1 · Ablasi cabang attention FUSegNet\n\n" +
       ("**DIAGNOSTIC FIXTURE: memakai ulang prediksi baseline untuk menguji analisis, bukan hasil training ablasi.**\n\n" if diagnostic else "") +
       "Tiga model: baseline P-scSE penuh, additive-only, maxout-only. Stage 0–3 diubah; "
       "stage terakhir tetap shorted. Notebook menyimpan seluruh output, termasuk contoh prediksi.")
    code(f"""from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs/baseline.json').exists())
sys.path.insert(0, str(ROOT / 'scripts'))
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from build_ablation_analysis import load_comparison, display_table, plot_curves, plot_categories, plot_errors, plot_examples
from cp1_analysis import configure_style
%matplotlib inline
configure_style()
c = load_comparison(ROOT / {str(batch.relative_to(ROOT))!r})
def table(frame):
    display(HTML(frame.to_html(index=False, border=0, na_rep='—', float_format=lambda x: f'{{x:.4f}}')))
print('1.206 pasangan mask sudah diverifikasi; tidak ada training atau inference di notebook ini.')""")
    md("## 1 · Hipotesis dan kontrol\n\nKombinasi additive/maxout diuji terhadap masing-masing cabang. "
       "Split, seed 42, constructor dan bobot awal, pretrained encoder, augmentasi, loss, optimizer, "
       "scheduler, aturan checkpoint, early stopping, threshold 0.5, serta evaluator tetap sama. "
       "Cabang dorman tetap terdaftar untuk menyamakan inisialisasi; parameter yang dipakai forward berkurang. "
       "Baseline test sudah pernah dianalisis; varian dan endpoint dikunci sebelum training baru.")
    code("table(c['summary'].query(\"split == 'validation'\")[[\n    'model','registered_parameters','forward_called_parameters','training_minutes','peak_cuda_gib'\n]])")
    md("## 2 · Validation dan pemilihan checkpoint\n\nCheckpoint mengikuti kebijakan loss/IoU penulis, "
       "bukan pemilihan berdasarkan hasil test atau maksimum metrik lain. F-score pada kurva memakai rata-rata batch.")
    code("table(display_table(c, 'validation'))\nplot_curves(c)\nplt.show()")
    md("## 3 · Test semua varian\n\nSeluruh varian dilaporkan dengan checkpoint terpilih masing-masing. "
       "PFOM utama memakai kebijakan edge kosong 1/0; proxy sentinel 2 terpisah.")
    code("table(display_table(c, 'test'))\ntable(c['summary'].query(\"split == 'test'\")[['model','pfom_positive_gt','pfom_sentinel2_proxy','missed_wounds','empty_gt_fp_images','empty_gt_fp_pixels']])")
    md("## 4 · Analisis menurut area GT dan kasus error\n\nC0 berisi GT kosong dan tidak dimasukkan "
       "pada grafik kelompok GT positif. Luas memakai kanvas penuh termasuk padding, bukan ukuran fisik luka.")
    code("plot_categories(c)\nplt.show()\nplot_errors(c)\nplt.show()\ntable(c['categories'])")
    md("## 5 · Contoh tetap dari baseline\n\nEnam contoh ditetapkan dari dua worst-positive, dua median-positive, "
       "dan dua empty-FP baseline sebelum training ablasi. Galeri tidak dipilih berdasarkan hasil varian baru. "
       "Crop sama pada seluruh kolom; hijau TP, merah FP, biru FN.")
    code("plot_examples(c, 0)\nplt.show()\nplot_examples(c, 3)\nplt.show()")
    md("## 6 · Batas interpretasi\n\nIni satu seed; perbedaan kecil belum mengukur variasi antar-training. "
       "Menghapus cabang juga mengurangi parameter aktif dan komputasi. Jumlah/split asli paper belum identik, "
       "duplikasi RGB antar-split tetap ada, dan independensi pasien belum terbukti. Setting PFOM lokal "
       "terdokumentasi; kesetaraan persis dengan Figure 6 paper belum terverifikasi. Perbandingan ini "
       "merupakan ablasi cabang dengan struktur decoder tetap, bukan penggunaan opsi scSE bawaan yang mengubah konvolusi.")
    md("## Referensi\n\nContoh gambar dan GT berasal dari [Chronic Wound / UWM](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/wound_dataset). "
       "Benchmark: [Dhar et al., FUSegNet](https://arxiv.org/abs/2305.02961v2); "
       "[kode referensi](https://github.com/mrinal054/FUSegNet). Konfigurasi eksperimen: "
       "[protokol ablasi](../docs/attention-ablations.md).")
    code("print('Protocol fingerprint:', c['metadata']['protocol_fingerprint'])\nprint('Initial-state SHA256:', c['metadata']['model_verification']['full_initial_state_sha256'])\nprint('Diagnostic fixture:', c['metadata']['diagnostic_fixture'])")
    for i, cell in enumerate(cells): cell['id'] = f'ablation-{i:02d}'
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "FUSeg · Analysis", "name": "fuseg-analysis", "language": "python"},
        "ablation": {"batch": "cp1-attention-01", "diagnostic_fixture": diagnostic}})
    prefix = batch / "kernel"
    os.environ["JUPYTER_PATH"] = str(prefix / "share/jupyter") + (os.pathsep + os.environ["JUPYTER_PATH"] if os.environ.get("JUPYTER_PATH") else "")
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--prefix", str(prefix),
                    "--name", "fuseg-analysis", "--display-name", "FUSeg · Analysis"], check=True)
    NotebookClient(notebook, timeout=300, kernel_name="fuseg-analysis", allow_errors=False,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    codes = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if (not all(cell.execution_count is not None for cell in codes)
            or any(o.output_type == "error" for cell in codes for o in cell.outputs)):
        raise RuntimeError("Ablation notebook did not fully execute")
    notebook.metadata['ablation']['all_cells_executed'] = True
    nbf.validate(notebook)
    destination = batch / "published/notebooks/02_cp1_attention_ablation.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)


def make_report(batch):
    context = load_comparison(batch)
    destination = batch / "published/reports/cp1-attention-ablation-01.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# CP1 · Ablasi cabang attention FUSegNet", "",
             "Semua run selesai. Stage 0–3 memakai kedua cabang, additive-only, atau maxout-only; stage terakhir tetap shorted.",
             "Split 707/124/278, seed 42, konfigurasi training/evaluator dan inisialisasi constructor dikontrol.", "",
             "[Notebook lengkap yang sudah dieksekusi](../notebooks/02_cp1_attention_ablation.ipynb) · [Protokol](../docs/attention-ablations.md)", ""]
    for split in ["validation", "test"]:
        lines += [f"## {split.title()}", "", "| Model | Epoch | Checkpoint | Image Dice (%) | Data Dice (%) | PFOM 1/0 |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: |"]
        for row in context["summary"].query("split == @split").itertuples():
            lines.append(f"| {row.model} | {row.epochs} | {row.selected_epoch} | {row.image_based_dice*100:.2f} | {row.data_based_dice*100:.2f} | {row.pfom_mean:.4f} |")
        lines.append("")
    lines += ["## Perubahan terhadap baseline pada test", ""]
    test = context["summary"].query("split == 'test'").set_index("variant")
    baseline = test.loc['full']
    for variant in ["additive_only", "maxout_only"]:
        row = test.loc[variant]
        lines.append(f"- {LABELS[variant]}: delta image Dice {(row.image_based_dice-baseline.image_based_dice)*100:+.2f} pp; delta PFOM {row.pfom_mean-baseline.pfom_mean:+.4f}; GT positif TP=0 {int(row.missed_wounds)}; GT kosong dengan FP {int(row.empty_gt_fp_images)}/{int(row.empty_gt_images)}.")
    lines += ["", "## Verifikasi dan batas interpretasi", "",
              "Semua 1.206 pasangan mask dari tiga model diverifikasi terhadap CSV dan JSON evaluasi akhirnya. PFOM memakai protokol baseline yang sama. Tidak ada pemilihan threshold atau varian berdasarkan test.", "",
              "Rekonstruksi baseline dan kedua varian memiliki hash bobot awal yang sama. Model dibangun lengkap sebelum jalur forward cabang diubah. Cabang dorman tetap terdaftar; parameter forward aktif baseline 64,898,248 dan varian 64,781,798. Posisi attention, satu conv aktif per stage, dan shorted stage terakhir tetap sama.", "",
              "Satu seed berarti variasi antar-training belum diukur. Ablasi mengubah komponen sekaligus kapasitas aktif; ini tidak mengisolasi operator agregasi pada kapasitas parameter identik. Test baseline sudah dianalisis sebelum eksperimen; seluruh hipotesis dan kedua varian ditetapkan sebelum run baru. Ketidakpastian split asli paper, duplikasi sumber, dan konvensi PFOM tetap berlaku.", "",
              "Contoh notebook memakai enam nama yang ditetapkan dari galeri baseline, tanpa pemilihan ulang berdasarkan hasil varian baru.", "",
              "Bukti numerik: [ringkasan](tables/cp1-attention-summary.csv), [kategori](tables/cp1-attention-categories.csv), [per gambar](tables/cp1-attention-per-image.csv), [metadata](tables/cp1-attention-metadata.json).", ""]
    if context["metadata"]["diagnostic_fixture"]:
        lines.insert(2, "**DIAGNOSTIC FIXTURE: baseline dipakai berulang; bukan hasil training ablasi.**")
    destination.write_text("\n".join(lines))
    tables = batch / "published/reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    for source, target in [("comparison.csv", "summary.csv"), ("categories.csv", "categories.csv"),
                           ("per-image.csv", "per-image.csv"), ("comparison.json", "metadata.json")]:
        shutil.copyfile(batch / source, tables / f"cp1-attention-{target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", default="outputs/ablations/cp1-attention-01")
    parser.add_argument("--smoke-analysis", action="store_true", help="Diagnostic fixture only; never publish copied baseline results")
    args = parser.parse_args()
    batch = ROOT / args.batch_dir
    if args.smoke_analysis and batch.name == "cp1-attention-01":
        raise ValueError("Use a separate diagnostic output directory")
    prepare_comparison(batch, diagnostic=args.smoke_analysis)
    make_notebook(batch)
    make_report(batch)
    if not args.smoke_analysis:
        files = [{"source": str(path.relative_to(batch)), "destination": str(path.relative_to(batch / "published"))}
                 for path in sorted((batch / "published").rglob("*")) if path.is_file()]
        write_json(batch / "publication-manifest.json", {"files": files})
    print(f"Analysis completed; diagnostic={args.smoke_analysis}; artifacts={batch.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
