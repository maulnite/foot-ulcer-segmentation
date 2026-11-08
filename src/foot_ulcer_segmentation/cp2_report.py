"""Post-hoc phase-P report. Reads completed runs; never selects or trains models."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from .common import ROOT, digest_file, write_json


def build_report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root).resolve()
    manifest = json.loads((root / "data/manifest.json").read_text())
    config = json.loads((root / "protocol.json").read_text())
    statuses, metrics, histories = {}, {}, {}
    for name in ("B0", "B1"):
        statuses[name] = json.loads((root / name / "status.json").read_text())
        if statuses[name]["status"] != "completed" or statuses[name]["diagnostic_smoke_only"]:
            raise ValueError("A real completed B0 and B1 are required; no placeholder report")
        if digest_file(root / name / "best.pth") != statuses[name]["best_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint checksum changed")
        metrics[name] = json.loads((root / name / "final_evaluation.json").read_text())
        histories[name] = json.loads((root / name / "history.json").read_text())
    controls = json.loads((root / "train_paired_controls.json").read_text())
    if not controls["initialization_exact"] or not controls["all_common_epoch_order_and_augmented_input_hashes_equal"]:
        raise ValueError("Controlled comparison failed")
    public = ROOT / "reports"
    figures = public / "figures/cp2-phase-p"
    tables = public / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    colors = {"B0": "#64748B", "B1": "#0F8B8D"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 13, "figure.facecolor": "white"})
    files = []

    def save(fig, name):
        for extension in ("png", "svg"):
            path = figures / f"{name}.{extension}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            files.append({"path": str(path.relative_to(ROOT)), "sha256": digest_file(path)})
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for name, ax in zip(("B0", "B1"), axes[0]):
        h = histories[name]
        epochs = [r["epoch"] for r in h]
        ax.plot(epochs, [r["train_loss"] for r in h], label="Train", color=colors[name])
        ax.plot(epochs, [r["validation_loss"] for r in h], label="Validation", color="#D97706")
        ax.set(title=f"{name}: objective sendiri (bukan skor segmentasi)", xlabel="Epoch", ylabel="Dice + focal loss")
        ax.legend()
        axes[1, 0].plot(epochs, [r["validation_image_dice"] for r in h], label=name, color=colors[name])
        chosen = statuses[name]["selected_epoch"]
        axes[1, 0].scatter(chosen, h[chosen - 1]["validation_image_dice"], color=colors[name], s=65, zorder=5)
        axes[1, 1].step(epochs, [r["learning_rate_used"] for r in h], where="post", label=name, color=colors[name])
    axes[1, 0].set(title="Pemilihan checkpoint: Image Dice validation", xlabel="Epoch", ylabel="Image Dice (0–1)", ylim=(0, 1))
    axes[1, 1].set(title="Scheduler memantau Image Dice validation", xlabel="Epoch", ylabel="Learning rate", yscale="log")
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    axes[1, 0].legend()
    axes[1, 1].legend()
    fig.suptitle(f"REAL — CP2 Chronic Wound | {len(manifest['splits']['train'])} train / {len(manifest['splits']['validation'])} validation | seed 42", fontsize=15)
    save(fig, "training_curves")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, split in zip(axes, ("validation", "test")):
        labels = ["Semua", "GT positif", "GT kosong", "Luka kecil ≤1%"]
        keys = ["all", "gt_positive", "gt_empty", "small"]
        x = np.arange(4)
        for shift, name in [(-.18, "B0"), (.18, "B1")]:
            values = []
            for key in keys:
                m = metrics[name][split]
                subgroup = m["subgroups"][key] if key == "small" else m[key]
                values.append(subgroup["image_based"]["dice"] if subgroup["images"] else np.nan)
            ax.bar(x + shift, values, width=.34, color=colors[name], label=name)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Mean Image Dice (0–1)",
               title=f"{split}: n={len(manifest['splits'][split])} RGB groups")
        ax.tick_params(axis="x", rotation=15)
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("REAL — CP2 B0 vs B1 | threshold ≥0,5 | checkpoint dipilih pada validation", fontsize=14)
    save(fig, "subgroup_comparison")

    paired_differences = {}
    for split in ("validation", "test"):
        rows = {}
        for name in ("B0", "B1"):
            source = root / name / f"evaluation-{split}.csv"
            with source.open() as stream:
                rows[name] = {r["name"]: r for r in csv.DictReader(stream)}
            shutil.copyfile(source, tables / f"cp2-phase-p-{name.lower()}-{split}.csv")
        if rows["B0"].keys() != rows["B1"].keys():
            raise ValueError("Baselines have different evaluated populations")
        difference = np.array([float(rows["B1"][k]["dice"]) - float(rows["B0"][k]["dice"])
                               for k in sorted(rows["B0"])])
        rng = np.random.default_rng(20261107)
        boot = [float(rng.choice(difference, len(difference), replace=True).mean()) for _ in range(2000)]
        paired_differences[split] = {"mean_image_dice_B1_minus_B0": float(difference.mean()),
                                    "descriptive_rgb_group_bootstrap_95_interval": np.percentile(boot, [2.5, 97.5]).tolist(),
                                    "B1_better_images": int((difference > 0).sum()),
                                    "B0_better_images": int((difference < 0).sum()),
                                    "tied_images": int((difference == 0).sum()),
                                    "unit": "one unique RGB group/image; NOT patient or training-seed uncertainty"}
    with (tables / "cp2-phase-p-membership.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "sample_id", "rgb_group", "gt_pixels", "canvas_pixels", "image_sha256", "mask_sha256"])
        writer.writeheader()
        for split, rows in manifest["splits"].items():
            writer.writerows({"split": split, **{k: r[k] for k in writer.fieldnames if k != "split"}} for r in rows)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "COMPLETED — PHASE P B0/B1",
               "run_root": str(root.relative_to(ROOT)), "manifest_fingerprint": manifest["fingerprint"],
               "manifest_file_sha256": digest_file(root / "data/manifest.json"),
               "data_audit": manifest["audit"], "protocol": config,
               "statuses": statuses, "metrics": metrics, "paired_controls": controls,
               "paired_differences": paired_differences, "figures": files}
    write_json(public / "cp2-phase-p-results.json", summary)
    lines = ["# CP2 — FASE P: audit, smoke test dan baseline B0/B1", "",
             "**Selesai pada data nyata.** Cakupan berhenti pada dua baseline. Tidak ada fase loss lanjutan, attention, boundary head, atau evaluasi eksternal yang dijalankan.", "",
             "## Protokol dan data", "",
             f"Manifest `{manifest['fingerprint']}`: **{len(manifest['splits']['train'])} train / {len(manifest['splits']['validation'])} validation / {len(manifest['splits']['test'])} source-test**. Seluruh 1.109 pasangan foto–mask Chronic Wound diperiksa pada grid 224×224. Ditemukan 30 kelompok RGB duplikat, 28 kelompok berkonflik anotasi, dan 17 kelompok lintas source train–test. Sebanyak 57 baris dalam kelompok berkonflik dikarantina; dua baris duplikat tanpa konflik diringkas menjadi wakil leksikografis. File mentah tetap utuh.", "",
             "Split seed 42 atas sorted RGB groups, holdout round(15%). Ini **duplicate-group split, bukan patient-level split**. Pemeriksaan near-duplicate memakai RGB-only content crop (menghindari dominasi padding hitam), pHash, MAE dan SSIM; tidak membuktikan tidak adanya crop atau foto pasien berulang. Source-test telah dianalisis dalam CP1 dan tidak dianggap uji konfirmatori yang belum pernah dilihat. Tidak ada dataset eksternal dipakai memilih konfigurasi.", "",
             "B0 dan B1 memakai FUSegNet–EfficientNet-B7 yang sama, 70.968.939 parameter terdaftar. Keluaran diekspos sebagai logits tanpa mengubah parameter; sigmoid dipakai satu kali untuk probabilitas. **B0** mempertahankan Dice(sigmoid(z)) + focal-with-logits(sigmoid(z)); **B1** memakai Dice(sigmoid(z)) + focal-with-logits(z). Primitive loss berasal dari source terverifikasi: alpha None, gamma 2, Dice epsilon 1, focal epsilon 1e-6, mean reduction, bobot 1+1, agregasi Dice seluruh batch.", "",
             "Satu berkas bobot awal bersama diverifikasi per tensor, termasuk buffers. Membership, urutan sample dan hash augmented input sama pada seluruh epoch yang sama-sama dijalankan. Adam LR 1e-4/WD 1e-5, batch 2, empat loader workers, augmentasi dan preprocessing source tidak diubah. PyTorch 1.11.0+cu113/torchvision 0.12.0+cu113; float32, tanpa AMP, flag TF32 lama tetap True. Tidak ada upgrade dependency.", "",
             "Checkpoint = Image Dice validation tertinggi; tie mempertahankan epoch lebih awal. ReduceLROnPlateau memantau Image Dice (max), factor 0,1/patience 10/min LR 1e-5; early stopping 30 epoch tanpa perbaikan, maksimal 200 epoch. Scheduler tidak memantau objective loss yang berbeda antarbaseline. Threshold **>=0,5**, ordinary confusion counts; per-image kedua mask kosong=1, tepat satu kosong=0; aggregate Dice null bila seluruh GT/prediksi kosong. Boundary IoU memakai inner erosion 3×3 dengan zero-padded exterior, radius round(1% diagonal), minimum satu piksel. PFOM menggunakan primitive CP1 dan kebijakan kosong yang sama.", "",
             "## Hasil terpilih", "",
             "| Baseline | Epoch dipilih / selesai | Validation Image Dice | Source-test Image Dice | Waktu training (menit) | Peak allocated / reserved (GiB) |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("B0", "B1"):
        s = statuses[name]
        lines.append(f"| {name} | {s['selected_epoch']} / {s['completed_epochs']} | {metrics[name]['validation']['all']['image_based']['dice']:.4f} | {metrics[name]['test']['all']['image_based']['dice']:.4f} | {s['seconds_this_invocation']/60:.2f} | {s['peak_cuda_allocated_bytes']/1024**3:.2f} / {s['peak_cuda_reserved_bytes']/1024**3:.2f} |")
    lines += ["", "![Training curves](figures/cp2-phase-p/training_curves.png)", "",
              "**Yang terlihat:** setiap objective mempunyai kurva train/validation sendiri, sedangkan Dice dan LR dibandingkan pada protokol yang sama. Titik menandai checkpoint yang dipilih.", "",
              "**Artinya:** koreksi focal dinilai menggunakan prediksi segmentasi, bukan hanya nilai loss yang memang berbeda definisi. Jadwal LR aktual dan penghentian dapat berbeda sebagai konsekuensi kurva validation.", "",
              "**Perlu diingat:** ini satu seed. Tidak ada pemilihan epoch dari source-test dan tidak ada klaim keunggulan umum dari satu perbandingan.", "",
              "| Split / subgroup | n | B0 Image Dice | B1 Image Dice | B0 / B1 data Dice | B0 / B1 precision | B0 / B1 recall | B0 / B1 Boundary IoU |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation", "test"):
        for key in ("all", "gt_positive", "gt_empty", "small"):
            a = metrics["B0"][split]["subgroups"][key] if key == "small" else metrics["B0"][split][key]
            b = metrics["B1"][split]["subgroups"][key] if key == "small" else metrics["B1"][split][key]
            if not a["images"]:
                continue
            fmt = lambda v: "undefined" if v is None else f"{v:.4f}"
            lines.append(f"| {split}/{key} | {a['images']} | {a['image_based']['dice']:.4f} | {b['image_based']['dice']:.4f} | {fmt(a['data_based']['dice'])} / {fmt(b['data_based']['dice'])} | {a['data_based']['precision']:.4f} / {b['data_based']['precision']:.4f} | {a['data_based']['recall']:.4f} / {b['data_based']['recall']:.4f} | {a['image_based']['boundary_iou']:.4f} / {b['image_based']['boundary_iou']:.4f} |")
    lines += ["", "![Subgroups](figures/cp2-phase-p/subgroup_comparison.png)", "",
              "**Yang terlihat:** semua, GT positif, GT kosong dan luka kecil memakai gambar yang sama untuk B0/B1. Luka kecil berarti luas GT >0 sampai 1% kanvas, bukan ukuran fisik atau kategori C1 penulis.", "",
              "**Artinya:** perubahan rata-rata perlu dibaca bersama kemampuan menemukan foreground dan false positive pada anotasi kosong. GT kosong bukan diagnosis pasien sehat.", "",
              "**Perlu diingat:** karantina mengubah populasi. Skor CP1 707/124/278 dan checkpoint policy sumber tidak menjadi pembanding langsung; konfigurasi/evaluator CP2 berbeda.", "",
              "## Ketahanan kontrol, biaya, dan keterbatasan", "",
              f"Seluruh {controls['common_epochs_with_identical_inputs']} epoch bersama memiliki hash urutan dan augmented input identik. Checkpoint replay setelah perturbasi parameter menghasilkan prediksi bit-identik pada kedua baseline; split disjoint untuk hash RGB yang diketahui. Normalisasi ImageNet tetap, tidak di-fit dari validation/test. Prediksi satu per gambar diekspor bersama probabilitas float32, mask threshold, GT, per-image confusion counts, subgroup, checksum, dan checkpoint identity.", "",
              "Training dilakukan **sekuensial dalam satu proses GPU**. GPU bersama masih memuat pekerjaan pengguna lain; waktu/latency bukan benchmark perangkat eksklusif. Peak allocated/reserved adalah allocator proses kita; `train_gpu_samples.jsonl` mencatat pemakaian keseluruhan GPU setiap lima detik (termasuk proses lain). Mulai concurrency berikutnya dengan satu job; lakukan profiling terkontrol sebelum menambah proses. Tidak ada alasan menaikkan batch/AMP hanya untuk mengisi VRAM.", "",
              "Ketidakpastian bootstrap pada JSON memakai unit RGB-group/image, bukan piksel; tidak mencakup pasien yang tidak diketahui atau variasi retraining antar-seed. Near-duplicate screening tidak menyelesaikan seluruh hubungan crop/pasien. Tidak ada bukti generalisasi eksternal, signifikansi klinis, atau novelty metode dari fase P ini.", "",
              f"Artefak lengkap: `{root.relative_to(ROOT)}/` (manifest, initial weights, smoke, B0/B1 best/last checkpoints, curves, status, persistent logs, GPU sampling). Bobot CP1 tidak dipakai sebagai inisialisasi train; hanya pretrained ImageNet source yang digunakan. Checkpoints/cache/raw data tetap lokal.", "",
              "[Hasil machine-readable](cp2-phase-p-results.json) · [Membership](tables/cp2-phase-p-membership.csv) · [Panduan](../docs/cp2-phase-p.md). Fase berikutnya menunggu review B0/B1, lalu konfirmasi beberapa seed sebelum memperluas matriks."]
    # Keep the separately recorded post-training interpretation on a later replay.
    supplement = root / "report_supplement.md"
    if supplement.exists():
        lines += ["", supplement.read_text().strip()]
    if (public / "cp2-phase-p-review.md").exists():
        lines += ["", "[Review akhir: kondisi awal, metrik tambahan, analisis error dan visual QA](cp2-phase-p-review.md)."]
    destination = public / "cp2-phase-p.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(destination)
    evidence = root / "report_generated.json"
    if evidence.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        evidence = root / "report_replays" / f"{stamp}.json"
    write_json(evidence, {"report": "reports/cp2-phase-p.md", "figures": files})
    return summary
