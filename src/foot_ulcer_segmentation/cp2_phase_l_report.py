"""CPU-only presentation of completed phase-L runs; never chooses or trains models."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from .common import ROOT, digest_file, write_json


NAMES = ("L0", "L1", "L2", "L3", "L4")
COLORS = {"L0": "#64748B", "L1": "#2563EB", "L2": "#7C3AED",
          "L3": "#D97706", "L4": "#0F8B8D"}


def read(path):
    return json.loads(Path(path).read_text())


def subgroup(report, key):
    return report["subgroups"][key] if key in ("small", "medium", "large") else report[key]


def format_metric(value):
    return "undefined" if value is None else f"{value:.4f}"


def dice_contributions(candidate, baseline):
    """Exact additive decomposition of mean-image Dice on the same cohort."""
    keys = ("all", "gt_positive", "gt_empty")
    if any(candidate[k]["images"] != baseline[k]["images"] for k in keys):
        raise ValueError("Dice decomposition requires the same evaluation population")
    count = baseline["all"]["images"]
    if not count or sum(baseline[k]["images"] for k in keys[1:]) != count:
        raise ValueError("Positive/empty groups must partition a nonempty cohort")
    parts = {k: baseline[k]["images"] / count * (
        candidate[k]["image_based"]["dice"] - baseline[k]["image_based"]["dice"])
        for k in keys[1:]}
    total = candidate["all"]["image_based"]["dice"] - baseline["all"]["image_based"]["dice"]
    np.testing.assert_allclose(sum(parts.values()), total, rtol=0, atol=1e-12)
    return {**parts, "total": total}


def _paired_effects(root, control, tables):
    effects = {}
    for split in ("validation", "test"):
        paired = {}
        for name in NAMES:
            source = (control / "B1" if name == "L0" else root / name) / f"evaluation-{split}.csv"
            with source.open(newline="") as stream:
                paired[name] = {r["name"]: r for r in csv.DictReader(stream)}
            shutil.copyfile(source, tables / f"cp2-phase-l-{name.lower()}-{split}.csv")
        identities = sorted(paired["L0"])
        if any(set(rows) != set(identities) for rows in paired.values()):
            raise ValueError("Loss candidates evaluated different images")
        effects[split] = {}
        for name in NAMES[1:]:
            delta = np.array([float(paired[name][k]["dice"]) - float(paired["L0"][k]["dice"])
                              for k in identities])
            rng = np.random.default_rng(20261108)
            bootstrap = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
            effects[split][name] = {
                "mean_image_dice_minus_L0": float(delta.mean()),
                "descriptive_rgb_group_bootstrap_95_interval": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                "better_images": int((delta > 0).sum()), "worse_images": int((delta < 0).sum()),
                "tied_images": int((delta == 0).sum()), "images": len(delta),
                "unit": "paired unique RGB group/image; NOT patient or training-seed uncertainty"}
    return effects


def build_report(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(directory).resolve()
    if read(root / "train_workflow.json")["status"] != "completed":
        raise ValueError("All four real-data training/evaluation stages must complete")
    config = read(root / "protocol.json")
    control = ROOT / config["phase_l"]["control_root"]
    manifest = read(root / "data/manifest.json")
    metrics, statuses, histories = {}, {}, {}
    for name in NAMES:
        run = control / "B1" if name == "L0" else root / name
        status = read(run / "status.json")
        if status["status"] != "completed" or status["diagnostic_smoke_only"]:
            raise ValueError("No smoke or placeholder result may enter the report")
        if digest_file(run / "best.pth") != status["best_checkpoint_sha256"]:
            raise ValueError("Completed selected checkpoint changed")
        statuses[name], histories[name] = status, read(run / "history.json")
        metrics[name] = {split: read(run / f"evaluation-{split}.json") for split in ("validation", "test")}
    selection = read(root / "selection.json")
    preflight = read(root / "pretraining_checks.json")
    test_count = preflight["tests"]["tests"] if isinstance(preflight["tests"], dict) else preflight["tests"]
    oracle_count = read(root / "boundary_loss_oracle.json")["fixtures"]
    if not selection["validation_only"]:
        raise ValueError("Selection was not validation-only")
    public = ROOT / "reports"
    figures, tables = public / "figures/cp2-phase-l", public / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    effects = _paired_effects(root, control, tables)
    assets = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})

    def save(figure, stem):
        for extension in ("png", "svg"):
            path = figures / f"{stem}.{extension}"
            figure.savefig(path, dpi=160, bbox_inches="tight")
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            assets.append({"path": str(path.relative_to(ROOT)), "sha256": digest_file(path)})
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, name in zip(axes.flat, NAMES[1:]):
        history = histories[name]
        epochs = [r["epoch"] for r in history]
        ax.plot(epochs, [r["train_loss"] for r in history], color=COLORS[name], label="Train")
        ax.plot(epochs, [r["validation_loss"] for r in history], color="#475569", linestyle="--", label="Validation")
        ax.axvline(statuses[name]["selected_epoch"], color=COLORS[name], alpha=.5, linestyle=":", label="Selected checkpoint")
        ax.set(title=f"{name}: objective sendiri", xlabel="Epoch", ylabel="Nilai objective (skala masing-masing)")
        ax.grid(alpha=.15)
        ax.legend(fontsize=9)
    figure.suptitle("REAL — CP2 loss screening | 671 train / 118 validation | seed 42", fontsize=15)
    save(figure, "training_losses")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name in NAMES:
        h = histories[name]
        epochs = [r["epoch"] for r in h]
        style = "--" if name == "L0" else "-"
        axes[0].plot(epochs, [r["validation_image_dice"] for r in h], color=COLORS[name], label=name, linestyle=style)
        chosen = statuses[name]["selected_epoch"]
        axes[0].scatter(chosen, h[chosen - 1]["validation_image_dice"], color=COLORS[name], s=36, zorder=4)
        axes[1].step(epochs, [r["learning_rate_used"] for r in h], where="post", color=COLORS[name], label=name, linestyle=style)
    axes[0].set(title="Checkpoint dari Image Dice validation", xlabel="Epoch", ylabel="Image Dice (0–1)", ylim=(0, 1))
    axes[1].set(title="Scheduler dengan aturan yang sama", xlabel="Epoch", ylabel="Learning rate", yscale="log")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=.15)
    figure.suptitle("REAL — Chronic Wound | L0=CACHED B1; L1–L4=FRESH training", fontsize=14)
    save(figure, "checkpoint_and_lr")

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row, split in enumerate(("validation", "test")):
        for column, (group, metric, title) in enumerate([
                ("all", "dice", "Semua gambar: Image Dice"),
                ("small", "dice", "Luka kecil ≤1%: Image Dice"),
                ("all", "boundary_iou", "Semua gambar: Boundary IoU")]):
            ax = axes[row, column]
            values = [subgroup(metrics[n][split], group)["image_based"][metric] for n in NAMES]
            ax.bar(NAMES, values, color=[COLORS[n] for n in NAMES])
            for index, value in enumerate(values):
                ax.text(index, value + .018, f"{value:.3f}", ha="center", fontsize=10)
            ax.set(ylim=(0, 1.04), ylabel="Mean per gambar (0–1)", xlabel="Objective",
                   title=f"{split}, n={subgroup(metrics['L0'][split], group)['images']}\n{title}")
            ax.grid(axis="y", alpha=.15)
    figure.suptitle("REAL — CP2 selected checkpoints | threshold ≥0,5 | native 224×224", fontsize=15)
    save(figure, "segmentation_and_boundary")

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for row, split in enumerate(("validation", "test")):
        for column, (group, key, title) in enumerate([
                ("gt_positive", "total_foreground_misses", "GT positif dengan TP=0"),
                ("gt_empty", "images_with_prediction", "GT kosong dengan foreground terprediksi")]):
            ax = axes[row, column]
            values = [metrics[n][split][group][key] for n in NAMES]
            ax.bar(NAMES, values, color=[COLORS[n] for n in NAMES])
            for index, value in enumerate(values):
                ax.text(index, value + .12, str(value), ha="center")
            ax.set(ylim=(0, max(2, max(values) + 1)), xlabel="Objective", ylabel="Jumlah gambar",
                   title=f"{split}: {title}\nn={metrics['L0'][split][group]['images']}")
            ax.yaxis.get_major_locator().set_params(integer=True)
            ax.grid(axis="y", alpha=.15)
    figure.suptitle("REAL — Kegagalan foreground dan FP | GT kosong bukan diagnosis sehat", fontsize=14)
    save(figure, "foreground_and_empty_errors")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, split in zip(axes, ("validation", "test")):
        for position, name in enumerate(NAMES[1:]):
            result = effects[split][name]
            mean = 100 * result["mean_image_dice_minus_L0"]
            low, high = [100 * v for v in result["descriptive_rgb_group_bootstrap_95_interval"]]
            ax.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="o", color=COLORS[name], capsize=4)
        ax.axvline(0, color="#64748B", linestyle="--")
        ax.set(yticks=range(4), yticklabels=NAMES[1:], xlabel="Beda mean Image Dice terhadap L0 (poin persentase)",
               ylabel="Objective", title=f"{split}: n={metrics['L0'][split]['all']['images']} RGB groups")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    figure.suptitle("REAL — Paired RGB-group bootstrap deskriptif 95% | bukan pasien / variasi seed", fontsize=14)
    save(figure, "paired_dice_effects")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, name in zip(axes, ("L3", "L4")):
        h = histories[name]
        for stage, style in (("train", "-"), ("validation", "--")):
            for key, color, label in [("regional", "#2563EB", "Regional"),
                                      ("boundary_weighted", "#D97706", "0,01 × boundary"),
                                      ("total", "#0F8B8D", "Total")]:
                ax.plot([r["epoch"] for r in h], [r[f"{stage}_components"][key] for r in h],
                        color=color, linestyle=style, label=f"{stage}: {label}")
        ax.axhline(0, linewidth=.8, color="#64748B")
        ax.set(title=f"{name}: skala komponen aktual", xlabel="Epoch", ylabel="Kontribusi ke total objective")
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    figure.suptitle("REAL — Boundary bertanda dapat negatif | λ tetap, tanpa normalisasi / warm-up", fontsize=14)
    save(figure, "boundary_component_scales")

    for name in NAMES:
        path = tables / f"cp2-phase-l-{name.lower()}-history.csv"
        keys = ["epoch", "train_loss", "validation_loss", "validation_image_dice", "learning_rate_used",
                "learning_rate_next", "selected", "epoch_compute_seconds", "step_seconds_mean"]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: row[key] for key in keys} for row in histories[name])
    with (tables / "cp2-phase-l-membership.csv").open("w", newline="") as stream:
        keys = ["split", "sample_id", "rgb_group", "gt_pixels", "canvas_pixels", "image_sha256", "mask_sha256"]
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for split, rows in manifest["splits"].items():
            writer.writerows({"split": split, **{k: r[k] for k in keys[1:]}} for r in rows)
    summary = {"status": "COMPLETED — PHASE L.1 REAL-DATA SCREENING", "created_utc": datetime.now(timezone.utc).isoformat(),
               "run_root": str(root.relative_to(ROOT)), "protocol": config, "selection": selection,
               "manifest_fingerprint": manifest["fingerprint"], "statuses": statuses, "metrics": metrics,
               "execution": {n: "CACHED phase-P B1" if n == "L0" else "FRESH full training" for n in NAMES},
               "paired_controls": read(root / "train_paired_controls.json"), "paired_effects": effects, "figures": assets,
               "conditional_L5_L7": "NOT RUN — L5–L7 are conditional isolated follow-ups, not the automatic screening matrix"}
    write_json(public / "cp2-phase-l-results.json", summary)
    lines = ["# CP2 Fase L — screening objective regional dan boundary", "",
             "**COMPLETED — empat model dilatih dan dievaluasi pada data nyata, satu seed 42.** L0 adalah B1 Fase P yang dimuat sebagai acuan historis; tidak dilatih ulang. Tidak ada perubahan attention, boundary head, dataset eksternal, atau konfirmasi multi-seed.", "",
             f"Run: `{root.relative_to(ROOT)}`. Manifest `{manifest['fingerprint']}` tetap 671 train / 118 validation / 261 source-test. Ini duplicate-group split, bukan patient-level. Seluruh data mentah dan hasil CP1/P dipertahankan.", "",
             "[Protokol pratraining](../docs/cp2-phase-l.md) · [Config](../configs/cp2/phase_l.json) · [Hasil JSON](cp2-phase-l-results.json)", "",
             "## Pertanyaan dan kontrol", "",
             "Apakah penalti FN lebih besar, eksponen Focal Tversky, atau signed-distance Boundary Loss memperbaiki luka kecil/batas tanpa menambah kegagalan foreground dan FP pada GT kosong? Empat kandidat mulai dari state awal identik B0/B1, bukan checkpoint B1 fine-tuned. Model/augmentasi/split/batch/optimizer/precision/evaluator sama. Checkpoint dan scheduler memakai Image Dice validation, early stopping 30/max 200. LR dan jumlah epoch aktual tetap dapat berbeda sebagai akibat validation.", "",
             "L1=Tversky q=1; L2=Focal Tversky q=0,75; L3=B1+0,01×Boundary; L4=L2+0,01×Boundary. αFP=0,3/βFN=0,7, batch-global, epsilon=1, stabilized power delta=1e-6. Boundary memakai native-pixel signed distance setelah augmentasi, mean p×φ, GT kosong zero map dengan regional penalty tetap aktif. Tidak ada normalization/warm-up. [Definisi dan referensi lengkap](../docs/cp2-phase-l.md#empat-objective-yang-dikunci). Nilai loss lintasobjective tidak dibandingkan sebagai skor kualitas segmentasi.", "",
             "## Hasil selected checkpoints", "",
             "| Objective | Epoch dipilih / selesai | Validation Image Dice | Source-test Image Dice | Waktu invocation terakhir (menit) | Peak allocated / reserved (GiB) |",
             "|---|---:|---:|---:|---:|---:|"]
    for n in NAMES:
        s = statuses[n]
        lines.append(f"| {n}{' (cached)' if n == 'L0' else ''} | {s['selected_epoch']} / {s['completed_epochs']} | {metrics[n]['validation']['all']['image_based']['dice']:.4f} | {metrics[n]['test']['all']['image_based']['dice']:.4f} | {s['seconds_this_invocation']/60:.2f} | {s['peak_cuda_allocated_bytes']/1024**3:.2f} / {s['peak_cuda_reserved_bytes']/1024**3:.2f} |")
    lines += ["", "![Training loss](figures/cp2-phase-l/training_losses.png)", "",
              "**Yang terlihat:** train/validation objective ditampilkan pada panel terpisah per loss. Garis vertikal adalah checkpoint dari Image Dice, bukan minimum loss.", "",
              "**Artinya:** skala objective dan kurva train–validation dapat berubah tanpa perubahan arsitektur. **Perlu diingat:** perbedaan nilai loss antar-panel tidak menunjukkan model mana lebih baik.", "",
              "![Checkpoint and LR](figures/cp2-phase-l/checkpoint_and_lr.png)", "",
              "**Yang terlihat:** checkpoint mengikuti puncak validation Image Dice; scheduler memakai aturan sama tetapi perubahan LR aktual dapat terjadi pada epoch berbeda. **Artinya:** ini kontras recipe training lengkap dengan monitor tetap. **Perlu diingat:** durasi dan jumlah update aktual berbeda akibat early stopping, sehingga belum merupakan kontrol fixed-step budget.", "",
              "## Subgroup dan boundary", "",
              "| Split / kelompok | n | Objective | Image Dice | Data Dice | Data IoU | Data precision | Data recall | Boundary IoU | PFOM |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation", "test"):
        for group in ("all", "gt_positive", "gt_empty", "small"):
            for name in NAMES:
                m = subgroup(metrics[name][split], group)
                values = [m['image_based']['dice'], m['data_based']['dice'], m['data_based']['iou'],
                          m['data_based']['precision'], m['data_based']['recall'],
                          m['image_based']['boundary_iou'], m['image_based']['pfom']]
                lines.append(f"| {split}/{group} | {m['images']} | {name} | " +
                             " | ".join(format_metric(value) for value in values) + " |")
    lines += ["", "![Segmentasi dan batas](figures/cp2-phase-l/segmentation_and_boundary.png)", "",
              "**Yang terlihat:** lima objective dibandingkan pada gambar yang sama, keseluruhan dan luka kecil. **Artinya:** mean per gambar dan aggregate-pixel Dice dapat bergerak berbeda. **Perlu diingat:** kecil berarti 0<luas GT≤1% kanvas yang mengandung padding, bukan ukuran fisik luka. Boundary IoU memakai toleransi 3 pixel; PFOM memakai empty-edge rule CP1 dan bukan pengganti region metrics.", "",
              "![Kesalahan foreground](figures/cp2-phase-l/foreground_and_empty_errors.png)", "",
              "**Yang terlihat:** TP=0 pada GT positif dan GT kosong dengan foreground terprediksi dihitung sebagai jumlah gambar. **Artinya:** recall/luka kecil perlu dibaca bersama false positives. **Perlu diingat:** mask kosong bukan diagnosis sehat; kualitas anotasi belum direview ulang oleh ahli. Semua label tetap dibekukan.", "",
              "## Pemilihan validation-only", "",
              f"Kandidat screening yang dikunci sebelum test baru: **{selection['selected_screening_candidate']}**. Guard mensyaratkan overall Dice lebih tinggi, small Dice tidak turun, TP=0 tidak bertambah, dan jumlah GT kosong dengan FP tidak bertambah terhadap L0. Jika tidak ada yang eligible, L0 dipertahankan. Ini heuristik konservatif untuk eksplorasi, bukan ambang klinis.", "",
              "| Kandidat | Overall Dice naik | Small Dice tidak turun | TP=0 tidak bertambah | Empty FP tidak bertambah | Eligible |",
              "|---|---|---|---|---|---|"]
    for name in NAMES[1:]:
        d = selection["decisions"][name]
        lines.append("| " + name + " | " + " | ".join("Ya" if v else "Tidak" for v in [*d["checks"].values(), d["eligible"]]) + " |")
    lines += ["", "Selection disimpan pada `selection.json` sebelum test predictions L1–L4. Semua empat test hasil tetap ditampilkan, termasuk kandidat yang tidak lolos. Source-test pernah dianalisis CP1/P dan tidak menjadi bukti konfirmatori atau dasar memilih loss.", "",
              "## Efek berpasangan dan interpretasi", "",
              "| Split | Kandidat − L0 | Mean delta Dice (poin persentase) | Bootstrap deskriptif 95% (poin) | Gambar lebih baik / lebih buruk / sama |",
              "|---|---|---:|---:|---:|"]
    for split in ("validation", "test"):
        for name in NAMES[1:]:
            effect = effects[split][name]
            lo, hi = effect["descriptive_rgb_group_bootstrap_95_interval"]
            lines.append(f"| {split} | {name} − L0 | {100*effect['mean_image_dice_minus_L0']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | {effect['better_images']} / {effect['worse_images']} / {effect['tied_images']} |")
    lines += ["", "![Paired effects](figures/cp2-phase-l/paired_dice_effects.png)", "",
              "**Yang terlihat:** tiap titik adalah selisih mean pada pasangan gambar yang sama; garis nol berarti rata-rata sama. **Artinya:** arah dan variasi error perlu dibaca bersama subgroup, bukan hanya ranking tunggal. **Perlu diingat:** 2.000 bootstrap unit RGB group/image, bukan pixel; hubungan pasien/crop tidak diketahui. Interval tidak mencakup retraining seed, checkpoint selection atau multiple comparisons, dan bukan klaim signifikansi.", "",
              "![Skala komponen](figures/cp2-phase-l/boundary_component_scales.png)", "",
              "**Yang terlihat:** kontribusi regional, boundary berbobot dan total berasal dari batch training/validation aktual. Boundary bertanda dapat negatif. **Artinya:** grafik membantu diagnosis faktor L6/L7 sebelum pengembangan tambahan. **Perlu diingat:** distance dihitung dalam pixel; λ=0,01 hanya satu titik screening, bukan nilai optimum. Tidak ada perubahan bobot berdasarkan test.", "",
              "## Biaya, reproducibility dan keterbatasan", "",
              "Training L1→L2→L3→L4 menggunakan satu proses GPU, batch 2/float32 tanpa AMP. Beban GPU bersama berubah antar-epoch dan antarkandidat, termasuk selama L2; L0 juga berasal dari waktu Fase P yang berbeda. Selisih wall time tidak mengisolasi overhead atau speedup loss. Peak allocated/reserved adalah allocator kita; log GPU keseluruhan termasuk proses lain. Tidak ada perubahan lingkungan atau pekerjaan pengguna lain.", "",
              f"Preflight: {test_count} tests lulus, empat real-data smoke lulus, {oracle_count} fixture oracle signed-distance/product-mean bit-identik dengan fungsi penulis yang dipin. Fixture sintetis hanya engineering tests. Baseline tetap utuh, shared initialization/membership/input hashes dibandingkan pada seluruh common epochs. Checkpoint reload diverifikasi bit-identik setelah perturbasi parameter.", "",
              "L.2 (L5–L7) **belum dijalankan**: faktor aggregation, normalization dan warm-up bersyarat pada diagnosis source validation dan memerlukan kontras terpisah; tidak digabung otomatis. Pada fixed-vs-warm-up, kontrol langkah/anggaran harus sama. Screening satu seed belum cukup untuk menetapkan loss terbaik secara umum; konfirmasi beberapa seed sebelum klaim metode atau melanjutkan kandidat penting. Tidak ada generalisasi pasien, klinis atau eksternal yang dibuktikan.", "",
              f"Artefak lengkap lokal: `{root.relative_to(ROOT)}/` (protocol, manifest, shared initial weights, smoke, checkpoints, per-image probabilities/masks/GT, histories, component losses, selection dan logs). Foto klinis/checkpoint besar tidak diunggah. Konfigurasi, per-image metric tables dan source dapat direproduksi mengikuti [panduan](../docs/cp2-phase-l.md).",
              "", "[QA akhir](cp2-phase-l-validation.json) · [Review visual](cp2-phase-l-quality.json). Berkas QA hanya diterbitkan setelah pemeriksaan aktual selesai."]
    destination = public / "cp2-phase-l.md"
    supplement = root / "report_supplement.md"
    if supplement.exists():
        lines += ["", supplement.read_text().strip()]
    if (public / "cp2-phase-l-review.md").exists():
        lines += ["", "[Review akhir dan keputusan lanjutan](cp2-phase-l-review.md)."]
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(destination)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(root / "report_records" / f"{stamp}.json", {"report": str(destination.relative_to(ROOT)), "figures": assets})
    return summary
