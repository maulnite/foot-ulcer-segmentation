"""Validate N1 artifacts and generate aggregate reports after full training."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_data import verify_cp2_manifest  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_n import check_pairing, preservation, read, sources, utc  # noqa: E402
from foot_ulcer_segmentation.cp2_presence import select_presence  # noqa: E402
from validate_cp2_phase_a import validate_exports  # noqa: E402
from validate_cp2_phase_l import compare, compare_csv  # noqa: E402


def validate(root):
    config = read(root / "protocol.json")
    registration = read(root / "preregistration.json")
    if registration["training_sources"] != sources() or registration["protocol_fingerprint"] != digest_json(config):
        raise ValueError("Preregistered source/config changed")
    if read(root / "inference_workflow.json")["status"] != "completed":
        raise ValueError("Real training and inference must complete")
    run = root / "N1"
    status = read(run / "status.json")
    assert status["status"] == "completed" and not status["diagnostic_smoke_only"]
    assert not status["standalone_smoke_training"]
    assert status["first_step_checks"]["finite_gradients"]
    assert all(status["first_step_checks"]["updated_parameters"].values())
    assert status["checkpoint_replay"]["primary_and_presence_bit_exact"]
    assert datetime.fromisoformat(registration["registered_before_full_training_utc"]) < datetime.fromisoformat(
        read(run / "environment.json")["created_at_utc"])
    manifest = verify_cp2_manifest(read(root / "data/manifest.json"))
    assert manifest["fingerprint"] == config["phase_n"]["control_manifest_fingerprint"]
    assert read(run / "split-manifest.json") == manifest
    identity = digest_json({"config": config, "manifest": manifest["fingerprint"], "sources": sources()})
    assert status["identity"] == identity and status["training_sources"] == sources()
    assert digest_file(run / "best.pth") == status["best_checkpoint_sha256"]
    assert digest_file(root / "initial_weights.pth") == config["phase_n"]["control_initial_weights_sha256"]
    control = ROOT / config["phase_n"]["control_root"] / "B1"
    assert digest_file(control / "best.pth") == config["phase_n"]["control_B1_checkpoint_sha256"]
    assert digest_file(control / "history.json") == read(root / "control_reference.json")["history_sha256"]
    initial = read(run / "initialization_verified.json")
    assert initial["all_original_named_tensors_equal"] and initial["head_parameters"] == 33
    assert initial["source_tensor_sha256"] == config["phase_n"]["control_initial_tensor_sha256"]
    history = read(run / "history.json")
    best = max(history, key=lambda row: row["validation_image_dice"])
    assert best["epoch"] == status["selected_epoch"] and len(history) == status["completed_epochs"]
    assert best["validation_image_dice"] == status["selected_validation_image_dice"]
    for name, epoch in (("best.pth", status["selected_epoch"]), ("last.pth", status["completed_epochs"])):
        checkpoint = torch.load(run / name, map_location="cpu")
        assert checkpoint["identity"] == identity and checkpoint["completed_epochs"] == epoch
        del checkpoint
    selected = read(root / "selection.json")
    selected_at = datetime.fromisoformat(selected["created_utc"]).timestamp()
    checked = []
    for split in ("validation", "test"):
        report = read(run / f"evaluation-{split}.json")
        assert report["checkpoint_sha256"] == status["best_checkpoint_sha256"]
        evaluator = validate_exports(run / "predictions" / split, manifest["splits"][split], report, config["evaluation"])
        with (run / f"evaluation-{split}.csv").open(newline="") as stream:
            compare_csv(list(csv.DictReader(stream)), evaluator.rows)
        if split == "validation":
            assert report["all"]["image_based"]["dice"] == status["selected_validation_image_dice"]
        if split == "test":
            for item in read(run / "predictions/test/manifest.json"):
                assert (run / "predictions/test" / item["file"]).stat().st_mtime >= selected_at
        with (run / f"presence-{split}.csv").open(newline="") as stream:
            presence = list(csv.DictReader(stream))
        rows = manifest["splits"][split]
        assert len(presence) == len(rows)
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for p, row in zip(presence, rows):
            assert p["sample_id"] == row["sample_id"]
            probability = float(p["presence_probability"])
            assert np.isfinite(probability) and 0 <= probability <= 1
            truth, prediction = int(row["gt_pixels"] > 0), int(probability >= .5)
            assert int(p["ground_truth_presence"]) == truth and int(p["predicted_presence"]) == prediction
            assert p["diagnostic_only_not_mask_gate"] == "True"
            counts[("tp" if prediction else "fn") if truth else ("fp" if prediction else "tn")] += 1
        presence_report = read(run / f"presence-{split}.json")
        assert counts == presence_report["counts"] and presence_report["images"] == len(rows)
        np.testing.assert_allclose(presence_report["balanced_accuracy"], .5 * (
            counts["tp"] / (counts["tp"] + counts["fn"]) + counts["tn"] / (counts["tn"] + counts["fp"])), atol=1e-12, rtol=0)
        checked.append({"split": split, "predictions": len(rows), "checksums_GT_metrics_CSV_presence": "passed"})
    profiles = read(root / "architecture_preflight.json")["profiles"]
    decision = select_presence(read(run / "evaluation-validation.json"), read(control / "evaluation-validation.json"),
                               profiles["N1"], profiles["N0"])
    compare(decision, {key: value for key, value in selected.items() if key != "created_utc"})
    paired = check_pairing(root, config)
    compare(paired, read(root / "paired_inputs.json"))
    result = {"status": "passed", "checked_utc": utc(), "prediction_checks": checked,
              "predictions_verified": sum(row["predictions"] for row in checked),
              "preregistered_before_training": True, "first_actual_step_verified": True,
              "separate_smoke_training_runs": 0, "training_runs": 1,
              "all_original_initial_tensors_match_B1": True, "paired_inputs": paired,
              "earliest_best_validation_checkpoint_verified": True,
              "selection_before_new_test": True, "checkpoint_reload": "main and presence bit exact",
              "historical_user_files": preservation(root), "patient_independence_claimed": False,
              "preflight": read(root / "pretraining_checks.json"), "visual_review": "PENDING — inspect after training"}
    write_json(root / "final_validation.json", result)
    return result


def plots(root, config):
    destination = root / "figures"
    destination.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 140, "savefig.bbox": "tight"})
    control = ROOT / config["phase_n"]["control_root"] / "B1"
    histories = {"B1 — CACHED": read(control / "history.json"), "N1 — FRESH": read(root / "N1/history.json")}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for (name, history), color in zip(histories.items(), ("#64748b", "#7c3aed")):
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["validation_image_dice"] * 100 for r in history], label=name, color=color)
        # B1 total is its main objective; N1 main is logged separately from presence BCE.
        axes[1].plot(epochs, [r["train_components"]["regional"] if "train_components" in r
                             else r["train_loss"] for r in history], label=name, color=color)
    new = list(histories.values())[1]
    axes[2].plot([r["epoch"] for r in new], [r["train_components"]["presence_bce"] for r in new], label="Train", color="#7c3aed")
    axes[2].plot([r["epoch"] for r in new], [r["validation_components"]["presence_bce"] for r in new], label="Validation", color="#0f766e")
    for ax, title, ylabel in zip(axes, ("Validation Image Dice", "Main segmentation loss saat training", "Auxiliary presence loss N1"),
                                ("Image Dice (%)", "Dice loss + focal loss", "BCE loss (unweighted)")):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.15)
        ax.legend(fontsize=10)
    axes[0].set_ylim(0, 100)
    fig.suptitle("Chronic Wound · CP2 · seed42 · validation n=118 · REAL / CACHED vs FRESH")
    fig.savefig(destination / "training.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, split in zip(axes, ("validation", "test")):
        for offset, (name, directory, color) in enumerate((("B1 — CACHED", control, "#64748b"), ("N1 — FRESH", root / "N1", "#7c3aed"))):
            r = read(directory / f"evaluation-{split}.json")
            values = [r["all"]["image_based"]["dice"], r["gt_positive"]["image_based"]["dice"],
                      r["subgroups"]["small"]["image_based"]["dice"], r["all"]["image_based"]["boundary_iou"]]
            ax.bar(np.arange(4) + (offset - .5) * .36, np.array(values) * 100, .36, label=name, color=color)
        ax.set_xticks(np.arange(4), ["Overall\nDice", "GT positif\nDice", "Luka kecil\nDice", "Boundary\nIoU"])
        ax.set(title=f"{split} · n={r['all']['images']}", ylabel="Skor (%)", ylim=(0, 100))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Chronic Wound · segmentasi luka · seed42 · test deskriptif setelah seleksi validation")
    fig.savefig(destination / "comparison.png")
    plt.close(fig)
    return list(destination.glob("*.png"))


def publish(root, qa):
    config, status = read(root / "protocol.json"), read(root / "N1/status.json")
    control = ROOT / config["phase_n"]["control_root"] / "B1"
    decision = read(root / "selection.json")
    figures = plots(root, config)
    report = ROOT / "reports/cp2-phase-n.md"
    if report.exists():
        raise FileExistsError("Preserve completed report")
    public_figures = ROOT / "reports/figures/cp2-phase-n"
    public_figures.mkdir(exist_ok=False)
    for figure in figures:
        shutil.copyfile(figure, public_figures / figure.name)
    lines = ["| Model | Split | Image Dice | Data Dice | GT+ Dice | Small Dice | Boundary IoU | Recall pixel GT+ | Miss | GT-empty FP images / total | GT-empty FP pixels |",
             "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    metrics = {}
    for name, directory in (("B1", control), ("N1", root / "N1")):
        metrics[name] = {}
        for split in ("validation", "test"):
            r = metrics[name][split] = read(directory / f"evaluation-{split}.json")
            lines.append(f"| {name} | {split} | {r['all']['image_based']['dice']*100:.2f}% | {r['all']['data_based']['dice']*100:.2f}% | "
                         f"{r['gt_positive']['image_based']['dice']*100:.2f}% | {r['subgroups']['small']['image_based']['dice']*100:.2f}% | "
                         f"{r['all']['image_based']['boundary_iou']*100:.2f}% | {r['gt_positive']['data_based']['recall']*100:.2f}% | "
                         f"{r['gt_positive']['total_foreground_misses']} | {r['gt_empty']['images_with_prediction']} / {r['gt_empty']['images']} | {r['gt_empty']['counts']['fp']} |")
            target = ROOT / f"reports/tables/cp2-phase-n-{name.lower()}-{split}.csv"
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(directory / f"evaluation-{split}.csv", target)
    public = {"status": "COMPLETED — REAL SINGLE-SEED SCREEN", "run": str(root.relative_to(ROOT)),
              "metrics": metrics, "selection": decision, "N1_status": status, "validation": qa,
              "profiles": read(root / "architecture_preflight.json"),
              "presence_head_diagnostic": {s: read(root / f"N1/presence-{s}.json") for s in ("validation", "test")}}
    write_json(ROOT / "reports/cp2-phase-n-results.json", public)
    write_json(ROOT / "reports/cp2-phase-n-validation.json", qa)
    text = f'''# CP2 Fase N — standalone presence supervision

**COMPLETED — satu full training N1, seed42; B1 cached tidak dilatih ulang. Tidak ada smoke training terpisah.**

Run: `{root.relative_to(ROOT)}`. [Preregistration](../docs/cp2-phase-n.md) · [Config](../configs/cp2/phase_n.json) · [Metrik JSON](cp2-phase-n-results.json).

## Pertanyaan dan metode

Apakah tugas tambahan mengenali keberadaan foreground membantu segmentasi mengurangi false positive pada anotasi kosong? Presence berarti ada piksel luka pada anotasi, bukan diagnosis pasien sehat/sakit.

Source FUSegNet/P-scSE dan objective utama B1 tetap. Fitur decoder stage3 [B,32,112,112] dirata-ratakan secara spasial menjadi [B,32], lalu Linear(32,1) menghasilkan satu presence logit per gambar. Tambahan 33 parameter. Objective = B1 + 0,1 × BCEWithLogits(mean), tanpa class balancing. Target presence dihitung dari mask sesudah augmentasi, termasuk luka satu piksel. Mask utama tidak dikalikan presence probability; auxiliary head dapat dilepas saat inference.

Split group RGB tetap 671/118/261, tanpa patient ID. Shared initial weights, epoch RNG, actual order/augmented bytes pada common epochs, Adam, LR, batch2, FP32/TF32, scheduler dan early stopping mengikuti B1. Max200, earliest strictly best validation Image Dice, Plateau patience10, early stopping30. Checkpoint aktual: epoch **{status['selected_epoch']}**, total **{status['completed_epochs']}** epoch. Satu run memerlukan **{status['seconds_this_invocation']/3600:.2f} jam invocation**, peak CUDA allocated/reserved **{status['peak_cuda_allocated_bytes']/1024**3:.2f}/{status['peak_cuda_reserved_bytes']/1024**3:.2f} GiB**. Ini biaya GPU shared, bukan benchmark latency eksklusif.

## Hasil seluruh kondisi

{chr(10).join(lines)}

![Kurva](figures/cp2-phase-n/training.png)

**Yang terlihat:** kurva B1 historis dan N1 baru, main loss terpisah dari auxiliary BCE. **Artinya:** checkpoint dipilih dari Image Dice validation, bukan total loss yang berbeda antar-kondisi. **Perlu diingat:** scheduler dan early stopping menghasilkan jumlah update berbeda; satu seed tidak menggambarkan variasi training.

![Perbandingan](figures/cp2-phase-n/comparison.png)

**Yang terlihat:** overall/GT-positif/luka-kecil Dice dan Boundary IoU pada dua split ditampilkan terpisah. **Artinya:** improvement rata-rata harus diperiksa bersama kegagalan mendeteksi luka dan FP pada GT kosong. **Perlu diingat:** source-test sudah terekspos dan hanya deskriptif. Hanya tujuh GT-empty validation; satu kasus sangat memengaruhi Image Dice.

## Keputusan source validation

Selected: **{decision['selected_candidate']}**. N1 eligible: **{decision['eligible']}**. Rincian guard: `{decision['checks']}`.

Semua guard historis dipertahankan; ditambahkan positive pixel recall tidak turun, empty-FP pixels tidak bertambah, dan FP images atau pixels wajib berkurang. Ini seleksi satu-seed eksploratif. Kelolosan menjadi bahan review konfirmasi sebelum kombinasi; tidak ada H2/A/H3/N2 yang dilatih otomatis. Jika gagal, B1 tetap baseline dan hasil negatif dipertahankan.

## QA dan keterbatasan

Unit tests/dependency/inisialisasi/structural forward dijalankan sebelum training; tidak ada run training pendek tambahan sesuai instruksi pengguna. First actual optimizer step memverifikasi gradien finite serta pembaruan encoder dan presence head; seluruh training steps tetap menolak nonfinite. Checkpoint main/presence reload bit-exact. {qa['predictions_verified']} prediksi diverifikasi dengan checksum, raw GT, threshold≥0,5, metrics dan CSV. Split/initial tensors dipasangkan dan selection disimpan sebelum prediksi test baru. [QA lengkap](cp2-phase-n-validation.json).

Presence head probabilities/confusion matrix tersimpan lokal sebagai diagnostik; tidak dipakai sebagai mask gate atau kriteria pemilihan checkpoint. Dataset train didominasi foreground positif; head yang menebak semua positif dapat memiliki accuracy tinggi, sehingga balanced accuracy/positive dan negative recall juga disimpan.

Tidak ada external tuning/training: CO2Wounds sudah diamati sebelumnya dan tidak dipakai memilih konfigurasi N. Pengujian eksternal konfirmatori perlu sumber lain yang belum digunakan untuk pengembangan. Tidak ada klaim patient independence, signifikansi klinis atau novelty global.

**Visual review figure: PENDING — perlu inspeksi manual setelah training selesai.** Semua grafik berasal dari artefak real-data; tidak ada media pasien dalam output publik. Seluruh {qa['historical_user_files']['files']} file historis/user dalam inventory tetap utuh.
'''
    with report.open("x") as stream:
        stream.write(text)
    print({"report": str(report.relative_to(ROOT)), "selected_candidate": decision["selected_candidate"],
           "eligible": decision["eligible"], "visual_review": "pending"}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "final_validation.json").exists():
        raise FileExistsError("Preserve completed validation/report; inspect before recovery")
    qa = validate(root)
    publish(root, qa)


if __name__ == "__main__":
    main()
