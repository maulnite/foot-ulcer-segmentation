"""Independently verify predictions and publish aggregate clean cascade results."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_cascade import check_run, decision, preserve, read, save_csv, utc  # noqa: E402
from foot_ulcer_segmentation.cp2_external import binary_mask, restore_probability, verify_dataset  # noqa: E402
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator  # noqa: E402


def compare(expected, actual):
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ValueError("Metric keys differ")
        for key in expected:
            compare(expected[key], actual[key])
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(expected, actual, rtol=0, atol=1e-12):
            raise ValueError(f"Metric differs: {expected} / {actual}")
    elif expected != actual:
        raise ValueError("Metric metadata differs")


def validate(root):
    config, manifest = check_run(root)
    if (root / "validation.json").exists():
        raise FileExistsError("Completed validation preserved; inspect saved result")
    state = read(root / "refiner/status.json")
    assert state["status"] == "completed" and state["checkpoint_reload_bit_exact"]
    assert all(state["first_step_checks"]["updated_parameters"].values())
    assert digest_file(root / "refiner/best.pth") == state["best_checkpoint_sha256"]
    history = read(root / "refiner/history.json")
    best = max(history, key=lambda r: r["validation_full_cascade_image_dice"])
    assert best["epoch"] == state["selected_epoch"] and len(history) == state["completed_epochs"]
    assert best["validation_full_cascade_image_dice"] == state["selected_validation_image_dice"]
    for name, epoch in (("best.pth", state["selected_epoch"]), ("last.pth", len(history))):
        checkpoint = torch.load(root / "refiner" / name, map_location="cpu")
        assert checkpoint["completed_epochs"] == epoch and checkpoint["identity"] == state["identity"]
        del checkpoint
    registration = read(root / "preregistration.json")
    environment = read(root / "refiner/environment.json")
    assert datetime.fromisoformat(registration["registered_utc"]) < datetime.fromisoformat(environment["created_at_utc"])
    selected = read(root / "selection.json")
    selected_time = datetime.fromisoformat(selected["locked_utc"]).timestamp()
    external = ROOT / config["cascade"]["external_root"]
    target = read(external / "dataset_manifest.json")
    verify_dataset(target)
    cache = {r["sample_id"]: r for r in read(external / "input_cache.json")}
    checks, results = [], {}
    for split in ("validation", "test", "CO2Wounds"):
        output = root / "evaluation" / split
        result = results[split] = read(output / "result.json")
        rows = manifest["splits"][split] if split != "CO2Wounds" else target["eligible"]
        index = read(output / "prediction_manifest.json")
        assert [r["sample_id"] for r in index] == [r["sample_id"] for r in rows]
        assert result["images"] == len(rows) and result["buffers_unchanged"] and not result["training_performed"]
        evaluators = {name: CP2Evaluator(config["evaluation"], boundaries=True) for name in ("coarse", "cascade")}
        for row, item in zip(rows, index):
            truth = binary_mask(cv2.imread(str(ROOT / row["mask_path"]), cv2.IMREAD_UNCHANGED))
            for name, evaluator in evaluators.items():
                path = output / item[name]["file"]
                assert digest_file(path) == item[name]["sha256"]
                if split != "validation":
                    assert path.stat().st_mtime >= selected_time
                with np.load(path, allow_pickle=False) as data:
                    probability = data["probability"]
                    assert probability.dtype == np.float32 and probability.shape == (224, 224)
                    native = restore_probability(probability, cache[row["sample_id"]]["geometry"]) if split == "CO2Wounds" else probability
                    np.testing.assert_array_equal(data["truth"], truth)
                    np.testing.assert_array_equal(data["prediction"], native >= .5)
                evaluator.add(row, truth, native)
        for name, evaluator in evaluators.items():
            compare(result["metrics"][name], evaluator.summary())
            with (output / f"{name}.csv").open(newline="") as stream:
                saved = list(csv.DictReader(stream))
            assert len(saved) == len(rows)
            for old, fresh in zip(saved, evaluator.rows):
                for key, value in fresh.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        assert abs(float(old[key]) - value) < 1e-12
                    else:
                        assert old[key] == str(value)
            checks.append({"split": split, "model": name, "predictions_recomputed": len(rows),
                           "hashes_truth_threshold_region_boundary_and_csv": "passed"})
    expected = decision(results["validation"]["metrics"]["coarse"], results["validation"]["metrics"]["cascade"])
    compare(expected, {key: selected[key] for key in expected})
    qa = {"status": "passed", "utc": utc(), "checks": checks,
          "predictions_recomputed": sum(c["predictions_recomputed"] for c in checks),
          "earliest_best_full_validation_checkpoint": True, "selection_before_test_and_external": True,
          "preservation": preserve(root), "checkpoint_reload_bit_exact": True,
          "no_patient_independence_claim": True, "new_training_runs": 1, "preflight": read(root / "preflight.json"),
          "visual_review": "PENDING — inspect generated figures"}
    write_json(root / "validation.json", qa)
    return config, results, qa


def publish(root, config, results, qa):
    report = ROOT / "reports/cp2-cascade-clean.md"
    destination = ROOT / "reports/figures/cp2-cascade-clean"
    if report.exists() or destination.exists():
        raise FileExistsError("Existing report/figures preserved")
    state, selected = read(root / "refiner/status.json"), read(root / "selection.json")
    history = read(root / "refiner/history.json")
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    destination.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_crop_loss"] for r in history], color="#2563eb")
    axes[0].set(title="Refiner belajar pada crop training", xlabel="Epoch", ylabel="B1 crop loss")
    axes[1].plot(epochs, [r["validation_full_cascade_image_dice"] * 100 for r in history], color="#0f766e", label="Cascade validation utuh")
    axes[1].axvline(state["selected_epoch"], color="#64748b", linestyle="--", label=f"Best epoch {state['selected_epoch']}")
    axes[1].set(title="Seleksi tanpa bbox GT validation", xlabel="Epoch", ylabel="Image Dice (%)", ylim=(0, 100))
    axes[1].legend(fontsize=10)
    fig.suptitle("Chronic Wound · clean CP2 · seed42 · train671 / validation118 · REAL")
    fig.savefig(destination / "training.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, split in zip(axes, ("validation", "test", "CO2Wounds")):
        metrics = results[split]["metrics"]
        for offset, name, color in ((-.18, "coarse", "#64748b"), (.18, "cascade", "#0f766e")):
            m = metrics[name]
            values = [m["all"]["image_based"]["dice"] * 100, m["all"]["data_based"]["dice"] * 100,
                      m["all"]["image_based"]["boundary_iou"] * 100]
            ax.bar(np.arange(3) + offset, values, width=.36, label=name, color=color)
        ax.set(xticks=range(3), xticklabels=["Image\nDice", "Data\nDice", "Boundary\nIoU"],
               ylabel="Skor (%)", ylim=(0, 100), title=f"{split} · n={results[split]['images']}")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10)
    fig.suptitle("Coarse B1 vs cascade · REAL / FRESH inference · source224 / CO2 native grid")
    fig.savefig(destination / "comparison.png", dpi=140)
    plt.close(fig)
    lines = ["| Split | Model | n | Image Dice | Data Dice | Small Dice | Boundary IoU | Miss GT+ | FP GT kosong |",
             "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    table = []
    for split, result in results.items():
        for name, m in result["metrics"].items():
            small = m["subgroups"]["small"]["image_based"]
            small_score = f"{small['dice']*100:.2f}%" if small else "N/A"
            lines.append(f"| {split} | {name} | {result['images']} | {m['all']['image_based']['dice']*100:.2f}% | "
                         f"{m['all']['data_based']['dice']*100:.2f}% | {small_score} | {m['all']['image_based']['boundary_iou']*100:.2f}% | "
                         f"{m['gt_positive']['total_foreground_misses']} | {m['gt_empty'].get('images_with_prediction',0)}/{m['gt_empty']['images']} |")
            table.append({"split": split, "model": name, "images": result["images"],
                          "image_dice": m["all"]["image_based"]["dice"], "data_dice": m["all"]["data_based"]["dice"]})
            shutil.copyfile(root / "evaluation" / split / f"{name}.csv", ROOT / "reports/tables" / f"cp2-cascade-clean-{split.lower()}-{name}.csv")
    selected_text = selected["selected"]
    text = f"""# Cascade Ganta pada split bersih Orie

**COMPLETED — ONE REAL REFINER TRAINING, seed42.** Run `{root.relative_to(ROOT)}`.
Coarse adalah B1 CP2 yang sudah completed dan dibekukan. Refiner baru memakai arsitektur Ganta dengan seluruh active initial tensors dipetakan dari initializer CP2; checkpoint terlatih partner tidak dipakai. Ini adaptasi desain Ganta ke protokol CP2, bukan replay historis atau klaim novelty.

## Data dan protokol

Train671 / validation118 / test261; manifest `{config['cascade']['manifest_fingerprint']}`. Kelompok RGB identik tidak melintasi partisi; **bukan patient-level**. Semua crop dari satu foto mengikuti foto sumber. Refiner dilatih pada crop GT training (margin0.3–0.8, peluang random crop0.2). Validation/test/external memakai proposal coarse saja: threshold0.3, komponen8-connected ≥4px, maksimal5, margin0.5, minimum sisi0.25, crop224. Area overlap mengambil maximum probability; luar crop tetap coarse. Final threshold≥0.5.

Loss B1 corrected-logit focal + source Dice, Adam1e-4/WD1e-5, batch2, FP32 tanpa AMP. Scheduler dan early stopping memonitor **mean Image Dice cascade pada 118 foto validation utuh**, tanpa bbox GT validation. Max200 epoch, patience30; earliest strict best. Seleksi klinis dikunci sebelum test/CO2. Batas tambahan parameter1% untuk eksperimen attention tunggal tidak diterapkan pada cascade dua model; biaya tambahan dilaporkan.

## Hasil

{chr(10).join(lines)}

Keputusan berdasarkan validation: **{selected_text}**. Guards: `{json.dumps(selected['guards'])}`.
CPU validation untuk keputusan deployment dapat sedikit berbeda dari GPU validation training; keduanya disimpan dan tidak dipakai untuk memilih ulang epoch.

![Training](figures/cp2-cascade-clean/training.png)

**Yang terlihat:** crop loss dan skor cascade validation lengkap sepanjang training; garis menandai checkpoint yang dipilih. **Artinya:** refiner dipilih menurut dampak pada foto utuh, bukan keberhasilan pada crop GT ideal. **Perlu diingat:** validation yang sama sudah dipakai dalam pengembangan CP2; hasil eksploratif satu seed.

![Perbandingan](figures/cp2-cascade-clean/comparison.png)

**Yang terlihat:** coarse dan cascade dievaluasi pada gambar identik dalam setiap panel. **Artinya:** perbedaan di dalam split menunjukkan dampak refinement untuk coarse yang sama. **Perlu diingat:** source memakai grid224; CO2 native resolution. Domain, anotasi dan grid berbeda, jadi selisih antardataset tidak mengisolasi domain shift.

## Biaya, integritas dan keterbatasan

Best epoch{state['selected_epoch']}, completed{state['completed_epochs']}epoch, training invocation{state['seconds_this_invocation']/3600:.2f}jam. Peak Torch allocated{state['peak_cuda_allocated_bytes']/2**30:.2f}GiB / reserved{state['peak_cuda_reserved_bytes']/2**30:.2f}GiB. Refiner64,898,248 active/trainable parameters, coarse64,898,248 active: cascade129,796,496 active, di luar tensor sumber yang terdaftar tetapi tidak dipanggil. Inference membutuhkan1+jumlah proposal forward; bukan model murah yang memenuhi guard attention1%. Timing CPU shared dan proposal counts ada di JSON; bukan benchmark latency GPU eksklusif.

{qa['predictions_recomputed']} prediction records diverifikasi ulang dari probability, raw GT, threshold, region/boundary metrics dan CSV. Checkpoint reload bit-exact pada device training. Historical/user files{qa['preservation']['files']} tidak berubah. Visual review: **PENDING**, perlu inspeksi dua PNG; pembuatan PNG belum dianggap review.

CO2Wounds-V2 Mendeleyv1:558 eligible dari607 labeled,49 konflik identik-RGB dikarantina;157 tanpa label tidak dievaluasi. Seluruh558 GT positif: tidak mengukur false positive GT kosong eksternal. License ambiguity sebelumnya tetap berlaku; media/probability/checkpoint tetap lokal. Source test dan CO2 sudah pernah dilihat, bukan untouched confirmatory datasets. Tidak ada fitting, threshold tuning atau seleksi konfigurasi dari CO2. Tidak menyimpulkan kesalahan paper atau generalisasi klinis dari satu run.

## Reproduksi

`configs/cp2/cascade_clean.json`, `docs/cp2-cascade-clean.md`, `scripts/run_cp2_cascade.py`. Run menolak overwrite dan memerlukan inspected `--resume` untuk checkpoint terputus. Gunakan environment proyek tanpa upgrade. Training Phase C tidak diubah; worker cascade menunggu workflow Phase C completed sebelum mengalokasikan CUDA.
"""
    report.write_text(text)
    write_json(ROOT / "reports/cp2-cascade-clean-results.json", {"run": str(root.relative_to(ROOT)), "training": state,
               "selection": selected, "results": results, "config": config})
    write_json(ROOT / "reports/cp2-cascade-clean-validation.json", qa)
    save_csv(ROOT / "reports/tables/cp2-cascade-clean-summary.csv", table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config, results, qa = validate(args.root.resolve())
    publish(args.root.resolve(), config, results, qa)
    print("Completed; scientific checks passed; visual review pending", flush=True)
