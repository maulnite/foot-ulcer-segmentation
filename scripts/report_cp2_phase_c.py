"""Validate one joint-model run and all three cached controls; report interactions."""

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
from foot_ulcer_segmentation.cp2_phase_c import (  # noqa: E402
    MODELS, directory, histories, interaction, pairing, select, sources,
)
from foot_ulcer_segmentation.cp2_phase_n import preservation, read, utc  # noqa: E402
from validate_cp2_phase_a import validate_exports  # noqa: E402
from validate_cp2_phase_l import compare, compare_csv  # noqa: E402


def validate(root, config):
    prereg = read(root / "preregistration.json")
    assert prereg["training_sources"] == sources()
    assert prereg["protocol_fingerprint"] == digest_json(config)
    assert read(root / "inference_workflow.json")["status"] == "completed"
    manifest = verify_cp2_manifest(read(root / "data/manifest.json"))
    assert manifest["fingerprint"] == config["phase_c"]["manifest_fingerprint"]
    assert digest_file(root / "initial_weights.pth") == config["phase_c"]["initial_weights_sha256"]
    run, state = root / "A1H2", read(root / "A1H2/status.json")
    assert state["status"] == "completed" and not state["diagnostic_smoke_only"]
    assert state["training_sources"] == sources() and not state["standalone_smoke_training"]
    assert state["first_step_checks"]["finite_nonzero_gradients"]
    assert all(state["first_step_checks"]["updated_parameters"].values())
    assert state["checkpoint_replay"]["primary_and_auxiliary_bit_exact"]
    assert state["checkpoint_replay"]["head_removal_primary_bit_exact"]
    assert read(run / "split-manifest.json") == manifest
    assert state["identity"] == digest_json({"config": config, "manifest": manifest["fingerprint"], "sources": sources()})
    assert datetime.fromisoformat(prereg["registered_before_full_training_utc"]) < datetime.fromisoformat(
        read(run / "environment.json")["created_at_utc"])
    assert digest_file(run / "best.pth") == state["best_checkpoint_sha256"]
    initial = read(run / "initialization_verified.json")
    assert initial["all_original_named_tensors_equal"] and initial["head_parameters"] == 33
    assert initial["source_tensor_sha256"] == config["phase_c"]["initial_tensor_sha256"]
    assert initial["local_tensor_sha256"] == config["phase_c"]["a1_local_tensor_sha256"]
    assert initial["head_tensor_sha256"] == config["phase_c"]["h2_head_tensor_sha256"]
    history = read(run / "history.json")
    best = max(history, key=lambda r: r["validation_image_dice"])
    assert best["epoch"] == state["selected_epoch"] and len(history) == state["completed_epochs"]
    assert best["validation_image_dice"] == state["selected_validation_image_dice"]
    for file, epoch in (("best.pth", state["selected_epoch"]), ("last.pth", state["completed_epochs"])):
        checkpoint = torch.load(run / file, map_location="cpu")
        assert checkpoint["identity"] == state["identity"] and checkpoint["completed_epochs"] == epoch
        del checkpoint
    for name, reference in read(root / "control_reference.json").items():
        path = directory(config, name, root)
        for file, sha in reference["artifact_sha256"].items():
            assert digest_file(path / file) == sha
    paired = pairing(histories(config, root, include_joint=True))
    compare(paired, read(root / "paired_inputs.json"))
    selected = read(root / "selection.json")
    selected_time = datetime.fromisoformat(selected["created_utc"]).timestamp()
    metrics, verified = {}, []
    for name in MODELS:
        path = directory(config, name, root)
        metrics[name] = {}
        for split in ("validation", "test"):
            result = metrics[name][split] = read(path / f"evaluation-{split}.json")
            assert result["split_fingerprint"] == manifest["fingerprint"]
            if name == "A1H2":
                assert result["checkpoint_sha256"] == state["best_checkpoint_sha256"]
            evaluator = validate_exports(path / "predictions" / split, manifest["splits"][split], result, config["evaluation"])
            with (path / f"evaluation-{split}.csv").open(newline="") as stream:
                compare_csv(list(csv.DictReader(stream)), evaluator.rows)
            if name == "A1H2" and split == "validation":
                assert result["all"]["image_based"]["dice"] == state["selected_validation_image_dice"]
            if name == "A1H2" and split == "test":
                assert all((path / "predictions/test" / row["file"]).stat().st_mtime >= selected_time
                           for row in read(path / "predictions/test/manifest.json"))
            verified.append({"model": name, "split": split, "predictions": len(evaluator.rows),
                             "checksums_GT_threshold_metrics_CSV": "passed"})
    decision = select({n: r["validation"] for n, r in metrics.items()}, read(root / "architecture_preflight.json")["profiles"])
    compare(decision, {k: v for k, v in selected.items() if k != "created_utc"})
    effects = {split: interaction({n: r[split] for n, r in metrics.items()}) for split in ("validation", "test")}
    compare(effects["validation"], read(root / "interaction-validation.json"))
    write_json(root / "interactions.json", effects)
    qa = {"status": "passed", "checked_utc": utc(), "predictions_verified": sum(r["predictions"] for r in verified),
          "fresh_predictions_verified": sum(r["predictions"] for r in verified if r["model"] == "A1H2"),
          "evaluations": verified, "preregistered_before_full_training": True,
          "cached_controls_preserved": True, "shared_source_local_and_head_initialization": True,
          "actual_first_step_finite_nonzero_gradients_and_updates": True,
          "checkpoint_replay": "main and auxiliary boundary outputs bit exact",
          "paired_inputs": paired, "earliest_best_checkpoint_verified": True,
          "selection_before_new_test": True, "patient_independence_claimed": False,
          "new_training_runs": 1, "standalone_smoke_training_runs": 0,
          "historical_user_files": preservation(root), "preflight": read(root / "pretraining_checks.json"),
          "visual_review": "PENDING — inspect both aggregate figures after training"}
    write_json(root / "final_validation.json", qa)
    return metrics, effects, qa


def plots(root, config, metrics, effects):
    destination = root / "figures"
    destination.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 140, "savefig.bbox": "tight"})
    colors = {"B1": "#64748b", "A1": "#2563eb", "H2": "#7c3aed", "A1H2": "#0f766e"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for name, history in histories(config, root, include_joint=True).items():
        label = name + (" · FRESH" if name == "A1H2" else " · CACHED")
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["validation_image_dice"] * 100 for r in history], label=label, color=colors[name])
        axes[1].plot(epochs, [r.get("train_components", {}).get("regional", r["train_loss"]) for r in history],
                     label=label, color=colors[name])
    axes[0].set(title="Checkpoint ditentukan oleh validation Image Dice", ylabel="Image Dice (%)", ylim=(0, 100))
    axes[1].set(title="Main segmentation loss; auxiliary tidak dijumlahkan", ylabel="Dice loss + focal loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.15)
        ax.legend(fontsize=10)
    fig.suptitle("Chronic Wound · CP2 A1×H2 · seed42 · validation n=118 · REAL")
    fig.savefig(destination / "training.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for i, name in enumerate(MODELS):
        r = metrics[name]["validation"]
        values = [r["all"]["image_based"]["dice"], r["subgroups"]["small"]["image_based"]["dice"],
                  r["all"]["image_based"]["boundary_iou"]]
        axes[0].bar(np.arange(3) + (i - 1.5) * .19, np.array(values) * 100, .19, label=name, color=colors[name])
    axes[0].set_xticks(np.arange(3), ["Overall Dice", "Small Dice", "Boundary IoU"])
    axes[0].set(title="Empat kondisi pada source validation yang sama", ylabel="Skor (%)", ylim=(0, 100))
    axes[0].legend(fontsize=10)
    keys = ("overall_image_dice_pp", "small_image_dice_pp", "boundary_iou_pp")
    values = [effects["validation"][k]["interaction"] for k in keys]
    axes[1].bar(np.arange(3), values, color=["#0f766e" if v >= 0 else "#b91c1c" for v in values])
    axes[1].axhline(0, color="#64748b", linewidth=1)
    axes[1].set_xticks(np.arange(3), ["Overall Dice", "Small Dice", "Boundary IoU"])
    axes[1].set(title="Interaksi deskriptif: A1H2 − A1 − H2 + B1", ylabel="Selisih interaksi (poin persentase)")
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Chronic Wound · segmentasi luka · seed42 · validation n=118 · CACHED controls / FRESH A1H2")
    fig.savefig(destination / "interaction.png")
    plt.close(fig)
    return list(destination.glob("*.png"))


def publish(root, config, metrics, effects, qa):
    state, decision = read(root / "A1H2/status.json"), read(root / "selection.json")
    figures = plots(root, config, metrics, effects)
    report = ROOT / "reports/cp2-phase-c.md"
    if report.exists():
        raise FileExistsError("Preserve completed phase C report")
    public_figures = ROOT / "reports/figures/cp2-phase-c"
    public_figures.mkdir(exist_ok=False)
    for p in figures:
        shutil.copyfile(p, public_figures / p.name)
    lines = ["| Model | Split | Image Dice | Data Dice | GT+ Dice | Small Dice | Boundary IoU | GT+ pixel recall | Miss | Empty FP images / total | Empty FP pixels |",
             "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in MODELS:
        for split, r in metrics[name].items():
            lines.append(f"| {name} | {split} | {r['all']['image_based']['dice']*100:.2f}% | {r['all']['data_based']['dice']*100:.2f}% | "
                         f"{r['gt_positive']['image_based']['dice']*100:.2f}% | {r['subgroups']['small']['image_based']['dice']*100:.2f}% | "
                         f"{r['all']['image_based']['boundary_iou']*100:.2f}% | {r['gt_positive']['data_based']['recall']*100:.2f}% | "
                         f"{r['gt_positive']['total_foreground_misses']} | {r['gt_empty']['images_with_prediction']} / {r['gt_empty']['images']} | {r['gt_empty']['counts']['fp']} |")
            target = ROOT / f"reports/tables/cp2-phase-c-{name.lower()}-{split}.csv"
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(directory(config, name, root) / f"evaluation-{split}.csv", target)
    write_json(ROOT / "reports/cp2-phase-c-results.json", {
        "status": "COMPLETED — REAL EXPLORATORY TWO-FACTOR SCREEN", "run": str(root.relative_to(ROOT)),
        "metrics": metrics, "interactions": effects, "selection": decision, "A1H2_status": state,
        "profiles": read(root / "architecture_preflight.json"), "validation": qa})
    write_json(ROOT / "reports/cp2-phase-c-validation.json", qa)
    d = effects["validation"]["overall_image_dice_pp"]
    text = f'''# CP2 Fase C — interaksi local attention A1 dan boundary supervision H2

**COMPLETED — satu full training A1H2 seed42. B1/A1/H2 cached; tidak dilatih ulang.**

Run: `{root.relative_to(ROOT)}`. [Preregistration](../docs/cp2-phase-c.md) · [Config](../configs/cp2/phase_c.json) · [JSON](cp2-phase-c-results.json).

## Pertanyaan dan metode

Apakah cabang lokal A1 dan supervisi batas H2 saling melengkapi? Kedua komponen belum lolos seluruh guard individual. Percobaan ini menguji hipotesis interaksi baru; hasil lama tetap berlaku. Tidak mengasumsikan manfaat komponen otomatis dijumlahkan.

A1: DWConv3×3 → PWConv1×1 → sigmoid spatial map, decoder stage2–3, fixed fusion dan gain3 yang sama dengan fase A. H2: auxiliary Conv2d(32,1,1) dari output decoder stage3, upsample logits ke224, target morphological gradient3×3 sesudah augmentasi. Objective = **B1(main,mask) +0,1×B1(aux,boundary)**, menggunakan Dice dan corrected-logit focal yang sama seperti fase H. Boundary logits tidak menjadi input attention; ini **bukan H3**. Auxiliary head dapat dilepas saat mask inference.

Split671/118/261, preprocessing/augmentasi, shared source/local/head initial tensors dan actual input bytes dipasangkan. Seed42, batch2/workers4, Adam1e-4/WD1e-5, FP32/TF32, max200, Plateau validation-Dice/patience10, early stopping30 dan earliest strictly best validation Image Dice sama dengan kontrol. Tidak ada warm-start dari checkpoint terlatih, loss L3, CBAM, presence head atau target tuning.

## Seluruh hasil

{chr(10).join(lines)}

![Training](figures/cp2-phase-c/training.png)

**Yang terlihat:** semua kurva kondisi, CACHED/FRESH dinyatakan; main loss dipisahkan dari auxiliary. **Artinya:** checkpoint ditentukan oleh validation Image Dice, bukan perbandingan total loss yang berbeda. **Perlu diingat:** aturan scheduler sama tetapi jumlah epoch/update aktual berbeda, sehingga ini perbandingan recipe, bukan fixed-update-budget experiment.

## Interaksi dan nilai tambahan

Validation Image Dice A1H2−B1 = **{d['A1H2_minus_B1']:+.3f} poin**; A1H2−A1 = **{d['A1H2_minus_A1']:+.3f}**; A1H2−H2 = **{d['A1H2_minus_H2']:+.3f}**. Interaksi = A1H2−A1−H2+B1 = **{d['interaction']:+.3f} poin**.

![Interaksi](figures/cp2-phase-c/interaction.png)

**Yang terlihat:** metrik keempat kondisi dan selisih interaksi numeriknya pada source validation. **Artinya:** interaksi positif berarti gabungan melampaui jumlah efek individual pada skala yang ditulis; belum tentu mengungguli kontrol terbaik atau lolos FP guard. **Perlu diingat:** satu seed, validation reuse, nonlinear metrics dan adaptive training membatasi interpretasi. Ini tidak membuktikan sinergi kausal, signifikansi statistik, generalisasi pasien atau novelty.

Semua delta, termasuk positive-pixel recall, GT-positive Dice, empty-FP images/pixels dan total miss tersedia di JSON. Source-test dilaporkan deskriptif setelah selection tersimpan; tidak digunakan mengubah konfigurasi atau keputusan.

## Seleksi dan biaya

Selected **{decision['selected_candidate']}**; eligible **{decision['eligible']}**. Original CP2 guards: `{decision['original_cp2_guards']}`. Lebih tinggi dari kedua individual pada overall validation: **{decision['higher_overall_than_both_individuals']}**.

Guard historis overall/small/boundary/misses/empty-FP/capacity dipertahankan. Tambahan nilai kombinasi mensyaratkan overall validation Dice melampaui A1 dan H2. Tidak meloloskan ulang individual yang gagal dan tidak memperluas matriks otomatis. Bila eligible, perlu review replikasi sebelum mengklaim peningkatan; jika gagal, B1 tetap baseline.

Epoch selected/completed **{state['selected_epoch']}/{state['completed_epochs']}**, waktu invocation **{state['seconds_this_invocation']/3600:.2f} jam**, peak allocated/reserved **{state['peak_cuda_allocated_bytes']/1024**3:.2f}/{state['peak_cuda_reserved_bytes']/1024**3:.2f} GiB**. Empat profil parameter/MAC training dan mask-only ada pada JSON. Waktu GPU shared bukan latency benchmark eksklusif. Boundary auxiliary tidak menambah parameter aktif saat mask-only inference.

## QA dan keterbatasan

{qa['predictions_verified']} prediksi total diverifikasi, terdiri dari {qa['fresh_predictions_verified']} fresh dan sisanya cached; checksum, raw GT, threshold≥0,5, metrik dan CSV cocok. Source, local tensors dan boundary head diinisialisasi persis kontrol. First actual full-training step memeriksa encoder/kedua local branch/boundary head; seluruh langkah menolak nonfinite. Checkpoint main/aux reload bit-exact dan earliest-best policy diverifikasi. Tidak ada smoke training terpisah. [QA](cp2-phase-c-validation.json).

Seluruh {qa['historical_user_files']['files']} historical/user files dalam inventory tetap utuh. Duplicate-RGB grouping bukan patient split; patient ID tidak tersedia. Tujuh GT-empty validation membatasi kepastian FP. CO2Wounds sudah dilihat dan tidak digunakan memilih konfigurasi C; evaluasi eksternal baru tidak dijalankan. Tidak ada media pasien dalam output publik.

**Visual review: PENDING — kedua figure harus diperiksa langsung setelah training.**
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
        raise FileExistsError("Preserve completed QA; inspect before isolated recovery")
    config = read(root / "protocol.json")
    metrics, effects, qa = validate(root, config)
    publish(root, config, metrics, effects, qa)


if __name__ == "__main__":
    main()
