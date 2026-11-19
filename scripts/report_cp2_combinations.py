"""Validate and report two real A6/H2 combinations against preserved controls."""

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
from foot_ulcer_segmentation.cp2_combinations import (  # noqa: E402
    CACHED, FRESH, contrasts, directory, histories, prepare, run_config, select, sources,
)
from foot_ulcer_segmentation.cp2_phase_c import pairing  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_n import preservation, read, utc  # noqa: E402
from validate_cp2_phase_a import validate_exports  # noqa: E402
from validate_cp2_phase_l import compare, compare_csv  # noqa: E402


def validate(root):
    registration, config = read(root / "preregistration.json"), read(root / "protocol.json")
    assert registration["sources"] == sources() and registration["config_fingerprint"] == digest_json(config)
    assert read(root / "inference_workflow.json")["status"] == "completed"
    if (root / "final_validation.json").exists():
        raise FileExistsError("Completed QA preserved; inspect its result")
    manifest = prepare(root, config)
    architecture = read(root / "architecture_preflight.json")
    assert architecture["complete_initial_states_equal"] and architecture["initial_main_and_auxiliary_forward_bit_exact"]
    states, proofs = {}, {}
    for name in FRESH:
        run = root / name
        state = states[name] = read(run / "status.json")
        assert state["status"] == "completed" and not state["diagnostic_smoke_only"]
        assert state["training_sources"] == sources()
        assert all(state["first_step_checks"]["updated_parameters"].values())
        assert all(state["checkpoint_replay"].values())
        assert digest_file(run / "best.pth") == state["best_checkpoint_sha256"]
        assert state["identity"] == digest_json({"config": run_config(config, name), "manifest": manifest["fingerprint"], "sources": sources()})
        assert read(run / "config.json") == run_config(config, name)
        assert read(run / "split-manifest.json") == manifest
        assert datetime.fromisoformat(registration["registered_before_training_utc"]) < datetime.fromisoformat(read(run / "environment.json")["created_at_utc"])
        proofs[name] = read(run / "initialization_verified.json")
        compare(proofs[name], architecture["initialization"][name])
        history = read(run / "history.json")
        best = max(history, key=lambda r: r["validation_image_dice"])
        assert best["epoch"] == state["selected_epoch"] and len(history) == state["completed_epochs"]
        assert best["validation_image_dice"] == state["selected_validation_image_dice"]
        for file, epoch in (("best.pth", state["selected_epoch"]), ("last.pth", state["completed_epochs"])):
            checkpoint = torch.load(run / file, map_location="cpu")
            assert checkpoint["identity"] == state["identity"] and checkpoint["completed_epochs"] == epoch
            del checkpoint
    assert proofs[FRESH[0]] == proofs[FRESH[1]]
    assert states[FRESH[0]]["identity"] != states[FRESH[1]]["identity"]
    for name, record in read(root / "control_reference.json").items():
        for file, expected in record["artifact_sha256"].items():
            assert digest_file(directory(root, config, name) / file) == expected
    paired = pairing(histories(root, config, FRESH))
    compare(paired, read(root / "paired_inputs.json"))
    selection = read(root / "selection.json")
    selected_time = datetime.fromisoformat(selection["created_utc"]).timestamp()
    metrics, checks = {}, []
    for name in (*CACHED, *FRESH):
        run = directory(root, config, name)
        metrics[name] = {}
        for split in ("validation", "test"):
            result = metrics[name][split] = read(run / f"evaluation-{split}.json")
            assert result["split_fingerprint"] == manifest["fingerprint"]
            if name in FRESH:
                assert result["checkpoint_sha256"] == states[name]["best_checkpoint_sha256"]
                if split == "validation":
                    assert result["all"]["image_based"]["dice"] == states[name]["selected_validation_image_dice"]
                else:
                    assert all((run / "predictions/test" / r["file"]).stat().st_mtime >= selected_time
                               for r in read(run / "predictions/test/manifest.json"))
            evaluated = validate_exports(run / "predictions" / split, manifest["splits"][split], result, config["evaluation"])
            with (run / f"evaluation-{split}.csv").open(newline="") as stream:
                compare_csv(list(csv.DictReader(stream)), evaluated.rows)
            checks.append({"condition": name, "split": split, "predictions": len(evaluated.rows),
                           "probability_hashes_raw_GT_threshold_metrics_CSV": "passed"})
    selected = select({n: r["validation"] for n, r in metrics.items()}, architecture["profiles"])
    compare(selected, {k: v for k, v in selection.items() if k != "created_utc"})
    effects = {split: contrasts({n: r[split] for n, r in metrics.items()}) for split in ("validation", "test")}
    compare(effects["validation"], read(root / "contrasts-validation.json"))
    write_json(root / "contrasts.json", effects)
    qa = {"status": "passed", "checked_utc": utc(), "checks": checks,
          "predictions_verified": sum(r["predictions"] for r in checks),
          "fresh_predictions_verified": sum(r["predictions"] for r in checks if r["condition"] in FRESH),
          "new_training_runs": 2, "standalone_smoke_training": False,
          "initial_models_and_auxiliary_heads_paired_exactly": True, "separate_loss_run_identities": True,
          "checkpoint_reload_bit_exact": True, "earliest_best_checkpoints_verified": True,
          "actual_input_pairing": paired, "source_validation_selection_before_new_test": True,
          "cached_controls_preserved": True, "historical_user_files": preservation(root),
          "preflight": read(root / "pretraining_checks.json"),
          "external_inference": False, "patient_level_claim": False,
          "full_three_factor_matrix": False, "visual_review": "PENDING — inspect both actual figures"}
    write_json(root / "final_validation.json", qa)
    return config, metrics, effects, qa


def publish(root, config, metrics, effects, qa):
    report = ROOT / "reports/cp2-phase-c-combinations.md"
    figures = ROOT / "reports/figures/cp2-phase-c-combinations"
    if report.exists() or figures.exists():
        raise FileExistsError("Existing completed combination report preserved")
    figures.mkdir(parents=True)
    colors = {"B1": "#64748b", "A6": "#2563eb", "H2": "#7c3aed", "L3": "#a16207",
              "B1A6H2": "#0f766e", "L3A6H2": "#be185d"}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for name, history in histories(root, config, FRESH).items():
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["validation_image_dice"] * 100 for r in history], color=colors[name],
                     linewidth=2 if name in FRESH else 1, label=name + (" FRESH" if name in FRESH else " CACHED"))
        if name in FRESH:
            axes[1].plot(epochs, [r["train_components"]["regional"] for r in history], color=colors[name], label=name)
    axes[0].set(title="Checkpoint dari validation Image Dice", xlabel="Epoch", ylabel="Image Dice (%)", ylim=(0, 100))
    axes[1].set(title="Main regional loss; boundary/auxiliary dipisahkan", xlabel="Epoch", ylabel="Source Dice loss + focal loss")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(fontsize=9)
    fig.suptitle("CP2 · Chronic Wound · train671 / validation118 · seed42 · REAL")
    fig.savefig(figures / "training.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, split in zip(axes, ("validation", "test")):
        for i, name in enumerate((*CACHED, *FRESH)):
            m = metrics[name][split]
            values = np.array([m["all"]["image_based"]["dice"], m["subgroups"]["small"]["image_based"]["dice"],
                               m["all"]["image_based"]["boundary_iou"]]) * 100
            ax.bar(np.arange(3) + (i - 2.5) * .13, values, .13, label=name, color=colors[name])
        ax.set(xticks=range(3), xticklabels=["Overall Dice", "Small Dice", "Boundary IoU"],
               ylim=(0, 100), ylabel="Skor (%)", title=f"{split} · n={118 if split=='validation' else 261}")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=10)
    fig.suptitle("REAL · CP2 matched source split · cached standalone / fresh combinations · seed42")
    fig.savefig(figures / "comparison.png", dpi=150)
    plt.close(fig)
    rows = ["| Model | Split | Image Dice | Data Dice | GT+ Dice | Small Dice | Boundary IoU | GT+ recall | Miss | Empty FP images | Empty FP pixels |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, splits in metrics.items():
        for split, m in splits.items():
            rows.append(f"| {name} | {split} | {m['all']['image_based']['dice']*100:.2f}% | {m['all']['data_based']['dice']*100:.2f}% | "
                        f"{m['gt_positive']['image_based']['dice']*100:.2f}% | {m['subgroups']['small']['image_based']['dice']*100:.2f}% | "
                        f"{m['all']['image_based']['boundary_iou']*100:.2f}% | {m['gt_positive']['data_based']['recall']*100:.2f}% | "
                        f"{m['gt_positive']['total_foreground_misses']} | {m['gt_empty']['images_with_prediction']}/{m['gt_empty']['images']} | {m['gt_empty']['counts']['fp']} |")
            source = directory(root, config, name) / f"evaluation-{split}.csv"
            shutil.copyfile(source, ROOT / "reports/tables" / f"cp2-combinations-{name.lower()}-{split}.csv")
    states = {n: read(root / n / "status.json") for n in FRESH}
    selection = read(root / "selection.json")
    cost = ["| Run | Best / completed epoch | Training hours | Peak allocated / reserved GiB | Active training params | Mask-only params |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    architecture = read(root / "architecture_preflight.json")
    for name, s in states.items():
        cost.append(f"| {name} | {s['selected_epoch']} / {s['completed_epochs']} | {s['seconds_this_invocation']/3600:.2f} | "
                    f"{s['peak_cuda_allocated_bytes']/2**30:.2f} / {s['peak_cuda_reserved_bytes']/2**30:.2f} | "
                    f"{architecture['profiles'][name]['active_forward_parameters']} | {architecture['mask_only_profiles'][name]['active_forward_parameters']} |")
    delta = effects["validation"]["image_dice_pp"]
    report.write_text(f"""# CP2 Fase C — dua kombinasi CBAM A6 dan H2

**COMPLETED — TWO REAL FULL TRAININGS, seed42.** Run `{root.relative_to(ROOT)}`. [Preregistration](../docs/cp2-phase-c-combinations.md) · [Config](../configs/cp2/phase_c_combinations.json).

## Pertanyaan dan kontrol

Apakah kandidat CBAM A6 dan H2 saling melengkapi, dan apakah mengganti main loss B1 menjadi L3 membantu arsitektur gabungan yang sama? B1/A6/H2/L3 cached, tidak dilatih ulang. Historical A1+H2, N1, konfirmasi seed lain dan cascade tetap utuh. Skor terbaik individual tidak menjamin kombinasi terbaik; ketidaklulusan guard historis tidak dihapus.

**B1A6H2:** B1(main mask) +0.1 B1(auxiliary morphological boundary). **L3A6H2:** objective pertama +0.01 mean(sigmoid(main logits) × signed-distance actual augmented GT). L3 sudah mencakup B1; tidak menjumlahkan B1 dua kali. L3 hanya pada main mask, bukan target auxiliary. Signed distance loss dan H2 boundary target mempunyai peran/definisi berbeda dan mungkin redundan.

Kedua run memakai CBAM A6 persis pada decoder stage2–3, termasuk gain3, dan H2 Conv1x1(32,1) dari feature112×112 yang di-upsample ke224. Boundary logits tidak mengarahkan attention; bukan H3. Whole initial model state identik; source tensors, A6 CBAM tensors dan H2 head persis kontrol historis, tanpa warm-start checkpoint terlatih. Main/aux forward awal bit-exact dan actual training input bytes/order dipasangkan pada common epochs.

Split671/118/261 dan seed42 tetap; duplicate-RGB grouping bukan patient-level. Adam1e-4/WD1e-5, batch2/workers4, FP32 tanpaAMP, TF32/augmentation/RNG/evaluator/threshold≥0.5 tetap. Checkpoint earliest strictly best validation Image Dice; Plateau factor0.1/patience10/minLR1e-5, max200epoch, early-stop30. Semua kriteria dipreregister; source-test dan CO2 yang sudah pernah dilihat tidak memilih konfigurasi. Tidak ada inference eksternal baru pada dua run ini.

## Hasil lengkap

{chr(10).join(rows)}

![Training](figures/cp2-phase-c-combinations/training.png)

**Yang terlihat:** validation curves enam kondisi dan regional loss kedua run baru. **Artinya:** checkpoint dipilih dari primary validation, sedangkan main signed-distance dan auxiliary losses dicatat terpisah dalam history JSON. **Perlu diingat:** scheduler sama aturannya, tetapi epoch/update aktual berbeda. Tidak membandingkan total loss dua resep sebagai ukuran kualitas yang sama.

![Perbandingan](figures/cp2-phase-c-combinations/comparison.png)

**Yang terlihat:** overall/small/boundary pada populasi yang identik per split. **Artinya:** perubahan dapat diperiksa bersama false-positive dan miss pada tabel, bukan hanya satu skor. **Perlu diingat:** hanya7 validation GT-empty dan satu seed; source-test sudah terekspos.

## Seleksi dan kontras

Dipilih **{selection['selected_candidate']}** dari source validation. Guard original terhadap B1: overall meningkat, small/boundary tidak turun, misses/empty-FP images tidak bertambah, active training parameters≤1.01×B1. Kombinasi juga harus melampaui standalone relevan pada overall validation (A6/H2, ditambah L3 untuk resep L3). Ranking hanya eligible; tie memilih loss B1 yang lebih sederhana. Jika semua gagal, baseline tetap B1. Keputusan disimpan sebelum semua test probabilities baru.

Validation Image Dice: B1A6H2−B1 **{delta['B1A6H2_minus_B1']:+.3f}pp**; B1A6H2−A6 **{delta['B1A6H2_minus_A6']:+.3f}pp**; L3A6H2−B1A6H2 **{delta['L3A6H2_minus_B1A6H2']:+.3f}pp**. Seluruh perbedaan, subgroup, FP/miss dan descriptive A6×H2 interaction tersedia di JSON.

Ini **bukan full2³ matrix**: L3+A6 dan L3+H2 belum diuji. Selisih dua run baru mengukur perubahan loss pada resep gabungan yang dipasangkan, bukan interaksi tiga-arah yang teridentifikasi atau bukti sinergi kausal. Belum ada konfirmasi multiseed atau validasi pasien independen.

## Biaya dan QA

{chr(10).join(cost)}

Waktu/VRAM di GPU shared, bukan latency benchmark paper. H2 head33parameter dapat dilepas tanpa perubahan primary mask; MAC Conv/Linear dan parameter terdaftar/aktif tersedia di JSON. Single GPU training sequential setelah cascade; tidak mengubah proses orang lain atau shared environment.

**{qa['predictions_verified']} predictions diverifikasi**, termasuk{qa['fresh_predictions_verified']}fresh. Checksums/rawGT/threshold/per-image dan aggregate metrics/CSV diverifikasi, earliest checkpoints dan strict reload main/aux bit-exact. Paired initializer, unique per-loss identities, actual input pairing, chronological selection dan seluruh{qa['historical_user_files']['files']}protected files lulus. [QA](cp2-phase-c-combinations-validation.json). **Visual review PENDING** sampai kedua actual PNG diperiksa; file gambar yang berhasil dibuat belum dianggap visual review.

Reproduksi: environment existing, `scripts/run_cp2_combinations.py --root <new append-only run> --stage prepare`, lalu `--stage queue`. Interrupted stages memerlukan inspected `--resume`; code/config/input/environment harus sama, completed stages tidak dilatih ulang. Checkpoints, source media, masks dan probability maps tetap lokal. Semua hasil merupakan eksplorasi satu sumber; tidak menyatakan keunggulan atas paper atau validitas klinis.
""")
    public = {"status": "completed", "run": str(root.relative_to(ROOT)), "metrics": metrics,
              "contrasts": effects, "selection": selection, "training": states,
              "architecture": architecture, "validation": qa}
    write_json(ROOT / "reports/cp2-phase-c-combinations-results.json", public)
    write_json(ROOT / "reports/cp2-phase-c-combinations-validation.json", qa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config, metrics, effects, qa = validate(args.root.resolve())
    publish(args.root.resolve(), config, metrics, effects, qa)
    print({"status": "completed", "selected": read(args.root / "selection.json")["selected_candidate"], "visual_review": "PENDING"})
