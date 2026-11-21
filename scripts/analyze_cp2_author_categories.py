"""CPU-only posthoc C0–C9 metrics from completed CP2 primary test caches.

No checkpoint loading, inference, training, model selection or threshold search.
Author JSON membership applies to the original source-test identities only.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_categories import (  # noqa: E402
    METRICS, author_cohort, author_membership, grouped_metrics, summarize_rows,
)
from foot_ulcer_segmentation.cp2_data import verify_cp2_manifest  # noqa: E402
from validate_cp2_phase_a import validate_exports  # noqa: E402
from validate_cp2_phase_l import compare, compare_csv  # noqa: E402

REFERENCE = "60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc"
AUTHOR = Path("source/upstream/FUSegNet") / REFERENCE / "categorized_oldDfu.json"
ALIASES = {"L0": "B1", "A0": "B1", "H0": "B1"}
PREFIX = "cp2-author-categories"


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_csv(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def phase_sources():
    sources, views, reports = {}, {}, {}
    control = ROOT / read(ROOT / "reports/cp2-phase-p-results.json")["run_root"] / "B1"
    for phase in ("P", "L", "A", "H"):
        report_path = ROOT / f"reports/cp2-phase-{phase.lower()}-results.json"
        report = read(report_path)
        run_root = ROOT / report["run_root"]
        reports[phase] = {"path": relative(report_path), "sha256": digest_file(report_path),
                          "run_root": relative(run_root)}
        views[phase] = []
        for alias, metrics in report["metrics"].items():
            key = ALIASES.get(alias, alias)
            directory = control if alias in ALIASES else run_root / alias
            state = read(directory / "status.json")
            if state["status"] != "completed" or state["diagnostic_smoke_only"]:
                raise ValueError("Only completed real-data primary experiments are eligible")
            compare(metrics["test"], read(directory / "evaluation-test.json"))
            record = {"directory": relative(directory), "phase": "P" if alias in ALIASES else phase,
                      "selected_epoch": state["selected_epoch"],
                      "checkpoint_sha256": state["best_checkpoint_sha256"],
                      "prediction_index_sha256": digest_file(directory / "predictions/test/manifest.json"),
                      "evaluation_sha256": digest_file(directory / "evaluation-test.json"),
                      "split_fingerprint": state["split_fingerprint"]}
            if key in sources:
                compare(record, sources[key])
            else:
                sources[key] = record
            views[phase].append({"alias": alias, "source_model": key,
                                 "cached_baseline_alias": alias in ALIASES})
    return sources, views, reports


def protected_files(root, reports):
    """Snapshot prior research and user WIP, excluding this separate new run."""
    path = root / "protected_before.json"
    if path.exists():
        return read(path)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    names = {name for name in tracked if name}
    roadmap = "Pengembangan_Novelty_dan_Eksperimen_CP2_FUSegNet.md"
    if (ROOT / roadmap).exists():
        names.add(roadmap)
    for report in reports.values():
        directory = ROOT / report["run_root"]
        names.update(relative(p) for p in directory.rglob("*") if p.is_file())
        old_inventory = directory / "protected_prior_sha256.json"
        if old_inventory.exists():
            names.update(read(old_inventory))
    with ThreadPoolExecutor(max_workers=6) as pool:
        inventory = dict(pool.map(lambda n: (n, digest_file(ROOT / n)), sorted(names)))
    write_json(path, inventory)
    write_json(root / "initial_state.json", {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "protected_files": len(inventory)})
    return inventory


def evaluate_sources(root, sources, manifest, settings, membership, inventory):
    metrics, all_rows, cache_checks = {}, [], []
    for name, source in sources.items():
        directory = ROOT / source["directory"]
        if inventory[relative(directory / "best.pth")] != source["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum differs from completed status")
        if read(directory / "split-manifest.json") != manifest:
            raise ValueError("Evaluation populations differ between models")
        expected = read(directory / "evaluation-test.json")
        evaluated = validate_exports(directory / "predictions/test", manifest["splits"]["test"], expected, settings)
        with (directory / "evaluation-test.csv").open(newline="") as stream:
            compare_csv(list(csv.DictReader(stream)), evaluated.rows)
        compare(summarize_rows(evaluated.rows), expected["all"])
        metrics[name] = grouped_metrics(evaluated.rows, membership)
        for row in evaluated.rows:
            category = membership[Path(row["name"]).name]
            all_rows.append({"source_model": name, "category": f"C{category}",
                             "paper_category": category + 1, **row})
        cache_checks.append({"source_model": name, "images": len(evaluated.rows),
                             "raw_GT_probability_checksums_threshold_all_metrics_and_original_CSV": "passed"})
        print(f"Verified cached {name}: {len(evaluated.rows)} test predictions and ten author groups", flush=True)
    return metrics, all_rows, cache_checks


def category_table(metrics, views):
    rows = []
    for phase, models in views.items():
        for model in models:
            key = model["source_model"]
            for category, value in metrics[key].items():
                item = {"phase": phase, "model": model["alias"], "source_model": key,
                        "cached_baseline_alias": model["cached_baseline_alias"],
                        "category": category, "paper_category": int(category[1:]) + 1,
                        "images": value["images"]}
                for field in METRICS:
                    item[f"image_{field}"] = value["image_based"][field] if value["images"] else None
                for field in ("dice", "iou", "precision", "recall"):
                    item[f"data_{field}"] = value["data_based"][field] if value["images"] else None
                for field in ("tp", "tn", "fp", "fn"):
                    item[field] = value.get("counts", {}).get(field)
                for field in ("total_foreground_misses", "images_with_prediction"):
                    item[field] = value.get(field)
                rows.append(item)
    return rows


def render_figures(root, metrics, views, cohort):
    output = root / "figures"
    output.mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    assets = []

    def save(fig, name):
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            if extension == "svg":
                path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
            assets.append({"file": path.name, "sha256": digest_file(path)})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    kept = [r["eligible_images"] for r in cohort]
    removed = [r["excluded_images"] for r in cohort]
    labels = [r["category"] for r in cohort]
    ax.bar(labels, kept, color="#2563EB", label="Eligible CP2 test, n=261")
    ax.bar(labels, removed, bottom=kept, color="#CBD5E1", label="Dikeluarkan audit, n=17")
    for i, (a, b) in enumerate(zip(kept, removed)):
        ax.text(i, a+b+.7, f"{a}/{a+b}", ha="center", fontsize=10)
    ax.set(title="REAL — Chronic Wound | author JSON membership; angka eligible/original",
           xlabel="Kategori kode penulis (C0 = GT kosong)", ylabel="Jumlah gambar", ylim=(0, 44))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=.15)
    save(fig, "cohort")

    for aggregate in ("image_based", "data_based"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, sharey=True)
        for ax, (phase, models) in zip(axes.flat, views.items()):
            for i, model in enumerate(models):
                key = model["source_model"]
                baseline = key == "B1"
                ax.plot(range(1, 10), [metrics[key][f"C{j}"][aggregate]["dice"] for j in range(1, 10)],
                        marker="o", markersize=4, color="#64748B" if baseline else plt.get_cmap("tab10")(i),
                        linewidth=2 if baseline else 1.5,
                        label=model["alias"] + (" = B1 cached" if model["cached_baseline_alias"] else ""))
            ax.set(title=f"Fase {phase} | same 261 test images",
                   xticks=range(1, 10), xticklabels=[f'C{i}\nn={cohort[i]["eligible_images"]}' for i in range(1, 10)],
                   xlabel="Kategori penulis / jumlah gambar", ylabel="Image Dice (0–1)" if aggregate == "image_based" else "Data Dice (0–1)",
                   ylim=(0, 1.03))
            ax.legend(loc="lower right", ncol=2, fontsize=9); ax.grid(alpha=.15)
        label = "Image Dice: rata-rata per gambar" if aggregate == "image_based" else "Data Dice: pooled TP/FP/FN per kategori"
        fig.suptitle(f"REAL / CACHED — Chronic Wound | {label}\nAuthor membership tetap; C0 dievaluasi terpisah; source-test deskriptif", fontsize=14)
        save(fig, aggregate + "_dice")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (phase, models) in zip(axes.flat, views.items()):
        values = [metrics[m["source_model"]]["C0"]["images_with_prediction"] for m in models]
        ax.bar([m["alias"] for m in models], values, color=["#64748B" if m["source_model"] == "B1" else "#2563EB" for m in models])
        for i, value in enumerate(values):
            ax.text(i, value+.15, str(value), ha="center")
        ax.set(title=f'Fase {phase} | C0, n={cohort[0]["eligible_images"]}', xlabel="Model (L0/A0/H0 = B1 cached)",
               ylabel="GT kosong dengan prediksi foreground", ylim=(0, cohort[0]["eligible_images"]+1))
        ax.yaxis.get_major_locator().set_params(integer=True); ax.grid(axis="y", alpha=.15)
    fig.suptitle("REAL / CACHED — Chronic Wound | threshold ≥0,5 | C0 false-positive images\nGT kosong adalah anotasi, bukan diagnosis sehat; source-test deskriptif", fontsize=13)
    save(fig, "empty_false_positives")
    return assets


def render_report(root, results):
    lines = ["# CP2 — evaluasi tambahan menurut kategori asli penulis", "",
             "**COMPLETED — CPU posthoc atas prediksi tersimpan; tanpa training, inference, atau pemilihan ulang.**", "",
             "[Metrik JSON](cp2-author-categories-results.json) · [QA](cp2-author-categories-validation.json) · [Tabel semua metrik](tables/cp2-author-categories-metrics.csv) · [Per gambar](tables/cp2-author-categories-per-image.csv)", "",
             "Analisis mencakup seluruh completed primary models fase P/L/A/H: **13 checkpoint unik, 3.393 prediksi test**. L0, A0 dan H0 adalah alias B1, bukan tiga training atau observasi independen baru. Tidak memasukkan smoke, inference interventions atau auxiliary heads sebagai model primary tambahan.", "",
             "## Keanggotaan dan populasi", "",
             "Keanggotaan mengikuti `categorized_oldDfu.json` penulis pada revision `" + REFERENCE + "`. Kode memakai C0–C9; Figure 5 paper memakai kategori 1–10, sehingga paper ID = kode ID + 1. Tidak merekonstruksi batas dari komentar kode/OCR yang ambigu. Rentang di bawah adalah nilai GT yang benar-benar teramati pada anggota CP2 tersisa, **bukan definisi cutoffs baru**.", "",
             "| Kode / paper | Original n | CP2 eligible n | Excluded n | Rentang GT teramati (% kanvas) |",
             "|---|---:|---:|---:|---:|"]
    for row in results["cohort"]:
        lo, hi = row["eligible_gt_fraction_min"], row["eligible_gt_fraction_max"]
        extent = f"{100*lo:.3f}–{100*hi:.3f}" if lo is not None else "tidak ada anggota"
        lines.append(f'| {row["category"]} / {row["paper_category"]} | {row["original_images"]} | {row["eligible_images"]} | {row["excluded_images"]} | {extent} |')
    lines += ["", "**Yang terlihat:** JSON memuat 278 original source-test IDs; 261 dipertahankan dan 17 tetap dikeluarkan sesuai audit konflik anotasi CP2. Tidak ada gambar yang dikembalikan untuk menyamakan support paper.", "",
              "**Artinya:** model-model CP2 dapat dibandingkan pada kategori dan cohort yang sama. Perbandingan numerik langsung dengan paper/CP1 tetap dibatasi perubahan cohort, split dan evaluator. Ukuran GT adalah luas pada kanvas berpadded 224×224, bukan cm² atau ukuran setiap luka klinis.", "",
              "**Perlu diingat:** JSON hanya memuat original test IDs. **Validation C0–C9: NOT COMPUTED — author membership unavailable for validation identities.** Kategori kecil/sedang/besar CP2 pada validation/test tetap utuh dan berbeda definisinya.", "",
              "![Support kategori](figures/cp2-author-categories/cohort.png)", "",
              "Batang memperlihatkan support original versus eligible; numerator/denominator di atas adalah eligible/original. Kategori tidak seimbang, sehingga perubahan pada beberapa gambar dapat menggeser mean kategori kecil.", "",
              "## Image Dice per kategori positif", "",
              "Semua angka persen. Ini mean Dice per gambar, bukan Data Dice; C0 dipisahkan karena empty rules membuat per-image Dice hanya 0 atau 1.", "",
              "| Model unik | " + " | ".join(f"C{i}" for i in range(1, 10)) + " |",
              "|---|" + "---:|"*9]
    for name, groups in results["metrics"].items():
        lines.append("| " + name + " | " + " | ".join(f'{100*groups[f"C{i}"]["image_based"]["dice"]:.2f}' for i in range(1, 10)) + " |")
    for aggregate, caption in [("image_based_dice", "Mean per gambar memberi bobot sama pada tiap anggota kategori. L0/A0/H0 mengulang cached B1 hanya untuk konteks fase."),
                               ("data_based_dice", "Data Dice dihitung dari TP/FP/FN gabungan dalam kategori, bukan mean Image Dice. Gambar dengan foreground lebih besar dapat memberi kontribusi hitungan lebih besar.")]:
        lines += ["", f"![{aggregate}](figures/cp2-author-categories/{aggregate}.png)", "", caption]
    baseline, first_head, boundary_head = (results["metrics"][n] for n in ("B1", "H1", "H2"))
    lines += ["", "## Contoh interpretasi hasil yang benar-benar dihitung", "",
              f'**Yang terlihat:** pada C1, n={results["cohort"][1]["eligible_images"]}, Image Dice B1 **{100*baseline["C1"]["image_based"]["dice"]:.2f}%** dan H2 **{100*boundary_head["C1"]["image_based"]["dice"]:.2f}%**. Namun pada C5, H2 **{100*boundary_head["C5"]["image_based"]["dice"]:.2f}%**, di bawah B1 **{100*baseline["C5"]["image_based"]["dice"]:.2f}%**. H1 pada C9 **{100*first_head["C9"]["image_based"]["dice"]:.2f}%**, dibanding B1 **{100*baseline["C9"]["image_based"]["dice"]:.2f}%**.', "",
              f'Data Dice C1 justru turun **{100*baseline["C1"]["data_based"]["dice"]:.2f}% → {100*boundary_head["C1"]["data_based"]["dice"]:.2f}%** pada B1→H2. Mean Image Dice yang naik tidak menjamin pooled pixel score naik; kedua agregasi memberi bobot berbeda pada kesalahan antargambar.', "",
              "**Artinya:** skor overall atau small CP2 dapat menyembunyikan tradeoff antarkategori penulis. Tidak ada klaim semua ukuran membaik; lihat seluruh baris dan metrik, bukan kategori yang paling menguntungkan.", "",
              f'**Perlu diingat:** C9 hanya mempunyai {results["cohort"][9]["eligible_images"]} gambar. Ini satu seed dan analisis test posthoc, bukan pemilihan model baru atau bukti signifikansi. B1 tetap baseline terpilih menurut guard validation fase sebelumnya.']
    lines += ["", "## GT kosong dan interpretasi", "",
              "![Empty FP](figures/cp2-author-categories/empty_false_positives.png)", "",
              "C0 memuat 16 gambar eligible. Nilai batang adalah jumlah gambar dengan setidaknya satu pixel prediksi foreground; bukan jumlah pixel FP. Tabel CSV juga menyimpan pixel FP, total foreground misses, Dice/IoU/precision/recall image-based dan data-based, Boundary IoU serta PFOM. Undefined aggregate Dice tetap null; tidak diisi nilai buatan.", "",
              "Perbandingan ini **posthoc/deskriptif** atas source-test yang sudah terekspos CP1/P/L/A/H. Tidak mengubah checkpoint, threshold ≥0,5, empty rules, kategori CP2 lama, atau keputusan guard validation. Tidak memilih kategori/model berdasarkan skor baru, tidak mengklaim generalisasi patient-level, dan tidak menyatakan kesetaraan protokol paper.", "",
              "## Reproduksi dan bukti", "",
              "```bash", ".venv-xfuseg/bin/python scripts/analyze_cp2_author_categories.py \\",
              "  --root outputs/cp2/author_categories/<new-timestamp>", "```", "",
              "Runner CPU-only memverifikasi completed status, shared cohort, checkpoint/index hashes, checksum 3.393 NPZ, raw GT, threshold, seluruh metrik lama dan per-image CSV; baru kemudian mengelompokkan ulang. Tidak memuat bobot model ke Python atau GPU. Run completed tidak ditimpa. `--publish` hanya untuk publikasi pertama; flag itu menolak overwrite laporan publik yang sudah ada. Semua artefak lama dan perubahan lokal pengguna dilindungi hash sebelum/sesudah; raw data, GT, NPZ dan checkpoints tetap lokal.", "",
              "Sumber: [paper Figure 5](https://arxiv.org/pdf/2305.02961v2#page=12), [JSON penulis](https://github.com/mrinal054/FUSegNet/blob/" + REFERENCE + "/categorized_oldDfu.json), [protokol CP2](../docs/cp2-phase-p.md)."]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def publish(root):
    pairs = [(root / "metrics.json", ROOT / f"reports/{PREFIX}-results.json"),
             (root / "validation.json", ROOT / f"reports/{PREFIX}-validation.json"),
             (root / "report.md", ROOT / f"reports/{PREFIX}.md"),
             (root / "per_category.csv", ROOT / f"reports/tables/{PREFIX}-metrics.csv"),
             (root / "per_image.csv", ROOT / f"reports/tables/{PREFIX}-per-image.csv")]
    pairs += [(p, ROOT / "reports/figures" / PREFIX / p.name) for p in (root / "figures").iterdir()]
    if any(target.exists() for _, target in pairs):
        raise FileExistsError("Public analysis already exists; preserve completed artifacts")
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.relative_to(ROOT)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "status.json").exists():
        raise FileExistsError("Preserve existing analysis attempt; use a new run directory")
    torch.set_num_threads(2)
    sources, views, reports = phase_sources()
    inventory = protected_files(root, reports)
    write_json(root / "status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        baseline = ROOT / reports["P"]["run_root"]
        manifest = verify_cp2_manifest(read(baseline / "data/manifest.json"))
        settings = read(baseline / "protocol.json")["evaluation"]
        membership = author_membership(read(ROOT / AUTHOR))
        cohort = author_cohort(manifest, membership)
        if len(membership) != 278 or sum(r["eligible_images"] for r in cohort) != 261:
            raise ValueError("Unexpected author/CP2 cohort; audit explicitly before changing scope")
        for phase in reports.values():
            protocol = read(ROOT / phase["run_root"] / "protocol.json")
            for key, value in settings.items():
                if key != "test_policy" and protocol["evaluation"][key] != value:
                    raise ValueError("Evaluator settings changed across phases")
        metrics, image_rows, checks = evaluate_sources(root, sources, manifest, settings, membership, inventory)
        flat = category_table(metrics, views)
        write_csv(root / "per_category.csv", flat)
        write_csv(root / "per_image.csv", image_rows)
        assets = render_figures(root, metrics, views, cohort)
        with ThreadPoolExecutor(max_workers=6) as pool:
            changed = [name for name, actual in pool.map(lambda n: (n, digest_file(ROOT / n)), inventory)
                       if actual != inventory[name]]
        if changed:
            raise ValueError(f"Protected historical/user files changed: {changed}")
        provenance = {"run_root": relative(root), "created_utc": datetime.now(timezone.utc).isoformat(),
                      "author_json": str(AUTHOR), "author_json_sha256": digest_file(ROOT / AUTHOR),
                      "source_reports": reports, "sources": sources,
                      "manifest_fingerprint": manifest["fingerprint"],
                      "posthoc_sources": {str(p): digest_file(ROOT / p) for p in (
                          Path("scripts/analyze_cp2_author_categories.py"),
                          Path("src/foot_ulcer_segmentation/cp2_categories.py"),
                          Path("tests/test_cp2_categories.py"))}}
        results = {"status": "COMPLETED — REAL CACHED PREDICTIONS, POSTHOC AUTHOR CATEGORIES",
                   "provenance": provenance, "split": "test", "units": "fraction", "threshold": .5,
                   "cohort": cohort, "metrics": metrics, "phase_views": views,
                   "unique_models": len(metrics), "unique_predictions_verified": len(image_rows),
                   "phase_alias_category_rows": len(flat), "figures": assets,
                   "validation_author_categories": "NOT COMPUTED — official membership covers original source-test only",
                   "no_retraining_inference_or_reselection": True,
                   "caveats": ["Previously exposed source-test, descriptive only", "Duplicate RGB group split, not patient-level",
                               "Author category identities retained; 17 excluded original test images not restored",
                               "Paper category ID = JSON key + 1; no threshold reconstruction",
                               "All original CP2 small/medium/large metrics and selection guards preserved"]}
        write_json(root / "metrics.json", results)
        validation = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
                      "historical_and_user_files_unchanged": len(inventory), "changed": changed,
                      "unique_models": len(metrics), "unique_predictions_verified": len(image_rows),
                      "cache_checks": checks, "all_original_overall_metrics_reproduced": True,
                      "author_category_partition_counts_and_weighted_metrics_consistent": True,
                      "alias_B1_not_retrained_or_double_counted": True,
                      "original_source_test": 278, "eligible_test": 261, "excluded_test": 17,
                      "no_training_no_inference_no_GPU": True, "direct_visual_review": "pending"}
        write_json(root / "validation.json", validation)
        render_report(root, results)
        if args.publish:
            publish(root)
        write_json(root / "status.json", {"status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
                                           "metrics_sha256": digest_file(root / "metrics.json")})
        print(f"Completed: {len(metrics)} unique models, {len(image_rows)} cached predictions; unchanged={len(inventory)}", flush=True)
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "error": str(error),
                                           "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
