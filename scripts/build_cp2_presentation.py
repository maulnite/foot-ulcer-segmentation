#!/usr/bin/env python3
"""Build P0 figures from immutable completed evidence; never train or use a GPU."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import cv2
import numpy as np
from PIL import Image
import torch

from foot_ulcer_segmentation.common import ROOT, digest_file, write_json
from foot_ulcer_segmentation.presentation import Evidence, choose, read
from foot_ulcer_segmentation import presentation_plots as plots
from foot_ulcer_segmentation.presentation_replay import cp1_cases, original_cascade_cases


def preservation(path):
    baseline = read(path)
    changed = [name for name, checksum in baseline.items()
               if not (ROOT/name).is_file() or digest_file(ROOT/name) != checksum]
    if changed:
        raise ValueError(f"Protected research changed: {changed[:5]}")
    return {"checked": len(baseline), "all_unchanged": True}


def case(evidence, names, sample, label):
    rgb, truth = evidence.source("validation", sample)
    return {"sample_id": sample, "label": label, "rgb": rgb, "truth": truth,
            "predictions": {name: evidence.case(name, "validation", sample)[2] for name in names},
            "metrics": {name: evidence.tables[name]["validation"][sample] for name in names},
            "epochs": {name: evidence.models[name].status["selected_epoch"] for name in names},
            "provenance": [evidence.provenance(name, "validation", sample) for name in names]}


def gain_and_regression(evidence, names, candidate, reference="B1"):
    baseline = evidence.tables[reference]["validation"]
    values = list(evidence.tables[candidate]["validation"].values())
    delta = lambda row: row["dice"]-baseline[row["name"]]["dice"]
    gain = choose(values, lambda row: 0 < row["gt_fraction"] <= .01 and delta(row) > 0,
                  lambda row: -delta(row))
    regression = choose(values, lambda row: row["gt_pixels"] > 0 and delta(row) < 0,
                        delta, excluded=[gain])
    return [case(evidence, names, gain, f"Gain luka kecil · {candidate}"),
            case(evidence, names, regression, f"Regresi · {candidate}")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cp2/presentation_p0.json")
    parser.add_argument("--run-root", required=True, help="Dedicated append-only presentation directory")
    args = parser.parse_args()
    started = perf_counter()
    settings = read(ROOT/args.config)
    torch.set_num_threads(settings["cpu_threads"])
    cv2.setNumThreads(1)
    run = ROOT/args.run_root
    run.mkdir(parents=True, exist_ok=True)
    protected = run/"protected_before.json"
    if not protected.exists():
        paths = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        write_json(protected, {path: digest_file(ROOT/path) for path in paths if (ROOT/path).is_file()})
    print("Checking protected inputs before rendering", flush=True)
    before = preservation(protected)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = run/stamp
    folder, traces = attempt/"figures", attempt/"traces"
    folder.mkdir(parents=True, exist_ok=False)
    traces.mkdir()
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "training_performed": False,
                  "device": "cpu", "cpu_threads": settings["cpu_threads"], "opencv_threads": 1,
                  "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "config": args.config, "config_sha256": digest_file(ROOT/args.config),
                  "source_attribution": settings["source_attribution"],
                  "user_authorized_selected_clinical_panels": settings["publish_clinical_analysis_panels"],
                  "selection_policy": settings["selection_policy"], "figures": {},
                  "deferred": settings["defer"], "preservation_before": before}
    code_paths = [Path(__file__), *[ROOT/"src/foot_ulcer_segmentation"/name for name in
                  ["presentation.py", "presentation_plots.py", "presentation_replay.py"]]]
    provenance["code_sha256"] = {str(path.relative_to(ROOT)): digest_file(path) for path in code_paths}

    def register(path, records, selection):
        paths = [path]
        if path.with_suffix(".svg").exists():
            paths.append(path.with_suffix(".svg"))
        for file in paths:
            item = {"file": file.name, "sha256": digest_file(file), "selection": selection,
                    "evidence": records, "source_attribution": settings["source_attribution"],
                    "code_sha256": provenance["code_sha256"], "base_commit": provenance["base_commit"]}
            if file.suffix == ".png":
                with Image.open(file) as image:
                    if image.size != (2400, 1350) or np.asarray(image).std() < 5:
                        raise ValueError("Blank or incorrectly sized presentation figure")
                item["pixels"] = [2400, 1350]
            provenance["figures"][file.name] = item
            write_json(folder/(file.name+".json"), item)

    def galleries(stem, title, subtitle, cases, names, rule):
        records = [record for item in cases for record in item["provenance"]]
        for suffix, kwargs in [("", {}), ("_zoom", {"zoom": True}), ("_masks", {"masks": True})]:
            register(plots.gallery(folder, stem+suffix, title, subtitle, cases, names, **kwargs), records, rule)

    print("Verifying completed CP2 checkpoints and all 5,685 prediction caches", flush=True)
    evidence = Evidence(settings)
    evidence.verify()
    for family, names, candidate, title in [
            ("06_phase_p_case_gallery", ["B0", "B1"], "B1", "Fase P · koreksi input focal: gain dan regresi lokal"),
            ("07_phase_l_case_gallery", ["B1", "L2", "L3"], "L3", "Fase L · perubahan loss pada gambar yang sama"),
            ("08_phase_a_case_gallery", ["B1", "A1", "A6"], "A6", "Fase A · attention: contoh kenaikan dan penurunan"),
            ("09_phase_h_case_gallery", ["B1", "H1", "H2"], "H2", "Fase H · auxiliary supervision dan mask utama")]:
        reference = "B0" if family == "06_phase_p_case_gallery" else "B1"
        selected = gain_and_regression(evidence, names, candidate, reference=reference)
        galleries(family, title, "Chronic Wound · validation n=118 · seed42 · REAL / CACHED",
                  selected, names, f"Maximum positive {candidate}−{reference} Dice gain among 0<GT≤1%; minimum {candidate}−{reference} Dice delta among GT-positive; ties by sample ID")

    n_rows = list(evidence.tables["N1"]["validation"].values())
    b1 = evidence.tables["B1"]["validation"]
    introduced = choose(n_rows, lambda r: r["gt_pixels"] == 0 and r["fp"] > 0 and b1[r["name"]]["fp"] == 0,
                        lambda r: r["name"])
    shared = choose(n_rows, lambda r: r["gt_pixels"] == 0 and r["fp"] > 0 and b1[r["name"]]["fp"] > 0,
                    lambda r: -r["fp"], excluded=[introduced])
    selected = [case(evidence, ["B1", "N1"], introduced, "FP baru pada N1"),
                case(evidence, ["B1", "N1"], shared, "FP pada kedua model")]
    subtitle = ("Chronic Wound · validation n=118, GT kosong n=7 · seed42 · REAL / CACHED\n"
                f"Mean Image Dice seluruh validation: B1 {evidence.metrics['B1']['validation']['image_dice']*100:.2f}% · N1 {evidence.metrics['N1']['validation']['image_dice']*100:.2f}%")
    galleries("10_phase_n_case_gallery", "Fase N · presence head belum menghilangkan false positive", subtitle,
              selected, ["B1", "N1"], "First lexicographic N1-introduced FP on GT-empty; maximum N1 FP among shared FP cases; ties lexicographic")

    names = ["B1", "A6", "H2", "A1H2"]
    selected = gain_and_regression(evidence, names, "A6")[:1]
    used = [item["sample_id"] for item in selected]
    empty = choose(list(b1.values()), lambda r: r["gt_pixels"] == 0,
                   lambda r: -max(evidence.tables[n]["validation"][r["name"]]["fp"] for n in names), excluded=used)
    regression = choose(list(b1.values()), lambda r: r["gt_pixels"] > 0,
                        lambda r: evidence.tables["A1H2"]["validation"][r["name"]]["dice"]-r["dice"], excluded=used+[empty])
    selected.extend([case(evidence, names, empty, "GT kosong · FP"), case(evidence, names, regression, "Regresi A1H2")])
    galleries("15_cross_model_real_cases", "Kasus nyata · empat model pada gambar yang sama",
              "Chronic Wound · validation n=118 · seed42 · REAL / CACHED\nA1H2 = kombinasi historis completed; final kombinasi A6+H2 dan Cascade CP2 masih ditunda",
              selected, names, "Maximum A6−B1 small-wound gain; maximum FP across displayed models on GT-empty; minimum A1H2−B1 positive-image delta; ties lexicographic")

    cp1, cp1_checks = cp1_cases(settings)
    galleries("03_cp1_failure_gallery", "CP1 · tiga kegagalan yang perlu dijelaskan",
              "Chronic Wound · test historis n=278 · seed42 · checkpoint epoch52 · REAL / CACHED",
              cp1, ["CP1"], cp1_checks["selection"])
    metadata = {name: {"epoch": model.status["selected_epoch"], "checkpoint_sha256": model.checkpoint_sha256,
                       "family": model.family, "seed": 42} for name, model in evidence.models.items()}
    cohort = {"models": metadata, "metrics_recomputed": evidence.metrics,
              "manifest_fingerprint": evidence.manifest["fingerprint"], "validation_images": 118, "source_test_images": 261,
              "source": "REAL / CACHED", "original_guards": {n: m.guards for n, m in evidence.models.items()}}
    register(plots.scoreboard(folder, list(metadata), evidence.metrics, metadata), cohort, "All 15 completed seed42 models on the identical clean cohort; source-test descriptive only")
    register(plots.tradeoffs(folder, list(metadata), evidence.metrics, {n: m.guards for n, m in evidence.models.items()}),
             cohort, "Validation deltas vs B1 with each phase's original guard; no model reselection")
    register(plots.combinations_context(folder, evidence.metrics), cohort,
             "Completed B1/A1/H2/A1H2 only; latest combinations explicitly excluded pending final QA")
    print("Replaying two original Ganta cases on CPU; current GPU training is untouched", flush=True)
    cascade, cascade_checks = original_cascade_cases(settings, traces)
    for zoom in [False, True]:
        register(plots.cascade_walkthrough(folder, cascade, zoom=zoom), cascade_checks,
                 "Maximum coarse-to-cascade Dice gain on original positive test images; maximum cascade FP on original GT-empty test images; ties lexicographic")
    provenance["validation"] = {"cp2": evidence.checks, "cp1": cp1_checks,
                                "original_cascade": cascade_checks,
                                "blank_size_checks": "passed", "visual_review": "PENDING: inspect every exported asset"}
    provenance["preservation_after"] = preservation(protected)
    provenance["wall_seconds"] = perf_counter()-started
    write_json(attempt/"manifest.json", provenance)
    print(f"Built {len(provenance['figures'])} assets: {attempt.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
