#!/usr/bin/env python3
"""Verify completed CP2 evidence and render slides13–16, without training/GPU.

Every invocation writes a fresh attempt. --reuse-verified re-renders only a
checksum-verified local payload, useful for typography corrections without
repeating model inference. It does not replace final preservation or visual QA.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import cv2
import numpy as np
from PIL import Image
import torch

from foot_ulcer_segmentation.common import ROOT, digest_file, write_json
from foot_ulcer_segmentation.presentation import read
from foot_ulcer_segmentation.presentation_final import FinalEvidence, miss_transitions
from foot_ulcer_segmentation import presentation_final_plots as plots


def capture_inventory(run, settings):
    """Baseline for a fresh replay; never replace an existing audit inventory."""
    target = run/"protected_before.json"
    if target.exists():
        return
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    paths = {ROOT/name for name in names if (ROOT/name).is_file()}
    historical = read(ROOT/settings["historical_settings"])
    reports = [f["report"] for f in historical["families"]]
    reports += [settings["combinations_report"], settings["cascade_report"]]
    for name in reports:
        report = read(ROOT/name)
        root = ROOT/report.get("run_root", report.get("run"))
        paths.update(p for p in root.rglob("*") if p.is_file())
    run.mkdir(parents=True, exist_ok=True)
    write_json(target, {str(p.relative_to(ROOT)): digest_file(p) for p in sorted(paths)})


def save_payload(folder, payload):
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"__array__": key}
        if isinstance(value, dict):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(v) for v in value]
        return value

    write_json(folder/"payload.json", pack(payload))
    np.savez_compressed(folder/"payload.npz", **arrays)
    return {name: digest_file(folder/name) for name in ("payload.json", "payload.npz")}


def load_payload(folder):
    checksums = read(folder/"manifest.json")["payload_sha256"]
    for name, checksum in checksums.items():
        if digest_file(folder/name) != checksum:
            raise ValueError("Verified renderer payload changed")
    with np.load(folder/"payload.npz", allow_pickle=False) as data:
        def unpack(value):
            if isinstance(value, dict) and set(value) == {"__array__"}:
                return data[value["__array__"]].copy()
            if isinstance(value, dict):
                return {k: unpack(v) for k, v in value.items()}
            return [unpack(v) for v in value] if isinstance(value, list) else value
        return unpack(read(folder/"payload.json"))


def verify_inputs(settings, attempt):
    evidence = FinalEvidence(settings)
    evidence.verify()
    evidence.verify_cascade()
    selected = evidence.select_cases()
    galleries = [evidence.gallery_case(*item) for item in selected]
    traces = evidence.trace([selected[0], selected[-1]], attempt/"traces")
    source_paths = [ROOT/settings[key] for key in ("historical_settings", "combinations_report", "cascade_report")]
    source_paths += [ROOT/f["report"] for f in evidence.settings["families"]]
    for model in evidence.models.values():
        source_paths += [model.root/"status.json", model.root/"history.json"]
        for split in ("validation", "test"):
            source_paths += [model.root/"predictions"/split/"manifest.json", model.root/f"evaluation-{split}.csv"]
    source_paths += [evidence.cascade_root/f for f in ("manifest.json", "protocol.json", "partner_source.py", "selection.json", "refiner/status.json", "refiner/history.json", "validation_coarse/manifest.json")]
    for split in ("validation", "test", "CO2Wounds"):
        source_paths += [evidence.cascade_root/"evaluation"/split/f for f in ("result.json", "prediction_manifest.json", "coarse.csv", "cascade.csv")]
    source_paths += [ROOT/evidence.config["cascade"]["external_root"]/f for f in ("dataset_manifest.json", "input_cache.json")]
    return {"galleries": galleries, "traces": traces, "metrics": evidence.metrics,
            "source_artifacts_sha256": {str(p.relative_to(ROOT)): digest_file(p) for p in source_paths},
            "models": evidence.model_metadata(), "case_metadata": [evidence.case_metadata(s) for s, _ in selected],
            "case_selection_rules": settings["case_selection"], "trace_checks": evidence.trace_checks,
            "checks": evidence.checks, "cohort_fingerprint": evidence.manifest["fingerprint"],
            "paired_coarse_metrics": {s: m["coarse"] for s, m in evidence.cascade_metrics.items()},
            "cascade_metrics": evidence.cascade_metrics, "external_rows": evidence.cascade_tables["CO2Wounds"],
            "external_miss_transitions": miss_transitions(*[evidence.cascade_tables["CO2Wounds"][n] for n in ("coarse", "cascade")]),
            "parameters": {"coarse_active": evidence.parameter_count, "refiner_active": evidence.parameter_count,
                           "total_active": evidence.parameter_count*2, "forwards": "1 + proposal count"},
            "canonical_cascade_report": evidence.cascade_report,
            "optional_total_miss_case": "NOT AVAILABLE: zero coarse/Cascade total foreground misses among111 validation positives",
            "B1_CPU_vs_historical_GPU": {s: {k: evidence.cascade_metrics[s]["coarse"][k]-evidence.metrics["B1"][s][k]
                                       for k in ("image_dice", "small_dice", "boundary_iou", "misses", "empty_fp_images")}
                                        for s in ("validation", "test")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cp2/presentation_final.json")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--reuse-verified", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    settings = read(ROOT/args.config)
    if settings["model_colors"] != plots.COLORS:
        raise ValueError("Plot colors differ from the registered presentation config")
    run = ROOT/args.run_root
    capture_inventory(run, settings)
    attempt = run/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt.mkdir(parents=True, exist_ok=False)
    folder = attempt/"figures"
    folder.mkdir()
    (attempt/"traces").mkdir()
    torch.set_num_threads(settings["cpu_threads"])
    cv2.setNumThreads(1)
    previous = read(args.reuse_verified/"manifest.json") if args.reuse_verified else None
    if previous and previous["config_sha256"] != digest_file(ROOT/args.config):
        raise ValueError("Re-render config differs from the verified experiment presentation")
    payload = load_payload(args.reuse_verified) if args.reuse_verified else verify_inputs(settings, attempt)
    for name, checksum in payload["source_artifacts_sha256"].items():
        if digest_file(ROOT/name) != checksum:
            raise ValueError("Source evidence changed since verification")
    checksums = save_payload(attempt, payload)
    code = [Path(__file__), *[ROOT/"src/foot_ulcer_segmentation"/n for n in
        ("presentation.py", "presentation_final.py", "presentation_final_plots.py", "cp2_cascade.py", "cp2_protocol.py", "cp2_external.py")]]
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "training_performed": False,
        "device": "cpu", "cpu_threads": settings["cpu_threads"], "opencv_threads": 1,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config": args.config, "config_sha256": digest_file(ROOT/args.config),
        "code_sha256": {str(p.relative_to(ROOT)): digest_file(p) for p in code},
        "payload_sha256": checksums, "reuse_verified_attempt": str(args.reuse_verified) if args.reuse_verified else None,
        "verification_parent_manifest_sha256": digest_file(args.reuse_verified/"manifest.json") if args.reuse_verified else None,
        "source_artifacts_sha256": payload["source_artifacts_sha256"],
        "cohort_fingerprint": payload["cohort_fingerprint"], "models": payload["models"], "figures": {}}
    provenance["verification_code_sha256"] = (previous.get("verification_code_sha256", previous["code_sha256"])
                                               if previous else provenance["code_sha256"])
    historical = read(ROOT/settings["historical_settings"])

    def register(files, slide, description, evidence, caveats):
        for file in files:
            item = {"figure": file.name, "slide": slide, "description": description,
                "sha256": digest_file(file), "source": ("SCHEMATIC of verified actual implementation" if slide == 13 and file.suffix == ".svg"
                    else "REAL / CACHED + FRESH CPU refiner" if slide == 13 else "REAL / CACHED; no additional inference"),
                "data_source": "CO2Wounds-V2 Mendeleyv1" if slide == 15 else "Chronic Wound clean CP2",
                "split": "external eligible558 native grid" if slide == 15 else "validation118; train671/source-test261 context",
                "metrics_units": settings["metric_units"], "threshold_rule": "probability >=0.5",
                "cohort_fingerprint": payload["cohort_fingerprint"], "models": payload["models"] if slide == 14 else
                    {n: payload["models"][n] for n in (["B1", "A6", "H2", "Cascade"] if slide == 16 else ["B1", "Cascade"])},
                "evidence": evidence, "caveats": caveats,
                "source_attribution": historical["source_attribution"] if slide != 15 else {
                    "dataset": "CO2Wounds-V2", "doi": "10.17632/s2w7rjwz49.1", "publication": "aggregate only; no photos or clinical mask panels"},
                "renderer": "scripts/build_cp2_final_presentation.py", "code_sha256": provenance["code_sha256"],
                "verification_code_sha256": provenance["verification_code_sha256"],
                "source_artifacts_sha256": payload["source_artifacts_sha256"],
                "config_sha256": provenance["config_sha256"], "base_commit": provenance["base_commit"]}
            if file.suffix == ".png":
                with Image.open(file) as im:
                    if min(im.size) < 1350 or im.width < 2400 or np.asarray(im).std() < 5:
                        raise ValueError("Empty or low resolution figure")
                    item["pixels"] = list(im.size)
            write_json(folder/(file.name+".json"), item)
            provenance["figures"][file.name] = item

    for kwargs in ({}, {"zoom": True}, {"masks": True}):
        register(plots.walkthrough(folder, payload["traces"], **kwargs), 13,
            "Six actual stages, strongest small-wound gain and largest positive-image regression",
            {"cases": [payload["case_metadata"][0], payload["case_metadata"][-1]], "fresh_replay": payload["trace_checks"],
             "selection": [settings["case_selection"][0], settings["case_selection"][-1]], "canonical_results": payload["cascade_metrics"]["validation"]},
            ["GT used only for posthoc illustration/evaluation; no GT proposals", "Crop logits/probabilities captured from exact selected checkpoint", "Zoom is display-only; scores on full224 grid", "Extreme examples are not a frequency estimate"])
    register(plots.architecture(folder, payload["parameters"]), 13, "Actual training/inference distinction and two-model cost",
        {"config": read(ROOT/"configs/cp2/cascade_clean.json"), "parameters": payload["parameters"]},
        ["Schematic, not a measured activation map", "Coarse frozen; only refiner trained; both frozen during this rendering"])
    cohort = {"metrics": payload["metrics"], "paired_CPU_coarse": payload["paired_coarse_metrics"],
              "GPU_CPU_replay_deltas": payload["B1_CPU_vs_historical_GPU"], "parameters": payload["parameters"]}
    cohort["complete_original_phase_selection_decisions"] = {f["id"]: read(ROOT/f["report"]).get("selection")
                                                            for f in historical["families"]}
    cohort["complete_original_phase_selection_decisions"]["C+"] = read(ROOT/settings["combinations_report"])["selection"]
    cohort["complete_original_phase_selection_decisions"]["Cascade"] = payload["canonical_cascade_report"]["selection"]
    caveats = ["Same118 validation image identities; no patient-level claim", "Original phase-specific guards retained; no retrospective reselection",
        "Single seed42 screen; three-seed confirmation remains separate", "Cascade is exempt from 1.01x single-model budget", "Historical B1 GPU metrics preserved; paired Cascade coarse CPU difference explicitly recorded"]
    register(plots.scoreboard(folder, list(payload["models"]), payload["metrics"], payload["models"]), 14,
             "18 completed conditions; validation Image Dice", cohort, caveats)
    register(plots.guard_tradeoffs(folder, list(payload["models"]), payload["metrics"], payload["models"], payload["paired_coarse_metrics"]["validation"]), 14,
             "Absolute overall/small/boundary/miss/empty-FP with original guards and budget applicability", cohort, caveats)
    external = {"metrics": payload["cascade_metrics"]["CO2Wounds"], "miss_transitions": payload["external_miss_transitions"],
                "canonical_result": payload["canonical_cascade_report"]["results"]["CO2Wounds"]}
    delta = np.array([payload["external_rows"]["cascade"][s]["dice"]-payload["external_rows"]["coarse"][s]["dice"]
                      for s in payload["external_rows"]["coarse"]])
    external["paired_image_dice_counts"] = {"improved": int((delta > 0).sum()), "equal": int((delta == 0).sum()),
                                            "regressed": int((delta < 0).sum()), "comparison": "unrounded full-image Dice"}
    caveats = ["All558 GT positive, no GT-empty subgroup", "Cross-source AND cross-pathology; native grid differs from source224", "Previously exposed exploratory target; no model selection from external results", "Aggregate only: no patient photos or derived clinical panels"]
    register(plots.external_comparison(folder, external["metrics"]), 15, "Aggregate gains and increased total misses on same558 images", external, caveats)
    register(plots.external_tradeoffs(folder, payload["external_miss_transitions"], payload["external_rows"]), 15,
             "Paired miss transitions and per-image Dice improvement/regression counts", external, caveats)
    for kwargs in ({}, {"zoom": True}, {"masks": True}):
        register(plots.gallery(folder, payload["galleries"], **kwargs), 16, "Four deterministic actual same-photo comparisons",
                 {"cases": payload["case_metadata"], "selection": settings["case_selection"],
                  "ties": settings["selection_ties"], "optional_total_miss": payload["optional_total_miss_case"]},
                 ["Posthoc extreme illustrations, not prevalence", "GT-empty is annotation status, not healthy diagnosis", "Full224 evaluation for every gallery, zoom is display-only", "B1 paired CPU coarse, A6/H2 historical GPU caches"])
    if len(provenance["figures"]) != 13:
        raise ValueError("Incomplete P0 package")
    provenance["checks"] = payload["checks"]
    provenance["trace_checks"] = payload["trace_checks"]
    provenance["visual_review"] = "PENDING: inspect every PNG and rasterized SVG"
    provenance["seconds"] = perf_counter()-started
    write_json(attempt/"manifest.json", provenance)
    print(f"Created13 final assets: {attempt.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
