#!/usr/bin/env python3
"""Finish reporting completed combinations without retraining or changing frozen code."""

import argparse
from pathlib import Path
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))

import torch
from foot_ulcer_segmentation.common import digest_file
from foot_ulcer_segmentation.cp2_phase_n import read, utc
from foot_ulcer_segmentation.publication_integrity import append_json, verify_preserved
import report_cp2_combinations as frozen
from run_cp2_combinations_overlap import annotate_completed_report

RELOCATIONS = {
    "../ganta/notebook/13_cascade_crop_refine_chronic_wound.py":
        "../ganta/notebook/scripts/13_cascade_crop_refine_chronic_wound.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    root, audit = args.root.resolve(), args.audit_root.resolve()
    torch.set_num_threads(2)
    if read(root/"inference_workflow.json")["status"] != "completed":
        raise ValueError("Existing completed real-data inference required")
    if not (audit/"protected_before.json").exists():
        raise ValueError("Save the publication baseline inventory first")
    if (root/"final_validation.json").exists():
        raise FileExistsError("Completed QA is preserved; inspect instead of repeating validation")
    preservation = verify_preserved(ROOT, read(root/"protected_before.json"), RELOCATIONS)
    append_json(audit/"relocation_verification.json", preservation)

    # Override only this reporting process. The registered source files,
    # checkpoint identities, model selection and active Cascade code stay frozen.
    frozen.preservation = lambda checked: verify_preserved(ROOT, read(checked/"protected_before.json"), RELOCATIONS)
    frozen.write_json = append_json
    print("Recomputing 2,274 cached/fresh predictions; CPU only, no training", flush=True)
    config, metrics, effects, qa = frozen.validate(root)
    frozen.publish(root, config, metrics, effects, qa)
    annotate_completed_report(root)
    preserved = verify_preserved(ROOT, read(audit/"protected_before.json"))
    append_json(audit/"publication_execution.json", {
        "status": "completed", "finished_utc": utc(), "run": str(root.relative_to(ROOT)),
        "training_or_new_inference_performed": False, "device": "cpu", "threads": 2,
        "frozen_report_sha256": digest_file(ROOT/"scripts/report_cp2_combinations.py"),
        "publication_dispatcher_sha256": digest_file(__file__),
        "preservation": preserved, "historical_preservation_with_relocation": preservation,
        "old_failed_workflow_and_logs_preserved": True,
        "canonical_final_validation_sha256": digest_file(root/"final_validation.json"),
        "selected_candidate": read(root/"selection.json")["selected_candidate"],
        "visual_review": "PENDING — inspect both figures before final publication"})
    print("Completed report generated; original failed workflow remains as historical evidence", flush=True)


if __name__ == "__main__":
    main()
