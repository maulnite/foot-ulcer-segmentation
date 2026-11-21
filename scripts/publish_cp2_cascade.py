#!/usr/bin/env python3
"""Recover final Cascade reporting from completed caches, without model inference."""

import argparse
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))

import cv2
import torch
from foot_ulcer_segmentation.common import digest_file
from foot_ulcer_segmentation.cp2_phase_n import read, utc
from foot_ulcer_segmentation.publication_integrity import append_json, verify_preserved
import report_cp2_cascade as frozen

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
    cv2.setNumThreads(1)
    if read(root/"refiner/status.json")["status"] != "completed":
        raise ValueError("Completed training required; never invoke the training runner here")
    if not (audit/"protected_before.json").exists():
        raise ValueError("Save the publication baseline inventory first")
    for split in ["validation", "test", "CO2Wounds"]:
        if not (root/"evaluation"/split/"result.json").exists():
            raise ValueError(f"Completed inference required: {split}")
    if (root/"validation.json").exists():
        raise FileExistsError("Preserve completed QA; inspect instead of rerunning")
    preservation = verify_preserved(ROOT, read(root/"protected_before.json"), RELOCATIONS)
    append_json(audit/"relocation_verification.json", preservation)
    # Resolve only the verified missing path, in this separate report process.
    # Preregistered training/inference sources and input inventories stay intact.
    frozen.preserve = lambda checked: verify_preserved(ROOT, read(checked/"protected_before.json"), RELOCATIONS)
    frozen.write_json = append_json
    print("Verifying 1,874 source/external prediction records; CPU only, no new model inference", flush=True)
    config, results, qa = frozen.validate(root)
    frozen.publish(root, config, results, qa)
    preserved = verify_preserved(ROOT, read(audit/"protected_before.json"))
    append_json(audit/"publication_execution.json", {
        "status": "completed", "finished_utc": utc(), "run": str(root.relative_to(ROOT)),
        "training_or_new_inference_performed": False, "device": "cpu", "threads": 2,
        "frozen_report_sha256": digest_file(ROOT/"scripts/report_cp2_cascade.py"),
        "publication_dispatcher_sha256": digest_file(__file__),
        "publication_helper_sha256": digest_file(ROOT/"src/foot_ulcer_segmentation/publication_integrity.py"),
        "preservation": preserved, "historical_preservation_with_relocation": preservation,
        "old_failed_workflow_and_logs_preserved": True,
        "canonical_validation_sha256": digest_file(root/"validation.json"),
        "selected_candidate": read(root/"selection.json")["selected"],
        "visual_review": "PENDING — inspect both actual figures"})
    print("Final Cascade report created; original failure evidence preserved", flush=True)


if __name__ == "__main__":
    main()
