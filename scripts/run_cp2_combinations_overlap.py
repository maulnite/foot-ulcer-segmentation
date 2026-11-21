"""User-authorized scheduling amendment; scientific experiment code stays frozen.

Release the original wait gate only. Both new models still train sequentially,
with their original configurations, identities, input pairing and selection.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
import run_cp2_combinations as frozen  # noqa: E402


def approved_gate(root, config):
    amendment = frozen.read(root / "execution_policy_override.json")
    if amendment["scope"] != "overlap one combination with cascade; combinations remain sequential":
        raise ValueError("Unexpected scheduling override")
    if digest_file(__file__) != amendment["launcher_sha256"]:
        raise ValueError("Scheduling launcher changed")
    if digest_file(root / "preregistration.json") != amendment["original_preregistration_sha256"]:
        raise ValueError("Original scientific registration changed")
    if digest_json(config) != amendment["scientific_config_fingerprint"]:
        raise ValueError("Scientific config changed")
    if frozen.sources() != frozen.read(root / "preregistration.json")["sources"]:
        raise ValueError("Frozen experiment code changed")
    if amendment["new_combination_order"] != list(frozen.FRESH):
        raise ValueError("Training order changed")
    if not amendment["queued_predecessor_verified_unstarted_and_stopped"]:
        raise ValueError("Previous queue must be safely handed over")
    write_json(root / "scheduling_gate_released.json", {
        "released_utc": datetime.now(timezone.utc).isoformat(),
        "override_sha256": digest_file(root / "execution_policy_override.json"),
        "scientific_config_unchanged": True, "experiment_sources_unchanged": True,
        "new_models_still_sequential": True,
        "existing_cuda_memory_gate_still_applies": True})
    print("User-authorized overlap: original cascade wait skipped; scientific protocol unchanged", flush=True)


def annotate_completed_report(root):
    """Correct the presentation's old scheduling sentence, without editing QA/metrics."""
    evidence = root / "execution_policy_presentation.json"
    if evidence.exists():
        return
    report = ROOT / "reports/cp2-phase-c-combinations.md"
    original = report.read_text()
    old = "Single GPU training sequential setelah cascade; tidak mengubah proses orang lain atau shared environment."
    new = ("Kedua kombinasi berjalan sequential, dengan izin overlap satu training kombinasi dan cascade. "
           "Penundaan menunggu cascade dibuka melalui scheduling amendment sebelum training pertama; "
           "konfigurasi ilmiah, inisialisasi, loss, data dan seleksi tidak berubah. "
           "Runtime/VRAM berasal dari GPU shared dengan concurrent workloads. "
           "Tidak mengubah proses orang lain atau shared environment.")
    if original.count(old) != 1:
        raise ValueError("Report scheduling text differs; inspect before presentation correction")
    before = hashlib.sha256(report.read_bytes()).hexdigest()
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text(original.replace(old, new))
    temporary.replace(report)
    override = frozen.read(root / "execution_policy_override.json")
    summary = {"kind": "user-authorized scheduling-only amendment", "approved_utc": override["approved_utc"],
               "max_concurrent_local_training_processes": 2,
               "scientific_config_unchanged": True, "new_models_trained_sequentially": True,
               "override_sha256": digest_file(root / "execution_policy_override.json"),
               "launcher_sha256": digest_file(__file__)}
    for name in ("cp2-phase-c-combinations-results.json", "cp2-phase-c-combinations-validation.json"):
        path = ROOT / "reports" / name
        value = frozen.read(path)
        value["execution_policy_amendment"] = summary
        if "validation" in value:
            value["validation"]["execution_policy_amendment"] = summary
        write_json(path, value)
    write_json(evidence, {"annotated_utc": datetime.now(timezone.utc).isoformat(),
               "canonical_final_validation_sha256_unchanged": digest_file(root / "final_validation.json"),
               "report_before_sha256": before, "report_after_sha256": digest_file(report),
               "scientific_metrics_changed": False, "scheduling_amendment": summary})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    # Preserve the original runner, registration and source snapshot. Only this
    # explicitly recorded dispatcher gate is replaced in the live module.
    frozen.wait = approved_gate
    sys.argv = [str(ROOT / "scripts/run_cp2_combinations.py"), "--root", str(root),
                "--stage", "queue", "--device", args.device]
    if args.resume:
        sys.argv.append("--resume")
    frozen.main()
    annotate_completed_report(root)


if __name__ == "__main__":
    main()
