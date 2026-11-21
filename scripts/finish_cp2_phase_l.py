"""CPU watcher: finish reports/QA after the single loss-screening trainer exits."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import write_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / ".completion.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    target = root / "completion_checks.json"
    if target.exists():
        raise FileExistsError("Preserve completed post-training evidence")
    while True:
        status = read(root / "train_workflow.json")
        if status["status"] == "failed":
            write_json(root / "completion_blocker.json", {
                "status": "NOT COMPLETED — TRAINING/EVALUATION FAILURE", "workflow": status})
            raise RuntimeError("Training workflow failed; no silent retry or configuration change")
        if status["status"] == "completed":
            break
        if not args.wait:
            raise RuntimeError("Training is still running")
        progress = {}
        for name in ("L1", "L2", "L3", "L4"):
            path = root / name / "status.json"
            if path.exists():
                record = read(path)
                progress[name] = {key: record.get(key) for key in ("status", "completed_epochs", "selected_epoch")}
        print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "training": progress}), flush=True)
        time.sleep(30)
    pid = read(root / "launch.json")["pid"]
    while Path(f"/proc/{pid}/stat").exists():
        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
            break
        time.sleep(2)
    # Keep the venv path: resolving its symlink would bypass pyvenv.cfg.
    train_python = str(ROOT / ".venv-training/bin/python")
    analysis_python = str(ROOT / ".venv-analysis/bin/python")
    subprocess.run([analysis_python, "scripts/report_cp2_phase_l.py", "--root", str(root)], cwd=ROOT, check=True)
    subprocess.run([train_python, "scripts/validate_cp2_phase_l.py", "--root", str(root),
                    "--inference-device", "cuda:0"], cwd=ROOT, check=True)
    samples = [json.loads(line) for line in (root / "train_gpu_samples.jsonl").read_text().splitlines()]
    resources = {"scope": "One sequential GPU trainer; unrelated workloads preserved",
                 "batch1_inference": read(root / "batch1_inference.json"), "objectives": {}}
    interruptions = read(root / "resource_interruptions.json")["events"] if (root / "resource_interruptions.json").exists() else []
    resources["resource_interruptions"] = interruptions
    for name in ("L1", "L2", "L3", "L4"):
        history = read(root / name / "history.json")
        status = read(root / name / "status.json")
        started = datetime.fromisoformat(read(root / name / "environment.json")["created_at_utc"])
        finished = datetime.fromisoformat(status["finished_utc"])
        scoped = [r for r in samples if started <= datetime.fromisoformat(r["utc"]) <= finished and r["gpu_csv"]]
        numeric = [[float(v.strip()) for v in row["gpu_csv"].split(",")[2:]] for row in scoped]
        resources["objectives"][name] = {
            "full_invocation_seconds": status["seconds_this_invocation"],
            "invocation_scope": "last resumed invocation" if any(e['objective_id'] == name for e in interruptions) else "single uninterrupted invocation",
            "completed_epoch_compute_seconds_sum": sum(r["epoch_compute_seconds"] for r in history),
            "completed_epoch_with_checkpoint_seconds_sum": sum(r.get("epoch_with_checkpoint_seconds", r["epoch_compute_seconds"]) for r in history),
            "environment_record_to_finish_seconds": (finished - started).total_seconds(),
            "epoch_seconds_median": float(np.median([r["epoch_compute_seconds"] for r in history])),
            "training_step_seconds_mean": float(np.mean([r["step_seconds_mean"] for r in history])),
            "peak_cuda_allocated_bytes": status["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": status["peak_cuda_reserved_bytes"],
            "sampled_gpu_overall_peak_used_MiB": max(v[0] for v in numeric),
            "sampled_gpu_utilization_mean_percent": float(np.mean([v[2] for v in numeric])),
            "gpu_overall_samples": len(numeric),
            "caveat": "GPU-wide sampling includes other processes. For resumed stages, allocator peaks cover the final invocation; failed-session peak was not saved. Environment-to-finish includes interruption pause."}
    write_json(root / "resources.json", resources)
    write_json(ROOT / "reports/cp2-phase-l-resources.json", resources)
    result = read(ROOT / "reports/cp2-phase-l-results.json")
    lines = ["", "## Angka error dan kontras objective", ""]
    for split in ("validation", "test"):
        baseline = result["metrics"]["L0"][split]
        lines += [f"### {split}", "",
                  "| Objective | Δ Image Dice vs L0 (poin) | Δ small Dice (poin) | Δ Boundary IoU (poin) | GT positif TP=0 | GT kosong dengan FP | FP pixel pada GT kosong |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name in ("L0", "L1", "L2", "L3", "L4"):
            m = result["metrics"][name][split]
            lines.append(f"| {name} | {100*(m['all']['image_based']['dice']-baseline['all']['image_based']['dice']):+.3f} | "
                         f"{100*(m['subgroups']['small']['image_based']['dice']-baseline['subgroups']['small']['image_based']['dice']):+.3f} | "
                         f"{100*(m['all']['image_based']['boundary_iou']-baseline['all']['image_based']['boundary_iou']):+.3f} | "
                         f"{m['gt_positive']['total_foreground_misses']} | "
                         f"{m['gt_empty']['images_with_prediction']}/{m['gt_empty']['images']} | {m['gt_empty']['counts']['fp']} |")
        lines.append("")
    lines += ["Kontras terisolasi dalam screen: L2−L1 mengubah eksponen; L3−L0 menambah boundary pada B1; L4−L2 menambah boundary pada Focal Tversky. "
              "Interpretasi mencakup konsekuensi scheduler/early stopping adaptif. Tidak ada atribusi ke loss saja pada fixed update budget yang belum diuji.", "",
              "## Biaya aktual dan QA akhir", "",
              "| Objective | Median epoch (s) | Mean training step (s) | Batch-1 median / P95 (ms) | Sampled overall GPU peak (MiB) |",
              "|---|---:|---:|---:|---:|"]
    for name, values in resources["objectives"].items():
        inference = resources["batch1_inference"][name]
        lines.append(f"| {name} | {values['epoch_seconds_median']:.2f} | {values['training_step_seconds_mean']:.3f} | "
                     f"{inference['latency_median_ms']:.2f} / {inference['latency_p95_ms']:.2f} | {values['sampled_gpu_overall_peak_used_MiB']:.0f} |")
    qa = read(root / "final_validation.json")
    lines += ["", f"QA memverifikasi **{qa['historical_and_user_files_unchanged']:,} file historis/pekerjaan pengguna tetap utuh** dan "
              f"**{qa['new_predictions_verified']:,} prediksi baru** beserta checksum, GT, threshold, per-image CSV, region/boundary/subgroup metrics dan checkpoint metadata. "
              "Selection dihitung ulang dari validation dan mendahului seluruh test probabilities baru. Tidak ada RGB-identical group overlap; independensi pasien tidak dapat dibuktikan.", "",
              "[Resource JSON](cp2-phase-l-resources.json) · [QA JSON](cp2-phase-l-validation.json). "
              "Concurrency tetap satu job sampai profiling terkontrol; tidak menaikkan batch size atau mengaktifkan AMP untuk mengisi VRAM."]
    if (root / "launch_failure.json").exists():
        lines += ["", "Satu launcher awal salah meresolusikan symlink interpreter virtualenv sehingga import cv2 gagal sebelum model/training dimulai. "
                  "Log gagal dipertahankan, path virtualenv diperbaiki, tanpa install/upgrade atau perubahan konfigurasi. "
                  "Isolated reference oracle awal juga membutuhkan alias typing.Set; perbaikannya tidak mengubah perhitungan loss. "
                  "Semua smoke/training berikutnya memakai environment yang benar."]
    if interruptions:
        lines += ["", "Training mengalami interupsi resource: " + "; ".join(
            f"{e['objective_id']} sesudah epoch {e['completed_epoch_boundary']}: shared memory DataLoader /dev/shm ENOSPC" for e in interruptions) +
            ". Ruang shared memory pulih sebelum resume. Optimizer, scheduler, RNG dan checkpoint identity diverifikasi; "
            "partial epoch dibuang lalu dimulai ulang dari akhir epoch lengkap. Worker count, data, loss, batch size, precision dan environment tidak diubah; "
            "completed L1 tidak dilatih ulang. Tidak ada shared-memory file milik pengguna lain dihapus atau mount diubah.", "",
            "Untuk stage yang di-resume, waktu invocation dan peak allocator pada tabel berasal dari invocation terakhir. "
            "Resource JSON juga memuat jumlah waktu semua completed epochs; itu belum mencakup waktu partial epoch yang gagal. "
            "Interval environment-record sampai selesai mencakup jeda diagnostik. Log GPU keseluruhan tetap append-only untuk seluruh percobaan. "
            "Jangan menganggap timing/peak tersebut pengukuran lengkap semua invocation atau benchmark GPU eksklusif."]
    supplement = root / "report_supplement.md"
    if supplement.exists():
        raise FileExistsError("Preserve completed post-training interpretation")
    temporary = supplement.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(supplement)
    with (ROOT / "reports/cp2-phase-l.md").open("a") as stream:
        stream.write("\n".join(lines) + "\n")
    write_json(target, {"status": "COMPLETED — PHASE L.1 TRAINING AND QA",
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "validation": "passed", "report": "reports/cp2-phase-l.md",
                        "visual_review": "pending direct visual inspection",
                        "L5_L7_architecture_external_or_multi_seed_executed": False})
    print("Completed actual L1-L4 training, evaluation and preservation QA", flush=True)


if __name__ == "__main__":
    main()
