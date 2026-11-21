"""Finish phase-P QA/resources after the single sequential trainer exits; never retrain."""

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
        raise FileExistsError("Preserve completed QA evidence")
    while True:
        state = read(root / "train_workflow.json")
        if state["status"] == "failed":
            write_json(root / "completion_blocker.json", {"status": "NOT COMPLETED — TRAINING FAILURE", "training": state})
            raise RuntimeError("Training failed; no retry or configuration change was made")
        if state["status"] == "completed":
            break
        if not args.wait:
            raise RuntimeError("Training still running")
        # CPU-only watcher; one GPU trainer still owns the sequential B0/B1 queue.
        print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "training": "running",
                          "progress": {n: read(root / n / "status.json").get("completed_epochs", 0)
                                       for n in ("B0", "B1") if (root / n / "status.json").exists()}}), flush=True)
        time.sleep(30)
    pid = read(root / "launch.json")["pid"]
    while Path(f"/proc/{pid}/stat").exists():
        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
            break
        time.sleep(2)
    python = ROOT / ".venv-training/bin/python"
    if not (root / "batch1_inference.json").exists():
        subprocess.run([str(python), "scripts/benchmark_cp2_inference.py", "--root", str(root)], cwd=ROOT, check=True)
    subprocess.run([str(python), "scripts/validate_cp2_phase_p.py", "--root", str(root)], cwd=ROOT, check=True)
    resources = {"scope": "One sequential GPU training process; existing unrelated GPU jobs retained",
                 "architecture": read(root / "architecture_verification.json"),
                 "batch1_inference": read(root / "batch1_inference.json"), "baselines": {}}
    gpu = [json.loads(line) for line in (root / "train_gpu_samples.jsonl").read_text().splitlines()]
    for name in ("B0", "B1"):
        history = read(root / name / "history.json")
        status = read(root / name / "status.json")
        started = datetime.fromisoformat(read(root / name / "environment.json")["created_at_utc"])
        finished = datetime.fromisoformat(status["finished_utc"])
        samples = [r for r in gpu if started <= datetime.fromisoformat(r["utc"]) <= finished]
        values = [[float(v.strip()) for v in row["gpu_csv"].split(",")[2:]] for row in samples if row["gpu_csv"]]
        resources["baselines"][name] = {
            "training_wall_seconds": (finished - started).total_seconds(),
            "epoch_seconds_median": float(np.median([r["epoch_compute_seconds"] for r in history])),
            "training_step_seconds_mean": float(np.mean([r["step_seconds_mean"] for r in history])),
            "peak_cuda_allocated_bytes": status["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": status["peak_cuda_reserved_bytes"],
            "gpu_overall_peak_used_MiB": max(v[0] for v in values),
            "gpu_overall_utilization_mean_percent": float(np.mean([v[2] for v in values])),
            "gpu_overall_samples": len(values),
            "gpu_overall_caveat": "Includes unrelated processes; sampled every 5 seconds, not our allocator footprint"}
    write_json(root / "resources.json", resources)
    write_json(ROOT / "reports/cp2-phase-p-resources.json", resources)
    results = read(ROOT / "reports/cp2-phase-p-results.json")
    lines = ["", "## Interpretasi hasil dan pemeriksaan akhir", ""]
    for split in ("validation", "test"):
        a, b = results["metrics"]["B0"][split], results["metrics"]["B1"][split]
        delta = b["all"]["image_based"]["dice"] - a["all"]["image_based"]["dice"]
        small_delta = b["subgroups"]["small"]["image_based"]["dice"] - a["subgroups"]["small"]["image_based"]["dice"]
        lines += [f"**{split}:** B1−B0 mean Image Dice = **{delta*100:+.2f} poin persentase**; khusus luka kecil {small_delta*100:+.2f} poin. "
                  f"GT positif dengan TP=0: B0 {a['gt_positive']['total_foreground_misses']}, B1 {b['gt_positive']['total_foreground_misses']}. "
                  f"GT kosong dengan prediksi foreground: B0 {a['gt_empty']['images_with_prediction']}/{a['gt_empty']['images']}, "
                  f"B1 {b['gt_empty']['images_with_prediction']}/{b['gt_empty']['images']}.", ""]
    lines += ["Koreksi input focal mempunyai arah/dampak seperti angka di atas pada satu seed dan protokol ini. "
              "Hasil tidak menjamin efek yang sama pada seed, split, pasien atau dataset lain. Tidak ada pemilihan threshold maupun checkpoint dari source-test.", "",
              "| Model | Median epoch (detik) | Mean training step (detik) | Batch-1 inference median / P95 (ms) | Peak pemakaian GPU keseluruhan (MiB) |", "|---|---:|---:|---:|---:|"]
    for name in ("B0", "B1"):
        r, latency = resources["baselines"][name], resources["batch1_inference"][name]
        lines.append(f"| {name} | {r['epoch_seconds_median']:.2f} | {r['training_step_seconds_mean']:.3f} | "
                     f"{latency['latency_median_ms']:.2f} / {latency['latency_p95_ms']:.2f} | {r['gpu_overall_peak_used_MiB']:.0f} |")
    verified = read(root / "final_validation.json")
    lines += ["", f"Validasi akhir memverifikasi **{verified['original_cp1_and_user_files_verified']:,} file CP1/pekerjaan pengguna** tanpa perubahan. "
              "Seluruh probabilitas diekspor diperiksa ulang terhadap checksum, GT asli, threshold, confusion counts, region/boundary metrics dan subgroups. "
              "Epoch checkpoint sama dengan earliest maximum Image Dice validation. Seluruh common-epoch inputs B0/B1 identik.", "",
              "Active forward parameters **64.898.248**, registered **70.968.939**; selisih adalah modul source yang tidak dipakai forward. "
              "Batch-1 latency memakai satu foto validation yang sama, lima warm-up dan 30 synchronized forward+sigmoid steps, tidak termasuk loading/export. "
              "Angka biaya berasal dari GPU bersama dan tidak mengukur kapasitas concurrency eksklusif. Rekomendasi tetap satu job sampai profiling concurrency terkontrol dilakukan.", "",
              "[QA akhir](cp2-phase-p-validation.json) · [Resources](cp2-phase-p-resources.json). Seluruh kegagalan pemeriksaan awal dan perbaikannya dicatat pada [preflight](cp2-phase-p-preflight.json); tidak ada perubahan objective/config diam-diam selama baseline."]
    report = ROOT / "reports/cp2-phase-p.md"
    supplement = root / "report_supplement.md"
    if supplement.exists():
        raise FileExistsError("Preserve the recorded post-training interpretation")
    temporary = supplement.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(supplement)
    with report.open("a") as stream:
        stream.write("\n".join(lines) + "\n")
    write_json(target, {"status": "COMPLETED — PHASE P B0/B1 TRAINING AND QA", "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "validation": "passed", "report": "reports/cp2-phase-p.md",
                        "phases_L_A_H_external_executed": False, "visual_review": "pending manual inspection of generated figures"})
    print("Completed phase P training, evaluation, resources and CP1 preservation QA", flush=True)


if __name__ == "__main__":
    main()
