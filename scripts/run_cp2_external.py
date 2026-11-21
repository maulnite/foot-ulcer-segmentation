"""One coordinated CPU workflow for frozen CP1/CP2 external inference."""

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_external import (  # noqa: E402
    binary_mask, buffers_fingerprint, load_frozen_model, predict_primary,
    prepare_dataset, read, restore_probability, verify_dataset,
)
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator  # noqa: E402


def runtime_sources():
    paths = ["scripts/run_cp2_external.py", "src/foot_ulcer_segmentation/cp2_external.py",
             "src/foot_ulcer_segmentation/reference.py", "src/foot_ulcer_segmentation/cp2_attention.py",
             "src/foot_ulcer_segmentation/cp2_heads.py", "src/foot_ulcer_segmentation/cp2_protocol.py",
             "src/foot_ulcer_segmentation/evaluation.py", "scripts/pfom_metrics.py",
             "configs/cp2/phase_a.json"]
    return {path: digest_file(ROOT / path) for path in paths}


def check_identity(run):
    if read(run / "execution_sources.json") != runtime_sources():
        raise ValueError("Frozen execution code changed")
    protocol = read(run / "preregistration.json")
    if digest_file(run / "models.json") != protocol["frozen_models_manifest_sha256"]:
        raise ValueError("Frozen model roster changed")
    for name, expected in read(run / "execution_inputs.json").items():
        if digest_file(run / name) != expected:
            raise ValueError(f"Frozen target geometry or protocol changed: {name}")


def save_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def worker(run, model_id, smoke=False, threads=4):
    check_identity(run)
    manifest = read(run / "dataset_manifest.json")
    verify_dataset(manifest)
    record = next(value for value in read(run / "models.json") if value["id"] == model_id)
    output = run / ("preflight" if smoke else "models") / model_id
    identity = digest_json({"model": record, "dataset": manifest["fingerprint"],
                            "sources": runtime_sources(), "mode": "smoke" if smoke else "full"})
    if (output / "result.json").exists():
        result = read(output / "result.json")
        if result["identity"] != identity or result["status"] != "completed":
            raise ValueError("Completed model output identity changed")
        if not smoke:
            for row in read(output / "prediction_manifest.json"):
                if digest_file(output / row["file"]) != row["sha256"]:
                    raise ValueError("Completed probability cache changed")
        print(f"CACHED {model_id}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    cv2.setNumThreads(1)
    model, model_proof = load_frozen_model(record)
    before = buffers_fingerprint(model)
    cache = {row["sample_id"]: row for row in read(run / "input_cache.json")}
    samples = manifest["eligible"]
    if smoke:
        samples = samples[:2]
    settings = {"prediction_threshold": .5, "boundary_radius_diagonal_fraction": .01}
    evaluator = CP2Evaluator(settings, boundaries=True)
    prediction_rows, inference_seconds = [], []
    for index, row in enumerate(samples):
        item = cache[row["sample_id"]]
        path = run / item["file"]
        if digest_file(path) != item["sha256"]:
            raise ValueError("Input tensor cache changed")
        inputs = torch.from_numpy(np.load(path, allow_pickle=False)[None])
        t0 = time.monotonic()
        probability = predict_primary(model, inputs, record["outputs_probabilities"])[0]
        inference_seconds.append(time.monotonic() - t0)
        if smoke:
            repeated = predict_primary(model, inputs, record["outputs_probabilities"])[0]
            if not np.array_equal(repeated, probability):
                raise ValueError("CPU checkpoint replay is not deterministic")
        else:
            native = restore_probability(probability, item["geometry"])
            truth = binary_mask(cv2.imread(str(ROOT / row["mask_path"]), cv2.IMREAD_UNCHANGED))
            evaluator.add(row, truth, native)
            destination = output / "probabilities" / (row["sample_id"] + ".npz")
            destination.parent.mkdir(exist_ok=True)
            np.savez_compressed(destination, probability=probability)
            prediction_rows.append({"sample_id": row["sample_id"],
                                    "file": str(destination.relative_to(output)),
                                    "sha256": digest_file(destination)})
        if (index + 1) % 50 == 0 or index == len(samples) - 1:
            write_json(output / "status.json", {"status": "running", "model": model_id,
                       "images_done": index + 1, "images_total": len(samples),
                       "seconds": time.monotonic() - started})
            print(f"{model_id} {index + 1}/{len(samples)} {time.monotonic() - started:.1f}s", flush=True)
    if buffers_fingerprint(model) != before:
        raise ValueError("Inference updated frozen BatchNorm buffers")
    if not smoke:
        save_csv(output / "per_image.csv", evaluator.rows)
        write_json(output / "prediction_manifest.json", prediction_rows)
    result = {"status": "completed", "identity": identity, "model": record,
              "model_proof": model_proof, "mode": "smoke" if smoke else "full",
              "dataset_fingerprint": manifest["fingerprint"], "images": len(samples),
              "buffers_unchanged": True, "parameters_frozen": not any(p.requires_grad for p in model.parameters()),
              "training_performed": False, "torch": torch.__version__, "device": "cpu",
              "threads": threads, "batch_size": 1, "precision": "float32",
              "mean_forward_ms_shared_cpu": float(np.mean(inference_seconds) * 1000),
              "seconds": time.monotonic() - started,
              "metrics": None if smoke else evaluator.summary()}
    write_json(output / "result.json", result)
    write_json(output / "status.json", {"status": "completed", "images": len(samples)})
    del model
    gc.collect()


def dispatch(run, smoke, workers, threads):
    check_identity(run)
    stage = "preflight" if smoke else "models"
    if not smoke and not read(run / "preflight.json")["passed"]:
        raise ValueError("All checkpoint smoke tests must pass before full inference")
    ids = [record["id"] for record in read(run / "models.json")]
    queue, active, finished, failures = list(ids), [], [], []
    logs = run / "logs"
    logs.mkdir(exist_ok=True)
    while queue or active:
        while queue and len(active) < workers:
            model_id = queue.pop(0)
            if (run / stage / model_id / "result.json").exists():
                worker(run, model_id, smoke, threads)
                finished.append(model_id)
                continue
            log_path = logs / f"{stage}-{model_id}.log"
            if log_path.exists():
                raise FileExistsError("Inspect previous failed worker before explicit new attempt")
            stream = log_path.open("x")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run", str(run),
                       "--stage", "worker", "--model", model_id, "--threads", str(threads)]
            if smoke:
                command.append("--smoke")
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(threads),
                           "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1"}
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            active.append((model_id, process, stream))
        for model_id, process, stream in list(active):
            code = process.poll()
            if code is not None:
                stream.close()
                active.remove((model_id, process, stream))
                (finished if code == 0 else failures).append(model_id)
                print(f"{stage}: {model_id} {'completed' if code == 0 else 'FAILED'}; {len(finished)}/{len(ids)}", flush=True)
        write_json(run / (stage + "_workflow.json"), {"status": "running" if active or queue else "completed" if not failures else "failed",
                   "finished": finished, "failed": failures,
                   "active": [{"model": name, "pid": process.pid} for name, process, _ in active],
                   "queued": queue, "workers": workers, "threads_per_worker": threads})
        if active:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"Worker failures: {failures}; see persistent logs")
    if smoke:
        write_json(run / "preflight.json", {"passed": True, "models": finished,
                   "checks": "strict state loading, finite probabilities, exact repeated CPU predictions, frozen parameters and BatchNorm"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "preflight", "infer", "worker"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.threads <= 4:
        parser.error("CPU budget is at most four workers × four threads")
    run = args.run.resolve()
    if args.stage == "prepare":
        if (run / "preflight.json").exists() or (run / "models").exists():
            check_identity(run)
            raise FileExistsError("Completed preparation is frozen; use existing inference stages")
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(1)
        # Resource amendment is frozen before any target prediction. No scientific settings change.
        write_json(run / "execution_policy.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                   "amends": "resource-only sequential-model field in original preregistration",
                   "reason": "GPU busy; CPU server has 256 logical cores and >1 TiB available RAM",
                   "workers": args.workers, "threads_per_worker": args.threads,
                   "max_total_cpu_threads": args.workers * args.threads,
                   "gpu_used": False, "batch_size": 1, "precision": "float32",
                   "data_exclusions": "exclude source exact overlaps and conflicting target masks; one representative of identical RGB+mask",
                   "no_changes": "frozen model roster, checkpoint, threshold, geometry, normalization, metrics"})
        manifest = prepare_dataset(run)
        write_json(run / "execution_sources.json", runtime_sources())
        write_json(run / "execution_inputs.json", {
            name: digest_file(run / name) for name in ["models.json", "preregistration.json",
            "execution_policy.json", "dataset_manifest.json", "input_cache.json"]})
        print(json.dumps({"eligible": len(manifest["eligible"]), "excluded": len(manifest["excluded"]),
                          "fingerprint": manifest["fingerprint"], "empty_masks": manifest["empty_masks"]}, indent=2))
    elif args.stage == "worker":
        worker(run, args.model, args.smoke, args.threads)
    else:
        dispatch(run, args.stage == "preflight", args.workers, args.threads)


if __name__ == "__main__":
    main()
