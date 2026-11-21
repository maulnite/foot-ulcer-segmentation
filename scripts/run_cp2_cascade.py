"""Prepare, validate and queue one clean-split Ganta-design refiner training."""

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_cascade import (  # noqa: E402
    CropTraining, cache_validation, check_run, decision, evaluate, mapped_model,
    namespace, prepare, preserve, read, refine, sources, train, utc,
)
from foot_ulcer_segmentation.cp2_external import verify_dataset  # noqa: E402
from foot_ulcer_segmentation.cp2_train import gpu_snapshot, make_cp2_loader  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402
from foot_ulcer_segmentation.reference import build_model  # noqa: E402


def preflight(root, config, manifest):
    if (root / "preflight.json").exists():
        raise FileExistsError("Preflight already exists; preserve previous evidence")
    setup(42, config["cpu_threads"])
    ns = namespace(root / "partner_source.py")
    c = config["cascade"]
    names = {"PROP_THR": "proposal_threshold", "MAX_COMPONENTS": "max_components",
             "MIN_COMPONENT_PX": "min_component_pixels", "CROP_SIZE": "crop_size",
             "MIN_BOX_FRAC": "minimum_box_fraction", "CROP_MARGIN": "train_crop_margin",
             "CROP_MARGIN_EVAL": "eval_crop_margin", "P_RANDOM_CROP": "random_crop_probability"}
    for original, configured in names.items():
        actual = list(ns[original]) if isinstance(ns[original], tuple) else ns[original]
        if actual != c[configured]:
            raise ValueError(f"Original Ganta rule changed: {original}")
    state = torch.load(ROOT / c["control_root"] / "initial_weights.pth", map_location="cpu")["state_dict"]
    model, proof = mapped_model(ns, state)
    source = build_model(config, False)
    source.load_state_dict(state, strict=True)
    source.eval()
    model.eval()
    loader, _ = make_cp2_loader(config, manifest, "train", smoke=True)
    images, _ = next(iter(loader))
    with torch.no_grad():
        left, right = source(images), model(images)
    if not torch.equal(left, right):
        raise ValueError("Mapped partner and canonical initial forwards differ")
    proof["real_training_input_initial_forward_bit_exact"] = True
    proof["real_images_checked"] = len(images)
    # Differential inference test: actual original partner function vs cached-
    # coarse implementation; uses synthetic structural fixtures, no ML score.
    class FixedCoarse(torch.nn.Module):
        def forward(self, x):
            output = torch.full((1, 1, 224, 224), -5.)
            output[:, :, 25:43, 35:55] = 2.
            output[:, :, 170:180, 170:190] = 2.
            return output
    coarse = FixedCoarse()
    with torch.no_grad():
        original, expected, n = ns["cascade_predict"](coarse, model, images[:1])
        actual, boxes = refine(model, images[:1], original, ns, torch.device("cpu"))
    if not np.array_equal(actual, expected) or len(boxes) != n:
        raise ValueError("Cascade inference differs from original partner rule")
    cropped = CropTraining(manifest["splits"]["train"], ns)
    checks = []
    for empty in (True, False):
        i = next(i for i, r in enumerate(cropped.rows) if (r["gt_pixels"] == 0) == empty)
        image, mask = cropped[i]
        checks.append({"original_empty": empty, "input_shape": list(image.shape),
                       "mask_shape": list(mask.shape), "finite": bool(np.isfinite(image).all()),
                       "binary_mask": bool(np.isin(mask, [0, 1]).all())})
    write_json(root / "architecture_preflight.json", {"initialization": proof,
               "original_inference_rule_bit_exact": True, "actual_training_crop_checks": checks,
               "pretraining_forward_only": True, "separate_smoke_training": False})
    del source, model, state, images, cropped
    gc.collect()
    # Cache only source-validation coarse outputs before training/model selection.
    cache_validation(root, config, manifest)
    target = read(ROOT / c["external_root"] / "dataset_manifest.json")
    if target["fingerprint"] != c["external_fingerprint"] or len(target["eligible"]) != 558:
        raise ValueError("Wrong external cohort")
    verify_dataset(target)
    with (root / "tests.log").open("x") as stream:
        subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "-m", "pytest", "-o", "addopts=", "-q",
                        f"--junitxml={root / 'tests.xml'}"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    suites = list(ET.parse(root / "tests.xml").getroot().iter("testsuite"))
    tests = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "errors", "failures", "skipped")}
    dependencies = {}
    for env in (".venv-training", ".venv-xfuseg"):
        result = subprocess.run([str(ROOT / env / "bin/python"), "-m", "pip", "check"],
                                capture_output=True, text=True, check=True)
        dependencies[env] = result.stdout.strip()
    result = {"status": "passed", "created_utc": utc(), "tests": tests,
              "preservation": preserve(root), "dependencies": dependencies,
              "source_membership": {s: len(r) for s, r in manifest["splits"].items()},
              "external_images": 558, "sources": sources(), "config_sha256": digest_json(config),
              "standalone_smoke_training": False, "ruff": "NOT RUN — unavailable; environments unchanged"}
    write_json(root / "preflight.json", result)
    inputs = read(root / "input_inventory.json")
    inputs[str((root / "validation_coarse/manifest.json").relative_to(ROOT))] = digest_file(root / "validation_coarse/manifest.json")
    for name in ("dataset_manifest.json", "input_cache.json"):
        path = ROOT / c["external_root"] / name
        inputs[str(path.relative_to(ROOT))] = digest_file(path)
    # The external utility is reused unchanged and must also stay pinned.
    inputs["src/foot_ulcer_segmentation/cp2_external.py"] = digest_file(ROOT / "src/foot_ulcer_segmentation/cp2_external.py")
    write_json(root / "preregistration.json", {"registered_utc": utc(), "sources": sources(),
               "config_sha256": digest_json(config), "inputs": inputs,
               "new_training_runs": 1, "coarse_retrained": False,
               "source_test_or_CO2_used_for_configuration_selection": False,
               "clinical_decision_rule": c["clinical_selection"]})
    for name in sources():
        destination = root / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    return result


def wait_for_phase_c(root, config):
    phase_c = ROOT / config["cascade"]["wait_for_phase_c"]
    while True:
        check_run(root)
        workflow = read(phase_c / "train_workflow.json")
        if workflow["status"] == "failed":
            raise RuntimeError("Phase C failed; cascade training not started; inspect before explicit resume")
        if workflow["status"] == "completed":
            break
        pid = workflow["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError("Phase C disappeared without completed workflow; cascade remains unstarted") from exc
        write_json(root / "workflow.json", {"status": "queued", "pid": os.getpid(),
                   "waiting_for": str(phase_c.relative_to(ROOT)), "phase_c_pid": pid,
                   "updated_utc": utc(), "training_started": False})
        time.sleep(30)
    # Check current global memory without altering any other user's jobs.
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    if int(result.stdout.splitlines()[0].strip()) < 8192:
        raise RuntimeError("Less than 8 GiB free VRAM; training not started")


def workflow(root, device, resume=False):
    config, _ = check_run(root)
    if (root / "workflow.json").exists() and read(root / "workflow.json")["status"] == "completed":
        print("CACHED completed cascade workflow", flush=True)
        return
    wait_for_phase_c(root, config)
    stop = threading.Event()

    def monitor():
        with (root / "gpu_samples.jsonl").open("a") as stream:
            while not stop.is_set():
                stream.write(json.dumps(gpu_snapshot()) + "\n")
                stream.flush()
                stop.wait(30)
    threading.Thread(target=monitor, daemon=True).start()
    try:
        write_json(root / "workflow.json", {"status": "training", "pid": os.getpid(), "started_utc": utc()})
        train(root, device, resume=resume)
        # Release CUDA before CPU inference, independently frozen from training.
        if torch.cuda.is_initialized():
            gc.collect()
            torch.cuda.empty_cache()
        torch.set_num_threads(config["cascade"]["inference_threads"])
        validation = evaluate(root, "validation")
        selection = root / "selection.json"
        if not selection.exists():
            selected = decision(validation["metrics"]["coarse"], validation["metrics"]["cascade"])
            write_json(selection, {"locked_utc": utc(), **selected,
                       "training_validation_image_dice": read(root / "refiner/status.json")["selected_validation_image_dice"],
                       "inference_cpu_validation_image_dice": validation["metrics"]["cascade"]["all"]["image_based"]["dice"]})
        for split in ("test", "CO2Wounds"):
            write_json(root / "workflow.json", {"status": "evaluating", "pid": os.getpid(), "split": split, "updated_utc": utc()})
            evaluate(root, split)
        with (root / "report.log").open("a") as stream:
            subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "-u", str(ROOT / "scripts/report_cp2_cascade.py"),
                            "--root", str(root)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        write_json(root / "workflow.json", {"status": "completed", "finished_utc": utc(),
                   "new_training_runs": 1, "coarse_retrained": False, "visual_review": "PENDING"})
    finally:
        stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cp2/cascade_clean.json")
    parser.add_argument("--stage", choices=("prepare", "queue", "status"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.stage == "prepare":
                config = read(args.config)
                manifest = prepare(root, config)
                print(json.dumps(preflight(root, config, manifest)), flush=True)
            elif args.stage == "queue":
                workflow(root, args.device, args.resume)
            else:
                print(json.dumps(read(root / "workflow.json"), indent=2))
        except Exception:
            write_json(root / f"failure-{time.time_ns()}.json", {"stage": args.stage, "utc": utc(), "traceback": traceback.format_exc()})
            if args.stage == "queue":
                write_json(root / "workflow.json", {"status": "failed", "pid": os.getpid(), "utc": utc(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
