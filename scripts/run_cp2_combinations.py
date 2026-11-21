"""Queue exactly B1+A6+H2 then L3+A6+H2; preserve all previous runs."""

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

import torch  # noqa: E402
from foot_ulcer_segmentation.common import digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_attention import model_profile  # noqa: E402
from foot_ulcer_segmentation.cp2_combinations import (  # noqa: E402
    CACHED, FRESH, build, contrasts, evaluate, histories, prepare, select, sources, train,
)
from foot_ulcer_segmentation.cp2_heads import build_head_model  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_c import pairing  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_n import preservation, read, utc  # noqa: E402
from foot_ulcer_segmentation.cp2_train import gpu_snapshot  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402


def architecture(root, config):
    setup(config["seed"], config["cpu_threads"])
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    fixture = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    profiles, proofs, outputs, mask_only = {}, {}, {}, {}
    for name in FRESH:
        model, objective, proofs[name] = build(config, name, initial)
        profiles[name] = model_profile(model, fixture)
        profiles[name].pop("decoder_calls")
        mask_only[name] = model_profile(model.base, fixture)
        mask_only[name].pop("decoder_calls")
        model.eval()
        with torch.no_grad():
            out, bare = model(fixture), model(fixture, auxiliary=False)
        if not torch.equal(out.main_logits, bare.main_logits):
            raise ValueError("Head removal changes primary output")
        if list(out.auxiliary_logits.shape) != [1, 1, 224, 224]:
            raise ValueError("Wrong H2 tensor shape")
        if model.base.decoder.blocks[3]._forward_hooks:
            raise ValueError("Read-only hook leaked")
        outputs[name] = (out.main_logits.clone(), out.auxiliary_logits.clone())
        del model, objective
    if proofs[FRESH[0]] != proofs[FRESH[1]]:
        raise ValueError("Two initial model states are not identical")
    if not all(torch.equal(a, b) for a, b in zip(outputs[FRESH[0]], outputs[FRESH[1]])):
        raise ValueError("Paired model main/aux initial forward differs")
    for name in ("B1", "H2"):
        model, _ = build_head_model(config, "H2" if name == "H2" else "H0", initial)
        profiles[name] = model_profile(model, fixture)
        profiles[name].pop("decoder_calls")
        del model
    profiles["L3"] = profiles["B1"]
    profiles["A6"] = mask_only[FRESH[0]]
    if any(profiles[n]["active_forward_parameters"] != profiles["A6"]["active_forward_parameters"] + 33 for n in FRESH):
        raise ValueError("Unexpected auxiliary parameter count")
    result = {"status": "passed", "profiles": profiles, "mask_only_profiles": mask_only,
              "initialization": proofs, "complete_initial_states_equal": True,
              "initial_main_and_auxiliary_forward_bit_exact": True,
              "head_removal_primary_bit_exact": True,
              "fixture": "SYNTHETIC linspace structural forward; no separate smoke training or scientific score"}
    write_json(root / "architecture_preflight.json", result)
    del initial
    gc.collect()
    return result


def preflight(root, config):
    gate = root / "pretraining_checks.json"
    if gate.exists():
        raise FileExistsError("Preflight already exists; preserve prior evidence")
    architecture(root, config)
    with (root / "pretraining_tests.log").open("x") as stream:
        subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "-m", "pytest", "-o", "addopts=", "-q",
                        f"--junitxml={root / 'pretraining_tests.xml'}"], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    suites = list(ET.parse(root / "pretraining_tests.xml").getroot().iter("testsuite"))
    tests = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "errors", "failures", "skipped")}
    dependencies = {}
    for env in (".venv-training", ".venv-xfuseg"):
        result = subprocess.run([str(ROOT / env / "bin/python"), "-m", "pip", "check"],
                                capture_output=True, text=True, check=True)
        dependencies[env] = result.stdout.strip()
    result = {"status": "passed", "utc": utc(), "tests": tests, "dependency_checks": dependencies,
              "preservation": preservation(root), "sources": sources(), "config_fingerprint": digest_json(config),
              "new_training_runs": 2, "standalone_smoke_training": False,
              "actual_first_step_checks": "encoder, CBAM channel and spatial gates at both stages, H2 head",
              "ruff": "NOT RUN — unavailable; shared environments unchanged"}
    write_json(gate, result)
    return result


def register(root, config):
    gate = read(root / "pretraining_checks.json")
    if gate["status"] != "passed" or gate["sources"] != sources() or gate["config_fingerprint"] != digest_json(config):
        raise ValueError("Passed matching preflight required")
    if (root / "preregistration.json").exists():
        raise FileExistsError("Completed preregistration preserved")
    for name in sources():
        destination = root / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write_json(root / "preregistration.json", {"registered_before_training_utc": utc(), "sources": sources(),
               "config_fingerprint": digest_json(config), "order": list(FRESH), "seed": 42,
               "cached_controls": list(CACHED), "new_training_runs": 2, "standalone_smoke_training": False})


def check(root):
    config = read(root / "protocol.json")
    registration = read(root / "preregistration.json")
    if registration["sources"] != sources() or registration["config_fingerprint"] != digest_json(config):
        raise ValueError("Registered code/config changed")
    manifest = prepare(root, config)
    return config, manifest


def wait(root, config):
    previous = ROOT / config["combinations"]["wait_for"]
    while True:
        state = read(previous / "workflow.json")
        if state["status"] == "completed":
            return
        if state["status"] == "failed":
            raise RuntimeError("Cascade workflow failed; new combinations remain unstarted pending inspection")
        pid = state["pid"]
        proc = Path(f"/proc/{pid}/cmdline")
        if not proc.exists() or b"run_cp2_cascade.py" not in proc.read_bytes():
            raise RuntimeError("Expected cascade process disappeared without completed workflow")
        write_json(root / "workflow.json", {"status": "queued", "pid": os.getpid(), "updated_utc": utc(),
                   "waiting_for": config["combinations"]["wait_for"], "previous_pid": pid,
                   "training_started": False, "order": list(FRESH)})
        time.sleep(30)


def workflow(root, device, resume):
    config, manifest = check(root)
    path = root / "workflow.json"
    if path.exists():
        old = read(path)
        if old["status"] == "completed":
            print("CACHED completed workflow", flush=True)
            return
        if old["status"] not in ("queued",) and not resume:
            raise FileExistsError("Inspect prior workflow before explicit --resume")
    wait(root, config)
    # The queue never creates a CUDA context while another local training runs.
    if not device.startswith("cuda") or not torch.cuda.is_available() or torch.cuda.mem_get_info(torch.device(device))[0] < 6 * 1024**3:
        raise RuntimeError("Need >=6GiB available CUDA memory; no batch/precision/device fallback")
    stop = threading.Event()

    def monitor():
        with (root / "gpu_samples.jsonl").open("a", buffering=1) as stream:
            while not stop.is_set():
                stream.write(json.dumps(gpu_snapshot()) + "\n")
                stop.wait(30)
    threading.Thread(target=monitor, daemon=True).start()
    try:
        for name in FRESH:
            write_json(path, {"status": "training", "pid": os.getpid(), "active_variant": name,
                       "started_or_resumed_utc": utc(), "order": list(FRESH), "new_training_runs": 2})
            train(root, config, manifest, name, device, resume=resume)
            evaluate(root, config, manifest, name, "validation", device)
            gc.collect()
            torch.cuda.empty_cache()
        reports = {n: r["validation"] for n, r in read(root / "control_reference.json").items()}
        reports.update({n: read(root / n / "evaluation-validation.json") for n in FRESH})
        write_json(root / "paired_inputs.json", pairing(histories(root, config, FRESH)))
        selected = select(reports, read(root / "architecture_preflight.json")["profiles"])
        selection_file = root / "selection.json"
        if selection_file.exists():
            if {k: v for k, v in read(selection_file).items() if k != "created_utc"} != selected:
                raise ValueError("Persisted validation-only selection differs")
        else:
            write_json(selection_file, {"created_utc": utc(), **selected})
        write_json(root / "contrasts-validation.json", contrasts(reports))
        for name in FRESH:
            evaluate(root, config, manifest, name, "test", device)
            gc.collect()
            torch.cuda.empty_cache()
        write_json(root / "inference_workflow.json", {"status": "completed", "finished_utc": utc()})
        with (root / "report_console.log").open("a") as stream:
            subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "-u", str(ROOT / "scripts/report_cp2_combinations.py"),
                            "--root", str(root)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        write_json(path, {"status": "completed", "finished_utc": utc(), "new_training_runs": 2,
                   "external_inference": False, "visual_review": "PENDING"})
    finally:
        stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "queue", "status"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cp2/phase_c_combinations.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "status":
        print(json.dumps(read(root / "workflow.json"), indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.stage == "prepare":
                config = read(args.config)
                prepare(root, config)
                result = preflight(root, config)
                register(root, config)
                print(json.dumps(result), flush=True)
            else:
                workflow(root, args.device, args.resume)
        except Exception:
            failure = {"status": "failed", "stage": args.stage, "pid": os.getpid(), "utc": utc(), "traceback": traceback.format_exc()}
            write_json(root / f"failure-{time.time_ns()}.json", failure)
            if args.stage == "queue":
                write_json(root / "workflow.json", failure)
            raise


if __name__ == "__main__":
    main()
