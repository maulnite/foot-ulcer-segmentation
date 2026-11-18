"""One real N1 full training; N0/B1 cached; no standalone smoke training."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_attention import model_profile  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_n import (  # noqa: E402
    check_pairing, evaluate, prepare, preservation, read, sources, train, utc,
)
from foot_ulcer_segmentation.cp2_presence import build_presence_model, select_presence  # noqa: E402
from foot_ulcer_segmentation.cp2_train import gpu_snapshot  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402


def architecture_check(root, config):
    setup(42, config["cpu_threads"])
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    inputs = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    profiles, proofs, values = {}, {}, {}
    for name in ("N0", "N1"):
        model, proofs[name] = build_presence_model(config, initial, auxiliary=name == "N1")
        profiles[name] = model_profile(model, inputs)
        profiles[name].pop("decoder_calls")
        model.eval()
        with torch.no_grad():
            result = model(inputs)
            without_head = model(inputs, auxiliary=False)
        if not torch.equal(result.main_logits, without_head.main_logits):
            raise ValueError("Presence head changes primary mask computation")
        if model.base.decoder.blocks[3]._forward_hooks:
            raise ValueError("Temporary hook leaked")
        values[name] = result.main_logits.clone()
        if name == "N1" and list(result.presence_logits.shape) != [1, 1]:
            raise ValueError("Presence tensor shape differs")
        del model
    if not torch.equal(values["N0"], values["N1"]):
        raise ValueError("Original B1/N1 initial forward differs")
    if profiles["N1"]["active_forward_parameters"] != profiles["N0"]["active_forward_parameters"] + 33:
        raise ValueError("Unexpected active parameter increase")
    result = {"status": "passed", "profiles": profiles, "initialization": proofs,
              "initial_primary_bit_exact": True, "head_removal_preserves_primary": True,
              "training_sources": sources(),
              "fixture": "SYNTHETIC linspace for structural forward checks only; no training or scientific scores"}
    write_json(root / "architecture_preflight.json", result)
    return result


def preflight(root, config):
    gate = root / "pretraining_checks.json"
    if gate.exists():
        result = read(gate)
        if result["training_sources"] != sources() or result["protocol_fingerprint"] != digest_json(config):
            raise ValueError("Prior preflight code/config differs; preserve and use a new run")
        return result
    preserved = preservation(root)
    architecture_check(root, config)
    with (root / "pretraining_tests.log").open("x") as stream:
        subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "-m", "pytest", "-o", "addopts=", "-q",
                        f"--junitxml={root / 'pretraining_tests.xml'}"],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    dependencies = {}
    for environment in (".venv-training", ".venv-xfuseg"):
        result = subprocess.run([str(ROOT / environment / "bin/python"), "-m", "pip", "check"],
                                capture_output=True, text=True, check=True)
        dependencies[environment] = result.stdout.strip()
    suites = list(ET.parse(root / "pretraining_tests.xml").getroot().iter("testsuite"))
    tests = {key: sum(int(suite.get(key, 0)) for suite in suites)
             for key in ("tests", "errors", "failures", "skipped")}
    result = {"status": "passed", "created_utc": utc(), "tests": tests,
              "preservation": preserved, "dependency_checks": dependencies,
              "training_sources": sources(), "protocol_fingerprint": digest_json(config),
              "standalone_smoke_training": False,
              "user_instruction": "Full training one seed without a separate smoke training run",
              "gpu_update_checks": "enforced on first actual full-training optimizer step",
              "ruff": "NOT RUN — unavailable in existing environments"}
    write_json(gate, result)
    return result


def register(root, config):
    gate = read(root / "pretraining_checks.json")
    if gate["status"] != "passed" or gate["training_sources"] != sources():
        raise ValueError("Passed matching preflight required")
    if gate["protocol_fingerprint"] != digest_json(config):
        raise ValueError("Configuration changed after preflight")
    destination = root / "preregistration.json"
    if destination.exists():
        previous = read(destination)
        if previous["training_sources"] != sources() or previous["protocol_fingerprint"] != digest_json(config):
            raise ValueError("Preregistered configuration/code changed")
        return
    for name in sources():
        target = root / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(destination, {
        "registered_before_full_training_utc": utc(), "training_sources": sources(),
        "protocol_fingerprint": digest_json(config),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "existing_validation_and_external_scores_observed": True,
        "selection_data": "source validation only, fixed original CP2 manifest",
        "standalone_smoke_training": False, "new_training_runs": 1,
        "automatic_combinations": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cp2/phase_n.json"))
    parser.add_argument("--stage", choices=("prepare", "preflight", "register", "train", "status"), default="prepare")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    root = args.root.resolve()
    config = read(args.config)
    if args.stage == "status":
        for name in ("train_workflow.json", "N1/status.json"):
            print(json.dumps({name: read(root / name) if (root / name).exists() else {"status": "not_started"}}))
        return
    lock = (root / ".phase_n.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = prepare(root, config)
    if args.stage == "prepare":
        print(json.dumps({"status": "prepared", "dataset": read(root / "dataset_summary.json")}), flush=True)
        return
    if args.stage == "preflight":
        print(json.dumps(preflight(root, config)), flush=True)
        return
    if args.stage == "register":
        register(root, config)
        print("REGISTERED one N1 full training, seed42; N0 cached; no smoke training or combinations", flush=True)
        return
    register(root, config)
    workflow = root / "train_workflow.json"
    if workflow.exists():
        previous = read(workflow)
        if previous["status"] == "completed":
            print("CACHED completed workflow; no retraining", flush=True)
            return
        if not args.resume:
            raise FileExistsError("Prior workflow exists; inspect before --resume")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.mem_get_info(torch.device(args.device))[0] < 6 * 1024**3:
            raise RuntimeError("Need >=6 GiB available CUDA memory; no batch/precision fallback")
    stop = threading.Event()

    def monitor():
        with (root / "gpu_samples.jsonl").open("a", buffering=1) as stream:
            while not stop.is_set():
                stream.write(json.dumps(gpu_snapshot()) + "\n")
                stop.wait(15)

    write_json(workflow, {"status": "running", "pid": os.getpid(), "started_utc": utc(),
                          "device": args.device, "new_training_runs": 1, "seed": 42,
                          "standalone_smoke_training": False})
    threading.Thread(target=monitor, daemon=True).start()
    try:
        train(root, config, manifest, args.device, resume=args.resume)
        validation = evaluate(root, config, manifest, "validation", args.device)
        write_json(root / "paired_inputs.json", check_pairing(root, config))
        profiles = read(root / "architecture_preflight.json")["profiles"]
        decision = select_presence(validation, read(root / "control_reference.json")["validation"],
                                   profiles["N1"], profiles["N0"])
        selected = root / "selection.json"
        if selected.exists():
            previous = read(selected)
            if {k: v for k, v in previous.items() if k != "created_utc"} != decision:
                raise ValueError("Persisted source-validation decision differs")
        else:
            write_json(selected, {"created_utc": utc(), **decision})
        evaluate(root, config, manifest, "test", args.device)
        write_json(root / "inference_workflow.json", {"status": "completed", "finished_utc": utc()})
        # Fresh CPU plotting/validation process after GPU work; no second training job.
        with (root / "report_console.log").open("a") as stream:
            subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), str(ROOT / "scripts/report_cp2_phase_n.py"),
                            "--root", str(root)], stdout=stream, stderr=subprocess.STDOUT, check=True)
        write_json(workflow, {"status": "completed", "pid": os.getpid(), "finished_utc": utc(),
                              "seed": 42, "new_training_runs": 1, "automatic_combinations": False})
    except Exception:
        failure = {"status": "failed", "pid": os.getpid(), "failed_utc": utc(), "traceback": traceback.format_exc()}
        write_json(workflow, failure)
        print(json.dumps(failure), flush=True)
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
