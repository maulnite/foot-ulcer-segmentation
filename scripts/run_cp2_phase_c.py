"""One full A1+H2 training with three cached controls and frozen CP2 protocol."""

import argparse
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
from foot_ulcer_segmentation.common import digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_attention import model_profile  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_c import (  # noqa: E402
    MODELS, build, evaluate, histories, interaction, pairing, prepare, select, sources, train,
)
from foot_ulcer_segmentation.cp2_phase_n import preservation, read, utc  # noqa: E402
from foot_ulcer_segmentation.cp2_train import gpu_snapshot  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402


def architecture(root, config):
    setup(42, config["cpu_threads"])
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    fixture = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    profiles, inference, proofs, outputs = {}, {}, {}, {}
    for name in MODELS:
        model, _, proofs[name] = build(config, name, initial)
        profiles[name] = model_profile(model, fixture)
        profiles[name].pop("decoder_calls")
        inference[name] = model_profile(model.base, fixture)
        inference[name].pop("decoder_calls")
        model.eval()
        with torch.no_grad():
            result, bare = model(fixture), model(fixture, auxiliary=False)
        if not torch.equal(result.main_logits, bare.main_logits):
            raise ValueError("Head removal changed primary mask")
        if name in ("H2", "A1H2") and list(result.auxiliary_logits.shape) != [1, 1, 224, 224]:
            raise ValueError("Wrong boundary output shape")
        if model.base.decoder.blocks[3]._forward_hooks:
            raise ValueError("Read-only hook leaked")
        outputs[name] = result.main_logits.clone()
        del model
    if not torch.equal(outputs["B1"], outputs["H2"]) or not torch.equal(outputs["A1"], outputs["A1H2"]):
        raise ValueError("Unpaired primary forward before training")
    if proofs["A1"]["local_tensor_sha256"] != proofs["A1H2"]["local_tensor_sha256"]:
        raise ValueError("Unpaired local initialization")
    if proofs["H2"]["head_tensor_sha256"] != proofs["A1H2"]["head_tensor_sha256"]:
        raise ValueError("Unpaired boundary head initialization")
    if profiles["A1H2"]["active_forward_parameters"] != profiles["A1"]["active_forward_parameters"] + 33:
        raise ValueError("Unexpected joint parameter count")
    if inference["A1H2"]["active_forward_parameters"] != inference["A1"]["active_forward_parameters"]:
        raise ValueError("Boundary head retained in mask-only inference profile")
    result = {"status": "passed", "profiles": profiles, "mask_only_profiles": inference,
              "initialization": proofs, "training_sources": sources(),
              "B1_H2_initial_primary_bit_exact": True, "A1_A1H2_initial_primary_bit_exact": True,
              "paired_original_local_and_boundary_initialization": True,
              "fixture": "SYNTHETIC linspace structural forward only; no training or scientific metrics"}
    write_json(root / "architecture_preflight.json", result)
    return result


def preflight(root, config):
    gate = root / "pretraining_checks.json"
    if gate.exists():
        old = read(gate)
        if old["training_sources"] != sources() or old["protocol_fingerprint"] != digest_json(config):
            raise ValueError("Previous preflight code/config changed; preserve prior attempt")
        return old
    protected = preservation(root)
    architecture(root, config)
    with (root / "pretraining_tests.log").open("x") as stream:
        subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "-m", "pytest", "-o", "addopts=", "-q",
                        f"--junitxml={root / 'pretraining_tests.xml'}"],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    dependencies = {}
    for env in (".venv-training", ".venv-xfuseg"):
        result = subprocess.run([str(ROOT / env / "bin/python"), "-m", "pip", "check"],
                                capture_output=True, text=True, check=True)
        dependencies[env] = result.stdout.strip()
    suites = list(ET.parse(root / "pretraining_tests.xml").getroot().iter("testsuite"))
    tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    result = {"status": "passed", "created_utc": utc(), "tests": tests,
              "preservation": protected, "dependency_checks": dependencies,
              "training_sources": sources(), "protocol_fingerprint": digest_json(config),
              "standalone_smoke_training": False, "fresh_training_runs": 1,
              "actual_first_step_checks": "encoder, both local branches and boundary head; finite nonzero gradients/updates",
              "ruff": "NOT RUN — unavailable; existing environments unchanged"}
    write_json(gate, result)
    return result


def register(root, config):
    gate = read(root / "pretraining_checks.json")
    if gate["status"] != "passed" or gate["training_sources"] != sources() or gate["protocol_fingerprint"] != digest_json(config):
        raise ValueError("Passed matching preflight required")
    output = root / "preregistration.json"
    if output.exists():
        old = read(output)
        if old["training_sources"] != sources() or old["protocol_fingerprint"] != digest_json(config):
            raise ValueError("Registered source/config changed")
        return
    for name in sources():
        target = root / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(output, {"registered_before_full_training_utc": utc(), "training_sources": sources(),
               "protocol_fingerprint": digest_json(config),
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
               "new_training_runs": 1, "seed": 42, "cached_models": list(MODELS[:-1]),
               "standalone_smoke_training": False, "selection": "source validation only",
               "existing_source_and_CO2Wounds_results_observed": True,
               "individual_historical_decisions_preserved": True, "automatic_expansion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cp2/phase_c.json"))
    parser.add_argument("--stage", choices=("prepare", "preflight", "register", "train", "status"), default="prepare")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    root, config = args.root.resolve(), read(args.config)
    if args.stage == "status":
        for name in ("train_workflow.json", "A1H2/status.json"):
            print(json.dumps({name: read(root / name) if (root / name).exists() else {"status": "not_started"}}))
        return
    lock = (root / ".phase_c.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = prepare(root, config)
    if args.stage == "prepare":
        print("PREPARED: fixed CP2 population and cached B1/A1/H2; one new A1H2", flush=True)
        return
    if args.stage == "preflight":
        print(json.dumps(preflight(root, config)), flush=True)
        return
    register(root, config)
    if args.stage == "register":
        print("REGISTERED before full training; no separate smoke run or automatic expansion", flush=True)
        return
    workflow = root / "train_workflow.json"
    if workflow.exists():
        if read(workflow)["status"] == "completed":
            print("CACHED completed workflow", flush=True)
            return
        if not args.resume:
            raise FileExistsError("Inspect prior workflow before explicit --resume")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.mem_get_info(torch.device(args.device))[0] < 6 * 1024**3:
            raise RuntimeError("Need >=6GiB available CUDA memory; no batch/precision fallback")
    stop = threading.Event()

    def monitor():
        with (root / "gpu_samples.jsonl").open("a", buffering=1) as stream:
            while not stop.is_set():
                stream.write(json.dumps(gpu_snapshot()) + "\n")
                stop.wait(15)

    write_json(workflow, {"status": "running", "pid": os.getpid(), "started_utc": utc(),
               "device": args.device, "fresh_model": "A1H2", "seed": 42, "new_training_runs": 1})
    threading.Thread(target=monitor, daemon=True).start()
    try:
        train(root, config, manifest, args.device, resume=args.resume)
        fresh = evaluate(root, config, manifest, "validation", args.device)
        reports = {name: item["validation"] for name, item in read(root / "control_reference.json").items()}
        reports["A1H2"] = fresh
        write_json(root / "paired_inputs.json", pairing(histories(config, root, include_joint=True)))
        decision = select(reports, read(root / "architecture_preflight.json")["profiles"])
        selected = root / "selection.json"
        if selected.exists():
            if {k: v for k, v in read(selected).items() if k != "created_utc"} != decision:
                raise ValueError("Persisted source-validation decision differs")
        else:
            write_json(selected, {"created_utc": utc(), **decision})
        write_json(root / "interaction-validation.json", interaction(reports))
        evaluate(root, config, manifest, "test", args.device)
        write_json(root / "inference_workflow.json", {"status": "completed", "finished_utc": utc()})
        with (root / "report_console.log").open("a") as stream:
            subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), str(ROOT / "scripts/report_cp2_phase_c.py"),
                            "--root", str(root)], stdout=stream, stderr=subprocess.STDOUT, check=True)
        write_json(workflow, {"status": "completed", "pid": os.getpid(), "finished_utc": utc(),
                   "fresh_model": "A1H2", "new_training_runs": 1, "automatic_expansion": False})
    except Exception:
        failure = {"status": "failed", "pid": os.getpid(), "failed_utc": utc(), "traceback": traceback.format_exc()}
        write_json(workflow, failure)
        print(json.dumps(failure), flush=True)
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
