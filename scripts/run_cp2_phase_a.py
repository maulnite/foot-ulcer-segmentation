"""Sequential attention screening: engineering gates, A1 diagnosis, bounded expansion."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import threading
import traceback
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_attention import CBAM, build_attention_model, model_profile  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_a import (  # noqa: E402
    EXPERIMENTS, evaluate_candidate, expansion_decision, phase_a_sources, prepare, read,
    run_experiment, select_candidate, verify_inputs,
)
from foot_ulcer_segmentation.cp2_train import gpu_snapshot  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402


def preserved(root):
    destination = root / "protected_prior_sha256.json"
    if destination.exists():
        expected = read(destination)
    else:
        expected = read(ROOT / "outputs/cp2/phase_l/20261108T065944Z/protected_prior_sha256.json")
        paths = set(expected)
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        paths.update(n for n in tracked if not any(v in n for v in ("phase_a", "phase-a", "cp2_attention.py")))
        for base in (ROOT / "outputs/cp2/phase_p", ROOT / "outputs/cp2/phase_l"):
            paths.update(str(p.relative_to(ROOT)) for p in base.rglob("*") if p.is_file())
        expected = {p: digest_file(ROOT / p) for p in sorted(paths)}
        write_json(destination, expected)
    with ThreadPoolExecutor(max_workers=6) as pool:
        changed = [p for p, same in pool.map(lambda p: (p, digest_file(ROOT / p) == expected[p]), expected) if not same]
    if changed:
        raise ValueError(f"Historical/user assets changed: {changed}")
    return len(expected)


def profile_initial_models(root, config):
    path = root / "architecture_preflight.json"
    if path.exists():
        previous = read(path)
        if previous.get("attention_source_sha256") == digest_file(ROOT / "src/foot_ulcer_segmentation/cp2_attention.py"):
            return previous
        archive = root / "engineering_attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        path.rename(archive / path.name)
    setup(config["seed"], config["cpu_threads"])
    initial = torch.load(root / "initial_weights.pth", map_location="cpu")
    inputs = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    profiles, proofs, outputs = {}, {}, {}
    for name in ("A0", *EXPERIMENTS):
        model, proofs[name] = build_attention_model(config, name, initial)
        profiles[name] = model_profile(model, inputs)
        with torch.no_grad():
            prediction = model(inputs)
        outputs[name] = hashlib.sha256(prediction.numpy().tobytes()).hexdigest()
        for stage in range(5):
            calls = profiles[name]["decoder_calls"]
            assert calls[f"decoder.blocks.{stage}.conv1"] == 1
            assert calls[f"decoder.blocks.{stage}.conv2"] == 0
            assert calls[f"decoder.blocks.{stage}.attention1"] == int(stage < 4 and not (name == "A6" and stage in (2, 3)))
            assert calls[f"decoder.blocks.{stage}.attention2"] == int(not (name == "A6" and stage in (2, 3)))
        del model
    assert len({outputs[n] for n in ("A1", "A2", "A3")}) == 1
    assert profiles["A0"]["registered_parameters"] == 70_968_939
    assert profiles["A0"]["active_forward_parameters"] == 64_898_248
    assert outputs["A5"] == outputs["A0"]
    assert len({proofs[n]["local_tensor_sha256"] for n in ("A1", "A2", "A3")}) == 1
    extra3 = profiles["A3"]["active_forward_parameters"] - profiles["A0"]["active_forward_parameters"]
    extra5 = profiles["A5"]["active_forward_parameters"] - profiles["A0"]["active_forward_parameters"]
    assert abs(extra5 / extra3 - 1) <= .05
    result = {"status": "passed", "profiles": profiles, "initialization": proofs,
              "attention_source_sha256": digest_file(ROOT / "src/foot_ulcer_segmentation/cp2_attention.py"),
              "initial_output_sha256": outputs, "engineering_input": "SYNTHETIC linspace shape probe, not scientific scores",
              "A1_A2_A3_initial_forward_exactly_equal": True, "A5_initial_forward_equals_A0": True,
              "capacity_added_active_parameter_relative_difference": abs(extra5 / extra3 - 1)}
    write_json(path, result)
    return result


def cbam_oracle(root):
    path = root / "cbam_oracle.json"
    if path.exists():
        previous = read(path)
        if previous.get("attention_source_sha256") == digest_file(ROOT / "src/foot_ulcer_segmentation/cp2_attention.py"):
            return previous
        archive = root / "engineering_attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        path.rename(archive / path.name)
    reference = root / "reference_audit/cbam.py"
    license_path = root / "reference_audit/LICENSE"
    if not reference.exists():
        import base64
        reference.parent.mkdir(exist_ok=True)
        for source, output in (("MODELS/cbam.py", reference), ("LICENSE", license_path)):
            metadata = json.loads(subprocess.check_output([
                "gh-orie", "api", "repos/Jongchan/attention-module/contents/" + source +
                "?ref=459efad0e05ee7dde50c41ca10a3d0800bc3792a"], text=True))
            output.write_bytes(base64.b64decode(metadata["content"]))
    if digest_file(license_path) != digest_file(ROOT / "licenses/cbam-MIT.txt"):
        raise ValueError("CBAM attribution/license differs")
    original_class = runpy.run_path(str(reference))["CBAM"]
    setup(42, 2)
    original, actual = original_class(32), CBAM(32)
    mappings = {"mlp.0": "ChannelGate.mlp.1", "mlp.2": "ChannelGate.mlp.3",
                "spatial.0": "SpatialGate.spatial.conv", "spatial.1": "SpatialGate.spatial.bn"}
    original_state = original.state_dict()
    actual.load_state_dict({name: original_state[next(old + name[len(new):] for new, old in mappings.items()
                                                    if name.startswith(new + "."))] for name in actual.state_dict()})
    for training in (False, True):
        original.train(training)
        actual.train(training)
        inputs = torch.randn(2, 32, 7, 9)
        torch.testing.assert_close(original(inputs), actual(inputs), rtol=0, atol=0)
    result = {"status": "passed", "reference_source_sha256": digest_file(reference),
              "attention_source_sha256": digest_file(ROOT / "src/foot_ulcer_segmentation/cp2_attention.py"),
              "official_revision": "459efad0e05ee7dde50c41ca10a3d0800bc3792a",
              "fixtures": "SYNTHETIC random tensor engineering checks, eval and train with BN",
              "unscaled_CBAM_forward_bit_exact_after_semantic_weight_mapping": True}
    write_json(path, result)
    return result


def preflight(root, config, manifest, device):
    gate = root / "pretraining_checks.json"
    if gate.exists():
        result = read(gate)
        if result["training_sources"] != phase_a_sources() or result["protocol_fingerprint"] != digest_json(config):
            raise ValueError("Preflight cannot be reused after code/config changes")
        return result
    count = preserved(root)
    profile_initial_models(root, config)
    cbam_oracle(root)
    with (root / "pretraining_tests.log").open("x") as stream:
        subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "-m", "pytest", "-o", "addopts=", "-q",
                        f"--junitxml={root / 'pretraining_tests.xml'}"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    suites = list(ET.parse(root / "pretraining_tests.xml").getroot().iter("testsuite"))
    tests = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "errors", "failures", "skipped")}
    if tests["errors"] or tests["failures"]:
        raise ValueError("Tests failed")
    for environment in (".venv-training", ".venv-analysis"):
        subprocess.run([str(ROOT / environment / "bin/python"), "-m", "pip", "check"], check=True)
    for name in EXPERIMENTS:
        run_experiment(config, manifest, root, name, device, smoke=True)
    paired = verify_inputs(config, root, list(EXPERIMENTS), smoke=True)
    write_json(root / "smoke_paired_controls.json", paired)
    result = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "training_sources": phase_a_sources(), "protocol_fingerprint": digest_json(config),
              "historical_files_unchanged": count, "tests": tests,
              "real_data_smoke_models": list(EXPERIMENTS), "manifest_fingerprint": manifest["fingerprint"]}
    write_json(gate, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cp2/phase_a.json"))
    parser.add_argument("--stage", choices=("prepare", "preflight", "train", "status"), default="prepare")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    root, config = args.root.resolve(), read(args.config)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "status":
        print(json.dumps({n: read(root / n / "status.json") if (root / n / "status.json").exists()
                          else {"status": "not_started"} for n in EXPERIMENTS}, indent=2))
        return
    lock = (root / ".phase_a.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = prepare(config, root)
    if args.stage == "prepare":
        print("Prepared immutable B1 control assets; no baseline retraining")
        return
    status_path = root / f"{args.stage}_workflow.json"
    if status_path.exists() and read(status_path)["status"] == "completed":
        if args.stage == "preflight":
            preflight(root, config, manifest, args.device)
        else:
            proof = read(root / "pretraining_checks.json")
            if proof["training_sources"] != phase_a_sources() or proof["protocol_fingerprint"] != digest_json(config):
                raise ValueError("Completed training uses different code/config; preserve its sources")
        print("CACHED completed workflow")
        return
    stop = threading.Event()

    def monitor():
        with (root / f"{args.stage}_gpu_samples.jsonl").open("a", buffering=1) as stream:
            while not stop.is_set():
                sample = gpu_snapshot()
                shm = os.statvfs("/dev/shm")
                sample["shm_free_bytes"] = shm.f_bavail * shm.f_frsize
                stream.write(json.dumps(sample) + "\n")
                stop.wait(5)

    write_json(status_path, {"status": "running", "pid": os.getpid(), "stage": args.stage,
                             "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        if args.device.startswith("cuda"):
            free, _ = torch.cuda.mem_get_info()
            if free < 6 * 1024**3:
                raise RuntimeError("Less than 6 GiB free GPU memory; no precision/batch fallback")
            threading.Thread(target=monitor, daemon=True).start()
        if args.stage == "preflight":
            preflight(root, config, manifest, args.device)
        else:
            gate = read(root / "pretraining_checks.json")
            if gate["status"] != "passed" or gate["training_sources"] != phase_a_sources() or gate["protocol_fingerprint"] != digest_json(config):
                raise ValueError("Frozen pretraining gate incompatible")
            snapshot = root / "training_code_snapshot"
            if not snapshot.exists():
                for relative in phase_a_sources():
                    destination = snapshot / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / relative, destination)
            run_experiment(config, manifest, root, "A1", args.device, resume=args.resume)
            first = evaluate_candidate(config, manifest, root, "A1", "validation", args.device)
            control = read(root / "control_reference.json")["validation"]
            decision = expansion_decision(first, control, config)
            decision_path = root / "expansion_decision.json"
            if decision_path.exists():
                previous = read(decision_path)
                if previous != {**decision, "created_utc": previous["created_utc"]}:
                    raise ValueError("A1 validation expansion decision changed")
            else:
                write_json(root / "A1_validation_diagnosis.json", {
                    "status": "A1 real training and validation completed before expansion",
                    "validation_only": True, "A1": first, "A0": control,
                    "decision": decision, "no_new_test_predictions_created": True})
                write_json(decision_path, decision)
            print("A1 VALIDATION DIAGNOSIS: " + json.dumps(decision), flush=True)
            names = ["A1", *decision["remaining_models"]]
            for name in names[1:]:
                run_experiment(config, manifest, root, name, args.device, resume=args.resume)
            validation = {name: evaluate_candidate(config, manifest, root, name, "validation", args.device) for name in names}
            paired = verify_inputs(config, root, names)
            write_json(root / "train_paired_controls.json", paired)
            selection = select_candidate(validation, control, read(root / "architecture_preflight.json")["profiles"])
            selection_path = root / "selection.json"
            if selection_path.exists():
                previous = read(selection_path)
                if previous != {**selection, "created_utc": previous["created_utc"]}:
                    raise ValueError("Persisted selection changed")
            else:
                write_json(selection_path, selection)
            for name in names:
                evaluate_candidate(config, manifest, root, name, "test", args.device)
                write_json(root / name / "final_evaluation.json", {
                    split: read(root / name / f"evaluation-{split}.json") for split in ("validation", "test")})
        write_json(status_path, {"status": "completed", "pid": os.getpid(), "stage": args.stage,
                                 "finished_utc": datetime.now(timezone.utc).isoformat()})
    except Exception:
        failure = {"status": "failed", "pid": os.getpid(), "stage": args.stage,
                   "utc": datetime.now(timezone.utc).isoformat(), "traceback": traceback.format_exc()}
        write_json(status_path, failure)
        print(json.dumps(failure), flush=True)
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
