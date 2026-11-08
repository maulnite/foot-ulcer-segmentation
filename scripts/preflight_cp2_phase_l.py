"""Reproducible prepare/tests/reference-oracle/smoke gate for an existing CP2 control."""

import argparse
import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, List, Set, Tuple, cast
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_losses import signed_distance  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_l import phase_l_sources, prepare, read  # noqa: E402


def preserve_inventory(root, config):
    destination = root / "protected_prior_sha256.json"
    if destination.exists():
        protected = read(destination)
    else:
        control = ROOT / config["phase_l"]["control_root"]
        paths = set(read(control / "protected_cp1_sha256.json"))
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        for name in tracked:
            # Current phase-L presentation/code may be improved on its own branch.
            # Its executed core is separately frozen by the source/config gate.
            if not any(token in name for token in ("cp2_phase_l", "cp2-phase-l", "cp2_losses", "cp2/phase_l.json")):
                paths.add(name)
        paths.update(str(p.relative_to(ROOT)) for p in control.rglob("*") if p.is_file())
        roadmap = ROOT / "Pengembangan_Novelty_dan_Eksperimen_CP2_FUSegNet.md"
        if roadmap.exists():
            paths.add(str(roadmap.relative_to(ROOT)))
        protected = {name: digest_file(ROOT / name) for name in sorted(paths)}
        write_json(destination, protected)
    with ThreadPoolExecutor(max_workers=6) as pool:
        actual = dict(pool.map(lambda name: (name, digest_file(ROOT / name)), protected))
    if actual != protected:
        raise ValueError("Historical/user files changed after this run's initial inventory")
    if not (root / "initial_audit.json").exists():
        write_json(root / "initial_audit.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv"], text=True),
            "disk": subprocess.check_output(["df", "-h", "."], text=True)})
    return len(protected)


def reference_oracle(root, config):
    target = root / "boundary_loss_oracle.json"
    if target.exists():
        result = read(target)
        if result["status"] != "passed":
            raise ValueError("Existing oracle did not pass")
        return result
    directory = root / "reference_audit"
    directory.mkdir(exist_ok=True)
    revision = config["phase_l"]["boundary"]["reference_revision"]
    files = {}
    for name in ("utils.py", "losses.py", "LICENSE"):
        metadata = json.loads(subprocess.check_output([
            "gh-orie", "api", f"repos/LIVIAETS/boundary-loss/contents/{name}?ref={revision}"], text=True))
        data = base64.b64decode(metadata["content"])
        path = directory / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Pinned boundary source changed locally")
        if not path.exists():
            path.write_bytes(data)
        files[name] = {"blob_sha": metadata["sha"], "sha256": digest_file(path), "url": metadata["html_url"]}
    namespace = {"np": np, "torch": torch, "Tensor": torch.Tensor,
                 "Tuple": Tuple, "List": List, "Iterable": Iterable, "Set": Set,
                 "cast": cast, "eucl_distance": distance_transform_edt, "einsum": torch.einsum}
    for filename, wanted in [("utils.py", ["uniq", "sset", "simplex", "one_hot", "one_hot2dist"]),
                             ("losses.py", ["SurfaceLoss"])]:
        for node in ast.parse((directory / filename).read_text()).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), f"pinned_{filename}", "exec"), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        surface = namespace["SurfaceLoss"](idc=[1])
    rng = np.random.default_rng(42)
    masks = [np.zeros((7, 9), np.uint8), np.ones((7, 9), np.uint8)]
    masks += [(rng.random((7, 9)) > .8).astype(np.uint8) for _ in range(38)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for mask in masks:
            onehot = np.stack([1 - mask, mask])
            expected = namespace["one_hot2dist"](onehot, dtype=np.float32)
            actual = signed_distance(mask)
            np.testing.assert_array_equal(actual, expected[1])
            probability = torch.tensor(rng.uniform(.01, .99, (1, 1, 7, 9)), dtype=torch.float32)
            logits_free_probabilities = torch.cat([1 - probability, probability], dim=1)
            first = (probability * torch.from_numpy(actual)[None, None]).mean()
            second = surface(logits_free_probabilities, torch.from_numpy(expected[None]))
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    result = {"status": "passed", "fixtures": len(masks), "fixtures_kind": "synthetic engineering masks only",
              "distance_maps": "bit exact", "surface_product_mean": "bit exact",
              "reference": {"repository": "LIVIAETS/boundary-loss", "revision": revision, "files": files}}
    write_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cp2/phase_l.json"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.chdir(ROOT)
    root, config = args.root.resolve(), read(args.config)
    gate = root / "pretraining_checks.json"
    if gate.exists():
        recorded = read(gate)
        if (recorded["status"] != "passed" or recorded["training_sources"] != phase_l_sources()
                or recorded["protocol_fingerprint"] != digest_json(config)):
            raise ValueError("Cannot reuse preflight after source/config changes")
        print("CACHED verified preflight; no GPU smoke or training repeated")
        return
    manifest = prepare(config, root)
    protected = preserve_inventory(root, config)
    oracle = reference_oracle(root, config)
    python = str(ROOT / ".venv-training/bin/python")
    with (root / "pretraining_tests.log").open("x") as stream:
        subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "-m", "pytest", "-o", "addopts=", "-q",
                        f"--junitxml={root / 'pretraining_tests.xml'}"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for environment in (".venv-training", ".venv-analysis"):
        subprocess.run([str(ROOT / environment / "bin/python"), "-m", "pip", "check"], cwd=ROOT, check=True)
    with (root / "smoke_console.log").open("x") as stream:
        subprocess.run([python, "-u", "scripts/run_cp2_phase_l.py", "--root", str(root),
                        "--config", str(args.config), "--stage", "smoke", "--device", args.device],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    suites = list(ET.parse(root / "pretraining_tests.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    if counts["errors"] or counts["failures"] or read(root / "smoke_paired_controls.json")["status"] != "passed":
        raise ValueError("Tests/smoke controls did not pass")
    write_json(gate, {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
                     "training_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                     "protocol_fingerprint": digest_json(config), "training_sources": phase_l_sources(),
                     "manifest_fingerprint": manifest["fingerprint"], "tests": counts,
                     "original_files_unchanged": protected, "oracle_fixtures": oracle["fixtures"],
                     "preflight_driver_sha256": digest_file(__file__)})
    print("Preflight passed: actual tests, pinned reference oracle, real-data smoke and preservation inventory")


if __name__ == "__main__":
    main()
