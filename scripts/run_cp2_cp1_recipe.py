"""Sequential targeted CP1-recipe control, H2 and A6; historical runs immutable."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_recipe import (  # noqa: E402
    EXPERIMENTS, evaluate_test, prepare, read, run_experiment, select_recipe, sources,
    verify_paired_inputs,
)


def preservation(root):
    path = root / "protected_before.json"
    if not path.exists():
        names = set(subprocess.check_output(["git", "ls-files"], text=True).splitlines())
        names.add("Pengembangan_Novelty_dan_Eksperimen_CP2_FUSegNet.md")
        for directory in ("outputs/runs", "outputs/cp2/phase_p", "outputs/cp2/phase_l",
                          "outputs/cp2/phase_a", "outputs/cp2/phase_h", "outputs/cp2/author_categories"):
            names.update(str(p.relative_to(ROOT)) for p in (ROOT / directory).rglob('*') if p.is_file())
        names = {n for n in names if (ROOT / n).is_file()}
        with ThreadPoolExecutor(max_workers=6) as pool:
            write_json(path, dict(pool.map(lambda n: (n, digest_file(ROOT / n)), sorted(names))))
    inventory = read(root / "protected_before.json")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda n: (n, digest_file(ROOT / n) == inventory[n]), inventory))
    changed = [n for n, same in results if not same]
    if changed:
        raise ValueError(f"Protected historical/user files changed: {changed}")
    return {"status": "passed", "unchanged_files": len(inventory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cp2/cp1_recipe.json"))
    parser.add_argument("--stage", choices=("prepare", "preflight", "train", "status"), default="prepare")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "status":
        print(json.dumps({n: read(root / n / "status.json") if (root / n / "status.json").exists()
                          else {"status": "not_started"} for n in EXPERIMENTS}, indent=2))
        return
    lock = (root / ".recipe.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(args.config)
    manifest = prepare(config, root)
    proof = preservation(root)
    preregistration = {"config": config, "config_fingerprint": digest_json(config),
                       "manifest_fingerprint": manifest["fingerprint"], "training_sources": sources()}
    path = root / "preregistration.json"
    if path.exists():
        old = read(path)
        if any(old[k] != v for k, v in preregistration.items()):
            raise ValueError("Preregistered source/config changed; use a new run")
    else:
        write_json(path, {**preregistration, "created_utc": datetime.now(timezone.utc).isoformat(),
                         "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
        for name in sources():
            target = root / "training_code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    if args.stage == "prepare":
        print(json.dumps({"status": "prepared", "preservation": proof,
                          "split_counts": {s: len(v) for s, v in manifest["splits"].items()}}))
        return
    record = root / f"{args.stage}_workflow.json"
    if record.exists() and read(record)['status'] == 'completed':
        if read(root / 'pretraining_checks.json')['training_sources'] != sources():
            raise ValueError('Completed workflow code/config changed')
        print('CACHED completed workflow; protected assets verified')
        return
    write_json(record, {"status": "running", "pid": os.getpid(), "stage": args.stage,
                        "updated_utc": datetime.now(timezone.utc).isoformat()})
    try:
        gate = root / "pretraining_checks.json"
        if args.stage == "preflight":
            if gate.exists():
                if read(gate)["training_sources"] != sources():
                    raise ValueError("Completed preflight source mismatch")
                print("CACHED verified preflight")
                return
            with (root / "tests.log").open("x") as stream:
                subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "-m", "pytest", "-o", "addopts=", "-q",
                                f"--junitxml={root / 'tests.xml'}"], stdout=stream, stderr=subprocess.STDOUT, check=True)
            for env in (".venv-training", ".venv-xfuseg"):
                subprocess.run([str(ROOT / env / "bin/python"), "-m", "pip", "check"], check=True)
            for name in EXPERIMENTS:
                run_experiment(config, manifest, root, name, args.device, smoke=True, resume=args.resume)
            paired = verify_paired_inputs(root, smoke=True)
            write_json(root / "smoke_paired_inputs.json", paired)
            write_json(gate, {"status": "passed", "training_sources": sources(),
                             "config_fingerprint": digest_json(config), "preservation": preservation(root),
                             "models": list(EXPERIMENTS), "paired_inputs": paired,
                             "created_utc": datetime.now(timezone.utc).isoformat()})
        else:
            if not gate.exists() or read(gate)["training_sources"] != sources():
                raise ValueError("Successful unchanged preflight required before training")
            for name in EXPERIMENTS:
                run_experiment(config, manifest, root, name, args.device, resume=args.resume)
            write_json(root / "train_paired_inputs.json", verify_paired_inputs(root))
            if not (root / "selection.json").exists():
                write_json(root / "selection.json", select_recipe(root))
            elif read(root / "selection.json")["decisions"] != select_recipe(root)["decisions"]:
                raise ValueError("Persisted selection differs from completed validation")
            for name in EXPERIMENTS:
                evaluate_test(config, manifest, root, name, args.device)
            if not (root / 'final_validation.json').exists():
                subprocess.run([str(ROOT / ".venv-xfuseg/bin/python"), "scripts/validate_cp2_cp1_recipe.py",
                                "--root", str(root)], check=True)
            subprocess.run([str(ROOT / ".venv-analysis/bin/python"), "scripts/report_cp2_cp1_recipe.py",
                            "--root", str(root)], check=True)
            write_json(root / "final_preservation.json", preservation(root))
        write_json(record, {"status": "completed", "pid": os.getpid(), "stage": args.stage,
                            "updated_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException:
        write_json(record, {"status": "failed", "pid": os.getpid(), "stage": args.stage,
                            "updated_utc": datetime.now(timezone.utc).isoformat(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
