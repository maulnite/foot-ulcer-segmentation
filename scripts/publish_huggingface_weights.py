"""Inventory, publish and verify original Orie/Ganta research checkpoints.

Preparation uses the existing torch environment; upload uses hf-orie's installed
Hub SDK and isolated token file. No training or checkpoint conversion is done.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".hdf5", ".onnx", ".pkl", ".joblib", ".bin"}
IGNORED_DIRS = {".git", "data", "__pycache__", "node_modules", "model-publication"}
SIDECARS = {"config.json", "protocol.json", "status.json", "history.json", "environment.json",
            "evaluation-validation.json", "evaluation-test.json", "validation.json",
            "oof-validation.json", "model-ablation.json", "identity.json", "initialization_verified.json"}
SECRET_PATTERN = re.compile(rb"(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_checkpoints(root):
    found, excluded = [], []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not d.startswith((".venv", "venv")))
        for filename in sorted(files):
            path = Path(directory) / filename
            if path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(root)
            reason = None
            if "cache" in relative.parts:
                reason = "upstream pretrained download cache; not an experiment-trained checkpoint"
            elif filename == "end-rng.pth":
                reason = "RNG-only state; no model parameters"
            if reason:
                excluded.append({"path": relative.as_posix(), "reason": reason})
            else:
                found.append(path)
    return sorted(found), excluded


def destination(contributor, relative):
    text = relative.as_posix()
    replacements = (("outputs/cp2/", "orie/cp2/"), ("outputs/runs/", "orie/cp1/"),
                    ("outputs/smoke/", "orie/engineering-smoke/")) if contributor == "orie" else (
                    ("output_chronic_wound/checkpoints/", "ganta/chronic-wound/"),
                    ("output_fuseg_2021/checkpoints/", "ganta/fuseg-2021/"))
    for prefix, replacement in replacements:
        if text.startswith(prefix):
            return replacement + text[len(prefix):]
    raise ValueError(f"Unclassified model path: {contributor}/{text}")


def role_for(path):
    if "initial" in path.name:
        return "initialization_only_not_trained"
    if any("smoke" in part.lower() for part in path.parts):
        return "engineering_smoke_not_research_result"
    if path.name == "last.pth":
        return "last_training_checkpoint_not_selected"
    if path.name.startswith("stage1"):
        return "cascade_coarse_stage"
    if path.name.startswith("stage2"):
        return "cascade_refiner_stage"
    return "historical_selected_checkpoint"


def sanitize(value, roots):
    if isinstance(value, dict):
        return {k: sanitize(v, roots) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, roots) for v in value]
    if isinstance(value, str):
        for owner, root in roots.items():
            value = value.replace(str(root), "${" + owner.upper() + "_PROJECT_ROOT}")
        value = value.replace(str(Path.home()), "${SERVER_USER_HOME}")
    return value


def checked_bytes(path):
    raw = Path(path).read_bytes()
    if SECRET_PATTERN.search(raw):
        raise ValueError("Credential-like content detected; file publication refused")
    return raw


def checkpoint_summary(path, torch):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Unrecognized checkpoint container")
    weights = next((state[k] for k in ("state_dict", "model_state_dict", "model")
                    if isinstance(state.get(k), dict)), state)
    tensors = {k: v for k, v in weights.items() if torch.is_tensor(v)}
    if not tensors:
        raise ValueError("Checkpoint does not contain model tensors")
    result = {"container_keys": list(state)[:30] if weights is not state else ["raw_state_dict"],
              "tensor_count": len(tensors), "state_tensor_elements_including_buffers": sum(t.numel() for t in tensors.values()),
              "tensor_dtypes": dict(Counter(str(t.dtype) for t in tensors.values())),
              "optimizer_included": "optimizer" in state or "optimizer_state_dict" in state,
              "tensor_keys_preview": list(tensors)[:6]}
    for key in ("epoch", "completed_epochs", "seed", "baseline", "variant"):
        if isinstance(state.get(key), (int, float, str, bool)):
            result[key + "_as_stored"] = state[key]
    if isinstance(state.get("policy"), dict):
        result["selection_policy_as_stored"] = {k: v for k, v in state["policy"].items()
                                                 if isinstance(v, (int, float, str, bool, type(None)))}
    del tensors, weights, state
    gc.collect()
    return result


def git_snapshot(root):
    def command(*args):
        return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    return {"commit": command("rev-parse", "HEAD"), "branch": command("branch", "--show-current"),
            "dirty_status": command("status", "--short")}


def preserve_snapshot(roots):
    records = []
    for owner, root in roots.items():
        for section in ("src", "scripts", "configs", "docs", "reports", "notebooks", "notebook", "licenses"):
            directory = root / section
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in {".py", ".json", ".md", ".csv", ".ipynb", ".txt", ".yaml", ".yml"}:
                    records.append({"owner": owner, "path": path.relative_to(root).as_posix(), "sha256": sha256(path)})
    return records


def make_readme(folder, manifest, completed=False):
    counts = Counter(record["owner"] for record in manifest["checkpoints"])
    roles = Counter(record["role"] for record in manifest["checkpoints"])
    tree = f"https://huggingface.co/{manifest['repo_id']}/tree/main"
    lines = ["---", "language:", "- en", "- id", "library_name: pytorch", "pipeline_tag: image-segmentation",
             "tags:", "- fusegnet", "- efficientnet", "- wound-segmentation", "- research-checkpoints", "---", "",
             "# Foot Ulcer Segmentation · Orie & Ganta", "",
             "Arsip bobot asli seluruh eksperimen yang tersedia pada inventaris server. "
             "Checkpoint dipublikasikan tanpa retraining, konversi bobot, atau pemilihan ulang model.", "",
             "**Status: " + ("COMPLETE — all listed files remotely verified." if completed else
                               "UPLOAD IN PROGRESS — inventory prepared; remote verification pending.") + "**", "",
             "| Koleksi | Jumlah checkpoint | Akses |", "| :--- | ---: | :--- |",
             f"| Orie · CP1 | {sum('/cp1/' in r['hub_path'] for r in manifest['checkpoints'])} | [Browse]({tree}/orie/cp1) |",
             f"| Orie · CP2 | {sum('/cp2/' in r['hub_path'] for r in manifest['checkpoints'])} | [Browse]({tree}/orie/cp2) |",
             f"| Ganta · FUSeg 2021 | {sum('/fuseg-2021/' in r['hub_path'] for r in manifest['checkpoints'])} | [Browse]({tree}/ganta/fuseg-2021) |",
             f"| Ganta · Chronic Wound | {sum('/chronic-wound/' in r['hub_path'] for r in manifest['checkpoints'])} | [Browse]({tree}/ganta/chronic-wound) |", "",
             f"**{len(manifest['checkpoints'])} checkpoint paths**; Orie {counts['orie']}, Ganta {counts['ganta']}. "
             f"Ukuran logis {manifest['logical_bytes'] / 2**30:.2f} GiB; "
             f"{manifest['unique_checkpoint_contents']} isi unik (file identik memakai storage deduplication Hub).", "",
             "## Start here", "",
             "- [Daftar lengkap dan link download per file](CATALOG.md)",
             "- [Machine-readable model index](model-index.json)",
             "- [SHA256SUMS](SHA256SUMS.txt)",
             "- [Source code dan konfigurasi Orie](reproducibility/orie)",
             "- [Source code Ganta](reproducibility/ganta/notebook/scripts)",
             "- [Laporan hasil Orie](evidence/orie/reports)",
             "- [Hasil dan audit Ganta](evidence/ganta)", "",
             "## Which files should I use?", "",
             "`best.pth` / `best_model*.pth` adalah pilihan historis masing-masing run. "
             "`last.pth` adalah state akhir training, bukan otomatis pilihan terbaik. "
             "`stage1_coarse.pth` dan `stage2_refiner.pth` harus digunakan bersama untuk Cascade Ganta. "
             "Cascade CP2 menggunakan **B1 coarse + refiner CP2**; keduanya tercatat pada [recipe](RECIPES.md). "
             "Lima fold x-FUSegNet dipertahankan terpisah, bukan satu model ensemble yang sudah digabung.", "",
             f"Inventaris juga menyimpan {roles['engineering_smoke_not_research_result']} checkpoint smoke dan "
             f"{roles['initialization_only_not_trained']} initializer dengan label eksplisit. "
             "File-file ini bukan hasil evaluasi penelitian. Run lama/parsial Ganta tetap diarsipkan; "
             "keberadaan checkpoint tidak membuktikan semua rencana training selesai.", "",
             "## Download and integrity", "", "```bash",
             f"hf-orie download {manifest['repo_id']} --include 'orie/cp2/phase_p/*/B1/best.pth' 'model-index.json' 'SHA256SUMS.txt' --local-dir downloaded-weights",
             "cd downloaded-weights", "sha256sum --ignore-missing -c SHA256SUMS.txt", "```", "",
             "Format asli PyTorch dipertahankan. Checkpoint dapat berisi `state_dict`, optimizer, scheduler, "
             "RNG, dan history; sebagian Ganta adalah state dict/stage checkpoint. Gunakan `torch.load(..., "
             "map_location='cpu')` hanya untuk file tepercaya, lalu pilih container yang dijelaskan di indeks. "
             "Jangan mengimpor entrypoint training partner untuk inference karena dapat menjalankan training.", "",
             "## Compatibility, provenance and limitations", "",
             "Orie memakai environment project yang terdokumentasi (CP2: torch 1.11.0+cu113 / author's SMP fork); "
             "Ganta memakai implementasi/environment berbeda. Snapshot source dan lockfiles disertakan. "
             "Epoch disimpan sebagaimana checkpoint asli; konvensi indeks epoch antar runner dapat berbeda. "
             "Untuk hasil final, gunakan status/metrics/report yang terkait, bukan menafsirkan epoch sendiri.", "",
             "CP2 utama menggunakan train671 / validation118 / test261 yang dikelompokkan menurut RGB identik, "
             "bukan patient-level. CP1-matched, x-FUSegNet, dan eksperimen Ganta memiliki protokol berbeda. "
             "Jangan meranking seluruh arsip sebagai satu benchmark. Test/external yang sudah dieksplorasi "
             "tidak menjadi untouched confirmatory cohort. Tidak ada klaim clinical validation.", "",
             "## Attribution and use", "",
             "Reference: Dhar et al., [FUSegNet](https://arxiv.org/abs/2305.02961), "
             "[author code](https://github.com/mrinal054/FUSegNet), Segmentation Models PyTorch, dan EfficientNet. "
             "[Repository penelitian](https://github.com/maulnite/foot-ulcer-segmentation). "
             "Upstream notices tersedia dalam snapshot. Arsip ini tidak menetapkan lisensi baru untuk bobot "
             "atau dataset; patuhi ketentuan sumber masing-masing. Tidak ada foto pasien, raw dataset, "
             "mask dataset, probability caches, ataupun kredensial yang disertakan.", "",
             "Kelengkapan berarti seluruh checkpoint yang ditemukan dan lolos audit pada snapshot ini. "
             "Folder eksperimen tanpa checkpoint tercatat di `model-index.json`; bobot yang sudah tidak ada "
             "tidak dibuat ulang. Checkpoint optimizer/resume dipertahankan byte-identical; "
             "ini tidak menjamin seluruh dependensi training historis dapat direkonstruksi.", ""]
    (folder / "README.md").write_text("\n".join(lines))


def prepare(args):
    import torch
    torch.set_num_threads(1)
    roots = {"orie": args.root.resolve(), "ganta": args.ganta_root.resolve()}
    args.run.mkdir(parents=True, exist_ok=False)
    folder = args.run / "hub"
    folder.mkdir()
    write_json(args.run / "protected-before.json", preserve_snapshot(roots))
    manifest = {"schema_version": 1, "created_utc": now(), "repo_id": args.repo_id,
                "source_git": {owner: git_snapshot(root) for owner, root in roots.items()},
                "checkpoints": [], "excluded_files": [], "checkpoint_directories_without_weights": []}
    inode_cache, summary_cache = {}, {}
    expected = {record["checkpoint"]: record["checkpoint_sha256"]
                for record in json.loads((roots['orie'] / 'configs/cp2/external.json').read_text())["models"]}
    for owner, root in roots.items():
        found, excluded = scan_checkpoints(root)
        manifest["excluded_files"].extend({"owner": owner, **row} for row in excluded)
        for i, path in enumerate(found, 1):
            relative = path.relative_to(root)
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            digest = inode_cache.get(identity)
            if digest is None:
                digest = sha256(path)
                inode_cache[identity] = digest
            if owner == "orie" and relative.as_posix() in expected and digest != expected[relative.as_posix()]:
                raise ValueError("Checkpoint differs from the frozen canonical registry")
            if digest not in summary_cache:
                summary_cache[digest] = checkpoint_summary(path, torch)
            hub_path = destination(owner, relative)
            target = folder / hub_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
            status_path = path.parent / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            record = {"owner": owner, "source_relative_path": relative.as_posix(), "hub_path": hub_path,
                      "role": role_for(path), "bytes": stat.st_size, "sha256": digest,
                      "source_mtime_ns": stat.st_mtime_ns, "checkpoint": summary_cache[digest],
                      "run_status_as_recorded": status.get("status", "checkpoint_present_completion_not_independently_verified"),
                      "selected_epoch_from_run_status": status.get("selected_epoch"),
                      "selected_validation_image_dice_from_run_status": status.get("selected_validation_image_dice"),
                      "download_url": f"https://huggingface.co/{args.repo_id}/resolve/main/{hub_path}"}
            manifest["checkpoints"].append(record)
            print(f"AUDITED {owner} {i}/{len(found)} {hub_path}", flush=True)
            write_json(args.run / "inventory-progress.json", {"at": now(), "audited": len(manifest['checkpoints'])})
            for sidecar in sorted(SIDECARS):
                source = path.parent / sidecar
                out = target.parent / sidecar
                if source.is_file() and not out.exists():
                    checked_bytes(source)
                    write_json(out, sanitize(json.loads(source.read_text()), roots))
            for ancestor in path.parents:
                if ancestor == root:
                    break
                for name in ("protocol.json", "config.json", "control_reference.json", "selection.json", "preregistration.json"):
                    source = ancestor / name
                    if not source.is_file():
                        continue
                    out = folder / destination(owner, source.relative_to(root))
                    if not out.exists():
                        publish_text(source, out, roots)
        if owner == "ganta":
            for output in ("output_chronic_wound", "output_fuseg_2021"):
                for directory in sorted((root / output / "checkpoints").iterdir()):
                    if directory.is_dir() and not any(p.suffix in EXTENSIONS for p in directory.iterdir() if p.is_file()):
                        manifest["checkpoint_directories_without_weights"].append(owner + "/" + directory.relative_to(root).as_posix())
    manifest["logical_bytes"] = sum(r["bytes"] for r in manifest["checkpoints"])
    unique = {r['sha256']: r['bytes'] for r in manifest['checkpoints']}
    manifest["unique_checkpoint_contents"] = len(unique)
    manifest["unique_checkpoint_bytes"] = sum(unique.values())
    for owner, root in roots.items():
        for section in ("src", "scripts", "configs", "licenses"):
            for path in sorted((root / section).rglob("*")):
                if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml", ".txt", ".md"}:
                    publish_text(path, folder / "reproducibility" / owner / path.relative_to(root), roots)
        if owner == "ganta":
            for path in sorted((root / "notebook/scripts").glob("*.py")):
                publish_text(path, folder / "reproducibility/ganta" / path.relative_to(root), roots)
            publish_text(root / "notebook/UPSTREAM_LICENSE.txt", folder / "reproducibility/ganta/notebook/UPSTREAM_LICENSE.txt", roots)
        else:
            for path in sorted((root / "source/upstream/FUSegNet").rglob("*")):
                if path.is_file() and (path.suffix == ".py" or path.name in {"LICENSE", "LICENSE.txt"}):
                    publish_text(path, folder / "reproducibility/orie" / path.relative_to(root), roots)
        for path in sorted(root.glob("requirements*.txt")):
            publish_text(path, folder / "reproducibility" / owner / path.name, roots)
        for path in sorted((root / "docs").glob("*.md")):
            publish_text(path, folder / "reproducibility" / owner / path.relative_to(root), roots)
        for path in sorted((root / "reports").glob("*")):
            if path.is_file() and path.suffix in {".json", ".md"}:
                publish_text(path, folder / "evidence" / owner / path.relative_to(root), roots)
        if owner == "ganta":
            for output in ("output_chronic_wound", "output_fuseg_2021"):
                for path in sorted((root / output / "results").rglob("*")):
                    if path.is_file() and path.suffix in {".json", ".csv", ".txt"}:
                        publish_text(path, folder / "evidence/ganta" / path.relative_to(root), roots)
    make_readme(folder, manifest)
    make_catalog(folder, manifest)
    (folder / ".gitattributes").write_text("*.pth filter=lfs diff=lfs merge=lfs -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n")
    write_json(folder / "model-index.json", manifest)
    (folder / "SHA256SUMS.txt").write_text("".join(f"{r['sha256']}  {r['hub_path']}\n" for r in manifest['checkpoints']))
    write_json(args.run / "preparation.json", {"status": "prepared", "at": now(), "checkpoints": len(manifest['checkpoints']),
               "logical_bytes": manifest['logical_bytes'], "unique_bytes": manifest['unique_checkpoint_bytes'], "gpu_used": False})
    print(json.dumps(json.loads((args.run / 'preparation.json').read_text())), flush=True)


def publish_text(source, target, roots):
    raw = checked_bytes(source)
    text = raw.decode("utf-8")
    if source.suffix == ".json":
        text = json.dumps(sanitize(json.loads(text), roots), indent=2, ensure_ascii=False) + "\n"
    else:
        text = sanitize(text, roots)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def make_catalog(folder, manifest):
    lines = ["# Checkpoint catalog", "", "Semua jalur merujuk ke file asli, bukan ekspor ulang. "
             "Epoch mengikuti metadata asli; lihat status/report untuk konvensi dan seleksi.", ""]
    for owner in ("orie", "ganta"):
        lines += ["## " + owner.title(), "", "| Checkpoint | Peran | Epoch asli / completed | GiB |",
                  "| :--- | :--- | :--- | ---: |"]
        for r in manifest["checkpoints"]:
            if r["owner"] != owner:
                continue
            summary = r["checkpoint"]
            epoch = summary.get("epoch_as_stored", summary.get("completed_epochs_as_stored", "—"))
            url = f"https://huggingface.co/{manifest['repo_id']}/blob/main/{r['hub_path']}"
            lines.append(f"| [{r['hub_path']}]({url}) | {r['role']} | {epoch} | {r['bytes']/2**30:.3f} |")
    lines += ["", "## Directories without checkpoint", "", "Folder berikut tersedia tetapi tidak memiliki bobot; "
              "tidak dibuat ulang dan tidak diklaim terunggah:", ""]
    lines += ["- `" + value + "`" for value in manifest["checkpoint_directories_without_weights"]]
    (folder / "CATALOG.md").write_text("\n".join(lines) + "\n")
    coarse = next(r['hub_path'] for r in manifest['checkpoints'] if '/phase_p/' in r['hub_path'] and r['hub_path'].endswith('/B1/best.pth'))
    refiner = next(r['hub_path'] for r in manifest['checkpoints'] if '/cascade/' in r['hub_path'] and r['hub_path'].endswith('/refiner/best.pth'))
    (folder / "RECIPES.md").write_text("# Model bundles\n\n## Cascade CP2\n\n"
        f"- Frozen coarse B1: [{coarse}]({coarse})\n- Refiner: [{refiner}]({refiner})\n"
        "- Inference: proposal threshold0.3, 8-connected components≥4px, max5, margin0.5, min side0.25, "
        "crop224; maximum probability in overlaps; preserve coarse outside crop; final probability≥0.5.\n"
        "- Selected refiner epoch46; GT crops only during training. No validation/test GT proposals.\n"
        "- Matching config: `reproducibility/orie/configs/cp2/cascade_clean.json`.\n\n"
        "## Original Ganta Cascade\n\nUse `stage1_coarse.pth` + `stage2_refiner.pth` from the **same** "
        "dataset/run directory. Do not mix this protocol with Cascade CP2. Matching scripts are under "
        "`reproducibility/ganta/notebook/scripts/13_cascade_crop_refine_*`.\n\n"
        "## x-FUSegNet\n\nUse five `best_model_1.pth` … `best_model_5.pth` from one run; average raw sigmoid "
        "maps, then threshold≥0.5. Historical partial runs remain distinguishable by source run IDs.\n")


def source_preserved(args, manifest, full_hash=False):
    roots = {"orie": args.root.resolve(), "ganta": args.ganta_root.resolve()}
    for row in manifest["checkpoints"]:
        path = roots[row["owner"]] / row["source_relative_path"]
        stat = path.stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["source_mtime_ns"]:
            raise ValueError("Source checkpoint changed after inventory")
        if full_hash and sha256(path) != row["sha256"]:
            raise ValueError("Source checkpoint content changed")
    protected = json.loads((args.run / "protected-before.json").read_text())
    for row in protected:
        if sha256(roots[row['owner']] / row['path']) != row['sha256']:
            raise ValueError("Protected research source/report/notebook changed")
    return len(protected)


def hub_api():
    from huggingface_hub import HfApi
    token_path = Path.home() / ".config/hf-orie/token"
    api = HfApi(token=token_path.read_text().strip())
    if api.whoami()["name"] != "maulnite":
        raise ValueError("Expected the isolated maulnite Hugging Face account")
    return api


def upload(args):
    folder = args.run / "hub"
    manifest = json.loads((folder / "model-index.json").read_text())
    assert manifest['repo_id'] == args.repo_id
    source_preserved(args, manifest)
    api = hub_api()
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=folder / "README.md", path_in_repo="README.md", repo_id=args.repo_id,
                    commit_message="docs: describe complete research checkpoint inventory before upload")
    write_json(args.run / "upload-status.json", {"status": "uploading", "started_utc": now(), "repo_id": args.repo_id})
    api.upload_large_folder(repo_id=args.repo_id, repo_type="model", folder_path=folder, num_workers=args.workers,
                            print_report=True, print_report_every=30, ignore_patterns=[".cache/**"])
    write_json(args.run / "upload-status.json", {"status": "uploaded_verification_pending", "at": now()})
    verify(args)


def verify(args):
    folder = args.run / "hub"
    manifest = json.loads((folder / "model-index.json").read_text())
    api = hub_api()
    revision = api.repo_info(args.repo_id, repo_type="model").sha
    remote = {item.path: item for item in api.list_repo_tree(args.repo_id, repo_type="model", recursive=True, revision=revision)
              if hasattr(item, "size")}
    checked = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(folder).parts:
            continue
        relative = path.relative_to(folder).as_posix()
        if relative == "publication-validation.json":
            continue
        item = remote.get(relative)
        if item is None or item.size != path.stat().st_size:
            raise ValueError(f"Missing or wrong-sized remote artifact: {relative}")
        expected = next((r["sha256"] for r in manifest['checkpoints'] if r['hub_path'] == relative), None)
        if item.lfs:
            expected = expected or sha256(path)
            if item.lfs.sha256 != expected:
                raise ValueError(f"Remote SHA256 mismatch: {relative}")
            method = "server_LFS_SHA256"
        else:
            raw = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if item.blob_id != blob:
                raise ValueError(f"Remote Git blob mismatch: {relative}")
            method = "server_Git_blob"
        checked.append({"path": relative, "bytes": item.size, "method": method})
    protected = source_preserved(args, manifest, full_hash=True)
    report = {"status": "PASSED", "verified_utc": now(), "repo_id": args.repo_id, "verified_revision": revision,
              "checkpoint_paths_verified": len(manifest['checkpoints']), "remote_files_verified": len(checked),
              "logical_checkpoint_bytes": manifest['logical_bytes'], "unique_checkpoint_contents": manifest['unique_checkpoint_contents'],
              "all_checkpoint_SHA256_match_server": True, "source_checkpoint_SHA256_rechecked": True,
              "protected_research_files_unchanged": protected, "training_executed": False, "GPU_used": False,
              "raw_datasets_or_media_included": False, "verified_files": checked}
    make_readme(folder, manifest, completed=True)
    write_json(folder / "publication-validation.json", report)
    commit = api.upload_folder(repo_id=args.repo_id, folder_path=folder, allow_patterns=["README.md", "publication-validation.json"],
                               commit_message="publication: verify every checkpoint and finalize research archive")
    report['final_revision'] = commit.oid
    report['url'] = f"https://huggingface.co/{args.repo_id}"
    write_json(args.run / 'publication-validation.json', report)
    write_json(args.run / 'upload-status.json', {"status":"completed", "at":now(), "url":report['url'], "revision":commit.oid})
    print(json.dumps({k:v for k,v in report.items() if k!='verified_files'},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "upload", "verify"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ganta-root", type=Path, default=Path(__file__).resolve().parents[2] / "ganta")
    parser.add_argument("--repo-id", default="maulnite/foot-ulcer-segmentation-weights")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
