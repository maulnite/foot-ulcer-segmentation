"""Paired real-data errors and deterministic LOCAL-only source-image galleries."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def select_cases(rows):
    """Symmetric rules, deterministic IDs, including failures when present."""
    names = sorted(rows)
    positive = [n for n in names if int(rows[n]["B0"]["gt_pixels"]) > 0]
    random_name = names[int(np.random.default_rng(42).integers(len(names)))]
    ranked = sorted(positive, key=lambda n: (
        (float(rows[n]["B0"]["dice"]) + float(rows[n]["B1"]["dice"])) / 2, n))
    failed = [n for n in positive if any(int(rows[n][m]["tp"]) == 0 for m in ("B0", "B1"))]
    empty = [n for n in names if int(rows[n]["B0"]["gt_pixels"]) == 0]
    empty_fp = [n for n in empty if any(int(rows[n][m]["fp"]) > 0 for m in ("B0", "B1"))]
    multi = [n for n in positive if int(rows[n]["B0"]["gt_components_8_connected"]) > 1]
    smallest = min(positive, key=lambda n: (int(rows[n]["B0"]["gt_pixels"]), n))
    return [("Random seed 42", random_name), ("Median paired Dice, GT positif", ranked[len(ranked) // 2]),
            ("Kegagalan TP=0 pada salah satu model", failed[0] if failed else None),
            ("GT kosong dengan FP; fallback GT kosong", (empty_fp or empty or [None])[0]),
            ("GT multi-komponen (8-connected)", multi[0] if multi else None),
            ("Luas GT positif terkecil", smallest)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if read(root / "train_workflow.json")["status"] != "completed":
        raise ValueError("Both real-data baselines and evaluation must finish first")
    manifest = read(root / "data/manifest.json")
    output = root / "error_analysis"
    output.mkdir(parents=True, exist_ok=False)
    public = ROOT / "reports/figures/cp2-phase-p"
    public.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    choices = {}
    for row_index, split in enumerate(("validation", "test")):
        rows = {}
        for name in ("B0", "B1"):
            with (root / name / f"evaluation-{split}.csv").open() as stream:
                for row in csv.DictReader(stream):
                    rows.setdefault(row["name"], {})[name] = row
        names = sorted(rows)
        if not all(set(r) == {"B0", "B1"} for r in rows.values()):
            raise ValueError("Unpaired comparison population")
        a = np.array([float(rows[n]["B0"]["dice"]) for n in names])
        b = np.array([float(rows[n]["B1"]["dice"]) for n in names])
        ax = axes[row_index, 0]
        palette = {"empty": "#94A3B8", "small": "#7C3AED", "medium": "#2563EB", "large": "#0F8B8D"}
        for group, color in palette.items():
            selected = np.array([rows[n]["B0"]["size_group"] == group for n in names])
            ax.scatter(a[selected], b[selected], color=color, alpha=.6, s=24,
                       label=f"{group}: n={int(selected.sum())}")
        ax.plot([0, 1], [0, 1], "--", color="#64748B", linewidth=1)
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="B0 Image Dice", ylabel="B1 Image Dice",
               title=f"{split}: paired errors, n={len(names)}")
        ax.set_aspect("equal")
        ax.legend(fontsize=9, loc="upper left")
        difference = b - a
        ax = axes[row_index, 1]
        ax.hist(difference, bins=np.linspace(-1, 1, 41), color="#0F8B8D", edgecolor="white")
        ax.axvline(0, linestyle="--", color="#64748B")
        ax.set(xlim=(-1, 1), xlabel="Image Dice B1 − B0", ylabel="Jumlah RGB groups / images",
               title=f"{split}: distribusi perubahan per gambar")
        ax.text(.03, .95, f"B1 lebih baik: {(difference > 0).sum()}\nB0 lebih baik: {(difference < 0).sum()}\nSama: {(difference == 0).sum()}",
                transform=ax.transAxes, va="top", fontsize=10)
        for item in axes[row_index]:
            item.grid(alpha=.15)
        cases = select_cases(rows)
        choices[split] = [{"rule": rule, "sample_id": name, "missing_case": name is None,
                           "B0_dice": float(rows[name]["B0"]["dice"]) if name else None,
                           "B1_dice": float(rows[name]["B1"]["dice"]) if name else None}
                          for rule, name in cases]
        gallery, cells = plt.subplots(len(cases), 4, figsize=(12, 3 * len(cases)), constrained_layout=True)
        records = {r["sample_id"]: r for r in manifest["splits"][split]}
        for index, (rule, name) in enumerate(cases):
            for cell in cells[index]:
                cell.set_xticks([])
                cell.set_yticks([])
            if name is None:
                cells[index, 0].text(.1, .5, f"{rule}\nTidak ada kasus", wrap=True)
                for cell in cells[index, 1:]:
                    cell.axis("off")
                continue
            original = np.array(Image.open(ROOT / records[name]["image_path"]).convert("RGB"))
            exported = {}
            for model in ("B0", "B1"):
                path = root / model / "predictions" / split / (Path(records[name]["name"]).stem + ".npz")
                exported[model] = np.load(path, allow_pickle=False)
            truth = exported["B0"]["truth"].astype(bool)
            cells[index, 0].imshow(original)
            cells[index, 0].set_title(f"{rule}\nFoto asli, {truth.shape[1]}×{truth.shape[0]}", fontsize=10)
            cells[index, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
            cells[index, 1].set_title(f"GT: {truth.sum()} pixels\n{truth.mean()*100:.3f}% kanvas", fontsize=10)
            for column, model in [(2, "B0"), (3, "B1")]:
                prediction = exported[model]["prediction"].astype(bool)
                colors = np.zeros((*truth.shape, 4), dtype=float)
                colors[truth & prediction] = [.06, .72, .50, .7]
                colors[truth & ~prediction] = [.86, .15, .15, .8]
                colors[~truth & prediction] = [.05, .65, .91, .8]
                cells[index, column].imshow(original)
                cells[index, column].imshow(colors)
                cells[index, column].set_title(f"{model} cached prediction\nDice {float(rows[name][model]['dice']):.3f}", fontsize=10)
        gallery.suptitle(f"REAL / LOCAL ONLY — {split}, CP2 B0/B1 | TP hijau · FN merah · FP biru\nPemilihan simetris dan deterministik; gambar tidak di-commit", fontsize=13)
        gallery.savefig(output / f"{split}_local_gallery.png", dpi=140, bbox_inches="tight")
        plt.close(gallery)
    fig.suptitle("REAL — CP2 paired Image Dice | threshold ≥0,5 | satu seed, kelompok RGB unik", fontsize=14)
    public_files = []
    for extension in ("png", "svg"):
        path = public / f"paired_errors.{extension}"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        public_files.append({"path": str(path.relative_to(ROOT)), "sha256": digest_file(path)})
    plt.close(fig)
    evidence = {"status": "completed", "source": "REAL / CACHED predictions; FRESH paired analysis",
                "source_media_policy": "Original photo/GT/prediction galleries local only; no new raw clinical imagery in Git",
                "case_selection": choices, "public_figures": public_files}
    write_json(output / "selection.json", evidence)
    write_json(ROOT / "reports/cp2-phase-p-error-analysis.json", evidence)
    print(json.dumps({"public_aggregate_figures": public_files, "local_galleries": str(output)}, indent=2))


if __name__ == "__main__":
    main()
