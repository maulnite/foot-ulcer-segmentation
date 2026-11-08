"""Deterministic, LOCAL-only clinical galleries from actual verified cached predictions."""

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


NAMES = ("L0", "L1", "L2", "L3", "L4")


def read(path):
    return json.loads(Path(path).read_text())


def select_cases(rows):
    identities = sorted(rows)
    positive = [key for key in identities if int(rows[key]["L0"]["gt_pixels"]) > 0]
    ranked = sorted(positive, key=lambda key: (
        np.mean([float(rows[key][n]["dice"]) for n in NAMES]), key))
    missed = [key for key in positive if any(int(rows[key][n]["tp"]) == 0 for n in NAMES)]
    empty = [key for key in identities if int(rows[key]["L0"]["gt_pixels"]) == 0]
    empty_fp = [key for key in empty if any(int(rows[key][n]["fp"]) > 0 for n in NAMES)]
    multiple = [key for key in positive if int(rows[key]["L0"]["gt_components_8_connected"]) > 1]
    return [
        ("Random seed 42", identities[int(np.random.default_rng(42).integers(len(identities)))]),
        ("Median mean Dice lima model, GT positif", ranked[len(ranked) // 2]),
        ("First ID TP=0 pada salah satu model", missed[0] if missed else None),
        ("First ID GT kosong + FP; fallback kosong", (empty_fp or empty or [None])[0]),
        ("First ID mask multi-komponen", multiple[0] if multiple else None),
        ("GT positif terkecil", min(positive, key=lambda key: (int(rows[key]["L0"]["gt_pixels"]), key))),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if read(root / "train_workflow.json")["status"] != "completed" or read(root / "final_validation.json")["status"] != "passed":
        raise ValueError("Actual completed training and artifact QA must precede image interpretation")
    output = root / "error_analysis"
    output.mkdir(exist_ok=False)
    config, manifest = read(root / "protocol.json"), read(root / "data/manifest.json")
    control = ROOT / config["phase_l"]["control_root"] / "B1"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    cases, files = {}, []
    for split in ("validation", "test"):
        rows = {}
        for name in NAMES:
            run = control if name == "L0" else root / name
            with (run / f"evaluation-{split}.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    rows.setdefault(row["name"], {})[name] = row
        if any(set(values) != set(NAMES) for values in rows.values()):
            raise ValueError("Unpaired image identities")
        choices = select_cases(rows)
        records = {r["sample_id"]: r for r in manifest["splits"][split]}
        figure, axes = plt.subplots(len(choices), 7, figsize=(20, 16), constrained_layout=True)
        cases[split] = []
        for index, (rule, identity) in enumerate(choices):
            for ax in axes[index]:
                ax.set_xticks([])
                ax.set_yticks([])
            if identity is None:
                axes[index, 0].text(.05, .5, rule + "\nTidak ada kasus", wrap=True)
                for ax in axes[index, 1:]:
                    ax.axis("off")
                cases[split].append({"rule": rule, "sample_id": None, "available": False})
                continue
            record = records[identity]
            with Image.open(ROOT / record["image_path"]) as image:
                original = np.asarray(image.convert("RGB"))
            arrays = {}
            for name in NAMES:
                run = control if name == "L0" else root / name
                with np.load(run / "predictions" / split / (Path(record["name"]).stem + ".npz"), allow_pickle=False) as data:
                    arrays[name] = {key: data[key] for key in ("prediction", "truth")}
            truth = arrays["L0"]["truth"].astype(bool)
            visible = np.any(original > 8, axis=2) | truth
            for name in NAMES:
                np.testing.assert_array_equal(arrays[name]["truth"], truth)
                visible |= arrays[name]["prediction"].astype(bool)
            ys, xs = np.where(visible)
            if len(xs):
                ymin, ymax = max(0, int(ys.min()) - 3), min(truth.shape[0], int(ys.max()) + 4)
                xmin, xmax = max(0, int(xs.min()) - 3), min(truth.shape[1], int(xs.max()) + 4)
            else:
                ymin, ymax, xmin, xmax = 0, truth.shape[0], 0, truth.shape[1]
            axes[index, 0].imshow(original)
            axes[index, 0].set_title(rule + f"\nFoto; zoom {xmax-xmin}×{ymax-ymin} pixel", fontsize=9)
            axes[index, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
            axes[index, 1].set_title(f"GT: {truth.sum()} pixel\n{truth.mean()*100:.3f}% kanvas asli", fontsize=9)
            for column, name in enumerate(NAMES, start=2):
                prediction = arrays[name]["prediction"].astype(bool)
                overlay = np.zeros((*truth.shape, 4), dtype=float)
                overlay[truth & prediction] = [.06, .72, .50, .7]
                overlay[truth & ~prediction] = [.86, .15, .15, .8]
                overlay[~truth & prediction] = [.05, .65, .91, .8]
                axes[index, column].imshow(original)
                axes[index, column].imshow(overlay)
                axes[index, column].set_title(f"{name}: Dice {float(rows[identity][name]['dice']):.3f}\nTP={rows[identity][name]['tp']} / FP={rows[identity][name]['fp']}", fontsize=9)
            for ax in axes[index]:
                ax.set_xlim(xmin - .5, xmax - .5)
                ax.set_ylim(ymax - .5, ymin - .5)
            cases[split].append({"rule": rule, "sample_id": identity, "available": True,
                                 "display_crop_xyxy": [xmin, ymin, xmax, ymax],
                                 "actual_model_and_metric_grid": [224, 224],
                                 "dice": {name: float(rows[identity][name]["dice"]) for name in NAMES}})
        figure.suptitle(f"REAL / CACHED / LOCAL ONLY — {split}, CP2 L0–L4 | TP hijau · FN merah · FP biru\n"
                        "Display zoom union RGB/GT/semua prediksi; model dan metrik tetap full 224×224; pemilihan simetris/deterministik", fontsize=13)
        path = output / f"{split}_local_gallery.png"
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        files.append({"path": str(path.relative_to(ROOT)), "sha256": digest_file(path)})
    evidence = {"status": "completed", "source": "real cached, checksum-validated original-image predictions",
                "media_policy": "Clinical source/GT/overlay galleries remain local; no clinical imagery committed",
                "selection_rules": "Same six symmetric deterministic rules, including failures; repeats across rules allowed",
                "case_selection": cases, "local_files": files}
    write_json(output / "selection.json", evidence)
    write_json(ROOT / "reports/cp2-phase-l-cases.json", evidence)
    print("Saved two LOCAL-only galleries and public numeric case-selection provenance")


if __name__ == "__main__":
    main()
