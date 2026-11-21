"""Posthoc descriptive contrasts; never changes training, selection or metrics."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_l_report import dice_contributions  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / "posthoc_analysis.json"
    if destination.exists():
        raise FileExistsError("Preserve completed posthoc evidence")
    if read(root / "final_validation.json")["status"] != "passed":
        raise ValueError("Completed real-data artifact QA required")
    source = ROOT / "reports/cp2-phase-l-results.json"
    result = read(source)
    if result["run_root"] != str(root.relative_to(ROOT)):
        raise ValueError("Public results belong to another run")
    metrics = result["metrics"]
    analysis = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "COMPLETED — DESCRIPTIVE POSTHOC ANALYSIS",
                "source_results_sha256": digest_file(source),
                "run_root": result["run_root"], "training_changed": False,
                "selection_changed": False, "dice_contributions": {}, "recipe_contrasts": {}}
    names = ("L1", "L2", "L3", "L4")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, split in zip(axes, ("validation", "test")):
        baseline = metrics["L0"][split]
        pieces = {name: dice_contributions(metrics[name][split], baseline) for name in names}
        analysis["dice_contributions"][split] = pieces
        positions = np.arange(len(names))
        ax.barh(positions - .15, [100 * pieces[n]["gt_positive"] for n in names],
                height=.25, color="#2563EB", label="Kontribusi GT positif")
        ax.barh(positions + .15, [100 * pieces[n]["gt_empty"] for n in names],
                height=.25, color="#DC2626", label="Kontribusi GT kosong")
        ax.scatter([100 * pieces[n]["total"] for n in names], positions,
                   marker="D", s=35, color="#1E293B", label="Delta keseluruhan", zorder=4)
        ax.axvline(0, color="#64748B", linestyle="--", linewidth=1)
        ax.set(yticks=positions, yticklabels=names, xlabel="Kontribusi terhadap Δ Image Dice vs L0 (poin persentase)",
               ylabel="Objective", title=f"{split}: {baseline['gt_positive']['images']} GT positif + {baseline['gt_empty']['images']} kosong")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
        analysis["recipe_contrasts"][split] = {}
        for first, second in (("L1", "L0"), ("L2", "L1"), ("L3", "L0"), ("L4", "L2")):
            a, b = metrics[first][split], metrics[second][split]
            analysis["recipe_contrasts"][split][f"{first}-{second}"] = {
                "image_dice": a["all"]["image_based"]["dice"] - b["all"]["image_based"]["dice"],
                "data_dice": a["all"]["data_based"]["dice"] - b["all"]["data_based"]["dice"],
                "data_precision": a["all"]["data_based"]["precision"] - b["all"]["data_based"]["precision"],
                "data_recall": a["all"]["data_based"]["recall"] - b["all"]["data_based"]["recall"],
                "small_image_dice": a["subgroups"]["small"]["image_based"]["dice"] - b["subgroups"]["small"]["image_based"]["dice"],
                "boundary_iou": a["all"]["image_based"]["boundary_iou"] - b["all"]["image_based"]["boundary_iou"]}
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.06), ncol=3, frameon=False)
    figure.suptitle("REAL / CACHED — Chronic Wound, seed 42 | Δ semua = kontribusi GT positif + GT kosong", fontsize=14)
    figures = []
    for extension in ("png", "svg"):
        path = ROOT / "reports/figures/cp2-phase-l" / f"dice_contributions.{extension}"
        if path.exists():
            raise FileExistsError("Preserve existing diagnostic figure")
        figure.savefig(path, dpi=160, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
        figures.append({"path": str(path.relative_to(ROOT)), "sha256": digest_file(path)})
    plt.close(figure)
    analysis.update({"figures": figures,
                     "interpretation": "Weighted positive/empty mean differences exactly sum to overall mean Dice difference; no subgroup exclusion or evaluator change",
                     "caveat": "Posthoc explanation, not new training/model selection. Unknown patient grouping, one seed, adaptive checkpoint budgets; exposed source-test is descriptive."})
    write_json(destination, analysis)
    write_json(ROOT / "reports/cp2-phase-l-analysis.json", analysis)
    print(analysis["status"])


if __name__ == "__main__":
    main()
