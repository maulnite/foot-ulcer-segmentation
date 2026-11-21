"""Scientific figures for CP2 slides 13–16; all numeric inputs are verified."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np

from .presentation import display_bounds, overlay
from .presentation_plots import TP, FP, FN, style

BASE, CASCADE = "#475569", "#0f766e"
COLORS = {"B0": "#94a3b8", "B1": BASE, "A6": "#7c3aed", "H2": "#2563eb",
          "B1A6H2": "#b45309", "L3A6H2": "#be185d", "Cascade": CASCADE}
DISPLAY = {"B1A6H2": "B1+A6+H2", "L3A6H2": "L3+A6+H2", "Cascade": "Cascade CP2"}


def configure():
    style()
    plt.rcParams.update({"svg.hashsalt": "cp2-final-figures", "font.size": 13})


def save(fig, folder, stem, vector=False, svg_only=False):
    files = [Path(folder)/(stem+suffix) for suffix in ([".svg"] if svg_only else [".png", ".svg"] if vector else [".png"])]
    for file in files:
        if file.exists():
            raise FileExistsError("Final figures are append-only; choose a fresh render directory")
        fig.savefig(file, dpi=160, facecolor="white", metadata={"Date": None} if file.suffix == ".svg" else None)
        if file.suffix == ".svg":
            file.write_text("\n".join(line.rstrip() for line in file.read_text().splitlines())+"\n")
    plt.close(fig)
    return files


def title(fig, heading, subtitle):
    fig.suptitle(heading, x=.5, y=.974, fontsize=24, fontweight="bold")
    fig.text(.5, .922, subtitle, ha="center", va="center", fontsize=13)


def legend(fig, y=.06, roi=False):
    handles = [Patch(color=TP, label="TP · overlap dengan GT"), Patch(color=FP, label="FP · di luar GT"),
               Patch(color=FN, label="FN · GT terlewat")]
    if roi:
        handles.append(Patch(color="#e69f00", label="ROI dari coarse ≥0,3"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=len(handles),
               frameon=False, fontsize=12)


def viewport(ax, bounds):
    x0, y0, x1, y1 = bounds
    ax.set_xlim(x0-.5, x1-.5)
    ax.set_ylim(y1-.5, y0-.5)
    ax.axis("off")


def walkthrough(folder, cases, zoom=False, masks=False):
    configure()
    fig = plt.figure(figsize=(20, 12))
    axes = [[fig.add_axes([.035+j*.16, bottom, .145, .22]) for j in range(6)]
            for bottom in (.545, .185)]
    title(fig, "Cascade CP2 · telusuri prediksi hingga refinement",
          "Chronic Wound · validation n=118 · B1 epoch25 (frozen) + refiner epoch46 · REAL / CACHED + FRESH CPU")
    mode = "ZOOM tampilan saja; ROI utuh tetap terlihat" if zoom else "Kanvas evaluasi utuh 224×224"
    fig.text(.5, .868, mode+" · proposal tanpa GT · seluruh crop aktual ditampilkan", ha="center", fontsize=13)
    for i, case in enumerate(cases):
        rgb, truth = case["rgb"], case["truth"]
        first, final = case["coarse"] >= .5, case["final"] >= .5
        if len(case["boxes"]) != 1:
            raise ValueError("This six-panel layout requires one actual crop; never hide additional crops")
        bounds = display_bounds(rgb, truth, [first, final]) if zoom else (0, 0, 224, 224)
        if zoom:
            x0, y0, x1, y1 = case["boxes"][0]
            bounds = (min(bounds[0], x0), min(bounds[1], y0), max(bounds[2], x1), max(bounds[3], y1))
        axes[i][0].imshow(rgb)
        axes[i][1].imshow(first if masks else overlay(rgb, truth, first), cmap="gray", vmin=0, vmax=1)
        axes[i][2].imshow(rgb)
        x0, y0, x1, y1 = case["boxes"][0]
        axes[i][2].add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False, edgecolor="#e69f00", linewidth=2.5))
        axes[i][3].imshow(case["crops"][0])
        axes[i][4].imshow(case["crop_probabilities"][0] >= .5, cmap="gray", vmin=0, vmax=1)
        axes[i][5].imshow(final if masks else overlay(rgb, truth, final), cmap="gray", vmin=0, vmax=1)
        b, c = case["metrics"]["B1"], case["metrics"]["Cascade"]
        headings = ["1 · Original image", f"2 · Coarse B1\nDice {b['dice']*100:.2f}%",
                    "3 · Proposal bounding box", "4 · Actual normalized crop\n→ RGB untuk tampilan",
                    "5 · Refiner prediction\nMask crop ≥0,5 (display)", f"6 · Final stitched mask\nDice {c['dice']*100:.2f}%"]
        for j, ax in enumerate(axes[i]):
            ax.set_title(headings[j], fontsize=13, pad=10)
            viewport(ax, (0, 0, 224, 224) if j in (3, 4) else bounds)
        y = .825 if i == 0 else .465
        fig.text(.04, y, f"{case['label']} · ID {Path(case['sample_id']).stem} · GT {int(truth.sum())} piksel",
                 fontsize=13, fontweight="bold")
        boxes = f"ROI xyxy=[{x0},{y0},{x1},{y1}) · resize {x1-x0}×{y1-y0} →224×224"
        detail = f"{boxes} · 2 forward · Δ Image Dice {(c['dice']-b['dice'])*100:+.2f} pp · FP {b['fp']}→{c['fp']} · FN {b['fn']}→{c['fn']}"
        fig.text(.5, .51 if i == 0 else .15, detail, ha="center", fontsize=12)
    if masks:
        fig.text(.5, .115, "Mask aktual: putih=foreground. Crop/refiner memakai grid crop; coarse/final memakai grid foto utuh.", ha="center", fontsize=12)
    else:
        legend(fig, y=.085, roi=True)
    fig.text(.5, .058, "Proposal ≥0,3 · 8-connected ≥4px · max5 ROI · margin0,5 · minimum sisi0,25 · final ≥0,5. Stitch memakai probability, bukan mask crop biner.", ha="center", fontsize=11.5)
    fig.text(.5, .03, "Overlap: maximum probability; di luar ROI: coarse. Dua kasus ekstrem posthoc, bukan frekuensi keberhasilan. Tidak ada retraining.", ha="center", fontsize=11.5)
    suffix = "_masks" if masks else "_zoom" if zoom else ""
    return save(fig, folder, "13_cascade_cp2_final_walkthrough"+suffix)


def architecture(folder, parameters):
    configure()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")

    def box(x, y, w, h, text, color="#edf2f7", edge=BASE, size=13):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10", facecolor=color,
                                   edgecolor=edge, linewidth=1.5))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size)

    def arrow(a, b, label=None):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, color=BASE, linewidth=1.7))
        if label:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.14, label, ha="center", fontsize=11)

    ax.text(8, 8.65, "Cascade CP2 · training dan inference memakai sumber crop berbeda", ha="center", fontsize=21, fontweight="bold")
    ax.text(.1, 7.9, "TRAINING · 671 foto · refiner saja diperbarui", fontsize=15, color=CASCADE, fontweight="bold")
    box(.2, 6.1, 2.2, 1.1, "RGB + GT\ntraining saja")
    box(3.2, 6.0, 3.0, 1.3, "GT-based crop\nmargin0,3–0,8\n20% random crop")
    box(7.0, 6.0, 3.0, 1.3, "REFINER · trainable\nCP2 initializer\nB1 corrected focal + Dice", "#d9f0e7", CASCADE, 12)
    box(11.0, 6.0, 4.4, 1.3, "Checkpoint: full-image validation\npredicted proposals saja\nearliest best Image Dice → epoch46", "#e8eefb", size=12)
    for a, b in [((2.5, 6.65), (3.1, 6.65)), ((6.3, 6.65), (6.9, 6.65)), ((10.1, 6.65), (10.9, 6.65))]:
        arrow(a, b)
    ax.plot([.1, 15.7], [5.4, 5.4], color="#cbd5e1", linewidth=1)
    ax.text(.1, 4.95, "VALIDATION / TEST / EXTERNAL · tidak ada GT dalam jalur inference", fontsize=15, color=BASE, fontweight="bold")
    box(.2, 3.0, 1.6, 1.2, "RGB\n224×224")
    box(2.3, 2.95, 2.4, 1.3, "COARSE B1\nFROZEN · epoch25\nprobability p", "#edf1f5", size=12)
    box(5.3, 2.8, 3.0, 1.6, "Predicted proposals\np ≥0,3 · 8-connected\n≥4px · max5 ROI\nmargin0,5 · minside0,25", "#fff3d9", "#a36b00", 11.5)
    box(8.9, 2.95, 2.8, 1.3, "Actual RGB crops\nresize224×224\nREFINER epoch46", "#d9f0e7", CASCADE, 12)
    box(12.3, 2.8, 3.1, 1.6, "Restore probability to ROI\nOverlap: max probability\nOutside ROI: coarse\nFinal threshold ≥0,5", "#d9f0e7", CASCADE, 11.5)
    for a, b in [((1.9, 3.6), (2.2, 3.6)), ((4.8, 3.6), (5.2, 3.6)), ((8.4, 3.6), (8.8, 3.6)), ((11.8, 3.6), (12.2, 3.6))]:
        arrow(a, b)
    ax.plot([3.5, 3.5, 13.8], [2.85, 2.25, 2.25], color=BASE, linewidth=1.5)
    arrow((13.8, 2.25), (13.8, 2.75))
    ax.text(8.5, 1.95, "Coarse probability dipertahankan di luar seluruh ROI; tanpa proposal → hasil coarse", ha="center", fontsize=12)
    text = f"{parameters['coarse_active']:,} coarse + {parameters['refiner_active']:,} refiner = {parameters['total_active']:,} parameter aktif"
    ax.text(8, 1.3, text+" · 1 + ROI forward/foto", ha="center", fontsize=13, fontweight="bold")
    ax.text(8, .85, "Budget attention tunggal ≤1,01× B1 TIDAK berlaku untuk Cascade dua model. Tambahan kapasitas adalah confound.", ha="center", fontsize=11.5)
    ax.text(8, .35, "SCHEMATIC · implementasi aktual terverifikasi · source cohort 671/118/261 · bukan patient-level atau validasi klinis", ha="center", fontsize=11.5)
    fig.subplots_adjust(left=.025, right=.99, top=.99, bottom=.01)
    return save(fig, folder, "13_cascade_cp2_architecture", svg_only=True)


def scoreboard(folder, names, metrics, metadata):
    configure()
    fig, ax = plt.subplots(figsize=(20, 12))
    fig.subplots_adjust(left=.16, right=.98, top=.81, bottom=.20)
    ypos = np.arange(len(names))
    values = [metrics[n]["validation"]["image_dice"]*100 for n in names]
    ax.barh(ypos, values, color=[COLORS.get(n, "#7b929b") for n in names], height=.68)
    ax.set(xlim=(0, 100), yticks=ypos, yticklabels=[f"{DISPLAY.get(n, n)} · e{metadata[n]['epoch']}" for n in names],
           xlabel="Validation Mean Image Dice (%)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    for y, value in zip(ypos, values):
        ax.text(value+.65, y, f"{value:.2f}%", va="center", fontsize=12)
    ax.axvline(metrics["B1"]["validation"]["image_dice"]*100, color=BASE, linestyle="--", alpha=.65)
    for tick, n in zip(ax.get_yticklabels(), names):
        if n in COLORS:
            tick.set_fontweight("bold")
            tick.set_color(COLORS[n])
    title(fig, "CP2 final · 18 kondisi completed pada validation yang identik",
          "Chronic Wound · train671 / validation118 / source-test261 · seed42 · REAL / CACHED · threshold ≥0,5")
    fig.text(.5, .867, "Urutan keluarga eksperimen; bukan ranking ulang berdasarkan source-test atau external. Guard ditampilkan terpisah.", ha="center", fontsize=13)
    fig.text(.5, .137, "B1 = acuan awal · A6/H2 = standalone menjanjikan, guard historis gagal · kombinasi terbaru tetap ditampilkan", ha="center", fontsize=13)
    fig.text(.5, .096, "Cascade lolos lima guard validation tersendiri; dua model / 129,80M parameter aktif. Budget attention 1,01× tidak berlaku.", ha="center", fontsize=13, color=CASCADE)
    fig.text(.5, .056, "B1 historis GPU 84,2765%; replay coarse CPU berpasangan Cascade 84,2773%. Selisih kecil ini dicatat; epoch tidak dipilih ulang.", ha="center", fontsize=11.5)
    fig.text(.5, .024, "Tidak mencampur CP1, Cascade Ganta asli, atau CO2. RGB-group split ≠ patient-level; repeated validation, eksploratif.", ha="center", fontsize=11.5)
    return save(fig, folder, "14_cp2_final_scoreboard", vector=True)


def guard_tradeoffs(folder, names, metrics, metadata, paired_base):
    configure()
    fig, ax = plt.subplots(figsize=(20, 12))
    fig.subplots_adjust(left=.15, right=.99, top=.79, bottom=.21)
    cells = np.zeros((len(names), 7, 4))
    labels = []
    for i, n in enumerate(names):
        m = metrics[n]["validation"]
        guards = metadata[n]["guards"]
        state = "ACUAN" if n == "B1" else "N/A" if not guards else "LULUS" if all(guards.values()) else "GAGAL"
        row = [f"{m[k]*100:.2f}%" for k in ("image_dice", "small_dice", "boundary_iou")]
        row += [str(m["misses"]), f"{m['empty_fp_images']}/{m['empty_images']}", state,
                "N/A · 2 model" if n == "Cascade" else "acuan" if n == "B1" else
                "lulus" if guards.get("capacity_budget") is True else "N/A"]
        base = paired_base if n == "Cascade" else metrics["B1"]["validation"]
        for j, key in enumerate(("image_dice", "small_dice", "boundary_iou", "misses", "empty_fp_images")):
            delta = m[key]-base[key]
            good = delta > 0 if j < 3 else delta < 0
            cells[i, j] = matplotlib.colors.to_rgba("#d9f0e7" if good else "#f8ded5" if delta else "#edf1f5")
        cells[i, 5] = matplotlib.colors.to_rgba("#d9f0e7" if state == "LULUS" else "#f8ded5" if state == "GAGAL" else "#edf1f5")
        cells[i, 6] = matplotlib.colors.to_rgba("#fff3d9" if n == "Cascade" else "#edf1f5")
        labels.append(row)
    ax.imshow(cells, aspect="auto", interpolation="nearest")
    ax.set(xticks=range(7), xticklabels=["Image Dice\n(%)", "Small Dice\n(%)", "Boundary IoU\n(%)", "GT+ TP=0\n(foto/111)", "GT-empty FP\n(foto/7)", "Guard asli\nfase terkait", "Budget ≤1,01×\nsingle-model"],
           yticks=range(len(names)), yticklabels=[DISPLAY.get(n, n) for n in names])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=10)
    for i, row in enumerate(labels):
        for j, label in enumerate(row):
            ax.text(j, i, label, ha="center", va="center", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(fig, "CP2 final · skor, error, dan guard penelitian dibaca bersama",
          "Validation n=118 · GT+111 / GT kosong7 / small36 · seed42 · REAL / CACHED · final threshold ≥0,5")
    fig.text(.5, .86, "Nilai absolut; warna relatif B1 (Cascade: coarse CPU berpasangan). Hijau membaik, jingga memburuk, abu-abu sama.", ha="center", fontsize=12)
    fig.text(.5, .144, "L/A/H/N/C mempertahankan guard aslinya; kombinasi baru juga harus melampaui standalone relevan. Tidak memilih ulang hasil historis.", ha="center", fontsize=12)
    fig.text(.5, .10, "Cascade: overall naik; small/boundary tidak turun; miss dan empty FP tidak bertambah. 129,80M parameter aktif / 1+n ROI forward.", ha="center", fontsize=12, color=CASCADE)
    fig.text(.5, .06, "Small: 0<GT≤1% kanvas224. TP=0 = tidak ada overlap dengan GT positif. FP pada GT kosong bukan diagnosis sehat.", ha="center", fontsize=12)
    fig.text(.5, .025, "Tidak semua keluarga memakai aturan seleksi identik. Kolom budget bukan syarat Cascade; tidak ada klaim patient-level atau signifikansi.", ha="center", fontsize=11.5)
    return save(fig, folder, "14_cp2_final_guard_tradeoffs", vector=True)


def external_comparison(folder, metrics):
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [2, 1]})
    fig.subplots_adjust(left=.075, right=.97, top=.78, bottom=.27, wspace=.27)
    keys = ["image_dice", "data_dice", "small_dice", "boundary_iou"]
    for name, offset, color, label in [("coarse", -.18, BASE, "B1 coarse · e25"), ("cascade", .18, CASCADE, "Cascade CP2 · e46")]:
        values = [metrics[name][k]*100 for k in keys]
        bars = axes[0].bar(np.arange(4)+offset, values, width=.36, color=color, label=label)
        axes[0].bar_label(bars, fmt="%.2f", padding=4, fontsize=11)
    axes[0].set(xticks=range(4), xticklabels=["Image\nDice", "Data\nDice", "Small\nDice", "Boundary\nIoU"],
                ylim=(0, 100), ylabel="Skor (%)", title="Metrik agregat meningkat")
    axes[0].legend(frameon=False, fontsize=11, loc="upper right")
    values = [metrics[n]["misses"] for n in ("coarse", "cascade")]
    bars = axes[1].bar([0, 1], values, color=[BASE, CASCADE], width=.55)
    axes[1].bar_label(bars, labels=[f"{v}/558\n({v/558*100:.1f}%)" for v in values], padding=6, fontsize=12)
    axes[1].set(xticks=[0, 1], xticklabels=["B1 coarse", "Cascade"], ylim=(0, 210),
                ylabel="GT+ tanpa overlap (foto)", title="Tetapi total miss bertambah")
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    title(fig, "CO2Wounds-V2 · skor naik, tetapi lebih banyak luka terlewat",
          "558 gambar berlabel yang sama · native-resolution evaluation · REAL / CACHED · threshold ≥0,5")
    fig.text(.5, .865, "Coarse B1 dan Cascade identik input/cohort; tidak ada training, tuning atau pemilihan model dari CO2.", ha="center", fontsize=12)
    fig.text(.5, .185, f"Δ Image Dice {(metrics['cascade']['image_dice']-metrics['coarse']['image_dice'])*100:+.2f} pp  ·  Δ total miss {values[1]-values[0]:+d} foto", ha="center", fontsize=17, fontweight="bold")
    fig.text(.5, .12, "Seluruh GT positif: tidak ada subgroup GT-empty. Small berdasarkan persentase kanvas native, bukan ukuran fisik.", ha="center", fontsize=11.5)
    fig.text(.5, .077, "Sumber dan patologi berubah bersama; grid native berbeda dari source224. External sudah pernah diperiksa, bukan confirmatory test baru.", ha="center", fontsize=11)
    fig.text(.5, .035, "Grafik agregat tanpa foto/mask pasien. Cascade dipilih melalui validation source; hasil eksternal tidak mengubah keputusan.", ha="center", fontsize=11)
    return save(fig, folder, "15_external_cascade_comparison")


def external_tradeoffs(folder, transitions, rows):
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.075, right=.97, top=.78, bottom=.28, wspace=.32)
    matrix = np.array([[transitions["overlap_both"], transitions["newly_missed"]],
                       [transitions["recovered"], transitions["both_missed"]]])
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=558)
    axes[0].set(xticks=[0, 1], xticklabels=["Ada overlap", "TP=0 (miss)"],
                yticks=[0, 1], yticklabels=["Ada overlap", "TP=0 (miss)"],
                xlabel="Cascade CP2", ylabel="Coarse B1", title="Transisi error pada gambar yang sama")
    for y in range(2):
        for x in range(2):
            axes[0].text(x, y, f"{matrix[y,x]} foto\n{matrix[y,x]/558*100:.1f}%", ha="center", va="center", fontsize=18,
                         color="white" if matrix[y, x] > 300 else "#18324b")
    deltas = np.array([rows["cascade"][s]["dice"]-rows["coarse"][s]["dice"] for s in rows["coarse"]])
    values = [(deltas > 0).sum(), (deltas == 0).sum(), (deltas < 0).sum()]
    bars = axes[1].bar([0, 1, 2], values, color=[CASCADE, "#94a3b8", FP], width=.6)
    axes[1].bar_label(bars, labels=[str(v) for v in values], padding=5, fontsize=14)
    axes[1].set(xticks=[0, 1, 2], xticklabels=["Dice naik", "Sama", "Dice turun"],
                ylim=(0, 558), ylabel="Jumlah gambar (dari558)", title="Rerata tidak mewakili setiap gambar")
    axes[1].grid(axis="y", alpha=.15)
    axes[1].set_axisbelow(True)
    title(fig, "CO2Wounds-V2 · siapa yang terbantu, siapa yang terlewat?",
          "558 pasangan prediksi asli · REAL / CACHED · satu foto = satu unit · bukan unit pixel atau crop")
    fig.text(.5, .865, "Total miss berarti GT positif dengan TP=0; model dapat tetap menghasilkan FP di lokasi lain.", ha="center", fontsize=12)
    fig.text(.5, .17, f"{transitions['recovered']} miss coarse pulih · {transitions['newly_missed']} miss baru setelah refinement · net {transitions['newly_missed']-transitions['recovered']:+d}", ha="center", fontsize=17, fontweight="bold")
    fig.text(.5, .107, "Distribusi paired Dice memakai seluruh558 gambar; bukan galeri contoh terbaik. Tidak ada GT-empty untuk mengukur penolakan negative images.", ha="center", fontsize=11)
    fig.text(.5, .063, "Domain, patologi dan grid evaluasi berbeda dari source. Target pernah diperiksa; hasil deskriptif dan tidak digunakan memilih konfigurasi.", ha="center", fontsize=11)
    fig.text(.5, .025, "Agregat saja: tidak mempublikasikan foto pasien atau derived clinical panels CO2Wounds.", ha="center", fontsize=11)
    return save(fig, folder, "15_external_cascade_tradeoffs")


def gallery(folder, cases, zoom=False, masks=False):
    configure()
    names = ["B1", "A6", "H2", "Cascade"]
    fig, axes = plt.subplots(4, 6, figsize=(20, 13))
    fig.subplots_adjust(left=.03, right=.985, top=.81, bottom=.11, hspace=.6, wspace=.08)
    title(fig, "Kasus nyata · model berbeda, foto dan GT yang sama",
          "Chronic Wound · validation n=118 · B1 e25 / A6 e110 / H2 e50 / refiner e46 · REAL / CACHED")
    fig.text(.5, .867, "ZOOM tampilan: viewport identik dan mempertahankan seluruh FP/FN" if zoom else "FULL FIELD · grid evaluasi 224×224 · threshold ≥0,5", ha="center", fontsize=13)
    for i, case in enumerate(cases):
        rgb, truth, predictions = case["rgb"], case["truth"], case["predictions"]
        bounds = display_bounds(rgb, truth, list(predictions.values())) if zoom else (0, 0, 224, 224)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"{case['label']}\nID {Path(case['sample_id']).stem[:12]}", fontsize=12)
        axes[i, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Ground truth\n{int(truth.sum())} px · {truth.mean()*100:.3f}%", fontsize=12)
        for j, name in enumerate(names, start=2):
            m = case["metrics"][name]
            axes[i, j].imshow(predictions[name] if masks else overlay(rgb, truth, predictions[name]), cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(f"{DISPLAY.get(name,name)} · Dice {m['dice']*100:.2f}%\nFP {m['fp']} · FN {m['fn']}", fontsize=12, color=COLORS[name])
        for ax in axes[i]:
            viewport(ax, bounds)
    if masks:
        fig.text(.5, .065, "Mask aktual: putih=foreground, hitam=background. B1 adalah coarse CPU berpasangan Cascade; A6/H2 cache GPU historis.", ha="center", fontsize=12)
    else:
        legend(fig, y=.05)
    fig.text(.5, .031, "Empat contoh ekstrem posthoc dipilih deterministik; bukan prevalensi atau seleksi model. GT kosong ≠ pasien sehat. Full sample IDs ada di metadata.", ha="center", fontsize=11)
    fig.text(.5, .011, "B1 memakai replay coarse CPU yang sama dengan Cascade. Seluruh metrik per gambar dihitung pada kanvas penuh, termasuk pada versi zoom.", ha="center", fontsize=10.5)
    suffix = "_masks" if masks else "_zoom" if zoom else ""
    return save(fig, folder, "16_cross_model_real_cases"+suffix)
