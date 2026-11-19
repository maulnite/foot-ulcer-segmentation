#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisis_empat_ide.py
=====================
Analisis untuk empat ide lanjutan segmentasi luka kecil (FUSegNet):

  A1  Diagnosis      : kegagalan lebih karena UKURAN luka atau KONTRAS luka-kulit?
  A2  Cascade        : berapa banyak komponen luka yang TERLEWAT TOTAL oleh model coarse
                       (cascade berbasis crop dari prediksi tidak bisa memperbaikinya)?
  A3  Copy-Paste     : sebaran ukuran luka di train, ukuran "kolam" luka kecil,
                       dan faktor oversampling yang diperlukan.
  A4  CL-attention   : bandingkan tiap model vs baseline secara berpasangan per citra
                       (Wilcoxon + bootstrap CI), per ukuran dan per kontras.

Semua dihitung dari mask:
  data/raw/<dataset>/<split>/{images,labels}/<nama>.png
  output_<dataset>/predictions/<model>_..._<timestamp>/<nama>.png

Cara pakai (dari folder notebook/ atau dari root proyek, base-dir dideteksi otomatis):
  python analisis_empat_ide.py --dataset both
  python analisis_empat_ide.py --dataset fuseg_2021 --models baseline,cl_dpsc,bw_ftl,cascade
  python analisis_empat_ide.py --dataset chronic_wound --run baseline=../output_chronic_wound/predictions/<folder>

Keluaran: <base-dir>/analisis_empat_ide/<dataset>/ (laporan.md, CSV, PNG)
Dependensi: numpy, pandas, scipy, pillow, matplotlib
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# KONFIGURASI (sesuai struktur proyek)
# ----------------------------------------------------------------------------
DATASETS = {
    "fuseg_2021": dict(raw="data/raw/fuseg_2021", eval_split="validation",
                       out="output_fuseg_2021"),
    "chronic_wound": dict(raw="data/raw/chronic_wound", eval_split="test",
                          out="output_chronic_wound"),
}
SMALL_FRAC = 0.01                      # "luka kecil": luas > 0 dan < 1% citra (sama dengan notebook)
AREA_EDGES = [0.0, 0.001, 0.01, 0.05, 1.01]
AREA_LABELS = ["<0.1%", "0.1-1%", "1-5%", ">5%"]
S8 = np.ones((3, 3), dtype=int)        # konektivitas-8
EPS = 1e-6


# ----------------------------------------------------------------------------
# UTIL I/O
# ----------------------------------------------------------------------------
def read_mask(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a.max(axis=2)
    thr = 127 if a.max() > 1 else 0
    return a > thr


def read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def index_dir(d, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    return {p.stem: p for p in sorted(Path(d).glob("*")) if p.suffix.lower() in exts}


def rgb2lab(rgb):
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (c @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def wound_contrast(img, gt):
    """Kontras luka vs cincin kulit di sekelilingnya (Lab). Return (deltaE, delta a*)."""
    area = int(gt.sum())
    if area == 0:
        return np.nan, np.nan
    r = int(np.clip(round(0.25 * np.sqrt(area)), 4, 25))
    ring = ndi.binary_dilation(gt, iterations=r) & ~gt
    if ring.sum() < 10:
        return np.nan, np.nan
    lab = rgb2lab(img)
    dw, dr = lab[gt].mean(0), lab[ring].mean(0)
    return float(np.linalg.norm(dw - dr)), float(dw[1] - dr[1])


def area_bin(frac):
    return pd.cut(frac, AREA_EDGES, labels=AREA_LABELS, right=False)


def fmt(df, nd=3):
    return "```\n" + df.round(nd).to_string() + "\n```\n"


# ----------------------------------------------------------------------------
# DATA GT (evaluasi)
# ----------------------------------------------------------------------------
def load_eval(base, cfg, min_comp):
    d = base / cfg["raw"] / cfg["eval_split"]
    imgs, labs = index_dir(d / "images"), index_dir(d / "labels")
    names = sorted(set(imgs) & set(labs))
    if not names:
        sys.exit(f"[ERROR] tidak ada pasangan images/labels di {d}")
    rows, gts = [], {}
    for s in names:
        gt = read_mask(labs[s])
        gts[s] = gt
        frac = gt.sum() / gt.size
        lab, n = ndi.label(gt, S8)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        de, da = wound_contrast(read_rgb(imgs[s]), gt)
        rows.append(dict(image=s, area_frac=frac, n_comp=int((sizes >= min_comp).sum()),
                         contrast_dE=de, contrast_da=da, H=gt.shape[0], W=gt.shape[1]))
    ev = pd.DataFrame(rows)
    ev["area_bin"] = area_bin(ev.area_frac)
    ev["small"] = (ev.area_frac > 0) & (ev.area_frac < SMALL_FRAC)
    return ev, gts


# ----------------------------------------------------------------------------
# EVALUASI MODEL DARI PNG PREDIKSI
# ----------------------------------------------------------------------------
def eval_model(run, gts, min_comp):
    rows, comps, fps = [], [], []
    missing = 0
    for s, gt in gts.items():
        p = run / f"{s}.png"
        if not p.exists():
            missing += 1
            continue
        pr = read_mask(p)
        if pr.shape != gt.shape:
            pr = np.array(Image.fromarray(pr.astype(np.uint8) * 255)
                          .resize(gt.shape[::-1], Image.NEAREST)) > 127
        tp = int((gt & pr).sum()); fp = int((~gt & pr).sum()); fn = int((gt & ~pr).sum())
        g, pp = tp + fn, tp + fp
        dice = 1.0 if g + pp == 0 else 2 * tp / (2 * tp + fp + fn)
        if g == 0:
            cat = "no_gt"
        elif tp == 0:
            cat = "miss_total"
        elif tp / g < 0.5:
            cat = "under_seg"
        elif tp / pp < 0.5:
            cat = "fp_heavy"
        else:
            cat = "ok"
        if pp > 0:
            ys, xs = np.where(pr)
            bbox = (ys.min(), ys.max(), xs.min(), xs.max())
        else:
            bbox = None
        labg, ng = ndi.label(gt, S8)
        sz = np.bincount(labg.ravel(), minlength=ng + 1)
        ov = np.bincount(labg[pr].ravel(), minlength=ng + 1)
        cms = ndi.center_of_mass(gt, labg, range(1, ng + 1)) if ng else []
        for i in range(1, ng + 1):
            if sz[i] >= min_comp:
                comps.append(dict(image=s, size_px=int(sz[i]), size_frac=sz[i] / gt.size,
                                  hit=bool(ov[i] > 0), cover=ov[i] / sz[i],
                                  cy=cms[i - 1][0], cx=cms[i - 1][1]))
        labp, npc = ndi.label(pr, S8)
        szp = np.bincount(labp.ravel(), minlength=npc + 1)
        ovp = np.bincount(labp[gt].ravel(), minlength=npc + 1)
        for i in range(1, npc + 1):
            if szp[i] >= min_comp and ovp[i] == 0:
                fps.append(dict(image=s, size_px=int(szp[i])))
        rows.append(dict(image=s, tp=tp, fp=fp, fn=fn, dice=dice,
                         precision=tp / pp if pp else np.nan,
                         recall=tp / g if g else np.nan, cat=cat,
                         n_fp_comp=sum(1 for f in fps if f["image"] == s),
                         pred_empty=pp == 0, bbox=bbox))
    if missing:
        print(f"   [WARN] {missing} citra tidak punya PNG prediksi di {run.name}")
    return pd.DataFrame(rows), pd.DataFrame(comps), pd.DataFrame(fps)


def find_run(pred_root, key, override):
    if key in override:
        return Path(override[key])
    cands = [d for d in pred_root.glob(f"{key}_*") if d.is_dir() and any(d.glob("*.png"))]
    if not cands:
        return None
    return sorted(cands, key=lambda d: d.name[-19:])[-1]


# ----------------------------------------------------------------------------
# A1  DIAGNOSIS: ukuran vs kontras
# ----------------------------------------------------------------------------
def a1_diagnosis(ev, res, mdl, outdir, rep):
    rep.append(f"## A1. Diagnosis kegagalan (model: `{mdl}`)\n")
    d = res[mdl][0].merge(ev, on="image")
    d = d[d.area_frac > 0].copy()
    d["deficit"] = 1 - d.dice
    t = d.groupby("area_bin", observed=True).agg(
        n=("dice", "size"), dice_mean=("dice", "mean"), dice_median=("dice", "median"),
        miss_total=("cat", lambda x: (x == "miss_total").mean()),
        under_seg=("cat", lambda x: (x == "under_seg").mean()),
        fp_heavy=("cat", lambda x: (x == "fp_heavy").mean()),
        deficit_sum=("deficit", "sum"))
    t["share_of_total_deficit"] = t.deficit_sum / t.deficit_sum.sum()
    t = t.drop(columns="deficit_sum")
    t.to_csv(outdir / "a1_per_area_bin.csv")
    rep.append("Dice dan jenis kegagalan per kelas luas luka (share_of_total_deficit = porsi "
               "total kehilangan Dice yang berasal dari kelas itu):\n")
    rep.append(fmt(t))

    rho_a, p_a = spearmanr(np.log10(d.area_frac), d.dice)
    dc = d.dropna(subset=["contrast_dE"])
    rho_c, p_c = spearmanr(dc.contrast_dE, dc.dice)
    rho_s, p_s = spearmanr(dc.loc[dc.small, "contrast_dE"], dc.loc[dc.small, "dice"]) \
        if dc.small.sum() > 5 else (np.nan, np.nan)
    corr = pd.DataFrame([
        dict(pasangan="Dice vs log10(luas), semua", rho=rho_a, p=p_a, n=len(d)),
        dict(pasangan="Dice vs kontras dE, semua", rho=rho_c, p=p_c, n=len(dc)),
        dict(pasangan="Dice vs kontras dE, hanya luka kecil", rho=rho_s, p=p_s,
             n=int(dc.small.sum()))])
    corr.to_csv(outdir / "a1_korelasi.csv", index=False)
    rep.append("Korelasi Spearman:\n")
    rep.append(fmt(corr, 4))

    med = dc.contrast_dE.median()
    dc = dc.assign(ukuran=np.where(dc.small, "kecil(<1%)", "besar"),
                   kontras=np.where(dc.contrast_dE < med, "rendah", "tinggi"))
    g = dc.groupby(["ukuran", "kontras"]).agg(n=("dice", "size"), dice_mean=("dice", "mean"),
                                              miss_total=("cat", lambda x: (x == "miss_total").mean()))
    g.to_csv(outdir / "a1_ukuran_x_kontras.csv")
    rep.append(f"Tabel 2x2 ukuran x kontras (batas kontras = median dE {med:.1f}). "
               "Kalau 'kontras rendah' jauh lebih buruk di kedua baris, penyebabnya kontras, bukan ukuran:\n")
    rep.append(fmt(g))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].scatter(d.area_frac * 100, d.dice, s=10, alpha=.6)
    ax[0].set_xscale("log"); ax[0].set_xlabel("luas luka (% citra)"); ax[0].set_ylabel("Dice per citra")
    ax[0].set_title("Dice vs ukuran")
    ax[1].scatter(dc.contrast_dE, dc.dice, s=10, alpha=.6, c=np.log10(dc.area_frac), cmap="viridis")
    ax[1].set_xlabel("kontras luka-kulit (dE Lab)"); ax[1].set_title("Dice vs kontras (warna = log luas)")
    cats = d.groupby(["area_bin", "cat"], observed=True).size().unstack(fill_value=0)
    (cats.div(cats.sum(1), axis=0)).plot(kind="bar", stacked=True, ax=ax[2], legend=True)
    ax[2].set_title("Jenis kegagalan per kelas luas"); ax[2].set_xlabel("")
    plt.tight_layout(); plt.savefig(outdir / "a1_diagnosis.png", dpi=130); plt.close()
    rep.append("Gambar: `a1_diagnosis.png`\n")


# ----------------------------------------------------------------------------
# A2  CASCADE: komponen luka yang terlewat total
# ----------------------------------------------------------------------------
def a2_cascade(ev, res, coarse_key, casc_key, outdir, rep, margin=0.2):
    rep.append(f"## A2. Batas atas cascade berbasis proposal prediksi "
               f"(coarse = `{coarse_key}`, refiner = `{casc_key}`)\n")
    if coarse_key not in res or casc_key not in res:
        rep.append("_Dilewati: prediksi coarse atau cascade tidak ditemukan._\n")
        return
    if coarse_key == "baseline":
        rep.append("> Catatan: PNG coarse dari cascade tidak disimpan, jadi **baseline dipakai sebagai "
                   "proksi** model coarse. Beri `--coarse-dir` jika punya PNG coarse yang sebenarnya.\n")
    ci, cc, cf = res[coarse_key]
    ri, rc, rf = res[casc_key]
    tot_px = cc.size_px.sum() if len(cc) else 0

    cc = cc.copy()
    cc["size_bin"] = area_bin(cc.size_frac)
    miss = cc[~cc.hit]
    t = cc.groupby("size_bin", observed=True).agg(
        komponen=("hit", "size"), terlewat=("hit", lambda x: int((~x).sum())),
        hit_rate=("hit", "mean"), mean_cover=("cover", "mean"))
    t["px_terlewat"] = cc[~cc.hit].groupby("size_bin", observed=True).size_px.sum()
    t = t.fillna(0)
    t.to_csv(outdir / "a2_komponen_coarse.csv")
    rep.append("Komponen luka GT yang tidak tersentuh sama sekali oleh coarse (hit = ada piksel prediksi yang overlap):\n")
    rep.append(fmt(t))
    rep.append(f"Total komponen GT: {len(cc)}; terlewat: {len(miss)} "
               f"({100 * len(miss) / max(len(cc), 1):.1f}%); piksel luka dalam komponen terlewat: "
               f"{100 * miss.size_px.sum() / max(tot_px, 1):.2f}% dari semua piksel luka.\n")

    # apakah cascade memulihkan?
    m = cc.merge(rc[["image", "size_px", "hit", "cover"]], on=["image", "size_px"],
                 suffixes=("_coarse", "_casc"))
    rec = m[~m.hit_coarse]
    rep.append(f"Dari komponen yang terlewat coarse, yang berhasil disentuh cascade: "
               f"{int(rec.hit_casc.sum())}/{len(rec)}. Sebaliknya, yang tadinya kena lalu hilang di cascade: "
               f"{int(((m.hit_coarse) & (~m.hit_casc)).sum())}.\n")

    # citra tanpa prediksi sama sekali -> tidak ada crop yang bisa dibuat
    gi = ci.merge(ev[["image", "area_frac", "H", "W"]], on="image")
    gi = gi[gi.area_frac > 0]
    empty = gi[gi.pred_empty]
    rep.append(f"Citra dengan GT tapi **prediksi coarse kosong** (cascade tidak punya crop): "
               f"{len(empty)}/{len(gi)}.\n")

    # komponen terlewat di luar bbox prediksi (+margin) pada citra yang punya prediksi
    out_cnt, in_cnt = 0, 0
    bbx = gi.set_index("image")
    for _, r in miss.iterrows():
        row = bbx.loc[r.image]
        if row.bbox is None or (isinstance(row.bbox, float)):
            continue
        y0, y1, x0, x1 = row.bbox
        mh, mw = margin * (y1 - y0 + 1), margin * (x1 - x0 + 1)
        inside = (y0 - mh <= r.cy <= y1 + mh) and (x0 - mw <= r.cx <= x1 + mw)
        in_cnt += inside
        out_cnt += (not inside)
    rep.append(f"Komponen terlewat pada citra yang *punya* prediksi: {in_cnt + out_cnt}; "
               f"di luar bbox prediksi (+{int(margin * 100)}% margin): {out_cnt}. "
               "Angka 'di luar' = kasus yang tidak akan terjangkau crop berbasis bbox prediksi "
               "(aturan crop asli notebook 13 bisa berbeda, ini perkiraan).\n")

    # dekomposisi kehilangan Dice per kategori
    d = ci.merge(ev[["image", "area_frac"]], on="image")
    d = d[d.area_frac > 0].copy(); d["deficit"] = 1 - d.dice
    dec = d.groupby("cat").agg(n=("dice", "size"), deficit_sum=("deficit", "sum"))
    dec["share_of_total_deficit"] = dec.deficit_sum / dec.deficit_sum.sum()
    dec = dec.drop(columns="deficit_sum")
    dec.to_csv(outdir / "a2_deficit_per_kategori.csv")
    rep.append("Kehilangan Dice coarse per jenis kegagalan. Kalau `miss_total` dominan, proposal "
               "berbasis ketidakpastian punya ruang; kalau `under_seg`/`fp_heavy` dominan, zoom saja sudah cukup:\n")
    rep.append(fmt(dec))

    fpt = pd.DataFrame({
        "coarse": [len(cf), int(cf.size_px.sum()) if len(cf) else 0],
        "cascade": [len(rf), int(rf.size_px.sum()) if len(rf) else 0]},
        index=["jumlah komponen FP", "piksel FP"])
    rep.append("Komponen prediksi yang tidak overlap GT (false positive terpisah):\n")
    rep.append(fmt(fpt, 0))


# ----------------------------------------------------------------------------
# A3  COPY-PASTE / OVERSAMPLING: statistik train
# ----------------------------------------------------------------------------
def a3_copypaste(base, cfg, ev, res, outdir, rep, min_comp, target):
    rep.append("## A3. Data train: kolam luka kecil untuk Copy-Paste / oversampling\n")
    labs = index_dir(base / cfg["raw"] / "train" / "labels")
    if not labs:
        rep.append("_Dilewati: folder train/labels tidak ditemukan._\n")
        return
    rows, comp_rows = [], []
    for s, p in labs.items():
        gt = read_mask(p)
        lab, n = ndi.label(gt, S8)
        sz = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        rows.append(dict(image=s, area_frac=gt.sum() / gt.size, n_comp=int((sz >= min_comp).sum())))
        for z in sz[sz >= min_comp]:
            comp_rows.append(dict(image=s, size_frac=z / gt.size))
    tr = pd.DataFrame(rows)
    tr["area_bin"] = area_bin(tr.area_frac)
    tc = pd.DataFrame(comp_rows)
    tr_pos = tr[tr.area_frac > 0]
    n_small = int(((tr_pos.area_frac < SMALL_FRAC)).sum())
    n_large = len(tr_pos) - n_small

    tb = pd.DataFrame({
        "train_n": tr_pos.area_bin.value_counts().reindex(AREA_LABELS).fillna(0).astype(int),
        "eval_n": ev[ev.area_frac > 0].area_bin.value_counts().reindex(AREA_LABELS).fillna(0).astype(int)})
    tb["train_%"] = 100 * tb.train_n / tb.train_n.sum()
    tb["eval_%"] = 100 * tb.eval_n / max(tb.eval_n.sum(), 1)
    if "baseline" in res:
        d = res["baseline"][0].merge(ev, on="image")
        tb["baseline_dice"] = d[d.area_frac > 0].groupby("area_bin", observed=True).dice.mean()
    tb.to_csv(outdir / "a3_sebaran_ukuran.csv")
    rep.append(f"Citra train berlabel: {len(tr_pos)} | citra luka kecil (<1%): {n_small} "
               f"({100 * n_small / max(len(tr_pos), 1):.1f}%).\n")
    rep.append(fmt(tb))
    if len(tc):
        small_c = tc[tc.size_frac < SMALL_FRAC]
        rep.append(f"Komponen luka (>= {min_comp}px): {len(tc)}; yang kecil (<1%): {len(small_c)} "
                   f"dari {small_c.image.nunique()} citra sumber. Ini ukuran 'kolam' yang bisa "
                   "ditempel (Copy-Paste) ke area kulit lain.\n")
    if n_small > 0:
        r = max(1.0, target / (1 - target) * n_large / n_small)
        rep.append(f"Agar citra luka kecil mencapai {int(target * 100)}% dari paparan per epoch, "
                   f"bobot sampling citra luka kecil ~{r:.2f}x terhadap citra besar "
                   f"(1.00 berarti sudah di atas target).\n")
    else:
        rep.append("Tidak ada citra luka kecil di train.\n")
    rep.append("> Catatan: ini seluruh folder `train` (sebelum split 85/15). Dan baca kolom "
               "`baseline_dice` bersama `train_n`: kalau kelas kecil punya contoh train yang cukup tapi "
               "Dice tetap rendah, masalahnya bukan kelangkaan data, dan oversampling kecil kemungkinan membantu.\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.logspace(-4.5, 0, 30)
    ax[0].hist(tr_pos.area_frac, bins=bins, alpha=.6, density=True, label="train")
    ax[0].hist(ev[ev.area_frac > 0].area_frac, bins=bins, alpha=.6, density=True, label="eval")
    ax[0].set_xscale("log"); ax[0].axvline(SMALL_FRAC, color="k", ls="--")
    ax[0].set_xlabel("luas luka (fraksi citra)"); ax[0].legend(); ax[0].set_title("Sebaran luas luka")
    tb[["train_%", "eval_%"]].plot(kind="bar", ax=ax[1]); ax[1].set_title("Proporsi per kelas luas")
    plt.tight_layout(); plt.savefig(outdir / "a3_sebaran_ukuran.png", dpi=130); plt.close()
    rep.append("Gambar: `a3_sebaran_ukuran.png`\n")


# ----------------------------------------------------------------------------
# A4  PERBANDINGAN BERPASANGAN vs baseline
# ----------------------------------------------------------------------------
def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def paired_stats(delta):
    delta = np.asarray(delta, float)
    if len(delta) == 0:
        return dict(n=0)
    try:
        p = wilcoxon(delta).pvalue if np.any(np.abs(delta) > EPS) else 1.0
    except ValueError:
        p = np.nan
    lo, hi = boot_ci(delta)
    return dict(n=len(delta), mean_delta=delta.mean(), ci95_lo=lo, ci95_hi=hi, p_wilcoxon=p,
                wins=int((delta > EPS).sum()), losses=int((delta < -EPS).sum()))


def _guess_cols(df):
    low = {c: c.lower() for c in df.columns}
    name = next((c for c in df.columns if any(k in low[c] for k in ("image", "name", "file", "id"))),
                df.columns[0])
    dice = next((c for c in df.columns if low[c] in ("dice_img", "dice_image", "dice")), None)
    if dice is None:
        dice = next((c for c in df.columns if "dice" in low[c] and "data" not in low[c]), None)
    return name, dice


def noise_floor(base, cfg, rep):
    res_dir = base / cfg["out"] / "results"
    a = sorted(res_dir.glob("baseline_*_per_image.csv"), key=lambda p: p.name)
    b = sorted(res_dir.glob("cascade_*_per_image_coarse.csv"), key=lambda p: p.name)
    if not a or not b:
        return
    try:
        da, db = pd.read_csv(a[-1]), pd.read_csv(b[-1])
        na, ca = _guess_cols(da); nb, cb = _guess_cols(db)
        if ca is None or cb is None:
            rep.append(f"_Noise floor dilewati: kolom Dice tidak terdeteksi. Kolom: {list(da.columns)}_\n")
            return
        da["k"] = da[na].astype(str).str.replace(r"\.\w+$", "", regex=True)
        db["k"] = db[nb].astype(str).str.replace(r"\.\w+$", "", regex=True)
        m = da.merge(db, on="k", suffixes=("_a", "_b"))
        delta = (m[f"{cb}_b" if cb == ca else cb] - m[f"{ca}_a" if cb == ca else ca]).dropna().values
        s = paired_stats(delta)
        rep.append("**Noise floor** (arsitektur identik, seed/run berbeda: `cascade_coarse` vs `baseline`, "
                   f"dari CSV; kolom tebakan nama=`{na}`, dice=`{ca}`/`{cb}`):\n")
        rep.append(fmt(pd.DataFrame([s])))
        rep.append("Selisih model lain yang tidak jauh lebih besar dari angka ini belum bisa diklaim.\n")
    except Exception as e:  # noqa
        rep.append(f"_Noise floor dilewati: {e}_\n")


def a4_paired(ev, res, models, outdir, base, cfg, rep):
    rep.append("## A4. Perbandingan berpasangan vs baseline (per citra, hanya citra ber-GT)\n")
    if "baseline" not in res:
        rep.append("_Dilewati: baseline tidak ditemukan._\n")
        return
    b = res["baseline"][0][["image", "dice", "cat"]].rename(columns={"dice": "d_b", "cat": "cat_b"})
    b = b.merge(ev, on="image")
    b = b[b.area_frac > 0]
    q = pd.qcut(b.contrast_dE, 3, labels=["kontras_rendah", "kontras_sedang", "kontras_tinggi"],
                duplicates="drop") if b.contrast_dE.notna().sum() > 9 else None
    b = b.assign(c_tert=q)
    rows, flips = [], {}
    for mname in models:
        if mname == "baseline" or mname not in res:
            continue
        m = b.merge(res[mname][0][["image", "dice", "cat"]].rename(columns={"dice": "d_m", "cat": "cat_m"}),
                    on="image")
        m["delta"] = m.d_m - m.d_b
        subsets = {"semua": m, "kecil(<1%)": m[m.small], "besar": m[~m.small]}
        for lb in AREA_LABELS:
            subsets[f"bin {lb}"] = m[m.area_bin == lb]
        if q is not None:
            for lb in m.c_tert.dropna().unique():
                subsets[str(lb)] = m[m.c_tert == lb]
        for sname, sub in subsets.items():
            rows.append(dict(model=mname, subset=sname, **paired_stats(sub.delta.values)))
        recov = ((m.cat_b == "miss_total") & (m.cat_m != "miss_total")).sum()
        lost = ((m.cat_b != "miss_total") & (m.cat_m == "miss_total")).sum()
        flips[mname] = (int(recov), int(lost))
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "a4_berpasangan_vs_baseline.csv", index=False)
    rep.append("`mean_delta` = rata-rata (Dice model - Dice baseline) per citra; CI = bootstrap 95%; "
               "p = Wilcoxon berpasangan. Dengan banyak subset dan model, p kecil di satu-dua baris bisa "
               "kebetulan, jadi cari pola yang konsisten, bukan satu angka.\n")
    for mname in out.model.unique():
        rep.append(f"### {mname} vs baseline\n")
        rep.append(fmt(out[out.model == mname].drop(columns="model").set_index("subset")))
        if mname in flips:
            rep.append(f"Citra yang di baseline gagal total (Dice-nya miss_total) tapi dipulihkan {mname}: "
                       f"{flips[mname][0]}; sebaliknya baru gagal total di {mname}: {flips[mname][1]}.\n")
    noise_floor(base, cfg, rep)

    sub = out[out.subset.isin(["kecil(<1%)", "besar"] + [f"bin {x}" for x in AREA_LABELS])]
    if len(sub):
        fig, ax = plt.subplots(figsize=(9, 4))
        piv = sub.pivot(index="subset", columns="model", values="mean_delta")
        piv.plot(kind="bar", ax=ax); ax.axhline(0, color="k", lw=.8)
        ax.set_ylabel("mean delta Dice vs baseline"); ax.set_title("Selisih Dice per kelas luas")
        plt.tight_layout(); plt.savefig(outdir / "a4_delta_per_kelas.png", dpi=130); plt.close()
        rep.append("Gambar: `a4_delta_per_kelas.png`\n")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def run_dataset(name, args, base):
    cfg = DATASETS[name]
    outdir = base / "analisis_empat_ide" / name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    ev, gts = load_eval(base, cfg, args.min_comp)
    print(f"   citra evaluasi: {len(ev)} | berlabel luka: {(ev.area_frac > 0).sum()} | "
          f"luka kecil: {int(ev.small.sum())}")

    pred_root = base / cfg["out"] / "predictions"
    override = dict(kv.split("=", 1) for kv in args.run)
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    res, used = {}, {}
    for k in keys:
        run = find_run(pred_root, k, override)
        if run is None:
            print(f"   [SKIP] {k}: folder prediksi tidak ketemu di {pred_root}")
            continue
        print(f"   {k}: {run.name}")
        res[k] = eval_model(run, gts, args.min_comp)
        used[k] = run.name
        res[k][0].drop(columns="bbox").to_csv(outdir / f"per_image_{k}.csv", index=False)
    if args.coarse_dir:
        print(f"   coarse (manual): {args.coarse_dir}")
        res["coarse"] = eval_model(Path(args.coarse_dir), gts, args.min_comp)
        used["coarse"] = str(args.coarse_dir)
    ev.drop(columns=["H", "W"]).to_csv(outdir / "gt_fitur_per_citra.csv", index=False)

    rep = [f"# Analisis empat ide: {name}\n",
           f"Evaluasi: `{cfg['raw']}/{cfg['eval_split']}` ({len(ev)} citra). "
           f"Luka kecil = luas > 0 dan < {SMALL_FRAC * 100:.0f}% citra. "
           f"Komponen < {args.min_comp}px diabaikan.\n",
           "Folder prediksi yang dipakai (otomatis: timestamp terbaru yang berisi PNG):\n",
           "```\n" + "\n".join(f"{k}: {v}" for k, v in used.items()) + "\n```\n",
           "> Satu run, satu seed. Selisih kecil belum bisa diklaim (lihat noise floor di A4).\n"]
    if "baseline" in res:
        a1_diagnosis(ev, res, "baseline", outdir, rep)
    else:
        rep.append("## A1. Dilewati: baseline tidak ditemukan.\n")
    coarse_key = "coarse" if "coarse" in res else "baseline"
    a2_cascade(ev, res, coarse_key, "cascade", outdir, rep)
    a3_copypaste(base, cfg, ev, res, outdir, rep, args.min_comp, args.target_small_share)
    a4_paired(ev, res, keys, outdir, base, cfg, rep)

    (outdir / "laporan.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"   -> {outdir / 'laporan.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="both", choices=["both", *DATASETS])
    ap.add_argument("--base-dir", default=None, help="root proyek (default: deteksi otomatis)")
    ap.add_argument("--models", default="baseline,cl_dpsc,bw_ftl,cascade",
                    help="prefix nama folder prediksi, pisah koma")
    ap.add_argument("--run", action="append", default=[], metavar="MODEL=FOLDER",
                    help="paksa folder prediksi tertentu, boleh diulang")
    ap.add_argument("--coarse-dir", default=None, help="folder PNG prediksi coarse (opsional, untuk A2)")
    ap.add_argument("--min-comp", type=int, default=20, help="abaikan komponen < N piksel")
    ap.add_argument("--target-small-share", type=float, default=0.5)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    if args.base_dir:
        base = Path(args.base_dir)
    else:
        base = Path(".") if Path("data/raw").exists() else Path("..")
    base = base.resolve()
    if not (base / "data" / "raw").exists():
        sys.exit(f"[ERROR] {base}/data/raw tidak ada. Jalankan dari root proyek atau notebook/, "
                 "atau beri --base-dir.")
    print(f"base-dir: {base}")
    for ds in (DATASETS if args.dataset == "both" else [args.dataset]):
        run_dataset(ds, args, base)


if __name__ == "__main__":
    main()
