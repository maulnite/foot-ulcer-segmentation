"""Presentation plots: real masks, shared coordinates and explicit evidence labels."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from .presentation import display_bounds, overlay

TP, FP, FN = "#009E73", "#D55E00", "#0072B2"
BASE, TEAL, PURPLE = "#64748b", "#0f766e", "#7c3aed"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "text.color": "#18324b", "axes.labelcolor": "#18324b",
                         "svg.fonttype": "none"})


def save(fig, folder, stem, vector=False):
    path = Path(folder)/f"{stem}.png"
    if path.exists():
        raise FileExistsError("Presentation assets are append-only")
    fig.savefig(path, dpi=150, facecolor="white")
    if vector:
        fig.savefig(Path(folder)/f"{stem}.svg", facecolor="white")
    plt.close(fig)
    return path


def gallery(folder, stem, title, subtitle, cases, model_names, zoom=False, masks=False):
    """Identical photo/GT/model columns; a full-field companion always accompanies zoom."""
    style()
    fig, axes = plt.subplots(len(cases), len(model_names)+2, figsize=(16, 9), squeeze=False)
    fig.subplots_adjust(left=.035, right=.985, top=.78, bottom=.17, wspace=.045, hspace=.50)
    for row_index, case in enumerate(cases):
        rgb, truth, predictions = case["rgb"], case["truth"], case["predictions"]
        bound = display_bounds(rgb, truth, [predictions[m] for m in model_names]) if zoom else (0, 0, 224, 224)
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].set_title(f"{case['label']}\nID {Path(case['sample_id']).stem[:9]}", fontsize=11, pad=7)
        axes[row_index, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title(f"GT · {int(truth.sum())} piksel\n{truth.mean()*100:.3f}% kanvas", fontsize=11, pad=7)
        for column, name in enumerate(model_names, start=2):
            prediction, metric = predictions[name], case["metrics"][name]
            ax = axes[row_index, column]
            if masks:
                ax.imshow(prediction, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(overlay(rgb, truth, prediction))
            ax.set_title(f"{name} · epoch {case['epochs'][name]}\nDice {metric['dice']*100:.2f}%\nFP {metric['fp']} · FN {metric['fn']}",
                         fontsize=10.5, pad=7)
        for ax in axes[row_index]:
            ax.set_xlim(bound[0]-.5, bound[2]-.5)
            ax.set_ylim(bound[3]-.5, bound[1]-.5)
            ax.axis("off")
    fig.suptitle(title, fontsize=21, fontweight="bold", y=.965)
    fig.text(.5, .895, subtitle, ha="center", fontsize=12)
    mode = "ZOOM TAMPILAN · ROI sama untuk semua model; seluruh FP/FN tetap di dalam ROI" if zoom else "FULL FIELD · grid evaluasi asli 224×224 · threshold ≥0,5"
    fig.text(.5, .85, mode, ha="center", fontsize=11, color=BASE)
    if masks:
        fig.text(.5, .102, "Mask prediksi asli: putih = foreground · hitam = background", ha="center", fontsize=12)
    else:
        fig.legend(handles=[Patch(color=TP, label="TP · benar pada GT"), Patch(color=FP, label="FP · di luar GT"),
                            Patch(color=FN, label="FN · GT yang terlewat")], loc="lower center",
                   bbox_to_anchor=(.5, .075), ncol=3, frameon=False, fontsize=12)
    fig.text(.5, .04, "Kasus dipilih posthoc dengan aturan tetap; contoh bukan estimasi prevalensi keberhasilan. GT kosong ≠ diagnosis sehat.",
             ha="center", fontsize=10.5)
    return save(fig, folder, stem)


def scoreboard(folder, names, metrics, metadata):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(left=.12, right=.96, top=.80, bottom=.14, wspace=.16)
    ypos = np.arange(len(names))
    for ax, split, n in zip(axes, ["validation", "test"], [118, 261]):
        values = [metrics[name][split]["image_dice"]*100 for name in names]
        ax.barh(ypos, values, color=[BASE if name in ["B0", "B1"] else TEAL for name in names], height=.65)
        ax.set(xlim=(0,100), yticks=ypos, yticklabels=[f"{name} · e{metadata[name]['epoch']}" for name in names],
               xlabel="Mean Image Dice (%)", title=f"{'Validation' if split == 'validation' else 'Source-test'} · n={n} foto")
        ax.axvline(metrics["B1"][split]["image_dice"]*100, color=PURPLE, linestyle="--", linewidth=1.4, label="B1")
        for y,value in zip(ypos,values):
            ax.text(value+.7,y,f"{value:.2f}",va="center",fontsize=11)
        ax.grid(axis="x",alpha=.14)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Perbandingan terpadu pada cohort CP2 yang sama",fontsize=21,fontweight="bold",y=.965)
    fig.text(.5,.90,"Chronic Wound · split 671/118/261 · seed42 · REAL / CACHED · threshold ≥0,5",ha="center",fontsize=13)
    fig.text(.5,.85,"15 model completed; semua angka dihitung ulang dari cache asli. B1 tetap baseline yang dipertahankan.",ha="center",fontsize=11)
    fig.text(.5,.075,"Source-test sudah terekspos: hasil deskriptif, bukan dasar pemilihan model. Split berdasarkan RGB duplikat, bukan pasien.",ha="center",fontsize=11)
    fig.text(.5,.035,"Ganta asli, CP1 dan dataset eksternal tidak dimasukkan sebagai baris ranking. Kombinasi terbaru dan cascade CP2 menunggu laporan final.",ha="center",fontsize=10.5)
    return save(fig,folder,"14_cp2_same_cohort_scoreboard",vector=True)


def tradeoffs(folder, names, metrics, guards):
    style()
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(left=.13,right=.98,top=.77,bottom=.20)
    base=metrics["B1"]["validation"]
    colors=np.zeros((len(names),6,4))
    texts=[]
    for i,name in enumerate(names):
        value=metrics[name]["validation"]
        deltas=[(value[key]-base[key])*(100 if column<3 else 1)
                for column,key in enumerate(["image_dice","small_dice","boundary_iou","misses","empty_fp_images"])]
        text=[f"{delta:+.2f}" if j<3 else f"{int(delta):+d}" for j,delta in enumerate(deltas)]
        for j,delta in enumerate(deltas):
            good=delta>0 if j<3 else delta<0
            colors[i,j]=matplotlib.colors.to_rgba("#d9f0e7" if good else "#f8ded5" if delta!=0 else "#edf1f5")
        eligible=all(guards[name].values()) if guards[name] else None
        text.append("ACUAN" if name=="B1" else "N/A" if eligible is None else "LULUS" if eligible else "TIDAK")
        colors[i,5]=matplotlib.colors.to_rgba("#d9f0e7" if eligible else "#edf1f5" if eligible is None else "#f8ded5")
        texts.append(text)
    ax.imshow(colors,aspect="auto",interpolation="nearest")
    ax.set(xticks=range(6),xticklabels=["Δ Image Dice\n(pp)","Δ Small Dice\n(pp)","Δ Boundary IoU\n(pp)","Δ TP=0\n(foto)","Δ Empty FP\n(foto)","Guard asli\nfase terkait"],yticks=range(len(names)),yticklabels=names)
    ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=10)
    for i,row in enumerate(texts):
        for j,label in enumerate(row):ax.text(j,i,label,ha="center",va="center",fontsize=11)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle("Kenaikan Dice tidak selalu meloloskan guard validation",fontsize=21,fontweight="bold",y=.97)
    fig.text(.5,.89,"Chronic Wound · validation n=118 (GT+111, GT kosong7, small36) · seed42 · REAL / CACHED",ha="center",fontsize=12)
    fig.text(.5,.84,"Selisih terhadap B1; hijau = arah membaik, jingga = memburuk. Angka selalu ditampilkan, tidak hanya warna.",ha="center",fontsize=11)
    fig.text(.5,.12,"Small: 0 < GT ≤1% kanvas berpadded. TP=0: GT positif tanpa overlap prediksi. Empty FP: mask kosong dengan prediksi foreground.",ha="center",fontsize=10.5)
    fig.text(.5,.075,"Kolom terakhir mempertahankan keputusan/guard fase asli; definisi guard antar-fase berbeda. Tidak menerapkan aturan baru atau memilih ulang model.",ha="center",fontsize=10.5)
    fig.text(.5,.035,"Belum mencakup kombinasi terbaru dan cascade CP2 final. Tidak ada klaim superioritas klinis atau generalisasi antar-pasien.",ha="center",fontsize=10.5)
    return save(fig,folder,"14_cp2_guard_tradeoffs",vector=True)


def combinations_context(folder, metrics):
    style()
    fig, axes=plt.subplots(1,2,figsize=(16,9),gridspec_kw={"width_ratios":[1,1.5]})
    fig.subplots_adjust(left=.055,right=.97,top=.78,bottom=.18,wspace=.15)
    names=["B1","A1","H2","A1H2"]
    matrix=np.array([[0,0],[1,0],[0,1],[1,1]])
    axes[0].imshow(matrix,cmap="Greys",vmin=0,vmax=1,aspect="auto")
    axes[0].set(xticks=[0,1],xticklabels=["Local A1","Boundary H2"],yticks=range(4),yticklabels=names,title="Faktor yang sudah selesai diuji")
    for i in range(4):
        for j in range(2):axes[0].text(j,i,"AKTIF" if matrix[i,j] else "—",ha="center",va="center",color="white" if matrix[i,j] else BASE)
    values=[[f"{metrics[n]['validation'][k]*100:.2f}%" if k not in ['misses','empty_fp_images'] else str(metrics[n]['validation'][k])
             for k in ['image_dice','small_dice','boundary_iou','misses','empty_fp_images']] for n in names]
    axes[1].axis("off")
    table=axes[1].table(cellText=values,rowLabels=names,colLabels=["Image Dice","Small Dice","Boundary IoU","Miss (foto)","Empty FP (foto)"],loc="center",cellLoc="center")
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1.05,2.7)
    axes[1].set_title("Validation n=118 · threshold ≥0,5")
    fig.suptitle("Konteks kombinasi yang sudah selesai: A1 + H2",fontsize=21,fontweight="bold",y=.96)
    fig.text(.5,.88,"Chronic Wound · REAL / CACHED · seed42 · bukan hasil final dua kombinasi A6+H2 terbaru",ha="center",fontsize=12)
    fig.text(.5,.11,"A1H2 tidak meloloskan guard asli; B1 dipertahankan. Individual controls dan kombinasi memakai gambar validation yang identik.",ha="center",fontsize=11)
    fig.text(.5,.055,"B1+A6+H2 dan L3+A6+H2: training selesai, laporan final belum lolos QA. Angkanya tidak dipakai untuk memilih kandidat atau mengisi figur final.",ha="center",fontsize=10.5)
    return save(fig,folder,"11_combinations_completed_context")


def cascade_walkthrough(folder, cases, zoom=False):
    style()
    fig=plt.figure(figsize=(16,9))
    grid=fig.add_gridspec(4,4,left=.04,right=.985,top=.77,bottom=.16,hspace=.36,wspace=.09,height_ratios=[2.2,1,2.2,1])
    headings=["1 · Foto asli", "2 · Coarse", "3 · Proposal dari coarse", "4 · Refiner → stitch"]
    for i,case in enumerate(cases):
        rgb,truth=case['rgb'],case['truth']
        first,final=case['coarse']>=.5,case['final']>=.5
        bounds=display_bounds(rgb,truth,[first,final]) if zoom else (0,0,224,224)
        for j in range(4):
            ax=fig.add_subplot(grid[i*2,j])
            ax.imshow(rgb if j in [0,2] else overlay(rgb,truth,first if j==1 else final))
            if j==2:
                for box_number,box in enumerate(case['boxes'],start=1):
                    x0,y0,x1,y1=box
                    ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor="#E69F00",linewidth=2))
                    ax.text(x0+1,y0+1,str(box_number),color="black",backgroundcolor="#E69F00",
                            fontsize=9,va="top",clip_on=True)
            metric=case['coarse_metrics'] if j==1 else case['final_metrics'] if j==3 else None
            ax.set_title(headings[j]+(f"\nDice {metric['dice']*100:.2f}% · FP {metric['fp']} · FN {metric['fn']}" if metric else ''),fontsize=11)
            ax.set_xlim(bounds[0]-.5,bounds[2]-.5);ax.set_ylim(bounds[3]-.5,bounds[1]-.5);ax.axis('off')
        ax=fig.add_subplot(grid[i*2+1,0]);ax.axis('off')
        ax.text(.5,.65,case['label']+f"\nID {Path(case['sample_id']).stem[:10]}\nGT {truth.sum()} piksel",ha='center',va='center',fontsize=11)
        ax=fig.add_subplot(grid[i*2+1,1]);ax.axis('off')
        ax.text(.5,.6,f"Proposal ≥0,3; final ≥0,5\n{len(case['boxes'])} ROI → 1+{len(case['boxes'])} forward\nGT hanya untuk evaluasi",ha='center',va='center',fontsize=11)
        for column,key in [(2,'crops'),(3,'crop_predictions')]:
            inner=grid[i*2+1,column].subgridspec(1,max(1,len(case[key])),wspace=.05)
            for n,value in enumerate(case[key]):
                ax=fig.add_subplot(inner[0,n]);ax.imshow(value,cmap='gray',vmin=0,vmax=1) if column==3 else ax.imshow(value)
                ax.set_title(f"{'Input crop' if column==2 else 'Mask refiner ROI'} {n+1}",fontsize=9);ax.axis('off')
    fig.suptitle("Cascade Ganta asli · telusuri crop hingga hasil akhir",fontsize=20,fontweight='bold',y=.967)
    fig.text(.5,.90,"Chronic Wound · test historis n=278 · seed42 · coarse epoch41 / refiner epoch27 · REAL / FRESH CPU",ha='center',fontsize=12)
    fig.text(.5,.85,"Dua kasus posthoc: gain Dice terbesar dan GT-empty dengan FP terbesar. Bukan adaptasi Orie atau pemilihan model baru.",ha='center',fontsize=10.5)
    fig.legend(handles=[Patch(color=TP,label='TP'),Patch(color=FP,label='FP'),Patch(color=FN,label='FN'),Patch(color='#E69F00',label='ROI prediksi coarse')],loc='lower center',bbox_to_anchor=(.5,.07),ncol=4,frameon=False,fontsize=11)
    fig.text(.5,.035,('ZOOM tampilan; full-field companion tetap tersedia. ' if zoom else 'FULL FIELD 224×224. ')+"Crop berasal dari prediksi, bukan GT; seluruh crop aktual dan mask refiner ditampilkan. Tidak ada retraining.",ha='center',fontsize=10.2)
    return save(fig,folder,'12_cascade_original_walkthrough'+('_zoom' if zoom else ''))
