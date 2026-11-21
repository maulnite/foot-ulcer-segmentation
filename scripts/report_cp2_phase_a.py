"""CPU report of completed, real attention screening; no training or selection."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402


COLORS = {"A0": "#64748B", "A1": "#2563EB", "A2": "#7C3AED", "A3": "#0F8B8D",
          "A5": "#B7791F", "A6": "#DB2777"}
BRANCH_COLORS = ("#2563EB", "#7C3AED", "#0F8B8D")


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if read(root / "train_workflow.json")["status"] != "completed":
        raise ValueError("Only completed actual screening can be presented")
    config, selection = read(root / "protocol.json"), read(root / "selection.json")
    names = ["A0", *selection["decisions"]]
    control = ROOT / config["phase_a"]["control_root"] / "B1"
    profiles = read(root / "architecture_preflight.json")["profiles"]
    analysis = read(ROOT / "reports/cp2-phase-a-analysis.json")
    if analysis["run_root"] != str(root.relative_to(ROOT)) or analysis["selection_sha256"] != digest_file(root / "selection.json"):
        raise ValueError("Posthoc analysis belongs to another experiment/selection")
    manifest = read(root / "data/manifest.json")
    metrics, status, histories, pairs, effects = {}, {}, {}, {}, {}
    figures, tables = ROOT / "reports/figures/cp2-phase-a", ROOT / "reports/tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)
    for name in names:
        run = control if name == "A0" else root / name
        status[name], histories[name] = read(run / "status.json"), read(run / "history.json")
        if status[name]["status"] != "completed" or status[name]["diagnostic_smoke_only"]:
            raise ValueError("No diagnostic/pending metric may be presented as full training")
        if digest_file(run / "best.pth") != status[name]["best_checkpoint_sha256"]:
            raise ValueError("Completed checkpoint changed")
        metrics[name] = {s:read(run / f"evaluation-{s}.json") for s in ("validation", "test")}
        pairs[name] = {}
        for split in ("validation", "test"):
            path = run / f"evaluation-{split}.csv"
            with path.open(newline="") as stream:
                pairs[name][split] = {r["name"]:r for r in csv.DictReader(stream)}
            shutil.copyfile(path, tables / f"cp2-phase-a-{name.lower()}-{split}.csv")
        history_file = tables / f"cp2-phase-a-{name.lower()}-history.csv"
        keys = ["epoch", "train_loss", "validation_loss", "validation_image_dice", "learning_rate_used",
                "learning_rate_next", "selected", "epoch_compute_seconds", "step_seconds_mean"]
        with history_file.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows({key:row[key] for key in keys} for row in histories[name])
    for split in ("validation", "test"):
        identities = sorted(pairs["A0"][split])
        effects[split] = {}
        for name in names[1:]:
            if set(pairs[name][split]) != set(identities):
                raise ValueError("Comparison population differs")
            delta = np.array([float(pairs[name][split][k]["dice"]) - float(pairs["A0"][split][k]["dice"]) for k in identities])
            rng = np.random.default_rng(20261108)
            bootstrap = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
            effects[split][name] = {"mean_delta_image_dice": float(delta.mean()),
                                    "descriptive_RGB_group_bootstrap_95": np.percentile(bootstrap, [2.5,97.5]).tolist(),
                                    "better": int((delta>0).sum()), "worse": int((delta<0).sum()), "tied": int((delta==0).sum()),
                                    "unit": "paired RGB group/image, not patient or seed uncertainty"}
    samples = [json.loads(line) for line in (root / "train_gpu_samples.jsonl").read_text().splitlines()]
    resources = {"scope":"one sequential GPU trainer; overall samples include other users; no exclusive-hardware timing claim",
                 "inference":analysis["inference"], "fresh_models":{}}
    for name in names[1:]:
        started=datetime.fromisoformat(read(root/name/"environment.json")["created_at_utc"])
        finished=datetime.fromisoformat(status[name]["finished_utc"])
        rows=[r for r in samples if started<=datetime.fromisoformat(r["utc"])<=finished and r["gpu_csv"]]
        values=[[float(v.strip()) for v in r["gpu_csv"].split(",")[2:]] for r in rows]
        resources["fresh_models"][name]={
            "completed_epochs":len(histories[name]),
            "sum_completed_epoch_compute_seconds":sum(r["epoch_compute_seconds"] for r in histories[name]),
            "median_epoch_compute_seconds":float(np.median([r["epoch_compute_seconds"] for r in histories[name]])),
            "mean_step_seconds":float(np.mean([r["step_seconds_mean"] for r in histories[name]])),
            "last_invocation_seconds":status[name]["seconds_this_invocation"],
            "environment_record_to_finish_seconds":(finished-started).total_seconds(),
            "peak_cuda_allocated_bytes_last_invocation":status[name]["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes_last_invocation":status[name]["peak_cuda_reserved_bytes"],
            "GPU_overall_sample_count":len(rows),
            "GPU_overall_peak_used_MiB":max(v[0] for v in values) if values else None,
            "GPU_overall_mean_utilization_percent":float(np.mean([v[2] for v in values])) if values else None,
            "minimum_sampled_free_shm_bytes":min(r["shm_free_bytes"] for r in rows) if rows else None,
            "caveat":"Invocation/allocator peaks cover the last process only if resumed; completed-epoch sums omit failed partial work. Overall GPU numbers include other processes."}
    resources["resource_interruptions"]=read(root/"resource_interruptions.json") if (root/"resource_interruptions.json").exists() else None
    write_json(ROOT/"reports/cp2-phase-a-resources.json",resources)
    with (tables / "cp2-phase-a-membership.csv").open("w", newline="") as stream:
        keys = ["split", "sample_id", "rgb_group", "gt_pixels", "canvas_pixels", "image_sha256", "mask_sha256"]
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for split, rows in manifest["splits"].items():
            writer.writerows({"split":split, **{k:r[k] for k in keys[1:]}} for r in rows)
    shutil.copyfile(ROOT / analysis["numeric_csv"], tables / "cp2-phase-a-attention-validation.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.titlesize":13,
                         "axes.spines.top":False, "axes.spines.right":False, "figure.facecolor":"white"})
    assets = []

    def save(fig, stem):
        for extension in ("png", "svg"):
            path = figures / f"{stem}.{extension}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            if extension == "svg":
                path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines())+"\n")
            assets.append({"path":str(path.relative_to(ROOT)), "sha256":digest_file(path)})
        plt.close(fig)

    fig, axes = plt.subplots(int(np.ceil(len(names)/2)), 2, figsize=(13,4*np.ceil(len(names)/2)), constrained_layout=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        h = histories[name]
        ax.plot([r["epoch"] for r in h], [r["train_loss"] for r in h], color=COLORS[name], label="Train")
        ax.plot([r["epoch"] for r in h], [r["validation_loss"] for r in h], color="#475569", linestyle="--", label="Validation")
        ax.axvline(status[name]["selected_epoch"], color=COLORS[name], linestyle=":", label="Selected checkpoint")
        ax.set(title=f"{name}: B1 objective"+(" (cached)" if name=="A0" else ""), xlabel="Epoch", ylabel="Dice + corrected focal loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("REAL — Chronic Wound | 671 train / 118 validation | seed 42", fontsize=15)
    save(fig,"training_curves")

    fig, axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for name in names:
        h=histories[name];epochs=[r["epoch"] for r in h]
        axes[0].plot(epochs,[r["validation_image_dice"] for r in h],label=name,color=COLORS[name],linestyle="--" if name=="A0" else "-")
        selected=status[name]["selected_epoch"]
        axes[0].scatter(selected,h[selected-1]["validation_image_dice"],color=COLORS[name],s=30,zorder=4)
        axes[1].step(epochs,[r["learning_rate_used"] for r in h],where="post",label=name,color=COLORS[name])
    axes[0].set(title="Checkpoint dipilih dari validation",xlabel="Epoch",ylabel="Image Dice",ylim=(0,1))
    axes[1].set(title="Aturan scheduler sama; jadwal aktual dapat berbeda",xlabel="Epoch",ylabel="Learning rate",yscale="log")
    for ax in axes:ax.legend(fontsize=9);ax.grid(alpha=.15)
    fig.suptitle("REAL — A0=CACHED B1; A1 dan kandidat perluasan=FRESH training",fontsize=14)
    save(fig,"checkpoint_and_lr")

    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for row,split in enumerate(("validation","test")):
        for column,(group,key,title) in enumerate((("all","dice","Semua gambar: Image Dice"),("small","dice","Luka kecil ≤1%: Image Dice"),("all","boundary_iou","Semua gambar: Boundary IoU"))):
            reports=[m[split]["subgroups"]["small"] if group=="small" else m[split]["all"] for m in metrics.values()]
            values=[r["image_based"][key] for r in reports];ax=axes[row,column]
            ax.bar(names,values,color=[COLORS[n] for n in names])
            for i,value in enumerate(values):ax.text(i,value+.015,f"{value:.3f}",ha="center",fontsize=9)
            ax.set(ylim=(0,1.04),xlabel="Arsitektur",ylabel="Mean per gambar (0–1)",title=f"{split}, n={reports[0]['images']}\n{title}")
            ax.grid(axis="y",alpha=.15)
    fig.suptitle("REAL — Chronic Wound selected checkpoints | threshold ≥0,5 | native 224×224",fontsize=15)
    save(fig,"segmentation_and_boundary")

    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for row,split in enumerate(("validation","test")):
        for column,(group,key,title) in enumerate((("gt_positive","total_foreground_misses","GT positif dengan TP=0"),("gt_empty","images_with_prediction","GT kosong dengan FP"))):
            ax=axes[row,column];values=[metrics[n][split][group][key] for n in names]
            ax.bar(names,values,color=[COLORS[n] for n in names])
            for i,value in enumerate(values):ax.text(i,value+.12,str(value),ha="center")
            ax.set(ylim=(0,max(2,max(values)+1)),xlabel="Arsitektur",ylabel="Jumlah gambar",title=f"{split}: {title}\nn={metrics['A0'][split][group]['images']}")
            ax.yaxis.get_major_locator().set_params(integer=True);ax.grid(axis="y",alpha=.15)
    fig.suptitle("REAL — Foreground miss dan empty FP | GT kosong bukan diagnosis sehat",fontsize=14)
    save(fig,"foreground_and_empty_errors")

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,split in zip(axes,("validation","test")):
        for position,name in enumerate(names[1:]):
            effect=effects[split][name];mean=100*effect["mean_delta_image_dice"];lo,hi=np.array(effect["descriptive_RGB_group_bootstrap_95"])*100
            ax.plot([lo,hi],[position,position],color=COLORS[name],linewidth=2)
            ax.scatter(mean,position,color=COLORS[name],s=40)
        ax.axvline(0,color="#64748B",linestyle="--")
        ax.set(yticks=range(len(names)-1),yticklabels=names[1:],xlabel="Δ Image Dice vs A0 (poin persentase)",ylabel="Arsitektur",title=f"{split}, n={metrics['A0'][split]['all']['images']} RGB groups")
        ax.invert_yaxis();ax.grid(axis="x",alpha=.15)
    fig.suptitle("REAL — Paired bootstrap deskriptif 95% | bukan pasien, variasi seed atau selection uncertainty",fontsize=13)
    save(fig,"paired_effects")

    gated=[n for n in names if n in ("A1","A2","A3")]
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,stage in zip(axes,("2","3")):
        bottom=np.zeros(len(gated))
        for key,label,color in zip(("w_additive_half","w_maxout","w_local"),("Additive / 2","Max-out","Local"),BRANCH_COLORS):
            values=[analysis["trained_validation"][n]["statistics"][stage]["all"][key]["mean"] for n in gated]
            ax.bar(gated,values,bottom=bottom,color=color,label=label)
            for i,(v,b) in enumerate(zip(values,bottom)):
                if v>.10:ax.text(i,b+v/2,f"{v:.2f}",ha="center",va="center",color="white",fontsize=11)
            bottom+=values
        ax.set(ylim=(0,1.03),xlabel="Fusion",ylabel="Mean normalized branch weight",title=f"Stage {stage}: validation n=118")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.06),ncol=3,frameon=False)
    fig.suptitle("REAL — Bobot fusion aktual | bukan bukti causal importance cabang",fontsize=14)
    save(fig,"fusion_weights")

    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for row,(kind,label) in enumerate((("initial_two_validation_images","Initial weights: dua gambar yang sama"),("trained_validation","Selected model: 118 validation images"))):
        for column,stage in enumerate((2,3)):
            values=[]
            for name in names:
                source=analysis[kind][name]
                if row==0:values.append(float(np.mean([v["fused_over_input_rms"] for v in source if v["stage"]==stage and v["fused_over_input_rms"] is not None])))
                else:values.append(source["statistics"][str(stage)]["all"]["fused_over_input_rms"]["mean"])
            ax=axes[row,column];ax.bar(names,values,color=[COLORS[n] for n in names])
            for i,value in enumerate(values):ax.text(i,value+.025,f"{value:.2f}",ha="center",fontsize=9)
            ax.set(ylim=(0,max(3.2,max(values)+.3)),xlabel="Arsitektur",ylabel="Mean RMS(fused) / RMS(input)",title=f"{label}\nStage {stage}")
            ax.grid(axis="y",alpha=.15)
    fig.suptitle("REAL — Audit skala aktual | source/A1–A3/CBAM bound ≤3; conv control tidak dibatasi sigmoid",fontsize=13)
    save(fig,"fusion_scale")

    if "A3" in names:
        with (ROOT/analysis["numeric_csv"]).open(newline="") as stream:gate_rows=[r for r in csv.DictReader(stream) if r["variant"]=="A3"]
        fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
        for ax,stage in zip(axes,("2","3")):
            rows=sorted([r for r in gate_rows if r["stage"]==stage],key=lambda r:r["sample_id"])
            matrix=np.array([[float(r[k]) for r in rows] for k in ("w_additive_half","w_maxout","w_local")])
            im=ax.imshow(matrix,aspect="auto",vmin=0,vmax=1,cmap="viridis",interpolation="nearest")
            ax.set(yticks=[0,1,2],yticklabels=["Add / 2","Max-out","Local"],xlabel="Urutan sorted sample ID (bukan waktu)",ylabel="Cabang",title=f"A3 stage {stage}, n={len(rows)}")
            fig.colorbar(im,ax=ax,label="Normalized weight (0–1)",shrink=.8)
        fig.suptitle("REAL — Variasi gate per gambar | warna hampir sama dapat berarti bobot hampir konstan",fontsize=13)
        save(fig,"input_gate_variation")

    interventions=analysis["interventions"]
    if interventions:
        labels,values=[],[]
        for name in ("A1","A3"):
            if name not in names:continue
            labels.append(name+"\noriginal");values.append(metrics[name]["validation"])
            for key,metric in interventions.items():
                if metric["variant"]==name:labels.append(name+"\n"+metric["intervention"].replace("_","\n"));values.append(metric)
        fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
        for ax,(group,key,title) in zip(axes,(("all","dice","Overall Dice, n=118"),("small","dice","Small Dice, n=36"),("all","boundary_iou","Boundary IoU, n=118"))):
            scores=[(m["subgroups"]["small"] if group=="small" else m["all"])["image_based"][key] for m in values]
            ax.bar(labels,scores,color=["#0F8B8D" if "original" in label else "#94A3B8" for label in labels])
            for i,v in enumerate(scores):ax.text(i,v+.015,f"{v:.3f}",ha="center",fontsize=9)
            ax.set(ylim=(0,1.04),xlabel="Inference condition",ylabel="Mean per gambar (0–1)",title=title)
            ax.tick_params(axis="x",labelsize=8);ax.grid(axis="y",alpha=.15)
        fig.suptitle("REAL — Validation-only intervention sesudah primary selection | tanpa retraining",fontsize=14)
        save(fig,"branch_interventions")

    summary={"status":"COMPLETED — REAL ATTENTION SCREENING", "created_utc":datetime.now(timezone.utc).isoformat(),
             "run_root":str(root.relative_to(ROOT)), "preregistration":read(root/"preregistration.json"),
             "protocol":config,"manifest_fingerprint":manifest["fingerprint"],"selection":selection,
             "expansion_decision":read(root/"expansion_decision.json"),"statuses":status,"metrics":metrics,
             "profiles":profiles,"resources":resources,"paired_effects":effects,"paired_inputs":read(root/"train_paired_controls.json"),
             "figures":assets,"execution":{n:"CACHED phase-P B1" if n=="A0" else "FRESH completed training" for n in names},
             "deferred":"A4, placement expansion, L3 combination, boundary head, multiseed and external evaluation NOT RUN"}
    write_json(ROOT/"reports/cp2-phase-a-results.json",summary)
    lines=["# CP2 Fase A — attention screening", "", "**COMPLETED — model yang dilaporkan benar-benar dilatih/evaluasi, seed 42.** A0 adalah B1 cached; tidak mengulang B0/B1. L3 tetap tersimpan untuk kombinasi mendatang.", "",
           f"Run `{root.relative_to(ROOT)}`. Preregistration commit `{summary['preregistration']['commit']}`. Manifest tetap **671/118/261**, duplicate-RGB-group split, **bukan patient-level**. Tidak ada external data atau perubahan precision/environment.", "",
           "[Protokol dan hipotesis](../docs/cp2-phase-a.md) · [Config](../configs/cp2/phase_a.json) · [Results JSON](cp2-phase-a-results.json) · [Gate/RMS/intervention JSON](cp2-phase-a-analysis.json)", "",
           "## Alur dan keputusan bertahap", "", "A1 diselesaikan dan dianalisis pada validation sebelum keputusan perluasan. Gate kelayakan tidak sama dengan guard menerima peningkatan. Kriteria/matriks tidak ditentukan dari source-test.", "",
           f"Keputusan perluasan A1: **{summary['expansion_decision']['expand']}**, checks `{summary['expansion_decision']['checks']}`. Executed fresh: {', '.join(names[1:])}. A4/spatial gate tetap deferred.", "",
           "Semua model memakai **B1 Dice + corrected-logit focal**; split, source preprocessing/augmentasi, shared initialization, batch 2, Adam, float32/TF32 flags, max 200/early-stop 30, validation-Dice checkpoint dan scheduler tetap. Core step/evaluator P/L digunakan ulang tanpa mengedit historical sources. Modul bersama diverifikasi per tensor, local A1/A2/A3 dipasangkan, initial forward A1/A2/A3 identik. Stage 2–3 saja berubah; final stage 4 tetap.", "",
           "## Hasil selected checkpoints", "", "| Model | Epoch dipilih / selesai | Val Image Dice | Test Image Dice | Small val / test | Boundary IoU val / test |", "|---|---:|---:|---:|---:|---:|"]
    for n in names:
        v,t=metrics[n]["validation"],metrics[n]["test"];s=status[n]
        lines.append(f"| {n} | {s['selected_epoch']} / {s['completed_epochs']} | {v['all']['image_based']['dice']:.4f} | {t['all']['image_based']['dice']:.4f} | {v['subgroups']['small']['image_based']['dice']:.4f} / {t['subgroups']['small']['image_based']['dice']:.4f} | {v['all']['image_based']['boundary_iou']:.4f} / {t['all']['image_based']['boundary_iou']:.4f} |")
    lines += ["",f"**Dipilih sebelum test baru: {selection['selected_candidate']}.** Guard: overall validation Dice naik; small/Boundary IoU tidak turun; total miss/empty FP tidak bertambah; active params ≤1,01×A0. Jika tidak ada eligible, A0 dipertahankan. Semua executed test results tetap dilaporkan. Source-test sudah dipakai CP1/P/L dan hasil berikut bersifat eksploratif, bukan confirmatory.","",
              "| Model | Eligible | Validation checks |","|---|---|---|"]
    for n,d in selection["decisions"].items():lines.append(f"| {n} | {d['eligible']} | `{d['checks']}` |")
    captions={"training_curves":"Objective tetap B1, sehingga definisi loss sama. Train memakai augmentasi; garis vertikal mengikuti puncak validation Image Dice, bukan minimum loss. Budget/jadwal aktual berbeda akibat early stopping.",
              "checkpoint_and_lr":"Kriteria checkpoint/scheduler sama, tetapi peak dan waktu penurunan LR dapat berbeda. Ini perbandingan recipe adaptif; bukan fixed-update-budget ablation.",
              "segmentation_and_boundary":"Semua model dibandingkan pada gambar yang sama. Small berarti ≤1% kanvas berpadded, bukan luas fisik. Baca overall bersama GT-positive/empty dan aggregate-pixel metrics.",
              "foreground_and_empty_errors":"TP=0 dan empty FP dihitung sebagai jumlah gambar. Nol yang dianotasi benar-benar berarti tidak ada kasus. GT kosong bukan diagnosis sehat; FP pixels dapat berbeda meski jumlah gambar sama.",
              "paired_effects":"Unit bootstrap adalah pasangan RGB group/gambar, bukan piksel. Interval tidak mencakup pasien/crop yang tidak diketahui, seed, checkpoint selection atau multiple comparisons; bukan bukti signifikansi klinis.",
              "fusion_weights":"Bobot aktual mengikuti normalisasi cabang dan gain 3 yang dipraspesifikasikan. Mean serupa tidak membuktikan gate identik; per-image variance/entropy ada pada JSON. Bobot bukan causal importance.",
              "fusion_scale":"Baris initial memakai dua validation images yang sama, tanpa fitting gain atau update bobot. Baris selected juga mencakup perubahan fitur upstream selama training. Bound/konvensi skala sama tidak menjamin distribusi aktivasi identik; CBAM midpoint berbeda dan conv residual tidak sigmoid-bounded.",
              "input_gate_variation":"Heatmap berisi bobot model A3 dari gambar nyata, urut sample ID, bukan temporal sequence. Warna yang hampir rata harus dibaca bersama nilai std/entropy; tidak boleh dipaksa menjadi bukti adaptivitas.",
              "branch_interventions":"Hanya inference validation setelah selection; drop-local menormalisasi bobot tersisa, fixed-prior mengganti learned gate. Tidak melatih ulang, tidak menunjukkan missing-branch robustness atau penyebab fisiologis."}
    for stem,caption in captions.items():
        if any(Path(a["path"]).stem==stem for a in assets):lines += ["",f"![{stem}](figures/cp2-phase-a/{stem}.png)","",caption]
    lines += ["","## Subgroups dan biaya","","| Split | Model | GT-positive Dice | GT-empty Dice | TP=0 / GT-positive n | Empty FP / empty n | Empty FP pixels | Data Dice | Pixel precision / recall | PFOM |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation","test"):
        for n in names:
            m=metrics[n][split]
            lines.append(f"| {split} | {n} | {m['gt_positive']['image_based']['dice']:.4f} | {m['gt_empty']['image_based']['dice']:.4f} | {m['gt_positive']['total_foreground_misses']} / {m['gt_positive']['images']} | {m['gt_empty']['images_with_prediction']} / {m['gt_empty']['images']} | {m['gt_empty']['counts']['fp']} | {m['all']['data_based']['dice']:.4f} | {m['all']['data_based']['precision']:.4f} / {m['all']['data_based']['recall']:.4f} | {m['all']['image_based']['pfom']:.4f} |")
    lines += ["","| Model | Registered / active parameters | Conv+Linear MACs/image | Training invocation (min) | Peak training alloc / reserved (GiB) | Batch-1 median / P95 (ms) |","|---|---:|---:|---:|---:|---:|"]
    for n in names:
        p,s,c=profiles[n],status[n],analysis["inference"][n]
        lines.append(f"| {n} | {p['registered_parameters']:,} / {p['active_forward_parameters']:,} | {p['conv_linear_MACs_per_image']:,} | {s['seconds_this_invocation']/60:.2f} | {s['peak_cuda_allocated_bytes']/1024**3:.2f} / {s['peak_cuda_reserved_bytes']/1024**3:.2f} | {c['median_ms']:.2f} / {c['p95_ms']:.2f} |")
    lines += ["","Active params dihitung dari actual forward calls, tidak memasukkan dormant source modules. A5 added active params dibandingkan A3 berjarak sekitar 1,6%, tetapi operasi dan kapasitas fungsional tidak identik. MACs hanya Conv2d/Linear, bukan seluruh pooling/normalisasi/elementwise. Latency mencakup forward+sigmoid, same-image/warmup 5/30 synchronized steps, tanpa data/GT/loss/export. GPU bersama dapat berubah; A0 training berasal dari waktu P yang berbeda. Waktu/peak invocation pada resume tidak mencakup semua percobaan; log dan status mentah tetap tersedia. [Resource JSON](cp2-phase-a-resources.json) memuat waktu per epoch/step, sampled overall GPU dan shared-memory headroom. Satu job tetap dipertahankan; VRAM kosong bukan alasan menaikkan batch/AMP/concurrency.","",
              "## Batas kesimpulan dan artefak","","Screening satu seed tidak menetapkan novelty, signifikansi atau manfaat klinis. Tidak mengubah annotation/empty rule berdasarkan model; pasien dan crop grouping belum diketahui. Source-test bukan untouched test. Gate/spatial map bukan causal explanation. L3 combination, A4, all-stage placement, H, multiseed dan external evaluation belum dijalankan.","",
              "Artefak lokal: protocol/manifest/initial weights, checkpoints/optimizer/scheduler/RNG, per-image probabilities/GT/masks, logs dan GPU/shared-memory samples. Foto/GT/overlay klinis tetap lokal. Public numeric membership, per-image tables, trained gate/RMS table, aggregate figures dan source disimpan di repository.","",
              "[QA](cp2-phase-a-validation.json) · [Review akhir](cp2-phase-a-review.md) · [Review visual/engineering](cp2-phase-a-quality.json). Berkas hanya diterbitkan setelah pemeriksaan aktual selesai."]
    path=ROOT/"reports/cp2-phase-a.md";temp=path.with_suffix(".md.tmp");temp.write_text("\n".join(lines)+"\n");temp.replace(path)
    write_json(root/"report_records"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json"),{"figures":assets,"result_sha256":digest_file(ROOT/"reports/cp2-phase-a-results.json")})
    print(summary["status"],flush=True)


if __name__=="__main__":
    main()
