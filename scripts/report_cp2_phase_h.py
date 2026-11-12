"""Render actual completed H results on CPU; no training, tuning or reselection."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402

COLORS = {'H0': '#64748B', 'H1': '#2563EB', 'H2': '#7C3AED'}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if read(root / 'train_workflow.json')['status'] != 'completed':
        raise ValueError('Require completed real screening')
    config, manifest, selection = read(root / 'protocol.json'), read(root / 'data/manifest.json'), read(root / 'selection.json')
    analysis = read(ROOT / 'reports/cp2-phase-h-analysis.json')
    if analysis['selection_sha256'] != digest_file(root / 'selection.json') or analysis['run_root'] != str(root.relative_to(ROOT)):
        raise ValueError('Posthoc belongs to another run/selection')
    control = ROOT / config['phase_h']['control_root'] / 'B1'
    names = ['H0', *selection['decisions']]
    metrics, statuses, histories, tables = {}, {}, {}, ROOT / 'reports/tables'
    tables.mkdir(exist_ok=True)
    paired, effects = {}, {}
    for name in names:
        run = control if name == 'H0' else root / name
        statuses[name], histories[name] = read(run / 'status.json'), read(run / 'history.json')
        assert statuses[name]['status'] == 'completed' and not statuses[name]['diagnostic_smoke_only']
        assert digest_file(run / 'best.pth') == statuses[name]['best_checkpoint_sha256']
        metrics[name] = {s: read(run / f'evaluation-{s}.json') for s in ('validation', 'test')}
        paired[name] = {}
        for split in ('validation', 'test'):
            path = run / f'evaluation-{split}.csv'
            with path.open(newline='') as stream:
                paired[name][split] = {r['name']: r for r in csv.DictReader(stream)}
            shutil.copyfile(path, tables / f'cp2-phase-h-{name.lower()}-{split}.csv')
        keys = ['epoch', 'train_loss', 'validation_loss', 'validation_image_dice', 'learning_rate_used',
                'learning_rate_next', 'selected', 'epoch_compute_seconds', 'step_seconds_mean']
        with (tables / f'cp2-phase-h-{name.lower()}-history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
            writer.writeheader()
            writer.writerows({k:r[k] for k in keys} for r in histories[name])
    for split in ('validation', 'test'):
        identities = sorted(paired['H0'][split])
        effects[split] = {}
        for name in names[1:]:
            assert set(paired[name][split]) == set(identities)
            delta = np.array([float(paired[name][split][k]['dice']) - float(paired['H0'][split][k]['dice']) for k in identities])
            rng = np.random.default_rng(20261112)
            bootstrap = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
            effects[split][name] = {'mean_delta_image_dice': float(delta.mean()),
                                    'descriptive_RGB_group_bootstrap_95': np.percentile(bootstrap, [2.5,97.5]).tolist(),
                                    'better': int((delta>0).sum()), 'worse': int((delta<0).sum()), 'tied': int((delta==0).sum()),
                                    'unit': 'paired RGB group/image, not patient/seed/selection uncertainty'}
    samples = [json.loads(line) for line in (root / 'train_gpu_samples.jsonl').read_text().splitlines()]
    resources = {'scope':'one sequential GPU trainer; overall GPU includes other jobs; cached B1 at a different time', 'fresh_models':{}}
    for name in names[1:]:
        started = datetime.fromisoformat(read(root/name/'environment.json')['created_at_utc'])
        finished = datetime.fromisoformat(statuses[name]['finished_utc'])
        sampled = [r for r in samples if started<=datetime.fromisoformat(r['utc'])<=finished and r['gpu_csv']]
        values = [[float(v.strip()) for v in r['gpu_csv'].split(',')[2:]] for r in sampled]
        resources['fresh_models'][name] = {
            'completed_epochs':len(histories[name]),
            'sum_completed_epoch_compute_seconds':sum(r['epoch_compute_seconds'] for r in histories[name]),
            'median_epoch_compute_seconds':float(np.median([r['epoch_compute_seconds'] for r in histories[name]])),
            'mean_step_seconds':float(np.mean([r['step_seconds_mean'] for r in histories[name]])),
            'last_invocation_seconds':statuses[name]['seconds_this_invocation'],
            'peak_allocated_bytes_last_invocation':statuses[name]['peak_cuda_allocated_bytes'],
            'peak_reserved_bytes_last_invocation':statuses[name]['peak_cuda_reserved_bytes'],
            'overall_GPU_peak_MiB':max(v[0] for v in values) if values else None,
            'overall_GPU_mean_util_percent':float(np.mean([v[2] for v in values])) if values else None,
            'minimum_sampled_free_shm_bytes':min(r['shm_free_bytes'] for r in sampled) if sampled else None,
            'sample_count':len(sampled), 'inference':analysis['models'][name]['mask_only_inference'],
            'caveat':'Last invocation excludes earlier interrupted work; overall GPU includes other users; inference head bypassed'}
    resources['interruptions'] = read(root/'resource_interruptions.json') if (root/'resource_interruptions.json').exists() else None
    write_json(ROOT/'reports/cp2-phase-h-resources.json', resources)
    with (tables/'cp2-phase-h-membership.csv').open('w', newline='') as stream:
        keys = ['split','sample_id','rgb_group','gt_pixels','canvas_pixels','image_sha256','mask_sha256']
        writer = csv.DictWriter(stream,fieldnames=keys,lineterminator='\n');writer.writeheader()
        for split,rows in manifest['splits'].items():
            writer.writerows({'split':split,**{k:r[k] for k in keys[1:]}} for r in rows)
    figures = ROOT/'reports/figures/cp2-phase-h';figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    assets=[]

    def save(fig,name):
        for extension in ('png','svg'):
            path=figures/f'{name}.{extension}';fig.savefig(path,dpi=160,bbox_inches='tight')
            if extension=='svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
            assets.append({'path':str(path.relative_to(ROOT)),'sha256':digest_file(path)})
        plt.close(fig)

    fig,axes=plt.subplots(len(names),2,figsize=(13,4*len(names)),constrained_layout=True,squeeze=False)
    for row,name in enumerate(names):
        h=histories[name];epochs=[r['epoch'] for r in h]
        for mode,style in (('train','-'),('validation','--')):
            regional=[r[f'{mode}_components'].get('regional',r[f'{mode}_loss']) if f'{mode}_components' in r else r[f'{mode}_loss'] for r in h]
            axes[row,0].plot(epochs,regional,style,color=COLORS[name] if mode=='train' else '#475569',label=mode)
            if name!='H0':axes[row,1].plot(epochs,[r[f'{mode}_components']['weighted_auxiliary'] for r in h],style,color=COLORS[name] if mode=='train' else '#475569',label=mode)
        axes[row,0].axvline(statuses[name]['selected_epoch'],linestyle=':',color=COLORS[name],label='Selected checkpoint')
        axes[row,0].set(title=f'{name}: main B1 objective'+(' (cached)' if name=='H0' else ''),xlabel='Epoch',ylabel='Main Dice + corrected focal loss')
        axes[row,0].legend(fontsize=9);axes[row,0].grid(alpha=.15)
        if name=='H0':axes[row,1].text(.5,.5,'H0: tidak ada auxiliary head',ha='center');axes[row,1].axis('off')
        else:
            axes[row,1].set(title=f'{name}: auxiliary '+('segmentation mask' if name=='H1' else 'boundary band'),xlabel='Epoch',ylabel='0.1 × auxiliary Dice + focal')
            axes[row,1].legend(fontsize=9);axes[row,1].grid(alpha=.15)
    fig.suptitle('REAL — Chronic Wound | 671 train / 118 validation | seed 42 | checkpoint dari MAIN Dice',fontsize=14)
    save(fig,'training_objectives')

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for name in names:
        h=histories[name];epochs=[r['epoch'] for r in h];selected=statuses[name]['selected_epoch']
        axes[0].plot(epochs,[r['validation_image_dice'] for r in h],color=COLORS[name],label=name)
        axes[0].scatter(selected,h[selected-1]['validation_image_dice'],color=COLORS[name],s=35)
        axes[1].step(epochs,[r['learning_rate_used'] for r in h],where='post',color=COLORS[name],label=name)
    axes[0].set(title='Checkpoint dipilih dari main validation Image Dice',xlabel='Epoch',ylabel='Image Dice (0–1)',ylim=(0,1))
    axes[1].set(title='Aturan scheduler sama; jadwal aktual dapat berbeda',xlabel='Epoch',ylabel='Learning rate',yscale='log')
    for ax in axes:ax.legend();ax.grid(alpha=.15)
    fig.suptitle('REAL — Chronic Wound | H0=CACHED B1; FRESH: '+', '.join(names[1:]),fontsize=14);save(fig,'checkpoint_and_lr')

    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    for row,split in enumerate(('validation','test')):
        for column,(group,key,title) in enumerate((('all','dice','Main Image Dice'),('small','dice','Small-wound Image Dice'),('all','boundary_iou','Main Boundary IoU'))):
            reports=[metrics[n][split]['subgroups']['small'] if group=='small' else metrics[n][split]['all'] for n in names]
            values=[r['image_based'][key] for r in reports];ax=axes[row,column]
            ax.bar(names,values,color=[COLORS[n] for n in names])
            for i,v in enumerate(values):ax.text(i,v+.015,f'{v:.3f}',ha='center')
            ax.set(title=f'{split}, n={reports[0]["images"]}\n{title}',xlabel='Model',ylabel='Mean per gambar (0–1)',ylim=(0,1.04));ax.grid(axis='y',alpha=.15)
    fig.suptitle('REAL — Chronic Wound | PRIMARY mask | threshold ≥0,5 | 224×224\nSmall: 0<GT≤1% kanvas; source-test deskriptif',fontsize=14);save(fig,'segmentation_and_boundary')

    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for row,split in enumerate(('validation','test')):
        for column,(group,key,title) in enumerate((('gt_positive','total_foreground_misses','GT positif dengan TP=0'),('gt_empty','images_with_prediction','GT kosong dengan FP'))):
            ax=axes[row,column];values=[metrics[n][split][group][key] for n in names]
            ax.bar(names,values,color=[COLORS[n] for n in names])
            for i,v in enumerate(values):ax.text(i,v+.12,str(v),ha='center')
            ax.set(title=f'{split}: {title}\nn={metrics["H0"][split][group]["images"]}',xlabel='Model',ylabel='Jumlah gambar',ylim=(0,max(2,max(values)+1)))
            ax.yaxis.get_major_locator().set_params(integer=True);ax.grid(axis='y',alpha=.15)
    fig.suptitle('REAL — Chronic Wound | PRIMARY mask errors\nGT kosong bukan diagnosis pasien sehat; source-test deskriptif',fontsize=14);save(fig,'foreground_and_empty_errors')

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,split in zip(axes,('validation','test')):
        for i,name in enumerate(names[1:]):
            e=effects[split][name];lo,hi=np.array(e['descriptive_RGB_group_bootstrap_95'])*100;mean=e['mean_delta_image_dice']*100
            ax.plot([lo,hi],[i,i],color=COLORS[name],linewidth=2);ax.scatter(mean,i,color=COLORS[name],s=40)
        ax.axvline(0,color=COLORS['H0'],linestyle='--');ax.set(yticks=range(len(names)-1),yticklabels=names[1:],xlabel='Δ Image Dice vs H0 (poin persentase)',ylabel='Model',title=f'{split}, n={metrics["H0"][split]["all"]["images"]} RGB groups');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    fig.suptitle('REAL — Chronic Wound | Paired bootstrap deskriptif 95%\nBukan patient, seed atau selection uncertainty; source-test deskriptif',fontsize=13);save(fig,'paired_effects')

    fig,ax=plt.subplots(figsize=(13,4.8),constrained_layout=True);ax.axis('off')
    boxes=[(.02,.72,.18,'Foto RGB\n[B,3,224,224]'),(.28,.72,.23,'FUSegNet / P-scSE\nsource forward tetap'),(.68,.72,.27,'Main logits [B,1,224,224]\nsigmoid → mask utama'),(.28,.30,.23,'Stage 3 feature\n[B,32,112,112]'),(.68,.30,.27,'Conv1×1, 33 parameter\nupsample → auxiliary loss ×0,1')]
    for x,y,w,text in boxes:ax.text(x+w/2,y,text,ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round,pad=.6',facecolor='#EFF6FF' if y>.5 else '#F3E8FF',edgecolor='#64748B'),transform=ax.transAxes)
    for first,last in [((.21,.72),(.27,.72)),((.52,.72),(.66,.72)),((.40,.58),(.40,.43)),((.53,.30),(.66,.30))]:
        ax.annotate('',xy=last,xytext=first,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#475569',lw=1.8))
    ax.text(.02,.02,'Training: B1(main,GT) + 0,1×B1(aux,target); H1 target mask, H2 target boundary band\nSaat inference auxiliary dapat dilepas; seluruh jalur mask utama tetap.',fontsize=11,transform=ax.transAxes)
    fig.suptitle('SCHEMATIC — arsitektur H; tensor shapes dan 33 parameter diverifikasi pada model aktual',fontsize=14);save(fig,'auxiliary_architecture')

    results={'status':'COMPLETED — REAL STANDALONE AUXILIARY SUPERVISION SCREENING','created_utc':datetime.now(timezone.utc).isoformat(),
             'run_root':str(root.relative_to(ROOT)),'preregistration':read(root/'preregistration.json'),'protocol':config,
             'manifest_fingerprint':manifest['fingerprint'],'selection':selection,'expansion_decision':read(root/'expansion_decision.json'),
             'statuses':statuses,'metrics':metrics,'profiles':read(root/'architecture_preflight.json'),'resources':resources,
             'paired_effects':effects,'paired_inputs':read(root/'train_paired_controls.json'),'figures':assets,
             'execution':{'H0':'CACHED phase-P B1',**{n:'FRESH real-data full training' for n in names[1:]}},
             'deferred':config['phase_h']['deferred']}
    write_json(ROOT/'reports/cp2-phase-h-results.json',results)
    lines=['# CP2 Fase H — auxiliary segmentation dan boundary supervision','',
           '**COMPLETED — kondisi yang dilaporkan benar-benar dilatih, seed 42.** H0=B1 cached; tidak mengulang baseline atau fase lama.','',
           f'Run `{root.relative_to(ROOT)}`. [Preregistered protocol](../docs/cp2-phase-h.md) · [Results JSON](cp2-phase-h-results.json) · [Auxiliary diagnosis](cp2-phase-h-analysis.json).','',
           'Manifest tetap **671/118/261**, duplicate-RGB-group split, bukan patient-level. Raw source tetap 831/278. Main B1, source architecture, shared initial tensors, loader/augmentation, float32/TF32, Adam/batch2 dan kebijakan checkpoint tetap.','',
           f'H1 diagnosis tersimpan sebelum H2; gate perluasan **{results["expansion_decision"]["expand"]}**. Executed: {", ".join(names[1:])}. H3 dan kombinasi ditunda.','',
           '| Model | Selected / completed epoch | Val Image Dice | Small val | Boundary IoU val | Test Image Dice | Empty FP val / test |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for n in names:
        v,t=metrics[n]['validation'],metrics[n]['test'];s=statuses[n]
        lines.append(f'| {n} | {s["selected_epoch"]} / {s["completed_epochs"]} | {v["all"]["image_based"]["dice"]:.4f} | {v["subgroups"]["small"]["image_based"]["dice"]:.4f} | {v["all"]["image_based"]["boundary_iou"]:.4f} | {t["all"]["image_based"]["dice"]:.4f} | {v["gt_empty"]["images_with_prediction"]}/7 · {t["gt_empty"]["images_with_prediction"]}/16 |')
    lines+=['',f'**Validation-only selection: {selection["selected_candidate"]}.** Semua guard dipertahankan; bukan memilih pemimpin source-test. Checkpoint berdasarkan main Image Dice, bukan auxiliary score atau minimum total loss. Source-test sudah terekspos CP1/P/L/A, bersifat deskriptif.','']
    for n,d in selection['decisions'].items():lines.append(f'- {n}: eligible **{d["eligible"]}**, checks `{d["checks"]}`.')
    captions={
      'training_objectives':'Kolom kiri adalah objective MAIN yang sama; kanan auxiliary berbobot 0,1. Target H1/H2 berbeda. Besar loss bukan norm gradien atau bukti kausal; train memakai augmentasi.',
      'checkpoint_and_lr':'Titik adalah selected main validation Dice, garis LR dari aturan scheduler yang sama. Jumlah update aktual berbeda karena early stopping; bukan fixed-budget ablation.',
      'segmentation_and_boundary':'Seluruh skor berasal dari PRIMARY mask pada gambar sama. Small berdasarkan kanvas berpadded, bukan luas fisik; baca bersama GT-positive/empty dan pixel metrics.',
      'foreground_and_empty_errors':'Nol diberi anotasi bila benar-benar tidak ada kasus. Total miss berarti GT positif dengan TP=0; GT kosong bukan diagnosis sehat, dan jumlah FP images tidak menggantikan FP pixels.',
      'paired_effects':'Resampling pasangan RGB-group/gambar, tidak mencakup pasien/crop yang tidak diketahui, seed, adaptive checkpoint selection atau multiple comparisons; bukan klaim signifikansi.',
      'auxiliary_architecture':'Head tambahan hanya memberi gradient saat training dan dapat dilepas. H2 berbeda dari L3: H2 memprediksi target batas pada head terpisah; L3 memberi distance loss pada mask utama.'}
    for stem,caption in captions.items():lines += ['',f'![{stem}](figures/cp2-phase-h/{stem}.png)','',caption]
    lines+=['','## Subgroups utama','',
            '| Split | Model | GT-positive Dice | GT-empty Dice | Total miss / positive n | Empty FP pixels | Data Dice | Pixel precision / recall | PFOM |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('validation','test'):
        for n in names:
            m=metrics[n][split]
            lines.append(f'| {split} | {n} | {m["gt_positive"]["image_based"]["dice"]:.4f} | {m["gt_empty"]["image_based"]["dice"]:.4f} | {m["gt_positive"]["total_foreground_misses"]}/{m["gt_positive"]["images"]} | {m["gt_empty"]["counts"]["fp"]} | {m["all"]["data_based"]["dice"]:.4f} | {m["all"]["data_based"]["precision"]:.4f} / {m["all"]["data_based"]["recall"]:.4f} | {m["all"]["image_based"]["pfom"]:.4f} |')
    lines+=['','## Auxiliary diagnosis sesudah selection','',
            'Tabel ini mengevaluasi task head masing-masing, **bukan kualitas mask utama**. H1 dan H2 mempunyai definisi target berbeda, sehingga angkanya tidak menjadi ranking model atau dasar reselection.','',
            '| Head | Target | Target pixel fraction | Aux Image Dice val | Aux pixel precision / recall |',
            '|---|---|---:|---:|---:|']
    for n in names[1:]:
        a=analysis['models'][n];m=a['auxiliary_metrics']['all']
        lines.append(f'| {n} | {"segmentation mask" if n=="H1" else "3×3 boundary band"} | {100*a["auxiliary_target_foreground_fraction"]:.3f}% | {m["image_based"]["dice"]:.4f} | {m["data_based"]["precision"]:.4f} / {m["data_based"]["recall"]:.4f} |')
    lines+=['','## Komputasi dan artefak','',
            '| Model | Train registered / active params | Mask-only active params | Last training invocation (min) | Peak train allocated / reserved (GiB) | Mask-only median / P95 (ms) |',
            '|---|---:|---:|---:|---:|---:|']
    profiles=results['profiles']
    for n in names:
        p=profiles['profiles'][n];i=profiles['mask_only_inference_profiles'][n];s=statuses[n];cost=analysis['models'][n]['mask_only_inference']
        lines.append(f'| {n} | {p["registered_parameters"]:,} / {p["active_forward_parameters"]:,} | {i["active_forward_parameters"]:,} | {s["seconds_this_invocation"]/60:.2f} | {s["peak_cuda_allocated_bytes"]/1024**3:.2f} / {s["peak_cuda_reserved_bytes"]/1024**3:.2f} | {cost["median_ms"]:.2f} / {cost["p95_ms"]:.2f} |')
    lines+=['','H0 timing training berasal dari P; GPU bersama berubah. H1/H2 inference architecture identik ketika head dilepas, jadi selisih latency tidak menunjukkan metode tertentu intrinsik lebih cepat. MACs dalam JSON hanya Conv/Linear, bukan seluruh operasi. [Resource JSON](cp2-phase-h-resources.json) menyimpan epoch/step time, overall GPU dan shared memory.','',
            'Checkpoints/optimizer/scheduler/RNG, probabilities/GT, auxiliary predictions, logs dan clinical galleries tetap lokal. Numeric tables dan aggregate charts publik. Satu seed, label ambigu/crop-patient dependence dan validation reuse membatasi generalisasi. Tidak ada novelty, manfaat klinis, atau generalisasi eksternal yang terbukti.','',
            '[Artifact QA](cp2-phase-h-validation.json) · [Review ilmiah](cp2-phase-h-review.md) · [Visual/engineering QA](cp2-phase-h-quality.json). Berkas ini baru diterbitkan setelah pemeriksaan aktual selesai.']
    (ROOT/'reports/cp2-phase-h.md').write_text('\n'.join(lines)+'\n')
    print('Rendered completed real H screening and six aggregate figures')


if __name__=='__main__':
    main()
