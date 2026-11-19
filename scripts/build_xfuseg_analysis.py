#!/usr/bin/env python3
"""Verify x-FUSegNet outputs, evaluate OOF/PFOM, execute and stage a notebook."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from PIL import Image

from cp1_analysis import ROOT, configure_style, TEAL, CORAL, NAVY
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json
from foot_ulcer_segmentation.evaluation import SegmentationEvaluator
from pfom_metrics import score_masks


def read(path):
    return json.loads(Path(path).read_text())


def prepare(batch, diagnostic=False):
    config = read(ROOT / "configs/xfusegnet.json")
    run = ROOT / config["run_dir"]
    split = read(ROOT / config["data"]["manifest"])
    if digest_json({k:v for k,v in split.items() if k != 'fingerprint'}) != split['fingerprint']:
        raise ValueError("Split fingerprint changed")
    pfom = read(ROOT / "configs/pfom.json")
    predictions = None if diagnostic else read(ROOT / config["workflow_dir"] / "prediction-manifest.json")
    if not diagnostic:
        if predictions['config_fingerprint'] != digest_json(config) or predictions['split_fingerprint'] != split['fingerprint']:
            raise ValueError("Inference protocol identity changed")
        status = read(run / 'status.json')
        if status['status'] != 'completed' or status['smoke_only']:
            raise ValueError("Full training has not finished")
        environment = read(run / 'environment.json')
        for name, expected in {**environment['pipeline_sha256'], **environment['xfuseg_runtime_sha256']}.items():
            if digest_file(ROOT / name) != expected:
                raise ValueError(f"Training source changed: {name}")
        for identity in predictions['checkpoints']:
            path = run / 'author-checkpoints' / f"best_model_{identity['fold']}.pth"
            if digest_file(path) != identity['checkpoint_sha256']:
                raise ValueError("Selected checkpoint changed")
        if (len(predictions['oof']) != 1010 or len(predictions['test_ensemble']) != 200
                or predictions['official_test_metrics'] is not None or predictions['oof_is_ensemble_evaluation']):
            raise ValueError("Invalid OOF/test declaration")
    batch.mkdir(parents=True, exist_ok=True)
    indexed = {} if diagnostic else {r['name']:r for r in predictions['oof']}
    samples = [r for r in split['labeled'] if not diagnostic or r['name'] in split['oof_examples']]
    oof, evaluator = [], SegmentationEvaluator(.5)
    for sample in samples:
        name = sample['name']
        fold = next(f['fold'] for f in split['folds'] if name in f['validation'])
        if digest_file(ROOT / sample['image_path']) != sample['image_sha256'] or digest_file(ROOT / sample['mask_path']) != sample['mask_sha256']:
            raise ValueError("Raw dataset changed")
        truth = np.asarray(Image.open(ROOT / sample['mask_path']).convert('L')) >= 127.5
        if diagnostic:
            probability = truth.astype(np.float32)
            folder=batch/'diagnostic-predictions/oof'; folder.mkdir(parents=True, exist_ok=True)
            path=folder/name; Image.fromarray(truth.astype(np.uint8)*255).save(path)
        else:
            record = indexed[name]
            if record['fold'] != fold or name in split['folds'][fold-1]['train']:
                raise ValueError("OOF prediction was made by a model trained on that sample")
            for kind in ['probability','mask']:
                if digest_file(ROOT / record[kind]['path']) != record[kind]['sha256']:
                    raise ValueError("Cached OOF prediction changed")
            probability = np.load(ROOT / record['probability']['path'], allow_pickle=False)
            path = ROOT / record['mask']['path']
        mask = np.asarray(Image.open(path)) > 0
        if probability.dtype != np.float32 or probability.shape != (512,512) or not np.array_equal(mask,probability >= .5):
            raise ValueError("OOF probability/mask mismatch")
        measured = evaluator.add(name, truth, probability)
        if not diagnostic:
            for key,value in measured.items():
                if key != 'name' and not np.isclose(value,indexed[name]['evaluation'][key],atol=1e-12,rtol=0):
                    raise ValueError("OOF metric parity failed")
        oof.append({**sample, **measured, **score_masks(truth, mask, beta=pfom['beta'], **pfom['canny']),
                    'fold':fold,'prediction_path':str(path.relative_to(ROOT)), 'gt_area_percent':truth.mean()*100})
    summary = evaluator.summary()
    if not diagnostic:
        expected = read(run/'oof-validation.json')
        if summary['offline_hard_line']['counts'] != expected['offline_hard_line']['counts']:
            raise ValueError("Aggregate OOF counts differ from recorded inference")
        for family in ['data_based','image_based']:
            for key,value in summary['offline_hard_line'][family].items():
                if not np.isclose(value,expected['offline_hard_line'][family][key],atol=1e-12,rtol=0):
                    raise ValueError("Aggregate OOF metrics differ from recorded inference")
    frame = pd.DataFrame(oof)
    folds=[]
    for fold in split['folds']:
        number=fold['fold']; rows=frame[frame.fold.eq(number)]
        counts={key:int(rows[key].sum()) for key in ['tp','tn','fp','fn']}
        data_dice=2*counts['tp']/(2*counts['tp']+counts['fp']+counts['fn']) if 2*counts['tp']+counts['fp']+counts['fn'] else None
        status={'completed_epochs':1,'selected_epoch':1,'seconds':0,'peak_cuda_allocated_bytes':0} if diagnostic else read(run/f'fold-{number}/status.json')
        folds.append({'fold':number,'train_images':808,'oof_images':len(rows),'epochs':status['completed_epochs'],
                      'selected_epoch':status['selected_epoch'],'training_minutes':status['seconds']/60,
                      'peak_cuda_gib':status['peak_cuda_allocated_bytes']/1024**3,'image_dice':rows.dice.mean(),
                      'data_dice':data_dice,'pfom':rows.pfom.mean(),
                      'oof_identical_rgb_to_train':fold['validation_images_identical_rgb_to_train']})
    test=[]
    test_index={} if diagnostic else {(r['fold'],r['name']):r for r in predictions['test_per_fold']}
    test_ensemble={} if diagnostic else {r['name']:r for r in predictions['test_ensemble']}
    for sample in split['test']:
        name=sample['name']
        if diagnostic and name not in split['test_examples']:
            continue
        if digest_file(ROOT/sample['image_path']) != sample['image_sha256']:
            raise ValueError("Raw test image changed")
        if diagnostic:
            mean=np.zeros((512,512),np.float32)
            folder=batch/'diagnostic-predictions/test';folder.mkdir(parents=True,exist_ok=True)
            path=folder/name;Image.fromarray(np.zeros((512,512),np.uint8)).save(path)
        else:
            individual=[]
            for fold in range(1,6):
                record=test_index[(fold,name)]['probability']
                if digest_file(ROOT/record['path']) != record['sha256']:
                    raise ValueError("Individual-fold test probability changed")
                individual.append(np.load(ROOT/record['path'],allow_pickle=False))
            mean=np.mean(np.array(individual),axis=0)
            record=test_ensemble[name]
            for kind in ['probability','mask']:
                if digest_file(ROOT/record[kind]['path']) != record[kind]['sha256']:
                    raise ValueError("Ensemble artifact changed")
            if not np.array_equal(mean,np.load(ROOT/record['probability']['path'],allow_pickle=False)):
                raise ValueError("Saved ensemble is not the exact five-model mean")
            path=ROOT/record['mask']['path']
        mask=np.asarray(Image.open(path))>0
        if not np.array_equal(mask,mean >= .5):
            raise ValueError("Ensemble threshold mismatch")
        test.append({**sample,'prediction_path':str(path.relative_to(ROOT)),'predicted_area_percent':mask.mean()*100})
    frame.to_csv(batch/'oof-per-image.csv',index=False,float_format='%.15g')
    pd.DataFrame(folds).to_csv(batch/'folds.csv',index=False)
    pd.DataFrame(test).to_csv(batch/'test-predictions.csv',index=False)
    memberships=[{'name':r['name'],'provided_split':r['provided_split'],'validation_fold':next(f['fold'] for f in split['folds'] if r['name'] in f['validation']),
                  'image_sha256':r['image_sha256'],'mask_sha256':r['mask_sha256']} for r in split['labeled']]
    pd.DataFrame(memberships).to_csv(batch/'split-membership.csv',index=False)
    write_json(batch/'analysis.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'diagnostic_fixture':diagnostic,
               'config':config,'config_fingerprint':digest_json(config),'split_fingerprint':split['fingerprint'],
               'oof_evaluation':summary,'oof_mean_pfom':frame.pfom.mean(),'oof_positive_gt_pfom':frame.loc[frame.gt_pixels.gt(0),'pfom'].mean(),
               'oof_sentinel2_proxy':frame.pfom_paper_sentinel2_proxy.mean(),'pfom_protocol':pfom,
               'official_test_metrics':None,'oof_is_ensemble_evaluation':False,
               'oof_examples':split['oof_examples'],'test_examples':split['test_examples'],
               'all_oof_pairs_verified':True,'all_test_ensembles_verified':True,
               'checkpoints':[] if diagnostic else predictions['checkpoints'],'source_sha256':digest_file(Path(__file__))})


def load_context(batch):
    batch=Path(batch)
    return {'batch':batch,'metadata':read(batch/'analysis.json'),'oof':pd.read_csv(batch/'oof-per-image.csv'),
            'folds':pd.read_csv(batch/'folds.csv'),'test':pd.read_csv(batch/'test-predictions.csv')}


def save_figure(context,figure,name):
    if context['metadata']['diagnostic_fixture'] and figure._suptitle is not None:
        figure._suptitle.set_text('DIAGNOSTIC FIXTURE · bukan hasil eksperimen\n'+figure._suptitle.get_text())
    path=context['batch']/'figures';path.mkdir(exist_ok=True)
    figure.savefig(path/(name+'.png'),dpi=200,bbox_inches='tight')
    return figure


def plot_curves(context):
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    run=ROOT/context['metadata']['config']['run_dir']
    for row in context['folds'].itertuples():
        if context['metadata']['diagnostic_fixture']:
            epochs,loss,dice=[1],[.5],[.5]
        else:
            history=read(run/f'fold-{row.fold}/history.json')
            epochs=[r['epoch'] for r in history];loss=[r['validation']['dice_loss + focal_loss'] for r in history]
            dice=[r['validation']['fscore'] for r in history]
        line=axes[0].plot(epochs,loss,label=f'Fold {row.fold}')[0]
        axes[1].plot(epochs,np.array(dice)*100,color=line.get_color())
        for ax in axes:
            ax.axvline(row.selected_epoch,color=line.get_color(),alpha=.4,linestyle=':')
    axes[0].set(title='Validation loss',ylabel='Dice loss + focal loss')
    axes[1].set(title='Validation F-score dalam log',ylabel='Batch F-score (%)')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.5)
    axes[0].legend(frameon=False)
    fig.suptitle('x-FUSegNet · lima model dengan checkpoint loss/IoU penulis',fontweight='bold')
    return save_figure(context,fig,'01_learning_curves')


def plot_oof(context):
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    rows=context['folds'];x=np.arange(len(rows));width=.35
    axes[0].bar(x-width/2,rows.image_dice*100,width,color=TEAL,label='Image Dice')
    axes[0].bar(x+width/2,rows.data_dice*100,width,color=NAVY,label='Data Dice')
    axes[0].set(xticks=x,xticklabels=[f'Fold {i}' for i in rows.fold],ylabel='OOF Dice (%)',ylim=(0,105))
    axes[0].legend(frameon=False)
    axes[1].scatter(context['oof'].gt_area_percent,context['oof'].dice*100,s=12,alpha=.5,color=TEAL)
    axes[1].set(xlabel='Area GT pada kanvas 512x512 (%)',ylabel='OOF image Dice (%)',ylim=(-3,103))
    for ax in axes:ax.grid(alpha=.4)
    fig.suptitle('Validasi internal OOF · setiap gambar memakai model fold yang menahannya',fontweight='bold')
    return save_figure(context,fig,'02_oof_validation')


def plot_gallery(context,split):
    names=context['metadata'][split+'_examples']
    rows=context['oof' if split=='oof' else 'test'].set_index('name')
    columns=4 if split=='oof' else 3
    fig,axes=plt.subplots(len(names),columns,figsize=(3.2*columns,3.1*len(names)),squeeze=False,layout='constrained')
    for index,name in enumerate(names):
        row=rows.loc[name]
        image=np.asarray(Image.open(ROOT/row.image_path).convert('RGB'))
        prediction=np.asarray(Image.open(ROOT/row.prediction_path))>0
        axes[index,0].imshow(image)
        if split=='oof':
            truth=np.asarray(Image.open(ROOT/row.mask_path).convert('L'))>=127.5
            axes[index,1].imshow(truth,cmap='gray',vmin=0,vmax=1)
            axes[index,2].imshow(prediction,cmap='gray',vmin=0,vmax=1)
            overlay=image.astype(float)/255
            for selection,color in [(truth&prediction,'#2EC27E'),(~truth&prediction,'#F15B5D'),(truth&~prediction,'#5EA5FF')]:
                import matplotlib.colors as colors
                overlay[selection]=overlay[selection]*.45+np.array(colors.to_rgb(color))*.55
            axes[index,3].imshow(overlay)
            caption=f'{name} · fold {row.fold}\nDice {row.dice*100:.2f}% · PFOM {row.pfom:.4f}'
            titles=['Gambar','Ground truth','Prediksi OOF','TP / FP / FN']
        else:
            axes[index,1].imshow(prediction,cmap='gray',vmin=0,vmax=1)
            axes[index,2].imshow(image)
            if prediction.any() and not prediction.all():
                axes[index,2].contour(prediction,levels=[.5],colors=[TEAL],linewidths=1)
            caption=f'{name}\nPrediksi area {row.predicted_area_percent:.2f}% · tanpa GT'
            titles=['Gambar test','Mean 5 model >= 0.5','Batas prediksi ensemble']
        axes[index,0].text(.5,-.04,caption,ha='center',va='top',transform=axes[index,0].transAxes,fontsize=9)
        for column,title in enumerate(titles):
            axes[index,column].set_axis_off()
            if index==0:axes[index,column].set_title(title,fontsize=11)
    fig.suptitle('Contoh OOF tetap · validasi lokal' if split=='oof' else 'Test ensemble · GT dan skor test tidak tersedia',fontweight='bold')
    if split=='oof':
        fig.legend(handles=[Patch(color=c,label=t) for c,t in [('#2EC27E','TP'),('#F15B5D','FP'),('#5EA5FF','FN')]],
                   loc='lower center',ncol=3,frameon=False)
    return save_figure(context,fig,'03_oof_examples' if split=='oof' else '04_test_ensemble_examples')


def build_notebook(batch):
    context=load_context(batch);diagnostic=context['metadata']['diagnostic_fixture'];cells=[]
    def md(value):cells.append(nbformat.v4.new_markdown_cell(value))
    def code(value):cells.append(nbformat.v4.new_code_cell(value))
    md('# CP1 · Reproduksi x-FUSegNet\n\n'+('**DIAGNOSTIC FIXTURE: prediksi sintetis untuk menguji tampilan; bukan hasil eksperimen.**\n\n' if diagnostic else '')+
       '**Lima model FUSegNet · native 512×512 · rata-rata probabilitas sebelum threshold 0.5.**\n\n'
       'OOF adalah validasi internal dari satu model fold per gambar. **Skor test challenge tidak tersedia** karena 200 mask test privat tidak ada di dataset lokal.')
    code("from pathlib import Path\nimport sys\nROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/xfusegnet.json').exists())\nsys.path.insert(0,str(ROOT/'scripts'))\n"
         "import matplotlib.pyplot as plt\nfrom IPython.display import display,HTML\nfrom build_xfuseg_analysis import load_context,plot_curves,plot_oof,plot_gallery\nfrom cp1_analysis import configure_style\n"
         "%matplotlib inline\nconfigure_style()\nc=load_context(ROOT/"+repr(str(batch.relative_to(ROOT)))+")\n"
         "def table(df):\n    display(HTML(df.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')))\n"
         "print('Diagnostic fixture:',c['metadata']['diagnostic_fixture'])\nprint('Split:',c['metadata']['split_fingerprint'])")
    md('## 1 · Metode dan kesetaraan kode\n\nLoop numerik dari `xfusegnet_train.py` pada revisi penulis yang dipin dieksekusi langsung; hook menambah status, checkpoint resume, dan pencatatan. '
       'Arsitektur, augmentasi x-FUSegNet, loss, optimizer, scheduler, dan kebijakan checkpoint sumber dipertahankan. Python shuffle seed 42 diikuti `KFold(5, shuffle=False)`. '
       'Global torch/numpy seed 42 ditambahkan untuk reproducibility; RNG berlanjut antar-fold. Urutan file asli penulis tidak tersedia. '
       'Pool 1.010 gambar berlabel berasal dari 810 provided-train + 200 provided-validation, masing-masing fold 808/202; test 200 terpisah.')
    code("table(c['folds'][['fold','train_images','oof_images','epochs','selected_epoch','training_minutes','peak_cuda_gib','oof_identical_rgb_to_train']])\nplot_curves(c)\nplt.show()")
    md('## 2 · Evaluasi OOF internal\n\nSetiap gambar dievaluasi oleh model yang tidak memakai filename tersebut dalam training fold-nya. '
       'Checkpoint dipilih pada validation fold yang sama, sehingga ini hasil validasi internal. OOF **bukan hasil ensemble lima model pada independent test set**. '
       'Mask training tetap dibagi 255 termasuk nilai 32/255 pada satu mask. Untuk metrik biner/PFOM saja, GT memakai threshold 0.5. PFOM empty-edge 1/0; proxy sentinel 2 terpisah.')
    code("m=c['metadata'];s=m['oof_evaluation']['offline_hard_line']\nprint('OOF image Dice:',f\"{s['image_based']['dice']*100:.2f}%\")\n"
         "print('OOF data Dice:',f\"{s['data_based']['dice']*100:.2f}%\")\nprint('OOF mean PFOM:',f\"{m['oof_mean_pfom']:.4f}\")\n"
         "print('Sentinel-2 proxy:',f\"{m['oof_sentinel2_proxy']:.4f}\")\ntable(c['folds'][['fold','image_dice','data_dice','pfom']])\nplot_oof(c)\nplt.show()")
    md('## 3 · Prediksi OOF\n\nLima nama ditetapkan sebelum training: gambar pertama validation masing-masing fold. Hijau TP, merah FP, biru FN. Tidak ada pemilihan ulang berdasarkan hasil model.')
    code("plot_gallery(c,'oof')\nplt.show()")
    md('## 4 · Ensemble pada 200 gambar test\n\nSetiap gambar diprediksi oleh lima checkpoint terpilih, kemudian `np.mean` pada lima probability map float32 dan threshold `>=0.5`. '
       'Semua mean dan PNG diperiksa ulang. GT test tidak tersedia; galeri berikut **tidak memiliki skor Dice/PFOM**. Lima nama contoh ditetapkan dari urutan filename sebelum training. '
       'Angka **89.23%** dalam paper adalah skor test challenge milik penulis; angka itu tidak diverifikasi oleh validasi OOF lokal.')
    code("print('Gambar test dengan ensemble:',len(c['test']))\nprint('Official test metrics:',c['metadata']['official_test_metrics'])\nplot_gallery(c,'test')\nplt.show()")
    md('## 5 · Batas interpretasi dan referensi\n\nSumber memiliki duplikasi RGB antar-fold yang dipertahankan agar mengikuti protokol sumber; jumlahnya dilaporkan per fold. '
       'Independensi pasien belum terverifikasi. Fold asli penulis, global seed asli, dan evaluasi test privat belum tersedia. Skor OOF 512×512 tidak dibandingkan langsung sebagai peningkatan atas test Chronic Wound 224×224. '
       'Reproduksi x-FUSegNet merupakan bagian reproduksi, belum merupakan novelty CP2.\n\n'
       '[Paper §3.2.4](https://arxiv.org/abs/2305.02961v2) · [Kode penulis](https://github.com/mrinal054/FUSegNet) · '
       '[Dataset FUSeg](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge) · [Protokol lokal](../docs/xfusegnet.md)')
    for index,cell in enumerate(cells):cell['id']=f'xfuseg-{index:02d}'
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'fuseg-analysis','display_name':'FUSeg · Analysis','language':'python'},
                                    'xfusegnet':{'diagnostic_fixture':diagnostic}})
    prefix=batch/'kernel';os.environ['JUPYTER_PATH']=str(prefix/'share/jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    subprocess.run([sys.executable,'-m','ipykernel','install','--prefix',str(prefix),'--name','fuseg-analysis','--display-name','FUSeg · Analysis'],check=True)
    NotebookClient(notebook,timeout=300,kernel_name='fuseg-analysis',allow_errors=False,resources={'metadata':{'path':str(ROOT)}}).execute()
    codes=[c for c in notebook.cells if c.cell_type=='code']
    if not all(c.execution_count is not None for c in codes) or any(o.output_type=='error' for c in codes for o in c.outputs):
        raise ValueError('Notebook execution incomplete')
    notebook.metadata['xfusegnet']['all_cells_executed']=True;nbformat.validate(notebook)
    target=batch/'published/notebooks/03_cp1_xfusegnet.ipynb';target.parent.mkdir(parents=True,exist_ok=True);nbformat.write(notebook,target)


def build_report(batch):
    c=load_context(batch);m=c['metadata'];s=m['oof_evaluation']['offline_hard_line'];diagnostic=m['diagnostic_fixture']
    lines=['# CP1 · Reproduksi x-FUSegNet','',
           '**DIAGNOSTIC FIXTURE: bukan hasil training.**' if diagnostic else '**Training lima fold, inference ensemble, dan analisis selesai.**','',
           '[Notebook executed](../notebooks/03_cp1_xfusegnet.ipynb) · [Protokol dan audit kesetaraan](../docs/xfusegnet.md)','',
           '## Cakupan','',
           'FUSeg 512×512: 1.010 gambar berlabel, lima fold 808 train / 202 validation, dan 200 gambar test tanpa mask. Loop training asli penulis dipakai dengan hook pencatatan/resume. Rata-rata lima probabilitas float32 mendahului threshold >=0.5.','',
           '## Validasi internal OOF','',
           f"- Image Dice: **{s['image_based']['dice']*100:.2f}%**.",f"- Data Dice: **{s['data_based']['dice']*100:.2f}%**.",
           f"- Mean PFOM (empty-edge 1/0): **{m['oof_mean_pfom']:.4f}**; sentinel-2 proxy **{m['oof_sentinel2_proxy']:.4f}**.",'',
           'Setiap gambar hanya diprediksi model fold yang menahannya. Checkpoint dipilih pada validation fold tersebut. Skor ini bukan evaluasi ensemble lima model pada independent test set.','',
           '| Fold | Epoch | Checkpoint | OOF image Dice (%) | OOF data Dice (%) | PFOM | Menit training |',
           '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in c['folds'].itertuples():
        lines.append(f'| {r.fold} | {r.epochs} | {r.selected_epoch} | {r.image_dice*100:.2f} | {r.data_dice*100:.2f} | {r.pfom:.4f} | {r.training_minutes:.1f} |')
    lines += ['', '## Test ensemble','',f"Tersedia **{len(c['test'])} prediksi ensemble test**. Semua mean probabilitas lima model dan PNG threshold diverifikasi. Tidak ada skor Dice/PFOM test lokal karena mask test tidak tersedia. **Skor paper 89.23% belum diverifikasi secara lokal.**",'',
              '## Batas kesetaraan','',
              'Urutan direktori/fold penulis tidak dipublikasikan; urutan lokal dipersistenkan sebelum training. Seed torch/numpy 42 adalah tambahan reproducibility, dengan RNG berlanjut antar-fold. Seluruh versi dependency training baseline tetap sama; dependency plotting tambahan berada di environment khusus.','',
              'Augmentasi dan mask fractional training mengikuti xfusegnet_train.py secara langsung. Evaluasi OOF/PFOM membinarisasi GT pada >=0.5 secara terpisah. Loss probability-input, checkpoint loss/IoU elif, dan metadata epoch off-by-one sumber dipertahankan. Tabel melaporkan jumlah epoch training aktual.','',
              'Duplikasi RGB antar-fold dipertahankan dan dihitung; independensi pasien belum terverifikasi. Validasi OOF memakai fold yang juga memilih checkpoint. Perbedaan dataset/resolusi/protokol menghalangi perbandingan langsung dengan hasil test Chronic Wound.','',
              'Bukti: [folds](tables/cp1-xfusegnet-folds.csv), [OOF per gambar](tables/cp1-xfusegnet-oof-per-image.csv), [keanggotaan fold](tables/cp1-xfusegnet-split-membership.csv), [metadata](tables/cp1-xfusegnet-metadata.json).','']
    target=batch/'published/reports/cp1-xfusegnet-01.md';target.parent.mkdir(parents=True,exist_ok=True);target.write_text('\n'.join(lines))
    tables=batch/'published/reports/tables';tables.mkdir(exist_ok=True)
    for source,target in [('folds.csv','folds.csv'),('oof-per-image.csv','oof-per-image.csv'),('test-predictions.csv','test-predictions.csv'),('split-membership.csv','split-membership.csv'),('analysis.json','metadata.json')]:
        shutil.copyfile(batch/source,tables/('cp1-xfusegnet-'+target))
    if not diagnostic:
        base=read(batch/'publication-base.json');readme=base['readme_text']
        marker='<!-- XFUSEGNET_STATUS -->';end='<!-- /XFUSEGNET_STATUS -->'
        replacement=(f"{marker}\n**x-FUSegNet reproduction completed:** five-fold training on FUSeg 512×512, "
                     f"**{s['image_based']['dice']*100:.2f}% internal OOF image Dice**, and 200 verified test ensemble predictions. "
                     "Official challenge test metrics remain unavailable because test masks are private. "
                     "[Open the executed notebook](notebooks/03_cp1_xfusegnet.ipynb) · [Read the report](reports/cp1-xfusegnet-01.md).\n"+end)
        if readme.count(marker)!=1 or readme.count(end)!=1:
            raise ValueError('README publication markers missing')
        prefix,rest=readme.split(marker,1);_,suffix=rest.split(end,1)
        (batch/'published/README.md').write_text(prefix+replacement+suffix)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',default='outputs/xfusegnet/cp1-xfusegnet-01')
    parser.add_argument('--smoke-analysis',action='store_true');args=parser.parse_args()
    batch=ROOT/args.batch_dir
    if args.smoke_analysis and batch.name=='cp1-xfusegnet-01':
        raise ValueError('Diagnostic fixture needs a separate directory')
    prepare(batch,diagnostic=args.smoke_analysis);build_notebook(batch);build_report(batch)
    if not args.smoke_analysis:
        files=[{'source':str(p.relative_to(batch)),'destination':str(p.relative_to(batch/'published'))} for p in sorted((batch/'published').rglob('*')) if p.is_file()]
        write_json(batch/'publication-manifest.json',{'files':files})
    print(f'Analysis complete; diagnostic={args.smoke_analysis}',flush=True)


if __name__=='__main__':main()
