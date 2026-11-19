#!/usr/bin/env python3
"""Build train-only CP1 EDA, scientific figures, report, and an executed notebook.

Run: .venv-analysis/bin/python scripts/build_cp1_eda.py
Input decoding and augmentation run in the existing frozen baseline environment.
No training entrypoint, model, inference, new split, or dependency installation.
"""

import argparse
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import FancyBboxPatch
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from PIL import Image

from cp1_eda import (ROOT, RUN, OUTPUT, CSV, SUMMARY, read, write, sha, load_identity,
                     area_statistics, area_bins, choose_examples, version_record,
                     capture_protected_inputs, verify_protected_inputs)

FIGURES = OUTPUT / 'figures'
PUBLIC_FIGURES = ROOT / 'reports/figures/cp1-eda-01'
PUBLIC_MANIFEST = ROOT / 'reports/tables/cp1-eda-split-manifest.json'
PUBLIC_PREPROCESSING = ROOT / 'reports/tables/cp1-eda-preprocessing.json'
PUBLIC_VERIFICATION = ROOT / 'reports/tables/cp1-eda-verification.json'
REPORT = ROOT / 'reports/cp1-eda-01.md'
NOTEBOOK = ROOT / 'notebooks/04_cp1_eda.ipynb'
COLORS = {'ivory':'#FAF7F2', 'ink':'#352B43', 'lavender':'#DDD5E8', 'copper':'#B86545', 'gray':'#746D78'}
FIGURE_NAMES = ['01_pixel_composition','02_wound_area_distribution','03_train_examples','04_preprocessing_augmentation']


def configure_style():
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white',
        'text.color':COLORS['ink'],'axes.labelcolor':COLORS['ink'],'xtick.color':COLORS['ink'],
        'ytick.color':COLORS['ink'],'font.family':'DejaVu Sans','font.size':15,
        'axes.titlesize':21,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':COLORS['lavender'],
        'svg.fonttype':'none','savefig.facecolor':'white','figure.dpi':110})


def number(value):
    return f'{int(value):,}'.replace(',','.')


def percent(value, digits=2):
    return f'{value:.{digits}f}'.replace('.',',') + '%'


def stats_table(summary):
    return pd.DataFrame([{'Cakupan':label,'n':summary[key]['n'], **{
        name:summary[key][field]*100 if summary[key][field] is not None else np.nan
        for name,field in [('Min (%)','min'),('Q1 (%)','q1'),('Median (%)','median'),
                           ('Q3 (%)','q3'),('Max (%)','max'),('Mean (%)','mean')]}}
        for label,key in [('Seluruh train','area_statistics_all'),('GT positif saja','area_statistics_positive')]])


def bin_table(summary):
    return pd.DataFrame([{'Bin luas kanvas':r['label'],'Jumlah gambar':r['count'],
                          '% dari seluruh train':r['percent_of_train']} for r in summary['area_bins']])


def source_wiring_check():
    """Check the project call sites without executing the training engine."""
    data = ast.parse((ROOT/'src/foot_ulcer_segmentation/data.py').read_text())
    engine = ast.parse((ROOT/'src/foot_ulcer_segmentation/engine.py').read_text())
    make = next(n for n in data.body if isinstance(n,ast.FunctionDef) and n.name=='make_dataset')
    defaults = dict(zip([a.arg for a in make.args.args][-len(make.args.defaults):], make.args.defaults))
    assert ast.literal_eval(defaults['augment']) is False
    calls = []
    for function in engine.body:
        if not isinstance(function,ast.FunctionDef):continue
        for call in ast.walk(function):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='make_dataset':
                augment=next((ast.literal_eval(k.value) for k in call.keywords if k.arg=='augment'),False)
                calls.append({'function':function.name,'line':call.lineno,'split_expression':ast.unparse(call.args[2]),'augment':augment})
    assert any(r['split_expression']=="'train'" and r['augment'] for r in calls)
    assert any(r['split_expression']=="'validation'" and not r['augment'] for r in calls)
    assert any(r['split_expression']=='split' and not r['augment'] for r in calls)
    return {'make_dataset_default_augment':False,'engine_calls':calls,
            'validation_and_evaluation_skip_training_augmentation':True,
            'verification':'AST of frozen project data.py and engine.py; no entrypoint imported'}


def load_verified(check_raw=True):
    """Reconcile CSV, JSON, saved membership, checksums, demos, and raw inputs."""
    manifest,config,environment,identity=load_identity()
    s=read(SUMMARY);f=pd.read_csv(CSV,float_precision='round_trip')
    m=read(ROOT/s['preprocessing_metadata_path'])
    assert sha(ROOT/'scripts/cp1_eda.py')==s['generator_source_sha256'], 'Worker source differs from the prepared EDA; run the full builder again'
    assert sha(ROOT/m['source_path'])==m['source_sha256'], 'Upstream baseline source differs from the prepared demo'
    assert sha(CSV)==s['per_image_csv_sha256']
    assert sha(ROOT/s['preprocessing_metadata_path'])==s['preprocessing_metadata_sha256']
    assert s['manifest']==identity
    rows=manifest['splits']['train']
    assert f.sample_id.tolist()==[r['name'] for r in rows]
    assert f.split.eq('train').all() and len(f)==s['n_images']==len(rows)
    assert f.sample_id.is_unique and f.image_path.is_unique and f.mask_path.is_unique
    assert set(s['mask_unique_values_raw'])=={0,255}
    assert f.foreground_pixels.ge(0).all() and f.background_pixels.ge(0).all()
    assert f.area_fraction.between(0,1).all()
    assert (f.foreground_pixels+f.background_pixels).eq(f.height*f.width).all()
    np.testing.assert_allclose(f.area_fraction,f.foreground_pixels/(f.height*f.width),rtol=0,atol=1e-16)
    assert f.empty_gt.eq(f.foreground_pixels==0).all()
    assert int(f.empty_gt.sum())==s['n_empty_gt']
    assert int((~f.empty_gt).sum())==s['n_positive_gt']
    assert int(f.foreground_pixels.sum())==s['total_foreground_pixels']
    assert int(f.background_pixels.sum())==s['total_background_pixels']
    assert int((f.height*f.width).sum())==s['total_mask_pixels']==s['total_foreground_pixels']+s['total_background_pixels']
    assert s['foreground_pct_global']==100*s['total_foreground_pixels']/s['total_mask_pixels'] or np.isclose(s['foreground_pct_global'],100*s['total_foreground_pixels']/s['total_mask_pixels'],rtol=0,atol=1e-13)
    assert np.isclose(s['foreground_pct_global']+s['background_pct_global'],100)
    for key,values in [('area_statistics_all',f.area_fraction),('area_statistics_positive',f.loc[~f.empty_gt,'area_fraction'])]:
        for field,value in area_statistics(values).items():
            if value is None:assert s[key][field] is None
            else:np.testing.assert_allclose(value,s[key][field],rtol=0,atol=1e-16)
    assert area_bins(f.area_fraction)==s['area_bins']
    assert choose_examples(f)==s['example_selection']['examples']
    train=f.set_index('sample_id')
    for example in s['example_selection']['examples']:
        row=train.loc[example['sample_id']]
        assert Path(row.image_path).name==Path(row.mask_path).name==row.name
        assert example['area_fraction']==row.area_fraction
    assert m['sample_id'] in train.index
    assert m['no_additional_resize'] and m['geometric']['same_sampled_parameters_verified']
    assert m['intensity']['mask_unchanged_verified'] and not m['validation_test_augmentation']['transform']['transforms']
    for group in m['tensors'].values():
        for field,item in group.items():
            assert sha(ROOT/item['path'])==item['sha256']
            if field in ['image','mask']:
                value=np.load(ROOT/item['path'],allow_pickle=False)
                assert list(value.shape)==item['shape'] and str(value.dtype)==item['dtype']=='float32'
    assert np.array_equal(np.load(ROOT/m['tensors']['original']['mask']['path']),
                          np.load(ROOT/m['tensors']['intensity']['mask']['path']))
    if check_raw:
        for row,expected in zip(f.itertuples(index=False),rows):
            assert row.image_path==expected['image_path'] and row.mask_path==expected['mask_path']
            for field in ['image','mask']:
                path=ROOT/getattr(row,field+'_path')
                assert sha(path)==expected[field+'_sha256']==getattr(row,field+'_sha256')
            with Image.open(ROOT/row.image_path) as image, Image.open(ROOT/row.mask_path) as mask:
                raw=np.asarray(mask)
                assert set(map(int,np.unique(raw))).issubset({0,255})
                if raw.ndim==3:
                    assert np.array_equal(raw[...,0],raw[...,1]) and np.array_equal(raw[...,0],raw[...,2])
                    raw=raw[...,0]
                assert raw.shape==(row.height,row.width) and image.size==(row.width,row.height)
                assert int((raw==255).sum())==row.foreground_pixels
    verification={'membership_matches_saved_run':True,'n_actual_train_verified':len(f),
        'pairs_unique_and_matching':True,'raw_image_mask_sha256_verified':bool(check_raw),
        'raw_pixel_counts_recomputed':bool(check_raw),'binary_values_checked_before_binarization':True,
        'pixel_accounting_valid':True,'area_fractions_in_range':True,'bins_cover_train_once':True,
        'all_and_positive_statistics_recomputed':True,'gallery_train_membership_verified':True,
        'demo_tensors_and_pairing_verified':True,'no_validation_test_pixel_statistics':True,
        'source_wiring':source_wiring_check()}
    return s,f,m,verification


def pixel_composition(s,f,m):
    fig,ax=plt.subplots(figsize=(13.8,5.8))
    fig.subplots_adjust(left=.055,right=.98,bottom=.24,top=.75)
    fg=s['foreground_pct_global'];bg=s['background_pct_global']
    ax.barh(0,fg,height=.46,color=COLORS['copper'])
    ax.barh(0,bg,left=fg,height=.46,color=COLORS['lavender'])
    ax.annotate(f"Luka: {percent(fg)}\n{number(s['total_foreground_pixels'])} piksel",
                xy=(fg/2,.22),xytext=(13,.65),fontsize=18,fontweight='bold',color=COLORS['copper'],
                arrowprops={'arrowstyle':'-','color':COLORS['copper'],'lw':1.6},ha='left')
    ax.text(53,0,f"Background: {percent(bg)}\n{number(s['total_background_pixels'])} piksel",
            ha='center',va='center',fontsize=20,fontweight='bold')
    ax.set(xlim=(0,100),ylim=(-.5,1.03),yticks=[],xlabel='Persentase seluruh piksel mask (%)')
    ax.set_xticks([0,25,50,75,100]);ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    for spine in ['left','bottom']:ax.spines[spine].set_visible(False)
    fig.text(.055,.94,'Background mendominasi target segmentasi',fontsize=25,weight='bold')
    fig.text(.055,.86,f"Chronic Wound · train aktual n={s['n_images']} · background:luka = {s['background_foreground_ratio']:.2f}:1",fontsize=18)
    fig.text(.055,.07,f"Denominator: {number(s['total_mask_pixels'])} piksel · seluruh kanvas 224 × 224, termasuk padding hitam.\nForeground = mask 255; background = 0. Bukan proporsi jaringan tubuh atau luas fisik.",fontsize=14,color=COLORS['gray'])
    return fig


def area_distribution(s,f,m):
    fig,(ax,pos)=plt.subplots(1,2,figsize=(15,6.6),gridspec_kw={'width_ratios':[1.5,1]})
    fig.subplots_adjust(left=.065,right=.97,bottom=.25,top=.79,wspace=.28)
    bins=s['area_bins'];x=np.arange(len(bins));counts=[r['count'] for r in bins]
    ax.bar(x,counts,color=[COLORS['gray']]+[COLORS['copper']]*5,width=.7)
    for i,row in enumerate(bins):
        ax.text(i,row['count']+8,f"{row['count']}\n({row['percent_of_train']:.1f}%)",ha='center',va='bottom',fontsize=13)
    ax.set_xticks(x,[r['label'].replace('–','–\n') for r in bins])
    ax.set_ylim(0,max(counts)*1.3);ax.set_ylabel('Jumlah gambar');ax.set_xlabel('Piksel luka / seluruh kanvas mask (%)')
    ax.set_title(f"Luas mask · seluruh train (n={len(f)})",loc='left',fontsize=18)
    ax.yaxis.grid(True,color=COLORS['lavender'],alpha=.4);ax.set_axisbelow(True)
    positive=f[~f.empty_gt]
    pos.scatter(positive.centroid_x_fraction*100,positive.centroid_y_fraction*100,s=14,
                color=COLORS['copper'],alpha=.38,edgecolors='none')
    pos.set(xlim=(0,100),ylim=(100,0),xlabel='Posisi x / lebar kanvas (%)',ylabel='Posisi y / tinggi kanvas (%)')
    pos.set_xticks([0,25,50,75,100]);pos.set_yticks([0,25,50,75,100]);pos.set_aspect('equal')
    pos.set_title(f"Centroid GT positif (n={len(positive)})",loc='left',fontsize=18)
    pos.grid(color=COLORS['lavender'],alpha=.5);pos.set_axisbelow(True)
    fig.text(.065,.94,'Luas dan posisi mask bervariasi pada kanvas foto',fontsize=24,weight='bold')
    fig.text(.065,.87,f"Train aktual · GT kosong {s['n_empty_gt']}/{s['n_images']} ({percent(s['empty_gt_percent'])}) · median semua gambar {percent(s['area_statistics_all']['median']*100)}",fontsize=17)
    fig.text(.065,.075,'Bin deskriptif, bukan kategori klinis atau C1–C9. Interval positif: batas bawah eksklusif, atas inklusif.\nCentroid = rata-rata pusat piksel foreground; satu titik per GT positif. Padding memengaruhi luas dan posisi.',fontsize=13.5,color=COLORS['gray'])
    return fig


def example_gallery(s,f,m):
    examples=s['example_selection']['examples'];lookup=f.set_index('sample_id')
    fig,axes=plt.subplots(2,len(examples),figsize=(15.5,8.7),squeeze=False)
    fig.subplots_adjust(left=.055,right=.98,bottom=.17,top=.77,hspace=.20,wspace=.10)
    for i,example in enumerate(examples):
        row=lookup.loc[example['sample_id']]
        with Image.open(ROOT/row.image_path) as image:axes[0,i].imshow(image.convert('RGB'))
        with Image.open(ROOT/row.mask_path) as mask:axes[1,i].imshow(mask.convert('L'),cmap='gray',vmin=0,vmax=255)
        short=example['label'].replace('Luas ','').replace(' · ','\n')
        axes[0,i].set_title(f"{short}\nLuas = {percent(row.area_fraction*100)}",fontsize=16,pad=10)
        for axis in axes[:,i]:axis.set_xticks([]);axis.set_yticks([])
        sid=example['sample_id'];wrapped=sid[:18]+'\n'+sid[18:]
        axes[1,i].set_xlabel(wrapped,fontsize=11,labelpad=8)
    axes[0,0].set_ylabel('Foto RGB',fontsize=18);axes[1,0].set_ylabel('GT biner',fontsize=18)
    fig.text(.055,.95,'Foto dan GT yang benar-benar berpasangan',fontsize=25,weight='bold')
    fig.text(.055,.89,f"Chronic Wound · train aktual n={s['n_images']} · semua contoh berasal dari manifest train",fontsize=17)
    fig.text(.055,.05,'Aturan tetap: terdekat Q10/Q50/Q90 luas GT positif; ID berbeda, tie diurutkan menurut filename.\nDitambah GT kosong pertama menurut filename. Foto tampil pada kanvas asli; tidak ada diagnosis klinis yang ditambahkan.',fontsize=13.5,color=COLORS['gray'])
    return fig


def augmentation_gallery(s,f,m):
    fig=plt.figure(figsize=(15.5,10.4))
    fig.text(.045,.965,'Pipeline asli dan demo augmentasi berpasangan',fontsize=25,weight='bold')
    steps=['OpenCV: BGR → RGB\nMask grayscale ÷255','TRAIN: geometri sinkron\nintensitas foto saja','Normalisasi ImageNet\nfoto saja','HWC → CHW\nfloat32 foto + mask']
    for i,label in enumerate(steps):
        x=.035+i*.245
        fig.add_artist(FancyBboxPatch((x,.82),.215,.08,boxstyle='round,pad=.007',
                       transform=fig.transFigure,facecolor=COLORS['ivory'],edgecolor=COLORS['lavender']))
        fig.text(x+.1075,.86,label,fontsize=14,va='center',ha='center')
        if i<3:fig.text(x+.23,.855,'→',fontsize=20,ha='center')
    fig.text(.045,.79,'Validation/test: tanpa augmentasi training. Tidak ada resize/padding tambahan yang aktif.',fontsize=15,color=COLORS['gray'])
    grid=fig.add_gridspec(2,3,left=.11,right=.90,bottom=.205,top=.70,hspace=.08,wspace=.10)
    geo=m['geometric']['replay']['transforms'][0]['params'];inten=m['intensity']['replay']['transforms'][0]['params']
    titles=['Asli',f"Demo geometri · rotasi {geo['angle']:.2f}°",'Demo intensitas · brightness/contrast']
    for i,kind in enumerate(['original','geometric','intensity']):
        for j,(field,label) in enumerate([('display_image','Foto RGB'),('display_mask','GT biner')]):
            ax=fig.add_subplot(grid[j,i]);item=m['tensors'][kind][field]
            with Image.open(ROOT/item['path']) as image:
                ax.imshow(image if j==0 else image.convert('L'),**({} if j==0 else {'cmap':'gray','vmin':0,'vmax':255}))
            ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_title(titles[i],fontsize=15,pad=10)
            if i==0:ax.set_ylabel(label,fontsize=17)
            if j==1:ax.set_xlabel(['Target asli','GT ikut berubah','GT identik dengan asli'][i],fontsize=14,labelpad=9)
    fig.text(.045,.145,f"ID train: {m['sample_id']} · foto ditampilkan sebelum normalisasi ImageNet.",fontsize=13)
    fig.text(.045,.102,f"Geometri: seed {m['geometric']['seed']}; rotate_limit ±30°, shift=0, scale=1. "
             f"Intensitas: seed {m['intensity']['seed']}; α={inten['alpha']:.4f}, β={inten['beta']:.4f} (batas ±0,2).",fontsize=12.5)
    fig.text(.045,.044,'DEMO TRANSFORMASI: cabang asli dipaksa aktif (p=1), bukan probabilitas training.\nAlbumentations 1.3.0 · OpenCV 4.6.0.66 · geometri memakai parameter sama; intensitas mengubah foto saja.',fontsize=13.5,color=COLORS['gray'])
    return fig


def save_figures(s,f,m):
    configure_style();FIGURES.mkdir(parents=True,exist_ok=True)
    PUBLIC_FIGURES.mkdir(parents=True,exist_ok=True)
    artifacts={}
    for name,plot in zip(FIGURE_NAMES,[pixel_composition,area_distribution,example_gallery,augmentation_gallery]):
        fig=plot(s,f,m)
        for suffix in ['png','svg']:
            path=FIGURES/(name+'.'+suffix);fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
            if suffix=='svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
            artifacts[str(path.relative_to(ROOT))]={'sha256':sha(path),'bytes':path.stat().st_size}
            shutil.copyfile(path,PUBLIC_FIGURES/path.name)
        plt.close(fig)
    return artifacts


def export_public_metadata(s):
    """Publish small provenance artifacts, preserving the actual run manifest bytes."""
    shutil.copyfile(RUN/'split-manifest.json',PUBLIC_MANIFEST)
    shutil.copyfile(ROOT/s['preprocessing_metadata_path'],PUBLIC_PREPROCESSING)
    assert sha(PUBLIC_MANIFEST)==s['manifest']['manifest_sha256']
    assert sha(PUBLIC_PREPROCESSING)==s['preprocessing_metadata_sha256']
    s['public_exports']={'figures_directory':str(PUBLIC_FIGURES.relative_to(ROOT)),
        'manifest_path':str(PUBLIC_MANIFEST.relative_to(ROOT)),
        'preprocessing_metadata_path':str(PUBLIC_PREPROCESSING.relative_to(ROOT)),
        'verification_path':str(PUBLIC_VERIFICATION.relative_to(ROOT)),
        'raw_data_checkpoints_and_preprocessing_tensors':'Remain local; not published'}


def md_table(frame,digits=4):
    def render(v):
        if isinstance(v,float):return f'{v:.{digits}f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(render,row))+' |' for row in frame.itertuples(index=False,name=None)])


def write_report(s,f,m,v):
    duplicate_images=sum(len(ids) for ids in s['within_train_rgb_duplicate_groups'])
    versions=s['dependencies'];geo=m['geometric']['replay']['transforms'][0]['params'];inten=m['intensity']['replay']['transforms'][0]['params']
    examples=pd.DataFrame([{'ID':e['sample_id'],'Alasan':e['label'],'Luas (%)':e['area_fraction']*100} for e in s['example_selection']['examples']])
    content=f"""# CP1 · EDA train aktual dan preprocessing baseline

## FAKTA DATASET DARI EDA KAMI

Yang dianalisis adalah **{s['n_images']} foto + {s['n_masks']} GT** Chronic Wound pada train aktual `cp1-baseline-01`, bukan seluruh 831 source train. Membership dibaca langsung dari manifest yang disalin ke run selesai; tidak ada split baru, shuffle ulang, deduplikasi, training, atau inference. Tidak ada statistik piksel validation/test yang dicampurkan.

### Identitas data dan audit

- Manifest aktual lokal: `{s['manifest']['manifest_path']}`. [Salinan publik byte-identik](tables/cp1-eda-split-manifest.json) disertakan agar membership dapat diaudit tanpa membentuk split baru.
- SHA-256 file: `{s['manifest']['manifest_sha256']}`.
- Fingerprint isi manifest: `{s['manifest']['manifest_fingerprint']}`; cocok dengan `environment.json` dan `status.json` run.
- Konfigurasi run: `{s['manifest']['config_path']}`; SHA-256 `{s['manifest']['config_sha256']}`.
- {s['n_unique_filenames']} filename unik. Semua pasangan memakai filename sama, path unik, SHA-256 foto/mask cocok manifest, resolusi **224 × 224** untuk foto dan mask.
- Mask mentah terverifikasi hanya memiliki nilai **{s['mask_unique_values_raw']}**. Mask RGB yang tersimpan memiliki tiga kanal identik. Nilai diperiksa sebelum binarisasi; foreground = grayscale **255**, background = **0**. Mask kosong ditentukan oleh tidak adanya piksel 255.
- Membership train/validation tidak beririsan menurut filename/path. Train/test diperiksa menurut path sumber, karena filename lintas direktori dapat berulang. Kesamaan isi RGB merupakan audit berbeda dan tetap menjadi keterbatasan.

### Angka utama

| Ukuran | Hasil |
| --- | --- |
| Jumlah foto / mask train | {s['n_images']} / {s['n_masks']} |
| GT kosong | {s['n_empty_gt']} ({percent(s['empty_gt_percent'])}) |
| GT positif | {s['n_positive_gt']} ({percent(100-s['empty_gt_percent'])}) |
| Seluruh piksel mask | {number(s['total_mask_pixels'])} |
| Piksel foreground / luka | {number(s['total_foreground_pixels'])} ({percent(s['foreground_pct_global'],4)}) |
| Piksel background / nonluka | {number(s['total_background_pixels'])} ({percent(s['background_pct_global'],4)}) |
| Rasio background:foreground | {s['background_foreground_ratio']:.4f}:1 |

![Komposisi piksel train](../outputs/analysis/cp1-eda-01/figures/01_pixel_composition.png)

### Definisi dan denominator

`foreground_pct_global = Σ foreground_pixels / Σ(H × W) × 100`.
`background_pct_global = Σ background_pixels / Σ(H × W) × 100`.
`area_fraction_i = foreground_pixels_i / (H_i × W_i)`; persentase per gambar = `100 × area_fraction_i`.
Seluruh kanvas **termasuk padding hitam bawaan** dihitung; denominator bukan area tubuh, bukan area jaringan, dan bukan luas fisik. Rasio BG:FG adalah total background dibagi total foreground; bila foreground nol, kode menghasilkan JSON `null`, bukan pembagian nol.

Rata-rata persentase per gambar seluruh train = **{percent(s['mean_per_image_area_percent_all'],4)}**, sama dengan persentase global karena semua resolusi seragam dan cakupan gambarnya sama. Rata-rata khusus GT positif mempunyai denominator gambar berbeda dan dilaporkan terpisah.

### Distribusi luas

Semua angka luas berikut dalam **persen seluruh kanvas mask**, kuantil linear (`numpy.quantile(method='linear')`).

{md_table(stats_table(s))}

{md_table(bin_table(s),2)}

Bin positif memakai interval `(batas bawah, batas atas]`; bin kosong tepat nol. Bin ini deskriptif, **bukan kategori klinis dan bukan otomatis C1–C9 penulis**. Jumlah bin = {sum(r['count'] for r in s['area_bins'])} gambar.

![Distribusi luas dan posisi](../outputs/analysis/cp1-eda-01/figures/02_wound_area_distribution.png)

Posisi dihitung sebagai centroid seluruh foreground, `mean(x+0.5)/W` dan `mean(y+0.5)/H`, pada {s['n_positive_gt']} GT positif. GT kosong tidak mempunyai centroid. Posisi relatif terhadap kanvas dipengaruhi crop/padding; bukan lokasi anatomis. Bila mask memiliki beberapa komponen, satu centroid merangkum semuanya.

### Contoh foto–GT yang dapat diulang

Ambil foto paling dekat dengan kuantil Q10/Q50/Q90 luas **GT positif**, tanpa mengulang ID; tie diurutkan menurut filename. Tambahkan GT kosong pertama secara leksikografis. Tidak ada pemilihan berdasarkan test atau hasil model. Target kuantil, ID, dan alasan lengkap tersimpan di JSON; pasangan benar diverifikasi terhadap manifest.

{md_table(examples)}

![Foto dan GT train](../outputs/analysis/cp1-eda-01/figures/03_train_examples.png)

Galeri memperlihatkan variasi tampilan/crop dan anotasi pada contoh terpilih, bukan sampel acak yang mewakili seluruh populasi. Tidak ada label diagnosis, kelas jaringan, atau penilaian pencahayaan yang ditambahkan; brightness global yang terpengaruh padding tidak dipakai sebagai bukti pencahayaan.

## PENJELASAN METODE DARI PAPER/KODE

Rujukan: [paper FUSegNet v2](https://arxiv.org/abs/2305.02961v2), **§3.2.3** (Dice + Focal Loss), **Table 1 dan §3.2.5** (augmentasi/pengaturan training), serta **§3.2.2** (P-scSE). EDA ini tidak menguji manfaat atau efisiensi P-scSE.

Pada §3.2.3, penulis mengaitkan Focal Loss dengan ketidakseimbangan background/foreground: kontribusi contoh mudah diturunkan agar pembelajaran memberi perhatian lebih pada contoh sulit. Focal Loss digabung dengan Dice Loss. Angka EDA di atas adalah hitungan dataset kami; uraian fungsi loss ini berasal dari paper, bukan hasil eksperimen EDA. Implementasi baseline yang terkunci tidak diubah; [audit kode/loss](../docs/reference-audit.md) tetap berlaku.

### Urutan aktual baseline Chronic Wound

1. `cv2.imread` foto, kemudian konversi **BGR → RGB** (`uint8`, HWC).
2. `cv2.imread(mask, 0)` untuk grayscale, **bagi 255**, lalu tambahkan kanal (`float64`, HWC1 sebelum konversi akhir).
3. Training: panggil augmentasi dengan `image=...` dan `mask=...`; geometri memakai parameter acak yang sama pada keduanya. Transformasi intensitas mengubah foto saja.
4. Foto: normalisasi pretrained ImageNet dari encoder EfficientNet-B7, **`(RGB/255 − mean) / std`**, mean `{m['normalization']['mean']}`, std `{m['normalization']['std']}`. Mask tidak mendapat normalisasi ImageNet.
5. Foto dan mask: **HWC → CHW**, `float32`; foto `[3,224,224]`, mask `[1,224,224]`.

Tidak ada `Resize` atau `PadIfNeeded` aktif tambahan; foto telah 224 × 224 di raw. Geometri tetap bisa mengubah isi kanvas dan mengisi border konstan hitam, dengan output berukuran sama. Validation/test menggunakan `get_validation_augmentation()` yang berisi Compose kosong, lalu normalisasi/tensor yang sama. Call site train/validation/evaluation diverifikasi melalui AST tanpa menjalankan engine.

### Konfigurasi augmentasi training dari sumber

| Tingkat | Konfigurasi aktual |
| --- | --- |
| Compose luar | p=0.9 |
| OneOf flip | p=0.5; HorizontalFlip p=0.8 / VerticalFlip p=0.4 |
| OneOf affine | p=0.9; scale ±0.5 saja / rotate ±30° saja / shift ±0.1 saja / ketiganya; setiap anak p=1, border_mode=0 |
| OneOf efek | p=0.2; Perspective / GaussNoise / Sharpen / Blur(blur_limit=3) / MotionBlur(blur_limit=3), setiap anak p=1 |
| OneOf intensitas | p=0.2; CLAHE / RandomBrightnessContrast(±0.2, ±0.2) / RandomGamma / HueSaturationValue, setiap anak p=1 |

`p` anak di dalam `OneOf` adalah **bobot pemilihan**, bukan probabilitas independen. Aktivasi bergantung Compose luar dan grup. Konfigurasi lengkap hasil serialisasi library, default parameter, jenis target, replay, dan hash source ada di [metadata preprocessing publik](tables/cp1-eda-preprocessing.json). Path raw/tensor di dalam metadata merujuk artefak lokal; tensor tidak dipublikasikan.

### Demo transformasi (bukan simulasi probabilitas training)

Sumber aman: AST loader proyek mengekstrak definisi Dataset/augmentasi/preprocessing dari `{m['source_path']}` (SHA-256 `{m['source_sha256']}`). Seluruh entrypoint training tidak diimpor; tidak ada model dibuat atau weights dimuat.

Demo memakai ID **`{m['sample_id']}`**, contoh Q90 positif train untuk keterbacaan:

- Geometri: cabang **rotate-only baseline** `ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=1, border_mode=0)`, seed **{m['geometric']['seed']}**; terambil angle **{geo['angle']:.6f}°**, scale **{geo['scale']}**, dx/dy **{geo['dx']}/{geo['dy']}**. Foto interpolasi linear (OpenCV 1), mask nearest-neighbor (implementasi `apply_to_mask`); parameter geometrik sama diverifikasi terhadap keluaran aktual.
- Intensitas: cabang **RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)**, seed **{m['intensity']['seed']}**; terambil alpha **{inten['alpha']:.6f}**, beta **{inten['beta']:.6f}**, `brightness_by_max=True`. Foto berubah, mask **bitwise identik** dengan asli.
- Cabang sumber diisolasi dalam `ReplayCompose(p=1)` agar aktif: **DEMO TRANSFORMASI**, bukan probabilitas training. Seed tetap; tidak dipilih melalui hasil model/test. Demos menampilkan foto **sebelum** normalisasi ImageNet; tensor sesudah normalisasi disimpan dan diverifikasi terpisah.
- Cabang rotate x-FUSegNet mempunyai `shift_limit=0.1`; baseline Chronic Wound **0**. Demo mengikuti baseline.

![Preprocessing dan demo](../outputs/analysis/cp1-eda-01/figures/04_preprocessing_augmentation.png)

## INTERPRETASI ATAU HIPOTESIS KAMI

- Dominasi piksel background ({percent(s['background_pct_global'])}) memberi **konteks empiris** untuk alasan Focal Loss di §3.2.3. EDA tidak menunjukkan loss tersebut lebih baik dari alternatif.
- Variasi luas/centroid mask pada kanvas dan tampilan foto pada galeri memberi konteks augmentasi geometrik/intensitas pada Table 1. Hipotesis bahwa augmentasi meningkatkan generalisasi perlu eksperimen terkontrol dengan protokol validasi yang tepat.
- Mask 0/255 menetapkan target **segmentasi biner luka vs background**, bukan diagnosis atau jenis jaringan. EDA tidak memvalidasi kualitas anotasi secara klinis.
- Manfaat Focal Loss, augmentasi, atau P-scSE tidak dapat disimpulkan dari grafik ini. Dibutuhkan perbandingan terkontrol; EDA ini tidak membuat hasil model baru.

## Keterbatasan

1. **Padding/crop** masuk denominator dan koordinat. Proporsi kanvas bukan luas tubuh/fisik; perubahan brightness juga bisa mengubah padding. Tidak ada kuantifikasi pencahayaan fisiologis/klinis.
2. **Duplikat dipertahankan**: ditemukan {len(s['within_train_rgb_duplicate_groups'])} grup RGB identik di dalam train, melibatkan {duplicate_images} gambar. Audit overlap sebelumnya melaporkan 2 gambar validation dan 16 test dengan RGB identik di actual train; angka itu hanya dirujuk dari [audit terdahulu](cp1-analysis-01.md), bukan dihitung sebagai EDA validation/test di sini. Tidak ada split diubah atau data dihapus.
3. **Independensi pasien belum terverifikasi**: identifier pasien/luka yang memadai tidak tersedia; audit RGB tidak mendeteksi seluruh kemiripan atau foto lain dari pasien yang sama. Nama file tidak dianggap ID pasien.
4. **Batas reproduksi**: yang pasti adalah 707 anggota baseline lokal tersimpan. Split train/validation asli penulis tidak tersedia; jumlah source publik 831/278 berbeda dari 810/200 yang disebut paper. Tidak diklaim identik dengan membership asli paper.
5. **Generalisasi/efikasi belum diuji oleh EDA ini**. Nilai area tidak punya kalibrasi fisik, kategori klinis, atau bukti metode lebih unggul. Semua data wajib dan pipeline demo tersedia; hal klinis/independensi pasien di atas tetap tidak dapat diverifikasi dari artefak ini.

## Rekomendasi dua slide untuk presentasi 10 menit

**Satu slide EDA (~50 detik).** Judul: “Background mendominasi, ukuran target bervariasi”. Gunakan grafik komposisi dan panel bin luas dari grafik distribusi; centroid/galeri detail dapat menjadi cadangan. Tampilkan hanya tiga angka kunci: luka **{percent(s['foreground_pct_global'])}**, background:luka **{s['background_foreground_ratio']:.1f}:1**, GT kosong **{s['n_empty_gt']}/{s['n_images']}**. Tulis footer “707 train aktual; seluruh kanvas 224×224 termasuk padding”. Narasi: “Ini konteks pemilihan Focal Loss dan augmentasi penulis, bukan bukti keunggulan metode.”

**Satu slide preprocessing (~50 detik).** Gunakan galeri preprocessing/augmentasi: asli → demo geometri (GT ikut berubah) → demo intensitas (GT tetap), plus alur BGR→RGB, mask÷255, train augmentation, ImageNet, CHW float32. Sebutkan “demo dipaksa aktif; validation/test tanpa augmentasi training; tidak ada resize tambahan”. Jangan tampilkan tensor ternormalisasi sebagai foto RGB biasa.

## Reproduksibilitas dan pemeriksaan

Jalankan dari root proyek:
```bash
.venv-analysis/bin/python scripts/build_cp1_eda.py
```
Builder memakai `.venv-training` yang sudah dipin untuk pembacaan sumber/augmentasi, `.venv-analysis` yang sudah dipin untuk plot dan eksekusi notebook. Tidak ada dependency diinstal atau diganti.

Versi data/augmentasi: {', '.join(f'{k}={val}' for k,val in versions['data_and_augmentation'].items() if k!='interpreter')}.

Versi plot/notebook: {', '.join(f'{k}={val}' for k,val in versions['plot_and_notebook'].items())}.

Audit otomatis: membership saved run, hash seluruh {s['n_images']} foto dan mask, mask 0/255, jumlah piksel, rentang area, kuantil semua/positif, jumlah bin, ID galeri, mask demo, tensor/dtype, dan pemanggilan augmentasi train vs validation/evaluation cocok. Artefak baseline dan file tracked selain output EDA dilindungi dengan snapshot hash/stat per build, termasuk notebook 01 yang sebelumnya sudah termodifikasi. [Hasil akhir pemeriksaan](tables/cp1-eda-verification.json) disertakan untuk audit.

Output:
- [Notebook yang sudah dieksekusi](../notebooks/04_cp1_eda.ipynb).
- [CSV per gambar](tables/cp1-eda-per-image.csv), [JSON ringkasan](tables/cp1-eda-summary.json).
- [Figure PNG dan SVG](figures/cp1-eda-01/), 300 dpi untuk PNG, latar putih; salinan lokal tetap di `outputs/analysis/cp1-eda-01/figures/`.
- [Manifest yang digunakan](tables/cp1-eda-split-manifest.json), [metadata dan replay](tables/cp1-eda-preprocessing.json), serta [verifikasi](tables/cp1-eda-verification.json).
- Foto/mask demo terpilih tertanam pada figure; input terpisah dan tensor preprocessing tetap lokal di `outputs/analysis/cp1-eda-01/preprocessing/`.
- [Builder](../scripts/build_cp1_eda.py), [worker aman](../scripts/cp1_eda.py).

Notebook, tabel, manifest, metadata, dan figure merupakan artefak publikasi Git. Raw dataset lengkap, checkpoint, tensor, dan dokumen pribadi tetap lokal. Foto/GT contoh berasal dari [dataset Chronic Wound publik UWM](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/wound_dataset); tidak ada hasil model baru dibuat untuk EDA ini.
"""
    content=content.replace('../outputs/analysis/cp1-eda-01/figures/','figures/cp1-eda-01/')
    REPORT.write_text(content)


def build_notebook():
    cells=[]
    def md(value):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(value).strip()))
    def code(value):cells.append(nbf.v4.new_code_cell(textwrap.dedent(value).strip()))
    md('''
    # CP1 · Data yang dipelajari baseline
    **EDA terarah untuk presentasi 10 menit · Chronic Wound · train aktual**

    **Alur:** fakta data → hubungan dengan metode penulis → demo preprocessing → batas interpretasi.
    Notebook ini sudah dieksekusi: tabel dan empat visual dapat dibaca langsung. Tidak menjalankan model,
    training, inference, atau pembentukan split. Semua hitungan memakai train dari manifest baseline tersimpan.
    ''')
    code('''
    from pathlib import Path
    import sys
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'scripts/build_cp1_eda.py').is_file())
    sys.path.insert(0, str(ROOT/'scripts'))
    from build_cp1_eda import load_verified, stats_table, bin_table, configure_style
    from cp1_eda import sha, CSV, SUMMARY
    import pandas as pd
    from IPython.display import display, HTML, Markdown, Image
    s, frame, pipeline, verification = load_verified(check_raw=True)
    display(HTML("""<style>
    .dataframe {border-collapse:collapse;font-size:14px} .dataframe th {background:#352B43;color:white;padding:9px}
    .dataframe td {padding:8px;border-bottom:1px solid #DDD5E8} .dataframe tr:nth-child(even){background:#FAF7F2}
    </style>"""))
    def table(df): display(HTML(df.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:.4f}')))
    def figure(name): display(Image(filename=str(ROOT/'reports/figures/cp1-eda-01'/name),width=1100))
    print(f"✓ {len(frame)} train aktual; pasangan, hash raw, hitungan piksel, statistik, dan demo terverifikasi.")
    print('Manifest:',s['manifest']['manifest_path'])
    print('SHA-256:',s['manifest']['manifest_sha256'])
    print('Membership dibaca dari manifest run, tanpa shuffle atau split baru.')
    ''')
    md('''
    ## FAKTA DATASET DARI EDA KAMI
    ### 1 · Target biner dan inventaris train
    Mask mentah diperiksa **sebelum** binarisasi; foreground = grayscale 255, background = 0.
    Kanal RGB mask tersimpan identik. Ini target segmentasi biner, bukan kategori klinis.
    ''')
    code('''
    table(pd.DataFrame([
        ('Foto train',s['n_images']),('Mask train',s['n_masks']),('Filename unik',s['n_unique_filenames']),
        ('Resolusi foto dan mask',str(s['resolutions'])),('Nilai unik mask mentah',str(s['mask_unique_values_raw'])),
        ('GT kosong',f"{s['n_empty_gt']} ({s['empty_gt_percent']:.2f}%)"),('GT positif',s['n_positive_gt'])
    ],columns=['Audit','Hasil']))
    ''')
    md('''
    ### 2 · Ketidakseimbangan piksel
    **Denominator:** seluruh piksel single-channel mask train, termasuk padding hitam bawaan.
    `foreground_pct_global = Σ foreground / Σ(H×W) × 100`. Ini **bukan** proporsi jaringan tubuh atau luas fisik.
    ''')
    code('''
    table(pd.DataFrame([
        ('Luka',s['total_foreground_pixels'],s['foreground_pct_global']),
        ('Background',s['total_background_pixels'],s['background_pct_global']),
        ('Seluruh mask',s['total_mask_pixels'],100.)
    ],columns=['Piksel','Jumlah','Persen kanvas']))
    display(Markdown(f"**Background:luka = {s['background_foreground_ratio']:.4f}:1.** "
                     "Jika total foreground nol, rasio disimpan sebagai null (tidak terdefinisi)."))
    figure('01_pixel_composition.png')
    ''')
    md('''
    ### 3 · Luas dan posisi: cakupan statistik dibedakan
    `area_fraction_i = foreground_pixels_i / (H_i × W_i)`. Tabel dalam **persen**; kuantil interpolasi linear.
    Semua gambar dan GT positif memiliki denominator gambar berbeda. Karena resolusi seluruh train seragam,
    persentase global sama dengan rata-rata persentase per gambar **untuk cakupan seluruh train yang sama**.
    Bin bersifat deskriptif, bukan klinis atau kategori C1–C9. Centroid adalah rata-rata pusat semua piksel
    foreground relatif terhadap kanvas; tidak terdefinisi pada GT kosong, bukan posisi anatomis.
    ''')
    code('''
    table(stats_table(s))
    table(bin_table(s))
    print(f"Mean seluruh train = {s['mean_per_image_area_percent_all']:.6f}%; global = {s['foreground_pct_global']:.6f}%")
    print('Jumlah bin =',sum(r['count'] for r in s['area_bins']))
    figure('02_wound_area_distribution.png')
    ''')
    md('''
    ### 4 · Foto–GT dipilih dengan aturan tetap
    Terdekat Q10/Q50/Q90 luas mask positif, ID berbeda, tie menurut filename; ditambah GT kosong pertama
    secara leksikografis. Tidak memilih dari validation/test atau berdasarkan hasil model. Galeri adalah contoh
    terpilih, bukan sampel acak yang merepresentasikan populasi. Tidak ada diagnosis atau klaim pencahayaan
    dari brightness global yang dipengaruhi padding.
    ''')
    code('''
    choices=pd.DataFrame(s['example_selection']['examples'])
    choices['luas_aktual_pct']=choices.area_fraction*100
    table(choices[['sample_id','label','target_area_fraction','luas_aktual_pct','reason']])
    figure('03_train_examples.png')
    ''')
    md('''
    ## PENJELASAN METODE DARI PAPER/KODE
    **Paper §3.2.3:** Dice + Focal Loss; focal memberi perhatian pada contoh sulit ketika background dominan.
    **Table 1 / §3.2.5:** augmentasi geometrik dan intensitas pada training; normalisasi/preprocessing.
    **§3.2.2:** arsitektur P-scSE. EDA ini tidak menguji keunggulan arsitektur tersebut.

    Alur kode: OpenCV BGR→RGB → mask grayscale÷255 + kanal → augmentasi training berpasangan →
    normalisasi ImageNet foto saja → HWC→CHW, float32. Input sudah 224×224; tidak ada resize atau
    PadIfNeeded aktif tambahan. Validation/test memakai Compose kosong, dilanjutkan preprocessing yang sama.

    Compose training p=0.9; grup OneOf flip p=0.5, affine p=0.9, efek p=0.2, intensitas p=0.2.
    `p` anak OneOf adalah bobot pemilihan. Konfigurasi lengkap/default library dan replay disimpan dalam metadata.
    ''')
    code('''
    display(Markdown(f"**Normalisasi:** mean `{pipeline['normalization']['mean']}`, std `{pipeline['normalization']['std']}`."))
    table(pd.DataFrame(pipeline['transform_targets']))
    tensor_rows=[]
    for kind,items in pipeline['tensors'].items():
        for field in ['image','mask']:
            item=items[field]
            tensor_rows.append({'contoh':kind,'target':field,**{k:item[k] for k in ['shape','dtype','min','max']}})
    table(pd.DataFrame(tensor_rows))
    table(pd.DataFrame(verification['source_wiring']['engine_calls']))
    ''')
    md('''
    ### Demo transformasi: geometri mengubah foto + GT, intensitas mengubah foto saja
    Cabang sumber asli diisolasi dan **dipaksa aktif** dalam ReplayCompose(p=1). Probabilitas ini **bukan**
    probabilitas training. Foto ditampilkan **sebelum normalisasi ImageNet**; tensor setelah normalisasi
    tersimpan terpisah. Tidak ada model dibangun atau weights dimuat. Definisi diekstrak melalui AST loader aman.
    Cabang rotate baseline mempunyai shift_limit=0; x-FUSegNet mempunyai 0.1. Demo utama mengikuti baseline.
    ''')
    code('''
    rows=[]
    for kind in ['geometric','intensity']:
        demo=pipeline[kind]
        rows.append({'demo':kind,'seed':demo['seed'],'konfigurasi':str(demo['transform']['transform']),
                     'parameter_aktual':str(demo['replay']['transforms'][0]['params'])})
    table(pd.DataFrame(rows))
    figure('04_preprocessing_augmentation.png')
    ''')
    md('''
    ## INTERPRETASI ATAU HIPOTESIS KAMI
    Dominasi background memberi konteks alasan Focal Loss; variasi luas/posisi dan tampilan contoh foto
    memberi konteks augmentasi penulis. **EDA tidak membuktikan Focal Loss, augmentasi, atau P-scSE lebih baik**;
    klaim itu membutuhkan eksperimen terkontrol. Mask biner menjelaskan target yang dipelajari model.

    **Batas:** seluruh kanvas termasuk padding; tidak ada ukuran fisik/diagnosis; duplikat dipertahankan;
    identitas pasien/luka yang memadai tidak tersedia sehingga independensi pasien belum terverifikasi.
    Audit terdahulu menemukan 2 validation dan 16 test dengan RGB identik di actual train
    ([laporan sebelumnya](../reports/cp1-analysis-01.md)); itu bukan statistik EDA train baru.
    Membership asli train/validation penulis tidak tersedia; ini manifest baseline lokal yang pasti dipakai.
    ''')
    code('''
    groups=s['within_train_rgb_duplicate_groups']
    print(f"Di dalam actual train: {len(groups)} grup RGB identik, melibatkan {sum(map(len,groups))} gambar. Tidak dihapus.")
    table(pd.DataFrame([{'grup':i+1,'sample_ids':', '.join(ids)} for i,ids in enumerate(groups)]))
    display(Markdown('**Versi environment yang dipakai:**'))
    table(pd.DataFrame([{'fungsi':group,'dependency':k,'versi':v} for group,items in s['dependencies'].items()
                        for k,v in items.items()]))
    print('Pemeriksaan inti:',verification)
    print('CSV SHA-256:',sha(CSV))
    ''')
    md('''
    ## Dua slide yang disarankan
    **Satu slide EDA (~50 detik):** komposisi piksel + panel bin luas. Sebut luka/background,
    rasio BG:FG, GT kosong; footer “707 train aktual, seluruh kanvas termasuk padding”.
    Tutup dengan “konteks metode penulis, belum bukti manfaat”. Centroid/galeri sebagai cadangan.

    **Satu slide preprocessing (~50 detik):** diagram alur + galeri asli/geometri/intensitas,
    dengan GT yang berubah/tetap. Sebut demo dipaksa aktif, val/test tanpa augmentasi training,
    tidak ada resize tambahan, dan normalisasi ImageNet untuk foto saja.

    Sumber: [paper](https://arxiv.org/abs/2305.02961v2), [laporan lengkap](../reports/cp1-eda-01.md),
    [CSV](../reports/tables/cp1-eda-per-image.csv), [JSON](../reports/tables/cp1-eda-summary.json),
    [manifest aktual](../reports/tables/cp1-eda-split-manifest.json),
    [metadata/replay](../reports/tables/cp1-eda-preprocessing.json), dan
    [figure PNG/SVG](../reports/figures/cp1-eda-01/).
    Foto/GT terpilih berasal dari [Chronic Wound publik UWM](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/wound_dataset).
    Raw dataset, checkpoint, dan tensor preprocessing tetap lokal. Untuk mengeksekusi ulang verifikasi,
    diperlukan raw data, snapshot source, dan artefak run baseline lokal; tidak perlu training/inference.
    ''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'fuseg-cp1-eda','display_name':'FUSeg · CP1 EDA (analysis)','language':'python'}})
    nbf.write(nb,NOTEBOOK)
    return nb


def execute_notebook(nb):
    prefix=OUTPUT/'kernel'
    subprocess.run([str(ROOT/'.venv-analysis/bin/python'),'-m','ipykernel','install','--prefix',str(prefix),
                    '--name','fuseg-cp1-eda','--display-name','FUSeg · CP1 EDA (analysis)'],check=True,capture_output=True,text=True)
    os.environ['JUPYTER_PATH']=str(prefix/'share/jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    executed=NotebookClient(nb,timeout=300,kernel_name='fuseg-cp1-eda',allow_errors=False,
                            resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(executed,NOTEBOOK)
    return executed


def verify_published_outputs(s,notebook):
    assert read(SUMMARY)==s
    report=REPORT.read_text()
    assert s['manifest']['manifest_sha256'] in report
    assert number(s['total_foreground_pixels']) in report and number(s['total_background_pixels']) in report
    assert 'FAKTA DATASET DARI EDA KAMI' in report and 'PENJELASAN METODE DARI PAPER/KODE' in report and 'INTERPRETASI ATAU HIPOTESIS KAMI' in report
    code_cells=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code_cells)
    assert all(o.output_type!='error' for c in code_cells for o in c.outputs)
    embedded=sum('image/png' in o.get('data',{}) for c in code_cells for o in c.outputs)
    assert embedded==len(FIGURE_NAMES)
    for name in FIGURE_NAMES:
        assert (FIGURES/(name+'.png')).is_file() and (FIGURES/(name+'.svg')).is_file()
    return {'all_notebook_code_cells_executed':True,'notebook_code_cells':len(code_cells),
            'notebook_embedded_png_figures':embedded,'report_numbers_match_json':True,
            'csv_json_raw_consistency':True,'figure_formats':['PNG 300 dpi','SVG'],
            'report_sha256':sha(REPORT),'notebook_sha256':sha(NOTEBOOK),'summary_sha256':sha(SUMMARY)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-prepared',action='store_true',help='Reuse and fully verify this EDA worker output; never reuse a split reconstruction')
    args=parser.parse_args()
    # Protect inputs for this invocation; generated EDA artifacts may be rebuilt.
    capture_protected_inputs()
    verify_protected_inputs()
    if not args.reuse_prepared:
        env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':''}
        subprocess.run([str(ROOT/'.venv-training/bin/python'),str(ROOT/'scripts/cp1_eda.py'),'--prepare'],cwd=ROOT,env=env,check=True)
    s,f,m,v=load_verified(check_raw=True)
    s['dependencies']['plot_and_notebook']={'python':sys.version.split()[0],**version_record(['numpy','pandas','Pillow','matplotlib','nbformat','nbclient','ipykernel'])}
    s['builder_source_sha256']=sha(Path(__file__))
    export_public_metadata(s)
    write(SUMMARY,s)
    figures=save_figures(s,f,m)
    write_report(s,f,m,v)
    notebook=execute_notebook(build_notebook())
    _,_,_,final=load_verified(check_raw=True)
    final['protected_inputs']=verify_protected_inputs()
    final['published_outputs']=verify_published_outputs(s,notebook)
    final['figure_artifacts']=figures
    final['no_training_no_inference_no_git_write']=True
    write(OUTPUT/'verification.json',final)
    shutil.copyfile(OUTPUT/'verification.json',PUBLIC_VERIFICATION)
    if (OUTPUT/'build-failure.json').is_file():
        previous=read(OUTPUT/'build-failure.json')
        previous.update({'resolved':True,'resolution':'Notebook tables use DataFrame.to_html; no Jinja2 installation or environment change required'})
        write(OUTPUT/'build-failure-resolved.json',previous)
        (OUTPUT/'build-failure.json').unlink()
    print(f"Completed: {len(f)} actual train; {len(figures)} exports; executed {NOTEBOOK.relative_to(ROOT)}",flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUTPUT/'build-failure.json',{'error':f'{type(error).__name__}: {error}','numbers_substituted':False})
        raise
