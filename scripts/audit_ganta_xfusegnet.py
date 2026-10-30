#!/usr/bin/env python3
"""Read-only audit of Ganta's script, saved results, and dataset identity.

Does not import the partner entrypoint, load checkpoints/pickle, train, infer,
reconstruct fold membership, change raw data, or write in the source project.
Requires the existing analysis environment: numpy, pandas and Pillow.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from PIL import Image

SCRIPT = Path('notebook/x-FUSegNet Reproduce.py')
RESULTS = Path('output_fuseg_2021/results')
EXTENSIONS = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def files_to_publish(root):
    return [root/SCRIPT]+[p for p in sorted((root/RESULTS).rglob('*')) if p.is_file()
        and '.ipynb_checkpoints' not in p.parts and p.suffix.lower() in {'.json','.csv','.png','.svg'}]


def file_stats(path):
    s=path.stat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}


def list_images(directory):
    return {p.name:p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}


def normalized_ast(node):
    class StripDocstrings(ast.NodeTransformer):
        def visit_Expr(self,item):
            if isinstance(item.value,ast.Constant) and isinstance(item.value.value,str):return None
            return self.generic_visit(item)
    return ast.dump(StripDocstrings().visit(ast.parse(ast.unparse(node))),include_attributes=False)


def source_comparison(ganta,orie,config):
    partner=ast.parse((ganta/SCRIPT).read_text())
    upstream=orie/'source/upstream/FUSegNet'/config['reference_revision']
    definitions={n.name:n for n in partner.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    lookup={}
    for relative in ['xfusegnet_train.py','segmentation_models_pytorch/decoders/unet/decoder.py',
                     'segmentation_models_pytorch/base/modules.py']:
        for node in ast.parse((upstream/relative).read_text()).body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)):lookup[node.name]=(relative,node)
    comparisons=[]
    for name in ['Dataset','get_training_augmentation','get_validation_augmentation','to_tensor','get_preprocessing',
                 'Conv2dReLU','SCSEModule','SCSEModule0','Attention','DecoderBlock','CenterBlock','UnetDecoder']:
        relative,reference=lookup[name];node=definitions[name]
        comparisons.append({'definition':name,'partner_line':node.lineno,'upstream_file':relative,
            'upstream_line':reference.lineno,'equal_ast_ignoring_docstrings':normalized_ast(node)==normalized_ast(reference)})
    seeds=[{'line':n.lineno,'call':ast.unparse(n)} for n in ast.walk(partner) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr in {'seed','manual_seed','manual_seed_all'}]
    return {'script_sha256':sha(ganta/SCRIPT),'syntax_valid':True,'entrypoint_imported':False,
            'upstream_revision':config['reference_revision'],'definitions':comparisons,'explicit_seed_calls':seeds,
            'source_claim_limit':'Static source comparison; no proof of identical runtime numerics across dependency versions'}


def notebook_evidence(ganta):
    evidence=[]
    for path in sorted((ganta/'notebook').glob('*.ipynb')):
        nb=read(path);cells=[c for c in nb['cells'] if c['cell_type']=='code'];errors=[]
        for i,c in enumerate(nb['cells']):
            for out in c.get('outputs',[]):
                if out.get('output_type')=='error':errors.append({'cell':i,'name':out.get('ename'),'value':out.get('evalue')})
        evidence.append({'path':str(path.relative_to(ganta)),'sha256':sha(path),'code_cells':len(cells),
            'executed_code_cells':sum(c.get('execution_count') is not None for c in cells),'stored_errors':errors,
            'executed_or_modified_by_this_audit':False,'published':False})
    return evidence


def audit_data(ganta,orie,manifest):
    payload={k:v for k,v in manifest.items() if k!='fingerprint'}
    actual=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if actual!=manifest['fingerprint']:raise ValueError('Orie saved manifest fingerprint is invalid')
    labeled={r['name']:r for r in manifest['labeled']};test={r['name']:r for r in manifest['test']}
    result={};source_mask_values=set();actual_shapes=set()
    raw_root=ganta/'data/raw/fuseg_2021'
    for split in ['train','validation','test']:
        images=list_images(raw_root/split/'images')
        expected={name:r for name,r in (test if split=='test' else labeled).items() if r['provided_split']==split}
        if set(images)!=set(expected):raise ValueError(f'Dataset membership differs in provided {split}')
        labels=list_images(raw_root/split/'labels') if (raw_root/split/'labels').is_dir() else {}
        if split!='test' and set(labels)!=set(images):raise ValueError(f'Unpaired masks: {split}')
        for name,path in images.items():
            row=expected[name]
            if sha(path)!=row['image_sha256'] or sha(orie/row['image_path'])!=row['image_sha256']:
                raise ValueError(f'Image bytes differ from Orie manifest: {split}/{name}')
            with Image.open(path) as image:actual_shapes.add(image.size)
            if split!='test':
                if sha(labels[name])!=row['mask_sha256'] or sha(orie/row['mask_path'])!=row['mask_sha256']:
                    raise ValueError(f'Mask bytes differ from Orie manifest: {split}/{name}')
                with Image.open(labels[name]) as opened:
                    mask=np.asarray(opened.convert('L'));source_mask_values.update(map(int,np.unique(mask)))
                    if mask.shape!=(512,512):raise ValueError('Unexpected native mask size')
        result[split]={'n_images':len(images),'n_masks':len(labels),'membership_identical_to_Orie_provided_split':True,
                       'image_and_available_mask_sha256_match_Orie_manifest':True}
    if actual_shapes!={(512,512)}:raise ValueError(f'Unexpected raw image resolutions: {actual_shapes}')
    train_ids=set(list_images(raw_root/'train/images'));heldout_ids=set(list_images(raw_root/'validation/images'))
    test_ids=set(list_images(raw_root/'test/images'))
    train_rgb={labeled[n]['rgb_sha256'] for n in train_ids}
    identical=[name for name in sorted(heldout_ids) if labeled[name]['rgb_sha256'] in train_rgb]
    result.update({'native_resolution':[512,512],'labeled_mask_unique_values':sorted(source_mask_values),
        'partner_evaluation_source_split':'provided validation; not official challenge test',
        'partner_training_pool_images':len(train_ids),'partner_evaluation_images':len(heldout_ids),
        'partner_evaluation_filename_overlap_with_partner_training':len(train_ids&heldout_ids),
        'partner_evaluation_identical_rgb_to_partner_training':{'count':len(identical),'sample_ids':identical},
        'partner_evaluation_is_in_Orie_labeled_pool':heldout_ids.issubset(labeled),
        'partner_evaluation_overlap_with_Orie_official_test':len(heldout_ids&test_ids),
        'Orie_fold_training_membership_for_partner_evaluation':[
            {'fold':r['fold'],'in_training':len(heldout_ids&set(r['train'])),
             'in_validation':len(heldout_ids&set(r['validation']))} for r in manifest['folds']],
        'partner_run_linked_fold_manifest_available':False,
        'partner_actual_fold_membership_reconstructed':False,
        'partner_configured_fold_sizes':{'train':len(train_ids)*4//5,'validation':len(train_ids)//5},
        'fold_sizes_basis':'Derived from supplied script KFold(5,shuffle=False) and 810 pool; saved fold IDs unavailable'})
    return result


def scores(tp,fp,fn,eps=1e-7):
    return {'dice':(2*tp+eps)/(2*tp+fp+fn+eps),'iou':(tp+eps)/(tp+fp+fn+eps),
            'precision':(tp+eps)/(tp+fp+eps),'recall':(tp+eps)/(tp+fn+eps)}


def verify_cached_metrics(ganta,run_info,reported,output):
    gt_dir=ganta/'data/raw/fuseg_2021/validation/labels';names=sorted(list_images(gt_dir))
    all_rows=[];verified={}
    for mode,label in [('avg','x-FUSegNet (Average)'),('voting','x-FUSegNet (Voting)')]:
        folder=ganta/'output_fuseg_2021/predictions'/(run_info['model_name']+'_'+mode)
        cached=list_images(folder)
        if set(cached)!=set(names):raise ValueError('Cached prediction membership differs from held-out set')
        rows=[]
        for name in names:
            gt_path=gt_dir/name;pr_path=cached[name]
            with Image.open(gt_path) as image:g=np.asarray(image.convert('L'))/255>=.5
            with Image.open(pr_path) as image:raw=np.asarray(image.convert('L'));p=raw>127
            if g.shape!=p.shape or not set(map(int,np.unique(raw))).issubset({0,255}):
                raise ValueError('Cached prediction shape or binary encoding differs')
            tp=int(np.logical_and(p,g).sum());fp=int(np.logical_and(p,~g).sum());fn=int(np.logical_and(~p,g).sum())
            rows.append({'sample_id':name,'evaluation_source':'provided_validation','ensemble':mode,
                'tp':tp,'fp':fp,'fn':fn,'empty_gt':not bool(g.any()),'empty_prediction':not bool(p.any()),
                'image_dice':scores(tp,fp,fn)['dice'],'image_iou':scores(tp,fp,fn)['iou'],
                'gt_sha256':sha(gt_path),'prediction_sha256':sha(pr_path)})
        sums={k:sum(r[k] for r in rows) for k in ['tp','fp','fn']};global_scores=scores(**sums)
        calculated={'Data Dice (%)':global_scores['dice']*100,'Data IoU (%)':global_scores['iou']*100,
                    'Data Prec (%)':global_scores['precision']*100,'Data Rec (%)':global_scores['recall']*100,
                    'Image Dice (%)':float(np.mean([r['image_dice'] for r in rows]))*100,
                    'Image IoU (%)':float(np.mean([r['image_iou'] for r in rows]))*100}
        row=reported.loc[reported.Model==label]
        if len(row)!=1:raise ValueError('Missing or ambiguous ensemble row in source metrics')
        errors={k:float(calculated[k]-row.iloc[0][k]) for k in calculated}
        if any(abs(v)>1e-9 for v in errors.values()):raise ValueError(f'CSV differs from cached PNG metrics: {errors}')
        verified[mode]={'n_images':len(rows),'counts':sums,'recomputed_metrics':calculated,
            'maximum_absolute_csv_error_percentage_points':max(map(abs,errors.values())),
            'n_empty_gt':sum(r['empty_gt'] for r in rows),'n_empty_prediction':sum(r['empty_prediction'] for r in rows),
            'gt_and_prediction_sha256_in_per_image_csv':True}
        all_rows.extend(rows)
    pd.DataFrame(all_rows).to_csv(output/'ganta-xfusegnet-audit-per-image.csv',index=False,float_format='%.17g')
    return {'policy':'GT grayscale/255 >=0.5; cached prediction >127; source eps=1e-7 added numerator and denominator; mean per-image includes empty GT',
            'ensembles':verified,'single_model_rows_recomputed':False,
            'limitations':'No raw five-model probabilities were loaded; mean/voting computation and model-to-PNG lineage not independently verified; no model inference',
            'per_image_csv':'ganta-xfusegnet-audit-per-image.csv'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ganta-root',required=True,type=Path);p.add_argument('--orie-root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    ganta,orie,output=a.ganta_root.resolve(),a.orie_root.resolve(),a.output_dir.resolve()
    if output.is_relative_to(ganta):raise ValueError('Audit output must be outside the read-only Ganta source directory')
    output.mkdir(parents=True,exist_ok=True)
    files=files_to_publish(ganta);before={str(p.relative_to(ganta)):{'sha256':sha(p),**file_stats(p)} for p in files}
    run_files=list((ganta/RESULTS).glob('*run_info.json'))
    if len(run_files)!=1:raise ValueError('Select a run explicitly when more than one results metadata file exists')
    info=read(run_files[0]);run_name=info['model_name']
    metric_path=ganta/RESULTS/(run_name+'_test_metrics.csv');reported=pd.read_csv(metric_path)
    folds=pd.read_csv(ganta/RESULTS/(run_name+'_kfold_summary.csv'))
    if sorted(folds['fold_idx(0-based)'].tolist())!=list(range(info['n_splits'])):raise ValueError('Unexpected fold summary')
    config=read(orie/'configs/xfusegnet.json');manifest_path=orie/config['data']['manifest'];manifest=read(manifest_path)
    candidate_path=ganta/'data/splits/fuseg_2021_xfusegnet/manifest.json'
    candidate=read(candidate_path) if candidate_path.is_file() else None
    candidate_info={'path':str(candidate_path.relative_to(ganta)),'exists':candidate is not None,
        'used_by_selected_python_script':False,'is_evidence_of_partner_810_pool_fold_assignment':False}
    if candidate is not None:
        candidate_info.update({'sha256':sha(candidate_path),'byte_identical_to_Orie_manifest':candidate_path.read_bytes()==manifest_path.read_bytes(),
            'n_labeled':len(candidate.get('labeled',[])),'shuffle_seed':candidate.get('data_settings',{}).get('shuffle_seed'),
            'fingerprint':candidate.get('fingerprint'),
            'interpretation':'An existing 1010-pool seed-42 manifest; selected partner script uses 810 files with seed 200 and never reads this manifest'})
    cp_dir=ganta/'output_fuseg_2021/checkpoints'/run_name
    checkpoint_stats={f'best_model_{i}.pth':file_stats(cp_dir/f'best_model_{i}.pth') for i in range(1,6)}
    audit={'created_at_utc':datetime.now(timezone.utc).isoformat(),'partner_run_id':run_name,
        'partner_source_directory_role':'Read-only external source; outputs generated in a separate publication worktree',
        'reference_Orie_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=orie,text=True).strip(),
        'reference_Orie_config_sha256':sha(orie/'configs/xfusegnet.json'),
        'reference_Orie_manifest_sha256':sha(manifest_path),'reference_Orie_manifest_fingerprint':manifest['fingerprint'],
        'source_files_to_publish':before,'partner_run_info':info,'source_comparison':source_comparison(ganta,orie,config),
        'notebook_evidence':notebook_evidence(ganta),'data_audit':audit_data(ganta,orie,manifest),
        'metrics_verification':verify_cached_metrics(ganta,info,reported,output),
        'reported_single_model_metrics':reported[~reported.Model.str.contains('Average|Voting')].to_dict('records'),
        'reported_fold_summary':folds.to_dict('records'),
        'saved_fold_id_manifest':'No run-linked 810-pool fold IDs found in supplied script/results/checkpoint artifacts; do not substitute the existing 1010-pool manifest or reconstruct folds',
        'existing_manifest_candidate':candidate_info,
        'checkpoint_policy_verification':'Static source if-loss / elif-IoU matches conceptual source policy; checkpoint contents not deserialized',
        'source_run_lineage_limit':'Run metadata has no source hash, run-linked saved split membership, full dependency lock, or RNG states; saved notebooks contain an interrupted training cell',
        'source_models_stat_before':checkpoint_stats,'checkpoints_loaded':False,'pickle_loaded':False,
        'training_executed':False,'model_inference_executed':False,'raw_data_modified':False,
        'audit_dependencies':{'python':platform.python_version(),**{n:metadata.version(n) for n in ['numpy','pandas','Pillow']}},
        'audit_script_sha256':sha(Path(__file__))}
    for relative,expected in before.items():
        path=ganta/relative
        if {'sha256':sha(path),**file_stats(path)}!=expected:raise ValueError(f'Partner source changed during audit: {relative}')
    for name,expected in checkpoint_stats.items():
        if file_stats(cp_dir/name)!=expected:raise ValueError('A partner checkpoint changed during this read-only audit')
    audit['source_files_and_checkpoint_stats_unchanged_during_audit']=True
    write(output/'ganta-xfusegnet-audit.json',audit)
    print(json.dumps({'status':'verified','published_source_files':len(files),'partner_evaluation_source':'provided validation',
        'data_inventory':{s:audit['data_audit'][s] for s in ['train','validation','test']},
        'verified_ensemble_metrics':audit['metrics_verification']['ensembles'],
        'heldout_identical_rgb_to_training':audit['data_audit']['partner_evaluation_identical_rgb_to_partner_training']['count']},indent=2))


if __name__=='__main__':main()
