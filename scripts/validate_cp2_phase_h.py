"""Independent artifact, target, checkpoint, chronology and preservation checks for H."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_data import verify_cp2_manifest  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_h import expansion_decision, phase_h_sources, read, select_candidate, verify_inputs  # noqa: E402
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator  # noqa: E402
from validate_cp2_phase_a import validate_exports  # noqa: E402
from validate_cp2_phase_l import compare, compare_csv  # noqa: E402


def auxiliary_exports(folder, rows, expected, settings, variant):
    index = read(folder / 'manifest.json')
    assert len(index) == len(rows)
    by_id = {r['sample_id']:r for r in rows}
    assert {r['sample_id'] for r in index} == set(by_id)
    evaluator = CP2Evaluator(settings, boundaries=True)
    for record in index:
        path = folder / record['file']
        assert digest_file(path) == record['sha256']
        row = by_id[record['sample_id']]
        with Image.open(ROOT / row['mask_path']) as im:
            mask = (np.asarray(im.convert('L')) / 255).astype(np.uint8)
        truth = mask
        if variant == 'H2':
            kernel = np.ones((3,3),np.uint8)
            truth = cv2.dilate(mask,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0) - cv2.erode(mask,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        with np.load(path,allow_pickle=False) as data:
            assert data['probability'].dtype == np.float32
            np.testing.assert_array_equal(data['truth'],truth)
            np.testing.assert_array_equal(data['prediction'],data['probability']>=.5)
            evaluator.add(row,truth,data['probability'])
    values = evaluator.summary()
    compare(values,{k:expected[k] for k in values})
    return evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    root = args.root.resolve();output = args.output or root/'final_validation.json'
    if output.exists():raise FileExistsError('Preserve completed QA; use a new output filename')
    config,gate = read(root/'protocol.json'),read(root/'pretraining_checks.json')
    assert gate['training_sources']==phase_h_sources()
    assert gate['protocol_fingerprint']==digest_json(config)
    assert read(root/'train_workflow.json')['status']=='completed'
    manifest = verify_cp2_manifest(read(root/'data/manifest.json'))
    assert manifest['fingerprint']==config['phase_h']['control_manifest_fingerprint']
    assert digest_file(root/'initial_weights.pth')==config['phase_h']['control_initial_weights_sha256']
    prior = read(root/'protected_prior_sha256.json')
    with ThreadPoolExecutor(max_workers=6) as pool:
        changed=[n for n,ok in pool.map(lambda n:(n,(ROOT/n).is_file() and digest_file(ROOT/n)==prior[n]),prior) if not ok]
    assert not changed,changed
    selection,expansion=read(root/'selection.json'),read(root/'expansion_decision.json')
    names=list(selection['decisions']);assert names==['H1',*expansion['remaining_models']]
    assert selection['validation_only'] and expansion['validation_only']
    control=ROOT/config['phase_h']['control_root']/'B1'
    assert digest_file(control/'best.pth')==config['phase_h']['control_B1_checkpoint_sha256']
    paired=verify_inputs(config,root,names);compare(paired,read(root/'train_paired_controls.json'))
    public=read(ROOT/'reports/cp2-phase-h-results.json')
    profiles=read(root/'architecture_preflight.json')
    compare(public['profiles'],profiles)
    assert profiles['paired_auxiliary_head_tensors'] and profiles['initial_main_predictions_H0_H1_H2_equal']
    assert profiles['head_removal_preserves_main_forward']
    compare(public['statuses']['H0'],read(control/'status.json'))
    for split in ('validation','test'):
        compare(public['metrics']['H0'][split],read(control/f'evaluation-{split}.json'))
    prereg=read(root/'preregistration.json')
    assert prereg['training_sources']==phase_h_sources() and prereg['protocol_fingerprint']==digest_json(config)
    registered=datetime.fromisoformat(prereg['registered_before_full_training_utc'])
    expanded=datetime.fromisoformat(expansion['created_utc']).timestamp()
    selected=datetime.fromisoformat(selection['created_utc']).timestamp()
    evaluations,validation,head_hashes=[],{},set()
    for name in names:
        run=root/name;s=read(run/'status.json')
        assert s['status']=='completed' and not s['diagnostic_smoke_only']
        compare(s,public['statuses'][name])
        identity=digest_json({'config':config,'architecture':name,'manifest':manifest['fingerprint'],'sources':phase_h_sources()})
        assert s['identity']==identity and s['training_sources']==phase_h_sources()
        assert read(run/'split-manifest.json')==manifest
        started=datetime.fromisoformat(read(run/'environment.json')['created_at_utc'])
        assert registered<started
        if name=='H2':assert started.timestamp()>=expanded
        h=read(run/'history.json');best=max(h,key=lambda x:x['validation_image_dice'])
        assert best['epoch']==s['selected_epoch'] and len(h)==s['completed_epochs']
        assert best['validation_image_dice']==s['selected_validation_image_dice']
        assert s['checkpoint_replay_exact'] and digest_file(run/'best.pth')==s['best_checkpoint_sha256']
        initialization=read(run/'initialization_verified.json')
        assert initialization['all_original_named_tensors_equal']
        assert initialization['source_tensor_sha256']==config['phase_h']['control_initial_tensor_sha256']
        assert initialization['head_parameters']==33
        head_hashes.add(initialization['head_tensor_sha256'])
        for file,epoch in (('best.pth',s['selected_epoch']),('last.pth',s['completed_epochs'])):
            checkpoint=torch.load(run/file,map_location='cpu')
            assert checkpoint['identity']==identity and checkpoint['completed_epochs']==epoch
            del checkpoint
        for split in ('validation','test'):
            expected=read(run/f'evaluation-{split}.json')
            evaluator=validate_exports(run/'predictions'/split,manifest['splits'][split],expected,config['evaluation'])
            compare(expected,public['metrics'][name][split]);compare(expected,read(run/'final_evaluation.json')[split])
            for path in (run/f'evaluation-{split}.csv',ROOT/f'reports/tables/cp2-phase-h-{name.lower()}-{split}.csv'):
                with path.open(newline='') as stream:compare_csv(list(csv.DictReader(stream)),evaluator.rows)
            if split=='validation':
                assert expected['all']['image_based']['dice']==s['selected_validation_image_dice'];validation[name]=expected
            else:
                assert all((run/'predictions/test'/item['file']).stat().st_mtime>=selected for item in read(run/'predictions/test/manifest.json'))
            evaluations.append({'variant':name,'split':split,'predictions':len(evaluator.rows),'checksums_GT_threshold_metrics_CSV':'passed'})
        cost=public['resources']['fresh_models'][name]
        assert cost['completed_epochs']==len(h)
        assert cost['sum_completed_epoch_compute_seconds']==sum(r['epoch_compute_seconds'] for r in h)
    assert len(head_hashes)==1
    baseline=read(control/'evaluation-validation.json')
    expected=expansion_decision(validation['H1'],baseline,config)
    compare({k:v for k,v in expansion.items() if k!='created_utc'},{k:v for k,v in expected.items() if k!='created_utc'})
    expected=select_candidate(validation,baseline,profiles['profiles'])
    compare({k:v for k,v in selection.items() if k!='created_utc'},{k:v for k,v in expected.items() if k!='created_utc'})
    index=read(root/'analysis_index.json');analysis_root=ROOT/index['directory']
    assert digest_file(analysis_root/'summary.json')==index['summary_sha256']
    analysis=read(analysis_root/'summary.json');compare(analysis,read(ROOT/'reports/cp2-phase-h-analysis.json'))
    assert analysis['selection_sha256']==digest_file(root/'selection.json')
    auxiliary=[]
    for name,value in analysis['models'].items():
        assert value['validation_images']==118 and value['with_without_head_main_bit_exact']
        assert value['hard_mask_replay_exact_images']==118
        assert value['maximum_primary_probability_difference']<=1e-6
        if name!='H0':
            e=auxiliary_exports(analysis_root/name/'auxiliary-validation',manifest['splits']['validation'],value['auxiliary_metrics'],config['evaluation'],name)
            with (analysis_root/name/'auxiliary-validation.csv').open(newline='') as stream:compare_csv(list(csv.DictReader(stream)),e.rows)
            auxiliary.append({'variant':name,'predictions':len(e.rows),'independent_cv2_target_oracle_metrics_and_checksum':'passed'})
    for f in public['figures']:assert digest_file(ROOT/f['path'])==f['sha256']
    result={'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),'run_root':str(root.relative_to(ROOT)),
            'historical_and_user_files_unchanged':len(prior),'changed':changed,
            'training_sources_and_protocol_unchanged':True,'preregistered_before_full_training':True,
            'all_original_initial_tensors_match_B1':True,'paired_H1_H2_head_initialization':True,
            'initial_main_H0_H1_H2_bit_exact':True,'removable_head_main_predictions_bit_exact':True,
            'paired_inputs':paired,'duplicate_rgb_overlap':0,'patient_independence_established':False,
            'expansion_recomputed_validation_only_and_before_H2':True,
            'selection_recomputed_validation_only_and_before_all_new_test_exports':True,
            'checkpoints_hashes_and_earliest_peak_epoch_verified':True,
            'primary_predictions_verified':sum(e['predictions'] for e in evaluations),'evaluations':evaluations,
            'auxiliary_validation_predictions_verified':sum(e['predictions'] for e in auxiliary),'auxiliary':auxiliary,
            'preflight':gate,'limitations':'one seed; RGB group not patient split; previously exposed test; no external data'}
    write_json(output,result);write_json(ROOT/'reports/cp2-phase-h-validation.json',result)
    print({k:result[k] for k in ('status','historical_and_user_files_unchanged','primary_predictions_verified','auxiliary_validation_predictions_verified')},flush=True)


if __name__=='__main__':
    main()
