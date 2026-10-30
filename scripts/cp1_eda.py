"""Read-only EDA of the saved baseline train membership and source input pipeline."""

import argparse
import ast
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/runs/cp1-baseline-01"
OUTPUT = ROOT / "outputs/analysis/cp1-eda-01"
TABLES = ROOT / "reports/tables"
CSV = TABLES / "cp1-eda-per-image.csv"
SUMMARY = TABLES / "cp1-eda-summary.json"
DEMO_SEED = 20261030
BIN_SPECS = [
    ("Kosong", 0., 0.), (">0–0,5%", 0., .005), (">0,5–1%", .005, .01),
    (">1–5%", .01, .05), (">5–10%", .05, .1), (">10%", .1, 1.),
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required existing artifact is missing: {path}")
    return json.loads(Path(path).read_text())


def write(path, value):
    def convert(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        raise TypeError(type(item).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert) + "\n")
    temporary.replace(path)


def area_statistics(fractions):
    values = np.asarray(fractions, dtype=np.float64)
    if values.size and (not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
        raise ValueError("Area fractions must be finite and within [0,1]")
    if not values.size:
        return {"n": 0, **{name: None for name in ["min", "q1", "median", "q3", "max", "mean"]}}
    quantiles = np.quantile(values, [0, .25, .5, .75, 1], method="linear")
    return {"n": int(values.size), **dict(zip(["min", "q1", "median", "q3", "max"], map(float, quantiles))),
            "mean": float(values.mean())}


def area_bins(fractions):
    values = np.asarray(fractions, dtype=np.float64)
    area_statistics(values)
    result = []
    for label, lower, upper in BIN_SPECS:
        condition = values == 0 if upper == 0 else (values > lower) & (values <= upper)
        count = int(condition.sum())
        result.append({"label": label, "lower_fraction": lower, "upper_fraction": upper,
                       "interval": "[0,0]" if upper == 0 else "(lower,upper]",
                       "count": count, "percent_of_train": count / len(values) * 100 if len(values) else None})
    if sum(row["count"] for row in result) != len(values):
        raise ValueError("Area bins do not cover every image exactly once")
    return result


def background_foreground_ratio(background, foreground):
    return background / foreground if foreground else None


def choose_examples(frame):
    positive = frame[~frame.empty_gt].copy()
    chosen, used = [], set()
    for quantile, label in [(.1, "Luas rendah · Q10 positif"), (.5, "Luas median · Q50 positif"), (.9, "Luas tinggi · Q90 positif")]:
        if positive.empty: break
        target = float(np.quantile(positive.area_fraction, quantile, method="linear"))
        candidates = positive[~positive.sample_id.isin(used)].copy()
        if candidates.empty: break
        candidates["distance"] = abs(candidates.area_fraction - target)
        row = candidates.sort_values(["distance", "sample_id"], kind="mergesort").iloc[0]
        chosen.append({"sample_id": row.sample_id, "label": label, "quantile": quantile,
                       "target_area_fraction": target, "area_fraction": float(row.area_fraction),
                       "reason": "Nearest positive-mask area to the linear quantile; distinct IDs; ties by filename"})
        used.add(row.sample_id)
    empty = frame[frame.empty_gt].sort_values("sample_id")
    if not empty.empty:
        chosen.append({"sample_id": empty.iloc[0].sample_id, "label": "GT kosong", "quantile": None,
                       "target_area_fraction": 0., "area_fraction": 0., "reason": "Lexicographically first empty GT in actual train"})
    return chosen


def load_identity():
    manifest_path = RUN / "split-manifest.json"
    manifest, config, environment, status = (read(manifest_path), read(RUN / "config.json"),
                                            read(RUN / "environment.json"), read(RUN / "status.json"))
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if json_digest(payload) != manifest["fingerprint"]:
        raise ValueError("Saved run manifest fingerprint is invalid")
    if manifest["fingerprint"] != environment["split_fingerprint"] or manifest["fingerprint"] != status["split_fingerprint"]:
        raise ValueError("Saved run manifest does not match the actual baseline run")
    if json_digest(config) != environment["config_fingerprint"] or json_digest(config) != status["config_fingerprint"]:
        raise ValueError("Saved baseline configuration identity differs")
    if status["status"] != "completed" or status["smoke_only"]:
        raise ValueError("Expected the completed full baseline")
    train = manifest["splits"]["train"]
    # 707 is the requested analysis scope, not a substituted computed result.
    if len(train) != status["split_counts"]["train"] or len(train) != 707:
        raise ValueError(f"Wrong analysis scope: saved run contains {len(train)} train samples")
    for key in ["name", "image_path", "mask_path"]:
        if len({r[key] for r in train}) != len(train):
            raise ValueError(f"Duplicate train membership key: {key}")
    train_paths = {r["image_path"] for r in train}
    for split in ["validation", "test"]:
        if train_paths & {r["image_path"] for r in manifest["splits"][split]}:
            raise ValueError("A saved image path has both train and held-out membership")
    if {r['name'] for r in train} & {r['name'] for r in manifest['splits']['validation']}:
        raise ValueError("Provided-train filenames were assigned to both train and validation")
    for relative, expected in environment["pipeline_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen baseline implementation changed: {relative}")
    original_manifest = ROOT / config["data"]["split_manifest"]
    identity = {"run_id": "cp1-baseline-01", "manifest_path": str(manifest_path.relative_to(ROOT)),
                "manifest_sha256": sha(manifest_path), "manifest_fingerprint": manifest["fingerprint"],
                "live_manifest_path": str(original_manifest.relative_to(ROOT)),
                "live_manifest_same_bytes": original_manifest.is_file() and original_manifest.read_bytes() == manifest_path.read_bytes(),
                "config_path": str((RUN / 'config.json').relative_to(ROOT)), "config_sha256": sha(RUN / 'config.json'),
                "config_fingerprint": json_digest(config), "baseline_pipeline_sha256": environment['pipeline_sha256'],
                "training_membership_reconstructed": False, "new_split_created": False,
                "train_test_filename_collisions": len({r['name'] for r in train} & {r['name'] for r in manifest['splits']['test']})}
    return manifest, config, environment, identity


def version_record(names):
    return {name: metadata.version(name) for name in names}


def prepare_eda():
    manifest, config, environment, identity = load_identity()
    for name, expected in environment['packages'].items():
        if metadata.version(name) != expected:
            raise ValueError(f"Use the unchanged frozen training environment: {name}")
    records, raw_values, dimensions, rgb_groups = [], set(), Counter(), defaultdict(list)
    for order, sample in enumerate(manifest['splits']['train']):
        image_path, mask_path = ROOT / sample['image_path'], ROOT / sample['mask_path']
        if image_path.name != sample['name'] or mask_path.name != sample['name']:
            raise ValueError("Filename pairing differs from the saved sample ID")
        for field in ['image', 'mask']:
            path = ROOT / sample[field + '_path']
            if not path.resolve().is_relative_to((ROOT / config['data']['root']).resolve()):
                raise ValueError("A train path resolves outside the declared dataset")
            if sha(path) != sample[field + '_sha256']:
                raise ValueError(f"Missing or changed raw {field}: {sample['name']}")
        with Image.open(image_path) as opened:
            image_mode = opened.mode; image = np.array(opened.convert('RGB'))
        with Image.open(mask_path) as opened:
            mask_mode = opened.mode; stored = np.array(opened)
        values = set(map(int, np.unique(stored))); raw_values.update(values)
        if not values.issubset({0, 255}):
            raise ValueError(f"Raw Chronic Wound mask is not 0/255: {sample['name']}, values={values}")
        if stored.ndim == 3:
            if stored.shape[2] != 3 or not (np.array_equal(stored[...,0],stored[...,1]) and np.array_equal(stored[...,0],stored[...,2])):
                raise ValueError("Mask channels disagree")
            mask = stored[...,0]
        elif stored.ndim == 2:
            mask = stored
        else:
            raise ValueError("Unexpected mask dimensions")
        height, width = mask.shape
        if image.shape[:2] != (height,width) or [height,width] != config['data']['image_size']:
            raise ValueError("Image/mask resolution differs from the run configuration")
        foreground = mask == 255
        count = int(foreground.sum()); total = height * width
        y,x = np.nonzero(foreground)
        rgb_hash = hashlib.sha256(image.tobytes()).hexdigest();rgb_groups[rgb_hash].append(sample['name'])
        dimensions[(height,width)] += 1
        records.append({'sample_id':sample['name'],'split':'train','height':height,'width':width,
                        'foreground_pixels':count,'background_pixels':total-count,'area_fraction':count/total,
                        'empty_gt':count==0,'centroid_x_fraction':float((x+.5).mean()/width) if count else None,
                        'centroid_y_fraction':float((y+.5).mean()/height) if count else None,
                        'image_path':sample['image_path'],'mask_path':sample['mask_path'],
                        'image_sha256':sample['image_sha256'],'mask_sha256':sample['mask_sha256'],
                        'image_mode':image_mode,'mask_mode':mask_mode,'rgb_sha256':rgb_hash,'manifest_train_order':order})
    frame = pd.DataFrame(records)
    n, n_empty = len(frame), int(frame.empty_gt.sum())
    total_fg, total_bg = int(frame.foreground_pixels.sum()), int(frame.background_pixels.sum())
    total = int((frame.height * frame.width).sum())
    if total_fg + total_bg != total: raise ValueError("Pixel accounting failed")
    examples = choose_examples(frame)
    uniform = len(dimensions) == 1
    mean_area_pct = float(frame.area_fraction.mean()*100)
    fg_pct = total_fg/total*100
    if uniform and not np.isclose(mean_area_pct,fg_pct,atol=1e-12,rtol=0):
        raise ValueError("Uniform-resolution global and per-image means disagree")
    duplicate_groups = [ids for ids in rgb_groups.values() if len(ids)>1]
    summary = {'created_at_utc':datetime.now(timezone.utc).isoformat(),'dataset':'Chronic Wound public source',
               'split':'actual_train','n_images':n,'n_masks':n,'n_empty_gt':n_empty,'n_positive_gt':n-n_empty,
               'empty_gt_fraction':n_empty/n,'empty_gt_percent':n_empty/n*100,'n_unique_filenames':frame.sample_id.nunique(),
               'resolutions':[{'height':h,'width':w,'n_images':count,'n_masks':count} for (h,w),count in sorted(dimensions.items())],
               'mask_unique_values_raw':sorted(raw_values),'total_foreground_pixels':total_fg,'total_background_pixels':total_bg,
               'total_mask_pixels':total,'foreground_pct_global':fg_pct,'background_pct_global':total_bg/total*100,
               'background_foreground_ratio':background_foreground_ratio(total_bg,total_fg),
               'zero_foreground_denominator_policy':'JSON null when total foreground is zero; ratio is undefined',
               'mean_per_image_area_percent_all':mean_area_pct,'uniform_resolution':uniform,
               'global_equals_mean_per_image_when_uniform':uniform and bool(np.isclose(fg_pct,mean_area_pct,atol=1e-12,rtol=0)),
               'area_statistics_units':'fraction of full H*W canvas; multiply by 100 for percent',
               'quantile_method':'numpy.quantile(method=linear)',
               'area_statistics_all':area_statistics(frame.area_fraction),
               'area_statistics_positive':area_statistics(frame.loc[~frame.empty_gt,'area_fraction']),
               'area_bins':area_bins(frame.area_fraction),
               'bin_interpretation':'Descriptive analysis bins, not clinical categories or author C1–C9',
               'centroid_definition':'Mean foreground pixel-center coordinates: mean(x+0.5)/W, mean(y+0.5)/H; undefined on empty masks',
               'centroid_statistics_positive':{axis:area_statistics(frame.loc[~frame.empty_gt,'centroid_'+axis+'_fraction']) for axis in ['x','y']},
               'denominator':'All single-channel mask pixels across actual train; full canvas including existing black padding',
               'mask_policy':'Verify stored raw values are a subset of {0,255} and RGB mask channels agree BEFORE binarization; foreground is grayscale ==255, background ==0',
               'dataset_unchanged':True,'deduplication_applied':False,'within_train_rgb_duplicate_groups':duplicate_groups,
               'overlap_limitations':'Existing cross-split RGB overlap audit is retained; no validation/test pixel EDA is computed; patient independence remains unverified',
               'prior_overlap_audit':'reports/cp1-analysis-01.md',
               'manifest':identity,'example_selection':{'seed':None,'rule':'Deterministic area-quantile proximity then lexical filename; distinct IDs, plus first lexical empty GT','examples':examples},
               'demo_seed':DEMO_SEED,'dependencies':{'data_and_augmentation':{'python':platform.python_version(),'interpreter':sys.executable,
                   **version_record(['numpy','pandas','Pillow','opencv-python-headless','albumentations','torch','torchvision'])}},
               'training_performed':False,'model_inference_performed':False,'validation_test_pixel_statistics_computed':False,
               'generator_source_sha256':sha(Path(__file__))}
    TABLES.mkdir(parents=True,exist_ok=True)
    frame.to_csv(CSV,index=False,float_format='%.17g')
    write(SUMMARY,summary)
    preprocessing = prepare_demo(frame,summary,config)
    summary['preprocessing_metadata_path']=str((OUTPUT/'preprocessing/metadata.json').relative_to(ROOT))
    summary['preprocessing_metadata_sha256']=sha(OUTPUT/'preprocessing/metadata.json')
    summary['per_image_csv_sha256']=sha(CSV)
    write(SUMMARY,summary)
    print(json.dumps({'status':'prepared','n_images':n,'foreground_pct':fg_pct,'n_empty_gt':n_empty,
                      'geometric_demo_seed':preprocessing['geometric']['seed'],'intensity_demo_seed':preprocessing['intensity']['seed']}),flush=True)


def prepare_demo(frame,summary,config):
    import albumentations as A
    import cv2
    from foot_ulcer_segmentation.reference import load_reference, REFERENCE
    smp, _, functions = load_reference()  # Definitions only; no entrypoint, model, or weights are run.
    training = functions['get_training_augmentation']()
    validation = functions['get_validation_augmentation']()
    source = REFERENCE/'fusegnet_train.py'
    params = smp.encoders.get_preprocessing_params(config['model']['encoder_name'],config['model']['encoder_weights'])
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config['model']['encoder_name'],config['model']['encoder_weights'])
    choices = [item for item in summary['example_selection']['examples'] if item['quantile'] is not None]
    if not choices: raise ValueError("No positive training GT is available for a visible paired transform demo")
    selected = choices[-1]  # High positive quantile for legibility, chosen only from actual train.
    row = frame.set_index('sample_id').loc[selected['sample_id']]
    bgr = cv2.imread(str(ROOT/row.image_path)); image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    grayscale=cv2.imread(str(ROOT/row.mask_path),0)
    if not set(map(int,np.unique(grayscale))).issubset({0,255}):raise ValueError('Demo raw mask is not binary 0/255')
    mask=np.expand_dims(grayscale/255.,-1)
    geometry=copy.deepcopy(training.transforms[1].transforms[1])
    intensity=copy.deepcopy(training.transforms[3].transforms[1])
    if not isinstance(geometry,A.ShiftScaleRotate) or not isinstance(intensity,A.RandomBrightnessContrast):
        raise ValueError('Unexpected baseline augmentation branch')
    if tuple(geometry.shift_limit_x)!=(0,0) or tuple(geometry.rotate_limit)!=(-30,30):
        raise ValueError('Expected the Chronic Wound rotate-only branch, not x-FUSegNet')
    if validation.transforms:raise ValueError('Expected empty validation/test augmentation')
    # Isolate the actual branch in a forced-active demo. Full train probabilities are recorded separately.
    demos={}
    for kind,transform,seed in [('geometric',geometry,DEMO_SEED),('intensity',intensity,DEMO_SEED+2)]:
        random.seed(seed);np.random.seed(seed)
        replay=A.ReplayCompose([transform],p=1.)
        result=replay(image=image.copy(),mask=mask.copy())
        applied=result['replay']['transforms'][0]
        if not applied['applied']:raise ValueError('Demo transform did not activate')
        if kind=='geometric':
            call=applied['params']
            expected_image=transform.apply(image.copy(),**call)
            expected_mask=transform.apply_to_mask(mask.copy(),**call)
            if not np.array_equal(expected_image,result['image']) or not np.array_equal(expected_mask,result['mask']):
                raise ValueError('Geometry does not use the same sampled parameters on the paired image and mask')
            if np.array_equal(mask,result['mask']):raise ValueError('Geometric demo did not visibly change GT')
        elif not np.array_equal(mask,result['mask']):
            raise ValueError('Intensity transform modified the target mask')
        if np.array_equal(image,result['image']):raise ValueError('Demo did not alter the image')
        if result['image'].shape!=image.shape or result['mask'].shape!=mask.shape:
            raise ValueError('Unexpected resizing in selected demo')
        demos[kind]={'image':result['image'],'mask':result['mask'],'seed':seed,'replay':result['replay']}
    values={'original':{'image':image,'mask':mask,'seed':None},**demos}
    output=OUTPUT/'preprocessing';output.mkdir(parents=True,exist_ok=True)
    tensor_rows={}
    for kind,value in values.items():
        folder=output/kind;folder.mkdir(exist_ok=True)
        Image.fromarray(value['image']).save(folder/'image.png')
        Image.fromarray((value['mask'][...,0]*255).astype(np.uint8)).save(folder/'mask.png')
        prepared=functions['get_preprocessing'](preprocessing_fn)(image=value['image'].copy(),mask=value['mask'].copy())
        expected=((value['image'].astype(np.float64)/255-np.array(params['mean']))/np.array(params['std'])).transpose(2,0,1).astype(np.float32)
        if not np.array_equal(expected,prepared['image']):raise ValueError('ImageNet normalization differs from the source implementation')
        if not np.array_equal(value['mask'].transpose(2,0,1).astype(np.float32),prepared['mask']):
            raise ValueError('Source mask tensor conversion differs')
        if prepared['image'].dtype!=np.float32 or prepared['mask'].dtype!=np.float32:
            raise ValueError('Unexpected preprocessing tensor dtype')
        tensor_rows[kind]={}
        for field in ['image','mask']:
            path=folder/(field+'_chw.npy');np.save(path,prepared[field],allow_pickle=False)
            tensor_rows[kind][field]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'shape':list(prepared[field].shape),
                                    'dtype':str(prepared[field].dtype),'min':float(prepared[field].min()),'max':float(prepared[field].max())}
        tensor_rows[kind]['display_image']={'path':str((folder/'image.png').relative_to(ROOT)),'sha256':sha(folder/'image.png')}
        tensor_rows[kind]['display_mask']={'path':str((folder/'mask.png').relative_to(ROOT)),'sha256':sha(folder/'mask.png')}
    dataset=functions['Dataset']([row.name],str((ROOT/row.image_path).parent),str((ROOT/row.mask_path).parent),
                augmentation=validation,preprocessing=functions['get_preprocessing'](preprocessing_fn))
    source_image,source_mask=dataset[0]
    if not np.array_equal(source_image,np.load(ROOT/tensor_rows['original']['image']['path'])) or not np.array_equal(source_mask,np.load(ROOT/tensor_rows['original']['mask']['path'])):
        raise ValueError('Unaugmented source Dataset output differs from traced preprocessing')
    nodes=ast.parse(source.read_text()).body
    definitions={node.name:{'line':node.lineno,'ast_sha256':hashlib.sha256(ast.dump(node).encode()).hexdigest()} for node in nodes if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ['Dataset','get_training_augmentation','get_validation_augmentation','get_preprocessing','to_tensor']}
    def leaves(transform):
        return [leaf for child in transform.transforms for leaf in leaves(child)] if hasattr(transform,'transforms') else [transform]
    transforms=[{'class':type(t).__name__,'p':t.p,'target_kind':'paired geometry' if isinstance(t,A.DualTransform) else 'image only',
                 'targets':sorted(t.targets.keys())} for t in leaves(training)]
    if any(type(t).__name__ in {'Resize','RandomResizedCrop','PadIfNeeded'} for t in leaves(training)):
        raise ValueError('Unexpected explicit resize/padding in baseline augmentation')
    record={'sample_id':row.name,'split':'actual_train','selection_reason':'Highest selected positive quantile (Q90), for visibility; no test used',
            'source_path':str(source.relative_to(ROOT)),'source_sha256':sha(source),'safe_loader':'foot_ulcer_segmentation.reference.load_reference extracts definitions via AST; training entrypoint not imported',
            'source_definitions':definitions,'sequence':['OpenCV read photo BGR -> RGB uint8 HWC','OpenCV mask grayscale /255 float64 HWC1',
               'Training augmentation with paired image/mask kwargs','ImageNet normalization on photo only','HWC -> CHW float32 for photo and mask'],
            'training_augmentation':A.to_dict(training),'transform_targets':transforms,
            'validation_test_augmentation':A.to_dict(validation),'validation_test_training_augmentation_applied':False,
            'normalization':params,'no_additional_resize':True,'native_image_shape':list(image.shape),'mask_before_tensor_shape':list(mask.shape),
            'model_instantiated':False,'training_executed':False,'inference_executed':False,
            'geometric':{'seed':demos['geometric']['seed'],'transform':A.to_dict(geometry),'replay':demos['geometric']['replay'],'same_sampled_parameters_verified':True},
            'intensity':{'seed':demos['intensity']['seed'],'transform':A.to_dict(intensity),'replay':demos['intensity']['replay'],'mask_unchanged_verified':True},
            'demo_probability_policy':'DEMO TRANSFORMASI: inner source branch isolated under ReplayCompose(p=1). This is NOT the probability of that branch in training.',
            'demos_displayed_before_imagenet_normalization':True,'tensors':tensor_rows,
            'x_fusegnet_difference':'Baseline rotate-only shift_limit=0; xfusegnet_train.py rotate branch shift_limit=0.1. Demo uses baseline.'}
    write(output/'metadata.json',record)
    return record


def capture_protected_inputs():
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    # These generated EDA outputs are deliberately overwritten on a rebuild.
    generated={'notebooks/04_cp1_eda.ipynb','reports/cp1-eda-01.md',
               'reports/tables/cp1-eda-per-image.csv','reports/tables/cp1-eda-summary.json',
               'reports/tables/cp1-eda-split-manifest.json','reports/tables/cp1-eda-preprocessing.json',
               'reports/tables/cp1-eda-verification.json'}
    files={name:sha(ROOT/name) for name in paths if name and (ROOT/name).is_file()
           and name not in generated and not name.startswith('reports/figures/cp1-eda-01/')}
    stats={}
    for path in RUN.iterdir():
        if path.is_file():
            stat=path.stat();stats[str(path.relative_to(ROOT))]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino}
    write(OUTPUT/'protected-inputs-before.json',{'existing_tracked_files_sha256':files,'baseline_artifacts_stat':stats})


def verify_protected_inputs():
    before=read(OUTPUT/'protected-inputs-before.json');changed=[]
    for name,expected in before['existing_tracked_files_sha256'].items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=expected:changed.append(name)
    for name,expected in before['baseline_artifacts_stat'].items():
        path=ROOT/name
        if not path.is_file():changed.append(name);continue
        stat=path.stat()
        if {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino}!=expected:changed.append(name)
    if changed:raise ValueError(f'Protected existing input changed during EDA: {changed}')
    return {'existing_tracked_files_checked':len(before['existing_tracked_files_sha256']),
            'baseline_artifacts_checked':len(before['baseline_artifacts_stat']),'all_unchanged':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if not args.prepare:parser.error('Use --prepare from the pinned training environment')
    try:prepare_eda()
    except Exception as error:
        write(OUTPUT/'failure.json',{'error':f'{type(error).__name__}: {error}','numbers_substituted':False});raise
