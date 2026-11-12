"""Post-selection replay, auxiliary-task diagnosis and removable-head inference cost."""

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_heads import build_head_model, morphological_boundary  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_h import phase_h_sources, read  # noqa: E402
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator  # noqa: E402
from foot_ulcer_segmentation.cp2_train import export_evaluation, gpu_snapshot, make_cp2_loader, sync  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-name', default='analysis')
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ('.', '..'):
        raise ValueError('Use a new single-component output directory')
    root, device = args.root.resolve(), torch.device(args.device)
    if read(root / 'train_workflow.json')['status'] != 'completed':
        raise ValueError('Complete primary selection and evaluations first')
    if read(root / 'pretraining_checks.json')['training_sources'] != phase_h_sources():
        raise ValueError('Executed training sources changed')
    config, manifest, selection = read(root / 'protocol.json'), read(root / 'data/manifest.json'), read(root / 'selection.json')
    output = root / args.output_name
    output.mkdir(exist_ok=False)
    setup(config['seed'], config['cpu_threads'])
    names = ['H0', *selection['decisions']]
    initial = torch.load(root / 'initial_weights.pth', map_location='cpu')
    control = ROOT / config['phase_h']['control_root'] / 'B1'
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'run_root': str(root.relative_to(ROOT)),
               'selection_sha256': digest_file(root / 'selection.json'), 'primary_selection_changed': False,
               'source_sha256': digest_file(__file__), 'models': {}}
    for name in names:
        model, _ = build_head_model(config, name, initial)
        run = control if name == 'H0' else root / name
        state = read(run / 'status.json')
        if digest_file(run / 'best.pth') != state['best_checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        checkpoint = torch.load(run / 'best.pth', map_location='cpu')
        if name == 'H0':
            model.base.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        model.to(device).eval()
        loader, rows = make_cp2_loader(config, manifest, 'validation')
        bit_exact = mask_exact = 0
        maximum_difference, offset = 0., 0
        target_pixels, target_canvas = 0, 0
        aux = CP2Evaluator(config['evaluation'], boundaries=True) if name != 'H0' else None
        aux_dir = output / name / 'auxiliary-validation'
        if aux is not None:
            aux_dir.mkdir(parents=True)
        aux_index = []
        first_image = None
        with torch.no_grad():
            for images, truth in loader:
                images = images.to(device)
                values = model(images)
                probabilities = values.sigmoid().cpu().numpy()
                bare = model(images, auxiliary=False).sigmoid().cpu().numpy()
                np.testing.assert_array_equal(probabilities, bare)
                if first_image is None:
                    first_image = images[:1].clone()
                if aux is not None:
                    target = truth if name == 'H1' else morphological_boundary(truth)
                    auxiliary = values.auxiliary_logits.sigmoid().cpu().numpy()
                for j, probability in enumerate(probabilities):
                    row = rows[offset]
                    path = run / 'predictions/validation' / (Path(row['name']).stem + '.npz')
                    with np.load(path, allow_pickle=False) as cached:
                        difference = float(np.max(np.abs(probability[0] - cached['probability'])))
                        maximum_difference = max(maximum_difference, difference)
                        bit_exact += int(difference == 0)
                        np.testing.assert_allclose(probability[0], cached['probability'], rtol=0, atol=1e-6)
                        np.testing.assert_array_equal(probability[0] >= .5, cached['prediction'])
                        mask_exact += 1
                    if aux is not None:
                        target_array = target[j, 0].numpy().astype(np.uint8)
                        aux.add(row, target_array, auxiliary[j, 0])
                        target_pixels += int(target_array.sum())
                        target_canvas += target_array.size
                        file = aux_dir / (Path(row['name']).stem + '.npz')
                        with file.with_suffix('.npz.tmp').open('wb') as stream:
                            np.savez_compressed(stream, probability=auxiliary[j, 0],
                                                prediction=(auxiliary[j, 0] >= .5).astype(np.uint8),
                                                truth=target_array)
                        file.with_suffix('.npz.tmp').replace(file)
                        aux_index.append({'sample_id': row['sample_id'], 'file': file.name, 'sha256': digest_file(file)})
                    offset += 1
        assert offset == len(rows)
        value = {'validation_images': offset, 'bit_exact_primary_probability_images': bit_exact,
                 'maximum_primary_probability_difference': maximum_difference,
                 'hard_mask_replay_exact_images': mask_exact, 'with_without_head_main_bit_exact': True,
                 'checkpoint_sha256': state['best_checkpoint_sha256'],
                 'auxiliary_target': config['phase_h']['targets'].get(name),
                 'auxiliary_target_foreground_fraction': target_pixels / target_canvas if target_canvas else None}
        if aux is not None:
            write_json(aux_dir / 'manifest.json', aux_index)
            value['auxiliary_metrics'] = export_evaluation(output / name / 'auxiliary-validation.json', aux, {
                'variant': name, 'split': 'validation', 'target': value['auxiliary_target'],
                'interpretation': 'AUXILIARY task only; H1 mask and H2 boundary scores are NOT directly comparable',
                'auxiliary_group_caveat': 'Size groups describe auxiliary target area; H2 band area is NOT original wound area',
                'checkpoint_sha256': state['best_checkpoint_sha256'], 'selection_sha256': summary['selection_sha256']})
        del images, values
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        times = []
        with torch.no_grad():
            for _ in range(5):
                model(first_image, auxiliary=False).sigmoid()
            sync(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(30):
                started = time.perf_counter()
                model(first_image, auxiliary=False).sigmoid()
                sync(device)
                times.append(time.perf_counter() - started)
        value['mask_only_inference'] = {
            'input_shape': list(first_image.shape), 'warmup': 5, 'timed_steps': 30,
            'median_ms': float(np.median(times) * 1000), 'p95_ms': float(np.percentile(times, 95) * 1000),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            'gpu_overall': gpu_snapshot() if device.type == 'cuda' else None,
            'scope': 'main forward+sigmoid only; head bypassed; no data/GT/export; shared GPU, not exclusive hardware'}
        summary['models'][name] = value
        del model, first_image
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'Verified real primary replay and auxiliary diagnosis {name}', flush=True)
    assert digest_file(root / 'selection.json') == summary['selection_sha256']
    summary.update(status='COMPLETED — real post-selection replay and auxiliary diagnosis',
                   caveats=['No retraining or primary reselection', 'Auxiliary targets differ; no H1-vs-H2 auxiliary metric ranking',
                            'GT-empty means annotation empty, not healthy patient; H2 target bands are not wound areas'])
    write_json(output / 'summary.json', summary)
    write_json(root / 'analysis_index.json', {'directory': str(output.relative_to(ROOT)), 'summary_sha256': digest_file(output / 'summary.json')})
    write_json(ROOT / 'reports/cp2-phase-h-analysis.json', summary)


if __name__ == '__main__':
    main()
