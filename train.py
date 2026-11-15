"""Train the fixed best CatBoost + independent image fusion; no search or Val tuning."""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import data as D
import features as F
from image_features import extract_embeddings
from image_model import fit_image, predict_image
from numeric import fit_numeric, predict_numeric
from reporting import report, predictions, error_summary
from utils import blend, digest, dump, file_hash, run_lock, write_csv, write_json


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', type=Path, default=Path('.'))
    p.add_argument('--train_csv', type=Path)
    p.add_argument('--validation_csv', type=Path)
    p.add_argument('--no_external_validation', action='store_true')
    p.add_argument('--image_root', type=Path)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--stage', choices=['all', 'cv', 'final'], default='all')
    p.add_argument('--split_manifest', type=Path)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--iterations', type=int, default=600)
    p.add_argument('--task_type', choices=['CPU', 'GPU'], default='CPU')
    p.add_argument('--devices', default='0')
    p.add_argument('--gpu_ram_part', type=float, default=.60)
    p.add_argument('--image_device', default='cpu')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--image_epochs', type=int, default=12)
    p.add_argument('--image_batch_size', type=int, default=128)
    p.add_argument('--encoder_batch_size', type=int, default=32)
    p.add_argument('--weights_path', type=Path, help='Optional official ResNet18 ImageNet state dict')
    p.add_argument('--image_weight', type=float, default=.25)
    p.add_argument('--allow_partial_images', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if min(a.iterations, a.jobs, a.image_epochs, a.image_batch_size, a.encoder_batch_size) < 1:
        p.error('Budgets and batch sizes must be positive')
    if a.folds < 2 or not 0 < a.gpu_ram_part <= 1 or not 0 <= a.image_weight <= 1:
        p.error('Invalid fold count, GPU fraction or image weight')
    if a.validation_csv and a.no_external_validation:
        p.error('--validation_csv and --no_external_validation conflict')
    return a


def numeric_options(a, seed):
    return dict(seed=seed, iterations=a.iterations, task_type=a.task_type,
                devices=a.devices, gpu_ram_part=a.gpu_ram_part, jobs=a.jobs, resume=a.resume)


def image_options(a, seed, checkpoint):
    return dict(checkpoint_dir=checkpoint, epochs=a.image_epochs,
                batch_size=a.image_batch_size, seed=seed, device=a.image_device,
                jobs=a.jobs, resume=a.resume)


def image_predict(state, frame, embeddings, rows, a):
    return predict_image(state, frame, embeddings, indices=rows,
                         device=a.image_device, batch_size=a.image_batch_size)


def extract(frame, directory, a, encoder_state=None):
    return extract_embeddings(frame, directory, device=a.image_device,
                              batch_size=a.encoder_batch_size, jobs=a.jobs,
                              encoder_state=encoder_state, weights_path=a.weights_path)


def cv(a, source, x, image_frame, image_rows, visual, splits, signature):
    n, m = len(source), len(image_frame)
    numeric_all = np.full((n, 6), np.nan)
    image = np.full((m, 6), np.nan)
    blank_image = np.full((m, 6), np.nan)
    fold_ids = np.zeros(n, int)
    fold_results = []
    for number, (tr, va) in enumerate(splits, 1):
        fold_ids[va] = number
        itr = np.flatnonzero(np.isin(image_rows, tr))
        iva = np.flatnonzero(np.isin(image_rows, va))
        if set(image_frame.iloc[itr].label) != set(range(1, 7)) or not len(iva):
            raise ValueError(f'Fold {number}: image training must contain all stages and assessment images')
        directory = a.output_dir / 'cv' / f'fold_{number}'
        marker = directory / 'complete.json'
        if a.resume and marker.is_file():
            old = json.loads(marker.read_text('utf-8'))
            if old['signature'] != signature:
                raise ValueError('Saved fold signature differs')
            artifact = joblib.load(directory / 'predictions.joblib')
            if artifact['sample_ids'] != source.iloc[va].sample_id.astype(str).tolist():
                raise ValueError('Saved fold row alignment differs')
            if not np.array_equal(artifact['image_rows'], iva):
                raise ValueError('Saved fold image alignment differs')
            numeric_all[va], image[iva], blank_image[iva] = (
                artifact['numeric'], artifact['image'], artifact['blank_image'])
            fold_results.append(old['metrics'])
            print(f'Reuse complete fold {number}/{len(splits)}', flush=True)
            continue
        print(f'\nFold {number}/{len(splits)}: numeric train={len(tr)}, image train={len(itr)}, '
              f'assessment images={len(iva)}', flush=True)
        models = fit_numeric(x, source.label, tr, directory / 'numeric',
                             f'{signature}:fold{number}', **numeric_options(a, a.seed + number))
        pn = predict_numeric(models, x.iloc[va], a.jobs)
        numeric_all[va] = pn
        # Release CatBoost GPU work before the image head. No simultaneous training.
        del models
        head = fit_image(image_frame, visual['embeddings'], itr, val_indices=iva,
                         **image_options(a, a.seed + number, directory / 'head_checkpoint'))
        dump(directory / 'image_head.joblib', head)
        pi = image_predict(head, image_frame, visual['embeddings'], iva, a)
        pb = image_predict(head, image_frame, visual['black'], iva, a)
        image[iva], blank_image[iva] = pi, pb
        pnum = numeric_all[image_rows[iva]]
        fused, blank = blend(pnum, pi, a.image_weight), blend(pnum, pb, a.image_weight)
        assessment = image_frame.iloc[iva].reset_index(drop=True)
        metrics = {name: report(assessment, prob) for name, prob in
                   [('numeric', pnum), ('image', pi), ('fused', fused), ('blank_fused', blank)]}
        fold_results.append(metrics)
        dump(directory / 'predictions.joblib', {
            'sample_ids': source.iloc[va].sample_id.astype(str).tolist(),
            'image_rows': iva, 'numeric': pn, 'image': pi, 'blank_image': pb})
        write_csv(directory / 'fused_predictions.csv', predictions(assessment, fused))
        write_json(marker, {'signature': signature, 'fold': number, 'metrics': metrics})
        print(f"Fold {number}: numeric={metrics['numeric']['macro_f1']:.6f}, "
              f"fused={metrics['fused']['macro_f1']:.6f}, "
              f"blank={metrics['blank_fused']['macro_f1']:.6f}", flush=True)
    matched_numeric = numeric_all[image_rows]
    fused = blend(matched_numeric, image, a.image_weight)
    blank = blend(matched_numeric, blank_image, a.image_weight)
    controls = {'numeric': matched_numeric, 'image': image, 'fused': fused, 'blank_fused': blank}
    results, comparison = {}, []
    for name, p in controls.items():
        results[name] = report(image_frame, p)
        results[name]['fold_macro_f1'] = [v[name]['macro_f1'] for v in fold_results]
        comparison.append({'model': name, 'rows': m, 'experiments': int(image_frame.experiment_id.nunique()),
                           'pooled_macro_f1': results[name]['macro_f1'],
                           'fold_mean_macro_f1': float(np.mean(results[name]['fold_macro_f1'])),
                           'fold_std_macro_f1': float(np.std(results[name]['fold_macro_f1']))})
        write_csv(a.output_dir / 'cv' / f'{name}_oof.csv', predictions(image_frame, p, fold_ids[image_rows]))
    write_csv(a.output_dir / 'cv' / 'numeric_oof_all_rows.csv', predictions(source, numeric_all, fold_ids))
    write_json(a.output_dir / 'cv' / 'numeric_all_rows_report.json', report(source, numeric_all))
    results.update(signature=signature, scope='Train-only experiment-group development OOF; readable-image cohort',
                   source_rows=n, image_rows=m, partial_images=m != n, original_validation_used_for_fit=False,
                   image_weight=a.image_weight,
                   image_blanking_delta=results['fused']['macro_f1'] - results['blank_fused']['macro_f1'],
                   target_093_exceeded=results['fused']['macro_f1'] > .93,
                   selection_note='Method/weight previously selected using Train development OOF; external Validation is separate.')
    write_csv(a.output_dir / 'cv' / 'comparison.csv', pd.DataFrame(comparison))
    write_json(a.output_dir / 'cv' / 'report.json', results)
    write_csv(a.output_dir / 'cv' / 'errors_by_experiment.csv', error_summary(image_frame, fused))
    print('\nMatched readable-image OOF comparison:', flush=True)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    return results


def save_bundle(directory, a, x, visual, head, signature):
    dump(directory / 'image_head.joblib', head)
    dump(directory / 'encoder.joblib', visual['encoder_state'])
    files = ['router.cbm', 'pair23.cbm', 'late456.cbm', 'image_head.joblib', 'encoder.joblib']
    metadata = {'format_version': 21, 'signature': signature, 'labels': list(range(1, 7)),
                'feature_names': list(x), 'categorical_features': F.CATEGORICAL_FEATURES,
                'image_weight': a.image_weight, 'numeric_weight': 1 - a.image_weight,
                'fit_scope': 'Train only; external Validation never fitted',
                'files': {name: file_hash(directory / name) for name in files}}
    write_json(directory / 'bundle.json', metadata)


def final_fit(a, source, x, image_frame, visual, signature):
    directory = a.output_dir / 'final_model'
    directory.mkdir(parents=True, exist_ok=True)
    models = fit_numeric(x, source.label, np.arange(len(source)), directory, f'{signature}:final',
                         **numeric_options(a, a.seed))
    head_file, bundle_file = directory / 'image_head.joblib', directory / 'bundle.json'
    if a.resume and bundle_file.is_file():
        from model import load_bundle
        bundle = load_bundle(directory)
        if bundle['metadata']['signature'] != signature:
            raise ValueError('Saved final bundle differs')
        head = bundle['head']
        print('Reuse complete final image model', flush=True)
    else:
        head = fit_image(image_frame, visual['embeddings'], np.arange(len(image_frame)),
                         **image_options(a, a.seed, directory / 'head_checkpoint'))
        save_bundle(directory, a, x, visual, head, signature)
    return models, head


def external_validation(a, path, source, models, head, visual, signature):
    validation = D.read_csv(path, labelled=True)
    overlap = set(source.experiment_id.astype(str)) & set(validation.experiment_id.astype(str))
    sample_overlap = set(source.sample_id.astype(str)) & set(validation.sample_id.astype(str))
    if overlap or sample_overlap:
        raise ValueError('External Validation overlaps Train experiments/sample IDs')
    # History includes only current/past rows of each external experiment.
    vx = F.build_features(validation)
    vf, vr, audit = D.inspect_images(validation, path, a.data_dir, a.image_root, a.allow_partial_images)
    out = a.output_dir / 'external_validation'
    write_json(out / 'image_audit.json', audit)
    pn = predict_numeric(models, vx, a.jobs)
    fused_all = pn.copy()  # Explicit partial mode: missing current images use numeric-only fallback.
    details = {'scope': 'Original external Validation; final Train-only model, fixed weight/epochs',
               'signature': signature, 'validation_csv_sha256': file_hash(path),
               'source_rows': len(validation), 'readable_image_rows': len(vf),
               'numeric_fallback_rows': len(validation) - len(vf),
               'numeric_all_rows': report(validation, pn), 'training_or_tuning_used_validation': False}
    if len(vf):
        vv = extract(vf, a.output_dir / 'cache' / 'external_validation', a, visual['encoder_state'])
        pi = image_predict(head, vf, vv['embeddings'], None, a)
        pb = image_predict(head, vf, vv['black'], None, a)
        pf, pblank = blend(pn[vr], pi, a.image_weight), blend(pn[vr], pb, a.image_weight)
        fused_all[vr] = pf
        details['matched_readable_rows'] = {name: report(vf, p) for name, p in
            [('numeric', pn[vr]), ('image', pi), ('fused', pf), ('blank_fused', pblank)]}
        details['image_blanking_delta'] = (details['matched_readable_rows']['fused']['macro_f1'] -
                                             details['matched_readable_rows']['blank_fused']['macro_f1'])
        write_csv(out / 'blank_fused_readable.csv', predictions(vf, pblank))
    details['fused_all_rows'] = report(validation, fused_all)
    write_csv(out / 'predictions.csv', predictions(validation, fused_all))
    write_csv(out / 'numeric_predictions.csv', predictions(validation, pn))
    write_csv(out / 'errors_by_experiment.csv', error_summary(validation, fused_all))
    write_json(out / 'report.json', details)
    print(f"External Validation macro F1={details['fused_all_rows']['macro_f1']:.6f}; "
          f"numeric={details['numeric_all_rows']['macro_f1']:.6f}; fallback={len(validation)-len(vf)}", flush=True)


def execute(a):
    train_csv = D.locate_csv(a.data_dir, 'train', a.train_csv)
    source = D.read_csv(train_csv, labelled=True)
    if set(source.label) != set(range(1, 7)):
        raise ValueError('Train must contain all six stages')
    val_csv = None if a.no_external_validation else D.locate_csv(
        a.data_dir, 'validation', a.validation_csv, required=bool(a.validation_csv))
    if val_csv:
        val = D.read_csv(val_csv, labelled=True)
        if (set(val.experiment_id.astype(str)) & set(source.experiment_id.astype(str)) or
                set(val.sample_id.astype(str)) & set(source.sample_id.astype(str))):
            raise ValueError('Validation must be disjoint from Train by experiment AND sample ID')
        del val
    print(f'Train: {len(source)} rows / {source.experiment_id.nunique()} experiments; '
          f'external Validation: {val_csv or "not found/disabled"}', flush=True)
    x = F.build_features(source)  # ALL numeric histories, before selecting readable image rows.
    image_frame, image_rows, audit = D.inspect_images(
        source, train_csv, a.data_dir, a.image_root, a.allow_partial_images)
    if not len(image_frame):
        raise ValueError('No readable Train images')
    manifest = D.make_splits(source, a.folds, a.seed, a.split_manifest)
    splits = D.row_splits(source, manifest)
    settings = {name: getattr(a, name) for name in [
        'folds', 'seed', 'iterations', 'task_type', 'devices', 'gpu_ram_part', 'jobs',
        'image_device', 'image_epochs', 'image_batch_size', 'encoder_batch_size',
        'image_weight', 'allow_partial_images']}
    config = {'format_version': 21, 'settings': settings, 'train_csv_sha256': file_hash(train_csv),
              'images': audit, 'manifest': manifest, 'feature_names': list(x),
              'weights_sha256': file_hash(a.weights_path) if a.weights_path else 'official ResNet18 IMAGENET1K_V1',
              'code': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'versions': {name: version(name) for name in
                           ['catboost', 'torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow', 'joblib']},
              'python': platform.python_version(),
              'fit_scope': 'Train only, no original Validation rows'}
    signature = digest(config)
    config_path = a.output_dir / 'run_config.json'
    if config_path.is_file():
        old = json.loads(config_path.read_text('utf-8'))
        if not a.resume or old['signature'] != signature:
            raise ValueError('Existing run differs or --resume missing; use a new output_dir')
    elif any(p.name != '.run.lock' and p.suffix != '.log' for p in a.output_dir.iterdir()):
        raise ValueError('Nonempty output_dir without a run_config; use a new directory')
    write_json(config_path, {'signature': signature, **config})
    write_json(a.output_dir / 'image_audit.json', audit)
    write_json(a.output_dir / 'split_manifest.json', manifest)
    for path in Path(__file__).parent.glob('*.py'):
        target = a.output_dir / 'source_snapshot' / path.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
    print(f'Readable images: {len(image_frame)}/{len(source)}. Image rows are never copied.', flush=True)
    if a.image_epochs != 12 or a.iterations != 600 or a.image_weight != .25:
        print('Custom budget/weight differs from the measured best configuration.', flush=True)
    visual = extract(image_frame, a.output_dir / 'cache' / 'train', a)
    if a.stage in ['all', 'cv']:
        cv(a, source, x, image_frame, image_rows, visual, splits, signature)
    if a.stage in ['all', 'final']:
        models, head = final_fit(a, source, x, image_frame, visual, signature)
        if val_csv:
            external_validation(a, val_csv, source, models, head, visual, signature)
        else:
            print('No external Validation evaluated. Train OOF is reported separately.', flush=True)
    write_json(a.output_dir / 'status.json', {'state': 'complete', 'stage': a.stage,
                                             'signature': signature, 'external_validation': str(val_csv) if val_csv else None})
    print(f'Done: {a.output_dir.resolve()}', flush=True)


def main():
    a = options()
    with run_lock(a.output_dir):
        try:
            execute(a)
        except BaseException as error:
            write_json(a.output_dir / 'last_failure.json', {'error': repr(error), 'stage': a.stage})
            raise


if __name__ == '__main__':
    main()
