"""Predict all six stages. Output columns/sample order follow sample_submission when given."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
from features import build_features
from image_features import extract_embeddings
from image_model import predict_image
from model import load_bundle
from numeric import predict_numeric
from reporting import predictions
from utils import blend, run_lock, write_csv, write_json


def make_submission(frame, labels, sample_submission=None, target_column='label'):
    table = pd.DataFrame({'sample_id': frame.sample_id.astype(str).to_numpy(),
                          target_column: np.asarray(labels, int)})
    if sample_submission:
        sample = pd.read_csv(sample_submission, dtype={'sample_id': 'string'})
        if 'sample_id' not in sample or not sample.sample_id.is_unique or sample.sample_id.isna().any():
            raise ValueError('sample_submission requires unique, nonmissing sample_id')
        targets = [c for c in sample if c != 'sample_id']
        if len(targets) != 1:
            raise ValueError('sample_submission must have sample_id and one target column')
        if set(sample.sample_id.astype(str)) != set(table.sample_id):
            raise ValueError('sample_submission IDs differ from Test CSV')
        target = targets[0]
        values = table.set_index('sample_id')[target_column]
        sample[target] = sample.sample_id.astype(str).map(values).astype(int)
        return sample
    return table


def execute(a):
    csv = D.locate_csv(a.data_dir, 'test', a.test_csv)
    frame = D.read_csv(csv, labelled=False)
    bundle = load_bundle(a.model_dir)  # Only load artifacts you trained/trust (joblib).
    x = build_features(frame)  # Full Test history; no fit, and no future timestamps.
    if list(x) != bundle['metadata']['feature_names']:
        raise ValueError('Inference feature order differs from training')
    pn = predict_numeric(bundle['numeric'], x, a.jobs)
    image_frame, rows, audit = D.inspect_images(frame, csv, a.data_dir,
                                               a.image_root, a.allow_partial_images)
    p = pn.copy()
    if len(image_frame):
        visual = extract_embeddings(image_frame, a.cache_dir, device=a.image_device,
                                    batch_size=a.encoder_batch_size, jobs=a.jobs,
                                    encoder_state=bundle['encoder_state'])
        pi = predict_image(bundle['head'], image_frame, visual['embeddings'],
                           device=a.image_device, batch_size=a.image_batch_size)
        p[rows] = blend(pn[rows], pi, bundle['metadata']['image_weight'])
    pred = p.argmax(1) + 1
    submission = make_submission(frame, pred, a.sample_submission, a.target_column)
    write_csv(a.output, submission)
    write_csv(a.output.with_name(a.output.stem + '_probabilities.csv'), predictions(frame, p))
    write_json(a.output.with_name(a.output.stem + '_audit.json'), {
        'model_signature': bundle['metadata']['signature'], 'rows': len(frame),
        'image_rows': len(rows), 'numeric_fallback_rows': len(frame) - len(rows),
        'image_weight': bundle['metadata']['image_weight'], 'image_audit': audit,
        'label_counts': {str(i): int((pred == i).sum()) for i in range(1, 7)}})
    print(f'Saved {a.output.resolve()} ({len(frame)} rows; numeric fallback={len(frame)-len(rows)})', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', type=Path, default=Path('.'))
    p.add_argument('--test_csv', type=Path)
    p.add_argument('--image_root', type=Path)
    p.add_argument('--model_dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sample_submission', type=Path)
    p.add_argument('--target_column', default='label')
    p.add_argument('--cache_dir', type=Path)
    p.add_argument('--image_device', default='cpu')
    p.add_argument('--encoder_batch_size', type=int, default=32)
    p.add_argument('--image_batch_size', type=int, default=128)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--allow_partial_images', action='store_true')
    a = p.parse_args()
    if min(a.jobs, a.encoder_batch_size, a.image_batch_size) < 1:
        p.error('Jobs/batch sizes must be positive')
    if a.target_column == 'sample_id':
        p.error('Target column cannot be sample_id')
    if a.cache_dir is None:
        a.cache_dir = a.output.parent / ('cache_' + a.output.stem)
    with run_lock(a.cache_dir):
        execute(a)


if __name__ == '__main__':
    main()
