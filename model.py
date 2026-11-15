"""Load a complete inference bundle without downloading pretrained weights."""
import json
from pathlib import Path

import joblib

from features import CATEGORICAL_FEATURES
from numeric import load_numeric
from utils import file_hash


def load_bundle(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'bundle.json').read_text('utf-8'))
    if meta.get('format_version') != 21 or meta.get('labels') != list(range(1, 7)):
        raise ValueError('Unsupported model bundle format or class order')
    if meta['categorical_features'] != CATEGORICAL_FEATURES:
        raise ValueError('Categorical feature schema differs')
    required = {'router.cbm', 'pair23.cbm', 'late456.cbm', 'image_head.joblib', 'encoder.joblib'}
    if set(meta['files']) != required:
        raise ValueError('Incomplete model bundle')
    for name, sha in meta['files'].items():
        if file_hash(directory / name) != sha:
            raise ValueError(f'Model checksum mismatch: {name}')
    return {'metadata': meta, 'numeric': load_numeric(directory),
            'head': joblib.load(directory / 'image_head.joblib'),
            'encoder_state': joblib.load(directory / 'encoder.joblib')}
