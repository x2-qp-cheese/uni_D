"""Atomic artifacts, process locking and stable run identities."""
import contextlib
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                              default=str), encoding='utf-8')
    tmp.replace(path)


def write_csv(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    table.to_csv(tmp, index=False)
    tmp.replace(path)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(value, tmp, compress=3)
    tmp.replace(path)


def normalized(values, width=6):
    p = np.asarray(values, dtype=float)
    if p.ndim != 2 or p.shape[1] != width or not np.isfinite(p).all():
        raise ValueError(f'Invalid probability matrix: {p.shape}')
    if (p < -1e-7).any() or (p.sum(1) <= 0).any():
        raise ValueError('Invalid probability values')
    p = np.maximum(p, 0)
    return p / p.sum(1, keepdims=True)


def combine(router, pair, late):
    router, pair, late = normalized(router, 3), normalized(pair, 2), normalized(late, 3)
    return normalized(np.column_stack([router[:, 0],
                                      router[:, 1:2] * pair,
                                      router[:, 2:3] * late]))


def blend(numeric, image, weight=.25):
    if not 0 <= weight <= 1:
        raise ValueError('image_weight must be 0..1')
    return normalized((1 - weight) * normalized(numeric) + weight * normalized(image))


@contextlib.contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.run.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Another process is using output_dir') from error
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError('Another process is using output_dir') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
