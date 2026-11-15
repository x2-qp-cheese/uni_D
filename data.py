"""Portable CSV/image loading and strictly grouped Train-only cross-validation.

Image inspection reads the originals in place. It never copies images or fits
anything. A partially available image cohort keeps positional links to the
complete numeric table, so sensor histories can still use every Train row.
"""
from __future__ import annotations

from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

import features as F


def read_csv(path, labelled=False):
    """Read the historical sensor schema and require portable image references."""
    frame = F.read_frame(Path(path), labelled=labelled).reset_index(drop=True)
    if 'image_path' not in frame:
        raise ValueError('Missing columns: [\'image_path\']')
    for column in ('sample_id', 'experiment_id', 'image_path'):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'{column} must be non-null and nonempty')
    # Match the feature builder's policy: indu_t may be absent for a method;
    # other sensor columns must contain finite numeric observations.
    for column in F.SENSORS:
        values = pd.to_numeric(frame[column], errors='raise').to_numpy(float)
        if column != 'indu_t' and not np.isfinite(values).all():
            raise ValueError(f'Unexpected missing/nonfinite sensor: {column}')
    for column, allowed in F.CATEGORY_VALUES.items():
        values = pd.to_numeric(frame[column], errors='raise').to_numpy(float)
        valid = np.isclose(values[:, None], np.asarray(allowed, float)[None, :],
                           rtol=0.0, atol=1e-6).any(axis=1)
        if not np.isfinite(values).all() or not valid.all():
            raise ValueError(f'Unexpected categorical code in {column}: {values[~valid][:5]}')
    if labelled:
        frame['label'] = pd.to_numeric(frame['label'], errors='raise').astype(np.int64)
    return frame


def _case_insensitive_file(base, relative):
    """Accept Train/Validation/Test capitalization on case-sensitive servers."""
    direct = base / relative
    if direct.is_file():
        return direct.resolve()
    current = base
    for part in Path(relative).parts:
        if not current.is_dir():
            return None
        exact = current / part
        if exact.exists():
            current = exact
            continue
        matches = [entry for entry in current.iterdir() if entry.name.casefold() == part.casefold()]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f'Ambiguous case-insensitive dataset path: {current / part}')
        current = matches[0]
    return current.resolve() if current.is_file() else None


def locate_csv(data_dir, split, explicit=None, required=True):
    """Locate a conventional split CSV, returning an absolute Path or None."""
    root = Path(data_dir).expanduser().resolve()
    if explicit is not None:
        supplied = Path(explicit).expanduser()
        candidates = [supplied] if supplied.is_absolute() else [supplied, root / supplied]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        if required:
            raise FileNotFoundError(f'Explicit {split} CSV does not exist: {explicit}')
        return None
    name = str(split).casefold()
    if name in ('val', 'valid', 'validation'):
        folders = ('validation', 'val', 'valid')
        files = ('validation.csv', 'val.csv', 'valid.csv')
    elif name in ('train', 'test'):
        folders, files = (name,), (f'{name}.csv',)
    else:
        raise ValueError(f'Unknown split: {split}; use train, validation, or test')
    relatives = [str(Path(folder) / file) for folder in folders for file in files]
    relatives += list(files)
    for relative in relatives:
        match = _case_insensitive_file(root, relative)
        if match is not None:
            return match
    if required:
        expected = ', '.join(str(root / item) for item in relatives)
        raise FileNotFoundError(f'Cannot locate {split} CSV; checked: {expected}')
    return None


def _image_candidates(value, csv_parent, data_dir, image_root):
    normalized = str(value).replace('\\', '/')
    windows = PureWindowsPath(normalized)
    original = Path(normalized).expanduser()
    absolute_reference = normalized.startswith('/') or original.is_absolute() or windows.is_absolute()
    # Strip an old machine's drive/anchor before joining a portable root.
    relative = normalized
    if windows.drive:
        relative = relative[len(windows.drive):]
    relative = relative.lstrip('/')
    relative_path = Path(relative)
    basename = relative_path.name
    candidates = []
    if image_root is not None:
        candidates.extend((image_root / relative_path, image_root / basename))
    candidates.extend((csv_parent / relative_path, csv_parent / 'images' / basename,
                       data_dir / relative_path, data_dir / 'images' / basename))
    if absolute_reference:
        candidates.append(original)
    seen = set()
    for candidate in candidates:
        # resolve(strict=False) also normalizes '..' without requiring existence.
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            yield resolved


def inspect_images(frame, csv_path, data_dir, image_root=None, allow_partial=False):
    """Decode originals and fingerprint actual bytes; retain source positions.

    Missing and zero-byte files may be dropped only with allow_partial=True.
    A nonempty corrupt image is always an error. Explicit image_root candidates
    precede stale original absolute paths, which makes copied datasets portable.
    Empty readable results are allowed in partial mode for numeric-only Test
    fallback; the training entry point decides whether that cohort is usable.
    """
    for column in ('sample_id', 'image_path'):
        if column not in frame:
            raise ValueError(f'Missing image inspection column: {column}')
    csv_parent = Path(csv_path).expanduser().resolve().parent
    root = Path(data_dir).expanduser().resolve()
    override = Path(image_root).expanduser().resolve() if image_root is not None else None
    positions, paths = [], []
    counts = {'missing': 0, 'zero': 0, 'corrupt': 0}
    examples = {key: [] for key in counts}
    shapes = Counter()
    fingerprint = hashlib.sha256()
    for position, (sample_id, value) in enumerate(zip(frame['sample_id'], frame['image_path'])):
        path = next((candidate for candidate in _image_candidates(value, csv_parent, root, override)
                     if candidate.is_file()), None)
        if path is None:
            counts['missing'] += 1
            if len(examples['missing']) < 5:
                examples['missing'].append({'sample_id': str(sample_id), 'image_path': str(value)})
            continue
        try:
            payload = path.read_bytes()
            if not payload:
                counts['zero'] += 1
                if len(examples['zero']) < 5:
                    examples['zero'].append({'sample_id': str(sample_id), 'image_path': str(path)})
                continue
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                image.load()
                width, height, mode = image.width, image.height, image.mode
                if width < 1 or height < 1:
                    raise ValueError('Image has nonpositive dimensions')
        except Exception as error:
            counts['corrupt'] += 1
            if len(examples['corrupt']) < 5:
                examples['corrupt'].append({'sample_id': str(sample_id), 'image_path': str(path),
                                            'error': f'{type(error).__name__}: {error}'})
            continue
        content_hash = hashlib.sha256(payload).hexdigest()
        record = json.dumps([str(sample_id), len(payload), content_hash],
                            ensure_ascii=False, separators=(',', ':'))
        fingerprint.update(record.encode('utf-8') + b'\n')
        shapes[f'{width}x{height}:{mode}'] += 1
        positions.append(position)
        paths.append(str(path))
    audit = {'rows': int(len(frame)), 'total': int(len(frame)), 'readable': len(positions),
             **counts, 'dropped': int(len(frame) - len(positions)),
             'allow_partial': bool(allow_partial), 'source_csv': str(Path(csv_path).resolve()),
             'image_root': str(override) if override is not None else None,
             'shape_counts': dict(sorted(shapes.items())),
             'image_fingerprint': fingerprint.hexdigest(),
             'image_fingerprint_sha256': fingerprint.hexdigest(),
             'image_fingerprint_method': 'ordered JSON [sample_id,byte_size,SHA256(actual bytes)] + LF',
             'examples': examples}
    summary = (f"total={len(frame)}, readable={len(positions)}, missing={counts['missing']}, "
               f"zero={counts['zero']}, corrupt={counts['corrupt']}")
    if counts['corrupt']:
        raise ValueError(f'Nonempty image read/decode failures ({summary}); '
                         f'first five: {json.dumps(examples["corrupt"], ensure_ascii=False)}')
    if not allow_partial and (counts['missing'] or counts['zero']):
        raise FileNotFoundError(f'Incomplete images ({summary}); first five per category: '
                                f'{json.dumps(examples, ensure_ascii=False)}. '
                                'Use allow_partial only to deliberately select readable rows.')
    source_positions = np.asarray(positions, dtype=np.int64)
    image_frame = frame.iloc[source_positions].copy().reset_index(drop=True)
    image_frame['image_path'] = pd.Series(paths, dtype=object)
    return image_frame, source_positions, audit


def _split_frame(frame):
    required = {'experiment_id', 'label'}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError('Grouped splitting needs a nonempty frame with experiment_id and label')
    if frame['experiment_id'].isna().any() or frame['experiment_id'].astype(str).str.strip().eq('').any():
        raise ValueError('experiment_id must be non-null and nonempty')
    labels = pd.to_numeric(frame['label'], errors='raise').to_numpy(float)
    if not np.isin(labels, np.arange(1, 7)).all():
        raise ValueError('Grouped splitting requires integer labels 1..6')
    return frame['experiment_id'].astype(str).to_numpy(), labels.astype(np.int64)


def _records(manifest):
    records = manifest.get('folds') if isinstance(manifest, dict) else manifest
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError('Split manifest must contain at least two fold records')
    return records


def _groups(record, index):
    if not isinstance(record, dict):
        raise ValueError(f'Manifest fold {index} is not an object')
    training = record.get('training_experiments', record.get('outer_training'))
    validation = record.get('validation_experiments', record.get('outer_assessment'))
    if not isinstance(training, (list, tuple)) or not isinstance(validation, (list, tuple)):
        raise ValueError('Manifest needs training_experiments and validation_experiments lists')
    tr, va = [str(value) for value in training], [str(value) for value in validation]
    if len(set(tr)) != len(tr) or len(set(va)) != len(va):
        raise ValueError(f'Manifest fold {index} repeats an experiment within a group list')
    fold = record.get('fold', index)
    if isinstance(fold, bool) or not isinstance(fold, (int, np.integer)) or int(fold) != index:
        raise ValueError('Manifest fold numbers must be consecutive JSON integers starting at 1')
    return set(tr), set(va)


def row_splits(frame, manifest):
    """Validate exhaustive experiment folds and return positional index arrays."""
    experiments, labels = _split_frame(frame)
    available = set(experiments)
    coverage = np.zeros(len(frame), dtype=np.int64)
    splits = []
    for index, record in enumerate(_records(manifest), 1):
        training, validation = _groups(record, index)
        if not training or not validation or training & validation:
            raise ValueError(f'Manifest fold {index} has empty or overlapping experiment groups')
        if training | validation != available:
            missing = sorted(available - (training | validation))[:5]
            unknown = sorted((training | validation) - available)[:5]
            raise ValueError(f'Manifest fold {index} does not exactly cover current experiments; '
                             f'missing={missing}, unknown={unknown}')
        tr = np.flatnonzero(np.isin(experiments, sorted(training)))
        va = np.flatnonzero(np.isin(experiments, sorted(validation)))
        if not len(tr) or not len(va) or np.intersect1d(tr, va).size:
            raise ValueError(f'Manifest fold {index} has empty or overlapping row sets')
        if set(labels[tr]) != set(range(1, 7)):
            raise ValueError(f'Manifest fold {index} training set must contain all six stages '
                             'for the router and stage 2/3 and 4/5/6 experts')
        coverage[va] += 1
        splits.append((tr, va))
    if not np.all(coverage == 1):
        raise ValueError('Every Train row/experiment must be held out exactly once')
    return splits


def _canonical_manifest(frame, value, method=None):
    splits = row_splits(frame, value)
    experiments = frame['experiment_id'].astype(str).to_numpy()
    metadata = {key: value for key, value in value.items() if key != 'folds'} if isinstance(value, dict) else {}
    if method is not None:
        metadata['method'] = method
    return {**metadata, 'folds': [
        {'fold': int(index), 'training_experiments': sorted(set(experiments[tr])),
         'validation_experiments': sorted(set(experiments[va])),
         'training_rows': int(len(tr)), 'validation_rows': int(len(va))}
        for index, (tr, va) in enumerate(splits, 1)]}


def make_splits(frame, folds=3, seed=42, manifest_path=None):
    """Reuse exact historical groups, or run its unchanged 16-candidate search.

    The supplied frame must be original Train only. Experiment IDs are never
    prefixed or rewritten: an overlap with any external split remains visible
    to the caller. All six original stages determine fold balance.
    """
    experiments, labels = _split_frame(frame)
    if isinstance(folds, bool) or not isinstance(folds, (int, np.integer)) or folds < 2:
        raise ValueError('folds must be an integer >= 2')
    folds, seed = int(folds), int(seed)
    if len(set(experiments)) < folds:
        raise ValueError('Fewer experiments than folds')
    if manifest_path is not None:
        value = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
        if len(_records(value)) != folds:
            raise ValueError(f'Manifest has {len(_records(value))} folds; set folds to that value')
        result = _canonical_manifest(frame, value, 'explicit imported experiment manifest')
        result['manifest_source'] = str(Path(manifest_path).resolve())
        return result
    reference = Path(__file__).with_name('reference_train_splits.json')
    if reference.is_file():
        value = json.loads(reference.read_text(encoding='utf-8-sig'))
        records = _records(value)
        reference_experiments = set()
        for index, record in enumerate(records, 1):
            tr_groups, va_groups = _groups(record, index)
            reference_experiments.update(tr_groups | va_groups)
        if len(records) == folds and reference_experiments == set(experiments):
            result = _canonical_manifest(frame, value, 'bundled historical Train experiment manifest')
            result['manifest_source'] = reference.name
            return result
    needed = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
    if any(column not in frame for column in needed):
        raise ValueError(f'Automatic grouped splitting needs condition columns: {needed}')
    condition = frame[needed].astype(str).agg('|'.join, axis=1)
    values, codes = np.unique(condition, return_inverse=True)
    totals = np.bincount(labels - 1, minlength=6)
    condition_totals = np.bincount(codes)
    best = None
    for offset in range(16):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + offset)
        splits = list(splitter.split(np.zeros(len(frame)), labels, experiments))
        if any(set(labels[tr]) != set(range(1, 7)) for tr, _ in splits):
            continue
        shares = [np.bincount(labels[va] - 1, minlength=6) / np.maximum(totals, 1)
                  for _, va in splits]
        condition_shares = [np.bincount(codes[va], minlength=len(values)) / condition_totals
                            for _, va in splits]
        objective = float(np.mean((np.asarray(shares) - 1 / folds) ** 2)
                          + .6 * np.mean((np.asarray(condition_shares) - 1 / folds) ** 2))
        if best is None or objective < best[0]:
            best = objective, seed + offset, splits
    if best is None:
        raise ValueError('No split with all six original stages in training; reduce folds')
    metadata = {'method': 'v11-compatible original-six-label stratification',
                'seed': int(best[1]), 'balance_objective': best[0], 'candidates': 16}
    records = [{'fold': index, 'training_experiments': sorted(set(experiments[tr])),
                'validation_experiments': sorted(set(experiments[va]))}
               for index, (tr, va) in enumerate(best[2], 1)]
    return _canonical_manifest(frame, {**metadata, 'folds': records})
