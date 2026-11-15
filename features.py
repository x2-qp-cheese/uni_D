"""Compact77 causal numeric features, computed on ALL rows before image selection.

Every feature for a row is computed from its current numeric values and
earlier rows of the same experiment. No future or cross-experiment values enter.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [1, 2, 3, 4, 5, 6]
SEED = 42
VERSION = 'uni-d-best-fusion-v21.0'

SENSORS = [
    'meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'meas_ch4', 'meas_co',
    'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no',
    'meas_no2', 'meas_so2', 'atmp_t', 'thermal_t', 'bat_cc', 'indu_t',
]
CATEGORY_VALUES = {
    'bat_class': [1, 2, 3],
    'soc_status': [1, 2],
    'tr_method': [1, 2],
    'bat_cc': [4.8, 6.0, 28.0, 39.0, 73.0, 280.0],
}
CATEGORICAL_FEATURES = ['bat_class_cat', 'bat_cc_cat', 'soc_status_cat', 'tr_method_cat']
TEMPORAL = [
    'meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'meas_ch4', 'meas_co',
    'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no',
    'meas_no2', 'meas_so2', 'thermal_t', 'indu_t',
]
WINDOWS_S = (15, 30, 60, 120)


def read_frame(path, labelled=False):
    frame = pd.read_csv(
        path,
        dtype={'sample_id': 'string', 'experiment_id': 'string'},
        float_precision='round_trip',
    )
    validate_frame(frame)
    if labelled:
        if 'label' not in frame:
            raise ValueError('Labelled split has no label column')
        y = pd.to_numeric(frame.label, errors='raise')
        if not y.isin(LABELS).all():
            raise ValueError('Labels must be integers 1..6')
    return frame


def validate_frame(frame):
    required = ['sample_id', 'experiment_id', 'relative_time_s']
    missing = set(required + SENSORS + list(CATEGORY_VALUES)) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('Empty input CSV')
    if frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError('sample_id must be non-null and unique')
    if frame.experiment_id.isna().any():
        raise ValueError('experiment_id cannot be missing')
    times = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('relative_time_s must be finite and nonnegative')
    if pd.DataFrame({'experiment': frame.experiment_id, 'time': times}).duplicated().any():
        raise ValueError('Duplicate timestamps within an experiment are ambiguous')


def resolve_paths(frame, roots):
    paths = []
    for value in frame.image_path:
        path = Path(str(value))
        candidates = [path] if path.is_absolute() else [Path(root) / path for root in roots]
        match = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if match is None:
            raise FileNotFoundError(f'Image not found: {value}; roots={roots}')
        paths.append(match)
    return paths


def resolve_image_paths(frame, roots):
    return pd.Series(resolve_paths(frame, roots), index=frame.index)


def temporal_features(times, values, window_s):
    """Return statistics over (t-window,t] and change from the last row <=t-window."""
    count = len(times)
    valid = np.isfinite(values)
    safe = np.where(valid, values, 0.0)
    sums = np.r_[0.0, np.cumsum(safe)]
    squares = np.r_[0.0, np.cumsum(safe * safe)]
    counts = np.r_[0, np.cumsum(valid)]
    left = np.searchsorted(times, times - window_s, side='right')
    end = np.arange(count) + 1
    window_count = counts[end] - counts[left]
    mean = np.divide(
        sums[end] - sums[left], window_count,
        out=np.full(count, np.nan), where=window_count > 0,
    )
    second_moment = np.divide(
        squares[end] - squares[left], window_count,
        out=np.full(count, np.nan), where=window_count > 0,
    )
    std = np.sqrt(np.maximum(0.0, second_moment - mean * mean))
    maximum = np.full(count, np.nan)
    minimum = np.full(count, np.nan)
    max_queue, min_queue = deque(), deque()
    for index in range(count):
        while max_queue and max_queue[0] < left[index]:
            max_queue.popleft()
        while min_queue and min_queue[0] < left[index]:
            min_queue.popleft()
        if valid[index]:
            while max_queue and values[max_queue[-1]] <= values[index]:
                max_queue.pop()
            while min_queue and values[min_queue[-1]] >= values[index]:
                min_queue.pop()
            max_queue.append(index)
            min_queue.append(index)
        if max_queue:
            maximum[index] = values[max_queue[0]]
            minimum[index] = values[min_queue[0]]
    lag = left - 1
    delta = np.full(count, np.nan)
    slope = np.full(count, np.nan)
    usable = np.flatnonzero((lag >= 0) & valid)
    usable = usable[valid[lag[usable]]]
    delta[usable] = values[usable] - values[lag[usable]]
    elapsed = times[usable] - times[lag[usable]]
    positive_elapsed = elapsed > 0
    slope[usable[positive_elapsed]] = (
        delta[usable[positive_elapsed]] / elapsed[positive_elapsed]
    )
    return {
        'mean': mean, 'std': std, 'min': minimum, 'max': maximum,
        'delta': delta, 'slope': slope,
    }


def expanding_features(times, values):
    """Causal features relative to the first valid observation and all past rows."""
    count = len(times)
    delta = np.full(count, np.nan)
    slope = np.full(count, np.nan)
    running_min = np.full(count, np.nan)
    running_max = np.full(count, np.nan)
    first_index = None
    current_min = current_max = np.nan
    for index in range(count):
        if not np.isfinite(values[index]):
            continue
        if first_index is None:
            first_index = index
            current_min = current_max = values[index]
        current_min = min(current_min, values[index])
        current_max = max(current_max, values[index])
        running_min[index] = current_min
        running_max[index] = current_max
        delta[index] = values[index] - values[first_index]
        elapsed = times[index] - times[first_index]
        if elapsed > 0:
            slope[index] = delta[index] / elapsed
    return {
        'from_start_delta': delta,
        'from_start_slope': slope,
        'running_min': running_min,
        'running_max': running_max,
    }


def _category_strings(frame, column, allowed):
    values = pd.to_numeric(frame[column], errors='raise').to_numpy(float)
    allowed_array = np.asarray(allowed, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'Categorical column contains missing values: {column}')
    valid = np.isclose(values[:, None], allowed_array[None, :], rtol=0.0, atol=1e-6).any(axis=1)
    if not valid.all():
        raise ValueError(f'Unexpected categorical code in {column}: {values[~valid][:5]}')
    return np.asarray([format(value, '.12g') for value in values], dtype=object)


def build_features(frame):
    """Exactly the 77-feature compact configuration audited on real Train.

    No label, ID, clock time, image statistics, whole-experiment size, or
    future observation is a feature. All histories restart per experiment.
    """
    validate_frame(frame)
    frame = frame.reset_index(drop=True)
    numeric = frame[SENSORS].apply(pd.to_numeric, errors='raise').astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    for c in SENSORS:
        if c != 'indu_t' and numeric[c].isna().any():
            raise ValueError(f'Unexpected missing sensor: {c}')
    result = {c: numeric[c].to_numpy() for c in SENSORS}
    for source, target in zip(['bat_class','bat_cc','soc_status','tr_method'], CATEGORICAL_FEATURES):
        result[target] = _category_strings(frame, source, CATEGORY_VALUES[source])
    times = frame.relative_time_s.to_numpy(float)
    groups = [np.asarray(i) for i in frame.groupby('experiment_id', sort=False).indices.values()]
    for signal in ['meas_v','meas_t1','meas_t2','meas_p','thermal_t']:
        values = numeric[signal].to_numpy()
        out = {f'{signal}_rate_per_s': np.full(len(frame), np.nan)}
        for window in (15, 60):
            for stat in ('mean','std','slope'):
                out[f'{signal}_{stat}_{window}s'] = np.full(len(frame), np.nan)
        for stat in ('from_start_delta','from_start_slope','running_min','running_max'):
            out[f'{signal}_{stat}'] = np.full(len(frame), np.nan)
        for group in groups:
            group = group[np.argsort(times[group], kind='stable')]
            t, v = times[group], values[group]
            out[f'{signal}_rate_per_s'][group[1:]] = np.diff(v)/np.diff(t)
            for window in (15,60):
                stats = temporal_features(t, v, window)
                for stat in ('mean','std','slope'):
                    out[f'{signal}_{stat}_{window}s'][group] = stats[stat]
            for stat, value in expanding_features(t, v).items():
                out[f'{signal}_{stat}'][group] = value
        result.update(out)
    table = pd.DataFrame(result)
    for c in CATEGORICAL_FEATURES:
        table[c] = table[c].astype(str)
    nums = [c for c in table if c not in CATEGORICAL_FEATURES]
    table[nums] = table[nums].replace([np.inf,-np.inf],np.nan)
    if table.shape[1] != 77:
        raise AssertionError('Compact schema must have 77 features')
    return table



def sample_weights(labels, alpha=.45, source_weight=None):
    labels=np.asarray(labels,int)
    base=np.ones(len(labels)) if source_weight is None else np.asarray(source_weight,float)
    counts=np.bincount(labels,weights=base,minlength=7)
    present=np.flatnonzero(counts>0)
    factors=np.zeros(7)
    factors[present]=(base.sum()/(len(present)*counts[present]))**alpha
    weights=base*factors[labels]
    return weights/weights.mean()
