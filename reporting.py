"""Matched-row reports. Conditional expert scores are labelled separately."""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from utils import normalized


def report(frame, p):
    p = normalized(p)
    y = frame.label.to_numpy(int)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError('Report requires nonempty aligned labelled rows')
    pred = p.argmax(1) + 1
    detail = classification_report(y, pred, labels=list(range(1, 7)),
                                   output_dict=True, zero_division=0)
    conditional = {}
    for name, labels in [('pair23', [2, 3]), ('late456', [4, 5, 6])]:
        mask = np.isin(y, labels)
        conditional[name] = None if not mask.any() else {
            'rows': int(mask.sum()),
            'macro_f1': float(f1_score(y[mask], np.asarray(labels)[
                p[mask][:, np.asarray(labels) - 1].argmax(1)], labels=labels,
                average='macro', zero_division=0)),
            'meaning': 'Known true group; argmax restricted to its stages, not routed accuracy'}
    return {'rows': len(y), 'experiments': int(frame.experiment_id.nunique()),
            'macro_f1': float(f1_score(y, pred, labels=list(range(1, 7)),
                                      average='macro', zero_division=0)),
            'classification_report': detail,
            'confusion_matrix': confusion_matrix(y, pred, labels=list(range(1, 7))).tolist(),
            'conditional': conditional}


def predictions(frame, p, folds=None):
    cols = [c for c in ['sample_id', 'experiment_id', 'relative_time_s', 'label',
                       'bat_cc', 'soc_status', 'tr_method'] if c in frame]
    result = frame[cols].reset_index(drop=True).copy()
    if folds is not None:
        result['fold'] = np.asarray(folds)
    p = normalized(p)
    if len(result) != len(p):
        raise ValueError('Prediction row alignment differs')
    result['predicted_label'] = p.argmax(1) + 1
    for i in range(6):
        result[f'p{i + 1}'] = p[:, i]
    return result


def error_summary(frame, p):
    table = predictions(frame, p)
    table['wrong'] = table.label != table.predicted_label
    result = table.groupby('experiment_id', sort=False).agg(
        rows=('wrong', 'size'), errors=('wrong', 'sum'))
    result['error_rate'] = result.errors / result.rows
    return result.sort_values('errors', ascending=False).reset_index()
