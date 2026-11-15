"""Three native CatBoost models; only supplied training rows enter a fit."""
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

from features import CATEGORICAL_FEATURES, sample_weights
from utils import combine, normalized


def parameters(seed=42, iterations=600, task_type='CPU', devices='0',
               gpu_ram_part=.60, jobs=4):
    p = dict(iterations=iterations, depth=5, learning_rate=.05, l2_leaf_reg=8,
             random_seed=seed, task_type=task_type, thread_count=jobs,
             border_count=128, max_ctr_complexity=1, boosting_type='Plain',
             nan_mode='Min', allow_writing_files=False)
    if task_type == 'GPU':
        p.update(devices=devices, gpu_ram_part=gpu_ram_part)
    return p


def fit_numeric(x, labels, rows, directory, identity, seed=42, iterations=600,
                task_type='CPU', devices='0', gpu_ram_part=.60, jobs=4,
                resume=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    y, rows = np.asarray(labels, int), np.asarray(rows, int)
    grouped = np.where(y == 1, 1, np.where(y <= 3, 2, 3))
    specs = [('router', grouped, rows, [1, 2, 3]),
             ('pair23', y, rows[np.isin(y[rows], [2, 3])], [2, 3]),
             ('late456', y, rows[np.isin(y[rows], [4, 5, 6])], [4, 5, 6])]
    models = {}
    for name, target, selected, classes in specs:
        if set(target[selected]) != set(classes):
            raise ValueError(f'{name}: training rows must contain {classes}')
        path = directory / f'{name}.cbm'
        fit_id = identity + ':' + name
        if resume and path.is_file():
            model = CatBoostClassifier()
            model.load_model(str(path))
            if model.get_metadata().get('fit_identity') != fit_id:
                raise ValueError(f'Saved numeric fit differs: {path}')
            if model.feature_names_ != list(x) or list(map(int, model.classes_)) != classes:
                raise ValueError('Saved CatBoost schema/classes differ')
            print(f'Reuse {path}', flush=True)
        else:
            p = parameters(seed, iterations, task_type, devices, gpu_ram_part, jobs)
            p['loss_function'] = 'Logloss' if len(classes) == 2 else 'MultiClass'
            model = CatBoostClassifier(**p, metadata={'fit_identity': fit_id})
            print(f'CatBoost {name}: {len(selected)} training rows', flush=True)
            model.fit(Pool(x.iloc[selected], label=target[selected],
                           weight=sample_weights(target[selected], .45),
                           cat_features=CATEGORICAL_FEATURES), verbose=200)
            temporary = path.with_suffix('.tmp.cbm')
            model.save_model(str(temporary))
            temporary.replace(path)
        models[name] = model
    return models


def load_numeric(directory):
    result = {}
    for name in ['router', 'pair23', 'late456']:
        model = CatBoostClassifier()
        model.load_model(str(Path(directory) / f'{name}.cbm'))
        result[name] = model
    return result


def predict_numeric(models, x, jobs=4):
    probabilities = []
    for name, classes in [('router', [1, 2, 3]), ('pair23', [2, 3]),
                          ('late456', [4, 5, 6])]:
        model = models[name]
        if list(map(int, model.classes_)) != classes or model.feature_names_ != list(x):
            raise ValueError(f'{name}: incompatible feature or class order')
        probabilities.append(normalized(model.predict_proba(x, thread_count=jobs), len(classes)))
    return combine(*probabilities)
