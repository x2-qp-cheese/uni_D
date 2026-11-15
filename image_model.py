"""Independent frozen-image temporal expert, faithful to v17 V00.

There is NO numeric/condition branch. The 64 numeric slots from the original
architecture are constant zeros. Main output is P(parent)*P(child), for six
classes. Current/past embeddings are sampled by actual seconds, not row steps.
"""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

VERSION = 'uni-d-v21-independent-v00-image-tcn-1'
OFFSETS = [120, 90, 60, 45, 30, 20, 10, 0]
VISUAL_DIM = 1024


def _validate_frame(frame):
    names = {'sample_id', 'experiment_id', 'relative_time_s'}
    if names-set(frame) or frame.empty:
        raise ValueError(f'Need nonempty frame with {sorted(names)}')
    if frame.sample_id.isna().any() or not frame.sample_id.is_unique or frame.experiment_id.isna().any():
        raise ValueError('Missing/duplicate sample IDs or missing experiment IDs')
    times = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('Times must be finite/nonnegative')
    if pd.DataFrame({'experiment': frame.experiment_id.astype(str), 'time': times}).duplicated().any():
        raise ValueError('Duplicate timestamps within an experiment')
    return times


def histories(frame):
    """All-stage causal contexts; no label filtering or future observations."""
    times = _validate_frame(frame)
    n = len(frame)
    indices = np.tile(np.arange(n)[:, None], (1, len(OFFSETS)))
    valid = np.zeros_like(indices, dtype=np.float32)
    ages = np.zeros_like(valid)
    for positions in frame.groupby('experiment_id', sort=False).indices.values():
        pos = np.asarray(positions)
        pos = pos[np.argsort(times[pos], kind='stable')]
        for j, lag in enumerate(OFFSETS):
            past = np.searchsorted(times[pos], times[pos]-lag, side='right')-1
            available = past >= 0
            rows = pos[available]
            indices[rows, j] = pos[past[available]]
            valid[rows, j] = 1
            ages[rows, j] = (times[rows]-times[indices[rows, j]])/120
    return indices, valid, ages


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.pad = 2*dilation
        self.net = nn.Sequential(nn.Conv1d(width, width, 3, dilation=dilation),
                                 nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(.15))

    def forward(self, x):
        return x+self.net(nn.functional.pad(x, (self.pad, 0)))


class ImageHead(nn.Module):
    """State names match historical temporal_head.Head(image_only=True)."""
    def __init__(self):
        super().__init__()
        width = 128
        self.image = nn.Sequential(nn.LayerNorm(VISUAL_DIM), nn.Linear(VISUAL_DIM, width), nn.GELU())
        self.number = None
        self.film = None
        self.input = nn.Linear(width+64+2, width)
        self.temporal = nn.Sequential(*[CausalBlock(width, dilation) for dilation in [1, 2, 4]])
        self.router = nn.Linear(width, 3)
        self.pair = nn.Linear(width, 2)
        self.late = nn.Linear(width, 3)
        self.project = nn.Linear(width, 64)  # Retained unused projection preserves historical initialization/state.

    def forward(self, batch):
        image = self.image(batch['visual'])*batch['context'][:, :, :1]
        number = image.new_zeros((*image.shape[:2], 64))
        sequence = self.input(torch.cat([image, number, batch['context']], dim=-1))
        feature = self.temporal(sequence.transpose(1, 2))[:, :, -1]
        return {'router': self.router(feature), 'pair': self.pair(feature), 'late': self.late(feature),
                'embedding': nn.functional.normalize(self.project(feature), dim=1)}


class _Sequences(Dataset):
    def __init__(self, rows, embeddings, history, labels=None):
        self.rows = np.asarray(rows, int)
        self.embeddings = embeddings
        self.indices, self.valid, self.ages = history
        self.labels = labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        mask = self.valid[row]
        visual = np.asarray(self.embeddings[self.indices[row]], np.float32)*mask[:, None]
        context = np.stack([mask, self.ages[row]], axis=1)
        result = {'visual': torch.from_numpy(visual), 'context': torch.from_numpy(context)}
        if self.labels is not None:
            result['target'] = torch.tensor(int(self.labels[row]), dtype=torch.long)
        return result


def _normalize(probability, width):
    p = np.asarray(probability, np.float64)
    if p.ndim != 2 or p.shape[1] != width or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f'Invalid N x {width} probabilities')
    total = p.sum(1, keepdims=True)
    if (total <= 0).any():
        raise ValueError('Zero probability mass')
    return p/total


def _probabilities(output):
    router = _normalize(output['router'].softmax(1).cpu().numpy(), 3)
    pair = _normalize(output['pair'].softmax(1).cpu().numpy(), 2)
    late = _normalize(output['late'].softmax(1).cpu().numpy(), 3)
    return _normalize(np.column_stack([router[:, 0], router[:, 1, None]*pair, router[:, 2, None]*late]), 6)


def _loss(output, labels, weights):
    group = torch.where(labels == 0, 0, torch.where(labels <= 2, 1, 2))
    per_row = nn.functional.cross_entropy(output['router'], group, reduction='none')
    pair = (labels == 1) | (labels == 2)
    late = labels >= 3
    if pair.any():
        per_row[pair] += nn.functional.cross_entropy(output['pair'][pair], labels[pair]-1, reduction='none')
    if late.any():
        per_row[late] += nn.functional.cross_entropy(output['late'][late], labels[late]-3, reduction='none')
    return (per_row*weights[labels]).mean()


def _indices(indices, n, name):
    value = np.asarray(indices)
    if value.ndim != 1 or value.dtype.kind not in 'iu' or len(value) == 0:
        raise ValueError(f'{name} must be a nonempty integer index vector')
    if (value < 0).any() or (value >= n).any() or len(np.unique(value)) != len(value):
        raise ValueError(f'Invalid/duplicate {name}')
    return value.astype(int, copy=False)


def _embeddings(values, n):
    if np.shape(values) != (n, VISUAL_DIM):
        raise ValueError(f'Expected embeddings shape {(n, VISUAL_DIM)}')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite image embeddings')
    return values


def _device(device):
    target = torch.device(device)
    if target.type not in ['cpu', 'cuda'] or (target.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Use available cpu or cuda[:index]')
    return target


def _fingerprint(frame, embeddings, rows, labels, parameters):
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(json.dumps({'rows': rows.tolist(), 'parameters': parameters,
                    'context': [[str(s), str(e), float(t)] for s, e, t in
                                zip(frame.sample_id, frame.experiment_id, frame.relative_time_s)]}, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(np.asarray(embeddings, np.float32)).tobytes())
    digest.update(np.ascontiguousarray(labels[rows], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _portable(model_state, history, signature, parameters):
    return {'version': VERSION, 'architecture': {'visual_dim': VISUAL_DIM, 'numeric_dim': 0,
             'condition_dim': 0, 'condition': False, 'ordinal': False, 'width': 128},
            'state': {key: value.detach().cpu().numpy().copy() for key, value in model_state.items()},
            'training_loss': list(history), 'offsets_seconds': list(OFFSETS),
            'signature': signature, 'parameters': parameters, 'scope': 'independent image-only 1/23/456 expert; no numeric values'}


def fit_image(frame, embeddings, train_indices, val_indices=None, black_embeddings=None,
              checkpoint_dir=None, epochs=12, batch_size=128, lr=.0003, weight_decay=.001,
              seed=43, device='cpu', jobs=4, resume=False):
    """Fit historical V00 exactly; val_indices are checked, never scored/selected.

    The complete frame supplies causal image history before selecting current
    training rows. All six labels must exist in train_indices. The optional
    checkpoint contains only this head/optimizer/RNG, not encoder/data arrays.
    Callers should pre-register their trial and guard the output directory.
    """
    if min(epochs, batch_size, jobs) < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError('Invalid image-head training budget')
    _validate_frame(frame)
    embeddings = _embeddings(embeddings, len(frame))
    if black_embeddings is not None:
        _embeddings(black_embeddings, len(frame))
    rows = _indices(train_indices, len(frame), 'train_indices')
    if 'label' not in frame:
        raise ValueError('Training frame needs label')
    raw = pd.to_numeric(frame.label, errors='coerce').to_numpy(float)
    if not np.isin(raw[rows], range(1, 7)).all() or set(raw[rows].astype(int)) != set(range(1, 7)):
        raise ValueError('Training rows must contain all six original labels 1..6')
    labels = np.where(np.isfinite(raw), raw-1, -1).astype(int)
    if val_indices is not None:
        validation = _indices(val_indices, len(frame), 'val_indices')
        if np.intersect1d(rows, validation).size or set(frame.iloc[rows].experiment_id.astype(str)) & set(frame.iloc[validation].experiment_id.astype(str)):
            raise ValueError('Training and validation experiments must be disjoint')
    target = _device(device)
    parameters = {'epochs': int(epochs), 'batch_size': int(batch_size), 'lr': float(lr),
                  'weight_decay': float(weight_decay), 'seed': int(seed), 'device': str(target),
                  'jobs': int(jobs), 'sampler': '.5 frame uniform + .5 experiment uniform',
                  'class_weight_power': .25, 'supcon_weight': 0, 'validation_selection': False}
    signature = _fingerprint(frame, embeddings, rows, labels, parameters)
    checkpoint = None if checkpoint_dir is None else Path(checkpoint_dir)/'image_head.pt'
    saved = None
    if checkpoint is not None and checkpoint.is_file():
        if not resume:
            raise ValueError('Image checkpoint exists; use resume=True or a new directory')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('signature') != signature:
            raise ValueError('Image checkpoint data/code/settings differ')
        if saved['epoch'] == epochs:
            return _portable(saved['model_state'], saved['history'], signature, parameters)
        if not 0 <= saved['epoch'] < epochs:
            raise ValueError('Invalid completed-epoch checkpoint')
    torch.set_num_threads(jobs)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if target.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = ImageHead().to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    experiment = frame.iloc[rows].experiment_id.astype(str).to_numpy()
    _, codes = np.unique(experiment, return_inverse=True)
    counts_exp = np.bincount(codes)
    sample = .5/len(rows)+.5/(len(counts_exp)*counts_exp[codes])
    sampler_generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.tensor(sample, dtype=torch.double), len(rows), replacement=True,
                                     generator=sampler_generator)
    dataset = _Sequences(rows, embeddings, histories(frame), labels)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    counts = np.bincount(labels[rows], minlength=6)
    factors = (len(rows)/(6*np.maximum(counts, 1)))**.25
    weights = torch.tensor(factors/factors.mean(), dtype=torch.float32, device=target)
    first_epoch = 0
    history = []
    if saved is not None:
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(target)
        sampler_generator.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if target.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'], target)
        first_epoch = int(saved['epoch'])
        history = list(saved['history'])
    for epoch in range(first_epoch, epochs):
        model.train()
        total = 0.
        number = 0
        for batch in loader:
            batch = {key: value.to(target) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = _loss(output, batch['target'], weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('Image head training loss became non-finite')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.item())*len(batch['target'])
            number += len(batch['target'])
        history.append(total/number)
        print(f'Independent image TCN epoch {epoch+1}/{epochs}: train loss={history[-1]:.5f}', flush=True)
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            payload = {'signature': signature, 'epoch': epoch+1, 'model_state': model.state_dict(),
                       'optimizer_state': optimizer.state_dict(), 'history': history,
                       'sampler_rng': sampler_generator.get_state(), 'torch_rng': torch.get_rng_state(),
                       'cuda_rng': torch.cuda.get_rng_state(target) if target.type == 'cuda' else None}
            temporary = checkpoint.with_suffix('.tmp.pt')
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
    result = _portable(model.state_dict(), history, signature, parameters)
    del optimizer, model
    if target.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def predict_image(state, frame, embeddings, indices=None, device='cpu', batch_size=128):
    """Predict six probabilities in indices order; build history on ALL frame rows."""
    _validate_frame(frame)
    embeddings = _embeddings(embeddings, len(frame))
    rows = np.arange(len(frame)) if indices is None else _indices(indices, len(frame), 'indices')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if state.get('offsets_seconds', OFFSETS) != OFFSETS:
        raise ValueError('Image expert temporal offsets differ')
    architecture = state.get('architecture', {})
    if architecture.get('visual_dim', VISUAL_DIM) != VISUAL_DIM or architecture.get('numeric_dim', 0) != 0 or architecture.get('condition', False) or architecture.get('ordinal', False):
        raise ValueError('Expected the independent image-only V00 architecture')
    target = _device(device)
    model = ImageHead()
    model.load_state_dict({key: torch.from_numpy(np.asarray(value).copy()) for key, value in state['state'].items()}, strict=True)
    model = model.to(target).eval()
    dataset = _Sequences(rows, embeddings, histories(frame))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(target) for key, value in batch.items()}
            predictions.append(_probabilities(model(batch)))
    result = np.concatenate(predictions, axis=0)
    del model
    if target.type == 'cuda':
        torch.cuda.empty_cache()
    return result
