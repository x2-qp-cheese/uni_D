"""Frozen two-view ResNet18 features for the v21 independent image expert.

RGB is never interpreted as Celsius. The views and preprocessing intentionally
match v17 V00: whole frame and fixed central 75%, aspect-preserving 224-square
letterboxes, ImageNet normalization, ResNet18 IMAGENET1K_V1, fc removed. The
central view is not a battery detector. No image-derived statistic is fitted on
the input split. Supplying encoder_state makes inference fully offline.
"""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

VERSION = 'uni-d-v21-frozen-r18-two-view-1'
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
SIZE = 224
DIMENSION = 1024
_CONTRACT = {'version': VERSION, 'size': SIZE, 'views': ['whole', 'center_75pct'],
             'interpolation': 'PIL_bilinear', 'letterbox_rgb': np.round(MEAN*255).astype(int).tolist(),
             'mean': MEAN.tolist(), 'std': STD.tolist(), 'encoder': 'resnet18_IMAGENET1K_V1_fc_identity'}


def _validate_frame(frame):
    required = {'sample_id', 'experiment_id', 'relative_time_s', 'image_path'}
    if required - set(frame) or frame.empty:
        raise ValueError(f'Need nonempty readable-image frame with {sorted(required)}')
    if frame.sample_id.isna().any() or not frame.sample_id.is_unique or frame.experiment_id.isna().any():
        raise ValueError('sample_id must be unique/non-null; experiment_id must be non-null')
    times = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('relative_time_s must be finite and nonnegative')
    if pd.DataFrame({'exp': frame.experiment_id.astype(str), 'time': times}).duplicated().any():
        raise ValueError('Duplicate timestamps inside an experiment are ambiguous')
    paths = [Path(str(p)) for p in frame.image_path]
    if any(not p.is_absolute() for p in paths):
        raise ValueError('image_path must be absolute; resolve readable images in the data loader')
    return paths, times


def _state_numpy(model):
    return {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}


def _state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = np.asarray(state[key])
        if value.dtype.hasobject:
            raise ValueError('Encoder state cannot contain object arrays')
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _load_encoder(encoder_state=None, weights_path=None):
    """Use portable state/local weights offline, or torchvision's official weights."""
    if encoder_state is not None:
        model = resnet18(weights=None)
        model.fc = nn.Identity()
        tensors = {key: torch.from_numpy(np.asarray(value).copy()) for key, value in encoder_state.items()}
        model.load_state_dict(tensors, strict=True)
    elif weights_path is not None:
        model = resnet18(weights=None)
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f'ResNet18 weights_path does not exist: {weights_path}')
        weights = torch.load(Path(weights_path), map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.fc = nn.Identity()
    else:
        # torchvision uses its normal cache or downloads the official file and
        # verifies the published filename hash. No workspace-specific paths.
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    model.requires_grad_(False)
    return model.eval()


def _views(image):
    """Return [whole, center] float32 RGB in 0..1, before standardization."""
    width, height = image.size
    result = []
    for roi in [(0, 0, width, height), (.125*width, .125*height, .875*width, .875*height)]:
        crop = image.crop(tuple(map(round, roi)))
        if min(crop.size) < 1:
            raise ValueError('Source image is too small for the central view')
        ratio = min(SIZE/crop.width, SIZE/crop.height)
        crop = crop.resize((max(1, round(crop.width*ratio)), max(1, round(crop.height*ratio))), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (SIZE, SIZE), tuple(np.round(MEAN*255).astype(int)))
        canvas.paste(crop, ((SIZE-crop.width)//2, (SIZE-crop.height)//2))
        result.append(np.asarray(canvas, np.float32).transpose(2, 0, 1)/255)
    return np.stack(result)


def _inspect(path):
    try:
        data = path.read_bytes()
        if not data:
            raise ValueError('zero-byte image')
        with Image.open(io.BytesIO(data)) as original:
            geometry = original.size
            original.verify()
        return hashlib.sha256(data).hexdigest(), tuple(map(int, geometry))
    except Exception as error:
        raise ValueError(f'Cannot inspect readable image {path}: {error}') from error


def _pixels(path_and_hash):
    path, expected = path_and_hash
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Image changed during extraction: {path}')
    with Image.open(io.BytesIO(data)) as original:
        value = _views(original.convert('RGB'))
    return (value-MEAN[None, :, None, None])/STD[None, :, None, None]


@contextlib.contextmanager
def _lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.features.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Another process is writing this feature fingerprint') from error
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def extract_embeddings(frame, cache_dir=None, device='cpu', batch_size=32, jobs=4,
                       encoder_state=None, weights_path=None):
    """Return embeddings[N,1024], black[N,1024], portable encoder_state and fingerprint.

    cache_dir is optional. Its fingerprinted child stores only feature arrays,
    unique black vectors and row indices. It never stores source/RGB images or
    repeated black vectors on disk. Black input preserves each original image's
    geometry and both crops, including the normal mean-colour letterbox bars.
    """
    if batch_size < 1 or jobs < 1:
        raise ValueError('batch_size and jobs must be positive')
    paths, times = _validate_frame(frame)
    target = torch.device(device)
    if target.type not in ['cpu', 'cuda'] or (target.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Use available cpu or cuda[:index]')
    torch.set_num_threads(jobs)
    model = _load_encoder(encoder_state, weights_path)
    portable = _state_numpy(model)
    encoder_hash = _state_hash(portable)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        inspected = list(pool.map(_inspect, paths))
    image_hashes = [value[0] for value in inspected]
    geometries = [value[1] for value in inspected]
    record = {'contract': _CONTRACT, 'encoder_sha256': encoder_hash,
              'rows': [[str(sid), str(exp), float(t), digest] for sid, exp, t, digest in
                       zip(frame.sample_id, frame.experiment_id, times, image_hashes)]}
    fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    directory = None if cache_dir is None else Path(cache_dir)/fingerprint

    def cached():
        meta_path = directory/'metadata.json'
        if not meta_path.is_file():
            return None
        saved = json.loads(meta_path.read_text('utf-8'))
        if saved.get('fingerprint') != fingerprint:
            raise ValueError('Feature cache fingerprint differs')
        for name in ['embeddings.npy', 'black_unique.npy', 'black_index.npy']:
            if _file_hash(directory/name) != saved['sha256'][name]:
                raise ValueError(f'Feature cache was modified: {name}')
        embeddings = np.load(directory/'embeddings.npy', mmap_mode='r')
        unique = np.load(directory/'black_unique.npy', mmap_mode='r')
        index = np.load(directory/'black_index.npy', allow_pickle=False)
        if embeddings.shape != (len(frame), DIMENSION) or embeddings.dtype != np.float32:
            raise ValueError('Cached embedding shape/dtype differs')
        if index.shape != (len(frame),) or index.dtype.kind not in 'iu' or (index < 0).any() or (index >= len(unique)).any():
            raise ValueError('Invalid black-vector row mapping')
        if unique.ndim != 2 or unique.shape[1] != DIMENSION or unique.dtype != np.float32:
            raise ValueError('Invalid unique-black feature cache')
        if not np.isfinite(embeddings).all() or not np.isfinite(unique).all():
            raise ValueError('Cached embeddings are non-finite')
        return {'embeddings': embeddings, 'black': np.asarray(unique[index], np.float32),
                'encoder_state': portable, 'fingerprint': fingerprint, 'metadata': saved}

    def compute():
        model.to(target).eval()
        if directory is None:
            embeddings = np.empty((len(frame), DIMENSION), np.float32)
        else:
            embeddings = np.lib.format.open_memmap(directory/'embeddings.tmp.npy', mode='w+',
                                                   dtype=np.float32, shape=(len(frame), DIMENSION))
        active_batch = int(batch_size)
        start = 0
        with ThreadPoolExecutor(max_workers=jobs) as pool, torch.inference_mode():
            while start < len(frame):
                end = min(len(frame), start+active_batch)
                arrays = list(pool.map(_pixels, zip(paths[start:end], image_hashes[start:end])))
                batch = None
                try:
                    batch = torch.from_numpy(np.stack(arrays)).to(target).flatten(0, 1)
                    value = model(batch).detach().cpu().numpy().reshape(end-start, DIMENSION)
                except torch.cuda.OutOfMemoryError:
                    del batch, arrays
                    if active_batch == 1:
                        raise
                    active_batch = max(1, active_batch//2)
                    torch.cuda.empty_cache()
                    continue
                embeddings[start:end] = value
                del batch, arrays
                start = end
                if start % 2048 < active_batch or start == len(frame):
                    print(f'Frozen ResNet18 images {start:,}/{len(frame):,}', flush=True)
            unique_geometry = sorted(set(geometries))
            geometry_index = {value: i for i, value in enumerate(unique_geometry)}
            black_index = np.asarray([geometry_index[value] for value in geometries], np.int32)
            black_unique = []
            for width, height in unique_geometry:
                value = _views(Image.new('RGB', (width, height), (0, 0, 0)))
                value = (value-MEAN[None, :, None, None])/STD[None, :, None, None]
                tensor = torch.from_numpy(value).to(target)
                black_unique.append(model(tensor).cpu().numpy().reshape(DIMENSION))
            black_unique = np.asarray(black_unique, np.float32)
        metadata = {'fingerprint': fingerprint, 'contract': _CONTRACT, 'rows': len(frame),
                    'encoder_sha256': encoder_hash, 'unique_black_geometries': len(black_unique),
                    'source_dimensions': [list(value) for value in unique_geometry],
                    'black_definition': 'actual black original-size RGB, same whole/center crops, letterbox, normalization and encoder',
                    'device_used': str(target), 'actual_batch_size': active_batch}
        if directory is not None:
            embeddings.flush()
            del embeddings
            (directory/'embeddings.tmp.npy').replace(directory/'embeddings.npy')
            np.save(directory/'black_unique.npy', black_unique)
            np.save(directory/'black_index.npy', black_index)
            metadata['sha256'] = {name: _file_hash(directory/name) for name in ['embeddings.npy', 'black_unique.npy', 'black_index.npy']}
            temporary = directory/'metadata.tmp.json'
            temporary.write_text(json.dumps(metadata, indent=2), 'utf-8')
            temporary.replace(directory/'metadata.json')
            embeddings = np.load(directory/'embeddings.npy', mmap_mode='r')
        return {'embeddings': embeddings, 'black': black_unique[black_index], 'encoder_state': portable,
                'fingerprint': fingerprint, 'metadata': metadata}

    try:
        if directory is not None:
            with _lock(directory):
                result = cached()
                return result if result is not None else compute()
        return compute()
    finally:
        del model
        if target.type == 'cuda':
            torch.cuda.empty_cache()
