"""No competition fitting or downloads. Run: python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D
import features as F
from test import make_submission
from utils import blend, combine


def frame(experiments=4, observations=12):
    records = []
    for e in range(experiments):
        for i in range(observations):
            row = {c: float(i + e) for c in F.SENSORS}
            row.update(sample_id=f's{e}_{i}', experiment_id=f'e{e}', relative_time_s=i * 5,
                       bat_class=1, bat_cc=4.8, soc_status=2, tr_method=1,
                       label=i % 6 + 1, image_path=f'images/s{e}_{i}.png')
            records.append(row)
    return pd.DataFrame(records)


class Invariants(unittest.TestCase):
    def test_causal_full_history_is_order_and_future_invariant(self):
        source = frame()
        full = F.build_features(source)
        self.assertEqual(full.shape[1], 77)
        subset = source[source.relative_time_s <= 25].copy()
        prefix = F.build_features(subset)
        pd.testing.assert_frame_equal(full.loc[subset.index].reset_index(drop=True), prefix)
        shuffled = source.sample(frac=1, random_state=5)
        transformed = F.build_features(shuffled)
        transformed.index = shuffled.index
        pd.testing.assert_frame_equal(full, transformed.sort_index())
        changed = source.copy()
        changed.loc[changed.experiment_id == 'e3', 'meas_t1'] = 99999
        pd.testing.assert_frame_equal(full.iloc[:36], F.build_features(changed).iloc[:36])
        self.assertNotIn('relative_time_s', full)
        self.assertNotIn('experiment_id', full)

    def test_group_folds_and_overlap_rejection(self):
        source = frame()
        manifest = D.make_splits(source, folds=2)
        splits = D.row_splits(source, manifest)
        seen = []
        for tr, va in splits:
            self.assertFalse(set(source.iloc[tr].experiment_id) & set(source.iloc[va].experiment_id))
            seen.extend(va)
        self.assertEqual(sorted(seen), list(range(len(source))))
        manifest['folds'][0]['training_experiments'] += manifest['folds'][0]['validation_experiments'][:1]
        with self.assertRaises(ValueError):
            D.row_splits(source, manifest)

    def test_missing_zero_corrupt_and_image_root_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / 'images'
            images.mkdir()
            source = frame(1, 3)
            for i in range(2):
                Image.new('RGB', (24, 32), (i * 30, 50, 100)).save(images / f's0_{i}.png')
            (images / 's0_2.png').touch()
            with self.assertRaises(FileNotFoundError):
                D.inspect_images(source, root / 'train.csv', root)
            selected, rows, audit = D.inspect_images(source, root / 'train.csv', root, allow_partial=True)
            self.assertEqual(rows.tolist(), [0, 1])
            self.assertEqual(audit['zero'], 1)
            self.assertEqual(len(selected), 2)
            before = audit['image_fingerprint']
            Image.new('RGB', (24, 32), (150, 50, 100)).save(images / 's0_0.png')
            _, _, after = D.inspect_images(source, root / 'train.csv', root, allow_partial=True)
            self.assertNotEqual(before, after['image_fingerprint'])
            (images / 's0_2.png').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                D.inspect_images(source, root / 'train.csv', root, allow_partial=True)

    def test_all_six_probabilities_and_submission_alignment(self):
        router = np.array([[.2, .3, .5]])
        pair = np.array([[.4, .6]])
        late = np.array([[.1, .3, .6]])
        p = combine(router, pair, late)
        np.testing.assert_allclose(p, [[.2, .12, .18, .05, .15, .30]])
        np.testing.assert_allclose(blend(p, np.eye(6)[[0]]), .75 * p + .25 * np.eye(6)[[0]])
        source = frame(1, 3)
        with tempfile.TemporaryDirectory() as folder:
            sample = Path(folder) / 'sample.csv'
            pd.DataFrame({'sample_id': ['s0_2', 's0_0', 's0_1'], 'stage': [0, 0, 0]}).to_csv(sample, index=False)
            result = make_submission(source, [2, 4, 6], sample)
            self.assertEqual(result.stage.tolist(), [6, 2, 4])
            self.assertEqual(list(result), ['sample_id', 'stage'])

    def test_image_history_does_not_use_future_or_other_experiment(self):
        from image_model import histories
        source = frame()
        indices, valid, ages = histories(source)
        for row in range(len(source)):
            for index in indices[row][valid[row].astype(bool)]:
                self.assertEqual(source.iloc[row].experiment_id, source.iloc[index].experiment_id)
                self.assertLessEqual(source.iloc[index].relative_time_s, source.iloc[row].relative_time_s)
        self.assertEqual(valid[0].sum(), 1)
        self.assertTrue(np.isfinite(ages).all())

    def test_interrupted_image_epoch_resumes_exactly(self):
        import image_model as I
        source = frame()
        embeddings = np.random.default_rng(7).standard_normal((len(source), 1024)).astype(np.float32)
        rows = np.arange(24)
        settings = dict(epochs=2, batch_size=12, seed=9, jobs=1)
        uninterrupted = I.fit_image(source, embeddings, rows, **settings)
        real_loss = I._loss
        calls = 0

        def interrupted_loss(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:  # First epoch/checkpoint complete, then interrupt next epoch.
                raise RuntimeError('synthetic interruption')
            return real_loss(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(I, '_loss', side_effect=interrupted_loss):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    I.fit_image(source, embeddings, rows, checkpoint_dir=folder, **settings)
            resumed = I.fit_image(source, embeddings, rows, checkpoint_dir=folder, resume=True, **settings)
            for key in uninterrupted['state']:
                np.testing.assert_array_equal(uninterrupted['state'][key], resumed['state'][key])
            changed = embeddings.copy()
            changed[0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'differ'):
                I.fit_image(source, changed, rows, checkpoint_dir=folder, resume=True, **settings)


if __name__ == '__main__':
    unittest.main()
