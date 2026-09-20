import unittest

import numpy as np
import torch

from data import SIZE, LoveDAPoints
from loveda import IGNORE, index_tiles


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = index_tiles()[:4]

    def test_shapes_and_dtypes(self):
        image, labels = LoveDAPoints(self.pairs)[0]
        self.assertEqual(image.shape, (3, SIZE, SIZE))
        self.assertEqual(labels.shape, (SIZE, SIZE))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.int64)

    def test_point_supervision_keeps_only_the_budget(self):
        _, labels = LoveDAPoints(self.pairs, n_points=20)[0]
        self.assertEqual(int((labels != IGNORE).sum()), 20)

    def test_dense_supervision_keeps_the_whole_mask(self):
        _, labels = LoveDAPoints(self.pairs, supervision="dense")[0]
        self.assertGreater(int((labels != IGNORE).sum()), SIZE * SIZE // 2)

    def test_points_agree_with_the_dense_mask(self):
        _, sparse = LoveDAPoints(self.pairs, n_points=20)[0]
        _, dense = LoveDAPoints(self.pairs, supervision="dense")[0]
        labelled = sparse != IGNORE
        self.assertTrue(torch.equal(sparse[labelled], dense[labelled]))

    def test_points_are_identical_across_epochs(self):
        """The same tile must yield the same points every time it is drawn.

        Re-drawing per epoch would let the effective label budget grow with
        training length, which is the factor the experiments hold fixed.
        """
        dataset = LoveDAPoints(self.pairs, n_points=20)
        self.assertTrue(torch.equal(dataset[0][1], dataset[0][1]))

    def test_seed_changes_the_points(self):
        a = LoveDAPoints(self.pairs, n_points=20, seed=0)[0][1]
        b = LoveDAPoints(self.pairs, n_points=20, seed=1)[0][1]
        self.assertFalse(torch.equal(a, b))

    def test_normalisation_is_roughly_standard(self):
        image, _ = LoveDAPoints(self.pairs)[0]
        self.assertLess(abs(float(image.mean())), 3.0)
        self.assertLess(float(image.abs().max()), 5.0)


if __name__ == "__main__":
    unittest.main()
