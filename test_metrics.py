import unittest

import numpy as np

from loveda import IGNORE
from metrics import ConfusionMatrix


class ConfusionMatrixTests(unittest.TestCase):
    def test_perfect_prediction_scores_one(self):
        targets = np.array([[0, 1, 2], [3, 4, 5]])
        matrix = ConfusionMatrix()
        matrix.update(targets, targets)
        self.assertAlmostEqual(matrix.miou(), 1.0)

    def test_ignored_pixels_do_not_enter_the_matrix(self):
        targets = np.array([0, 1, IGNORE, IGNORE])
        predictions = np.array([0, 1, 5, 5])
        matrix = ConfusionMatrix()
        matrix.update(predictions, targets)
        self.assertEqual(matrix.matrix.sum(), 2)
        self.assertAlmostEqual(matrix.miou(), 1.0)

    def test_absent_classes_are_nan_not_zero(self):
        targets = np.array([0, 0, 1])
        matrix = ConfusionMatrix()
        matrix.update(targets, targets)
        scores = matrix.iou()
        self.assertFalse(np.isnan(scores[:2]).any())
        self.assertTrue(np.isnan(scores[2:]).all())

    def test_iou_matches_a_hand_computed_case(self):
        # class 0: 2 correct, 1 predicted as 1 -> TP=2 FN=1 FP=0 -> 2/3
        # class 1: 1 correct, plus the stray above -> TP=1 FN=0 FP=1 -> 1/2
        targets = np.array([0, 0, 0, 1])
        predictions = np.array([0, 0, 1, 1])
        matrix = ConfusionMatrix()
        matrix.update(predictions, targets)
        scores = matrix.iou()
        self.assertAlmostEqual(scores[0], 2 / 3)
        self.assertAlmostEqual(scores[1], 1 / 2)
        self.assertAlmostEqual(matrix.miou(), (2 / 3 + 1 / 2) / 2)

    def test_batches_accumulate(self):
        matrix = ConfusionMatrix()
        matrix.update(np.array([0]), np.array([0]))
        matrix.update(np.array([1]), np.array([1]))
        self.assertEqual(matrix.matrix.sum(), 2)
        self.assertAlmostEqual(matrix.miou(), 1.0)


if __name__ == "__main__":
    unittest.main()
