"""mIoU accumulated over a confusion matrix.

The matrix is built first and the metrics read off it afterwards, so every
per-class number the report needs comes from one pass, and classes are scored
over the whole split rather than averaged across batches. Averaging per-batch
IoUs would over-weight batches that happened to contain a rare class.
"""

import numpy as np
from loveda import CLASSES, IGNORE


class ConfusionMatrix:

    def __init__(self, n_classes=len(CLASSES)):
        self.n_classes = n_classes
        self.matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, predictions, targets):
        predictions = np.asarray(predictions, dtype=np.int64).ravel()
        targets = np.asarray(targets, dtype=np.int64).ravel()

        keep = targets != IGNORE
        flat = targets[keep] * self.n_classes + predictions[keep]
        self.matrix += np.bincount(flat, minlength=self.n_classes**2).reshape(
            self.n_classes, self.n_classes
        )

    def iou(self):
        true_positives = np.diag(self.matrix)
        union = self.matrix.sum(1) + self.matrix.sum(0) - true_positives
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, true_positives / union, np.nan)

    def miou(self):

        return float(np.nanmean(self.iou()))

    def report(self):
        scores = dict(zip(CLASSES, self.iou()))
        scores["mIoU"] = self.miou()
        return scores
