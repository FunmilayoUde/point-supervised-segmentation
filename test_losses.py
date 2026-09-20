import unittest

import torch
import torch.nn.functional as F

from losses import partial_focal_ce
from loveda import IGNORE


def logits_and_targets(seed=0, batch=2, classes=7, size=8):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, classes, size, size, generator=generator)
    targets = torch.randint(0, classes, (batch, size, size), generator=generator)
    return logits, targets


class PartialCrossEntropyTests(unittest.TestCase):
    def test_matches_cross_entropy_when_every_pixel_is_labelled(self):
        logits, targets = logits_and_targets()
        expected = F.cross_entropy(logits, targets, reduction="mean")
        actual = partial_focal_ce(logits, targets, gamma=0.0)
        self.assertAlmostEqual(actual.item(), expected.item(), places=5)

    def test_matches_ignore_index_when_pixels_are_masked_out(self):
        logits, targets = logits_and_targets()
        targets[0, :4] = IGNORE
        targets[1, :, 3:] = IGNORE
        expected = F.cross_entropy(
            logits, targets, ignore_index=IGNORE, reduction="mean"
        )
        actual = partial_focal_ce(logits, targets, gamma=0.0)
        self.assertAlmostEqual(actual.item(), expected.item(), places=5)

    def test_unlabelled_pixels_receive_no_gradient(self):
        logits, targets = logits_and_targets()
        logits.requires_grad_(True)
        labelled = torch.zeros_like(targets, dtype=torch.bool)
        labelled[0, 2, 5] = True
        labelled[1, 6, 1] = True
        targets = torch.where(labelled, targets, torch.full_like(targets, IGNORE))

        partial_focal_ce(logits, targets).backward()

        touched = logits.grad.abs().sum(dim=1) > 0
        self.assertTrue(torch.equal(touched, labelled))

    def test_value_does_not_change_with_the_number_of_points(self):
        """Identical predictions everywhere: 2 labels and 200 must score the same.

        This is the denominator under test. Dividing by pixel count instead of
        label count would make these differ by two orders of magnitude.
        """
        logits = torch.zeros(1, 4, 16, 16)
        logits[:, 1] = 2.0
        dense = torch.ones(1, 16, 16, dtype=torch.long)

        sparse = torch.full_like(dense, IGNORE)
        sparse[0, 0, :2] = 1
        few = partial_focal_ce(logits, sparse)
        many = partial_focal_ce(logits, dense)
        self.assertAlmostEqual(few.item(), many.item(), places=6)

    def test_returns_zero_and_stays_differentiable_without_labels(self):
        logits, targets = logits_and_targets()
        logits.requires_grad_(True)
        targets = torch.full_like(targets, IGNORE)

        loss = partial_focal_ce(logits, targets)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(logits.grad.abs().sum().item(), 0.0)

    def test_focal_term_discounts_confident_correct_pixels(self):
        logits = torch.tensor([[[[6.0]], [[0.0]], [[0.0]]]])
        targets = torch.zeros(1, 1, 1, dtype=torch.long)
        plain = partial_focal_ce(logits, targets, gamma=0.0)
        focal = partial_focal_ce(logits, targets, gamma=2.0)
        self.assertLess(focal.item(), plain.item())

    def test_batch_reduction_pools_labels_instead_of_averaging_images(self):
        logits, targets = logits_and_targets()
        targets[0, 1:] = IGNORE
        targets[1, 2:] = IGNORE

        pooled = partial_focal_ce(logits, targets, gamma=0.0)
        per_image = torch.stack(
            [partial_focal_ce(logits[i : i + 1], targets[i : i + 1], gamma=0.0)
             for i in range(2)]
        ).mean()

        # 8 labels in image 0 against 16 in image 1, so the two disagree.
        self.assertNotAlmostEqual(pooled.item(), per_image.item(), places=4)

        counts = (targets != IGNORE).flatten(1).sum(1).float()
        manual = torch.stack(
            [partial_focal_ce(logits[i : i + 1], targets[i : i + 1], gamma=0.0)
             for i in range(2)]
        )
        self.assertAlmostEqual(
            pooled.item(), ((manual * counts).sum() / counts.sum()).item(), places=5
        )


if __name__ == "__main__":
    unittest.main()
