"""Partial cross entropy over point labels.

Implements:

    pfCE = sum(FocalLoss(pred, GT) * MASK_labelled) / sum(MASK_labelled)

Every pixel produces a loss, the unlabelled ones are multiplied away, and the
sum is divided by how many labels there actually were rather than by how many
pixels the tile holds. That denominator is the whole point. Normalising by
pixel count would shrink the loss as the point budget shrinks, quietly scaling
the effective learning rate with the very factor the experiments vary, and the
density curve would then measure two things at once.

gamma=0 collapses the focal term to plain cross entropy, so the same function
covers partial CE and partial focal CE, and the gamma ablation is one argument.

None of this is trusted: test_losses.py checks it against F.cross_entropy.
"""

import torch
import torch.nn.functional as F

from loveda import IGNORE


GAMMA = 2.0


def partial_focal_ce(logits, targets, gamma=GAMMA, ignore_index=IGNORE, weight=None):
    """Mean focal loss over the labelled pixels, ignoring everything else.

    logits are (B, C, H, W) float, targets are (B, H, W) int64 holding class
    ids at labelled pixels and ignore_index everywhere else.

    Reduction runs over the whole batch: losses are summed and divided by the
    total label count, not averaged per image and then across images. A tile
    that drew three points should not weigh as much as one that drew fifty.
    """
    labelled = targets != ignore_index
    n_labelled = labelled.sum()
    if n_labelled == 0:
        # Multiply rather than return a fresh zero, so the graph stays
        # connected and backward() still works on a batch with no labels.
        return logits.sum() * 0.0

    # gather() cannot index with ignore_index, so park the ignored pixels on
    # class 0 and let the mask discard whatever loss they produce.
    safe = torch.where(labelled, targets, torch.zeros_like(targets))

    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, safe.unsqueeze(1)).squeeze(1)

    loss = -((1.0 - log_pt.exp()) ** gamma) * log_pt
    if weight is not None:
        loss = loss * weight[safe]

    return (loss * labelled).sum() / n_labelled


def labelled_fraction(targets, ignore_index=IGNORE):
    """Share of pixels carrying a label, for logging the supervision actually used."""
    return (targets != ignore_index).float().mean().item()
