"""The training loop, and the sanity check that validates it.

Running this module performs the overfit check rather than a real experiment.
One tile is fitted twice: once against its dense mask, once against twenty
points. Dense supervision must drive the loss to near zero and mIoU on that
same tile to near one; if it cannot memorise a single image, the fault is in
the loop and no result computed on top of it means anything.

The point-supervised pass then fits its twenty pixels just as tightly while
dense mIoU stays far behind. That gap is not a bug, it is the problem point
supervision poses, measured before any method is applied to it.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
)

from data import LoveDAPoints
from losses import partial_focal_ce
from loveda import CLASSES, index_tiles
from metrics import ConfusionMatrix


ARCH = "resnet50"
LR = 1e-4
BATCH = 4


def device():
    """MPS where available, else CPU. No CUDA on this machine."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(arch=ARCH, n_classes=len(CLASSES)):
    """DeepLabv3 on an ImageNet-pretrained backbone, head resized to our classes.

    weights=None with weights_backbone set is the combination that matters:
    the decoder starts fresh for seven land-cover classes while the encoder
    arrives already knowing edges and textures. Twenty labelled pixels cannot
    teach an encoder from scratch, so pretraining is a precondition of the
    experiment rather than a tuning choice.
    """
    if arch == "resnet50":
        return deeplabv3_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
            num_classes=n_classes,
            aux_loss=False,
        )
    if arch == "mobilenet":
        return deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
            num_classes=n_classes,
            aux_loss=False,
        )
    raise ValueError(f"unknown arch {arch!r}")


def freeze_batchnorm(model):
    """Hold BatchNorm in inference mode for the whole of fine-tuning.

    Two reasons, one hard and one statistical. DeepLabv3's ASPP pools to 1x1,
    so a batch of one carries a single value per channel and BatchNorm cannot
    form a variance at all -- it raises. And at batch 4 the running statistics
    would be re-estimated from very few samples, throwing away the ImageNet
    estimates that are the entire reason for loading a pretrained backbone.
    The affine weights stay trainable; only the statistics are pinned.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train(model, loader, steps, lr=LR, gamma=0.0, dev=None, log_every=0):
    """Optimise for a fixed number of steps, cycling the loader as needed."""
    dev = dev or device()
    model.to(dev).train()
    freeze_batchnorm(model)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    stream = _cycle(loader)
    for step in range(1, steps + 1):
        images, targets = next(stream)
        logits = model(images.to(dev))["out"]
        loss = partial_focal_ce(logits, targets.to(dev), gamma=gamma)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        history.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"    step {step:>4}  loss {loss.item():.4f}")
    return history


@torch.no_grad()
def evaluate(model, loader, dev=None):
    """Confusion matrix over dense masks. Points never appear here."""
    dev = dev or device()
    model.to(dev).eval()
    matrix = ConfusionMatrix()
    for images, targets in loader:
        predictions = model(images.to(dev))["out"].argmax(1).cpu()
        matrix.update(predictions, targets)
    return matrix


def _cycle(loader):
    while True:
        yield from loader


if __name__ == "__main__":
    STEPS = 60
    dev = device()
    pairs = index_tiles()[:1]
    print(f"device: {dev}   arch: {ARCH}   overfitting {pairs[0][0].name}\n")

    dense_loader = DataLoader(
        LoveDAPoints(pairs, supervision="dense"), batch_size=1
    )

    for supervision, dataset in (
        ("dense mask", LoveDAPoints(pairs, supervision="dense")),
        ("20 points", LoveDAPoints(pairs, n_points=20, strategy="uniform")),
    ):
        torch.manual_seed(0)
        model = build_model()
        loader = DataLoader(dataset, batch_size=1)

        print(f"  [{supervision}]")
        history = train(model, loader, STEPS, log_every=20)
        matrix = evaluate(model, dense_loader, dev)

        _, targets = dataset[0]
        fitted = evaluate(model, DataLoader(dataset, batch_size=1), dev)
        print(f"    final loss on its own supervision : {history[-1]:.4f}")
        print(f"    mIoU on the supervision it saw    : {fitted.miou():.3f}")
        print(f"    mIoU against the DENSE mask       : {matrix.miou():.3f}\n")
