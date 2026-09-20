"""What the predictions actually look like.

Numbers say point supervision reaches 92% of the ceiling; they do not say what
the remaining 8% looks like on the ground. This trains the best point-supervised
configuration once, keeps the checkpoint, and renders held-out tiles as image,
dense truth, the handful of pixels that trained the model, and its prediction.

The third panel is the one to dwell on. Two hundred dots against a full mask is
the entire argument for the partial loss, and it is far more legible as a
picture than as a percentage.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import LoveDAPoints
from experiment import EPOCHS, GAMMA, LR, BATCH, splits
from losses import partial_focal_ce
from loveda import CLASSES, IGNORE, OUTPUT, PALETTE
from points import sample_points, tile_rng
from train import build_model, device, evaluate, freeze_batchnorm


STRATEGY = "uniform"
POINTS = 200
SEED = 0
SHOW_TILES = (0, 1, 2)
CHECKPOINT = OUTPUT / f"{STRATEGY}_n{POINTS}_seed{SEED}.pt"


def paint(labels):
    """Class ids to RGB, with IGNORE rendered as the no-data colour."""
    return PALETTE[np.where(labels == IGNORE, len(CLASSES), labels)]


def fit(train_pairs, val_pairs, dev):
    """Train the showcase configuration, reusing a checkpoint if one exists."""
    model = build_model("mobilenet").to(dev)
    if CHECKPOINT.exists():
        model.load_state_dict(torch.load(CHECKPOINT, map_location=dev))
        return model

    torch.manual_seed(SEED)
    model = build_model("mobilenet").to(dev)
    loader = DataLoader(
        LoveDAPoints(train_pairs, n_points=POINTS, strategy=STRATEGY, seed=SEED),
        batch_size=BATCH, shuffle=True,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        freeze_batchnorm(model)
        for images, targets in loader:
            loss = partial_focal_ce(
                model(images.to(dev))["out"], targets.to(dev), gamma=GAMMA
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        print(f"  epoch {epoch:>2}/{EPOCHS}")

    torch.save(model.state_dict(), CHECKPOINT)
    return model


@torch.no_grad()
def panel(model, val_pairs, dev, indices=SHOW_TILES, out_path=None):
    """Image, dense truth, the training points, and the prediction."""
    dense = LoveDAPoints(val_pairs, supervision="dense")
    model.eval()

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4.1 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, index in enumerate(indices):
        image, truth = dense[index]
        prediction = model(image.unsqueeze(0).to(dev))["out"].argmax(1)[0].cpu().numpy()
        truth = truth.numpy()
        points = sample_points(
            truth.astype(np.uint8), POINTS, STRATEGY, tile_rng(SEED, index)
        )

        raw = np.asarray(plt.imread(val_pairs[index][0]))
        rows_, cols_ = np.nonzero(points != IGNORE)

        axes[row, 0].imshow(raw)
        axes[row, 1].imshow(paint(truth))
        axes[row, 2].imshow(raw)
        axes[row, 2].scatter(
            cols_ * raw.shape[1] / points.shape[1],
            rows_ * raw.shape[0] / points.shape[0],
            c=PALETTE[points[rows_, cols_]] / 255, s=14,
            edgecolors="black", linewidths=0.4,
        )
        axes[row, 3].imshow(paint(prediction))

        if row == 0:
            for ax, title in zip(
                axes[row],
                ("image", "dense truth (never trained on)",
                 f"{POINTS} points (what trained it)", "prediction"),
            ):
                ax.set_title(title, fontsize=11)
        for ax in axes[row]:
            ax.axis("off")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    dev = device()
    train_pairs, val_pairs = splits()
    print(f"training {STRATEGY} N={POINTS} seed={SEED} on {dev}")

    model = fit(train_pairs, val_pairs, dev)
    scores = evaluate(
        model, DataLoader(LoveDAPoints(val_pairs, supervision="dense"), batch_size=BATCH), dev
    ).report()
    print(f"held-out mIoU {scores['mIoU']:.4f}")

    panel(model, val_pairs, dev, out_path=OUTPUT / "fig3_qualitative.png")
    print(f"wrote fig3_qualitative.png and {CHECKPOINT.name} to {OUTPUT}")
