"""Loading LoveDA tiles, and proving the data path before a model exists.

Indexes LoveDA tiles, remaps the raw 0-7 mask values onto contiguous training
indices, and renders image/label overlays so alignment can be judged by eye.
Nothing here imports torch. A mask that is misaligned or mis-indexed has to be
caught now: once training starts it surfaces only as a mediocre mIoU, with no
indication of which stage is at fault.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
# The dataset is ~6 GB and lives outside the repo. LOVEDA_ROOT overrides the
# default for a clone that keeps it somewhere else.
ROOT = Path(os.environ.get("LOVEDA_ROOT", HERE.parent / "data" / "LoveDA"))
OUTPUT = HERE / "outputs"
SPLIT = "Train"
SCENE = "Urban"  # One domain only. LoveDA's Urban/Rural gap is a domain-shift
# benchmark in its own right, and mixing them would confound label density.

CLASSES = ("background", "building", "road", "water", "barren", "forest", "agriculture")
IGNORE = 255  # No-data here; points.py reuses it for unsampled pixels.

# LoveDA's published colours, with black appended for no-data.
PALETTE = np.array(
    [
        [255, 255, 255],
        [255, 0, 0],
        [255, 255, 0],
        [0, 0, 255],
        [159, 129, 183],
        [0, 255, 0],
        [255, 195, 128],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)


def index_tiles(root=ROOT, split=SPLIT, scene=SCENE):
    """Return sorted (image, mask) path pairs for one split and scene."""
    image_dir = Path(root) / split / scene / "images_png"
    mask_dir = Path(root) / split / scene / "masks_png"

    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"no tiles under {image_dir}")

    pairs = [(image, mask_dir / image.name) for image in images]
    orphans = [mask for _, mask in pairs if not mask.exists()]
    if orphans:
        raise FileNotFoundError(f"{len(orphans)} masks missing, first is {orphans[0]}")
    return pairs


def read_image(path):
    """Load a tile as uint8 HxWx3."""
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path):
    """Load a raw LoveDA mask as uint8 HxW with values in 0-7."""
    raw = np.asarray(Image.open(path))
    if raw.ndim != 2:
        raise ValueError(f"{path.name}: expected a single channel, got {raw.shape}")
    if raw.max() > len(CLASSES):
        raise ValueError(f"{path.name}: label {raw.max()} above 0-{len(CLASSES)}")
    return raw


def to_train_ids(raw):
    """Shift LoveDA's 1-indexed labels down, sending no-data to IGNORE.

    Cross entropy wants targets in [0, C-1], and LoveDA spends index 0 on
    no-data rather than on a class. Doing the shift at load time gives every
    later stage one convention to reason about, and leaves no-data already
    marked IGNORE before points.py draws points, so a sampled point cannot land
    on a pixel that was never annotated in the first place.
    """
    labels = raw.astype(np.int16) - 1
    labels[raw == 0] = IGNORE
    return labels.astype(np.uint8)


def colorize(labels):
    """Map train ids, and IGNORE, onto RGB for display."""
    return PALETTE[np.where(labels == IGNORE, len(CLASSES), labels)]


def class_pixel_counts(pairs, limit=None):
    """Count pixels per class across tiles; the last bin holds no-data."""
    counts = np.zeros(len(CLASSES) + 1, dtype=np.int64)
    for _, mask_path in pairs[:limit]:
        labels = to_train_ids(read_mask(mask_path))
        binned = np.where(labels == IGNORE, len(CLASSES), labels)
        counts += np.bincount(binned.ravel(), minlength=len(CLASSES) + 1)
    return counts


def show_pair(image_path, mask_path, alpha=0.45, out_path=None):
    """Render the tile, its colourised labels, and the two blended.

    The blend is the panel that earns its place. Side-by-side panels let a
    small registration error pass as a plausible-looking pair; an overlay puts
    building edges either on the buildings or visibly beside them.
    """
    image = read_image(image_path)
    painted = colorize(to_train_ids(read_mask(mask_path)))
    blended = ((1 - alpha) * image + alpha * painted).astype(np.uint8)

    panels = (image, painted, blended)
    titles = (image_path.name, "labels", f"overlay (alpha={alpha})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    HISTOGRAM_TILES = 200

    pairs = index_tiles()
    image = read_image(pairs[0][0])
    labels = to_train_ids(read_mask(pairs[0][1]))

    print(f"{SPLIT}/{SCENE}: {len(pairs)} tiles")
    print(f"image: {image.shape} {image.dtype}")
    print(f"mask:  {labels.shape} {labels.dtype}")
    print(f"train ids in first tile: {sorted(np.unique(labels).tolist())}")

    counts = class_pixel_counts(pairs, limit=HISTOGRAM_TILES)
    shares = counts / counts.sum()
    names = list(CLASSES) + ["no-data"]

    print(f"\n--- class balance over {min(HISTOGRAM_TILES, len(pairs))} tiles ---")
    for name, count, share in zip(names, counts, shares):
        print(f"  {name:<12} {share:8.3%} {count:>13,}")

    OUTPUT.mkdir(exist_ok=True)
    pd.DataFrame({"class": names, "pixels": counts, "share": shares}).to_csv(
        OUTPUT / "class_balance.csv", index=False
    )
    show_pair(*pairs[0], out_path=OUTPUT / "tile_overlay.png")
    print(f"\nwrote class_balance.csv, tile_overlay.png to {OUTPUT}")
