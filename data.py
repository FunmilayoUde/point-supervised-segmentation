"""Tiles and their point labels, as a torch Dataset.

Each 1024 tile is resized to 256 before points are drawn, so the budget is
counted against the image the network actually sees and the coverage numbers in
points.py carry over unchanged. Order matters here: sampling first and resizing
afterwards would drop or duplicate points depending on the interpolation, and
the realised budget would no longer be the one under study.

Points are drawn from a stream keyed on (seed, tile index), so a tile yields
the same points on every epoch without anything being cached. Re-drawing each
epoch would leak roughly budget x epochs distinct labels into training and
quietly dissolve the experiment.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from loveda import read_mask, to_train_ids
from points import sample_points, tile_rng


SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LoveDAPoints(Dataset):
    """Tiles paired with simulated point labels, or with the dense mask.

    supervision="points" yields the sparse map that training consumes;
    supervision="dense" yields the full mask, which only evaluation may see.
    """

    def __init__(self, pairs, supervision="points", n_points=20,
                 strategy="uniform", seed=0, size=SIZE):
        if supervision not in ("points", "dense"):
            raise ValueError(f"unknown supervision {supervision!r}")
        self.pairs = pairs
        self.supervision = supervision
        self.n_points = n_points
        self.strategy = strategy
        self.seed = seed
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask = Image.fromarray(read_mask(mask_path))
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        labels = to_train_ids(np.asarray(mask))
        if self.supervision == "points":
            labels = sample_points(
                labels, self.n_points, self.strategy, tile_rng(self.seed, index)
            )

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        return (
            torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
            torch.from_numpy(labels.astype(np.int64)),
        )
