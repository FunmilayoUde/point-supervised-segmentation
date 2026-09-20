# Partial cross-entropy for point-supervised land-cover segmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22861209.svg)](https://doi.org/10.5281/zenodo.22861209)

How far does a segmentation model get when it is trained on a handful of
labelled pixels per image instead of a full mask, and does it matter how those
pixels are chosen?

On LoveDA urban tiles, **200 labelled pixels out of 65,536 — about 0.3% —
recover 92% of fully supervised mIoU.** Class-balanced sampling, the obvious
remedy for the severe class imbalance in this data, does not improve mean mIoU
at any budget tested; it reduces the spread across seeds by roughly a factor of
three.

**[Read the report (PDF, 10 pages)](outputs/report/Partial_CE_Point_Supervision_Report.pdf)**

![mIoU against label budget, uniform and stratified sampling, with the fully
supervised ceiling and majority-class floor marked](outputs/fig1_budget_curve.png)

## The loss

    pfCE = sum(FocalLoss(pred, GT) * MASK_labelled) / sum(MASK_labelled)

Per-pixel focal loss, masked to the annotated pixels, normalised by how many
labels there were rather than by how many pixels the tile holds. Unlabelled
pixels receive no gradient. The denominator matters: normalising by pixel count
would shrink the loss as the budget shrinks, which would scale the effective
learning rate with the factor under study.

`losses.py` is checked against `F.cross_entropy` in `test_losses.py`: at
`gamma=0` it agrees with `reduction="mean"` on a full mask and with
`ignore_index=255` on a partial one, gradients at unlabelled positions are
exactly zero, and the loss is invariant to how many points were drawn.

## Layout

| File | |
|---|---|
| `loveda.py` | Tile loading, label remapping, class balance |
| `points.py` | Uniform and stratified point sampling, coverage analysis |
| `losses.py` | Partial focal cross-entropy |
| `data.py` | Dataset pairing tiles with points or dense masks |
| `metrics.py` | Confusion-matrix mIoU |
| `train.py` | Training loop; run directly for the single-tile sanity check |
| `experiment.py` | The 27-run grid, resumable |
| `extra_seeds.py` | Three more seeds per strategy at N=50 |
| `figures.py`, `qualitative.py` | Figures 1–3 |
| `sampling_budget.py` | Probability a budget misses a class |
| `build_report.py` | Regenerates the PDF from the results |

Results land in `outputs/` as `results.csv` and `sweep.log`.

## Running it

Download LoveDA `Train.zip` and `Val.zip` from
[Zenodo](https://zenodo.org/records/5706578) and unzip alongside the repo:

    ../data/LoveDA/Train/Urban/{images_png,masks_png}

or set `LOVEDA_ROOT` to wherever you keep it.

    pip install torch torchvision numpy pandas pillow matplotlib reportlab
    python -m unittest discover        # 19 tests
    python train.py                    # single-tile sanity check
    python experiment.py               # the grid, ~2.5 h on an M-series GPU
    python extra_seeds.py
    python figures.py && python qualitative.py
    python build_report.py

`experiment.py` skips configurations already present in `results.csv`, so it
survives an interrupted run.

## Caveats

Runs are not bit-reproducible. Reductions on the Metal backend are not
deterministic, so retraining a configuration with the same seed lands close to
but not on the recorded number. Point sampling itself is deterministic, keyed
on `(seed, tile index)`. Section 6 of the report covers this and the other
limitations.

## Citing this work

Archived on Zenodo. The DOI below resolves to the most recent version; each
release also has its own.

    Ude, O. (2026). Partial cross-entropy for point-supervised land-cover
    segmentation: the effect of label budget and sampling strategy (v1.0.0).
    Zenodo. https://doi.org/10.5281/zenodo.22861209

```bibtex
@software{ude2026partialce,
  author    = {Ude, Oluwatifunmilayo},
  title     = {Partial cross-entropy for point-supervised land-cover
               segmentation: the effect of label budget and sampling strategy},
  year      = {2026},
  version   = {v1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22861209},
  url       = {https://github.com/FunmilayoUde/point-supervised-segmentation}
}
```

## Licence

The code in this repository is MIT licensed; see [LICENSE](LICENSE).

That does **not** extend to the data or to figures derived from it.

LoveDA is distributed under CC BY-NC-SA 4.0 and is licensed for academic use
only. Imagery originates from the Google Earth platform and remains subject to
its terms. Cite Wang et al., *LoveDA: A Remote Sensing Land-Cover Dataset for
Domain Adaptive Semantic Segmentation*, NeurIPS 2021 Datasets and Benchmarks
Track ([arXiv:2110.08733](https://arxiv.org/abs/2110.08733)).

The loss is from Tang et al., *Normalized Cut Loss for Weakly-supervised CNN
Segmentation*, CVPR 2018 ([arXiv:1804.01346](https://arxiv.org/abs/1804.01346)).
Full references are in the report.
