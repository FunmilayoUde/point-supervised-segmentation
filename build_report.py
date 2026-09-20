"""Assemble the technical report as a PDF.

Equations are rendered through matplotlib's mathtext rather than typed as
ASCII, and the figures are the same PNGs the experiment scripts produce, so
the document regenerates from the results without a manual step.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

FONTS = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
REPORT = OUT / "report"
PDF = REPORT / "Partial_CE_Point_Supervision_Report.pdf"
REPORT.mkdir(parents=True, exist_ok=True)


def render_equation(tex, path, fontsize):
    """Typeset one equation to a transparent PNG via mathtext."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=fontsize, color="black")
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


render_equation(r"$L(p,\,y) \;=\; -\,(1-p_y)^{\gamma}\,\log p_y$", REPORT / "eq1.png", 17)
render_equation(r"$\mathrm{pfCE} \;=\; \frac{\sum (L \odot M)}{\sum M}$", REPORT / "eq2.png", 19)

for name, fn in [("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"),
                 ("Serif-Italic", "DejaVuSerif-Italic.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                 ("Sans", "DejaVuSans.ttf"), ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, fn)))

INK, MUTED, RULE = colors.HexColor("#15191c"), colors.HexColor("#5a6168"), colors.HexColor("#c9ced3")

def S(name, **kw):
    base = dict(fontName="Serif", fontSize=9.6, leading=14.6, textColor=INK,
                spaceAfter=7, alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)

BODY     = S("body")
FIRST    = S("first", spaceAfter=7)
TITLE    = S("title", fontName="Sans-Bold", fontSize=16.5, leading=21.5,
             alignment=TA_CENTER, spaceAfter=10)
BYLINE   = S("byline", fontSize=9.8, alignment=TA_CENTER, textColor=MUTED, spaceAfter=2)
H1       = S("h1", fontName="Sans-Bold", fontSize=12.6, leading=16, spaceBefore=15,
             spaceAfter=6, alignment=0, keepWithNext=1)
H2       = S("h2", fontName="Sans-Bold", fontSize=10.4, leading=13.5, spaceBefore=10,
             spaceAfter=4, alignment=0, keepWithNext=1)
CAP      = S("cap", fontSize=8.5, leading=12, alignment=TA_CENTER, textColor=MUTED,
             spaceBefore=5, spaceAfter=10)
TCAP     = S("tcap", fontSize=8.5, leading=12, alignment=0, textColor=MUTED,
             spaceBefore=8, spaceAfter=5)
BULLET   = S("bullet", leftIndent=15, bulletIndent=4, spaceAfter=4)
REF      = S("ref", fontSize=8.8, leading=12.6, leftIndent=22, firstLineIndent=-22,
             spaceAfter=5, alignment=0)

def img(path, width):
    w, h = PILImage.open(path).size
    return Image(path, width=width, height=width * h / w)

def para(text, style=BODY):
    return Paragraph(text, style)

def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]

def table(data, widths, align_right=(), header=False):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Serif"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("LEADING", (0, 0), (-1, -1), 11.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
    ]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), "Sans-Bold"),
                  ("FONTSIZE", (0, 0), (-1, 0), 8.0),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK)]
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


doc = BaseDocTemplate(str(PDF), pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm,
                      topMargin=2.1*cm, bottomMargin=2.1*cm,
                      title="Partial Cross-Entropy for Point-Supervised Land-Cover Segmentation",
                      author="Oluwatifunmilayo Ude")
W = doc.width

def footer(canvas, d):
    canvas.saveState()
    canvas.setFont("Serif", 8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0] / 2, 1.3*cm, str(d.page))
    canvas.restoreState()

doc.addPageTemplates([PageTemplate(id="main",
    frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")],
    onPage=footer)])

s = []
A = s.append

A(para("Partial Cross-Entropy for Point-Supervised Land-Cover Segmentation: "
       "The Effect of Label Budget and Sampling Strategy", TITLE))
A(para("Oluwatifunmilayo Ude", BYLINE))
A(para("20 September 2026", BYLINE))
A(Spacer(1, 14))

# ---------------------------------------------------------------- 1
A(para("1. Introduction", H1))
A(para("Semantic segmentation requires a class label for every pixel. Training such a model "
  "normally requires ground truth of the same form: a mask in which a human has labelled every "
  "pixel. For remote sensing imagery this is costly. A LoveDA tile is 1024 &#215; 1024 pixels, and the "
  "urban training split contains 1,156 tiles. Producing dense masks at this scale is frequently "
  "the limiting factor in a land-cover mapping project."))
A(para("Point annotation reduces that cost. An annotator clicks a small number of pixels per image "
  "and labels each one. The label map that results is almost entirely unobserved. With twenty "
  "points on a 256 &#215; 256 image, 20 pixels of 65,536 are labelled and 65,516 are not. Those "
  "unlabelled pixels carry no class assignment of any kind; they are not background. Cross-entropy "
  "provides no mechanism for representing an unknown label, and assigning unlabelled pixels to a "
  "background index introduces incorrect supervision across more than 99.9% of the image."))
A(para("Partial cross-entropy restricts the loss to annotated pixels. The per-pixel loss is "
  "multiplied by a binary mask of labelled positions, summed, and divided by the number of "
  "labelled pixels. Gradients at unlabelled positions are zero. The choice of denominator is "
  "important here. Normalising by the total pixel count would make the loss magnitude decrease as "
  "annotation grows sparser, so the effective gradient scale would vary with the annotation budget "
  "rather than being held constant across it."))
A(para("Two questions remain open once the loss is defined: how many points are needed, and how "
  "they should be chosen. This report implements partial cross-entropy with focal weighting, "
  "simulates point labels by sampling from dense LoveDA masks, and evaluates both factors. Points "
  "per tile are varied over 5, 20, 50 and 200. Points are drawn either uniformly at random or with "
  "the budget divided equally between the classes present in a tile. Thirty-three training runs "
  "cover this grid, with a fully supervised model and a majority-class predictor as upper and lower "
  "reference points. At 200 points per tile, approximately 0.3% of pixels, mIoU reaches 92% of the "
  "fully supervised result. Class-balanced sampling does not improve mean mIoU at any budget "
  "tested. It does reduce the standard deviation across seeds by roughly a factor of three, which "
  "suggests its value lies in consistency rather than in accuracy."))

A(para("1.1 Related work", H2))
A(para("Point supervision for segmentation was introduced by Bearman et al. [4], who showed that "
  "under a fixed annotation budget a model trained from single points per object outperforms one "
  "trained from image-level labels. The loss used here, cross-entropy evaluated only at labelled "
  "pixels, was formalised as partial cross-entropy by Tang et al. [2], who paired it with a "
  "normalised-cut regulariser over the unlabelled pixels; a companion paper extends the approach to "
  "other regularisers [3]. The focal weighting is due to Lin et al. [7]."))
A(para("This report does not propose a new loss. It takes partial cross-entropy as given and "
  "measures how its performance depends on two properties of the annotation itself: how many points "
  "are collected per image, and how those points are distributed across classes. The regularisation "
  "terms that accompany partial cross-entropy in [2] and [3] are deliberately omitted, so that the "
  "effect of the annotation budget is not confounded with the effect of a second loss term."))

# ---------------------------------------------------------------- 2
A(para("2. Method", H1))
A(para("2.1 Partial cross-entropy", H2))
A(para("Let <i>p</i> be the predicted class distribution at a pixel and <i>y</i> its ground-truth "
  "class. The focal loss [7] weights the standard cross-entropy term by how confident the prediction "
  "already is:"))
A(Spacer(1, 3)); A(img(f"{OUT}/report/eq1.png", 268)); A(Spacer(1, 6))
A(para("with &#947; = 2 throughout. Setting &#947; = 0 recovers ordinary cross-entropy."))
A(para("Partial cross-entropy applies this per-pixel loss only where a label exists. Writing "
  "<i>M</i> for the binary mask that is 1 at annotated pixels and 0 elsewhere, and summing over all "
  "pixels in the batch:"))
A(Spacer(1, 3)); A(img(f"{OUT}/report/eq2.png", 132)); A(Spacer(1, 6))
A(para("Two properties follow. Pixels with <i>M</i> = 0 contribute nothing to the numerator and "
  "receive no gradient, so the network is unconstrained wherever no annotation exists. The "
  "denominator counts annotated pixels rather than total pixels, which holds the magnitude of the "
  "loss independent of annotation density."))
A(para("The second property is what makes a comparison across budgets valid. If the denominator "
  "were <i>H</i> &#215; <i>W</i>, a tile with 5 labelled pixels would yield a loss forty times "
  "smaller than one with 200, and gradient magnitude would fall as the budget fell. Any difference "
  "measured between budgets would then be partly a difference in effective learning rate rather "
  "than a difference in supervision."))

A(para("2.2 Representing unlabelled pixels", H2))
A(para("LoveDA encodes its seven classes as 1&#8211;7 and reserves 0 for regions with no data. "
  "Labels are remapped at load time to 0&#8211;6, with no-data assigned the sentinel 255. Point "
  "sampling draws only from pixels that are not 255, so a simulated point cannot land on a region "
  "that was never annotated. The same sentinel marks pixels excluded by sampling, so the loss and "
  "the evaluation metric operate under one convention rather than two."))

A(para("2.3 Verification", H2))
A(para("The implementation was checked against PyTorch's [10] own cross-entropy before being used:"))
A(KeepTogether(bullets([
  'With &#947; = 0 and a fully labelled mask, pfCE equals '
  '<font face="Mono" size="8.6">F.cross_entropy(reduction="mean")</font> to floating-point precision.',
  'With &#947; = 0 and a partially labelled mask, it equals '
  '<font face="Mono" size="8.6">F.cross_entropy(ignore_index=255)</font>.',
  'Gradients with respect to the logits are exactly zero at unlabelled positions.',
  'For fixed predictions, the loss value is invariant to the number of sampled points.'])))
A(para("The first two checks constrain the masking and the normalisation independently."))

# ---------------------------------------------------------------- 3
A(para("3. Experimental setup", H1))
A(para("3.1 Data and splits", H2))
A(para("LoveDA [1] contains 5,987 aerial images at 0.3 m ground sampling distance, collected over "
  "Nanjing, Changzhou and Wuhan and divided into urban and rural scenes. Seven land-cover classes "
  "are annotated: background, building, road, water, barren, forest and agriculture. All "
  "experiments here use the urban scene only. The two scenes were constructed to differ in class "
  "distribution for domain-adaptation benchmarking, so combining them would introduce a source of "
  "variation alongside the two factors under study."))
A(para("The published validation split was not used. Train/Urban is 49.7% background and 2.1% "
  "agriculture by valid pixel count, while Val/Urban is 26.0% and 25.7%. The two cover different "
  "geographic areas. That separation is deliberate in a dataset built for domain adaptation, but it "
  "is unsuitable here, because a difference in mIoU between two point budgets would be confounded "
  "with a change in the class distribution itself."))
A(para("Training and validation tiles were instead drawn from Train/Urban by a fixed random "
  "permutation (seed 1234): 400 tiles for training and a disjoint 250 for validation. Class "
  "frequencies agree between the two to within 2.5 percentage points on every class."))

A(para("3.2 Simulating point annotation", H2))
A(para("Dense masks exist for every tile, so point supervision is simulated by sampling from them "
  "and discarding the rest. Tiles are resized from 1024 &#215; 1024 to 256 &#215; 256 before "
  "sampling, using bilinear interpolation for the image and nearest-neighbour for the mask. "
  "Sampling after the resize keeps the budget counted against the image the network actually "
  "receives."))
A(para("Two strategies were compared. Uniform sampling draws <i>N</i> pixels at random from the "
  "labelled pixels of a tile, without replacement. Stratified sampling divides <i>N</i> equally "
  "among the classes present in that tile and distributes any remainder to randomly chosen classes. "
  "A class absent from a tile cannot be sampled under either strategy."))
A(para("Each tile's points come from a random stream keyed on the pair (seed, tile index), so a "
  "tile yields identical points on every epoch. Redrawing them each epoch would expose the network "
  "to <i>N</i> &#215; epochs distinct labels over a run, which would make the nominal budget "
  "meaningless."))

A(para("3.3 Model and training", H2))
A(para("The network is DeepLabv3 [5] with a MobileNetV3-Large [6] backbone initialised from "
  "ImageNet [9] weights, with a randomly initialised head sized to seven classes. Pretraining is a precondition "
  "rather than a tuning choice, since twenty labelled pixels per image cannot train an encoder from "
  "scratch."))
A(para("Batch-normalisation statistics are frozen throughout and only the affine parameters "
  "trained. DeepLabv3's ASPP module pools to 1 &#215; 1 spatially, which leaves a single value per "
  "channel and makes the batch variance undefined at small batch sizes. At batch size 4 the running "
  "estimates would also be re-derived from very few samples, discarding the ImageNet statistics "
  "that motivate a pretrained backbone in the first place."))
A(para("A larger backbone was considered. ResNet-50 measured 1.92 s per training step on the "
  "available hardware against 0.24 s for MobileNetV3. At that cost the full grid would have taken "
  "roughly 56 hours rather than 2.5, and a controlled comparison across 33 runs was judged more "
  "informative than two runs with a larger encoder."))

cfg = [["Dataset", "LoveDA, urban scene"],
       ["Ground sampling distance", "0.3 m"],
       ["Tile size", "1024² resized to 256²"],
       ["Classes", "7, plus an ignored no-data label"],
       ["Training tiles", "400"],
       ["Validation tiles", "250, disjoint, seed 1234"],
       ["Architecture", "DeepLabv3, MobileNetV3-Large"],
       ["Initialisation", "ImageNet backbone, random head"],
       ["Batch normalisation", "Statistics frozen, affine trained"],
       ["Optimiser", "AdamW [8], lr 3 × 10⁻⁴"],
       ["Batch size", "4"],
       ["Epochs", "12"],
       ["Loss", "Partial focal cross-entropy, γ = 2"],
       ["Selection", "Best epoch by dense validation mIoU"]]
A(KeepTogether([para("<b>Table 1.</b> Experimental configuration.", TCAP),
                table(cfg, [W*0.36, W*0.64])]))

A(para("3.4 Evaluation", H2))
A(para("All evaluation is against dense masks on the held-out tiles. Points are used for training "
  "only."))
A(para("Predictions and ground truth are accumulated into a single 7 &#215; 7 confusion matrix over "
  "the whole validation split. Per-class IoU is computed as TP / (TP + FP + FN) and mIoU as their "
  "mean. Pixels carrying the ignore sentinel are excluded. Classes absent from the ground truth "
  "return no value rather than zero, so they do not depress the mean."))
A(para("Model selection uses the best dense validation mIoU across the 12 epochs; training loss is "
  "not used. A preliminary check showed why. Fitting a single tile to twenty points drove the "
  "training loss to 0.0002 while mIoU against that same tile's dense mask stood at 0.237."))

# ---------------------------------------------------------------- 4
A(para("4. Hypotheses", H1))
A(para("All three predictions were fixed before any model was trained. They follow from two "
  "measurements taken on the training split alone: the class distribution, and the probability that "
  "a class receives no points at all under a given budget."))
A(para("4.1 Basis", H2))
A(para("Class frequencies in the training split are heavily skewed. Measured over valid pixels: "
  "background 49.7%, building 21.3%, road 9.0%, barren 7.7%, forest 6.3%, water 3.9%, agriculture "
  "2.1%."))
A(para("Under uniform sampling each of <i>N</i> points lands on class <i>c</i> with probability "
  "equal to that class's share of the tile, so the probability a tile receives no point of class "
  "<i>c</i> is (1 &#8722; share<sub>c</sub>)<super>N</super>. Table 2 evaluates this at the budgets "
  "used."))
A(para("At five points per tile, agriculture is unlabelled in 90% of tiles and water in 82%. At "
  "twenty points the figures are 66% and 46%. A class the network is rarely shown is a class it "
  "cannot learn. Stratified sampling removes this failure for any class present in a tile, since it "
  "guarantees every present class at least one point whenever <i>N</i> is at least the number of "
  "classes present."))

cov = [["Class", "Share", "N = 5", "N = 20", "N = 50", "N = 200"],
       ["Agriculture", "2.1%", "90.0%", "65.5%", "34.8%", "1.5%"],
       ["Water", "3.9%", "82.1%", "45.5%", "14.0%", "0.0%"],
       ["Forest", "6.3%", "72.1%", "27.0%", "3.8%", "0.0%"],
       ["Barren", "7.7%", "66.9%", "20.0%", "1.8%", "0.0%"],
       ["Road", "9.0%", "62.5%", "15.2%", "0.9%", "0.0%"],
       ["Building", "21.3%", "30.2%", "0.8%", "0.0%", "0.0%"],
       ["Background", "49.7%", "3.2%", "0.0%", "0.0%", "0.0%"]]
A(KeepTogether([
  para("<b>Table 2.</b> Probability that a tile receives no point of a given class under uniform "
       "sampling.", TCAP),
  table(cov, [W*0.22, W*0.13, W*0.1625, W*0.1625, W*0.1625, W*0.1625],
        align_right=(1, 2, 3, 4, 5), header=True)]))

A(para("4.2 H1: performance rises with budget and saturates", H2))
A(para("mIoU should increase with points per tile, with diminishing returns at the top of the "
  "range. The curve should climb steeply between 5 and 50 points and flatten by 200. Falsified by a "
  "flat curve across the range, or one still rising linearly at 200 points."))
A(para("4.3 H2: stratified sampling outperforms uniform at small budgets", H2))
A(para("Uniform sampling leaves rare classes unlabelled in most tiles at small <i>N</i>; stratified "
  "sampling does not. The advantage should be largest at <i>N</i> = 5 and should shrink as <i>N</i> "
  "grows, since uniform coverage improves with budget while stratified coverage is already "
  "complete. Falsified by uniform matching or exceeding stratified at <i>N</i> = 5."))
A(para("4.4 H3: rare classes stay poorly segmented under both strategies", H2))
A(para("Stratification can only allocate points to classes that appear in a tile. Whether a class "
  "appears at all is a property of the data, not of the sampling rule. Agriculture is absent from a "
  "large fraction of urban tiles however points are drawn, so its IoU should remain well below that "
  "of common classes at every budget and under both strategies. Water should behave the same way, "
  "less severely. Falsified by either class approaching the IoU that full supervision achieves on "
  "it."))

# ---------------------------------------------------------------- 5
A(para("5. Results", H1))
A(para("5.1 Validation of the training loop", H2))
A(para("Before any comparison, a single tile was fitted twice to confirm the pipeline could learn "
  "at all. Trained on that tile's dense mask, the model reached 0.902 mIoU against it. Trained on "
  "twenty points from the same tile, training loss fell to 0.0002 and mIoU on those twenty pixels "
  "reached 1.000, while mIoU against the dense mask was 0.237."))
A(para("The first run rules out faults in the optimiser, the gradient path and the data loader. The "
  "second isolates the problem the rest of this report measures: a model can fit its supervision "
  "perfectly and still recover little of the scene."))

A(para("5.2 Effect of label budget", H2))
A(para("Full supervision reached 0.417 &#177; 0.031 mIoU over three seeds. A majority-class "
  "predictor scored 0.068. All point-supervised results fall between these."))

res = [["Points per tile", "Uniform", "Stratified"],
       ["5", "0.215 ± 0.017", "0.157 ± 0.023"],
       ["20", "0.309 ± 0.071", "0.266 ± 0.044"],
       ["50", "0.341 ± 0.084", "0.346 ± 0.024"],
       ["200", "0.384 ± 0.035", "0.346 ± 0.012"],
       ["Fully supervised", "0.417 ± 0.031", ""],
       ["Majority class", "0.068", ""]]
A(KeepTogether([
  para("<b>Table 3.</b> mIoU by budget and sampling strategy, mean ± standard deviation across "
       "seeds. <i>N</i> = 50 uses six seeds per strategy; all other cells use three.", TCAP),
  table(res, [W*0.34, W*0.33, W*0.33], align_right=(1, 2), header=True)]))

A(para("Uniform sampling rises monotonically across the range and flattens at the top: 0.215, "
  "0.309, 0.341, 0.384. At 200 points per tile it reaches 92% of the fully supervised result using "
  "200 labelled pixels out of 65,536, about 0.3% of the image. <b>H1 is supported.</b>"))
A(para("Stratified sampling rises to 0.346 at 50 points and does not move at 200. Equal per-class "
  "quotas impose a ceiling of their own, since raising <i>N</i> cannot increase how often the "
  "common classes are seen relative to the rare ones."))
A(KeepTogether([img(f"{OUT}/fig1_budget_curve.png", W*0.86),
  para("<b>Figure 1.</b> mIoU against label budget for both sampling strategies. Error bars show "
       "standard deviation across seeds. The fully supervised ceiling and majority-class floor are "
       "marked.", CAP)]))

A(para("5.3 Effect of sampling strategy", H2))
A(para("Stratified sampling was worse than uniform at 5 points (&#8722;0.058) and at 20 points "
  "(&#8722;0.044), level at 50, and worse again at 200 (&#8722;0.037). It did not exceed uniform at "
  "any budget by more than the seed spread. <b>H2 is not supported, and the effect runs opposite to "
  "the prediction: stratification was worst exactly where it was expected to help most.</b>"))
A(para("The per-class figures at <i>N</i> = 5 show the mechanism. With seven classes present in a "
  "typical tile, stratification gives each class one pixel. Agriculture moves from 0.0 to 0.9 IoU, "
  "which is no gain in practice, while water falls from 52.3 to 27.8 and background from 36.8 to "
  "27.5."))

pc = [["Configuration", "Backg", "Build", "Road", "Water", "Barren", "Forest", "Agri"],
      ["Fully supervised", "48.6", "37.3", "45.4", "66.8", "31.8", "41.9", "20.3"],
      ["Uniform, N = 5", "36.8", "28.5", "13.9", "52.3", "14.5", "4.3", "0.0"],
      ["Stratified, N = 5", "27.5", "25.8", "12.1", "27.8", "7.2", "8.6", "0.9"],
      ["Uniform, N = 200", "45.8", "38.1", "38.5", "64.1", "33.0", "34.1", "15.1"],
      ["Stratified, N = 200", "28.2", "40.3", "41.3", "53.4", "30.0", "30.2", "19.1"]]
A(KeepTogether([
  para("<b>Table 4.</b> Per-class IoU (×100) on held-out dense masks.", TCAP),
  table(pc, [W*0.235] + [W*0.1093]*7, align_right=tuple(range(1, 8)), header=True)]))

A(para("A single labelled pixel is not enough to learn a class from, but the pixels given up to "
  "provide it were enough to lose ground on the classes that already had signal. Stratification "
  "only pays once the per-class quota is large enough to be informative, which on this data is "
  "above twenty points per tile."))

A(para("5.4 Rare classes", H2))
A(para("At 200 points, water reaches 64.1 against a fully supervised 66.8, and agriculture 19.1 "
  "against 20.3. Both are within 96% and 94% of the ceiling. <b>H3 is not supported.</b>"))
A(para("The prediction failed for a reason not anticipated when it was made. Agriculture scores "
  "20.3 under full supervision, far below every other class. Its low IoU is a property of the class "
  "in this data rather than a consequence of sparse supervision, and the hypothesis mistook a low "
  "ceiling for a supervision failure. Point supervision recovers agriculture about as well as it "
  "recovers anything else."))
A(KeepTogether([img(f"{OUT}/fig2_class_breakdown.png", W*0.86),
  para("<b>Figure 2.</b> Per-class IoU for both strategies at 200 points per tile, against the "
       "fully supervised ceiling. Classes are ordered by fully supervised IoU.", CAP)]))

A(para("5.5 Variance across seeds", H2))
A(para("Stratified sampling produced consistently tighter results. At 50 points, measured over six "
  "seeds per strategy, uniform had a standard deviation of 0.084 against 0.024 for stratified, and "
  "a range of 0.230 against 0.063. At 200 points the ratio was about three to one."))
A(para("The uniform runs at <i>N</i> = 50 were 0.396, 0.365, 0.177, 0.364, 0.336 and 0.407. The low "
  "run is isolated: the remaining five average 0.374 and the median across all six is 0.365. "
  "Stratified runs at the same budget spanned 0.305 to 0.367."))
A(para("That single run is the failure mode stratification is designed to prevent, and adding three "
  "seeds per strategy made the difference in spread more credible rather than less. The value of "
  "class-balanced sampling on this data is not a higher expected score but a narrower distribution "
  "of outcomes."))

A(para("5.6 Qualitative behaviour", H2))
A(para("Predictions from a point-supervised model place regions correctly and bound them poorly. "
  "Roads, large cleared areas and building clusters appear in approximately the right positions, "
  "but predicted regions are smooth where the ground truth is angular, and thin features such as "
  "narrow watercourses are lost."))
A(para("Sampled points sit in the interiors of regions, so the loss receives no information about "
  "where one class stops and another begins. The residual gap to the fully supervised ceiling is "
  "largely a boundary gap."))
A(KeepTogether([img(f"{OUT}/fig3_qualitative.png", W),
  para("<b>Figure 3.</b> Held-out tiles under uniform sampling at 200 points per tile. Left to "
       "right: image, dense ground truth, the 200 training points, and the prediction.", CAP)]))

# ---------------------------------------------------------------- 6
A(para("6. Limitations", H1))
A(para("Runs are not bit-reproducible. Retraining uniform sampling at 200 points with seed 0 "
  "produced 0.4304 mIoU against 0.3992 recorded in the sweep, with identical code and identical "
  "sampling. Reductions on the Metal backend are not deterministic. The seed spreads reported in "
  "Section 5.5 therefore conflate variation from the point draw with variation from the hardware, "
  "and the two cannot be separated from these runs."))
A(para("The fully supervised ceiling may be understated. The 0.4304 figure above exceeds the "
  "ceiling mean of 0.417, which sits inside its standard deviation but suggests 12 epochs may not "
  "be enough for the dense model to converge. If the ceiling is low, the 92% figure is generous."))
A(para("Most cells use three seeds. Only <i>N</i> = 50 was measured with six. Adjacent budgets "
  "should not be ordered on these data without more runs."))
A(para("Tiles were downsampled from 1024² to 256², reducing effective resolution from "
  "0.3 m to 1.2 m. Fine boundary structure is lost before training begins, which plausibly "
  "interacts with the boundary behaviour described in Section 5.6."))
A(para("One scene and one architecture were tested. Whether the ordering between strategies holds "
  "on rural tiles, or with a larger encoder, is untested."))

# ---------------------------------------------------------------- 7
A(para("7. Conclusion", H1))
A(para("Partial cross-entropy allows a segmentation network to be trained from sparse point labels "
  "by restricting the loss to annotated pixels and normalising by their count. On LoveDA urban "
  "tiles, 200 points per image recovers 92% of fully supervised mIoU from roughly 0.3% of the "
  "labels."))
A(para("Two of the three hypotheses were wrong. Class-balanced sampling was expected to help most "
  "at small budgets and instead hurt most there, because splitting a small budget across seven "
  "classes leaves each class with too little signal to learn from while removing signal from the "
  "classes that had it. Rare classes were expected to stay depressed under every strategy and did "
  "not; agriculture's weak IoU turned out to be a property of the class rather than of the "
  "supervision, since full supervision scores it barely higher."))
A(para("The result that was not predicted is the more useful one. Stratified sampling did not raise "
  "the expected score but reduced its spread by a factor of three, and the one uniform run that "
  "failed badly is the outcome stratification exists to avoid. For an annotation campaign, that "
  "reframes the choice from which strategy scores higher to which fails less often."))
A(para("The evidence supports uniform sampling where mean performance is the objective, stratified "
  "sampling where consistency across runs matters more, and neither strategy below about twenty "
  "points per tile, where a budget divided among seven classes teaches none of them."))

A(para("References", H1))
A(para("[1]&nbsp;&nbsp;Wang, J., Zheng, Z., Ma, A., Lu, X., Zhong, Y. (2021). <i>LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation.</i> NeurIPS Datasets and Benchmarks Track. arXiv:2110.08733.", REF))
A(para("[2]&nbsp;&nbsp;Tang, M., Djelouah, A., Perazzi, F., Boykov, Y., Schroers, C. (2018). <i>Normalized Cut Loss for Weakly-supervised CNN Segmentation.</i> CVPR. arXiv:1804.01346.", REF))
A(para("[3]&nbsp;&nbsp;Tang, M., Perazzi, F., Djelouah, A., Ben Ayed, I., Schroers, C., Boykov, Y. (2018). <i>On Regularized Losses for Weakly-supervised CNN Segmentation.</i> ECCV.", REF))
A(para("[4]&nbsp;&nbsp;Bearman, A., Russakovsky, O., Ferrari, V., Fei-Fei, L. (2016). <i>What's the Point: Semantic Segmentation with Point Supervision.</i> ECCV, pp. 549&#8211;565.", REF))
A(para("[5]&nbsp;&nbsp;Chen, L.-C., Papandreou, G., Schroff, F., Adam, H. (2017). <i>Rethinking Atrous Convolution for Semantic Image Segmentation.</i> arXiv:1706.05587.", REF))
A(para("[6]&nbsp;&nbsp;Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., et al. (2019). <i>Searching for MobileNetV3.</i> ICCV. arXiv:1905.02244.", REF))
A(para("[7]&nbsp;&nbsp;Lin, T.-Y., Goyal, P., Girshick, R., He, K., Doll&#225;r, P. (2017). <i>Focal Loss for Dense Object Detection.</i> ICCV. arXiv:1708.02002.", REF))
A(para("[8]&nbsp;&nbsp;Loshchilov, I., Hutter, F. (2019). <i>Decoupled Weight Decay Regularization.</i> ICLR. arXiv:1711.05101.", REF))
A(para("[9]&nbsp;&nbsp;Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., Fei-Fei, L. (2009). <i>ImageNet: A Large-Scale Hierarchical Image Database.</i> CVPR.", REF))
A(para("[10]&nbsp;&nbsp;Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019). <i>PyTorch: An Imperative Style, High-Performance Deep Learning Library.</i> NeurIPS.", REF))
A(Spacer(1, 8))
A(para("LoveDA is distributed under CC BY-NC-SA 4.0 and is licensed for academic use only. Imagery "
  "shown in Figure 3 originates from the Google Earth platform and remains subject to its terms of "
  "use.", S("note", fontSize=8.4, leading=12, textColor=MUTED)))

doc.build(s)
print("built:", PDF)
