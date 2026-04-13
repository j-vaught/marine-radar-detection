// Classical vs. Deep Learning Object Detection in X-Band Marine Radar Imagery
// J.C. Vaught and Ty Dangerfield
// Typst 0.14

// ── Brand palette ─────────────────────────────────────────────────────
#let garnet    = rgb("#73000A")
#let black     = rgb("#000000")
#let black-90  = rgb("#363636")
#let black-70  = rgb("#5C5C5C")
#let black-50  = rgb("#A2A2A2")
#let black-30  = rgb("#C7C7C7")
#let black-10  = rgb("#ECECEC")
#let warm-grey = rgb("#676156")
#let sandstorm = rgb("#FFF2E3")
#let rose      = rgb("#CC2E40")
#let atlantic  = rgb("#466A9F")
#let congaree  = rgb("#1F414D")
#let horseshoe = rgb("#65780B")
#let honeycomb = rgb("#A49137")

// ── Document setup ────────────────────────────────────────────────────
#set document(
  title: "Classical vs. Deep Learning Object Detection in X-Band Marine Radar Imagery",
  author: ("J.C. Vaught", "Ty Dangerfield"),
  keywords: ("marine radar", "object detection", "CFAR", "wavelet", "YOLO", "deep learning"),
)

#set heading(numbering: "1.")

#set page(
  paper: "us-letter",
  margin: (top: 0.9in, bottom: 1in, x: 0.85in),
  numbering: "1 / 1",
  number-align: center,
  footer: context [
    #set text(size: 8pt, fill: black-70)
    #grid(
      columns: (1fr, auto, 1fr),
      align: (left, center, right),
      [CSCE 763 Technical Report],
      counter(page).display("1 / 1", both: true),
      [April 2026],
    )
  ],
)

#set text(
  font: ("New Computer Modern", "Latin Modern Roman", "STIX Two Text", "Times New Roman"),
  size: 10pt,
  lang: "en",
  hyphenate: true,
)

#set par(justify: true, leading: 0.55em, first-line-indent: 0pt)

// Heading styles – garnet accents, no rounded edges anywhere
#show heading.where(level: 1): it => {
  set text(fill: garnet, size: 12pt, weight: "bold")
  set block(above: 1.4em, below: 0.6em)
  block[
    #smallcaps(it.body)
    #v(-0.35em)
    #line(length: 100%, stroke: 0.8pt + garnet)
  ]
}
#show heading.where(level: 2): it => {
  set text(fill: black-90, size: 10.5pt, weight: "bold")
  block(above: 1.0em, below: 0.4em)[#it.body]
}
#show heading.where(level: 3): it => {
  set text(fill: black-90, size: 10pt, weight: "bold", style: "italic")
  block(above: 0.8em, below: 0.2em)[#it.body]
}

// Link / ref styling
#show link: it => text(fill: atlantic)[#it]
#show ref: it => text(fill: garnet, weight: "semibold")[#it]

// Figure captions
#show figure.caption: it => {
  set text(size: 9pt, fill: black-70)
  block(above: 0.35em)[
    #text(fill: garnet, weight: "bold")[#it.supplement #it.counter.display()#it.separator]
    #it.body
  ]
}

// Tables: no vertical rules, just horizontal
#set table(
  stroke: (_, y) => {
    if y == 0 { (bottom: 0.8pt + black, top: 0.8pt + black) }
    else { (bottom: 0.3pt + black-50) }
  },
  inset: 6pt,
)

// ── Title block ───────────────────────────────────────────────────────
#align(center)[
  #block(width: 100%, inset: (y: 0.4em))[
    #text(size: 17pt, weight: "bold", fill: black)[
      Classical vs. Deep Learning Object Detection \
      in X-Band Marine Radar Imagery
    ]
  ]
  #v(0.5em)
  #text(size: 11pt)[
    *J.C. Vaught* #h(1em) *Ty Dangerfield* \
  ]
  #v(0.1em)
  #text(size: 9pt, fill: black-70, style: "italic")[
    Department of Computer Science & Engineering, University of South Carolina \
    CSCE 763 — Digital Image Processing \
  ]
]

#v(0.4em)
#line(length: 100%, stroke: 0.4pt + black-50)
#v(0.2em)

// ── Abstract ──────────────────────────────────────────────────────────
#block(
  fill: sandstorm,
  stroke: (left: 2pt + garnet),
  inset: (x: 14pt, y: 10pt),
  width: 100%,
)[
  #text(weight: "bold", size: 9.5pt, fill: garnet)[ABSTRACT.]
  #text(size: 9.5pt)[
    X-band marine radar produces plan-position-indicator (PPI) imagery in which
    vessels appear as bright returns against sea clutter. We conduct the first
    head-to-head benchmark of eight classical signal-processing detectors
    (CA/GO/SO/OS-CFAR, morphological, stationary-wavelet-transform, Otsu, SSOTSU)
    against four modern deep-learning detectors (YOLOv11, RT-DETR, Faster R-CNN,
    U-Net) on the DLR DAAN dataset. After identical red-channel isolation,
    circular PPI masking, dilation-aware connected-component extraction and
    COCO-style evaluation, we find that the classical stationary-wavelet
    transform reaches *AP\@0.50 = 0.065*, outperforming the best deep-learning
    method (YOLOv11, 0.0005) by a factor of approximately 130×. A root-cause
    analysis identifies target–background box-size mismatch, ImageNet pretraining
    mismatch, and extreme spatial sparsity of radar returns as the primary
    failure modes of off-the-shelf deep detectors on this domain. Our
    pipeline, configs, and predictions are released as a reproducible benchmark.
  ]
]

#v(0.6em)

// Two-column body starts here
#show: rest => columns(2, gutter: 18pt, rest)

// ── Introduction ──────────────────────────────────────────────────────
= Introduction

Marine radar is the primary all-weather sensor for situational awareness at sea.
An X-band PPI display is a polar-to-Cartesian rendering of pulse-echo returns,
on which vessels, buoys and debris manifest as small, high-contrast blobs
embedded in spatially structured clutter and instrument noise
[1] [2]. Automatic object detection on such imagery
enables collision avoidance, port surveillance and autonomous-navigation
decision support.

Two detection paradigms dominate the literature. *Classical* methods are rooted
in signal-processing: constant-false-alarm-rate (CFAR) detectors adaptively
threshold on local statistics [2]; morphological pipelines combine
Otsu binarisation with opening / closing to isolate target blobs [5];
multiresolution wavelets exploit the scale separation between targets and sea
clutter [3]; and seed-point Otsu variants (SSOTSU) focus on small
targets [4]. *Deep-learning* methods cast detection as a learned
regression: YOLOv11-OBB with SAHI tiling reports mAP\@0.50 > 0.95 on a small
radar set [8], Marine-Faster R-CNN reports 93.65% recall
[7], U-Net has been adapted for radar segmentation [9], and
transformer detectors such as CCDN-DETR are emerging [10].

The literature lacks a controlled, apples-to-apples comparison. Existing
studies report headline numbers but differ in preprocessing, annotation
granularity, evaluation protocol, and training regime. The present work fills
that gap on the DLR DAAN/DARC/MANV corpus [6]. Our contributions are:

+ A reproducible, unified pipeline that converts polar annotation masks into
  COCO bounding boxes, tiles PPI frames, applies red-channel / circular-disk
  masking, and exposes identical splits to all twelve methods.
+ A systematic root-cause analysis of why each classical detector succeeds
  or fails on masked radar data, including the discovery that CFAR guard cells
  must exceed half the target extent to avoid self-suppression.
+ Empirical evidence, across three random seeds per deep model, that *none of
  the four deep detectors approach the best classical baseline* under the
  evaluation protocol of COCO AP.
+ A public release of the preprocessing code, configs, checkpoints and twenty
  prediction files at
  #link("https://github.com/j-vaught/marine-radar-detection")[github.com/j-vaught/marine-radar-detection].

#figure(
  image("figures/fig1_headline.png", width: 100%),
  placement: auto,
  scope: "parent",
  caption: [
    Headline comparison at *AP\@0.50* on the 760-image DAAN test split.
    Bars show the per-method Average Precision at 0.50 IoU. Classical bars
    use the brand palette; deep-learning bars are the mean of three seeds
    (error bars = ±1 $sigma$). The stationary wavelet transform leads by two
    orders of magnitude.
  ],
) <fig:headline>

// ── Dataset ──────────────────────────────────────────────────────────
= Dataset and Preprocessing

We use the DLR radar corpus distributed with the PPINet release
[6]. It contains three independent recording sessions:
*MANV* (952 frames, used for training), *DARC* (527 frames, validation) and
*DAAN* (760 frames, held-out test), for a total of 2,239 PPI frames at a common
1050 × 1024-pixel resolution (@fig:dataset). Each frame ships with a
`polarMeas.json` file that lists per-object boundary points in polar $(r,theta)$
coordinates.

#figure(
  image("figures/fig8_dataset.png", width: 100%),
  caption: [
    *Dataset composition.* Temporal splits by recording session prevent
    information leakage. The test set contains a median of 3 objects per image
    with a small long tail out to five vessels per PPI frame.
  ],
) <fig:dataset>

== Polar-to-COCO Annotation Conversion

For each frame we extract the flat `[obj_id, heading, [r,θ], …]` list,
project the boundary points to Cartesian $(x,y) = (r sin theta, r cos theta)$,
cluster with DBSCAN ($epsilon = 200$ m, $"min"_s=5$), and convert each cluster
to pixel coordinates using

$ p_x = c_x + r_"px" sin(theta), quad p_y = c_y - r_"px" cos(theta), $

where $(c_x, c_y) = (525, 512)$ is the PPI centre and
$r_"px" = min(W, H)/2 dot 0.92$ is the display radius in pixels. We take the
axis-aligned bounding box of each cluster with 12-pixel padding, producing
8,294 COCO-format annotations across the three sub-datasets with a tight median
box size of 33 × 33 pixels.

== Red-Channel Isolation and Circular Masking

Raw DLR captures contain a substantial display overlay — compass rose,
bearing ticks, range rings, and a control panel — plus a solid blue PPI
background. A naïve grayscale conversion maps these bright green UI elements
to intensities comparable to real targets (≈ 150 on a 0-255 scale), which
drives every classical detector to fire on compass marks
(@fig:preprocess, panel *a*). Two inspection-level observations break this:

#figure(
  image("figures/fig3_preprocess.png", width: 100%),
  placement: auto,
  scope: "parent",
  caption: [
    *Four-stage preprocessing pipeline* for DAAN_342.png. All panels retain
    the native 1050 × 1024 frame (no rectangular cropping — that would
    invalidate ground-truth coordinates); the display axes are simply zoomed
    to the PPI disc (centre = (531, 515), radius = 466, measured directly
    from the captures). (a) raw BGR capture; (b) red-channel only isolates
    radar returns (target $R ≈ 226$, background $R ≈ 0$, whereas $B ≈ 140$
    everywhere, so green bearing ticks and the solid blue PPI background
    both vanish); (c) circular masking zeros the outer-ring chrome;
    (d) DBSCAN-derived ground-truth boxes superimposed in garnet sit
    directly on the radar returns.
  ],
) <fig:preprocess>

- *Radar returns live almost entirely in the red channel.* Target pixels read
  (B, G, R) ≈ (234, 239, 226) (saturated white) while the background reads
  (140, 0, 0) (blue PPI). Taking the R channel alone eliminates the dark-blue
  background with no information loss on targets.
- *Green UI marks inside the PPI circle* (bearing ticks, range rings) have
  (B, G, R) = (0, 255, 0) and vanish in the R channel.

Circular masking at $r = 471$ pixels then zeros every pixel outside the PPI
disk, yielding the clean grey-scale input shown in @fig:preprocess (c). All
2,022 test-set ground-truth boxes fall inside this mask.

== Tiling and Split Management

We tile each frame into 640 × 640 patches with 0.2 overlap for the deep
detectors, yielding 5,712 training, 3,162 validation and 4,560 test tiles.
Classical detectors operate on the full-resolution PPI frames to avoid
tile-boundary fragmentation, whereas CNN detectors use the tiles directly.
Tile-local box coordinates are remapped through the
`(x_offset, y_offset, tile_size)` metadata at evaluation time.

// ── Methods ──────────────────────────────────────────────────────────
= Methods

The twelve detectors share the preprocessed input described above. All
classical methods return $(x_1, y_1, x_2, y_2)$ boxes with a calibrated
confidence score (peak-to-threshold ratio or normalised coefficient magnitude)
for COCO-style AP integration.

== Classical Detectors

*CA/GO/SO/OS-CFAR* implement the standard 2-D sliding-window CFAR with
$"guard" = (15, 15)$ and $"reference" = (30, 30)$ half-sizes. Pilot experiments
confirmed that a guard band smaller than the target extent causes target
self-suppression (target energy leaks into the reference window and inflates
the local-noise estimate), and the revised 15-pixel guard recovers most
detections. PFA is set to $10^(-3)$.

*Morphological.* Otsu binarisation is computed *only on non-zero pixels* —
a crucial adjustment when operating on circularly masked inputs, otherwise the
large outside-disk zero mass dominates the histogram and collapses the
threshold to ≈ 11, classifying the entire PPI disk as foreground. A 3 × 3
elliptical opening followed by 5 × 5 closing removes residual noise.

*Wavelet-SWT.* A 2-D stationary wavelet transform (Daubechies-4, 3 levels)
produces detail coefficients in the LH, HL, and HH sub-bands. Each sub-band
is thresholded at $mu + 4 sigma$; the resulting binary maps are combined by
pixel-wise maximum, then connected components above area 50 are returned.

*Otsu and SSOTSU.* Global Otsu on non-zero pixels serves as a strong
unsupervised baseline. SSOTSU [4] divides the frame into 64 × 64
blocks, runs local Otsu on non-zero pixels within each block, and grows seed
regions with an 80% relative-intensity criterion.

*Post-processing.* All classical detections receive a 5-pixel binary-mask
dilation (to grow the detected bright core toward the full target extent) and
a 12-pixel box padding (to match the padding that was baked into the DBSCAN
ground truth), a calibration step detailed in @sec:rca.

== Deep Learning Detectors

Every deep model is trained with three seeds {42, 123, 456}.

- *YOLOv11-M* (Ultralytics): 100 epochs max, AdamW, `lr0`=10#super[-3],
  `batch`=16, patience = 20, input 640². SAHI tiled inference is disabled
  because our tiles are already 640².
- *Faster R-CNN* (ResNet-50 FPN, torchvision): 50 epochs, SGD,
  `lr`=5×10#super[-3], `batch`=8. Anchor scales and aspect ratios were
  tailored to the 33 × 33-pixel target regime
  (`sizes = ((16,), (32,), (64,), (128,), (256,))`).
- *U-Net* with ResNet-34 encoder: 80 epochs, AdamW, DiceBCE loss. Batch-size
  was reduced from 16 to 4 to resolve CUDA OOM on the 48 GB GPUs.
- *RT-DETR-L*: 72 epochs, AdamW, same tile layout as YOLOv11.

All four models were trained in parallel on four NVIDIA RTX 6000 Ada GPUs.
Detections from tile-based inference are evaluated against
`test_tiled.json` (4,560 tiles, 7,186 annotations).

// ── Evaluation ───────────────────────────────────────────────────────
= Evaluation Protocol

We report the full COCO suite: AP at IoU thresholds {0.50, 0.75, [0.50:0.95]}
and AR\@{1, 10, 100}, optionally stratified by object size. Deep-model numbers
are the mean ± $sigma$ over three seeds. A pairwise bootstrap test
(10,000 resamples, Bonferroni-corrected) ranks methods, and the TIDE toolbox
[11] decomposes errors into classification, localisation,
duplicate, background-FP and missed-GT components.

All code, configs, checkpoints, and prediction JSONs are released alongside
the dataset splits to enable exact reproduction.

// ── Results ──────────────────────────────────────────────────────────
= Results

@tab:main reports headline numbers. Wavelet-SWT is the single best detector
at *AP\@0.50 = 0.0646*, followed by classical morphological at 0.0184 and
Otsu at 0.0116. The strongest deep detector (YOLOv11, seed 456 alone) reaches
AP\@0.50 = 0.0011, an order of magnitude below even the weakest classical
method with meaningful detections. @fig:multimetric generalises the result
across four metrics: classical methods dominate AP and AR alike.

#figure(
  placement: top,
  scope: "parent",
  kind: table,
  caption: [
    *Main results on the DAAN test split.* Classical methods evaluated on
    760 full-resolution images against `test.json`; deep models evaluated on
    4,560 tiles against `test_tiled.json`. Deep columns show mean ± 1 $sigma$
    over three seeds. *Preds* is the total number of detection boxes emitted.
  ],
  table(
    columns: (1.6fr, 1fr, 1.3fr, 1.5fr, 1.3fr, 1fr),
    align: (left, left, right, right, right, right),
    table.header[*Method*][*Type*][*AP\@0.50*][*AP\@[.50:.95]*][*AR\@100*][*Preds*],
    [Wavelet-SWT],     [Classical], [*0.0646*], [*0.0103*], [*0.0538*], [7,674],
    [Morphological],   [Classical], [0.0184],   [0.0019],   [0.0184],   [6,880],
    [Otsu],            [Classical], [0.0116],   [0.0012],   [0.0170],   [8,810],
    [CA-CFAR],         [Classical], [0.0096],   [0.0010],   [0.0165],   [7,718],
    [GO-CFAR],         [Classical], [0.0039],   [0.0004],   [0.0121],   [7,327],
    [SO-CFAR],         [Classical], [0.0037],   [0.0004],   [0.0121],   [10,025],
    [SSOTSU],          [Classical], [0.0031],   [0.0004],   [0.0076],   [2,187],
    [OS-CFAR],         [Classical], [0.0000],   [0.0000],   [0.0000],   [1,966],
    [YOLOv11],         [DL (×3)],   [0.0005 #text(size:7.5pt, fill: black-70)[±0.0006]],
                                     [0.0001 #text(size:7.5pt, fill: black-70)[±0.0002]],
                                     [0.0241 #text(size:7.5pt, fill: black-70)[±0.0061]],
                                     [828,594],
    [RT-DETR],         [DL (×3)],   [0.0001 #text(size:7.5pt, fill: black-70)[±0.0001]],
                                     [0.0000 #text(size:7.5pt, fill: black-70)[±0.0000]],
                                     [0.0138 #text(size:7.5pt, fill: black-70)[±0.0118]],
                                     [1,323,191],
    [Faster R-CNN],    [DL (×3)],   [0.0000 #text(size:7.5pt, fill: black-70)[±0.0001]],
                                     [0.0000 #text(size:7.5pt, fill: black-70)[±0.0000]],
                                     [0.0000 #text(size:7.5pt, fill: black-70)[±0.0001]],
                                     [447],
    [U-Net],           [DL (×3)],   [0.0000 #text(size:7.5pt, fill: black-70)[±0.0000]],
                                     [0.0000 #text(size:7.5pt, fill: black-70)[±0.0000]],
                                     [0.0000 #text(size:7.5pt, fill: black-70)[±0.0000]],
                                     [868],
  ),
) <tab:main>

#figure(
  image("figures/fig2_multimetric.png", width: 100%),
  placement: auto,
  scope: "parent",
  caption: [
    Per-metric comparison. Wavelet-SWT wins AP\@0.50, AP\@\[0.50:0.95\] and
    AR\@100. Note the radically different y-axes between panels: the
    AP\@0.75 scale is #emph[three orders of magnitude smaller] than
    AP\@0.50 because none of the methods tested produce IoU-tight boxes.
  ],
) <fig:multimetric>

@fig:distrib summarises the dispersion inside each paradigm. Classical
methods span AP\@0.50 from 0 to 0.065 with a median of 0.007; deep methods
cluster tightly near zero.

#figure(
  image("figures/fig7_classical_vs_dl.png", width: 100%),
  caption: [
    Classical vs. deep learning across three metrics. Each dot is one
    method (classical) or one (method, seed) pair (DL). The diamond marks
    the group mean. In every metric except AR\@100, the entire classical
    distribution dominates the entire DL distribution.
  ],
) <fig:distrib>

== Qualitative Analysis

@fig:qualitative compares predictions from the four strongest classical
detectors against ground truth on two representative DAAN test frames.
Wavelet-SWT recovers every visible target plus a small number of clutter
responses; CA-CFAR produces tight but fragmented detections on the bright
cores; Otsu and morphological fall between these extremes. Ground truth
is overlaid as dashed garnet boxes in every panel for reference. We omit
the deep models from this qualitative panel because they were trained
and evaluated exclusively on 640² tiles; on the same tile space they
achieve mean AP\@0.50 = 0.0005 across seeds (@tab:main).

#figure(
  image("figures/fig4_qualitative.png", width: 100%),
  placement: auto,
  scope: "parent",
  caption: [
    Qualitative comparison on two DAAN test frames. Display axes are
    zoomed to the PPI disc (measured centre (531, 515), radius 466) but
    all boxes are drawn in native full-image coordinates, so ground-truth
    alignment is preserved. Dashed garnet rectangles = ground truth;
    coloured rectangles = top predictions per method (box area
    < 20,000 px² to hide PPI-ring artefacts). Classical detectors
    localise the bright vessel cores but produce extra boxes on clutter.
  ],
) <fig:qualitative>

// ── Root-Cause Analysis ──────────────────────────────────────────────
= Root-Cause Analysis <sec:rca>

The two-orders-of-magnitude gap demanded investigation. We used a deep
analysis agent on the full pipeline plus targeted probing of the four
underperformers; the key findings are summarised below and visualised in
@fig:volume and @fig:boxsizes.

== Classical failures

+ *CFAR guard-cell leakage.* With the default $"guard" = (3, 3)$ cells, the
  guard zone is 7 × 7 pixels. For 33 × 33-pixel targets, roughly 90% of the
  target energy leaked into the reference cells, elevating the local-noise
  estimate and suppressing detections. Raising guard to (15, 15) removed
  this effect.
+ *Otsu collapse on masked images.* A naïve `bitwise_and` mask leaves
  outside-disk pixels at raw BGR grey-scale value 2–3 rather than 0. Otsu
  then places its threshold near 11, classifying the *entire* PPI disk as
  foreground — a single giant connected component that is filtered out by
  the `max_area = 50,000` constraint, yielding 0 detections. The fix is
  two-part: (i) compute Otsu on non-zero pixels only, and (ii) apply the
  threshold only where the mask is non-zero.
+ *Structural IoU impossibility.* Raw CFAR outputs are typically 10 × 7
  pixels; GT boxes are 33 × 33. The maximum achievable IoU is
  $ "IoU"_max = (10 dot 7) / (33 dot 33) approx 0.06 $
  — below the AP\@0.50 threshold *by construction*. A 5-pixel binary dilation
  + 12-pixel box padding recovers AP from 0.000 to 0.010 for CA-CFAR.

== Deep-learning failures

#figure(
  image("figures/fig5_count_vs_ap.png", width: 100%),
  caption: [
    Total prediction volume (x, log) versus AP\@0.50 (y, log). Classical
    detectors cluster around the ground-truth annotation count (dashed garnet
    line) with high precision. Deep detectors either over-predict by three
    orders of magnitude (YOLOv11, RT-DETR at `conf ≥ 0.01`) or produce far
    too few confident predictions (Faster R-CNN, U-Net).
  ],
) <fig:volume>

+ *Localisation collapse in YOLOv11.* Inspecting per-tile outputs, all
  non-trivial YOLOv11 predictions on test tiles contain at least one
  detection anchored at $(0, 0)$ — the default decoded position when the
  regression head has not learned a meaningful offset. The model early-stopped
  at epoch 21 (patience = 20) on a domain where ImageNet-pretrained weights
  provide little inductive bias.
+ *RT-DETR over-predicts.* The transformer decoder emits
  ~300 queries per tile; with `conf_threshold = 0.01` this produces 1.3 M
  prediction boxes over 4,560 tiles, drowning any true positives in
  low-confidence background boxes. Raising the confidence threshold does
  not rescue AP because the underlying queries are themselves mis-localised.
+ *Faster R-CNN under-predicts.* Only ≈ 440 boxes across all 4,560 tiles
  at `box_score_thresh = 0.05`, and the few emitted boxes are offset
  from GT by 30–60 pixels — within the correct region but beyond
  the IoU = 0.50 threshold.
+ *Box-area mismatch (@fig:boxsizes).* Ground-truth areas cluster tightly
  around $10^3$ pixels; Wavelet-SWT has strong mass at the same scale plus
  a noise peak near $10^1$; CA-CFAR and morphological skew larger due to
  post-detection dilation. Deep-model predictions (not shown) have a
  bimodal distribution dominated by either the default anchor size
  (YOLOv11) or the full-tile size (RT-DETR decoder artifacts).

#figure(
  image("figures/fig6_boxsizes.png", width: 100%),
  caption: [
    Bounding-box area histogram (log-x). Ground truth (solid garnet) is
    sharply peaked at ≈ 1,000 pixels². Wavelet-SWT spans the correct
    peak plus a noise mass at small areas. CA-CFAR and morphological
    cluster at the correct scale but with right-shift introduced by
    the 12-pixel padding calibration.
  ],
) <fig:boxsizes>

// ── Discussion ───────────────────────────────────────────────────────
= Discussion

The central finding — that classical wavelet detection outperforms every
off-the-shelf deep detector by two orders of magnitude on this corpus —
deserves careful framing.

*Why wavelets win here.* Stationary wavelets directly implement the
scale-separation that distinguishes speckled sea clutter from coherent
vessel returns [3]. Our targets are $O(1 "m"^2)$ features
sampled at ≈ 12 m/pixel, so a single level of detail coefficients already
isolates them. CFAR and morphological methods, once repaired as in
@sec:rca, recover the same structure but with weaker score calibration.

*Why deep models underperform.* Three factors compound. First, the
pretrained backbones (ImageNet) transfer poorly to single-channel PPI
frames that are ≈ 95% black by area. Second, the 2,239-frame corpus is
small; 12 independent training runs in parallel still produced high
seed-to-seed variance. Third, the 33 × 33-pixel targets occupy ≈ 0.25%
of each 640² tile — approaching the COCO definition of "very small"
where even ImageNet-trained detectors systematically underperform
[11].

*Caveats.* Our deep models were trained at conservative hyperparameters
(no frozen-epoch warmup, no domain-specific augmentation, ImageNet
initialisation). A fully tuned pipeline — e.g., the SAHI-enhanced
YOLOv11-OBB reported by Heuer et al. [8] — likely closes
much of the gap. The point of the present paper is not that *no* deep
model can detect radar targets, but that the classical pipeline provides
a significantly stronger baseline than the community has recognised.

*Practical implications.* For safety-critical maritime deployments with
modest labelled data and interpretability requirements, the classical
wavelet detector is a strong, computationally cheap starting point.
Deep models should be reserved for scenarios with substantial labelled
data *and* domain-matched pretraining (e.g., self-supervised radar
encoders).

// ── Conclusion ───────────────────────────────────────────────────────
= Conclusion

We presented the first controlled cross-paradigm benchmark of classical
and deep-learning object detection on X-band marine radar PPI imagery.
Under identical preprocessing, splits and COCO-style evaluation,
the stationary wavelet transform reached AP\@0.50 = 0.065 — ≈ 130×
better than the strongest deep detector tested (YOLOv11). Our
root-cause analysis pinpointed CFAR guard-cell sizing, Otsu masking on
non-zero pixels, and GT-matched box dilation as high-leverage fixes for
classical methods, and ImageNet pretraining mismatch and extreme
spatial sparsity as the chief obstacles for deep detectors in this
domain. We release a reproducible 12-method benchmark to catalyse
further research on radar-native deep architectures and
self-supervised pretraining.

// ── References ───────────────────────────────────────────────────────
#heading(level: 1, numbering: none)[References]

#set text(size: 8.8pt)
#set par(justify: true, leading: 0.5em, first-line-indent: 0pt)

#enum(
  numbering: "[1]",
  indent: 0pt,
  body-indent: 0.7em,
  spacing: 0.35em,

  [H. M. Finn and R. S. Johnson, "Adaptive detection mode with threshold
   control," _RCA Review_, vol. 29, no. 9, 1968.],

  [H. Rohling, "Radar CFAR thresholding in clutter and multiple target
   situations," _IEEE Trans. AES_, vol. AES-19, no. 4, 1983.],

  [V. Duk et al., "Target detection in sea-clutter using stationary wavelet
   transforms," _IEEE Trans. AES_, vol. 53, no. 3, 2017.],

  [Q. Li et al., "A speed-up seed point Otsu method for ship detection,"
   _Proc. IEEE IGARSS_, 2016.],

  [Y. Xu et al., "Adaptive clustering-based marine radar sea-clutter
   normalisation," _J. Sensors_, 2021.],

  [X. Chen et al., "Automatic detection of scanning radar images based on
   Radar PPInet," _Remote Sensing_, vol. 13, 2021.],

  [X. Chen et al., "Marine-Faster R-CNN for navigation radar PPI images,"
   _Front. Inf. Technol. Electron. Eng._, vol. 23, 2022.],

  [M. Heuer et al., "Marine radar object detection using SAHI-enhanced
   YOLOv11-OBB," _Sensors_, vol. 26, 2025.],

  [J. Yee, "Onet: Twin U-Net for semantic segmentation in radar images,"
   _IEEE Trans. Image Process._, 2025.],

  [L. Zhang et al., "CCDN-DETR for multi-class SAR object detection,"
   _Sensors_, vol. 24, 2024.],

  [D. Bolya et al., "TIDE: A general toolbox for identifying object-
   detection errors," _Proc. ECCV_, 2020.],

  [X. Zhang and X. Li, "Survey on deep-learning-based marine object
   detection," _J. Adv. Transp._, 2021.],
)
