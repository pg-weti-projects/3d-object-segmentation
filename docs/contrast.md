# Contrast between the bladder and the tissue

The analytical methods (Chan-Vese, geodesic active contour, GrowCut) used to
start from a small ball around the seed point. A ball says nothing about where
the bladder ends, so each method had to discover that from one point, and on our
scans they mostly did not.

`src/models/contrast.py` measures the difference between the bladder and the
tissue once, inside the box, and `src/models/contrast_init.py` hands the result
to the methods (`init_from: contrast`). The old behaviour is still the default
(`init_from: seed_ball`).

## What is measured

Inside the volume the method receives (in semi-automatic mode: the contents of
the box):

| quantity | meaning |
|---|---|
| object level | median in a small ball around the seed, i.e. the bladder |
| tissue level | median of the tissue class of an Otsu split |
| threshold | between the two, `threshold_fraction` of the way from the bladder |
| separability | the gap between the levels as a share of the dynamic range (1-99 percentile) |
| CNR | the same gap in units of the noise at the seed (MAD) |

`separability` decides the regime, because it does not depend on the unit or the
offset of the scan (density in kg/m³, HU, or intensities rescaled to [0, 1]).

| regime | separability | meaning |
|---|---|---|
| `high` | >= 0.30 | gas, or any bladder a threshold separates on its own |
| `low` | 0.05 - 0.30 | the boundary is driven by the shape, not by the threshold |
| `none` | < 0.05, or the levels in the wrong order | nothing usable; the result is reported with a warning |

The boundaries are on round numbers on purpose: with three species there is not
enough data to place them more precisely.

## Measured on the scans of the project

`python scripts/measure_contrast.py` (the box is the one of the experiment config
for Oreo, and the bounding box of the reference mask grown by 25% for the other
two - roughly what a user would draw by hand)

| scan | gas | separability | CNR | regime |
|---|---|---|---|---|
| Oreo (box of `check_oreo_box.yaml`) | yes | 0.88 | infinite (the gas is uniform) | `high` |
| Gadus Morhua (box around the reference mask, 2x downsampled) | no | 0.09 | 6.0 | `low` |
| Bluefin tuna (box around the reference mask, body series at 3.5 mm) | 6% of the mask | 0.01 | 0.4 | `none` |
| Bluefin tuna (the same body at 1 mm, 918 slices) | 7% of the mask | 1.00 | - | `high`, not meaningful (see below) |

### The Bluefin tuna scan

**A correction.** Until the loader ordered slices along the slice normal, this
series was loaded mirrored in z against its reference masks, so earlier versions
of this page measured the wrong part of the fish. Their conclusions - that the
mask was as dense as the tissue around it, and that its boundary was not an edge -
do not hold. Everything below is measured on the corrected alignment, and claims
that were not measured again have been removed.

**The bladder is brighter than the tissue around it, not darker**, and that is why
the regime reads `none`: these methods look for a darker region. Two annotations
of the organ exist, agreeing at Dice 0.864. On the 3.5 mm series:

| mask | volume | median | in a shell around it | 75th / 95th percentile | gas (below -300 HU) |
|---|---|---|---|---|---|
| `Segmentation` | 402.5 mL | 131 HU | 19 HU | 493 / 3285 HU | 6.3%, 25.5 mL |
| `Series_3_segmentation` | 320.6 mL | 70 HU | 64 HU | 289 / 925 HU | 7.5%, 24.0 mL |

The estimate itself needs care on this scan. With the seed at the centroid of
`Segmentation` it finds -10.7 HU against 15.0 HU of tissue, `none`. In the box of
`Series_3_segmentation`, and at 1 mm in both, it reports separability 1.00 and
`high`. That is an artefact, not a finding: the Otsu split takes a very dense
structure as the tissue (a tissue level of about 4700-4900 HU).

**The gas does not explain the mask.** A method that found every gas voxel inside
the mask and nothing else would reach Dice 0.119 (`Segmentation`) or 0.139
(`Series_3_segmentation`), and 0.124 or 0.145 at 1 mm. The box holds 1005 mL of gas
in total, nearly all of it the air around the fish, at Dice 0.036 with the mask.

**The mask is inside the fish.** Of `Segmentation`, 99.9% lies inside the body (everything
above -300 HU, largest component, slices filled). Of its 25.5 mL of gas, 25.0 mL is
enclosed, in 12 parts, and 0.5 mL connects to the air outside.

**The boundary is an edge.** The in-plane gradient by signed distance from the
boundary, in mm (negative is inside), on the 3.5 mm series:

| mask | -20..-10 | -10..-4 | -4..0 | 0..4 | 4..10 | 10..20 |
|---|---|---|---|---|---|---|
| `Segmentation` | 240 | 539 | **1170** | **633** | 110 | 77 |
| `Series_3_segmentation` | 106 | 238 | **727** | **972** | 409 | 123 |

And on the 1 mm series, where slices are 2 mm thick instead of 7 mm:

| mask | on the boundary | 2-6 mm inside | 2-6 mm outside | 15 mm outside | boundary in the top 10% of gradient |
|---|---|---|---|---|---|
| `Segmentation` | 1615 | 1001 | 192 | 93 | 63% (5% at 15 mm) |
| `Series_3_segmentation` | 1287 | 448 | 766 | 115 | 59% (6% at 15 mm) |

So the outline is visible at both resolutions. The tables also hint at how the two
masks differ: past the boundary of `Segmentation` the gradient falls away, while past
that of `Series_3_segmentation` it stays high, as if one includes a dense layer and
the other stops inside it (see "What the three masks actually outline").

**An intensity rule has something to follow, in the other direction.** A single
threshold keeping what is *brighter* reaches Dice 0.494 on `Segmentation` and 0.395
on `Series_3_segmentation`, against 0.123 and 0.122 for one keeping what is darker.
That is a ceiling for one threshold, not a result of any method here.

It is worth stating in the report which of the three masks outlines what. The way
forward is to ask what each annotation outlined, and from which series: series 02
and 03 share the grid of the masks, and which one they were drawn on is not
established.

Four things had to be fixed before any of this could be measured, and all four
are worth knowing:

* the scan arrives as a flat dump of 2430 files with no extension and seven
  series mixed together; `scripts/to_dicom_folders.py` splits it into one folder
  per series (the body series `02_CUERPO_CUERPO_7_0`, 263 slices, is on the grid of
  the reference masks, as is the bone series `03_HUESO_HUESO_7_0`, and
  `04_CUERPO_CUERPO_2_0` has the same body at 918 slices of 1 mm);
* its slices overlap - 7 mm thick, reconstructed every 3.5 mm - and the loader
  read the thickness, which made the volume twice as long as its own reference
  mask and sent earlier attempts looking for a permutation of the axes that does
  not exist. With the distance taken from the slice positions, the volume is
  (263, 512, 512) at 3.5 mm, the grid of the masks;
* its slice normal points to -z, and the loader sorted by ascending z, which left
  the series mirrored against its masks. Slices are now ordered along the normal,
  and the runner warns when a mask looks mirrored against its scan (`alignment` in
  `metrics.json`);
* loading the 1 mm series ran out of memory: every slice was parsed and kept,
  then converted into a second list, then stacked into a third copy, and the
  whole series was read again as bytes to hash it. The slices now go one at a
  time into the output array, and the 918-slice series loads in about 4 seconds
  in 1 GB instead of being killed at 3 GB.

## The same method, three different targets

A Dice column across the three species compares three different tasks, so it is
worth being explicit about what the methods do and what each dataset asks of
them.

**What the methods do is the same everywhere.** Chan-Vese, the geodesic active
contour, GrowCut and `box_region` all answer one question: *which connected
region around the seed point is darker than the tissue around it*. They differ
only in how they settle its boundary. None of them looks for "gas"; gas is
simply what that region happens to be when the bladder holds any.

**What each dataset asks for is not the same.** Measured on the volume inside the
box, with the whole annotation against a shell of tissue around it:

| dataset | the annotation is | contrast of the whole target | is the target the darker region? |
|---|---|---|---|
| Oreo | the gas space (its 5th to 95th percentile is pure gas) | +0.886 | yes, it is exactly that region |
| Gadus | a fluid-filled organ, no gas | +0.075 | yes, but only just |
| Bluefin tuna | an organ, about 6-8% gas, denser than the tissue around it | not measured again | no - on median it is brighter |

So on the tuna the methods look for something the target is not: on median the organ
is brighter than the tissue around it, not darker. That is a property of the scan,
not a choice we made.

**How far the data can be pushed.** `scripts/threshold_ceiling.py` sweeps a
single intensity threshold over the whole range, under the same restriction to
the body the methods use, and reports the best Dice it can reach. That is the
ceiling for this whole family of methods:

| dataset | contrast | regime | best possible Dice | `box_region` reaches |
|---|---|---|---|---|
| Oreo | 0.882 | high | 0.940 | **0.949** |
| Gadus | 0.088 | low | 0.663 | 0.643 |
| Bluefin tuna | 0.013 | none | 0.123 | 0.100 |

On Oreo and Gadus the method is within a few points of the best any threshold could
do, and on Oreo it is at it. The spread between 0.95, 0.64 and 0.10 is therefore not
a story about the method working on one fish and failing on another - it is the
spread of how much of each annotated target a rule looking for a darker region can
see at all. Reporting the ceiling next to the Dice keeps that honest: 0.100 out of a
possible 0.123 says something quite different from 0.100 on its own.

The script sweeps only darker thresholds, and for the tuna that is the wrong
direction (see above). For `Series_3_segmentation` its restriction to the body also
collapses onto the dense structure, so its figure there is not meaningful.

## What the three masks actually outline

The tuna mask was compared with the other two in terms of each scan's own
intensity distribution, which makes the three comparable despite their different
units (density in kg/m3, raw TIFF values, HU):

| mask | volume | mask p5 / p50 / p95 | above the scan's p99 | parts | holes |
|---|---|---|---|---|---|
| Oreo | 31.2 mL | 1 / 1 / 1 (pure gas) | 0.1% | 4 | 0 |
| Gadus Morhua | 2.4 mL | 18263 / 18513 / 19090 (one soft-tissue band) | 1.3% | 1 | 0 |
| Bluefin tuna | 402.5 mL | -491 / 131 / 3285 | **44.5% = 179 mL** | 1 | 0 |

The first two are homogeneous: the Oreo mask is the gas space and nothing else,
the Gadus mask is a narrow band of soft tissue with no gas in it. Both look like
a careful outline of one structure.

The tuna mask is not homogeneous either way: 44.5% of it (179 mL) is denser than
99% of the whole scan, and the largest such part, 176 mL, runs through all of its
slices. Most of what is above 1000 HU in the box is one structure of 59 mL, median
2190 HU, with almost the extent of the bladder (301 x 93 x 99 mm). It lies 91.5%
inside `Segmentation` and 21.6% inside `Series_3_segmentation`.

So the conclusion is narrower than any single sentence about "the tuna mask": **the
annotations fit their scan once it is loaded in the right order, and the gradient
peaks on their boundaries, but what the dense structure is, and whether it belongs
to the target, is not established.** That is a question for whoever drew them, and
until it is answered the tuna's target is not of the same kind as the other two.

## What it changes for each method

Each method needed one more thing than the region itself, found by measuring:

* **Chan-Vese** binarizes each slice to an intensity window, and the window from
  a 3 mm ball in gas is almost empty, while on a cropped scan the window also
  catches the air around the body, so the object region ran out through it. With
  `init_from: contrast` the binary image is the thresholded region limited to the
  body, each slice's contour starts on its outline, and `data_term_sign` switches
  to `chan_vese` (the `paper` sign only moves the contour outwards, which a
  rectangle inside the object needs but a contour already on the boundary does
  not - it collapses to nothing, Dice 0.02 instead of 0.94).
* **Geodesic active contour** gets the region as its initial level set, and the
  balloon force is turned off, because a contour that starts in the right place
  only has to settle on the boundary; with the default balloon it keeps inflating
  until it fills the box.
* **GrowCut** gets the inside of the region as object seeds and the tissue around
  it as background seeds, instead of one ball of object seeds.

Both overrides apply only when they were not asked for explicitly.

## Results

`python scripts/compare_methods.py` (same box, same seed, same voxels for every
method; Dice against the reference mask)

| method | Oreo, seed ball | Oreo, contrast | Gadus, seed ball | Gadus, contrast | Tuna, seed ball | Tuna, contrast |
|---|---|---|---|---|---|---|
| `box_region` | - | **0.949** | - | 0.628 | - | 0.100 |
| Chan-Vese | 0.085 | 0.939 | 0.233 | 0.494 | 0.217 | 0.095 |
| Geodesic active contour | 0.180 | 0.772 | 0.210 | **0.727** | not run | not run |
| GrowCut | 0.585 | 0.851 | 0.597 | 0.586 | 0.485 | 0.101 |

The tuna columns use the `Segmentation` mask with the seed at its centroid. The
contrast starts now return large masks with low Dice rather than empty ones. The
geodesic method did not finish in 600 s on the tuna's 110 x 144 x 161 box on the
development machine, so it is reported as not run.

The tuna case is in the scripts as `tuna` (the 3.5 mm series on the grid of the masks)
and `tuna_1mm` (the same body at 1 mm, with the mask resampled onto it);
`compare_methods.py` skips `tuna_1mm` unless it is asked for by name, because of
its size.

Where the bladder is gas, the threshold alone is already close to the reference
and `box_region` is the most accurate. Where it is not (Gadus), the edge-based
contour of the geodesic method refines the boundary better than any threshold,
and it is the most accurate of the four. GrowCut gains a lot on gas and nothing
on Gadus, where its own ball of seeds was already inside a region of almost
uniform brightness.

These runs are on a development machine and with `compute_spacing_mm: null`, so
the times they print are not the official timings of the project.
