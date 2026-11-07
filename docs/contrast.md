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
| Bluefin tuna (box around the reference mask, body series at 3.5 mm) | 16% of the mask | 0.00 | -1.1 | `none` |
| Bluefin tuna (the same body at 1 mm, 918 slices) | 16% of the mask | 0.00 | -1.0 | `none` |

### The Bluefin tuna scan

The tuna is `none` for a reason that is worth stating plainly: **what its
reference mask outlines is as dense as the tissue around it.** In a box around
that mask, the seed region sits at 23 HU and the tissue at 18 HU, so the
"bladder" is in fact 5 HU *brighter* - there is nothing for any intensity-based
method to separate. Inside the mask the median is 17 HU, the 95th percentile
326 HU, and only 16% of it is gas (below -300 HU); the mask is 402 mL while the
enclosed gas pockets are tens of millilitres.

The mask was drawn by hand from anatomy, which a density threshold cannot
reproduce. Methods that start from the contrast therefore return an empty mask
and say why, instead of a shape they cannot justify; the best any of them does
from a seed ball is Dice 0.29, by filling the box with soft tissue.

**The 1 mm series gives the same answer, and an upper bound.** The same body is
in the dump at 1 mm as well (918 slices, `04_CUERPO_CUERPO_2_0`, the same z range
from the same origin), so the mask can be resampled onto it and measured again:
the seed region is 22.4 HU against 16.2 HU of tissue, still the wrong way round,
still `none`. Of the 402 mL of that mask, 63 mL is gas (below -300 HU), so **a
method that found every gas voxel inside the mask and nothing else would reach
Dice 0.271** - that is the ceiling for any density-based method on this
annotation. For comparison, the box holds 1792 mL of gas in total, almost all of
it the air around the fish, connected into one component that overlaps the mask
by Dice 0.06.

**The mask is where it should be, though.** The obvious suspicion - that it is
misaligned - was checked and does not hold. It lines up with the volume grid
without any permutation, and against a body mask (everything above -300 HU,
largest component, holes filled) it sits:

| | |
|---|---|
| inside the body | 84.3% (63 mL outside) |
| inside the body's silhouette (per-slice convex hull, which closes channels) | 98.4% (6.3 mL outside, in 48 scattered parts, largest 5.8 mL) |
| HU of the part "outside the body" | -960, i.e. air |

So the 63 mL is not the mask poking out of the fish: it is gas *within* the
fish's outline that is connected to the air outside, which is why a filled body
mask excludes it and why `use_hull` deliberately refuses to follow it. In a dead,
processed fish that is what a bladder open to the outside looks like.

What is left is a mask of 403 mL, correctly placed, of which about 340 mL has the
density of soft tissue and 63 mL is gas open to the outside. Whether that means a
collapsed bladder filled with fluid, or an outline of the whole organ including
its wall and contents, is a question for whoever drew it - not something a
segmentation parameter can settle.

**Nor can it be tuned.** Parameters decide which image feature a method
follows, so before calling the tuna hopeless, the three features a method could
possibly follow were measured at the boundary of its reference mask. None of them
is there (`scripts/` has no case for this; the numbers come from a one-off probe
on the 3.5 mm series, box of the mask grown by 25%):

| feature | at the mask boundary | control |
|---|---|---|
| gradient magnitude (is a bladder wall visible?) | 3.0 | 5.3 inside the mask, 3.2 just outside, 2.3 on a surface shifted by 12 voxels |
| enclosed gas to grow or inflate from | 3.1 mL in 8 parts, Dice 0.001 (its convex hull: 0.001) | of the 63 mL of gas-density voxels in the mask, nearly all are connected to the air outside the fish |
| local standard deviation in 5 mm (texture) | 13.0 inside the mask | 7.6 just outside; thresholding it gives at best 974 mL at Dice 0.170 |

The boundary of this mask is not an edge, is not a gas border, and is not a
texture border - the gradient on it is *lower* than inside the mask and no higher
than on an arbitrary surface 12 voxels away. There is no setting of
`threshold_fraction`, `balloon`, `gac_alpha` or seed radius that makes a method
follow something the image does not contain. What the mask does have is more
structure *inside* it than around it, which is consistent with a hand-drawn
outline of an organ whose wall this scan does not resolve.

If the annotation is the whole organ rather than its gas space, the thing to
follow would be the bladder wall - so that was measured too, on both series, as
the in-plane gradient (the wall is thin, and a 7 mm slice averages it away):

| series | on the mask boundary | inside | just outside | a surface 15 voxels away |
|---|---|---|---|---|
| 3.5 mm | 96.5 | 244.5 | 104.1 | 68.0 |
| 1 mm | 118.3 | 114.3 | 130.8 | 93.0 |

At 1 mm, where slices are 2 mm thick instead of 7 mm, the boundary is
indistinguishable from the tissue on either side of it, and only 22% of it falls
in the top 10% of the gradient, against 17% for an arbitrary surface nearby. A
visible wall would put that share at 60-90%. So the organ's outline is not in
this scan either, at either resolution.

That matters beyond the tuna: if the tuna mask is an organ outline and the Oreo
mask is a gas space (there the bladder is gas at density 1 and a threshold alone
reaches Dice 0.95), the two datasets do not define the same target, and a single
Dice column across species compares two different tasks. It is worth stating in
the report which of the three masks outlines what.

The honest ways forward are therefore not parameters: ask what the annotation
outlined (and from which series), or give the problem to a model that carries a
shape prior rather than an intensity rule - which is what the learned methods in
the runner's table are for.

Three things had to be fixed before any of this could be measured, and all three
are worth knowing:

* the scan arrives as a flat dump of 2430 files with no extension and seven
  series mixed together; `scripts/to_dicom_folders.py` splits it into one folder
  per series (the body series `02_CUERPO_CUERPO_7_0`, 263 slices, is the one the
  reference mask belongs to, and `04_CUERPO_CUERPO_2_0` has the same body at
  918 slices of 1 mm);
* its slices overlap - 7 mm thick, reconstructed every 3.5 mm - and the loader
  read the thickness, which made the volume twice as long as its own reference
  mask and sent earlier attempts looking for a permutation of the axes that does
  not exist. With the distance taken from the slice positions, the volume is
  (263, 512, 512) at 3.5 mm and the mask lines up with it directly;
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
| Bluefin tuna | an organ of which 12% is gas | +0.002 | no - only the gas part is |

So on the tuna the methods are not "focusing on gas instead of the organ" by
mistake: the organ is not a darker region at all, and the only darker thing
inside it is its remaining gas. That is a property of the scan, not a choice we
made.

**How far the data can be pushed.** `scripts/threshold_ceiling.py` sweeps a
single intensity threshold over the whole range, under the same restriction to
the body the methods use, and reports the best Dice it can reach. That is the
ceiling for this whole family of methods:

| dataset | contrast | regime | best possible Dice | `box_region` reaches |
|---|---|---|---|---|
| Oreo | 0.882 | high | 0.940 | **0.949** |
| Gadus | 0.088 | low | 0.663 | 0.643 |
| Bluefin tuna | 0.000 | none | 0.200 | 0.163 |

On every dataset the method is within a few points of the best any threshold
could do, and on Oreo it is at it. The spread between 0.95, 0.64 and 0.16 is
therefore not a story about the method working on one fish and failing on
another - it is the spread of how much of each annotated target an intensity
rule can see at all. Reporting the ceiling next to the Dice keeps that honest:
0.163 out of a possible 0.200 says something quite different from 0.163 on its
own.

## What the three masks actually outline

The tuna mask was compared with the other two in terms of each scan's own
intensity distribution, which makes the three comparable despite their different
units (density in kg/m3, raw TIFF values, HU):

| mask | volume | mask p5 / p50 / p95 | above the scan's p99 | parts | holes |
|---|---|---|---|---|---|
| Oreo | 31.2 mL | 1 / 1 / 1 (pure gas) | 0.1% | 4 | 0 |
| Gadus Morhua | 2.4 mL | 18263 / 18513 / 19090 (one soft-tissue band) | 1.3% | 1 | 0 |
| Bluefin tuna | 402.5 mL | -983 / 17 / 326 (air to bone) | **8.0% = 32 mL** | 1 | 0 |

The first two are homogeneous: the Oreo mask is the gas space and nothing else,
the Gadus mask is a narrow band of soft tissue with no gas in it. Both look like
a careful outline of one structure.

The tuna mask is not. It spans the full range of the scan, from air at -983 HU to
bone at +326 HU, and 8% of it (32 mL) is denser than 99% of the whole scan. Of
that, 19.7 mL is a single connected blob of bone density on slices 107-139 - a
swim bladder contains no bone, so that part is an error under any definition of
the target. Shrinking the mask does not reach a cleaner core either: eroded by
12 mm it is down to 21 mL and still 44% muscle density.

Two candidates for review were derived and left in `results/tuna_mask_review/`
(the reference itself is untouched): the mask without the bone blob (374 mL) and
the gas inside the mask (44.7 mL). Running the methods against them settles what
is wrong and what is not:

| target | best Dice | what it means |
|---|---|---|
| the reference, 402.5 mL | empty mask | the bladder at the seed is denser than the tissue |
| without the bone, 374 mL | 0.301 (GrowCut) | removing the bone changes nothing; 125 mL of muscle density remains |
| the gas only, 44.7 mL | 0.736 (GrowCut) | **circular** - that target was made with the same threshold the method uses |

The last row is not a result and must not be reported as one; it only shows that
the pipeline works on this scan once the target is something the image contains.

So the conclusion is narrower than "the tuna mask is wrong" and sharper than "the
tuna is hard": **the tuna annotation cannot be verified or reproduced from its
own scan, and it contains 20 mL of a structure that is certainly not bladder.**
Whether the rest of it is a collapsed, fluid-filled bladder drawn from anatomy or
an outline that took in the surrounding tissue is a question for whoever drew it.
Until that is answered, the tuna does not belong in a results table next to the
other two, because its target is not of the same kind.

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
| `box_region` | - | **0.949** | - | 0.628 | - | empty |
| Chan-Vese | 0.085 | 0.939 | 0.233 | 0.494 | 0.087 | empty |
| Geodesic active contour | 0.180 | 0.772 | 0.210 | **0.727** | not run | not run |
| GrowCut | 0.585 | 0.851 | 0.597 | 0.586 | 0.288 | empty |

"empty" is the honest answer on the tuna, as explained above - and no method can
do better than Dice 0.271 there anyway. The geodesic method was not run on it
because at 0.839 mm isotropic the box is 10.8 million voxels, which the
development machine will not finish in reasonable time.

The tuna case is in the scripts as `tuna` (the 3.5 mm series the mask belongs to)
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
