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
