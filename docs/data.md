# Data

Where the scans go, what the dataset scanner accepts, and how the Bluefin tuna
dump is prepared. Raw scans are not tracked by git: `data/raw/` is ignored, so
every scan has to be put there by hand.

## One name per specimen

A specimen is one folder name, and that name is the same under `data/raw/` and
under `data/reference_masks/`:

```text
data/raw/DD2210_Bluefin_Tuna/      data/reference_masks/DD2210_Bluefin_Tuna/
data/raw/Gadus_Morhua/             data/reference_masks/Gadus_Morhua/
data/raw/Oreo_fish/                data/reference_masks/Oreo_fish/
```

Nothing enforces this, so it is worth keeping by hand. The tuna used to be
`DD2210_Blue_Tuna` among the scans and `DD2210_Bluefin_Tuna` among the masks,
which left half the configs pointing at a folder that did not exist. Spelling
counts too: `Gadus_morhua.nii.gz` is the file, and three configs asking for
`Gadus_Morhua.nii.gz` ran on macOS, where the filesystem ignores case, and
failed on Linux and in CI.

`data/raw/` is not tracked by git, so renaming a folder here renames it only on
your own machine. Everyone who already has the scans has to rename their copy
to match, or their configs stop resolving.

## Layout the scanner accepts

The supported layout is flat:

```text
data/raw/<dataset_name>/volume.nii
data/raw/<dataset_name>/metadata.json
data/raw/<dataset_name>/dicom_scan/slice_0001.dcm
data/raw/<dataset_name>/tiff_scan/slice_0001.tif
```

`.json` and uncompressed `.nii` files belong directly in the dataset folder, and
a scan subfolder may hold only one level of DICOM (`.dcm`) or TIFF (`.tif`,
`.tiff`) slices. Nested folders, JSON or NIfTI inside a scan folder, compressed
files such as `.nii.gz` and `.json.gz`, and unknown extensions are listed but
marked **invalid**, so a problem is visible without an invalid source being
processed.

## Reference masks

`data/reference_masks/<specimen>/` holds the ground-truth masks drawn in 3D
Slicer, one folder per specimen. A run is named after this folder on the
Experiments page, because the same scan reaches the runner under several file
names while its mask does not.

The Bluefin tuna has two annotations of the same scan on the same grid.
`Series_3_segmentation.nii.gz` is the one to measure against; it is contained in
`Segmentation.nii.gz` to 97.5%, and the 90 mL the larger one adds have a median
of 912 HU with 70% of them above 300 HU, which is the skeleton rather than the
bladder.

## Preparing the Bluefin tuna DICOM dump

The tuna scan arrives as a flat folder of files without an extension, with seven
series mixed together. `load_volume(path, "dicom")` reads `*.dcm` from a folder
and treats it as one volume, so the dump is split first:

```bash
python scripts/to_dicom_folders.py data/raw/DD2210_Bluefin_Tuna/DD2210 --dry-run
python scripts/to_dicom_folders.py data/raw/DD2210_Bluefin_Tuna/DD2210
```

The files already are DICOM; they only get a name the loader recognises, and one
folder per series. `--dry-run` prints what the dump holds:

| folder | description | slices | size | thickness | z step |
|---|---|---|---|---|---|
| `02_CUERPO_CUERPO_7_0` | CUERPO CUERPO 7.0 | 263 | 512x512 | 7 mm | 3.5 mm |
| `03_HUESO_HUESO_7_0` | HUESO HUESO 7.0 | 263 | 512x512 | 7 mm | 3.5 mm |
| `04_CUERPO_CUERPO_2_0` | CUERPO CUERPO 2.0 | 918 | 512x512 | 2 mm | 1 mm |
| `06_HUESO_HUESO_2_0` | HUESO HUESO 2.0 | 918 | 512x512 | 2 mm | 1 mm |

(plus two scout/report series and one reformat). `CUERPO` is the body kernel and
`HUESO` the bone kernel; the reference mask belongs to `02_CUERPO_CUERPO_7_0`,
and `04_CUERPO_CUERPO_2_0` is the same body at 1 mm, covering the same z range
from the same origin.

These slices overlap: 7 mm thick, reconstructed every 3.5 mm. The distance
between slices is therefore taken from their positions
(`ImagePositionPatient`) rather than from their thickness, and the loader warns
when a series is not evenly spaced. A long series is also read slice by slice
into the output array, so the 918-slice one loads in about 1 GB.
