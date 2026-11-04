"""scripts/prepare_data.py on tiny synthetic originals; the real scans are not needed."""

from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile

import nibabel as nib
import numpy as np
import pydicom
import pytest
import tifffile
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from scripts import prepare_data as P
from src.api.datasets import list_datasets
from src.core.loaders import load_volume

SIZE = 8
BODY, BONE, OTHER = 2, 3, 7


def dicom_bytes(series: int, index: int, z: float, description: str = "BODY 7.0") -> bytes:
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.MediaStorageSOPClassUID = CTImageStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.SeriesNumber = series
    dataset.SeriesInstanceUID = f"1.2.3.{series}"
    dataset.SeriesDescription = description
    dataset.InstanceNumber = index + 1
    # Like the real tuna: instance 1 at the top, and a slice normal pointing to -z.
    dataset.ImagePositionPatient = [0.0, 0.0, float(z)]
    dataset.ImageOrientationPatient = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    dataset.PixelSpacing = [0.5, 0.5]
    dataset.SliceThickness = 7.0
    dataset.Rows = SIZE
    dataset.Columns = SIZE
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = 0.0
    dataset.PixelData = np.full((SIZE, SIZE), index, dtype=np.int16).tobytes()
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


@pytest.fixture
def tuna_originals(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    with tarfile.open(originals / P.TUNA_ARCHIVE, "w:gz") as archive:
        members = {}
        for series in (BODY, BONE, OTHER):
            for index in range(3):
                members[f"A{series}{index:03d}"] = dicom_bytes(series, index, 100.0 - 3.5 * index)
        members["._A2000"] = b"\x00\x05\x16\x07 macOS resource fork"
        members["notes"] = b"not a scan"
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return originals


def tiff_bytes(image: np.ndarray, dpi: float = 254.0) -> bytes:
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, image, resolution=(dpi, dpi), resolutionunit="INCH")
    return buffer.getvalue()


def media_csv(unit: str = "Mm", spacing: float = 0.1) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["title", "x_pixel_spacing", "y_pixel_spacing", "z_pixel_spacing", "unit"])
    writer.writerow(["Whole Body", spacing, spacing, spacing, unit])
    return buffer.getvalue()


def write_gadus_archive(originals, slices, *, unit: str = "Mm") -> None:
    originals.mkdir(exist_ok=True)
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as stack:
        for index, image in enumerate(slices):
            stack.writestr(f"Fish/Fish_{index:04d}.tif", tiff_bytes(image))
        stack.writestr("__MACOSX/Fish/._Fish_0000.tif", b"junk")
    with zipfile.ZipFile(originals / P.GADUS_ARCHIVE, "w") as outer:
        outer.writestr("Media/Fish.zip", inner.getvalue())
        outer.writestr("media-manifest.csv", media_csv(unit))


def gadus_stack(depth: int = 5, rows: int = 5, columns: int = 7) -> list[np.ndarray]:
    values = np.arange(depth * rows * columns, dtype=np.uint16).reshape(depth, rows, columns)
    return [values[index] for index in range(depth)]


@pytest.fixture
def oreo_originals(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir(exist_ok=True)
    density = np.full((4, 5, 6), 1.225)
    speed = np.full((4, 5, 6), 343.0)
    density[1:3, 1:4, 1:5] = 1050.0
    speed[1:3, 1:4, 1:5] = 1500.0
    payload = {
        "specimens": [
            {
                "shapes": [
                    {
                        "voxel_size": [0.003, 0.003, 0.003],
                        "voxel_size_units": "m",
                        "mass_density": density.tolist(),
                        "mass_density_units": "kg/m^3",
                        "sound_speed_compressional": speed.tolist(),
                        "sound_speed_compressional_units": "m/s",
                    }
                ]
            }
        ]
    }
    (originals / P.OREO_JSON).write_text(json.dumps(payload), encoding="utf-8")
    return originals


def write_mask(masks_root, relative: str, shape_xyz, affine) -> None:
    path = masks_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape_xyz, dtype=np.uint8), np.asarray(affine, dtype=float)), str(path))


def manifest(raw) -> dict:
    return json.loads((raw / P.MANIFEST).read_text(encoding="utf-8"))["outputs"]


def valid_sources(raw, dataset: str) -> set[str]:
    described = next(item for item in list_datasets(raw) if item["id"] == dataset)
    return {source["id"] for source in described["sources"] if source["valid"] is True}


# --- tuna ---------------------------------------------------------------------------------------


def test_macos_files_and_other_files_are_not_unpacked(tuna_originals, tmp_path):
    target = tmp_path / "dump"
    counts = P.unpack_dicom_dump(tuna_originals / P.TUNA_ARCHIVE, target, lambda number: number == BODY)
    assert counts == {"written": 3, "other_series": 6, "macos_files": 1, "not_dicom": 1}
    assert sorted(path.name for path in target.iterdir()) == ["A2000", "A2001", "A2002"]


def test_tar_paths_cannot_leave_the_target(tmp_path):
    archive = tmp_path / "evil.tar"
    content = dicom_bytes(BODY, 0, 0.0)
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo("../../outside")
        info.size = len(content)
        stream.addfile(info, io.BytesIO(content))
    P.unpack_dicom_dump(archive, tmp_path / "x" / "dump", lambda number: True)
    assert (tmp_path / "x" / "dump" / "outside").exists()
    assert not (tmp_path / "outside").exists()


def test_tuna_keeps_the_default_series_in_the_panel_layout(tuna_originals, tmp_path):
    raw = tmp_path / "raw"
    written = P.prepare_tuna(tuna_originals, raw, masks_root=tmp_path / "none", series=(BODY, BONE))
    assert [path.name for path in written] == ["02_BODY_7_0", "03_BODY_7_0"]
    assert valid_sources(raw, P.TUNA_DATASET) == {"02_BODY_7_0", "03_BODY_7_0"}
    assert not list(raw.glob(".prepare_*"))
    volume = load_volume(raw / P.TUNA_DATASET / "02_BODY_7_0")
    assert volume.shape == (3, SIZE, SIZE)
    assert volume.spacing_mm == (3.5, 0.5, 0.5)
    entry = manifest(raw)[f"{P.TUNA_DATASET}/02_BODY_7_0"]
    assert entry["source"]["file"] == P.TUNA_ARCHIVE
    assert entry["source"]["unpacked"]["macos_files"] == 1
    assert entry["series"]["number"] == BODY
    assert entry["shape_zyx"] == [3, SIZE, SIZE]
    assert entry["output_sha256"] == volume.source_hash


def test_tuna_can_keep_every_series(tuna_originals, tmp_path):
    raw = tmp_path / "raw"
    P.prepare_tuna(tuna_originals, raw, masks_root=tmp_path / "none", series=None)
    assert valid_sources(raw, P.TUNA_DATASET) == {"02_BODY_7_0", "03_BODY_7_0", "07_BODY_7_0"}


def test_a_mirrored_mask_is_reported_and_not_corrected(tuna_originals, tmp_path, capsys):
    raw, masks_root = tmp_path / "raw", tmp_path / "masks"
    affine = [[0, -0.5, 0, 0], [-0.5, 0, 0, 0], [0, 0, -3.5, 100.0], [0, 0, 0, 1]]  # k = 0 at the top
    write_mask(masks_root, P.TUNA_MASKS[0], (SIZE, SIZE, 3), affine)
    P.prepare_tuna(tuna_originals, raw, masks_root=masks_root, series=(BODY,))
    entry = manifest(raw)[f"{P.TUNA_DATASET}/02_BODY_7_0"]
    assert entry["reference_masks"][0]["mirrored_in_z_against_loaded_series"] is True
    assert any("mirrored in z" in warning for warning in entry["warnings"])
    assert "mirrored in z" in capsys.readouterr().err
    assert entry["orientation"]["z_of_first_loaded_slice"] == 93.0
    assert entry["orientation"]["z_of_first_instance"] == 100.0
    assert entry["orientation"]["slice_normal"] == [0.0, 0.0, -1.0]
    # The pixels stay in the order the loader has always given: ascending z.
    volume = load_volume(raw / P.TUNA_DATASET / "02_BODY_7_0")
    assert volume.data[:, 0, 0].tolist() == [2.0, 1.0, 0.0]


def test_a_mask_in_the_same_order_is_not_called_mirrored(tuna_originals, tmp_path):
    raw, masks_root = tmp_path / "raw", tmp_path / "masks"
    affine = [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 3.5, 93.0], [0, 0, 0, 1]]
    write_mask(masks_root, P.TUNA_MASKS[0], (SIZE, SIZE, 3), affine)
    P.prepare_tuna(tuna_originals, raw, masks_root=masks_root, series=(BODY,))
    entry = manifest(raw)[f"{P.TUNA_DATASET}/02_BODY_7_0"]
    assert entry["reference_masks"][0]["mirrored_in_z_against_loaded_series"] is False
    assert entry["warnings"] == []


def test_rerun_skips_force_rebuilds_and_the_original_is_untouched(tuna_originals, tmp_path):
    raw = tmp_path / "raw"
    archive = tuna_originals / P.TUNA_ARCHIVE
    before = (archive.read_bytes(), archive.stat().st_mtime_ns)
    P.prepare_tuna(tuna_originals, raw, masks_root=tmp_path / "none", series=(BODY,))
    marker = raw / P.TUNA_DATASET / "02_BODY_7_0" / "A2000.dcm"
    stamp = marker.stat().st_mtime_ns
    assert P.prepare_tuna(tuna_originals, raw, masks_root=tmp_path / "none", series=(BODY,)) == []
    assert marker.stat().st_mtime_ns == stamp
    stale = marker.with_name("stale.dcm")
    stale.write_bytes(marker.read_bytes())
    assert P.prepare_tuna(tuna_originals, raw, masks_root=tmp_path / "none", series=(BODY,), force=True)
    assert not stale.exists()
    assert (archive.read_bytes(), archive.stat().st_mtime_ns) == before


def test_a_second_series_can_be_added_without_rebuilding_the_first(tuna_originals, tmp_path):
    raw = tmp_path / "raw"
    P.prepare_tuna(tuna_originals, raw, masks_root=tmp_path / "none", series=(BODY,))
    written = P.prepare_tuna(tuna_originals, raw, masks_root=tmp_path / "none", series=(BODY, BONE))
    assert [path.name for path in written] == ["03_BODY_7_0"]
    assert set(manifest(raw)) == {f"{P.TUNA_DATASET}/02_BODY_7_0", f"{P.TUNA_DATASET}/03_BODY_7_0"}


# --- Gadus --------------------------------------------------------------------------------------


def test_block_mean_averages_and_drops_what_does_not_fit():
    image = np.array([[0, 2, 9], [4, 6, 9], [9, 9, 9]], dtype=np.uint16)
    assert P.block_mean(image, 2).tolist() == [[3.0]]
    assert P.block_mean(image, 1).tolist() == image.tolist()


def test_averaged_slices_average_along_z_and_drop_the_rest():
    slices = [np.full((2, 2), value, dtype=np.uint16) for value in (0, 2, 10, 20, 99)]
    result = list(P.averaged_slices(slices, 2))
    assert [float(image[0, 0]) for image in result] == [1.0, 15.0]
    assert all(image.dtype == np.float32 for image in result)


def test_averaging_does_not_overflow_uint16():
    slices = [np.full((2, 2), 65535, dtype=np.uint16)] * 2
    assert float(next(P.averaged_slices(slices, 2))[0, 0]) == 65535.0


def test_tiff_names_are_in_slice_order_without_macos_files(tmp_path):
    path = tmp_path / "stack.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("a/s_10.tif", "a/s_2.tif", "a/._s_1.tif", "__MACOSX/a/s_1.tif", "a/readme.txt"):
            archive.writestr(name, b"")
    with zipfile.ZipFile(path) as archive:
        assert P.tiff_names(archive) == ["a/s_2.tif", "a/s_10.tif"]


def test_spacing_comes_from_the_manifest_and_needs_millimetres(tmp_path):
    path = tmp_path / "media.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("m.csv", media_csv("Mm", 0.25))
    with zipfile.ZipFile(path) as archive:
        assert P.manifest_spacing(archive) == ((0.25, 0.25, 0.25), "m.csv")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("m.csv", media_csv("Cm"))
    with zipfile.ZipFile(path) as archive:
        with pytest.raises(ValueError, match="expected mm"):
            P.manifest_spacing(archive)


def test_write_nifti_refuses_a_wrong_number_of_slices(tmp_path):
    with pytest.raises(ValueError, match="expected 2"):
        P.write_nifti(tmp_path / "x.nii", [np.zeros((2, 3))], (2, 2, 3), (1.0, 1.0, 1.0))


def test_gadus_end_to_end_values_axes_spacing_and_manifest(tmp_path):
    originals, raw = tmp_path / "originals", tmp_path / "raw"
    stack = gadus_stack()
    write_gadus_archive(originals, stack)
    written = P.prepare_gadus(originals, raw, masks_root=tmp_path / "none", factor=2)
    assert [path.name for path in written] == ["gadus_morhua_0.200mm.nii"]
    assert valid_sources(raw, P.GADUS_DATASET) == {"gadus_morhua_0.200mm.nii"}

    volume = load_volume(written[0])
    assert volume.shape == (2, 2, 3)
    assert volume.spacing_mm == pytest.approx((0.2, 0.2, 0.2))
    full = np.stack(stack).astype(float)
    expected = full[:2, :2, :2].mean()
    assert volume.data[0, 0, 0] == pytest.approx(expected)
    expected = full[2:4, 2:4, 4:6].mean()
    assert volume.data[1, 1, 2] == pytest.approx(expected)

    entry = manifest(raw)[f"{P.GADUS_DATASET}/gadus_morhua_0.200mm.nii"]
    assert entry["averaging"]["factor"] == 2
    assert entry["averaging"]["original_spacing_mm_xyz"] == [0.1, 0.1, 0.1]
    assert entry["averaging"]["dropped_trailing_voxels_zyx"] == [1, 1, 1]
    assert entry["orientation"]["known"] is False
    assert entry["output_sha256"] == volume.source_hash
    assert "media-manifest.csv" in entry["spacing_source"]


def test_gadus_warns_when_the_reference_mask_is_on_another_grid(tmp_path, capsys):
    originals, raw, masks_root = tmp_path / "originals", tmp_path / "raw", tmp_path / "masks"
    write_gadus_archive(originals, gadus_stack())
    write_mask(masks_root, P.GADUS_MASKS[0], (3, 2, 2), np.diag([0.1, 0.1, 0.1, 1]))
    P.prepare_gadus(originals, raw, masks_root=masks_root, factor=2)
    entry = manifest(raw)[f"{P.GADUS_DATASET}/gadus_morhua_0.200mm.nii"]
    assert any("another grid" in warning and P.GADUS_MASKS[0] in warning for warning in entry["warnings"])
    assert "another grid" in capsys.readouterr().err


def test_gadus_rerun_skips_and_force_rebuilds(tmp_path):
    originals, raw = tmp_path / "originals", tmp_path / "raw"
    write_gadus_archive(originals, gadus_stack())
    first = P.prepare_gadus(originals, raw, masks_root=tmp_path / "none", factor=2)[0]
    first.write_bytes(b"changed")
    assert P.prepare_gadus(originals, raw, masks_root=tmp_path / "none", factor=2) == []
    assert first.read_bytes() == b"changed"
    P.prepare_gadus(originals, raw, masks_root=tmp_path / "none", factor=2, force=True)
    assert load_volume(first).shape == (2, 2, 3)


def test_gadus_stops_on_a_slice_of_another_size_and_leaves_no_output(tmp_path):
    originals, raw = tmp_path / "originals", tmp_path / "raw"
    stack = gadus_stack()
    stack[1] = np.zeros((4, 4), dtype=np.uint16)
    write_gadus_archive(originals, stack)
    with pytest.raises(ValueError, match="expected"):
        P.prepare_gadus(originals, raw, masks_root=tmp_path / "none", factor=2)
    assert not (raw / P.GADUS_DATASET).exists()
    assert not list(raw.glob(".prepare_*"))


# --- Oreo ---------------------------------------------------------------------------------------


def test_compare_geometry_sees_the_same_and_a_different_shape():
    first = np.full((4, 4, 4), 1.0)
    second = np.full((4, 4, 4), 300.0)
    first[1:3, 1:3, 1:3] = 1000.0
    second[1:3, 1:3, 1:3] = 1500.0
    same = P.compare_geometry(first, second)
    assert same["differs"] is False and same["dice_of_non_air"] == 1.0
    second[0, 0, 0] = 1500.0
    different = P.compare_geometry(first, second)
    assert different["differs"] is True and different["air_disagreement_voxels"] == 1
    assert P.compare_geometry(first, np.zeros((2, 2, 2)))["same_shape"] is False


def test_oreo_writes_one_json_per_array_next_to_the_tracked_files(oreo_originals, tmp_path, capsys):
    raw = tmp_path / "raw"
    tracked = raw / P.OREO_DATASET / "oreo_fish.json.gz"
    tracked.parent.mkdir(parents=True)
    tracked.write_bytes(b"tracked")
    written = P.prepare_oreo(oreo_originals, raw, masks_root=tmp_path / "none")
    assert sorted(path.name for path in written) == ["mass_density.json", "sound_speed_compressional.json"]
    assert tracked.read_bytes() == b"tracked"
    assert valid_sources(raw, P.OREO_DATASET) == {"mass_density.json", "sound_speed_compressional.json"}
    assert "is the same" in capsys.readouterr().out

    density = load_volume(raw / P.OREO_DATASET / "mass_density.json")
    speed = load_volume(raw / P.OREO_DATASET / "sound_speed_compressional.json")
    assert density.shape == speed.shape == (4, 5, 6)
    assert density.spacing_mm == pytest.approx((3.0, 3.0, 3.0))
    assert (density.units, speed.units) == ("kg/m^3", "m/s")
    assert density.data[0, 0, 0] == pytest.approx(1.225) and speed.data[0, 0, 0] == 343.0

    entry = manifest(raw)[f"{P.OREO_DATASET}/mass_density.json"]
    assert entry["geometry_comparison"]["differs"] is False
    assert entry["output_sha256"] == density.source_hash


def test_oreo_rerun_skips(oreo_originals, tmp_path):
    raw = tmp_path / "raw"
    P.prepare_oreo(oreo_originals, raw, masks_root=tmp_path / "none")
    assert P.prepare_oreo(oreo_originals, raw, masks_root=tmp_path / "none") == []


# --- the command --------------------------------------------------------------------------------


def test_main_reports_a_missing_original_without_a_traceback(tmp_path, capsys):
    code = P.main(["--originals", str(tmp_path / "nothing"), "--raw", str(tmp_path / "raw"), "--only", "oreo"])
    assert code == 1
    assert "original not found" in capsys.readouterr().err


def test_main_runs_only_what_is_asked(oreo_originals, tmp_path):
    raw = tmp_path / "raw"
    assert P.main(["--originals", str(oreo_originals), "--raw", str(raw), "--only", "oreo"]) == 0
    assert set(path.name for path in raw.iterdir()) == {P.OREO_DATASET, P.MANIFEST}
