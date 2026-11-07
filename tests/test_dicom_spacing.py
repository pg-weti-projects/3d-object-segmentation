"""The distance between DICOM slices comes from where the slices are, not from
how thick they are (they overlap in the Bluefin tuna series)."""

from __future__ import annotations

import logging

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from src.core.loaders import load_volume


def write_slice(
    folder,
    index: int,
    position_z: float,
    *,
    thickness: float = 7.0,
    spacing_between: float | None = None,
    pixel_spacing: tuple[float, float] = (0.8, 0.8),
    size: int = 8,
) -> None:
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.MediaStorageSOPClassUID = CTImageStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.InstanceNumber = index + 1
    dataset.ImagePositionPatient = [0.0, 0.0, float(position_z)]
    dataset.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    dataset.SliceThickness = float(thickness)
    if spacing_between is not None:
        dataset.SpacingBetweenSlices = float(spacing_between)
    dataset.Rows = size
    dataset.Columns = size
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1024.0
    dataset.PixelData = (np.full((size, size), index, dtype=np.int16)).tobytes()
    target = str(folder / f"slice_{index:03d}.dcm")
    try:  # pydicom >= 3
        dataset.save_as(target, enforce_file_format=True)
    except TypeError:  # pydicom 2.x
        dataset.save_as(target, write_like_original=False)


def series(folder, positions, **kwargs):
    folder.mkdir(parents=True, exist_ok=True)
    for index, position in enumerate(positions):
        write_slice(folder, index, position, **kwargs)
    return folder


class TestSeriesSpacing:
    def test_overlapping_slices_use_their_distance_not_their_thickness(self, tmp_path):
        # The tuna body series: 7 mm slices reconstructed every 3.5 mm.
        folder = series(tmp_path / "overlapping", [0.0, 3.5, 7.0, 10.5], thickness=7.0)

        volume = load_volume(folder, "dicom")

        assert volume.shape == (4, 8, 8)
        assert volume.spacing_mm == pytest.approx((3.5, 0.8, 0.8))

    def test_slices_in_any_order_give_the_same_spacing(self, tmp_path):
        folder = series(tmp_path / "shuffled", [10.5, 0.0, 7.0, 3.5], thickness=7.0)

        volume = load_volume(folder, "dicom")

        assert volume.spacing_mm == pytest.approx((3.5, 0.8, 0.8))

    def test_spacing_between_slices_is_confirmed_by_the_positions(self, tmp_path):
        # The Oreo scan: the scanner wrote SpacingBetweenSlices and it agrees.
        folder = series(tmp_path / "agreeing", [0.0, 3.0, 6.0], thickness=3.0, spacing_between=3.0)

        assert load_volume(folder, "dicom").spacing_mm == pytest.approx((3.0, 0.8, 0.8))

    def test_a_single_slice_falls_back_to_the_thickness(self, tmp_path):
        folder = series(tmp_path / "one", [0.0], thickness=5.0)

        assert load_volume(folder, "dicom").spacing_mm == pytest.approx((5.0, 0.8, 0.8))

    def test_a_hole_in_the_series_is_reported(self, tmp_path, caplog):
        # The tuna 2.0 series has gaps of 1 mm and one of 16 mm.
        folder = series(tmp_path / "hole", [0.0, 1.0, 2.0, 18.0], thickness=2.0)

        with caplog.at_level(logging.WARNING, logger="src.core.loaders"):
            volume = load_volume(folder, "dicom")

        assert volume.spacing_mm == pytest.approx((1.0, 0.8, 0.8))
        assert any("not evenly spaced" in record.message for record in caplog.records)

    def test_an_even_series_is_not_reported(self, tmp_path, caplog):
        folder = series(tmp_path / "even", [0.0, 2.0, 4.0, 6.0], thickness=2.0)

        with caplog.at_level(logging.WARNING, logger="src.core.loaders"):
            load_volume(folder, "dicom")

        assert not [record for record in caplog.records if "not evenly spaced" in record.message]
