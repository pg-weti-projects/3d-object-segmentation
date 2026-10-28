from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from src.core.preprocessor import PreprocessingConfig, preprocess
from src.core.volume import Volume
from src.models.chan_vese import ObjectNotFoundError
from src.models.factory import make_segmenter
from src.models.grid import SeedPoint, WorkVolume, voxel_to_mm
from src.models.roi import roi_from_seed
from src.models.segment import seed_from_zyx, segment_volume, to_work_volume
from src.pipeline.experiment import AlgorithmConfig, PreprocessingSettings
from src.pipeline.metrics import compute_metrics

SPACING_ZYX = (0.5, 0.5, 0.5)
HU_RANGE = (-1000.0, 200.0)

ALGORITHMS = {
    "chan_vese": {"data_term_sign": "chan_vese", "init_anchor": "center", "remove_outside_air": True},
    "geodesic_active_contour": {"domain_size_mm": None},
    "growcut": {},
}


def make_fish_hu(
    spacing_zyx: tuple[float, float, float] = SPACING_ZYX,
) -> tuple[Volume, np.ndarray, tuple[int, int, int]]:
    shape_zyx = tuple(
        int(round(extent / s)) for extent, s in zip((14.0, 20.0, 28.0), spacing_zyx, strict=True)
    )
    axes = [(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape_zyx, spacing_zyx, strict=True)]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    data = np.full(shape_zyx, -1000.0, dtype=np.float32)
    data[(x / 12) ** 2 + (y / 8) ** 2 + (z / 6) ** 2 <= 1] = 40.0
    bladder = ((x - 1) / 6) ** 2 + ((y + 0.5) / 4) ** 2 + (z / 3.5) ** 2 <= 1
    data[bladder] = -1000.0
    seed = tuple(int(round(float(c))) for c in np.argwhere(bladder).mean(axis=0))
    return Volume(data, spacing_zyx, "HU", "dicom", "0" * 64), bladder, seed  # type: ignore[return-value]


def preprocessed(
    invert_hu: bool = False,
) -> tuple[Volume, np.ndarray, tuple[int, int, int], PreprocessingConfig]:
    volume, bladder, seed = make_fish_hu()
    config = PreprocessingConfig(hu_range=HU_RANGE, invert_hu=invert_hu)
    return preprocess(volume, config), bladder, seed, config


class TestWorkVolume:
    def test_axes_and_affine(self):
        data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        volume = Volume(data, (3.0, 2.0, 1.0), "HU", "dicom", "0" * 64)
        work = to_work_volume(volume)
        assert work.shape == (4, 3, 2)
        assert work.spacing == (1.0, 2.0, 3.0)
        assert np.array_equal(np.diag(work.affine), [1.0, 2.0, 3.0, 1.0])
        z, y, x = 1, 2, 3
        assert work.data[x, y, z] == pytest.approx(data[z, y, x] / data.max())

    def test_scaling_in_hu_window(self):
        volume = Volume(
            np.array([[[-1000.0, -400.0, 200.0, 900.0]]], np.float32), (1, 1, 1), "HU", "dicom", "0" * 64
        )
        work = to_work_volume(volume, HU_RANGE)
        assert work.data[:, 0, 0] == pytest.approx([0.0, 0.5, 1.0, 1.0])
        assert work.meta["hu_range"] == list(HU_RANGE)

    def test_invert_undone(self):
        plain, _, _, cfg = preprocessed(invert_hu=False)
        inverted, _, _, cfg_inv = preprocessed(invert_hu=True)
        a = to_work_volume(plain, cfg.hu_range, cfg.invert_hu)
        b = to_work_volume(inverted, cfg_inv.hu_range, cfg_inv.invert_hu)
        assert np.allclose(a.data, b.data, atol=1e-6)
        assert a.data.min() == pytest.approx(0.0)

    def test_invalid_window(self):
        volume, _, _ = make_fish_hu()
        with pytest.raises(ValueError):
            to_work_volume(volume, (200.0, -1000.0))
        with pytest.raises(ValueError):
            to_work_volume(volume, None, invert_hu=True)
        flat = Volume(np.zeros((2, 2, 2), np.float32), (1, 1, 1), "HU", "dicom", "0" * 64)
        with pytest.raises(ValueError):
            to_work_volume(flat)


class TestSeed:
    def test_zyx_to_xyz(self):
        volume, _, _ = make_fish_hu()
        work = to_work_volume(volume, HU_RANGE)
        seed = seed_from_zyx((3, 4, 5), work)
        assert seed.voxel == (5, 4, 3) and seed.source == "manual"
        assert seed.mm == pytest.approx((2.5, 2.0, 1.5))

    @pytest.mark.parametrize("bad", [(0, 0), (0, 0, 999), (-1, 0, 0), (0.5, 0, 0)])
    def test_invalid_seed(self, bad):
        volume, _, _ = make_fish_hu()
        with pytest.raises(ValueError):
            seed_from_zyx(bad, to_work_volume(volume, HU_RANGE))


class TestSegmentVolume:
    @pytest.mark.parametrize("name", sorted(ALGORITHMS))
    def test_gas_bladder(self, name):
        volume, bladder, seed, cfg = preprocessed()
        result = segment_volume(volume, AlgorithmConfig(name, ALGORITHMS[name]), seed, cfg)
        assert result.algorithm == name and result.status == "ok"
        assert result.mask.shape == volume.shape and result.mask.dtype == bool
        assert result.phi.shape == volume.shape and np.array_equal(result.mask, result.phi > 0)
        metrics = compute_metrics(result.mask, bladder, volume.spacing_mm)
        assert metrics["dice"] > 0.8, (name, metrics)
        assert result.params_used["seed_zyx"] == list(seed)
        json.dumps(result.params_used)

    @pytest.mark.parametrize("name", sorted(ALGORITHMS))
    def test_invert_hu_gives_same_mask(self, name):
        volume, _, seed, cfg = preprocessed(invert_hu=False)
        inverted, _, _, cfg_inv = preprocessed(invert_hu=True)
        algo = {"name": name, "parameters": ALGORITHMS[name]}
        a = segment_volume(volume, algo, seed, cfg)
        b = segment_volume(inverted, algo, seed, cfg_inv)
        assert np.array_equal(a.mask, b.mask)
        assert b.params_used["intensity_scaling"]["invert_hu_undone"] is True

    def test_unknown_algorithm_and_parameter(self):
        volume, _, seed, cfg = preprocessed()
        with pytest.raises(ValueError):
            segment_volume(volume, AlgorithmConfig("unet_2d"), seed, cfg)
        with pytest.raises(ValueError, match="literówka"):
            segment_volume(volume, AlgorithmConfig("growcut", {"fg_radius": 1.0}), seed, cfg)

    def test_chan_vese_seed_outside_object(self):
        volume, _, _, cfg = preprocessed()
        params = {**ALGORITHMS["chan_vese"], "roi_radius_mm": 0.5}
        with pytest.raises(ObjectNotFoundError):
            segment_volume(volume, AlgorithmConfig("chan_vese", params), (14, 20, 1), cfg)

    @pytest.mark.parametrize("name", sorted(ALGORITHMS))
    def test_anisotropic_spacing_matches_direct_call(self, name):
        volume, bladder, seed = make_fish_hu((1.0, 0.5, 0.4))
        cfg = PreprocessingConfig(hu_range=HU_RANGE)
        pre = preprocess(volume, cfg)
        result = segment_volume(pre, AlgorithmConfig(name, ALGORITHMS[name]), seed, cfg)
        assert result.mask.shape == volume.shape

        lo, hi = HU_RANGE
        data = ((np.clip(pre.data, lo, hi) - lo) / (hi - lo)).astype(np.float32)
        work = WorkVolume(np.ascontiguousarray(data.transpose(2, 1, 0)), np.diag([0.4, 0.5, 1.0, 1.0]), "HU")
        voxel = (seed[2], seed[1], seed[0])
        point = SeedPoint("manual", voxel, voxel_to_mm(work.affine, voxel))
        roi = roi_from_seed(work, point, ALGORITHMS[name] if name == "chan_vese" else None)
        direct = make_segmenter(name, ALGORITHMS[name]).segment(work, roi)
        assert np.array_equal(result.mask, direct.mask.data.transpose(2, 1, 0))
        if name != "growcut":
            assert compute_metrics(result.mask, bladder, volume.spacing_mm)["dice"] > 0.75, name

    def test_chan_vese_slices_along_z(self):
        volume, bladder, seed, cfg = preprocessed()
        result = segment_volume(volume, AlgorithmConfig("chan_vese", ALGORITHMS["chan_vese"]), seed, cfg)
        z_with_mask = np.flatnonzero(result.mask.any(axis=(1, 2)))
        z_bladder = np.flatnonzero(bladder.any(axis=(1, 2)))
        assert result.params_used["slice_range"] == [int(z_bladder[0]), int(z_bladder[-1])]
        assert set(z_with_mask) <= set(z_bladder)

    def test_preprocessing_settings_from_experiment_config(self):
        volume, bladder, seed = make_fish_hu()
        settings = PreprocessingSettings(hu_range=HU_RANGE, invert_hu=True)
        cfg = PreprocessingConfig(hu_range=HU_RANGE, invert_hu=True)
        result = segment_volume(
            preprocess(volume, cfg), {"name": "growcut", "parameters": {}}, seed, settings
        )
        assert result.params_used["intensity_scaling"] == {
            "hu_range": list(HU_RANGE),
            "invert_hu_undone": True,
        }
        assert compute_metrics(result.mask, bladder, volume.spacing_mm)["dice"] > 0.8

    def test_without_preprocessing_warns_and_uses_minmax(self, caplog):
        volume, bladder, seed = make_fish_hu()
        with caplog.at_level(logging.WARNING, logger="src.models.segment"):
            result = segment_volume(volume, AlgorithmConfig("growcut"), seed)
        assert any("hu_range" in r.getMessage() for r in caplog.records)
        assert result.params_used["intensity_scaling"]["hu_range"] is None
        assert compute_metrics(result.mask, bladder, volume.spacing_mm)["dice"] > 0.8

    def test_input_volume_not_modified(self):
        volume, _, seed, cfg = preprocessed()
        before = volume.data.copy()
        segment_volume(volume, AlgorithmConfig("growcut"), seed, cfg)
        assert np.array_equal(volume.data, before)
