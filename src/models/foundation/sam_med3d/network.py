"""
Loading SAM-Med3D and predicting: the only part that needs torch, imported lazily.

The network works in (x, y, z), with cubes of SIZE^3 and clicks as voxel indices;
converting to and from the project's (z, y, x) is the job of cube.py.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import sys
import time
import types
from collections import OrderedDict
from typing import Any, Protocol

import numpy as np

from . import assets as assets_module

logger = logging.getLogger(__name__)

_UPSTREAM_PACKAGE = "_sam_med3d_upstream"
_CACHED_EMBEDDINGS = 4


class Network(Protocol):
    def embed(self, cube_xyz: np.ndarray) -> Any: ...

    def predict(self, embedding: Any, click_xyz: tuple[int, int, int]) -> np.ndarray: ...


class SamMed3D:
    """The upstream network, unmodified. One positive click, no box, the first mask prompt empty,
    exactly as `sam_model_infer` in upstream's utils/infer_utils.py."""

    def __init__(self, source, weights, device: str = "cpu", threads: int | None = None) -> None:
        import torch

        self._torch = torch
        if threads:
            torch.set_num_threads(int(threads))
        self._device = torch.device(device)
        build = _load_upstream_builder(source)
        model = build(checkpoint=None)
        # memory-mapped, so that the weights do not sit in memory twice while they are copied in
        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        self._model = model.to(self._device).eval()

    def embed(self, cube_xyz: np.ndarray) -> Any:
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(cube_xyz, dtype=np.float32))[None, None].to(self._device)
        with torch.no_grad():
            return self._model.image_encoder(tensor)

    def predict(self, embedding: Any, click_xyz: tuple[int, int, int]) -> np.ndarray:
        torch = self._torch
        size = int(self._model.prompt_encoder.input_image_size[0])
        coordinates = torch.tensor([[list(click_xyz)]], dtype=torch.float32, device=self._device)
        labels = torch.ones((1, 1), dtype=torch.int64, device=self._device)
        empty_mask = torch.zeros((1, 1, size // 4, size // 4, size // 4), device=self._device)
        with torch.no_grad():
            sparse, dense = self._model.prompt_encoder(points=[coordinates, labels], boxes=None, masks=empty_mask)
            low_resolution, _ = self._model.mask_decoder(
                image_embeddings=embedding,
                image_pe=self._model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,  # one mask, as upstream's train.py:239
            )
            upscaled = torch.nn.functional.interpolate(
                low_resolution, size=(size, size, size), mode="trilinear", align_corners=False
            )
            return torch.sigmoid(upscaled)[0, 0].cpu().numpy()


class CachedEncoder:
    """Keeps the encoder's output per cube, by its content, so that another click in the same cube
    does not rerun the encoder. Only the latest few are kept, and this lives as long as the process."""

    def __init__(self, network: Network, limit: int = _CACHED_EMBEDDINGS) -> None:
        self.network = network
        self.limit = limit
        self.last_was_cached = False
        self._embeddings: OrderedDict[str, Any] = OrderedDict()

    def embed(self, cube_xyz: np.ndarray) -> Any:
        key = hashlib.sha256(np.ascontiguousarray(cube_xyz, dtype=np.float32).tobytes()).hexdigest()
        self.last_was_cached = key in self._embeddings
        if self.last_was_cached:
            self._embeddings.move_to_end(key)
        else:
            self._embeddings[key] = self.network.embed(cube_xyz)
            while len(self._embeddings) > self.limit:
                self._embeddings.popitem(last=False)
        return self._embeddings[key]

    def predict(self, embedding: Any, click_xyz: tuple[int, int, int]) -> np.ndarray:
        return self.network.predict(embedding, click_xyz)


_loaded: dict[tuple[str, int | None], CachedEncoder] = {}


def get_network(device: str = "cpu", threads: int | None = None) -> CachedEncoder:
    """The network, verified against its pins and loaded once per process. The first call is slow."""
    key = (device, threads)
    if key not in _loaded:
        assets = assets_module.require()
        started = time.perf_counter()
        logger.info("SAM-Med3D: loading the weights; this is slow only on the first run in a process")
        _loaded[key] = CachedEncoder(SamMed3D(assets.source, assets.weights, device, threads))
        logger.info("SAM-Med3D: loaded in %.1f s", time.perf_counter() - started)
    return _loaded[key]


def _load_upstream_builder(source) -> Any:
    """Upstream's build_sam3D_vit_b_ori, the 128^3 architecture of this checkpoint, without running
    upstream's package __init__: that imports torchvision and its own top-level `utils`, which the
    3D model does not use. A private package name keeps it from colliding with anything else."""
    if _UPSTREAM_PACKAGE not in sys.modules:
        package = types.ModuleType(_UPSTREAM_PACKAGE)
        package.__path__ = [str(source / "segment_anything")]
        sys.modules[_UPSTREAM_PACKAGE] = package
    return importlib.import_module(f"{_UPSTREAM_PACKAGE}.build_sam3D").build_sam3D_vit_b_ori
