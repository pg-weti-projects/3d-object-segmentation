"""
Fine-tuning SAM-Med3D's prompt encoder and mask decoder on a frozen image encoder.

Training cubes are made as `pipeline.segment` makes them (a cube on a box, the scan around it, the
automatic intensity shift, upstream's z-score), from jittered boxes and with augmentation. Each cube is
encoded once and cached; training then runs only the decoder, with a new click inside the target at
every step. The recipe is fixed in md-files/sam_med3d_finetune_protocol.md, sections 2-3.

No torch at import: only `train` and the functions it calls import it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from src.core.volume import Volume

from . import assets, cube as geometry, network
from .pipeline import AUTO, auto_shift, prepare

logger = logging.getLogger(__name__)

RECIPE: dict[str, Any] = {
    # boxes (section 2.1, step 1)
    "box_growth": [0.10, 0.50],  # of the bladder's extent, per axis, half on each side
    "box_shift": 0.10,  # of the box's size, per axis, either way
    "long_box_share": 0.25,  # this share of samples grows the box along the bladder's longest axis instead
    "long_box_growth": [0.50, 1.00],
    # augmentation (step 3)
    "small_rotation_deg": 10.0,  # in the axial (y, x) plane
    "scale": [0.9, 1.1],
    "intensity_scale": [0.9, 1.1],
    "intensity_offset": 0.10,  # of the box's std
    "noise": 0.02,  # of the box's std
    # training (section 2.2)
    "steps": 2000,
    "batch": 4,
    "lr": 8e-5,
    "weight_decay": 0.1,
    "betas": [0.9, 0.999],
    "cubes_per_fish": 256,
    "synthetic_share": 1 / 3,  # arm B: of the training cubes
    "seed": 0,
    # synthetic bladders (section 3.1)
    "synthetic_scale": [0.7, 1.3],
    "synthetic_rotation_deg": 30.0,
    "synthetic_elastic": 0.10,  # of the bladder's size
    "synthetic_edge_sigma_voxels": 1.0,
    "synthetic_clearance_voxels": 2,
}
KINDS = ("real", "synthetic")
LOG_EVERY = 50


def recipe_hash(recipe: Mapping[str, Any] = RECIPE) -> str:
    """Names the cache: a change of the recipe never reuses cubes made under another one."""
    return hashlib.sha256(json.dumps(dict(recipe), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Specimen:
    """A scan and its reference bladder, both on the scan's (z, y, x) grid."""

    name: str
    volume: Volume
    mask: np.ndarray
    _tissue: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, bool)
        if self.mask.shape != self.volume.shape or not self.mask.any():
            raise ValueError(f"{self.name}: the mask must be non-empty and on the scan's grid {self.volume.shape}")

    _apart: np.ndarray | None = field(default=None, repr=False)
    _tissue_voxels: np.ndarray | None = field(default=None, repr=False)

    def tissue_voxels(self, clearance: int) -> np.ndarray:
        """The flat indices of `tissue(clearance)`, where a synthetic bladder's centre is drawn from."""
        if self._tissue_voxels is None:
            self._tissue_voxels = np.flatnonzero(self.tissue(clearance))
        return self._tissue_voxels

    def apart(self, clearance: int) -> np.ndarray:
        """At least `clearance` voxels from the real bladder."""
        if self._apart is None:
            self._apart = ~ndi.binary_dilation(self.mask, iterations=clearance) if clearance else ~self.mask
        return self._apart

    def tissue(self, clearance: int) -> np.ndarray:
        """Inside the fish (above Otsu's split of the scan), at least `clearance` voxels from outside it."""
        if self._tissue is None:
            from skimage.filters import threshold_otsu

            values = self.volume.data.ravel()[:: max(1, self.volume.data.size // 2_000_000)]
            inside = self.volume.data > float(threshold_otsu(values))
            self._tissue = ndi.binary_erosion(inside, iterations=clearance) if clearance else inside
        return self._tissue


def sample_seed(recipe: Mapping[str, Any], specimen: str, kind: str, index: int) -> np.random.Generator:
    """The random numbers of one sample: the same sample whenever it is made, in any order."""
    return np.random.default_rng([int(recipe["seed"]), zlib.crc32(specimen.encode()), KINDS.index(kind), int(index)])


# --- boxes and cubes ------------------------------------------------------------------------------


def bounding_box(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    found = np.argwhere(mask)
    return found.min(0), found.max(0) + 1


def jittered_box_mm(mask: np.ndarray, spacing, shape, rng: np.random.Generator, recipe: Mapping[str, Any] = RECIPE):
    """A box a user might draw around the bladder: grown and shifted at random, clipped to the scan."""
    spacing, limit = np.asarray(spacing, float), np.asarray(shape) * np.asarray(spacing, float)
    low, high = (np.asarray(v, float) * spacing for v in bounding_box(mask))
    extent = high - low
    growth = rng.uniform(*recipe["box_growth"], size=3)
    long_box = rng.random() < recipe["long_box_share"]
    if long_box:
        axis = int(np.argmax(extent))
        growth[axis] = rng.uniform(*recipe["long_box_growth"])
    low, high = low - growth * extent / 2, high + growth * extent / 2
    shift = rng.uniform(-recipe["box_shift"], recipe["box_shift"], size=3) * (high - low)
    low, high = np.clip(low + shift, 0, limit), np.clip(high + shift, 0, limit)
    return low, high, {"growth": growth.tolist(), "shift_mm": shift.tolist(), "long_box": bool(long_box)}


def cube_coordinates_mm(cube: geometry.Cube, angle_deg: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Where each cube voxel's centre lies in the scan (mm, z y x): the cube's grid rotated by `angle_deg`
    in the (y, x) plane and scaled by `scale`, both about the cube's centre."""
    centre = np.asarray(cube.corner_mm) + cube.side_mm / 2
    axes = [cube.centres_mm(axis) - centre[axis] for axis in range(3)]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.stack([centre[0] + scale * z, centre[1] + scale * (cos * y - sin * x), centre[2] + scale * (sin * y + cos * x)])


def region_for(points_mm: np.ndarray, shape, spacing, sigma) -> tuple[slice, slice, slice]:
    """The scan voxels the points need, plus the reach of the smoothing filter (as cube.region_around)."""
    region = []
    for axis, (size, step) in enumerate(zip(shape, spacing)):
        margin = int(4 * sigma[axis] + 0.5) + 2
        start = max(int(np.floor(points_mm[axis].min() / step)) - margin, 0)
        stop = min(int(np.ceil(points_mm[axis].max() / step)) + margin, size)
        region.append(slice(start, max(stop, start + 1)))
    return tuple(region)


def sample_points(data: np.ndarray, offset_voxels, spacing, points_mm: np.ndarray, order: int, shape) -> tuple[np.ndarray, np.ndarray]:
    """`data` (a region of a scan of `shape` starting at `offset_voxels`) at the points: trilinear (order 1)
    or nearest (order 0), holding the edge value inside the scan, as `cube.sample`; and which points lie
    inside the scan."""
    inside = np.ones(points_mm.shape[1:], bool)
    indices = []
    for axis, (step, size, start) in enumerate(zip(spacing, shape, offset_voxels)):
        inside &= (points_mm[axis] >= 0) & (points_mm[axis] < size * step)
        index = points_mm[axis] / step - 0.5 - start
        indices.append(np.clip(index, 0, data.shape[axis] - 1) if order == 1 else np.clip(np.floor(index + 0.5), 0, data.shape[axis] - 1))
    values = ndi.map_coordinates(np.asarray(data, np.float32), indices, order=order, mode="nearest", prefilter=False)
    values = values.astype(np.float32)
    values[~inside] = 0
    return values, inside


@dataclass(frozen=True)
class Augmentation:
    angle_deg: float = 0.0
    scale: float = 1.0
    quarter_turns: int = 0
    flips: tuple[bool, bool, bool] = (False, False, False)
    intensity_scale: float = 1.0
    intensity_offset: float = 0.0  # of the box's std
    noise: float = 0.0  # of the box's std

    @classmethod
    def draw(cls, rng: np.random.Generator, recipe: Mapping[str, Any] = RECIPE) -> "Augmentation":
        return cls(
            angle_deg=float(rng.uniform(-recipe["small_rotation_deg"], recipe["small_rotation_deg"])),
            scale=float(rng.uniform(*recipe["scale"])),
            quarter_turns=int(rng.integers(4)),
            flips=tuple(bool(v) for v in rng.random(3) < 0.5),
            intensity_scale=float(rng.uniform(*recipe["intensity_scale"])),
            intensity_offset=float(rng.uniform(-recipe["intensity_offset"], recipe["intensity_offset"])),
            noise=float(recipe["noise"]),
        )


def turn(array: np.ndarray, augmentation: Augmentation) -> np.ndarray:
    """The exact part of the augmentation: quarter turns in the (y, x) plane, then flips."""
    array = np.rot90(array, augmentation.quarter_turns, axes=(1, 2))
    for axis, flip in enumerate(augmentation.flips):
        if flip:
            array = np.flip(array, axis)
    return np.ascontiguousarray(array)


def make_sample(specimen: Specimen, rng: np.random.Generator, recipe: Mapping[str, Any] = RECIPE,
                augmentation: Augmentation | None = None, box_mm=None) -> dict[str, Any]:
    """One training cube: the network's input (z, y, x, normalised as `segment` does) and its target,
    the reference inside the box, on the cube's grid."""
    volume, spacing = specimen.volume, tuple(float(s) for s in specimen.volume.spacing_mm)
    if box_mm is None:
        low, high, box_meta = jittered_box_mm(specimen.mask, spacing, volume.shape, rng, recipe)
    else:
        (low, high), box_meta = (np.asarray(v, float) for v in box_mm), {"given": True}
    augmentation = augmentation if augmentation is not None else Augmentation.draw(rng, recipe)
    box = tuple(slice(int(np.floor(a / s + 1e-9)), max(int(np.ceil(b / s - 1e-9)), int(np.floor(a / s + 1e-9)) + 1)) for a, b, s in zip(low, high, spacing))
    cube = geometry.cube_on_box(tuple(low), tuple(high))

    points = cube_coordinates_mm(cube, augmentation.angle_deg, augmentation.scale)
    sigma = geometry.antialias_sigma(spacing, cube.spacing_mm * augmentation.scale)
    region = region_for(points, volume.shape, spacing, sigma)
    offset = [w.start for w in region]
    smoothed, _ = geometry.antialias(volume.data[region], spacing, cube.spacing_mm * augmentation.scale)
    values, inside = sample_points(smoothed, offset, spacing, points, 1, volume.shape)
    del smoothed
    in_box = np.zeros(volume.shape, bool)
    in_box[box] = True
    target_scan = (specimen.mask & in_box)[region]
    target, _ = sample_points(target_scan.astype(np.float32), offset, spacing, points, 0, volume.shape)
    target = (target > 0.5) & inside

    # The automatic shift is decided on the scan as it is, as at inference; the intensity augmentation comes
    # after it, so that it never moves air across 0 and changes that decision.
    box_values = np.asarray(volume.data[box], np.float64)
    rule = auto_shift(box_values)
    std = float(box_values.std()) or 1.0
    a, b = augmentation.intensity_scale, augmentation.intensity_offset * std
    shifted = values[inside] + rule["shift"]
    values[inside] = a * shifted + b + rng.normal(0.0, augmentation.noise * std, int(inside.sum())).astype(np.float32)
    prepared, normalisation = prepare(values, inside, 0.0)

    prepared, target = turn(prepared, augmentation), turn(target, augmentation)
    meta = {
        "specimen": specimen.name,
        "box_mm": [list(map(float, low)), list(map(float, high))],
        "box": box_meta,
        "augmentation": augmentation.__dict__ | {"flips": list(augmentation.flips)},
        "cube_spacing_mm": cube.spacing_mm,
        "intensity_shift": rule["shift"],
        "target_voxels": int(target.sum()),
        "normalisation": {key: normalisation[key] for key in ("mean", "std", "share_of_cube_above_0")},
    }
    return {"cube": prepared.astype(np.float32), "target": target, "meta": meta}


def click_from_target(target_zyx: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
    """A random target voxel, as the network's (x, y, z)."""
    found = np.flatnonzero(target_zyx)
    if found.size == 0:
        raise ValueError("The sample has no target voxel to click")
    z, y, x = np.unravel_index(int(found[rng.integers(found.size)]), target_zyx.shape)
    return int(x), int(y), int(z)


# --- synthetic bladders (protocol section 3.1) ----------------------------------------------------


def _rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(0, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def deformed_shape(mask: np.ndarray, spacing, rng: np.random.Generator, recipe: Mapping[str, Any] = RECIPE):
    """The reference shape scaled per axis, rotated and displaced by a smooth random field, on a local
    grid of the scan's voxels. Each source voxel is mapped forward at 8 points, and the result closed, so
    that scaling up leaves no holes. Returns the shape and the parameters drawn."""
    spacing = np.asarray(spacing, float)
    low, high = bounding_box(mask)
    extent_mm = (high - low) * spacing
    scale = rng.uniform(*recipe["synthetic_scale"], size=3)
    forward = _rotation(rng, recipe["synthetic_rotation_deg"]) @ np.diag(scale)
    amplitude = rng.uniform(0, recipe["synthetic_elastic"]) * float(extent_mm.max())

    voxels = np.argwhere(mask[tuple(slice(a, b) for a, b in zip(low, high))]).astype(np.float64)
    offsets = np.array(np.meshgrid(*([[0.25, 0.75]] * 3), indexing="ij")).reshape(3, -1).T
    points = ((voxels[:, None, :] + offsets[None]) * spacing).reshape(-1, 3) - extent_mm / 2  # mm from the shape's centre
    coarse = rng.normal(size=(3, 5, 5, 5))
    where = ((points + extent_mm / 2) / np.maximum(extent_mm, 1e-9) * 4).T
    field_ = np.stack([ndi.map_coordinates(c, where, order=3, mode="nearest") for c in coarse], axis=1)
    field_ *= amplitude / max(float(np.abs(field_).max()), 1e-9)
    moved = (points + field_) @ forward.T

    half = np.ceil(np.abs(moved).max(0) / spacing).astype(int) + 2
    index = np.floor(moved / spacing + half).astype(int)
    shape = np.zeros(tuple(2 * half + 1), bool)
    shape[tuple(index.T)] = True
    shape = ndi.binary_fill_holes(ndi.binary_closing(shape, iterations=1))
    return shape, {"scale": scale.tolist(), "elastic_mm": amplitude}


def synthetic_specimen(specimen: Specimen, rng: np.random.Generator, recipe: Mapping[str, Any] = RECIPE,
                       attempts: int = 400) -> Specimen | None:
    """The scan with one more bladder: the reference shape deformed (`deformed_shape`), placed in the
    fish's tissue apart from the real bladder, its voxels drawn from the real bladder's values and blended
    in over a soft edge. Its mask is the new bladder only; `real_mask` keeps the real one. None when no
    place fits."""
    clearance = int(recipe["synthetic_clearance_voxels"])
    shape, drawn_meta = deformed_shape(specimen.mask, specimen.volume.spacing_mm, rng, recipe)
    tissue, apart = specimen.tissue(clearance), specimen.apart(clearance)
    occupied = np.argwhere(shape)
    half = (np.asarray(shape.shape) - 1) // 2
    size = np.asarray(specimen.volume.shape)
    candidates = specimen.tissue_voxels(clearance)
    if candidates.size == 0:
        return None
    for _ in range(attempts):
        centre = np.array(np.unravel_index(int(candidates[rng.integers(candidates.size)]), tuple(size)))
        where = occupied - half + centre
        if (where < 0).any() or (where >= size).any():
            continue
        index = tuple(where.T)
        if tissue[index].all() and apart[index].all():
            break
    else:
        return None

    data = np.array(specimen.volume.data, np.float32)
    sigma = float(recipe["synthetic_edge_sigma_voxels"])
    reach = int(np.ceil(4 * sigma)) + 1
    window = tuple(slice(max(int(a) - reach, 0), min(int(b) + reach + 1, n)) for a, b, n in zip(where.min(0), where.max(0), size))
    mask = np.zeros(specimen.volume.shape, bool)
    mask[index] = True
    soft = ndi.gaussian_filter(mask[window].astype(np.float32), sigma)
    drawn = rng.choice(np.asarray(specimen.volume.data[specimen.mask], np.float32), size=soft.shape).astype(np.float32)
    blended = soft * drawn + (1 - soft) * data[window]
    data[window] = np.where(specimen.mask[window], data[window], blended)  # the soft edge never reaches the real bladder
    volume = Volume(data, specimen.volume.spacing_mm, specimen.volume.units, specimen.volume.source_format, specimen.volume.source_hash)
    synthetic = Specimen(f"{specimen.name}+synthetic", volume, mask)
    synthetic.real_mask = specimen.mask  # type: ignore[attr-defined]
    synthetic.meta = {**drawn_meta, "centre_voxel": centre.tolist(), "voxels": int(mask.sum())}  # type: ignore[attr-defined]
    return synthetic


def make_synthetic_sample(specimen: Specimen, rng: np.random.Generator, recipe: Mapping[str, Any] = RECIPE) -> dict[str, Any] | None:
    """A training cube on a box around a synthetic bladder; the click is drawn inside the synthetic one."""
    synthetic = synthetic_specimen(specimen, rng, recipe)
    if synthetic is None:
        return None
    low, high, box_meta = jittered_box_mm(synthetic.mask, synthetic.volume.spacing_mm, synthetic.volume.shape, rng, recipe)
    both = Specimen(synthetic.name, synthetic.volume, synthetic.mask | synthetic.real_mask)  # type: ignore[attr-defined]
    sample = make_sample(both, rng, recipe, box_mm=(low, high))
    clickable = make_sample(synthetic, np.random.default_rng(0), recipe, Augmentation(**{**sample["meta"]["augmentation"], "flips": tuple(sample["meta"]["augmentation"]["flips"])}), box_mm=(low, high))
    sample["click_target"] = clickable["target"]
    sample["meta"]["box"] = box_meta
    sample["meta"]["synthetic"] = synthetic.meta  # type: ignore[attr-defined]
    return sample


# --- the cache of encoded cubes ---------------------------------------------------------------------


def cache_folder(root: Path, specimen: str, kind: str, recipe: Mapping[str, Any] = RECIPE) -> Path:
    return Path(root) / recipe_hash(recipe) / specimen / kind


SYNTHETIC_ATTEMPTS = 4  # tries per synthetic cube wanted: a slender fish has little room for a second bladder


def encode(specimen: Specimen, kind: str, count: int, net, root: Path, recipe: Mapping[str, Any] = RECIPE) -> list[Path]:
    """Make and encode `count` samples of one kind, saving each as soon as it is done. Synthetic samples
    that find no place are skipped, up to SYNTHETIC_ATTEMPTS tries per sample wanted. Samples already in the
    cache are kept, so an interrupted run goes on where it stopped."""
    folder = cache_folder(root, specimen.name, kind, recipe)
    folder.mkdir(parents=True, exist_ok=True)
    (folder.parent.parent / "recipe.json").write_text(json.dumps(dict(recipe), indent=1), encoding="utf-8")
    files, skipped, index = [], 0, 0
    limit = count if kind == "real" else count * SYNTHETIC_ATTEMPTS
    while len(files) < count and index < limit:
        path = folder / f"{index:04d}.npz"
        skip_marker = folder / f"{index:04d}.none"
        index += 1
        if path.is_file():
            files.append(path)
            continue
        if skip_marker.is_file():
            skipped += 1
            continue
        rng = sample_seed(recipe, specimen.name, kind, index - 1)
        sample = make_sample(specimen, rng, recipe) if kind == "real" else make_synthetic_sample(specimen, rng, recipe)
        if sample is None:
            skip_marker.write_text("no place for a synthetic bladder", encoding="utf-8")
            skipped += 1
            continue
        embedding = np.asarray(_to_numpy(net.embed(geometry.to_network_axes(sample["cube"]))), np.float32)
        click_target = sample.get("click_target", sample["target"])
        partial = path.with_suffix(".part.npz")
        np.savez(partial, embedding=embedding, target=np.packbits(sample["target"]), click_target=np.packbits(click_target),
                 meta=json.dumps(sample["meta"]))
        partial.replace(path)
        files.append(path)
        logger.info("%s %s: %d/%d (try %d)", specimen.name, kind, len(files), count, index)
    if skipped:
        logger.warning("%s: %d of %d %s tries found no place", specimen.name, skipped, index, kind)
    return files


def training_files(root: Path, names: Iterable[str], synthetic: bool, recipe: Mapping[str, Any] = RECIPE) -> tuple[list[Path], dict[str, int]]:
    """The real cubes of the training fish and, for arm B, synthetic ones making up `synthetic_share` of
    all: split evenly over the training fish that have synthetic cubes (a fish whose scan has no room for
    a second bladder has none), the first ones of each, in the order they were made."""
    names = list(names)
    files = [path for name in names for path in specimen_files(root, [name], ["real"], recipe)]
    counts = {f"{name}/real": len(specimen_files(root, [name], ["real"], recipe)) for name in names}
    if synthetic:
        share = float(recipe["synthetic_share"])
        wanted = round(share / (1 - share) * len(files))
        available = {name: specimen_files(root, [name], ["synthetic"], recipe) for name in names}
        hosts = [name for name in names if available[name]]
        if not hosts:
            raise ValueError(f"No synthetic cubes for any of {names}")
        quota = -(-wanted // len(hosts))
        chosen = []
        for name in hosts:
            take = available[name][: min(quota, wanted - len(chosen))]
            if len(take) < min(quota, wanted - len(chosen)):
                raise ValueError(f"{name}: {len(available[name])} synthetic cubes, {quota} needed; encode more")
            chosen += take
            counts[f"{name}/synthetic"] = len(take)
        files += chosen
    return files, counts


def _to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def load_sample(path: Path) -> dict[str, Any]:
    with np.load(path) as stored:
        size = geometry.SIZE
        target = np.unpackbits(stored["target"])[: size**3].reshape((size,) * 3).astype(bool)
        click = np.unpackbits(stored["click_target"])[: size**3].reshape((size,) * 3).astype(bool)
        return {"embedding": stored["embedding"], "target": target, "click_target": click, "meta": json.loads(str(stored["meta"]))}


# --- training -----------------------------------------------------------------------------------------


def dice_ce_loss(logits, target):
    """Dice + binary cross-entropy on logits, as MONAI's DiceCELoss(sigmoid=True, squared_pred=True), which
    upstream trains with (train.py:136): smoothing 1e-5, both terms weighted 1, mean over the batch."""
    import torch

    probability = torch.sigmoid(logits)
    dims = tuple(range(2, logits.ndim))
    intersection = (probability * target).sum(dims)
    denominator = (probability**2).sum(dims) + (target**2).sum(dims)
    dice = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return dice.mean() + torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def train(files: list[Path], net, recipe: Mapping[str, Any] = RECIPE, log=None) -> list[dict[str, float]]:
    """Train `net`'s prompt encoder and mask decoder on cached samples; the image encoder stays frozen and
    is not run. Returns the mean loss over every LOG_EVERY steps."""
    import torch

    if not files:
        raise ValueError("No training samples")
    torch.manual_seed(int(recipe["seed"]))
    rng = np.random.default_rng(int(recipe["seed"]))
    model = net.model
    for parameter in model.image_encoder.parameters():
        parameter.requires_grad_(False)
    trained = [*model.prompt_encoder.parameters(), *model.mask_decoder.parameters()]
    optimiser = torch.optim.AdamW(trained, lr=recipe["lr"], weight_decay=recipe["weight_decay"], betas=tuple(recipe["betas"]))
    model.prompt_encoder.train()
    model.mask_decoder.train()
    device = next(model.mask_decoder.parameters()).device

    order: list[int] = []
    history, running = [], []
    for step in range(1, int(recipe["steps"]) + 1):
        optimiser.zero_grad()
        total = 0.0
        for _ in range(int(recipe["batch"])):
            if not order:
                order = list(rng.permutation(len(files)))
            sample = load_sample(files[order.pop()])
            embedding = torch.from_numpy(sample["embedding"]).to(device)
            target = torch.from_numpy(np.ascontiguousarray(geometry.to_network_axes(sample["target"]), np.float32))[None, None].to(device)
            loss = dice_ce_loss(net.logits(embedding, click_from_target(sample["click_target"], rng)), target) / int(recipe["batch"])
            loss.backward()
            total += float(loss.detach().cpu())
        optimiser.step()
        running.append(total)
        if step % LOG_EVERY == 0:
            history.append({"step": step, "loss": float(np.mean(running))})
            running = []
            message = f"step {step}/{recipe['steps']}: loss {history[-1]['loss']:.4f}"
            logger.info(message)
            if log:
                log(message)
    model.eval()
    return history


def save_tuned(name: str, net, manifest: Mapping[str, Any]) -> Path:
    """A named tuned set: the tuned parts and a manifest that checks them (assets.require_tuned)."""
    import torch

    folder = assets.tuned_folder(name)
    if folder.exists():
        raise FileExistsError(f"Tuned weights '{name}' exist already in {folder}; choose another name")
    folder.mkdir(parents=True)
    weights = folder / assets.TUNED_FILE
    torch.save(network.tuned_state(net.model), weights)
    full = {**manifest, "name": name, "base_weights_sha256": assets.WEIGHTS_SHA256, "sha256": assets.sha256_file(weights)}
    (folder / assets.TUNED_MANIFEST).write_text(json.dumps(full, indent=1), encoding="utf-8")
    return folder


def specimen_files(root: Path, names: Iterable[str], kinds: Iterable[str], recipe: Mapping[str, Any] = RECIPE) -> list[Path]:
    return sorted(path for name in names for kind in kinds for path in cache_folder(root, name, kind, recipe).glob("[0-9][0-9][0-9][0-9].npz"))


__all__ = ["RECIPE", "Specimen", "Augmentation", "make_sample", "make_synthetic_sample", "synthetic_specimen", "encode",
           "train", "save_tuned", "dice_ce_loss", "click_from_target", "load_sample", "specimen_files", "AUTO"]
