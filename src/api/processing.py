from pathlib import Path
import tempfile

import numpy as np
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.core import PreprocessingConfig, load_volume, preprocess


ALLOWED_EXTENSIONS = {".dcm", ".json", ".tif", ".tiff"}


def process_uploads(uploads: list[FileStorage], form) -> dict:
    """Load uploaded files, preprocess the volume, and return view data."""
    suffixes = {Path(file.filename).suffix.lower() for file in uploads}
    if not suffixes.issubset(ALLOWED_EXTENSIONS):
        raise ValueError("Supported files are .dcm, .json, .tif, and .tiff.")

    config = preprocessing_config(form)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        volume = _load_uploaded_volume(uploads, suffixes, directory)
        processed = preprocess(volume, config)

    return {
        "input_shape": volume.shape,
        "output_shape": processed.shape,
        "input_spacing": volume.spacing_mm,
        "output_spacing": processed.spacing_mm,
        "units": processed.units,
        "source_format": processed.source_format,
        "source_hash": processed.source_hash,
        "minimum": float(processed.data.min()),
        "maximum": float(processed.data.max()),
        "slice_index": processed.shape[0] // 2,
        "slice_data": slice_for_preview(processed.data),
    }


def preprocessing_config(form) -> PreprocessingConfig:
    """Build preprocessing settings from submitted form values."""
    target = tuple(float(value.strip()) for value in form["target_spacing"].split(","))
    hu_range = tuple(float(value.strip()) for value in form["hu_range"].split(","))
    if len(target) != 3 or len(hu_range) != 2:
        raise ValueError("Invalid preprocessing parameter format.")
    return PreprocessingConfig(
        target_spacing=target,
        hu_range=hu_range,
        invert_hu=form.get("invert_hu") == "on",
        denoise=form.get("denoise") == "on",
    )


def slice_for_preview(data: np.ndarray) -> list[list[float]]:
    """Normalize the middle axial slice for browser rendering."""
    image = np.asarray(data[data.shape[0] // 2], dtype=np.float32)
    low, high = float(image.min()), float(image.max())
    if high > low:
        image = (image - low) / (high - low)
    else:
        image = np.zeros_like(image)
    return image.round(4).tolist()


def _load_uploaded_volume(uploads: list[FileStorage], suffixes: set[str], directory: Path) -> np.ndarray:
    """Load a volume from uploaded files, either a single file or multiple DICOM slices."""
    if len(uploads) == 1:
        path = directory / secure_filename(uploads[0].filename)
        uploads[0].save(path)
        return load_volume(path)
    if suffixes == {".dcm"}:
        for index, uploaded in enumerate(uploads):
            path = directory / f"{index:04d}_{secure_filename(uploaded.filename)}"
            uploaded.save(path)
        return load_volume(directory)
    raise ValueError("Select one file or multiple DICOM slices.")
