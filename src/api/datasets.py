from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pydicom
import tifffile
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.core import load_tiff_series, load_volume


DATA_EXTENSIONS = {".dcm", ".tif", ".tiff"}
SUPPORTED_UPLOAD_EXTENSIONS = DATA_EXTENSIONS | {".json", ".nii"}
IGNORED_NAMES = {".DS_Store"}


class DatasetError(ValueError):
    """Raised when a dataset source cannot be used."""


class DatasetCatalog:
    """Background-loaded view of the server-side datasets."""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self._lock = threading.Lock()
        self._datasets: list[dict[str, Any]] = []
        self._status = "pending"
        self._error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._status == "loading":
                return
            self._status, self._error = "loading", None
        threading.Thread(target=self.refresh, name="dataset-index", daemon=True).start()

    def refresh_async(self) -> None:
        """Refresh the catalog without blocking the HTTP request."""
        self.start()

    def refresh(self) -> None:
        with self._lock:
            if self._status != "loading":
                self._status, self._error = "loading", None
        try:
            datasets = list_datasets(self.raw_dir)
        except (OSError, ValueError) as error:
            with self._lock:
                self._status, self._error = "error", str(error)
            return
        with self._lock:
            self._datasets, self._status = datasets, "ready"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"status": self._status, "error": self._error, "datasets": list(self._datasets)}


def import_dataset(
    raw_dir: str | Path,
    dataset_id: str,
    uploads: list[FileStorage],
    folder_uploads: list[FileStorage] | None = None,
) -> dict[str, Any]:
    """Store uploaded files as a new server-side dataset."""
    if not dataset_id or dataset_id != Path(dataset_id).name:
        raise DatasetError("Dataset name must be a single safe folder name.")
    safe_id = secure_filename(dataset_id)
    if not safe_id or safe_id != dataset_id:
        raise DatasetError("Dataset name contains unsupported characters.")
    folder_uploads = [
        upload for upload in (folder_uploads or []) if not _is_ignored_upload(upload)
    ]
    uploads = [
        upload for upload in [*uploads, *folder_uploads] if not _is_ignored_upload(upload)
    ]
    if not uploads:
        raise DatasetError("Select at least one supported file.")
    for upload in uploads:
        _validate_upload_extension(upload)
    for upload in folder_uploads:
        _validate_upload_extension(upload, DATA_EXTENSIONS, "Folder uploads accept only")

    root = Path(raw_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = _safe_child(root, safe_id)
    if destination.exists():
        raise DatasetError(f"Dataset '{safe_id}' already exists.")

    staged = root / f".{safe_id}.uploading"
    try:
        staged.mkdir()
        for upload in uploads:
            relative_name = _upload_relative_name(upload)
            target = _safe_child(staged, relative_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            upload.save(target)
        staged.rename(destination)
    except (OSError, ValueError) as error:
        _remove_tree(staged)
        raise DatasetError(f"Could not save dataset: {error}") from error

    return next(item for item in list_datasets(root) if item["id"] == safe_id)


def delete_dataset(raw_dir: str | Path, dataset_id: str) -> None:
    """Delete one complete dataset from the server data directory."""
    root = Path(raw_dir).resolve()
    dataset = _safe_child(root, dataset_id)
    if not dataset.is_dir():
        raise DatasetError(f"Dataset '{dataset_id}' does not exist.")
    _remove_tree(dataset)


def rename_dataset(raw_dir: str | Path, dataset_id: str, new_dataset_id: str) -> None:
    """Rename a dataset directory without allowing paths outside raw_dir."""
    if not new_dataset_id or new_dataset_id != Path(new_dataset_id).name:
        raise DatasetError("Dataset name must be a single safe folder name.")
    safe_id = secure_filename(new_dataset_id)
    if not safe_id or safe_id != new_dataset_id:
        raise DatasetError("Dataset name contains unsupported characters.")
    root = Path(raw_dir).resolve()
    source = _safe_child(root, dataset_id)
    destination = _safe_child(root, safe_id)
    if not source.is_dir():
        raise DatasetError(f"Dataset '{dataset_id}' does not exist.")
    if destination.exists() and destination != source:
        raise DatasetError(f"Dataset '{safe_id}' already exists.")
    try:
        source.rename(destination)
    except OSError as error:
        raise DatasetError(f"Could not rename dataset: {error}") from error


def delete_source(raw_dir: str | Path, dataset_id: str, source: str) -> None:
    """Delete one source item from a dataset."""
    root = Path(raw_dir).resolve()
    dataset = _safe_child(root, dataset_id)
    if not dataset.is_dir():
        raise DatasetError(f"Dataset '{dataset_id}' does not exist.")
    item = _safe_child(dataset, source)
    if not item.exists() or item == dataset:
        raise DatasetError(f"Source '{source}' does not exist.")
    if item.is_dir():
        _remove_tree(item)
    else:
        item.unlink()


def list_datasets(raw_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(raw_dir)
    if not root.is_dir():
        return []
    return [_describe_dataset(path, root) for path in sorted(root.iterdir()) if path.is_dir()]


def resolve_source(raw_dir: str | Path, dataset_id: str, source: str) -> Path:
    root = Path(raw_dir).resolve()
    dataset_dir = _safe_child(root, dataset_id)
    if not dataset_dir.is_dir():
        raise DatasetError(f"Dataset '{dataset_id}' does not exist.")
    path = _safe_child(dataset_dir, source)
    if not path.exists():
        raise DatasetError(f"Source '{source}' does not exist in dataset '{dataset_id}'.")
    if path.is_file() and not _is_single_file_source(path):
        raise DatasetError("Only JSON, NIfTI, and data files can be selected.")
    return path


def load_dataset(raw_dir: str | Path, dataset_id: str, source: str):
    path = resolve_source(raw_dir, dataset_id, source)
    if path.is_file() and path.name.lower().endswith(".json.gz"):
        raise DatasetError("Compressed JSON is detected but is not supported yet.")
    try:
        if path.is_dir() and _contains_suffix(path, {".tif", ".tiff"}):
            return load_tiff_series(
                [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}]
            )
        return load_volume(path)
    except (OSError, ValueError) as error:
        raise DatasetError(f"Source '{source}' is invalid: {error}") from error


def _describe_dataset(path: Path, root: Path) -> dict[str, Any]:
    sources = []
    for child in sorted(path.iterdir()):
        if child.name in IGNORED_NAMES:
            continue
        source = _describe_source(child, path)
        if source is not None:
            sources.append(source)
    return {"id": path.name, "path": str(path.relative_to(root)), "sources": sources}


def _describe_source(path: Path, dataset_dir: Path) -> dict[str, Any] | None:
    relative = str(path.relative_to(dataset_dir))
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file() and item.name not in IGNORED_NAMES]
        suffixes = {item.suffix.lower() for item in files}
        if suffixes & {".dcm"}:
            kind, label = "dicom-folder", "DICOM data folder"
        elif suffixes & {".tif", ".tiff"}:
            kind, label = "tiff-folder", "TIFF data folder"
        else:
            kind, label = "folder", "Folder"
        source = _source(path, relative, kind, label, len(files), sum(item.stat().st_size for item in files))
        if any(item.parent != path for item in files):
            source["valid"] = False
            source["error"] = "Nested folders are not supported in Raw Data."
        elif any(item.suffix.lower() not in DATA_EXTENSIONS for item in files):
            source["valid"] = False
            source["error"] = "Folders may contain only DICOM or TIFF data files."
        return source

    suffix = path.suffix.lower()
    if path.name.lower().endswith(".json.gz"):
        kind, label = "json-gz", "Compressed JSON file (.json.gz)"
    elif path.name.lower().endswith(".nii.gz"):
        kind, label = "nifti-gz", "Compressed NIfTI volume (.nii.gz)"
    elif suffix == ".nii":
        kind, label = "nifti", "NIfTI volume (.nii)"
    elif suffix == ".json":
        kind, label = "json", "JSON file"
    elif suffix == ".tar":
        kind, label = "archive", "TAR Archive"
    elif suffix in DATA_EXTENSIONS:
        kind, label = "file", f"Data file ({suffix})"
    else:
        file_type = f"{suffix} file" if suffix else "file without extension"
        kind, label = "unsupported", f"Unsupported file ({file_type})"
        source = _source(path, relative, kind, label, 1, path.stat().st_size)
        source["valid"] = False
        source["error"] = "This file type is not supported in Raw Data."
        return source
    return _source(path, relative, kind, label, 1, path.stat().st_size)


def _source(path: Path, relative: str, kind: str, label: str, count: int, size: int) -> dict[str, Any]:
    valid = None
    error = None
    if kind in {"json-gz", "nifti-gz", "archive"}:
        valid, error = False, "Compressed files are not supported in Raw Data."
    elif kind in {"dicom-folder", "tiff-folder", "json", "nifti"}:
        try:
            _validate_source(path, kind)
            valid = True
        except (OSError, ValueError, pydicom.errors.InvalidDicomError) as exc:
            valid, error = False, str(exc)
    return {
        "id": relative,
        "label": label,
        "kind": kind,
        "valid": valid,
        "error": error,
        "file_count": count,
        "size_bytes": size,
    }


def _validate_source(path: Path, kind: str) -> None:
    """Validate source structure without loading a complete large volume."""
    if kind in {"json", "nifti"}:
        load_volume(path)
        return
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in DATA_EXTENSIONS
    )
    if not files:
        raise ValueError("source contains no supported data files")
    if kind == "dicom-folder":
        pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
    elif kind == "tiff-folder":
        with tifffile.TiffFile(files[0]) as image:
            if not image.pages:
                raise ValueError("TIFF source contains no pages")


def _safe_child(parent: Path, name: str) -> Path:
    candidate = (parent / name).resolve()
    if candidate != parent and parent not in candidate.parents:
        raise DatasetError("Invalid dataset path.")
    return candidate


def _upload_relative_name(upload: FileStorage) -> str:
    filename = upload.filename or ""
    parts = [part for part in filename.replace("\\", "/").split("/") if part not in {"", ".", ".."}]
    if not parts:
        raise DatasetError("An uploaded file has no valid name.")
    safe_parts = [secure_filename(part) for part in parts]
    if any(not part for part in safe_parts):
        raise DatasetError("An uploaded file has an unsupported name.")
    return str(Path(*safe_parts))


def _validate_upload_extension(
    upload: FileStorage,
    allowed_extensions: set[str] = SUPPORTED_UPLOAD_EXTENSIONS,
    error_prefix: str = "Accepted file types are",
) -> None:
    relative_name = _upload_relative_name(upload)
    lower_name = relative_name.lower()
    suffix = ".nii.gz" if lower_name.endswith(".nii.gz") else Path(relative_name).suffix.lower()
    if suffix not in allowed_extensions:
        accepted = ", ".join(sorted(allowed_extensions))
        raise DatasetError(
            f"Unsupported file '{relative_name}'. {error_prefix}: {accepted}."
        )


def _is_ignored_upload(upload: FileStorage) -> bool:
    return Path(upload.filename or "").name in IGNORED_NAMES


def _is_single_file_source(path: Path) -> bool:
    return (
        path.suffix.lower() in DATA_EXTENSIONS | {".json", ".nii"}
    )


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    path.rmdir()


def _contains_suffix(path: Path, suffixes: set[str]) -> bool:
    return any(item.is_file() and item.suffix.lower() in suffixes for item in path.rglob("*"))
