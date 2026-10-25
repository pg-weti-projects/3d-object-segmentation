from __future__ import annotations

import threading
import uuid
import json
import struct
import zlib
from typing import Any
from pathlib import Path
import numpy as np

from .datasets import load_dataset
from .processing import process_volume
from .datasets import DatasetError


class ProcessingJobStore:
    """Run volume preprocessing in background threads and retain its result."""

    def __init__(self, results_dir: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._results_dir = Path(results_dir) if results_dir is not None else None
        if self._results_dir is not None:
            self._load_persisted_jobs()

    def start(self, raw_dir, dataset_id: str, source: str, form: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "result": None,
                "error": None,
                "dataset_id": dataset_id,
                "source": source,
            }
        thread = threading.Thread(
            target=self._run,
            args=(job_id, raw_dir, dataset_id, source, form),
            name=f"preprocess-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {"id": job_id, "status": job["status"], "error": job["error"]}

    def result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] != "complete":
                raise ValueError("Processing has not completed.")
            return job["result"]

    def source_info(self, job_id: str) -> dict[str, str | None]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {"dataset_id": job.get("dataset_id"), "source": job.get("source")}

    def preview_slice(self, job_id: str, plane: str, index: int) -> list[list[float]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] != "complete":
                raise ValueError("Processing has not completed.")
            result = job["result"]
        slices_path = self._results_dir / "preprocessed" / f"{job_id}.npy" if self._results_dir else None
        if slices_path and slices_path.exists():
            data = np.load(slices_path, mmap_mode="r")
            axis = {"axial": 0, "coronal": 1, "sagittal": 2}.get(plane)
            if axis is None:
                raise ValueError("Unknown preview plane.")
            if index < 0 or index >= data.shape[axis]:
                raise ValueError("Preview slice is out of range.")
            return np.take(data, index, axis=axis).tolist()
        slices = result.get("slices", [])
        if plane == "axial":
            return slices[index]
        if plane == "coronal":
            return [row[index] for row in slices]
        if plane == "sagittal":
            return [[row[index] for row in layer] for layer in slices]
        raise ValueError("Unknown preview plane.")

    def preview_png(self, job_id: str, plane: str, index: int) -> bytes:
        """Return one normalized preview slice as a compact grayscale PNG."""
        image = np.asarray(self.preview_slice(job_id, plane, index), dtype=np.float32)
        minimum, maximum = float(np.nanmin(image)), float(np.nanmax(image))
        if maximum > minimum:
            image = (image - minimum) / (maximum - minimum)
        else:
            image = np.zeros_like(image)
        pixels = np.clip(image * 255, 0, 255).astype(np.uint8)
        height, width = pixels.shape
        raw = b"".join(b"\x00" + row.tobytes() for row in pixels)

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
            )

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, level=1))
            + chunk(b"IEND", b"")
        )

    def completed_for_sources(self) -> dict[str, str]:
        """Return the latest persisted result job for each dataset source."""
        with self._lock:
            completed = [
                (job_id, job)
                for job_id, job in self._jobs.items()
                if job["status"] == "complete" and job.get("dataset_id") and job.get("source")
            ]
        return {
            f"{job['dataset_id']}::{job['source']}": job_id
            for job_id, job in completed
        }

    def active_for_sources(self) -> dict[str, str]:
        """Return queued or running jobs indexed by their dataset source."""
        with self._lock:
            active = [
                (job_id, job)
                for job_id, job in self._jobs.items()
                if job["status"] in {"queued", "processing"}
                and job.get("dataset_id") and job.get("source")
            ]
        return {
            f"{job['dataset_id']}::{job['source']}": job_id
            for job_id, job in active
        }

    def _run(self, job_id, raw_dir, dataset_id, source, form) -> None:
        self._update(job_id, status="processing")
        try:
            volume = load_dataset(raw_dir, dataset_id, source)
            result = process_volume(volume, form, include_slices=False)
        except (DatasetError, OSError, ValueError) as error:
            self._update(job_id, status="failed", error=str(error))
        else:
            self._update(job_id, status="complete", result=result)
            self._persist(job_id)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def _persist(self, job_id: str) -> None:
        if self._results_dir is None:
            return
        with self._lock:
            job = dict(self._jobs[job_id])
        directory = self._results_dir / "preprocessed"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{job_id}.json"
        temporary = directory / f".{job_id}.json.tmp"
        result = dict(job["result"])
        preview_data = result.pop("_preview_data", None)
        if preview_data is None:
            preview_data = np.asarray(result.pop("slices"), dtype=np.float32)
        else:
            preview_data = np.asarray(preview_data, dtype=np.float32)
        np.save(directory / f"{job_id}.npy", preview_data)
        job["result"] = result
        temporary.write_text(json.dumps(job), encoding="utf-8")
        temporary.replace(target)

    def _load_persisted_jobs(self) -> None:
        directory = self._results_dir / "preprocessed"
        if not directory.is_dir():
            return
        for path in directory.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job.get("status") == "complete" and isinstance(job.get("result"), dict):
                    slices_path = path.with_suffix(".npy")
                    slices = job["result"].pop("slices", None)
                    if slices is not None and not slices_path.exists():
                        np.save(slices_path, np.asarray(slices, dtype=np.float32))
                        path.write_text(json.dumps(job), encoding="utf-8")
                    self._jobs[path.stem] = job
            except (OSError, ValueError, TypeError):
                continue
