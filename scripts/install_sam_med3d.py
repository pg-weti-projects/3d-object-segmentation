"""
scripts/install_sam_med3d.py

Installs what the sam_med3d method needs, for the server this runs on: torch, the
upstream source at its pinned commit, and the weights with their SHA-256 checked.
Then it runs one prediction to confirm. Safe to re-run: what is already right is left
alone. Torch stays out of requirements.txt, because the other methods do not need it.

Usage:
    python scripts/install_sam_med3d.py
    python scripts/install_sam_med3d.py --assets /some/folder     # also: SAM_MED3D_ASSETS
    python scripts/install_sam_med3d.py --torch-index-url https://download.pytorch.org/whl/cpu
    python scripts/install_sam_med3d.py --no-check                # skip the test prediction
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.models.foundation.sam_med3d import assets  # noqa: E402
except ModuleNotFoundError:  # a server without the project's requirements yet: main() installs them first
    assets = None

GIT = ["git", "-c", "http.sslBackend=schannel"] if sys.platform == "win32" else ["git"]
REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def ensure_requirements() -> None:
    global assets
    if assets is not None:
        return
    print("requirements: installing the project's requirements ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)], check=True)
    from src.models.foundation.sam_med3d import assets as installed

    assets = installed


def ensure_torch(index_url: str | None = None) -> None:
    if importlib.util.find_spec("torch") is not None:
        print("torch: already installed")
        return
    command = [sys.executable, "-m", "pip", "install", "torch"]
    if index_url:
        command += ["--index-url", index_url]
    print("torch: installing ...")
    subprocess.run(command, check=True)


def ensure_source(source: Path) -> None:
    if assets.source_commit(source) == assets.SOURCE_COMMIT:
        print(f"source: {source} is at the pinned commit")
        return
    if not source.exists():
        print(f"source: cloning into {source} ...")
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([*GIT, "clone", assets.SOURCE_URL, str(source)], check=True)
    elif assets.source_commit(source) is None:
        raise RuntimeError(f"{source} exists but is not a git clone; move it away and run this again")
    else:
        status = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], capture_output=True, text=True)
        if status.stdout.strip():
            raise RuntimeError(f"{source} has local changes; resolve them and run this again")
    subprocess.run([*GIT, "-C", str(source), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", assets.SOURCE_COMMIT], check=True)
    print(f"source: {source} is now at the pinned commit")


def download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        written, shown = 0, 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            written += len(chunk)
            if written - shown >= 50 * 1024 * 1024:
                shown = written
                print(f"weights: {written // 2**20} of {total // 2**20} MB", flush=True)


def ensure_weights(weights: Path, fetch: Callable[[str, Path], None] = download) -> None:
    if weights.is_file() and assets.sha256_file(weights) == assets.WEIGHTS_SHA256:
        print(f"weights: {weights} has the pinned SHA-256")
        return
    weights.parent.mkdir(parents=True, exist_ok=True)
    partial = weights.with_name(weights.name + ".part")
    print(f"weights: downloading to {weights} ...")
    try:
        fetch(assets.WEIGHTS_URL, partial)
        digest = assets.sha256_file(partial)
        if digest != assets.WEIGHTS_SHA256:
            raise RuntimeError(f"downloaded weights have SHA-256 {digest}, expected {assets.WEIGHTS_SHA256}")
        partial.replace(weights)
    finally:
        partial.unlink(missing_ok=True)
    print("weights: done")


def confirm() -> None:
    """One real prediction on a small synthetic scan; it says nothing about quality."""
    import numpy as np

    from src.core.volume import Volume
    from src.models.foundation.sam_med3d import segment

    print("check: one prediction, on the CPU; the first load takes a while ...", flush=True)
    data = np.full((40, 40, 40), -1000.0, dtype=np.float32)
    data[8:32, 8:32, 8:32] = 40.0
    data[16:24, 16:24, 16:24] = 300.0
    volume = Volume(data, (3.0, 3.0, 3.0), "HU", "synthetic", "0" * 64)
    started = time.perf_counter()
    mask = segment(volume, {"cube": "click", "seed_zyx": [20, 20, 20]})
    print(f"check: ok, {time.perf_counter() - started:.1f} s, mask of shape {mask.shape}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, help="where the source and weights go (also: SAM_MED3D_ASSETS)")
    parser.add_argument("--torch-index-url", help="pip index for torch, for example the CPU-only one")
    parser.add_argument("--no-check", action="store_true", help="skip the test prediction")
    arguments = parser.parse_args(argv)
    if arguments.assets:
        os.environ["SAM_MED3D_ASSETS"] = str(arguments.assets)
    try:
        ensure_requirements()
        located = assets.locate()
        ensure_torch(arguments.torch_index_url)
        ensure_source(located.source)
        ensure_weights(located.weights)
        assets.require()
        if not arguments.no_check:
            confirm()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("SAM-Med3D is installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
