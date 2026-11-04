"""
The upstream source and weights live outside the repository, in `sam-med3d-assets`
next to it, or wherever SAM_MED3D_ASSETS points. Both are pinned. No torch here, so
that the method can say what is missing without having it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

ASSETS_VARIABLE = "SAM_MED3D_ASSETS"
DEFAULT_ASSETS = Path(__file__).resolve().parents[4].parent / "sam-med3d-assets"

SOURCE_URL = "https://github.com/uni-medical/SAM-Med3D.git"
SOURCE_COMMIT = "f3de1fa10da98e46f49f176773d2b1e306ba131f"
SOURCE_DIR = "SAM-Med3D"

WEIGHTS_REPOSITORY_COMMIT = "fc482a040ea69cdc9ae576a1d6bd9db02ab1994c"
WEIGHTS_URL = f"https://huggingface.co/blueyo0/SAM-Med3D/resolve/{WEIGHTS_REPOSITORY_COMMIT}/sam_med3d_turbo.pth"
WEIGHTS_SHA256 = "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c"
WEIGHTS_FILE = Path("SAM-Med3D-weights") / "sam_med3d_turbo.pth"

INSTALL_COMMAND = "python scripts/install_sam_med3d.py"


class NotInstalledError(RuntimeError):
    """SAM-Med3D cannot run here. The message says what is missing and what to do."""


@dataclass(frozen=True)
class Assets:
    source: Path
    weights: Path


def locate() -> Assets:
    root = Path(os.environ.get(ASSETS_VARIABLE) or DEFAULT_ASSETS)
    return Assets(root / SOURCE_DIR, root / WEIGHTS_FILE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit(source: Path) -> str | None:
    """The commit a clone is on, or None when it is not a git clone."""
    try:
        found = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=False
        )
    except OSError:
        return None
    return found.stdout.strip() if found.returncode == 0 else None


def missing() -> list[str]:
    """What stops the method from running, cheaply: no hashing of the weights here."""
    assets = locate()
    problems = []
    if importlib.util.find_spec("torch") is None:
        problems.append("the Python package torch")
    if not (assets.source / "segment_anything" / "build_sam3D.py").is_file():
        problems.append(f"the upstream source in {assets.source}")
    if not assets.weights.is_file():
        problems.append(f"the weights in {assets.weights}")
    return problems


def require() -> Assets:
    """The assets, verified against the pins, or a NotInstalledError that says what to do."""
    problems = missing()
    if problems:
        raise NotInstalledError(
            f"SAM-Med3D is not installed: missing {', '.join(problems)}. "
            f"Install it with: {INSTALL_COMMAND} (or set {ASSETS_VARIABLE} to where the assets are)."
        )
    assets = locate()
    commit = source_commit(assets.source)
    if commit != SOURCE_COMMIT:
        raise NotInstalledError(
            f"SAM-Med3D source in {assets.source} is at commit {commit}, expected {SOURCE_COMMIT}. "
            f"Fix it with: {INSTALL_COMMAND}"
        )
    digest = sha256_file(assets.weights)
    if digest != WEIGHTS_SHA256:
        raise NotInstalledError(
            f"SAM-Med3D weights in {assets.weights} have SHA-256 {digest}, expected {WEIGHTS_SHA256}. "
            f"Fix them with: {INSTALL_COMMAND}"
        )
    return assets
