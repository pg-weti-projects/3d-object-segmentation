import numpy as np
import pytest

smp = pytest.importorskip("segmentation_models_pytorch")
torch = pytest.importorskip("torch")
from src.core.volume import Volume
from src.models import unet

WINDOW = [3815, 20800]


def _volume(data):
    return Volume(data, (1.0, 0.8, 0.8), "raw", "nifti", "0" * 64)


def test_checkpoint_loading_and_repeatability(tmp_path):
    torch.manual_seed(0)
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=1, classes=1)
    checkpoint = str(tmp_path / "unet.pt")
    preprocessing = {"img_size": 64, "eval_margin": 0.2, "encoder": "resnet18"}
    torch.save(
        {"model": model.state_dict(), "preprocessing": preprocessing}, checkpoint
    )
    data = np.random.default_rng(0).uniform(3000, 22000, (5, 30, 50)).astype(np.float32)
    parameters = {"checkpoint": checkpoint, "window": WINDOW}

    first = unet.segment(_volume(data), parameters)

    assert first.shape == data.shape and first.dtype == np.uint8
    assert np.array_equal(first, unet.segment(_volume(data), parameters))
