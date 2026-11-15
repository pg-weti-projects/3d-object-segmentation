import functools

import numpy as np
import segmentation_models_pytorch as smp
import torch.nn.functional as F

import torch
from src.core.volume import Volume

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def segment(volume: Volume, parameters: dict) -> np.ndarray:
    if "window" not in parameters:
        raise ValueError(
            "unet needs parameters.window: the [low, high] intensity range it was trained on"
        )
    low, high = parameters["window"]
    model, preprocessing = _load(
        parameters.get("checkpoint", "src/models/bladder_unet_2d.pt")
    )
    size = preprocessing["img_size"]

    depth, height, width = volume.shape
    side = round(max(height, width) * (1 + 2 * preprocessing["eval_margin"]))
    top, left = (side - height) // 2, (side - width) // 2
    x = torch.zeros(depth, 1, side, side)
    x[:, 0, top : top + height, left : left + width] = torch.from_numpy(
        (np.clip(volume.data, low, high) - low) / (high - low)
    )

    logits = []
    with torch.no_grad():
        for batch in x.split(16):
            batch = F.interpolate(batch.to(DEVICE), size=(size, size), mode="bilinear")
            logits.append(
                F.interpolate(model(batch), size=(side, side), mode="bilinear").cpu()
            )
    inside = torch.cat(logits)[:, 0, top : top + height, left : left + width]
    return (inside > 0).numpy().astype(np.uint8)


@functools.lru_cache(maxsize=1)
def _load(checkpoint: str):
    saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model = smp.Unet(
        saved["preprocessing"]["encoder"],
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    model.load_state_dict(saved["model"])
    return model.to(DEVICE).eval(), saved["preprocessing"]
