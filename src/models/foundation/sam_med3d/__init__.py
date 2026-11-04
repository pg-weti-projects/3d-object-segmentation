"""SAM-Med3D as a segmentation method: one click, one 128^3 cube, no training.

    cube.py      where the cube goes, sampling and mapping back (no torch)
    assets.py    finding and verifying the upstream source and weights (no torch)
    network.py   loading the network and predicting (the only part that needs torch)
    pipeline.py  segment(volume, parameters), the entry point of the runner
"""

from .pipeline import segment

__all__ = ["segment"]
