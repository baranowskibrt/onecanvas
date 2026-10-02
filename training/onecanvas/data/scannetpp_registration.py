"""ScanNet++ iPhone-to-mesh registration, and which scenes have one.

`scans/segments_anno.json` annotates the laser-scan mesh. `pose_intrinsic_imu.json`
is the ARKit trajectory, a different coordinate system, and nothing in the
loader related the two. Every ScanNet++ object annotation therefore landed
somewhere the object is not, by up to a metre horizontally, with taught boxes
inside walls. `scripts/fit_scannetpp_arkit_to_mesh.py` recovers the relation
from the dataset's own `iphone/colmap` registration and writes the registry
this module reads.

THERE IS NO LIST OF BAD SCENES IN THIS FILE OR ANY OTHER, deliberately. A scene
is usable exactly when the registry holds a transform for it, and the registry
is regenerated from the data by the fitter. A scene refuses because its own
numbers refuse, so a re-download, a dataset fix or a better fitter changes the
answer without anyone editing code. The refusal reasons are carried in the
registry's `refused` map for reporting, never consulted for a decision.

Why scenes refuse at all: ARKit tracking breaks partway through some captures,
so no single rigid transform describes the whole trajectory and the fit lands
between a clean stretch and a broken one. Measured over the refused scenes, the
clean stretch registers to about 2 cm while the broken stretch is metres out.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "scannetpp_arkit_to_mesh.json")

_cache: Optional[dict] = None


def _registry() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(REGISTRY_PATH) as fh:
                _cache = json.load(fh)
        except FileNotFoundError:
            # An absent registry makes every ScanNet++ scene unregistered, which
            # is the same path a refused scene takes. It does NOT silently fall
            # back to the old rotation-only placement: that placement was the
            # defect, and reviving it here would hide a missing registry behind
            # plausible-looking boxes.
            _cache = {"scenes": {}, "refused": {}}
    return _cache


def arkit_to_mesh(scene_id: str) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """``(R, scale, t)`` taking an ARKit camera pose into the mesh frame.

    Returned decomposed rather than as one 4x4 because the two halves apply to
    different things. ``R`` rotates the camera's orientation and ``scale`` and
    ``t`` place its centre. The scale corrects ARKit's drift in POSITION, which
    is metric to a few percent over a capture. It must not reach the depth map,
    which comes from the LiDAR and is metric on its own, nor multiply into the
    rotation block, which would leave it non-orthonormal for everything
    downstream that assumes otherwise.
    """
    rec = _registry()["scenes"].get(str(scene_id))
    if rec is None:
        return None
    T = np.asarray(rec["transform"], dtype=np.float64).reshape(4, 4)
    s = float(rec["scale"])
    return T[:3, :3] / s, s, T[:3, 3].copy()


def is_registered(scene_id: str) -> bool:
    """Whether this scene's annotations can be placed on its canvas at all."""
    return str(scene_id) in _registry()["scenes"]


def refusal_reason(scene_id: str) -> Optional[str]:
    """Why the fitter refused this scene, for reports. Never a control flow key."""
    return _registry()["refused"].get(str(scene_id))
