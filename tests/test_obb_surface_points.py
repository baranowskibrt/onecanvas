"""Regression test for the OBB surface-point marker sampler.

Guards the self-shadowing bug where ``_common._sample_obb_surface_points``
imported the ``utils.bbox`` dense face sampler and then defined a wrapper of
the same name, so the wrapper called itself recursively and every synthetic
box collapsed to its 8 corners instead of a dense surface cloud. With the fix
in place the wrapper returns ``n_total`` points sampled across the OBB faces.
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from onecanvas.data.spatial_pretraining._common import _sample_obb_surface_points


def test_surface_points_dense_not_corners_only():
    center = torch.tensor([1.0, 2.0, 3.0])
    R = torch.eye(3)
    dims = (2.0, 3.0, 1.5)
    pts = _sample_obb_surface_points(center, R, dims, n_total=100)
    assert pts.shape == (100, 3)
    # Corners-only collapse (the shadowing bug) yields <= 8 unique rows.
    assert torch.unique(pts, dim=0).shape[0] > 8


def test_surface_points_corners_floor():
    center = torch.zeros(3)
    R = torch.eye(3)
    dims = (1.0, 1.0, 1.0)
    pts = _sample_obb_surface_points(center, R, dims, n_total=4)
    # Budget below 8 floors to the 8 corners.
    assert pts.shape == (8, 3)


if __name__ == "__main__":
    test_surface_points_dense_not_corners_only()
    test_surface_points_corners_floor()
    print("PASS")
