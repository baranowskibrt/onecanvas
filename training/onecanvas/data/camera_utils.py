"""Camera intrinsics and depth-map geometry utilities."""

import torch

# Original full-resolution dimensions (before any border crop or resize).
_ORIG_W, _ORIG_H = 1296, 968


def adjust_intrinsics_for_resize(intrinsics, orig_w, orig_h, new_w, new_h):
    """Scale intrinsics when the image has been resized from orig_w×orig_h to new_w×new_h.

    The stored intrinsics are in post-border-crop coordinates of the *original*
    resolution.  Because the border-crop ratio is the same for both resolutions the
    crop fraction cancels and the effective scale is simply new/orig for each axis.

    Supports both:
      - 1D [fx, fy, cx, cy] format (produced by _compute_da3_geometry)
      - 2D 3×3 camera matrix format

    Transforms: fx *= sx, cx *= sx, fy *= sy, cy *= sy.
    """
    if intrinsics is None:
        return None
    intr = torch.as_tensor(intrinsics, dtype=torch.float32).clone()
    sx = new_w / orig_w
    sy = new_h / orig_h
    # 1D [fx, fy, cx, cy] format
    if intr.ndim == 1 and intr.shape[0] == 4:
        intr[0] *= sx  # fx
        intr[1] *= sy  # fy
        intr[2] *= sx  # cx
        intr[3] *= sy  # cy
        return intr
    # 3×3 (or larger) camera matrix format
    if intr.ndim != 2 or intr.shape[0] < 3 or intr.shape[1] < 3:
        return intr
    intr[0, 0] *= sx  # fx
    intr[0, 2] *= sx  # cx
    intr[1, 1] *= sy  # fy
    intr[1, 2] *= sy  # cy
    return intr


def adjust_intrinsics_for_crop(intrinsics, left, top):
    """Shift principal point after image/depth crop while keeping focal lengths."""
    if intrinsics is None:
        return None

    intr = torch.as_tensor(intrinsics).clone()
    if intr.ndim != 2:
        return intr

    if intr.shape[0] >= 3 and intr.shape[1] >= 3:
        intr[0, 2] -= float(left)
        intr[1, 2] -= float(top)
    return intr


def crop_depth_map(depth, left, top, right, bottom):
    """Crop depth map using image crop bounds; supports [H,W] and [C,H,W]."""
    if depth is None:
        return None

    d = torch.as_tensor(depth)
    if d.ndim == 2:
        return d[top:bottom, left:right]
    if d.ndim == 3:
        return d[:, top:bottom, left:right]
    return d
