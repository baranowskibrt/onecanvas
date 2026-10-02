"""Projection mapping functions: equirectangular panorama."""

import torch


def world_to_spherical(local_pts, horiz_eps=1e-8):
    """Convert scene-local xyz to equirectangular spherical coordinates.

    Canvas-side convention, shared by ``compute_equirectangular_mapping`` (viz)
    and ``reprojection.scene_reprojection`` (the production canvas): the last
    axis holds (X, Y, Z) and is remapped to x_c = X (right), y_c = -Z (down),
    z_c = Y (forward). Indexes the last axis only, so it works for both flat
    ``[N, 3]`` point lists and per-frame ``[H, W, 3]`` grids.

    NOTE: this is NOT the only spherical helper in the repo. The probe sampler
    stack has its own ``spatial_pretraining._common._world_to_spherical`` that
    uses a different frame (Y as vertical via ``asin``) and expects callers to
    pre-swap axes into an intermediate frame. The two are intentionally separate
    (different validity policy, no far-plane cap on the sampler side); do not
    assume changing one updates the other.

    Returns ``(longitude, latitude, radial_depth)`` with longitude in [-pi, pi]
    (0 = forward), latitude in [-pi/2, pi/2] (0 = horizon), and radial_depth the
    Euclidean distance from the panorama center.
    """
    x_c = local_pts[..., 0]    # "right"
    y_c = -local_pts[..., 2]   # camera convention (positive = down)
    z_c = local_pts[..., 1]    # "forward"

    radial_depth = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
    horiz = torch.sqrt(x_c**2 + z_c**2).clamp(min=horiz_eps)
    longitude = torch.atan2(x_c, z_c)          # [-pi, pi], 0 = forward
    latitude = torch.atan2(y_c, horiz)          # [-pi/2, pi/2], 0 = horizon
    return longitude, latitude, radial_depth


def compute_equirectangular_mapping(pts, width=256, height=64, center_point=None,
                                     return_angles=False):
    """VISUALIZATION ONLY: the model canvas is never rasterized. This builds a pixel map for debug/viz images only.

    Project 3D points onto a full-sphere equirectangular (360x180) panorama.

    Returns a (flat_indices, validity_mask, radial_depths) tuple.

    If return_angles=True, also returns (longitude, latitude) as the 4th and 5th values.

    Coordinate convention:
        raw input:  col-0 = X, col-1 = Y, col-2 = Z
        after remap: x_c = col-0,  y_c = -col-2,  z_c = col-1
        → z_c points "forward", x_c points "right", y_c points "up"

    Longitude  = atan2(x_c, z_c)         ∈ [-π, π]   (0 = forward)
    Latitude   = atan2(y_c, √(x_c²+z_c²)) ∈ [-π/2, π/2] (0 = horizon)

    Pixel mapping (top-left origin):
        u = (longitude + π) / (2π) * width      →  [0, width)
        v = (π/2 + latitude) / π  * height      →  [0, height)  (top = zenith)
    """
    device = pts.device

    if center_point is not None:
        local_pts = pts - center_point
    else:
        local_pts = pts

    # Coordinate remap + spherical projection (shared convention)
    longitude, latitude, radial_depth = world_to_spherical(local_pts)

    # Map to pixel coordinates
    u = ((longitude + torch.pi) / (2.0 * torch.pi) * width).to(torch.int32)
    v = ((torch.pi / 2.0 + latitude) / torch.pi * height).to(torch.int32)

    # Clamp u to handle the exact +pi edge case
    u = u.clamp(0, width - 1)

    # Validity mask
    valid = (radial_depth > 0.1) & (radial_depth < 50.0) & (v >= 0) & (v < height)

    # Flat index into H x W grid
    N = pts.shape[0]
    all_indices = torch.zeros(N, dtype=torch.int32, device=device)
    all_mask = valid
    all_depths = torch.where(valid, radial_depth, torch.full_like(radial_depth, float('inf')))

    valid_idx = valid.nonzero(as_tuple=False).view(-1)
    if valid_idx.numel() > 0:
        all_indices[valid_idx] = v[valid_idx] * width + u[valid_idx]

    if return_angles:
        return all_indices, all_mask, all_depths, longitude, latitude
    return all_indices, all_mask, all_depths
