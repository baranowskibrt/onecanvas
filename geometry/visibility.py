"""Exact per-camera visibility for synthetic probing markers.

Visibility of a 3D point p from a camera with pose c2w, intrinsics K, and image
dims (W, H) is defined as:
  1. p projects to a pixel (u, v) with 0 <= u < W and 0 <= v < H
  2. z_cam > 0 (in front of the camera)
  3. no synthetic occluder AABB intersects the open segment from the camera
     origin to p

Occlusion is optional. When no occluders are passed the test reduces to
frustum-only visibility.

Used by probe tasks that need clean visibility labels (appearance_order,
visibility_from_camera). Real-scene depth maps are deliberately NOT consulted:
the noise they carry would poison the labels. Occlusion is controlled at
sample time with synthetic AABBs dropped into the scene.
"""

from typing import Optional

import torch


def project_world_to_camera(
    pts_world: torch.Tensor,   # [M, 3]
    pose_c2w: torch.Tensor,    # [4, 4]
    intrinsics: torch.Tensor,  # [4]  (fx, fy, cx, cy)
):
    """Project world-space points into one camera's image plane.

    Returns:
        u, v:  [M] pixel coords (float)
        z_cam: [M] depth in camera frame (positive = in front)
    """
    w2c = torch.linalg.inv(pose_c2w.to(torch.float32))
    pts_h = torch.cat([pts_world, torch.ones_like(pts_world[..., :1])], dim=-1)  # [M, 4]
    cam = (pts_h @ w2c.T)[..., :3]  # [M, 3]
    z = cam[..., 2]
    fx, fy, cx, cy = intrinsics.tolist()
    u = fx * cam[..., 0] / z.clamp_min(1e-6) + cx
    v = fy * cam[..., 1] / z.clamp_min(1e-6) + cy
    return u, v, z


def in_frustum(
    pts_world: torch.Tensor,   # [M, 3]
    pose_c2w: torch.Tensor,    # [4, 4]
    intrinsics: torch.Tensor,  # [4]
    image_dim: torch.Tensor,   # [2]  (W, H)
) -> torch.Tensor:
    """True iff each point lies in the camera's frustum."""
    u, v, z = project_world_to_camera(pts_world, pose_c2w, intrinsics)
    W, H = image_dim.tolist()
    return (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def segment_aabb_intersect(
    p0: torch.Tensor,        # [M, 3]  segment start
    p1: torch.Tensor,        # [M, 3]  segment end
    aabb_min: torch.Tensor,  # [O, 3]
    aabb_max: torch.Tensor,  # [O, 3]
) -> torch.Tensor:
    """Vectorized slab-test for line segments against axis-aligned boxes.

    Returns [M, O] bool: whether segment m intersects box o at some t in [0, 1].
    The segment's endpoints are treated as exclusive — an AABB that merely
    touches an endpoint (t == 0 or t == 1) counts as no intersection, so placing
    the source/target inside an occluder does not spuriously block visibility.
    """
    M = p0.shape[0]
    O = aabb_min.shape[0]
    if O == 0:
        return torch.zeros(M, 0, dtype=torch.bool, device=p0.device)

    d = p1 - p0                                     # [M, 3]
    # Broadcast to [M, O, 3].
    p0b = p0.unsqueeze(1).expand(M, O, 3)
    db  = d.unsqueeze(1).expand(M, O, 3)
    lo  = aabb_min.unsqueeze(0).expand(M, O, 3)
    hi  = aabb_max.unsqueeze(0).expand(M, O, 3)

    # Slab intersection t-ranges per axis. For an axis with d == 0, the
    # segment is parallel to that slab — hit iff p0 already lies inside the
    # slab. We mark the t-range as [-inf, +inf] when inside, [+inf, -inf]
    # when outside (empty interval collapses the AND across axes).
    eps = 1e-8
    inv_d = torch.where(db.abs() > eps, 1.0 / db, torch.zeros_like(db))
    t0 = (lo - p0b) * inv_d
    t1 = (hi - p0b) * inv_d
    t_near = torch.minimum(t0, t1)
    t_far  = torch.maximum(t0, t1)

    # Handle parallel case: where d ≈ 0, check p0 ∈ [lo, hi].
    parallel = db.abs() <= eps
    inside_slab = (p0b >= lo) & (p0b <= hi)
    t_near = torch.where(parallel & inside_slab, torch.full_like(t_near, -float("inf")), t_near)
    t_far  = torch.where(parallel & inside_slab, torch.full_like(t_far,   float("inf")),  t_far)
    t_near = torch.where(parallel & ~inside_slab, torch.full_like(t_near,  float("inf")), t_near)
    t_far  = torch.where(parallel & ~inside_slab, torch.full_like(t_far, -float("inf")),  t_far)

    t_enter = t_near.max(dim=-1).values   # [M, O]
    t_exit  = t_far.min(dim=-1).values    # [M, O]

    # Strict (>, <) excludes grazing endpoints so segments starting/ending on
    # a box face aren't flagged as blocked.
    return (t_enter < t_exit) & (t_exit > 0.0) & (t_enter < 1.0)


def is_visible(
    pts_world: torch.Tensor,              # [M, 3]
    pose_c2w: torch.Tensor,               # [4, 4]
    intrinsics: torch.Tensor,             # [4]
    image_dim: torch.Tensor,              # [2] (W, H)
    occluder_min: Optional[torch.Tensor] = None,   # [O, 3]
    occluder_max: Optional[torch.Tensor] = None,   # [O, 3]
) -> torch.Tensor:
    """Visibility of M points from one camera, with optional occluder AABBs."""
    vis = in_frustum(pts_world, pose_c2w, intrinsics, image_dim)
    if occluder_min is None or occluder_max is None or occluder_min.numel() == 0:
        return vis

    cam_origin = pose_c2w[:3, 3].to(pts_world.dtype).to(pts_world.device)
    p0 = cam_origin.unsqueeze(0).expand_as(pts_world)
    hit = segment_aabb_intersect(p0, pts_world, occluder_min, occluder_max)  # [M, O]
    blocked = hit.any(dim=-1)
    return vis & ~blocked


def visibility_matrix(
    pts_world: torch.Tensor,     # [M, 3]
    poses_c2w: torch.Tensor,     # [N, 4, 4]
    intrinsics: torch.Tensor,    # [N, 4]
    image_dims: torch.Tensor,    # [N, 2]  (W, H) per frame
    occluder_min: Optional[torch.Tensor] = None,   # [O, 3]
    occluder_max: Optional[torch.Tensor] = None,   # [O, 3]
) -> torch.Tensor:
    """Per-point, per-frame visibility matrix.

    Returns [M, N] bool: ``vis[m, k]`` iff camera k observes point m.
    """
    N = poses_c2w.shape[0]
    M = pts_world.shape[0]
    vis = torch.zeros(M, N, dtype=torch.bool, device=pts_world.device)
    for k in range(N):
        vis[:, k] = is_visible(
            pts_world,
            poses_c2w[k],
            intrinsics[k],
            image_dims[k],
            occluder_min=occluder_min,
            occluder_max=occluder_max,
        )
    return vis


def first_visible_frame(
    pts_world: torch.Tensor,     # [M, 3]
    poses_c2w: torch.Tensor,     # [N, 4, 4]
    intrinsics: torch.Tensor,    # [N, 4]
    image_dims: torch.Tensor,    # [N, 2]  (W, H) per frame
    occluder_min: Optional[torch.Tensor] = None,   # [O, 3]
    occluder_max: Optional[torch.Tensor] = None,   # [O, 3]
    vis: Optional[torch.Tensor] = None,            # [M, N] precomputed
) -> torch.Tensor:
    """Earliest frame index in which each world-space point is visible.

    Returns [M] long; entries are -1 for points no camera sees. Accepts an
    optional precomputed visibility matrix to avoid duplicate work.
    """
    if vis is None:
        vis = visibility_matrix(
            pts_world, poses_c2w, intrinsics, image_dims,
            occluder_min=occluder_min, occluder_max=occluder_max,
        )
    M, N = vis.shape
    frame_range = torch.arange(N, device=vis.device, dtype=torch.long)
    sentinel = torch.full((M, N), N, dtype=torch.long, device=vis.device)
    t_or_sentinel = torch.where(vis, frame_range.unsqueeze(0).expand(M, N), sentinel)
    first = t_or_sentinel.min(dim=1).values
    first = torch.where(first == N, torch.full_like(first, -1), first)
    return first
