"""Shared helpers and constants for the spatial-pretraining dataset."""

import math
import os
import random
import sys
from copy import copy

import torch
from torch.utils.data import Dataset

from ..dataset_utils import prepare_depths, stack_tensor_list
from ..curriculum_task_registry import (
    get_probe_task,
    load_plugins_from_env,
    registered_strip_tasks,
)
from geometry import compute_scene_aabb_from_depths, visibility_matrix, first_visible_frame
from utils.bbox import (
    _sample_obb_surface_points as _obb_face_points,
    format_multi_metric_bbox,
    format_multi_metric_bbox_json,
    obb_closest_surface_points,
    obb_surface_distance,
)



_INLINE_PATCH_PLACEHOLDER = "<|object_ref_start|>"


def make_asset_paste(scene, rng, label, asset, center, yaw=None):
    """Build one real-asset paste dict shared across the sampler families.

    ``t_start`` is a random valid frame-band start given the asset's frame-index
    spread. ``yaw`` is the paste orientation in radians; when None a random yaw
    in [-pi, pi] is drawn (the observability-sampler convention). The rng call
    order (randint, then the optional uniform) is load-bearing for seeded
    reproducibility, do not reorder.
    """
    spread = int(asset["frame_indices"].max().item()
                 - asset["frame_indices"].min().item())
    t_max = max(0, int(scene.n_images) - 1 - spread)
    t_start = rng.randint(0, t_max) if t_max > 0 else 0
    if yaw is None:
        yaw = rng.uniform(-math.pi, math.pi)
    return {
        "asset": asset, "target_center": center,
        "yaw_rad": float(yaw), "t_start": int(t_start),
        "label": label,
    }


def _canvas_local_to_world(pts_local, center_point, yaw_angle):
    """Invert the compute_scene_geometry canvas-local transform.

    scene_reprojection.py maps each world point via:
      local_pre = (p_world - center_point)
      (rx, ry)  = rotate [local_pre.x, local_pre.y] by yaw_angle around +Z
      rz        = local_pre.z
      x_c, y_c, z_c = (rx, -rz, ry)   # canvas-local = (x_c, y_c, z_c)

    This helper runs the exact inverse so synthetic box centers sampled in
    canvas-local coords can be tested for visibility against world-space
    camera poses.

    pts_local: [..., 3] in (x_c, y_c, z_c).
    Returns: [..., 3] in world coords.
    """
    if pts_local.dim() == 1:
        squeeze = True
        pts_local = pts_local.unsqueeze(0)
    else:
        squeeze = False
    x_c = pts_local[..., 0]
    y_c = pts_local[..., 1]
    z_c = pts_local[..., 2]
    # Undo coordinate swap: (rx, ry, rz) = (x_c, z_c, -y_c).
    rx = x_c
    ry = z_c
    rz = -y_c
    if yaw_angle is not None and yaw_angle != 0.0:
        c = math.cos(yaw_angle)
        s = math.sin(yaw_angle)
        # Forward was: rx =  c*lx - s*ly,  ry =  s*lx + c*ly.
        # Inverse:     lx =  c*rx + s*ry,  ly = -s*rx + c*ry.
        lx = c * rx + s * ry
        ly = -s * rx + c * ry
    else:
        lx, ly = rx, ry
    lz = rz
    local_pre = torch.stack([lx, ly, lz], dim=-1)
    world = local_pre + center_point.to(local_pre.dtype)
    if squeeze:
        world = world.squeeze(0)
    return world


def _sample_obb_surface_points(center, R, dims, n_total):
    """Sample 8 OBB corners + uniform surface points for use as canvas markers.

    Real lifted features come from visible object surfaces, so surface sampling
    matches the distribution the model sees in normal scene features. For tasks
    whose GT is a surface-to-surface quantity (dist_box, rel_dist_box*) this
    also makes the observable signal directly agree with the label, the closest
    marker pair is an upper bound on the true closest-point distance. The 8
    corners pin the exact extremes, eliminating the sampling-noise floor on
    closest-point distance and longest-side readouts.

    Returns exactly ``max(8, n_total)`` points: the 8 OBB corners followed by
    ``n_total - 8`` uniformly-sampled surface points (or just the corners when
    ``n_total <= 8``).

    Args:
        center: [3] torch.Tensor, OBB center in world frame.
        R:      [3, 3] torch.Tensor, OBB rotation.
        dims:   (dx, dy, dz) tuple of full-length extents.
        n_total: int, total point budget (floored to 8).
    Returns:
        [K, 3] torch.Tensor of world-frame points, K = max(8, n_total).
    """
    n_total = max(8, int(n_total))
    corners = _obb_world_corners(center, dims, R)  # [8, 3]
    n_face = n_total - 8
    if n_face == 0:
        return corners
    n_per_face = max(1, (n_face + 5) // 6)  # round up so we have >= n_face
    face_pts = _obb_face_points(center, R, dims, n_per_face)
    return torch.cat([corners, face_pts[:n_face]], dim=0)


def _world_to_spherical(world_pts):
    """Convert [N, 3] world-frame points to (lat, lon, depth) with validity mask.

    No upper depth bound — the depth Fourier encoder saturates safely up to
    100 m (model_adapters/qwen3_vl/model.py:112) and lat/lon are direction-
    only, so far points still project cleanly. Only the degenerate
    near-origin case (depth < 0.1 m) is filtered out.
    """
    depths = world_pts.norm(dim=-1).clamp(min=0.01)
    lats = torch.asin((world_pts[:, 1] / depths).clamp(-1, 1))
    lons = torch.atan2(world_pts[:, 0], world_pts[:, 2])
    valid = depths > 0.1
    return lats, lons, depths, valid


def _real_asset_paste_matrix(yaw_rad):
    """OBB-local → canvas-intermediate transform for a real-asset paste.

    Asset ``xyz_offsets`` are stored in the OBB's local frame derived from the
    Z-up scene world (see ``scripts/extract_real_object_assets.py``: the
    ``R = _euler_zxy_to_rotation_matrix(rx, ry, rz)`` is built for Z-up world
    coords, so for upright OBBs — the vast majority, with ry=rz≈0 — the local
    third axis aligns with world Z = vertical). A paste yaw must therefore
    rotate around OBB-local axis 2, not axis 1.

    The canvas consumes the intermediate frame, where ``(X_i, Y_i, Z_i) =
    (X_w, -Z_w, Y_w)`` (see ``reprojection/scene_reprojection.py:109-111``).
    The combined transform here is (1) yaw around world Z, then (2) the
    Z-up → intermediate axis swap, so callers compute paste positions as
    ``intermediate_pts = xyz_offsets @ M.T + target_center``.

    Earlier versions of this code used ``[[c, 0, -s], [0, 1, 0], [s, 0, c]]``
    (rotation around axis 1), which was meant as "yaw around vertical" but
    actually pitched the object around its OBB-local Y axis — pasting chairs
    on their side / upside down. Fix landed alongside this helper.
    """
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return torch.tensor(
        [[ c,  s, 0.0],
         [0.0, 0.0, -1.0],
         [-s,  c, 0.0]],
        dtype=torch.float32,
    )


def _asset_paste_world_pts(asset, target_center, yaw_rad):
    """Intermediate-frame xyz of an asset's feature-bearing patches at paste time.

    Mirrors the placement transform in ``_append_real_object_assets`` and
    applies the same canvas-projection validity filter. The returned points
    are exactly the backprojected per-patch positions that get rendered onto
    the canvas (modulo the later cap_per_paste subsampling), so distances
    over them are GT-aligned with what the model can read.
    """
    M = _real_asset_paste_matrix(yaw_rad)
    world_pts = asset["xyz_offsets"].float() @ M.T + target_center.float().unsqueeze(0)
    _, _, _, valid = _world_to_spherical(world_pts)
    return world_pts[valid]


def _build_camera_frustum_points(center_3d, forward_3d, right_3d, up_3d,
                                 hfov_half, vfov_half, near=0.10, far=1.20):
    """Camera-frustum wireframe points: 4 near corners + 4 far corners + 4
    midpoints along the far edges + 1 forward-tip point.

    The shape on the panoramic canvas: a tight cluster at the camera
    position (near plane) plus a broader spread fanning toward the viewing
    direction (far plane). The H/W spread of the far points encodes both
    orientation and FOV; the near cluster anchors position.

    Args:
        center_3d:  [3] camera position in canvas-local coords.
        forward_3d: [3] unit forward direction.
        right_3d:   [3] unit right direction (perpendicular to forward).
        up_3d:      [3] unit up direction (perpendicular to forward + right).
        hfov_half:  horizontal half-FOV in radians.
        vfov_half:  vertical half-FOV in radians.
        near:       near-plane depth (m).
        far:        far-plane depth (m).
    Returns:
        [13, 3] torch.Tensor of canvas-local points.
    """
    th = math.tan(hfov_half)
    tv = math.tan(vfov_half)
    pts = []
    # 4 near-plane corners.
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            p = center_3d + near * (forward_3d + sx * th * right_3d + sy * tv * up_3d)
            pts.append(p)
    # 4 far-plane corners.
    far_corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            p = center_3d + far * (forward_3d + sx * th * right_3d + sy * tv * up_3d)
            far_corners.append(p)
            pts.append(p)
    # 4 midpoints along the far-plane edges (TL-TR, TR-BR, BR-BL, BL-TL).
    # Helps the model resolve the FOV rim instead of only 4 isolated corners.
    edge_pairs = [(0, 1), (1, 3), (3, 2), (2, 0)]
    for a, b in edge_pairs:
        pts.append(0.5 * (far_corners[a] + far_corners[b]))
    # Forward-tip: a single point along the central axis at far depth.
    pts.append(center_3d + far * forward_3d)
    return torch.stack(pts, dim=0)


def _camera_frame_from_forward(forward_3d):
    """Build orthonormal (right, up) given a forward direction. Right is in
    the world horizontal plane (cross with world-up), up is forward x right
    so the camera "up" is the geometric up of the frustum (not world up)."""
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=forward_3d.dtype)
    fwd = forward_3d / forward_3d.norm().clamp(min=1e-6)
    # If forward is nearly parallel to world up, fall back to world-x as the
    # reference axis so cross() doesn't degenerate.
    if abs(float((fwd * world_up).sum().item())) > 0.95:
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=forward_3d.dtype)
    else:
        ref = world_up
    right = torch.linalg.cross(fwd, ref)
    right = right / right.norm().clamp(min=1e-6)
    up = torch.linalg.cross(right, fwd)
    up = up / up.norm().clamp(min=1e-6)
    return fwd, right, up


# EmbodiedScan class label -> plural form used in question templates for
# *_real probe tasks. Hardcoded so EM evaluation isn't tripped by "couchs" /
# "shelfs" / etc.
_PLURAL_LABELS = {
    "chair":     "chairs",
    "table":     "tables",
    "couch":     "couches",
    "bed":       "beds",
    "cabinet":   "cabinets",
    "desk":      "desks",
    "shelf":     "shelves",
    "sink":      "sinks",
    "microwave": "microwaves",
    "dresser":   "dressers",
    "stool":     "stools",
}


def _scene_iqr_aabb(scene):
    """IQR of cartesian patch positions reconstructed from scene.{lat,lon,depth}.

    Returns (q25 [3], q75 [3]) tensors in the panorama intermediate frame.
    Used by _sample_collision_free_center to pick placement domain matching
    the synthetic-OBB samplers (_sample_object_counting / _sample_multi_box).
    """
    cos_lat = torch.cos(scene.latitude)
    pts_all = torch.stack([
        scene.depth * cos_lat * torch.sin(scene.longitude),
        scene.depth * torch.sin(scene.latitude),
        scene.depth * cos_lat * torch.cos(scene.longitude),
    ], dim=-1)
    return pts_all.quantile(0.25, dim=0), pts_all.quantile(0.75, dim=0)


def _scene_p5_p95_aabb(scene):
    """Robust full-scene AABB (5th/95th percentile of patch positions).

    Wider than IQR; used as a fallback placement domain when the IQR is
    too tight to fit large real-asset bounding spheres. Percentile (not
    raw min/max) keeps a few outlier patches from inflating the box.
    """
    cos_lat = torch.cos(scene.latitude)
    pts_all = torch.stack([
        scene.depth * cos_lat * torch.sin(scene.longitude),
        scene.depth * torch.sin(scene.latitude),
        scene.depth * cos_lat * torch.cos(scene.longitude),
    ], dim=-1)
    return pts_all.quantile(0.05, dim=0), pts_all.quantile(0.95, dim=0)


def _sample_collision_free_center(
    placed_centers, placed_radii, r_sphere, rng,
    aabb_min, aabb_max, pack_factor=2.0, max_tries=256,
    fallback_aabb=None,
):
    """Sample a [3] float32 center inside the aabb that doesn't collide.

    Bounding-sphere overlap test: rejects candidates whose distance to any
    already-placed center is less than (pack_factor / 2) * (r_sphere + r_other).
    pack=2.0 = no overlap (dist >= r_a + r_b). Same convention as the synthetic-OBB
    samplers' inline checks. Returns None after max_tries rejections.

    aabb_min / aabb_max are 3-element tensors or sequences (typically the
    q25 / q75 from _scene_iqr_aabb). If fallback_aabb=(min, max) is given
    and the primary domain fails, retries inside the fallback (typically
    the full scene AABB) so large assets aren't capped by the tight IQR.
    """
    # Inflate any axis narrower than ~3 sphere-radii around its midpoint, so
    # the placement domain can actually fit r_sphere alongside neighbors.
    # A tight scene IQR (e.g. 1 m) can't accommodate large furniture (r~1 m)
    # otherwise — the helper would burn its retry budget rejecting every
    # candidate.
    def _inflate(amn_, amx_):
        min_side = 6.0 * r_sphere
        for k in range(3):
            side = amx_[k] - amn_[k]
            if side < min_side:
                mid = 0.5 * (amn_[k] + amx_[k])
                half = 0.5 * min_side
                amn_[k] = mid - half
                amx_[k] = mid + half
        return amn_, amx_

    def _draw(amn_, amx_):
        amn_, amx_ = _inflate(list(amn_), list(amx_))
        for _ in range(int(max_tries)):
            c = torch.tensor([
                rng.uniform(amn_[0], amx_[0]),
                rng.uniform(amn_[1], amx_[1]),
                rng.uniform(amn_[2], amx_[2]),
            ], dtype=torch.float32)
            ok = True
            for c2, r2 in zip(placed_centers, placed_radii):
                if (c - c2).norm().item() < (pack_factor / 2.0) * (r_sphere + r2):
                    ok = False
                    break
            if ok:
                return c
        return None

    amn = [float(aabb_min[i].item() if hasattr(aabb_min[i], "item") else aabb_min[i]) for i in range(3)]
    amx = [float(aabb_max[i].item() if hasattr(aabb_max[i], "item") else aabb_max[i]) for i in range(3)]
    c = _draw(amn, amx)
    if c is not None:
        return c
    if fallback_aabb is not None:
        fmin, fmax = fallback_aabb
        amn = [float(fmin[i].item() if hasattr(fmin[i], "item") else fmin[i]) for i in range(3)]
        amx = [float(fmax[i].item() if hasattr(fmax[i], "item") else fmax[i]) for i in range(3)]
        return _draw(amn, amx)
    return None


def _draw_colinear_centers(rng, n_boxes, scene, min_gap=0.30, max_gap=1.20):
    """Draw n_boxes OBB centers on a single ray from the panorama center.

    All centers share (lat, lon) and differ only in depth, so they project
    to the same bearing on the equirectangular canvas and their canvas
    footprints overlap. This is the shortcut-buster for
    appearance_order_box / dist_box / object_counting: the 2D canvas
    region can no longer distinguish markers, leaving depth / T /
    per-patch xyz as the only signal.

    All depths are strictly positive (d_k >= 0.5 m), so the panorama
    center (origin of the canvas-centered frame) sits at one end of the
    line and every box lies on the +u side, projecting to the same
    panorama bearing u (never the antipode).

    Ray direction is drawn from a random valid scene patch so the centers
    land in an inhabited region of the canvas. Depths are placed sequentially
    along the ray: first depth from the IQR of scene-point magnitudes,
    subsequent depths stepped by Uniform(min_gap, max_gap). Order is
    shuffled so "nearest" is not always index 0.

    Works under any canvas centering policy (scene-center, agent-pose,
    camera-pose). scene.latitude/longitude/depth are always computed
    relative to the active canvas center (see
    reprojection/scene_reprojection.py), so drawing colinear in that
    frame always yields colinearity through the actual panorama center.

    Args:
        rng: random.Random instance.
        n_boxes: number of centers to draw.
        scene: object with .latitude, .longitude, .depth tensors over valid
               scene patches (used for ray direction and IQR).
        min_gap, max_gap: per-step depth gap range (metres).
    Returns:
        list of n_boxes [3]-float32 torch.Tensors in the canvas-centered
        frame, or None if the scene has too few valid points.
    """
    n_valid = int(scene.latitude.numel())
    if n_valid < 1:
        return None
    anchor = rng.randrange(n_valid)
    lat = float(scene.latitude[anchor].item())
    lon = float(scene.longitude[anchor].item())
    cl = math.cos(lat)
    u = torch.tensor(
        [cl * math.sin(lon), math.sin(lat), cl * math.cos(lon)],
        dtype=torch.float32,
    )
    u = u / u.norm().clamp(min=1e-6)

    depth_t = scene.depth
    d_q25 = float(depth_t.quantile(0.25).item())
    d_q75 = float(depth_t.quantile(0.75).item())
    if not (d_q25 > 0.1 and d_q75 > d_q25):
        d_q25, d_q75 = 0.8, 2.5
    d0 = rng.uniform(max(0.5, d_q25), max(d_q25 + 0.2, d_q75))

    depths = [d0]
    for _ in range(n_boxes - 1):
        depths.append(depths[-1] + rng.uniform(min_gap, max_gap))
    rng.shuffle(depths)
    return [u * d for d in depths]


# Turn sector boundaries (45° / 135°) with 15° sampling margins.
#   Classification:  Left/Right = 45°–135°,  Back = >135°  (dot < cos 135° ≈ -0.707)
#   Sampling accept: Left/Right = 60°–120°  (dot in (-0.5,  0.5))
#                    Back       = >150°      (dot < cos 150° ≈ -0.866)
#   Sampling reject: <60° (near-straight) and 120°–150° (near the 135° boundary)
_TURN_MIN_DOT      =  math.cos(math.radians(60))   # ≈  0.500 — sampling: reject angle < 60°
_TURN_LR_MAX_DOT   = -math.cos(math.radians(60))   # ≈ -0.500 — sampling: reject L/R if dot below this
_TURN_BACK_DOT     =  math.cos(math.radians(150))  # ≈ -0.866 — sampling: accept Back if dot below this
_TURN_BACK_CLASS_DOT = math.cos(math.radians(135)) # ≈ -0.707 — classification boundary for Turn Back


def _horizontal_turn(p_prev, p_here, p_next):
    """Given three 3D points (y is up), project the incoming heading
    (p_here - p_prev) and the outgoing direction (p_next - p_here) onto the
    horizontal xz-plane and return one of "Turn Left" / "Turn Right" / "Turn Back".
    Sectors: Left/Right = 45°–135°, Back = >135°. Near-straight (<45°) falls
    back to "Turn Right" (degenerate; sampling should have excluded these).
    """
    hx = float(p_here[0].item() - p_prev[0].item())
    hz = float(p_here[2].item() - p_prev[2].item())
    tx = float(p_next[0].item() - p_here[0].item())
    tz = float(p_next[2].item() - p_here[2].item())
    h_n = math.sqrt(hx * hx + hz * hz)
    t_n = math.sqrt(tx * tx + tz * tz)
    if h_n < 1e-3 or t_n < 1e-3:
        return "Turn Right"
    hx /= h_n; hz /= h_n
    tx /= t_n; tz /= t_n
    dot   = hx * tx + hz * tz
    cross = hx * tz - hz * tx
    if dot < _TURN_BACK_CLASS_DOT:
        return "Turn Back"
    return "Turn Left" if cross > 0 else "Turn Right"


def _rel_dir_label(p_ref, p_fwd, p_tgt, task):
    """Compute the egocentric direction of p_tgt relative to a viewer standing
    at p_ref and facing p_fwd. xz horizontal plane; y up. Returns one of:
      - rel_dir_easy:    "left" / "right"
      - rel_dir_medium:  "left" / "right" / "back"
      - rel_dir_hard:    "front-left" / "front-right" / "back-left" / "back-right"
      - rel_dir_4way:    "front" / "back" / "left" / "right"
    Callers should have enforced angular margins in _sample_patches so the
    label is well-defined (away from cardinal axes).
    """
    hx = float(p_fwd[0].item() - p_ref[0].item())
    hz = float(p_fwd[2].item() - p_ref[2].item())
    tx = float(p_tgt[0].item() - p_ref[0].item())
    tz = float(p_tgt[2].item() - p_ref[2].item())
    h_n = math.sqrt(hx * hx + hz * hz)
    t_n = math.sqrt(tx * tx + tz * tz)
    if h_n < 1e-3 or t_n < 1e-3:
        hx, hz = 1.0, 0.0
        tx, tz = 1.0, 0.0
    else:
        hx /= h_n; hz /= h_n
        tx /= t_n; tz /= t_n
    dot   = hx * tx + hz * tz
    cross = hx * tz - hz * tx
    left = cross > 0
    if task == "rel_dir_easy":
        return "left" if left else "right"
    if task == "rel_dir_medium":
        if dot < math.cos(math.radians(135)):  # angle > 135°
            return "back"
        return "left" if left else "right"
    if task == "rel_dir_4way":
        # 4 cardinal quadrants of 90° each (±45° boundaries).
        c45 = math.cos(math.radians(45))
        if dot >  c45: return "front"
        if dot < -c45: return "back"
        return "left" if left else "right"
    # rel_dir_hard
    front = dot > 0
    if front and left:    return "front-left"
    if front and not left: return "front-right"
    if not front and left: return "back-left"
    return "back-right"


_OBB_CORNER_SIGNS = torch.tensor([
    [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
    [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
], dtype=torch.float32)


def _obb_world_corners(center, dims, R):
    """Return world-space corners of an OBB as a [8, 3] tensor."""
    dx, dy, dz = dims
    local = _OBB_CORNER_SIGNS * torch.tensor(
        [dx / 2.0, dy / 2.0, dz / 2.0], dtype=torch.float32,
    )
    return (R @ local.T).T + center


def _rel_dir_boxes_consistent(centers, dims_list, rotations, task):
    """Return True iff the ``task`` label is unambiguous across all OBB corner
    triples of the (ref, fwd, tgt) boxes. A box with non-negligible extent can
    straddle a quadrant boundary even when its center is well inside one
    sector, so the center-based label alone is not enough for box variants.

    We build the 9 sample points per box (8 corners + center), form every
    (ref, fwd, tgt) triple (729 combos), project to the xz plane, and verify
    every combo produces the same label as the center-triple. Yaw-only (y is
    dropped), matching _rel_dir_label's convention.
    """
    ref_pts = torch.cat(
        [_obb_world_corners(centers[0], dims_list[0], rotations[0]),
         centers[0].unsqueeze(0)], dim=0)  # [9, 3]
    fwd_pts = torch.cat(
        [_obb_world_corners(centers[1], dims_list[1], rotations[1]),
         centers[1].unsqueeze(0)], dim=0)
    tgt_pts = torch.cat(
        [_obb_world_corners(centers[2], dims_list[2], rotations[2]),
         centers[2].unsqueeze(0)], dim=0)

    r = ref_pts[:, None, None, :]  # [9, 1, 1, 3]
    f = fwd_pts[None, :, None, :]  # [1, 9, 1, 3]
    t = tgt_pts[None, None, :, :]  # [1, 1, 9, 3]

    hx = f[..., 0] - r[..., 0]
    hz = f[..., 2] - r[..., 2]
    tx = t[..., 0] - r[..., 0]
    tz = t[..., 2] - r[..., 2]

    h_n = torch.sqrt(hx * hx + hz * hz)
    t_n = torch.sqrt(tx * tx + tz * tz)
    # Corner combos with a degenerate zero-length heading or target vector
    # can't contribute a well-defined label; fail closed.
    if bool(((h_n < 1e-3) | (t_n < 1e-3)).any()):
        return False

    hx_n = hx / h_n
    hz_n = hz / h_n
    tx_n = tx / t_n
    tz_n = tz / t_n
    dot = hx_n * tx_n + hz_n * tz_n
    cross = hx_n * tz_n - hz_n * tx_n

    if task == "rel_dir_easy":
        return bool((cross > 0).all()) or bool((cross < 0).all())
    if task == "rel_dir_medium":
        boundary = math.cos(math.radians(135))
        back = dot < boundary
        if bool(back.all()):
            return True
        if bool(back.any()):
            return False
        return bool((cross > 0).all()) or bool((cross < 0).all())
    if task == "rel_dir_4way":
        # Every corner-triple must fall into the same one of 4 sectors.
        c45 = math.cos(math.radians(45))
        if bool((dot >  c45).all()):
            return True
        if bool((dot < -c45).all()):
            return True
        if bool(((dot.abs() <= c45) & (cross > 0)).all()):
            return True
        if bool(((dot.abs() <= c45) & (cross < 0)).all()):
            return True
        return False
    # rel_dir_hard
    front = dot > 0
    left = cross > 0
    if not (bool(front.all()) or bool((~front).all())):
        return False
    if not (bool(left.all()) or bool((~left).all())):
        return False
    return True


def _rel_dir_camera_boxes_consistent(centers, dims_list, rotations, task):
    """True iff the ``task`` label is unambiguous across all (tgt_corner,
    piv_corner) pairs of the 2 OBBs. Heading is the camera's canvas-local
    forward (+Z after reorient), not a third anchor box, so only 2 boxes and
    81 corner pairs are checked.

    ``centers`` / ``dims_list`` / ``rotations`` must be ordered [tgt, piv] to
    match the sampler's ``[i_tgt, i_piv]`` return order. ``task`` is the
    suffix-stripped name (e.g. ``rel_dir_camera_easy``).
    """
    tgt_pts = torch.cat(
        [_obb_world_corners(centers[0], dims_list[0], rotations[0]),
         centers[0].unsqueeze(0)], dim=0)  # [9, 3]
    piv_pts = torch.cat(
        [_obb_world_corners(centers[1], dims_list[1], rotations[1]),
         centers[1].unsqueeze(0)], dim=0)  # [9, 3]

    t = tgt_pts[:, None, :]   # [9, 1, 3]
    p = piv_pts[None, :, :]   # [1, 9, 3]
    tx = t[..., 0] - p[..., 0]   # [9, 9]
    tz = t[..., 2] - p[..., 2]
    t_n = torch.sqrt(tx * tx + tz * tz)
    if bool((t_n < 1e-3).any()):
        return False
    tx_n = tx / t_n
    tz_n = tz / t_n
    # Camera heading is (hx, hz) = (0, 1), so dot = tz_n, cross = -tx_n.
    dot = tz_n
    cross = -tx_n

    if task == "rel_dir_camera_easy":
        return bool((cross > 0).all()) or bool((cross < 0).all())
    if task == "rel_dir_camera_medium":
        boundary = math.cos(math.radians(135))
        back = dot < boundary
        if bool(back.all()):
            return True
        if bool(back.any()):
            return False
        return bool((cross > 0).all()) or bool((cross < 0).all())
    # rel_dir_camera_hard: 4-way front/back × left/right.
    front = dot > 0
    left = cross > 0
    if not (bool(front.all()) or bool((~front).all())):
        return False
    if not (bool(left.all()) or bool((~left).all())):
        return False
    return True


def _ego_hour(dot, cross):
    """Convert normalized (dot, cross) in the ego xz frame to an integer hour 1..12.

    Convention matches _rel_dir_label: cross>0 means LEFT, cross<0 means RIGHT.
    Clockwise angle from forward (12 o'clock): 3 = right, 6 = back, 9 = left.
    """
    theta_cw = math.atan2(-cross, dot)              # in (-pi, pi]
    hour_float = (theta_cw / (2 * math.pi)) * 12.0  # -6..6
    hour = int(round(hour_float)) % 12
    return 12 if hour == 0 else hour


def _ego_cardinal_side(dot, cross, sin_margin):
    """4-way cardinal sector (right/left/front/back) from a normalized (dot, cross).

    Boundaries are the 45° diagonals (|dot| == |cross|). Returns None when the
    point lies within ``sin_margin`` of any diagonal (ambiguous sector). Uses
    the same left/right convention as _rel_dir_label (cross>0 => left).
    """
    if abs(abs(dot) - abs(cross)) < sin_margin:
        return None
    if -cross >= abs(dot):
        return "right"
    if cross >= abs(dot):
        return "left"
    if dot >= abs(cross):
        return "front"
    return "back"


def _rel_dir_oclock_boxes_consistent(centers, dims_list, rotations):
    """True iff every (ref_corner, fwd_corner, tgt_corner) triple across the 3
    boxes yields the same o'clock hour. Stricter than _rel_dir_boxes_consistent:
    the 30° sectors are narrow enough that a box center well inside one hour
    can still have corners spilling into the adjacent hour.
    """
    ref_pts = torch.cat(
        [_obb_world_corners(centers[0], dims_list[0], rotations[0]),
         centers[0].unsqueeze(0)], dim=0)
    fwd_pts = torch.cat(
        [_obb_world_corners(centers[1], dims_list[1], rotations[1]),
         centers[1].unsqueeze(0)], dim=0)
    tgt_pts = torch.cat(
        [_obb_world_corners(centers[2], dims_list[2], rotations[2]),
         centers[2].unsqueeze(0)], dim=0)

    r = ref_pts[:, None, None, :]
    f = fwd_pts[None, :, None, :]
    t = tgt_pts[None, None, :, :]

    hx = f[..., 0] - r[..., 0]
    hz = f[..., 2] - r[..., 2]
    tx = t[..., 0] - r[..., 0]
    tz = t[..., 2] - r[..., 2]
    h_n = torch.sqrt(hx * hx + hz * hz)
    t_n = torch.sqrt(tx * tx + tz * tz)
    if bool(((h_n < 1e-3) | (t_n < 1e-3)).any()):
        return False
    hx_n = hx / h_n; hz_n = hz / h_n
    tx_n = tx / t_n; tz_n = tz / t_n
    dot = hx_n * tx_n + hz_n * tz_n
    cross = hx_n * tz_n - hz_n * tx_n
    theta_cw = torch.atan2(-cross, dot)              # [9, 9, 9]
    hour_float = (theta_cw / (2 * math.pi)) * 12.0
    hour = torch.round(hour_float).to(torch.int64) % 12
    hour = torch.where(hour == 0, torch.full_like(hour, 12), hour)
    return bool((hour == hour.flatten()[0]).all())


def _count_side_box_consistent(ref_c, ref_d, ref_R,
                               fwd_c, fwd_d, fwd_R,
                               tgt_c, tgt_d, tgt_R,
                               sin_margin):
    """Return the unambiguous cardinal side ("right"/"left"/"front"/"back") of
    the target box relative to a viewer standing at ref_c and facing fwd_c, or
    None if any (ref_corner, fwd_corner, tgt_corner) triple lies within
    ``sin_margin`` of a 45° diagonal or disagrees on the sector.
    """
    ref_pts = torch.cat(
        [_obb_world_corners(ref_c, ref_d, ref_R), ref_c.unsqueeze(0)], dim=0)
    fwd_pts = torch.cat(
        [_obb_world_corners(fwd_c, fwd_d, fwd_R), fwd_c.unsqueeze(0)], dim=0)
    tgt_pts = torch.cat(
        [_obb_world_corners(tgt_c, tgt_d, tgt_R), tgt_c.unsqueeze(0)], dim=0)

    r = ref_pts[:, None, None, :]
    f = fwd_pts[None, :, None, :]
    t = tgt_pts[None, None, :, :]

    hx = f[..., 0] - r[..., 0]
    hz = f[..., 2] - r[..., 2]
    tx = t[..., 0] - r[..., 0]
    tz = t[..., 2] - r[..., 2]
    h_n = torch.sqrt(hx * hx + hz * hz)
    t_n = torch.sqrt(tx * tx + tz * tz)
    if bool(((h_n < 1e-3) | (t_n < 1e-3)).any()):
        return None
    hx_n = hx / h_n; hz_n = hz / h_n
    tx_n = tx / t_n; tz_n = tz / t_n
    dot = hx_n * tx_n + hz_n * tz_n
    cross = hx_n * tz_n - hz_n * tx_n

    # 45° diagonal margin: |dot|-|cross| too close to 0 is ambiguous.
    if bool((torch.abs(torch.abs(dot) - torch.abs(cross)) < sin_margin).any()):
        return None

    right = (-cross >= torch.abs(dot))
    left  = (cross  >= torch.abs(dot))
    front = (dot    >= torch.abs(cross)) & ~right & ~left
    back  = ~(right | left | front)
    if bool(right.all()): return "right"
    if bool(left.all()):  return "left"
    if bool(front.all()): return "front"
    if bool(back.all()):  return "back"
    return None


def _mcq_shuffle(correct_value, distractors, rng):
    """Build an MCQ: shuffle [correct_value, *distractors] into A/B/C/... options
    and return (letter, options_list) where letter is the letter of correct_value
    in the shuffled order and options_list is ["A. <text>", "B. <text>", ...].
    correct_value and each distractor may be any string (object label, permutation,
    turn pair) — identity comparison is by position in the list before shuffling.
    """
    items = [correct_value] + list(distractors)
    order = list(range(len(items)))
    rng.shuffle(order)
    letters = ["A", "B", "C", "D", "E", "F"]
    options = [f"{letters[pos]}. {items[src]}" for pos, src in enumerate(order)]
    correct_letter = letters[order.index(0)]
    return correct_letter, options


_ROTATION_MODES = ("aligned", "yaw", "random")


def _rotation_matrix(rotation_mode, rng):
    """Return a 3x3 rotation matrix for the specified mode.

    - "aligned": identity (no rotation).
    - "yaw": rotation about the vertical (Y) axis only by a random yaw.
      Matches the VSI-Bench gravity-aligned object convention (rx=ry=0).
    - "random": fully random 3D rotation via Rodrigues (random axis + angle).

    Used by all box-family probing samplers so they share a single rotation
    policy controllable via a task-name suffix (_aligned / _yaw / _random).
    """
    if rotation_mode == "aligned":
        return torch.eye(3)
    if rotation_mode == "yaw":
        yaw = rng.uniform(0, 2 * math.pi)
        c_y, s_y = math.cos(yaw), math.sin(yaw)
        return torch.tensor([
            [ c_y, 0.0,  s_y],
            [ 0.0, 1.0,  0.0],
            [-s_y, 0.0,  c_y],
        ], dtype=torch.float32)
    if rotation_mode == "random":
        ax = torch.randn(3)
        ax = ax / ax.norm()
        angle = rng.uniform(0, 2 * math.pi)
        K = torch.tensor([
            [0, -ax[2], ax[1]],
            [ax[2], 0, -ax[0]],
            [-ax[1], ax[0], 0],
        ])
        return torch.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    raise ValueError(f"Unknown rotation_mode: {rotation_mode!r}")


def _xz_footprint_area(world_corners):
    """XZ-plane area of the 8 OBB corners' convex hull (for random rotation).

    Falls back to None on degenerate hulls; caller should then retry the OBB.
    """
    from scipy.spatial import ConvexHull
    xz = world_corners[:, [0, 2]].numpy()
    try:
        return float(ConvexHull(xz).volume)  # 'volume' = 2D area in scipy
    except Exception:
        return None


# Default rotation mode per box-task base name (used when the task name has no
# rotation suffix, so existing curricula keep their historical behavior).
_DEFAULT_ROTATION_BY_TASK = {
    "box_floor_area":           "aligned",   # room-size probe (VSI room_size)
    "box_floor_area_nonrect":   "yaw",       # L/T-shaped room (two contiguous rects)
    "box_floor_area_irregular": "yaw",       # N in {2,3,4} rects, may be multi-component
    "multi_box_grounding":      "yaw",       # predict AABB around painted OBBs
}


# Box-task bases eligible for a rotation suffix. Non-box tasks never use it.
_BOX_TASK_BASES = frozenset(_DEFAULT_ROTATION_BY_TASK.keys())


def _parse_box_task(task):
    """Return (base_task, rotation_mode) for a box-family task name.

    Accepts both plain names ("box_floor_area") and suffixed names
    ("box_floor_area_yaw"). For non-box tasks returns (task, None).
    """
    for mode in _ROTATION_MODES:
        suffix = f"_{mode}"
        if task.endswith(suffix):
            base = task[:-len(suffix)]
            if base in _BOX_TASK_BASES:
                return base, mode
    if task in _BOX_TASK_BASES:
        return task, _DEFAULT_ROTATION_BY_TASK[task]
    return task, None


# Probe tasks whose answer *depends on* the MRoPE T-axis (frame index / time).
# For these, the marker's T value MUST stay at its real source-frame T — we
# can't shuffle it or the task becomes unlearnable.
#
# Every other probe task is a "spatial" task: the answer is a pure function of
# the markers' 3D positions, not their temporal origin. For those, we shuffle
# the marker T-axis so the model can't shortcut the spatial reasoning by
# reading the source frame index (which correlates with physical position via
# the video trajectory — early frames see one region of the room, late frames
# see another).
_PROBE_TEMPORAL_TASKS = frozenset({
    "appearance_order", "appearance_order_box",
})


def _is_temporal_probe_task(task: str) -> bool:
    base, _ = _parse_box_task(task)
    return base in _PROBE_TEMPORAL_TASKS or task in _PROBE_TEMPORAL_TASKS


_OBB_ONLY_SUFFIX = "_obb_only"


def _strip_display_suffix(task: str) -> str:
    """Remove the ``_obb_only`` alias suffix if present.

    Used when the same underlying probe task appears in the task mix both
    with and without canvas stripping (e.g. ``rel_dist_box`` +
    ``rel_dist_box_obb_only``). The suffix only affects the ``canvas_obb_only``
    flag (matched against ``curriculum_canvas_obb_only_tasks`` membership) and the
    task label in metrics; sampling and Q/A dispatch use the base task name.
    """
    if task.endswith(_OBB_ONLY_SUFFIX):
        return task[:-len(_OBB_ONLY_SUFFIX)]
    return task


def _sample_marker_t_override(scene, patch_indices, frame_indices, rng):
    """Pick a canvas-patch index per marker whose T-value will be used in
    place of the marker's own source-frame T.

    For each marker k with source frame f_k we pick a random canvas patch
    from a DIFFERENT frame (any frame ≠ f_k). The adapter then uses that
    patch's T position for the marker token, so the marker's T axis no
    longer leaks f_k — which, in a walking-through-the-room video, is
    correlated with the marker's physical position and would otherwise let
    the model shortcut spatial reasoning.

    Falls back to the marker's own patch index when the scene has only one
    frame (nothing to shuffle to).

    Returns a Python list of ints of length ``len(patch_indices)``.
    """
    n_valid = int(scene.n_valid)
    if n_valid == 0:
        return list(patch_indices)
    frame_idx_t = scene.frame_index
    unique_frames = torch.unique(frame_idx_t)
    if unique_frames.numel() <= 1:
        return list(patch_indices)
    t_indices = []
    for src_patch, f in zip(patch_indices, frame_indices):
        mask = (frame_idx_t != f).nonzero(as_tuple=True)[0]
        if mask.numel() == 0:
            t_indices.append(int(src_patch))
        else:
            t_indices.append(int(mask[rng.randrange(mask.numel())].item()))
    return t_indices

__all__ = [
    'make_asset_paste',
    '_canvas_local_to_world',
    '_sample_obb_surface_points',
    '_world_to_spherical',
    '_real_asset_paste_matrix',
    '_asset_paste_world_pts',
    '_build_camera_frustum_points',
    '_camera_frame_from_forward',
    '_scene_iqr_aabb',
    '_scene_p5_p95_aabb',
    '_sample_collision_free_center',
    '_draw_colinear_centers',
    '_horizontal_turn',
    '_rel_dir_label',
    '_obb_world_corners',
    '_rel_dir_boxes_consistent',
    '_rel_dir_camera_boxes_consistent',
    '_ego_hour',
    '_ego_cardinal_side',
    '_rel_dir_oclock_boxes_consistent',
    '_count_side_box_consistent',
    '_mcq_shuffle',
    '_rotation_matrix',
    '_xz_footprint_area',
    '_parse_box_task',
    '_is_temporal_probe_task',
    '_strip_display_suffix',
    '_sample_marker_t_override',
    '_INLINE_PATCH_PLACEHOLDER',
    '_PLURAL_LABELS',
    '_TURN_MIN_DOT',
    '_TURN_LR_MAX_DOT',
    '_TURN_BACK_DOT',
    '_TURN_BACK_CLASS_DOT',
    '_OBB_CORNER_SIGNS',
    '_ROTATION_MODES',
    '_DEFAULT_ROTATION_BY_TASK',
    '_BOX_TASK_BASES',
    '_PROBE_TEMPORAL_TASKS',
    '_OBB_ONLY_SUFFIX',
]
