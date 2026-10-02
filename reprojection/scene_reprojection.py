"""Model-agnostic scene reprojection: features + geometry -> ReprojectedScene."""

import math
import warnings

import torch
import torch.nn.functional as F

from geometry.projection_maps import world_to_spherical

from .types import ReprojectedScene, SceneGeometry

# --- Geometry constants (shared across the reprojection path) ---------------
# Raw depth maps are stored in millimetres; divide to get metres.
_DEPTH_MM_TO_M = 1000.0
# A patch is kept only when its metric depth falls in this range. Below the near
# plane is sensor noise / zero-fill; beyond the far plane is unreliable depth.
_VALID_DEPTH_MIN_M = 0.1
_VALID_DEPTH_MAX_M = 50.0
# Floor applied to the horizontal radius before atan2 to avoid a 0/0 at the poles.
_HORIZ_EPS = 1e-8
# Floor applied to returned radial depth so downstream log/scale ops stay finite.
_MIN_RADIAL_DEPTH_M = 0.01


def _in_depth_band(depth_mm):
    """The same validity test the patch loop applies, on a millimetre map."""
    z = depth_mm / _DEPTH_MM_TO_M
    return (z > _VALID_DEPTH_MIN_M) & (z < _VALID_DEPTH_MAX_M) & torch.isfinite(z)


def _coarsen_patch_depth(depth_mm, K):
    """Give every ``K x K`` block of patches one shared depth sample.

    Operates on the per-patch depth grid (already nearest-sampled to
    ``H_feat x W_feat``), and holds the VALIDITY MASK exactly fixed:

      * the representative of a block is the median over its VALID entries, so
        it is an actual observed depth (no synthesised value lying on no
        surface) and depth holes are not spread across the block;
      * a patch that had no usable depth keeps its own value, so it stays
        exactly as invalid as it was, and it is never promoted by a neighbour.

    Holding the mask fixed is the point of the knob. A plain subsample loses
    patches wherever a hole wins a block (measured: 6% of patches at K=2 rising
    to 33% at K=8), which puts canvas token density back into the experiment --
    the one confound this perturbation exists to remove.
    """
    H, W = depth_mm.shape
    ph, pw = (-H) % K, (-W) % K
    d = depth_mm
    if ph or pw:
        d = F.pad(d[None, None], (0, pw, 0, ph), mode='replicate')[0, 0]
    Hp, Wp = d.shape
    blocks = d.view(Hp // K, K, Wp // K, K).permute(0, 2, 1, 3).reshape(
        Hp // K, Wp // K, K * K)
    masked = torch.where(_in_depth_band(blocks), blocks,
                         torch.full_like(blocks, float('nan')))
    rep = masked.nanmedian(dim=-1).values
    coarse = rep.repeat_interleave(K, 0).repeat_interleave(K, 1)[:H, :W]
    keep_own = ~_in_depth_band(depth_mm) | ~torch.isfinite(coarse)
    return torch.where(keep_own, depth_mm, coarse)


def compute_scene_geometry(
    depths,         # [N_imgs, H_depth, W_depth] or list — MILLIMETRES (see below)
    poses,          # [N_imgs, 4, 4]  camera-to-world extrinsics
    intrinsics,     # [N_imgs, 4]  (fx, fy, cx, cy)
    image_dims,     # [N_imgs, 2]  (W, H)
    H_feat,         # int — per-frame feature grid height (post merge_size)
    W_feat,         # int — per-frame feature grid width  (post merge_size)
    device="cpu",
    center_override=None,  # [3] tensor — use this instead of mean of cameras
    yaw_angle=None,        # float in [-pi, pi] — rotate panoramic forward direction
    depth_downsample=1,    # int >= 1 — coarsen the depth grid to (H_feat/K, W_feat/K)
):
    """Geometry-only scene reprojection.

    Computes per-patch spherical coordinates (longitude, latitude, depth) and
    the validity mask, without touching any visual features. Returned
    ``valid_indices`` index the flat ``N_imgs * H_feat * W_feat`` patch grid;
    ``reproject_scene`` uses them to gather features in a second step.

    Used directly by datasets that pick patches at sample time but defer
    feature extraction to the live visual encoder in ``model.forward()``
    (geometric probing, future scene-conditioned synthetic Q-A datasets).

    Depths must be in MILLIMETRES (sensor-PNG convention): they are divided by
    1000 below and only patches with metric depth in
    ``(_VALID_DEPTH_MIN_M, _VALID_DEPTH_MAX_M)`` survive. Passing metres-valued
    depths collapses everything to sub-millimetre and returns ``n_valid == 0``.

    ``depth_downsample`` (default 1 = off) coarsens the depth grid: every
    ``K x K`` block of patches is given one shared depth sample, so patches
    land in 3D as if the depth map had ``K`` times fewer samples per axis.
    See ``_coarsen_patch_depth`` — it holds the validity mask exactly fixed, so
    the patch COUNT and the features are untouched and only 3D placement
    degrades. That is what separates it from input image resolution, which
    changes how many patches exist in the first place.
    """
    depth_downsample = int(depth_downsample or 1)
    if depth_downsample < 1:
        raise ValueError(f"depth_downsample must be >= 1, got {depth_downsample}")

    N_imgs = len(depths) if not torch.is_tensor(depths) else depths.shape[0]

    # Loud early check on depth units (cheap: first finite frame only). A
    # metres-valued input would silently drop every patch, which is an hour of
    # confused debugging for a first-time user.
    for _d in depths:
        if torch.is_tensor(_d):
            # Sensor PNGs arrive as uint16. Boolean indexing is not implemented
            # for UInt16 tensors on CPU, so inspect the same values as float.
            _d_float = _d.float()
            _finite = _d_float[torch.isfinite(_d_float)]
            if _finite.numel():
                _dmax = float(_finite.max())
                if _dmax < _VALID_DEPTH_MAX_M:
                    warnings.warn(
                        f"compute_scene_geometry: input depths look like metres "
                        f"(max {_dmax:.3g} < {_VALID_DEPTH_MAX_M}); this API expects "
                        f"MILLIMETRES (sensor-PNG convention) and divides by 1000, so "
                        f"metric input drops every patch (n_valid == 0). Multiply "
                        f"depths by 1000.",
                        RuntimeWarning, stacklevel=2,
                    )
                break

    # --- Compute scene center (mean of camera positions, or override) ---
    valid_poses = [p for p in poses if p is not None and torch.all(torch.isfinite(p))]
    if center_override is not None:
        center_point = center_override.float()
    elif valid_poses:
        translations = torch.stack([p[:3, 3] for p in valid_poses])
        center_point = torch.mean(translations, dim=0)
    else:
        center_point = torch.zeros(3)

    # --- Precompute yaw rotation (world XY plane, around Z axis) ---
    _yaw_cos = _yaw_sin = None
    if yaw_angle is not None and yaw_angle != 0.0:
        _yaw_cos = math.cos(yaw_angle)
        _yaw_sin = math.sin(yaw_angle)

    # --- Per-image unprojection to world coordinates ---
    all_longitude = []
    all_latitude = []
    all_depth = []
    all_valid = []
    all_frame_idx = []

    for i in range(N_imgs):
        depth_map = depths[i].to(device).float()
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        elif depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(0)
        # Each patch's depth is read at its centre pixel ('nearest-exact'), the
        # pixel its ray below passes through. 'nearest' reads the patch's
        # top-left pixel, which puts tokens a median 5.8 cm off their surface
        # on the ScanNet 7x10 training grid, against 0.1 cm at the centre.
        depth_resized = F.interpolate(depth_map, size=(H_feat, W_feat), mode='nearest-exact').squeeze()
        if depth_resized.dim() == 3:
            depth_resized = depth_resized[0]
        if depth_downsample > 1:
            depth_resized = _coarsen_patch_depth(depth_resized, depth_downsample)
        zi = depth_resized / _DEPTH_MM_TO_M

        pose = poses[i].to(device)
        # Use .tolist() for float64 arithmetic, matching original code exactly
        fx, fy, cx, cy = intrinsics[i].tolist()
        orig_w, orig_h = image_dims[i].tolist()

        scale_x = W_feat / orig_w
        scale_y = H_feat / orig_h
        s_fx, s_fy = fx * scale_x, fy * scale_y
        s_cx, s_cy = cx * scale_x, cy * scale_y

        v, u = torch.meshgrid(
            torch.arange(H_feat, device=device, dtype=torch.float32) + 0.5,
            torch.arange(W_feat, device=device, dtype=torch.float32) + 0.5,
            indexing='ij',
        )

        xi = (u - s_cx) * zi / s_fx
        yi = (v - s_cy) * zi / s_fy

        pts_cam = torch.stack([xi, yi, zi, torch.ones_like(zi)], dim=-1)
        pts_world = (pts_cam.view(-1, 4) @ pose.T)[:, :3].view(H_feat, W_feat, 3)

        local_pts = pts_world - center_point.to(device)

        # --- Optional yaw rotation (panoramic augmentation) ---
        if _yaw_cos is not None:
            rotated_x = _yaw_cos * local_pts[..., 0] - _yaw_sin * local_pts[..., 1]
            rotated_y = _yaw_sin * local_pts[..., 0] + _yaw_cos * local_pts[..., 1]
            local_pts = torch.stack([rotated_x, rotated_y, local_pts[..., 2]], dim=-1)

        longitude, latitude, radial_depth = world_to_spherical(local_pts, horiz_eps=_HORIZ_EPS)

        valid = (zi > _VALID_DEPTH_MIN_M) & (zi < _VALID_DEPTH_MAX_M) & torch.isfinite(radial_depth)

        all_longitude.append(longitude.reshape(-1))
        all_latitude.append(latitude.reshape(-1))
        all_depth.append(radial_depth.reshape(-1))
        all_valid.append(valid.reshape(-1))
        all_frame_idx.append(torch.full((H_feat * W_feat,), i, dtype=torch.float32, device=device))

    # --- Flatten across images ---
    longitude_flat = torch.cat(all_longitude)    # [N_total]
    latitude_flat = torch.cat(all_latitude)      # [N_total]
    depth_flat = torch.cat(all_depth)            # [N_total]
    valid_flat = torch.cat(all_valid)            # [N_total]
    frame_idx_flat = torch.cat(all_frame_idx)    # [N_total]

    valid_indices = valid_flat.nonzero(as_tuple=True)[0]
    N_valid = valid_indices.numel()

    # Stash intrinsics and image_dims so samplers can run visibility-exact
    # probes (first_visible_frame) without re-threading the assets dict.
    intr_t = intrinsics if torch.is_tensor(intrinsics) else None
    dims_t = image_dims if torch.is_tensor(image_dims) else None
    if intr_t is None and isinstance(intrinsics, (list, tuple)) and len(intrinsics) > 0:
        try:
            intr_t = torch.stack([
                x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
                for x in intrinsics
            ])
        except Exception:
            intr_t = None
    if dims_t is None and isinstance(image_dims, (list, tuple)) and len(image_dims) > 0:
        try:
            dims_t = torch.stack([
                x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
                for x in image_dims
            ])
        except Exception:
            dims_t = None

    return SceneGeometry(
        longitude=longitude_flat[valid_indices],
        latitude=latitude_flat[valid_indices],
        depth=depth_flat[valid_indices].clamp(min=_MIN_RADIAL_DEPTH_M),
        frame_index=frame_idx_flat[valid_indices],
        n_valid=N_valid,
        n_images=N_imgs,
        center_point=center_point,
        valid_indices=valid_indices,
        poses=poses,
        intrinsics=intr_t,
        image_dims=dims_t,
        yaw_angle=yaw_angle,
    )


def reproject_scene(
    features,       # [N_imgs, N_layers, H_feat, W_feat, C]
    depths,         # [N_imgs, H_depth, W_depth] or [N_imgs, 1, H, W] — MILLIMETRES
    poses,          # [N_imgs, 4, 4]  camera-to-world extrinsics
    intrinsics,     # [N_imgs, 4]  (fx, fy, cx, cy)
    image_dims,     # [N_imgs, 2]  (W, H)
    device="cpu",
    center_override=None,  # [3] tensor — use this instead of mean of cameras
    yaw_angle=None,        # float in [-pi, pi] — rotate panoramic forward direction
    depth_downsample=1,    # int >= 1 — coarsen the depth grid (see compute_scene_geometry)
):
    """Reproject multi-view features to 3D and return a ReprojectedScene.

    Model-agnostic entry point.  Delegates the geometric layout (lat/lon/depth/
    validity) to ``compute_scene_geometry`` and then gathers features at the
    valid patch indices.  No position scaling or VLM-specific logic happens
    here.

    ``depths`` are in MILLIMETRES and ``poses`` are camera-to-world extrinsics
    (see ``compute_scene_geometry``).
    """
    N_imgs, N_layers, H_feat, W_feat, C = features.shape

    geom = compute_scene_geometry(
        depths=depths,
        poses=poses,
        intrinsics=intrinsics,
        image_dims=image_dims,
        H_feat=H_feat,
        W_feat=W_feat,
        device=device,
        center_override=center_override,
        yaw_angle=yaw_angle,
        depth_downsample=depth_downsample,
    )

    # --- Extract valid patch features at the indices computed above ---
    N_total = N_imgs * H_feat * W_feat
    feats = features.permute(0, 2, 3, 1, 4).reshape(N_total, N_layers, C)
    valid_feats = feats[geom.valid_indices]

    embeds = valid_feats[:, 0, :]
    aux_layers = [valid_feats[:, l, :] for l in range(1, N_layers)]

    return ReprojectedScene(
        embeds=embeds,
        aux_layers=aux_layers,
        longitude=geom.longitude,
        latitude=geom.latitude,
        depth=geom.depth,
        frame_index=geom.frame_index,
        n_valid=geom.n_valid,
        n_images=geom.n_images,
        center_point=geom.center_point,
        poses=geom.poses,
        intrinsics=geom.intrinsics,
        image_dims=geom.image_dims,
        yaw_angle=geom.yaw_angle,
        valid_indices=geom.valid_indices,
    )
