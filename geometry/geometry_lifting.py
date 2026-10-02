"""3D geometry lifting: unproject 2D pixels + depth to 3D world coordinates."""

import torch
import torch.nn.functional as F


def lift_to_3d_with_intrinsics(image, depth, pose, intrinsics, stride=1):
    fx, fy, cx, cy = intrinsics[0]
    device = depth.device

    # Prep depth for interpolation (B, C, H, W)
    depth_tensor = depth.unsqueeze(0).unsqueeze(0).float()
    orig_h, orig_w = image.shape[:2]

    target_h, target_w = orig_h // stride, orig_w // stride

    depth_resized = F.interpolate(depth_tensor, size=(target_h, target_w), mode='nearest')
    depth_torch = depth_resized.squeeze() / 1000.0  # MILLIMETRES -> metres (API-wide contract)

    s_fx, s_fy, s_cx, s_cy = fx / stride, fy / stride, cx / stride, cy / stride
    curr_h, curr_w = target_h, target_w

    img_sliced = image[::stride, ::stride]
    if img_sliced.ndim == 3 and img_sliced.shape[-1] == 3:
        source_values = img_sliced.reshape(-1, 3).float() / 255.0
    else:
        # Feature maps or single-channel inputs: normalise and broadcast to 3 columns
        flat = img_sliced.reshape(-1).float()
        gray_vals = flat[:, None] / (flat.abs().max() + 1e-6)
        source_values = gray_vals.repeat(1, 3)

    # Use torch.meshgrid with indexing='xy' to match np.meshgrid
    v, u = torch.meshgrid(torch.arange(curr_h, device=device),
                          torch.arange(curr_w, device=device),
                          indexing='ij')

    z = depth_torch.flatten()
    u, v = u.flatten(), v.flatten()

    x = (u - s_cx) * z / s_fx
    y = (v - s_cy) * z / s_fy

    pts_cam = torch.stack((x, y, z, torch.ones_like(z)), dim=-1).to(pose.dtype).to(pose.device)

    pts_world = (pts_cam @ pose.T)[:, :3]

    mask = (z > 0.1) & (z < 50.0)

    pts_world[~mask] = float('nan')
    return pts_world, source_values


def get_scene_center(poses):
    valid_poses = [p for p in poses if p is not None and torch.all(torch.isfinite(p))]
    if not valid_poses:
        return torch.zeros(3)
    translations = torch.stack([p[:3, 3] for p in valid_poses])
    return torch.mean(translations, dim=0)


def compute_scene_aabb_from_depths(depths, poses, intrinsics, image_dims, grid=16):
    """Scene AABB by unprojecting a sparse depth grid per frame to world.

    Returns (lo, hi) as [3] tensors, or (None, None) if no valid points.
    Subsampling keeps this cheap (grid=16 -> 256 pts/frame).
    """
    lo_list, hi_list = [], []
    N = len(depths) if not torch.is_tensor(depths) else depths.shape[0]
    for i in range(N):
        depth_map = depths[i] if i < len(depths) else None
        pose = poses[i] if i < len(poses) else None
        intr = intrinsics[i] if i < len(intrinsics) else None
        dims = image_dims[i] if i < len(image_dims) else None
        if depth_map is None or pose is None or intr is None or dims is None:
            continue
        if not torch.is_tensor(pose) or not torch.all(torch.isfinite(pose)):
            continue
        if depth_map.dim() == 2:
            dm = depth_map.unsqueeze(0).unsqueeze(0)
        elif depth_map.dim() == 3:
            dm = depth_map.unsqueeze(0)
        else:
            dm = depth_map
        dm = F.interpolate(dm.float(), size=(grid, grid), mode='nearest').squeeze()
        if dm.dim() == 3:
            dm = dm[0]
        zi = dm / 1000.0  # MILLIMETRES -> metres (API-wide contract)

        fx, fy, cx, cy = intr.tolist() if torch.is_tensor(intr) else list(intr)
        orig_w, orig_h = dims.tolist() if torch.is_tensor(dims) else list(dims)
        sx, sy = grid / orig_w, grid / orig_h
        s_fx, s_fy = fx * sx, fy * sy
        s_cx, s_cy = cx * sx, cy * sy

        v, u = torch.meshgrid(
            torch.arange(grid, dtype=torch.float32) + 0.5,
            torch.arange(grid, dtype=torch.float32) + 0.5,
            indexing='ij',
        )
        xi = (u - s_cx) * zi / s_fx
        yi = (v - s_cy) * zi / s_fy
        pts_cam = torch.stack([xi, yi, zi, torch.ones_like(zi)], dim=-1).view(-1, 4).to(pose.dtype)
        pts_world = (pts_cam @ pose.T)[:, :3]

        mask = (zi.view(-1) > 0.1) & (zi.view(-1) < 50.0) & torch.isfinite(pts_world).all(dim=1)
        if mask.any():
            pw = pts_world[mask]
            lo_list.append(pw.min(dim=0).values)
            hi_list.append(pw.max(dim=0).values)

    if not lo_list:
        return None, None
    lo = torch.stack(lo_list).min(dim=0).values
    hi = torch.stack(hi_list).max(dim=0).values
    return lo.float(), hi.float()
