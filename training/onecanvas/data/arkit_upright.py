"""Gravity uprighting for ARKitScenes frames, in the ORIGINAL -> UPRIGHT
direction used by the data loader.

ARKitScenes stores every frame in the sensor's native landscape buffer no
matter how the phone was held, so gravity points sideways in a large fraction
of scenes (measured on the 150 VSI-Bench ARKit scenes: 66 upright, 58 at 90
degrees, 26 at 180). The GT poses encode the roll correctly, so the
reconstructed 3D is right and only the *appearance* fed to the vision tower is
wrong.

This module rotates the image, the depth map, the intrinsics and the pose
together, by the same k clockwise quarter turns. When all four rotate, the
lifted world points are bit-for-bit the geometry you had before; the only
thing that changes is which way is up in the pixels the vision tower sees.
`_self_test` asserts exactly that.

Conventions:
  theta  gravity direction in image space, degrees, OpenCV axes (x right,
         y down). 0 = gravity points image-down (already upright),
         90 = gravity points image-right, 180 = upside-down.
  k      number of CLOCKWISE quarter turns applied to upright the frame,
         k = theta / 90 (mod 4).

Gravity comes from `lowres_wide.traj` (the IMU-derived ARKit trajectory), NOT
from the `sky_direction` column of ARKitScenes' `metadata.csv`: that label is
flatly wrong on 20 of these 150 scenes.

Self-test:
    python training/onecanvas/data/arkit_upright.py
"""

import os

import numpy as np

QUARTERS = (0, 90, -90, 180)

# Original camera coords -> upright camera coords, for ONE clockwise quarter
# turn of the image. The camera frame turns with the image: the upright image's
# "right" is the original's "up", its "down" is the original's "right".
_M1 = np.array([[0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]])


def _rodrigues(aa):
    th = float(np.linalg.norm(aa))
    if th < 1e-8:
        return np.eye(3)
    a = np.asarray(aa, float) / th
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _quantise(theta):
    return min(QUARTERS, key=lambda t: abs(((theta - t) + 180) % 360 - 180))


def scene_theta_from_traj(scene_dir):
    """Circular-mean gravity angle over a scene's ARKit trajectory.

    Returns (theta_quantised, frac_frames_agreeing), or (None, None) when the
    scene has no `lowres_wide.traj` (i.e. it is not an ARKitScenes scene).

    The ARKit world is z-up, so world -z is gravity; .traj lines are
    world-to-camera angle-axis, so R @ (0,0,-1) is gravity in camera coords.
    """
    traj = os.path.join(str(scene_dir), "lowres_wide.traj")
    if not os.path.exists(traj):
        return None, None
    angs = []
    with open(traj) as fh:
        for line in fh:
            q = line.split()
            if len(q) != 7:
                continue
            g = _rodrigues(np.array([float(x) for x in q[1:4]])) @ np.array([0.0, 0.0, -1.0])
            if np.hypot(g[0], g[1]) > 1e-6:
                angs.append(np.arctan2(g[0], g[1]))
    if not angs:
        return None, None
    A = np.array(angs)
    mean = (np.degrees(np.arctan2(np.sin(A).mean(), np.cos(A).mean())) + 180) % 360 - 180
    q = _quantise(mean)
    frac = float(np.mean([_quantise(np.degrees(a)) == q for a in A]))
    return int(q), frac


def theta_to_k(theta):
    """Clockwise quarter turns needed to upright a frame at this gravity angle."""
    return int(round(theta / 90.0)) % 4


def scene_upright_k(scene_dir):
    """k for a scene directory, or None if it is not an ARKitScenes scene.

    The None return is what keeps ScanNet and ScanNet++ structurally untouched:
    they have no `lowres_wide.traj`, so they can never be rotated by accident.
    """
    theta, _ = scene_theta_from_traj(scene_dir)
    return None if theta is None else theta_to_k(theta)


def rotate_image_cw(img, k):
    """PIL image, k clockwise quarter turns. PIL rotates CCW, hence the sign."""
    k %= 4
    return img if k == 0 else img.rotate(-90 * k, expand=True)


def rotate_array_cw(arr, k):
    """k clockwise quarter turns on the last two dims of a numpy array or
    torch tensor. Both libraries' rot90 turn COUNTER-clockwise for positive k,
    so clockwise is -k."""
    k %= 4
    if k == 0:
        return arr
    if isinstance(arr, np.ndarray):
        return np.ascontiguousarray(np.rot90(arr, -k, axes=(-2, -1)))
    import torch
    # torch.rot90 goes through flip, which has no kernel for the small
    # unsigned integer dtypes ("flip_cpu not implemented for 'UInt16'"). Raw
    # millimetre depth arrives as uint16 on the predicted-geometry path, so
    # round-trip those through int32: every uint16 value is exactly
    # representable, and the cast back is lossless.
    if arr.dtype in (torch.uint16, torch.uint32, torch.uint64):
        _dt = arr.dtype
        return torch.rot90(arr.to(torch.int64), -k, dims=(-2, -1)).contiguous().to(_dt)
    return torch.rot90(arr, -k, dims=(-2, -1)).contiguous()


def axis_map(k):
    """3x3 mapping original camera coords -> upright camera coords."""
    M = np.eye(3)
    for _ in range(k % 4):
        M = _M1 @ M
    return M


def intrinsics_orig_to_upright(fx, fy, cx, cy, width, height, k):
    """(fx,fy,cx,cy) in ORIGINAL image pixels -> UPRIGHT image pixels.

    CONVENTION, and it matters. `reproject_scene` samples pixel centres at
    ``col + 0.5`` and scales the principal point by ``W_feat / orig_w``, i.e.
    it uses CORNER-ORIGIN continuous coordinates: the image spans [0, W] and a
    centred principal point is cx = W/2. In those coordinates one clockwise
    quarter turn maps (u, v) -> (H - v, u), so

        fx' = fy,  fy' = fx,  cx' = height - cy,  cy' = cx

    which is what this returns. Solving the lift equations directly:
    x' = (u' - cx')z/fx' must equal -y = (cy - v)z/fy for every v, which forces
    fx' = fy and cx' = height - cy.

    Note this is ``height - cy``, NOT ``(height - 1) - cy``. The one-pixel
    variant belongs to the integer-centre convention (pixel i sits at i, a
    centred principal point is (W-1)/2) used by the geometry track's
    agentic-onecanvas/scripts/arkit_upright.py. Both are self-consistent; only
    this one leaves `reproject_scene`'s world points where they were. Using the
    other here shifts every ARKit ray by one pixel, silently.

    `width`/`height` are the ORIGINAL dims and swap on every turn.
    Returns (fx, fy, cx, cy, width, height) with the upright dims.
    """
    for _ in range(k % 4):
        fx, fy = fy, fx
        cx, cy = height - cy, cx
        width, height = height, width
    return fx, fy, cx, cy, width, height


def pose_orig_to_upright(pose, k):
    """4x4 cam->world in the ORIGINAL image frame -> the UPRIGHT image frame.

    world = pose_orig @ x_orig = pose_orig @ M.T @ x_up, so
    pose_up = pose_orig @ M.T. This is the inverse of the geometry track's
    `pose_upright_to_orig` (= pose_up @ M), which is the direction that ships
    in agentic-onecanvas/scripts/arkit_upright.py -- do not mix them up.

    Accepts a numpy array or a torch tensor and returns the same type.
    """
    k %= 4
    if k == 0:
        return pose
    M4 = np.eye(4)
    M4[:3, :3] = axis_map(k).T
    if isinstance(pose, np.ndarray):
        return pose @ M4.astype(pose.dtype)
    import torch
    return pose @ torch.from_numpy(M4).to(dtype=pose.dtype, device=pose.device)


# ------------------------------------------------------------------ self-test

def _lift(depth, fx, fy, cx, cy, pose, W, H):
    """A faithful miniature of `reproject_scene`'s per-frame unprojection:
    pixel centres at col+0.5, intrinsics scaled by grid/orig, camera->world by
    the pose. Returns an [H, W, 3] world-point map."""
    v, u = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    sx, sy = W / W, H / H  # grid == image here; kept to mirror the real scaling
    x = (u - cx * sx) * depth / (fx * sx)
    y = (v - cy * sy) * depth / (fy * sy)
    pts = np.stack([x, y, depth, np.ones_like(depth)], -1)
    return (pts.reshape(-1, 4) @ pose.T)[:, :3].reshape(H, W, 3)


def _self_test():
    from PIL import Image
    import torch

    rng = np.random.default_rng(0)
    W, H = 64, 48
    fx, fy, cx, cy = 53.0, 52.8, 31.2, 23.6
    depth = rng.uniform(0.6, 5.0, (H, W))

    for k in range(4):
        # 1) The image rotation really is the index map (u,v) -> (H-1-v, u),
        #    i.e. the continuous corner-origin map (u,v) -> (H-v, u).
        base = (np.arange(W * H, dtype=np.int64).reshape(H, W) % 251).astype(np.uint8)
        img = Image.fromarray(np.stack([base] * 3, -1))
        rot = np.asarray(rotate_image_cw(img, k))[:, :, 0]
        exp_shape = (H, W) if k % 2 == 0 else (W, H)
        assert rot.shape == exp_shape, f"k={k}: image shape {rot.shape} != {exp_shape}"
        for (u, v) in [(0, 0), (W - 1, 0), (7, 11), (W - 3, H - 5)]:
            uu, vv, ww, hh = u, v, W, H
            for _ in range(k):
                uu, vv = (hh - 1) - vv, uu
                ww, hh = hh, ww
            assert rot[vv, uu] == base[v, u], f"k={k}: pixel ({u},{v}) landed wrong"

        # 2) numpy and torch array rotation agree with the PIL rotation, so the
        #    depth map turns exactly the way the image does.
        assert np.array_equal(rotate_array_cw(base, k), rot), f"k={k}: numpy rot mismatch"
        assert np.array_equal(
            rotate_array_cw(torch.from_numpy(base), k).numpy(), rot), f"k={k}: torch rot mismatch"

        # 3) THE POINT OF ALL THIS: lift the frame both ways and check the
        #    world points are the same points, not merely a similar cloud.
        #    Undo the rotation on the upright map and compare elementwise.
        pose_orig = np.eye(4)
        pose_orig[:3, :3] = _rodrigues(rng.normal(size=3))
        pose_orig[:3, 3] = rng.normal(size=3)
        pose_up = pose_orig_to_upright(pose_orig, k)
        fxu, fyu, cxu, cyu, wu, hu = intrinsics_orig_to_upright(fx, fy, cx, cy, W, H, k)
        assert (wu, hu) == exp_shape[::-1], f"k={k}: dims {(wu, hu)}"

        w_orig = _lift(depth, fx, fy, cx, cy, pose_orig, W, H)
        w_up = _lift(rotate_array_cw(depth, k), fxu, fyu, cxu, cyu, pose_up, wu, hu)
        # rotate_array_cw acts on the last two dims, so move the xyz axis out.
        w_up_back = np.moveaxis(rotate_array_cw(np.moveaxis(w_up, -1, 0), -k), 0, -1)
        err = np.abs(w_orig - w_up_back).max()
        assert err < 1e-9, f"k={k}: world points moved by {err:.3e} m"

        # 4) torch pose path matches the numpy one.
        t_up = pose_orig_to_upright(torch.from_numpy(pose_orig).float(), k).numpy()
        assert np.allclose(t_up, pose_up, atol=1e-5), f"k={k}: torch pose mismatch"

    print("arkit_upright self-test OK (k=0,1,2,3: image, depth, intrinsics, pose; "
          "lifted world points identical to <1e-9 m)")


if __name__ == "__main__":
    _self_test()
