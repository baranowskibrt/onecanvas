#!/usr/bin/env python3
"""Guard the --depth-downsample contract.

The knob exists to perturb 3D placement and NOTHING else, so that a counting
result under it cannot be explained by canvas token density. Three properties
carry that claim, and all three are easy to break with a one-line change to
the coarsening:

  1. K=1 is a byte-exact no-op.
  2. The validity mask -- hence the patch count and which features survive --
     is identical at every K, including across depth holes.
  3. K>1 actually moves patches, and moves them more as K grows.

Run: python tests/regression/check_depth_downsample.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from reprojection import compute_scene_geometry  # noqa: E402

N, H_FEAT, W_FEAT = 3, 15, 20


def scene():
    """Depth in MILLIMETRES: a back wall, a nearer slab, and a sensor hole."""
    d = torch.full((N, 240, 320), 3200.0)
    d[:, 70:170, 110:210] = 1100.0     # foreground object
    d[:, 190:240, 0:90] = 0.0          # zero-fill hole, as sensor PNGs have
    d[1, 30:60, 250:320] = 0.0
    poses = torch.eye(4).repeat(N, 1, 1)
    poses[1, 0, 3] = 0.6
    poses[2, 1, 3] = -0.4
    intr = torch.tensor([[250.0, 250.0, 160.0, 120.0]] * N)
    dims = torch.tensor([[320.0, 240.0]] * N)
    return d, poses, intr, dims


def geom(k):
    d, poses, intr, dims = scene()
    kw = {} if k is None else {"depth_downsample": k}
    return compute_scene_geometry(d, poses, intr, dims, H_FEAT, W_FEAT, **kw)


def main():
    ref = geom(None)          # no argument at all: the pre-knob code path
    one = geom(1)
    fails = []

    if not torch.equal(ref.valid_indices, one.valid_indices) or \
            not torch.allclose(ref.depth, one.depth):
        fails.append("K=1 is not a no-op vs omitting the argument")

    prev_move = -1.0
    for k in (2, 4, 8):
        g = geom(k)
        if not torch.equal(g.valid_indices, one.valid_indices):
            fails.append(
                f"K={k} changed the validity mask "
                f"({g.n_valid} valid vs {one.n_valid} at K=1) -- patch count "
                f"must not depend on K, or token density confounds the result")
        if not torch.isfinite(g.depth).all():
            fails.append(f"K={k} produced non-finite depths")
        n = min(g.depth.numel(), one.depth.numel())
        move = float((g.depth[:n] - one.depth[:n]).abs().mean())
        if move <= prev_move:
            fails.append(
                f"K={k} did not perturb more than the previous K "
                f"(mean |dz| {move:.4f} <= {prev_move:.4f}) -- a knob that "
                f"does nothing would read as 'counting is insensitive'")
        print(f"  K={k}: n_valid={g.n_valid} (K=1: {one.n_valid})  "
              f"mean |dz| vs K=1 = {move:.4f} m")
        prev_move = move

    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        return 1
    print("All depth_downsample checks passed.")
    return 0


if __name__ == "__main__":
    print("Checking depth_downsample contract...")
    sys.exit(main())
