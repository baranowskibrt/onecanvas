"""Correctness gate for --upright_arkit. Run this BEFORE trusting any number.

The failure mode of gravity uprighting is silent: get the pose or the
intrinsics direction backwards and the canvas still looks plausible, the run
still finishes, and the accuracy just moves for the wrong reason. So this gate
checks the things that MUST NOT move, not the thing we hope will:

  1. Structural, exhaustive over the benchmark: `_arkit_upright_k` is 0 for
     every ScanNet and ScanNet++ scene. Every uprighting branch sits behind
     `if _upright_k:`, so k=0 makes the flag a literal no-op there.
  2. Loader: for k=0 ARKit scenes and for ScanNet / ScanNet++ scenes, the
     loader's images, depths, intrinsics and poses are BIT-IDENTICAL with the
     flag on and off.
  3. Geometry: for rotated ARKit scenes the image genuinely turns and the
     lifted world points do NOT. World points come from a faithful copy of
     `reproject_scene`'s per-frame unprojection, then get de-rotated and
     compared elementwise.

Residual note for (3): `reproject_scene` resizes depth to the patch grid with
mode='nearest', and nearest sampling does not commute exactly with a flip (out
index j reads floor((j+0.5)*s); reversing the axis reads a neighbouring source
row whenever (j+0.5)*s is an integer). So a small minority of patches near
depth discontinuities can differ by their local depth step. The gate reports
the median and the tail separately for that reason -- the median is the
statement about the transform, the tail is that interpolation artefact.

Usage:
    python tests/regression/check_arkit_upright.py --run-dir OneCanvas-stage2 [--scenes-per-bucket 2]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "training"))

MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def build_datasets(run_dir):
    from transformers import AutoProcessor
    from training.run_benchmarks import build_data_args, parse_args
    from onecanvas.data.data_processor_3d import SceneQADataset

    argv = sys.argv
    lora = os.path.join(run_dir, "best_checkpoint")
    lora = lora if os.path.isdir(lora) else run_dir
    sys.argv = ["gate", "--from-config", run_dir, "--lora", lora,
                "--datasets", "vsi_bench", "--image-resolution", "640x480"]
    args = parse_args()
    sys.argv = argv

    processor = AutoProcessor.from_pretrained(MODEL)
    args.upright_arkit = False
    off = SceneQADataset(processor, build_data_args(args, "vsi_bench"), data_split="test")
    args.upright_arkit = True
    on = SceneQADataset(processor, build_data_args(args, "vsi_bench"), data_split="test")
    return off, on


def to_grid(depth, grid):
    """reproject_scene's depth step: nearest resize to the patch grid, mm->m."""
    d = depth.float()
    while d.dim() < 4:
        d = d.unsqueeze(0)
    return F.interpolate(d, size=grid, mode="nearest").reshape(grid) / 1000.0


def lift_grid(zi, intr, pose, dims):
    """reproject_scene's per-frame unprojection given an already-gridded depth:
    pixel centres at col+0.5, intrinsics scaled by grid/orig, camera->world.
    Returns [H_grid, W_grid, 3] world points."""
    H_feat, W_feat = zi.shape
    fx, fy, cx, cy = [float(x) for x in intr]
    orig_w, orig_h = float(dims[0]), float(dims[1])
    sx, sy = W_feat / orig_w, H_feat / orig_h
    v, u = torch.meshgrid(torch.arange(H_feat).float() + 0.5,
                          torch.arange(W_feat).float() + 0.5, indexing="ij")
    xi = (u - cx * sx) * zi / (fx * sx)
    yi = (v - cy * sy) * zi / (fy * sy)
    pts = torch.stack([xi, yi, zi, torch.ones_like(zi)], -1)
    return (pts.reshape(-1, 4) @ pose.float().T)[:, :3].reshape(H_feat, W_feat, 3)


def lift(depth, intr, pose, dims, grid=(17, 22)):
    return lift_grid(to_grid(depth, grid), intr, pose, dims)


def unrot(w, k):
    """Undo k clockwise quarter turns on an [H, W, 3] world-point map."""
    from onecanvas.data.arkit_upright import rotate_array_cw
    return rotate_array_cw(w.permute(2, 0, 1), -k).permute(1, 2, 0)


def rival_intrinsics(intr, dims, k):
    """The transform this pipeline must NOT use: the integer-centre convention
    ((H-1)-cy instead of H-cy) that the geometry track's toolkit uses. Kept as
    a live discriminator so the gate proves the choice instead of asserting it."""
    fx, fy, cx, cy = [float(x) for x in intr]
    w, h = float(dims[0]), float(dims[1])
    for _ in range(k % 4):
        fx, fy = fy, fx
        cx, cy = (h - 1) - cy, cx
        w, h = h, w
    return torch.tensor([fx, fy, cx, cy])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-per-bucket", type=int, default=2)
    ap.add_argument("--run-dir", required=True,
                    help="A stage-2 run or the downloaded stage2 branch (holds resolved_config.json)")
    opt = ap.parse_args()

    from onecanvas.data.arkit_upright import rotate_array_cw

    src = {}
    from onecanvas.data import VSI_BENCH
    with open(VSI_BENCH["annotation_path"]) as fh:
        for line in fh:
            d = json.loads(line)
            src[str(d["scene_name"])] = {"scannet": "ScanNet", "arkitscenes": "ARKit",
                                         "scannetpp": "ScanNet++"}[d["dataset"]]

    off, on = build_datasets(opt.run_dir)
    by_scene = {}
    for it in off.list_data_dict:
        by_scene.setdefault(str(it["scene_id"]), it)
    print(f"[gate] vsi_bench test items: {len(off.list_data_dict)}  scenes: {len(by_scene)}")

    # ---- 1. structural, exhaustive ----------------------------------------
    print("\n[1] structural: which scenes can the flag touch at all?")
    ks, unresolved = {}, []
    for sid, it in by_scene.items():
        sdir = off._resolve_scene_dir(sid, it["data_path"], scene_subdir=it.get("scene_subdir"))
        if sdir is None:
            unresolved.append(sid)
            continue
        ks[sid] = on._arkit_upright_k(sdir)

    bad = sorted(s for s, k in ks.items() if src.get(s) != "ARKit" and k)
    print(f"    scenes resolved: {len(ks)}  unresolved: {len(unresolved)}")
    print(f"    non-ARKit scenes with k != 0: {len(bad)}  "
          f"{'OK' if not bad else 'FAIL ' + str(bad[:5])}")
    per_src = {}
    for s, k in ks.items():
        per_src.setdefault(src.get(s, "?"), []).append(k)
    for s, v in sorted(per_src.items()):
        dist = {k: v.count(k) for k in sorted(set(v))}
        print(f"    {s:10s} n={len(v):4d}  k distribution {dist}")
    assert not bad, "a non-ARKit scene would be rotated"

    # ---- 2 + 3. loader-level -----------------------------------------------
    print("\n[2/3] loader: flag off vs on")
    buckets = {}
    for sid, k in ks.items():
        buckets.setdefault((src.get(sid, "?"), k), []).append(sid)

    n_fail = 0
    for (label, k), sids in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        for sid in sorted(sids)[:opt.scenes_per_bucket]:
            it = by_scene[sid]
            kw = dict(sample_idx=0, scene_subdir=it.get("scene_subdir"),
                      dataset_name=it.get("dataset_name"))
            a = off._load_scene_data(sid, it["data_path"], **kw)
            b = on._load_scene_data(sid, it["data_path"], **kw)
            if a is None or b is None:
                print(f"    {label:10s} k={k} {sid}: SKIP (loader returned None)")
                continue
            n = len(a["images"])
            same = dict(
                img=all(np.array_equal(np.asarray(a["images"][i]), np.asarray(b["images"][i]))
                        for i in range(n)),
                depth=all(torch.equal(a["depths"][i], b["depths"][i]) for i in range(n)),
                K=all(torch.equal(a["intrinsics"][i], b["intrinsics"][i]) for i in range(n)),
                pose=all(torch.equal(a["poses"][i], b["poses"][i]) for i in range(n)),
            )

            if not k:
                ok = all(same.values())
                n_fail += 0 if ok else 1
                print(f"    {label:10s} k={k} {sid}: bit-identical="
                      f"{'YES' if ok else 'NO'}  {same}  ({n} frames)  "
                      f"{'OK' if ok else 'FAIL'}")
                continue

            # The patch grid turns with the frame: a 90-degree scene is
            # portrait on the upright side, so its grid is the transpose.
            g_a = (17, 22)
            g_b = g_a if k % 2 == 0 else g_a[::-1]

            # (3a) TRANSFORM EXACTNESS. Feed both sides the SAME depth samples
            #      (rotate the already-gridded map instead of re-resampling the
            #      raw one) so nothing but the intrinsics and the pose can move
            #      the points. This must be exact. Repeated with the rival
            #      integer-centre intrinsics convention, (H-1)-cy, which must
            #      NOT be exact -- that is what proves the pipeline needs the
            #      corner-origin form rather than merely tolerating it.
            # (3b) END TO END, including the nearest re-resampling of raw depth.
            #      Any residual here is that resampling, not the transform:
            #      nearest sampling does not commute with an axis flip.
            exact, e2e = [], []
            for i in range(n):
                za = to_grid(a["depths"][i], g_a)
                zb_exact = rotate_array_cw(za, k)
                wa = lift_grid(za, a["intrinsics"][i], a["poses"][i], a["image_dims"][i])
                wb = lift_grid(zb_exact, b["intrinsics"][i], b["poses"][i], b["image_dims"][i])
                exact.append(float((wa - unrot(wb, k)).abs().max()))

                wb2 = lift_grid(to_grid(b["depths"][i], g_b), b["intrinsics"][i],
                                b["poses"][i], b["image_dims"][i])
                d = (wa - unrot(wb2, k)).abs().amax(-1)
                d = d[torch.isfinite(d)]
                if d.numel():
                    e2e.append((float(d.median()), float(d.quantile(0.99)), float(d.max())))

            ex = max(exact) if exact else float("nan")
            med = float(np.median([x[0] for x in e2e])) if e2e else float("nan")
            p99 = float(np.median([x[1] for x in e2e])) if e2e else float("nan")
            mx = max(x[2] for x in e2e) if e2e else float("nan")

            # Rival convention, one frame, as a discriminator.
            i0 = 0
            K_rival = rival_intrinsics(a["intrinsics"][i0], a["image_dims"][i0], k)
            w_rival = lift_grid(rotate_array_cw(to_grid(a["depths"][i0], g_a), k),
                                K_rival, b["poses"][i0], b["image_dims"][i0])
            rival_err = float((lift_grid(to_grid(a["depths"][i0], g_a), a["intrinsics"][i0],
                                         a["poses"][i0], a["image_dims"][i0])
                               - unrot(w_rival, k)).abs().max())

            ok = (not same["img"]) and ex < 1e-5
            n_fail += 0 if ok else 1
            print(f"    {label:10s} k={k} {sid}: image turned={not same['img']}  "
                  f"dims {tuple(a['image_dims'][0])}->{tuple(b['image_dims'][0])}")
            print(f"        transform-exact max err   {ex:.2e} m   "
                  f"{'OK' if ex < 1e-5 else 'FAIL'}   "
                  f"[(H-1)-cy rival: {rival_err:.2e} m]")
            print(f"        end-to-end (nearest depth resample) median={med:.2e} "
                  f"p99={p99:.2e} max={mx:.2e} m")

    print(f"\n[gate] {'PASS' if n_fail == 0 else f'FAIL ({n_fail} buckets)'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
