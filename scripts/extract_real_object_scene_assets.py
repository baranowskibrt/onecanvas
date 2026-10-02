"""Whole-scene real-object asset bank extraction.

Sister of scripts/extract_real_object_assets.py. Instead of one .pt per
EmbodiedScan OBB (tight bbox + 5 cm inflation), this saves one .pt per
ScanNet scene containing every valid patch in the scene plus the OBB
metadata for every class-allowlisted EmbodiedScan annotation.

Per-scene .pt at <output_dir>/<scene_id>.pt:
    {
        "features":        [N_valid, N_layers, C] bf16,
        "world_xyz":       [N_valid, 3] float32,    # axis-aligned ScanNet world frame
        "frame_indices":   [N_valid] int16,
        "n_source_images": int,
        "source_scene":    str,
        "obbs": [
            {
                "target_id":  int,
                "label":      str,
                "center":     [3] float32,
                "dims":       [3] float32,           # tight (no inflation baked in)
                "euler_zxy":  [3] float32,
            }, ...
        ],
    }

The bank consumer (training.onecanvas.data.real_object_scene_bank.RealObjectSceneBank)
chooses inflation at paste time by intersecting world_xyz with an inflated
OBB local-frame box, so the bank file is inflation-free.

Per-scene resume markers under <output_dir>/_scenes_done/<scene_id>.json
make the run safe to interrupt.

scene_index.json side-output at the bank root, built from markers on
completion: a flat list of {label, scene_path, scene_id, target_id,
obb_index} entries for the runtime per-class lookup table.
"""

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# This script lives in <repo>/scripts/, so the repo root is two levels up.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# utils/__init__.py drags in onecanvas.* — load bbox.py directly to dodge that.
_BBOX_PATH = os.path.join(REPO, "utils", "bbox.py")
_spec_b = importlib.util.spec_from_file_location("_oc_bbox_scene", _BBOX_PATH)
_bbox_mod = importlib.util.module_from_spec(_spec_b)
_spec_b.loader.exec_module(_bbox_mod)
_euler_zxy_to_rotation_matrix = _bbox_mod._euler_zxy_to_rotation_matrix

# Reuse the per-object harvest's I/O + visual-tower helpers verbatim so
# coordinate frames match bit-for-bit.
_HARVEST_PATH = os.path.join(REPO, "scripts", "extract_real_object_assets.py")
_spec_h = importlib.util.spec_from_file_location("_extract_assets_scene", _HARVEST_PATH)
_h = importlib.util.module_from_spec(_spec_h)
_spec_h.loader.exec_module(_h)
load_scene_frames = _h.load_scene_frames
load_embodied_scan_obbs = _h.load_embodied_scan_obbs
load_visual_tower = _h.load_visual_tower
run_visual_tower = _h.run_visual_tower
reconstruct_world_pts = _h.reconstruct_world_pts
SCANNET_PRE = _h.SCANNET_PRE
DEFAULT_CLASSES = _h.DEFAULT_CLASSES

from reprojection.scene_reprojection import compute_scene_geometry  # noqa: E402


def harvest_scene(scene_id, scene_dir, scene_obbs, processor, model, device, args):
    """Returns ({scene_payload}, list[(target_id, label, n_kept, drop_reason)]) or None."""
    target_res = tuple(int(x) for x in args.image_resolution.lower().split("x"))
    scene = load_scene_frames(scene_id, scene_dir, args.num_images, target_res)
    if scene is None:
        return None
    n_imgs = len(scene["images"])

    features, H_feat, W_feat = run_visual_tower(processor, model, scene["images"], device)
    geom = compute_scene_geometry(
        depths=scene["depths"],
        poses=scene["poses"],
        intrinsics=scene["intrinsics"],
        image_dims=scene["image_dims"],
        H_feat=H_feat,
        W_feat=W_feat,
        device="cpu",
        center_override=None,
        yaw_angle=None,
    )
    if geom.n_valid == 0:
        return None
    world_pts = reconstruct_world_pts(geom)            # [N_valid, 3] cpu float32
    frame_index = geom.frame_index.long().cpu()        # [N_valid]

    N_imgs, N_layers, _, _, C = features.shape
    feats_flat = features.permute(0, 2, 3, 1, 4).reshape(N_imgs * H_feat * W_feat, N_layers, C)
    valid_feats = feats_flat[geom.valid_indices.cpu()].contiguous()  # [N_valid, N_layers, C] bf16

    # Per-OBB inclusion check just decides whether to keep the OBB metadata
    # entry. The scene's features array is invariant to which OBBs survive.
    world_pts_np = world_pts.numpy().astype(np.float64)
    frame_index_np = frame_index.numpy()
    obbs_meta = []
    dropped = []
    for tid, label, obb in scene_obbs:
        cx, cy, cz, dx, dy, dz, rx, ry, rz = obb
        center = np.array([cx, cy, cz], dtype=np.float64)
        dims = np.array([dx, dy, dz], dtype=np.float64)
        R = _euler_zxy_to_rotation_matrix(rx, ry, rz)
        # Tight inclusion (no extractor-side inflation; bank handles that).
        local_xyz = (world_pts_np - center) @ R
        half = dims / 2.0
        mask = (np.abs(local_xyz) <= half).all(axis=1)
        n_kept = int(mask.sum())
        if n_kept < args.min_patches:
            dropped.append((tid, label, n_kept, "min_patches"))
            continue
        n_distinct = len(set(frame_index_np[mask].tolist()))
        if n_distinct < args.min_frames:
            dropped.append((tid, label, n_distinct, "min_frames"))
            continue
        obbs_meta.append({
            "target_id":  int(tid),
            "label":      str(label),
            "center":     torch.tensor(center, dtype=torch.float32),
            "dims":       torch.tensor(dims,   dtype=torch.float32),
            "euler_zxy":  torch.tensor([rx, ry, rz], dtype=torch.float32),
        })

    payload = {
        "features":        valid_feats,                                       # bf16
        "world_xyz":       world_pts.float().contiguous(),                    # float32
        "frame_indices":   frame_index.short().contiguous(),                  # int16
        "n_source_images": int(n_imgs),
        "source_scene":    str(scene_id),
        "obbs":            obbs_meta,
    }
    return payload, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="Destination directory for the scene asset bank.")
    ap.add_argument("--classes", default=DEFAULT_CLASSES)
    ap.add_argument("--min_patches", type=int, default=6)
    ap.add_argument("--min_frames", type=int, default=2)
    ap.add_argument("--num_images", type=int, default=32)
    ap.add_argument("--image_resolution", default="320x240")
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max_scenes", type=int, default=0,
                    help="Cap (0 = all). Useful for smoke tests.")
    ap.add_argument("--scenes", default="",
                    help="Comma-separated scene IDs. Overrides automatic discovery.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report_every", type=int, default=20)
    args = ap.parse_args()

    classes = set(c.strip() for c in args.classes.split(",") if c.strip())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[args] output_dir={output_dir} classes={sorted(classes)} "
          f"num_images={args.num_images} res={args.image_resolution}")

    obbs = load_embodied_scan_obbs(classes)

    if args.scenes:
        scene_ids = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        preprocessed = set(os.listdir(SCANNET_PRE)) if os.path.isdir(SCANNET_PRE) else set()
        scene_ids = sorted(s for s in obbs.keys() if s in preprocessed)
        if args.max_scenes > 0:
            scene_ids = scene_ids[:args.max_scenes]
        print(f"[args] auto-discovered {len(scene_ids)} ScanNet scenes "
              f"(intersection of EmbodiedScan and preprocessed)")

    processor, model = load_visual_tower(args.model_name, args.device)

    progress_dir = output_dir / "_scenes_done"
    progress_dir.mkdir(parents=True, exist_ok=True)

    n_scenes_processed = 0
    n_scenes_skipped_resume = 0
    n_obbs_kept = 0
    n_obbs_dropped = 0
    t_start = time.time()
    for i, sid in enumerate(scene_ids):
        marker = progress_dir / f"{sid}.json"
        if marker.exists():
            n_scenes_skipped_resume += 1
            continue

        scene_dir = os.path.join(SCANNET_PRE, sid)
        if not os.path.isdir(scene_dir):
            continue
        scene_obbs = obbs.get(sid, [])
        if not scene_obbs:
            continue
        try:
            res = harvest_scene(sid, scene_dir, scene_obbs, processor, model, args.device, args)
        except Exception as e:
            print(f"[error] {sid}: {type(e).__name__}: {e}")
            continue
        if res is None:
            with open(marker, "w") as f:
                json.dump({"scene_id": sid, "scene_path": "",
                           "obbs": [], "skipped": True}, f)
            continue
        payload, dropped = res
        scene_path = output_dir / f"{sid}.pt"
        torch.save(payload, str(scene_path))
        n_scenes_processed += 1
        n_obbs_kept += len(payload["obbs"])
        n_obbs_dropped += len(dropped)

        marker_entry = {
            "scene_id":   sid,
            "scene_path": str(scene_path),
            "obbs": [
                {"label": o["label"], "target_id": int(o["target_id"]),
                 "obb_index": j}
                for j, o in enumerate(payload["obbs"])
            ],
            "skipped": False,
        }
        tmp = marker.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(marker_entry, f)
        tmp.rename(marker)

        # The mmap-friendly per-scene tensor stays resident as a python
        # dict reference until we drop it; let GC reclaim before the next
        # scene to keep RSS bounded.
        del payload
        gc.collect()

        if (i + 1) % args.report_every == 0 or (i + 1) == len(scene_ids):
            dt = time.time() - t_start
            rate = n_scenes_processed / max(dt, 1e-6)
            print(f"[{i+1}/{len(scene_ids)}] new={n_scenes_processed} "
                  f"resume_skip={n_scenes_skipped_resume} "
                  f"obbs_kept={n_obbs_kept} obbs_dropped={n_obbs_dropped} "
                  f"rate={rate:.2f} sc/s")

    # Build scene_index.json from the union of all markers (so prior runs
    # are folded in too).
    scene_index = []
    for marker_path in sorted(progress_dir.glob("*.json")):
        with open(marker_path) as f:
            data = json.load(f)
        scene_path = data.get("scene_path", "")
        if not scene_path:
            continue
        for entry in data.get("obbs", []):
            scene_index.append({
                "label":      entry["label"],
                "scene_path": scene_path,
                "scene_id":   data["scene_id"],
                "target_id":  int(entry["target_id"]),
                "obb_index":  int(entry["obb_index"]),
            })

    index_path = output_dir / "scene_index.json"
    with open(index_path, "w") as f:
        json.dump(scene_index, f)

    by_class = defaultdict(int)
    for a in scene_index:
        by_class[a["label"]] += 1
    print("\n=== Summary ===")
    print(f"scenes processed: {n_scenes_processed}")
    print(f"obbs kept:        {n_obbs_kept}")
    print(f"obbs dropped:     {n_obbs_dropped}")
    print("by class:")
    for lbl, n in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {lbl:12s} {n:6d}")
    print(f"index: {index_path}")
    print(f"wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
