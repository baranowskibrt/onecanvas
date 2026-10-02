"""Phase 1: harvest real-object assets from ScanNet scenes.

For each ScanNet scene that has EmbodiedScan OBB annotations:
  1. Run the Qwen3-VL visual tower on N frames (matches training resolution).
  2. Lift each feature patch to world XYZ via depth + GT pose (axis-aligned ScanNet frame).
  3. For each EmbodiedScan OBB matching the class allowlist, gather the features /
     xyz / frame_index of patches whose world position falls inside the OBB.
  4. Save per-asset .pt files keyed by class label, plus a flat asset_index.json.

Asset xyz_offsets are stored in the OBB's local frame (canonical pose) so that
the runtime paste step can apply a random yaw around world Z. frame_indices
are stored raw (0..N_images-1) so the runtime paste step can preserve the
relative T-spread.

Output: <output_dir>/<label>/<scene_id>_obj<target_id>.pt + asset_index.json.
"""

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# This script lives in <repo>/scripts/, so the repo root is two levels up.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# utils/__init__.py pulls in onecanvas.* — load bbox.py directly to avoid that.
_BBOX_PATH = os.path.join(REPO, "utils", "bbox.py")
_spec = importlib.util.spec_from_file_location("_oc_bbox", _BBOX_PATH)
_bbox_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bbox_mod)
_euler_zxy_to_rotation_matrix = _bbox_mod._euler_zxy_to_rotation_matrix

from reprojection.scene_reprojection import compute_scene_geometry  # noqa: E402

# Dataset trees live under a single root, resolved the same way as
# training/onecanvas/data/__init__.py:
#   1. ONECANVAS_DATA_ROOT env var (explicit override, always wins).
#   2. a `datasets/` directory sibling to the repo root (common checkout layout).
#   3. otherwise raise with a clear message.
_DATA_ROOT = os.environ.get("ONECANVAS_DATA_ROOT")
if not _DATA_ROOT:
    _sibling_datasets = os.path.join(os.path.dirname(REPO), "datasets")
    if os.path.isdir(_sibling_datasets):
        _DATA_ROOT = _sibling_datasets
if not _DATA_ROOT:
    raise RuntimeError(
        "ONECANVAS_DATA_ROOT is not set. Export it to the directory that "
        "contains scannet/, vlm_annotations/, etc., before running this "
        "extractor.\n"
        "  example: export ONECANVAS_DATA_ROOT=/path/to/your/datasets"
    )

SCANNET_PRE = f"{_DATA_ROOT}/scannet/scannet_preprocessed"
EMBODIED_SCAN_DIR = f"{_DATA_ROOT}/vlm_annotations/embodiedscan"
DEFAULT_CLASSES = "chair,table,couch,bed,cabinet,desk,shelf,sink,microwave,dresser,stool"
# Widened class list — every EmbodiedScan ScanNet label with >=150
# samples that's a reasonable discrete-instance counting target. Original
# 11-class default left intact for backward compatibility with existing
# asset bank dirs; pass --classes "$WIDE_CLASSES" (or this constant via
# script call) when re-extracting a fresh wider bank.
# VSI-vocabulary coverage is incomplete (no `counter`, `refrigerator`,
# `tv`, `nightstand`, `bookshelf`, `ceiling light` in EmbodiedScan
# ScanNet; closest are `cabinet`, `microwave`, `monitor`, `dresser`,
# `shelf`, `light`).
WIDE_CLASSES = (
    "chair,table,couch,bed,cabinet,desk,shelf,sink,microwave,dresser,stool,"
    "door,window,bin,picture,box,bottle,lamp,pillow,towel,book,backpack,"
    "monitor,curtain,cup,bag,plant,shoe,keyboard,mirror,light,speaker,toy,"
    "telephone,mouse,laptop,radiator,fan,printer,clock,wardrobe,oven,"
    "blackboard,basket,bowl,toilet,bicycle"
)
OBB_RE = re.compile(r"\(([^)]+)\)")


# --------------------------------------------------------------------------
# EmbodiedScan loading
# --------------------------------------------------------------------------

def parse_obb(answer_str):
    m = OBB_RE.search(answer_str)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) != 9:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None


def load_embodied_scan_obbs(class_allowlist):
    """Returns dict[scene_id -> list[(target_id, label, obb)]] filtered + deduped by target_id."""
    by_scene = defaultdict(dict)
    n_total = 0
    for split in ("train", "val"):
        path = f"{EMBODIED_SCAN_DIR}/embodiedscan_scannet_{split}.jsonl"
        with open(path) as fh:
            for line in fh:
                item = json.loads(line)
                lbl = item["object_label"]
                if class_allowlist and lbl not in class_allowlist:
                    continue
                obb = parse_obb(item["answers"][0])
                if obb is None:
                    continue
                tid = item["target_id"]
                sid = item["scene_id"]
                if tid not in by_scene[sid]:
                    by_scene[sid][tid] = (lbl, obb)
                    n_total += 1
    out = {k: [(tid, lbl, obb) for tid, (lbl, obb) in v.items()] for k, v in by_scene.items()}
    print(f"[obbs] {len(out)} scenes, {n_total} unique OBBs after class filter and dedup")
    return out


# --------------------------------------------------------------------------
# ScanNet scene I/O (axis-aligned frame, matches wants_aligned=True path)
# --------------------------------------------------------------------------

def parse_scene_info(info_path):
    """Returns (depth_K[fx,fy,cx,cy], axisAlignment[4x4], E_ctd_inv[4x4])."""
    vals = {}
    with open(info_path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    depth_K = (
        float(vals["fx_depth"]),
        float(vals["fy_depth"]),
        float(vals["mx_depth"]),
        float(vals["my_depth"]),
    )
    if "axisAlignment" in vals:
        axis = torch.tensor([float(x) for x in vals["axisAlignment"].split()],
                            dtype=torch.float32).reshape(4, 4)
    else:
        axis = torch.eye(4, dtype=torch.float32)
    if "colorToDepthExtrinsics" in vals:
        nums = [float(x) for x in vals["colorToDepthExtrinsics"].split()]
        E_ctd = torch.tensor(nums, dtype=torch.float32).reshape(4, 4) if len(nums) == 16 else torch.eye(4)
    else:
        E_ctd = torch.eye(4, dtype=torch.float32)
    return depth_K, axis, torch.linalg.inv(E_ctd)


def list_scene_frames(scene_dir):
    pose_dir = os.path.join(scene_dir, "pose")
    depth_dir = os.path.join(scene_dir, "depth")
    if not (os.path.isdir(pose_dir) and os.path.isdir(depth_dir)):
        return []
    pose_stems = {os.path.splitext(p)[0] for p in os.listdir(pose_dir) if p.endswith(".txt")}
    depth_stems = {os.path.splitext(p)[0] for p in os.listdir(depth_dir) if p.endswith(".png")}
    return sorted(pose_stems & depth_stems, key=lambda s: int(s))


def find_color_path(scene_dir, stem, target_res):
    """Prefer pre-resized image at target_res, else any color/, else fail."""
    res_str = f"{target_res[0]}x{target_res[1]}"
    for cand in (f"color_{res_str}", "color_640x480", "color"):
        p = os.path.join(scene_dir, cand, f"{stem}.jpg")
        if os.path.exists(p):
            return p
    return None


def load_scene_frames(scene_id, scene_dir, num_images, target_res):
    """Returns dict with images (PIL list at target_res), depths, poses, intrinsics, image_dims, frame_stems."""
    info_path = os.path.join(scene_dir, f"{scene_id}.txt")
    if not os.path.exists(info_path):
        return None
    depth_K, axis, E_ctd_inv = parse_scene_info(info_path)

    stems = list_scene_frames(scene_dir)
    if not stems:
        return None
    if len(stems) > num_images:
        idx = np.linspace(0, len(stems) - 1, num_images, dtype=int)
        stems = [stems[i] for i in idx]

    images, depths, poses, intrinsics, image_dims, kept_stems = [], [], [], [], [], []
    for stem in stems:
        cp = find_color_path(scene_dir, stem, target_res)
        if cp is None:
            continue
        try:
            pil = Image.open(cp).convert("RGB")
            if pil.size != tuple(target_res):
                pil = pil.resize(tuple(target_res), Image.LANCZOS)
        except Exception:
            continue

        dpath = os.path.join(scene_dir, "depth", f"{stem}.png")
        ppath = os.path.join(scene_dir, "pose", f"{stem}.txt")
        try:
            depth = torch.from_numpy(np.array(Image.open(dpath)).astype(np.float32))  # mm
            mat = torch.from_numpy(np.loadtxt(ppath)).float()
        except Exception:
            continue
        if mat.shape != (4, 4) or not torch.isfinite(mat).all():
            continue
        pose_world = axis @ mat @ E_ctd_inv

        images.append(pil)
        depths.append(depth)
        poses.append(pose_world)
        # depth-camera intrinsics, applied at sensor resolution (depth.shape == (H, W))
        intrinsics.append(torch.tensor(list(depth_K), dtype=torch.float32))
        image_dims.append(torch.tensor([depth.shape[-1], depth.shape[-2]], dtype=torch.float32))  # (W, H)
        kept_stems.append(stem)

    if len(images) < 2:
        return None
    return {
        "images": images,
        "depths": torch.stack(depths),
        "poses": torch.stack(poses),
        "intrinsics": torch.stack(intrinsics),
        "image_dims": torch.stack(image_dims),
        "frame_stems": kept_stems,
    }


# --------------------------------------------------------------------------
# Visual tower
# --------------------------------------------------------------------------

def load_visual_tower(model_name, device):
    """Load Qwen3-VL processor + full model (LLM stays loaded; ~16 GB bf16 on A6000)."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"[model] loading {model_name} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    print(f"[model] loaded in {time.time()-t0:.1f}s on {device}")
    return processor, model


def run_visual_tower(processor, model, images, device):
    """Returns (features [N_imgs, N_layers, H_feat, W_feat, C] cpu/bf16, H_feat, W_feat).

    Uses model.get_image_features(...) — the same entry point inference/pipeline.py
    and curriculum_marker_stash._build_stash_inner use, so feature semantics match training.
    """
    proc = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device, dtype=torch.bfloat16)
    image_grid_thw = proc["image_grid_thw"].to(device)
    # After merge_size=2 collapse, feature grid per image is (H/2, W/2).
    h_feat = (image_grid_thw[:, 1] // 2).tolist()
    w_feat = (image_grid_thw[:, 2] // 2).tolist()
    if not all(h == h_feat[0] and w == w_feat[0] for h, w in zip(h_feat, w_feat)):
        raise RuntimeError(f"Heterogeneous feature grids per image: h={h_feat}, w={w_feat}")
    H_feat, W_feat = h_feat[0], w_feat[0]
    tokens_per_img = [H_feat * W_feat] * len(images)

    with torch.no_grad():
        out = model.get_image_features(pixel_values, image_grid_thw)
    # out: BaseModelOutputWithDeepstackFeatures
    #   .pooler_output: tuple of per-image tensors of shape [H*W, C]
    #   .deepstack_features: list of [N_total, C] tensors per deepstack layer
    embeds_per_img = out.pooler_output
    deepstack = list(getattr(out, "deepstack_features", []) or [])

    layer0 = torch.stack([emb.view(H_feat, W_feat, -1) for emb in embeds_per_img])  # [N_imgs, H, W, C]
    layers = [layer0]
    for layer_tensor in deepstack:
        chunks = torch.split(layer_tensor, tokens_per_img, dim=0)
        layers.append(torch.stack([ch.view(H_feat, W_feat, -1) for ch in chunks]))
    features = torch.stack(layers, dim=1).to("cpu", dtype=torch.bfloat16)  # [N_imgs, N_layers, H, W, C]
    return features, H_feat, W_feat


# --------------------------------------------------------------------------
# Geometry: spherical -> world XYZ
# --------------------------------------------------------------------------

def reconstruct_world_pts(geom):
    """Invert compute_scene_geometry's spherical projection (yaw_angle=None assumed).

    geom returns local (scene-centered) coords as:
        x_c = local_x;  y_c = -local_z;  z_c = local_y
        depth = sqrt(x_c^2 + y_c^2 + z_c^2)
        longitude = atan2(x_c, z_c)
        latitude  = atan2(y_c, sqrt(x_c^2 + z_c^2))
    """
    lon = geom.longitude
    lat = geom.latitude
    d = geom.depth
    cos_lat = torch.cos(lat)
    x_c = d * cos_lat * torch.sin(lon)
    y_c = d * torch.sin(lat)
    z_c = d * cos_lat * torch.cos(lon)
    local = torch.stack([x_c, z_c, -y_c], dim=-1)
    return (local + geom.center_point.to(local.device)).cpu()


# --------------------------------------------------------------------------
# Per-scene harvest
# --------------------------------------------------------------------------

def harvest_scene(scene_id, scene_dir, scene_obbs, processor, model, device, args):
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
    world_pts = reconstruct_world_pts(geom)  # [N_valid, 3]
    frame_index = geom.frame_index.long().cpu()  # [N_valid]

    # Gather valid features
    N_imgs, N_layers, _, _, C = features.shape
    feats_flat = features.permute(0, 2, 3, 1, 4).reshape(N_imgs * H_feat * W_feat, N_layers, C)
    valid_feats = feats_flat[geom.valid_indices.cpu()]  # [N_valid, N_layers, C] bf16

    saved = []
    dropped = []
    for tid, label, obb in scene_obbs:
        cx, cy, cz, dx, dy, dz, rx, ry, rz = obb
        center = np.array([cx, cy, cz], dtype=np.float64)
        dims = np.array([dx, dy, dz], dtype=np.float64)
        R = _euler_zxy_to_rotation_matrix(rx, ry, rz)
        # OBB-local coords for inclusion test
        local_xyz = (world_pts.numpy().astype(np.float64) - center) @ R
        half = dims / 2.0 + args.inflation
        mask = (np.abs(local_xyz) <= half).all(axis=1)
        n_kept = int(mask.sum())
        if n_kept < args.min_patches:
            dropped.append((tid, label, n_kept, "min_patches"))
            continue
        kept_frames = frame_index.numpy()[mask]
        n_distinct = len(set(kept_frames.tolist()))
        if n_distinct < args.min_frames:
            dropped.append((tid, label, n_distinct, "min_frames"))
            continue
        saved.append({
            "target_id": tid,
            "label": label,
            "features": valid_feats[mask].clone().contiguous(),       # [K, N_layers, C] bf16
            "xyz_offsets": torch.from_numpy(local_xyz[mask]).float(), # [K, 3] OBB-local
            "frame_indices": torch.from_numpy(kept_frames).short(),    # [K] int16, raw
            "n_source_images": int(n_imgs),
            "bbox_dims": torch.tensor(dims, dtype=torch.float32),
            "obb_euler_zxy": torch.tensor([rx, ry, rz], dtype=torch.float32),
            "source_scene": scene_id,
        })

    return saved, dropped, scene["frame_stems"]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="Destination directory for the asset bank.")
    ap.add_argument("--classes", default=DEFAULT_CLASSES)
    ap.add_argument("--inflation", type=float, default=0.05)
    ap.add_argument("--min_patches", type=int, default=6)
    ap.add_argument("--min_frames", type=int, default=2)
    ap.add_argument("--num_images", type=int, default=32)
    ap.add_argument("--image_resolution", default="320x240")
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max_scenes", type=int, default=0,
                    help="Cap (0 = all).  Useful for smoke tests.")
    ap.add_argument("--scenes", default="",
                    help="Comma-separated scene IDs.  Overrides automatic discovery.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report_every", type=int, default=20)
    args = ap.parse_args()

    classes = set(c.strip() for c in args.classes.split(",") if c.strip())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[args] output_dir={output_dir} classes={sorted(classes)} num_images={args.num_images} res={args.image_resolution}")

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

    # Per-scene progress markers under _scenes_done/<scene_id>.json so the run
    # can be resumed safely after a crash. Each marker holds the index entries
    # produced for that scene; the final asset_index.json is built by merging
    # all markers at the end.
    progress_dir = output_dir / "_scenes_done"
    progress_dir.mkdir(parents=True, exist_ok=True)

    n_assets_saved = 0
    n_dropped = 0
    n_scenes_processed = 0
    n_scenes_skipped_resume = 0
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
            # Mark as done with empty entries so we don't retry on resume.
            with open(marker, "w") as f:
                json.dump({"scene_id": sid, "entries": [], "skipped": True}, f)
            continue
        saved, dropped, frame_stems = res
        n_scenes_processed += 1
        n_dropped += len(dropped)
        scene_entries = []
        for asset in saved:
            label = asset["label"]
            tid = asset["target_id"]
            class_dir = output_dir / label
            class_dir.mkdir(parents=True, exist_ok=True)
            asset_path = class_dir / f"{sid}_obj{int(tid):04d}.pt"
            torch.save(asset, str(asset_path))
            scene_entries.append({
                "label": label,
                "path": str(asset_path),
                "n_patches": int(asset["features"].shape[0]),
                "n_source_images": int(asset["n_source_images"]),
                "source_scene": sid,
                "target_id": int(tid),
            })
            n_assets_saved += 1
        # Atomic-ish marker write (write to .tmp then rename).
        tmp = marker.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"scene_id": sid, "entries": scene_entries, "skipped": False}, f)
        tmp.rename(marker)
        if (i + 1) % args.report_every == 0 or (i + 1) == len(scene_ids):
            dt = time.time() - t_start
            rate = n_scenes_processed / max(dt, 1e-6)
            print(f"[{i+1}/{len(scene_ids)}] new={n_scenes_processed} "
                  f"resume_skip={n_scenes_skipped_resume} "
                  f"assets_saved={n_assets_saved} dropped={n_dropped} "
                  f"rate={rate:.2f} sc/s")

    # Aggregate all markers (including ones from prior runs) into asset_index.json.
    asset_index = []
    for marker_path in sorted(progress_dir.glob("*.json")):
        with open(marker_path) as f:
            data = json.load(f)
        asset_index.extend(data.get("entries", []))

    index_path = output_dir / "asset_index.json"
    with open(index_path, "w") as f:
        json.dump(asset_index, f)
    by_class = defaultdict(int)
    for a in asset_index:
        by_class[a["label"]] += 1
    print("\n=== Summary ===")
    print(f"scenes processed: {n_scenes_processed}")
    print(f"assets saved:     {n_assets_saved}")
    print(f"assets dropped:   {n_dropped}")
    print("by class:")
    for lbl, n in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {lbl:12s} {n:6d}")
    print(f"index: {index_path}")
    print(f"wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
