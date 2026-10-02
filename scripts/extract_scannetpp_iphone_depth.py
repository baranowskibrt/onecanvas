#!/usr/bin/env python
"""Extract ScanNet++ iPhone depth.bin -> per-frame uint16 PNGs.

Produces ``<data_root>/scannetpp/data/<scene>/iphone/depth/frame_NNNNNN.png``
(uint16, millimeters, 256x192), the depth source the OneCanvas dataloader's
ScanNet++ GT-calibration path reads (``depth_subdir: 'depth'`` at
training/onecanvas/data/data_processor_3d.py:1024; frame availability check
at :1009). Frame stems match the keys of ``pose_intrinsic_imu.json``
(consumed raw at data_processor_3d.py:962-998) and the extracted rgb/ frames
(scripts/extract_scannetpp_iphone_rgb.py).

Upstream source: the official ScanNet++ dataset download
(https://kaldir.vc.in.tum.de/scannetpp/ after accepting the ScanNet++ terms
of use), asset ``iphone/depth.bin`` per scene.

depth.bin layout (official scannetpp toolkit `extract_depth`):
  repeat:
    uint32 LE size
    <size> bytes of lz4.block-compressed uint16 depth (192 x 256, mm)

Only frames that have an extracted RGB (rgb/, rgb_640x480/, or rgb_320x240/)
are written; if no RGB folder exists yet, every frame is extracted. PNG
compression level differs from PNG defaults (speed), but the decoded uint16
array is exactly what depth.bin stores, so pixel content is bit-identical
to the original preprocessing.

Requires the ``lz4`` package (pip install lz4).

Example:
  python scripts/extract_scannetpp_iphone_depth.py --scene 00777c41d4
  python scripts/extract_scannetpp_iphone_depth.py --shard 0/4
"""
import argparse
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import lz4.block
import numpy as np
from tqdm import tqdm

DEPTH_H, DEPTH_W = 192, 256
UNCOMPRESSED_SIZE = DEPTH_H * DEPTH_W * 2  # uint16


def resolve_data_root(cli_value=None):
    """Same resolution order as training/onecanvas/data/__init__.py:
    CLI override, then ONECANVAS_DATA_ROOT, then a `datasets/` directory
    sibling to the repo root."""
    if cli_value:
        return cli_value
    root = os.environ.get("ONECANVAS_DATA_ROOT")
    if root:
        return root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sibling = os.path.join(os.path.dirname(repo_root), "datasets")
    if os.path.isdir(sibling):
        return sibling
    raise SystemExit(
        "Could not resolve the dataset root. Set ONECANVAS_DATA_ROOT or pass "
        "--data_root."
    )


def _write_one(args):
    out_path, payload = args
    if os.path.exists(out_path):
        return False
    decoded = lz4.block.decompress(payload, uncompressed_size=UNCOMPRESSED_SIZE)
    arr = np.frombuffer(decoded, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)
    # cv2.imwrite is ~5x faster than imageio for uint16 PNG. Use low
    # compression (PNG level 1) for another ~3x.
    cv2.imwrite(out_path, arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return True


def _list_needed_frames(scene_dir: Path) -> set:
    """Frame stems for which an RGB exists (we only need depth for those)."""
    needed = set()
    for sub in ("rgb", "rgb_640x480", "rgb_320x240"):
        d = scene_dir / "iphone" / sub
        if d.is_dir():
            for p in d.iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    needed.add(p.stem)
            if needed:
                break
    return needed


def extract_scene(scene_dir: Path, force: bool = False, pool=None,
                  out_root: Path | None = None) -> tuple[int, int]:
    depth_bin = scene_dir / "iphone" / "depth.bin"
    out_dir = (out_root or scene_dir / "iphone") / "depth"
    if not depth_bin.exists():
        return 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = _list_needed_frames(scene_dir)
    # If no RGB folder found, fall back to extracting everything.
    filter_active = bool(needed)

    tasks = []
    frame_id = 0
    skipped = 0
    with open(depth_bin, "rb") as f:
        while True:
            size_bytes = f.read(4)
            if len(size_bytes) == 0:
                break
            if len(size_bytes) < 4:
                raise RuntimeError(f"{depth_bin}: truncated length prefix at frame {frame_id}")
            size = struct.unpack("<I", size_bytes)[0]
            payload = f.read(size)
            if len(payload) < size:
                raise RuntimeError(f"{depth_bin}: truncated payload at frame {frame_id}")

            stem = f"frame_{frame_id:06d}"
            frame_id += 1
            if filter_active and stem not in needed:
                continue
            out_path = str(out_dir / f"{stem}.png")
            if os.path.exists(out_path) and not force:
                skipped += 1
                continue
            tasks.append((out_path, payload))

    if not tasks:
        return 0, skipped
    if pool is None:
        for t in tasks:
            _write_one(t)
    else:
        list(pool.map(_write_one, tasks, chunksize=8))
    return len(tasks), skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_root", default=None,
                    help="ScanNet++ scene root, i.e. <dataset_root>/scannetpp/data "
                         "(default: derived from ONECANVAS_DATA_ROOT or the datasets/ "
                         "directory sibling to the repo).")
    ap.add_argument("--scene", default=None, help="Single scene id (otherwise all).")
    ap.add_argument("--out_root", default=None,
                    help="Write depth/ under this directory instead of the scene's "
                         "iphone/ dir (verification runs).")
    ap.add_argument("--force", action="store_true", help="Re-extract even if output PNG exists.")
    ap.add_argument("--shard", default="0/1", help="Shard as i/N to process scenes[i::N].")
    ap.add_argument("--workers", type=int, default=16, help="Parallel frame writers per scene.")
    args = ap.parse_args()

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(resolve_data_root()) / "scannetpp" / "data"
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted([d.name for d in data_root.iterdir() if (d / "iphone" / "depth.bin").exists()])
        i, n = (int(x) for x in args.shard.split("/"))
        scenes = scenes[i::n]
        print(f"shard {i}/{n}: {len(scenes)} scenes")

    out_root = Path(args.out_root) if args.out_root else None

    total_w = total_s = 0
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for sid in tqdm(scenes, desc="scenes"):
            try:
                w, s = extract_scene(data_root / sid, force=args.force, pool=pool,
                                     out_root=out_root)
                total_w += w
                total_s += s
            except Exception as e:
                print(f"[{sid}] ERROR: {e}", file=sys.stderr)
    finally:
        if pool is not None:
            pool.shutdown()
    print(f"Done. wrote={total_w} skipped={total_s} scenes={len(scenes)}")


if __name__ == "__main__":
    main()
