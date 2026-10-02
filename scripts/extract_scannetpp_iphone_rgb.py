#!/usr/bin/env python
"""Extract ScanNet++ iPhone rgb video -> subsampled per-frame JPEGs.

Produces ``<data_root>/scannetpp/data/<scene>/iphone/rgb/frame_NNNNNN.jpg``
at the video's native resolution (1920x1440), where NNNNNN is the frame's
ORIGINAL index in the video. The OneCanvas dataloader finds this directory
via ``_resolve_image_sources`` (training/onecanvas/data/data_processor_3d.py:518,
"rgb"; the resized rgb_<W>x<H>/ variants at :506 come from
scripts/resize_frames.py). Stem alignment is what matters downstream:
``pose_intrinsic_imu.json`` (consumed raw at data_processor_3d.py:962-998)
has one entry per original frame index ("frame_000000", ...), and the
loader keeps only pose entries whose stem also exists in rgb/ and depth/
(data_processor_3d.py:1004-1009), so filenames must keep original indices.

Upstream source: the official ScanNet++ dataset download
(https://kaldir.vc.in.tum.de/scannetpp/ after accepting the ScanNet++ terms
of use), asset ``iphone/rgb.<ext>`` per scene (plus pose_intrinsic_imu.json).

CONTAINER: ScanNet++ used to ship ``rgb.mp4`` and now ships ``rgb.mkv``. The
tree the paper's numbers were measured on was built from the .mp4 era. Both
names are accepted here, in VIDEO_NAMES order. This matters because the old
code matched ``rgb.mp4`` only, so against a current download it found zero
scenes, printed "0 scenes with iphone/rgb.mp4", wrote nothing and exited 0.
The switch is a repackaging, not a re-capture: pose_intrinsic_imu.json entry
counts are unchanged (verified on 8 scenes), so the frame selection below picks
the same indices from either container.

Frame selection (default): ``numpy.linspace(0, n_frames - 1, 500, dtype=int)``
- 500 frames evenly spaced over the whole video. This is the rule the
original preprocessed tree follows (verified against it; consecutive kept
indices differ by n_frames/499, e.g. 0, 20, 40, 60, 81, ... for a
10110-frame video - NOT a fixed stride). ``n_frames`` is taken from the
scene's pose_intrinsic_imu.json entry count when present (authoritative,
one entry per frame), else from cv2's frame count. Pass ``--stride N`` to
keep every N-th frame instead (fixed stride, not the original convention).

Frames are decoded sequentially with cv2.VideoCapture (no per-frame seeking,
which is unreliable and slow on long videos) and saved with JPEG quality 95.

Outputs ARE byte-identical to the published tree. Verified 2026-08-25 by
downloading 8 scenes fresh from ScanNet++ and rebuilding: rgb/, rgb_640x480/
and rgb_320x240/ all came out 4000/4000 byte-identical. (An earlier version of
this docstring claimed the opposite, that re-encoding could not reproduce the
original bytes. That was a guess, and it was wrong: the decode is
standard-defined and cv2's JPEG encoder path is stable here.)

This ALSO writes ``iphone/rgb_640x480/`` in the same pass, from the same
decoded frame. That is not a convenience. The original tree's rgb_640x480/
was produced here rather than by a later resize of rgb/, and it has to be,
because rgb/ is a lossy JPEG of the same frame: resizing rgb/ instead carries
that encode error into every downsampled pixel and never reproduces the
original bytes, best case mean pixel delta 1.1. This was first inferred without
a video to check against, from two clues: in the published tree rgb/ and
rgb_640x480/ frame files are written 24 ms apart by one process, and a phase
search over the exact 3x downscale picks out (dy=1, dx=1), the block centre,
which is precisely cv2.resize's sample point at an integer ratio. A fresh
download later confirmed it outright, 4000/4000 byte-identical.

At 1920x1440 -> 640x480 the ratio is exactly 3, so cv2.resize with
INTER_LINEAR degenerates to point sampling. That is deliberate here. It looks
worse than an antialiased downscale and it is the operation the published
numbers were measured on, so it is what the release reproduces. Do not
"improve" it to INTER_AREA or PIL LANCZOS without re-running the ScanNet++
share of the paper evals.

The 320x240 size is NOT built here; scripts/resize_frames.py chains it off
rgb_640x480/ (verified 80/80 byte-identical).

Example:
  python scripts/extract_scannetpp_iphone_rgb.py --scene 00777c41d4
  python scripts/extract_scannetpp_iphone_rgb.py            # all scenes
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

JPEG_QUALITY = 95
DEFAULT_NUM_FRAMES = 500
# Accepted container filenames, in preference order. .mkv is what ScanNet++
# serves now; .mp4 is what the published tree was built from.
VIDEO_NAMES = ("rgb.mkv", "rgb.mp4")
# Written alongside the native frame, from the same decoded array. See the
# module docstring for why this cannot be a later resize of rgb/.
RESIZED_OUTPUT = (640, 480)


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


def scene_frame_count(scene_dir: Path, cap: cv2.VideoCapture) -> int:
    """Total original frame count: pose_intrinsic_imu.json entry count when
    available (one entry per original frame), else cv2's estimate."""
    json_path = scene_dir / "iphone" / "pose_intrinsic_imu.json"
    if json_path.exists():
        with open(json_path) as f:
            return len(json.load(f))
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def select_frames(n_frames: int, num_frames: int, stride) -> set:
    if stride:
        return set(range(0, n_frames, stride))
    return set(np.linspace(0, n_frames - 1, num_frames, dtype=int).tolist())


def find_video(scene_dir: Path):
    """The scene's iPhone RGB video, whichever container it shipped in."""
    for name in VIDEO_NAMES:
        p = scene_dir / "iphone" / name
        if p.exists():
            return p
    return None


def extract_scene(scene_dir: Path, num_frames: int, stride,
                  out_root: Path | None = None, force: bool = False) -> tuple[int, int]:
    video_path = find_video(scene_dir)
    if video_path is None:
        return 0, 0
    base_dir = out_root or scene_dir / "iphone"
    out_dir = base_dir / "rgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    rw, rh = RESIZED_OUTPUT
    resized_dir = base_dir / f"rgb_{rw}x{rh}"
    resized_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    try:
        n_frames = scene_frame_count(scene_dir, cap)
        keep = select_frames(n_frames, num_frames, stride)

        written = skipped = 0
        # Sequential decode; do NOT seek per frame (unreliable on long videos).
        for idx in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                print(f"  WARNING: {video_path} ended at frame {idx} "
                      f"(expected {n_frames})", file=sys.stderr)
                break
            if idx not in keep:
                continue
            name = f"frame_{idx:06d}.jpg"
            out_path = out_dir / name
            resized_path = resized_dir / name
            if out_path.exists() and resized_path.exists() and not force:
                skipped += 1
                continue
            if force or not out_path.exists():
                cv2.imwrite(str(out_path), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if force or not resized_path.exists():
                # From `frame`, the decoded array, never from out_path.
                cv2.imwrite(str(resized_path),
                            cv2.resize(frame, RESIZED_OUTPUT,
                                       interpolation=cv2.INTER_LINEAR),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            written += 1
        return written, skipped
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_root", default=None,
                    help="ScanNet++ scene root, i.e. <dataset_root>/scannetpp/data "
                         "(default: derived from ONECANVAS_DATA_ROOT or the datasets/ "
                         "directory sibling to the repo).")
    ap.add_argument("--scene", default=None, help="Single scene id (otherwise all).")
    ap.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                    help="Number of evenly spaced frames to keep per scene "
                         "(linspace over original indices; default 500, the "
                         "convention of the original preprocessed tree).")
    ap.add_argument("--stride", type=int, default=None,
                    help="Keep every N-th original frame instead of the evenly "
                         "spaced default. Overrides --num-frames.")
    ap.add_argument("--out_root", default=None,
                    help="Write rgb/ and rgb_640x480/ under this directory instead "
                         "of the scene's iphone/ dir (verification runs).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output JPEGs.")
    args = ap.parse_args()

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(resolve_data_root()) / "scannetpp" / "data"

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted(d.name for d in data_root.iterdir()
                        if find_video(d) is not None)
        if not scenes:
            # Exiting 0 here is how the .mp4 -> .mkv rename went unnoticed:
            # "0 scenes", nothing written, success.
            raise SystemExit(
                f"No scene under {data_root} has an iPhone RGB video "
                f"({' or '.join(VIDEO_NAMES)}). Nothing to extract. Check the "
                f"download completed and that --data_root points at "
                f"<dataset_root>/scannetpp/data."
            )
        print(f"{len(scenes)} scenes with an iPhone RGB video "
              f"({' / '.join(VIDEO_NAMES)})")

    out_root = Path(args.out_root) if args.out_root else None
    total_w = total_s = 0
    for sid in tqdm(scenes, desc="scenes"):
        try:
            w, s = extract_scene(data_root / sid, args.num_frames, args.stride,
                                 out_root=out_root, force=args.force)
            total_w += w
            total_s += s
        except Exception as e:
            print(f"[{sid}] ERROR: {e}", file=sys.stderr)
    print(f"Done. wrote={total_w} skipped={total_s} scenes={len(scenes)}")


if __name__ == "__main__":
    main()
