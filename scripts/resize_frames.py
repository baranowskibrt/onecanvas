#!/usr/bin/env python
"""Build the resized frame directories the OneCanvas dataloader looks for.

For every scene under the dataset-specific root, resizes each frame with
PIL LANCZOS and writes it next to the source directory using the exact
naming convention ``_resolve_image_sources`` matches
(training/onecanvas/data/data_processor_3d.py:501-529):

  --dataset scannet          <data_root>/scannet/scannet_preprocessed/<scene>/
                             color/          -> color_<W>x<H>/          (*.jpg)
  --dataset scannetpp_iphone <data_root>/scannetpp/data/<scene>/iphone/
                             rgb/            -> rgb_<W>x<H>/            (*.jpg)
  --dataset scannetpp_dslr   <data_root>/scannetpp/data/<scene>/dslr/
                             resized_undistorted_images/
                               -> resized_undistorted_images_<W>x<H>/   (*.JPG)

Source frames come from scripts/download_extract_scannet.py (ScanNet color/),
scripts/extract_scannetpp_iphone_rgb.py (iPhone rgb/ and rgb_640x480/), and the
official ScanNet++ toolkit's undistortion step (DSLR
resized_undistorted_images/).

Paper conventions: training reads 320x240 everywhere, eval reads 640x480 on
VSI-Bench and SPBench, so build both sizes for scenes those benchmarks use.

THE RESIZE CHAIN IS NOT THE SAME FOR EVERY DATASET, and the difference is not
cosmetic. Verified byte-for-byte against the tree the paper numbers were
measured on (see RESIZE_SOURCE below):

  ScanNet           color/ -> color_640x480/   and   color/ -> color_320x240/
                    Both resized from the native frame. 15/15 and 30/30
                    byte-identical.
  ScanNet++ iPhone  rgb_640x480/ -> rgb_320x240/
                    320x240 is resized from the 640x480 frame, NOT from rgb/.
                    80/80 byte-identical over 8 scenes. Resizing it from rgb/
                    instead reproduces nothing (mean pixel delta 2.1).
                    rgb_640x480/ itself is NOT built here at all: it is written
                    by extract_scannetpp_iphone_rgb.py off the decoded video
                    frame, because it never passed through rgb/'s JPEG encode.

Output filenames and image content match the original preprocessing
(same PIL resize + save call, default JPEG quality). JPEG re-encoding is
pixel-deterministic for a given PIL version but not guaranteed
byte-identical across PIL/libjpeg versions.

Example:
  python scripts/resize_frames.py --dataset scannet --size 320x240
  python scripts/resize_frames.py --dataset scannetpp_iphone --size 640x480 \
      --root /path/to/datasets/scannetpp/data
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

NUM_WORKERS = 10

# dataset -> (root relative to data_root, scene subdir or None, native frame dirname)
DATASETS = {
    "scannet": ("scannet/scannet_preprocessed", None, "color"),
    "scannetpp_iphone": ("scannetpp/data", "iphone", "rgb"),
    "scannetpp_dslr": ("scannetpp/data", "dslr", "resized_undistorted_images"),
}

# (dataset, target size) -> directory to resize FROM, when it is not the native
# frame directory. Output naming always follows the native dirname, so a chained
# build still writes rgb_320x240/ and never rgb_640x480_320x240/.
RESIZE_SOURCE = {
    ("scannetpp_iphone", (320, 240)): "rgb_640x480",
}

# Sizes this script must refuse to build because the original tree did not build
# them here. rgb_640x480/ comes off the decoded video frame in
# extract_scannetpp_iphone_rgb.py and cannot be reproduced from rgb/ by any
# resize: rgb/ is a lossy JPEG of that same frame, so a rebuild from it carries
# the encode error (best case mean pixel delta 1.1, byte-identical never).
NOT_BUILT_HERE = {
    ("scannetpp_iphone", (640, 480)): (
        "rgb_640x480/ is written by scripts/extract_scannetpp_iphone_rgb.py "
        "directly from the decoded rgb.mp4 frame, in the same pass that writes "
        "rgb/. Rebuilding it from rgb/ here would silently produce frames that "
        "do not match the ones the published numbers were measured on. Run the "
        "extraction script instead."
    ),
}


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
        "--data_root / --root."
    )


def resize_image(img_path: Path, out_path: Path, target_size) -> str:
    if out_path.exists():
        return f"Skipped {out_path.name} (already exists)"
    with Image.open(img_path) as img:
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        resized_img.save(out_path)
    return f"Resized {img_path}"


def process_scene(scene_dir: Path, src_name: str, out_base: str, target_size,
                  quiet: bool) -> None:
    src_dir = scene_dir / src_name
    if not src_dir.exists():
        # A chained build whose source is missing means the user skipped a step,
        # which is not the same thing as a scene that simply is not here. Saying
        # so beats writing nothing and reporting success.
        if src_name != out_base and (scene_dir / out_base).is_dir():
            raise SystemExit(
                f"{scene_dir}: need {src_name}/ to build "
                f"{out_base}_{target_size[0]}x{target_size[1]}/, but it does not "
                f"exist. Build that size first."
            )
        return

    images = sorted([f for f in src_dir.iterdir() if f.is_file()])
    if not images:
        return

    out_dir = scene_dir / f"{out_base}_{target_size[0]}x{target_size[1]}"
    out_dir.mkdir(exist_ok=True)

    print(f"Processing {scene_dir} ({len(images)} images)...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(resize_image, img, out_dir / img.name, target_size): img
            for img in images
        }
        for future in as_completed(futures):
            result = future.result()
            if result and not quiet:
                print(f"  {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                        help="Which preprocessed tree to resize.")
    parser.add_argument("--size", required=True,
                        help="Target size as WxH, e.g. 320x240 or 640x480.")
    parser.add_argument("--data_root", default=None,
                        help="Dataset root (default: ONECANVAS_DATA_ROOT, else the "
                             "datasets/ directory sibling to the repo).")
    parser.add_argument("--root", default=None,
                        help="Override the scene-tree root directly (e.g. a scratch "
                             "copy); bypasses --data_root resolution.")
    parser.add_argument("--scene", default=None,
                        help="Process a single scene id (default: all scenes under the root).")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print per-scene progress, not per-frame lines.")
    args = parser.parse_args()

    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size must be WxH, got {args.size!r}")
    target_size = (w, h)

    rel_root, scene_subdir, out_base = DATASETS[args.dataset]
    blocked = NOT_BUILT_HERE.get((args.dataset, target_size))
    if blocked:
        raise SystemExit(f"Refusing to build {args.dataset} @ {args.size}: {blocked}")
    src_name = RESIZE_SOURCE.get((args.dataset, target_size), out_base)
    if args.root:
        root_dir = Path(args.root)
    else:
        root_dir = Path(resolve_data_root(args.data_root)) / rel_root
    if not root_dir.is_dir():
        raise SystemExit(f"Scene root does not exist: {root_dir}")

    if args.scene:
        scene_ids = [args.scene]
    else:
        scene_ids = sorted(d.name for d in root_dir.iterdir() if d.is_dir())

    for scene_id in scene_ids:
        scene_dir = root_dir / scene_id
        if scene_subdir:
            scene_dir = scene_dir / scene_subdir
        if not scene_dir.is_dir():
            continue
        process_scene(scene_dir, src_name, out_base, target_size, args.quiet)


if __name__ == "__main__":
    main()
