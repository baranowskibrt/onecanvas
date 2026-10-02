#!/usr/bin/env python
"""Download ARKitScenes assets into the tree the OneCanvas dataloader expects.

Everything lands under ``<data_root>/arkitscenes/raw/{Training,Validation}/<video_id>/``
(the "arkitscenes" root in training/onecanvas/data/__init__.py:65). Three
subcommands, all wrapping the official ARKitScenes downloader
(``download_data.py`` from https://github.com/apple/ARKitScenes, pass its
clone via --arkitscenes-repo):

  lowres   Downloads the raw lowres assets:
             lowres_wide/             RGB PNGs (image source,
                                      data_processor_3d.py:518)
             lowres_depth/            uint16 mm depth PNGs
                                      (data_processor_3d.py:842)
             lowres_wide_intrinsics/  .pincam files (data_processor_3d.py:841)
             lowres_wide.traj         GT trajectory (data_processor_3d.py:840)

  vga      Downloads vga_wide/ + vga_wide_intrinsics/, then trims each scene
           to at most --max-frames evenly spaced frames (numpy.linspace over
           the sorted frame list; unkept frames and their .pincam files are
           deleted) and resizes the kept PNGs IN PLACE to --resize (default
           320x240, PIL LANCZOS). This reproduces the on-disk ``vga_wide``
           convention of the original tree (320x240 despite the name; see the
           comment at data_processor_3d.py:1085). PNG re-encoding is
           pixel-deterministic for a given PIL version but not guaranteed
           byte-identical across PIL versions.

  vga640   Downloads vga_wide at its NATIVE 640x480 into ``vga_wide_640x480/``,
           keeping exactly the frame stems already present in the scene's
           (sampled) ``vga_wide/`` so geometry files stay aligned. This
           directory is NOT part of the standard ARKitScenes download; the
           loader matches it as f"vga_wide_{image_resolution}" at
           data_processor_3d.py:507 (the 640x480 eval resolution for
           VSI-Bench). Run ``vga`` first.

Scene list: --scene-list is a text file with one video_id per line, or a CSV
with a ``video_id`` column (optional ``fold`` column). When folds are not
given, they are resolved from --metadata-csv (default
``<data_root>/arkitscenes/raw/metadata.csv``; the ARKitScenes metadata.csv
with ``video_id`` and ``fold`` columns, obtainable per the ARKitScenes repo's
DATA.md). For VSI-Bench, the needed ids are the ``scene_name`` values with
``dataset == "arkitscenes"`` in ``vlm_annotations/vsi-bench/test.jsonl``.

License: ARKitScenes is provided by Apple for non-commercial research use
only. Downloading means you accept the ARKitScenes license and terms of use
(see LICENSE in https://github.com/apple/ARKitScenes).

Examples:
  python scripts/download_arkitscenes.py lowres \
      --scene-list vsi_arkit_ids.txt --arkitscenes-repo /path/to/ARKitScenes
  python scripts/download_arkitscenes.py vga \
      --scene-list vsi_arkit_ids.txt --arkitscenes-repo /path/to/ARKitScenes
  python scripts/download_arkitscenes.py vga640 \
      --scene-list vsi_arkit_ids.txt --arkitscenes-repo /path/to/ARKitScenes
"""

import argparse
import concurrent.futures
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Exact frame subsample of the tree the published numbers were measured on.
FRAME_MANIFEST = Path(__file__).resolve().parent / "assets" / "arkitscenes_vga_frames.json.gz"


def load_frame_manifest(path=None):
    """video_id -> the exact vga_wide frame stems the paper's tree kept.

    The `vga` subcommand keeps 256 evenly spaced frames and deletes the rest
    IN PLACE. That is destructive in a way that matters for reproduction: the
    kept set depends on how many frames the download contained, and after the
    prune that count is gone. It cannot be recovered from anything left on
    disk, and ARKitScenes' metadata.csv does not record it, so a rebuild
    cannot be checked against the original, and a rebuild against a different
    ARKitScenes revision would silently select different frames.

    This manifest pins the answer for the 150 ARKitScenes scenes VSI-Bench
    evaluates, so those reproduce exactly. Scenes not listed (the training
    pool) fall back to the linspace rule, and say so.
    """
    path = Path(path) if path else FRAME_MANIFEST
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as f:
        return json.load(f).get("scenes", {})


LOWRES_ASSETS = ["lowres_wide", "lowres_depth", "lowres_wide_intrinsics", "lowres_wide.traj"]
VGA_ASSETS = ["vga_wide", "vga_wide_intrinsics"]


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


def read_scene_list(path: Path) -> dict:
    """Return {video_id: fold_or_None} from a txt (one id per line) or CSV
    (video_id[,fold]) file."""
    scenes: dict = {}
    with open(path, newline="") as f:
        first = f.readline()
        f.seek(0)
        if "," in first and "video_id" in first:
            for row in csv.DictReader(f):
                vid = str(row["video_id"]).strip()
                if vid:
                    scenes[vid] = (row.get("fold") or "").strip() or None
        else:
            for line in f:
                vid = line.strip().split(",")[0]
                if vid and vid != "video_id":
                    scenes[vid] = None
    return scenes


def resolve_folds(scenes: dict, metadata_csv: Path) -> dict:
    """Fill in missing folds from metadata.csv; drop (and warn about) ids
    without metadata."""
    missing = [v for v, fold in scenes.items() if fold is None]
    if missing:
        if not metadata_csv.exists():
            raise SystemExit(
                f"{len(missing)} scene ids need a fold but metadata csv not found: "
                f"{metadata_csv}. Provide --metadata-csv or a scene-list CSV with "
                f"a fold column."
            )
        meta = {}
        with open(metadata_csv, newline="") as f:
            for row in csv.DictReader(f):
                meta[str(row["video_id"]).strip()] = row["fold"].strip()
        no_meta = [v for v in missing if v not in meta]
        if no_meta:
            print(f"WARNING: {len(no_meta)} scene ids not in metadata, skipped: "
                  f"{sorted(no_meta)[:5]}...")
        scenes = {v: (fold if fold else meta.get(v))
                  for v, fold in scenes.items() if fold or v in meta}
    bad = {fold for fold in scenes.values() if fold not in ("Training", "Validation")}
    if bad:
        raise SystemExit(f"Unexpected fold values: {bad}")
    return scenes


def run_official_downloader(arkit_repo: Path, video_id: str, fold: str,
                            download_dir: Path, assets: list) -> bool:
    """Invoke the official ARKitScenes download_data.py for one scene.
    It places files at {download_dir}/raw/{fold}/{video_id}/."""
    with tempfile.TemporaryDirectory(dir=download_dir) as td:
        tmp_csv = Path(td) / f"{video_id}.csv"
        with open(tmp_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "fold"])
            w.writerow([video_id, fold])
        cmd = [
            sys.executable, str(arkit_repo / "download_data.py"),
            "raw",
            "--video_id_csv", str(tmp_csv),
            "--download_dir", str(download_dir),
            "--raw_dataset_assets", *assets,
        ]
        result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"[{video_id}] FAILED (rc={result.returncode}):\n{result.stderr[-400:]}")
        return False
    return True


def remove_zips(scene_dir: Path, video_id: str):
    """Delete any leftover zip files under a scene directory."""
    removed = []
    for z in scene_dir.rglob("*.zip"):
        z.unlink()
        removed.append(z.name)
    if removed:
        print(f"  [{video_id}] Removed zips: {removed}")


def resize_image_inplace(img_path: Path, target_size):
    img = Image.open(img_path)
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
        img.save(img_path)


def sample_scene(scene_dir: Path, video_id: str, max_frames: int, target_size,
                 manifest=None):
    """Trim vga_wide (and matching intrinsics) to the paper's frame subset, then
    resize kept frames to target_size in place.

    Prefers the shipped manifest so the selection matches the published tree
    exactly. Falls back to max_frames evenly spaced frames, the rule that
    produced the manifest in the first place but whose result depends on the
    download's frame count.
    """
    img_dir = scene_dir / "vga_wide"
    intr_dir = scene_dir / "vga_wide_intrinsics"
    images = sorted(img_dir.glob("*.png")) if img_dir.exists() else []
    if not images:
        return

    pinned = (manifest or {}).get(video_id)
    if pinned:
        want = {f"{video_id}_{t}" for t in pinned}
        keep_set = {img for img in images if img.stem in want}
        missing = want - {img.stem for img in images}
        if missing:
            # Pruning now would bake in a subset that is neither the manifest's
            # nor the linspace rule's, and nothing downstream could tell.
            raise SystemExit(
                f"[{video_id}] download is missing {len(missing)} of "
                f"{len(want)} frames the published tree used (e.g. "
                f"{sorted(missing)[:3]}). Refusing to prune: the ARKitScenes "
                f"revision on disk differs from the one the paper used."
            )
    else:
        indices = np.linspace(0, len(images) - 1, max_frames, dtype=int)
        keep_set = {images[i] for i in indices}
        print(f"  [{video_id}] not in the frame manifest, falling back to "
              f"{max_frames} evenly spaced frames (selection is not pinned)")
    removed = 0
    for img in images:
        if img not in keep_set:
            img.unlink(missing_ok=True)
            (intr_dir / (img.stem + ".pincam")).unlink(missing_ok=True)
            removed += 1
        else:
            resize_image_inplace(img, target_size)
    print(f"  [{video_id}] Sampled {len(keep_set)}/{len(images)} frames "
          f"(removed {removed}), resized to {target_size}.")


# ---------------------------------------------------------------- subcommands

def cmd_lowres(args, scenes, raw_root):
    def _one(item):
        vid, fold = item
        scene_dir = raw_root / fold / vid
        if all((scene_dir / a).exists() for a in LOWRES_ASSETS):
            print(f"  [{vid}] already complete")
            return True
        if args.dry_run:
            print(f"  [{vid}] would download {LOWRES_ASSETS} -> {scene_dir}")
            return True
        ok = run_official_downloader(args.arkitscenes_repo, vid, fold,
                                     raw_root.parent, LOWRES_ASSETS)
        if ok:
            remove_zips(scene_dir, vid)
        return ok
    return _run_parallel(_one, scenes, args.workers)


def cmd_vga(args, scenes, raw_root):
    target_size = _parse_size(args.resize)
    manifest = {} if args.no_frame_manifest else load_frame_manifest(args.frame_manifest)
    if manifest:
        print(f"frame manifest: {len(manifest)} scenes pinned to the published subset")
    else:
        print("frame manifest: NOT in use, frame selection will not be pinned")

    def _one(item):
        vid, fold = item
        scene_dir = raw_root / fold / vid
        if (scene_dir / "vga_wide").exists():
            # Re-sample already-downloaded scenes (idempotent) and clean zips.
            remove_zips(scene_dir, vid)
            sample_scene(scene_dir, vid, args.max_frames, target_size, manifest)
            return True
        if args.dry_run:
            print(f"  [{vid}] would download {VGA_ASSETS} -> {scene_dir}")
            return True
        print(f"[{vid}] Downloading...")
        ok = run_official_downloader(args.arkitscenes_repo, vid, fold,
                                     raw_root.parent, VGA_ASSETS)
        if not ok:
            return False
        remove_zips(scene_dir, vid)
        sample_scene(scene_dir, vid, args.max_frames, target_size, manifest)
        return True
    return _run_parallel(_one, scenes, args.workers)


def cmd_vga640(args, scenes, raw_root):
    def _one(item):
        vid, fold = item
        scene_dir = raw_root / fold / vid
        src = scene_dir / "vga_wide"
        dst = scene_dir / "vga_wide_640x480"

        if not src.exists():
            print(f"  [{vid}] SKIP: vga_wide/ missing (run the vga subcommand first)")
            return True
        want_stems = {f.stem for f in src.glob("*.png")}
        if not want_stems:
            print(f"  [{vid}] SKIP: vga_wide/ empty")
            return True
        if dst.exists():
            have = {f.stem for f in dst.glob("*.png")}
            if have == want_stems:
                print(f"  [{vid}] already complete ({len(have)} frames)")
                return True
        if args.dry_run:
            print(f"  [{vid}] would download {len(want_stems)} frames -> {dst}")
            return True

        # Per-scene staging dir so parallel workers don't collide on raw/.
        staging = raw_root.parent / f"_staging_{vid}"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            ok = run_official_downloader(args.arkitscenes_repo, vid, fold,
                                         staging, ["vga_wide"])
            if not ok:
                return False
            dl_dir = staging / "raw" / fold / vid / "vga_wide"
            if not dl_dir.exists():
                print(f"  [{vid}] FAILED: expected {dl_dir} after download")
                return False
            # Delete unwanted frames in place, then rename to final location.
            for f in list(dl_dir.glob("*.png")):
                if f.stem not in want_stems:
                    f.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            dl_dir.rename(dst)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        kept = len(list(dst.glob("*.png")))
        print(f"  [{vid}] done: {kept}/{len(want_stems)} frames -> {dst}")
        return kept > 0
    return _run_parallel(_one, scenes, args.workers)


def _parse_size(text):
    try:
        w, h = (int(x) for x in text.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--resize must be WxH, got {text!r}")
    return (w, h)


def _run_parallel(fn, scenes, workers):
    items = sorted(scenes.items())
    ok = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item[0] for item in items}
        for fut in concurrent.futures.as_completed(futures):
            vid = futures[fut]
            try:
                if fut.result():
                    ok += 1
                else:
                    fail += 1
            except Exception as exc:
                print(f"[{vid}] Exception: {exc}")
                fail += 1
    print(f"\nDone: {ok} ok, {fail} failed")
    return fail == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["lowres", "vga", "vga640"],
                        help="Which asset set to download (see module docstring).")
    parser.add_argument("--scene-list", required=True, type=Path,
                        help="Text file with one video_id per line, or CSV with "
                             "video_id[,fold] columns.")
    parser.add_argument("--arkitscenes-repo", required=True, type=Path,
                        help="Path to a clone of https://github.com/apple/ARKitScenes "
                             "(its download_data.py is invoked per scene).")
    parser.add_argument("--data_root", default=None,
                        help="Dataset root (default: ONECANVAS_DATA_ROOT, else the "
                             "datasets/ directory sibling to the repo).")
    parser.add_argument("--metadata-csv", type=Path, default=None,
                        help="ARKitScenes metadata.csv for fold resolution "
                             "(default: <data_root>/arkitscenes/raw/metadata.csv).")
    parser.add_argument("--max-frames", type=int, default=256,
                        help="vga: max frames kept per scene after download (evenly "
                             "spaced; default 256, the original tree's convention).")
    parser.add_argument("--frame-manifest", default=None,
                        help="vga: path to a frame manifest pinning the kept "
                             "vga_wide stems per scene (default: the one shipped "
                             "in scripts/assets/).")
    parser.add_argument("--no-frame-manifest", action="store_true",
                        help="vga: ignore the manifest and select frames with the "
                             "linspace rule. The result then depends on the "
                             "download's frame count and may not match the tree "
                             "the published numbers were measured on.")
    parser.add_argument("--resize", default="320x240",
                        help="vga: in-place resize target WxH for kept frames "
                             "(default 320x240, the original tree's convention).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent scene downloads.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.arkitscenes_repo / "download_data.py").exists():
        raise SystemExit(
            f"download_data.py not found in {args.arkitscenes_repo}. "
            f"Clone https://github.com/apple/ARKitScenes there first."
        )

    data_root = resolve_data_root(args.data_root)
    raw_root = Path(data_root) / "arkitscenes" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    metadata_csv = args.metadata_csv or (raw_root / "metadata.csv")

    scenes = read_scene_list(args.scene_list)
    print(f"Scene list: {len(scenes)} video ids")
    scenes = resolve_folds(scenes, metadata_csv)
    folds = {}
    for fold in scenes.values():
        folds[fold] = folds.get(fold, 0) + 1
    print(f"Resolved folds: {folds}")

    handler = {"lowres": cmd_lowres, "vga": cmd_vga, "vga640": cmd_vga640}[args.command]
    success = handler(args, scenes, raw_root)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
