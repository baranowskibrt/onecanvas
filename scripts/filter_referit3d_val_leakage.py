#!/usr/bin/env python3
"""Filter Nr3D and Sr3D to remove ScanRefer val scene leakage.

Produces:
    <data_root>/vlm_annotations/nr3d/nr3d_train.jsonl
    <data_root>/vlm_annotations/sr3d/sr3d_train.jsonl

Consumed by the NR3D_TRAIN / SR3D_TRAIN registry entries in
training/onecanvas/data/__init__.py, whose comment declares these filtered
variants MANDATORY for any grounding training mix.

Inputs are the raw nr3d.jsonl / sr3d.jsonl produced by
scripts/convert_grounding_datasets.py (upstream: the ReferIt3D csvs from
https://referit3d.github.io/ plus ScanNet-derived bboxes) and the
scanrefer_val.jsonl from the same converter (upstream: ScanRefer,
https://github.com/daveredrum/ScanRefer).

Raw `nr3d.jsonl` and `sr3d.jsonl` are defined over the full ScanNet train set
and predate ScanRefer's val partition. They share 130 / 116 of the 141
ScanRefer val scenes respectively, so training on the raw files leaks val
scenes into the training mix.

This script writes scene-filtered "_train" variants alongside the originals:
    nr3d_train.jsonl   = nr3d.jsonl  - scanrefer_val scenes
    sr3d_train.jsonl   = sr3d.jsonl  - scanrefer_val scenes

ScanRefer val == Multi3DRefer val (same 141 scenes), so excluding scanrefer_val
also excludes multi3drefer_val. No need to filter against both.

Items are re-serialized with json.dumps, preserving key order, so a kept line
is byte-identical to its source line (in particular, a per-item data_path key
is passed through if the source has one; the portable converters omit it).

Usage:
    python scripts/filter_referit3d_val_leakage.py
    python scripts/filter_referit3d_val_leakage.py --data-root /path/to/datasets

Idempotent -- overwrites the _train files on each run and prints stats.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def resolve_data_root(cli_value=None):
    """Mirror training/onecanvas/data/__init__.py: CLI > env > sibling datasets/."""
    if cli_value:
        return cli_value
    root = os.environ.get("ONECANVAS_DATA_ROOT")
    if root:
        return root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sibling = os.path.join(os.path.dirname(repo_root), "datasets")
    if os.path.isdir(sibling):
        return sibling
    raise SystemExit(
        "Cannot resolve dataset root: pass --data-root or export ONECANVAS_DATA_ROOT."
    )


def load_scene_ids(jsonl_path: str) -> set:
    s: set = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s.add(json.loads(line)["scene_id"])
    return s


def filter_jsonl(src: str, dst: str, exclude: set):
    """Stream src to dst, dropping items whose scene_id is in `exclude`.

    Returns (kept_items, dropped_items, kept_scenes, dropped_scenes).
    """
    kept_items = 0
    dropped_items = 0
    kept_scenes: set = set()
    dropped_scenes: set = set()
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("scene_id")
            if sid in exclude:
                dropped_items += 1
                dropped_scenes.add(sid)
                continue
            fout.write(json.dumps(d) + "\n")
            kept_items += 1
            kept_scenes.add(sid)
    return kept_items, dropped_items, len(kept_scenes), len(dropped_scenes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter Nr3D/Sr3D against ScanRefer val scenes")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (default: $ONECANVAS_DATA_ROOT or sibling datasets/)")
    parser.add_argument("--scanrefer-val", default=None,
                        help="scanrefer_val.jsonl providing the exclusion scene set "
                             "(default: <data_root>/vlm_annotations/scanrefer/scanrefer_val.jsonl)")
    parser.add_argument("--nr3d", default=None,
                        help="Raw nr3d.jsonl (default: <data_root>/vlm_annotations/nr3d/nr3d.jsonl)")
    parser.add_argument("--sr3d", default=None,
                        help="Raw sr3d.jsonl (default: <data_root>/vlm_annotations/sr3d/sr3d.jsonl)")
    parser.add_argument("--nr3d-out", default=None,
                        help="Output (default: nr3d_train.jsonl next to --nr3d)")
    parser.add_argument("--sr3d-out", default=None,
                        help="Output (default: sr3d_train.jsonl next to --sr3d)")
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    scanrefer_val = args.scanrefer_val or os.path.join(
        root, "vlm_annotations", "scanrefer", "scanrefer_val.jsonl")
    nr3d_raw = args.nr3d or os.path.join(root, "vlm_annotations", "nr3d", "nr3d.jsonl")
    sr3d_raw = args.sr3d or os.path.join(root, "vlm_annotations", "sr3d", "sr3d.jsonl")
    nr3d_out = args.nr3d_out or os.path.join(os.path.dirname(nr3d_raw), "nr3d_train.jsonl")
    sr3d_out = args.sr3d_out or os.path.join(os.path.dirname(sr3d_raw), "sr3d_train.jsonl")

    for p in (scanrefer_val, nr3d_raw, sr3d_raw):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing {p}")

    val_scenes = load_scene_ids(scanrefer_val)
    print(f"scanrefer_val scenes to exclude: {len(val_scenes)}")
    print()

    for src, dst, name in (
        (nr3d_raw, nr3d_out, "nr3d"),
        (sr3d_raw, sr3d_out, "sr3d"),
    ):
        kept_items, dropped_items, kept_scenes, dropped_scenes = filter_jsonl(
            src, dst, val_scenes
        )
        total_items = kept_items + dropped_items
        print(f"{name}:")
        print(f"  src: {src}")
        print(f"  dst: {dst}")
        print(
            f"  items: kept {kept_items} / {total_items} "
            f"({100 * kept_items / total_items:.1f}%), dropped {dropped_items}"
        )
        print(f"  scenes: kept {kept_scenes}, leaked-out {dropped_scenes}")
        print()

    print("Done. The NR3D_TRAIN / SR3D_TRAIN registry entries consume the _train files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
