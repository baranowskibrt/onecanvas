#!/usr/bin/env python3
"""Convert Multi3DRefer annotations to project JSONL format (standalone variant).

Multi3DRefer (Zhang et al., CVPR 2023) extends ScanRefer with multi-object
referring expressions. Each description may refer to zero, one, or multiple
objects in a ScanNet scene.

NOTE ON PROVENANCE: the multi3drefer_{train,val}.jsonl files actually shipped
under <data_root>/vlm_annotations/multi3drefer/ were produced by
scripts/convert_grounding_datasets.py (referring templates, '; ' separator,
source/ann_id/object_ids/eval_type metadata, bboxes from the ScanNet
mesh-derived cache). This script is the standalone ALTERNATIVE converter: it
works from pre-extracted *_aligned_bbox.npy files (or a ScanRefer-style bbox
JSON) without downloading ScanNet meshes, and emits a slightly different
schema (explicit instruction prompt, ' | ' separator, question_type
grounding_{ZT,ST,MT}). Use convert_grounding_datasets.py to regenerate the
shipped files; use this one when you only have VoteNet-style instance bbox
dumps.

Output goes to <data_root>/vlm_annotations/multi3drefer/ by default; the
MULTI3DREFER / MULTI3DREFER_TRAIN registry entries in
training/onecanvas/data/__init__.py consume multi3drefer_{val,train}.jsonl
from that directory.

Upstream sources:
    1. Multi3DRefer annotations:
       git clone https://github.com/3dlg-hcvc/M3DRef-CLIP
       Annotations in: M3DRef-CLIP/data/multi3drefer/
       Files: multi3drefer_{train,val}.json
       (default location here: <data_root>/vlm_annotations/multi3drefer/)

    2. ScanNet axis-aligned bounding boxes (one of):
       a) Raw ScanNet scans directory containing scene*/scene*_aligned_bbox.npy
       b) Pre-extracted: scannet_instance_data/ with scene*_aligned_bbox.npy
       c) ScanRefer-style JSON with object bounding boxes

Output JSONL format (one per line):
    {
        "scene_id": "scene0000_00",
        "question": "<description>\\nLocate all matching objects. ...",
        "answers": ["(cx, cy, cz, w, h, d) | (cx, cy, cz, w, h, d)"],
        "question_type": "grounding_MT",
        "eval_type": "MT-MC"
    }

Bounding boxes are bare metric float tuples in the ScanNet axis-aligned frame
(uncentered) -- same convention as scanrefer/nr3d/sr3d. The data processor
subtracts the per-sample camera-mean at load time so the bbox is in the same
panorama-centered frame as the features.

data_path: NO per-item "data_path" key is written by default (embedding an
absolute machine-local path breaks portability; the loader falls back to the
registry-configured root only when the item has NO data_path key). Pass an
explicit --data-path to embed one anyway.

Usage:
    python scripts/prepare_multi3drefer.py --scannet-data /path/to/scannet_instance_data/
    python scripts/prepare_multi3drefer.py \\
        --multi3drefer-dir /path/to/multi3drefer/ \\
        --scannet-data /path/to/scannet/scans/ \\
        --output /path/to/vlm_annotations/multi3drefer/ \\
        --splits val train
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np


PROMPT_TEMPLATE = (
    "{description}\n"
    "Locate all matching objects in the scene. "
    "Provide each 3D bounding box as (cx, cy, cz, w, h, d), separated by ' | '. "
    "If no objects match, respond 'none'."
)


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


def load_multi3drefer_annotations(json_path):
    """Load Multi3DRefer annotations from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} annotations from {json_path}")
    return data


def find_bbox_file(scannet_data, scene_id):
    """Find axis-aligned bbox file for a scene. Supports multiple directory layouts."""
    candidates = [
        # Raw ScanNet: scans/scene0000_00/scene0000_00_aligned_bbox.npy
        os.path.join(scannet_data, scene_id, f"{scene_id}_aligned_bbox.npy"),
        # Pre-extracted: scannet_instance_data/scene0000_00_aligned_bbox.npy
        os.path.join(scannet_data, f"{scene_id}_aligned_bbox.npy"),
        # Alternative naming
        os.path.join(scannet_data, scene_id, "aligned_bbox.npy"),
        # Flat directory
        os.path.join(scannet_data, f"{scene_id}_bbox.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_scene_bboxes(bbox_path):
    """Load axis-aligned bounding boxes from a .npy file.

    Standard format: [N, 7] with columns (cx, cy, cz, dx, dy, dz, label).
    Some files have [N, 8+] with extra columns (e.g., object_id).

    Returns dict mapping object_index (0-based) to (cx, cy, cz, dx, dy, dz).
    """
    bboxes = np.load(bbox_path)
    result = {}
    for i in range(len(bboxes)):
        result[i] = tuple(float(v) for v in bboxes[i, :6])
    return result


def load_scanrefer_bboxes(scanrefer_json):
    """Load bounding boxes from ScanRefer-format JSON (fallback).

    Returns nested dict: {scene_id: {object_id: (cx, cy, cz, dx, dy, dz)}}.
    """
    with open(scanrefer_json) as f:
        data = json.load(f)
    bboxes = defaultdict(dict)
    for item in data:
        sid = item["scene_id"]
        oid = int(item["object_id"])
        if "bbox" in item:
            bb = item["bbox"]
            bboxes[sid][oid] = tuple(float(v) for v in bb[:6])
    return dict(bboxes)


def format_bbox(bbox_tuple):
    """Format a bounding box as (cx, cy, cz, w, h, d) string."""
    cx, cy, cz, dx, dy, dz = bbox_tuple
    return f"({cx:.2f}, {cy:.2f}, {cz:.2f}, {dx:.2f}, {dy:.2f}, {dz:.2f})"


def build_bbox_index(scannet_data):
    """Build a {scene_id: {object_idx: bbox}} index from all available bbox files."""
    index = {}
    if os.path.isdir(scannet_data):
        for entry in os.listdir(scannet_data):
            scene_id = entry.replace("_aligned_bbox.npy", "").replace("_bbox.npy", "")
            bbox_file = find_bbox_file(scannet_data, scene_id)
            if bbox_file:
                index[scene_id] = load_scene_bboxes(bbox_file)
    return index


def convert_annotations(annotations, bbox_index, data_path, eval_types=None):
    """Convert Multi3DRefer annotations to project JSONL format.

    Args:
        annotations: List of Multi3DRefer annotation dicts.
        bbox_index: {scene_id: {object_id: (cx, cy, cz, dx, dy, dz)}}.
        data_path: Per-item data_path to embed, or None to omit the key.
        eval_types: If set, only include annotations of these eval types.

    Returns:
        List of converted annotation dicts, stats dict.
    """
    converted = []
    stats = Counter()

    for ann in annotations:
        scene_id = ann["scene_id"]
        description = ann["description"]
        object_ids = ann.get("object_ids", [])
        eval_type = ann.get("eval_type", "unknown")

        if eval_types and eval_type not in eval_types:
            continue

        stats["total"] += 1

        scene_bboxes = bbox_index.get(scene_id, {})
        if not scene_bboxes and object_ids:
            stats["missing_scene_bbox"] += 1
            continue

        resolved_bboxes = []
        missing_obj = False
        for oid in object_ids:
            if oid in scene_bboxes:
                resolved_bboxes.append(scene_bboxes[oid])
            else:
                missing_obj = True
                stats["missing_object_bbox"] += 1
                break

        if missing_obj:
            continue

        if len(resolved_bboxes) == 0:
            answer_text = "none"
        else:
            answer_text = " | ".join(format_bbox(bb) for bb in resolved_bboxes)

        if eval_type.startswith("ZT"):
            question_type = "grounding_ZT"
        elif eval_type.startswith("ST"):
            question_type = "grounding_ST"
        elif eval_type.startswith("MT"):
            question_type = "grounding_MT"
        else:
            question_type = "grounding"

        item = {
            "scene_id": scene_id,
            "question": PROMPT_TEMPLATE.format(description=description),
            "answers": [answer_text],
            "question_type": question_type,
            "eval_type": eval_type,
        }
        if data_path:
            item["data_path"] = data_path
        converted.append(item)

        stats[eval_type] += 1
        stats["converted"] += 1

    return converted, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Convert Multi3DRefer to project JSONL")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (default: $ONECANVAS_DATA_ROOT or sibling datasets/)")
    parser.add_argument("--multi3drefer-dir", default=None,
                        help="Directory with multi3drefer_{train,val}.json files "
                             "(default: <data_root>/vlm_annotations/multi3drefer)")
    parser.add_argument("--scannet-data", required=True,
                        help="ScanNet scans dir or pre-extracted bbox dir with *_aligned_bbox.npy")
    parser.add_argument("--scanrefer-json", default=None,
                        help="Optional: ScanRefer JSON with bbox info (fallback)")
    parser.add_argument("--output", default=None,
                        help="Output directory for JSONL files "
                             "(default: <data_root>/vlm_annotations/multi3drefer)")
    parser.add_argument("--splits", nargs="+", default=["val"],
                        help="Splits to convert (default: val)")
    parser.add_argument("--data-path", default=None,
                        help="If set, embed this per-item 'data_path' like the "
                             "historical files did. Default: omit the key so the "
                             "loader falls back to the configured data root.")
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    m3dr_dir = args.multi3drefer_dir or os.path.join(root, "vlm_annotations", "multi3drefer")
    out_dir = args.output or m3dr_dir

    os.makedirs(out_dir, exist_ok=True)

    print(f"Building bbox index from {args.scannet_data} ...")
    bbox_index = build_bbox_index(args.scannet_data)
    print(f"  Found bbox data for {len(bbox_index)} scenes")

    if args.scanrefer_json and os.path.exists(args.scanrefer_json):
        print(f"Loading ScanRefer bboxes as fallback from {args.scanrefer_json} ...")
        sr_bboxes = load_scanrefer_bboxes(args.scanrefer_json)
        for sid, objs in sr_bboxes.items():
            if sid not in bbox_index:
                bbox_index[sid] = objs
            else:
                for oid, bb in objs.items():
                    if oid not in bbox_index[sid]:
                        bbox_index[sid][oid] = bb
        print(f"  Total scenes with bbox data: {len(bbox_index)}")

    if not bbox_index:
        print("\nERROR: No bounding box data found. You need one of:")
        print("  1. ScanNet scans/ dir with scene*/scene*_aligned_bbox.npy")
        print("  2. Pre-extracted dir with scene*_aligned_bbox.npy files")
        print("  3. ScanRefer JSON with bbox fields (--scanrefer-json)")
        sys.exit(1)

    for split in args.splits:
        json_path = os.path.join(m3dr_dir, f"multi3drefer_{split}.json")
        if not os.path.exists(json_path):
            print(f"WARNING: {json_path} not found, skipping split '{split}'")
            continue

        annotations = load_multi3drefer_annotations(json_path)
        converted, stats = convert_annotations(annotations, bbox_index, args.data_path)

        out_path = os.path.join(out_dir, f"multi3drefer_{split}.jsonl")
        with open(out_path, "w") as f:
            for item in converted:
                f.write(json.dumps(item) + "\n")

        print(f"\n=== Split: {split} ===")
        print(f"  Total annotations: {stats.get('total', 0)}")
        print(f"  Converted: {stats.get('converted', 0)}")
        print(f"  Missing scene bbox: {stats.get('missing_scene_bbox', 0)}")
        print(f"  Missing object bbox: {stats.get('missing_object_bbox', 0)}")
        print(f"  By eval type:")
        for k, v in sorted(stats.items()):
            if k not in ("total", "converted", "missing_scene_bbox", "missing_object_bbox"):
                print(f"    {k}: {v}")
        print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
