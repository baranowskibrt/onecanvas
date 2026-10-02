#!/usr/bin/env python3
"""Convert EmbodiedScan v2 visual grounding annotations to OneCanvas JSONL format.

Produces (under <data_root>/vlm_annotations/embodiedscan/):
    embodiedscan_scannet_{train,val}.jsonl
    embodiedscan_3rscan_{train,val}.jsonl
    embodiedscan_matterport3d_{train,val}.jsonl
    embodiedscan_arkitscenes_{train,val}.jsonl
    embodiedscan_all_{train,val}.jsonl          (concatenation of the above)

Consumers: no fixed entry in training/onecanvas/data/__init__.py registers
these files directly; scripts/extract_real_object_assets.py reads
embodiedscan_scannet_{split}.jsonl to harvest real-object OBB assets, and
training mixes reference the per-source files ad hoc.

Upstream source: EmbodiedScan v2 official data release
(https://github.com/OpenRobotLab/EmbodiedScan, download requires their
agreement form), files:
  - embodiedscan_infos_{train,val,test}.pkl: per-scene metadata with 9-DoF OBB
    instances
  - embodiedscan_{train,val}_vg.json: visual grounding text annotations
Place them under <data_root>/embodiedscan/embodiedscan-v2/.

Each VG entry references a scan_id and target_id; the target_id maps to a
bbox_id in the infos pkl. We combine these to produce grounding JSONL entries
with OBB bounding boxes.

Bbox format (9-DoF OBB): (cx, cy, cz, dx, dy, dz, rx, ry, rz)
  - (cx, cy, cz): center in the aligned coordinate frame
  - (dx, dy, dz): full size dimensions along OBB local axes
  - (rx, ry, rz): Euler rotation angles (ZXY convention)

The bboxes in the infos pkl are already in the axis-aligned frame (verified
by comparing with ScanRefer AABBs for the same ScanNet scenes).

data_path: unlike the historical files, NO per-item "data_path" key is written
by default. Embedding it froze absolute machine-local paths into the jsonl
(the loader only falls back to the configured root when the item has NO
data_path key), which broke portability. Pass --emit-data-paths to restore the
historical behavior: each item then carries the source tree derived from the
data root (scannet/scannet_preprocessed/, arkitscenes/raw/, 3rscan/,
matterport3d/), and for ARKitScenes the Training/ or Validation/ subdir is
moved from the scene_id into the data_path (the loader's scene index expects
the bare numeric scene ID; when walking arkitscenes/raw/ WITHOUT per-item
data_path it strips the Training/Validation level by itself, so bare
scene_ids resolve either way).

Reproducibility: question templates come from one seeded RNG stream shared
across the train and val splits (train first). Convert both splits in one run
(the default) to reproduce the shipped val files.

Usage:
    python scripts/convert_embodiedscan_grounding.py
    python scripts/convert_embodiedscan_grounding.py --split train
    python scripts/convert_embodiedscan_grounding.py --split val
"""

import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


# EmbodiedScan source prefix -> subtree of the data root holding that source's
# preprocessed scenes (only used with --emit-data-paths).
SOURCE_SUBTREES = {
    "scannet": os.path.join("scannet", "scannet_preprocessed"),
    "arkitscenes": os.path.join("arkitscenes", "raw"),
    "3rscan": "3rscan",
    "matterport3d": "matterport3d",
}

# Referring expression grounding templates (same style as convert_grounding_datasets.py).
REFERRING_TEMPLATES_OBB = [
    "Locate the object described: \"{desc}\". Provide its oriented 3D bounding box.",
    "Find what is described: \"{desc}\". Give its oriented 3D bounding box.",
    "Where is the following: \"{desc}\"? Output the oriented 3D bounding box.",
]


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


def load_scanrefer_val_scenes(scanrefer_val_jsonl) -> set:
    """Load ScanRefer val scene IDs to exclude from training."""
    scenes = set()
    if os.path.exists(scanrefer_val_jsonl):
        with open(scanrefer_val_jsonl) as f:
            for line in f:
                d = json.loads(line)
                scenes.add(d.get("scene_id"))
    return scenes


def build_scene_bbox_map(infos: dict) -> Tuple[Dict, Dict]:
    """Build mapping: scan_id -> {bbox_id: bbox_3d_list} and scan_id -> axis_align.

    Returns:
        bbox_map: {scan_id: {bbox_id: [9 floats]}}
        align_map: {scan_id: 4x4 ndarray}
    """
    bbox_map = {}
    align_map = {}

    # Build label_id -> name mapping from metainfo
    categories = infos.get("metainfo", {}).get("categories", {})
    # categories is {name: label_id}, invert it
    id_to_name = {v: k for k, v in categories.items()}

    for entry in infos["data_list"]:
        scan_id = entry["sample_idx"]
        bboxes = {}
        for inst in entry.get("instances", []):
            bbox_id = inst["bbox_id"]
            bboxes[bbox_id] = {
                "bbox_3d": inst["bbox_3d"],
                "label": id_to_name.get(inst["bbox_label_3d"], "object"),
            }
        bbox_map[scan_id] = bboxes
        if "axis_align_matrix" in entry:
            align_map[scan_id] = np.array(entry["axis_align_matrix"])

    return bbox_map, align_map


def obb_str(bbox: list) -> str:
    """Format 9-DoF OBB as string: (cx, cy, cz, dx, dy, dz, rx, ry, rz)."""
    return "({})".format(", ".join(f"{v:.3f}" for v in bbox))


def convert_split(split: str, bbox_map: dict, scanrefer_val_scenes: set,
                  rng: random.Random, es_dir: str, out_dir: str,
                  data_path_map: dict) -> dict:
    """Convert one split (train or val) and write per-source JSONL files.

    data_path_map: {source: absolute data_path} to embed per item, or {} to
    omit the data_path key (the portable default).

    Returns: {source: count} summary.
    """
    vg_path = os.path.join(es_dir, f"embodiedscan_{split}_vg.json")
    if not os.path.exists(vg_path):
        print(f"VG file not found: {vg_path}")
        return {}

    with open(vg_path) as f:
        vg_data = json.load(f)
    print(f"Loaded {len(vg_data)} VG entries for {split}")

    # Collect entries per source
    source_entries: Dict[str, List[dict]] = defaultdict(list)
    skipped_no_bbox = 0
    skipped_val_leak = 0
    skipped_no_source = 0

    for item in vg_data:
        scan_id = item["scan_id"]  # e.g., "scannet/scene0191_00"
        parts = scan_id.split("/", 1)
        if len(parts) != 2:
            skipped_no_source += 1
            continue
        source, scene_name = parts

        # Skip ScanRefer val scenes in training split
        if split == "train" and source == "scannet" and scene_name in scanrefer_val_scenes:
            skipped_val_leak += 1
            continue

        # Look up bbox
        if scan_id not in bbox_map:
            skipped_no_bbox += 1
            continue
        target_id = item["target_id"]
        if target_id not in bbox_map[scan_id]:
            skipped_no_bbox += 1
            continue

        bbox_info = bbox_map[scan_id][target_id]
        bbox_3d = bbox_info["bbox_3d"]

        # Build question from the referring expression
        text = item["text"]
        template = rng.choice(REFERRING_TEMPLATES_OBB)
        question = template.format(desc=text)

        if source not in SOURCE_SUBTREES:
            skipped_no_source += 1
            continue
        data_path = data_path_map.get(source)

        # ARKitScenes scene_name comes in as "Training/40776204" or
        # "Validation/41069021". The data_processor's _scene_id_from_rel_path
        # strips that prefix when building its scene index (returning the bare
        # numeric ID), so leaving the prefix in scene_id makes the lookup miss.
        # Keep scene_id bare; when embedding data_path, move the prefix there.
        if source == "arkitscenes" and "/" in scene_name:
            split_dir, scene_name = scene_name.split("/", 1)
            if data_path is not None:
                data_path = os.path.join(data_path, split_dir) + "/"

        entry = {
            "scene_id": scene_name,
            "question": question,
            "answers": [obb_str(bbox_3d)],
        }
        if data_path is not None:
            entry["data_path"] = data_path
        entry.update({
            "question_type": "grounding",
            "source": f"embodiedscan_{source}",
            "bbox_format": "obb",
            "object_label": bbox_info["label"],
            "target_id": target_id,
        })

        source_entries[source].append(entry)

    # Write per-source JSONL files
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for source, entries in source_entries.items():
        out_path = os.path.join(out_dir, f"embodiedscan_{source}_{split}.jsonl")
        with open(out_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        summary[source] = len(entries)
        print(f"  {source} {split}: {len(entries)} entries -> {out_path}")

    # Also write a combined file
    all_entries = []
    for entries in source_entries.values():
        all_entries.extend(entries)
    combined_path = os.path.join(out_dir, f"embodiedscan_all_{split}.jsonl")
    with open(combined_path, "w") as f:
        for e in all_entries:
            f.write(json.dumps(e) + "\n")
    print(f"  ALL {split}: {len(all_entries)} entries -> {combined_path}")

    if skipped_no_bbox:
        print(f"  Skipped (no bbox): {skipped_no_bbox}")
    if skipped_val_leak:
        print(f"  Skipped (val leakage): {skipped_val_leak}")
    if skipped_no_source:
        print(f"  Skipped (unknown source): {skipped_no_source}")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (default: $ONECANVAS_DATA_ROOT or sibling datasets/)")
    parser.add_argument("--embodiedscan-dir", default=None,
                        help="Dir with embodiedscan_infos_*.pkl and *_vg.json "
                             "(default: <data_root>/embodiedscan/embodiedscan-v2)")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: <data_root>/vlm_annotations/embodiedscan)")
    parser.add_argument("--scanrefer-val", default=None,
                        help="scanrefer_val.jsonl used for train-split leakage filtering "
                             "(default: <data_root>/vlm_annotations/scanrefer/scanrefer_val.jsonl)")
    parser.add_argument("--emit-data-paths", action="store_true",
                        help="Embed a per-item data_path derived from the data root "
                             "(historical behavior; default omits the key)")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both",
                        help="Which split to convert (default: both). NOTE: the "
                             "template RNG stream runs across train then val, so "
                             "reproducing the shipped val files requires 'both'.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    es_dir = args.embodiedscan_dir or os.path.join(root, "embodiedscan", "embodiedscan-v2")
    out_dir = args.out_dir or os.path.join(root, "vlm_annotations", "embodiedscan")
    scanrefer_val_jsonl = args.scanrefer_val or os.path.join(
        root, "vlm_annotations", "scanrefer", "scanrefer_val.jsonl")

    data_path_map = {}
    if args.emit_data_paths:
        data_path_map = {
            source: os.path.join(root, subtree) + "/"
            for source, subtree in SOURCE_SUBTREES.items()
        }

    rng = random.Random(args.seed)

    # Load EmbodiedScan infos for all available splits
    print("Loading EmbodiedScan infos...")
    bbox_map = {}
    for s in ["train", "val", "test"]:
        pkl_path = os.path.join(es_dir, f"embodiedscan_infos_{s}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                infos = pickle.load(f)
            bm, _ = build_scene_bbox_map(infos)
            bbox_map.update(bm)
            print(f"  Loaded {len(infos['data_list'])} scenes from {s} split")

    print(f"Total scenes with bboxes: {len(bbox_map)}")

    # Load ScanRefer val scenes
    scanrefer_val = load_scanrefer_val_scenes(scanrefer_val_jsonl)
    print(f"ScanRefer val scenes to exclude: {len(scanrefer_val)}")

    # Convert
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        print(f"\n--- Converting {split} split ---")
        convert_split(split, bbox_map, scanrefer_val, rng, es_dir, out_dir,
                      data_path_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
