#!/usr/bin/env python3
"""Convert ScanRefer, Multi3DRefer, Nr3D, and Sr3D to the grounding JSONL format
used by the OneCanvas pipeline.

Produces (under <data_root>/vlm_annotations/):
    scanrefer/scanrefer_{train,val,test}.jsonl
    multi3drefer/multi3drefer_{val,train}.jsonl
    nr3d/nr3d.jsonl
    sr3d/sr3d.jsonl
    scannet_object_bboxes.json          (per-scene AABB cache, reused across runs)

Consumed by these registry entries in training/onecanvas/data/__init__.py:
    SCANREFER_TRAIN / SCANREFER_VAL   ("scanrefer_train" / "scanrefer_val")
    MULTI3DREFER / MULTI3DREFER_TRAIN ("multi3drefer" / "multi3drefer_train")
    nr3d.jsonl / sr3d.jsonl are the RAW files; the NR3D_TRAIN / SR3D_TRAIN
    entries consume the val-leakage-filtered *_train.jsonl variants produced
    from them by scripts/filter_referit3d_val_leakage.py (run it after this).

Upstream sources:
    ScanRefer:    https://github.com/daveredrum/ScanRefer (download form),
                  files ScanRefer_filtered_{train,val,test}.json
                  -> <data_root>/scanrefer/
    Multi3DRefer: https://github.com/3dlg-hcvc/M3DRef-CLIP release
                  (multi3drefer_train_val.zip), files multi3drefer_{train,val}.json
                  -> <data_root>/vlm_annotations/multi3drefer/
    Nr3D / Sr3D:  https://referit3d.github.io/ downloads,
                  nr3d.csv -> <data_root>/referit3d/nr3d.csv
                  sr3d.csv -> <data_root>/referit3d/Sr3D/sr3d.csv
    ScanNet meshes + segmentation (for the bbox cache): downloaded on demand
                  from the official ScanNet server (requires having accepted
                  the ScanNet terms of use) into <data_root>/scannet/scannet_annotations/.

Each output line:
  {"scene_id": "...", "question": "...", "answers": ["(cx, cy, cz, w, h, d)"],
   "question_type": "grounding", "source": ..., ...}

Bounding boxes are axis-aligned, in ScanNet's aligned coordinate frame (after
applying the per-scene axisAlignment transform from the .txt metadata).

data_path: unlike the historical files, NO per-item "data_path" key is written
by default. Embedding it froze an absolute machine-local path into the jsonl
(the loader at data_processor_3d.py only falls back to the registry-configured
root when the item has NO data_path key), which broke portability. Pass an
explicit --data-path to restore the old embedding behavior.

Reproducibility: question templates are drawn from one seeded RNG stream that
runs across all datasets in a fixed order, and items are skipped when their
scene is missing from the bbox cache, so byte-identical regeneration of a
historical file requires the SAME bbox cache contents. Use --frozen-cache to
convert strictly against the shipped scannet_object_bboxes.json (no downloads,
no recompute). A fresh cache rebuild may cover more scenes and therefore yield
a superset of items with different (but distribution-identical) template draws.

Usage:
    python scripts/convert_grounding_datasets.py --frozen-cache
    python scripts/convert_grounding_datasets.py [--workers 16] [--skip-download]
"""

import argparse
import csv
import gzip
import json
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

SCANNET_BASE_URL = "http://kaldir.vc.in.tum.de/scannet/v2/scans"

# Referring expression grounding templates (natural language description)
REFERRING_TEMPLATES = [
    "Locate the object described: \"{desc}\". Provide its 3D bounding box.",
    "Find what is described: \"{desc}\". Give its 3D bounding box.",
    "Where is the following: \"{desc}\"? Output the 3D bounding box.",
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


class Paths:
    """All input/output locations, derived from the data root + CLI overrides."""

    def __init__(self, args):
        root = resolve_data_root(args.data_root)
        self.scannet_preprocessed = args.scannet_preprocessed or os.path.join(
            root, "scannet", "scannet_preprocessed")
        self.annot_cache = args.annot_cache or os.path.join(
            root, "scannet", "scannet_annotations")
        scanrefer_dir = args.scanrefer_dir or os.path.join(root, "scanrefer")
        self.scanrefer = {
            split: os.path.join(scanrefer_dir, f"ScanRefer_filtered_{split}.json")
            for split in ("train", "val", "test")
        }
        m3dr_dir = args.multi3drefer_dir or os.path.join(
            root, "vlm_annotations", "multi3drefer")
        self.multi3drefer = {
            split: os.path.join(m3dr_dir, f"multi3drefer_{split}.json")
            for split in ("val", "train")
        }
        referit3d_dir = args.referit3d_dir or os.path.join(root, "referit3d")
        self.nr3d_csv = os.path.join(referit3d_dir, "nr3d.csv")
        self.sr3d_csv = os.path.join(referit3d_dir, "Sr3D", "sr3d.csv")
        self.out_dir = args.out_dir or os.path.join(root, "vlm_annotations")
        # The bbox cache is an INPUT in --frozen-cache mode, so it hangs off the
        # data root rather than the output directory. Those are the same path
        # when --out-dir is left alone, which is why deriving it from out_dir
        # looked fine: redirect the output and the frozen cache silently went
        # missing, and every item was then skipped into an empty file.
        self.bbox_cache = args.bbox_cache or self._default_bbox_cache(root)

    @staticmethod
    def _default_bbox_cache(root):
        """The data-root copy if present, else the one shipped in this repo.

        --frozen-cache cannot work without this file and a rebuild deliberately
        does NOT reproduce it (a fresh cache covers more scenes and moves the
        seeded template draws), so the shipped copy is the only way a user who
        has not built one can regenerate the published grounding jsonls.
        """
        local = os.path.join(root, "vlm_annotations", "scannet_object_bboxes.json")
        if os.path.exists(local):
            return local
        shipped = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "assets", "scannet_object_bboxes.json.gz")
        return shipped if os.path.exists(shipped) else local


# ---------------------------------------------------------------------------
# ScanNet annotation download + bbox computation
# ---------------------------------------------------------------------------

def download_scene_files(scene_id, annot_cache):
    """Download aggregation.json, segs.json, and mesh PLY for one scene."""
    scene_dir = os.path.join(annot_cache, scene_id)
    os.makedirs(scene_dir, exist_ok=True)

    files = {
        "agg": f"{scene_id}_vh_clean.aggregation.json",
        "segs": f"{scene_id}_vh_clean_2.0.010000.segs.json",
        "ply": f"{scene_id}_vh_clean_2.ply",
    }

    for key, fname in files.items():
        out_path = os.path.join(scene_dir, fname)
        if os.path.exists(out_path):
            continue
        url = f"{SCANNET_BASE_URL}/{scene_id}/{fname}"
        try:
            urllib.request.urlretrieve(url, out_path)
        except Exception as e:
            print(f"  Failed to download {url}: {e}")
            if os.path.exists(out_path):
                os.remove(out_path)
            return None

    return scene_dir


def parse_ply_vertices(ply_path):
    """Extract vertex positions from a binary little-endian PLY file.
    ScanNet meshes: x,y,z (float32) + r,g,b,a (uint8) per vertex."""
    with open(ply_path, "rb") as f:
        num_verts = 0

        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                num_verts = int(line.split()[-1])
            elif line == "end_header":
                break

        # Standard ScanNet: float x, float y, float z, uchar r, g, b, a = 16 bytes
        dt = np.dtype([("xyz", np.float32, 3), ("rgba", np.uint8, 4)])
        raw = f.read(num_verts * dt.itemsize)

    return np.frombuffer(raw, dtype=dt)["xyz"]


def compute_scene_bboxes(scene_id, axis_align, annot_cache):
    """Compute axis-aligned bounding boxes for all objects in a scene.
    Returns dict: {object_id: [cx, cy, cz, w, h, d], ...}
    """
    scene_dir = os.path.join(annot_cache, scene_id)
    agg_path = os.path.join(scene_dir, f"{scene_id}_vh_clean.aggregation.json")
    segs_path = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json")
    ply_path = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")

    if not all(os.path.exists(p) for p in [agg_path, segs_path, ply_path]):
        return {}

    with open(agg_path) as f:
        agg = json.load(f)
    with open(segs_path) as f:
        seg_data = json.load(f)

    seg_indices = np.array(seg_data["segIndices"])
    vertices = parse_ply_vertices(ply_path)

    # Apply axis alignment
    ones = np.ones((len(vertices), 1), dtype=np.float32)
    verts_h = np.hstack([vertices, ones])  # (N, 4)
    aligned = (axis_align @ verts_h.T).T[:, :3]  # (N, 3)

    bboxes = {}
    for sg in agg["segGroups"]:
        obj_id = sg["objectId"]
        mask = np.isin(seg_indices, sg["segments"])
        obj_verts = aligned[mask]
        if len(obj_verts) == 0:
            continue

        mins = obj_verts.min(axis=0)
        maxs = obj_verts.max(axis=0)
        center = ((mins + maxs) / 2).tolist()
        size = (maxs - mins).tolist()
        bboxes[str(obj_id)] = [
            round(center[0], 2), round(center[1], 2), round(center[2], 2),
            round(size[0], 2), round(size[1], 2), round(size[2], 2),
        ]

    return bboxes


def get_axis_alignment(scene_id, scannet_preprocessed):
    """Read axis alignment matrix from scene .txt file."""
    txt_path = os.path.join(scannet_preprocessed, scene_id, f"{scene_id}.txt")
    if not os.path.exists(txt_path):
        return None
    with open(txt_path) as f:
        for line in f:
            if line.startswith("axisAlignment"):
                vals = [float(x) for x in line.split("=")[1].strip().split()]
                return np.array(vals, dtype=np.float32).reshape(4, 4)
    return None


def build_bbox_cache(scene_ids, paths, workers=16, skip_download=False,
                     frozen=False):
    """Download ScanNet annotations and compute bboxes for all needed scenes.
    Returns and caches: {scene_id: {object_id_str: [cx,cy,cz,w,h,d], ...}, ...}

    With frozen=True the cache file is used exactly as-is (no downloads, no
    recompute, no rewrite) -- the mode that reproduces the shipped jsonls.
    """
    cache = {}
    if os.path.exists(paths.bbox_cache):
        opener = gzip.open if paths.bbox_cache.endswith(".gz") else open
        with opener(paths.bbox_cache, "rt") as f:
            cache = json.load(f)

    if frozen:
        # Without a cache every lookup misses, every item is "skipped", and the
        # converter writes empty jsonls and exits 0. That is indistinguishable
        # from success in a log and downstream it just looks like a small
        # dataset, so it has to be an error here.
        if not cache:
            raise SystemExit(
                f"--frozen-cache needs the bbox cache, but "
                f"{paths.bbox_cache} is "
                f"{'empty' if os.path.exists(paths.bbox_cache) else 'missing'}. "
                f"Every item would be skipped and the outputs would be empty. "
                f"Point --bbox-cache at scannet_object_bboxes.json, or drop "
                f"--frozen-cache to build one (which yields a superset of items "
                f"with different template draws, see the module docstring)."
            )
        print(f"Frozen bbox cache: {len(cache)} scenes from {paths.bbox_cache}")
        return cache

    # Filter to scenes we actually have preprocessed data for
    available = set(os.listdir(paths.scannet_preprocessed))
    needed = [s for s in scene_ids if s in available and s not in cache]

    if not needed:
        print(f"Bbox cache has all {len(scene_ids)} scenes (or they're unavailable).")
        return cache

    print(f"Need bboxes for {len(needed)} scenes ({len(cache)} already cached)...")

    # Download annotation files
    if not skip_download:
        print(f"Downloading ScanNet annotations for {len(needed)} scenes...")
        os.makedirs(paths.annot_cache, exist_ok=True)
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_scene_files, s, paths.annot_cache): s
                       for s in needed}
            for i, fut in enumerate(as_completed(futures)):
                scene = futures[fut]
                result = fut.result()
                if result is None:
                    failed.append(scene)
                if (i + 1) % 50 == 0:
                    print(f"  Downloaded {i+1}/{len(needed)}...")

        if failed:
            print(f"  Failed to download {len(failed)} scenes: {failed[:10]}...")
            needed = [s for s in needed if s not in set(failed)]

    # Compute bboxes
    print(f"Computing bboxes for {len(needed)} scenes...")
    for i, scene_id in enumerate(needed):
        axis_align = get_axis_alignment(scene_id, paths.scannet_preprocessed)
        if axis_align is None:
            continue
        bboxes = compute_scene_bboxes(scene_id, axis_align, paths.annot_cache)
        if bboxes:
            cache[scene_id] = bboxes
        if (i + 1) % 100 == 0:
            print(f"  Computed {i+1}/{len(needed)}...")

    # Save cache
    with open(paths.bbox_cache, "w") as f:
        json.dump(cache, f)
    print(f"Bbox cache saved: {len(cache)} scenes -> {paths.bbox_cache}")

    return cache


def bbox_str(bbox):
    """Format bbox as string: (cx, cy, cz, w, h, d)"""
    return "({}, {}, {}, {}, {}, {})".format(*bbox)


# ---------------------------------------------------------------------------
# Dataset converters
# ---------------------------------------------------------------------------

def convert_scanrefer(bbox_cache, paths, data_path, split="val"):
    """Convert ScanRefer split to grounding JSONL."""
    src = paths.scanrefer[split]
    if not os.path.exists(src):
        print(f"ScanRefer {split}: source file not found ({src}), skipping.")
        return None

    with open(src) as f:
        data = json.load(f)

    out_path = os.path.join(paths.out_dir, "scanrefer", f"scanrefer_{split}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    written = 0
    skipped = 0
    with open(out_path, "w") as out:
        for item in data:
            scene_id = item["scene_id"]
            obj_id = str(item["object_id"])

            if scene_id not in bbox_cache or obj_id not in bbox_cache[scene_id]:
                skipped += 1
                continue

            bbox = bbox_cache[scene_id][obj_id]
            template = random.choice(REFERRING_TEMPLATES)
            question = template.format(desc=item["description"])

            entry = {
                "scene_id": scene_id,
                "question": question,
                "answers": [bbox_str(bbox)],
            }
            if data_path:
                entry["data_path"] = data_path
            entry.update({
                "question_type": "grounding",
                "source": "scanrefer",
                "ann_id": item.get("ann_id"),
                "object_id": int(obj_id),
            })
            out.write(json.dumps(entry) + "\n")
            written += 1

    print(f"ScanRefer {split}: {written} written, {skipped} skipped -> {out_path}")
    return out_path


def convert_multi3drefer(bbox_cache, paths, data_path, split="val"):
    """Convert Multi3DRefer to grounding JSONL."""
    src = paths.multi3drefer[split]
    with open(src) as f:
        data = json.load(f)

    out_path = os.path.join(paths.out_dir, "multi3drefer", f"multi3drefer_{split}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    written = 0
    skipped = 0
    with open(out_path, "w") as out:
        for item in data:
            scene_id = item["scene_id"]
            obj_ids = item["object_ids"]

            # Get bboxes for all referenced objects
            if scene_id not in bbox_cache:
                skipped += 1
                continue

            bboxes = []
            for oid in obj_ids:
                oid_str = str(oid)
                if oid_str in bbox_cache[scene_id]:
                    bboxes.append(bbox_cache[scene_id][oid_str])

            eval_type = item.get("eval_type", "")
            # Zero-target: description matches no objects in the scene. The
            # parser handles "none" as the empty-list answer (utils/bbox.py).
            if not bboxes and eval_type.startswith("zt_"):
                answer = "none"
            elif not bboxes:
                skipped += 1
                continue
            elif len(bboxes) == 1:
                answer = bbox_str(bboxes[0])
            else:
                answer = "; ".join(bbox_str(b) for b in bboxes)

            template = random.choice(REFERRING_TEMPLATES)
            question = template.format(desc=item["description"])

            entry = {
                "scene_id": scene_id,
                "question": question,
                "answers": [answer],
            }
            if data_path:
                entry["data_path"] = data_path
            entry.update({
                "question_type": "grounding",
                "source": "multi3drefer",
                "ann_id": item.get("ann_id"),
                "object_ids": obj_ids,
                "eval_type": item.get("eval_type"),
            })
            out.write(json.dumps(entry) + "\n")
            written += 1

    print(f"Multi3DRefer {split}: {written} written, {skipped} skipped -> {out_path}")
    return out_path


def convert_referit3d(csv_path, dataset_name, bbox_cache, paths, data_path):
    """Convert Nr3D or Sr3D CSV to grounding JSONL."""
    out_path = os.path.join(paths.out_dir, dataset_name, f"{dataset_name}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    written = 0
    skipped = 0
    with open(csv_path) as f, open(out_path, "w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = row["scan_id"]
            obj_id = str(row["target_id"])

            if scene_id not in bbox_cache or obj_id not in bbox_cache[scene_id]:
                skipped += 1
                continue

            bbox = bbox_cache[scene_id][obj_id]
            template = random.choice(REFERRING_TEMPLATES)
            question = template.format(desc=row["utterance"])

            entry = {
                "scene_id": scene_id,
                "question": question,
                "answers": [bbox_str(bbox)],
            }
            if data_path:
                entry["data_path"] = data_path
            entry.update({
                "question_type": "grounding",
                "source": dataset_name,
                "object_id": int(obj_id),
            })
            out.write(json.dumps(entry) + "\n")
            written += 1

    print(f"{dataset_name}: {written} written, {skipped} skipped -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_all_scene_ids(paths):
    """Gather all unique scene IDs referenced across all datasets."""
    scenes = set()

    # ScanRefer
    for path in paths.scanrefer.values():
        if os.path.exists(path):
            with open(path) as f:
                for item in json.load(f):
                    scenes.add(item["scene_id"])

    # Multi3DRefer
    for path in paths.multi3drefer.values():
        if os.path.exists(path):
            with open(path) as f:
                for item in json.load(f):
                    scenes.add(item["scene_id"])

    # Nr3D / Sr3D
    for csv_path in [paths.nr3d_csv, paths.sr3d_csv]:
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    scenes.add(row["scan_id"])

    return scenes


def main():
    parser = argparse.ArgumentParser(description="Convert grounding datasets to JSONL")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (default: $ONECANVAS_DATA_ROOT or sibling datasets/)")
    parser.add_argument("--out-dir", default=None,
                        help="Output vlm_annotations dir (default: <data_root>/vlm_annotations)")
    parser.add_argument("--scanrefer-dir", default=None,
                        help="Dir with ScanRefer_filtered_{train,val,test}.json "
                             "(default: <data_root>/scanrefer)")
    parser.add_argument("--multi3drefer-dir", default=None,
                        help="Dir with multi3drefer_{train,val}.json "
                             "(default: <data_root>/vlm_annotations/multi3drefer)")
    parser.add_argument("--referit3d-dir", default=None,
                        help="Dir with nr3d.csv and Sr3D/sr3d.csv "
                             "(default: <data_root>/referit3d)")
    parser.add_argument("--scannet-preprocessed", default=None,
                        help="ScanNet preprocessed scenes dir "
                             "(default: <data_root>/scannet/scannet_preprocessed)")
    parser.add_argument("--annot-cache", default=None,
                        help="Download cache for ScanNet mesh/segmentation files "
                             "(default: <data_root>/scannet/scannet_annotations)")
    parser.add_argument("--bbox-cache", default=None,
                        help="Per-scene AABB cache json "
                             "(default: <out_dir>/scannet_object_bboxes.json)")
    parser.add_argument("--data-path", default=None,
                        help="If set, embed this per-item 'data_path' like the "
                             "historical files did. Default: omit the key so the "
                             "loader falls back to the configured data root.")
    parser.add_argument("--workers", type=int, default=16, help="Download threads")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip annotation download (compute from annot cache only)")
    parser.add_argument("--frozen-cache", action="store_true",
                        help="Use the bbox cache file as-is: no downloads, no "
                             "recompute. Required to reproduce the shipped jsonls "
                             "byte-for-byte (modulo data_path).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = Paths(args)
    random.seed(args.seed)

    # Collect all needed scenes
    all_scenes = collect_all_scene_ids(paths)
    print(f"Total unique scenes across all datasets: {len(all_scenes)}")

    # Build bbox cache
    bbox_cache = build_bbox_cache(all_scenes, paths, workers=args.workers,
                                  skip_download=args.skip_download,
                                  frozen=args.frozen_cache)

    # Convert each dataset. NOTE: one RNG stream runs across all datasets in
    # this exact order; do not reorder if reproducibility matters.
    print("\n--- Converting datasets ---")

    for split in ["train", "val", "test"]:
        convert_scanrefer(bbox_cache, paths, args.data_path, split)

    for split in ["val", "train"]:
        if os.path.exists(paths.multi3drefer[split]):
            convert_multi3drefer(bbox_cache, paths, args.data_path, split)

    if os.path.exists(paths.nr3d_csv):
        convert_referit3d(paths.nr3d_csv, "nr3d", bbox_cache, paths, args.data_path)

    if os.path.exists(paths.sr3d_csv):
        convert_referit3d(paths.sr3d_csv, "sr3d", bbox_cache, paths, args.data_path)

    print("\nDone! If nr3d/sr3d were converted, now run "
          "scripts/filter_referit3d_val_leakage.py to produce the *_train variants.")


if __name__ == "__main__":
    main()
