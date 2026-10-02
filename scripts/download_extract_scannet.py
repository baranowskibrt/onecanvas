#!/usr/bin/env python
"""Download ScanNet scenes referenced by the training/eval annotations and
extract them into the preprocessed tree the OneCanvas dataloader reads.

Produces, per scene, under ``<out_dir>/<scene_id>/`` (default out_dir:
``<data_root>/scannet/scannet_preprocessed/``):

  color/<i>.jpg      per-frame RGB (JPEG quality=90), consumed via
                     ``_resolve_image_sources`` in
                     training/onecanvas/data/data_processor_3d.py:501-529
                     (the ``color`` / ``color_<W>x<H>`` entries; build the
                     resized variants with scripts/resize_frames.py)
  depth/<i>.png      per-frame uint16 depth in millimeters, consumed by the
                     GT-calibration path (data_processor_3d.py:684)
  pose/<i>.txt       4x4 camera-to-world matrices (data_processor_3d.py:680)
  <scene_id>.txt     scene metadata incl. axisAlignment
                     (data_processor_3d.py:648-676)

With ``--download-meshes`` it additionally downloads
``<scene_id>_vh_clean_2.ply`` into
``<data_root>/scannet/scannet_annotations/<scene_id>/``, which
data_processor_3d.py:709-724 reads to recover SQA3D agent poses
(the SQA3D ``bs_center`` offset).

Upstream download source: the official ScanNet release server. ScanNet data
requires accepting the ScanNet terms of use (fill the form linked from
https://github.com/ScanNet/ScanNet); ``--base-url`` is the download address
you are given after signing the agreement (the directory that contains
``v1/scans/``, ``v2/scans/``, ``v2/scans_test/``).

Scene IDs are collected from the annotation files of ScanQA, SQA3D,
VSI-Bench, SPBench, VLM-3R, and ViCA-322k, resolved relative to the data
root (see ``--annotation_files``); download those annotation sets first
(paths as in training/onecanvas/data/__init__.py).

The per-frame extraction (JPEG quality=90, uint16 PNG depth, ``%f``-formatted
pose text) matches the tree this repo was trained from exactly, up to JPEG
encoder version: color frames are re-encoded from the .sens JPEG stream, so
they are pixel-deterministic for a given PIL version but not guaranteed
byte-identical across PIL/libjpeg versions.

Example:
  python scripts/download_extract_scannet.py \
      --base-url http://<address-from-scannet-agreement>/ \
      --download-meshes
"""

import argparse
import io
import json
import os
import ssl
import sys
import tempfile
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scannet_sensordata import SensorData  # noqa: E402

ssl._create_default_https_context = ssl._create_unverified_context


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
        "--data_root (the directory that contains scannet/, vlm_annotations/, ...)."
    )


def get_scene_ids_from_file(path):
    """Extract ScanNet scene IDs from a JSON, JSONL, or Parquet annotation file."""
    ids = set()
    ext = os.path.splitext(path)[1].lower()

    if ext == '.parquet':
        import pandas as pd
        df = pd.read_parquet(path)
        # SPBench uses 'scene_name'; filter to ScanNet only
        if 'scene_name' in df.columns:
            col = 'dataset' if 'dataset' in df.columns else None
            rows = df[df[col] == 'scannet'] if col else df
            ids.update(rows['scene_name'].dropna().tolist())

    elif ext == '.jsonl':
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # VSI-Bench: {"dataset": "scannet", "scene_name": ...}
                if item.get('dataset', 'scannet') != 'scannet':
                    continue
                for key in ('scene_name', 'scene_id'):
                    if key in item:
                        ids.add(item[key])
                        break

    else:  # .json
        with open(path, 'r') as f:
            content = json.load(f)
        # SQA3D: {"questions": [...]}
        if isinstance(content, dict) and 'questions' in content:
            data = content['questions']
        else:
            data = content
        for item in data:
            # Filter to scannet-only for multi-source datasets
            source = item.get('data_source') or item.get('dataset')
            if source and source != 'scannet':
                continue
            # Try scene_id, scene_name, then extract from video path
            for key in ('scene_id', 'scene_name'):
                if key in item:
                    ids.add(item[key])
                    break
            else:
                video = item.get('video', '')
                if video:
                    stem = os.path.splitext(video.replace('\\', '/').split('/')[-1])[0]
                    if stem.startswith('scene'):
                        ids.add(stem)

    return ids


def get_all_scene_ids(annotation_files):
    all_ids = set()
    for path in annotation_files:
        if not os.path.exists(path):
            print(f"  WARNING: annotation file missing, skipped: {path}")
            continue
        ids = get_scene_ids_from_file(path)
        print(f"  {os.path.basename(path)}: {len(ids)} scenes")
        all_ids.update(ids)
    return sorted(all_ids)


def get_release_scans(release_file):
    scan_lines = urllib.request.urlopen(release_file)
    return [line.decode('utf8').rstrip('\n') for line in scan_lines]


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)


def process_and_clean(scan_id, scan_out_dir):
    sens_file = os.path.join(scan_out_dir, scan_id + '.sens')
    print(f"--- Extracting {scan_id} ---")

    try:
        sd = SensorData(sens_file)
        rgb_out = os.path.join(scan_out_dir, 'color')
        depth_out = os.path.join(scan_out_dir, 'depth')
        pose_out = os.path.join(scan_out_dir, 'pose')

        os.makedirs(rgb_out, exist_ok=True)
        os.makedirs(depth_out, exist_ok=True)
        os.makedirs(pose_out, exist_ok=True)

        sd.export_poses(pose_out)

        num_frames = len(sd.frames)
        for i in range(num_frames):
            frame = sd.frames[i]

            # RGB Extraction
            color_data = frame.decompress_color(sd.color_compression_type)
            if isinstance(color_data, np.ndarray):
                img = Image.fromarray(color_data)
            else:
                img = Image.open(io.BytesIO(color_data))
            img.save(os.path.join(rgb_out, f"{i}.jpg"), quality=90)

            # Depth Extraction
            depth_data = frame.decompress_depth(sd.depth_compression_type)
            if isinstance(depth_data, np.ndarray):
                depth_array = depth_data
            else:
                depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape(sd.depth_height, sd.depth_width)

            depth_img = Image.fromarray(depth_array)
            depth_img.save(os.path.join(depth_out, f"{i}.png"))

        print(f"Success. Deleting {sens_file}")
        os.remove(sens_file)

    except Exception as e:
        print(f"Processing Error for {scan_id}: {str(e)}")


def default_annotation_files(data_root):
    ann = os.path.join(data_root, "vlm_annotations")
    scanqa_files = [
        f"{ann}/ScanQA/data/qa/ScanQA_v1.0_train.json",
        f"{ann}/ScanQA/data/qa/ScanQA_v1.0_val.json",
        f"{ann}/ScanQA/data/qa/ScanQA_v1.0_test_w_obj.json",
        f"{ann}/ScanQA/data/qa/ScanQA_v1.0_test_wo_obj.json",
    ]
    sqa3d_files = [
        f"{ann}/sqa_task/balanced/v1_balanced_questions_train_scannetv2.json",
        f"{ann}/sqa_task/balanced/v1_balanced_questions_val_scannetv2.json",
        f"{ann}/sqa_task/balanced/v1_balanced_questions_test_scannetv2.json",
    ]
    vsibench_files = [
        f"{ann}/vsi-bench/test.jsonl",
    ]
    spbench_files = [
        f"{ann}/spbench/SPBench-MV.parquet",
        f"{ann}/spbench/SPBench-SI.parquet",
    ]
    vlm3r_files = [
        f"{ann}/vlm3r_data/vsibench_train/merged_qa_scannet_train.json",
        f"{ann}/vlm3r_data/vsibench_train/merged_qa_route_plan_train.json",
    ]
    vica_scannet_files = [
        f"{ann}/vica-322k/scannet/base/obj_appearance_order.json",
        f"{ann}/vica-322k/scannet/base/object_abs_distance.json",
        f"{ann}/vica-322k/scannet/base/object_count.json",
        f"{ann}/vica-322k/scannet/base/object_relative_distance.json",
        f"{ann}/vica-322k/scannet/base/object_size_estimation.json",
        f"{ann}/vica-322k/scannet/base/room_size.json",
        f"{ann}/vica-322k/scannet/complex/conversation.json",
        f"{ann}/vica-322k/scannet/complex/furniture.json",
        f"{ann}/vica-322k/scannet/complex/important_daily_necessities.json",
        f"{ann}/vica-322k/scannet/complex/spatial_description.json",
        f"{ann}/vica-322k/scannet/complex/usage.json",
        f"{ann}/vica-322k/scannet/complex/wheelchair_user.json",
    ]
    return (scanqa_files + sqa3d_files + vsibench_files + spbench_files
            + vlm3r_files + vica_scannet_files)


def download_mesh(base_url, scan_id, is_test, mesh_root):
    """Download <scene>_vh_clean_2.ply into <mesh_root>/<scene>/ (the path
    data_processor_3d.py:709-724 reads for SQA3D agent poses)."""
    mesh_name = f"{scan_id}_vh_clean_2.ply"
    out_file = os.path.join(mesh_root, scan_id, mesh_name)
    if os.path.isfile(out_file):
        return
    release_path = 'v2/scans_test' if is_test else 'v2/scans'
    url = f"{base_url}{release_path}/{scan_id}/{mesh_name}"
    download_file(url, out_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument('--base-url', required=True,
                        help="ScanNet release download address (given after accepting the "
                             "ScanNet terms of use); the directory containing v1/scans/, "
                             "v2/scans/, v2/scans_test/. Include the trailing slash.")
    parser.add_argument('--data_root', default=None,
                        help="Dataset root (default: ONECANVAS_DATA_ROOT, else the "
                             "datasets/ directory sibling to the repo).")
    parser.add_argument('-o', '--out_dir', default=None,
                        help="Scene output directory (default: <data_root>/scannet/scannet_preprocessed).")
    parser.add_argument('--annotation_files', nargs='+', default=None,
                        help="Annotation files to collect scene IDs from (JSON/JSONL/Parquet). "
                             "Defaults to all ScanQA, SQA3D, VSI-Bench, SPBench, VLM-3R, and "
                             "ViCA-322k ScanNet splits under <data_root>/vlm_annotations/.")
    parser.add_argument('--download-meshes', action='store_true',
                        help="Additionally download <scene>_vh_clean_2.ply into "
                             "<data_root>/scannet/scannet_annotations/<scene>/ "
                             "(needed for SQA3D agent poses).")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    base_url = args.base_url if args.base_url.endswith('/') else args.base_url + '/'
    out_dir = args.out_dir or os.path.join(data_root, "scannet", "scannet_preprocessed")
    annotation_files = args.annotation_files or default_annotation_files(data_root)
    mesh_root = os.path.join(data_root, "scannet", "scannet_annotations")

    print("Fetching scan lists...")
    v2_test_scans = set(get_release_scans(base_url + 'v2/scans_test.txt'))

    print("Collecting scene IDs from annotation files...")
    scene_ids = get_all_scene_ids(annotation_files)
    total = len(scene_ids)
    print(f"Found {total} unique scenes total. Starting sequential processing...")

    for index, scan_id in enumerate(scene_ids):
        scan_out_dir = os.path.join(out_dir, scan_id)
        is_test = scan_id in v2_test_scans

        if args.download_meshes:
            try:
                download_mesh(base_url, scan_id, is_test, mesh_root)
            except Exception as e:
                print(f"Mesh download failed for {scan_id}: {e}")

        # Skip if already unpacked
        color_path = os.path.join(scan_out_dir, 'color')
        if os.path.exists(color_path) and os.listdir(color_path):
            print(f"[{index+1}/{total}] Skipping {scan_id} (already exists).")
            continue

        print(f"[{index+1}/{total}] Downloading {scan_id} ({'test' if is_test else 'train'})...")

        try:
            # .txt metadata: both v2/scans and v2/scans_test host it
            release_path = 'v2/scans_test' if is_test else 'v2/scans'
            txt_url = f"{base_url}{release_path}/{scan_id}/{scan_id}.txt"
            download_file(txt_url, os.path.join(scan_out_dir, scan_id + '.txt'))

            # .sens files: test scans live under v2/scans_test, train/val under v1/scans
            if is_test:
                sens_url = f"{base_url}v2/scans_test/{scan_id}/{scan_id}.sens"
            else:
                sens_url = f"{base_url}v1/scans/{scan_id}/{scan_id}.sens"
            download_file(sens_url, os.path.join(scan_out_dir, scan_id + '.sens'))

            process_and_clean(scan_id, scan_out_dir)

        except Exception as e:
            print(f"Failed {scan_id}: {e}")


if __name__ == "__main__":
    main()
