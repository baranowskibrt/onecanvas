# Dataset preparation

Everything the training and eval code reads lives under one root:

```bash
export ONECANVAS_DATA_ROOT=/path/to/datasets   # default: a `datasets/` dir next to this repo
```

The datasets themselves are NOT distributed with this repo. This page lists,
per dataset, the public download source and the exact commands that turn the
raw download into the tree the loaders expect. Every command below ships in
`scripts/`. Unless noted otherwise, all annotation files are consumed in their
published upstream form.

Target tree (only the parts the code reads):

```
$ONECANVAS_DATA_ROOT/
  scannet/
    scannet_preprocessed/<scene_id>/{color,color_640x480,color_320x240,depth,pose,<scene_id>.txt}
    scannet_annotations/<scene_id>/<scene_id>_vh_clean_2.ply
  scannetpp/data/<scene_id>/iphone/{rgb,rgb_640x480,rgb_320x240,depth,pose_intrinsic_imu.json}
  scannetpp/data/<scene_id>/dslr/resized_undistorted_images[_<WxH>]/
  arkitscenes/raw/{Training,Validation}/<video_id>/
      {lowres_wide,lowres_depth,lowres_wide_intrinsics,lowres_wide.traj,vga_wide,vga_wide_640x480}
  vlm_annotations/{ScanQA,sqa_task,vsi-bench,spbench,vica-322k,spatialladder-26k,
                   vlm3r_data,MV-ScanQA,embodiedscan,scanrefer,multi3drefer,nr3d,sr3d}
```

## Install the preprocessing dependencies first

Everything below runs from `scripts/`, which needs deps the core install does
not carry:

```bash
pip install -e ".[data]"
```

opencv-python, lz4, pandas and pyarrow. Skipping this is the most common way
these commands fail, with a bare `ModuleNotFoundError` several steps in.

## Scene data

### ScanNet (v2)

Sign the [ScanNet terms of use](https://github.com/ScanNet/ScanNet#scannet-data)
to receive the download address, then:

```bash
python scripts/download_extract_scannet.py \
    --base-url http://<address-from-scannet-agreement>/ --download-meshes
python scripts/resize_frames.py --dataset scannet --size 640x480
python scripts/resize_frames.py --dataset scannet --size 320x240
```

Download the annotations first ([Benchmark and training
annotations](#benchmark-and-training-annotations)), because the first command
collects the scene ids from the annotation files under `vlm_annotations/`. It downloads `<scene>.sens` + `<scene>.txt`, and
extracts `color/{i}.jpg` (quality 90), `depth/{i}.png` (uint16, millimetres),
and `pose/{i}.txt`. `--download-meshes` additionally fetches
`<scene>_vh_clean_2.ply` into `scannet_annotations/`, which SQA3D agent-pose
recovery reads. The resize passes create the `color_<WxH>/` dirs the loader
selects via `image_resolution` (training default 320x240, VSI-Bench and
SPBench eval 640x480).

No crop is ever baked into the extracted files. The 3% border crop
(`border_crop_ratio`, default 0.03) is applied at load time, and only on the
predicted-geometry path. The loader crops the RGB frame to match the geometry
file it picked, so the crop follows the file name. Picking
`da3_geometry_balanced_256_crop003_metric.pt` applies 0.03, and picking
`da3_geometry_balanced_256_metric.pt` applies 0.0. Under the default
`use_gt_all=True` with GT depth and poses on disk, the loader skips the
geometry file entirely and applies NO crop. Do not expect the paper's 3% crop
on the default GT path.

### ScanNet++ (iPhone + DSLR)

Download with the official [ScanNet++ toolkit](https://github.com/scannetpp/scannetpp)
(requires their data agreement), including the iPhone RGB video, `depth.bin`,
`pose_intrinsic_imu.json`, and the undistorted DSLR images.

**The RGB video container changed upstream.** ScanNet++ used to ship `rgb.mp4`
and now ships `rgb.mkv`. Both are accepted. It is a repackaging, not a
re-capture: `pose_intrinsic_imu.json` entry counts are unchanged (checked on 8
scenes), so the frame selection picks the same indices either way, and a rebuild
from the current `.mkv` reproduces the published frames byte-for-byte.

Then:

```bash
python scripts/extract_scannetpp_iphone_rgb.py      # rgb.mkv|rgb.mp4 -> rgb/ AND rgb_640x480/ (500 frames, linspace)
python scripts/extract_scannetpp_iphone_depth.py    # depth.bin -> depth/frame_NNNNNN.png (uint16 mm)
python scripts/resize_frames.py --dataset scannetpp_iphone --size 320x240
python scripts/resize_frames.py --dataset scannetpp_dslr --size 320x240
```

**The resize chain differs from ScanNet's, and the order above matters.**
ScanNet builds both sizes from `color/`. ScanNet++ iPhone does not:

- `rgb_640x480/` is written by the extraction script, from the decoded video
  frame, in the same pass as `rgb/`. It is not a resize of `rgb/` and cannot
  be, because `rgb/` is a lossy JPEG of that same frame. `resize_frames.py`
  refuses `--dataset scannetpp_iphone --size 640x480` for that reason.
- `rgb_320x240/` is resized from `rgb_640x480/`, not from `rgb/`. Run the
  extraction first or the resize exits with an error.

Both rules were recovered by matching the tree the published numbers were
measured on, and the 320x240 chain reproduces it byte-for-byte (4000/4000
frames over 8 scenes). Building 320x240 from `rgb/` instead produces frames
that look fine and are not the ones the paper measured, worth 99 changed
predictions out of 506 on the ScanNet++ share of VSI-Bench.

At 1920x1440 the 640x480 downscale is an exact 3x, so `cv2.resize` point-samples
the centre of each 3x3 block rather than averaging it. That is what the
published frames used. An antialiased downscale there is a different image, so
do not substitute one without re-running the ScanNet++ share of the evals.

Frame stems keep the ORIGINAL video frame index, matching the per-frame keys
in `pose_intrinsic_imu.json` (consumed raw).

The whole iPhone RGB chain reproduces byte-for-byte. Verified 2026-08-25 by
downloading 8 scenes fresh and rebuilding with only these scripts: `rgb/`,
`rgb_640x480/` and `rgb_320x240/` each came out 4000/4000 byte-identical to the
tree the published numbers were measured on.

### ARKitScenes

Clone [apple/ARKitScenes](https://github.com/apple/ARKitScenes) (their
license applies), then:

```bash
python scripts/download_arkitscenes.py lowres --scene-list <ids.txt> --arkitscenes-repo /path/to/ARKitScenes
python scripts/download_arkitscenes.py vga    --scene-list <ids.txt> --arkitscenes-repo /path/to/ARKitScenes
python scripts/download_arkitscenes.py vga640 --scene-list <ids.txt> --arkitscenes-repo /path/to/ARKitScenes
```

`lowres` fetches the raw pose/intrinsics/depth assets, which the loader reads
in unmodified ARKit format. `vga` downloads `vga_wide`, keeps 256 frames and
downsamples them in place to 320x240 (PIL LANCZOS, verified byte-identical to
the published tree over 96 frames from 12 scenes). `vga640` builds
`vga_wide_640x480/` (native-resolution frames stem-matched to the depth
maps). That directory is not part of the standard ARKitScenes layout but is
what the 640x480 evals read. Without it the 640x480 evals stop with a
resolution error, so build it before running them.
`vga640` takes its stem list from `vga_wide/` on disk, so the two directories
cannot drift apart, but it does mean `vga` has to run first.

**Which 256 frames is pinned by a shipped manifest, and it has to be.** `vga`
deletes the frames it does not keep IN PLACE, so the frame count that the
even-spacing rule divided is destroyed by the step that used it. Nothing on
disk records it afterwards and ARKitScenes' `metadata.csv` has no frame-count
column, so a rebuild can be neither re-derived nor checked, and a download of a
different ARKitScenes revision would silently select different frames.
`scripts/assets/arkitscenes_vga_frames.json.gz` therefore carries the exact
stems for the 150 ARKitScenes scenes VSI-Bench evaluates (119 KB), and `vga`
uses it by default. Scenes outside it (the training pool) fall back to even
spacing and print that they did. A download that cannot supply a manifested
frame is an error rather than a smaller prune, since a third frame set that
matches neither rule is the one failure nothing downstream could detect.

## Benchmark and training annotations

Placed under `$ONECANVAS_DATA_ROOT/vlm_annotations/<name>/`. Consumed as
published, no conversion: **SQA3D** (`sqa_task/balanced/` from the official
release), **ScanQA** (clone of the ScanQA repo), **VSI-Bench**
(`nyu-visionx/VSI-Bench`, upstream already ships `test.jsonl`), **ViCA-322K**,
**SpatialLadder-26k**, **VLM-3R** (`Journey9ni/VLM-3R-DATA`, the
`vsibench_train/merged_qa_*.json` files are upstream), **MV-ScanQA**
(`kmichiru/MV-ScanQA`).

Converted locally, converter ships here:

```bash
# SPBench (hongxingli/SPBench): parquet -> spbench_si.jsonl / spbench_mv.jsonl
python scripts/convert_spbench.py

# EmbodiedScan: official pkl annotations -> per-object OBB jsonls
python scripts/convert_embodiedscan_grounding.py

# ScanRefer / Nr3D / Sr3D / Multi3DRefer -> onecanvas grounding jsonl schema
python scripts/convert_grounding_datasets.py --frozen-cache
# then the MANDATORY val-leakage filter for the ReferIt3D training splits
python scripts/filter_referit3d_val_leakage.py
```

All four converters were re-run from scratch on 2026-08-25 and regenerate the
exact files this code was trained and evaluated with: byte-identical for
SPBench and the leakage filter, identical modulo the removed `data_path` field
for the 10 EmbodiedScan splits and all six grounding sets. The converters do
not emit per-item `data_path` keys, so the output is portable across machines.

`--frozen-cache` reproduces the shipped grounding files from the pinned
793-scene bbox cache. **That cache ships with the release** as
`scripts/assets/scannet_object_bboxes.json.gz` (284 KB), and the converter
falls back to it when `<data_root>/vlm_annotations/scannet_object_bboxes.json`
is absent, which is the normal state for a fresh checkout. It has to ship,
because a rebuilt cache covers more scenes and moves the seeded template draws:
fine for new work, but not the file the published splits came from. Without a
cache, `--frozen-cache` now exits with an error rather than writing empty
jsonls and reporting success, which is what it used to do.

## No vision-feature pre-extraction is required

Nothing in this release loads precomputed vision-tower features. The QA
dataloader hands the model raw frames and the vision tower runs live inside
`forward()`. `scripts/precompute.py --features` still writes
`qwen3_vl_features_*.pt`, but no loader reads them, so skip it.

The one precomputed-feature artifact that IS used is the stage-1 curriculum's
OBB feature stash, and it builds itself on first use (see "Derived artifacts"
below). You never invoke it directly.

Verified: an eval on a tree containing no feature files at all reproduced the
author tree's predictions exactly, on every question of both benchmarks tested.

## Predicted geometry (optional for the default paths)

The default configuration (`use_gt_all=True`, which every shipped run script
uses) reads GT sensor depth and poses, so most scenes need no precompute. Two
exceptions and one opt-in:

- Seven ScanNet++ iPhone scenes have broken ARKit GT poses
  (`DA3_FALLBACK_SCENES` in `data_processor_3d.py`). For those the loader
  reads `da3_geometry_balanced_256_metric.pt` even under `--use_gt_all`, and
  logs each such scene. Without the file those scenes cannot load, and a
  benchmark run stops instead of scoring a partial split.
- `scannetpp_pose_frame` selects the ScanNet++ world frame. The released model
  records `arkit` (ARKit's frame turned z-up, every scene has GT poses). Under
  `mesh`, poses are registered to the annotated mesh, and scenes without a
  registration, all 50 VSI-Bench ScanNet++ scenes among them, fall back to the
  DA3 file above.
- The grounding datasets read `*_metric_aligned.pt`.
- The `--no-use-gt-all` / predicted-geometry arms read the full
  `da3_geometry_balanced_256*` family.

All of these come from one script (GPU required):

```bash
python scripts/precompute.py --dataset scannet     --nonaligned --aligned
python scripts/precompute.py --dataset scannetpp   --nonaligned --aligned
python scripts/precompute.py --dataset arkitscenes --nonaligned --split Training
```

`_metric` files are pinned to `depth-anything/DA3NESTED-GIANT-LARGE-1.1` (the
non-nested DA3-Large is not metric scaled and must never write `_metric`
files). Each output gets a sidecar json recording the model id.

The `--predicted-geometry` eval variants (`*_eval32*`, `dvlt_*`,
`mapanything_*`, `*_scalecal*`, `*_hires*`, `*_upright*`) were produced by
research tooling outside this release and are NOT reproducible from this
repo. The paper's headline numbers do not use them.

`scripts/audit_poses.py <dataset>` regenerates the pose-quality audits behind
`docs/bad_scenes.md`.

## Derived artifacts built automatically

The stage-1 OBB feature stash is built on first use from the ViCA scene
sources into `${XDG_CACHE_HOME:-~/.cache}/onecanvas_features/`. It needs the
ScanNet, ScanNet++, and ARKitScenes trees above plus the ViCA annotations and
a GPU. The real-object asset banks (optional curriculum variant) are built by
`scripts/extract_real_object_assets.py` / `extract_real_object_scene_assets.py`
from the converted EmbodiedScan jsonls plus the `_aligned` geometry.

## Traps worth knowing

- The loader checks the size of the frames it opens and stops when it differs
  from the requested resolution (`--image_resolution_fallback warn` accepts it
  instead). Build the `_<WxH>` dirs, and for ARKitScenes the `vga` and `vga640`
  steps, for the resolutions you run.
- Scenes missing the files a path needs are dropped when the dataset is built,
  with a warning. A benchmark run then stops, because its question count no
  longer matches the complete split (3,519 SQA3D, 5,130 VSI-Bench, 1,328
  SPBench). During a benchmark, a question whose scene fails to load is retried
  and then stops the run, never replaced by another question.
