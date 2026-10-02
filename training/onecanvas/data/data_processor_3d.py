
import glob
import json
import math
import os
import random
import re
import time
import warnings
from bisect import bisect_left
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from . import data_list
from . import scannetpp_registration
from model_adapters import get_adapter
from .camera_utils import (
    adjust_intrinsics_for_resize,
    adjust_intrinsics_for_crop,
    crop_depth_map,
    _ORIG_W,
    _ORIG_H,
)
from .dataset_utils import (
    IGNORE_INDEX,
    pad_and_cat,
    _load_pt_mmap,
    rank0_print,
    stack_tensor_list,
    prepare_depths,
    _stratified_sample,
)
from .data_collator import FlattenedDataCollatorForSupervisedDataset
from utils.bbox import (
    format_multi_metric_bbox,
    format_multi_metric_bbox_json,
    format_multi_metric_obb_json,
    parse_multi_3d_bbox,
    parse_multi_3d_obb,
    format_pano_bbox_json,
    world_bbox_to_pano_bbox,
)
from geometry import get_scene_center, compute_scene_aabb_from_depths


# Dummy image H=W in vision patches. The dummy only makes the tokenizer
# emit an image_pad region; prepare_batch replaces it with the exact
# projected token count, so the size is arbitrary. Fixed so the processed
# dummy can be cached once in __init__.
DUMMY_IMAGE_PATCHES = 64


# Used by the pano_grounding_format path to fill the "label" field of the
# Qwen3-VL-style bbox_3d JSON. The exact string matters less than preserving
# the JSON shape — the pretrained model expects *some* short label here.
_GROUNDING_HEAD_NOUN_RE = re.compile(
    r'(?:there is a |there is an |there is the |a |an |the )?'
    r'([a-zA-Z][a-zA-Z\- ]{2,30}?)(?:[,. ]|$)',
    re.IGNORECASE,
)


# Scenes where ground-truth camera poses (ARKit / BundleFusion) diverged during
# capture. When `--use_gt_all True` is set, these scenes fall back to the
# DA3-predicted poses stored in the precomputed .pt, because the DA3 estimates
# are internally consistent even when the sensor tracker lost lock.
DA3_FALLBACK_SCENES = frozenset({
    "7dab70c8c8",  # ScanNet++ iPhone: ARKit span 189 km
    "d755b3d9d8",  # ScanNet++ iPhone: ARKit span 72 km
    "120acffd90",  # ScanNet++ iPhone: ARKit span 680 m
    "cc0aa81452",  # ScanNet++ iPhone: ARKit span 332 m
    "46001f434d",  # ScanNet++ iPhone: ARKit span 203 m
    "ab4f373966",  # ScanNet++ iPhone: ARKit span 158 m
    "02a980c994",  # ScanNet++ iPhone: slow drift, z-axis span 172 m
})


def _extract_short_label(item):
    """Best-effort short object label for pano-format grounding output.

    Preference order:
      1. item["object_name"] (Multi3DRefer / Nr3D carry this explicitly)
      2. first double-quoted phrase in the question, reduced to head noun
      3. "object"
    """
    obj_name = item.get("object_name")
    if obj_name:
        return str(obj_name).strip()[:32] or "object"
    q = str(item.get("question") or "")
    m = re.search(r'"([^"]{1,80})"', q)
    if m:
        phrase = m.group(1).lower().strip()
        m2 = _GROUNDING_HEAD_NOUN_RE.match(phrase)
        if m2:
            return m2.group(1).strip()[:32] or "object"
        return (phrase.split(",")[0].split(".")[0].strip()[:32] or "object")
    return "object"


class _BoundedCache(OrderedDict):
    """A memo dict with a size cap: inserting past ``maxlen`` evicts the
    least recently inserted entry. Drop-in for the plain dict the per-scene
    calib cache used to be, so its twenty-odd read and write sites did not
    change. Not a correctness cache: a miss reloads the same files and
    returns the same values."""

    def __init__(self, maxlen: int):
        super().__init__()
        self.maxlen = int(maxlen)

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxlen:
            self.popitem(last=False)


class SceneQADataset(Dataset):
    def __init__(self, processor, data_args, data_split="train",sort=False, select_images_randomly=False):
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        
        self.model_type = data_args.model_type
        self.adapter = get_adapter(processor, data_args)
        self.with_precomputed_geometry = bool(getattr(data_args, "with_precomputed_geometry", True))

        # self.num_images = getattr(data_args, "num_images", 8)
        self.num_images = getattr(data_args, "num_images", 20)
        # self.num_images = getattr(data_args, "num_images", 21)
        self.border_crop_ratio = float(getattr(data_args, "border_crop_ratio", 0.03))
        self.border_crop_ratio = max(0.0, min(0.45, self.border_crop_ratio))
        self._image_resolution = getattr(data_args, "image_resolution", "640x480")
        # NO SILENT FALLBACKS. "error" (default) refuses to read frames at a
        # resolution the run was not configured for; "warn" logs every
        # occurrence and counts them. See _check_frame_resolution.
        self._image_resolution_fallback = str(
            getattr(data_args, "image_resolution_fallback", "error")).lower()
        self._resolution_mismatch_count = 0
        self._resolution_checked_scenes = {}
        # "exact" is every pre-existing run. "cap" reads the best source a scene
        # HAS and treats image_resolution as a ceiling; see
        # _check_frame_resolution and _effective_frame_resolution.
        self._image_resolution_policy = str(
            getattr(data_args, "image_resolution_policy", "exact")).lower()
        if self._image_resolution_policy not in ("exact", "cap"):
            raise ValueError(
                "image_resolution_policy must be 'exact' or 'cap', got "
                f"{self._image_resolution_policy!r}")
        # {(native_w, native_h) -> {"effective": (w, h), "scenes": n}}. The
        # achieved size is a RESULT of what is installed, so a cap run has to
        # report it rather than restate the configured target.
        self._resolution_achieved = {}

        dataset_names = data_args.dataset_use.split(",")
        dataset_configs = data_list(dataset_names)
        self.sort = sort
        self._scene_asset_cache = {}
        self._root_scene_index_cache = {}
        # Per-scene file listing cache. Bounded LRU because each entry stores
        # the full {stem -> path} map of a scene (~1000-5000 frames, hundreds
        # of KB of Python strings). Unbounded growth caused worker OOMs deep
        # into long runs (steps 25k+) once enough unique scenes had been hit.
        self._image_index_cache: "OrderedDict" = OrderedDict()
        self._image_index_cache_max = int(os.environ.get("ONECANVAS_IMAGE_INDEX_CACHE_MAX", "256"))
        # Runtime sample-debug prints (scene_id + question) for dataloader iteration.
        # Defaults keep logging bounded while still giving visibility during training.
        self._debug_print_limit = int(os.getenv("QWEN3D_DEBUG_PRINT_SAMPLES", "200"))
        self._debug_print_every = int(os.getenv("QWEN3D_DEBUG_PRINT_EVERY", "1000"))
        self._debug_print_count = 0
        self._missing_root_warnings = set()
        self._skipped_missing_root = 0
        self._skipped_missing_geometry = 0
        self._skipped_scenes_by_dataset = {}   # {dataset_name: {scene_id: reason}}
        self._skipped_count_by_dataset = {}    # {dataset_name: int}

        # Optional question_type allowlist. Must be set BEFORE the
        # _load_annotations loop below, which is where it is applied.
        _qtf = getattr(data_args, "question_type_filter", None)
        self._question_type_filter = (
            {t.strip() for t in str(_qtf).split(",") if t.strip()} if _qtf else None
        )

        # Optional scene allowlist, same lifecycle as the question_type filter.
        # Point it at a file of scene ids (one per line) to evaluate ONLY those
        # scenes. Exists so a fix that provably touches a known subset (e.g.
        # --upright-arkit, which changes nothing outside gravity-rotated ARKit
        # scenes) can be re-measured on that subset instead of paying for the
        # whole benchmark.
        _sf = getattr(data_args, "scene_filter_file", None)
        if _sf:
            with open(_sf) as _fh:
                self._scene_filter = {ln.strip() for ln in _fh if ln.strip()
                                      and not ln.startswith("#")}
            rank0_print(f"[data] scene_filter_file={_sf}: "
                        f"{len(self._scene_filter)} scenes allowed")
        else:
            self._scene_filter = None

        self.list_data_dict = []
        self.sample_weights: list = []
        self._using_weighted_sampling = False
        self.data_split = data_split
        # self.sample_num = data_args.val_sample_num if data_split == "val" or data_split == "test" else None
        self.sample_num = data_args.val_sample_num if data_split == "val" else None
        self.offset = data_args.dataset_offset if hasattr(data_args, "dataset_offset") else 0
        # MUST be set before the _load_annotations loop: _build_scene_dir_index
        # reads it to decide whether a scene is indexed on its image dirs (the
        # GT path) or gated on a da3_geometry_balanced_256*_metric.pt file. It
        # used to be assigned further down in __init__, so the index always took
        # the geometry-gated branch and every scene without predicted geometry
        # was dropped as missing_scene_dir -- silently, since skips only warn.
        # A tree built exactly as docs/DATA.md describes (GT depth and poses, no
        # precompute) therefore loaded ZERO samples. It went unnoticed because
        # the development trees all carry DA3 .pt files from other experiments.
        self._use_gt_all = bool(getattr(data_args, "use_gt_all", False))
        for dataset_name, config in zip(dataset_names, dataset_configs):
            config = dict(config)
            config["_dataset_name"] = dataset_name
            self._load_annotations(config)

        if self._skipped_missing_root or self._skipped_missing_geometry:
            total_skipped = self._skipped_missing_root + self._skipped_missing_geometry
            total_raw = len(self.list_data_dict) + total_skipped
            skip_pct = 100.0 * total_skipped / total_raw if total_raw > 0 else 0.0
            _severity = "WARNING" if skip_pct > 1.0 else "INFO"
            # Break the count down by the reason actually recorded. The old
            # message called every non-root skip "missing geometry", which sent
            # debugging after DA3 .pt files when the real reason was an
            # unresolved scene directory.
            _by_reason = {}
            for _scenes in self._skipped_scenes_by_dataset.values():
                for _reason in _scenes.values():
                    _kind = str(_reason).split(":", 1)[0] or "unknown"
                    _by_reason[_kind] = _by_reason.get(_kind, 0) + 1
            _detail = ", ".join(f"{k}: {v} scenes" for k, v in sorted(_by_reason.items())) \
                or f"missing roots: {self._skipped_missing_root}"
            print(
                f"[data] [{_severity}] skipped {total_skipped}/{total_raw} samples ({skip_pct:.1f}%) "
                f"with unavailable scene assets ({_detail})"
            )
            for ds_name, scenes in sorted(self._skipped_scenes_by_dataset.items()):
                sample_count = self._skipped_count_by_dataset.get(ds_name, len(scenes))
                unique_scenes = len(scenes)
                print(f"[data]   {ds_name}: {sample_count} samples skipped across {unique_scenes} scenes")
                for scene_id, reason in sorted(scenes.items())[:5]:
                    print(f"[data]     {scene_id}: {reason}")
                if unique_scenes > 5:
                    print(f"[data]     ... and {unique_scenes - 5} more scenes")
            if skip_pct > 10.0:
                print(
                    f"[data] [WARNING] {skip_pct:.0f}% of samples skipped — many missing scene assets. "
                    f"Check NFS connectivity and that scenes are preprocessed."
                )
            if getattr(self.data_args, "strict_data_loading", False) and total_skipped > 0:
                raise RuntimeError(
                    f"strict_data_loading: {total_skipped} samples skipped due to missing scene assets. "
                    f"Fix the data or disable --strict_data_loading to continue."
                )

        if self.sample_weights and len(self.sample_weights) != len(self.list_data_dict):
            raise RuntimeError(
                f"sample_weights length {len(self.sample_weights)} != "
                f"list_data_dict length {len(self.list_data_dict)}; "
                "mixed @N and unweighted datasets resolved incorrectly."
            )

        if self.sort:
            self.list_data_dict.sort(key=lambda x: str(x.get("scene_id", "")))

        if self.sample_num is not None:
            stratified = getattr(data_args, "stratified_eval", True)
            if stratified:
                self.list_data_dict = _stratified_sample(
                    self.list_data_dict, self.sample_num, seed=42
                )
            else:
                self.list_data_dict = self.list_data_dict[self.offset:self.offset + self.sample_num]

        if self.data_split in ("val", "test"):
            self.sample_weights = []

        if len(self.list_data_dict) == 0:
            # Point at the reason that was actually recorded. Annotations
            # loading fine and then every scene being dropped for missing
            # assets is the common case, and blaming annotation paths for it
            # sends debugging in the wrong direction.
            _dropped = self._skipped_missing_root + self._skipped_missing_geometry
            if _dropped:
                raise ValueError(
                    f"Dataset is empty: all {_dropped} annotated samples were dropped because "
                    f"their scene assets could not be resolved (see the [data] skip breakdown "
                    f"above). The annotations were read fine. Check ONECANVAS_DATA_ROOT and "
                    f"that each scene directory holds the frame directories the loader looks "
                    f"for, per docs/DATA.md."
                )
            raise ValueError("Dataset is empty. Check your annotation paths.")

        print(f"Successfully loaded {len(self.list_data_dict)} samples.")
        self.with_answer = getattr(data_args, "with_answer", True)
        self.select_images_randomly = select_images_randomly
        self._projection_mode = getattr(data_args, "projection_mode", "equirectangular")
        if self._projection_mode not in ("equirectangular", "corrected_equirectangular"):
            raise ValueError(
                f"Only equirectangular projection is supported (got "
                f"{self._projection_mode!r})."
            )
        # Dummy image: gets the tokenizer to emit <|vision_start|>...<|vision_end|>
        # with image_pad tokens.  prepare_batch splices in the exact number of
        # projected tokens, so the dummy size doesn't matter.
        _patch = getattr(processor.image_processor, "patch_size", 16)
        _merge = getattr(processor.image_processor, "merge_size", 2)
        self._dummy_grid_scale = _patch * _merge
        self.dummy_image = Image.new(
            'RGB',
            (DUMMY_IMAGE_PATCHES * self._dummy_grid_scale, DUMMY_IMAGE_PATCHES * self._dummy_grid_scale),
            (0, 0, 0),
        )

        # Projection config — used when projection is run in the dataloader (CPU).
        self._rope_pos_range = float(getattr(data_args, "rope_pos_range", 100.0))
        self._temporal_max_range = float(getattr(data_args, "temporal_max_range", 100.0))
        self._temporal_raw_frame_index = bool(getattr(data_args, "temporal_raw_frame_index", False))
        self._use_resized_images = bool(getattr(data_args, "use_resized_images", True))
        _obs_res = str(getattr(data_args, "observation_max_resolution", "") or "").strip().lower()
        self._observation_max_resolution = (
            tuple(int(v) for v in _obs_res.split("x")) if "x" in _obs_res else None)
        _max_res_str = str(getattr(data_args, "max_image_resolution", "") or "").strip().lower()
        if _max_res_str and "x" in _max_res_str:
            _mw, _mh = _max_res_str.split("x")
            self._max_image_resolution = (int(_mw), int(_mh))
        else:
            self._max_image_resolution = None
        if self._image_resolution_policy == "cap" and self._max_image_resolution is None:
            raise ValueError(
                "image_resolution_policy='cap' needs max_image_resolution set: "
                "the cap is what downscales an oversized source, and without it "
                "the run would read native frames while claiming the target size")
        # ARKitScenes gravity uprighting. ARKit stores frames in the sensor's
        # native landscape buffer regardless of how the phone was held, so
        # gravity points sideways on 84 of the 150 VSI-Bench ARKit scenes and
        # the vision tower encodes rooms on their side. GT poses carry the roll
        # correctly, so only the APPEARANCE is wrong. Rotating image, depth,
        # intrinsics and pose together leaves the lifted 3D identical.
        # Default OFF so every pre-existing number reproduces unchanged.
        self._upright_arkit = bool(getattr(data_args, "upright_arkit", False))
        # Causal check: force this many clockwise quarter turns on every ARKit
        # scene instead of the gravity-derived k. Slice the already-upright
        # (k=0) scenes out of the results and they should DEGRADE.
        self._upright_force_turns = int(getattr(data_args, "upright_force_turns", 0) or 0)
        self._arkit_k_cache = {}
        self._depth_embed_min = float(getattr(data_args, "depth_embed_min", 0.3))
        # Fallback TRUE, matching the DataArguments default: an object that
        # does not carry the attribute at all must not land silently in the
        # amputated regime (see argument.py, 2026-08-01).
        self._depth_embed_mode = "cartesian_fourier" if getattr(data_args, "use_depth_embedding", True) else "off"
        self._feature_set = getattr(data_args, "feature_set", "balanced")
        self._inline_patch_override_rope = bool(
            getattr(data_args, "inline_patch_override_rope", True))
        self._probe_zero_visual = bool(getattr(data_args, "curriculum_zero_visual", False))
        self._probe_single_patch_canvas = bool(getattr(data_args, "curriculum_single_patch_canvas", False))
        self._pano_grounding_format = bool(getattr(data_args, "pano_grounding_format", False))
        self._metric_json_grounding_format = bool(getattr(data_args, "metric_json_grounding_format", True))
        if self._pano_grounding_format and self._metric_json_grounding_format:
            raise RuntimeError(
                "pano_grounding_format and metric_json_grounding_format are mutually exclusive."
            )

        # GT depth override: load sensor depth PNGs instead of DAv3 depth from .pt
        self._use_gt_depth = bool(getattr(data_args, "use_gt_depth", False))
        # Full GT bypass: poses + intrinsics + depth from ScanNet sensor files,
        # no DA3 geometry used at all. Forces use_gt_depth=True below.
        # Already set above, before _load_annotations, because the scene-dir
        # index needs it. Do not move it back down here.
        assert self._use_gt_all == bool(getattr(data_args, "use_gt_all", False))
        if self._use_gt_all:
            self._use_gt_depth = True
        self._gt_strict = bool(getattr(data_args, "gt_strict", False))
        self._gt_strict_warned: set = set()
        # predicted-geometry override. When set (e.g. "da3_eval32"
        # or "mapanything_eval32") the loader prefers the predictor's eval32
        # geometry .pt (keyed on the exact frames the GT eval selects) over the
        # default balanced_256 DA3 files, so a GT-vs-predicted eval compares
        # identical photos with only the geometry source changed. Must be run
        # with use_gt_all=False so poses/intrinsics/depth all come from the .pt.
        self._predicted_geometry = getattr(data_args, "predicted_geometry", None) or None
        # Depth-resolution probe (eval only): coarsen the depth grid to
        # (H_feat/K, W_feat/K) inside compute_scene_geometry. Travels to the
        # live-features path through _reprojection_config(), not through the
        # batch, because the coarsening is expressed in feature-grid cells and
        # only reproject_scene knows H_feat/W_feat.
        self._depth_downsample = int(getattr(data_args, "depth_downsample", 1) or 1)
        # SPBench rides on ScanNet scene dirs but pins its own per-scene frame
        # UNIONS (own `*_spbench_crop003_metric.pt`), while VSI/SQA3D use the
        # fullspan-linspace `*_eval32_crop003_metric.pt` in the SAME dir. Under a
        # predicted-geometry override both exist, so the candidate order must put
        # the right one first for the active benchmark.
        self._is_spbench_eval = "spbench" in str(getattr(data_args, "dataset_use", "")).lower()
        # Per-scene GT calib cache (populated on first load of a scene).
        # BOUNDED since 2026-09-02. Each entry holds EVERY frame's pose of a
        # scene (ARKit / ScanNet++ add per-frame intrinsics) as one small tensor
        # per frame: ~4700 tensors and ~2.5 MB of Python objects for a ScanNet
        # scene. As a plain dict a worker kept every scene it ever touched, and
        # a random sampler over 4.5M curriculum items reaches every unique scene
        # in the run, in each of 16 workers. Every long stage-1 run of the
        # from-zero family was OOM-killed by its own cgroup near step 19000
        # (jobs 2899640, 2904634, 2906956), host memory climbing ~9 GB/h from a
        # 50 GB start, and finished only by resuming. The image-index cache
        # above hit the identical failure earlier and was bounded; this one was
        # not. A worker only needs the scene it is building, so the bound is
        # small, and a memo cannot change a result.
        self._gt_scene_calib_cache: "OrderedDict" = _BoundedCache(maxlen=32)
        # Scenes whose GT poses are broken (see DA3_FALLBACK_SCENES) and that
        # we've already logged a fallback warning for. Deduped so training logs
        # don't spam one line per sample.
        self._da3_fallback_logged: set = set()

        # Panoramic augmentation (training-only, disabled for grounding samples)
        self._panoramic_augment_center = bool(getattr(data_args, "panoramic_augment_center", False))
        self._panoramic_augment_center_sigma = float(getattr(data_args, "panoramic_augment_center_sigma", 0.0))
        self._panoramic_augment_center_uniform = bool(getattr(data_args, "panoramic_augment_center_uniform", False))
        self._panoramic_augment_center_uniform_scene = bool(getattr(data_args, "panoramic_augment_center_uniform_scene", False))
        self._panoramic_augment_center_inflate = float(getattr(data_args, "panoramic_augment_center_inflate", 1.0))
        self._panoramic_augment_yaw = bool(getattr(data_args, "panoramic_augment_yaw", False))
        self._aug_seed = int(getattr(data_args, "dataset_sampling_seed", 42))
        # SQA3D: place panorama at the situated agent position + rotation.
        self._sqa3d_use_agent_pose = bool(getattr(data_args, "sqa3d_use_agent_pose", False))
        self._sqa3d_agent_yaw_offset = float(getattr(data_args, "sqa3d_agent_yaw_offset", 0.0))
        self._sqa3d_pose_debug_logged = 0
        # Inference-time canvas-origin ablation (see argument.py for semantics).
        self._sqa3d_canvas_center_mode = str(getattr(data_args, "sqa3d_canvas_center_mode", "auto"))
        _valid_modes = {"auto", "agent_pose", "scene_center", "random_camera", "outside_bbox"}
        if self._sqa3d_canvas_center_mode not in _valid_modes:
            raise ValueError(
                f"sqa3d_canvas_center_mode must be one of {sorted(_valid_modes)}, "
                f"got {self._sqa3d_canvas_center_mode!r}"
            )
        self._sqa3d_center_mode_debug_logged = 0
        # SPBench-SI: place panorama at the single pinned camera pose + orientation.
        self._spbench_use_camera_pose = bool(getattr(data_args, "spbench_use_camera_pose", False))
        self._spbench_camera_yaw_offset = float(getattr(data_args, "spbench_camera_yaw_offset", 0.0))
        self._spbench_pose_debug_logged = 0
        # Global canvas yaw offset (seam probe). Rotates
        # the whole canvas rigidly, moving ONLY where the equirectangular
        # longitude wrap falls relative to the scene. See the apply site.
        self._panoramic_eval_yaw_offset = float(
            getattr(data_args, "panoramic_eval_yaw_offset", 0.0))
        self._eval_yaw_offset_logged = 0
        self._predgeo_anchor_logged = 0
        self._declared_anchor_logged = 0

        # Pre-process the dummy 360 image ONCE so __getitem__ never repeats the
        # expensive image-processing pass (~2 s/sample) for an image that never changes.
        _img_w = DUMMY_IMAGE_PATCHES * self._dummy_grid_scale
        _img_h = DUMMY_IMAGE_PATCHES * self._dummy_grid_scale
        self._dummy_img_w = _img_w
        self._dummy_img_h = _img_h

        # Run the image processor in isolation (fast path: no text).
        # This gives us pixel_values + image_grid_thw without any text overhead.
        _img_only = self.processor.image_processor(
            images=[self.dummy_image],
            max_pixels=_img_w * _img_h,
            return_tensors="pt",
        )
        self._cached_pixel_values   = _img_only["pixel_values"]
        self._cached_image_grid_thw = _img_only["image_grid_thw"]
        print(f"[init] cached 360-image pixel_values shape: {self._cached_pixel_values.shape}, "
              f"image_grid_thw: {self._cached_image_grid_thw}")

        _ac = self.adapter.config
        print(f"[init] adapter: feature_prefix={_ac.feature_prefix}, "
              f"image_pad={_ac.image_pad_token_id}, video_pad={_ac.video_pad_token_id}, "
              f"vision_start={_ac.vision_start_token_id}, "
              f"assistant={_ac.assistant_token_id}, im_end={_ac.im_end_token_id}")


    def _arkit_upright_k(self, scene_dir):
        """Clockwise quarter turns needed to upright this scene's frames, or 0.

        Returns 0 for every non-ARKitScenes scene: the gravity angle is read
        from ``lowres_wide.traj``, which only ARKitScenes has, so ScanNet and
        ScanNet++ can never be rotated by accident. Cached per scene because
        it reads the whole trajectory file.
        """
        if not self._upright_arkit:
            return 0
        if scene_dir in self._arkit_k_cache:
            return self._arkit_k_cache[scene_dir]
        from onecanvas.data.arkit_upright import scene_upright_k
        k = scene_upright_k(scene_dir)
        if k is None:
            k = 0                                   # not an ARKitScenes scene
        elif self._upright_force_turns:
            k = self._upright_force_turns % 4       # causal check
        self._arkit_k_cache[scene_dir] = k
        return k

    def _compute_border_crop_box(self, width, height):
        if self.border_crop_ratio <= 0:
            return (0, 0, width, height)

        crop_x = int(round(width * self.border_crop_ratio))
        crop_y = int(round(height * self.border_crop_ratio))

        max_crop_x = max((width - 2) // 2, 0)
        max_crop_y = max((height - 2) // 2, 0)
        crop_x = min(crop_x, max_crop_x)
        crop_y = min(crop_y, max_crop_y)

        left, top = crop_x, crop_y
        right, bottom = width - crop_x, height - crop_y
        return (left, top, right, bottom)

    @staticmethod
    def _scene_id_from_rel_path(rel_path):
        if not rel_path or rel_path == ".":
            return None

        parts = rel_path.split(os.sep)
        if parts[-1] == "iphone" and len(parts) >= 2:
            return parts[-2]
        if parts[0] in {"Training", "Validation", "Test"} and len(parts) >= 2:
            return parts[1]

        scene_name = parts[-1]
        if scene_name.endswith("_unified_resolution"):
            return scene_name[: -len("_unified_resolution")]
        return scene_name

    def _build_scene_dir_index(self, data_path):
        # When running fully from GT (no DA3 .pt files), index by image-dir
        # presence instead. Otherwise gate on a precomputed geometry file so
        # half-preprocessed scenes don't appear in the index.
        gt_only = bool(getattr(self, "_use_gt_all", False))
        # Must stay in sync with _resolve_image_sources, which accepts both the
        # bare directory name and a resolution-suffixed variant (color_320x240,
        # resized_undistorted_images_320x240, ...). Matching only the bare names
        # here dropped 50 ScanNet++ DSLR scenes that ship ONLY the resized dir,
        # so the index called them missing while the loader could have read them.
        image_dirnames = (
            "color", "rgb", "lowres_wide", "vga_wide",
            "resized_undistorted_images", "resized_images",
        )

        # Bare name, or the name plus a resolution suffix. Anchored on a
        # <W>x<H> suffix rather than any suffix so ARKitScenes' sibling
        # lowres_wide_intrinsics/ is not mistaken for a frame directory.
        _image_dir_re = re.compile(
            r"^(?:%s)(?:_\d+x\d+)?$" % "|".join(re.escape(b) for b in image_dirnames)
        )

        def _is_image_dir(name):
            return _image_dir_re.match(name) is not None

        scene_dirs = {}
        for root, dirs, files in os.walk(data_path, followlinks=True):
            rel_path = os.path.relpath(root, data_path)
            depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
            if depth > 2:
                dirs[:] = []
                continue

            if gt_only:
                has_assets = any(_is_image_dir(d) for d in dirs)
            else:
                has_assets = (
                    "da3_geometry_balanced_256_crop003_metric.pt" in files
                    or "da3_geometry_balanced_256_metric.pt" in files
                )

            if has_assets:
                scene_id = self._scene_id_from_rel_path(rel_path)
                if scene_id is not None:
                    scene_dirs.setdefault(str(scene_id), root)
                dirs[:] = []

        return scene_dirs

    def _resolve_scene_dir(self, scene_id, data_path, scene_subdir=None):
        if isinstance(data_path, (list, tuple)):
            for dp in data_path:
                result = self._resolve_scene_dir(scene_id, dp, scene_subdir=scene_subdir)
                if result is not None:
                    return result
            return None

        if not data_path or not os.path.isdir(data_path):
            return None

        # When a specific subdirectory is requested (e.g. "dslr"), bypass the
        # generic index which may return "iphone/" instead via setdefault.
        if scene_subdir:
            candidate = os.path.join(data_path, str(scene_id), scene_subdir)
            if os.path.isdir(candidate):
                return candidate
            return None

        scene_dir_index = self._root_scene_index_cache.get(data_path)
        if scene_dir_index is None:
            scene_dir_index = self._build_scene_dir_index(data_path)
            self._root_scene_index_cache[data_path] = scene_dir_index

        return scene_dir_index.get(str(scene_id))

    def _check_frame_resolution(self, scene_dir, source):
        """Refuse to read frames at a resolution the run did not ask for.

        Between 2026-08-06 and 2026-08-23 every VSI-Bench 640x480 run read
        ARKitScenes at 320x240, because `vga_wide_640x480/` was a dangling link
        into a decommissioned volume and the loader quietly fell back to
        `vga_wide/`. Nothing logged it, so a third of the benchmark ran at half
        resolution behind plausible-looking numbers.

        Checking that the directory NAME matched was what allowed that: the
        real question is the size of the pixels actually opened, so this peeks
        at one frame per scene. It stays silent when a differently-named
        directory happens to hold correctly-sized frames, which is the normal
        ARKit training case (`vga_wide/` is already 320x240 in place).
        """
        if source is None:
            return
        if self._image_resolution_fallback == "off" and self._image_resolution_policy != "cap":
            return
        try:
            want_w, want_h = (int(v) for v in str(self._image_resolution).lower().split("x"))
        except (ValueError, AttributeError):
            return  # no resolution configured, nothing to enforce

        cached = self._resolution_checked_scenes.get(scene_dir)
        if cached is not None:
            got = cached
        else:
            image_dir, image_ext = source
            try:
                names = sorted(glob.glob(os.path.join(image_dir, f"*{image_ext}")))
                if not names:
                    return
                with Image.open(names[0]) as im:
                    got = im.size
            except Exception:
                return  # a read problem is reported elsewhere, not here
            self._resolution_checked_scenes[scene_dir] = got

        if self._image_resolution_policy == "cap":
            # The configured size is a CEILING. `source` is already the best
            # directory this scene has (an exact-size resized dir when it
            # exists, otherwise the un-suffixed native one), so a smaller `got`
            # means nothing better is installed and upsampling it would buy
            # 4x the canvas tokens and no extra evidence. Record what was
            # actually achieved; a run reports that instead of its target.
            effective = self._effective_frame_resolution(got)
            row = self._resolution_achieved.setdefault(
                tuple(got), {"effective": effective, "scenes": 0})
            if cached is None:                      # first sighting of this scene
                row["scenes"] += 1
            return

        if got == (want_w, want_h):
            return

        msg = (
            f"frame resolution mismatch in {scene_dir}: configured "
            f"image_resolution={want_w}x{want_h} but the frames actually being read "
            f"are {got[0]}x{got[1]} (source: {source[0]}). The run would silently "
            f"evaluate at a resolution it was not configured for. Build the "
            f"missing resized directory (see docs/DATA.md), or pass "
            f"--image_resolution_fallback warn to accept it, which logs and "
            f"counts every occurrence."
        )
        if self._image_resolution_fallback != "warn":
            raise RuntimeError(msg)
        self._resolution_mismatch_count += 1
        # Deliberately print, not warnings.warn: Python dedups warnings per
        # process, which would turn a per-scene rate into a single line.
        print(f"[data] RESOLUTION FALLBACK ({self._resolution_mismatch_count}): {msg}")

    def _effective_frame_resolution(self, got):
        """The size frames reach the tower at, after the configured cap."""
        if self._max_image_resolution is None:
            return tuple(int(v) for v in got)
        w, h = (int(v) for v in got)
        mw, mh = self._max_image_resolution
        scale = min(mw / w, mh / h)
        if self._image_resolution_policy == "cap":
            scale = min(scale, 1.0)
        return (max(1, int(round(w * scale))), max(1, int(round(h * scale))))

    def resolution_achieved_summary(self):
        """What a cap run actually read, keyed by the size on disk.

        Empty under the 'exact' policy, where the configured resolution and the
        achieved one are the same number by construction.
        """
        return {
            f"{w}x{h}": {"effective": f"{row['effective'][0]}x{row['effective'][1]}",
                         "scenes": int(row["scenes"])}
            for (w, h), row in sorted(self._resolution_achieved.items())
        }

    def _resolve_image_sources(self, scene_dir):
        res = self._image_resolution
        resized_source = None
        full_source = None

        for dirname, ext in ((f"color_{res}", ".jpg"), (f"rgb_{res}", ".jpg"),
                              (f"vga_wide_{res}", ".png"),
                              # ScanNet++ DSLR resized directories
                              (f"resized_undistorted_images_{res}", ".JPG"),
                              (f"resized_undistorted_images_{res}", ".jpg"),
                              (f"resized_images_{res}", ".JPG"),
                              (f"resized_images_{res}", ".jpg")):
            candidate = os.path.join(scene_dir, dirname)
            if os.path.isdir(candidate):
                resized_source = (candidate, ext)
                break

        for dirname, ext in (("color", ".jpg"), ("rgb", ".jpg"), ("lowres_wide", ".png"), ("vga_wide", ".png"),
                              ("resized_undistorted_images", ".JPG"), ("resized_undistorted_images", ".jpg"),
                              ("resized_images", ".JPG"), ("resized_images", ".jpg")):
            candidate = os.path.join(scene_dir, dirname)
            if os.path.isdir(candidate):
                full_source = (candidate, ext)
                break

        if full_source is None:
            full_source = resized_source

        return resized_source, full_source

    @staticmethod
    def _sort_frame_keys(frame_keys):
        def sort_key(value):
            if isinstance(value, (int, float)):
                return (0, float(value), str(value))

            text = str(value)
            match = re.search(r"(\d+(?:\.\d+)?)$", text)
            if match:
                return (0, float(match.group(1)), text)

            return (1, float("inf"), text)

        return sorted(frame_keys, key=sort_key)

    def _maybe_debug_print_sample(self, item, curr_idx):
        if self._debug_print_limit <= 0 and self._debug_print_every <= 0:
            return

        worker = get_worker_info()
        # Print from worker 0 only to avoid duplicated multi-worker spam.
        if worker is not None and worker.id != 0:
            return

        should_print = False
        if self._debug_print_count < self._debug_print_limit:
            should_print = True
        elif self._debug_print_every > 0 and self._debug_print_count % self._debug_print_every == 0:
            should_print = True

        if should_print:
            q = str(item.get("question", "")).replace("\n", " ").strip()
            q = q[:220]
            ds = item.get("dataset_name", "unknown")
            print(
                f"[data][{self.data_split}] dataset={ds} idx={curr_idx} "
                f"scene_id={item.get('scene_id', 'NA')} question={q}"
            )

        self._debug_print_count += 1

    @staticmethod
    def _extract_numeric_suffix(value):
        text = str(value)
        match = re.search(r"(\d+(?:\.\d+)?)$", text)
        return float(match.group(1)) if match else None

    def _get_image_index(self, image_dir, image_ext):
        cache_key = (image_dir, image_ext)
        cached = self._image_index_cache.get(cache_key)
        if cached is not None:
            self._image_index_cache.move_to_end(cache_key)
            return cached

        exact_map = {}
        numeric_pairs = []
        for entry in os.scandir(image_dir):
            if not entry.is_file() or not entry.name.endswith(image_ext):
                continue
            stem = os.path.splitext(entry.name)[0]
            exact_map[stem] = entry.path
            numeric_value = self._extract_numeric_suffix(stem)
            if numeric_value is not None:
                numeric_pairs.append((numeric_value, entry.path))

        numeric_pairs.sort(key=lambda x: x[0])
        numeric_values = [x[0] for x in numeric_pairs]
        numeric_paths = [x[1] for x in numeric_pairs]
        index = (exact_map, numeric_values, numeric_paths)
        self._image_index_cache[cache_key] = index
        if len(self._image_index_cache) > self._image_index_cache_max:
            self._image_index_cache.popitem(last=False)
        return index

    def _resolve_frame_image_path(self, image_dir, image_ext, frame_key):
        key_str = str(frame_key)
        exact_map, numeric_values, numeric_paths = self._get_image_index(image_dir, image_ext)

        # Fast path for datasets where geometry keys match image stems exactly.
        exact_path = exact_map.get(key_str)
        if exact_path is not None:
            return exact_path

        # ARKit scenes often have slight timestamp drift between geometry keys and
        # image filenames (e.g. ...316 vs ...333). Fall back to nearest timestamp.
        numeric_key = self._extract_numeric_suffix(key_str)
        if numeric_key is None or len(numeric_values) == 0:
            return None

        insert_pos = bisect_left(numeric_values, numeric_key)
        candidates = []
        if insert_pos < len(numeric_values):
            candidates.append((abs(numeric_values[insert_pos] - numeric_key), numeric_paths[insert_pos]))
        if insert_pos > 0:
            candidates.append((abs(numeric_values[insert_pos - 1] - numeric_key), numeric_paths[insert_pos - 1]))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _load_scannet_gt_scene_calib(self, scene_id: str, scene_dir: str,
                                     wants_aligned: bool) -> Optional[dict]:
        """Load ScanNet GT calibration for --use_gt_all mode.

        Returns dict with:
          depth_K: [fx, fy, cx, cy] for full depth-sensor frame (typically 640x480)
          poses:   {stem: 4x4 depth-sensor pose in world coords}
                   For wants_aligned: axisAlignment @ pose_color @ E_ctd^-1.
                   Else:              pose_color @ E_ctd^-1.
        Returns None if any required file/field is missing (caller falls back
        to the DA3 .pt).
        """
        _cache_key = (scene_id, bool(wants_aligned))
        if _cache_key in self._gt_scene_calib_cache:
            return self._gt_scene_calib_cache[_cache_key]

        info_path = os.path.join(scene_dir, f"{scene_id}.txt")
        if not os.path.exists(info_path):
            self._gt_scene_calib_cache[_cache_key] = None
            return None
        vals: Dict[str, str] = {}
        with open(info_path) as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip()
        try:
            depth_K = torch.tensor([
                float(vals["fx_depth"]), float(vals["fy_depth"]),
                float(vals["mx_depth"]), float(vals["my_depth"]),
            ], dtype=torch.float32)
        except KeyError:
            self._gt_scene_calib_cache[_cache_key] = None
            return None
        E_ctd_raw = vals.get("colorToDepthExtrinsics")
        if E_ctd_raw:
            nums = [float(x) for x in E_ctd_raw.split()]
            E_ctd = torch.tensor(nums, dtype=torch.float32).reshape(4, 4) if len(nums) == 16 else torch.eye(4)
        else:
            E_ctd = torch.eye(4, dtype=torch.float32)
        E_ctd_inv = torch.linalg.inv(E_ctd)

        axis = torch.eye(4, dtype=torch.float32)
        if wants_aligned and "axisAlignment" in vals:
            ax_nums = [float(x) for x in vals["axisAlignment"].split()]
            if len(ax_nums) == 16:
                axis = torch.tensor(ax_nums, dtype=torch.float32).reshape(4, 4)

        pose_dir = os.path.join(scene_dir, "pose")
        if not os.path.isdir(pose_dir):
            self._gt_scene_calib_cache[_cache_key] = None
            return None
        depth_dir = os.path.join(scene_dir, "depth")
        depth_stems = set()
        if os.path.isdir(depth_dir):
            for d in os.scandir(depth_dir):
                if d.name.endswith(".png"):
                    depth_stems.add(os.path.splitext(d.name)[0])
        poses: Dict[str, torch.Tensor] = {}
        for p in sorted(os.scandir(pose_dir), key=lambda e: e.name):
            if not p.name.endswith(".txt"):
                continue
            stem = os.path.splitext(p.name)[0]
            if depth_stems and stem not in depth_stems:
                continue
            try:
                mat = torch.from_numpy(np.loadtxt(p.path)).float()
            except Exception:
                continue
            if mat.shape != (4, 4) or not torch.isfinite(mat).all():
                continue
            poses[stem] = axis @ mat @ E_ctd_inv

        out = {"depth_K": depth_K, "poses": poses}
        self._gt_scene_calib_cache[_cache_key] = out
        return out

    def _load_scannet_bs_center(self, scene_id: str) -> Optional[torch.Tensor]:
        """Return (verts_raw.max + verts_raw.min) / 2 — the offset SQA3D
        subtracts when storing positions. Raw-frame aabb mid of the
        _vh_clean_2.ply mesh.

        SQA3D stores agent position as ``raw_position - bs_center`` (see
        ``ScanQA/lib/sepdataset.py`` in the official repo), so to recover the
        raw position we do ``raw = stored + bs_center``. The mesh lives under
        ``${ONECANVAS_DATA_ROOT}/scannet/scannet_annotations/{scene_id}/``.
        """
        cache_key = (scene_id, "bs_center")
        if cache_key in self._gt_scene_calib_cache:
            return self._gt_scene_calib_cache[cache_key]
        from . import _DATA_ROOT
        mesh_root = f"{_DATA_ROOT}/scannet/scannet_annotations"
        ply_path = os.path.join(mesh_root, scene_id, f"{scene_id}_vh_clean_2.ply")
        if not os.path.exists(ply_path):
            self._gt_scene_calib_cache[cache_key] = None
            return None
        try:
            from plyfile import PlyData
        except ImportError:
            # Don't silently disable GT calibration: warn once (default filter
            # dedupes per call site) so the fallback to depth-derived centers is
            # visible. plyfile ships in onecanvas[train].
            warnings.warn(
                "plyfile is not installed; GT scene-center calibration is "
                "disabled and centers fall back to depth-derived estimates. "
                "Install it (pip install plyfile, or onecanvas[train]).",
                RuntimeWarning,
            )
            self._gt_scene_calib_cache[cache_key] = None
            return None
        try:
            ply = PlyData.read(ply_path)
            xs = np.asarray(ply['vertex']['x'])
            ys = np.asarray(ply['vertex']['y'])
            zs = np.asarray(ply['vertex']['z'])
            mid = torch.tensor([
                (float(xs.min()) + float(xs.max())) * 0.5,
                (float(ys.min()) + float(ys.max())) * 0.5,
                (float(zs.min()) + float(zs.max())) * 0.5,
            ], dtype=torch.float32)
        except Exception:
            # Corrupt / unreadable mesh: degrade gracefully as before.
            mid = None
        self._gt_scene_calib_cache[cache_key] = mid
        return mid

    def _predgeo_anchor_transform(self, scene_id, scene_dir, frame_keys,
                                  pred_poses) -> Optional[dict]:
        """GT-world -> predicted-frame similarity fit for anchor re-expression
        under --predicted-geometry.

        The SQA3D agent pose is annotated in the GT ScanNet world frame, but
        with predicted geometry the canvas lives in the predictor's own
        (similarity-equivalent) frame, so applying the annotation as-is
        misplaces and misorients the anchor in every scene. Fit
        ``pred ~ s * R @ gt + t`` (Umeyama) on the camera centers of the
        SELECTED eval frames: GT centers from the same calibration the GT eval
        anchors against (pose_color @ E_ctd^-1, no axis alignment), predicted
        centers from the loaded geometry .pt. Registration uses only camera
        poses of the input frames — no answer-bearing information.

        Returns {s, R, t, yaw, rmse, n} or None (missing GT calib / <3 pairs);
        cached per scene. ``yaw`` is R's Z-azimuth component, added to the
        agent yaw so the canvas forward tracks the rotated frame.
        """
        cache_key = (scene_id, "predgeo_anchor")
        if cache_key in self._gt_scene_calib_cache:
            return self._gt_scene_calib_cache[cache_key]
        result = None
        calib = None
        if scene_dir:
            calib = self._load_scannet_gt_scene_calib(
                scene_id, scene_dir, wants_aligned=False)
        if calib is not None:
            gt_c, pr_c = [], []
            for key, pp in zip(frame_keys, pred_poses):
                gp = calib["poses"].get(str(key))
                if gp is None or pp is None or not torch.is_tensor(pp):
                    continue
                if torch.isfinite(gp).all() and torch.isfinite(pp).all():
                    gt_c.append(gp[:3, 3].double().numpy())
                    pr_c.append(pp[:3, 3].double().numpy())
            if len(gt_c) >= 3:
                src, dst = np.stack(gt_c), np.stack(pr_c)
                mu_s, mu_d = src.mean(0), dst.mean(0)
                sc, dc = src - mu_s, dst - mu_d
                norm = float(np.linalg.norm(sc))
                if norm > 1e-9:
                    s = float(np.linalg.norm(dc)) / norm
                    U, _, Vt = np.linalg.svd((s * sc).T @ dc)
                    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
                    R = Vt.T @ D @ U.T
                    t = mu_d - s * R @ mu_s
                    rmse = float(np.sqrt((((s * (src @ R.T)) + t - dst) ** 2).sum(1).mean()))
                    result = {
                        "s": s, "R": R, "t": t,
                        "yaw": float(math.atan2(R[1, 0], R[0, 0])),
                        "rmse": rmse, "n": len(gt_c),
                    }
        if result is not None and self._predgeo_anchor_logged < 5:
            self._predgeo_anchor_logged += 1
            print(
                f"[predgeo-anchor] scene={scene_id} n={result['n']} "
                f"scale={result['s']:.4f} yaw={result['yaw']:.4f} "
                f"fit_rmse={result['rmse']:.4f}m"
            )
        self._gt_scene_calib_cache[cache_key] = result
        return result

    def _load_arkitscenes_gt_scene_calib(self, scene_id: str, scene_dir: str,
                                         wants_aligned: bool) -> Optional[dict]:
        """Load ARKitScenes GT calibration for --use_gt_all mode.

        Returns dict with:
          poses:             {stem: 4x4 world-pose, y-up ARKit → z-up}
          intrinsics_per_frame: {stem: [fx, fy, cx, cy]} from per-frame .pincam
          depth_subdir:      'lowres_depth' (uint16 mm PNGs)

        Stems match DA3 .pt key format: "<scene_id>_<timestamp>" where
        timestamp is three-decimal seconds. Pose logic matches
        scripts/precompute.py:_load_arkit_gt_poses (same 100 ms match tolerance
        and Rx(+90°) y-up → z-up correction). `wants_aligned` is ignored
        (ARKit has no axis-alignment step).
        """
        _cache_key = (scene_id, "arkit", bool(wants_aligned))
        if _cache_key in self._gt_scene_calib_cache:
            return self._gt_scene_calib_cache[_cache_key]

        traj_path = os.path.join(scene_dir, "lowres_wide.traj")
        intr_dir  = os.path.join(scene_dir, "lowres_wide_intrinsics")
        depth_dir = os.path.join(scene_dir, "lowres_depth")
        if not (os.path.exists(traj_path) and os.path.isdir(intr_dir)
                and os.path.isdir(depth_dir)):
            self._gt_scene_calib_cache[_cache_key] = None
            return None

        traj_ts = []
        traj_T  = []
        with open(traj_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 7:
                    continue
                ts = float(parts[0])
                aa = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
                t  = np.array([float(x) for x in parts[4:7]], dtype=np.float64)
                angle = np.linalg.norm(aa)
                if angle > 1e-8:
                    axis = aa / angle
                    K = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                else:
                    R = np.eye(3)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3]  = t
                traj_ts.append(ts)
                traj_T.append(T)
        if not traj_ts:
            self._gt_scene_calib_cache[_cache_key] = None
            return None
        traj_ts_arr = np.array(traj_ts)

        # ARKit .traj conventions (empirically verified):
        #   - world-to-camera (invert for c2w)
        #   - world z-up, cam OpenCV-compatible (NO cam flip needed)

        prefix = scene_id + "_"
        poses: Dict[str, torch.Tensor] = {}
        intrinsics: Dict[str, torch.Tensor] = {}
        for entry in os.scandir(intr_dir):
            if not entry.name.endswith(".pincam"):
                continue
            stem = entry.name[: -len(".pincam")]
            if not stem.startswith(prefix):
                continue
            try:
                ts = float(stem[len(prefix):])
            except ValueError:
                continue
            idx = int(np.abs(traj_ts_arr - ts).argmin())
            if abs(traj_ts_arr[idx] - ts) > 0.1:
                continue
            try:
                with open(entry.path) as fh:
                    nums = fh.read().split()
                # pincam: W H fx fy cx cy
                if len(nums) < 6:
                    continue
                fx, fy, cx, cy = (float(nums[2]), float(nums[3]),
                                  float(nums[4]), float(nums[5]))
            except Exception:
                continue
            poses[stem] = torch.from_numpy(np.linalg.inv(traj_T[idx])).float()
            intrinsics[stem] = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

        if not poses:
            self._gt_scene_calib_cache[_cache_key] = None
            return None

        # Filter to frames whose depth PNGs (and RGB) actually exist on disk.
        # ARKit .pincam / .traj entries occasionally reference timestamps with
        # no matching lowres_depth file, which would otherwise surface later
        # as "gt_all could not resolve depth" skip-scene warnings.
        _available: Optional[set] = None
        for _subdir in ("lowres_depth", "lowres_wide"):
            _dir = os.path.join(scene_dir, _subdir)
            if os.path.isdir(_dir):
                _stems = {os.path.splitext(f)[0] for f in os.listdir(_dir)}
                _available = _stems if _available is None else _available & _stems
        if _available is not None:
            poses = {k: v for k, v in poses.items() if k in _available}
            intrinsics = {k: v for k, v in intrinsics.items() if k in _available}
        if not poses:
            self._gt_scene_calib_cache[_cache_key] = None
            return None

        out = {
            "poses": poses,
            "intrinsics_per_frame": intrinsics,
            "depth_subdir": "lowres_depth",
        }
        self._gt_scene_calib_cache[_cache_key] = out
        return out

    def _load_scannetpp_gt_scene_calib(self, scene_id: str, scene_dir: str,
                                       wants_aligned: bool) -> Optional[dict]:
        """Load ScanNet++ iPhone GT calibration for --use_gt_all mode.

        Returns dict with:
          poses:             {stem: 4x4 world-pose from pose_intrinsic_imu.json}
          intrinsics_per_frame: {stem: [fx, fy, cx, cy]}
          depth_subdir:      'depth' (uint16 mm PNGs from extract_scannetpp_iphone_depth.py)

        Stem format "frame_NNNNNN" matches both the JSON keys and the DA3 .pt keys.
        `wants_aligned` is ignored (ScanNet++ iphone has no axis-alignment step;
        raw poses match the existing _aligned.pt files per precompute.py).
        DSLR scenes are not supported — returns None so loader falls back to DA3.
        """
        _pose_frame = str(getattr(self.data_args, "scannetpp_pose_frame", "mesh"))
        if _pose_frame not in ("mesh", "arkit"):
            raise ValueError(f"scannetpp_pose_frame must be 'mesh' or 'arkit', got {_pose_frame!r}")
        _cache_key = (scene_id, "scannetpp", bool(wants_aligned), _pose_frame)
        if _cache_key in self._gt_scene_calib_cache:
            return self._gt_scene_calib_cache[_cache_key]

        # DSLR branch: no depth/pose in the same format; fall back.
        if os.path.basename(scene_dir.rstrip(os.sep)) == "dslr":
            self._gt_scene_calib_cache[_cache_key] = None
            return None

        json_path = os.path.join(scene_dir, "pose_intrinsic_imu.json")
        if not os.path.exists(json_path):
            self._gt_scene_calib_cache[_cache_key] = None
            return None
        with open(json_path) as f:
            data = json.load(f)

        # ScanNet++ iPhone JSON pose conventions (empirically verified):
        #   - camera-to-world
        #   - cam frame already OpenCV-compatible (no cam flip)
        #
        # The world frame is ARKit's, which is NOT the frame `segments_anno.json`
        # annotates. This used to apply a bare Rx(+90°) to get from y-up to z-up
        # and stop there, as if the two frames differed by a height convention.
        # They differ by 151 degrees of yaw and a translation as well, so every
        # object annotation landed up to a metre from its object. The per-scene
        # registration below replaces that rotation entirely: it already carries
        # the y-up-to-z-up part, fitted rather than assumed.
        if _pose_frame == "arkit":
            # ARKit's y-up world turned z-up, the frame checkpoints trained
            # before the registration saw. Every scene has it.
            reg = (((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), 1.0, (0.0, 0.0, 0.0))
        else:
            reg = scannetpp_registration.arkit_to_mesh(scene_id)
        if reg is None:
            # No registration means annotations cannot be placed on this scene's
            # canvas, so the GT path refuses and the caller falls back to DA3 as
            # it does for any scene without GT assets. Returning the old
            # unregistered poses instead would keep training on boxes in walls.
            self._gt_scene_calib_cache[_cache_key] = None
            return None
        R_reg, s_reg, t_reg = reg
        R_reg = torch.tensor(R_reg, dtype=torch.float32)
        t_reg = torch.tensor(t_reg, dtype=torch.float32)
        poses: Dict[str, torch.Tensor] = {}
        intrinsics: Dict[str, torch.Tensor] = {}
        for stem, entry in data.items():
            pose = entry.get("pose")
            K    = entry.get("intrinsic")
            if pose is None or K is None:
                continue
            T = torch.tensor(pose, dtype=torch.float32)
            if T.shape != (4, 4) or not torch.isfinite(T).all():
                continue
            fx, fy = float(K[0][0]), float(K[1][1])
            cx, cy = float(K[0][2]), float(K[1][2])
            # JSON K is for full 1920x1440 RGB; rescale to 256x192 depth grid
            # so downstream code (scales K by img_res/depth_res) is correct.
            full_w = max(1.0, cx * 2.0)
            full_h = max(1.0, cy * 2.0)
            sx = 256.0 / full_w
            sy = 192.0 / full_h
            # Scale belongs to the camera CENTRE, not to its orientation and not
            # to the depth map. It is ARKit's few-percent drift in position; the
            # LiDAR depth is metric independently, and multiplying it into the
            # rotation block would hand every downstream consumer a rotation
            # that is not orthonormal.
            M = torch.eye(4, dtype=torch.float32)
            M[:3, :3] = R_reg @ T[:3, :3]
            M[:3, 3] = s_reg * (R_reg @ T[:3, 3]) + t_reg
            poses[stem] = M
            intrinsics[stem] = torch.tensor(
                [fx * sx, fy * sy, cx * sx, cy * sy], dtype=torch.float32,
            )

        if not poses:
            self._gt_scene_calib_cache[_cache_key] = None
            return None

        # Filter to frames that actually exist on disk.  The pose JSON has
        # one entry per raw iPhone frame (8k-18k), but rgb/ and depth/ are
        # subsampled to ~500 frames each.  Keep only the intersection so the
        # sampler never picks a frame that lacks an image or depth map.
        _available: Optional[set] = None
        for _subdir in ("rgb", "depth"):
            _dir = os.path.join(scene_dir, _subdir)
            if os.path.isdir(_dir):
                _stems = {os.path.splitext(f)[0] for f in os.listdir(_dir)}
                _available = _stems if _available is None else _available & _stems
        if _available is not None:
            poses = {k: v for k, v in poses.items() if k in _available}
            intrinsics = {k: v for k, v in intrinsics.items() if k in _available}
        if not poses:
            self._gt_scene_calib_cache[_cache_key] = None
            return None

        out = {
            "poses": poses,
            "intrinsics_per_frame": intrinsics,
            "depth_subdir": "depth",
        }
        self._gt_scene_calib_cache[_cache_key] = out
        return out

    def _load_gt_scene_calib(self, scene_id: str, scene_dir: str,
                             wants_aligned: bool,
                             dataset_name: Optional[str] = None) -> Optional[dict]:
        """Dispatcher over per-dataset GT-calib loaders.

        Dispatches by scene_dir path (most reliable — a single dataset_name like
        `vsi_bench` mixes scenes from all 3 datasets). Returns None for scenes
        whose GT assets are absent, and the caller falls back to DA3.
        """
        sd = scene_dir.replace(os.sep, "/")
        if "/arkitscenes/" in sd:
            return self._load_arkitscenes_gt_scene_calib(scene_id, scene_dir, wants_aligned)
        if "/scannetpp/" in sd:
            return self._load_scannetpp_gt_scene_calib(scene_id, scene_dir, wants_aligned)
        # Fallback: ScanNet (the only remaining dataset with a GT loader).
        return self._load_scannet_gt_scene_calib(scene_id, scene_dir, wants_aligned)

    def _geometry_candidates(self, wants_aligned: bool):
        """Return the ordered list of (filename, crop_ratio) geometry candidates.

        For grounding samples (``wants_aligned=True``) we need GT-pose-derived
        axis-aligned geometry (``_aligned.pt``); ScanRefer/Multi3DRefer/Nr3D/Sr3D
        bboxes are produced in the same axis-aligned ScanNet frame, so the
        loaded poses and the GT bboxes match without any rotation.

        For QA samples (``wants_aligned=False``) we use the default DA3
        ``_metric.pt`` which is gravity-aligned by ``_compute_da3_geometry`` but
        otherwise in DA3's predicted-pose frame. QA does not need coordinate
        accuracy, only consistent panorama orientation.

        Falling back from ``_aligned.pt`` to ``_metric.pt`` for grounding is
        almost always a bug — the calling code logs a warning when it happens.
        """
        crop = self.border_crop_ratio
        primary = [
            ('da3_geometry_balanced_256_crop003_metric.pt', crop),
            ('da3_geometry_balanced_256_metric.pt',         0.0),
        ]
        # predicted-geometry override. Prepend the predictor's
        # eval32 files (keyed on the exact GT-eval frames). Per-dataset naming:
        # scannet uses crop003, scannetpp/arkit use crop 0.0, SPBench has its own
        # union-stem file; all three variants are listed and the first that
        # exists on disk for a given scene wins (the same try-each-candidate
        # mechanism the balanced_256 pair already relies on).
        if self._predicted_geometry and not wants_aligned:
            _pred_prefix = {
                'da3_eval32': 'da3',
                'mapanything_eval32': 'mapanything',
                'da3_eval32ctx256': 'da3',
                # DVLT / Deja View (unposed pure RGB). DVLT is non-metric, so the
                # *_metric.pt it points at is the DA3-anchored rescale
                # (dvlt_geometry_eval32_metric.pt from scripts/dvlt_da3anchor.py):
                # per-scene scale = median DA3/DVLT depth, so QA differences vs DA3
                # isolate pose / collective-reconstruction quality, not scale.
                'dvlt_eval32': 'dvlt',
                # Full-resolution / gravity-upright regeneration (2026-07-24).
                # ARKit's on-disk `vga_wide` is 320x240 while the eval reads
                # `vga_wide_640x480`, so every earlier predicted-geometry arm
                # reconstructed from a quarter of the pixels the model sees;
                # ARKit is additionally uprighted by k*90 (LabelMaker protocol)
                # because 100% of its hard reconstruction collapses were
                # 90-degree-rotated scenes.
                'da3_hires': 'da3',
                'dvlt_hires': 'dvlt',
                # da3_hires plus one per-sensor metric-scale constant on the
                # ScanNet files (0.9669, calibrated on 876 non-eval scenes by
                # scripts/predgeo_scale_calibrate.py in agentic-onecanvas).
                # ARKit/ScanNet++ hires files are reused unchanged: their
                # measured bias is <1%, so there is nothing to calibrate.
                'da3_hires_scalecal': 'da3',
                # Per-scene GT scale: DIAGNOSTIC CEILING for scale corrections
                # (privileged -- peeks at each eval scene's GT depth). Never a
                # quotable row; bounds how much of the predicted-vs-GT gap is
                # scale at all.
                'da3_hires_scaleoracle': 'da3',
                # 32 frames + gravity-uprighting + vga_wide_640x480, NO context.
                # Isolates uprighting from frame count: on the 84 rotated ARKit
                # scenes uprighting alone closes 99% of the 32->286 trajectory
                # gap (median 0.416 -> 0.055 m vs 0.050 m at 286 frames), so
                # "ARKit needs 256 frames" may be an artifact of sideways images.
                'da3_upright32': 'da3',
                # ScanNet at 32 frames but process_res 504 / img_subdir=color,
                # i.e. the hires arm's RESOLUTION with none of its 282 context.
                # SQA3D's 32-vs-286 gap bundled both; this splits them.
                'da3_hires32': 'da3',
                # MATCHED DVLT: 256-frame context, ARKit gravity-uprighted,
                # vga_wide_640x480 -- the twin of da3_hires, so a DVLT-vs-DA3
                # difference is the PREDICTOR and nothing else. Every earlier
                # DVLT row was 32 frames against DA3 arms with up to 286.
                'dvlt_ctx256': 'dvlt',
            }.get(self._predicted_geometry)
            if _pred_prefix is None:
                raise ValueError(
                    f"Unknown predicted_geometry={self._predicted_geometry!r}; "
                    "expected 'da3_eval32', 'mapanything_eval32', "
                    "'da3_eval32ctx256', 'dvlt_eval32', 'da3_upright32', "
                    "'da3_hires' or 'dvlt_hires'."
                )
            _scannet = (f'{_pred_prefix}_geometry_eval32_crop003_metric.pt', crop)
            _spbench = (f'{_pred_prefix}_geometry_spbench_crop003_metric.pt', crop)
            _other = (f'{_pred_prefix}_geometry_eval32_metric.pt', 0.0)  # scannetpp/arkit
            if self._is_spbench_eval:
                predicted = [_spbench, _scannet, _other]
            else:
                predicted = [_scannet, _spbench, _other]
            if self._predicted_geometry == 'dvlt_ctx256':
                # Per-dataset names again: ARKit carries `_upright`, ScanNet++
                # does not, ScanNet carries crop003. All three are the DA3-ANCHORED
                # metric rescale -- raw DVLT is ~1.9x off scale.
                for _c, _cr in (
                        ('dvlt_geometry_eval32ctx256_crop003_metric.pt', crop),
                        ('dvlt_geometry_eval32ctx256_upright_hires_metric.pt', 0.0),
                        ('dvlt_geometry_eval32ctx256_hires_metric.pt', 0.0)):
                    predicted.insert(0, (_c, _cr))
            if self._predicted_geometry == 'da3_hires32':
                # ScanNet-only arm; scenes without it fall through to the usual
                # 32-frame files, so nothing outside SQA3D's 66 scenes changes.
                predicted.insert(
                    0, (f'{_pred_prefix}_geometry_eval32_hires32_crop003_metric.pt', crop))
            if self._predicted_geometry == 'da3_upright32':
                # ARKit only: its uprighted 32-frame file must be tried BEFORE
                # the plain eval32 name, since ARKit scenes carry both. ScanNet
                # and ScanNet++ have no uprighted file and fall through to their
                # usual 32-frame arms, which is the intent -- neither dataset is
                # rotated, so this arm changes ARKit and nothing else.
                predicted.insert(
                    0, (f'{_pred_prefix}_geometry_eval32_upright_hires_metric.pt', 0.0))
            if self._predicted_geometry in ('da3_hires', 'dvlt_hires',
                                            'da3_hires_scalecal',
                                            'da3_hires_scaleoracle'):
                # Per-dataset names differ (ARKit carries `_upright`, the others
                # do not), so list every variant and let the existing
                # first-one-that-exists-per-scene mechanism pick. Scenes with no
                # hires file fall through to the arms below, so coverage
                # degrades gracefully instead of erroring.
                _hi = [
                    (f'{_pred_prefix}_geometry_eval32ctx_upright_hires_metric.pt', 0.0),
                    (f'{_pred_prefix}_geometry_eval32ctx_hires_metric.pt', 0.0),
                    # DVLT: the DA3-anchored METRIC rescale, never the raw file
                    # (raw DVLT is ~1.8x off scale).
                    (f'{_pred_prefix}_geometry_eval32_hires_metric.pt', 0.0),
                    (f'{_pred_prefix}_geometry_eval32ctx_hires_crop003_metric.pt', crop),
                    (f'{_pred_prefix}_geometry_spbenchctx_hires_crop003_metric.pt', crop),
                ]
                if self._is_spbench_eval:
                    _hi = _hi[-1:] + _hi[:-1]
                if self._predicted_geometry.endswith(('scalecal', 'scaleoracle')):
                    # Calibrated ScanNet files take priority over EVERY plain
                    # hires entry (prepended after the spbench rotation above,
                    # otherwise the uncalibrated spbenchctx file would win the
                    # first-exists race on every SPBench scene). ARKit /
                    # ScanNet++ scenes have no scalecal file and fall through
                    # to the plain hires list.
                    _sfx = self._predicted_geometry.rsplit('_', 1)[-1]
                    _cal = [
                        (f'{_pred_prefix}_geometry_eval32ctx_hires_{_sfx}_crop003_metric.pt', crop),
                        (f'{_pred_prefix}_geometry_spbenchctx_hires_{_sfx}_crop003_metric.pt', crop),
                    ]
                    if self._is_spbench_eval:
                        _cal = _cal[::-1]
                    _hi = _cal + _hi
                predicted = _hi + predicted
            if self._predicted_geometry.endswith('ctx256'):
                # Predictor-context arm: same eval32 frames, predicted with
                # ~256-frame context . Per-dataset naming mirrors
                # the eval32 files: scannet crop003, scannetpp/arkit crop 0.0,
                # SPBench union-stem. Scenes without a ctx256 file fall through
                # to the plain eval32 files below so coverage degrades
                # gracefully.
                _ctx_scannet = (f'{_pred_prefix}_geometry_eval32ctx256_crop003_metric.pt', crop)
                _ctx = (f'{_pred_prefix}_geometry_eval32ctx256_metric.pt', 0.0)
                _ctx_spb = (f'{_pred_prefix}_geometry_spbenchctx256_crop003_metric.pt', crop)
                if self._is_spbench_eval:
                    predicted = [_ctx_spb, _ctx_scannet, _ctx] + predicted
                else:
                    predicted = [_ctx_scannet, _ctx, _ctx_spb] + predicted
            return predicted + primary  # balanced_256 as last-ditch fallback
        if wants_aligned:
            return [
                ('da3_geometry_balanced_256_crop003_metric_aligned.pt', crop),
                ('da3_geometry_balanced_256_metric_aligned.pt',         0.0),
                *primary,  # fallback (will warn at the call site)
            ]
        return primary

    def _scene_has_required_assets(self, scene_id, data_path, scene_subdir=None):
        """Check whether a scene can be loaded before it reaches __getitem__."""
        # Fast path: callers that retry broken scenes at __getitem__ time (e.g.
        # SpatialPretrainingDataset) can skip this validation entirely. The check
        # is millions of NFS stat calls for ScanNet (one os.scandir over each
        # scene's color_*/ dir per annotation, ~9M calls / 89504 scenes), so
        # bypassing it cuts dataset construction from minutes to seconds.
        if getattr(self.data_args, "skip_asset_validation", False):
            return (True, None)

        cache_key = (self.data_split, str(data_path), str(scene_id), scene_subdir)
        cached = self._scene_asset_cache.get(cache_key)
        if cached is not None:
            return cached

        if isinstance(data_path, (list, tuple)):
            has_any = any(os.path.isdir(dp) for dp in data_path if dp)
        else:
            has_any = bool(data_path) and os.path.isdir(data_path)
        if not has_any:
            result = (False, f"missing_root:{data_path}")
            self._scene_asset_cache[cache_key] = result
            return result

        scene_dir = self._resolve_scene_dir(scene_id, data_path, scene_subdir=scene_subdir)
        if scene_dir is None:
            if isinstance(data_path, (list, tuple)):
                result = (False, f"missing_scene_dir:{[os.path.join(dp, str(scene_id)) for dp in data_path]}")
            else:
                result = (False, f"missing_scene_dir:{os.path.join(data_path, str(scene_id))}")
            self._scene_asset_cache[cache_key] = result
            return result

        _, full_source = self._resolve_image_sources(scene_dir)
        if full_source is None:
            result = (False, f"missing_images:{scene_dir}")
            self._scene_asset_cache[cache_key] = result
            return result

        image_dir, image_ext = full_source
        exact_map, _, _ = self._get_image_index(image_dir, image_ext)
        if len(exact_map) == 0:
            result = (False, f"missing_images:{scene_dir}")
            self._scene_asset_cache[cache_key] = result
            return result

        result = (True, None)

        self._scene_asset_cache[cache_key] = result
        return result


    @staticmethod
    def _normalize_annotation(item, config):
        """Normalise heterogeneous annotation formats to the canonical schema.

        Canonical schema: {scene_id, question, answers, data_path}

        Handles:
          - VLM3R / VICA-322k: ``scene_name``, ``conversations``, ``video``, ``data_source``
          - VSI-Bench: ``scene_name``, ``question``, ``ground_truth``, ``options``, ``dataset``
          - ScanQA / MV-ScanQA / SQA3D: already canonical (pass-through)
        """
        # 1. Resolve scene_id ---------------------------------------------------
        if "scene_id" not in item:
            if "scene_name" in item:
                item["scene_id"] = item["scene_name"]
            elif "image" in item:
                imgs = item["image"]
                if isinstance(imgs, str):
                    imgs = [imgs]
                if isinstance(imgs, list) and len(imgs) > 0:
                    first_img = str(imgs[0])
                    m = re.search(r"(scene\d{4}_\d{2})", first_img)
                    if m:
                        item["scene_id"] = m.group(1)
                    else:
                        parts = first_img.replace("\\", "/").split("/")
                        # HoLi-Spatial format: images/<scene_id>/<optional_prefix_>DSC*.jpg
                        if len(parts) >= 3 and parts[0] == "images":
                            item["scene_id"] = parts[1]
                            item["scene_subdir"] = "dslr"
                            item["pinned_stems"] = [
                                os.path.splitext(img.replace("\\", "/").split("/")[-1])[0]
                                for img in imgs
                            ]
                        else:
                            item["scene_id"] = parts[0]
            elif "video" in item:
                # e.g. "arkitscenes/47332778.mp4" → "47332778"
                video_stem = item["video"].replace("\\", "/").split("/")[-1]
                item["scene_id"] = os.path.splitext(video_stem)[0]

        # 2. Resolve question / answers from conversations ----------------------
        if "conversations" in item and "question" not in item:
            convs = item["conversations"]
            human_turns = [c.get("value", "") for c in convs if c.get("from") == "human"]
            gpt_turns   = [c.get("value", "") for c in convs if c.get("from") == "gpt"]
            q = human_turns[0] if human_turns else ""
            # Strip media tokens that are irrelevant at text-only processing time
            q = re.sub(r"<(?:image|video)>\s*\n?", "", q).strip()
            item["question"] = q
            item["answers"]  = gpt_turns

        # SpatialLadder cold-start format uses prompt/solution fields.
        if "question" not in item and "prompt" in item:
            item["question"] = str(item["prompt"]).strip()

        # 3. Resolve answers from ground_truth ----------------------------------
        if "ground_truth" in item and "answers" not in item:
            item["answers"] = [str(item["ground_truth"])]

        # Many datasets provide singular "answer" instead of list-like "answers".
        if "answers" not in item and "answer" in item:
            ans = item["answer"]
            if isinstance(ans, str):
                item["answers"] = [ans]
            elif isinstance(ans, list):
                if ans and all(isinstance(ele, dict) for ele in ans):
                    labels = [str(ele.get("label", "")).strip() for ele in ans if ele.get("label")]
                    item["answers"] = [", ".join(labels)] if labels else [json.dumps(ans)]
                else:
                    item["answers"] = [str(ele) for ele in ans]
            else:
                item["answers"] = [str(ans)]

        # SpatialLadder cold-start format stores answer in XML-like tags.
        if "answers" not in item and "solution" in item:
            solution = str(item["solution"])
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", solution, flags=re.IGNORECASE | re.DOTALL)
            item["answers"] = [m.group(1).strip() if m else solution.strip()]

        # 4. Append MCQ options to the question text ----------------------------
        if "options" in item and item.get("question"):
            opts = item["options"]
            if opts:
                if all(isinstance(o, str) and re.match(r"^[A-Za-z][\.|\)]\s", o.strip()) for o in opts):
                    opts_str = "\n".join(opts)
                else:
                    opts_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
                item["question"] = f"{item['question']}\n{opts_str}"

        # 5. Resolve data_path from data_path_map ------------------------------
        if "data_path_map" in config and "data_path" not in item:
            dpm = config["data_path_map"]
            source = (
                item.get("data_source")
                or item.get("dataset")
            )
            if source is None and "video" in item:
                parts = item["video"].replace("\\", "/").split("/")
                source = parts[0] if len(parts) >= 2 else None
            if source and source in dpm:
                item["data_path"] = dpm[source]

        # 6. Preserve or derive question_type --------------------------------
        if "question_type" not in item or item.get("question_type") == "N/A":
            # Use raw_question (without situation prefix) so the first word is the question word.
            q_text = item.get("raw_question") or item.get("question", "")
            q_text = q_text.strip()
            first_word = q_text.split()[0].capitalize() if q_text else "Others"
            _SQA3D_CATEGORIES = {"What", "Is", "How", "Can", "Which"}
            item["question_type"] = first_word if first_word in _SQA3D_CATEGORIES else "Others"

        return item

    def _load_annotations(self, config):
        annotations = []
        target_scene = None
        skipped_missing_root = 0
        skipped_missing_geometry = 0

        questions_path = f"question_path_{self.data_split}"
        answers_path = f"answer_path_{self.data_split}"

        if "annotation_path" in config:
            path = config["annotation_path"]
            if not os.path.exists(path):
                return

            # Collect annotation files: single file or all JSON/JSONL under a directory
            if os.path.isdir(path):
                ann_files = sorted(
                    glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
                    + glob.glob(os.path.join(path, "**", "*.json"), recursive=True)
                )
            else:
                ann_files = [path]

            # Optional question_type filtering (e.g. HoLi-Spatial QA splits).
            _qt_include = config.get("question_type_include")
            _qt_exclude = config.get("question_type_exclude")

            for ann_file in ann_files:
                if ann_file.endswith(".jsonl"):
                    with open(ann_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            data = json.loads(line)
                            if target_scene is not None and data.get("scene_id") != target_scene:
                                continue
                            qt = data.get("question_type")
                            if _qt_include is not None and qt not in _qt_include:
                                continue
                            if _qt_exclude is not None and qt in _qt_exclude:
                                continue
                            annotations.append(data)
                else:
                    with open(ann_file, "r") as f:
                        raw_data = json.load(f)
                    if isinstance(raw_data, list):
                        annotations.extend(raw_data)
                    else:
                        annotations.append(raw_data)
                    
        elif questions_path in config and answers_path in config:
            with open(config[questions_path], "r") as f:
                questions = json.load(f)["questions"]
            with open(config[answers_path], "r") as f:
                answers = json.load(f)["annotations"]

            ans_dict = {str(a["question_id"]): a for a in answers}
            for q in questions:
                if target_scene is not None and q.get("scene_id") != target_scene:
                    continue

                qid = str(q["question_id"])
                anno = ans_dict.get(qid, {})
                formatted_answers = [a["answer"] for a in anno.get("answers", [])]
                merged = {
                    "question_id": qid,
                    "scene_id": q["scene_id"],
                    "question": f"{q.get('situation', '')} {q['question']}".strip(),
                    "raw_question": q["question"],
                    "answers": formatted_answers,
                    "question_type": q.get("question_type") or anno.get("question_type", "N/A"),
                    "data_path": config["data_path"]
                }
                # SQA3D: preserve the situated agent pose (position + rotation)
                # from the annotation JSON so sqa3d_use_agent_pose can place the
                # panorama at the agent's situation location/orientation.
                if "position" in anno:
                    merged["agent_position"] = anno["position"]
                if "rotation" in anno:
                    merged["agent_rotation"] = anno["rotation"]
                if config.get("force_agent_pose"):
                    merged["force_agent_pose"] = True
                annotations.append(merged)

        dataset_weight = config.get("dataset_weight")
        sampling_rate = config.get("sampling_rate", 1.0)

        # Legacy %N path: deterministic subsampling, only used when no @N weight given.
        if dataset_weight is None and sampling_rate < 1.0:
            num_to_sample = max(1, int(len(annotations) * sampling_rate))
            # Keep dataset-level subsampling deterministic across runs.
            sampling_seed = int(getattr(self.data_args, "dataset_sampling_seed", 42))
            _ds_name_for_seed = str(config.get("_dataset_name", "unknown"))
            split_name = str(self.data_split)
            stable_key = f"{_ds_name_for_seed}:{split_name}"
            stable_hash = sum(ord(ch) for ch in stable_key)
            rng = random.Random(sampling_seed + stable_hash)
            annotations = rng.sample(annotations, num_to_sample)

        _start_idx = len(self.list_data_dict)
        _skip_pinned = config.get("skip_pinned_frames", False)

        # Fields needed at __getitem__ time. Everything else is discarded after
        # normalization to avoid holding multi-KB raw annotation data (conversations,
        # video paths, options, …) across 8 GPU processes × hundreds of thousands of
        # samples.
        _KEEP_FIELDS = frozenset({
            "scene_id", "question", "answers", "data_path", "dataset_name",
            "question_type", "raw_question", "question_id",
            "pinned_stems", "images", "scene_subdir",
            "bbox_format",
            # Generic RGB-D QA (see SINGLE_FRAME_QA): the frame(s) and their
            # geometry travel with the item, so keep them through the prune.
            # "frames" is the multi-frame variant (list of per-frame dicts);
            # "keep_box_order" pins multi-box grounding answers to annotation
            # order (e.g. grasp-then-release) instead of near-to-far.
            "single_frame", "rgb", "depth", "pose_c2w", "intrinsics", "image_wh",
            "frames", "keep_box_order",
            # A per-item tool list, rendered into the system prompt by the
            # chat template (see the branch of this name further down). Kept
            # through the prune because a tool block that is silently dropped
            # does not raise: the sample just trains an action call whose
            # schema the prompt never documented.
            "tools",
            # A DECLARED canvas origin, honoured verbatim (see the branch of
            # this name further down). Kept through the prune because an anchor
            # that is silently dropped does not raise: the sample just gets the
            # mean-camera origin instead, and every answer stored against the
            # declared one is then wrong by the offset between the two.
            "canvas_center", "canvas_yaw",
            # A per-item SYSTEM message and the declared reference points /
            # feature pool that bind an item's named points to fixed pool rows.
            # Kept through the prune for the same reason as the two above: a
            # field that is silently dropped does not raise, the sample just
            # trains on a different prompt or a resampled binding than it was
            # generated with, and nothing in the record says so.
            "system_prompt", "reference_points", "feature_pool",
            "reference_point_frame",
            # SQA3D situated agent pose (used when sqa3d_use_agent_pose is on).
            "agent_position", "agent_rotation",
            # Per-item gate: set by the sqa3d_agent_pose dataset entry so pose
            # conditioning fires on those items even when the global flag is off.
            "force_agent_pose",
        })

        _qt_filter = self._question_type_filter
        _qt_dropped = 0

        for ann in annotations:
            # Normalise heterogeneous annotation formats to canonical schema.
            ann = self._normalize_annotation(ann, config)

            # question_type allowlist (e.g. a counting-only eval). Applied here,
            # before the per-scene asset check, so a filtered run neither pays
            # for scenes it will not use nor reports skip counts for them.
            if _qt_filter is not None and ann.get("question_type") not in _qt_filter:
                _qt_dropped += 1
                continue

            # Scene allowlist, applied in the same place and for the same
            # reason: a filtered run must not pay for scenes it will not use.
            if self._scene_filter is not None \
                    and str(ann.get("scene_id")) not in self._scene_filter:
                _qt_dropped += 1
                continue

            # When skip_pinned_frames is set, drop pinned_stems so the loader
            # samples num_images frames from the full scene instead of only
            # the images referenced in the annotation.
            if _skip_pinned:
                ann.pop("pinned_stems", None)

            # Single-frame RGB-D QA: the frame + geometry live in the item, so
            # there is no scene directory to resolve or asset-check. Keep the
            # geometry fields and append directly.
            if config.get("single_frame"):
                ann["single_frame"] = True
                ann.setdefault("data_path", "")
                # A MULTI-OBSERVATION CONVERSATION carries its turns and one
                # photo per observed turn instead of a question and an
                # answer. It used to be dropped here without a word for
                # having no `question`, so it is admitted on its own fields
                # and keeps them through the prune.
                _is_obs_conv = bool(ann.get("observations")) and bool(
                    ann.get("conversations"))
                if "scene_id" not in ann or (
                        "question" not in ann and not _is_obs_conv):
                    continue
                ann["dataset_name"] = config.get("_dataset_name", "unknown")
                _keep_sf = (_KEEP_FIELDS | {"observations", "conversations",
                                            "schema"}
                            if _is_obs_conv else _KEEP_FIELDS)
                for k in list(ann.keys()):
                    if k not in _keep_sf:
                        del ann[k]
                self.list_data_dict.append(ann)
                continue

            # Fall back to config-level data_path when the item has no mapping.
            if "data_path" not in ann:
                if "data_path" in config:
                    ann["data_path"] = config["data_path"]
                else:
                    # Cannot determine data_path — skip this item.
                    continue

            if "scene_id" not in ann or "question" not in ann:
                continue

            available, reason = self._scene_has_required_assets(ann["scene_id"], ann["data_path"],
                                                                 scene_subdir=ann.get("scene_subdir"))
            if not available:
                _ds_name = config.get("_dataset_name", "unknown")
                _scene_id = ann.get("scene_id", "?")
                if reason and reason.startswith("missing_root:"):
                    skipped_missing_root += 1
                    missing_root = reason.split(":", 1)[1]
                    if missing_root not in self._missing_root_warnings:
                        print(f"[data] data root not found: {missing_root}")
                        self._missing_root_warnings.add(missing_root)
                else:
                    skipped_missing_geometry += 1
                ds_skips = self._skipped_scenes_by_dataset.setdefault(_ds_name, {})
                ds_skips[_scene_id] = reason or "unknown"
                self._skipped_count_by_dataset[_ds_name] = self._skipped_count_by_dataset.get(_ds_name, 0) + 1
                continue

            ann["dataset_name"] = config.get("_dataset_name", "unknown")
            # Drop raw annotation fields no longer needed to save memory.
            # EXCEPTION: a toolcall_marker episode rebuilds its whole multi-turn
            # place_marker/python sequence at __getitem__ time, so it must keep
            # the raw conversation + marker events (and the eval/identity-check
            # metadata) through the prune. These records are small and rare
            # (~28k), so the memory the global prune protects is untouched; QA
            # items still drop conversations.
            _keep = _KEEP_FIELDS
            if ann.get("question_type") == "toolcall_marker_points":
                _keep = _KEEP_FIELDS | {
                    "conversations", "marker_events", "gt_points",
                    "gt_points_alt", "toolcall_kind", "episode_kind", "arms",
                    "scene_name",
                    # Selects the <tools> block in _toolcall_marker_inputs.
                    # Dropping it silently rendered delta-grammar episodes with
                    # the TWO-tool prompt: the model would be supervised to emit
                    # adjust_marker while its own system message never declared
                    # that op exists, and at inference the op would be
                    # undeclared. The transcript looks perfect either way, which
                    # is why this needed a gate rather than a read-through
                    # (agentic scripts/check_real_marker_delta_sample.py).
                    "delta_grammar"}
            for k in list(ann.keys()):
                if k not in _keep:
                    del ann[k]
            self.list_data_dict.append(ann)

        self._skipped_missing_root += skipped_missing_root
        self._skipped_missing_geometry += skipped_missing_geometry

        ds_name = config.get("_dataset_name", "unknown")
        n_loaded = len(self.list_data_dict) - _start_idx
        print(f"[data] {ds_name}: {n_loaded} items loaded ({self.data_split})")
        if _qt_filter is not None:
            print(f"[data] {ds_name}: question_type_filter="
                  f"{sorted(_qt_filter)} kept {n_loaded}, dropped {_qt_dropped}")
        if dataset_weight is not None and n_loaded > 0:
            self.sample_weights.extend([dataset_weight / n_loaded] * n_loaded)
            self._using_weighted_sampling = True
        elif n_loaded > 0 and self._using_weighted_sampling:
            # Dataset has no @N weight but others in this run do — treat as @1.
            self.sample_weights.extend([1.0 / n_loaded] * n_loaded)
            print(f"[data] {ds_name}: no @weight given, defaulting to @1 in weighted run")

    def __len__(self):
        return len(self.list_data_dict)

    def _reprojection_config(self):
        """Gather adapter config from dataset attributes into a plain dict."""
        return {
            "rope_pos_range": self._rope_pos_range,
            "temporal_max_range": self._temporal_max_range,
            "temporal_raw_frame_index": self._temporal_raw_frame_index,
            "depth_embed_mode": self._depth_embed_mode,
            "depth_embed_min": self._depth_embed_min,
            "depth_downsample": int(getattr(self, "_depth_downsample", 1) or 1),
            "inline_patch_override_rope": bool(
                getattr(self, "_inline_patch_override_rope", True)),
            "image_pad_token_id": self.adapter.config.image_pad_token_id,
            "curriculum_zero_visual": bool(getattr(self, "_probe_zero_visual", False)),
            "curriculum_single_patch_canvas": bool(getattr(self, "_probe_single_patch_canvas", False)),
        }

    def _load_single_frame_assets(self, item):
        """Assets for a generic RGB-D QA item (see SINGLE_FRAME_QA).

        The frame(s) + geometry travel in the annotation, so there is no scene
        dir to resolve. Two item shapes are supported:
          * single-frame: top-level ``rgb/depth/pose_c2w/intrinsics/image_wh``
          * multi-frame: ``frames``, a list of dicts with those same five keys
            (a pan scan, a short clip, ...). Everything downstream is already
            list-based, so N frames need no other change.
        Returns the same asset dict shape as ``_load_scene_data``'s
        precomputed-geometry path. Depth is stored in the annotation in METRES
        and returned in MILLIMETRES to satisfy the reprojection's API-wide
        ``depth / 1000`` contract (matching ``_use_gt_all``). Origin is left
        implicit: ``get_scene_center`` is the mean camera position, so for one
        frame -- or a fixed-position scan whose frames share one camera
        position -- a world-coord grounding answer is re-centered to the camera.
        """
        frames = item.get("frames") or [
            fr for obs in (item.get("observations") or []) for fr in obs["frames"]
        ] or [item]
        poses, intrs, dims, images, depths = [], [], [], [], []
        for fr in frames:
            img = Image.open(fr["rgb"]).convert("RGB")
            # `.npz` is the deflate-compressed form (key "depth"), bit-exact
            # and 6.6x smaller than the `.npy` for a 512x384 float16 frame.
            if str(fr["depth"]).endswith(".npz"):
                with np.load(fr["depth"]) as saved:
                    depth_m = saved["depth"].astype(np.float32)
            else:
                depth_m = np.load(fr["depth"]).astype(np.float32)
            W, H = int(fr["image_wh"][0]), int(fr["image_wh"][1])
            if (img.width, img.height) != (W, H):
                img = img.resize((W, H), Image.LANCZOS)
            intrinsics = fr["intrinsics"]
            cap = getattr(self, "_observation_max_resolution", None)
            if cap is not None:
                # THE OBSERVATION CAP (`observation_max_resolution`). A 1280x960
                # photo is 1200 canvas tokens; a 14-photo trace at that size did
                # not fit a 48 GB GPU (2026-09-26). Pixels scale about the corner
                # origin reproject_scene uses, so fx, fy, cx, cy scale exactly.
                s = min(cap[0] / W, cap[1] / H, 1.0)
                if s < 1.0:
                    nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
                    img = img.resize((nw, nh), Image.LANCZOS)
                    if depth_m.shape != (H, W):
                        raise ValueError(
                            f"depth {depth_m.shape} does not match image_wh {W}x{H}")
                    depth_m = np.asarray(Image.fromarray(depth_m, mode="F").resize(
                        (nw, nh), Image.NEAREST), dtype=np.float32)
                    sx, sy = nw / W, nh / H
                    if len(intrinsics) != 4 or isinstance(intrinsics[0], list):
                        raise ValueError(
                            f"observation intrinsics {intrinsics} are not fx,fy,cx,cy")
                    fx, fy, cx, cy = (float(v) for v in intrinsics)
                    intrinsics = [fx * sx, fy * sy, cx * sx, cy * sy]
            poses.append(torch.tensor(fr["pose_c2w"], dtype=torch.float32))
            intrs.append(torch.tensor(intrinsics, dtype=torch.float32))
            dims.append(img.size)
            images.append(img)
            depths.append(torch.from_numpy(depth_m * 1000.0))
        return {
            "poses": poses,
            "intrinsics": intrs,
            "image_dims": dims,
            "images": images,
            "depths": depths,
            "features": None,
        }

    def _load_scene_data(self, scene_id, data_path, sample_idx=0, pinned_frames=None,
                         scene_subdir=None, wants_aligned=False, dataset_name=None,
                         frame_rng=None):
        self._last_load_failure = None
        self._last_invalid_frames = []
        self._last_recovered_assets = []
        scene_dir = self._resolve_scene_dir(scene_id, data_path, scene_subdir=scene_subdir)
        if scene_dir is None:
            self._last_load_failure = f"scene_dir_not_found:{data_path}/{scene_id}"
            return None
        self._last_scene_dir = scene_dir

        # Per-scene gt_all override: scenes with known-broken GT poses
        # (ARKit/BundleFusion divergence) fall back to DA3-predicted poses
        # even when --use_gt_all is set. See DA3_FALLBACK_SCENES above.
        _use_gt_all = self._use_gt_all and scene_id not in DA3_FALLBACK_SCENES
        if self._use_gt_all and not _use_gt_all and scene_id not in self._da3_fallback_logged:
            self._da3_fallback_logged.add(scene_id)
            rank0_print(
                f"[data] scene {scene_id} is in DA3_FALLBACK_SCENES "
                f"(GT poses broken); using DA3 .pt poses instead."
            )


        # --- GT calibration (use_gt_all only) ---
        # Loaded BEFORE the .pt so that, when use_gt_all + no precomputed
        # features, we can skip the .pt entirely and pull frame keys from
        # gt_calib["poses"]. Scenes without GT assets fall back to .pt.
        gt_calib: Optional[dict] = None
        if _use_gt_all:
            gt_calib = self._load_gt_scene_calib(
                scene_id, scene_dir, wants_aligned=wants_aligned,
                dataset_name=dataset_name,
            )
            if gt_calib is None and self._gt_strict:
                if scene_id not in self._gt_strict_warned:
                    self._gt_strict_warned.add(scene_id)
                    rank0_print(
                        f"[gt_strict] WARNING: scene {scene_id} has no GT calib "
                        f"(scene_dir={scene_dir}); skipping instead of falling back "
                        f"to DA3."
                    )
                self._last_load_failure = f"gt_strict_no_calib:{scene_id}"
                return None

        _skip_pt = gt_calib is not None

        _geo_candidates = self._geometry_candidates(wants_aligned)
        poses_path = None
        _scene_crop_ratio = 0.0
        _picked_aligned = False
        if not _skip_pt:
            for _fname, _crop in _geo_candidates:
                _candidate = os.path.join(scene_dir, _fname)
                if os.path.exists(_candidate):
                    poses_path = _candidate
                    _scene_crop_ratio = _crop
                    _picked_aligned = "_aligned" in _fname
                    break
            if self._predicted_geometry and poses_path is not None \
                    and not getattr(self, "_predicted_geometry_logged", False):
                self._predicted_geometry_logged = True
                rank0_print(
                    f"[predicted_geometry] {self._predicted_geometry}: loading "
                    f"'{os.path.basename(poses_path)}' (crop={_scene_crop_ratio}) "
                    f"e.g. scene {scene_id}; use_gt_all={self._use_gt_all} "
                    f"use_gt_depth={self._use_gt_depth}"
                )
            if poses_path is None and gt_calib is None:
                self._last_load_failure = f"no_geometry_file:{scene_dir}"
                return None
            if (poses_path is not None and gt_calib is None and self._use_gt_all
                    and not self._predicted_geometry):
                _warned = self.__dict__.setdefault("_predicted_fallback_logged", set())
                if scene_id not in _warned:
                    _warned.add(scene_id)
                    rank0_print(
                        f"[geometry] {scene_id}: no ground-truth calibration, using "
                        f"predicted geometry from {os.path.basename(poses_path)}")
            if (poses_path is None and gt_calib is not None):
                # No .pt but we have GT — proceed via skip-pt path.
                _skip_pt = True
            elif wants_aligned and not _picked_aligned and gt_calib is None:
                # Grounding sample resolved to a non-aligned geometry file
                # WITHOUT use_gt_all override; bbox/feature frames will not
                # match. Warn loudly (deduped).
                if not hasattr(self, "_grounding_fallback_warned"):
                    self._grounding_fallback_warned = set()
                if scene_id not in self._grounding_fallback_warned:
                    self._grounding_fallback_warned.add(scene_id)
                    rank0_print(
                        f"[grounding] WARNING: scene {scene_id} has no _aligned.pt; "
                        f"falling back to {os.path.basename(poses_path)}. "
                        f"This sample's coordinate frame will not match the GT bbox. "
                        f"Regenerate _aligned.pt for this scene to fix."
                    )

        if _skip_pt:
            # Pure GT path, no .pt at all. Frame keys come from GT pose dict.
            geometry_data = {}
            all_keys = self._sort_frame_keys(list(gt_calib["poses"].keys()))
        else:
            # Use mmap so the OS only faults in pages actually accessed,
            # avoiding a full file read on every __getitem__ call.
            geometry_data = _load_pt_mmap(poses_path)
            all_keys = self._sort_frame_keys(
                [k for k in geometry_data.keys() if not k.startswith("__")]
            )

        if len(all_keys) == 0:
            _src = poses_path if not _skip_pt else "GT pose dict"
            rank0_print(f"[data] WARNING: 0 usable keys for {scene_id} (src={_src})")
            self._last_load_failure = f"no_usable_keys:{scene_id}(src={_src})"
            del geometry_data
            return None

        # Filter out frames with invalid (NaN/inf) poses before sampling. On
        # ScanNet ~2% of frames overall have BundleFusion tracking failures,
        # and up to 46% on the worst scenes — sampling from the raw pool
        # silently turns those into dead slots (the frame contributes no 3D
        # points but still consumes encoder compute).
        _raw_n = len(all_keys)
        if gt_calib is not None:
            # use_gt_all: keep only keys with a valid GT pose. The calib loader
            # has already filtered non-finite / malformed pose files.
            _valid_gt = set(gt_calib["poses"].keys())
            all_keys = [k for k in all_keys if k in _valid_gt]
        else:
            all_keys = [
                k for k in all_keys
                if k in geometry_data
                and "pose" in geometry_data[k]
                and torch.is_tensor(geometry_data[k]["pose"])
                and torch.isfinite(geometry_data[k]["pose"]).all()
            ]
        if len(all_keys) < _raw_n and not hasattr(self, "_bad_pose_warned"):
            self._bad_pose_warned = set()
        if len(all_keys) < _raw_n and scene_id not in self._bad_pose_warned:
            self._bad_pose_warned.add(scene_id)
            _dropped = _raw_n - len(all_keys)
            if _dropped / _raw_n > 0.2:
                rank0_print(
                    f"[data] scene {scene_id}: {_dropped}/{_raw_n} frames "
                    f"({100*_dropped/_raw_n:.1f}%) dropped for NaN/inf poses"
                )
        if len(all_keys) == 0:
            rank0_print(f"[data] WARNING: all poses invalid for {scene_id}, skipping")
            self._last_load_failure = f"all_poses_invalid:{scene_id}"
            del geometry_data
            return None

        # Numerical-observation consumers need the actual valid source-frame
        # population, not the 32-frame visual sample.  This is provenance only.
        # The list is never exposed to the model prompt.
        self._last_all_frame_keys = list(all_keys)

        resized_image_source, full_image_source = self._resolve_image_sources(scene_dir)
        if full_image_source is None:
            self._last_load_failure = f"no_images:{scene_dir}"
            del geometry_data
            return None
        total_frames = len(all_keys)

        # If caller pinned specific frames (e.g. SPBench-SI), use those directly.
        if pinned_frames:
            geometry_key_set = set(str(k) for k in all_keys)
            matched = []
            missing_pins = []
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            for requested in pinned_frames:
                key = os.path.basename(str(requested))
                if key not in geometry_key_set:
                    stem, ext = os.path.splitext(key)
                    if ext.lower() in image_extensions and stem in geometry_key_set:
                        key = stem
                if key in geometry_key_set:
                    if key not in matched:
                        matched.append(key)
                else:
                    missing_pins.append(str(requested))
            if not matched:
                self._last_load_failure = (
                    f"no_pinned_frames_resolved:{scene_id}:"
                    f"{missing_pins[:8]}")
                del geometry_data
                return None
            selected_keys = matched
        elif total_frames > self.num_images:
            # SpaceMind-style: sample num_images + 2 buckets uniformly and drop
            # the first/last, which on ScanNet-like sequences tend to be device
            # initialization (pointed at floor, pose not yet stable) or the
            # operator ending the scan.
            # Full-span sampler: pick num_images frames evenly across the full
            # sequence (np.linspace(0, T-1, num_images), so frame 0 and frame
            # T-1 are both included). This is now the DEFAULT for every run
            # (training and eval) so train-time and eval-time framing match.
            # Set ONECANVAS_FULLSPAN_SAMPLER=0 to fall back to the legacy
            # +2-buckets / drop-ends SpaceMind sampler.
            _fullspan = os.environ.get("ONECANVAS_FULLSPAN_SAMPLER", "1") == "1"
            _pad = 0 if _fullspan else 2
            if self.select_images_randomly:
                # Draw one frame per temporal interval. Curriculum callers pass
                # their per-draw RNG so the selection reproduces from the
                # dataset seed and item index. Other callers retain the prior
                # process RNG behavior.
                _rng = frame_rng if frame_rng is not None else random
                boundaries = np.linspace(0, total_frames, self.num_images + _pad + 1, dtype=int)
                selected_keys = []
                _lo = 0 if _fullspan else 1
                _hi = self.num_images if _fullspan else self.num_images + 1
                for i in range(_lo, _hi):
                    start = boundaries[i]
                    end = boundaries[i + 1]
                    idx = _rng.randint(start, end - 1)
                    selected_keys.append(all_keys[idx])
            elif not self.with_precomputed_geometry:
                image_dir, image_ext = full_image_source
                all_f = glob.glob(os.path.join(image_dir, f"*{image_ext}"))
                if len(all_f) == 0:
                    self._last_load_failure = f"no_image_files:{image_dir}"
                    return None
                stems = self._sort_frame_keys(
                    [os.path.splitext(os.path.basename(p))[0] for p in all_f]
                )
                _idx_full = np.linspace(0, len(stems) - 1, num=self.num_images + _pad, dtype=int)
                indices = _idx_full if _fullspan else _idx_full[1:-1]
                selected_keys = [stems[idx] for idx in indices]
            else:
                _idx_full = np.linspace(0, len(all_keys) - 1, num=self.num_images + _pad, dtype=int)
                indices = _idx_full if _fullspan else _idx_full[1:-1]
                selected_keys = [all_keys[idx] for idx in indices]


        else:
            selected_keys = all_keys

        scene_dims, scene_pil_images = [], []
        scene_depths, scene_intrinsics = [], []
        loop_keys = selected_keys

        # Prefer the pre-resized 640×480 colour images when available (controlled by use_resized_images flag).
        _use_resized = self._use_resized_images and resized_image_source is not None
        self._check_frame_resolution(
            scene_dir, resized_image_source if _use_resized else full_image_source)

        # ARKit gravity uprighting: one trajectory read per scene, not per
        # frame. 0 for every non-ARKit scene, and for ARKit scenes already
        # shot upright, in which case nothing below fires at all.
        _upright_k = self._arkit_upright_k(scene_dir)
        # Stashed with `_last_frame_files`: the files on disk are in the
        # ORIGINAL orientation and the poses and intrinsics returned below are
        # upright, so a rebuild from those files must turn them the same way.
        self._last_upright_k = int(_upright_k or 0)

        # Determine the original image dimensions for intrinsics adjustment.
        # The hardcoded _ORIG_W/_ORIG_H are ScanNet-specific (1296×968). For other
        # datasets (e.g. ScanNet++ at 1920×1440) we peek at the first full-res image
        # so the resize scale factor is correct.
        _scene_orig_w, _scene_orig_h = _ORIG_W, _ORIG_H
        if _use_resized and full_image_source is not None:
            _full_dir, _full_ext = full_image_source
            _full_samples = sorted(glob.glob(os.path.join(_full_dir, f"*{_full_ext}")))
            if _full_samples:
                try:
                    with Image.open(_full_samples[0]) as _peek:
                        _scene_orig_w, _scene_orig_h = _peek.size
                except Exception:
                    pass

        # The (key, colour file, depth file) triples this call actually OPENED,
        # in frame order, stashed for provenance readers the way
        # `_last_scene_dir` is. A rebuild that re-derives the files from the
        # key cannot get ARKitScenes right: the key is a lowres_depth
        # timestamp and the colour frame is the NEAREST vga_wide one, matched
        # below, so an exact stem lookup finds 3 of 32.
        self._last_frame_files = []
        loaded_keys = []

        def _canonical_candidates(path):
            """Read-only mirrors for recovery, never another scene or frame."""
            roots = [p for p in os.environ.get(
                "ONECANVAS_CANONICAL_DATA_ROOTS", "").split(os.pathsep) if p]
            try:
                rel = os.path.relpath(path, os.path.abspath(data_path))
            except ValueError:
                rel = None
            out = []
            for root in roots:
                if rel and not rel.startswith(".."):
                    out.append(os.path.join(root, rel))
                out.append(os.path.join(root, scene_id,
                                        os.path.relpath(path, scene_dir)))
            return list(dict.fromkeys(p for p in out
                                      if os.path.abspath(p) != os.path.abspath(path)))

        def _open_rgb(path, key):
            attempts = [path]
            # A corrupt resized cache is recoverable from the full-resolution
            # source for the same key. It is resized below with matching K.
            if _use_resized and full_image_source is not None:
                fdir, fext = full_image_source
                full = self._resolve_frame_image_path(fdir, fext, key)
                if full and full not in attempts:
                    attempts.append(full)
            attempts.extend(_canonical_candidates(path))
            errors = []
            for candidate in attempts:
                if not candidate or not os.path.isfile(candidate):
                    continue
                try:
                    image = Image.open(candidate).convert("RGB")
                    if candidate != path:
                        self._last_recovered_assets.append({
                            "frame_key": str(key), "kind": "rgb",
                            "failed_asset": path, "recovered_asset": candidate,
                        })
                    return image, candidate
                except Exception as exc:
                    errors.append(f"{candidate}:{type(exc).__name__}:{exc}")
            self._last_invalid_frames.append({
                "frame_key": str(key), "kind": "rgb",
                "failed_asset": path, "errors": errors or ["asset missing"],
            })
            return None, path

        def _open_depth(path, key):
            errors = []
            for candidate in [path] + _canonical_candidates(path):
                if not candidate or not os.path.isfile(candidate):
                    continue
                try:
                    value = np.array(Image.open(candidate))
                    if value.dtype != np.uint16:
                        raise ValueError(
                            f"sensor depth PNG must be uint16 millimetres, got "
                            f"{value.dtype}")
                    if candidate != path:
                        self._last_recovered_assets.append({
                            "frame_key": str(key), "kind": "depth",
                            "failed_asset": path, "recovered_asset": candidate,
                        })
                    return value, candidate
                except Exception as exc:
                    errors.append(f"{candidate}:{type(exc).__name__}:{exc}")
            self._last_invalid_frames.append({
                "frame_key": str(key), "kind": "depth",
                "failed_asset": path, "errors": errors or ["asset missing"],
            })
            return None, path

        for key in loop_keys:
            _depth_file = None
            if _use_resized:
                image_dir, image_ext = resized_image_source
            else:
                image_dir, image_ext = full_image_source

            img_path = self._resolve_frame_image_path(image_dir, image_ext, key)
            if img_path is None and _use_resized and full_image_source is not None:
                image_dir, image_ext = full_image_source
                img_path = self._resolve_frame_image_path(image_dir, image_ext, key)
            if img_path is None:
                continue

            img, img_path = _open_rgb(img_path, key)
            if img is None:
                continue

            # --- On-the-fly resize ---
            # If max_image_resolution is set and the loaded image doesn't match,
            # resize (up or down) while preserving aspect ratio. Intrinsics will
            # be rescaled below (for the non-gt-all path; gt_all derives intrinsics
            # from the final img.width / depth_w ratio so needs no extra step).
            _pre_cap_w, _pre_cap_h = img.width, img.height
            if self._max_image_resolution is not None:
                _mw, _mh = self._max_image_resolution
                if img.width != _mw or img.height != _mh:
                    _scale = min(_mw / img.width, _mh / img.height)
                    # Under 'cap' the configured size is a ceiling, so a source
                    # already below it is kept: an upsample manufactures no
                    # evidence and costs 4x the canvas tokens to say so.
                    if self._image_resolution_policy == "cap":
                        _scale = min(_scale, 1.0)
                    if _scale != 1.0:
                        _new_w = max(1, int(round(img.width * _scale)))
                        _new_h = max(1, int(round(img.height * _scale)))
                        img = img.resize((_new_w, _new_h), Image.LANCZOS)

            # --- Depth loading ---
            # Default: DAv3 depth from the geometry .pt file.
            # use_gt_depth: load sensor depth from <depth_subdir>/{key}.png.
            # The subdir is "depth" for ScanNet / ScanNet++, "lowres_depth" for
            # ARKitScenes; comes from the GT-calib dict, with fallback to "depth".
            _depth_subdir = (gt_calib.get("depth_subdir", "depth")
                             if gt_calib is not None else "depth")
            if self._use_gt_depth:
                _gt_depth_path = os.path.join(scene_dir, _depth_subdir, f"{key}.png")
                _depth_available = (os.path.exists(_gt_depth_path) or any(
                    os.path.exists(p) for p in
                    _canonical_candidates(_gt_depth_path)))
                if _depth_available:
                    _depth_file = _gt_depth_path
                    # Sensor depth PNGs are uint16 MILLIMETRES by dataset
                    # contract (ScanNet `depth/`, ScanNet++ iPhone `depth/`,
                    # ARKitScenes `lowres_depth/`). Keep that source unit on
                    # every pose path: `compute_scene_geometry` owns the one
                    # and only mm -> m conversion. Previously the DA3-pose
                    # fallback and plain `use_gt_depth` path divided here as
                    # well, so reprojection divided a second time and dropped
                    # the entire cloud as sub-millimetre geometry.
                    _gt_arr, _depth_file = _open_depth(_gt_depth_path, key)
                    if _gt_arr is None:
                        continue
                    depth = torch.from_numpy(_gt_arr.astype(np.float32))
                else:
                    depth = geometry_data[key]["depth"] if key in geometry_data else None
            else:
                depth = geometry_data[key]["depth"] if key in geometry_data else None
            if _use_gt_all and gt_calib is not None and depth is not None:
                # Pick per-frame K (ARKit/ScanNet++) or scene-level depth_K (ScanNet).
                _K_pf = gt_calib.get("intrinsics_per_frame")
                if _K_pf is not None and key in _K_pf:
                    K_d = _K_pf[key]
                else:
                    K_d = gt_calib.get("depth_K")
                if K_d is None:
                    intrinsics = geometry_data[key]["intrinsics"] if key in geometry_data else None
                else:
                    # Scale K so (u,v) in RGB-image space maps through the sensor's
                    # focal length. Lift interpolates depth to image dims.
                    d_h, d_w = int(depth.shape[-2]), int(depth.shape[-1])
                    sxd = img.width / d_w
                    syd = img.height / d_h
                    intrinsics = torch.stack([K_d[0] * sxd, K_d[1] * syd,
                                               K_d[2] * sxd, K_d[3] * syd])
            else:
                intrinsics = geometry_data[key]["intrinsics"] if key in geometry_data else None

            # When using the pre-resized images the stored intrinsics must be scaled
            # to match the new pixel space. _scene_orig_w/h is derived from the actual
            # full-res images (not the hardcoded ScanNet 1296×968 constant) so that
            # datasets with different native resolutions (e.g. ScanNet++ at 1920×1440)
            # are handled correctly.
            if _use_resized and intrinsics is not None and not (_use_gt_all and gt_calib is not None):
                intrinsics = adjust_intrinsics_for_resize(
                    intrinsics, _scene_orig_w, _scene_orig_h, img.width, img.height
                )
            elif (not _use_resized) and intrinsics is not None \
                    and not (_use_gt_all and gt_calib is not None) \
                    and (img.width != _pre_cap_w or img.height != _pre_cap_h):
                # Loaded from full-res then capped on the fly: intrinsics were
                # stored in pre-cap (full-res) pixel coords, so rescale.
                intrinsics = adjust_intrinsics_for_resize(
                    intrinsics, _pre_cap_w, _pre_cap_h, img.width, img.height
                )

            # Always crop the RGB image to match the resolution used during precomputation.
            # For on-the-fly geometry (not with_precomputed_geometry), depth and intrinsics
            # were computed from the uncropped image and need cropping too.
            # For precomputed geometry, depth and intrinsics are already saved at cropped
            # resolution — UNLESS we loaded GT depth from disk (uncropped sensor depth).
            if _scene_crop_ratio > 0:
                _img_w_pre, _img_h_pre = img.width, img.height
                left, top, right, bottom = self._compute_border_crop_box(_img_w_pre, _img_h_pre)
                img = img.crop((left, top, right, bottom))
                if self._use_gt_depth and depth is not None:
                    # GT depth is at sensor resolution (e.g. 480x640). The crop
                    # box is in resized-RGB space — scale it by depth_size /
                    # img_pre_crop_size (NOT by depth_size / orig_full_res,
                    # which is the old bug).
                    _dh, _dw = depth.shape[-2], depth.shape[-1]
                    if (_dh, _dw) != (_img_h_pre, _img_w_pre):
                        _sy, _sx = _dh / _img_h_pre, _dw / _img_w_pre
                        depth = crop_depth_map(
                            depth,
                            int(round(left * _sx)), int(round(top * _sy)),
                            int(round(right * _sx)), int(round(bottom * _sy)),
                        )
                    else:
                        depth = crop_depth_map(depth, left, top, right, bottom)
                    # DA3 intrinsics from the .pt are already post-crop (no
                    # adjustment). use_gt_all uses full-FoV K → shift cx/cy.
                    if _use_gt_all and gt_calib is not None and intrinsics is not None:
                        intrinsics = intrinsics.clone()
                        intrinsics[2] = intrinsics[2] - left
                        intrinsics[3] = intrinsics[3] - top
                elif not self.with_precomputed_geometry:
                    depth = crop_depth_map(depth, left, top, right, bottom)
                    intrinsics = adjust_intrinsics_for_crop(intrinsics, left, top)

            # In use_gt_all mode there is no .pt fallback — if depth or
            # intrinsics are still None this frame is unusable.
            if _use_gt_all and (depth is None or intrinsics is None):
                rank0_print(
                    f"[data] WARNING: gt_all could not resolve "
                    f"{'depth' if depth is None else 'intrinsics'} for "
                    f"frame {key} in scene {scene_id} — skipping scene")
                self._last_load_failure = f"gt_all_missing_{'depth' if depth is None else 'intrinsics'}:{scene_id}/{key}"
                return None

            # --- ARKit gravity uprighting ---
            # Rotate image, depth and intrinsics by the same k clockwise
            # quarter turns (the pose is rotated once per scene, below).
            # Applied LAST, after the on-the-fly resize and the border crop, so
            # both of those keep operating in the frame they were written for
            # and no orientation-sensitive branch upstream has to change. The
            # intrinsics here are already in final image-pixel coordinates on
            # every path, which is exactly the space the transform expects.
            # With all four rotating together the lifted world points are
            # unchanged — asserted in arkit_upright._self_test.
            if _upright_k:
                from onecanvas.data.arkit_upright import (
                    intrinsics_orig_to_upright, rotate_array_cw, rotate_image_cw)
                _iw, _ih = img.width, img.height
                img = rotate_image_cw(img, _upright_k)
                if depth is not None:
                    depth = rotate_array_cw(depth, _upright_k)
                if intrinsics is not None:
                    _fx, _fy, _cx, _cy, _, _ = intrinsics_orig_to_upright(
                        float(intrinsics[0]), float(intrinsics[1]),
                        float(intrinsics[2]), float(intrinsics[3]),
                        _iw, _ih, _upright_k)
                    intrinsics = torch.tensor(
                        [_fx, _fy, _cx, _cy],
                        dtype=getattr(intrinsics, "dtype", torch.float32))

            scene_pil_images.append(img)
            scene_dims.append(img.size)
            scene_depths.append(depth)
            scene_intrinsics.append(intrinsics)
            loaded_keys.append(key)
            self._last_frame_files.append((str(key), img_path, _depth_file))

        if len(scene_pil_images) == 0:
            rank0_print(f"[data] WARNING: 0 images loaded for scene {scene_id} "
                        f"(selected_keys={len(selected_keys)}, scene_dir={scene_dir})")
            self._last_load_failure = f"no_images_loaded:{scene_id}(dir={scene_dir})"
            return None

        # Optionally load precomputed visual features (saved by features_saving.py).
        # Shape on disk per key: [NumLayers, H_feat, W_feat, C]
        # After stacking selected frames: [N_images, NumLayers, H_feat, W_feat, C]
        if not self.with_precomputed_geometry:
            del geometry_data  # release mmap reference so /dev/shm is not held by worker
            return {
                "image_dims": scene_dims,
                "images": scene_pil_images,
                "features": None,
            }
        else:
            if _use_gt_all and gt_calib is not None:
                poses = [gt_calib["poses"][key] for key in loaded_keys]
            else:
                poses = [geometry_data[key]["pose"] for key in loaded_keys]

            # ARKit gravity uprighting, pose half: cam->world in the original
            # image frame -> the upright one. pose_up = pose_orig @ M.T, which
            # exactly cancels the axis map the rotated pixels introduce, so
            # world points stay put. Builds new tensors, so the gt_calib pose
            # cache is never mutated.
            if _upright_k:
                from onecanvas.data.arkit_upright import pose_orig_to_upright
                poses = [pose_orig_to_upright(p, _upright_k) for p in poses]

            del geometry_data  # release mmap reference so /dev/shm is not held by worker
            # Stashed beside `_last_scene_dir` for provenance readers that only
            # see the finished sample (the assets dict never reaches it). A
            # colour rebuild of the canvas's own frames has to pair these keys
            # with disk stems; re-deriving the selection from the directory
            # listing instead diverged on ARKitScenes and ScanNet++, where
            # `all_keys` is the geometry file's key list and not the listing.
            if not (len(loaded_keys) == len(scene_pil_images)
                    == len(scene_depths) == len(scene_intrinsics) == len(poses)):
                raise RuntimeError(
                    f"unaligned scene inputs for {scene_id}: keys={len(loaded_keys)} "
                    f"rgb={len(scene_pil_images)} depth={len(scene_depths)} "
                    f"intrinsics={len(scene_intrinsics)} poses={len(poses)}"
                )
            self._last_frame_keys = list(loaded_keys)
            return {
                "poses": poses,
                "intrinsics": scene_intrinsics,
                "image_dims": scene_dims,
                "images": scene_pil_images,
                "depths": scene_depths,
                "features": None,
                # For predicted-geometry anchor re-expression (SQA3D agent
                # pose): the selected frame stems and scene dir let the anchor
                # code pair GT pose txts with the predicted poses above.
                "frame_keys": list(loaded_keys),
                "scene_dir": scene_dir,
            }


    def _marker_pool(self):
        """Lazy-load + cache the pinned marker pool (sha1-asserted at first
        use, work order section 3). Marker features are tower outputs of
        rendered patterns, never the OBB stash and never raw noise."""
        if getattr(self, "_marker_feats", None) is None:
            from onecanvas.data.marker_paste import load_marker_pool
            # TOOLMARK_BADGE_GLYPHS is the pre-rename (2026-07-30) env name;
            # in-flight jobs 2842471/2842466 still export it. Drop after.
            path = (os.environ.get("TOOLMARK_MARKER_POOL")
                    or os.environ.get("TOOLMARK_BADGE_GLYPHS"))
            if not path:
                raise RuntimeError(
                    "TOOLMARK_MARKER_POOL is unset; the toolcall_marker loader "
                    "needs the pinned marker pool.")
            self._marker_feats, self._marker_pool_sha1 = load_marker_pool(path)
        return self._marker_feats

    def _declared_feature_pool(self, spec):
        """Load and cache one DECLARED feature pool, sha1-asserted at first use.

        A ``[N, N_layers, D]`` tensor addressed BY ROW. The item names the file,
        its sha1 and its shape, and this refuses anything else rather than
        loading whatever is at the path: a pool that silently changed under a
        dataset would leave every row's feature bound to different content than
        the row was generated with, which is exactly the drift the row's own
        declaration exists to prevent.
        """
        import torch

        path = str(spec.get("path") or "")
        if not path:
            raise ValueError("a declared feature pool carries no path")
        cache = getattr(self, "_declared_pools", None)
        if cache is None:
            cache = self._declared_pools = {}
        if path not in cache:
            from onecanvas.data.marker_paste import stash_file_sha1
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"declared feature pool missing at {path}. This loader "
                    f"does not build one and does not substitute another: a "
                    f"row whose features come from a different pool than it "
                    f"was generated with is silently wrong.")
            want_sha1 = str(spec.get("sha1") or "")
            got_sha1 = stash_file_sha1(path)
            if want_sha1 and got_sha1 != want_sha1:
                raise ValueError(
                    f"declared feature pool {path} hashes {got_sha1} and the "
                    f"dataset was built against {want_sha1}")
            blob = torch.load(path, map_location="cpu", weights_only=False)
            rows = blob["patches"] if isinstance(blob, dict) else blob
            cache[path] = rows
        rows = cache[path]
        for key, index in (("n_rows", 0), ("n_layers", 1), ("d", 2)):
            want = spec.get(key)
            if want is not None and int(want) != int(rows.shape[index]):
                raise ValueError(
                    f"declared feature pool {path} has {key}="
                    f"{int(rows.shape[index])} and the dataset declares "
                    f"{int(want)}")
        return rows

    def _declared_reference_inputs(self, item, input_ids, labels):
        """Bind an item's DECLARED reference points to its prompt placeholders.

        THE GENERIC HALF OF AN ACTION-SPACE DATASET'S MODEL INPUT. The item
        supplies, in prompt order, a list of points that its own prompt refers
        to by name, each with the row of a declared feature pool that carries
        its appearance. The prompt renders each one as the same two placeholder
        tokens the toolcall-marker path uses -- ``<|object_ref_start|>`` for the
        inline copy and ``<|object_ref_end|>`` for the canvas token -- and this
        converts them to image_pad and emits the ``marker_tok_*`` keys the
        forward hands to ``prepare_batch``.

        THE MAPPING IS NEVER RESAMPLED HERE. The generating process froze which
        pool row belongs to which reference id and wrote it into the row, so
        training reads the same feature, at the same inline slot, at the same
        canvas placement, with the same positional conventions and the same
        auxiliary feature layers as generation did. Drawing a fresh row at load
        time would train on a binding the observation never had.

        THE POINTS ARE IN THE ITEM'S OWN DECLARED CANVAS FRAME and are used
        verbatim. A row that declares reference points must declare
        ``canvas_center`` with them, because a point expressed against one
        origin and a canvas reprojected around another are two different
        places. Nothing here transforms them, so there is no second definition
        of the frame to drift from the first.

        Anything missing REFUSES. A dropped field here is not a degraded
        sample, it is a different sample.
        """
        import torch

        rows = list(item.get("reference_points") or [])
        if not rows:
            raise ValueError(
                f"{item.get('scene_id')!r} declares an empty reference_points "
                f"list; omit the field instead")
        if item.get("canvas_center") is None:
            raise ValueError(
                f"{item.get('scene_id')!r} declares reference points without a "
                f"canvas_center. Their coordinates are expressed in that "
                f"frame, so a row that omits it cannot place them.")
        frame = str(item.get("reference_point_frame") or "canvas")
        if frame != "canvas":
            raise ValueError(
                f"{item.get('scene_id')!r} declares reference_point_frame "
                f"{frame!r}; this loader consumes points already expressed in "
                f"the item's declared canvas frame and transforms nothing")
        pool = self._declared_feature_pool(dict(item.get("feature_pool") or {}))

        points, feats = [], []
        for index, row in enumerate(rows):
            point = row.get("point")
            pool_row = row.get("pool_row")
            if point is None or len(point) != 3 or pool_row is None:
                raise ValueError(
                    f"reference point {index} of {item.get('scene_id')!r} is "
                    f"missing its point or its pool_row: {row}")
            if not (0 <= int(pool_row) < int(pool.shape[0])):
                raise ValueError(
                    f"reference point {row.get('id')} of "
                    f"{item.get('scene_id')!r} names pool row {pool_row}, "
                    f"outside the pool's {int(pool.shape[0])} rows")
            points.append([float(v) for v in point])
            feats.append(pool[int(pool_row)])

        ids0 = input_ids[0].clone()
        img_id = int(self.adapter.config.image_pad_token_id)
        _tk = self.processor.tokenizer
        inline_id = _tk.convert_tokens_to_ids("<|object_ref_start|>")
        canvas_id = _tk.convert_tokens_to_ids("<|object_ref_end|>")
        inline_pos = (ids0 == inline_id).nonzero(as_tuple=True)[0].tolist()
        canvas_pos = (ids0 == canvas_id).nonzero(as_tuple=True)[0].tolist()
        merged = sorted([(int(p), False) for p in inline_pos]
                        + [(int(p), True) for p in canvas_pos])
        want = [flag for _row in rows for flag in (False, True)]
        got = [flag for _p, flag in merged]
        if got != want:
            raise ValueError(
                f"{item.get('scene_id')!r} declares {len(rows)} reference "
                f"point(s), so its prompt must carry that many inline/canvas "
                f"placeholder PAIRS in order. It carries {len(inline_pos)} "
                f"inline and {len(canvas_pos)} canvas placeholders in the "
                f"order {got}. Do not pad or truncate; fix the render.")

        positions = [p for p, _flag in merged]
        for p in positions:
            ids0[p] = img_id
        # SUPERVISE ONLY THE ASSISTANT ANSWER. The placeholders sit in the
        # user turn, which mask_labels already masked, and this states it
        # rather than relying on where the render happened to put them.
        labels0 = labels[0].clone()
        labels0[torch.tensor(positions, dtype=torch.long)] = -100

        pair_feats, pair_points, is_canvas = [], [], []
        for index, (_p, flag) in enumerate(merged):
            pair_feats.append(feats[index // 2])
            pair_points.append(points[index // 2])
            is_canvas.append(bool(flag))
        keys = {
            "marker_tok_positions": torch.tensor(positions, dtype=torch.long),
            "marker_tok_features": torch.stack(pair_feats),
            "marker_tok_is_canvas": torch.tensor(is_canvas, dtype=torch.bool),
            "marker_tok_points": torch.tensor(pair_points,
                                              dtype=torch.float32),
        }
        return ids0.unsqueeze(0), labels0.unsqueeze(0), keys

    def _toolcall_marker_inputs(self, item, aug_center, aug_yaw):
        """Build a whole-episode marker sample (work order section 5): multi-turn
        tokenized inputs with per-turn supervise masking, dummy-image first/last
        idx, and the marker_tok_* keys the forward hands to prepare_batch.

        Coords in every gpt place line AND every marker_events point are
        center/yawed per sample (mirror of the toolcall_points branch). The tool
        response turns carry the marker PAIR (two <|object_ref_start|>
        placeholders, converted to image_pad so masked_scatter fills them). One
        fresh marker per place op, drawn shuffled without replacement.
        """
        import hashlib as _hashlib
        import numpy as _np
        from onecanvas.data.marker_paste import MARKER_TOOLS, MARKER_TOOLS_DELTA

        convs = item["conversations"]
        center = aug_center
        cy = math.cos(aug_yaw) if aug_yaw else 1.0
        sy = math.sin(aug_yaw) if aug_yaw else 0.0
        _pt_re = re.compile(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
            r"\s*(-?\d+(?:\.\d+)?)\s*\]")

        def _xf(x, y, z):
            x = float(x) - float(center[0])
            y = float(y) - float(center[1])
            z = float(z) - float(center[2])
            x, y = cy * x - sy * y, sy * x + cy * y
            return x, y, z

        def _fmt_triple(x, y, z):
            # Render coords EXACTLY as the shared formatter (markers.format_place_call
            # -> json.dumps(round(v, 2))): template-faithful JSON numbers, NOT
            # "%.2f" (which prints trailing zeros the chat template's tojson drops
            # and the rollout driver never emits). One rendering style across
            # builder / loader / rollout so BC targets and decode inputs match.
            return "[" + ", ".join(json.dumps(round(float(v), 2))
                                   for v in (x, y, z)) + "]"

        def _xf_text(m):
            return _fmt_triple(*_xf(m.group(1), m.group(2), m.group(3)))

        def _adjust_value(t):
            """Render an ``adjust_marker`` turn, deriving the delta HERE.

            A DELTA IS A VECTOR, NOT A POINT. The generic ``_pt_re.sub(_xf_text,
            ...)`` below rewrites every bracketed triple in a gpt turn through
            ``_xf``, which SUBTRACTS the frame centre -- correct for a position,
            catastrophic for a difference (wrong by whole metres, silently, and
            only on corrected episodes). So adjust turns never go through that
            path: the annotation stores the two ENDPOINTS and the delta is the
            difference of their transformed values, which picks up the rotation
            and can never pick up the translation.

            Differenced AFTER rounding, on purpose: both endpoints print at 2 dp,
            so this guarantees
                placed_as_printed + delta_as_printed == gt_as_printed
            exactly. Differencing in the raw frame and rounding afterwards
            instead leaves up to 1 cm against the printed target, which is half
            the readout error this track measures.
            """
            frm = [round(v, 2) for v in _xf(*t["adjust_from"])]
            to = [round(v, 2) for v in _xf(*t["adjust_to"])]
            delta = [round(to[i] - frm[i], 2) for i in range(3)]
            body = json.dumps({"name": "adjust_marker", "arguments": {
                "marker_id": t["marker_id"], "delta": delta}})
            call = f"<tool_call>\n{body}\n</tool_call>"
            # Decomposed-read correction (rung-1 rebuild): flagged per turn BY
            # THE BUILDER, so historical delta data renders byte-identically.
            # Before the call, supervise the two intermediate reads the bare
            # delta regression collapses -- marker position, target position --
            # then the delta as their difference. All three triples print the
            # SAME 2-dp values the call carries, so the rounding identity
            # (placed + delta == gt, exactly as printed) is untouched.
            if t.get("decompose"):
                prose = (f"Marker {t['marker_id']} is at {_fmt_triple(*frm)}. "
                         f"The target is at {_fmt_triple(*to)}. "
                         f"Delta: {_fmt_triple(*delta)}.\n")
                return prose + call
            return call

        # THE EPISODE'S OWN TOOL CONTRACT, when it carries one. Records built
        # under the grounding-interface recipe are encoded through the shared
        # codec AFTER the frame conversion, never by this method's hard-coded
        # canonical name and argument construction. Without that, a renamed
        # first turn would revert to `place_marker` the moment the model read
        # the next prompt.
        _iface = item.get("interface_spec")

        def _iface_item_geometry(op_item, geometry):
            """One item's episode-frame geometry, from its typed world values."""
            frame = op_item.get("frame") or "world"
            if geometry == "position":
                raw = op_item.get("position_world")
                if raw is None:
                    raw, frame = op_item["position"], "episode"
                if frame == "episode":
                    return [round(float(v), 2) for v in raw[:3]]
                if frame != "world":
                    raise ValueError(f"unknown geometry frame {frame!r}")
                return [round(v, 2) for v in _xf(*raw[:3])]
            raw = op_item.get("box_world")
            if raw is None:
                raw, frame = op_item["box"], "episode"
            raw = list(raw)
            if frame == "episode":
                return [round(float(v), 2) for v in raw[:7]]
            if frame != "world":
                raise ValueError(f"unknown geometry frame {frame!r}")
            return ([round(v, 2) for v in _xf(*raw[:3])]
                    + [round(float(v), 2) for v in raw[3:6]]
                    + [round(float(raw[6]) + float(aug_yaw or 0.0), 2)])

        def _interface_value(t):
            """A typed operation rendered under this episode's own contract."""
            from onecanvas.data.tool_interface import encode_operation
            op = t.get("operation")
            if not isinstance(op, dict) or _iface is None:
                return None
            name = op.get("name")
            if name in ("place_marker", "place_box"):
                geometry = "position" if name == "place_marker" else "box"
                frame = op.get("frame") or "world"
                items = op.get("items")
                if items is None:
                    key = "marker_id" if name == "place_marker" else "box_id"
                    items = [{"id": op[key],
                              f"{geometry}_world": op[f"{geometry}_world"],
                              "frame": frame}]
                typed = {"name": name, "items": [
                    {"id": str(it["id"]),
                     geometry: _iface_item_geometry(dict(it, frame=it.get(
                         "frame") or frame), geometry)}
                    for it in items]}
            elif name == "python":
                typed = {"name": name, "code": op["code"]}
            elif name == "remove_box":
                typed = {"name": name, "box_id": op["box_id"]}
            elif name == "adjust_marker":
                frm = [round(v, 2) for v in _xf(*op["from_world"])]
                to = [round(v, 2) for v in _xf(*op["to_world"])]
                typed = {"name": name, "marker_id": op["marker_id"],
                         "delta": [round(to[i] - frm[i], 2) for i in range(3)]}
            else:
                raise ValueError(
                    f"unsupported typed operation {name!r} in {item.get('id')}")
            return encode_operation(typed, _iface)

        def _operation_value(t):
            """Render typed operation geometry into the current episode frame.

            New records never run arbitrary assistant text through a coordinate
            regex.  World geometry remains typed storage, while the rendered
            call contains episode-frame geometry.  The regex path below is
            retained only for historical records without ``operation``.
            """
            op = t.get("operation")
            if not isinstance(op, dict):
                return None
            if _iface is not None:
                return _interface_value(t)
            name = op.get("name")
            if name == "place_marker":
                p = [round(v, 2) for v in _xf(*op["position_world"])]
                body = {"name": name, "arguments": {
                    "marker_id": op["marker_id"], "position": p}}
            elif name == "place_box":
                raw = list(op["box_world"])
                p = [round(v, 2) for v in _xf(*raw[:3])]
                # The episode transform rotates world XY by aug_yaw, so the
                # oriented box's yaw follows the identical rotation.
                box = p + [round(float(v), 2) for v in raw[3:6]] + [
                    round(float(raw[6]) + float(aug_yaw or 0.0), 2)]
                body = {"name": name, "arguments": {
                    "box_id": op["box_id"], "box": box}}
            elif name == "adjust_marker":
                frm = [round(v, 2) for v in _xf(*op["from_world"])]
                to = [round(v, 2) for v in _xf(*op["to_world"])]
                delta = [round(to[i] - frm[i], 2) for i in range(3)]
                body = {"name": name, "arguments": {
                    "marker_id": op["marker_id"], "delta": delta}}
            elif name == "remove_box":
                body = {"name": name, "arguments": {"box_id": op["box_id"]}}
            elif name == "python":
                body = {"name": name, "arguments": {"code": op["code"]}}
            else:
                raise ValueError(
                    f"unsupported typed operation {name!r} in {item.get('id')}")
            return "<tool_call>\n" + json.dumps(body) + "\n</tool_call>"

        def _legacy_tool_value(t):
            """Typed conversion for historical calls that predate metadata."""
            value = t.get("value")
            if not isinstance(value, str) or "<tool_call>" not in value:
                return None
            try:
                raw = value.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
                body = json.loads(raw.strip())
            except (IndexError, ValueError, TypeError):
                return None
            name = body.get("name")
            args = body.get("arguments") or {}
            if name == "place_marker" and len(args.get("position") or []) >= 3:
                args["position"] = [round(v, 2) for v in _xf(*args["position"][:3])]
            elif name == "place_box" and len(args.get("box") or []) >= 7:
                box = list(args["box"])
                args["box"] = ([round(v, 2) for v in _xf(*box[:3])]
                               + [round(float(v), 2) for v in box[3:6]]
                               + [round(float(box[6]) + float(aug_yaw or 0.0), 2)])
            elif name in {"python", "remove_box"}:
                pass
            else:
                return None
            body["arguments"] = args
            return "<tool_call>\n" + json.dumps(body) + "\n</tool_call>"

        # Tool list follows the DATASET, not the individual episode: flagged on
        # the record by the annotation builder so every episode in a delta build
        # advertises the same three tools. If the third entry appeared only in
        # episodes that happen to carry a correction, the model could read "a
        # correction is due" straight off its own system prompt.
        # A record may carry its OWN tool list (the agentic box families
        # advertise place_box; a benchmark item rolled out through the tool
        # loop carries the block its checkpoint trained with). Absent, this is
        # byte-identical to before: the marker list, delta or absolute.
        _tools = item.get("tools") or (
            MARKER_TOOLS_DELTA if item.get("delta_grammar") else MARKER_TOOLS)

        # Canvas twins land at EXACTLY the 2-decimal point the transcript prints
        # (round after transform), so the twin marks where the printed marker is.
        #
        # ONE OP MAY PASTE MORE THAN ONE TWIN. `point` is the one-twin shorthand
        # every record written so far uses; `points` is the general form, for a
        # place_box that renders its eight CORNERS instead of a single token at
        # the centre. Pasting only the centre means the extent and the yaw never
        # reach the canvas at all, so the model cannot see the quantity that IS
        # the answer on the size and room-area families. Reading both keeps every
        # existing record byte-identical.
        mk_twins = []
        for ev in item.get("marker_events", []):
            _pts = ev.get("points")
            if _pts is None:
                _pts = [ev["point"]] if ev.get("point") is not None else []
            mk_twins.append([[round(v, 2) for v in _xf(*p)] for p in _pts])
        # The op's ANCHOR, which the text marker carries. First entry by
        # convention: for a corner paste the builder puts the centre there.
        mk_points = [t[0] if t else None for t in mk_twins]

        # Build multi-turn messages (system <tools> rendered by apply_chat_template).
        messages = [{"role": "user", "content": [
            {"type": "image", "image": self.dummy_image},
            {"type": "text", "text": convs[0]["value"]}]}]
        gpt_supervise = []
        for t in convs[1:]:
            if t.get("from") == "gpt":
                # VERBATIM TURNS ARE NOT RE-RENDERED. A rollout turn is the
                # model's own text, and the next prompt has to show it exactly
                # as it was emitted: same exposed function name, same layout,
                # same whitespace and key order. Re-encoding it from the typed
                # geometry would produce an equivalent call the model never
                # wrote. The typed world geometry rides along on the turn, so a
                # deliberate replay into another frame can still re-encode.
                if t.get("verbatim"):
                    messages.append({"role": "assistant", "content": t["value"]})
                    gpt_supervise.append(bool(t.get("supervise")))
                    continue
                # adjust turns carry endpoints, not text: their only bracketed
                # triple is a delta, which _xf_text would translate.
                _typed = _operation_value(t)
                _content = (_typed if _typed is not None else
                            (_adjust_value(t) if t.get("adjust_from") is not None
                             else (_legacy_tool_value(t) or t["value"])))
                messages.append({"role": "assistant", "content": _content})
                gpt_supervise.append(bool(t.get("supervise")))
            elif t.get("from") == "tool":
                inner = t["value"]
                if inner.startswith("<tool_response>") and inner.endswith("</tool_response>"):
                    inner = inner[len("<tool_response>"):-len("</tool_response>")]
                messages.append({"role": "tool", "content": inner})

        text = self.processor.apply_chat_template(
            messages, tools=_tools, tokenize=False,
            add_generation_prompt=False, **self.adapter.config.chat_template_kwargs)
        processed = self.processor(
            text=[text], images=[self.dummy_image], return_tensors="pt",
            max_pixels=self._dummy_img_w * self._dummy_img_h)
        processed["pixel_values"] = self._cached_pixel_values
        processed["image_grid_thw"] = self._cached_image_grid_thw

        input_ids, labels = self.adapter.mask_labels(processed.input_ids)
        ids0 = input_ids[0].clone()
        img_id = self.adapter.config.image_pad_token_id
        _tk = self.processor.tokenizer
        text_id = _tk.convert_tokens_to_ids("<|object_ref_start|>")   # text marker
        canvas_id = _tk.convert_tokens_to_ids("<|object_ref_end|>")   # canvas twin

        # Dummy-image block = the FIRST contiguous image_pad run (before the
        # object_ref -> image_pad conversion, so it is only the dummy image).
        img_pos = (ids0 == img_id).nonzero(as_tuple=True)[0]
        first_idx, last_idx = int(img_pos[0].item()), int(img_pos[-1].item())

        # Marker placeholders in text order. DISTINCT tokens make each slot
        # self-describing: <|object_ref_start|> = text marker (is_canvas False),
        # <|object_ref_end|> = canvas twin (is_canvas True). Merge by position, so
        # is_canvas is read off token identity, not assumed from pair order.
        start_pos = (ids0 == text_id).nonzero(as_tuple=True)[0].tolist()
        end_pos = (ids0 == canvas_id).nonzero(as_tuple=True)[0].tolist()
        n_inline = int(item.get("n_inline", 0))
        if len(start_pos) < n_inline:
            raise ValueError(
                f"expected {n_inline} inline references but found "
                f"{len(start_pos)} in {item.get('id')}")
        inline_pos, marker_start = start_pos[:n_inline], start_pos[n_inline:]
        mk_slots = sorted([(p, False) for p in marker_start]
                          + [(p, True) for p in end_pos])
        mk_positions = [p for p, _ in mk_slots]
        placeholder_is_canvas = [c for _, c in mk_slots]
        for p in inline_pos + mk_positions:
            ids0[p] = img_id
        input_ids = ids0.unsqueeze(0)

        # Per-turn supervise masking: mask_labels un-masked EVERY assistant turn;
        # re-mask the supervise=False ones (bad placements, imperfect corrections).
        labels0 = labels[0].clone()
        if inline_pos:
            labels0[torch.tensor(inline_pos, dtype=torch.long)] = -100
        asst_id = self.adapter.config.assistant_token_id
        im_end = self.adapter.config.im_end_token_id
        idl = ids0.tolist()
        L, pos, k = len(idl), 0, 0
        while pos < L:
            if idl[pos] == asst_id:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and idl[ans_end] != im_end:
                    ans_end += 1
                if k < len(gpt_supervise) and not gpt_supervise[k]:
                    labels0[ans_start:ans_end + 2] = -100
                k += 1
                pos = ans_end
            pos += 1
        labels = labels0.unsqueeze(0)

        # marker_tok_* keys: one fresh marker per op (shuffled without
        # replacement), shared across the op's text marker and every twin it
        # pastes, so eight corners of one box read as one object rather than
        # eight. Slots are canvas twins, each OPTIONALLY preceded by its text
        # partner: a place op emits <|object_ref_start|> then its twins, while
        # the python tool's witness paste (`paste()` in a snippet) emits BARE
        # <|object_ref_end|> twins with no text partner. Every canvas twin
        # consumes one twin point, in text order. This is the synthetic
        # render's rule (agentic marker_episode_inputs); until 2026-09-06 this
        # path asserted a text partner per op and died on the first witness
        # paste rolled out through a real scene (VSI-Bench tool loop, vsi#37).
        # For records where every op has a text partner the output is
        # byte-identical to before.
        n_place = len(mk_twins)
        twin_event = [j for j, _t in enumerate(mk_twins) for _ in _t]
        flat_twins = [p for _t in mk_twins for p in _t]
        n_twin = sum(placeholder_is_canvas)
        if n_twin != len(flat_twins):
            raise ValueError(
                f"{n_twin} canvas twins in the transcript for {len(flat_twins)} "
                f"twin points across {n_place} marker events "
                f"({[len(t) for t in mk_twins]} per event) in {item.get('id')}; "
                f"is_canvas={placeholder_is_canvas}. The transcript's "
                "<tool_response> and the marker_events disagree about how many "
                "twins each op pastes.")
        for _i, _c in enumerate(placeholder_is_canvas):
            if not _c and (_i + 1 >= len(placeholder_is_canvas)
                           or not placeholder_is_canvas[_i + 1]):
                raise ValueError(
                    f"text marker at slot {_i} is not immediately followed by its "
                    f"canvas twin in {item.get('id')}; is_canvas={placeholder_is_canvas}")
        markers = self._marker_pool()
        events = list(item.get("marker_events", []))
        # Stable identity allocation.  An update reuses its id's feature.  Two
        # ids at one coordinate remain visually distinct.  Witness events have
        # their own event identity because they are measurements, not boxes.
        identity_keys = [
            str(ev.get("marker_id")) if ev.get("marker_id") is not None
            else f"__witness_{j}"
            for j, ev in enumerate(events)
        ]
        unique_keys = list(dict.fromkeys(identity_keys))
        if len(unique_keys) > int(markers.shape[0]):
            raise ValueError(
                f"{item.get('id')} renders {len(unique_keys)} marker identities "
                f"but pool capacity is {int(markers.shape[0])}")
        seed = int(_hashlib.sha1(str(item.get("id", "")).encode()).hexdigest()[:8], 16)
        order = _np.random.default_rng(seed).permutation(markers.shape[0])[:len(unique_keys)]
        feature_for = {key: int(order[i]) for i, key in enumerate(unique_keys)}

        # Only the most recent active event for an id remains on the spatial
        # canvas.  Historical response tokens stay in the transcript but are
        # rendered as non-canvas marker tokens.  A removal leaves no active
        # event for that id.
        active_event = {j: True for j, key in enumerate(identity_keys)
                        if key.startswith("__witness_")}
        latest = {}
        for j, ev in enumerate(events):
            mid = ev.get("marker_id")
            if mid is None:
                continue
            if ev.get("removed") or ev.get("action") == "removed":
                latest.pop(str(mid), None)
            else:
                latest[str(mid)] = j
        active_event.update({j: True for j in latest.values()})
        # A text token takes the feature and anchor point of the op whose twin
        # follows it, which is what binds the pair; a twin takes its own point.
        feats, pts, canvas_flags, ev = [], [], [], 0
        for _c in placeholder_is_canvas:
            j = twin_event[ev]
            feats.append(markers[feature_for[identity_keys[j]]])
            pts.append(flat_twins[ev] if _c else mk_points[j])
            canvas_flags.append(bool(_c and active_event.get(j, False)))
            if _c:
                ev += 1
        marker_keys = {
            "marker_tok_positions": torch.tensor(mk_positions, dtype=torch.long),
            "marker_tok_features": (torch.stack(feats) if feats
                                    else torch.zeros(0, markers.shape[1], markers.shape[2])),
            "marker_tok_is_canvas": torch.tensor(canvas_flags, dtype=torch.bool),
            "marker_tok_points": (torch.tensor(pts, dtype=torch.float32)
                                  if pts else torch.zeros(0, 3)),
            "inline_patch_positions": torch.tensor(
                inline_pos, dtype=torch.long),
        }
        return (input_ids, labels, first_idx, last_idx, int(input_ids.shape[-1]),
                processed.attention_mask, marker_keys)

    #: Pixel side of the placeholder image each observed turn carries. Any
    #: size works (prepare_batch_blocks replaces the run with the real canvas),
    #: and a small one keeps a many-photo conversation's pre-projection length
    #: far below model_max_length: 128 px at 16 px patches merged 2x2 is 16
    #: placeholder tokens per observation.
    _OBSERVATION_DUMMY_PX = 128

    def _observation_conversation_item(self, item, assets, attempts):
        """ONE CANVAS PER OBSERVATION: a conversation whose turns bring photos.

        A row of this kind is a sequence of human turns, assistant turns and
        tool results in which every human turn that names an `observation`
        brings that observation's own photo(s) and its own measured reference
        points. Each observation becomes its own canvas block, reprojected
        around the row's ONE declared origin (`canvas_center`, `canvas_yaw`),
        and the block sits at the start of the human turn that names it, so
        the assistant's move k attends to photos 0..k and to none taken after
        it under the ordinary causal mask.

        MARKERS BIND PER TURN. A turn's marker pairs (`<|object_ref_start|>`
        inline copy, `<|object_ref_end|>` canvas token) bind in text order to
        THAT turn's observation's `reference_points`, each with the declared
        pool row the row froze for it.

        LOSS ON EVERY ASSISTANT TURN whose `supervise` is not False. Everything
        else is masked: system, tools, human turns, tool results, placeholders.

        Nothing here is resampled or augmented. The origin, the points and the
        pool rows are the row's own, on either split.
        """
        import torch
        from PIL import Image

        observations = list(item["observations"])
        convs = list(item["conversations"])
        if item.get("canvas_center") is None:
            raise ValueError(
                f"{item.get('scene_id')!r} is a multi-observation row without a "
                f"canvas_center; its points are expressed in that frame")
        frame = str(item.get("reference_point_frame") or "canvas")
        if frame != "canvas":
            raise ValueError(
                f"{item.get('scene_id')!r} declares reference_point_frame "
                f"{frame!r}; this path consumes canvas-frame points only")
        center = torch.tensor([float(v) for v in item["canvas_center"]],
                              dtype=torch.float32)
        yaw = float(item.get("canvas_yaw") or 0.0)
        counts = [len(obs.get("frames") or []) for obs in observations]
        if any(c <= 0 for c in counts):
            raise ValueError(f"{item.get('scene_id')!r}: an observation has no frame")
        if sum(counts) != len(assets["images"]):
            raise ValueError(
                f"{item.get('scene_id')!r}: {sum(counts)} observation frames but "
                f"{len(assets['images'])} loaded images")
        frame_start = [sum(counts[:k]) for k in range(len(counts))]

        px = int(self._OBSERVATION_DUMMY_PX)
        dummy = Image.new("RGB", (px, px), (0, 0, 0))
        messages, named, gpt_supervise = [], [], []
        _system = str(item.get("system_prompt") or "")
        if _system:
            messages.append({"role": "system", "content": _system})
        for turn in convs:
            who = turn.get("from")
            if who == "human":
                content = []
                if turn.get("observation") is not None:
                    named.append(int(turn["observation"]))
                    content.append({"type": "image", "image": dummy})
                content.append({"type": "text", "text": str(turn["value"])})
                messages.append({"role": "user", "content": content})
            elif who == "gpt":
                messages.append({"role": "assistant", "content": str(turn["value"])})
                gpt_supervise.append(turn.get("supervise", True) is not False)
            elif who == "tool":
                messages.append({"role": "tool", "content": str(turn["value"])})
            else:
                raise ValueError(f"{item.get('scene_id')!r}: unknown turn {who!r}")
        if named != list(range(len(observations))):
            raise ValueError(
                f"{item.get('scene_id')!r}: human turns name observations {named}, "
                f"and every observation must be named once, in order")
        # A ROW THAT ENDS ON A HUMAN TURN ASKS FOR THE NEXT MOVE. It is rendered
        # with the generation prompt and carries no supervised token, which is
        # how a live rollout call is decoded through this same path (the policy
        # service builds it with agentic_onecanvas.body.trace_input). A training
        # row ends on its last assistant turn and is rendered exactly as before.
        generation = bool(convs) and convs[-1].get("from") == "human"

        text = self.processor.apply_chat_template(
            messages, tools=item.get("tools") or None, tokenize=False,
            add_generation_prompt=generation,
            **self.adapter.config.chat_template_kwargs)
        processed = self.processor(
            text=[text], images=[dummy] * len(observations), return_tensors="pt",
            max_pixels=px * px)
        input_ids, labels = self.adapter.mask_labels(processed.input_ids)
        ids0 = input_ids[0].clone()
        labels0 = labels[0].clone()
        img_id = int(self.adapter.config.image_pad_token_id)

        # The dummy runs, before any placeholder becomes image_pad.
        pad = (ids0 == img_id).nonzero(as_tuple=True)[0].tolist()
        runs = []
        for p in pad:
            if runs and p == runs[-1][1] + 1:
                runs[-1][1] = p
            else:
                runs.append([p, p])
        if len(runs) != len(observations):
            raise ValueError(
                f"{item.get('scene_id')!r}: {len(runs)} placeholder image runs "
                f"for {len(observations)} observations")

        # Markers, bound per turn in text order.
        _tk = self.processor.tokenizer
        inline_id = _tk.convert_tokens_to_ids("<|object_ref_start|>")
        canvas_id = _tk.convert_tokens_to_ids("<|object_ref_end|>")
        slots = sorted(
            [(int(p), False) for p in (ids0 == inline_id).nonzero(as_tuple=True)[0]]
            + [(int(p), True) for p in (ids0 == canvas_id).nonzero(as_tuple=True)[0]])
        pool = (self._declared_feature_pool(dict(item.get("feature_pool") or {}))
                if slots else None)
        positions, is_canvas, feats, points, block_of = [], [], [], [], []
        for k, obs in enumerate(observations):
            lo = runs[k][1]
            hi = runs[k + 1][0] if k + 1 < len(runs) else len(ids0)
            mine = [(p, c) for p, c in slots if lo < p < hi]
            refs = list(obs.get("reference_points") or [])
            want = [flag for _r in refs for flag in (False, True)]
            if [c for _p, c in mine] != want:
                raise ValueError(
                    f"{item.get('scene_id')!r}: observation {k} declares "
                    f"{len(refs)} reference points and its turn carries "
                    f"placeholders {[c for _p, c in mine]}. Do not pad or "
                    f"truncate; fix the render.")
            for j, (p, c) in enumerate(mine):
                ref = refs[j // 2]
                row = int(ref["pool_row"])
                if not (0 <= row < int(pool.shape[0])):
                    raise ValueError(
                        f"reference {ref.get('id')} names pool row {row}, outside "
                        f"the pool's {int(pool.shape[0])} rows")
                positions.append(p)
                is_canvas.append(c)
                feats.append(pool[row])
                points.append([float(v) for v in ref["point"]])
                block_of.append(k)
        if len(positions) != len(slots):
            raise ValueError(
                f"{item.get('scene_id')!r}: {len(slots) - len(positions)} marker "
                f"placeholder(s) lie before the first observation")
        for p in positions:
            ids0[p] = img_id
            labels0[p] = -100

        # Per-turn supervision: mask_labels opened every assistant turn.
        asst_id = self.adapter.config.assistant_token_id
        im_end = self.adapter.config.im_end_token_id
        idl = ids0.tolist()
        pos, k = 0, 0
        while pos < len(idl):
            if idl[pos] == asst_id:
                start = pos + 2
                end = start
                while end < len(idl) and idl[end] != im_end:
                    end += 1
                if k < len(gpt_supervise) and not gpt_supervise[k]:
                    labels0[start:end + 2] = -100
                k += 1
                pos = end
            pos += 1
        if generation:
            labels0[:] = -100

        # The real photos, for the live visual encoder only.
        raw_messages = [{"role": "user", "content": [
            *[{"type": "image", "image": im} for im in assets["images"]],
            {"type": "text", "text": ""}]}]
        raw_text = self.processor.apply_chat_template(
            raw_messages, tokenize=False, add_generation_prompt=False,
            **self.adapter.config.chat_template_kwargs)
        processed_raw = self.processor(
            text=[raw_text], images=assets["images"], return_tensors="pt")

        blocks = torch.tensor(
            [[runs[k][0], runs[k][1], frame_start[k], frame_start[k] + counts[k]]
             for k in range(len(observations))], dtype=torch.long)
        gpt_values = [str(t["value"]) for t in convs if t.get("from") == "gpt"]
        human_values = [str(t["value"]) for t in convs if t.get("from") == "human"]
        data_dict = {
            "pixel_values": processed_raw["pixel_values"],
            "image_grid_thw": processed_raw["image_grid_thw"],
            # [1, 1, L], the shape the single-canvas path returns
            # (`input_ids_360.unsqueeze(0)` on a [1, L] tensor).
            "input_ids": ids0.view(1, 1, -1),
            "attention_mask": processed.attention_mask.unsqueeze(0),
            "labels": labels0.view(1, 1, -1),
            "image_dims": stack_tensor_list(assets.get("image_dims")),
            "answer": gpt_values[-1] if gpt_values and not generation else "",
            "question": human_values[0] if human_values else "",
            "question_type": item.get("question_type"),
            "scene_id": item["scene_id"],
            "image_token_first_idx": torch.tensor(runs[0][0], dtype=torch.long),
            "image_token_last_idx": torch.tensor(runs[0][1], dtype=torch.long),
            "input_seq_len": torch.tensor(int(ids0.shape[-1]), dtype=torch.long),
            "canvas_blocks": blocks,
            "marker_tok_positions": torch.tensor(positions, dtype=torch.long),
            "marker_tok_features": (torch.stack(feats) if feats else None),
            "marker_tok_is_canvas": torch.tensor(is_canvas, dtype=torch.bool),
            "marker_tok_points": (torch.tensor(points, dtype=torch.float32)
                                  if points else torch.zeros(0, 3)),
            "marker_tok_block": torch.tensor(block_of, dtype=torch.long),
            "images": assets["images"],
            "n_source_images": torch.tensor(len(assets["images"]), dtype=torch.long),
            "_dataloader_retries": attempts,
            "aug_center_override": center,
            "aug_yaw_angle": torch.tensor(yaw, dtype=torch.float32),
        }
        if self.with_precomputed_geometry:
            data_dict["depths"] = stack_tensor_list(prepare_depths(assets.get("depths", [])))
            data_dict["poses"] = stack_tensor_list(assets.get("poses"))
            data_dict["intrinsics"] = stack_tensor_list(assets.get("intrinsics"))
        return data_dict

    def __getitem__(self, i):
        curr_idx = i
        attempts = 0
        max_samples = len(self.list_data_dict)
        _failed_scenes = {}  # scene_id -> reason, for this __getitem__ call

        while attempts < max_samples:
            item = self.list_data_dict[curr_idx]
            _wants_aligned = item.get("question_type") == "grounding"
            try:
                if item.get("single_frame"):
                    assets = self._load_single_frame_assets(item)
                else:
                    assets = self._load_scene_data(item["scene_id"], item["data_path"], sample_idx=i,
                                                    pinned_frames=item.get("pinned_stems") or item.get("images"),
                                                    scene_subdir=item.get("scene_subdir"),
                                                    wants_aligned=_wants_aligned,
                                                    dataset_name=item.get("dataset_name"))
            except (OSError, IOError) as e:
                self._last_load_failure = f"io_error:{item['scene_id']}:{e}"
                assets = None
            if assets is not None:
                break
            # Track per-scene failures for diagnostics
            _sid = item["scene_id"]
            if _sid not in _failed_scenes:
                _failed_scenes[_sid] = self._last_load_failure or "unknown"
            if self.data_split == "test":
                # A benchmark scores each question on its own scene. Retry the
                # same item (flaky shared storage), never substitute another one,
                # which would duplicate one question and drop this one unnoticed.
                if attempts >= 2:
                    raise RuntimeError(
                        f"test item {i} (scene {_sid}) failed to load three times: "
                        f"{self._last_load_failure or 'unknown'}")
                attempts += 1
                continue
            curr_idx = (curr_idx + 1) % max_samples
            attempts += 1

        if assets is None:
            raise RuntimeError("No valid scene data found")

        # One bounded warning (once per worker) when the loader had to skip
        # past unreadable scenes to find a usable one. A few retries are
        # expected on flaky shared storage. Persistent failures show up as a
        # repeated first-failure scene id here and in the init-time skip report.
        if attempts > 0 and not getattr(self, "_retry_warned", False):
            self._retry_warned = True
            _sid, _reason = next(iter(_failed_scenes.items()),
                                 (item["scene_id"], "unknown"))
            rank0_print(
                f"[data] WARNING: skipped {attempts} unreadable scene(s) before "
                f"loading {item['scene_id']}; first failure: {_sid} ({_reason}). "
                f"Consider lowering dataloader_num_workers if this persists."
            )

        self._maybe_debug_print_sample(item, curr_idx)

        if item.get("observations"):
            return self._observation_conversation_item(item, assets, attempts)

        prompt_text = item["question"]

        answers = item.get("answers") or [""]
        # During training, randomly sample from all annotator answers to reduce bias
        # from always using index 0; use index 0 deterministically at eval time.
        answer_text = random.choice(answers) if self.data_split == "train" else answers[0]

        # --- Panoramic augmentation (training-only) ---
        # Sampled here (before bbox centering) so grounding GT coordinates
        # can be shifted to match the augmented panorama origin.
        _aug_center = None
        _aug_yaw = None
        _is_grounding = item.get("question_type") == "grounding"
        if self.data_split == "train":
            _aug_rng = random.Random(self._aug_seed + i)
            if self._panoramic_augment_center_uniform_scene:
                # Uniform in the scene-extent AABB (unprojected depth points),
                # which typically extends well beyond the camera-translation
                # AABB — cameras usually don't reach into every corner/wall.
                _poses_list = assets.get("poses", [])
                _depths_list = assets.get("depths", [])
                _intr_list = assets.get("intrinsics", [])
                _dims_list = assets.get("image_dims", [])
                _lo, _hi = (None, None)
                if _depths_list and _poses_list and _intr_list and _dims_list:
                    _lo, _hi = compute_scene_aabb_from_depths(
                        _depths_list, _poses_list, _intr_list, _dims_list, grid=16,
                    )
                if _lo is not None:
                    if self._panoramic_augment_center_inflate != 1.0:
                        _mid = (_lo + _hi) * 0.5
                        _half = (_hi - _lo) * 0.5 * self._panoramic_augment_center_inflate
                        _lo, _hi = _mid - _half, _mid + _half
                    _aug_center = torch.tensor(
                        [_aug_rng.uniform(_lo[0].item(), _hi[0].item()),
                         _aug_rng.uniform(_lo[1].item(), _hi[1].item()),
                         _aug_rng.uniform(_lo[2].item(), _hi[2].item())],
                        dtype=torch.float32,
                    )
            elif self._panoramic_augment_center_uniform:
                # Uniform in the XYZ axis-aligned bbox of camera translations.
                # Reprojection has no z-buffering so centers inside furniture /
                # walls are valid — they just reshuffle angular coverage.
                _poses_list = assets.get("poses", [])
                _valid = [p for p in _poses_list if p is not None and torch.is_tensor(p) and torch.all(torch.isfinite(p))]
                if _valid:
                    _translations = torch.stack([p[:3, 3] for p in _valid])
                    _lo = _translations.min(dim=0).values
                    _hi = _translations.max(dim=0).values
                    if self._panoramic_augment_center_inflate != 1.0:
                        _mid = (_lo + _hi) * 0.5
                        _half = (_hi - _lo) * 0.5 * self._panoramic_augment_center_inflate
                        _lo, _hi = _mid - _half, _mid + _half
                    _aug_center = torch.tensor(
                        [_aug_rng.uniform(_lo[0].item(), _hi[0].item()),
                         _aug_rng.uniform(_lo[1].item(), _hi[1].item()),
                         _aug_rng.uniform(_lo[2].item(), _hi[2].item())],
                        dtype=torch.float32,
                    )
            elif self._panoramic_augment_center_sigma > 0:
                # Gaussian center in the XY (horizontal) plane only.
                # Z stays at camera mean to avoid looking at ceiling/floor.
                _poses_list = assets.get("poses", [])
                _valid = [p for p in _poses_list if p is not None and torch.is_tensor(p) and torch.all(torch.isfinite(p))]
                if _valid:
                    _translations = torch.stack([p[:3, 3] for p in _valid])
                    _mean = _translations.mean(dim=0)
                    _scene_radius = (_translations - _mean).norm(dim=1).max().item()
                    _std = self._panoramic_augment_center_sigma * max(_scene_radius, 0.1)
                    _aug_center = torch.tensor(
                        [_aug_rng.gauss(_mean[0].item(), _std),
                         _aug_rng.gauss(_mean[1].item(), _std),
                         _mean[2].item()],
                        dtype=torch.float32,
                    )
            elif self._panoramic_augment_center:
                _poses_list = assets.get("poses", [])
                _valid = [p for p in _poses_list if p is not None and torch.is_tensor(p) and torch.all(torch.isfinite(p))]
                if _valid:
                    _aug_center = _aug_rng.choice(_valid)[:3, 3].clone()
            if self._panoramic_augment_yaw:
                if _is_grounding:
                    # 90-degree snaps only: keeps axis-aligned bboxes valid
                    # (arbitrary yaw would require OBB or inflate the AABB).
                    _aug_yaw = _aug_rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2])
                else:
                    _aug_yaw = _aug_rng.uniform(-math.pi, math.pi)

        # --- SQA3D: place panorama at situated agent pose ---
        # Overrides any panoramic_augment_* sampling above. Self-gates on the
        # presence of agent_position/agent_rotation in the item (only SQA3D
        # carries these after normalization), so leaving the flag on for
        # mixed-dataset runs is safe.
        #
        # SQA3D convention (from their ScanQA/lib/sepdataset.py):
        #   raw_position = sqa3d_position + bs_center_raw
        # where bs_center_raw = (verts_raw.max + verts_raw.min) / 2 of the
        # {scene}_vh_clean_2.ply mesh. Rotation is in raw frame directly
        # (quaternion → yaw = 2*atan2(qz, qw)) and SQA3D's identity
        # orientation = agent facing -X. Panorama forward at yaw=0 points
        # world +Y, so to make panorama forward = agent forward we set
        #   yaw_panorama = -π/2 - yaw_sqa3d
        # This was empirically verified on scene0702, scene0050, scene0169:
        # all landmarks land at the expected columns (toilet paper on left,
        # soap on right, toolboxes front, ottoman right, trash can front).
        # Resolve the effective center mode for this sample. "auto" defers to
        # the legacy sqa3d_use_agent_pose flag (training-side default); explicit
        # modes override for inference-time ablation.
        _mode = self._sqa3d_canvas_center_mode
        if _mode == "auto":
            _use_agent_pose = bool(self._sqa3d_use_agent_pose or item.get("force_agent_pose"))
            _alt_origin = None
        elif _mode == "agent_pose":
            _use_agent_pose = True
            _alt_origin = None
        elif _mode == "scene_center":
            _use_agent_pose = False
            _alt_origin = None
        else:
            _use_agent_pose = False
            _alt_origin = _mode  # "random_camera" | "outside_bbox"

        if _use_agent_pose:
            _pos = item.get("agent_position")
            _rot = item.get("agent_rotation")
            _sqa3d_pos = None
            _sqa3d_yaw = None
            if _pos is not None:
                _sqa3d_pos = torch.tensor(
                    [float(_pos["x"]), float(_pos["y"]), float(_pos["z"])],
                    dtype=torch.float32,
                )
            if _rot is not None:
                _qz = float(_rot["_z"])
                _qw = float(_rot["_w"])
                _sqa3d_yaw = 2.0 * math.atan2(_qz, _qw)

            _bs_center = None
            if _sqa3d_pos is not None:
                _bs_center = self._load_scannet_bs_center(item["scene_id"])
                if _bs_center is None:
                    # SQA3D positions are relative to the mesh's bounding-box
                    # centre. Without the mesh the canvas lands in the wrong place
                    # and the question is still answered, so refuse instead.
                    raise RuntimeError(
                        f"{item['scene_id']}: the SQA3D agent pose needs the ScanNet mesh "
                        f"{item['scene_id']}_vh_clean_2.ply and the plyfile package to "
                        "place the canvas. Download the meshes as in docs/DATA.md.")

            if _bs_center is not None and _sqa3d_pos is not None:
                _sqa3d_pos = _sqa3d_pos + _bs_center

            # --predicted-geometry: the agent pose above is in the GT world
            # frame, but the canvas is built in the predictor's own frame.
            # Re-express position AND yaw through the per-scene GT->predicted
            # similarity fit (see _predgeo_anchor_transform).
            if self._predicted_geometry and _sqa3d_pos is not None:
                _fit = self._predgeo_anchor_transform(
                    item["scene_id"], assets.get("scene_dir"),
                    assets.get("frame_keys") or [], assets.get("poses") or [])
                if _fit is not None:
                    _p = _sqa3d_pos.double().numpy()
                    _sqa3d_pos = torch.tensor(
                        _fit["s"] * (_fit["R"] @ _p) + _fit["t"],
                        dtype=torch.float32)
                    if _sqa3d_yaw is not None:
                        _sqa3d_yaw = _sqa3d_yaw + _fit["yaw"]

            if _sqa3d_pos is not None:
                _aug_center = _sqa3d_pos
            if _sqa3d_yaw is not None:
                _aug_yaw = (-math.pi / 2) - _sqa3d_yaw + self._sqa3d_agent_yaw_offset
                _aug_yaw = ((_aug_yaw + math.pi) % (2 * math.pi)) - math.pi

            # Diagnostic: print a handful of agent-vs-camera frames so the
            # user can sanity-check coord alignment before trusting numbers.
            if self._sqa3d_pose_debug_logged < 5 and _aug_center is not None:
                _poses_list = assets.get("poses", [])
                _valid = [p for p in _poses_list if p is not None and torch.is_tensor(p) and torch.all(torch.isfinite(p))]
                if _valid:
                    _cams = torch.stack([p[:3, 3] for p in _valid])
                    _cmin = _cams.min(dim=0).values.tolist()
                    _cmax = _cams.max(dim=0).values.tolist()
                    _cmean = _cams.mean(dim=0).tolist()
                    _tag = "bs_applied" if _bs_center is not None else "no_bs_center"
                    print(
                        f"[sqa3d-pose] scene={item['scene_id']} {_tag} "
                        f"agent_xyz=[{_aug_center[0].item():.3f},{_aug_center[1].item():.3f},{_aug_center[2].item():.3f}] "
                        f"yaw={_aug_yaw:.3f} "
                        f"cam_mean=[{_cmean[0]:.3f},{_cmean[1]:.3f},{_cmean[2]:.3f}] "
                        f"cam_aabb=[{_cmin[0]:.2f},{_cmin[1]:.2f},{_cmin[2]:.2f}]→"
                        f"[{_cmax[0]:.2f},{_cmax[1]:.2f},{_cmax[2]:.2f}]"
                    )
                    self._sqa3d_pose_debug_logged += 1

        # --- SQA3D canvas-origin ablation: random_camera / outside_bbox ---
        # Inference-time only (driven by --sqa3d-canvas-center-mode). Gates on
        # presence of agent_position so non-SQA3D items in mixed evals fall
        # through to the default (scene-center) origin.
        if _alt_origin is not None and item.get("agent_position") is not None:
            _poses_list = assets.get("poses", [])
            _valid_poses = [
                p for p in _poses_list
                if p is not None and torch.is_tensor(p) and torch.all(torch.isfinite(p))
            ]
            if _valid_poses:
                _trans = torch.stack([p[:3, 3] for p in _valid_poses]).float()
                if _alt_origin == "random_camera":
                    # Deterministic per-sample seed so re-runs reproduce.
                    _alt_rng = random.Random(self._aug_seed + i + 7919)
                    _alt_idx = _alt_rng.randrange(_trans.shape[0])
                    _aug_center = _trans[_alt_idx].clone()
                elif _alt_origin == "outside_bbox":
                    _xy_center = _trans[:, :2].mean(dim=0)
                    _xy_radius = (_trans[:, :2] - _xy_center).norm(dim=1).max().item()
                    _z_mean = float(_trans[:, 2].mean().item())
                    _aug_center = torch.tensor(
                        [float(_xy_center[0].item()) + 1.5 * _xy_radius,
                         float(_xy_center[1].item()),
                         _z_mean],
                        dtype=torch.float32,
                    )
                # No yaw rotation for alternative origins — keeps the
                # comparison about position only.
                _aug_yaw = None

                if self._sqa3d_center_mode_debug_logged < 5 and _aug_center is not None:
                    _cmean = _trans.mean(dim=0).tolist()
                    print(
                        f"[sqa3d-center-mode={_alt_origin}] scene={item['scene_id']} "
                        f"origin_xyz=[{_aug_center[0].item():.3f},{_aug_center[1].item():.3f},{_aug_center[2].item():.3f}] "
                        f"cam_mean=[{_cmean[0]:.3f},{_cmean[1]:.3f},{_cmean[2]:.3f}]"
                    )
                    self._sqa3d_center_mode_debug_logged += 1

        # --- SPBench-SI: place panorama at the pinned camera pose ---
        # SPBench-SI annotations pin a single frame via `images: [frame.jpg]`
        # and ask questions in that camera's egocentric frame ("From the
        # camera's perspective, is X to Y's left/right/front/back?"). Gate on
        # len(images)==1 so SPBench-MV (8 frames, object-relative questions)
        # falls through to scene-center.
        # ScanNet/ARKit/SNPP poses are camera→world in OpenCV convention, so
        # camera forward = +Z in camera coords. World forward = pose[:3,:3] @ +Z.
        # Panorama forward at yaw=0 points world +Y (same convention as SQA3D
        # at L1944), so yaw = atan2(fx, fy) aligns canvas forward with camera
        # forward. Sign/offset can be nudged via spbench_camera_yaw_offset.
        if self._spbench_use_camera_pose and len(item.get("images", [])) == 1:
            _poses = assets.get("poses", [])
            if _poses and torch.is_tensor(_poses[0]) and torch.all(torch.isfinite(_poses[0])):
                _P = _poses[0].float()
                _aug_center = _P[:3, 3].clone()
                _fwd_world = _P[:3, :3] @ torch.tensor([0.0, 0.0, 1.0])
                _aug_yaw = math.atan2(float(_fwd_world[0]), float(_fwd_world[1])) + self._spbench_camera_yaw_offset
                _aug_yaw = ((_aug_yaw + math.pi) % (2 * math.pi)) - math.pi
                if self._spbench_pose_debug_logged < 5:
                    print(
                        f"[spbench-pose] scene={item['scene_id']} "
                        f"image={item['images'][0]} "
                        f"cam_xyz=[{_aug_center[0].item():.3f},{_aug_center[1].item():.3f},{_aug_center[2].item():.3f}] "
                        f"fwd_world=[{float(_fwd_world[0]):.3f},{float(_fwd_world[1]):.3f},{float(_fwd_world[2]):.3f}] "
                        f"yaw={_aug_yaw:.3f}"
                    )
                    self._spbench_pose_debug_logged += 1

        # --- DECLARED canvas anchor (per item, honoured verbatim) ------------
        # An annotation may name the canvas origin it was written against.
        # Self-gating on the field's presence, like the SQA3D agent-pose branch
        # above, so leaving it absent changes nothing for any existing dataset.
        #
        # WHY A DATASET WOULD DECLARE ONE. Every existing anchor is CHOSEN by
        # the loader: the mean camera position, a random camera under
        # `panoramic_augment_center`, the situated agent, the queried camera.
        # None of those is reproducible per item, and an experiment whose
        # variable IS the origin needs a fixed train/val/test set of origins,
        # not a fresh draw per epoch. The first consumer is the
        # agentic-onecanvas waypoint-grounding pilot, which holds the scene and
        # the instruction fixed and varies only this.
        #
        # IT OVERRIDES THE `panoramic_augment_*` SAMPLING ABOVE, both halves,
        # and the yaw override is NOT optional. A declared anchor with a
        # randomly drawn yaw is a half-declared frame, so an answer stored
        # against the declaration would be right about the origin and wrong
        # about the bearing, on a fraction of samples, with no error anywhere.
        # `canvas_yaw` therefore defaults to 0.0 rather than to whatever the
        # augmentation happened to pick.
        if item.get("canvas_center") is not None:
            _cc = list(item["canvas_center"])
            if len(_cc) != 3:
                raise ValueError(
                    f"canvas_center must be 3 numbers, got {len(_cc)} on "
                    f"scene {item.get('scene_id')!r}")
            _aug_center = torch.tensor([float(v) for v in _cc],
                                       dtype=torch.float32)
            _aug_yaw = float(item.get("canvas_yaw") or 0.0)
            if self._declared_anchor_logged < 5:
                print(f"[declared-anchor] scene={item.get('scene_id')} "
                      f"xyz=[{_cc[0]:.3f},{_cc[1]:.3f},{_cc[2]:.3f}] "
                      f"yaw={_aug_yaw:.4f}")
                self._declared_anchor_logged += 1

        # --- Global canvas yaw offset (seam probe) ------
        # Rotates the ENTIRE canvas by a fixed angle. This is a rigid rotation
        # about the canvas vertical axis, so every distance, size and
        # inter-object relation is unchanged and every ground-truth answer stays
        # correct. The ONLY thing it moves is where the equirectangular
        # longitude wrap (the "seam", W = 0 = 100 in MRoPE terms) falls relative
        # to the scene. Sweeping it and comparing per-question accuracy measures
        # what the seam costs on real benchmark data.
        #
        # Placed HERE deliberately, on both sides:
        #   * AFTER the per-dataset anchor branches, so it COMPOSES with SQA3D
        #     agent-pose yaw and SPBench camera-pose yaw instead of replacing
        #     them. Datasets that pick no yaw (VSI-Bench / vlm3r) start from 0.
        #   * BEFORE the grounding block below, which rotates GT boxes by
        #     _aug_yaw. Applying it later would render a rotated canvas against
        #     unrotated GT boxes and silently corrupt every grounding answer.
        if self._panoramic_eval_yaw_offset:
            _aug_yaw = (((_aug_yaw or 0.0) + self._panoramic_eval_yaw_offset
                         + math.pi) % (2 * math.pi)) - math.pi
            if self._eval_yaw_offset_logged < 3:
                print(f"[eval-yaw-offset] offset="
                      f"{self._panoramic_eval_yaw_offset:.4f} rad -> "
                      f"aug_yaw={_aug_yaw:.4f} scene={item.get('scene_id')}")
                self._eval_yaw_offset_logged += 1

        # --- Grounding: translate raw bbox to scene-centered frame ---
        # Grounding JSONLs store bboxes as bare metric float tuples in the
        # scene's axis-aligned world frame (uncentered). The panorama is rendered
        # with the mean of the selected camera positions as origin, so we
        # subtract that same scene center from the bbox center to put both in
        # the same frame. Sizes and rotation angles are unchanged.
        #
        # Two formats are supported, keyed by the ``bbox_format`` field in
        # each annotation entry (defaults to "aabb" when absent):
        #   AABB (6-value): (cx, cy, cz, w, h, d)  — ScanRefer, Nr3D, Sr3D, ...
        #   OBB  (9-value): (cx, cy, cz, dx, dy, dz, rx, ry, rz) — EmbodiedScan, HoLi-Spatial raw
        _is_obb = item.get("bbox_format") == "obb"
        if (_wants_aligned
                and "<|box_start|>" not in answer_text
                and "bbox_3d" not in answer_text
                and assets.get("poses") is not None):
            # Parse raw boxes from the answer text.
            if _is_obb:
                _raw_boxes = parse_multi_3d_obb(answer_text)
            else:
                _raw_boxes = parse_multi_3d_bbox(answer_text)

            # Use augmented center when active so GT bbox matches the
            # shifted panorama origin; otherwise fall back to camera mean.
            center = _aug_center if _aug_center is not None else get_scene_center(assets["poses"])

            if _is_obb:
                # OBB: center-subtract the (cx, cy, cz) part; keep size + rotation.
                _centered_list = [
                    (b[0] - float(center[0]),
                     b[1] - float(center[1]),
                     b[2] - float(center[2]),
                     b[3], b[4], b[5],
                     b[6], b[7], b[8])
                    for b in _raw_boxes
                ]
                # Apply yaw rotation to OBB: rotate center in XY plane AND
                # add yaw offset to the rz Euler angle. Unlike AABBs, OBBs
                # don't need 90-deg snap restriction — arbitrary yaw is valid.
                if _aug_yaw is not None and _aug_yaw != 0.0:
                    _cos_y = math.cos(_aug_yaw)
                    _sin_y = math.sin(_aug_yaw)
                    _centered_list = [
                        (_cos_y * cx - _sin_y * cy,
                         _sin_y * cx + _cos_y * cy,
                         cz,
                         dx, dy, dz,
                         rx, ry, rz + _aug_yaw)
                        for cx, cy, cz, dx, dy, dz, rx, ry, rz in _centered_list
                    ]
                # Near-to-far ordering (skipped when the annotation's box
                # order is semantic, e.g. grasp-then-release)
                if len(_centered_list) > 1 and not item.get("keep_box_order"):
                    _centered_list.sort(key=lambda b: b[0]**2 + b[1]**2 + b[2]**2)
                # Format as OBB JSON
                if self._metric_json_grounding_format:
                    _label = _extract_short_label(item)
                    answer_text = format_multi_metric_obb_json(_centered_list, label=_label)
                else:
                    # Legacy box-token format not supported for OBB; fall back to JSON.
                    answer_text = format_multi_metric_obb_json(_centered_list)
            else:
                # AABB path (existing behavior)
                _centered_list = [
                    (b[0] - float(center[0]),
                     b[1] - float(center[1]),
                     b[2] - float(center[2]),
                     b[3], b[4], b[5])
                    for b in _raw_boxes
                ]
                # Apply yaw rotation to centered bbox (90-degree snaps for grounding).
                # reproject_scene rotates in the XY plane (ScanNet: X=right, Y=forward,
                # Z=up), so we rotate (cx, cy) and swap (dx, dy) to match.
                if _aug_yaw is not None and _aug_yaw != 0.0:
                    _cos_y = math.cos(_aug_yaw)
                    _sin_y = math.sin(_aug_yaw)
                    _centered_list = [
                        (_cos_y * cx - _sin_y * cy,
                         _sin_y * cx + _cos_y * cy,
                         cz,
                         abs(_cos_y) * dx + abs(_sin_y) * dy,
                         abs(_sin_y) * dx + abs(_cos_y) * dy,
                         dz)
                        for cx, cy, cz, dx, dy, dz in _centered_list
                    ]
                # Canonical near-to-far ordering for multi-box targets
                # (skipped when the annotation's box order is semantic,
                # e.g. grasp-then-release).
                if len(_centered_list) > 1 and not item.get("keep_box_order"):
                    _centered_list.sort(key=lambda b: b[0]**2 + b[1]**2 + b[2]**2)
                if self._pano_grounding_format:
                    if len(_centered_list) > 1:
                        rank0_print(
                            f"[grounding][warn] pano format dropped {len(_centered_list) - 1} "
                            f"box(es) for scene={item.get('scene_id')}"
                        )
                    if _centered_list:
                        _pano_box = world_bbox_to_pano_bbox(_centered_list[0])
                        _label = _extract_short_label(item)
                        answer_text = format_pano_bbox_json(_pano_box, label=_label)
                elif self._metric_json_grounding_format:
                    _label = _extract_short_label(item)
                    answer_text = format_multi_metric_bbox_json(_centered_list, label=_label)
                else:
                    answer_text = format_multi_metric_bbox(_centered_list)
            if self._debug_print_count % self._debug_print_every == 0 or self._debug_print_count < self._debug_print_limit:
                _ds = item.get("dataset_name", "unknown")
                _q = str(item.get("question", "")).replace("\n", " ").strip()[:120]
                rank0_print(f"[grounding][{self.data_split}] {_ds} | Q: {_q} | A: {answer_text}")

        # --- Toolcall point labels: raw world points -> sample canvas frame ---
        # Matched tool-call annotations (question_type == "toolcall_points",
        # agentic-onecanvas scripts/build_toolcall_matched_annotations.py)
        # store distance("name", [x,y,z], ...) / direction(...) calls with
        # points in the RAW ScanNet world frame, i.e. the same frame as the
        # non-grounding QA poses and _aug_center. Mirror of the OBB path
        # above: center-subtract, then CONTINUOUS yaw rotation in XY (points
        # need no 90-degree snap, there is no AABB validity constraint), z
        # stays up. wants_aligned stays False so poses remain raw, matching
        # the QA/eval canvases. Added 2026-07-16 for the matched retrain.
        if (item.get("question_type") == "toolcall_points"
                and assets.get("poses") is not None):
            _pt_re = re.compile(
                r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
                r"\s*(-?\d+(?:\.\d+)?)\s*\]")
            center = (_aug_center if _aug_center is not None
                      else get_scene_center(assets["poses"]))
            _cy = math.cos(_aug_yaw) if _aug_yaw else 1.0
            _sy = math.sin(_aug_yaw) if _aug_yaw else 0.0

            def _xform_pt(m):
                x = float(m.group(1)) - float(center[0])
                y = float(m.group(2)) - float(center[1])
                z = float(m.group(3)) - float(center[2])
                x, y = _cy * x - _sy * y, _sy * x + _cy * y
                return f"[{x:.2f}, {y:.2f}, {z:.2f}]"

            answer_text = _pt_re.sub(_xform_pt, answer_text)
            if self._debug_print_count < self._debug_print_limit:
                rank0_print(
                    f"[toolcall_points][{self.data_split}] "
                    f"scene={item.get('scene_id')} A: {answer_text[:160]}")

        # For Multi3DRefer, route eval_type (st_wo_d / st_w_d / mt / zt_wo_d /
        # zt_w_d) into question_type so the MetricTracker per-type breakdown
        # produces one row per case type instead of a single "grounding" row.
        # Local to multi3drefer items only — other grounding datasets keep the
        # literal "grounding" question_type.
        _resolved_qtype = item.get("question_type", "unknown")
        if (_resolved_qtype == "grounding"
                and item.get("source") == "multi3drefer"
                and item.get("eval_type")):
            _resolved_qtype = item["eval_type"]

        # ==================================================================
        # VANILLA QWEN3-VL BASELINE — no canvas, no 3D, no markers.
        # Standard multi-image SFT inputs only. Returns the minimal field
        # set the stock Qwen3VLForConditionalGeneration forward needs.
        # ==================================================================
        if getattr(self.data_args, "vanilla_qwen3vl", False):
            messages = [{"role": "user", "content": [
                *[{"type": "image", "image": img} for img in assets["images"]],
                {"type": "text", "text": prompt_text},
            ]}]
            if self.with_answer:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": answer_text}]})
            # One-step path: apply_chat_template(tokenize=True, return_dict=True)
            # is the canonical pattern for Qwen3-VL multi-image input. It
            # ensures the number of <|image_pad|> tokens in input_ids exactly
            # matches what the vision tower produces, which the stock forward
            # validates. The two-step pattern (tokenize=False -> processor())
            # used by the canvas dataset is fine for PATH B canvas because
            # the canvas forward replaces tokens with its own projection
            # before that validation runs; vanilla doesn't.
            processed = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=not self.with_answer,
                return_dict=True,
                return_tensors="pt",
                **self.adapter.config.chat_template_kwargs,
            )
            input_ids, labels = self.adapter.mask_labels(processed["input_ids"])
            return {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": processed["attention_mask"].unsqueeze(0),
                "labels": labels.unsqueeze(0),
                "pixel_values": processed["pixel_values"],
                "image_grid_thw": processed["image_grid_thw"],
                "answer": answer_text,
                "question": item["question"],
                "question_type": _resolved_qtype,
                "scene_id": item["scene_id"],
                "_dataloader_retries": attempts,
            }

        # A PER-ITEM TOOL LIST, rendered into the system prompt by the chat
        # template. Self-gating on the field, like the declared-anchor branch
        # above, so leaving it absent is byte-identical to before for every
        # existing dataset.
        #
        # WHY A DATASET WOULD CARRY ONE. An action-space dataset supervises a
        # tool call, and the schema of that call (its argument names, frame and
        # units) belongs in the context the model is conditioned on, not only
        # in the answer. Without it the model is trained to emit a call its own
        # prompt never documented, and a rollout generated WITH the tool block
        # is then reading a different prompt than training wrote. The
        # toolcall_marker branch already does this for synthetic canvases; this
        # is the same capability on the generic real-scene path, and it names
        # no dataset and no consumer.
        _item_tools = item.get("tools") or None

        # A PER-ITEM SYSTEM MESSAGE, rendered ahead of the user turn. Self-gated
        # on the field, exactly like the tool list above, so an item without one
        # is byte-identical to before for every existing dataset.
        #
        # WHY A DATASET WOULD CARRY ONE. An action-space dataset is generated
        # with its instructions in a SYSTEM message, and the field was already
        # being written into rows while nothing here consumed it: the loader
        # dropped it in the annotation prune and never rendered it, so training
        # conditioned on the user turn alone while generation conditioned on
        # system plus user. The two prompts differed, silently, and the
        # difference was invisible in the JSONL. This names no dataset and no
        # consumer.
        _item_system = str(item.get("system_prompt") or "")
        _system_turn = ([{"role": "system", "content": _item_system}]
                        if _item_system else [])

        # 1. Process Origin Images — the forward pass extracts features live.
        messages = _system_turn + [{"role": "user", "content": [*[{"type": "image", "image": img} for img in assets["images"]], {"type": "text", "text": prompt_text}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=_item_tools, **self.adapter.config.chat_template_kwargs)
        processed_raw_images = self.processor(text=[text], images=assets["images"], return_tensors="pt")

        # 2. Process Dummy Image
        # This establishes the grid structure for the projected features
        messages_360 = _system_turn + [
            {"role": "user", "content": [{"type": "image", "image": self.dummy_image}, {"type": "text", "text": prompt_text}]},
        ]
        if self.with_answer:
            messages_360.append({"role": "assistant", "content": [{"type": "text", "text": answer_text}]})

        text = self.processor.apply_chat_template(messages_360, tokenize=False, add_generation_prompt=True, tools=_item_tools, **self.adapter.config.chat_template_kwargs)
        # Override max_pixels so the dummy 360 image is never downscaled by the processor.
        # Its exact size encodes the projection grid tokens, so any
        # resize would produce the wrong number of position IDs.
        _img_w = self._dummy_img_w
        _img_h = self._dummy_img_h
        # Still pass images= so the tokenizer correctly inserts image-pad tokens.
        # pixel_values are overwritten with the cached copy immediately after so the
        # image_processor result (the slow part) is only ever computed once (in __init__).
        processed_360 = self.processor(
            text=[text], images=[self.dummy_image], return_tensors="pt",
            max_pixels=_img_w * _img_h,
        )
        processed_360["pixel_values"]   = self._cached_pixel_values
        processed_360["image_grid_thw"] = self._cached_image_grid_thw

        input_ids_360, labels_360 = self.adapter.mask_labels(processed_360.input_ids)

        # Precompute image token positions on CPU so forward() never needs a GPU nonzero()
        # sync (each GPU nonzero()+.item() stalls the entire CUDA stream). The canvas
        # image_pad tokens form a single contiguous run.
        #
        # READ BEFORE ANY PLACEHOLDER CONVERSION. The declared-reference branch
        # below turns its placeholders into image_pad too, so the dummy block
        # has to be located while it is still the only image_pad run.
        _img_pos = (input_ids_360[0] == self.adapter.config.image_pad_token_id).nonzero(as_tuple=True)[0]
        _img_first_idx = _img_pos[0].item()
        _img_last_idx = _img_pos[-1].item()
        _input_seq_len = input_ids_360.shape[-1]  # true length before collator padding

        # Declared reference points: an item that names points in its prompt and
        # binds each to a fixed row of a declared feature pool. Self-gated on
        # the field, so every existing dataset is untouched.
        _ref_keys = None
        if item.get("reference_points"):
            input_ids_360, labels_360, _ref_keys = \
                self._declared_reference_inputs(item, input_ids_360, labels_360)

        # toolcall_marker (v2): the single-turn block above assumed one Q/A; a
        # marker episode is a whole multi-turn place_marker/python sequence with
        # per-turn supervise masking and inline marker PAIR tokens. Rebuild the
        # dummy-side inputs (the real-image canvas above is shared and unchanged).
        _marker_keys = None
        if item.get("question_type") == "toolcall_marker_points":
            # Resolve the canvas origin the SAME way the toolcall_points branch
            # does: the aug center when augmenting (train), else the scene
            # centroid (val / aug off). The marker points must live in exactly
            # the centered/yawed frame the canvas is reprojected around, or the
            # twins land off the scene (trap 4). Without this fallback val
            # samples (aug_center is None) crash in the coord transform.
            _mk_center = (_aug_center if _aug_center is not None
                          else get_scene_center(assets["poses"]))
            (input_ids_360, labels_360, _img_first_idx, _img_last_idx,
             _input_seq_len, _marker_attn, _marker_keys) = \
                self._toolcall_marker_inputs(item, _mk_center, _aug_yaw)
            processed_360["attention_mask"] = _marker_attn

        # (_aug_center and _aug_yaw were sampled earlier, before bbox centering.)

        # ==================================================================
        # Raw data for forward-pass projection: the model extracts visual
        # features and projects the canvas live during prefill.
        # ==================================================================
        data_dict = {
            "pixel_values": processed_raw_images["pixel_values"],
            "image_grid_thw": processed_raw_images["image_grid_thw"],
            "input_ids": input_ids_360.unsqueeze(0),
            "attention_mask": processed_360.attention_mask.unsqueeze(0),
            "labels": labels_360.unsqueeze(0),
            "image_dims": stack_tensor_list(assets.get("image_dims")),
            "answer": answer_text,
            "question": item["question"],
            "question_type": _resolved_qtype,
            "scene_id": item["scene_id"],
            # CPU-precomputed scalars; collator will torch.stack these into [B] tensors.
            "image_token_first_idx": torch.tensor(_img_first_idx, dtype=torch.long),
            "image_token_last_idx":  torch.tensor(_img_last_idx,  dtype=torch.long),
            "input_seq_len":         torch.tensor(_input_seq_len, dtype=torch.long),
        }

        if _marker_keys is not None:
            data_dict.update(_marker_keys)
        if _ref_keys is not None:
            data_dict.update(_ref_keys)
        data_dict["images"] = assets["images"]
        # Number of source images that landed on the canvas. Equals
        # len(item["images"]) for pinned-frame datasets (SPBench-SI=1,
        # SPBench-MV=8) and 0 elsewhere. Used post-hoc to split SPBench eval
        # results into SI vs MV without re-reading annotation jsonls.
        data_dict["n_source_images"] = torch.tensor(len(item.get("images") or []), dtype=torch.long)
        data_dict["_dataloader_retries"] = attempts
        # Panoramic augmentation params for PATH B (model.forward projection)
        if _aug_center is not None:
            data_dict["aug_center_override"] = _aug_center  # [3] tensor
        if _aug_yaw is not None:
            data_dict["aug_yaw_angle"] = torch.tensor(_aug_yaw, dtype=torch.float32)
        if not self.with_precomputed_geometry:
            pass
        else:
            data_dict["depths"] = stack_tensor_list(prepare_depths(assets.get("depths", [])))
            data_dict["poses"] = stack_tensor_list(assets.get("poses"))
            data_dict["intrinsics"] = stack_tensor_list(assets.get("intrinsics"))

        return data_dict


# Back-compat alias: this dataset serves every real-scene QA source (SQA3D,
# ScanQA, VSI-Bench, ViCA, ...), not just ScanQA. Callers may still import the
# old name.
ScanQALazyDataset = SceneQADataset


def make_supervised_data_module(
    processor,
    data_args,
    select_images_randomly=False,
    with_test=False,
    build_train_dataset=True,
    build_eval_dataset=True,
) -> Dict:
    import re as _re
    _dataset_names = [_re.sub(r"[@%][\d.]+$", "", n) for n in data_args.dataset_use.split(",")]

    has_geometric_probing = "geometric_probing" in _dataset_names
    has_other = any(n != "geometric_probing" for n in _dataset_names)

    # Geometric-probing-only shortcut (on-the-fly synthetic geometric Q-A).
    if has_geometric_probing and not has_other:
        from onecanvas.data.spatial_pretraining import SpatialPretrainingDataset
        train_dataset = SpatialPretrainingDataset(processor, data_args, data_split="train") if build_train_dataset else None
        val_dataset   = SpatialPretrainingDataset(processor, data_args, data_split="val")   if build_eval_dataset else None
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer, data_packing=False)
        data_dict = {"train_dataset": train_dataset, "eval_dataset": val_dataset, "data_collator": data_collator}
        if with_test:
            data_dict["test_dataset"] = SpatialPretrainingDataset(processor, data_args, data_split="val")
        return data_dict

    # Strip geometric_probing from the list passed to SceneQADataset (it has
    # no JSON annotation path — it's built by its own Dataset class). We
    # concatenate it back afterwards.
    _orig_dataset_use = data_args.dataset_use
    _needs_restore = False

    if has_geometric_probing:
        import re as _re_gp
        _gp_spec = next(s for s in data_args.dataset_use.split(",") if s.startswith("geometric_probing"))
        _gp_weight_match = _re_gp.search(r"@([\d.]+)$", _gp_spec)
        _gp_rate_match   = _re_gp.search(r"%(\d+)$", _gp_spec)
        _gp_weight = float(_gp_weight_match.group(1)) if _gp_weight_match else None
        _gp_rate   = int(_gp_rate_match.group(1)) / 100.0 if _gp_rate_match else 1.0

    if has_geometric_probing:
        _filtered = [
            s for s in data_args.dataset_use.split(",")
            if not s.startswith("geometric_probing")
        ]
        data_args.dataset_use = ",".join(_filtered)
        _needs_restore = True

    train_dataset = None
    val_dataset = None

    if build_train_dataset:
        train_dataset = SceneQADataset(
            processor,
            data_args=data_args,
            data_split="train",
            select_images_randomly=select_images_randomly,
        )

    if build_eval_dataset:
        val_dataset = SceneQADataset(processor, data_args=data_args, data_split="val", sort=True)

    if with_test:
        # ONECANVAS_SQA3D_EVAL_SPLIT lets an eval/benchmark run load a different
        # split as the "test" set (e.g. "train" for an overfitting check).
        # Defaults to the real test split, so normal runs are unaffected.
        _eval_split = os.environ.get("ONECANVAS_SQA3D_EVAL_SPLIT", "test")
        test_dataset = SceneQADataset(processor, data_args=data_args, data_split=_eval_split, sort=True)

    # Restore original dataset_use and concatenate extra datasets if needed.
    if _needs_restore:
        data_args.dataset_use = _orig_dataset_use

    # Concat geometric_probing onto the QA mix. SceneQADataset has no
    # annotation file for it, so without this branch it silently loads 0
    # items (see make_supervised_data_module header comment). Train-time
    # sample_weights are carried through a ConcatDataset subclass so that
    # WeightedTrainer's @weight sampling keeps working.
    if has_geometric_probing:
        from onecanvas.data.spatial_pretraining import SpatialPretrainingDataset
        from torch.utils.data import ConcatDataset

        class _WeightedConcatDataset(ConcatDataset):
            """ConcatDataset that exposes .sample_weights for WeightedTrainer."""
            def __init__(self, datasets, sample_weights):
                super().__init__(datasets)
                self.sample_weights = sample_weights

        if build_train_dataset and train_dataset is not None:
            gp_train = SpatialPretrainingDataset(processor, data_args, data_split="train")
            n_gp = len(gp_train)
            _w = _gp_weight if _gp_weight is not None else 1.0
            print(f"[data] geometric_probing@{_w}: {n_gp} items concatenated (train)")

            qa_weights = list(getattr(train_dataset, "sample_weights", []) or [])
            if qa_weights:
                # QA is in weighted mode → append probe weights proportionally.
                gp_weights = [_w / n_gp] * n_gp
                train_dataset = _WeightedConcatDataset(
                    [train_dataset, gp_train], qa_weights + gp_weights
                )
            else:
                # Unweighted QA run → keep concat unweighted too.
                train_dataset = ConcatDataset([train_dataset, gp_train])

        if build_eval_dataset and val_dataset is not None:
            gp_val = SpatialPretrainingDataset(processor, data_args, data_split="val")
            val_dataset = ConcatDataset([val_dataset, gp_val])

    if train_dataset is not None:
        _n = len(train_dataset.list_data_dict) if hasattr(train_dataset, "list_data_dict") else len(train_dataset)
        print(f"[debug] train_dataset samples: {_n}")
    data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer, data_packing=False)
    data_dict = {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
    }

    if with_test:
        data_dict["test_dataset"] = test_dataset

    return data_dict
