"""Spatial-pretraining dataset: synthetic geometric Q-A over real ScanNet scenes."""

import math
import os
import random
import sys
from collections import Counter
from copy import copy

import torch
from torch.utils.data import Dataset, get_worker_info

from ..dataset_utils import prepare_depths, stack_tensor_list
from ..curriculum_task_registry import (
    get_probe_task,
    load_plugins_from_env,
    registered_strip_tasks,
)
from geometry import compute_scene_aabb_from_depths, visibility_matrix, first_visible_frame
from utils.bbox import (
    _sample_obb_surface_points,
    format_multi_metric_bbox,
    format_multi_metric_bbox_json,
    obb_closest_surface_points,
    obb_surface_distance,
)


from ._common import *
from .qa import QAMixin
from .patches import PatchesMixin
from .samplers_metric import MetricMixin
from .samplers_direction import DirectionMixin
from .samplers_navigation import NavigationMixin
from .samplers_observability import ObservabilityMixin
from .samplers_counting import CountingMixin
from .samplers_placement import PlacementMixin


class SpatialPretrainingDataset(QAMixin, PatchesMixin, MetricMixin, DirectionMixin, NavigationMixin, ObservabilityMixin, CountingMixin, PlacementMixin, Dataset):
    """Synthetic geometric Q-A pairs over real ScanNet scenes (live features)."""

    def __init__(self, processor, data_args, data_split="train"):
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.data_split = data_split

        # Build a private ScanQALazyDataset bound to vica_scannet_base just for
        # scene asset loading. We override its dataset_use field locally so we
        # don't disturb the calling code's data_args.
        #
        # skip_asset_validation: ScanQALazyDataset normally calls
        # _scene_has_required_assets() per annotation at construction time,
        # which os.scandir's every scene's color_*/ frame directory over NFS
        # — ~9M stat calls for ScanNet's 89504 scenes, several minutes. The
        # probe retries broken scenes at __getitem__ time
        # (curriculum_scene_retry_attempts), so the upfront check is dead
        # weight here.
        from ..data_processor_3d import ScanQALazyDataset

        _da = copy(data_args)
        _da.dataset_use = str(getattr(
            data_args, "curriculum_scene_sources",
            "vica_scannet_base,vica_arkit_base,vica_snpp_base",
        ))
        _da.skip_asset_validation = True
        self._scene_src = ScanQALazyDataset(
            processor, _da, data_split=data_split, sort=(data_split != "train"),
        )
        self._resample_frames = bool(
            data_split == "train"
            and getattr(data_args, "curriculum_resample_frames", False)
        )
        self._scene_src.select_images_randomly = self._resample_frames
        self._frame_draw_rng = None
        self._frame_draw_worker = None
        self.adapter = self._scene_src.adapter

        # Probe config
        self.num_samples_per_scene = max(
            1, int(getattr(data_args, "curriculum_samples_per_scene", 8))
        )
        # How many random scenes ONE sample may try. See the argument's help:
        # the task is held across retries, so the budget belongs to the
        # least-accepting task in the mixture, not to the average one.
        self._scene_retry_attempts = max(
            1, int(getattr(data_args, "curriculum_scene_retry_attempts", 128))
        )
        _types_str = str(getattr(data_args, "curriculum_task_types", "dist_box"))
        self.task_types = [t.strip() for t in _types_str.split(",") if t.strip()]
        if not self.task_types:
            self.task_types = ["dist_box"]

        # Shuffle marker T-axis for spatial tasks to remove the source-frame
        # shortcut. See _sample_marker_t_override / _PROBE_TEMPORAL_TASKS.
        self._shuffle_marker_t = bool(getattr(data_args, "curriculum_shuffle_marker_t", True))

        # Per-task canvas stripping. Tasks in this set get a pruned canvas
        # (only marker-referenced patches + synthetic OBB points) to remove
        # the room-geometry shortcut. Intended for route_plan_* where the
        # real scene's doors / walls / floors would let the model shortcut
        # navigation reasoning. Other tasks keep the full scene canvas.
        # Load any externally-registered probe tasks (e.g. downstream
        # experiments) named in $ONECANVAS_PROBE_TASK_PLUGINS before the strip
        # set is built. No-op when the env var is unset, so normal training is
        # unaffected. See curriculum_task_registry.py.
        load_plugins_from_env()
        _strip_str = str(getattr(data_args, "curriculum_canvas_obb_only_tasks", ""))
        self._canvas_obb_only_tasks = frozenset(
            t.strip() for t in _strip_str.split(",") if t.strip()
        ).union(registered_strip_tasks())

        # Resolve optional task-specific scene pools once. A sparse task then
        # draws directly from its accepted pool, while every task without a
        # pool keeps the full scene source. This selection happens after the
        # task draw and before any scene load. It does not turn the pool into
        # another accept/retry gate.
        by_scene = {}
        for scene_idx, item in enumerate(self._scene_src.list_data_dict):
            by_scene.setdefault(item["scene_id"], []).append(scene_idx)
        self._task_scene_indices = {}
        self._task_scene_weights = {}
        for task in set(self.task_types):
            spec = get_probe_task(task)
            if spec is None or spec.scene_ids is None:
                continue
            requested = tuple(dict.fromkeys(str(s) for s in spec.scene_ids))
            groups = tuple(tuple(by_scene[scene_id]) for scene_id in requested
                           if scene_id in by_scene)
            indices = tuple(i for group in groups for i in group)
            missing = tuple(s for s in requested if s not in by_scene)
            if not indices:
                raise RuntimeError(
                    f"curriculum task {task!r} has a task-specific scene pool "
                    "but none of its scenes are present after the dataset's "
                    f"ordinary filters. Requested {len(requested)} scene(s)."
                )
            if spec.scene_groups:
                grouped = []
                for source_group in spec.scene_groups:
                    source_indices = tuple(
                        tuple(by_scene[str(scene_id)]) for scene_id in source_group
                        if str(scene_id) in by_scene
                    )
                    if source_indices:
                        grouped.append(source_indices)
                weights = tuple(float(w) for w in (spec.scene_group_weights or ()))
                if len(weights) != len(spec.scene_groups):
                    raise RuntimeError(
                        f"curriculum task {task!r} has {len(spec.scene_groups)} "
                        f"scene groups but {len(weights)} weights"
                    )
                kept_weights = tuple(
                    weight for group, weight in zip(spec.scene_groups, weights)
                    if any(str(scene_id) in by_scene for scene_id in group)
                )
                if not grouped or len(grouped) != len(kept_weights):
                    raise RuntimeError(
                        f"curriculum task {task!r} has an empty weighted scene group"
                    )
                groups = tuple(grouped)
                self._task_scene_weights[task] = kept_weights
            self._task_scene_indices[task] = groups
            # SCENES AND GROUPS ARE DIFFERENT NUMBERS, and this line used to
            # report `len(groups)` as "scene(s)". Without weighted source
            # groups that is the scene count; WITH them it is the number of
            # source groups, so a 226,329-item pool over 1,096 rooms printed
            # "from 3 scene(s)" and read as a pool of three rooms.
            n_scenes = sum(1 for s in requested if s in by_scene)
            print(
                f"[SpatialPretrainingDataset] task scene pool {task}: "
                f"{len(indices)} item(s) from {n_scenes} scene(s) in "
                f"{len(groups)} weighted group(s), "
                f"{len(missing)} absent after ordinary filters",
                file=sys.stderr, flush=True,
            )

        # Variable num_images: parse "2,8,16,32" into a list of ints.
        # Empty string = use the global num_images (no variation).
        _range_str = str(getattr(data_args, "curriculum_num_images_range", ""))
        self._num_images_choices = None
        if _range_str.strip():
            self._num_images_choices = sorted(
                int(x.strip()) for x in _range_str.split(",") if x.strip()
            )
        self._base_num_images = self._scene_src.num_images

        # Chain-of-thought: output intermediate 3D coords before distance.
        self._use_cot = bool(getattr(data_args, "curriculum_use_cot", False))

        # Decimal precision for distance / depth answers.
        self._dist_decimals = int(getattr(data_args, "curriculum_dist_decimals", 1))

        # Per-sample total budget for synthetic OBB body patches. Split
        # uniformly across boxes via _per_box_budget(), floored at 8 corners.
        self._max_body_patches_per_sample = int(
            getattr(data_args, "curriculum_max_body_patches_per_sample", 100))

        # Stochastic difficulty mix for the counting family. See
        # _sample_object_counting / _sample_rel_dir_count_side for the
        # easy/medium/hard schedule.
        self._counting_difficulty_mix = bool(getattr(
            data_args, "curriculum_counting_difficulty_mix", False))

        # Colinear-centers shortcut-buster: per-sample probability of forcing
        # all task OBB centers onto a single ray from the panorama center
        # (shared lat/lon, depth-only distinguished). Applies to dist_box,
        # rel_dist_box, object_counting (+ parity/mod3 siblings),
        # appearance_order_box, and multi_box_grounding. Default 0.1 so new
        # runs pick this up automatically; set to 0.0 to disable.
        self._colinear_centers_prob = float(getattr(
            data_args, "curriculum_colinear_centers_prob", 0.1))

        # Answer format for multi_box_grounding: JSON (Qwen3-VL bbox_3d) vs
        # legacy <|box_start|>(...)<|box_end|> token format. Matches the
        # grounding training flag so the probe's bbox answers train in the
        # same format the downstream grounding task uses.
        self._metric_json_grounding_format = bool(getattr(
            data_args, "metric_json_grounding_format", True))

        # Minimum frame gap for frame_order task (fraction of total frames).
        self._min_frame_gap = float(getattr(data_args, "curriculum_min_frame_gap", 0.0))

        # 3D radius (meters) for local min-T in appearance_order probe.
        self._appearance_radius = float(getattr(data_args, "curriculum_appearance_radius", 0.5))

        # Spread each appearance_order_box surface point across the interval
        # [T_min, num_frames-1] instead of pinning the whole box body at T=0.
        # Forces the model to aggregate box-wise and take the min T, rather
        # than reading a single scalar T off the marker token.
        self._appearance_spread_enable = bool(
            getattr(data_args, "curriculum_appearance_spread_enable", True)
        )

        # Number of boxes/markers in appearance_order* probes (default 4 keeps
        # legacy behaviour). Assigned T_min values are {k, k+1, ..., k+N-1}.
        self._appearance_n_boxes = max(2, int(
            getattr(data_args, "curriculum_appearance_n_boxes", 4)
        ))

        # Open-ended answer format: emit digit-string permutation (e.g. "2413")
        # instead of 4-way MCQ letter. Drops chance baseline from 25% to 1/N!.
        self._appearance_open_ended = bool(
            getattr(data_args, "curriculum_appearance_open_ended", False)
        )

        # When enabled, appearance_order_real uses the same T design as
        # synthetic appearance_order_box (consecutive t_starts, uniform
        # per-patch T over [t_start, n_imgs-1], min(T) is the only
        # differentiating signal).
        self._appearance_real_uniform_t = bool(
            getattr(data_args, "curriculum_appearance_real_uniform_t", False)
        )

        # Per-sample shared real-asset subsample fraction. Draws
        # f ~ LogUniform[subsample_min, 1.0] once per sample and scales every
        # paste's cap_per_paste by f. Decouples total target-class patch
        # volume from instance count N so the model can't use
        # 'volume / per-instance density = N' as a counting shortcut.
        self._real_asset_global_subsample = bool(
            getattr(data_args, "curriculum_real_asset_global_subsample", False)
        )
        self._real_asset_subsample_min = float(
            getattr(data_args, "curriculum_real_asset_subsample_min", 0.05)
        )
        # Clamp to (0, 1] to avoid log(0) and accidental disable via 0.0.
        if self._real_asset_subsample_min <= 0.0:
            self._real_asset_subsample_min = 0.05
        if self._real_asset_subsample_min > 1.0:
            self._real_asset_subsample_min = 1.0

        # Per-item RNG seeding. We do NOT keep a shared self._rng because
        # PyTorch DataLoader pickles the dataset to each worker, so every
        # worker starts with identical RNG state. Build a fresh
        # Random(seed + offset + idx) inside __getitem__ instead.
        self._base_seed = int(getattr(data_args, "dataset_sampling_seed", 42))
        self._seed_offset = 0 if data_split == "train" else 999
        # Distinct offset from the per-item rng seed (base + seed_offset + idx).
        # On the train split seed_offset is 0, so _aug_seed == _base_seed would
        # make aug_rng = Random(base + idx) replay the item rng's own first
        # draws (correlated canvas center/yaw vs. sampled geometry). The 0x5EED
        # offset decorrelates the two streams. curriculum_legacy_aug_seed
        # restores the correlated stream (and eval-split augmentation) that the
        # shipped stage-1 checkpoints actually trained with — reproduction runs
        # only.
        self._legacy_aug_seed = bool(getattr(
            data_args, "curriculum_legacy_aug_seed", False,
        ))
        self._aug_seed = (self._base_seed if self._legacy_aug_seed
                          else self._base_seed + 0x5EED)

        # Panoramic canvas augmentation (training-only). Mirrors the
        # data_processor_3d.py flags so probe runs can match the canvas
        # distribution the downstream QA / grounding runs train on.
        # Applied by passing center_override / yaw_angle to
        # compute_scene_geometry (so probe patch sampling stays consistent
        # with the rendered canvas) AND by surfacing aug_center_override /
        # aug_yaw_angle in the returned batch (so model.forward's
        # reproject_scene renders the same augmented canvas).
        self._panoramic_augment_center = bool(getattr(data_args, "panoramic_augment_center", False))
        self._panoramic_augment_center_sigma = float(getattr(data_args, "panoramic_augment_center_sigma", 0.0))
        self._panoramic_augment_center_uniform = bool(getattr(data_args, "panoramic_augment_center_uniform", False))
        self._panoramic_augment_center_uniform_scene = bool(getattr(data_args, "panoramic_augment_center_uniform_scene", False))
        self._panoramic_augment_center_inflate = float(getattr(data_args, "panoramic_augment_center_inflate", 1.0))
        self._panoramic_augment_yaw = bool(getattr(data_args, "panoramic_augment_yaw", False))

        # Probe marker stash: pre-built pool of scene-agnostic patch features.
        # When enabled, probe samples harvest marker + synthetic-OBB features
        # from this pool instead of loading scene images and running the ViT
        # per sample. Cache is built at train.py startup by
        # maybe_build_obb_feature_stash(); we just read the tensor here.
        self._obb_feature_stash = None
        if bool(getattr(data_args, "curriculum_obb_feature_stash_enable", False)):
            from ..curriculum_obb_feature_stash import load_obb_feature_stash
            self._obb_feature_stash = load_obb_feature_stash(
                data_args, self.adapter.config.feature_prefix,
                processor=self.processor,
            )

        # Real-object asset bank for *_real curriculum tasks. When
        # enabled, the *_real samplers paste harvested assets onto the
        # canvas with their own per-patch features (vs synthetic OBBs that
        # clone one feature across every body point).
        self._real_asset_bank = None
        if bool(getattr(data_args, "real_object_assets_enable", False)):
            classes_str = str(getattr(data_args, "real_object_assets_classes", "") or "")
            classes = {c.strip() for c in classes_str.split(",") if c.strip()} or None
            scene_root = str(getattr(data_args, "real_object_scene_assets_root", "") or "")
            if scene_root:
                from ..real_object_scene_bank import RealObjectSceneBank
                self._real_asset_bank = RealObjectSceneBank(
                    root=scene_root, classes=classes,
                )
            else:
                from ..real_object_asset_bank import RealObjectAssetBank
                _root = getattr(data_args, "real_object_assets_root", "") or \
                        os.environ.get("ONECANVAS_ASSETS_ROOT", "")
                if not _root:
                    raise RuntimeError(
                        "Real-asset bank is enabled but no path was supplied. "
                        "Pass --real_object_assets_root /path/to/asset_bank, "
                        "or export ONECANVAS_ASSETS_ROOT before launching."
                    )
                self._real_asset_bank = RealObjectAssetBank(
                    root=_root,
                    classes=classes,
                )
        # Max paste-time OBB inflation (whole-scene bank only; the legacy
        # per-object bank treats inflation_frac as a no-op). Per-sample
        # draw inflation_frac ~ Uniform[0, max] happens in the *_real
        # samplers; the grounding / appearance-order samplers force 0.0.
        self._real_asset_inflation_max_frac = float(getattr(
            data_args, "real_object_paste_inflation_max_frac", 0.0,
        ))
        # Per-SAMPLE total cap on real-asset patches. Split evenly across the
        # sample's pastes at append time. Padded-batch compute pays for the
        # longest sample in the effective batch, so one 600-patch outlier
        # taxes everyone — bound the total instead of per paste. <= 0 disables.
        self._real_asset_max_per_sample = int(getattr(
            data_args, "real_object_assets_max_patches_per_sample", 200,
        ))
        # Other-class real-asset distractor range for the *_real samplers.
        self._real_asset_distract_min = int(getattr(
            data_args, "real_object_assets_distractors_min", 0,
        ))
        self._real_asset_distract_max = int(getattr(
            data_args, "real_object_assets_distractors_max", 0,
        ))
        if self._real_asset_distract_max < self._real_asset_distract_min:
            self._real_asset_distract_max = self._real_asset_distract_min

        self._n_scenes = len(self._scene_src.list_data_dict)
        if self._n_scenes == 0:
            raise ValueError(
                "SpatialPretrainingDataset: no scenes loaded from vica_scannet_base. "
                "Check that the underlying dataset is configured correctly."
            )
        val_n = getattr(data_args, "val_sample_num", None)
        if data_split != "train" and val_n:
            self._effective_len = min(val_n, self._n_scenes * self.num_samples_per_scene)
        else:
            self._effective_len = self._n_scenes * self.num_samples_per_scene

        _img_desc = (f"variable {self._num_images_choices}"
                     if self._num_images_choices else f"fixed {self._base_num_images}")
        print(
            f"[SpatialPretrainingDataset] {data_split}: {self._n_scenes} scenes x "
            f"{self.num_samples_per_scene} samples = {self._effective_len} effective items, "
            f"task_types={self.task_types}, num_images={_img_desc}"
            f"{', cot=True' if self._use_cot else ''}"
        )

    def __len__(self):
        return self._effective_len

    def __getitem__(self, idx):
        # Reproducibility: pin the ambient torch RNG for this item so the few
        # curriculum draws that read the *global* torch generator (OBB
        # surface-point sampling in ``utils.bbox``, the ``random`` rotation
        # mode, and the sparse-corner ``torch.randperm``) are deterministic at
        # a fixed ``dataset_sampling_seed`` instead of a function of the
        # worker's draw history. ``fork_rng`` save/restores the caller's RNG,
        # so a ``num_workers=0`` run leaves the model's stream untouched. Torch
        # and python use independent PRNGs, so seeding both from the item seed
        # does not correlate them, and the per-item python ``rng`` order below
        # is unaffected (the load-bearing sampler draw order is preserved).
        replay_frame_seed = None
        if isinstance(idx, tuple):
            idx, replay_frame_seed = idx
        idx = int(idx)
        frame_seed = replay_frame_seed
        if self._resample_frames and frame_seed is None:
            worker = get_worker_info()
            worker_key = None if worker is None else (worker.id, worker.seed)
            if self._frame_draw_rng is None or self._frame_draw_worker != worker_key:
                worker_seed = self._base_seed if worker is None else int(worker.seed)
                self._frame_draw_rng = random.Random(
                    worker_seed + self._seed_offset + 0x4F4E4543414E5641
                )
                self._frame_draw_worker = worker_key
            frame_seed = self._frame_draw_rng.randrange(2**63)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._base_seed + self._seed_offset + idx)
            return self._getitem_impl(idx, frame_seed=frame_seed)

    def _getitem_impl(self, idx, frame_seed=None):
        rng = random.Random(self._base_seed + self._seed_offset + idx)
        frame_rng = random.Random(frame_seed) if frame_seed is not None else None
        patch_seed = None if frame_seed is None else frame_seed ^ 0x5041544348524E47
        patch_rng = random.Random(patch_seed) if patch_seed is not None else rng

        # Variable num_images: pick a random value from the configured range
        # for this sample. Override on _scene_src so _load_scene_data uses it.
        # Safe: DataLoader pickles dataset per worker, only one __getitem__
        # runs at a time per worker.
        if self._num_images_choices and self.data_split == "train":
            chosen_num = rng.choice(self._num_images_choices)
            self._scene_src.num_images = chosen_num
        else:
            self._scene_src.num_images = self._base_num_images

        # Draw the probe task ONCE up front so scene retries preserve it.
        # Previously `task` was re-drawn inside _build_sample on every retry,
        # which biased the task distribution away from tasks with stricter
        # scene/patch requirements.
        task = rng.choice(self.task_types)

        task_scene_groups = self._task_scene_indices.get(task)
        task_scene_weights = self._task_scene_weights.get(task)

        attempts = 0
        max_attempts = self._scene_retry_attempts
        reasons = Counter()
        while attempts < max_attempts:
            attempts += 1
            # Use random offsets so a cluster of broken consecutive scenes
            # can't exhaust the retry budget.
            if task_scene_groups:
                if task_scene_weights:
                    source_group = rng.choices(
                        task_scene_groups, weights=task_scene_weights, k=1
                    )[0]
                    group = rng.choice(source_group)
                elif attempts <= 1:
                    group = task_scene_groups[idx % len(task_scene_groups)]
                else:
                    group = rng.choice(task_scene_groups)
                scene_idx = group[rng.randrange(len(group))]
            elif attempts <= 1:
                scene_idx = idx % self._n_scenes
            else:
                scene_idx = (idx + rng.randrange(self._n_scenes)) % self._n_scenes
            item = self._scene_src.list_data_dict[scene_idx]
            scene_id_try = item["scene_id"]

            assets = self._scene_src._load_scene_data(
                scene_id_try, item["data_path"], sample_idx=idx,
                pinned_frames=item.get("pinned_stems") or item.get("images"),
                scene_subdir=item.get("scene_subdir"),
                wants_aligned=False,
                frame_rng=frame_rng,
            )
            if assets is None:
                reasons["assets=None"] += 1
                print(
                    f"[SpatialPretrainingDataset] switching scene (idx={idx} task={task} "
                    f"attempt={attempts}): scene {scene_id_try} assets=None",
                    file=sys.stderr, flush=True,
                )
                continue
            if not assets.get("images"):
                reasons["no images"] += 1
                print(
                    f"[SpatialPretrainingDataset] switching scene (idx={idx} task={task} "
                    f"attempt={attempts}): scene {scene_id_try} has no images",
                    file=sys.stderr, flush=True,
                )
                continue

            depths_t = stack_tensor_list(prepare_depths(assets.get("depths", [])))
            poses_t = stack_tensor_list(assets.get("poses"))
            intrinsics_t = stack_tensor_list(assets.get("intrinsics"))
            image_dims_t = stack_tensor_list(assets.get("image_dims"))
            if depths_t is None or poses_t is None or intrinsics_t is None or image_dims_t is None:
                reasons["missing depths/poses/intrinsics"] += 1
                print(
                    f"[SpatialPretrainingDataset] switching scene (idx={idx} task={task} "
                    f"attempt={attempts}): scene {scene_id_try} missing depths/poses/intrinsics",
                    file=sys.stderr, flush=True,
                )
                continue

            try:
                return self._build_sample(
                    assets=assets,
                    depths_t=depths_t,
                    poses_t=poses_t,
                    intrinsics_t=intrinsics_t,
                    image_dims_t=image_dims_t,
                    rng=patch_rng,
                    scene_id=scene_id_try,
                    task=task,
                    idx=idx,
                    frame_seed=frame_seed,
                    patch_seed=patch_seed,
                    source_item=item,
                )
            except RuntimeError as e:
                # Patch sampling miss or tokenizer marker mismatch — retry next scene
                # with the SAME task.
                reasons[str(e).split("(")[0].strip()] += 1
                print(
                    f"[SpatialPretrainingDataset] switching scene (idx={idx} task={task} "
                    f"attempt={attempts}): scene {scene_id_try} build_sample failed: {e}",
                    file=sys.stderr, flush=True,
                )
                continue

        # THE CENSUS IS PART OF THE ERROR. An exhausted draw kills the run, and
        # the one thing the reader needs is whether the task's own accept gates
        # refused every scene (raise curriculum_scene_retry_attempts, or loosen
        # the gate) or the scene assets were unreadable (a data problem). Both
        # used to read as the same one-line failure with 128 stderr lines
        # scattered through a 1.4 MB log to reconstruct it from.
        census = ", ".join(f"{n}x {r}" for r, n in reasons.most_common(5))
        raise RuntimeError(
            f"SpatialPretrainingDataset: failed to load any usable scene for task "
            f"{task!r} after {max_attempts} attempts (idx={idx}). Rejections: "
            f"{census or 'none recorded'}. If the gates refused every scene, the "
            f"task's per-scene acceptance is below ~{3.0 / max_attempts:.1%} and "
            f"--curriculum_scene_retry_attempts is too low for this mixture."
        )

    def _per_box_budget(self, n_boxes: int):
        """Compute the per-box body-patch budget for the current sample.

        Splits ``self._max_body_patches_per_sample`` uniformly across boxes,
        floored at 8 (the OBB corner count) so every box always gets its 8
        corners. The caller passes this to _sample_obb_surface_points as
        ``n_total``; the sampler allocates 8 corners + (n_total - 8) face
        samples."""
        n_boxes = max(1, int(n_boxes))
        return max(8, self._max_body_patches_per_sample // n_boxes)

    def _reprojection_config(self):
        """Delegate to the inner ScanQALazyDataset.

        Train.py's live-features wiring loop ([train.py:1014-1024]) walks the
        train/eval datasets looking for a ``_reprojection_config`` method to
        attach to ``model.reprojection_config``. Without this delegation the
        loop returns ``None``, the model never gets its reprojection config,
        and the first PATH B forward call crashes with
        "Live-features path requires self.reprojection_config to be set."
        """
        return self._scene_src._reprojection_config()

    def _build_sample(self, assets, depths_t, poses_t, intrinsics_t, image_dims_t,
                      rng, scene_id, task, idx=-1, frame_seed=None,
                      patch_seed=None, source_item=None):
        """PATH B sample: raw images + geometry + inline patch marker indices.

        Mirrors data_processor_3d.py's PATH B branch (raw image processing for
        the live visual encoder + dummy 360 image for the canvas slot in
        input_ids), with inline patch markers spliced into the question text.

        Patch picking happens here (after ``processed_raw_images`` so we know
        the per-frame feature grid) rather than in ``__getitem__``: that lets
        us call ``compute_scene_geometry`` with the actual ``H_feat``/``W_feat``
        the live visual encoder will produce, instead of relying on a stub.
        """
        from reprojection import compute_scene_geometry

        with_answer = (self.data_split == "train")

        # 1. Process per-frame images for the live visual encoder. Use the
        # bare image_processor (no text, no chat template) — it returns the
        # same `pixel_values` and `image_grid_thw` that the model.forward()
        # PRE-PATH-A branch consumes ([model.py:319-347]) and skips the
        # tokenizer + template work, which the probe doesn't need on the
        # per-frame side (the canvas slot in input_ids comes from the dummy
        # 360 image processed separately below).
        processed_raw_images = self.processor.image_processor(
            images=assets["images"], return_tensors="pt",
        )

        # Derive (H_feat, W_feat) from the actual per-frame image_grid_thw the
        # processor produced. Mirrors model.py:341-347 which uses
        # `(gt[:, 1] // merge, gt[:, 2] // merge)` for the same per-frame grid.
        merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        gt = processed_raw_images["image_grid_thw"]  # [N_imgs, 3]: T, H_raw, W_raw
        h_per = (gt[:, 1] // merge_size).tolist()
        w_per = (gt[:, 2] // merge_size).tolist()
        H_feat, W_feat = h_per[0], w_per[0]
        if not all(h == H_feat and w == W_feat for h, w in zip(h_per, w_per)):
            # Heterogeneous frame shapes within a sample would also break
            # the live forward path ([model.py:345-347] has the same assert).
            raise RuntimeError(
                f"SpatialPretrainingDataset: heterogeneous frame feature grids in one sample "
                f"({h_per=}, {w_per=}); the live forward path requires homogeneous shapes."
            )

        # 2. Panoramic augmentation (training-only). Sampled here so the
        # per-item spherical coords geom computes — used downstream for patch
        # sampling and synthetic-OBB placement — match the canvas the model's
        # reproject_scene will render with aug_center_override / aug_yaw_angle.
        # rel_dir_camera_* overrides the random panoramic aug with a camera-
        # pose-based aug so the canvas is reoriented onto a real scene camera,
        # matching the eval-time SPBench-SI canvas when spbench_use_camera_pose
        # is on.
        _rel_dir_cam_idx = None
        if (task.startswith("rel_dir_camera_")
                or task.startswith("object_class_rel_dir_camera_")):
            _aug_center, _aug_yaw, _rel_dir_cam_idx = self._sample_camera_pose_aug(assets, idx)
            if _aug_center is None:
                # No valid poses — retry with a different scene.
                raise RuntimeError(
                    f"SpatialPretrainingDataset: scene {scene_id} has no valid poses for "
                    f"camera-aug task (task={task})."
                )
        else:
            _aug_center, _aug_yaw = self._sample_panoramic_aug(assets, idx)

        # 3. Geometry-only reprojection. No features needed: lat/lon/depth/
        # n_valid/frame_index/center_point/poses are pure functions of geometry,
        # and the live forward path will rebuild scene.embeds from real visual
        # features at the same patch indices we pick here.
        geom = compute_scene_geometry(
            depths=depths_t,
            poses=poses_t,
            intrinsics=intrinsics_t,
            image_dims=image_dims_t,
            H_feat=H_feat,
            W_feat=W_feat,
            device="cpu",
            center_override=_aug_center,
            yaw_angle=_aug_yaw,
        )

        # The numerical observation stream is independent of the VLM image
        # sample and of the task.  Every argument below is BOUND HERE, before
        # the task sampler receives geometry, so no question, target box or
        # answer can affect frame selection or the depth grid.
        #
        # THE CALL IS DEFERRED, and that is a cost fix rather than a semantic
        # one.  A 128-frame build is 95 to 136 s per scene at a 256x192 native
        # grid, and the in-situ samplers reject roughly thirteen of every
        # fourteen scenes they draw on gates that never touch the cloud
        # (layout, naming pool, visibility).  Building eagerly paid the whole
        # cost on every rejected draw: the first review harvest under this
        # stream produced 12 records in 12 minutes across four processes.  The
        # thunk is called by `episode_cloud` the first time a builder actually
        # reads observations, so a rejected layout costs nothing and an
        # accepted one gets exactly the same bytes.
        if (str(task).startswith("insitu_")
                and os.environ.get("ONECANVAS_NUMERICAL_OBSERVATIONS", "0") == "1"):
            from onecanvas.data.numerical_observations import (
                load_numerical_observations)
            cache_root = os.environ.get("ONECANVAS_NUMERICAL_CACHE_ROOT")
            if not cache_root:
                raise RuntimeError(
                    "ONECANVAS_NUMERICAL_OBSERVATIONS=1 requires "
                    "ONECANVAS_NUMERICAL_CACHE_ROOT")
            _obs_args = dict(
                scene_source=self._scene_src, item=source_item,
                visual_frame_keys=tuple(
                    getattr(self._scene_src, "_last_frame_keys", None) or []),
                cache_root=cache_root,
                max_frames=int(os.environ.get(
                    "ONECANVAS_NUMERICAL_MAX_FRAMES", "128")),
                grid_cap=(
                    int(os.environ.get("ONECANVAS_NUMERICAL_MAX_WIDTH", "640")),
                    int(os.environ.get("ONECANVAS_NUMERICAL_MAX_HEIGHT", "480"))),
            )
            geom._numerical_observation_loader = (
                lambda a=_obs_args: load_numerical_observations(
                    a["scene_source"], a["item"],
                    visual_frame_keys=a["visual_frame_keys"],
                    cache_root=a["cache_root"], max_frames=a["max_frames"],
                    grid_cap=a["grid_cap"]))

        # rel_dir_camera_* stashes the canvas-local viewer (camera) position +
        # forward direction on the geometry object so the question-answer
        # branch can compute the ego label without re-deriving the pose. The
        # canvas is already reoriented so the camera sits at the origin with
        # forward along +Z in canvas-local coords.
        if task.startswith("rel_dir_camera_") or task.startswith("object_class_rel_dir_camera_"):
            geom._rel_dir_camera_pos = torch.zeros(3, dtype=torch.float32)
            geom._rel_dir_camera_fwd = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            geom._rel_dir_camera_frame_idx = _rel_dir_cam_idx

        # 3. Sample patch indices from the geometric scene for the given task.
        # `task` is drawn in __getitem__ so it stays fixed across scene retries.
        patch_indices, frame_indices = self._sample_patches(task, geom, rng)
        if patch_indices is None:
            # The sampler returned None — this usually means the task's layout
            # constraints (e.g. waypoint clearance, asset placement, frame
            # spread, marker count) couldn't be satisfied for this scene's
            # geometry, NOT necessarily that n_valid is too low. The retry
            # loop in __getitem__ will pick a different scene with the same
            # task. n_valid is reported only as one possible cause.
            raise RuntimeError(
                f"task layout did not fit (task={task}, n_valid={geom.n_valid})"
            )
        # Place M ∈ [min, max] distractor OBBs with unique stash features per
        # distractor. No-op when curriculum_num_distractors_max <= 0 or for
        # *floor_area* tasks. Must run AFTER _sample_patches so collision +
        # target-AABB checks see the task's placed boxes.
        self._place_distractor_boxes(geom, task, rng)
        question_text, answer_text = self._build_question_and_answer(
            task, geom, patch_indices, frame_indices, rng,
        )
        task_label = f"curriculum_{task}"

        # Per-task patch-count diagnostic: total tokens the model sees for
        # this sample's markers + synthetic OBB body + real-asset pastes.
        # With canvas_obb_only these are the ONLY patches emitted.
        # Useful for deciding per-task lower bounds on
        # --curriculum_max_body_patches_per_sample. Deduplicated on task only:
        # one line per task per worker (counts vary continuously per task,
        # so the previous (task, count) dedup barely fired).
        _mbs = getattr(geom, "_multi_box_spherical", None)
        if _mbs is not None:
            _n_obb_body = sum(int(t.shape[0]) for t in _mbs)
        else:
            _synth = getattr(geom, "_synthetic_spherical", None)
            _n_obb_body = int(_synth.shape[0]) if _synth is not None else 0
        _n_markers = len(patch_indices)
        _walls = getattr(geom, "_extra_obb_spherical", None) or []
        _n_walls = sum(int(t.shape[0]) for t in _walls)
        # Real-asset paste patch count. Mirrors the pool-prune cap
        # applied at the actual paste site: _max_per_sample is the TOTAL
        # per-sample cap; budget_real = max_per_sample - (markers + obb +
        # walls); real patches collectively pruned uniformly down to
        # budget_real. Pre-projection upper bound; actual count may be
        # slightly lower if patches fall off-canvas.
        _real_pastes = getattr(geom, "_real_asset_pastes", None) or []
        _max_ps = int(self._real_asset_max_per_sample)
        _avail_real = sum(int(p["asset"]["features"].shape[0])
                          for p in _real_pastes)
        if _max_ps > 0:
            _budget_real = max(0, _max_ps - (_n_markers + _n_obb_body + _n_walls))
            _n_real = min(_avail_real, _budget_real)
        else:
            _n_real = _avail_real
        _total_patches = _n_markers + _n_obb_body + _n_walls + _n_real
        # Gate to rank 0 so the warmup spam doesn't multiply by 8 ranks.
        # Per-task dedupe is per-worker-process; with 6 workers per rank and
        # ~30 tasks this caps at ~180 lines during warmup, then silent.
        _rank0 = int(os.environ.get("RANK", "0")) == 0
        if _rank0:
            _seen = getattr(type(self), "_logged_patch_counts", None)
            if _seen is None:
                _seen = set()
                type(self)._logged_patch_counts = _seen
            if task not in _seen:
                _seen.add(task)
                print(
                    f"[probe-patches] task={task:<38} "
                    f"patches={_total_patches:<4} (markers={_n_markers}, "
                    f"obb_body={_n_obb_body}, walls={_n_walls}, "
                    f"real_assets={_n_real})"
                )

        # Decorrelate the marker's T-axis from its source frame for spatial
        # probe tasks. See _sample_marker_t_override: we replace the marker's
        # T with the T of a random canvas patch from a different frame, so
        # the model can't shortcut spatial reasoning by reading the source
        # frame index (which leaks physical position via video trajectory).
        inline_patch_t_indices = None
        if (self._shuffle_marker_t
                and not _is_temporal_probe_task(task)
                and len(patch_indices) > 0):
            inline_patch_t_indices = torch.tensor(
                _sample_marker_t_override(geom, patch_indices, frame_indices, rng),
                dtype=torch.long,
            )

        # Per-point T spread for appearance_order_box: each synthetic box
        # surface point draws T ~ Uniform{T_min_k, ..., num_frames-1} so the
        # box body is visible across many frames with T_min as the earliest.
        # Also override each marker's T to a frame > T_min_k so the marker
        # alone can't leak the answer — the model must aggregate across the
        # body patches and take the minimum.
        synthetic_patch_t_overrides = None
        if (task in ("appearance_order", "appearance_order_box")
                and self._appearance_spread_enable
                and getattr(geom, "_multi_box_spherical", None) is not None):
            n_frames = int(geom.frame_index.max().item()) + 1
            per_box_pts = [t.shape[0] for t in geom._multi_box_spherical]
            # For appearance_order the sampler stashes the exact per-box
            # visibility mask; draw body-point T values only from frames
            # that actually observe the box. For appearance_order_box there
            # is no visibility notion — the box's feature source was
            # captured at T_min and we just pretend the texture "persists"
            # across later frames to force min-aggregation.
            vis_rows = getattr(geom, "_appearance_visible_frames", None)
            t_override_rows: list = []
            for k, n_pts in enumerate(per_box_pts):
                t_min_k = int(frame_indices[k])
                if vis_rows is not None:
                    # Per-box visible frames (guaranteed >= 2 by the sampler).
                    vis_k = vis_rows[k].nonzero(as_tuple=True)[0].tolist()
                    vals = [rng.choice(vis_k) for _ in range(n_pts)]
                else:
                    hi = n_frames - 1
                    if hi <= t_min_k:
                        vals = [t_min_k] * n_pts
                    else:
                        vals = [rng.randint(t_min_k, hi) for _ in range(n_pts)]
                # Force at least one point at T_min so the min is
                # always exactly the ground-truth answer.
                if vals:
                    vals[0] = t_min_k
                    rng.shuffle(vals)
                t_override_rows.extend(vals)
            synthetic_patch_t_overrides = torch.tensor(t_override_rows, dtype=torch.long)

            # Marker T override: pick a real-scene patch at frame > T_min_k
            # (fallback: any frame != T_min_k) so marker T doesn't leak T_min.
            # For appearance_order the marker should additionally NOT sit at
            # any frame where the box is visible — if it did, the model
            # could grab that one T as a shortcut for t_min_k.
            frame_idx_t = geom.frame_index
            marker_t_indices: list = []
            for k in range(len(patch_indices)):
                t_min_k = int(frame_indices[k])
                if vis_rows is not None:
                    # Prefer any patch whose frame is NOT in the box's
                    # visible set. Fallback to !=t_min_k, then any patch.
                    vis_k_set = set(vis_rows[k].nonzero(as_tuple=True)[0].tolist())
                    frame_idx_list = frame_idx_t.tolist()
                    non_vis_mask = torch.tensor(
                        [int(f) not in vis_k_set for f in frame_idx_list],
                        dtype=torch.bool,
                    )
                    cand = non_vis_mask.nonzero(as_tuple=True)[0]
                    if cand.numel() == 0:
                        cand = (frame_idx_t != t_min_k).nonzero(as_tuple=True)[0]
                else:
                    cand = (frame_idx_t > t_min_k).nonzero(as_tuple=True)[0]
                    if cand.numel() == 0:
                        cand = (frame_idx_t != t_min_k).nonzero(as_tuple=True)[0]
                if cand.numel() == 0:
                    marker_t_indices.append(int(patch_indices[k]))
                else:
                    marker_t_indices.append(int(cand[rng.randrange(cand.numel())].item()))
            inline_patch_t_indices = torch.tensor(marker_t_indices, dtype=torch.long)

        # 4. Process the dummy 360 image: produces input_ids/attention_mask/
        # labels carrying the canvas image-pad slot. The live forward pass
        # splices that slot with N_valid projected tokens via prepare_batch.
        messages_360 = [
            {"role": "user", "content": [
                {"type": "image", "image": self._scene_src.dummy_image},
                {"type": "text", "text": question_text},
            ]},
        ]
        if with_answer:
            messages_360.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
            )
        text_360 = self.processor.apply_chat_template(
            messages_360, tokenize=False, add_generation_prompt=True,
            **self.adapter.config.chat_template_kwargs,
        )
        _img_w = self._scene_src._dummy_img_w
        _img_h = self._scene_src._dummy_img_h
        processed_360 = self.processor(
            text=[text_360], images=[self._scene_src.dummy_image], return_tensors="pt",
            max_pixels=_img_w * _img_h,
        )
        processed_360["pixel_values"] = self._scene_src._cached_pixel_values
        processed_360["image_grid_thw"] = self._scene_src._cached_image_grid_thw

        input_ids_360, labels_360 = self.adapter.mask_labels(processed_360.input_ids)

        # Swap inline patch-marker placeholders -> <|image_pad|>.
        _OBJ_REF_ID = self.adapter.config.object_ref_start_token_id
        _img_pad_id = self.adapter.config.image_pad_token_id
        _marker_mask = input_ids_360[0] == _OBJ_REF_ID
        n_marker_tokens = int(_marker_mask.sum().item())
        if n_marker_tokens != len(patch_indices):
            raise RuntimeError(
                f"SpatialPretrainingDataset: marker count mismatch "
                f"(found {n_marker_tokens}, expected {len(patch_indices)}). "
                f"Question: {question_text!r}"
            )
        input_ids_360[0, _marker_mask] = _img_pad_id
        labels_360[0, _marker_mask] = -100

        # Detect main image region vs inline marker positions: the main image
        # region is the first contiguous run of image_pad tokens; any image_pad
        # after a gap is an inline patch marker.
        _img_pos = (input_ids_360[0] == _img_pad_id).nonzero(as_tuple=True)[0]
        _img_first_idx = int(_img_pos[0].item())
        _inline_patch_positions: list = []
        if len(_img_pos) > 1:
            _diffs = _img_pos[1:] - _img_pos[:-1]
            _gaps = (_diffs > 1).nonzero(as_tuple=True)[0]
            if len(_gaps) > 0:
                _img_last_idx = int(_img_pos[_gaps[0]].item())
                _inline_patch_positions = _img_pos[_gaps[0] + 1:].tolist()
            else:
                _img_last_idx = int(_img_pos[-1].item())
        else:
            _img_last_idx = int(_img_pos[-1].item())

        if len(_inline_patch_positions) != len(patch_indices):
            raise RuntimeError(
                f"SpatialPretrainingDataset: detected {len(_inline_patch_positions)} "
                f"inline marker positions but expected {len(patch_indices)}."
            )

        _input_seq_len = input_ids_360.shape[-1]

        # Marker-stash FAST PATH: when enabled, construct the ReprojectedScene
        # directly from stash features + synthetic geometry in the dataloader
        # worker, run strip-canvas / synthetic-OBB-append / prepare_batch here
        # (CPU work parallelized across workers), and emit a PATH A batch.
        # The forward then skips PRE-PATH-A entirely — no ViT, no
        # reproject_scene, no per-sample CPU geometry on the main GPU process.
        if self._obb_feature_stash is not None:
            return self._build_sample_stash_fast(
                geom=geom,
                patch_indices=patch_indices,
                frame_indices=frame_indices,
                inline_patch_t_indices=inline_patch_t_indices,
                synthetic_patch_t_overrides=synthetic_patch_t_overrides,
                input_ids_360=input_ids_360,
                labels_360=labels_360,
                attention_mask_360=processed_360.attention_mask,
                inline_patch_positions=_inline_patch_positions,
                first_idx=_img_first_idx,
                last_idx=_img_last_idx,
                input_seq_len=_input_seq_len,
                question_text=question_text,
                answer_text=answer_text,
                task_label=task_label,
                scene_id=scene_id,
                task=task,
                rng=rng,
                idx=idx,
            )

        # Real-object asset pastes for the live PATH B forward: precomputed
        # rows (features + spherical + T), because the forward's strip/append
        # block has neither the asset bank nor an rng. See _real_paste_rows
        # for why the slow path must carry these at all. n_nonreal mirrors
        # the fast path's budget base: the dedup'd keep-set plus the
        # synthetic OBB body rows.
        def _rp_ids(v):
            if v is None:
                return set()
            return set(int(i) for i in (v.tolist() if hasattr(v, "tolist") else v))

        _rp_nonreal = len(
            _rp_ids(patch_indices)
            | _rp_ids(getattr(geom, "_multi_box_feature_sources", None))
            | _rp_ids(getattr(geom, "_synthetic_per_point_feature_sources", None))
            | _rp_ids(getattr(geom, "_distractor_feature_sources", None))
            | _rp_ids(getattr(geom, "_extra_obb_feature_sources", None))
        )
        _rp_synth_sph = getattr(geom, "_multi_box_spherical", None)
        if _rp_synth_sph is not None:
            _rp_nonreal += int(sum(t.shape[0] for t in _rp_synth_sph))
        elif getattr(geom, "_synthetic_spherical", None) is not None:
            _rp_nonreal += int(geom._synthetic_spherical.shape[0])
        _rp_rows = self._real_paste_rows(geom, rng, _rp_nonreal)

        data_dict = {
            # Raw image branch (live visual encoder input).
            "pixel_values":   processed_raw_images["pixel_values"],
            "image_grid_thw": processed_raw_images["image_grid_thw"],
            # Canvas slot — input_ids carry the dummy 360 image-pad region.
            "input_ids":      input_ids_360.unsqueeze(0),
            "attention_mask": processed_360.attention_mask.unsqueeze(0),
            "labels":         labels_360.unsqueeze(0),
            # Geometry consumed by the live forward path.
            "depths":     depths_t,
            "poses":      poses_t,
            "intrinsics": intrinsics_t,
            "image_dims": image_dims_t,
            # Image-pad region bookkeeping (so forward() can locate the slot
            # without re-running nonzero() on GPU).
            "image_token_first_idx": torch.tensor(_img_first_idx, dtype=torch.long),
            "image_token_last_idx":  torch.tensor(_img_last_idx,  dtype=torch.long),
            "input_seq_len":         torch.tensor(_input_seq_len, dtype=torch.long),
            # Inline patch markers — variable-length per sample. The live
            # forward path slices these per-sample and forwards them to
            # adapter.prepare_batch.
            "inline_patch_positions":     torch.tensor(_inline_patch_positions, dtype=torch.long),
            "inline_patch_indices": torch.tensor(patch_indices, dtype=torch.long),
            # Optional per-marker override for the MRoPE T-axis: for spatial
            # probe tasks, these point at canvas patches from OTHER frames so
            # the marker's T no longer leaks its own source frame. None for
            # temporal tasks (same_frame / frame_order / appearance_order*).
            "inline_patch_t_indices": inline_patch_t_indices,
            # Direct per-marker T integers (None in the non-stash PATH B flow).
            "inline_patch_t_values": None,
            # Marker-stash overlay fields: used only by the old forward-side
            # stash branch (now superseded by _build_sample_stash_fast, which
            # returns a PATH A dict). Kept at None so flag-off path is a no-op.
            "stash_overlay_indices":  None,
            "stash_overlay_features": None,
            # Per-sample canvas-strip flag. When True (set for tasks listed
            # in --curriculum_canvas_obb_only_tasks), the model.forward PATH B
            # removes all real-scene canvas patches except the marker
            # references, then appends the synthetic OBB points. Used to
            # drop walls/floors for route tasks where scene context would
            # leak the answer.
            "canvas_obb_only": bool(task in self._canvas_obb_only_tasks),
            # patch_exists / patch_exists_pair probe: canvas patch indices
            # to zero out after copying features to marker tokens.
            # _hide_patch_indices (list of specific indices) takes priority
            # over the boolean _hide_source_patches (all-or-nothing).
            "hide_source_patches": torch.tensor(
                getattr(geom, "_hide_patch_indices",
                        patch_indices if getattr(geom, "_hide_source_patches", False) else []),
                dtype=torch.long,
            ),
            # box_size / object_counting probes: synthetic patch data for scene extension.
            # None for other tasks (collator keeps as list via _list_keys).
            # Multi-box tasks (dist_box / rel_dist_box*) use per-box feature sources;
            # single-box tasks (box_size / object_counting) use the scalar path.
            "synthetic_patch_spherical": (
                torch.cat(geom._multi_box_spherical, dim=0)
                if getattr(geom, "_multi_box_spherical", None) is not None
                else getattr(geom, "_synthetic_spherical", None)
            ),
            "synthetic_patch_feature_source": torch.tensor(
                getattr(geom, "_synthetic_feature_source", -1), dtype=torch.long),
            # Per-point random feature sources (box_floor_area family): one
            # scene-patch index per synthetic surface point so the slab has
            # heterogeneous textures, distribution-matching a real room.
            # None for other tasks (collator keeps as list via _list_keys).
            "synthetic_patch_per_point_sources": getattr(
                geom, "_synthetic_per_point_feature_sources", None),
            # Per-box feature sources and point counts for multi-box distance tasks.
            # Empty tensors for single-box tasks (collator keeps as list via _list_keys).
            "synthetic_patch_feature_sources_per_box": torch.tensor(
                getattr(geom, "_multi_box_feature_sources", []), dtype=torch.long),
            "synthetic_patch_box_sizes": torch.tensor(
                [t.shape[0] for t in geom._multi_box_spherical]
                if getattr(geom, "_multi_box_spherical", None) is not None else [],
                dtype=torch.long,
            ),
            # Per-point MRoPE T override for synthetic canvas patches. Used
            # by appearance_order_box to give each box body a Uniform{T_min_k,
            # ..., num_frames-1} T spread so the model must take the min T
            # across the box, not read a single scalar off the marker.
            # None for other tasks (collator keeps it as a Python list).
            "synthetic_patch_t_overrides": synthetic_patch_t_overrides,
            # Real-object asset paste rows for PATH B (None for tasks without
            # pastes; collator keeps them as per-sample lists via _list_keys,
            # forward slices [b] and concatenates onto the reprojected scene).
            "real_paste_embeds":      (_rp_rows or {}).get("embeds"),
            "real_paste_spherical":   (_rp_rows or {}).get("spherical"),
            "real_paste_frame_index": (_rp_rows or {}).get("frame_index"),
            # Metadata.
            "answer":        answer_text,
            "question":      question_text,
            "question_type": task_label,
            "scene_id":      scene_id,
            "images":        assets.get("images", []),
            # Reproduction key: the inputs that pin this sample (patch picks,
            # MCQ shuffle, frame choice). At a fixed base_seed and unchanged
            # dataset code, dataset[idx] regenerates the same sample; the torch
            # draws are pinned per item in __getitem__.
            "_repro": {
                "idx": int(idx),
                "split": self.data_split,
                "base_seed": self._base_seed,
                "seed_offset": self._seed_offset,
                "dataset_use": getattr(self.data_args, "dataset_use", None),
                "task": task,
                "scene_id": scene_id,
                "patch_indices": [int(x) for x in patch_indices],
                "frame_indices": [int(x) for x in (frame_indices or [])],
                "num_images": int(self._scene_src.num_images),
                "source_frame_keys": [str(k) for k in (
                    getattr(self._scene_src, "_last_frame_keys", None) or []
                )],
                "frame_resampling": bool(self._resample_frames),
                "frame_rng_seed": frame_seed,
                "frame_rng_replay": "dataset[(idx, frame_rng_seed)]",
                "patch_rng_seed": patch_seed,
            },
            # Visualizer-only extras. Small Python-object dict; collator keeps it
            # as a list via _list_keys. Training code ignores it.
            "_debug": {
                "center_point": (
                    geom.center_point.detach().cpu().clone()
                    if geom.center_point is not None else None
                ),
                "patch_latitude":  geom.latitude.detach().cpu().clone(),
                "patch_longitude": geom.longitude.detach().cpu().clone(),
                "patch_depth":     geom.depth.detach().cpu().clone(),
                "patch_frame_index": geom.frame_index.detach().cpu().clone(),
                "multi_box_centers":   [c.detach().cpu().clone() for c in
                                        getattr(geom, "_multi_box_centers", []) or []],
                "multi_box_dims":      list(getattr(geom, "_multi_box_dims", []) or []),
                "multi_box_rotations": [R.detach().cpu().clone() for R in
                                        getattr(geom, "_multi_box_rotations", []) or []],
                # Canonical N+2 unique path centers for route_plan_*_box
                # (distinct from the expanded multi_box_centers list, which
                # has duplicates matching the question's placeholder order).
                # Visualizer uses these to draw turn arrows in path order.
                "route_plan_box_path_centers": [
                    c.detach().cpu().clone()
                    for c in getattr(geom, "_route_plan_box_path_centers", []) or []
                ],
                "route_plan_box_path_dims": list(
                    getattr(geom, "_route_plan_box_path_dims", []) or []
                ),
                "route_plan_box_classes": list(
                    getattr(geom, "_route_plan_box_classes", []) or []
                ),
                "route_plan_reface": bool(
                    getattr(geom, "_route_plan_reface", False)
                ),
                "box_dims":            getattr(geom, "_box_dims", None),
                "dist_box_value":      getattr(geom, "_dist_box_value", None),
                "rel_dist_box_answer": getattr(geom, "_rel_dist_box_answer", None),
                "rel_dist_box_dists":  getattr(geom, "_rel_dist_box_dists", None),
                "depth_compare_threshold": getattr(geom, "_depth_compare_threshold", None),
                "hide_patch_indices":  list(getattr(geom, "_hide_patch_indices", []) or []),
                "hide_source_patches": bool(getattr(geom, "_hide_source_patches", False)),
                "distractor_centers": [
                    c.detach().cpu().clone()
                    for c in getattr(geom, "_distractor_centers", []) or []
                ],
                "distractor_dims":    list(getattr(geom, "_distractor_dims", []) or []),
                "distractor_rotations": [
                    R.detach().cpu().clone()
                    for R in getattr(geom, "_distractor_rotations", []) or []
                ],
                "distractor_feature_sources": list(
                    int(i) for i in (getattr(geom, "_distractor_feature_sources", []) or [])
                ),
                # Extra task-attached OBB structures (extension seam, e.g.
                # corridor walls placed by a plugin task). Not referenced by
                # the prompt: rendered as canvas structure the model must
                # read visually.
                "extra_obb_centers": [
                    c.detach().cpu().clone()
                    for c in getattr(geom, "_extra_obb_centers", []) or []
                ],
                "extra_obb_dims": list(getattr(geom, "_extra_obb_dims", []) or []),
                "extra_obb_rotations": [
                    R.detach().cpu().clone()
                    for R in getattr(geom, "_extra_obb_rotations", []) or []
                ],
                "extra_obb_feature_sources": list(
                    int(i) for i in (getattr(geom, "_extra_obb_feature_sources", []) or [])
                ),
                # Floor-area slab OBBs (debug-only). Visualizer renders each as
                # a wireframe + surface cloud so the room-size geometry is
                # visible even when curriculum_max_body_patches_per_sample is low.
                "floor_area_obbs": [
                    (c.detach().cpu().clone(),
                     tuple(float(v) for v in dims),
                     R.detach().cpu().clone())
                    for (c, dims, R) in getattr(geom, "_floor_area_obbs", []) or []
                ],
                # rel_dir_camera_* canvas-local viewer pose (intermediate
                # frame). Visualizer draws a forward arrow so the "front"
                # axis is visible to the user. None for other tasks.
                "rel_dir_camera_pos": (
                    getattr(geom, "_rel_dir_camera_pos", None).detach().cpu().clone()
                    if getattr(geom, "_rel_dir_camera_pos", None) is not None else None
                ),
                "rel_dir_camera_fwd": (
                    getattr(geom, "_rel_dir_camera_fwd", None).detach().cpu().clone()
                    if getattr(geom, "_rel_dir_camera_fwd", None) is not None else None
                ),
                "count_side_label":     getattr(geom, "_count_side_label", None),
                "count_side_answer":    getattr(geom, "_count_side_answer", None),
                "count_side_n_targets": getattr(geom, "_count_side_n_targets", None),
                "count_side_target_sides": list(
                    getattr(geom, "_count_side_target_sides", []) or []
                ),
                "counting_diff_level":         getattr(geom, "_counting_diff_level", None),
                "counting_diff_sparse_K":      getattr(geom, "_counting_diff_sparse_K", None),
                "counting_diff_pack_factor":   getattr(geom, "_counting_diff_pack_factor", None),
                "counting_diff_per_obb_dims":  getattr(geom, "_counting_diff_per_obb_dims", None),
                "counting_diff_los_pair_count": getattr(geom, "_counting_diff_los_pair_count", None),
                "counting_diff_n_max_excl":    getattr(geom, "_counting_diff_n_max_excl", None),
                "counting_diff_N_target":      getattr(geom, "_counting_diff_N_target", None),
                "counting_diff_flag_on":       getattr(geom, "_counting_diff_flag_on", None),
                "counting_class_label":        getattr(geom, "_counting_class_label", None),
                # Real-object asset pastes. Slim view of geom._real_asset_pastes
                # (drops the heavy per-patch features tensor) so the visualizer
                # can render each pasted asset at target_center with the correct
                # yaw and look up dense pointclouds via (source_scene, target_id).
                "real_asset_pastes": [
                    {
                        "source_scene":  str(p["asset"]["source_scene"]),
                        "target_id":     int(p["asset"]["target_id"]),
                        "label":         str(p["label"]),
                        "target_center": p["target_center"].detach().cpu().clone(),
                        "yaw_rad":       float(p["yaw_rad"]),
                        "bbox_dims":     p["asset"]["bbox_dims"].detach().cpu().clone(),
                        "t_start":       int(p["t_start"]),
                    }
                    for p in (getattr(geom, "_real_asset_pastes", None) or [])
                ],
                # visibility_camera_fov_real synthetic camera (visualizer draws frustum).
                "vis_cam_real_pos_im": (
                    getattr(geom, "_vis_cam_real_pos_im", None).detach().cpu().clone()
                    if getattr(geom, "_vis_cam_real_pos_im", None) is not None else None
                ),
                "vis_cam_real_fwd_im": (
                    getattr(geom, "_vis_cam_real_fwd_im", None).detach().cpu().clone()
                    if getattr(geom, "_vis_cam_real_fwd_im", None) is not None else None
                ),
                "vis_cam_real_mode": getattr(geom, "_vis_cam_real_mode", None),
                "vis_cam_real_target_yaw_deg": getattr(
                    geom, "_vis_cam_real_target_yaw_deg", None),
                "vis_cam_real_target_pitch_deg": getattr(
                    geom, "_vis_cam_real_target_pitch_deg", None),
                "vis_cam_real_occ_center": (
                    getattr(geom, "_vis_cam_real_occ_center", None
                            ).detach().cpu().clone()
                    if getattr(geom, "_vis_cam_real_occ_center", None)
                    is not None else None
                ),
                "vis_cam_real_occ_dims": getattr(
                    geom, "_vis_cam_real_occ_dims", None),
                "vis_cam_real_occ_R": (
                    getattr(geom, "_vis_cam_real_occ_R", None
                            ).detach().cpu().clone()
                    if getattr(geom, "_vis_cam_real_occ_R", None)
                    is not None else None
                ),
                # appearance_order_camera virtual-camera markers. The 4 cameras
                # carry the entire T-axis appearance-order signal; the visualizer
                # draws each as a small sphere + forward arrow with frame_index
                # and target class printed in the .txt sidecar.
                "app_cam_real_classes": list(
                    getattr(geom, "_app_cam_real_classes", []) or []
                ),
                "app_cam_real_gt_order": list(
                    getattr(geom, "_app_cam_real_gt_order", []) or []
                ),
                "app_cam_real_camera_positions_im": [
                    p.detach().cpu().clone() for p in
                    (getattr(geom, "_app_cam_real_camera_positions_im", []) or [])
                ],
                "app_cam_real_camera_forwards_im": [
                    f.detach().cpu().clone() for f in
                    (getattr(geom, "_app_cam_real_camera_forwards_im", []) or [])
                ],
                "app_cam_real_camera_frame_indices": list(
                    int(t) for t in (getattr(geom, "_app_cam_real_camera_frame_indices", []) or [])
                ),
                "app_cam_real_object_t_shared": getattr(
                    geom, "_app_cam_real_object_t_shared", None
                ),
                # object_class_appearance_order_real named classes + GT order.
                "appearance_order_real_classes": list(
                    getattr(geom, "_appearance_order_real_classes", []) or []
                ),
                "appearance_order_real_gt_order": list(
                    getattr(geom, "_appearance_order_real_gt_order", []) or []
                ),
            },
        }

        # Panoramic augmentation params for PATH B (model.forward projection).
        # The same (center, yaw) was already applied to compute_scene_geometry
        # above so probe patch sampling and the rendered canvas agree.
        if _aug_center is not None:
            data_dict["aug_center_override"] = _aug_center
        if _aug_yaw is not None:
            data_dict["aug_yaw_angle"] = torch.tensor(_aug_yaw, dtype=torch.float32)

        return data_dict

    def _build_sample_stash_fast(
        self,
        geom,
        patch_indices,
        frame_indices,
        inline_patch_t_indices,
        synthetic_patch_t_overrides,
        input_ids_360,
        labels_360,
        attention_mask_360,
        inline_patch_positions,
        first_idx,
        last_idx,
        input_seq_len,
        question_text,
        answer_text,
        task_label,
        scene_id,
        task,
        rng,
        idx,
        marker_tokens=None,
    ):
        """PATH A emitter for marker-stash samples.

        Replicates the forward-side canvas_obb_only + synthetic-OBB-append +
        prepare_batch pipeline entirely in the dataloader worker, so that
        model.forward consumes a fully projected batch (projection_done=True)
        and skips PRE-PATH-A entirely. No ViT, no reproject_scene, no CPU
        Python loop over 32 frames on the main GPU process — all that CPU
        work gets parallelized across the dataloader workers instead.

        Mirrors the structure of model_adapters/qwen3_vl/model.py:781-1007:
        build keep-set → slice scene → synthetic OBB append → prepare_batch.
        """
        from reprojection.types import ReprojectedScene

        # --- 1. Per-marker direct T values (stash has no canvas T to read).
        if (inline_patch_t_indices is not None
                and len(inline_patch_t_indices) == len(patch_indices)):
            inline_patch_t_values = geom.frame_index[
                inline_patch_t_indices
            ].long().tolist()
        else:
            _n_frames_virt = int(geom.n_images)
            inline_patch_t_values = [
                rng.randrange(_n_frames_virt) for _ in range(len(patch_indices))
            ]

        # --- 2. Keep-set: every canvas index the stripped/appended scene
        #        will reference. Mirrors model.py:781-801.
        _keep_seq = [int(i) for i in patch_indices]
        _mbfs = getattr(geom, "_multi_box_feature_sources", None)
        if _mbfs:
            _keep_seq.extend(int(i) for i in _mbfs)
        _sfs_raw = getattr(geom, "_synthetic_feature_source", None)
        _sfs_val = None
        if _sfs_raw is not None:
            _sfs_val = int(_sfs_raw.item() if hasattr(_sfs_raw, "item") else _sfs_raw)
            if _sfs_val >= 0:
                _keep_seq.append(_sfs_val)
        _pps = getattr(geom, "_synthetic_per_point_feature_sources", None)
        if _pps is not None and len(_pps) > 0:
            _keep_seq.extend(int(i) for i in _pps)

        _hide_raw = getattr(geom, "_hide_patch_indices", None)
        _hide_all = bool(getattr(geom, "_hide_source_patches", False))
        if _hide_raw is not None and len(_hide_raw) > 0:
            _hide_list = [int(i) for i in _hide_raw]
        elif _hide_all:
            _hide_list = [int(i) for i in patch_indices]
        else:
            _hide_list = []
        if _hide_list:
            _keep_seq.extend(_hide_list)

        # Distractor feature sources: each distractor box clones its own
        # unique stash draw from this canvas position, so the keep_set must
        # include them.
        _dist_fs = getattr(geom, "_distractor_feature_sources", None) or []
        if _dist_fs:
            _keep_seq.extend(int(i) for i in _dist_fs)

        _wall_fs = getattr(geom, "_extra_obb_feature_sources", None) or []
        if _wall_fs and not getattr(geom, "_wall_sources_inline", False):
            _keep_seq.extend(int(i) for i in _wall_fs)

        if getattr(geom, "_keep_insertion_order", False):
            # Append-stable keep order (opt-in per task, set on the geom):
            # the kept source patches lead the compact canvas in FIRST-
            # REFERENCE order instead of sorted canvas order, so a source
            # added on a later episode step appends rows instead of
            # re-sorting rows already committed to a KV cache.
            keep = list(dict.fromkeys(_keep_seq))
        else:
            keep = sorted(set(_keep_seq))
        remap = {old: new for new, old in enumerate(keep)}
        keep_t = torch.tensor(keep, dtype=torch.long)

        # --- 3. Harvest stash features at each kept position.
        _N_stash = int(self._obb_feature_stash.shape[0])
        if getattr(geom, "_stash_pick_by_source", False):
            # Append-stable draw (opt-in per task, set on the geom by the
            # task sampler): key each kept position's stash row on the
            # canvas position itself, not on its rank in sorted(keep) and
            # not on the shared rng stream. A feature source added on a
            # later episode step then cannot re-deal features already
            # committed to a KV cache, which the agentic multi-turn loop
            # requires. Consumes NO draws from `rng`, so the draw stream
            # after this point is also independent of len(keep).
            stash_pick = torch.tensor(
                [random.Random(self._base_seed * 1_000_003 + int(k))
                 .randrange(_N_stash) for k in keep],
                dtype=torch.long,
            )
        else:
            stash_pick = torch.tensor(
                [rng.randrange(_N_stash) for _ in range(len(keep))],
                dtype=torch.long,
            )
        picked = self._obb_feature_stash[stash_pick]  # [K, N_layers, D]
        embeds = picked[:, 0, :].float()
        aux_layers = [picked[:, l, :].float() for l in range(1, picked.shape[1])]

        # --- 4. Compact ReprojectedScene — geometry from geom at keep,
        #        features from the stash. Marker-only extras carry through
        #        from geom so the cam-marker block in prepare_batch emits
        #        N_canvas + N_extra markers (e.g. 32 + 2 endpoint extras for
        #        ONECANVAS_MARKER_EXTRA_ENDPOINTS=1) instead of dropping the
        #        extras on this fast path.
        scene = ReprojectedScene(
            embeds=embeds,
            aux_layers=aux_layers,
            longitude=geom.longitude[keep_t].clone(),
            latitude=geom.latitude[keep_t].clone(),
            depth=geom.depth[keep_t].clone(),
            frame_index=geom.frame_index[keep_t].clone(),
            n_valid=len(keep),
            n_images=geom.n_images,
            center_point=geom.center_point,
            poses=geom.poses,
            intrinsics=geom.intrinsics,
            image_dims=geom.image_dims,
            yaw_angle=geom.yaw_angle,
        )

        # --- 5. Synthetic OBB append. Mirrors model.py:818-992.
        _multi_box_sph = getattr(geom, "_multi_box_spherical", None)
        if _multi_box_sph is not None:
            _synth_sph = torch.cat(_multi_box_sph, dim=0)
            _box_sizes = [int(t.shape[0]) for t in _multi_box_sph]
        else:
            _synth_sph = getattr(geom, "_synthetic_spherical", None)
            _box_sizes = None

        if _synth_sph is not None and _synth_sph.shape[0] > 0:
            if _mbfs and _box_sizes is not None and len(_mbfs) == len(_box_sizes):
                # Multi-box path: each box's body points take that box's
                # feature ref (cloned n_pts times).
                offset = 0
                all_lat, all_lon, all_dep = [], [], []
                all_emb = []
                all_ds_rows = [[] for _ in scene.aux_layers]
                for src_raw, n_pts in zip(_mbfs, _box_sizes):
                    src_idx = remap[int(src_raw)]
                    pts_k = _synth_sph[offset:offset + n_pts]
                    all_lat.append(pts_k[:, 0])
                    all_lon.append(pts_k[:, 1])
                    all_dep.append(pts_k[:, 2])
                    all_emb.append(
                        scene.embeds[src_idx].unsqueeze(0).expand(n_pts, -1).clone()
                    )
                    for li, layer in enumerate(scene.aux_layers):
                        all_ds_rows[li].append(
                            layer[src_idx].unsqueeze(0).expand(n_pts, -1).clone()
                        )
                    offset += n_pts
                n_new = offset
                if (synthetic_patch_t_overrides is not None
                        and int(synthetic_patch_t_overrides.numel()) == n_new):
                    _new_t = synthetic_patch_t_overrides.to(
                        dtype=scene.frame_index.dtype
                    )
                else:
                    _new_t = torch.zeros(n_new, dtype=scene.frame_index.dtype)
                scene.latitude = torch.cat([scene.latitude] + all_lat)
                scene.longitude = torch.cat([scene.longitude] + all_lon)
                scene.depth = torch.cat([scene.depth] + all_dep)
                scene.frame_index = torch.cat([scene.frame_index, _new_t])
                scene.embeds = torch.cat([scene.embeds] + all_emb, dim=0)
                scene.aux_layers = [
                    torch.cat([layer] + all_ds_rows[li], dim=0)
                    for li, layer in enumerate(scene.aux_layers)
                ]
                scene.n_valid += n_new
            else:
                # Legacy single-source path (box_size / object_counting /
                # box_floor_area_*). Per-point sources clone from the
                # compact-scene feature at each per-point's remapped idx;
                # single-source broadcasts one ref_idx feature n_synth times.
                n_synth = int(_synth_sph.shape[0])
                if _pps is not None and len(_pps) == n_synth:
                    pp_remap = torch.tensor(
                        [remap[int(i)] for i in _pps], dtype=torch.long,
                    )
                    synth_emb = scene.embeds[pp_remap].clone()
                    synth_ds = [layer[pp_remap].clone() for layer in scene.aux_layers]
                else:
                    if _sfs_val is None or _sfs_val < 0:
                        raise RuntimeError(
                            f"stash-fast: single-source synth for task={task!r} "
                            "needs _synthetic_feature_source (>=0) but got none"
                        )
                    ref_idx = remap[_sfs_val]
                    synth_emb = (
                        scene.embeds[ref_idx].unsqueeze(0).expand(n_synth, -1).clone()
                    )
                    synth_ds = [
                        layer[ref_idx].unsqueeze(0).expand(n_synth, -1).clone()
                        for layer in scene.aux_layers
                    ]
                scene.latitude = torch.cat([scene.latitude, _synth_sph[:, 0]])
                scene.longitude = torch.cat([scene.longitude, _synth_sph[:, 1]])
                scene.depth = torch.cat([scene.depth, _synth_sph[:, 2]])
                scene.frame_index = torch.cat([
                    scene.frame_index,
                    torch.zeros(n_synth, dtype=scene.frame_index.dtype),
                ])
                scene.embeds = torch.cat([scene.embeds, synth_emb], dim=0)
                scene.aux_layers = [
                    torch.cat([layer, synth_ds[li]], dim=0)
                    for li, layer in enumerate(scene.aux_layers)
                ]
                scene.n_valid += n_synth

        # --- 5b. Distractor OBB append. Each distractor clones its own unique
        #         stash feature (from _distractor_feature_sources[i], remapped
        #         into the compact scene) across its body points. Per-distractor
        #         T is a single random frame index, broadcast to that box's body,
        #         so the M distractor Ts spread across the frame range without
        #         a shared-T shortcut.
        _dist_sph = getattr(geom, "_distractor_spherical", None) or []
        _dist_fs_list = getattr(geom, "_distractor_feature_sources", None) or []
        if _dist_sph and _dist_fs_list and len(_dist_sph) == len(_dist_fs_list):
            _n_frames_virt = int(geom.n_images)
            d_all_lat, d_all_lon, d_all_dep = [], [], []
            d_all_t = []
            d_all_emb = []
            d_all_ds_rows = [[] for _ in scene.aux_layers]
            for pts_k, src_raw in zip(_dist_sph, _dist_fs_list):
                n_pts = int(pts_k.shape[0])
                if n_pts <= 0:
                    continue
                src_idx = remap[int(src_raw)]
                d_all_lat.append(pts_k[:, 0])
                d_all_lon.append(pts_k[:, 1])
                d_all_dep.append(pts_k[:, 2])
                t_k = int(rng.randrange(_n_frames_virt)) if _n_frames_virt > 0 else 0
                d_all_t.append(torch.full(
                    (n_pts,), t_k, dtype=scene.frame_index.dtype,
                ))
                d_all_emb.append(
                    scene.embeds[src_idx].unsqueeze(0).expand(n_pts, -1).clone()
                )
                for li, layer in enumerate(scene.aux_layers):
                    d_all_ds_rows[li].append(
                        layer[src_idx].unsqueeze(0).expand(n_pts, -1).clone()
                    )
            if d_all_lat:
                scene.latitude = torch.cat([scene.latitude] + d_all_lat)
                scene.longitude = torch.cat([scene.longitude] + d_all_lon)
                scene.depth = torch.cat([scene.depth] + d_all_dep)
                scene.frame_index = torch.cat([scene.frame_index] + d_all_t)
                scene.embeds = torch.cat([scene.embeds] + d_all_emb, dim=0)
                scene.aux_layers = [
                    torch.cat([layer] + d_all_ds_rows[li], dim=0)
                    for li, layer in enumerate(scene.aux_layers)
                ]
                scene.n_valid += sum(int(t.shape[0]) for t in d_all_lat)

        # --- 5c. Extra task-attached OBB structures (extension seam, e.g.
        #         corridor walls placed by a plugin task). Same append
        #         pattern as the distractor block above: each OBB's body
        #         points clone their feature source's embeds/aux_layers and
        #         pick a random frame index per OBB. These structures are
        #         unreferenced by the prompt; they exist purely as canvas
        #         structure the model must read visually.
        _wall_sph = getattr(geom, "_extra_obb_spherical", None) or []
        _wall_fs_list = getattr(geom, "_extra_obb_feature_sources", None) or []
        if _wall_sph and _wall_fs_list and len(_wall_sph) == len(_wall_fs_list):
            _n_frames_virt = int(geom.n_images)
            # Opt-in per-OBB T override (set on the geom by the task
            # sampler, aligned with the PLACED list): lets a task put
            # the OBB's T coordinate on a semantic axis (e.g. the episode
            # step it was observed on) instead of a random frame.
            # Also consumes no rng draws, keeping the stream append-stable
            # for downstream multi-turn loops.
            _wall_t_over = getattr(geom, "_extra_obb_t_indices", None)
            if _wall_t_over is not None and len(_wall_t_over) != len(_wall_sph):
                raise ValueError(
                    f"_extra_obb_t_indices has {len(_wall_t_over)} entries "
                    f"for {len(_wall_sph)} placed extra OBBs")
            # Opt-in inline wall sources: emit each wall's feature-source
            # row IMMEDIATELY BEFORE that wall's body rows instead of in
            # the up-front keep block. With the block layout, a later
            # episode step's new sources insert between the keep block and
            # the wall block, breaking the row prefix a KV-cache-reusing
            # loop depends on; inline, every append is terminal. Requires
            # the per-source stash draw so the source row's content does
            # not depend on the shared rng stream.
            _wall_inline = bool(getattr(geom, "_wall_sources_inline", False))
            if _wall_inline and not getattr(geom, "_stash_pick_by_source", False):
                raise ValueError(
                    "_wall_sources_inline requires _stash_pick_by_source")
            _inline_emitted = {}
            w_all_lat, w_all_lon, w_all_dep = [], [], []
            w_all_t = []
            w_all_emb = []
            w_all_ds_rows = [[] for _ in scene.aux_layers]
            for wall_i, (pts_k, src_raw) in enumerate(zip(_wall_sph, _wall_fs_list)):
                n_pts = int(pts_k.shape[0])
                if n_pts <= 0:
                    continue
                if _wall_inline:
                    src = int(src_raw)
                    if src not in _inline_emitted:
                        _pick = random.Random(
                            self._base_seed * 1_000_003 + src).randrange(_N_stash)
                        _prow = self._obb_feature_stash[_pick]
                        _inline_emitted[src] = (
                            _prow[0].float(),
                            [_prow[l].float() for l in range(1, _prow.shape[0])],
                        )
                        w_all_lat.append(geom.latitude[src].reshape(1))
                        w_all_lon.append(geom.longitude[src].reshape(1))
                        w_all_dep.append(geom.depth[src].reshape(1))
                        w_all_t.append(
                            geom.frame_index[src].reshape(1)
                            .to(scene.frame_index.dtype))
                        w_all_emb.append(_inline_emitted[src][0].unsqueeze(0))
                        for li in range(len(scene.aux_layers)):
                            w_all_ds_rows[li].append(
                                _inline_emitted[src][1][li].unsqueeze(0))
                    src_emb, src_aux = _inline_emitted[src]
                else:
                    src_idx = remap[int(src_raw)]
                    src_emb = scene.embeds[src_idx]
                    src_aux = [layer[src_idx] for layer in scene.aux_layers]
                w_all_lat.append(pts_k[:, 0])
                w_all_lon.append(pts_k[:, 1])
                w_all_dep.append(pts_k[:, 2])
                if _wall_t_over is not None:
                    t_k = min(max(int(_wall_t_over[wall_i]), 0),
                              max(_n_frames_virt - 1, 0))
                else:
                    t_k = int(rng.randrange(_n_frames_virt)) if _n_frames_virt > 0 else 0
                w_all_t.append(torch.full(
                    (n_pts,), t_k, dtype=scene.frame_index.dtype,
                ))
                w_all_emb.append(
                    src_emb.unsqueeze(0).expand(n_pts, -1).clone()
                )
                for li in range(len(scene.aux_layers)):
                    w_all_ds_rows[li].append(
                        src_aux[li].unsqueeze(0).expand(n_pts, -1).clone()
                    )
            if w_all_lat:
                scene.latitude = torch.cat([scene.latitude] + w_all_lat)
                scene.longitude = torch.cat([scene.longitude] + w_all_lon)
                scene.depth = torch.cat([scene.depth] + w_all_dep)
                scene.frame_index = torch.cat([scene.frame_index] + w_all_t)
                scene.embeds = torch.cat([scene.embeds] + w_all_emb, dim=0)
                scene.aux_layers = [
                    torch.cat([layer] + w_all_ds_rows[li], dim=0)
                    for li, layer in enumerate(scene.aux_layers)
                ]
                scene.n_valid += sum(int(t.shape[0]) for t in w_all_lat)

        # --- 5b. Real-object asset paste. Each paste is a harvested asset
        #         (per-patch features + xyz_offsets + frame_indices) that
        #         the *_real samplers attached to geom._real_asset_pastes.
        #         Appended AFTER strip + synthetic OBBs, so the asset
        #         patches survive the canvas-strip automatically.
        _real_pastes = getattr(geom, "_real_asset_pastes", None) or []
        if _real_pastes:
            # TOTAL per-sample cap (markers + obb_body + walls + real). The
            # non-real components are already in the scene at this point
            # (see "appended AFTER strip + synthetic OBBs" comment above), so
            # subtract them from _max_per_sample to get the budget left for
            # real-asset patches. Then distribute that budget across pastes
            # by uniform-random pruning of the pooled candidate patches:
            # build a flat (paste_idx, _) sequence of all available patches,
            # shuffle, take the first `budget_real`, count per paste. This
            # is equivalent to "append everything, then drop random until
            # total = cap" and gives small pastes their full count instead
            # of capping them to a per-paste share they don't need.
            _max_per_sample = int(self._real_asset_max_per_sample)
            _nonreal = int(scene.n_valid)
            if _max_per_sample > 0:
                _budget_real = max(0, _max_per_sample - _nonreal)
            else:
                _budget_real = -1  # sentinel: disabled (no cap)
            _n_avail = [int(p["asset"]["features"].shape[0])
                        for p in _real_pastes]
            _total_avail = sum(_n_avail)
            if _budget_real < 0 or _budget_real >= _total_avail:
                # Uncapped: -1 sentinel skips the per-paste subsample.
                _per_paste_cap = [-1] * len(_real_pastes)
            elif _budget_real == 0:
                _per_paste_cap = [0] * len(_real_pastes)
            else:
                _pool = [pi for pi, n in enumerate(_n_avail) for _ in range(n)]
                rng.shuffle(_pool)
                _kept = _pool[:_budget_real]
                _per_paste_cap = [0] * len(_real_pastes)
                for pi in _kept:
                    _per_paste_cap[pi] += 1
            # Optional per-sample shared subsample fraction. Drawn ONCE
            # per sample so all pastes share f, breaking the 'total volume /
            # per-instance density = count' shortcut without averaging out
            # under the law of large numbers (per-instance jitter would).
            if self._real_asset_global_subsample:
                _f_min = self._real_asset_subsample_min
                _f = math.exp(rng.uniform(math.log(_f_min), 0.0))
                _scaled_cap = []
                for _cap_i, _n_avail_i in zip(_per_paste_cap, _n_avail):
                    if _cap_i == 0:
                        _scaled_cap.append(0)
                    elif _cap_i < 0:
                        # Uncapped path: convert to a finite fraction-of-avail cap.
                        _scaled_cap.append(max(1, int(round(_n_avail_i * _f))))
                    else:
                        _scaled_cap.append(max(1, int(round(_cap_i * _f))))
                _per_paste_cap = _scaled_cap

            geom._real_asset_paste_ranges = []
            for _paste, _cap in zip(_real_pastes, _per_paste_cap):
                _rng_pair = self._append_real_object_assets(
                    scene, _paste, rng, cap_per_paste=_cap,
                )
                geom._real_asset_paste_ranges.append(_rng_pair)

        # --- 6. Remap marker patch_indices + hide indices into compact scene.
        _imp_patch_remap = [remap[int(i)] for i in patch_indices]
        _hide_remap = [remap[int(i)] for i in _hide_list if int(i) in remap]

        # --- 7. prepare_batch — identical call signature to
        #        data_processor_3d.py:2234 PATH A and model.py:994 forward.
        proj = self.adapter.prepare_batch(
            scene=scene,
            input_ids=input_ids_360,
            attention_mask=attention_mask_360,
            labels=labels_360,
            first_idx=first_idx,
            last_idx=last_idx,
            config=self._scene_src._reprojection_config(),
            inline_patch_positions=inline_patch_positions,
            inline_patch_indices=_imp_patch_remap,
            inline_patch_t_values=inline_patch_t_values,
            hide_source_patches=_hide_remap,
            # toolcall_marker track: text-marker + canvas-twin tokens with
            # explicit features. prepare_batch owns the paste (adapter.py's
            # marker block), so the fast path only has to hand it through --
            # the same dict model.forward builds on the slow PATH B. Without
            # this the stash path could not serve marker episodes at all and
            # they had to run with the stash disabled, paying a full 32-frame
            # ViT + reprojection per sample for a ~100-patch synthetic canvas.
            marker_tokens=marker_tokens,
        )

        data_dict = {
            "projection_done": True,
            "projected_input_ids":        proj["input_ids"],
            "projected_position_ids":     proj["position_ids"],
            "projected_attention_mask":   proj["attention_mask"],
            "projected_labels":           proj["labels"],
            "projected_embeds":           proj["embeds"],
            "projected_aux_layers":        proj["aux_layers"],
            "projected_depth_bins":       proj["depth_bins"],
            "projected_ray_dirs":         proj.get("ray_dirs"),
            "projected_inline_patch_indices_local": proj.get("inline_patch_indices_local"),
            "rope_deltas":                proj["rope_deltas"],
            # Metadata — ScanQALazyDataset's PATH A keeps these for eval + logs.
            "answer":        answer_text,
            "question":      question_text,
            "question_type": task_label,
            "scene_id":      scene_id,
            "images":        [],
            "_dataloader_retries": 0,
            "_repro": {
                "idx": int(idx),
                "split": self.data_split,
                "base_seed": self._base_seed,
                "seed_offset": self._seed_offset,
                "task": task,
                "scene_id": scene_id,
                "patch_indices": [int(x) for x in patch_indices],
                "frame_indices": [int(x) for x in (frame_indices or [])],
                "num_images": int(self._scene_src.num_images),
                "stash": True,
            },
        }
        return data_dict

    def _append_real_object_assets(self, scene, paste, rng, cap_per_paste=-1):
        """Append one harvested asset's K patches to the compact scene.

        Differs from the synthetic-OBB append in two ways:
          1. The asset carries its own per-patch features, so we do NOT
             clone-from-canvas. We append asset["features"][valid] directly
             to scene.embeds and scene.aux_layers.
          2. T values use the asset's preserved frame-spread shifted by
             paste["t_start"] (compressed proportionally if it overflows
             scene.n_images), so the model sees the object spread over
             multiple frames like a natural object. Collapsing to a single
             T would defeat the experimental signal.

        paste: {"asset": dict, "target_center": [3] tensor, "yaw_rad": float,
                "t_start": int, "label": str}

        cap_per_paste semantics:
          -1  -> uncapped (keep every projected-valid patch)
           0  -> drop entirely (skip this paste)
          >0  -> randomly subsample down to that count

        Returns (start_idx, end_idx) into the post-append scene.embeds index
        space, or None if no patches projected validly OR cap_per_paste == 0.
        """
        cap_per_paste = int(cap_per_paste)
        if cap_per_paste == 0:
            return None  # explicit "drop this paste" — caller used pool-prune
        asset = paste["asset"]
        target_center = paste["target_center"].float()
        yaw = float(paste["yaw_rad"])
        t_start = int(paste["t_start"])

        # 1. Place: rotate OBB-local offsets by a random yaw around the
        #    object's vertical axis, then translate to target. xyz_offsets
        #    are stored in OBB-local-from-Z-up-world (see
        #    extract_real_object_assets.py), so for upright OBBs (the vast
        #    majority) local axis 2 is vertical. _real_asset_paste_matrix
        #    folds in the Z-up → intermediate-frame axis swap so the result
        #    is ready for _world_to_spherical / target_center addition.
        M = _real_asset_paste_matrix(yaw)
        xyz_local = asset["xyz_offsets"].float()                  # [K, 3]
        world_pts = xyz_local @ M.T + target_center.unsqueeze(0)

        # 2. Project to canvas spherical coords.
        lat, lon, depth, valid = _world_to_spherical(world_pts)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return None

        # 2b. Per-paste patch cap (computed from the per-sample budget at the
        #     call site, so multi-paste tasks split the budget evenly). Big
        #     classes (bed/couch/desk: 130-250 patches mean) would otherwise
        #     dominate sequence length and tax the whole batch via padding.
        #     Class identity is recoverable from ~30 patches via shape-feature
        #     signal (median asset has ~30 patches anyway).
        keep_idx = valid.nonzero(as_tuple=True)[0]
        if cap_per_paste > 0 and n_valid > cap_per_paste:
            # Use the per-sample rng so the sampling is reproducible across
            # workers given a fixed seed.
            perm = torch.randperm(
                n_valid,
                generator=torch.Generator().manual_seed(rng.randrange(2**31)),
            )
            keep_idx = keep_idx[perm[:cap_per_paste]]
            n_valid = cap_per_paste

        # 3. T remap.
        #    Default: preserve the asset's per-patch frame-spread so the
        #    object "appears over" multiple frames; compress proportionally
        #    if the spread doesn't fit. Optional t_spread_cap further bounds
        #    the effective spread.
        #    Override (t_force_uniform_to_n_imgs=True, used by the appearance-
        #    order task): drop the asset's natural T entirely; each kept
        #    patch's T is drawn uniformly from [t_start, n_imgs-1], with at
        #    least one patch forced to t_start so min(T) is exactly the GT.
        #    This mirrors synthetic appearance_order_box and gives heavy
        #    T-overlap between concurrent objects so only first-appearance
        #    differs.
        tgt_n = int(scene.n_images)
        if paste.get("t_force_uniform_to_n_imgs", False):
            n_keep = int(keep_idx.numel())
            hi = tgt_n - 1
            if hi <= t_start or n_keep == 0:
                t_target_keep = torch.full((n_keep,), t_start, dtype=torch.long)
            else:
                seed = rng.randrange(2**31)
                gen = torch.Generator().manual_seed(seed)
                t_target_keep = torch.randint(
                    t_start, hi + 1, (n_keep,), generator=gen, dtype=torch.long
                )
                t_target_keep[0] = t_start  # guarantee min(T) = t_start
            # Skip the index_by-keep_idx step below by writing scene.frame_index
            # directly with t_target_keep.
            t_target = None  # signal to the append step that we already have keep-indexed T
        else:
            src_t = asset["frame_indices"].long()
            t_norm = src_t - src_t.min()
            spread = int(t_norm.max().item())
            max_room = max(0, tgt_n - 1 - t_start)
            budget = max_room
            cap = paste.get("t_spread_cap")
            if cap is not None:
                budget = min(budget, max(0, int(cap)))
            if spread <= budget:
                t_target = t_start + t_norm
            else:
                t_target = t_start + (
                    t_norm.float() * budget / max(spread, 1)
                ).round().long()
            t_target = t_target.clamp(0, tgt_n - 1)
            t_target_keep = t_target[keep_idx]

        # 4. Append per-patch features. asset["features"] is [K, N_layers, C] bf16.
        feats = asset["features"][keep_idx].float()               # [K_v, N_layers, C]
        start_idx = int(scene.n_valid)
        scene.embeds = torch.cat([scene.embeds, feats[:, 0]], dim=0)
        new_aux_layers = []
        for li, layer in enumerate(scene.aux_layers):
            new_aux_layers.append(torch.cat([layer, feats[:, li + 1]], dim=0))
        scene.aux_layers = new_aux_layers
        scene.latitude = torch.cat([scene.latitude, lat[keep_idx]], dim=0)
        scene.longitude = torch.cat([scene.longitude, lon[keep_idx]], dim=0)
        scene.depth = torch.cat([scene.depth, depth[keep_idx]], dim=0)
        scene.frame_index = torch.cat(
            [scene.frame_index, t_target_keep.to(scene.frame_index.dtype)],
            dim=0,
        )
        scene.n_valid = int(scene.n_valid + n_valid)

        end_idx = int(scene.n_valid)
        return (start_idx, end_idx)

    def _real_paste_rows(self, geom, rng, n_nonreal):
        """Real-asset paste rows for the SLOW (PATH B) path, precomputed.

        WHY THIS EXISTS (2026-08-16). ``_append_real_object_assets`` had
        exactly one call site, inside ``_build_sample_stash_fast``, so the
        real-object canvas content existed ONLY on the stash fast path:
        model.forward's PATH B strips the canvas and appends synthetic OBBs
        but knew nothing about real pastes. Any real_* sample routed through
        the slow path -- which is what a rollout harness must do, since the
        projected fast-path sample carries the supervised transcript and
        would leak the answer -- put an EMPTY canvas in front of the model.
        The toolmark_families_real eval measured exactly that
        (famacc_real_ckpt40000_blindcanvas in agentic-onecanvas): pred-GT
        placement correlation ~0, room-scale prior boxes.

        The rows are computed HERE, not in the forward, so the placement
        math and the budget arithmetic exist ONCE: this method drives the
        SAME ``_append_real_object_assets`` the fast path calls, on a row
        accumulator, and mirrors ``_build_sample_stash_fast`` step 5b's
        budget block. PATH B then only concatenates
        (``real_paste_embeds`` / ``real_paste_spherical`` /
        ``real_paste_frame_index`` in model.py).

        ``n_nonreal`` is the count of non-real canvas rows the forward will
        keep (keep-set plus synthetic OBB rows), so the patch budget splits
        the same way it does on the fast path. For the real_* box families
        every non-real channel is empty and it is 0.

        Deliberately does NOT set ``geom._real_asset_paste_ranges``: the
        fast path's ranges index the post-append scene, and the final row
        offsets on PATH B are only known inside the forward. A consumer
        that needs them on this path should derive them from the per-paste
        row counts, not read a half-meaningful attribute.

        Returns None (no pastes, or none survived projection/budget), or a
        dict of tensors ready to concatenate:
        ``embeds`` [K, N_layers, C] float, ``spherical`` [K, 3] float
        (lat, lon, depth), ``frame_index`` [K] long.
        """
        _real_pastes = getattr(geom, "_real_asset_pastes", None) or []
        if not _real_pastes:
            return None
        from types import SimpleNamespace
        feats0 = _real_pastes[0]["asset"]["features"]
        n_layers, C = int(feats0.shape[1]), int(feats0.shape[2])
        shim = SimpleNamespace(
            embeds=torch.zeros(0, C),
            aux_layers=[torch.zeros(0, C) for _ in range(n_layers - 1)],
            latitude=torch.zeros(0),
            longitude=torch.zeros(0),
            depth=torch.zeros(0),
            frame_index=torch.zeros(0, dtype=torch.long),
            n_valid=0,
            n_images=int(geom.n_images),
        )

        # Budget block, mirroring _build_sample_stash_fast step 5b (see the
        # comments there for the pool-prune rationale).
        _max_per_sample = int(self._real_asset_max_per_sample)
        if _max_per_sample > 0:
            _budget_real = max(0, _max_per_sample - int(n_nonreal))
        else:
            _budget_real = -1  # sentinel: disabled (no cap)
        _n_avail = [int(p["asset"]["features"].shape[0]) for p in _real_pastes]
        _total_avail = sum(_n_avail)
        if _budget_real < 0 or _budget_real >= _total_avail:
            _per_paste_cap = [-1] * len(_real_pastes)
        elif _budget_real == 0:
            _per_paste_cap = [0] * len(_real_pastes)
        else:
            _pool = [pi for pi, n in enumerate(_n_avail) for _ in range(n)]
            rng.shuffle(_pool)
            _kept = _pool[:_budget_real]
            _per_paste_cap = [0] * len(_real_pastes)
            for pi in _kept:
                _per_paste_cap[pi] += 1
        if self._real_asset_global_subsample:
            _f_min = self._real_asset_subsample_min
            _f = math.exp(rng.uniform(math.log(_f_min), 0.0))
            _scaled_cap = []
            for _cap_i, _n_avail_i in zip(_per_paste_cap, _n_avail):
                if _cap_i == 0:
                    _scaled_cap.append(0)
                elif _cap_i < 0:
                    _scaled_cap.append(max(1, int(round(_n_avail_i * _f))))
                else:
                    _scaled_cap.append(max(1, int(round(_cap_i * _f))))
            _per_paste_cap = _scaled_cap

        for _paste, _cap in zip(_real_pastes, _per_paste_cap):
            self._append_real_object_assets(shim, _paste, rng, cap_per_paste=_cap)
        if shim.n_valid == 0:
            return None
        return {
            "embeds": torch.stack([shim.embeds] + shim.aux_layers, dim=1),
            "spherical": torch.stack(
                [shim.latitude, shim.longitude, shim.depth], dim=1),
            "frame_index": shim.frame_index.long(),
        }

    def _draw_real_asset_inflation_frac(self, rng, force_tight: bool = False) -> float:
        """Per-sample paste-time OBB inflation draw.

        Returns 0.0 (tight) when ``force_tight`` is set or when the max is
        non-positive; otherwise draws uniformly from [0, max]. Each sampler
        calls this once at the top of the sample build and threads the
        value to every ``bank.sample(...)`` call within the sample so all
        pastes in one sample share the same visual scale (target +
        distractors look like they came from a coherent scene). The legacy
        per-object bank ignores ``inflation_frac``; the whole-scene bank
        uses it to widen the OBB inclusion mask.

        ``force_tight=True`` for samplers where the pasted region is the
        supervision target (object_class_grounding_real,
        object_class_appearance_order_real).
        """
        if force_tight:
            return 0.0
        max_frac = float(self._real_asset_inflation_max_frac)
        if max_frac <= 0.0:
            return 0.0
        return rng.uniform(0.0, max_frac)

    def _sample_camera_pose_aug(self, assets, idx):
        """Sample (aug_center, aug_yaw, cam_frame_idx) by reorienting the canvas
        onto one of the scene's real camera poses.

        Used by rel_dir_camera_* probe tasks to train the model on the same
        camera-egocentric canvas that SPBench-SI's eval uses when
        spbench_use_camera_pose is on. Formula matches
        data_processor_3d.py's spbench block exactly, so training + eval
        canvases are in the same distribution.

        Returns (None, None, None) if no valid poses are available.
        """
        poses_list = assets.get("poses", [])
        valid = [
            (i, p) for i, p in enumerate(poses_list)
            if p is not None and torch.is_tensor(p) and torch.all(torch.isfinite(p))
        ]
        if not valid:
            return None, None, None
        aug_rng = random.Random(self._aug_seed + int(idx))
        cam_idx, P = aug_rng.choice(valid)
        P = P.float()
        aug_center = P[:3, 3].clone()
        fwd_world = P[:3, :3] @ torch.tensor([0.0, 0.0, 1.0])
        aug_yaw = math.atan2(float(fwd_world[0]), float(fwd_world[1]))
        aug_yaw = ((aug_yaw + math.pi) % (2 * math.pi)) - math.pi
        return aug_center, aug_yaw, cam_idx

    def _sample_panoramic_aug(self, assets, idx):
        """Sample (aug_center, aug_yaw) for panoramic canvas augmentation.

        Mirrors data_processor_3d.py's panoramic_augment_* sampling policy so
        probe training matches the canvas distribution of downstream QA /
        grounding runs (which warm-start from the probe checkpoint). Returns
        (None, None) for the eval split (augmentation is training-only) or when
        all flags are off / assets insufficient.
        """
        # Legacy mode augments the eval split too (the internal code had no
        # split guard, so the shipped runs' gen evals ran on augmented
        # canvases; their aug rng was decorrelated there by the 999 eval
        # seed offset on the item rng).
        if self.data_split != "train" and not self._legacy_aug_seed:
            return None, None
        if not (self._panoramic_augment_center
                or self._panoramic_augment_center_sigma > 0
                or self._panoramic_augment_center_uniform
                or self._panoramic_augment_center_uniform_scene
                or self._panoramic_augment_yaw):
            return None, None

        aug_rng = random.Random(self._aug_seed + int(idx))
        aug_center = None
        aug_yaw = None

        if self._panoramic_augment_center_uniform_scene:
            poses_list = assets.get("poses", [])
            depths_list = assets.get("depths", [])
            intr_list = assets.get("intrinsics", [])
            dims_list = assets.get("image_dims", [])
            if depths_list and poses_list and intr_list and dims_list:
                lo, hi = compute_scene_aabb_from_depths(
                    depths_list, poses_list, intr_list, dims_list, grid=16,
                )
                if lo is not None:
                    if self._panoramic_augment_center_inflate != 1.0:
                        mid = (lo + hi) * 0.5
                        half = (hi - lo) * 0.5 * self._panoramic_augment_center_inflate
                        lo, hi = mid - half, mid + half
                    aug_center = torch.tensor(
                        [aug_rng.uniform(lo[0].item(), hi[0].item()),
                         aug_rng.uniform(lo[1].item(), hi[1].item()),
                         aug_rng.uniform(lo[2].item(), hi[2].item())],
                        dtype=torch.float32,
                    )
        elif self._panoramic_augment_center_uniform:
            poses_list = assets.get("poses", [])
            valid = [p for p in poses_list
                     if p is not None and torch.is_tensor(p) and torch.all(torch.isfinite(p))]
            if valid:
                translations = torch.stack([p[:3, 3] for p in valid])
                lo = translations.min(dim=0).values
                hi = translations.max(dim=0).values
                if self._panoramic_augment_center_inflate != 1.0:
                    mid = (lo + hi) * 0.5
                    half = (hi - lo) * 0.5 * self._panoramic_augment_center_inflate
                    lo, hi = mid - half, mid + half
                aug_center = torch.tensor(
                    [aug_rng.uniform(lo[0].item(), hi[0].item()),
                     aug_rng.uniform(lo[1].item(), hi[1].item()),
                     aug_rng.uniform(lo[2].item(), hi[2].item())],
                    dtype=torch.float32,
                )
        elif self._panoramic_augment_center_sigma > 0:
            poses_list = assets.get("poses", [])
            valid = [p for p in poses_list
                     if p is not None and torch.is_tensor(p) and torch.all(torch.isfinite(p))]
            if valid:
                translations = torch.stack([p[:3, 3] for p in valid])
                mean = translations.mean(dim=0)
                scene_radius = (translations - mean).norm(dim=1).max().item()
                std = self._panoramic_augment_center_sigma * max(scene_radius, 0.1)
                aug_center = torch.tensor(
                    [aug_rng.gauss(mean[0].item(), std),
                     aug_rng.gauss(mean[1].item(), std),
                     mean[2].item()],
                    dtype=torch.float32,
                )
        elif self._panoramic_augment_center:
            poses_list = assets.get("poses", [])
            valid = [p for p in poses_list
                     if p is not None and torch.is_tensor(p) and torch.all(torch.isfinite(p))]
            if valid:
                aug_center = aug_rng.choice(valid)[:3, 3].clone()

        if self._panoramic_augment_yaw:
            aug_yaw = aug_rng.uniform(-math.pi, math.pi)

        return aug_center, aug_yaw
