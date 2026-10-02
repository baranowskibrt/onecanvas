"""PlacementMixin: spatial-pretraining samplers."""

import math
import os
import random
import sys
from copy import copy

import torch
from torch.utils.data import Dataset

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

# How a sample's body-patch budget is split across its boxes.
#   even (default) -- equal share per box, no size term. What every run so far
#                     trained on.
#   sqrt           -- share proportional to sqrt(surface area).
#   area           -- share proportional to surface area, constant density.
#
# OFF by default because it changes the canvas point distribution and therefore
# the training data, and flipping a data default under queued jobs is how a run
# silently stops being the run its card describes.
#
# MEASURED by dumping the SAME four families three ways (30 boxes each, the
# per-sample total held at ~204 in all three, so this is a redistribution and
# not a budget increase). density = points per m^2, spread = max/min over boxes:
#   even  87.1x spread, corr(points, area) +0.44
#   sqrt  38.4x spread, corr +0.83
#   area  20.1x spread, corr +0.88
# `area` is the only one that makes density genuinely constant, and it pays by
# taking the smallest boxes down to their 8 corners and little else: on a skewed
# area draw, which is what a log-uniform dims draw produces, the small boxes
# starve. `sqrt` removes about half the gradient and never starves one, which is
# why it is the recommended setting rather than the mathematically pure one.
#
# Simulated allocations predicted 84.8 / 17.1 / 13.4x; the dumps came back
# worse for sqrt and area because real dims are more extreme than the simulation
# assumed and the 8-corner floor is a bigger share of a small box than modelled.
# The measured row is the one to quote.
_BODY_ALLOC = os.environ.get("ONECANVAS_BODY_PATCH_ALLOC", "even").lower()
_BODY_ALLOC_POW = {"area": 1.0, "sqrt": 0.5}.get(_BODY_ALLOC)
_AREA_PROPORTIONAL_BODY = _BODY_ALLOC_POW is not None


class PlacementMixin:
    def _sample_multi_box(self, scene, rng, n_boxes, patch_indices_hint=None,
                          center_hints=None, rotation_mode="random",
                          dim_hints=None, budget_n_boxes=None):
        """Generate ``n_boxes`` non-overlapping OBBs, each with a distinct visual feature source.

        Each box gets:
        - An independently random center (IQR-bounded), dims, and rotation.
        - A reference patch from a different source frame to give visually distinct content.

        Stores on ``scene``:
          _multi_box_spherical:      list of [N_pts_i, 3] tensors (lat, lon, depth) per box.
          _multi_box_feature_sources: list of n_boxes ref patch indices.
          _multi_box_centers:        list of n_boxes [3] tensors (world-centered Cartesian).
          _multi_box_dims:           list of n_boxes (dx, dy, dz) tuples.
          _multi_box_rotations:      list of n_boxes [3, 3] rotation matrices.

        If ``patch_indices_hint`` is provided (list of ``n_boxes`` real patch indices),
        boxes are placed at those patches' Cartesian positions (so the outer sampler's
        angular / turn constraints over point centers carry over), each box uses the
        hinted patch as its feature source, and dims are capped so bounding spheres
        can't overlap. Used by ``rel_dir_*_box``.

        If ``center_hints`` is provided (list of ``n_boxes`` [3] tensors), boxes are
        placed at those centers and feature sources are drawn from unused scene frames
        (no coupling between box position and feature patch). Used by
        ``route_plan_*_box`` (class-conditional box-path sampler).

        If ``dim_hints`` is provided (list of ``n_boxes`` (dx, dy, dz) tuples), the
        random dim sampling is skipped and the hinted dims are used directly. Must be
        combined with ``center_hints`` (not compatible with ``patch_indices_hint``
        which applies its own dim cap). Used by ``visibility_from_pose`` to place a
        small observer marker alongside a standard-sized target marker.

        Returns:
          (patch_indices, frame_indices) — one real-patch index per box, or (None, None)
          on failure after retries.
        """
        use_hint = patch_indices_hint is not None
        use_center_hint = center_hints is not None
        use_dim_hint = dim_hints is not None
        if use_hint and use_center_hint:
            raise ValueError("cannot pass both patch_indices_hint and center_hints")
        if use_dim_hint and use_hint:
            raise ValueError("dim_hints must be combined with center_hints, not patch_indices_hint")
        if use_dim_hint and len(dim_hints) != n_boxes:
            return None, None
        hint_centers = None
        hint_frames = None
        d_upper_cap = math.log(3.26)
        if use_hint:
            if len(patch_indices_hint) != n_boxes:
                return None, None
            hint_centers = []
            hint_frames = []
            for pi in patch_indices_hint:
                la = float(scene.latitude[pi].item())
                lo = float(scene.longitude[pi].item())
                de = float(scene.depth[pi].item())
                cl = math.cos(la)
                hint_centers.append(torch.tensor(
                    [de * cl * math.sin(lo), de * math.sin(la), de * cl * math.cos(lo)],
                    dtype=torch.float32,
                ))
                hint_frames.append(int(scene.frame_index[pi].item()))
        if use_center_hint:
            if len(center_hints) != n_boxes:
                return None, None
            hint_centers = [c.detach().clone().to(torch.float32) for c in center_hints]
        if hint_centers is not None:
            min_pair = float("inf")
            for i in range(n_boxes):
                for j in range(i + 1, n_boxes):
                    d = (hint_centers[i] - hint_centers[j]).norm().item()
                    if d < min_pair:
                        min_pair = d
            # Bounding-sphere radius r = sqrt(dx²+dy²+dz²)/2 ≤ d_max * sqrt(3)/2.
            # Two boxes don't overlap iff r_i + r_j < ||c_i - c_j||, so with
            # d_max shared across boxes we need d_max < min_pair / sqrt(3).
            # Factor 0.9 keeps a small airgap.
            d_cap = min_pair * 0.9 / math.sqrt(3)
            if d_cap < 0.12:
                return None, None
            d_upper_cap = math.log(min(3.26, d_cap))

        lat_t, lon_t, depth_t = scene.latitude, scene.longitude, scene.depth
        cos_lat = torch.cos(lat_t)
        pts_all = torch.stack([
            depth_t * cos_lat * torch.sin(lon_t),
            depth_t * torch.sin(lat_t),
            depth_t * cos_lat * torch.cos(lon_t),
        ], dim=-1)  # [N_valid, 3]

        q25 = pts_all.quantile(0.25, dim=0)
        q75 = pts_all.quantile(0.75, dim=0)

        # Need at least n_boxes distinct source frames (only when drawing
        # feature sources ourselves — with a hint the caller already picked
        # distinct-frame patches).
        if not use_hint:
            n_frames = int(scene.frame_index.max().item()) + 1
            if n_frames < n_boxes:
                return None, None

        for _outer in range(20):
            placed_centers = []
            placed_bsphere_radii = []
            boxes_spherical = []
            boxes_feature_sources = []
            boxes_centers = []
            boxes_dims = []
            boxes_rotations = []
            used_frames = []

            ok = True
            for _box_i in range(n_boxes):
                placed = False
                for _attempt in range(50):
                    # 1. Center: hinted or random within scene IQR.
                    if hint_centers is not None:
                        center = hint_centers[_box_i].clone()
                    else:
                        center = torch.tensor([
                            rng.uniform(q25[0].item(), q75[0].item()),
                            rng.uniform(q25[1].item(), q75[1].item()),
                            rng.uniform(q25[2].item(), q75[2].item()),
                        ], dtype=torch.float32)

                    # 2. OBB dims: hinted (caller specifies exact per-box dims) or
                    # random log-uniform matching box_size range.
                    if use_dim_hint:
                        dx, dy, dz = (float(v) for v in dim_hints[_box_i])
                    else:
                        d_max = math.exp(rng.uniform(math.log(0.12), d_upper_cap))
                        d2 = math.exp(rng.uniform(math.log(0.07), math.log(d_max)))
                        d3 = math.exp(rng.uniform(math.log(0.07), math.log(d_max)))
                        dims = [d_max, d2, d3]
                        rng.shuffle(dims)
                        dx, dy, dz = dims
                    r_sphere = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / 2.0

                    # 3. Rotation (aligned / yaw / random) per caller's choice.
                    R = _rotation_matrix(rotation_mode, rng)

                    # 4. Non-overlap check (bounding-sphere). With hinted centers the
                    # d_upper_cap already guarantees this, but we keep the check as
                    # a safety net for corner cases (shuffled dims can still exceed
                    # d_max cap on other axes up to d_max, so sqrt(3)*d_max is the
                    # true upper bound on the diagonal — same as the cap).
                    if any(
                        (center - c).norm().item() < r_sphere + r2
                        for c, r2 in zip(placed_centers, placed_bsphere_radii)
                    ):
                        continue

                    # 5. Sample surface points on the OBB and convert to spherical coords.
                    world_pts = _sample_obb_surface_points(
                        center, R, (dx, dy, dz),
                        n_total=self._per_box_budget(
                            budget_n_boxes if budget_n_boxes is not None else n_boxes))
                    lats, lons, depths, valid = _world_to_spherical(world_pts)
                    if valid.sum() < 4:
                        continue

                    # 6. Pick feature source: hinted patch (preserves caller's
                    # distinct-frame choice) or search for an unused frame.
                    # center_hints decouples box position from feature source —
                    # feature patch is drawn from an unused frame like the no-hint path.
                    if use_hint:
                        ref_idx = int(patch_indices_hint[_box_i])
                        ref_frame = hint_frames[_box_i]
                    else:
                        for _ref_try in range(30):
                            ref_idx = rng.randrange(scene.n_valid)
                            ref_frame = int(scene.frame_index[ref_idx].item())
                            if ref_frame not in used_frames:
                                break
                        else:
                            continue

                    spherical_k = torch.stack(
                        [lats[valid], lons[valid], depths[valid]], dim=-1)

                    placed_centers.append(center)
                    placed_bsphere_radii.append(r_sphere)
                    boxes_spherical.append(spherical_k)
                    boxes_feature_sources.append(ref_idx)
                    boxes_centers.append(center)
                    boxes_dims.append((dx, dy, dz))
                    boxes_rotations.append(R)
                    used_frames.append(ref_frame)
                    placed = True
                    break

                if not placed:
                    ok = False
                    break

            if ok and _AREA_PROPORTIONAL_BODY:
                # CONSTANT SURFACE DENSITY, per sample (2026-08-10).
                #
                # `_per_box_budget` splits the sample's body-patch budget evenly
                # by BOX COUNT, with no size term, so a box gets the same number
                # of surface points whether it is 0.002 m^2 or 3.0 m^2. Measured
                # over 118 boxes of a 17-family dump: corr(points, volume) =
                # +0.20, the largest volume quartile carries 1.26x the points of
                # the smallest for 75x the volume, and density spans 1 to 69933
                # points per m^3. A small object is therefore rendered as a dense
                # blob and a large one as a sparse shell, which is backwards from
                # a real lifted-depth canvas and hands the model a spurious cue
                # that runs INVERSE to size on the very families that ask about
                # size.
                #
                # Fixed by allocating the SAME budget in proportion to surface
                # area, which is what the points are samples OF. Per sample, not
                # global: a fixed points-per-m^2 constant preserved the mean but
                # swung the per-sample total from 30 to 400 against a cap of 200,
                # and sequence length is the thing the budget exists to bound.
                # Within a sample density is now constant; across samples the
                # total is unchanged, which is the pair of properties that
                # matters since the model never compares two samples.
                #
                # SECOND PASS, because the areas are not known until every box
                # has survived its rejection loop. The first pass keeps its own
                # sampling: it feeds the `valid.sum() < 4` rejection test, which
                # is a placement criterion and must not move. A re-sample that
                # comes back degenerate keeps the first-pass points rather than
                # dropping a box that was already accepted.
                _areas = []
                for (_dx, _dy, _dz) in boxes_dims:
                    _a = 2.0 * (_dx * _dy + _dy * _dz + _dx * _dz)
                    _areas.append(_a ** _BODY_ALLOC_POW)
                _tot = float(sum(_areas))
                _n = len(boxes_dims)
                _budget = max(8 * _n, int(self._max_body_patches_per_sample))
                if _tot > 0 and _n:
                    for _i in range(_n):
                        _n_i = max(8, int(round(_budget * _areas[_i] / _tot)))
                        _pts = _sample_obb_surface_points(
                            boxes_centers[_i], boxes_rotations[_i],
                            boxes_dims[_i], n_total=_n_i)
                        _la, _lo, _de, _va = _world_to_spherical(_pts)
                        if int(_va.sum()) >= 4:
                            boxes_spherical[_i] = torch.stack(
                                [_la[_va], _lo[_va], _de[_va]], dim=-1)

            if ok:
                # For distance tasks (dist_box / rel_dist_box*), augment each box's
                # marker points with the closest surface point to every other box.
                # The inter-box nearest pair IS the GT, so adding these two points
                # makes the observable nearest-marker distance coincide with the
                # label up to the 300-sample surface-approximation error (same
                # approximation the GT itself uses). Harmless for non-distance
                # multi-box tasks: the extra points still live on the box surface
                # and carry the same feature as the rest of that box's markers.
                for i in range(len(boxes_centers)):
                    for j in range(i + 1, len(boxes_centers)):
                        p_i, p_j, _ = obb_closest_surface_points(
                            boxes_centers[i], boxes_dims[i], boxes_rotations[i],
                            boxes_centers[j], boxes_dims[j], boxes_rotations[j],
                        )
                        extra = torch.stack([p_i, p_j], dim=0)  # [2, 3]
                        lats_e, lons_e, depths_e, valid_e = _world_to_spherical(extra)
                        if valid_e[0]:
                            add_i = torch.stack(
                                [lats_e[0:1], lons_e[0:1], depths_e[0:1]], dim=-1)
                            boxes_spherical[i] = torch.cat(
                                [boxes_spherical[i], add_i], dim=0)
                        if valid_e[1]:
                            add_j = torch.stack(
                                [lats_e[1:2], lons_e[1:2], depths_e[1:2]], dim=-1)
                            boxes_spherical[j] = torch.cat(
                                [boxes_spherical[j], add_j], dim=0)

                scene._multi_box_spherical = boxes_spherical
                scene._multi_box_feature_sources = boxes_feature_sources
                scene._multi_box_centers = boxes_centers
                scene._multi_box_dims = boxes_dims
                scene._multi_box_rotations = boxes_rotations
                frame_indices = [
                    int(scene.frame_index[ri].item()) for ri in boxes_feature_sources
                ]
                return boxes_feature_sources, frame_indices

        return None, None

    def _sample_object_class_grounding_real(self, scene, rng):
        """Real-asset twin of multi_box_grounding: find every instance of one class.

        Pastes N in {1, 2, 3, 4} target-class assets (the synthetic sampler's
        N weights, [15, 45, 30, 10]) at IQR collision-free centers with random
        yaw, plus other-class distractors drawn from the curriculum's real
        distractor range with a floor of one, because "find every X" needs at
        least one object that is not an X. The QA builder emits one
        axis-aligned box per target instance in near-to-far order from the
        canvas origin, the same canonical order and answer format as
        multi_box_grounding, scored by F1@0.25 (utils.metrics.curriculum_score).
        Each box wraps the asset's RENDERED patches (_asset_paste_world_pts,
        the points dist_real measures over), so the target is exactly what the
        canvas shows. Inflation is forced tight because the pasted region is
        the supervision target.

        Stores ``scene._grounding_real_target_label`` and
        ``scene._grounding_real_target_world_pts_list`` (list of [K_i, 3] in
        the panorama intermediate frame). Returns ([], []): the referent is a
        class name, no inline markers.
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng, force_tight=True)
        if bank is None or len(bank.labels()) < 2:
            return None, None

        labels = bank.labels()
        target_label = rng.choice(labels)
        other_labels = [l for l in labels if l != target_label]
        n_targets = rng.choices([1, 2, 3, 4], weights=[15, 45, 30, 10])[0]
        q25, q75 = _scene_iqr_aabb(scene)

        placed_centers: list = []
        placed_radii: list = []
        pastes: list = []
        target_pts: list = []

        def _paste(asset, center, yaw, label):
            spread = int(asset["frame_indices"].max().item()
                         - asset["frame_indices"].min().item())
            t_max = max(0, int(scene.n_images) - 1 - spread)
            t_start = rng.randint(0, t_max) if t_max > 0 else 0
            return {
                "asset": asset, "target_center": center,
                "yaw_rad": float(yaw), "t_start": int(t_start),
                "label": label,
            }

        for _ in range(n_targets):
            asset = bank.sample(target_label, rng, inflation_frac=inflation_frac)
            r_sphere = float(asset["bbox_dims"].max().item()) / 2.0 + 0.20
            center = _sample_collision_free_center(
                placed_centers, placed_radii, r_sphere, rng,
                aabb_min=q25, aabb_max=q75,
            )
            if center is None:
                # No room for another instance: keep what fits. The caller
                # retries the sample when nothing fits at all.
                break
            yaw = rng.uniform(-math.pi, math.pi)
            pts = _asset_paste_world_pts(asset, center, yaw)
            if pts.numel() == 0:
                continue
            placed_centers.append(center)
            placed_radii.append(r_sphere)
            pastes.append(_paste(asset, center, yaw, target_label))
            target_pts.append(pts)

        if not target_pts:
            return None, None

        n_distract = rng.randint(max(1, self._real_asset_distract_min),
                                 max(1, self._real_asset_distract_max))
        for _ in range(n_distract):
            d_label = rng.choice(other_labels)
            asset = bank.sample(d_label, rng, inflation_frac=inflation_frac)
            r_sphere = float(asset["bbox_dims"].max().item()) / 2.0 + 0.20
            center = _sample_collision_free_center(
                placed_centers, placed_radii, r_sphere, rng,
                aabb_min=q25, aabb_max=q75,
            )
            if center is None:
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            placed_centers.append(center)
            placed_radii.append(r_sphere)
            pastes.append(_paste(asset, center, yaw, d_label))

        scene._real_asset_pastes = pastes
        scene._grounding_real_target_label = target_label
        scene._grounding_real_target_world_pts_list = target_pts
        return [], []

    def _place_distractor_boxes(self, scene, task, rng):
        """Place M ∈ [min, max] distractor OBBs on the canvas with unique stash features.

        Distractors are non-overlapping with each other and with all already-placed
        task OBBs. Each distractor has its own unique feature source (canvas patch
        index), drawn from an unused frame so the per-canvas-patch stash draw in
        ``_build_sample_stash_fast`` produces visually distinct content across
        distractors. Distractors are NOT referenced by the task prompt; they
        are pure visual noise that the model must learn to ignore based on the
        task's ``<patch>`` token class.

        Attaches to ``scene`` (when M > 0 and at least one distractor placed):
          _distractor_spherical:       list of [N_pts_i, 3] tensors.
          _distractor_feature_sources: list of M canvas patch indices.
          _distractor_centers:         list of M [3] tensors.
          _distractor_dims:            list of M (dx, dy, dz) tuples.
          _distractor_rotations:       list of M [3, 3] matrices.

        No-op when ``curriculum_num_distractors_max <= 0`` or for tasks whose
        base name contains ``floor_area`` (room-size semantics — extra markers
        change what "the room" is). For ``visibility_from_pose`` distractors
        are still added but are forbidden from intersecting the p1->p2 cone
        of sight, so the task's precomputed yes/no label remains valid. For
        ``route_plan_*`` distractors are kept off the walked-path segments
        (m_face, reface only, is excluded so distractors may still land near
        it) — a distractor on a walked segment would contradict the prompt's
        "Go forward until [wp_k]" instruction even though it leaves the GT
        turn angles unchanged.
        """
        M_max = int(getattr(self.data_args, "curriculum_num_distractors_max", 0))
        M_min = max(0, int(getattr(self.data_args, "curriculum_num_distractors_min", 0)))
        if M_max <= 0:
            return
        if "floor_area" in task:
            return
        # Opt-out seam for externally-registered tasks: a plugin sampler that
        # already fills the non-referenced background role itself (e.g. with
        # the _extra_obb_* structures) sets this on the scene, since adding
        # free-floating distractor OBBs on top would clutter the canvas
        # without adding a shortcut-killing signal.
        if getattr(scene, "_suppress_obb_distractors", False):
            return
        # *_real tasks ship their own real-asset distractors
        # of different classes via the *_real samplers; adding synthetic OBB
        # distractors on top would mix two distractor mechanisms and dilute
        # the per-patch-feature signal that's the point of this curriculum.
        if task.endswith("_real"):
            return
        if scene.n_valid < 1:
            return
        M_min = min(M_min, M_max)
        M = rng.randint(M_min, M_max)
        if M <= 0:
            return

        base_task, _ = _parse_box_task(_strip_display_suffix(task))

        # --- Existing task-box collision info (centers + bounding-sphere radii).
        task_centers = []
        task_radii = []
        _mbc = getattr(scene, "_multi_box_centers", None) or []
        _mbd = getattr(scene, "_multi_box_dims", None) or []
        for c, dims in zip(_mbc, _mbd):
            task_centers.append(c.detach().clone().to(torch.float32))
            dx, dy, dz = (float(v) for v in dims)
            task_radii.append(math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / 2.0)

        # For visibility_from_pose: distractors MUST NOT lie anywhere near the
        # p1->p2 line of sight, or they would occlude rays that the task's
        # visible_frac was computed assuming only the designated occluder is
        # present — silently flipping "yes" labels to "no". Capture p1, p2
        # centers and use a conservative cylindrical clearance (p2 has a
        # 0.30 m AABB so half-diagonal ≈ 0.26 m bounds the ray spread).
        visibility_p1 = visibility_p2 = None
        if base_task == "visibility_from_pose" and _mbc and len(_mbc) >= 2:
            visibility_p1 = _mbc[0].detach().clone().to(torch.float32)
            visibility_p2 = _mbc[1].detach().clone().to(torch.float32)

        # For multi_box_grounding*: distractors must be outside the target
        # AABB (GT bbox wraps task OBBs; distractors inside would be false
        # positives against the model's predicted bbox).
        aabb_min = aabb_max = None
        if base_task == "multi_box_grounding" and _mbc and _mbd:
            _mbr = getattr(scene, "_multi_box_rotations", None) or []
            corners = []
            for c, dims, R in zip(_mbc, _mbd, _mbr):
                dx, dy, dz = (float(v) for v in dims)
                local = torch.tensor([
                    [sx * dx / 2.0, sy * dy / 2.0, sz * dz / 2.0]
                    for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
                ], dtype=torch.float32)
                world = (R.to(torch.float32) @ local.T).T + c.to(torch.float32)
                corners.append(world)
            all_corners = torch.cat(corners, dim=0)
            aabb_min = all_corners.min(dim=0).values
            aabb_max = all_corners.max(dim=0).values

        # For route_plan_*: keep distractors off the walked-path segments.
        # The prompt "Go forward until [wp_k]" is incoherent if a distractor
        # straddles m_k -> m_{k+1}, even though the GT turn angles are
        # waypoint-to-waypoint and don't numerically depend on what's between.
        # m_face (reface only, path index 1) is NOT walked, so it's excluded
        # from the segment list — distractors may still land near it.
        route_segments = []
        if base_task.startswith("route_plan_"):
            path_centers = getattr(scene, "_route_plan_box_path_centers", None) or []
            path_dims = getattr(scene, "_route_plan_box_path_dims", None) or []
            is_reface = bool(getattr(scene, "_route_plan_reface", False))
            if len(path_centers) >= 2 and len(path_dims) == len(path_centers):
                if is_reface:
                    # path order: [m0, m_face, m1, m2, ..., m_{N+1}]
                    walked = [0] + list(range(2, len(path_centers)))
                else:
                    # path order: [m0, m1, m2, ..., m_{N+1}]
                    walked = list(range(len(path_centers)))
                for i in range(len(walked) - 1):
                    a, b = walked[i], walked[i + 1]
                    p1 = path_centers[a].detach().clone().to(torch.float32)
                    p2 = path_centers[b].detach().clone().to(torch.float32)
                    dims_a = path_dims[a]
                    dims_b = path_dims[b]
                    r_a = math.sqrt(sum(float(v) ** 2 for v in dims_a)) / 2.0
                    r_b = math.sqrt(sum(float(v) ** 2 for v in dims_b)) / 2.0
                    route_segments.append((p1, p2, max(r_a, r_b)))

        # --- Canvas patch indices already used as task feature sources (avoid
        # reuse so distractors land on different stash draws by construction).
        used_refs = set()
        used_frames = set()
        _mbfs = getattr(scene, "_multi_box_feature_sources", None) or []
        for ri in _mbfs:
            used_refs.add(int(ri))
            used_frames.add(int(scene.frame_index[int(ri)].item()))
        _sfs = getattr(scene, "_synthetic_feature_source", None)
        if _sfs is not None:
            sfs_int = int(_sfs.item() if hasattr(_sfs, "item") else _sfs)
            if sfs_int >= 0:
                used_refs.add(sfs_int)
                used_frames.add(int(scene.frame_index[sfs_int].item()))

        # --- IQR of valid patch positions for center sampling.
        lat_t = scene.latitude
        lon_t = scene.longitude
        depth_t = scene.depth
        cos_lat = torch.cos(lat_t)
        pts_all = torch.stack([
            depth_t * cos_lat * torch.sin(lon_t),
            depth_t * torch.sin(lat_t),
            depth_t * cos_lat * torch.cos(lon_t),
        ], dim=-1)
        q25 = pts_all.quantile(0.25, dim=0)
        q75 = pts_all.quantile(0.75, dim=0)

        placed_centers = list(task_centers)
        placed_radii = list(task_radii)

        d_spherical = []
        d_feature_sources = []
        d_centers = []
        d_dims = []
        d_rotations = []

        # Fixed small body budget per distractor (8 corners + 8 face samples).
        # Keeps the total extra patches bounded at M * 16 ≤ 64 regardless of the
        # task's own budget.
        _dist_body_n = 16

        for _di in range(M):
            placed = False
            for _attempt in range(60):
                # Dims: modest range so distractors are visible but don't dominate.
                d_max = math.exp(rng.uniform(math.log(0.20), math.log(1.0)))
                d2 = math.exp(rng.uniform(math.log(0.10), math.log(d_max)))
                d3 = math.exp(rng.uniform(math.log(0.10), math.log(d_max)))
                dims_list = [d_max, d2, d3]
                rng.shuffle(dims_list)
                dx, dy, dz = dims_list
                r_sphere = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / 2.0

                R = _rotation_matrix("yaw", rng)

                center = torch.tensor([
                    rng.uniform(q25[0].item(), q75[0].item()),
                    rng.uniform(q25[1].item(), q75[1].item()),
                    rng.uniform(q25[2].item(), q75[2].item()),
                ], dtype=torch.float32)

                if any(
                    (center - c).norm().item() < r_sphere + r2
                    for c, r2 in zip(placed_centers, placed_radii)
                ):
                    continue

                if aabb_min is not None:
                    clamped = torch.max(torch.min(center, aabb_max), aabb_min)
                    if (center - clamped).norm().item() < r_sphere:
                        continue

                # visibility_from_pose: keep the distractor's bounding sphere
                # clear of every p1->q ray where q ranges over p2's AABB.
                # Closest-point-on-segment distance must exceed the distractor
                # sphere radius plus p2's half-diagonal (~0.26 m) plus a small
                # airgap. Conservative (treats rays as a cylinder of max half
                # width rather than a cone), which is fine — we just retry.
                if visibility_p1 is not None:
                    seg = visibility_p2 - visibility_p1
                    seg_len_sq = float((seg * seg).sum().item())
                    if seg_len_sq < 1e-9:
                        closest = visibility_p1
                    else:
                        t_along = float(((center - visibility_p1) * seg).sum().item()) / seg_len_sq
                        t_along = max(0.0, min(1.0, t_along))
                        closest = visibility_p1 + t_along * seg
                    dist_to_seg = float((center - closest).norm().item())
                    if dist_to_seg < r_sphere + 0.26 + 0.05:
                        continue

                # route_plan_*: reject candidates that intrude on any walked
                # segment. Same closest-point-on-segment math as the
                # visibility block above; clearance = r_distractor + max
                # waypoint bounding-sphere radius + 0.05 m airgap.
                if route_segments:
                    too_close = False
                    for (rp1, rp2, r_wp) in route_segments:
                        seg = rp2 - rp1
                        seg_len_sq = float((seg * seg).sum().item())
                        if seg_len_sq < 1e-9:
                            closest = rp1
                        else:
                            t_along = float(((center - rp1) * seg).sum().item()) / seg_len_sq
                            t_along = max(0.0, min(1.0, t_along))
                            closest = rp1 + t_along * seg
                        dist_to_seg = float((center - closest).norm().item())
                        if dist_to_seg < r_sphere + r_wp + 0.05:
                            too_close = True
                            break
                    if too_close:
                        continue

                world_pts = _sample_obb_surface_points(
                    center, R, (dx, dy, dz), n_total=_dist_body_n,
                )
                lats, lons, depths, valid = _world_to_spherical(world_pts)
                if int(valid.sum().item()) < 4:
                    continue

                # Pick a distractor ref_idx: prefer unused frame so stash picks
                # yield fresh content; fall back to any unused canvas index.
                # route_plan opt-in: clone a landmark feature with probability
                # curriculum_distractor_route_clone_landmark_p so the distractor is
                # visually identical to a waypoint and disambiguation has to
                # come from the inline marker IDs / OBB positions.
                ref_idx = None
                _clone_p = float(getattr(
                    self.data_args,
                    "curriculum_distractor_route_clone_landmark_p",
                    0.0,
                ))
                if (
                    _clone_p > 0.0
                    and base_task.startswith("route_plan_")
                    and _mbfs
                    and rng.random() < _clone_p
                ):
                    ref_idx = int(rng.choice(_mbfs))
                else:
                    for _rtry in range(30):
                        cand = rng.randrange(scene.n_valid)
                        if int(cand) in used_refs:
                            continue
                        cand_frame = int(scene.frame_index[cand].item())
                        if cand_frame in used_frames:
                            continue
                        ref_idx = int(cand)
                        used_refs.add(ref_idx)
                        used_frames.add(cand_frame)
                        break
                    if ref_idx is None:
                        for _rtry in range(30):
                            cand = rng.randrange(scene.n_valid)
                            if int(cand) in used_refs:
                                continue
                            ref_idx = int(cand)
                            used_refs.add(ref_idx)
                            break
                if ref_idx is None:
                    continue

                spherical_k = torch.stack(
                    [lats[valid], lons[valid], depths[valid]], dim=-1,
                )
                d_spherical.append(spherical_k)
                d_feature_sources.append(ref_idx)
                d_centers.append(center)
                d_dims.append((float(dx), float(dy), float(dz)))
                d_rotations.append(R)
                placed_centers.append(center)
                placed_radii.append(r_sphere)
                placed = True
                break

            if not placed:
                # Couldn't fit this distractor; stop early (< M_target is fine).
                break

        if d_spherical:
            scene._distractor_spherical = d_spherical
            scene._distractor_feature_sources = d_feature_sources
            scene._distractor_centers = d_centers
            scene._distractor_dims = d_dims
            scene._distractor_rotations = d_rotations
