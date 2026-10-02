"""PatchesMixin: spatial-pretraining samplers."""

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


class PatchesMixin:
    def _sample_patches(self, task, scene, rng):
        """Sample patch indices + their source frame indices for the given task."""
        if scene.n_valid < 2:
            return None, None

        # Strip the "_obb_only" alias suffix first so sampling sees the base
        # task name. The suffix only affects canvas-stripping / metric label.
        task = _strip_display_suffix(task)

        # Parse rotation-mode suffix for box-family tasks (aligned / yaw / random).
        # Non-box tasks are left untouched; box tasks without a suffix use the
        # per-task default from _DEFAULT_ROTATION_BY_TASK.
        base_task, rotation_mode = _parse_box_task(task)
        if rotation_mode is not None:
            task = base_task

        if task == "appearance_order":
            # Synthetic-target + exact-visibility variant. Four synthetic
            # OBB markers are placed at random scene positions. For each
            # marker the ground-truth first-appearance time is the earliest
            # frame whose camera frustum contains the marker's center
            # (computed in world space from real camera poses and intrinsics).
            # No depth-map occlusion test — that's noisy by construction; we
            # trade it for a clean analytic label.
            #
            # The full per-box visibility mask is stashed on the scene so the
            # downstream T-spread block can draw body-point T values ONLY
            # from frames that actually observe the box. Assigning T_n to a
            # body point when camera T_n does not see the box would be
            # self-contradictory supervision.
            #
            # Retry until all four first-visible-frames are distinct AND
            # every box is visible from >= 2 frames (otherwise the T-spread
            # collapses to a single value and the task degenerates to
            # "read the only T on the box").
            if (scene.poses is None or scene.intrinsics is None
                    or scene.image_dims is None):
                return None, None
            N = self._appearance_n_boxes
            for _try in range(50):
                pis, _src_frames = self._sample_multi_box(scene, rng, n_boxes=N)
                if pis is None:
                    return None, None
                centers_local = torch.stack(
                    [c.float() for c in scene._multi_box_centers], dim=0)  # [N, 3]
                centers_world = _canvas_local_to_world(
                    centers_local, scene.center_point, scene.yaw_angle,
                )
                vis = visibility_matrix(
                    centers_world,
                    scene.poses.float(),
                    scene.intrinsics.float(),
                    scene.image_dims.float(),
                )  # [N, N_imgs]
                vis_counts = vis.sum(dim=1)
                if (vis_counts < 2).any():
                    continue
                first = first_visible_frame(centers_world, None, None, None, vis=vis)  # [N]
                if (first < 0).any():
                    continue
                first_list = first.tolist()
                if len(set(first_list)) < N:
                    continue
                # Shuffle the slots but keep the (box, first_visible, vis_row)
                # alignment so the T-spread block reads the right visibility
                # row per slot.
                order = list(range(N))
                rng.shuffle(order)
                patch_indices = [int(pis[i]) for i in order]
                frame_indices = [int(first_list[i]) for i in order]
                scene._appearance_visible_frames = vis[order]  # [N, N_imgs]
                # Reorder scene._multi_box_* to match the shuffled slot order so
                # downstream per-slot lookups (body points, feature sources,
                # dims, rotations) line up with patch_indices / frame_indices.
                scene._multi_box_spherical = [scene._multi_box_spherical[i] for i in order]
                scene._multi_box_feature_sources = [scene._multi_box_feature_sources[i] for i in order]
                scene._multi_box_centers = [scene._multi_box_centers[i] for i in order]
                scene._multi_box_dims = [scene._multi_box_dims[i] for i in order]
                scene._multi_box_rotations = [scene._multi_box_rotations[i] for i in order]
                return patch_indices, frame_indices
            return None, None
        elif task in ("rel_dir_easy", "rel_dir_medium", "rel_dir_hard", "rel_dir_4way"):
            # 3-patch egocentric direction task. Enforce distinct source frames.
            # Balanced sampling: fix ref+fwd first, then pick a target label
            # uniformly at random and search for a tgt patch that lands in that
            # quadrant. This avoids the natural front-heavy bias that arises from
            # purely random patch sampling. Margin = 15° from any decision boundary.
            if scene.n_valid < 3:
                return None, None
            frame_index = scene.frame_index
            lat_t = scene.latitude
            lon_t = scene.longitude
            depth_t = scene.depth
            cos_lat = torch.cos(lat_t)

            def _xz(i):
                d_i = float(depth_t[i].item())
                cl = float(cos_lat[i].item())
                lo = float(lon_t[i].item())
                return d_i * cl * math.sin(lo), d_i * cl * math.cos(lo)

            sin_margin = math.sin(math.radians(15.0))
            boundary_cos = math.cos(math.radians(135))  # used by medium
            c30 = math.cos(math.radians(30))            # used by 4way (45°±15° margin)
            c60 = math.cos(math.radians(60))            # used by 4way

            if task == "rel_dir_easy":
                target_labels = ["left", "right"]
            elif task == "rel_dir_medium":
                target_labels = ["left", "right", "back"]
            elif task == "rel_dir_4way":
                target_labels = ["front", "back", "left", "right"]
            else:
                target_labels = ["front-left", "front-right", "back-left", "back-right"]

            for _ in range(30):
                # Sample ref and fwd from different frames with enough distance
                ia, ib = rng.sample(range(scene.n_valid), 2)
                if int(frame_index[ia].item()) == int(frame_index[ib].item()):
                    continue
                ax, az = _xz(ia); bx, bz = _xz(ib)
                hx, hz = bx - ax, bz - az
                h_n = math.sqrt(hx * hx + hz * hz)
                if h_n < 0.3:
                    continue
                hx /= h_n; hz /= h_n

                # Pick a target label uniformly and search for a matching tgt
                target_label = rng.choice(target_labels)
                fa, fb = int(frame_index[ia].item()), int(frame_index[ib].item())
                cands = [j for j in range(scene.n_valid)
                         if j != ia and j != ib
                         and int(frame_index[j].item()) not in (fa, fb)]
                rng.shuffle(cands)
                for ic in cands[:80]:
                    cx, cz = _xz(ic)
                    tx, tz = cx - ax, cz - az
                    t_n = math.sqrt(tx * tx + tz * tz)
                    if t_n < 0.3:
                        continue
                    tx /= t_n; tz /= t_n
                    dot   = hx * tx + hz * tz
                    cross = hx * tz - hz * tx

                    if task == "rel_dir_easy":
                        if abs(cross) < sin_margin:
                            continue
                        if target_label == "left"  and cross <= 0: continue
                        if target_label == "right" and cross >= 0: continue
                    elif task == "rel_dir_medium":
                        if abs(dot - boundary_cos) < sin_margin:
                            continue
                        if abs(cross) < sin_margin and dot > boundary_cos:
                            continue
                        if target_label == "back":
                            if dot >= boundary_cos: continue
                        else:
                            if dot < boundary_cos: continue
                            if target_label == "left"  and cross <= 0: continue
                            if target_label == "right" and cross >= 0: continue
                    elif task == "rel_dir_4way":
                        # 45° boundaries with 15° buffer → solid front/back
                        # zones are ±30° cones (dot > c30 / dot < -c30);
                        # solid left/right zones are 60°-120° sectors
                        # (|dot| < c60 = cos(60°) = 0.5).
                        if target_label == "front":
                            if dot < c30: continue
                        elif target_label == "back":
                            if dot > -c30: continue
                        else:
                            if abs(dot) > c60: continue
                            if target_label == "left"  and cross <= 0: continue
                            if target_label == "right" and cross >= 0: continue
                    else:  # hard
                        if abs(dot) < sin_margin:   continue
                        if abs(cross) < sin_margin: continue
                        want_front = target_label.startswith("front")
                        want_left  = target_label.endswith("left")
                        if want_front  and dot   <= 0: continue
                        if not want_front and dot >= 0: continue
                        if want_left   and cross <= 0: continue
                        if not want_left  and cross >= 0: continue

                    fc = int(frame_index[ic].item())
                    return [ia, ib, ic], [fa, fb, fc]
            return None, None
        elif task in ("rel_dir_camera_easy", "rel_dir_camera_medium", "rel_dir_camera_hard"):
            # SPBench-SI style "From the camera's perspective, is X to Y's left/
            # right/front/back?". Viewer = the camera the canvas was reoriented
            # onto (origin + canvas-forward stored on scene._rel_dir_camera_*).
            # Pick a pivot patch (Y) and a target patch (X) whose direction
            # from Y — measured in the camera's axes (== canvas axes by
            # construction) — lands in the requested quadrant. Keeps the 15°
            # margin rejection used by the non-camera variants.
            if scene.n_valid < 2:
                return None, None
            frame_index = scene.frame_index
            lat_t = scene.latitude
            lon_t = scene.longitude
            depth_t = scene.depth
            cos_lat = torch.cos(lat_t)

            def _xz(i):
                d_i = float(depth_t[i].item())
                cl = float(cos_lat[i].item())
                lo = float(lon_t[i].item())
                return d_i * cl * math.sin(lo), d_i * cl * math.cos(lo)

            sin_margin = math.sin(math.radians(15.0))
            boundary_cos = math.cos(math.radians(135))

            if task == "rel_dir_camera_easy":
                target_labels = ["left", "right"]
            elif task == "rel_dir_camera_medium":
                target_labels = ["left", "right", "back"]
            else:
                target_labels = ["front-left", "front-right", "back-left", "back-right"]

            # Camera-axes forward in canvas-local coords (+Z after reorient).
            hx, hz = 0.0, 1.0

            for _ in range(30):
                i_piv = rng.randrange(scene.n_valid)
                ax, az = _xz(i_piv)
                target_label = rng.choice(target_labels)
                fa = int(frame_index[i_piv].item())
                cands = [j for j in range(scene.n_valid)
                         if j != i_piv and int(frame_index[j].item()) != fa]
                rng.shuffle(cands)
                for i_tgt in cands[:80]:
                    cx, cz = _xz(i_tgt)
                    tx, tz = cx - ax, cz - az
                    t_n = math.sqrt(tx * tx + tz * tz)
                    if t_n < 0.3:
                        continue
                    tx /= t_n; tz /= t_n
                    dot   = hx * tx + hz * tz
                    cross = hx * tz - hz * tx

                    if task == "rel_dir_camera_easy":
                        if abs(cross) < sin_margin:
                            continue
                        if target_label == "left"  and cross <= 0: continue
                        if target_label == "right" and cross >= 0: continue
                    elif task == "rel_dir_camera_medium":
                        if abs(dot - boundary_cos) < sin_margin:
                            continue
                        if abs(cross) < sin_margin and dot > boundary_cos:
                            continue
                        if target_label == "back":
                            if dot >= boundary_cos: continue
                        else:
                            if dot < boundary_cos: continue
                            if target_label == "left"  and cross <= 0: continue
                            if target_label == "right" and cross >= 0: continue
                    else:  # hard
                        if abs(dot) < sin_margin:   continue
                        if abs(cross) < sin_margin: continue
                        want_front = target_label.startswith("front")
                        want_left  = target_label.endswith("left")
                        if want_front  and dot   <= 0: continue
                        if not want_front and dot >= 0: continue
                        if want_left   and cross <= 0: continue
                        if not want_left  and cross >= 0: continue

                    ft = int(frame_index[i_tgt].item())
                    # Return [target, pivot] so the first inline marker in the
                    # prompt ("is <marker0> to <marker1>'s ...") is the target.
                    return [i_tgt, i_piv], [ft, fa]
            return None, None
        elif task in (
            "rel_dir_camera_easy_box",
            "rel_dir_camera_medium_box",
            "rel_dir_camera_hard_box",
        ):
            # Box variant of rel_dir_camera_*. Pick (target, pivot) via the
            # point sampler, then wrap both in OBBs whose feature source is
            # the referenced patch. Retry until every (tgt_corner, piv_corner)
            # pair produces the same label under the camera's canvas-local
            # forward (+Z after reorient). Matches the corner-consistency
            # regime used by the world-frame rel_dir_*_box variants.
            patch_task = task[:-len("_box")]
            for _retry in range(20):
                patch_indices, frame_indices = self._sample_patches(
                    patch_task, scene, rng,
                )
                if patch_indices is None:
                    return None, None
                box_pis, _ = self._sample_multi_box(
                    scene, rng,
                    n_boxes=len(patch_indices),
                    patch_indices_hint=patch_indices,
                )
                if box_pis is None:
                    continue
                if _rel_dir_camera_boxes_consistent(
                    scene._multi_box_centers,
                    scene._multi_box_dims,
                    scene._multi_box_rotations,
                    patch_task,
                ):
                    return patch_indices, frame_indices
            return None, None
        elif task in (
            "rel_dir_easy_box", "rel_dir_medium_box", "rel_dir_hard_box",
            "rel_dir_4way_box",
        ):
            # rel_dir_*_box: reuse the patch-task sampler for center placement
            # (so the angular-margin rejection sampling carries over), then
            # wrap each sampled patch in an OBB whose feature source is that
            # same patch. Down-stream label logic reads the 3D centers of
            # patch_indices, which coincide with the box centers by construction.
            # The center-only 15° margin is insufficient for boxes with extent
            # (corners can cross quadrant boundaries). Retry until every corner
            # triple gives the same label — see _rel_dir_boxes_consistent.
            patch_task = task[:-len("_box")]
            for _retry in range(20):
                patch_indices, frame_indices = self._sample_patches(patch_task, scene, rng)
                if patch_indices is None:
                    return None, None
                box_pis, _ = self._sample_multi_box(
                    scene, rng,
                    n_boxes=len(patch_indices),
                    patch_indices_hint=patch_indices,
                )
                if box_pis is None:
                    continue
                if _rel_dir_boxes_consistent(
                    scene._multi_box_centers,
                    scene._multi_box_dims,
                    scene._multi_box_rotations,
                    patch_task,
                ):
                    return patch_indices, frame_indices
            return None, None
        elif task == "rel_dir_oclock_box":
            # Situated o'clock direction. 3 marker patches (ref, fwd, tgt)
            # wrapped in boxes; hour is unambiguous across all 729 corner
            # triples. Sampling picks the hour uniformly and rejects targets
            # outside a ±7.5° band around the hour center (30° sectors are
            # narrow — smaller margin than rel_dir_easy's 15°).
            if scene.n_valid < 3:
                return None, None
            frame_index = scene.frame_index
            depth_t = scene.depth
            cos_lat = torch.cos(scene.latitude)
            lon_t = scene.longitude

            def _xz(i):
                d_i = float(depth_t[i].item())
                cl = float(cos_lat[i].item())
                lo = float(lon_t[i].item())
                return d_i * cl * math.sin(lo), d_i * cl * math.cos(lo)

            accept_half = math.radians(7.5)  # ±7.5° around each hour center
            # Uniform-hour sampling: fix target_hour up front, then search
            # (ia, ib, ic) triples until one lands within 7.5° of that hour.
            # Aggressive retry (60 outer pairs × 200 candidates) counters the
            # scene's natural front-heavy distribution for rare back hours.
            target_hour = rng.randint(1, 12)
            target_theta = (target_hour % 12) * (math.pi / 6)
            for _retry in range(20):
                patch_indices = None
                frame_indices = None
                for _outer in range(60):
                    ia, ib = rng.sample(range(scene.n_valid), 2)
                    if int(frame_index[ia].item()) == int(frame_index[ib].item()):
                        continue
                    ax, az = _xz(ia); bx, bz = _xz(ib)
                    hx, hz = bx - ax, bz - az
                    h_n = math.sqrt(hx * hx + hz * hz)
                    if h_n < 0.3:
                        continue
                    hx /= h_n; hz /= h_n

                    fa, fb = int(frame_index[ia].item()), int(frame_index[ib].item())
                    cands = [j for j in range(scene.n_valid)
                             if j != ia and j != ib
                             and int(frame_index[j].item()) not in (fa, fb)]
                    rng.shuffle(cands)
                    for ic in cands[:200]:
                        cx, cz = _xz(ic)
                        tx, tz = cx - ax, cz - az
                        t_n = math.sqrt(tx * tx + tz * tz)
                        if t_n < 0.3:
                            continue
                        tx /= t_n; tz /= t_n
                        dot = hx * tx + hz * tz
                        cross = hx * tz - hz * tx
                        theta_cw = math.atan2(-cross, dot)
                        theta_cw_pos = theta_cw if theta_cw >= 0 else theta_cw + 2 * math.pi
                        delta = abs(theta_cw_pos - target_theta)
                        delta = min(delta, 2 * math.pi - delta)
                        if delta < accept_half:
                            fc = int(frame_index[ic].item())
                            patch_indices = [ia, ib, ic]
                            frame_indices = [fa, fb, fc]
                            break
                    if patch_indices is not None:
                        break
                if patch_indices is None:
                    return None, None
                box_pis, _ = self._sample_multi_box(
                    scene, rng,
                    n_boxes=len(patch_indices),
                    patch_indices_hint=patch_indices,
                )
                if box_pis is None:
                    continue
                if _rel_dir_oclock_boxes_consistent(
                    scene._multi_box_centers,
                    scene._multi_box_dims,
                    scene._multi_box_rotations,
                ):
                    return patch_indices, frame_indices
            return None, None
        elif task == "rel_dir_count_side_box":
            return self._sample_rel_dir_count_side(scene, rng)
        elif task in (
            "route_plan_1turn_box", "route_plan_2turn_box",
            "route_plan_3turn_box", "route_plan_4turn_box",
        ):
            # route_plan_*_box: pure class-conditional box-path generation.
            # Boxes are placed freely in 3D (decoupled from scene surface
            # patches), with yaw-only rotation matching VSI gravity-aligned
            # convention. The sampled class per turn is the exact GT answer;
            # _build_question_and_answer reads canonical centers from
            # scene._route_plan_box_path_centers and cached classes from
            # scene._route_plan_box_classes.
            #
            # Reface is randomized internally (50/50) per sample — half the
            # samples get the VSI "start facing an off-path anchor" layout
            # (N+3 unique boxes, N+1 turns), half get the simple "start
            # facing wp1" layout (N+2 unique boxes, N turns). The Q&A
            # builder reads scene._route_plan_reface to pick the template.
            # Non-reface expansion: [start, wp1, goal, wp1, wp2, ..., wpN, goal]
            # Reface expansion:     [m0, m_face, goal, m1, m2, ..., wpN, goal]
            n_turns = int(task[len("route_plan_"):len("route_plan_") + 1])
            reface = rng.random() < 0.5
            if reface:
                centers, classes = self._sample_box_path_reface(scene, rng, n_turns)
            else:
                centers, classes = self._sample_box_path(scene, rng, n_turns)
            if centers is None:
                return None, None
            patch_indices, frame_indices = self._sample_multi_box(
                scene, rng,
                n_boxes=len(centers),
                center_hints=centers,
                rotation_mode="yaw",
                dim_hints=None,
            )
            if patch_indices is None:
                return None, None

            scene._route_plan_reface = reface
            scene._route_plan_box_classes = classes
            scene._route_plan_box_path_centers = list(scene._multi_box_centers)
            scene._route_plan_box_path_dims = list(scene._multi_box_dims)

            if reface:
                # unique: 0=m0, 1=m_face, 2..N+1=m1..m_N, N+2=goal.
                order = [0, 1, n_turns + 2] + list(range(2, n_turns + 3))
            else:
                # unique: 0=start, 1..N=wp1..wpN, N+1=goal.
                order = [0, 1, n_turns + 1] + list(range(1, n_turns + 2))
            scene._multi_box_centers = [scene._multi_box_centers[i] for i in order]
            scene._multi_box_dims = [scene._multi_box_dims[i] for i in order]
            scene._multi_box_rotations = [scene._multi_box_rotations[i] for i in order]
            scene._multi_box_spherical = [scene._multi_box_spherical[i] for i in order]
            scene._multi_box_feature_sources = [
                scene._multi_box_feature_sources[i] for i in order
            ]
            patch_indices = [patch_indices[i] for i in order]
            frame_indices = [frame_indices[i] for i in order]
            return patch_indices, frame_indices
        elif task == "box_floor_area":
            return self._sample_box_floor_area(scene, rng, rotation_mode=rotation_mode)
        elif task == "box_floor_area_nonrect":
            return self._sample_box_floor_area_nonrect(scene, rng, rotation_mode=rotation_mode)
        elif task == "box_floor_area_irregular":
            return self._sample_box_floor_area_irregular(scene, rng, rotation_mode=rotation_mode)
        elif task in (
            "object_counting",
            "object_counting_parity", "object_counting_parity_box",
            "object_counting_mod3", "object_counting_mod3_box",
        ):
            # parity / mod3 share the object_counting sampler — they differ only
            # in the Q&A reduction of scene._count.
            return self._sample_object_counting(scene, rng)
        elif task == "dist_box":
            return self._sample_dist_box(scene, rng)
        elif task == "rel_dist_box":
            return self._sample_rel_dist_box(scene, rng)
        elif task == "multi_box_grounding":
            # N ∈ {1,2,3,4} weighted toward 2-3 (matches Multi3DRefer / EmbodiedScan
            # target counts). _sample_multi_box enforces non-overlap; N=4 may fail
            # and caller retries via _get_valid_task_sample.
            n_boxes = rng.choices([1, 2, 3, 4], weights=[15, 45, 30, 10])[0]
            colinear_on = (
                self._colinear_centers_prob > 0.0
                and n_boxes >= 2
                and rng.random() < self._colinear_centers_prob
            )
            scene._grounding_colinear_on = bool(colinear_on)
            if colinear_on:
                centers = _draw_colinear_centers(
                    rng, n_boxes=n_boxes, scene=scene)
                if centers is not None:
                    return self._sample_multi_box(
                        scene, rng, n_boxes=n_boxes,
                        rotation_mode=rotation_mode,
                        center_hints=centers,
                    )
            return self._sample_multi_box(
                scene, rng, n_boxes=n_boxes, rotation_mode=rotation_mode,
            )
        elif task == "visibility_from_pose":
            return self._sample_visibility_from_pose(scene, rng)
        elif task in (
            "object_class_counting_real",
            "object_class_counting_parity_real",
            "object_class_counting_mod3_real",
        ):
            # parity / mod3 share the real-counting sampler — they differ only
            # in the QA reduction of scene._count (mirrors the synthetic
            # object_counting / parity_box / mod3_box dispatch above).
            if self._real_asset_bank is None:
                raise RuntimeError(
                    f"{task} requires --real_object_assets_enable True"
                )
            return self._sample_object_class_counting_real(scene, rng)
        elif task == "object_class_grounding_real":
            if self._real_asset_bank is None:
                raise RuntimeError(
                    "object_class_grounding_real requires --real_object_assets_enable True"
                )
            return self._sample_object_class_grounding_real(scene, rng)
        elif task == "object_class_appearance_order_real":
            if self._real_asset_bank is None:
                raise RuntimeError(
                    "object_class_appearance_order_real requires --real_object_assets_enable True"
                )
            return self._sample_object_class_appearance_order_real(scene, rng)
        elif task == "object_class_dist_real":
            if self._real_asset_bank is None:
                raise RuntimeError(
                    "object_class_dist_real requires --real_object_assets_enable True"
                )
            return self._sample_object_class_dist_real(scene, rng)
        elif task == "object_class_size_real":
            if self._real_asset_bank is None:
                raise RuntimeError(
                    "object_class_size_real requires --real_object_assets_enable True"
                )
            return self._sample_object_class_size_real(scene, rng)
        elif task == "object_class_rel_dist_real":
            if self._real_asset_bank is None:
                raise RuntimeError(
                    "object_class_rel_dist_real requires --real_object_assets_enable True"
                )
            return self._sample_object_class_rel_dist_real(scene, rng)
        elif task == "object_class_rel_dir_count_side_real":
            if self._real_asset_bank is None:
                raise RuntimeError(
                    "object_class_rel_dir_count_side_real requires --real_object_assets_enable True"
                )
            return self._sample_object_class_rel_dir_count_side_real(scene, rng)
        elif task in (
            "object_class_rel_dir_easy_real",
            "object_class_rel_dir_medium_real",
            "object_class_rel_dir_hard_real",
            "object_class_rel_dir_4way_real",
        ):
            if self._real_asset_bank is None:
                raise RuntimeError(
                    f"{task} requires --real_object_assets_enable True"
                )
            return self._sample_object_class_rel_dir_real(scene, rng, task)
        elif task in (
            "object_class_rel_dir_camera_easy_real",
            "object_class_rel_dir_camera_medium_real",
            "object_class_rel_dir_camera_hard_real",
        ):
            if self._real_asset_bank is None:
                raise RuntimeError(
                    f"{task} requires --real_object_assets_enable True"
                )
            return self._sample_object_class_rel_dir_camera_real(scene, rng, task)
        elif task == "object_class_rel_dir_oclock_real":
            if self._real_asset_bank is None:
                raise RuntimeError(
                    "object_class_rel_dir_oclock_real requires --real_object_assets_enable True"
                )
            return self._sample_object_class_rel_dir_oclock_real(scene, rng)
        elif task in (
            "object_class_route_plan_1turn_real",
            "object_class_route_plan_2turn_real",
            "object_class_route_plan_3turn_real",
            "object_class_route_plan_4turn_real",
        ):
            if self._real_asset_bank is None:
                raise RuntimeError(
                    f"{task} requires --real_object_assets_enable True"
                )
            n_turns = int(task[len("object_class_route_plan_"):
                               len("object_class_route_plan_") + 1])
            return self._sample_object_class_route_plan_real(
                scene, rng, n_turns,
            )
        elif task == "object_class_visibility_from_pose_real":
            if self._real_asset_bank is None:
                raise RuntimeError(
                    "object_class_visibility_from_pose_real requires --real_object_assets_enable True"
                )
            return self._sample_object_class_visibility_from_pose_real(
                scene, rng,
            )
        else:
            # Fall through to externally-registered tasks (see
            # curriculum_task_registry.py). Empty registry -> same ValueError as
            # before, so core behaviour is unchanged for built-in curricula.
            spec = get_probe_task(task)
            if spec is not None:
                if spec.requires_real_assets and self._real_asset_bank is None:
                    raise RuntimeError(
                        f"{task} requires --real_object_assets_enable True"
                    )
                return spec.sample(self, scene, rng)
            raise ValueError(f"Unknown probe task: {task}")
