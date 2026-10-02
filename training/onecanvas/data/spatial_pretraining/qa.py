"""QAMixin: spatial-pretraining samplers."""

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


class QAMixin:
    def _build_question_and_answer(self, task, scene, patch_indices, frame_indices, rng):
        """Generate question text + ground truth answer for the given task."""
        # Strip the "_obb_only" alias suffix first so the branches below see
        # the base task name. The suffix only affects canvas-stripping.
        task = _strip_display_suffix(task)
        # Strip rotation-mode suffix for box-family tasks so the existing
        # task-name branches below handle _aligned / _yaw / _random uniformly.
        # The sampler has already baked the mode-appropriate GT into scene._*.
        base_task, _rot_mode = _parse_box_task(task)
        if _rot_mode is not None:
            task = base_task
        # Route rel_dir_*_box to its patch counterpart's label branch. The
        # label depends only on the 3D centers of patch_indices, which equal
        # the box centers by construction (_sample_multi_box with
        # patch_indices_hint places each box at the referenced patch's
        # Cartesian position).
        if task in (
            "rel_dir_easy_box", "rel_dir_medium_box", "rel_dir_hard_box",
            "rel_dir_4way_box",
            "rel_dir_camera_easy_box", "rel_dir_camera_medium_box",
            "rel_dir_camera_hard_box",
            "rel_dir_oclock_box", "rel_dir_count_side_box",
            "object_counting_parity_box", "object_counting_mod3_box",
        ):
            task = task[:-len("_box")]
        # box_floor_area_nonrect (L/T-shaped room) and box_floor_area_irregular
        # (N in {2,3,4} rects, may be multi-component) share the question text
        # and GT attribute (_floor_area_m2) with the single-rect variant.
        if task in ("box_floor_area_nonrect", "box_floor_area_irregular"):
            task = "box_floor_area"
        # route_plan_*_box uses free-placement boxes (decoupled from patch
        # positions). Turn labels depend on the N+2 (or N+3 for reface)
        # UNIQUE path centers (scene._route_plan_box_path_centers), not the
        # expanded scene._multi_box_centers list which has duplicates to
        # match the VSI-style template placeholder count.
        is_route_plan_box = task in (
            "route_plan_1turn_box", "route_plan_2turn_box",
            "route_plan_3turn_box", "route_plan_4turn_box",
        )
        is_route_plan_real = task in (
            "object_class_route_plan_1turn_real",
            "object_class_route_plan_2turn_real",
            "object_class_route_plan_3turn_real",
            "object_class_route_plan_4turn_real",
        )
        if is_route_plan_box or is_route_plan_real:
            pts = torch.stack(scene._route_plan_box_path_centers, dim=0)  # [N+2 (or N+3 for waypoint reface), 3]
            if is_route_plan_box:
                task = task[:-len("_box")]
        else:
            # Reconstruct world-centered 3D points for the sampled patches.
            # frame_index_x_c convention follows scene_reprojection.py:76-84.
            lat = scene.latitude[patch_indices]                # [K]
            lon = scene.longitude[patch_indices]               # [K]
            d = scene.depth[patch_indices]                     # [K]
            cos_lat = torch.cos(lat)
            pts = torch.stack([
                d * cos_lat * torch.sin(lon),   # x_c
                d * torch.sin(lat),              # y_c
                d * cos_lat * torch.cos(lon),    # z_c
            ], dim=-1)                                         # [K, 3]

        df = self._dist_decimals  # shorthand for format strings

        if task in ("appearance_order", "appearance_order_box"):
            # N-patch temporal sorting. Presents N markers (canonical slot order
            # p1..pN) and asks for their first-appearance order in the video.
            # For appearance_order the frame_indices are the local-min-T (over
            # a 3D neighbourhood) per marker; for appearance_order_box they are
            # the box's assigned T_min.
            #
            # Two answer formats:
            #   - MCQ: 4-way letter, legacy VSI-Bench Appr.Order protocol
            #     (correct + 3 random-permutation distractors). Chance = 25%
            #     regardless of N.
            #   - Open-ended: concatenated digit string "2413" meaning
            #     p2<p4<p1<p3. Chance = 1/N! (24x harder at N=4, 120x at N=5).
            N = pts.shape[0]
            assert N >= 2
            slot_labels = [f"p{i+1}" for i in range(N)]
            sorted_slots = sorted(range(N), key=lambda k: frame_indices[k])
            marker_list = ", ".join(
                f"p{i+1}={_INLINE_PATCH_PLACEHOLDER}" for i in range(N)
            )
            if self._appearance_open_ended:
                # Digit-string permutation. Fits vsibench_fuzzy_match's
                # first-token exact-match protocol (single bare token, no
                # punctuation). Digits are 1-indexed to match slot labels.
                a = "".join(str(k + 1) for k in sorted_slots)
                q = (
                    f"What is the first-time appearance order of the following patches "
                    f"in the video: {marker_list}? "
                    f"Answer as a single string of digits (1-indexed), earliest first. "
                    f"Example for {N} patches: '{''.join(str(i+1) for i in range(N))}' "
                    f"means p1 first, then p2, ..., then p{N}."
                )
            else:
                correct_perm = ", ".join(slot_labels[k] for k in sorted_slots)
                seen = {tuple(sorted_slots)}
                distractors = []
                guard = 0
                while len(distractors) < 3 and guard < 200:
                    guard += 1
                    perm = list(range(N))
                    rng.shuffle(perm)
                    if tuple(perm) in seen:
                        continue
                    seen.add(tuple(perm))
                    distractors.append(", ".join(slot_labels[k] for k in perm))
                letter, options = _mcq_shuffle(correct_perm, distractors, rng)
                q = (
                    f"What is the first-time appearance order of the following patches "
                    f"in the video: {marker_list}? " + "  ".join(options)
                )
                a = letter
        elif task in ("route_plan_1turn", "route_plan_2turn",
                      "route_plan_3turn", "route_plan_4turn"):
            # N-turn navigation MCQ. Two variants:
            #  (a) base: markers = [m0 (start), m1..m_N (waypoints), m_{N+1}
            #      (goal)], N+2 total. Robot walks m0→m1→...→m_{N+1}; initial
            #      heading is implicitly toward m1 (facing==wp1 convention).
            #      Step 1 is "Go forward until <m1>"; turn k is at m_k,
            #      computed as _horizontal_turn(m_{k-1}, m_k, m_{k+1}).
            #  (b) reface: markers = [m0, m_face, m1..m_N, m_{N+1}], N+3 total.
            #      Robot starts at m0 FACING m_face (m_face is off the path).
            #      Step 1 is [please fill in] — the start-turn reorienting
            #      from facing m_face to heading toward m1 — so the answer has
            #      N+1 turns. Mirrors VSI Route Plan phrasing where the
            #      facing object is not the first forward destination.
            #
            # Answer vocabulary per turn:
            #   reface start-turn (at m0): {Turn Left, Turn Right, Turn Back}
            #     — robot at m0 facing m_face, reorients to head toward m1.
            #     Back is semantically clean here (standing still, turn around).
            #   all path-turns (base turn at m1..m_N, reface path-turn at
            #     m1..m_N): {Turn Left, Turn Right} — any path-turn comes
            #     *after* a walked segment, so a Back would send the robot
            #     back through (or just past) a prior waypoint. Base has no
            #     start-turn slot, so Back never appears in base route_plans.
            # MCQ distractors sample from the same restricted space to avoid
            # a "Back only ever appears at the start-turn" shortcut.
            #
            # Class-conditional box-path sampler pre-commits each turn's
            # class; we cross-check that the horizontal-plane classifier
            # agrees (catches bugs in the sampler / classifier thresholds).
            # Reface can be indicated either by the task name (legacy
            # task names like route_plan_Nturn_reface_box) OR by the
            # scene-stashed flag set by the unified dispatcher.
            is_reface = task.endswith("_reface") or bool(
                getattr(scene, "_route_plan_reface", False)
            )
            base_rp = task[:-len("_reface")] if task.endswith("_reface") else task
            n_turns = {"route_plan_1turn": 1, "route_plan_2turn": 2,
                       "route_plan_3turn": 3, "route_plan_4turn": 4}[base_rp]
            if is_reface:
                assert pts.shape[0] == n_turns + 3
                # pts = [m0, m_face, m1, ..., m_{N+1}]. For the start-turn,
                # incoming heading is (m_face - m0); pass a virtual p_prev
                # (2*m0 - m_face) so _horizontal_turn's (p_here - p_prev)
                # recovers that heading.
                start_turn = _horizontal_turn(
                    2.0 * pts[0] - pts[1], pts[0], pts[2]
                )
                # Path-turns run along the walked path m0 → m1 → ... → m_{N+1},
                # skipping m_face (which is off-path). Build the walked-path
                # sequence explicitly so turn-k at m_k reads (m_{k-1}, m_k, m_{k+1}).
                path_seq = [pts[0]] + [pts[i] for i in range(2, n_turns + 3)]
                path_turns = [
                    _horizontal_turn(path_seq[i], path_seq[i + 1], path_seq[i + 2])
                    for i in range(n_turns)
                ]
                turns = [start_turn] + path_turns
            else:
                assert pts.shape[0] == n_turns + 2
                turns = [_horizontal_turn(pts[i], pts[i + 1], pts[i + 2])
                         for i in range(n_turns)]
            cached = getattr(scene, "_route_plan_box_classes", None)
            if cached is not None and len(cached) == len(turns):
                for k, (got, want) in enumerate(zip(turns, cached)):
                    assert got == want, (
                        f"turn {k+1}: sampled {want!r} != classifier {got!r}"
                    )
                turns = list(cached)
            total_turns = len(turns)
            correct = ", ".join(turns) if total_turns > 1 else turns[0]

            def _sample_turn_sequence():
                if is_reface:
                    # start-turn (3-way) + all path-turns (2-way)
                    seq = [rng.choice(["Turn Left", "Turn Right", "Turn Back"])]
                    for _ in range(n_turns):
                        seq.append(rng.choice(["Turn Left", "Turn Right"]))
                else:
                    # all path-turns (2-way)
                    seq = [rng.choice(["Turn Left", "Turn Right"])
                           for _ in range(n_turns)]
                return ", ".join(seq) if len(seq) > 1 else seq[0]

            # Non-reface 1-turn has only {L, R} in the restricted space, so
            # at most 1 valid distractor. All other configurations support 3.
            if total_turns == 1 and not is_reface:
                n_distractors = 1
            else:
                n_distractors = 2 if total_turns == 1 else 3
            seen = {correct}
            distractors = []
            attempts = 0
            while len(distractors) < n_distractors and attempts < 80:
                attempts += 1
                cand = _sample_turn_sequence()
                if cand in seen:
                    continue
                seen.add(cand)
                distractors.append(cand)
            # Fallback: for 1-turn, pull from the remaining class in the
            # restricted space; for multi-turn, make any unseen valid sequence.
            if total_turns == 1:
                classes = (("Turn Left", "Turn Right", "Turn Back") if is_reface
                           else ("Turn Left", "Turn Right"))
                while len(distractors) < n_distractors:
                    added = False
                    for t in classes:
                        if t not in seen:
                            seen.add(t)
                            distractors.append(t)
                            added = True
                            break
                    if not added:
                        break
            else:
                while len(distractors) < n_distractors:
                    fallback = ", ".join(
                        ["Turn Left"] + ["Turn Right"] * (total_turns - 1)
                    )
                    if fallback in seen:
                        fallback = ", ".join(
                            ["Turn Right"] + ["Turn Left"] * (total_turns - 1)
                        )
                    seen.add(fallback)
                    distractors.append(fallback)
            letter, options = _mcq_shuffle(correct, distractors, rng)
            # VSI-style question. Placeholder order matches the expanded
            # patch_indices layout from _sample_patches:
            #   base:   [m0, m1, goal, m1, m2, ..., goal]        (N+4 slots, N+2 unique)
            #   reface: [m0, m_face, goal, m1, m2, ..., goal]    (N+4 slots, N+3 unique)
            # Template mirrors real VSI Route Plan phrasing.
            if is_reface:
                # alternating [please fill in] / Go forward, starting with
                # fill-in; 2*(n_turns+1) steps total.
                steps = []
                for k in range(n_turns + 1):
                    steps.append(f"{2*k+1}. [please fill in].")
                    if k < n_turns:
                        steps.append(
                            f"{2*k+2}. Go forward until {_INLINE_PATCH_PLACEHOLDER}."
                        )
                    else:
                        steps.append(
                            f"{2*k+2}. Go forward until {_INLINE_PATCH_PLACEHOLDER}. "
                            f"You have reached the final destination."
                        )
            else:
                steps = [
                    f"1. Go forward until {_INLINE_PATCH_PLACEHOLDER}."
                ]
                for k in range(1, n_turns):
                    steps.append(f"{2*k}. [please fill in].")
                    steps.append(f"{2*k+1}. Go forward until {_INLINE_PATCH_PLACEHOLDER}.")
                steps.append(f"{2*n_turns}. [please fill in].")
                steps.append(
                    f"{2*n_turns+1}. Go forward until {_INLINE_PATCH_PLACEHOLDER}. "
                    f"You have reached the final destination."
                )
            q = (
                f"You are a robot beginning at {_INLINE_PATCH_PLACEHOLDER} "
                f"and facing {_INLINE_PATCH_PLACEHOLDER}. You want to "
                f"navigate to {_INLINE_PATCH_PLACEHOLDER}. You will perform "
                f"the following actions (Note: for each [please fill in], "
                f"choose either 'turn back,' 'turn left,' or 'turn right.'): "
                + " ".join(steps) + " "
                + "  ".join(options)
            )
            a = letter
        elif task in ("rel_dir_easy", "rel_dir_medium", "rel_dir_hard", "rel_dir_4way"):
            # Egocentric direction MCQ. 3 patches: ref (viewpoint), fwd (heading
            # anchor), tgt (asked-about). Compute horizontal-plane heading
            # (fwd - ref) and target direction (tgt - ref), then derive
            # left/right/back/front labels. Matches VSI Rel.Dir phrasing.
            assert pts.shape[0] == 3
            label = _rel_dir_label(pts[0], pts[1], pts[2], task)
            if task == "rel_dir_easy":
                distractor_pool = ["left", "right"]
            elif task == "rel_dir_medium":
                distractor_pool = ["left", "right", "back"]
            elif task == "rel_dir_4way":
                distractor_pool = ["front", "back", "left", "right"]
            else:
                distractor_pool = ["front-left", "front-right", "back-left", "back-right"]
            distractors = [x for x in distractor_pool if x != label]
            letter, options = _mcq_shuffle(label, distractors, rng)
            if task == "rel_dir_easy":
                question_tail = "is the target to the left or the right? "
            elif task == "rel_dir_medium":
                question_tail = "is the target to the back, left, or right? "
            elif task == "rel_dir_4way":
                question_tail = "is the target to the front, back, left, or right? "
            else:
                question_tail = (
                    "is the target to the front-left, front-right, back-left, "
                    "or back-right? "
                )
            q = (
                f"If you are standing at {_INLINE_PATCH_PLACEHOLDER} and "
                f"facing {_INLINE_PATCH_PLACEHOLDER}, with the target at "
                f"{_INLINE_PATCH_PLACEHOLDER}, "
                + question_tail
                + "  ".join(options)
            )
            a = letter
        elif task in ("rel_dir_camera_easy", "rel_dir_camera_medium", "rel_dir_camera_hard"):
            # SPBench-SI phrasing: "From the camera's perspective, is <target>
            # to <pivot>'s <direction>?". The sampler returns
            # patch_indices = [target, pivot] so marker substitution places the
            # target placeholder first (matches the SPBench word order).
            # pts[0] = target, pts[1] = pivot. The canvas has been reoriented
            # onto the chosen camera so "front/back/left/right" = the camera's
            # axes (origin + canvas-forward stored on scene._rel_dir_camera_*).
            assert pts.shape[0] == 2
            base = task.replace("rel_dir_camera_", "rel_dir_")
            p_ref = pts[1]                                    # pivot
            p_fwd = p_ref + scene._rel_dir_camera_fwd         # camera forward from pivot
            label = _rel_dir_label(p_ref, p_fwd, pts[0], base)
            if base == "rel_dir_easy":
                distractor_pool = ["left", "right"]
            elif base == "rel_dir_medium":
                distractor_pool = ["left", "right", "back"]
            else:
                distractor_pool = ["front-left", "front-right", "back-left", "back-right"]
            distractors = [x for x in distractor_pool if x != label]
            letter, options = _mcq_shuffle(label, distractors, rng)
            if base == "rel_dir_easy":
                tail = "left or right? "
            elif base == "rel_dir_medium":
                tail = "left, right, or back? "
            else:
                tail = "front-left, front-right, back-left, or back-right? "
            q = (
                f"From the camera's perspective, is "
                f"{_INLINE_PATCH_PLACEHOLDER} to "
                f"{_INLINE_PATCH_PLACEHOLDER}'s "
                + tail
                + "  ".join(options)
            )
            a = letter
        elif task == "rel_dir_oclock":
            # Situated o'clock direction (SQA3D phrasing, e.g. "couch in my
            # about eleven o'clock direction"). 3 markers: ref, fwd, tgt.
            # Hour is the clockwise angle from forward, snapped to 1..12.
            assert pts.shape[0] == 3
            hx = float(pts[1, 0].item() - pts[0, 0].item())
            hz = float(pts[1, 2].item() - pts[0, 2].item())
            tx = float(pts[2, 0].item() - pts[0, 0].item())
            tz = float(pts[2, 2].item() - pts[0, 2].item())
            h_n = math.sqrt(hx * hx + hz * hz)
            t_n = math.sqrt(tx * tx + tz * tz)
            hx /= h_n; hz /= h_n; tx /= t_n; tz /= t_n
            hour = _ego_hour(hx * tx + hz * tz, hx * tz - hz * tx)
            q = (
                f"If you are standing at {_INLINE_PATCH_PLACEHOLDER} and "
                f"facing {_INLINE_PATCH_PLACEHOLDER}, in what o'clock "
                f"direction is {_INLINE_PATCH_PLACEHOLDER}? Answer with an "
                f"hour from 1 to 12 (e.g. \"3 o'clock\")."
            )
            a = f"{hour} o'clock"
        elif task == "rel_dir_count_side":
            # Situated count on one cardinal side (SQA3D phrasing, e.g.
            # "How many chairs are to the right?"). 3 markers: ref, fwd, and
            # the shared-class feature source; N synthetic boxes at sampled
            # positions (not referenced by markers) are counted.
            assert pts.shape[0] == 3
            side = scene._count_side_label
            side_tail = {
                "right": "are on my right",
                "left":  "are on my left",
                "front": "are in front of me",
                "back":  "are behind me",
            }[side]
            q = (
                f"I am standing at {_INLINE_PATCH_PLACEHOLDER} and facing "
                f"{_INLINE_PATCH_PLACEHOLDER}. How many "
                f"{_INLINE_PATCH_PLACEHOLDER} {side_tail}? "
                f"Answer with a single integer."
            )
            a = f"{scene._count_side_answer}"
        elif task == "object_counting_parity":
            q = (
                f"Is the number of {_INLINE_PATCH_PLACEHOLDER} in the room "
                f"odd or even?"
            )
            a = "even" if scene._count % 2 == 0 else "odd"
        elif task == "object_counting_mod3":
            q = (
                f"Is the number of {_INLINE_PATCH_PLACEHOLDER} in the room "
                f"a multiple of three?"
            )
            a = "yes" if scene._count % 3 == 0 else "no"
        elif task == "box_floor_area":
            area = scene._floor_area_m2
            q = (
                f"What is the size of this room in square meters? "
                f"If multiple rooms are shown, estimate the size of "
                f"the combined space."
            )
            a = f"{area:.1f}"
        elif task == "object_counting":
            q = f"How many {_INLINE_PATCH_PLACEHOLDER} are in the room?"
            a = f"{scene._count}"
        elif task == "dist_box":
            q = (
                f"Measuring from the closest point of each object, what is the "
                f"distance between {_INLINE_PATCH_PLACEHOLDER} and "
                f"{_INLINE_PATCH_PLACEHOLDER} (in meters)?"
            )
            a = f"{scene._dist_box_value:.{self._dist_decimals}f}"
        elif task == "rel_dist_box":
            q = (
                f"Measuring from the closest point of each object, which of these "
                f"objects is the closest to {_INLINE_PATCH_PLACEHOLDER}? "
                f"A. {_INLINE_PATCH_PLACEHOLDER}  "
                f"B. {_INLINE_PATCH_PLACEHOLDER}  "
                f"C. {_INLINE_PATCH_PLACEHOLDER}  "
                f"D. {_INLINE_PATCH_PLACEHOLDER}"
            )
            a = scene._rel_dist_box_answer
        elif task == "visibility_from_pose":
            q = (
                f"You are standing at {_INLINE_PATCH_PLACEHOLDER}. "
                f"Can you see {_INLINE_PATCH_PLACEHOLDER} from this position? "
                f"Answer yes or no."
            )
            a = scene._visibility_answer
        elif task == "multi_box_grounding":
            # Predict the axis-aligned 3D bounding box around each painted OBB
            # marker. For rotation_mode=aligned the AABB == OBB; for yaw the
            # AABB is the looser axis-aligned envelope over the 8 rotated
            # corners. Output format matches the grounding training target
            # (Qwen3-VL bbox_3d JSON or legacy <|box_start|>(...)<|box_end|>).
            #
            # Frame conversion: scene._multi_box_centers live in the panorama
            # intermediate frame (x_c, y_c, z_c) = (rotated_X, -Z, rotated_Y);
            # real grounding emits bbox_3d in the aug-centered, yaw-rotated
            # world frame (rotated_X, rotated_Y, Z). Invert the permutation
            # from scene_reprojection.py:99-101 so the probe answer lives in
            # the same coord frame the model is trained to emit for
            # scanrefer/multi3drefer/nr3d/sr3d.
            n = len(scene._multi_box_centers)
            # Sort near-to-far by distance from canvas origin to match the
            # canonical ordering used by multi3drefer/scanrefer/nr3d/sr3d
            # (see data_processor_3d.py:2115). Reorder all five _multi_box_*
            # lists together so marker tokens in the text stay bound to the
            # right source patches (same pattern as route_plan at L3651).
            order = sorted(range(n), key=lambda i: float((scene._multi_box_centers[i] ** 2).sum()))
            scene._multi_box_centers         = [scene._multi_box_centers[i]         for i in order]
            scene._multi_box_dims            = [scene._multi_box_dims[i]            for i in order]
            scene._multi_box_rotations       = [scene._multi_box_rotations[i]       for i in order]
            scene._multi_box_feature_sources = [scene._multi_box_feature_sources[i] for i in order]
            scene._multi_box_spherical       = [scene._multi_box_spherical[i]       for i in order]
            labels = ["A", "B", "C", "D"][:n]
            parts = [f"{lab}: {_INLINE_PATCH_PLACEHOLDER}" for lab in labels]
            q = (
                "Provide the 3D bounding box of each highlighted region, "
                "in the same order: " + ", ".join(parts) + "."
            )
            centered_boxes = []
            for c, d, R in zip(
                scene._multi_box_centers,
                scene._multi_box_dims,
                scene._multi_box_rotations,
            ):
                corners = _obb_world_corners(c, d, R)
                mn = corners.amin(dim=0)
                mx = corners.amax(dim=0)
                cc = (mn + mx) / 2
                dd = mx - mn
                # Permute intermediate (x_c, y_c, z_c) -> world (X, Y, Z):
                #   world.X = x_c; world.Y = z_c; world.Z = -y_c.
                # Dims are non-negative extents, so the Z dim picks |y_c|.
                centered_boxes.append((
                    float(cc[0]), float(cc[2]), float(-cc[1]),
                    float(dd[0]), float(dd[2]), float(dd[1]),
                ))
            if self._metric_json_grounding_format:
                a = format_multi_metric_bbox_json(centered_boxes, decimals=2)
            else:
                a = format_multi_metric_bbox(centered_boxes, decimals=2)
        elif task == "object_class_counting_real":
            label = scene._counting_class_label
            plural = _PLURAL_LABELS.get(label, label + "s")
            q = f"How many {plural} are visible in this scene?"
            a = f"{scene._count}"
        elif task == "object_class_counting_parity_real":
            label = scene._counting_class_label
            plural = _PLURAL_LABELS.get(label, label + "s")
            q = f"Is the number of {plural} visible in this scene odd or even?"
            a = "even" if scene._count % 2 == 0 else "odd"
        elif task == "object_class_counting_mod3_real":
            label = scene._counting_class_label
            plural = _PLURAL_LABELS.get(label, label + "s")
            q = f"Is the number of {plural} visible in this scene a multiple of three?"
            a = "yes" if scene._count % 3 == 0 else "no"
        elif task == "object_class_grounding_real":
            # Real-asset twin of multi_box_grounding: one axis-aligned box per
            # target-class instance, near-to-far from the canvas origin, same
            # intermediate -> world permutation (world.X = x_c, world.Y = z_c,
            # world.Z = -y_c) and same answer format. The referent is the
            # class name, no inline marker, so the prompt cannot label the
            # instances A, B, C the way the synthetic one does.
            label = scene._grounding_real_target_label
            plural = _PLURAL_LABELS.get(label, label + "s")
            centered_boxes = []
            for pts in scene._grounding_real_target_world_pts_list:
                mn = pts.amin(dim=0)
                mx = pts.amax(dim=0)
                cc = (mn + mx) / 2.0
                dd = mx - mn
                centered_boxes.append((
                    float(cc[0]), float(cc[2]), float(-cc[1]),
                    float(dd[0]), float(dd[2]), float(dd[1]),
                ))
            centered_boxes.sort(key=lambda b: b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
            q = (
                f"Find every {label} in the scene. Provide the 3D bounding box "
                f"of each in near-to-far order. (There may be one or several "
                f"{plural}.)"
            )
            if self._metric_json_grounding_format:
                a = format_multi_metric_bbox_json(centered_boxes, decimals=2)
            else:
                a = format_multi_metric_bbox(centered_boxes, decimals=2)
        elif task == "object_class_appearance_order_real":
            # 4-way MCQ matching the synthetic appearance_order_box format:
            # correct full-permutation + 3 random non-GT permutations,
            # shuffled into A/B/C/D options, letter answer.
            classes = scene._appearance_order_real_classes
            gt_order = scene._appearance_order_real_gt_order
            correct_perm = ", ".join(gt_order)
            seen = {tuple(gt_order)}
            distractors = []
            guard = 0
            while len(distractors) < 3 and guard < 200:
                guard += 1
                p = list(gt_order)
                rng.shuffle(p)
                if tuple(p) in seen:
                    continue
                seen.add(tuple(p))
                distractors.append(", ".join(p))
            letter, options = _mcq_shuffle(correct_perm, distractors, rng)
            object_list = ", ".join(classes)
            q = (
                f"What is the first-time appearance order of these objects "
                f"in the video: {object_list}? " + "  ".join(options)
            )
            a = letter
        elif task == "object_class_dist_real":
            # Real-asset version of dist_box. No inline markers — both
            # objects are referenced by class name.
            la = scene._dist_real_label_a
            lb = scene._dist_real_label_b
            q = (
                f"Measuring from the closest point of each object, what is "
                f"the distance between the {la} and the {lb} (in meters)?"
            )
            a = f"{scene._dist_real_value:.{self._dist_decimals}f}"
        elif task == "object_class_size_real":
            # Real-asset version of VSI-Bench object_size_estimation.
            # Question is verbatim from VSI's lmms_eval prompts; answer is
            # a bare integer cm matching round(max(axesLengths) * 100).
            label = scene._size_real_label
            q = (
                f"What is the length of the longest dimension (length, "
                f"width, or height) of the {label}, measured in centimeters?"
            )
            a = str(scene._size_real_value)
        elif task == "object_class_rel_dist_real":
            # Real-asset version of rel_dist_box (4-way MCQ over candidate
            # classes). Letter answer aligns with the order the candidates
            # were sampled in (A = first cand, ..., D = fourth cand).
            tgt = scene._rel_dist_real_target_label
            cands = scene._rel_dist_real_cand_labels
            q = (
                f"Measuring from the closest point of each object, which of "
                f"these objects ({cands[0]}, {cands[1]}, {cands[2]}, {cands[3]}) "
                f"is the closest to the {tgt}?\n"
                f"A. {cands[0]}\n"
                f"B. {cands[1]}\n"
                f"C. {cands[2]}\n"
                f"D. {cands[3]}"
            )
            a = scene._rel_dist_real_answer
        elif task == "object_class_rel_dir_count_side_real":
            # Real-asset version of rel_dir_count_side_box. Fully real:
            # ref + fwd anchors and counted instances all referenced by
            # class name (no inline markers).
            side = scene._count_side_real_label
            side_tail = {
                "right": "are on my right",
                "left":  "are on my left",
                "front": "are in front of me",
                "back":  "are behind me",
            }[side]
            label = scene._count_side_real_class_label
            plural = _PLURAL_LABELS.get(label, label + "s")
            ref_label = scene._count_side_real_label_ref
            fwd_label = scene._count_side_real_label_fwd
            q = (
                f"I am standing at the {ref_label} and facing the "
                f"{fwd_label}. How many {plural} {side_tail}? "
                f"Answer with a single integer."
            )
            a = f"{scene._count_side_real_answer}"
        elif task in ("object_class_rel_dir_easy_real",
                      "object_class_rel_dir_medium_real",
                      "object_class_rel_dir_hard_real",
                      "object_class_rel_dir_4way_real"):
            # Real-asset version of rel_dir_{easy,medium,hard,4way}_box.
            # All three objects (ref / fwd / tgt) referenced by class name.
            label = scene._rel_dir_real_label
            base = scene._rel_dir_real_task
            la = scene._rel_dir_real_label_ref
            lb = scene._rel_dir_real_label_fwd
            lc = scene._rel_dir_real_label_tgt
            if base == "rel_dir_easy":
                distractor_pool = ["left", "right"]
                tail = f"is the {lc} to the left or the right? "
            elif base == "rel_dir_medium":
                distractor_pool = ["left", "right", "back"]
                tail = f"is the {lc} to the back, left, or right? "
            elif base == "rel_dir_4way":
                distractor_pool = ["front", "back", "left", "right"]
                tail = (
                    f"is the {lc} to the front, back, left, or right? "
                )
            else:
                distractor_pool = ["front-left", "front-right",
                                   "back-left", "back-right"]
                tail = (
                    f"is the {lc} to the front-left, front-right, "
                    f"back-left, or back-right? "
                )
            distractors = [x for x in distractor_pool if x != label]
            letter, options = _mcq_shuffle(label, distractors, rng)
            q = (
                f"If you are standing at the {la} and facing the {lb}, "
                + tail + "  ".join(options)
            )
            a = letter
        elif task in ("object_class_rel_dir_camera_easy_real",
                      "object_class_rel_dir_camera_medium_real",
                      "object_class_rel_dir_camera_hard_real"):
            # Real-asset version of rel_dir_camera_{easy,medium,hard}_box.
            # SPBench-SI phrasing using class names for pivot + target.
            label = scene._rel_dir_camera_real_label
            base = scene._rel_dir_camera_real_task
            l_piv = scene._rel_dir_camera_real_label_pivot
            l_tgt = scene._rel_dir_camera_real_label_target
            if base == "rel_dir_easy":
                distractor_pool = ["left", "right"]
                tail = "left or right? "
            elif base == "rel_dir_medium":
                distractor_pool = ["left", "right", "back"]
                tail = "left, right, or back? "
            else:
                distractor_pool = ["front-left", "front-right",
                                   "back-left", "back-right"]
                tail = (
                    "front-left, front-right, back-left, or back-right? "
                )
            distractors = [x for x in distractor_pool if x != label]
            letter, options = _mcq_shuffle(label, distractors, rng)
            q = (
                f"From the camera's perspective, is the {l_tgt} to the "
                f"{l_piv}'s " + tail + "  ".join(options)
            )
            a = letter
        elif task == "object_class_rel_dir_oclock_real":
            # Real-asset version of rel_dir_oclock_box (SQA3D situated).
            la = scene._rel_dir_oclock_real_label_ref
            lb = scene._rel_dir_oclock_real_label_fwd
            lc = scene._rel_dir_oclock_real_label_tgt
            hour = int(scene._rel_dir_oclock_real_hour)
            q = (
                f"If you are standing at the {la} and facing the {lb}, in "
                f"what o'clock direction is the {lc}? Answer with an hour "
                f"from 1 to 12 (e.g. \"3 o'clock\")."
            )
            a = f"{hour} o'clock"
        elif task in (
            "object_class_route_plan_1turn_real",
            "object_class_route_plan_2turn_real",
            "object_class_route_plan_3turn_real",
            "object_class_route_plan_4turn_real",
        ):
            # Real-asset version of route_plan_*_box. Waypoints carry class
            # names instead of inline markers; turn-class GT is computed from
            # the same path centers using _horizontal_turn (matches synthetic
            # variant). Reface flag stashed on scene by the sampler.
            n_turns = int(scene._route_plan_real_n_turns)
            is_reface = bool(scene._route_plan_real_reface)
            wp_labels = list(scene._route_plan_real_waypoint_labels)
            if is_reface:
                # pts = [m0, m_face, m1, ..., m_{N+1}].
                start_turn = _horizontal_turn(
                    2.0 * pts[0] - pts[1], pts[0], pts[2]
                )
                path_seq = [pts[0]] + [pts[i] for i in range(2, n_turns + 3)]
                path_turns = [
                    _horizontal_turn(path_seq[i], path_seq[i + 1],
                                     path_seq[i + 2])
                    for i in range(n_turns)
                ]
                turns = [start_turn] + path_turns
            else:
                turns = [
                    _horizontal_turn(pts[i], pts[i + 1], pts[i + 2])
                    for i in range(n_turns)
                ]
            cached = list(scene._route_plan_real_classes)
            for k, (got, want) in enumerate(zip(turns, cached)):
                assert got == want, (
                    f"route_plan_real turn {k+1}: classifier {got!r} != "
                    f"sampler {want!r}"
                )
            turns = list(cached)
            total_turns = len(turns)
            correct = ", ".join(turns) if total_turns > 1 else turns[0]

            def _sample_turn_sequence_real():
                if is_reface:
                    seq = [rng.choice(["Turn Left", "Turn Right",
                                       "Turn Back"])]
                    for _ in range(n_turns):
                        seq.append(rng.choice(["Turn Left", "Turn Right"]))
                else:
                    seq = [rng.choice(["Turn Left", "Turn Right"])
                           for _ in range(n_turns)]
                return ", ".join(seq) if len(seq) > 1 else seq[0]

            if total_turns == 1 and not is_reface:
                n_distractors = 1
            else:
                n_distractors = 2 if total_turns == 1 else 3
            seen = {correct}
            distractors = []
            attempts = 0
            while len(distractors) < n_distractors and attempts < 80:
                attempts += 1
                cand = _sample_turn_sequence_real()
                if cand in seen:
                    continue
                seen.add(cand)
                distractors.append(cand)
            if total_turns == 1:
                classes = (("Turn Left", "Turn Right", "Turn Back")
                           if is_reface else ("Turn Left", "Turn Right"))
                while len(distractors) < n_distractors:
                    added = False
                    for t in classes:
                        if t not in seen:
                            seen.add(t)
                            distractors.append(t)
                            added = True
                            break
                    if not added:
                        break
            else:
                while len(distractors) < n_distractors:
                    fallback = ", ".join(
                        ["Turn Left"] + ["Turn Right"] * (total_turns - 1)
                    )
                    if fallback in seen:
                        fallback = ", ".join(
                            ["Turn Right"]
                            + ["Turn Left"] * (total_turns - 1)
                        )
                    seen.add(fallback)
                    distractors.append(fallback)
            letter, options = _mcq_shuffle(correct, distractors, rng)
            if is_reface:
                # Reface: wp_labels = [m0, m_face, m1, ..., m_{N+1}].
                # Walked path = [m0, m1, ..., m_{N+1}], indices 0, 2..N+2.
                start_label = wp_labels[0]
                facing_label = wp_labels[1]
                walked_labels = [wp_labels[0]] + [
                    wp_labels[i] for i in range(2, n_turns + 3)
                ]
                goal_label = walked_labels[-1]
                steps = []
                for k in range(n_turns + 1):
                    steps.append(f"{2*k+1}. [please fill in].")
                    if k < n_turns:
                        steps.append(
                            f"{2*k+2}. Go forward until the "
                            f"{walked_labels[k + 1]}."
                        )
                    else:
                        steps.append(
                            f"{2*k+2}. Go forward until the "
                            f"{walked_labels[k + 1]}. You have reached "
                            f"the final destination."
                        )
                q = (
                    f"You are a robot beginning at the {start_label} and "
                    f"facing the {facing_label}. You want to navigate to "
                    f"the {goal_label}. You will perform the following "
                    f"actions (Note: for each [please fill in], choose "
                    f"either 'turn back,' 'turn left,' or 'turn right.'): "
                    + " ".join(steps) + " "
                    + "  ".join(options)
                )
            else:
                # Non-reface: wp_labels = [m0, m1, ..., m_{N+1}].
                start_label = wp_labels[0]
                facing_label = wp_labels[1]
                goal_label = wp_labels[-1]
                steps = [
                    f"1. Go forward until the {wp_labels[1]}."
                ]
                for k in range(1, n_turns):
                    steps.append(f"{2*k}. [please fill in].")
                    steps.append(
                        f"{2*k+1}. Go forward until the {wp_labels[k + 1]}."
                    )
                steps.append(f"{2*n_turns}. [please fill in].")
                steps.append(
                    f"{2*n_turns+1}. Go forward until the "
                    f"{wp_labels[-1]}. You have reached the final "
                    f"destination."
                )
                q = (
                    f"You are a robot beginning at the {start_label} and "
                    f"facing the {facing_label}. You want to navigate to "
                    f"the {goal_label}. You will perform the following "
                    f"actions (Note: for each [please fill in], choose "
                    f"either 'turn back,' 'turn left,' or 'turn right.'): "
                    + " ".join(steps) + " "
                    + "  ".join(options)
                )
            a = letter
        elif task == "object_class_visibility_from_pose_real":
            # Real-asset visibility: viewer + target by class name; occluder
            # stays synthetic (real scene objects bias the prior toward "yes").
            l_view = scene._visibility_real_label_viewer
            l_tgt = scene._visibility_real_label_target
            q = (
                f"You are standing at the {l_view}. Can you see the "
                f"{l_tgt} from this position? Answer yes or no."
            )
            a = scene._visibility_real_answer
        else:
            # Externally-registered tasks (curriculum_task_registry.py). Empty
            # registry -> same ValueError as before.
            spec = get_probe_task(task)
            if spec is not None:
                q, a = spec.qa(self, scene, task)
            else:
                raise ValueError(f"Unknown probe task: {task}")

        return q, a
