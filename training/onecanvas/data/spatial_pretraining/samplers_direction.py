"""DirectionMixin: spatial-pretraining samplers."""

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


class DirectionMixin:
    def _sample_object_class_rel_dir_count_side_real(self, scene, rng):
        """Real-asset version of rel_dir_count_side_box (fully real).

        All four roles are real-asset pastes referenced by class name:
          - ref / fwd: distinct-class anchor objects defining the egocentric
            pose ("standing at the chair, facing the table")
          - target class: N instances of one shared class to count on the
            asked side
          - distractors: 0..3 instances of OTHER classes (different from
            ref/fwd/target) that should NOT be counted

        Per-target unanimous side check uses ``_count_side_box_consistent``
        with each ref/fwd/target OBB taken as (asset_center, asset.bbox_dims,
        R_yaw) — same world-frame OBB the paste step paints. Markers list
        is empty since every reference is a class name in the prompt.

        Stores ``scene._count_side_real_label`` (side),
        ``scene._count_side_real_answer`` (int),
        ``scene._count_side_real_class_label`` (target class label),
        ``scene._count_side_real_label_ref`` / ``_label_fwd`` (anchor labels),
        ``scene._count_side_real_n_targets`` (int N).
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        # Need >= 3 distinct classes (ref, fwd, target). Distractors come
        # from other_labels, so 0 distractors is OK with exactly 3 classes.
        if bank is None or len(bank.labels()) < 3:
            return None, None

        sin_margin = math.sin(math.radians(15.0))
        q25, q75 = _scene_iqr_aabb(scene)

        side_label = rng.choice(["right", "left", "front", "back"])
        N = rng.choices([3, 4, 5, 6], weights=[30, 35, 25, 10])[0]

        def _R_yaw(yaw):
            c, s = math.cos(yaw), math.sin(yaw)
            return torch.tensor(
                [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]],
                dtype=torch.float32,
            )

        def _make_paste(label, asset, center, yaw):
            return make_asset_paste(scene, rng, label, asset, center, yaw)

        for _outer in range(20):
            # Sample 3 distinct anchor + target classes; remaining classes
            # are the distractor pool.
            ref_label, fwd_label, target_class = rng.sample(bank.labels(), 3)
            other_labels = [
                l for l in bank.labels()
                if l not in (ref_label, fwd_label, target_class)
            ]
            asset_ref = bank.sample(ref_label, rng, inflation_frac=inflation_frac)
            asset_fwd = bank.sample(fwd_label, rng, inflation_frac=inflation_frac)
            r_ref = float(asset_ref["bbox_dims"].max().item()) / 2.0 + 0.20
            r_fwd = float(asset_fwd["bbox_dims"].max().item()) / 2.0 + 0.20

            # Place ref then fwd; require horizontal separation ≥ 0.5 m so
            # the heading vector is well-defined and the model can read
            # ref→fwd as a clear facing direction.
            center_ref = _sample_collision_free_center(
                [], [], r_ref, rng, aabb_min=q25, aabb_max=q75,
            )
            if center_ref is None:
                continue
            center_fwd = None
            for _f_try in range(20):
                cf = _sample_collision_free_center(
                    [center_ref], [r_ref], r_fwd, rng,
                    aabb_min=q25, aabb_max=q75,
                )
                if cf is None:
                    break
                hx0 = float(cf[0].item() - center_ref[0].item())
                hz0 = float(cf[2].item() - center_ref[2].item())
                if math.sqrt(hx0 * hx0 + hz0 * hz0) >= 0.5:
                    center_fwd = cf
                    break
            if center_fwd is None:
                continue

            yaw_ref = rng.uniform(-math.pi, math.pi)
            yaw_fwd = rng.uniform(-math.pi, math.pi)
            R_ref = _R_yaw(yaw_ref)
            R_fwd = _R_yaw(yaw_fwd)
            dims_ref = asset_ref["bbox_dims"].float()
            dims_fwd = asset_fwd["bbox_dims"].float()

            ax_c = float(center_ref[0].item())
            az_c = float(center_ref[2].item())
            bx_c = float(center_fwd[0].item())
            bz_c = float(center_fwd[2].item())
            hx = bx_c - ax_c; hz = bz_c - az_c
            h_n = math.sqrt(hx * hx + hz * hz)
            if h_n < 1e-3:
                continue
            hx /= h_n; hz /= h_n

            placed_centers: list = [center_ref, center_fwd]
            placed_radii: list = [r_ref, r_fwd]
            target_pastes: list = []
            target_sides: list = []

            for _t_try in range(N * 30):
                if len(target_pastes) >= N:
                    break
                asset = bank.sample(target_class, rng, inflation_frac=inflation_frac)
                r_sphere = float(asset["bbox_dims"].max().item()) / 2.0 + 0.20
                center = _sample_collision_free_center(
                    placed_centers, placed_radii, r_sphere, rng,
                    aabb_min=q25, aabb_max=q75,
                )
                if center is None:
                    continue
                tx = float(center[0].item()) - ax_c
                tz = float(center[2].item()) - az_c
                t_n = math.sqrt(tx * tx + tz * tz)
                if t_n < 0.3:
                    continue
                tx /= t_n; tz /= t_n
                center_dot = hx * tx + hz * tz
                center_cross = hx * tz - hz * tx
                if _ego_cardinal_side(center_dot, center_cross,
                                      sin_margin) is None:
                    continue

                yaw = rng.uniform(-math.pi, math.pi)
                R_t = _R_yaw(yaw)
                dims_t = asset["bbox_dims"].float()
                tgt_side = _count_side_box_consistent(
                    center_ref, dims_ref, R_ref,
                    center_fwd, dims_fwd, R_fwd,
                    center, dims_t, R_t,
                    sin_margin,
                )
                if tgt_side is None:
                    continue

                spread = int(asset["frame_indices"].max().item()
                             - asset["frame_indices"].min().item())
                t_max = max(0, int(scene.n_images) - 1 - spread)
                t_start = rng.randint(0, t_max) if t_max > 0 else 0

                placed_centers.append(center)
                placed_radii.append(r_sphere)
                target_pastes.append({
                    "asset": asset, "target_center": center,
                    "yaw_rad": float(yaw), "t_start": int(t_start),
                    "label": target_class,
                })
                target_sides.append(tgt_side)

            if len(target_pastes) < N:
                continue
            if len(set(target_sides)) < 2:
                continue
            count = sum(1 for s in target_sides if s == side_label)
            if count == 0 or count == N:
                continue

            distract_pastes: list = []
            if other_labels:
                n_distract = rng.randint(0, 3)
                for _ in range(n_distract):
                    d_label = rng.choice(other_labels)
                    d_asset = bank.sample(d_label, rng, inflation_frac=inflation_frac)
                    d_r = float(d_asset["bbox_dims"].max().item()) / 2.0 + 0.20
                    d_c = _sample_collision_free_center(
                        placed_centers, placed_radii, d_r, rng,
                        aabb_min=q25, aabb_max=q75,
                    )
                    if d_c is None:
                        continue
                    d_yaw = rng.uniform(-math.pi, math.pi)
                    d_spread = int(d_asset["frame_indices"].max().item()
                                   - d_asset["frame_indices"].min().item())
                    d_tmax = max(0, int(scene.n_images) - 1 - d_spread)
                    d_tstart = rng.randint(0, d_tmax) if d_tmax > 0 else 0
                    placed_centers.append(d_c)
                    placed_radii.append(d_r)
                    distract_pastes.append({
                        "asset": d_asset, "target_center": d_c,
                        "yaw_rad": float(d_yaw), "t_start": int(d_tstart),
                        "label": d_label,
                    })

            scene._real_asset_pastes = (
                [_make_paste(ref_label, asset_ref, center_ref, yaw_ref),
                 _make_paste(fwd_label, asset_fwd, center_fwd, yaw_fwd)]
                + target_pastes + distract_pastes
            )
            scene._count_side_real_label = side_label
            scene._count_side_real_answer = int(count)
            scene._count_side_real_class_label = target_class
            scene._count_side_real_label_ref = ref_label
            scene._count_side_real_label_fwd = fwd_label
            scene._count_side_real_n_targets = int(N)
            return [], []

        return None, None

    def _sample_object_class_rel_dir_real(self, scene, rng, task):
        """Real-asset version of rel_dir_{easy,medium,hard,4way}_box.

        Pastes 3 distinct-class real assets (ref / fwd / tgt) at IQR-collision-
        free centers. Direction (tgt_center - ref_center) relative to
        (fwd_center - ref_center) is computed in the xz horizontal plane,
        matching the synthetic version's center-only convention. Rejection
        margins are identical to the synthetic samplers (15° from each label
        boundary), since the question phrasing keys on object identity (class
        name), so the model groups each object by class features and the
        canonical reference position is the asset's center, not its surface.

        Question placeholders (0): all three objects referenced by class name.

        Stores ``scene._rel_dir_real_label`` (string label),
        ``scene._rel_dir_real_task`` (variant: rel_dir_easy / medium / hard /
        4way), ``scene._rel_dir_real_label_ref/fwd/tgt`` (class names).
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        if bank is None or len(bank.labels()) < 3:
            return None, None

        labels_pool = bank.labels()
        q25, q75 = _scene_iqr_aabb(scene)

        sin_margin = math.sin(math.radians(15.0))
        boundary_cos = math.cos(math.radians(135))
        c30 = math.cos(math.radians(30))
        c60 = math.cos(math.radians(60))
        base = task.replace("object_class_", "").replace("_real", "")

        if base == "rel_dir_easy":
            target_labels = ["left", "right"]
        elif base == "rel_dir_medium":
            target_labels = ["left", "right", "back"]
        elif base == "rel_dir_4way":
            target_labels = ["front", "back", "left", "right"]
        else:  # hard
            target_labels = ["front-left", "front-right",
                             "back-left", "back-right"]

        def _R_yaw(yaw):
            c, s = math.cos(yaw), math.sin(yaw)
            return torch.tensor(
                [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]],
                dtype=torch.float32,
            )

        def _make_paste(label, asset, center, yaw):
            return make_asset_paste(scene, rng, label, asset, center, yaw)

        for _outer in range(30):
            la, lb, lc = rng.sample(labels_pool, 3)
            asset_a = bank.sample(la, rng, inflation_frac=inflation_frac)
            asset_b = bank.sample(lb, rng, inflation_frac=inflation_frac)
            r_a = float(asset_a["bbox_dims"].max().item()) / 2.0 + 0.20
            r_b = float(asset_b["bbox_dims"].max().item()) / 2.0 + 0.20

            # Place ref then fwd; require horizontal separation ≥ 0.3 m so
            # the heading vector is well-defined.
            center_a = _sample_collision_free_center(
                [], [], r_a, rng, aabb_min=q25, aabb_max=q75,
            )
            if center_a is None:
                continue
            center_b = None
            for _b_try in range(20):
                cb = _sample_collision_free_center(
                    [center_a], [r_a], r_b, rng,
                    aabb_min=q25, aabb_max=q75,
                )
                if cb is None:
                    break
                hx = float(cb[0].item() - center_a[0].item())
                hz = float(cb[2].item() - center_a[2].item())
                if math.sqrt(hx * hx + hz * hz) >= 0.3:
                    center_b = cb
                    break
            if center_b is None:
                continue

            ax = float(center_a[0].item()); az = float(center_a[2].item())
            bx = float(center_b[0].item()); bz = float(center_b[2].item())
            hx = bx - ax; hz = bz - az
            h_n = math.sqrt(hx * hx + hz * hz)
            hx /= h_n; hz /= h_n

            target_label = rng.choice(target_labels)
            asset_c = bank.sample(lc, rng, inflation_frac=inflation_frac)
            r_c = float(asset_c["bbox_dims"].max().item()) / 2.0 + 0.20

            chosen_center = None
            for _c_try in range(60):
                cc = _sample_collision_free_center(
                    [center_a, center_b], [r_a, r_b], r_c, rng,
                    aabb_min=q25, aabb_max=q75,
                )
                if cc is None:
                    continue
                cx = float(cc[0].item()); cz = float(cc[2].item())
                tx = cx - ax; tz = cz - az
                t_n = math.sqrt(tx * tx + tz * tz)
                if t_n < 0.3:
                    continue
                tx /= t_n; tz /= t_n
                dot = hx * tx + hz * tz
                cross = hx * tz - hz * tx

                ok = False
                if base == "rel_dir_easy":
                    if abs(cross) < sin_margin:
                        continue
                    if target_label == "left" and cross > 0:
                        ok = True
                    elif target_label == "right" and cross < 0:
                        ok = True
                elif base == "rel_dir_medium":
                    if abs(dot - boundary_cos) < sin_margin:
                        continue
                    if abs(cross) < sin_margin and dot > boundary_cos:
                        continue
                    if target_label == "back":
                        ok = dot < boundary_cos
                    else:
                        if dot < boundary_cos:
                            continue
                        if target_label == "left" and cross > 0:
                            ok = True
                        elif target_label == "right" and cross < 0:
                            ok = True
                elif base == "rel_dir_4way":
                    if target_label == "front":
                        ok = dot >= c30
                    elif target_label == "back":
                        ok = dot <= -c30
                    else:
                        if abs(dot) > c60:
                            continue
                        if target_label == "left" and cross > 0:
                            ok = True
                        elif target_label == "right" and cross < 0:
                            ok = True
                else:  # hard
                    if abs(dot) < sin_margin or abs(cross) < sin_margin:
                        continue
                    want_front = target_label.startswith("front")
                    want_left = target_label.endswith("left")
                    if want_front and dot <= 0:
                        continue
                    if not want_front and dot >= 0:
                        continue
                    if want_left and cross <= 0:
                        continue
                    if not want_left and cross >= 0:
                        continue
                    ok = True

                if ok:
                    chosen_center = cc
                    break

            if chosen_center is None:
                continue

            yaw_a = rng.uniform(-math.pi, math.pi)
            yaw_b = rng.uniform(-math.pi, math.pi)
            yaw_c = rng.uniform(-math.pi, math.pi)
            scene._real_asset_pastes = [
                _make_paste(la, asset_a, center_a, yaw_a),
                _make_paste(lb, asset_b, center_b, yaw_b),
                _make_paste(lc, asset_c, chosen_center, yaw_c),
            ]
            scene._rel_dir_real_label = target_label
            scene._rel_dir_real_task = base
            scene._rel_dir_real_label_ref = la
            scene._rel_dir_real_label_fwd = lb
            scene._rel_dir_real_label_tgt = lc

            other_labels = [l for l in labels_pool if l not in (la, lb, lc)]
            if other_labels:
                placed_centers = [center_a, center_b, chosen_center]
                placed_radii = [r_a, r_b, r_c]
                n_distract = rng.randint(self._real_asset_distract_min,
                                         self._real_asset_distract_max)
                for _ in range(n_distract):
                    d_label = rng.choice(other_labels)
                    d_asset = bank.sample(d_label, rng, inflation_frac=inflation_frac)
                    r_d = float(d_asset["bbox_dims"].max().item()) / 2.0 + 0.20
                    d_center = _sample_collision_free_center(
                        placed_centers, placed_radii, r_d, rng,
                        aabb_min=q25, aabb_max=q75,
                    )
                    if d_center is None:
                        continue
                    d_yaw = rng.uniform(-math.pi, math.pi)
                    placed_centers.append(d_center)
                    placed_radii.append(r_d)
                    scene._real_asset_pastes.append(
                        _make_paste(d_label, d_asset, d_center, d_yaw)
                    )
            return [], []

        return None, None

    def _sample_object_class_rel_dir_camera_real(self, scene, rng, task):
        """Real-asset version of rel_dir_camera_{easy,medium,hard}_box.

        Two distinct-class real-asset pastes (pivot + target). The canvas has
        already been reoriented onto a real scene camera by the dataset's
        rel_dir_camera_* prefix branch (camera at origin, forward = +Z), so
        the heading vector is hard-coded to canvas +Z. Direction
        (target_center - pivot_center) is checked against the chosen
        target_label with the same 15° margin as the synthetic variant.

        Question placeholders (0): pivot + target referenced by class name.

        Stores ``scene._rel_dir_camera_real_label``,
        ``scene._rel_dir_camera_real_task``,
        ``scene._rel_dir_camera_real_label_pivot/target``.
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        if bank is None or len(bank.labels()) < 2:
            return None, None

        labels_pool = bank.labels()
        q25, q75 = _scene_iqr_aabb(scene)

        sin_margin = math.sin(math.radians(15.0))
        boundary_cos = math.cos(math.radians(135))
        base = task.replace("object_class_", "").replace("_real", "")
        canvas_base = base.replace("rel_dir_camera_", "rel_dir_")

        if canvas_base == "rel_dir_easy":
            target_labels = ["left", "right"]
        elif canvas_base == "rel_dir_medium":
            target_labels = ["left", "right", "back"]
        else:  # hard
            target_labels = ["front-left", "front-right",
                             "back-left", "back-right"]

        # Camera-axes forward in canvas-local coords (matches the synthetic
        # rel_dir_camera_* branch at L5731).
        hx, hz = 0.0, 1.0

        def _make_paste(label, asset, center, yaw):
            return make_asset_paste(scene, rng, label, asset, center, yaw)

        for _outer in range(30):
            l_piv, l_tgt = rng.sample(labels_pool, 2)
            asset_piv = bank.sample(l_piv, rng, inflation_frac=inflation_frac)
            asset_tgt = bank.sample(l_tgt, rng, inflation_frac=inflation_frac)
            r_piv = float(asset_piv["bbox_dims"].max().item()) / 2.0 + 0.20
            r_tgt = float(asset_tgt["bbox_dims"].max().item()) / 2.0 + 0.20

            center_piv = _sample_collision_free_center(
                [], [], r_piv, rng, aabb_min=q25, aabb_max=q75,
            )
            if center_piv is None:
                continue
            ax = float(center_piv[0].item())
            az = float(center_piv[2].item())

            target_label = rng.choice(target_labels)
            chosen_center = None
            for _t_try in range(60):
                cc = _sample_collision_free_center(
                    [center_piv], [r_piv], r_tgt, rng,
                    aabb_min=q25, aabb_max=q75,
                )
                if cc is None:
                    continue
                cx = float(cc[0].item()); cz = float(cc[2].item())
                tx = cx - ax; tz = cz - az
                t_n = math.sqrt(tx * tx + tz * tz)
                if t_n < 0.3:
                    continue
                tx /= t_n; tz /= t_n
                dot = hx * tx + hz * tz
                cross = hx * tz - hz * tx

                ok = False
                if canvas_base == "rel_dir_easy":
                    if abs(cross) < sin_margin:
                        continue
                    if target_label == "left" and cross > 0:
                        ok = True
                    elif target_label == "right" and cross < 0:
                        ok = True
                elif canvas_base == "rel_dir_medium":
                    if abs(dot - boundary_cos) < sin_margin:
                        continue
                    if abs(cross) < sin_margin and dot > boundary_cos:
                        continue
                    if target_label == "back":
                        ok = dot < boundary_cos
                    else:
                        if dot < boundary_cos:
                            continue
                        if target_label == "left" and cross > 0:
                            ok = True
                        elif target_label == "right" and cross < 0:
                            ok = True
                else:  # hard
                    if abs(dot) < sin_margin or abs(cross) < sin_margin:
                        continue
                    want_front = target_label.startswith("front")
                    want_left = target_label.endswith("left")
                    if want_front and dot <= 0:
                        continue
                    if not want_front and dot >= 0:
                        continue
                    if want_left and cross <= 0:
                        continue
                    if not want_left and cross >= 0:
                        continue
                    ok = True

                if ok:
                    chosen_center = cc
                    break

            if chosen_center is None:
                continue

            yaw_piv = rng.uniform(-math.pi, math.pi)
            yaw_tgt = rng.uniform(-math.pi, math.pi)
            scene._real_asset_pastes = [
                _make_paste(l_piv, asset_piv, center_piv, yaw_piv),
                _make_paste(l_tgt, asset_tgt, chosen_center, yaw_tgt),
            ]
            scene._rel_dir_camera_real_label = target_label
            scene._rel_dir_camera_real_task = canvas_base
            scene._rel_dir_camera_real_label_pivot = l_piv
            scene._rel_dir_camera_real_label_target = l_tgt

            other_labels = [l for l in labels_pool if l not in (l_piv, l_tgt)]
            if other_labels:
                placed_centers = [center_piv, chosen_center]
                placed_radii = [r_piv, r_tgt]
                n_distract = rng.randint(self._real_asset_distract_min,
                                         self._real_asset_distract_max)
                for _ in range(n_distract):
                    d_label = rng.choice(other_labels)
                    d_asset = bank.sample(d_label, rng, inflation_frac=inflation_frac)
                    r_d = float(d_asset["bbox_dims"].max().item()) / 2.0 + 0.20
                    d_center = _sample_collision_free_center(
                        placed_centers, placed_radii, r_d, rng,
                        aabb_min=q25, aabb_max=q75,
                    )
                    if d_center is None:
                        continue
                    d_yaw = rng.uniform(-math.pi, math.pi)
                    placed_centers.append(d_center)
                    placed_radii.append(r_d)
                    scene._real_asset_pastes.append(
                        _make_paste(d_label, d_asset, d_center, d_yaw)
                    )
            return [], []

        return None, None

    def _sample_object_class_rel_dir_oclock_real(self, scene, rng):
        """Real-asset version of rel_dir_oclock_box (SQA3D-situated).

        Three distinct-class real-asset pastes (ref / fwd / tgt). Target hour
        is drawn uniformly from 1..12, and a center is rejection-sampled until
        the angular bin (snapped via _ego_hour) matches the chosen hour. Uses
        a 15° margin from each hour boundary so the snapped hour stays well-
        defined under small perturbations.

        Question placeholders (0): all three referenced by class name.

        Stores ``scene._rel_dir_oclock_real_hour`` (int 1..12),
        ``scene._rel_dir_oclock_real_label_ref/fwd/tgt`` (class names).
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        if bank is None or len(bank.labels()) < 3:
            return None, None

        labels_pool = bank.labels()
        q25, q75 = _scene_iqr_aabb(scene)

        # Snap window for hour h: theta_cw ∈ [(h - 0.5) * 30°, (h + 0.5) * 30°),
        # acceptance window after 15° margin: ±10° around the hour center
        # (effective bin half-width 10° instead of 15°).
        accept_half = math.radians(10.0)

        def _R_yaw(yaw):
            c, s = math.cos(yaw), math.sin(yaw)
            return torch.tensor(
                [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]],
                dtype=torch.float32,
            )

        def _make_paste(label, asset, center, yaw):
            return make_asset_paste(scene, rng, label, asset, center, yaw)

        for _outer in range(30):
            la, lb, lc = rng.sample(labels_pool, 3)
            asset_a = bank.sample(la, rng, inflation_frac=inflation_frac)
            asset_b = bank.sample(lb, rng, inflation_frac=inflation_frac)
            r_a = float(asset_a["bbox_dims"].max().item()) / 2.0 + 0.20
            r_b = float(asset_b["bbox_dims"].max().item()) / 2.0 + 0.20

            center_a = _sample_collision_free_center(
                [], [], r_a, rng, aabb_min=q25, aabb_max=q75,
            )
            if center_a is None:
                continue
            center_b = None
            for _b_try in range(20):
                cb = _sample_collision_free_center(
                    [center_a], [r_a], r_b, rng,
                    aabb_min=q25, aabb_max=q75,
                )
                if cb is None:
                    break
                hx0 = float(cb[0].item() - center_a[0].item())
                hz0 = float(cb[2].item() - center_a[2].item())
                if math.sqrt(hx0 * hx0 + hz0 * hz0) >= 0.3:
                    center_b = cb
                    break
            if center_b is None:
                continue

            ax = float(center_a[0].item()); az = float(center_a[2].item())
            bx = float(center_b[0].item()); bz = float(center_b[2].item())
            hx = bx - ax; hz = bz - az
            h_n = math.sqrt(hx * hx + hz * hz)
            hx /= h_n; hz /= h_n

            target_hour = rng.randint(1, 12)
            target_theta = (target_hour % 12) * (2 * math.pi / 12)

            asset_c = bank.sample(lc, rng, inflation_frac=inflation_frac)
            r_c = float(asset_c["bbox_dims"].max().item()) / 2.0 + 0.20

            chosen_center = None
            for _t_try in range(60):
                cc = _sample_collision_free_center(
                    [center_a, center_b], [r_a, r_b], r_c, rng,
                    aabb_min=q25, aabb_max=q75,
                )
                if cc is None:
                    continue
                cx = float(cc[0].item()); cz = float(cc[2].item())
                tx = cx - ax; tz = cz - az
                t_n = math.sqrt(tx * tx + tz * tz)
                if t_n < 0.3:
                    continue
                tx /= t_n; tz /= t_n
                dot = hx * tx + hz * tz
                cross = hx * tz - hz * tx
                # Match _ego_hour: theta_cw = atan2(-cross, dot), folded to [0, 2π).
                theta_cw = math.atan2(-cross, dot)
                theta_cw_pos = theta_cw if theta_cw >= 0 else theta_cw + 2 * math.pi
                delta = abs(theta_cw_pos - target_theta)
                delta = min(delta, 2 * math.pi - delta)
                if delta < accept_half:
                    chosen_center = cc
                    break

            if chosen_center is None:
                continue

            yaw_a = rng.uniform(-math.pi, math.pi)
            yaw_b = rng.uniform(-math.pi, math.pi)
            yaw_c = rng.uniform(-math.pi, math.pi)
            scene._real_asset_pastes = [
                _make_paste(la, asset_a, center_a, yaw_a),
                _make_paste(lb, asset_b, center_b, yaw_b),
                _make_paste(lc, asset_c, chosen_center, yaw_c),
            ]
            scene._rel_dir_oclock_real_hour = int(target_hour)
            scene._rel_dir_oclock_real_label_ref = la
            scene._rel_dir_oclock_real_label_fwd = lb
            scene._rel_dir_oclock_real_label_tgt = lc

            other_labels = [l for l in labels_pool if l not in (la, lb, lc)]
            if other_labels:
                placed_centers = [center_a, center_b, chosen_center]
                placed_radii = [r_a, r_b, r_c]
                n_distract = rng.randint(self._real_asset_distract_min,
                                         self._real_asset_distract_max)
                for _ in range(n_distract):
                    d_label = rng.choice(other_labels)
                    d_asset = bank.sample(d_label, rng, inflation_frac=inflation_frac)
                    r_d = float(d_asset["bbox_dims"].max().item()) / 2.0 + 0.20
                    d_center = _sample_collision_free_center(
                        placed_centers, placed_radii, r_d, rng,
                        aabb_min=q25, aabb_max=q75,
                    )
                    if d_center is None:
                        continue
                    d_yaw = rng.uniform(-math.pi, math.pi)
                    placed_centers.append(d_center)
                    placed_radii.append(r_d)
                    scene._real_asset_pastes.append(
                        _make_paste(d_label, d_asset, d_center, d_yaw)
                    )
            return [], []

        return None, None

    def _sample_rel_dir_count_side(self, scene, rng):
        """Sampler for rel_dir_count_side_box.

        Combines 2 anchor boxes (ref, fwd) wrapping real patches with N
        target boxes that share a single feature source (same-class cluster,
        SQA3D-style "how many chairs are on my right"). The asked-about side
        is drawn uniformly from {right, left, front, back}; target positions
        are rejection-sampled against a 15° margin from the 45° diagonals and
        checked for per-target unanimity across all OBB corner triples.

        Question placeholders (3): ref, fwd, class_feature_source. The N
        target boxes appear on the canvas but are not marker-referenced.

        When ``self._counting_difficulty_mix`` is set, applies a reduced
        lever set of L1 (sparse corner-only markers), L2 (tight 3D packing),
        and L4 (per-OBB independent dims). L3 / L5 are skipped because the
        side-classification GT margin would be unstable under canvas-collinear
        partners or larger N. Easy stays bit-exact.
        """
        if scene.n_valid < 3:
            return None, None
        frame_index = scene.frame_index
        depth_t = scene.depth
        cos_lat = torch.cos(scene.latitude)
        lon_t = scene.longitude
        sin_margin = math.sin(math.radians(15.0))

        def _xz_patch(i):
            d_i = float(depth_t[i].item())
            cl = float(cos_lat[i].item())
            lo = float(lon_t[i].item())
            return d_i * cl * math.sin(lo), d_i * cl * math.cos(lo)

        pts_all = torch.stack([
            scene.depth * cos_lat * torch.sin(lon_t),
            scene.depth * torch.sin(scene.latitude),
            scene.depth * cos_lat * torch.cos(lon_t),
        ], dim=-1)
        q25 = pts_all.quantile(0.25, dim=0)
        q75 = pts_all.quantile(0.75, dim=0)

        # Difficulty knobs (no rng calls when the flag is off → bit-exact).
        if self._counting_difficulty_mix:
            level = rng.choices(
                ["easy", "medium", "hard"], weights=[0.50, 0.30, 0.20], k=1,
            )[0]
        else:
            level = "easy"

        sparse_K = 0
        pack_factor = 2.0
        per_obb_dims = False
        if level == "medium":
            if rng.random() < 0.5:
                sparse_K = 4
            pack_factor = rng.uniform(1.4, 2.0)
        elif level == "hard":
            sparse_K = rng.choice([2, 3, 4])
            pack_factor = rng.uniform(0.9, 1.4)
            per_obb_dims = True

        side_label = rng.choice(["right", "left", "front", "back"])
        N = rng.choices([3, 4, 5, 6], weights=[30, 35, 25, 10])[0]

        # Shared target dims (object_counting-style cap so N boxes fit).
        max_dim_cap = max(0.25, min(3.26, 1.8 / math.sqrt(N + 2)))

        for _outer in range(20):
            # Sample ref/fwd anchors (rel_dir_easy style: distinct frames, min sep).
            ia = ib = None
            for _a_try in range(30):
                i0, i1 = rng.sample(range(scene.n_valid), 2)
                if int(frame_index[i0].item()) == int(frame_index[i1].item()):
                    continue
                ax0, az0 = _xz_patch(i0); bx0, bz0 = _xz_patch(i1)
                if math.sqrt((bx0 - ax0) ** 2 + (bz0 - az0) ** 2) < 0.3:
                    continue
                ia, ib = i0, i1
                break
            if ia is None:
                continue
            fa, fb = int(frame_index[ia].item()), int(frame_index[ib].item())
            ax, az = _xz_patch(ia); bx, bz = _xz_patch(ib)
            hx, hz = bx - ax, bz - az
            h_n = math.sqrt(hx * hx + hz * hz)
            hx /= h_n; hz /= h_n

            # Class feature source (distinct frame from ref & fwd).
            class_cands = [j for j in range(scene.n_valid)
                           if j != ia and j != ib
                           and int(frame_index[j].item()) not in (fa, fb)]
            if not class_cands:
                continue
            class_feat = rng.choice(class_cands)
            fc = int(frame_index[class_feat].item())

            # Place anchor boxes via _sample_multi_box (patch-hinted positions).
            # budget_n_boxes = N + 2 so each anchor + each target shares the
            # per-sample body-patch budget uniformly (N sampled targets below
            # plus the 2 anchors placed here).
            anchor_pis, _ = self._sample_multi_box(
                scene, rng, n_boxes=2,
                patch_indices_hint=[ia, ib],
                rotation_mode="yaw",
                budget_n_boxes=N + 2,
            )
            if anchor_pis is None:
                continue
            anchor_centers = list(scene._multi_box_centers)
            anchor_dims = list(scene._multi_box_dims)
            anchor_rotations = list(scene._multi_box_rotations)
            anchor_spherical = list(scene._multi_box_spherical)
            anchor_sources = list(scene._multi_box_feature_sources)

            # Shared seed dims (also used per-target when per_obb_dims=False
            # — preserves easy-mode bit-exactness).
            def _draw_target_dims():
                dm = math.exp(rng.uniform(math.log(0.25), math.log(max_dim_cap)))
                da = math.exp(rng.uniform(math.log(0.07), math.log(dm)))
                db = math.exp(rng.uniform(math.log(0.07), math.log(dm)))
                triple = [dm, da, db]
                rng.shuffle(triple)
                return tuple(triple)

            seed_dims = _draw_target_dims()
            seed_r_sphere = math.sqrt(sum(v ** 2 for v in seed_dims)) / 2.0
            anchor_sphere_r = [
                math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) / 2.0
                for d in anchor_dims
            ]

            target_centers = []
            target_rotations = []
            target_spherical = []
            target_sides = []
            target_dims_list = []
            target_radii = []

            _n_total_default = self._per_box_budget(N + 2)

            for _t_try in range(N * 30):
                if len(target_centers) >= N:
                    break
                # L4: per-OBB dims when active; else seed_dims for all targets.
                dims_k = _draw_target_dims() if per_obb_dims else seed_dims
                r_sphere_k = math.sqrt(sum(v ** 2 for v in dims_k)) / 2.0

                # Yaw-only rotation (gravity-aligned, matches VSI convention).
                yaw = rng.uniform(0, 2 * math.pi)
                cy, sy = math.cos(yaw), math.sin(yaw)
                R = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
                center = torch.tensor([
                    rng.uniform(q25[0].item(), q75[0].item()),
                    rng.uniform(q25[1].item(), q75[1].item()),
                    rng.uniform(q25[2].item(), q75[2].item()),
                ], dtype=torch.float32)

                # L2: tight packing — pack=2.0 recovers the original
                # touching-sphere bound (r_a + r_b) bit-exact.
                too_close = False
                for c, r_a in zip(anchor_centers, anchor_sphere_r):
                    if (center - c).norm().item() < (pack_factor / 2.0) * (r_sphere_k + r_a):
                        too_close = True
                        break
                if too_close:
                    continue
                for c, r_other in zip(target_centers, target_radii):
                    if (center - c).norm().item() < (pack_factor / 2.0) * (r_sphere_k + r_other):
                        too_close = True
                        break
                if too_close:
                    continue

                # Quick center-based side check (fast reject).
                tx = float(center[0].item()) - ax
                tz = float(center[2].item()) - az
                t_n = math.sqrt(tx * tx + tz * tz)
                if t_n < 0.3:
                    continue
                tx /= t_n; tz /= t_n
                center_dot = hx * tx + hz * tz
                center_cross = hx * tz - hz * tx
                center_side = _ego_cardinal_side(center_dot, center_cross, sin_margin)
                if center_side is None:
                    continue

                # Full box-corner unanimity check (729 triples per target).
                tgt_side = _count_side_box_consistent(
                    anchor_centers[0], anchor_dims[0], anchor_rotations[0],
                    anchor_centers[1], anchor_dims[1], anchor_rotations[1],
                    center, dims_k, R,
                    sin_margin,
                )
                if tgt_side is None:
                    continue

                # Marker emission. L1 (sparse): only K random corners; bypass
                # the max(8, ...) floor in _sample_obb_surface_points so the
                # marker cluster genuinely tests xyz-adjacency clustering.
                if sparse_K > 0:
                    all_corners = _obb_world_corners(center, dims_k, R)  # [8, 3]
                    K_use = min(sparse_K, 8)
                    perm = torch.randperm(8)[:K_use]
                    world_pts = all_corners[perm]
                    valid_min = min(K_use, 2)
                else:
                    world_pts = _sample_obb_surface_points(
                        center, R, dims_k, n_total=_n_total_default,
                    )
                    valid_min = 4
                lats, lons, depths, valid = _world_to_spherical(world_pts)
                if int(valid.sum().item()) < valid_min:
                    continue

                target_centers.append(center)
                target_rotations.append(R)
                target_spherical.append(torch.stack(
                    [lats[valid], lons[valid], depths[valid]], dim=-1))
                target_sides.append(tgt_side)
                target_dims_list.append(dims_k)
                target_radii.append(r_sphere_k)

            if len(target_centers) < N:
                continue
            if len(set(target_sides)) < 2:
                # Degenerate (all targets on one side) — resample.
                continue

            count = sum(1 for s in target_sides if s == side_label)
            # Reject trivial answers (0 or N) so the task can't be solved by a
            # constant "0" or "N" predictor. Both "0 on my right" and "all
            # N on my right" are easier to guess than the informative middle.
            if count == 0 or count == N:
                continue

            scene._multi_box_centers = anchor_centers + target_centers
            scene._multi_box_dims = anchor_dims + target_dims_list
            scene._multi_box_rotations = anchor_rotations + target_rotations
            scene._multi_box_spherical = anchor_spherical + target_spherical
            scene._multi_box_feature_sources = anchor_sources + [class_feat] * N

            # Hide the class feature source patch and nearby same-frame patches
            # (matches object_counting: the natural-position marker shouldn't
            # count as an extra instance).
            class_lat = scene.latitude[class_feat]
            class_lon = scene.longitude[class_feat]
            class_frame = scene.frame_index[class_feat]
            dlat = scene.latitude - class_lat
            dlon = scene.longitude - class_lon
            ang_dist = torch.sqrt(dlat ** 2 + dlon ** 2)
            hide_mask = (scene.frame_index == class_frame) & (ang_dist < math.radians(20))
            scene._hide_patch_indices = hide_mask.nonzero(as_tuple=True)[0].tolist()

            scene._count_side_label = side_label
            scene._count_side_answer = int(count)
            scene._count_side_n_targets = int(N)
            scene._count_side_target_sides = list(target_sides)
            # Visualizer-only: mirror the counting-family difficulty record.
            # Reduced lever set (L1, L2, L4); L3/L5 are not used in this task.
            scene._counting_diff_level = level
            scene._counting_diff_sparse_K = int(sparse_K)
            scene._counting_diff_pack_factor = float(pack_factor)
            scene._counting_diff_per_obb_dims = bool(per_obb_dims)
            scene._counting_diff_los_pair_count = 0
            scene._counting_diff_n_max_excl = 0  # N drawn from a fixed 3..6 set
            scene._counting_diff_N_target = int(N)
            scene._counting_diff_flag_on = bool(self._counting_difficulty_mix)

            return [ia, ib, class_feat], [fa, fb, fc]

        return None, None
