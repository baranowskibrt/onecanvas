"""MetricMixin: spatial-pretraining samplers."""

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


class MetricMixin:
    def _sample_box_floor_area_nonrect(self, scene, rng, rotation_mode="yaw"):
        """Non-rectangular room probe: two contiguous axis-aligned rects.

        Models L-shaped / T-shaped / offset-rectangle layouts that are common
        in real architecture. Two flat slab OBBs share one full edge (no gap,
        no overlap). Both use the same feature source (they're one room). GT
        is the sum of individual rectangle areas — matches VSI's "combined
        space" phrasing.

        Per-rect dims dx/dz log-uniform in [1.5 m, 10 m]; height dy uniform
        in [2.0 m, 3.5 m] (full ceiling height, distribution-matched to
        real rooms). Both rects share the same yaw angle (if
        rotation_mode="yaw") so they stay contiguous after rotation.
        """
        lat_t, lon_t, depth_t = scene.latitude, scene.longitude, scene.depth
        cos_lat = torch.cos(lat_t)
        pts_all = torch.stack([
            depth_t * cos_lat * torch.sin(lon_t),
            depth_t * torch.sin(lat_t),
            depth_t * cos_lat * torch.cos(lon_t),
        ], dim=-1)
        q25 = pts_all.quantile(0.25, dim=0)
        q75 = pts_all.quantile(0.75, dim=0)
        floor_y = float(pts_all[:, 1].quantile(0.05).item())

        for _ in range(30):
            # Per-rect dims (smaller range than single-rect task since two
            # combine; total area is still in the learnable room-size range).
            dx_A = math.exp(rng.uniform(math.log(1.5), math.log(10.0)))
            dz_A = math.exp(rng.uniform(math.log(1.5), math.log(10.0)))
            dx_B = math.exp(rng.uniform(math.log(1.5), math.log(10.0)))
            dz_B = math.exp(rng.uniform(math.log(1.5), math.log(10.0)))
            dy = math.exp(rng.uniform(math.log(2.0), math.log(3.5)))

            # Build the L-shape in an intermediate frame where A is at the
            # origin, then translate so the MIDPOINT of the combined shape
            # lands at a random scene-centre position.  Anchoring on the
            # midpoint (not on c_A alone) keeps both arms equidistant from
            # the camera, so neither arm dominates the canvas.
            c_A = torch.zeros(3, dtype=torch.float32)
            c_A[1] = floor_y + dy / 2.0

            # Pick contiguity direction: rect B shares an edge with rect A
            # along +X, -X, +Z, or -Z. For +X: B is to the right of A, with
            # B's -X face flush against A's +X face → c_B.x = c_A.x + dx_A/2 + dx_B/2.
            side = rng.choice(["+x", "-x", "+z", "-z"])
            if side == "+x":
                c_B = c_A.clone()
                c_B[0] += (dx_A + dx_B) / 2.0
                # Slide B along Z by a random offset in [-(dz_B-0.3), +(dz_B-0.3)]
                # clipped to keep shared edge nondegenerate (at least 0.3 m overlap).
                z_shift_max = max(0.0, (min(dz_A, dz_B) - 0.3) / 2.0)
                c_B[2] += rng.uniform(-z_shift_max, z_shift_max) if z_shift_max > 0 else 0.0
            elif side == "-x":
                c_B = c_A.clone()
                c_B[0] -= (dx_A + dx_B) / 2.0
                z_shift_max = max(0.0, (min(dz_A, dz_B) - 0.3) / 2.0)
                c_B[2] += rng.uniform(-z_shift_max, z_shift_max) if z_shift_max > 0 else 0.0
            elif side == "+z":
                c_B = c_A.clone()
                c_B[2] += (dz_A + dz_B) / 2.0
                x_shift_max = max(0.0, (min(dx_A, dx_B) - 0.3) / 2.0)
                c_B[0] += rng.uniform(-x_shift_max, x_shift_max) if x_shift_max > 0 else 0.0
            else:  # "-z"
                c_B = c_A.clone()
                c_B[2] -= (dz_A + dz_B) / 2.0
                x_shift_max = max(0.0, (min(dx_A, dx_B) - 0.3) / 2.0)
                c_B[0] += rng.uniform(-x_shift_max, x_shift_max) if x_shift_max > 0 else 0.0

            # Translate so the XZ midpoint of the L lands at a random IQR position.
            mid_xz = torch.tensor([
                rng.uniform(q25[0].item(), q75[0].item()),
                0.0,
                rng.uniform(q25[2].item(), q75[2].item()),
            ], dtype=torch.float32)
            joint_mid = (c_A + c_B) / 2.0
            shift = mid_xz - joint_mid
            shift[1] = 0.0  # keep Y (floor height) unchanged
            c_A = c_A + shift
            c_B = c_B + shift

            # Shared rotation (yaw only, applied about the midpoint of A and B
            # so both rects stay contiguous after rotation).
            R = _rotation_matrix(rotation_mode, rng)
            midpoint = (c_A + c_B) / 2.0
            c_A = (R @ (c_A - midpoint)) + midpoint
            c_B = (R @ (c_B - midpoint)) + midpoint

            _n_total = self._per_box_budget(2)
            pts_A = _sample_obb_surface_points(c_A, R, (dx_A, dy, dz_A), n_total=_n_total)
            lats_A, lons_A, depths_A, valid_A = _world_to_spherical(pts_A)
            pts_B = _sample_obb_surface_points(c_B, R, (dx_B, dy, dz_B), n_total=_n_total)
            lats_B, lons_B, depths_B, valid_B = _world_to_spherical(pts_B)
            if valid_A.sum() < 4 or valid_B.sum() < 4:
                continue

            ref_idx = rng.randrange(scene.n_valid)
            total_area = float(dx_A * dz_A + dx_B * dz_B)

            spherical_A = torch.stack(
                [lats_A[valid_A], lons_A[valid_A], depths_A[valid_A]], dim=-1)
            spherical_B = torch.stack(
                [lats_B[valid_B], lons_B[valid_B], depths_B[valid_B]], dim=-1)
            spherical_all = torch.cat([spherical_A, spherical_B], dim=0)

            # Per-point random feature sources: heterogeneous textures across
            # both rects (see _sample_box_floor_area for rationale).
            n_all = int(spherical_all.shape[0])
            per_point_sources = torch.tensor(
                [rng.randrange(scene.n_valid) for _ in range(n_all)],
                dtype=torch.long,
            )

            scene._floor_area_m2 = total_area
            scene._box_dims = (dx_A, dy, dz_A)  # debug: rect-A dims
            # Single synthetic path (both rects concatenated) — the two rects
            # share one ref_idx so the multi-box split carries no extra info.
            scene._synthetic_spherical = spherical_all
            scene._synthetic_feature_source = ref_idx
            scene._synthetic_per_point_feature_sources = per_point_sources
            # Both slab OBBs for the visualizer (debug only).
            scene._floor_area_obbs = [
                (c_A, (float(dx_A), float(dy), float(dz_A)), R),
                (c_B, (float(dx_B), float(dy), float(dz_B)), R),
            ]
            # Question template has no marker placeholder (room-size is a
            # whole-scene question), so return no patch indices.
            return [], []

        return None, None

    def _sample_object_class_dist_real(self, scene, rng):
        """Real-asset version of dist_box.

        Pastes 2 distinct-class real assets at IQR-collision-free centers;
        GT distance is the min pairwise Euclidean distance between the two
        assets' actual feature-bearing world-frame patches (what the canvas
        renders), not over the asset bbox's surface. Rejects pairs closer
        than 0.05 m (mirrors dist_box's near-touch reject).

        Stores ``scene._dist_real_value`` (float, metres),
        ``scene._dist_real_label_a`` and ``scene._dist_real_label_b``.
        Question carries no inline markers (objects referenced by class name).
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        if bank is None or len(bank.labels()) < 2:
            return None, None

        labels = bank.labels()
        q25, q75 = _scene_iqr_aabb(scene)

        for _ in range(20):
            la, lb = rng.sample(labels, 2)
            asset_a = bank.sample(la, rng, inflation_frac=inflation_frac)
            asset_b = bank.sample(lb, rng, inflation_frac=inflation_frac)
            r_a = float(asset_a["bbox_dims"].max().item()) / 2.0 + 0.20
            r_b = float(asset_b["bbox_dims"].max().item()) / 2.0 + 0.20

            center_a = _sample_collision_free_center(
                [], [], r_a, rng, aabb_min=q25, aabb_max=q75,
            )
            if center_a is None:
                continue
            center_b = _sample_collision_free_center(
                [center_a], [r_a], r_b, rng, aabb_min=q25, aabb_max=q75,
            )
            if center_b is None:
                continue

            yaw_a = rng.uniform(-math.pi, math.pi)
            yaw_b = rng.uniform(-math.pi, math.pi)
            pts_a = _asset_paste_world_pts(asset_a, center_a, yaw_a)
            pts_b = _asset_paste_world_pts(asset_b, center_b, yaw_b)
            if pts_a.numel() == 0 or pts_b.numel() == 0:
                continue
            dist = float(torch.cdist(pts_a, pts_b).min().item())
            if dist <= 0.05:
                continue

            spread_a = int(asset_a["frame_indices"].max().item()
                           - asset_a["frame_indices"].min().item())
            spread_b = int(asset_b["frame_indices"].max().item()
                           - asset_b["frame_indices"].min().item())
            t_max_a = max(0, int(scene.n_images) - 1 - spread_a)
            t_max_b = max(0, int(scene.n_images) - 1 - spread_b)
            t_start_a = rng.randint(0, t_max_a) if t_max_a > 0 else 0
            t_start_b = rng.randint(0, t_max_b) if t_max_b > 0 else 0

            scene._real_asset_pastes = [
                {
                    "asset": asset_a, "target_center": center_a,
                    "yaw_rad": float(yaw_a), "t_start": int(t_start_a),
                    "label": la,
                },
                {
                    "asset": asset_b, "target_center": center_b,
                    "yaw_rad": float(yaw_b), "t_start": int(t_start_b),
                    "label": lb,
                },
            ]
            scene._dist_real_value = float(dist)
            scene._dist_real_label_a = la
            scene._dist_real_label_b = lb

            other_labels = [l for l in labels if l not in (la, lb)]
            if other_labels:
                placed_centers = [center_a, center_b]
                placed_radii = [r_a, r_b]
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
                    d_spread = int(d_asset["frame_indices"].max().item()
                                   - d_asset["frame_indices"].min().item())
                    d_t_max = max(0, int(scene.n_images) - 1 - d_spread)
                    d_t_start = rng.randint(0, d_t_max) if d_t_max > 0 else 0
                    placed_centers.append(d_center)
                    placed_radii.append(r_d)
                    scene._real_asset_pastes.append({
                        "asset": d_asset, "target_center": d_center,
                        "yaw_rad": float(d_yaw), "t_start": int(d_t_start),
                        "label": d_label,
                    })
            return [], []

        return None, None

    def _sample_object_class_size_real(self, scene, rng):
        """Real-asset object-size estimation matching VSI-Bench
        ``object_size_estimation``.

        Pastes exactly one target-class real asset on the canvas (VSI's
        single-instance-per-category constraint) plus 1-3 distractor pastes
        of OTHER classes. The answer is the longest OBB side in cm:
        ``round(max(asset['bbox_dims']) * 100)``. ``bbox_dims`` is
        full-extent in metres (verified against extract_real_object_assets.py
        which builds half-extents locally as ``dims / 2.0``), so this matches
        VSI's ``round(max(axesLengths) * 100)`` GT unit-for-unit.

        Stores ``scene._size_real_value`` (int, cm) and
        ``scene._size_real_label`` (str). No inline markers — object is
        referenced by class name.
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        if bank is None or not bank.labels():
            return None, None

        labels = bank.labels()
        target_label = rng.choice(labels)
        asset = bank.sample(target_label, rng, inflation_frac=inflation_frac)

        q25, q75 = _scene_iqr_aabb(scene)
        r_target = float(asset["bbox_dims"].max().item()) / 2.0 + 0.20
        center = _sample_collision_free_center(
            [], [], r_target, rng, aabb_min=q25, aabb_max=q75,
        )
        if center is None:
            return None, None

        yaw = rng.uniform(-math.pi, math.pi)
        spread = int(asset["frame_indices"].max().item()
                     - asset["frame_indices"].min().item())
        t_max = max(0, int(scene.n_images) - 1 - spread)
        t_start = rng.randint(0, t_max) if t_max > 0 else 0

        placed_centers = [center]
        placed_radii = [r_target]
        scene._real_asset_pastes = [{
            "asset": asset,
            "target_center": center,
            "yaw_rad": float(yaw),
            "t_start": int(t_start),
            "label": target_label,
        }]

        # Distractor pastes of OTHER classes (1..3) so the canvas is not
        # an empty room with one object. VSI's single-instance constraint
        # only restricts the target class, so different-class distractors
        # are safe and mirror counting_real / grounding_real practice.
        other_labels = [l for l in labels if l != target_label]
        n_distract = rng.randint(1, 3) if other_labels else 0
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
            d_spread = int(d_asset["frame_indices"].max().item()
                           - d_asset["frame_indices"].min().item())
            d_t_max = max(0, int(scene.n_images) - 1 - d_spread)
            d_t_start = rng.randint(0, d_t_max) if d_t_max > 0 else 0
            placed_centers.append(d_center)
            placed_radii.append(r_d)
            scene._real_asset_pastes.append({
                "asset": d_asset,
                "target_center": d_center,
                "yaw_rad": float(d_yaw),
                "t_start": int(d_t_start),
                "label": d_label,
            })

        size_cm = int(round(float(asset["bbox_dims"].max().item()) * 100.0))
        scene._size_real_value = size_cm
        scene._size_real_label = target_label
        return [], []

    def _sample_object_class_rel_dist_real(self, scene, rng):
        """Real-asset version of rel_dist_box (metric MCQ).

        Pastes 1 target asset + 4 distinct-class candidate assets. Per-pair
        distance is the min pairwise Euclidean distance between the two
        assets' actual feature-bearing world-frame patches (what the canvas
        renders), not over the asset bbox's surface. Rejects samples whose
        closest-vs-second-closest gap is < 0.05 m (mirrors rel_dist_box's
        tie-break gap). Letter answer is the candidate index (A..D) of the
        closest one.

        Stores ``scene._rel_dist_real_answer`` (letter),
        ``scene._rel_dist_real_target_label``,
        ``scene._rel_dist_real_cand_labels`` (4 labels in A-D order),
        ``scene._rel_dist_real_dists`` (4 distances).
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        if bank is None or len(bank.labels()) < 5:
            return None, None

        labels = bank.labels()
        q25, q75 = _scene_iqr_aabb(scene)

        for _ in range(30):
            chosen = rng.sample(labels, 5)
            target_label = chosen[0]
            cand_labels = chosen[1:]
            assets = [bank.sample(l, rng, inflation_frac=inflation_frac) for l in chosen]
            radii = [float(a["bbox_dims"].max().item()) / 2.0 + 0.20
                     for a in assets]

            placed_centers: list = []
            placed_radii: list = []
            ok_place = True
            for r in radii:
                c = _sample_collision_free_center(
                    placed_centers, placed_radii, r, rng,
                    aabb_min=q25, aabb_max=q75,
                )
                if c is None:
                    ok_place = False
                    break
                placed_centers.append(c)
                placed_radii.append(r)
            if not ok_place:
                continue

            yaws = [rng.uniform(-math.pi, math.pi) for _ in range(5)]
            asset_pts = [
                _asset_paste_world_pts(assets[k], placed_centers[k], yaws[k])
                for k in range(5)
            ]
            if any(p.numel() == 0 for p in asset_pts):
                continue
            dists = [
                float(torch.cdist(asset_pts[0], asset_pts[1 + j]).min().item())
                for j in range(4)
            ]
            sorted_dists = sorted(dists)
            if sorted_dists[1] - sorted_dists[0] < 0.05:
                continue

            winner_idx = dists.index(min(dists))
            scene._rel_dist_real_answer = ["A", "B", "C", "D"][winner_idx]
            scene._rel_dist_real_target_label = target_label
            scene._rel_dist_real_cand_labels = list(cand_labels)
            scene._rel_dist_real_dists = [float(d) for d in dists]

            pastes = []
            for k in range(5):
                spread = int(assets[k]["frame_indices"].max().item()
                             - assets[k]["frame_indices"].min().item())
                t_max = max(0, int(scene.n_images) - 1 - spread)
                t_start = rng.randint(0, t_max) if t_max > 0 else 0
                pastes.append({
                    "asset": assets[k],
                    "target_center": placed_centers[k],
                    "yaw_rad": float(yaws[k]),
                    "t_start": int(t_start),
                    "label": chosen[k],
                })
            scene._real_asset_pastes = pastes

            other_labels = [l for l in labels if l not in chosen]
            if other_labels:
                d_placed_centers = list(placed_centers)
                d_placed_radii = list(placed_radii)
                n_distract = rng.randint(self._real_asset_distract_min,
                                         self._real_asset_distract_max)
                for _ in range(n_distract):
                    d_label = rng.choice(other_labels)
                    d_asset = bank.sample(d_label, rng, inflation_frac=inflation_frac)
                    r_d = float(d_asset["bbox_dims"].max().item()) / 2.0 + 0.20
                    d_center = _sample_collision_free_center(
                        d_placed_centers, d_placed_radii, r_d, rng,
                        aabb_min=q25, aabb_max=q75,
                    )
                    if d_center is None:
                        continue
                    d_yaw = rng.uniform(-math.pi, math.pi)
                    d_spread = int(d_asset["frame_indices"].max().item()
                                   - d_asset["frame_indices"].min().item())
                    d_t_max = max(0, int(scene.n_images) - 1 - d_spread)
                    d_t_start = rng.randint(0, d_t_max) if d_t_max > 0 else 0
                    d_placed_centers.append(d_center)
                    d_placed_radii.append(r_d)
                    scene._real_asset_pastes.append({
                        "asset": d_asset, "target_center": d_center,
                        "yaw_rad": float(d_yaw), "t_start": int(d_t_start),
                        "label": d_label,
                    })
            return [], []

        return None, None

    def _sample_dist_box(self, scene, rng):
        """Sample 2 OBBs with distinct features; compute surface-to-surface distance.

        Stores ``scene._dist_box_value`` (float, metres).
        Returns (patch_indices [2], frame_indices [2]).
        """
        colinear_on = (
            self._colinear_centers_prob > 0.0
            and rng.random() < self._colinear_centers_prob
        )
        scene._dist_box_colinear_on = bool(colinear_on)
        for _ in range(10):
            if colinear_on:
                centers = _draw_colinear_centers(rng, n_boxes=2, scene=scene)
                if centers is None:
                    return None, None
                patch_indices, frame_indices = self._sample_multi_box(
                    scene, rng, n_boxes=2, center_hints=centers)
            else:
                patch_indices, frame_indices = self._sample_multi_box(
                    scene, rng, n_boxes=2)
            if patch_indices is None:
                return None, None
            dist = obb_surface_distance(
                scene._multi_box_centers[0], scene._multi_box_dims[0], scene._multi_box_rotations[0],
                scene._multi_box_centers[1], scene._multi_box_dims[1], scene._multi_box_rotations[1],
            )
            if dist > 0.05:
                scene._dist_box_value = dist
                return patch_indices, frame_indices
            # Boxes too close / overlapping — retry with fresh placement.
        return None, None

    def _sample_rel_dist_box(self, scene, rng):
        """Sample 5 OBBs (target + 4 candidates); enforce >=0.05 m winner–runner-up gap.

        Stores ``scene._rel_dist_box_answer`` ("A"/"B"/"C"/"D").
        Returns (patch_indices [5], frame_indices [5]).
        """
        colinear_on = (
            self._colinear_centers_prob > 0.0
            and rng.random() < self._colinear_centers_prob
        )
        scene._rel_dist_box_colinear_on = bool(colinear_on)
        for _ in range(30):
            if colinear_on:
                centers = _draw_colinear_centers(rng, n_boxes=5, scene=scene)
                if centers is None:
                    return None, None
                patch_indices, frame_indices = self._sample_multi_box(
                    scene, rng, n_boxes=5, center_hints=centers)
            else:
                patch_indices, frame_indices = self._sample_multi_box(
                    scene, rng, n_boxes=5)
            if patch_indices is None:
                return None, None
            dists = [
                obb_surface_distance(
                    scene._multi_box_centers[0], scene._multi_box_dims[0], scene._multi_box_rotations[0],
                    scene._multi_box_centers[1 + j], scene._multi_box_dims[1 + j],
                    scene._multi_box_rotations[1 + j],
                )
                for j in range(4)
            ]
            sorted_dists = sorted(dists)
            if sorted_dists[1] - sorted_dists[0] >= 0.05:
                scene._rel_dist_box_answer = ["A", "B", "C", "D"][dists.index(min(dists))]
                scene._rel_dist_box_dists = dists
                return patch_indices, frame_indices
        return None, None
