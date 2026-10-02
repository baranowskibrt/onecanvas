"""NavigationMixin: spatial-pretraining samplers."""

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
    format_multi_metric_bbox,
    format_multi_metric_bbox_json,
    obb_closest_surface_points,
    obb_surface_distance,
)


from ._common import *


class NavigationMixin:
    def _sample_object_class_route_plan_real(self, scene, rng, n_turns):
        """Real-asset version of route_plan_*turn_box.

        Reuses the synthetic path geometry (``_sample_box_path`` /
        ``_sample_box_path_reface``) but pastes a distinct-class real asset at
        each waypoint instead of placing synthetic OBB markers. Reface is
        randomized internally 50/50 to match the synthetic dispatcher.

        Each waypoint is referenced by its asset's class label in the prompt
        ("Go forward until the chair") instead of by an inline marker.
        Pairwise asset bounding-sphere clearance is enforced after path
        sampling; failures retry the path generation.

        Question placeholders (0): all waypoints referenced by class name.

        Stores ``scene._route_plan_real_classes`` (list of n_turns class
        strings, the GT turn sequence), ``scene._route_plan_real_reface``
        (bool), ``scene._route_plan_real_waypoint_labels`` (list of class
        labels in waypoint order: [start, wp1, ..., wpN, goal] for non-reface
        OR [m0, m_face, m1, ..., wpN, goal] for reface).
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        if bank is None:
            return None, None

        reface = rng.random() < 0.5
        # Non-reface: n_turns + 2 unique waypoints. Reface: n_turns + 3.
        n_waypoints = n_turns + 3 if reface else n_turns + 2
        if len(bank.labels()) < n_waypoints:
            return None, None

        airgap = 0.20

        for _outer in range(20):
            # Resample classes + assets each outer attempt — different draw
            # might fit the same path (e.g., all small classes vs. mixed).
            wp_labels = rng.sample(bank.labels(), n_waypoints)
            wp_assets = [bank.sample(l, rng, inflation_frac=inflation_frac) for l in wp_labels]
            wp_radii = [
                float(a["bbox_dims"].max().item()) / 2.0 + airgap
                for a in wp_assets
            ]

            # Scale path step by drawn radii so adjacent-pair clearance
            # `d >= r_a + r_b` is satisfied by construction regardless of
            # which pair lands adjacent. The default `6/(n+1)` cap was
            # tuned for synthetic OBBs with sub-metre footprints; with the
            # 100+-class real-asset bank (bed/desk/cabinet up to ~2 m) it
            # forces 95%+ rejection. d_max generous (>= 3 m) so non-adjacent
            # pairs spread apart and rarely intersect either.
            r_max = max(wp_radii)
            d_min_eff = max(0.6, 2.0 * r_max + 0.10)
            d_max_eff = max(d_min_eff + 0.5, 3.0)

            # Path sample — synthetic geometry. Reuse existing sampler so the
            # turn-class GT and visibility math stay bit-exact.
            if reface:
                centers, classes = self._sample_box_path_reface(
                    scene, rng, n_turns,
                    d_min_override=d_min_eff, d_max_override=d_max_eff,
                )
            else:
                centers, classes = self._sample_box_path(
                    scene, rng, n_turns,
                    d_min_override=d_min_eff, d_max_override=d_max_eff,
                )
            if centers is None:
                continue
            if len(centers) != n_waypoints:
                continue

            # Pairwise asset clearance: every waypoint pair must be far
            # enough apart that real-asset bounding spheres don't overlap.
            ok = True
            for i in range(n_waypoints):
                if not ok:
                    break
                for j in range(i + 1, n_waypoints):
                    d = float((centers[i] - centers[j]).norm().item())
                    if d < wp_radii[i] + wp_radii[j]:
                        ok = False
                        break
            if not ok:
                continue

            pastes: list = []
            for k in range(n_waypoints):
                asset = wp_assets[k]
                spread = int(asset["frame_indices"].max().item()
                             - asset["frame_indices"].min().item())
                t_max = max(0, int(scene.n_images) - 1 - spread)
                t_start = rng.randint(0, t_max) if t_max > 0 else 0
                pastes.append({
                    "asset": asset,
                    "target_center": centers[k],
                    "yaw_rad": float(rng.uniform(-math.pi, math.pi)),
                    "t_start": int(t_start),
                    "label": wp_labels[k],
                })

            scene._real_asset_pastes = pastes
            scene._route_plan_real_classes = list(classes)
            scene._route_plan_real_reface = bool(reface)
            scene._route_plan_real_waypoint_labels = list(wp_labels)
            scene._route_plan_real_n_turns = int(n_turns)
            # Mirror the synthetic stash so distractor placement / debug viz
            # see the same waypoint geometry. Dims here are placeholder
            # (asset bbox already lives on _real_asset_pastes); the value
            # only matters for distractor segment-clearance.
            scene._route_plan_box_path_centers = [c.clone() for c in centers]
            scene._route_plan_box_path_dims = [
                tuple(float(v) for v in a["bbox_dims"].tolist())
                for a in wp_assets
            ]
            scene._route_plan_box_classes = list(classes)
            scene._route_plan_reface = bool(reface)

            # Distractors: avoid every walked segment so the prompt
            # "Go forward until [wp_k]" stays coherent. Walked index list
            # mirrors _sample_distractor_boxes route_segments at L6272:
            # reface = [0] + range(2, N), non-reface = range(N).
            other_labels = [l for l in bank.labels() if l not in wp_labels]
            if other_labels:
                if reface:
                    walked = [0] + list(range(2, n_waypoints))
                else:
                    walked = list(range(n_waypoints))
                segments = []
                for i in range(len(walked) - 1):
                    a, b = walked[i], walked[i + 1]
                    p1 = centers[a].detach().clone().to(torch.float32)
                    p2 = centers[b].detach().clone().to(torch.float32)
                    segments.append((p1, p2, max(wp_radii[a], wp_radii[b])))
                q25_d, q75_d = _scene_iqr_aabb(scene)
                placed_centers = list(centers)
                placed_radii = list(wp_radii)
                n_distract = rng.randint(self._real_asset_distract_min,
                                         self._real_asset_distract_max)
                for _ in range(n_distract):
                    d_label = rng.choice(other_labels)
                    d_asset = bank.sample(d_label, rng, inflation_frac=inflation_frac)
                    r_d = float(d_asset["bbox_dims"].max().item()) / 2.0 + 0.20
                    d_center = None
                    for _try in range(20):
                        cand = _sample_collision_free_center(
                            placed_centers, placed_radii, r_d, rng,
                            aabb_min=q25_d, aabb_max=q75_d,
                        )
                        if cand is None:
                            continue
                        too_close = False
                        for (rp1, rp2, r_wp) in segments:
                            seg = rp2 - rp1
                            seg_len_sq = float((seg * seg).sum().item())
                            if seg_len_sq < 1e-9:
                                closest = rp1
                            else:
                                t_along = float(((cand - rp1) * seg).sum().item()) / seg_len_sq
                                t_along = max(0.0, min(1.0, t_along))
                                closest = rp1 + t_along * seg
                            dist_to_seg = float((cand - closest).norm().item())
                            if dist_to_seg < r_d + r_wp + 0.05:
                                too_close = True
                                break
                        if too_close:
                            continue
                        d_center = cand
                        break
                    if d_center is None:
                        continue
                    d_asset_obj = d_asset
                    d_spread = int(d_asset_obj["frame_indices"].max().item()
                                   - d_asset_obj["frame_indices"].min().item())
                    d_t_max = max(0, int(scene.n_images) - 1 - d_spread)
                    d_t_start = rng.randint(0, d_t_max) if d_t_max > 0 else 0
                    placed_centers.append(d_center)
                    placed_radii.append(r_d)
                    scene._real_asset_pastes.append({
                        "asset": d_asset_obj,
                        "target_center": d_center,
                        "yaw_rad": float(rng.uniform(-math.pi, math.pi)),
                        "t_start": int(d_t_start),
                        "label": d_label,
                    })
            return [], []

        return None, None

    def _sample_box_path(self, scene, rng, n_turns,
                          d_min_override=None, d_max_override=None):
        """Class-conditional box-path sampler for route_plan_*_box.

        Generates ``n_turns + 2`` box centers forming a navigable path:
          m0 (start) → m1 (wp1 / facing target) → m2 (wp2) → ... → m_{N+1} (goal)

        Turn classes are drawn by construction (no rejection loop):
          - Turn 1 (at m1): uniform over {Left, Right, Back}.
          - Turn k >= 2 (at m_k): uniform over {Left, Right}.
        "Back" is reserved for turn 1 because mid-path U-turns re-traverse a
        waypoint just visited, which is incoherent with the "go forward until"
        wording.

        Angles sampled strictly inside each class sector (30° deadbands at
        60/120/150° match the sampling margins used by _horizontal_turn), so
        every sampled class matches the classifier's verdict by construction.

        d_{min,max}_override: optional per-call step bounds. Used by the
        real-asset variant to enlarge the step so adjacent waypoints clear
        their bounding-sphere radii. When None, defaults scale with n_turns.

        Returns: (centers, classes) where centers is a list of n_turns+2
        [3] tensors in world-centered Cartesian, and classes is a list of
        n_turns class strings ("Turn Left" / "Turn Right" / "Turn Back").
        Returns (None, None) if no path fits in the scene bounds after retries.
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

        # Per-segment step range scaled by n_turns so total path stays ~6 m
        # regardless of N. Keeps markers within a coherent cluster on the
        # canvas even though we no longer enforce scene-bbox containment.
        if d_min_override is not None:
            d_min = float(d_min_override)
            d_max = float(d_max_override) if d_max_override is not None else d_min + 0.5
        else:
            d_min = 0.6
            d_max = max(d_min + 0.1, min(3.0, 6.0 / (n_turns + 1)))

        # Pre-commit the target class sequence UP FRONT so the outer
        # containment-retry loop doesn't distort the accepted distribution
        # across classes. (Back paths double back toward the start and fit
        # the scene bbox more often than L/R paths, so resampling the class
        # per attempt would bias the accepted distribution toward Back.)
        # Non-reface: no Back anywhere — the first [please fill in] sits at
        # WP1 *after* walking m0→m1, so Back would force WP2 back through (or
        # just past) the start, which reads as incoherent navigation.
        target_classes = [rng.choice(["Turn Left", "Turn Right"])]
        for _ in range(n_turns - 1):
            target_classes.append(rng.choice(["Turn Left", "Turn Right"]))

        def _theta_for(cls):
            if cls == "Turn Left":
                return rng.uniform(math.radians(60), math.radians(120))
            if cls == "Turn Right":
                return -rng.uniform(math.radians(60), math.radians(120))
            # Turn Back: ±(150°..180°)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            return sign * rng.uniform(math.radians(150), math.radians(180))

        for _outer in range(100):
            m0 = torch.tensor([
                rng.uniform(q25[0].item(), q75[0].item()),
                rng.uniform(q25[1].item(), q75[1].item()),
                rng.uniform(q25[2].item(), q75[2].item()),
            ], dtype=torch.float32)
            # Initial heading in xz-plane (y is up/gravity).
            ang0 = rng.uniform(0, 2 * math.pi)
            hx, hz = math.cos(ang0), math.sin(ang0)
            centers = [m0]
            d1 = rng.uniform(d_min, d_max)
            m1 = m0 + torch.tensor([hx * d1, 0.0, hz * d1], dtype=torch.float32)
            centers.append(m1)

            for cls in target_classes:
                theta = _theta_for(cls)
                c_t, s_t = math.cos(theta), math.sin(theta)
                hx_new = c_t * hx - s_t * hz
                hz_new = s_t * hx + c_t * hz
                hx, hz = hx_new, hz_new
                d = rng.uniform(d_min, d_max)
                m_next = centers[-1] + torch.tensor(
                    [hx * d, 0.0, hz * d], dtype=torch.float32,
                )
                centers.append(m_next)

            # Route-plan markers are synthetic OBBs on the panoramic canvas.
            # Only filter the degenerate near-origin case; no upper bound —
            # _world_to_spherical accepts any depth > 0.1 and the model's
            # depth encoder saturates safely beyond 100 m. Scene-bbox
            # containment is NOT required — the canvas covers the full
            # sphere and far markers still render cleanly.
            if not all(float(c.norm().item()) > 0.1 for c in centers):
                continue
            return centers, list(target_classes)

        return None, None

    def _sample_box_path_reface(self, scene, rng, n_turns,
                                 d_min_override=None, d_max_override=None):
        """Class-conditional box-path sampler for route_plan_*_reface_box.

        Adds a "facing anchor" m_face off the walked path so step 1 of the
        VSI-style template is a [please fill in] instead of a "Go forward
        until <wp1>". Real VSI Route Plan questions include both forms —
        facing==wp1 (covered by _sample_box_path) and facing!=wp1 (this
        sampler) — so training on both lets the model learn the start-turn
        semantics.

        Layout: m0 (start) → m_face (placed along the initial facing
        direction, NOT on the forward path) → [start-turn at m0] → m1 (wp1)
        → [path-turn at m1] → m2 → ... → m_{N+1} (goal).

        Turn classes pre-committed by construction:
          - Start-turn (at m0):   uniform over {Left, Right, Back}.
          - Path-turn 1 (at m1):  uniform over {Left, Right, Back} — same
            rationale as turn 1 in _sample_box_path (prior "waypoint" is
            the start, a Back here is a single U-turn leg, not mid-path
            re-traversal).
          - Path-turn k >= 2:     uniform over {Left, Right}.

        Returns: (centers, classes) where
          centers = [m0, m_face, m1, m2, ..., m_{N+1}]  (length n_turns + 3)
          classes = [start_turn, path_turn_1, ..., path_turn_N]  (len n_turns + 1)
        Returns (None, None) if no path fits the scene bounds after retries.
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

        # Reface paths have one extra waypoint (m_face) so divisor is
        # (n_turns + 2). Total path stays ~6 m for all N. Real-asset
        # callers pass d_*_override to scale the step by drawn radii.
        if d_min_override is not None:
            d_min = float(d_min_override)
            d_max = float(d_max_override) if d_max_override is not None else d_min + 0.5
        else:
            d_min = 0.6
            d_max = max(d_min + 0.1, min(3.0, 6.0 / (n_turns + 2)))

        # Pre-commit class sequence: start-turn (3-way) + all path-turns
        # (2-way). Same rationale as _sample_box_path for not resampling per
        # retry. Back is allowed ONLY as the start-turn (robot standing at
        # m0 facing m_face, reorients to head toward m1) — a Back on any
        # walked waypoint would send the path back through a prior point.
        start_class = rng.choice(["Turn Left", "Turn Right", "Turn Back"])
        path_classes = [rng.choice(["Turn Left", "Turn Right"])]
        for _ in range(n_turns - 1):
            path_classes.append(rng.choice(["Turn Left", "Turn Right"]))

        def _theta_for(cls):
            if cls == "Turn Left":
                return rng.uniform(math.radians(60), math.radians(120))
            if cls == "Turn Right":
                return -rng.uniform(math.radians(60), math.radians(120))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            return sign * rng.uniform(math.radians(150), math.radians(180))

        for _outer in range(100):
            m0 = torch.tensor([
                rng.uniform(q25[0].item(), q75[0].item()),
                rng.uniform(q25[1].item(), q75[1].item()),
                rng.uniform(q25[2].item(), q75[2].item()),
            ], dtype=torch.float32)
            # Initial heading (the facing direction) in xz-plane.
            ang_face = rng.uniform(0, 2 * math.pi)
            hx, hz = math.cos(ang_face), math.sin(ang_face)
            d_face = rng.uniform(d_min, d_max)
            m_face = m0 + torch.tensor(
                [hx * d_face, 0.0, hz * d_face], dtype=torch.float32,
            )

            # Apply start-turn at m0: rotate heading from "toward m_face" to
            # "toward m1", then walk to m1.
            theta0 = _theta_for(start_class)
            c0, s0 = math.cos(theta0), math.sin(theta0)
            hx, hz = c0 * hx - s0 * hz, s0 * hx + c0 * hz
            d1 = rng.uniform(d_min, d_max)
            m1 = m0 + torch.tensor([hx * d1, 0.0, hz * d1], dtype=torch.float32)

            path_centers = [m1]
            for cls in path_classes:
                theta = _theta_for(cls)
                c_t, s_t = math.cos(theta), math.sin(theta)
                hx, hz = c_t * hx - s_t * hz, s_t * hx + c_t * hz
                d = rng.uniform(d_min, d_max)
                m_next = path_centers[-1] + torch.tensor(
                    [hx * d, 0.0, hz * d], dtype=torch.float32,
                )
                path_centers.append(m_next)

            centers = [m0, m_face] + path_centers  # length n_turns + 3

            # No scene-bbox containment: synthetic markers only need a valid
            # projection depth (see non-reface sampler above for rationale).
            if not all(float(c.norm().item()) > 0.1 for c in centers):
                continue
            return centers, [start_class] + list(path_classes)

        return None, None
