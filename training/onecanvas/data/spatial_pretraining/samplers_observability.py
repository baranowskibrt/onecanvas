"""ObservabilityMixin: spatial-pretraining samplers."""

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


class ObservabilityMixin:
    def _sample_object_class_appearance_order_real(self, scene, rng):
        """4-way appearance-order MCQ over real-asset pastes.

        Two T designs (gated on self._appearance_real_uniform_t):
          * Bucket design (legacy, default): each asset confined to its own
            non-overlapping T bucket of n_imgs/N frames with a 1-frame gap.
            Each asset preserves its natural T-spread within the bucket
            (paste-time t_spread_cap compresses if it overflows).
          * Uniform design (when the flag is True): consecutive
            t_starts {k, k+1, ..., k+N-1}, each asset's per-patch T uniform
            over [t_start, n_imgs-1] with one patch forced to t_start. Heavy
            T-overlap; min(T) is the only differentiating signal. Mirrors
            synthetic appearance_order_box.

        N distinct-class assets, GT permutation uniform across N! options;
        question asks first-time appearance order; answer A/B/C/D.
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng, force_tight=True)
        N = int(self._appearance_n_boxes)  # default 4 to match synthetic
        if bank is None or len(bank.labels()) < N:
            return None, None

        n_imgs = int(scene.n_images)
        if self._appearance_real_uniform_t:
            if n_imgs < N:
                return None, None
        else:
            # Bucket design needs N buckets of >= 1 frame plus (N-1) gaps.
            gap = 1
            if n_imgs < N + (N - 1) * gap:
                return None, None
            bucket_size = (n_imgs - (N - 1) * gap) // N
            if bucket_size < 1:
                return None, None

        # Sample N distinct classes; one asset each.
        classes = rng.sample(bank.labels(), N)
        assets = [bank.sample(c, rng, inflation_frac=inflation_frac) for c in classes]

        # Random class-to-bucket permutation: perm[i] is the index (in
        # ``classes``) of the class assigned to the i-th (earliest-first)
        # bucket. The GT order is therefore [classes[perm[0]], ...,
        # classes[perm[N-1]]].
        perm = list(range(N))
        rng.shuffle(perm)

        q25, q75 = _scene_iqr_aabb(scene)
        full_aabb = _scene_p5_p95_aabb(scene)

        # T-start assignment depends on design.
        if self._appearance_real_uniform_t:
            # Consecutive {k, k+1, ..., k+N-1}, earliest-first; assets spread
            # to n_imgs-1 with paste-time uniform T override.
            k = rng.randrange(0, n_imgs - (N - 1))
            t_starts_sorted = [k + i for i in range(N)]

        # Outer restart: a bad early placement can box the IQR and prevent
        # later assets from fitting. Reshuffle the bucket->class permutation
        # and retry from scratch a few times before giving up.
        pastes = None
        for _restart in range(5):
            placed_centers: list = []
            placed_radii: list = []
            attempt_pastes: list = []
            success = True
            for bucket_idx, class_idx in enumerate(perm):
                asset = assets[class_idx]
                label = classes[class_idx]

                if self._appearance_real_uniform_t:
                    t_start = t_starts_sorted[bucket_idx]
                    paste_t_extra = {"t_force_uniform_to_n_imgs": True}
                else:
                    t_lo = bucket_idx * (bucket_size + gap)
                    t_hi = t_lo + bucket_size - 1
                    t_start = rng.randint(t_lo, t_hi)
                    t_cap = max(0, t_hi - t_start)  # spread budget within bucket
                    paste_t_extra = {"t_spread_cap": int(t_cap)}

                r_sphere = float(asset["bbox_dims"].max().item()) / 2.0 + 0.20
                center = _sample_collision_free_center(
                    placed_centers, placed_radii, r_sphere, rng,
                    aabb_min=q25, aabb_max=q75,
                    fallback_aabb=full_aabb,
                )
                if center is None:
                    success = False
                    break
                placed_centers.append(center)
                placed_radii.append(r_sphere)
                attempt_pastes.append({
                    "asset": asset,
                    "target_center": center,
                    "yaw_rad": float(rng.uniform(-math.pi, math.pi)),
                    "t_start": int(t_start),
                    "label": label,
                    **paste_t_extra,
                })
            if success:
                pastes = attempt_pastes
                break
            rng.shuffle(perm)
        if pastes is None:
            return None, None

        scene._real_asset_pastes = pastes
        # Canonical class list (used to render the prompt's "given these N
        # objects" prefix) and the GT permutation (used by the QA branch to
        # build the correct option + 3 random-permutation distractors).
        # GT tracks the (possibly reshuffled-by-restart) final perm.
        scene._appearance_order_real_classes = list(classes)
        scene._appearance_order_real_gt_order = [classes[perm[i]] for i in range(N)]

        # Distractor pastes of OTHER classes (0..3) so the canvas isn't a
        # closed-world "only the named objects" scene; mirrors size_real.
        # Distractors get random T_start over the full frame range (no bucket
        # alignment) since they're not part of the appearance-order GT.
        other_labels = [l for l in bank.labels() if l not in classes]
        if other_labels:
            placed_centers = [p["target_center"] for p in pastes]
            placed_radii = [
                float(p["asset"]["bbox_dims"].max().item()) / 2.0 + 0.20
                for p in pastes
            ]
            n_distract = rng.randint(self._real_asset_distract_min,
                                     self._real_asset_distract_max)
            for _ in range(n_distract):
                d_label = rng.choice(other_labels)
                d_asset = bank.sample(d_label, rng, inflation_frac=inflation_frac)
                r_d = float(d_asset["bbox_dims"].max().item()) / 2.0 + 0.20
                d_center = _sample_collision_free_center(
                    placed_centers, placed_radii, r_d, rng,
                    aabb_min=q25, aabb_max=q75,
                    fallback_aabb=full_aabb,
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

    def _sample_object_class_visibility_from_pose_real(self, scene, rng):
        """Real-asset version of visibility_from_pose.

        Two real-asset pastes (viewer pose + target object, distinct classes)
        with a synthetic axis-aligned occluder OBB between them. The occluder
        stays synthetic by user policy: real scene objects (chairs, tables,
        ...) are usually visible, so making the blocker a real object would
        bias the prior toward "yes" in a way that real photographs do not.
        Visibility math is identical to the synthetic version: sample query
        points within the target asset's axis-aligned bbox envelope (taken
        as bbox_dims/2 in the asset-local frame, conservative w.r.t. yaw)
        and count rays from viewer-center to query that escape the occluder
        AABB. The 0.10-visible threshold for "yes" carries over.

        Question placeholders (0): viewer + target referenced by class name.

        Stores ``scene._visibility_real_answer`` ("yes" / "no"),
        ``scene._visibility_real_label_viewer``,
        ``scene._visibility_real_label_target``.
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        if bank is None or len(bank.labels()) < 2:
            return None, None

        lat_t, lon_t, depth_t = scene.latitude, scene.longitude, scene.depth
        cos_lat = torch.cos(lat_t)
        pts_all = torch.stack([
            depth_t * cos_lat * torch.sin(lon_t),
            depth_t * torch.sin(lat_t),
            depth_t * cos_lat * torch.cos(lon_t),
        ], dim=-1)
        q25 = pts_all.quantile(0.25, dim=0)
        q75 = pts_all.quantile(0.75, dim=0)

        def sample_interior():
            return torch.tensor([
                rng.uniform(q25[0].item(), q75[0].item()),
                rng.uniform(q25[1].item(), q75[1].item()),
                rng.uniform(q25[2].item(), q75[2].item()),
            ], dtype=torch.float32)

        def random_direction():
            theta = rng.uniform(0.0, 2.0 * math.pi)
            u = rng.random()
            phi = math.acos(1.0 - 2.0 * u)
            return torch.tensor([
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            ], dtype=torch.float32)

        def segment_aabb_intersects(p1, p2, center, half_ext):
            t_near, t_far = -float("inf"), float("inf")
            for a in range(3):
                p1a = float(p1[a].item())
                da = float(p2[a].item() - p1a)
                ca = float(center[a].item())
                ha = float(half_ext[a])
                if abs(da) < 1e-6:
                    if not (ca - ha <= p1a <= ca + ha):
                        return False
                    continue
                t1 = (ca - ha - p1a) / da
                t2 = (ca + ha - p1a) / da
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > t_near:
                    t_near = t1
                if t2 < t_far:
                    t_far = t2
            return t_near < t_far and t_near < 1.0 and t_far > 0.0

        def visible_frac(p1, p2_center, p2_half, occ_center, occ_half,
                         n_samples=60):
            p2cx = float(p2_center[0].item())
            p2cy = float(p2_center[1].item())
            p2cz = float(p2_center[2].item())
            p2hx, p2hy, p2hz = p2_half
            hits = 0
            for _ in range(n_samples):
                q = torch.tensor([
                    rng.uniform(p2cx - p2hx, p2cx + p2hx),
                    rng.uniform(p2cy - p2hy, p2cy + p2hy),
                    rng.uniform(p2cz - p2hz, p2cz + p2hz),
                ], dtype=torch.float32)
                if not segment_aabb_intersects(p1, q, occ_center, occ_half):
                    hits += 1
            return hits / n_samples

        def cross3(a, b):
            ax, ay, az = (float(a[0].item()), float(a[1].item()),
                          float(a[2].item()))
            bx, by, bz = (float(b[0].item()), float(b[1].item()),
                          float(b[2].item()))
            return torch.tensor([
                ay * bz - az * by,
                az * bx - ax * bz,
                ax * by - ay * bx,
            ], dtype=torch.float32)

        target_label = "yes" if rng.random() < 0.5 else "no"
        yes_mode = None
        if target_label == "yes":
            yes_mode = "clear" if rng.random() < 0.5 else "partial"

        for _outer in range(60):
            l_view, l_tgt = rng.sample(bank.labels(), 2)
            asset_view = bank.sample(l_view, rng, inflation_frac=inflation_frac)
            asset_tgt = bank.sample(l_tgt, rng, inflation_frac=inflation_frac)
            tgt_dims = asset_tgt["bbox_dims"].float().tolist()
            # Use asset half-extents in OBB-local frame as the visibility
            # query domain. Conservative w.r.t. yaw (slightly underestimates
            # the rotated envelope's worst-case extent), matches the way
            # _append_real_object_assets paints the asset around its center.
            tgt_half = tuple(float(d) / 2.0 for d in tgt_dims)
            tgt_max_half = max(tgt_half)

            for _inner in range(30):
                p1_try = sample_interior()
                d_dist = rng.uniform(2.5, 5.0)
                direction = random_direction()
                p2_try = p1_try + d_dist * direction

                if abs(float(direction[1].item())) < 0.9:
                    up = torch.tensor([0.0, 1.0, 0.0])
                else:
                    up = torch.tensor([1.0, 0.0, 0.0])
                perp1 = up - (up * direction).sum() * direction
                perp1 = perp1 / perp1.norm().clamp(min=1e-6)
                perp2 = cross3(direction, perp1)
                perp2 = perp2 / perp2.norm().clamp(min=1e-6)

                # Scale occluder roughly to the target's half-extent so it
                # can plausibly occlude. Bounded to the synthetic range so
                # the math (block_margin etc.) stays well-behaved.
                s = max(0.30, min(1.20, 1.4 * tgt_max_half))
                r_occ = math.sqrt(3.0) * s / 2.0
                airgap = 0.05
                # Asset bounding spheres for separation from the occluder.
                r_view = float(asset_view["bbox_dims"].max().item()) / 2.0
                r_tgt = float(asset_tgt["bbox_dims"].max().item()) / 2.0
                t_min = (r_view + r_occ + airgap) / d_dist
                t_max = 1.0 - (r_tgt + r_occ + airgap) / d_dist
                lo = max(t_min, 0.25)
                hi = min(t_max, 0.75)
                if hi - lo < 0.05:
                    continue
                t_along = rng.uniform(lo, hi)

                theta = rng.uniform(0.0, 2.0 * math.pi)
                perp_dir = math.cos(theta) * perp1 + math.sin(theta) * perp2

                block_center = 0.5 * s
                block_margin = max(0.02, tgt_max_half * t_along)

                if target_label == "no":
                    upper = block_center - 1.2 * block_margin
                    if upper <= 0.01:
                        continue
                    perp_off = rng.uniform(0.0, upper)
                elif yes_mode == "clear":
                    base = block_center + 1.5 * block_margin
                    perp_off = rng.uniform(base + 0.3, base + 1.8)
                else:  # yes partial
                    perp_off = rng.uniform(
                        block_center - 0.5 * block_margin,
                        block_center + 0.5 * block_margin,
                    )

                occ_c = (p1_try + t_along * (p2_try - p1_try)
                         + perp_off * perp_dir)
                occ_half = (s / 2.0, s / 2.0, s / 2.0)
                vf = visible_frac(p1_try, p2_try, tgt_half, occ_c, occ_half)

                accept = False
                if target_label == "no":
                    accept = vf < 0.05
                elif yes_mode == "clear":
                    accept = vf >= 0.95
                else:
                    accept = 0.10 <= vf <= 0.80
                if not accept:
                    continue

                # Synthetic axis-aligned occluder OBB.
                self._sample_multi_box(
                    scene, rng,
                    n_boxes=1,
                    center_hints=[occ_c],
                    dim_hints=[(s, s, s)],
                    rotation_mode="aligned",
                )

                # Real-asset pastes for viewer + target.
                def _make_paste(label, asset, center):
                    return make_asset_paste(scene, rng, label, asset, center)

                scene._real_asset_pastes = [
                    _make_paste(l_view, asset_view, p1_try),
                    _make_paste(l_tgt, asset_tgt, p2_try),
                ]
                scene._visibility_real_answer = target_label
                scene._visibility_real_label_viewer = l_view
                scene._visibility_real_label_target = l_tgt

                # Distractors: avoid the p1->p2 line of sight (mirrors
                # _sample_distractor_boxes visibility block at L6394) so
                # they cannot occlude rays the GT visible_frac assumed
                # only the synthetic occluder blocked.
                other_labels = [l for l in bank.labels()
                                if l not in (l_view, l_tgt)]
                if other_labels:
                    r_p2_diag = math.sqrt(sum(d * d for d in tgt_dims)) / 2.0
                    placed_centers = [p1_try, p2_try]
                    placed_radii = [
                        float(asset_view["bbox_dims"].max().item()) / 2.0 + 0.20,
                        float(asset_tgt["bbox_dims"].max().item()) / 2.0 + 0.20,
                    ]
                    seg = p2_try - p1_try
                    seg_len_sq = float((seg * seg).sum().item())
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
                                aabb_min=q25, aabb_max=q75,
                            )
                            if cand is None:
                                continue
                            if seg_len_sq < 1e-9:
                                closest = p1_try
                            else:
                                t_along = float(((cand - p1_try) * seg).sum().item()) / seg_len_sq
                                t_along = max(0.0, min(1.0, t_along))
                                closest = p1_try + t_along * seg
                            dist_to_seg = float((cand - closest).norm().item())
                            if dist_to_seg < r_d + r_p2_diag + 0.05:
                                continue
                            d_center = cand
                            break
                        if d_center is None:
                            continue
                        placed_centers.append(d_center)
                        placed_radii.append(r_d)
                        scene._real_asset_pastes.append(
                            _make_paste(d_label, d_asset, d_center)
                        )
                return [], []

        return None, None

    def _sample_visibility_from_pose(self, scene, rng):
        """Two-marker yes/no visibility probe with a synthetic third box.

        Places three synthetic boxes on the panoramic canvas:
          - p1: observer pin (0.10 m)
          - p2: target (0.30 m)
          - a third "distractor" box that may or may not occlude the
            line of sight from p1 to p2.

        The third box is ALWAYS present so the box count (3) does not leak
        the label. Visibility is a function of what fraction of p2 is
        reachable from p1 without the ray being blocked by the third box:

          visible_frac = Pr[q ~ uniform(p2_aabb)][
              segment p1 -> q does not intersect third_box_aabb
          ]

          label = "yes" iff visible_frac >= 0.10, else "no"

        Partial occlusion (>=10% of p2 still reachable) counts as visible.

        Shortcut-free sampling: blocker side ``s``, along-segment position
        ``t_along``, and segment length ``d_dist`` are drawn from a SINGLE
        shared distribution across both labels, so their marginal does not
        reveal the answer. The only label-conditional parameter is the
        perpendicular offset ``perp_off`` of the blocker center from the
        line of sight. In the small-angle approximation the blocker covers
        p2 fully iff ``perp_off < s/2 - 0.15 * t_along``, partially iff
        ``perp_off`` straddles ``s/2``, and not at all iff
        ``perp_off > s/2 + 0.15 * t_along``:
          * "no"           — perp_off well inside the full-block region.
          * "yes partial"  — perp_off straddles the boundary so only part
                             of p2's angular extent is covered.
          * "yes clear"    — perp_off well outside the partial band.

        50/50 yes/no balance; "yes" cases split ~50/50 between clear and
        partial.

        Stashes on scene:
          _visibility_answer: "yes" | "no"
          _visibility_p1_idx: 0   (index into _multi_box_* lists)
          _visibility_p2_idx: 1
          _visibility_occluder_idx: 2

        Returns (patch_indices [2], frame_indices [2]). The third box is
        painted on the canvas as synthetic OBB points but does not get an
        inline marker in the prompt text.
        """
        lat_t, lon_t, depth_t = scene.latitude, scene.longitude, scene.depth
        cos_lat = torch.cos(lat_t)
        pts_all = torch.stack([
            depth_t * cos_lat * torch.sin(lon_t),
            depth_t * torch.sin(lat_t),
            depth_t * cos_lat * torch.cos(lon_t),
        ], dim=-1)  # [N_valid, 3]
        q25 = pts_all.quantile(0.25, dim=0)
        q75 = pts_all.quantile(0.75, dim=0)

        def sample_interior():
            return torch.tensor([
                rng.uniform(q25[0].item(), q75[0].item()),
                rng.uniform(q25[1].item(), q75[1].item()),
                rng.uniform(q25[2].item(), q75[2].item()),
            ], dtype=torch.float32)

        def random_direction():
            theta = rng.uniform(0.0, 2.0 * math.pi)
            u = rng.random()
            phi = math.acos(1.0 - 2.0 * u)
            return torch.tensor([
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            ], dtype=torch.float32)

        def segment_aabb_intersects(p1, p2, center, half_ext):
            """True iff segment p1->p2 intersects the axis-aligned box."""
            t_near, t_far = -float("inf"), float("inf")
            for a in range(3):
                p1a = float(p1[a].item())
                da = float(p2[a].item() - p1a)
                ca = float(center[a].item())
                ha = float(half_ext[a])
                if abs(da) < 1e-6:
                    if not (ca - ha <= p1a <= ca + ha):
                        return False
                    continue
                t1 = (ca - ha - p1a) / da
                t2 = (ca + ha - p1a) / da
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > t_near:
                    t_near = t1
                if t2 < t_far:
                    t_far = t2
            return t_near < t_far and t_near < 1.0 and t_far > 0.0

        def visible_frac(p1, p2_center, p2_half, occ_center, occ_half, n_samples=60):
            """Fraction of p1->q rays (q uniform in p2 AABB) NOT blocked by occ."""
            p2cx = float(p2_center[0].item())
            p2cy = float(p2_center[1].item())
            p2cz = float(p2_center[2].item())
            p2hx, p2hy, p2hz = p2_half
            hits = 0
            for _ in range(n_samples):
                q = torch.tensor([
                    rng.uniform(p2cx - p2hx, p2cx + p2hx),
                    rng.uniform(p2cy - p2hy, p2cy + p2hy),
                    rng.uniform(p2cz - p2hz, p2cz + p2hz),
                ], dtype=torch.float32)
                if not segment_aabb_intersects(p1, q, occ_center, occ_half):
                    hits += 1
            return hits / n_samples

        def cross3(a, b):
            ax, ay, az = float(a[0].item()), float(a[1].item()), float(a[2].item())
            bx, by, bz = float(b[0].item()), float(b[1].item()), float(b[2].item())
            return torch.tensor([
                ay * bz - az * by,
                az * bx - ax * bz,
                ax * by - ay * bx,
            ], dtype=torch.float32)

        target_label = "yes" if rng.random() < 0.5 else "no"
        yes_mode = None
        if target_label == "yes":
            yes_mode = "clear" if rng.random() < 0.5 else "partial"

        p1_sel, p2_sel = None, None
        occ_center, occ_side = None, None
        p2_half_ext = (0.15, 0.15, 0.15)

        for _ in range(150):
            p1_try = sample_interior()
            # Shared across both labels — no label-conditional d_dist leak.
            d_dist = rng.uniform(2.5, 5.0)
            direction = random_direction()
            p2_try = p1_try + d_dist * direction

            # Orthonormal basis perpendicular to segment direction.
            if abs(float(direction[1].item())) < 0.9:
                up = torch.tensor([0.0, 1.0, 0.0])
            else:
                up = torch.tensor([1.0, 0.0, 0.0])
            perp1 = up - (up * direction).sum() * direction
            perp1 = perp1 / perp1.norm().clamp(min=1e-6)
            perp2 = cross3(direction, perp1)
            perp2 = perp2 / perp2.norm().clamp(min=1e-6)

            # Shared size and along-segment position. The marginal over
            # (s, t_along, d_dist) is IDENTICAL for "yes" and "no"; only
            # perp_off below depends on the label.
            s = rng.uniform(0.30, 0.70)
            r_occ = math.sqrt(3.0) * s / 2.0
            airgap = 0.05
            # Keep the occluder's bounding sphere clear of p1 (r≈0.087) and
            # p2 (r≈0.260) so _sample_multi_box's non-overlap check passes.
            t_min = (0.087 + r_occ + airgap) / d_dist
            t_max = 1.0 - (0.260 + r_occ + airgap) / d_dist
            lo = max(t_min, 0.25)
            hi = min(t_max, 0.75)
            if hi - lo < 0.05:
                continue
            t_along = rng.uniform(lo, hi)

            theta = rng.uniform(0.0, 2.0 * math.pi)
            perp_dir = math.cos(theta) * perp1 + math.sin(theta) * perp2

            # Small-angle geometry: blocker angular half-width from p1 is
            # s/(2*t*d), p2 angular half-width is 0.15/d. Full block iff
            # perp_off < s/2 - 0.15*t_along; full clear iff
            # perp_off > s/2 + 0.15*t_along. Partial straddles s/2.
            block_center = 0.5 * s
            block_margin = max(0.02, 0.15 * t_along)

            if target_label == "no":
                # Well inside the full-block region.
                upper = block_center - 1.2 * block_margin
                if upper <= 0.01:
                    # Blocker too small at this t_along to reliably occlude.
                    continue
                perp_off = rng.uniform(0.0, upper)
            elif yes_mode == "clear":
                # Well outside the partial band so segment cannot clip box.
                base = block_center + 1.5 * block_margin
                perp_off = rng.uniform(base + 0.3, base + 1.8)
            else:  # yes partial
                # Straddles s/2: blocker edge cuts into p2's angular extent.
                perp_off = rng.uniform(block_center - 0.5 * block_margin,
                                        block_center + 0.5 * block_margin)

            occ_c = p1_try + t_along * (p2_try - p1_try) + perp_off * perp_dir
            occ_half = (s / 2.0, s / 2.0, s / 2.0)
            vf = visible_frac(p1_try, p2_try, p2_half_ext, occ_c, occ_half)

            if target_label == "no":
                if vf < 0.05:
                    p1_sel, p2_sel = p1_try, p2_try
                    occ_center, occ_side = occ_c, (s, s, s)
                    break
            elif yes_mode == "clear":
                if vf >= 0.95:
                    p1_sel, p2_sel = p1_try, p2_try
                    occ_center, occ_side = occ_c, (s, s, s)
                    break
            else:  # partial
                if 0.10 <= vf <= 0.80:
                    p1_sel, p2_sel = p1_try, p2_try
                    occ_center, occ_side = occ_c, (s, s, s)
                    break

        if p1_sel is None:
            return None, None

        p1_dims = (0.10, 0.10, 0.10)
        p2_dims = (0.30, 0.30, 0.30)
        patch_indices, frame_indices = self._sample_multi_box(
            scene, rng, n_boxes=3,
            center_hints=[p1_sel, p2_sel, occ_center],
            dim_hints=[p1_dims, p2_dims, occ_side],
            rotation_mode="aligned",
        )
        if patch_indices is None:
            return None, None

        scene._visibility_answer = target_label
        scene._visibility_p1_idx = 0
        scene._visibility_p2_idx = 1
        scene._visibility_occluder_idx = 2
        # Only p1 and p2 get inline markers in the prompt; the third box
        # is painted on the canvas as synthetic OBB points without a
        # marker token so the prompt length is constant across labels.
        return patch_indices[:2], frame_indices[:2]
