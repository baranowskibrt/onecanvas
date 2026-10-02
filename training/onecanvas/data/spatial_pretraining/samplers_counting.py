"""CountingMixin: spatial-pretraining samplers."""

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


class CountingMixin:
    def _sample_object_counting(self, scene, rng):
        """Generate N non-overlapping synthetic OBBs (same shape/features) for counting.

        All OBBs share one reference patch (ref_idx) as their feature source, so the
        model sees N identical-looking objects.  OBB max-dim is capped by N_target to
        keep high-count placement tractable.  The actual placed count (≤ N_target) is
        stored in scene._count and used as the ground-truth answer.

        When ``self._counting_difficulty_mix`` is set, draws an easy/medium/hard
        difficulty per call and stochastically applies: sparse corner-only
        markers (L1, K in {2,3,4}); tight 3D packing (L2); canvas-collinear OBB
        pairs along the world-origin ray (L3, depth-only distinguished); per-OBB
        independent dims (L4); extended N range up to 25 (L5). Easy keeps the
        pre-flag code path and rng stream byte-identical.
        """
        # ------------------------------------------------------------------
        # Difficulty knobs. Easy keeps the pre-flag behaviour bit-exact by
        # consuming no extra rng.* calls before the original N_target draw.
        # ------------------------------------------------------------------
        if self._counting_difficulty_mix:
            level = rng.choices(
                ["easy", "medium", "hard"], weights=[0.50, 0.30, 0.20], k=1,
            )[0]
        else:
            level = "easy"

        sparse_K = 0           # L1: 0 = full body budget; >0 = K random corners
        pack_factor = 2.0      # L2: scales touching-sphere bound (2.0 = touching)
        per_obb_dims = False   # L4: resample dims per OBB
        los_pair_count = 0     # L3: # of canvas-collinear partners
        n_max_excl = 16        # L5: exclusive upper bound for log-uniform N draw

        if level == "medium":
            # Half of medium samples drop to 4-corner sparse markers; tight
            # packing relaxes from 2.0 toward 1.4 * (r_a + r_b).
            if rng.random() < 0.5:
                sparse_K = 4
            pack_factor = rng.uniform(1.4, 2.0)
        elif level == "hard":
            sparse_K = rng.choice([2, 3, 4])
            pack_factor = rng.uniform(0.9, 1.4)
            per_obb_dims = True
            if rng.random() < 0.5:
                los_pair_count = 1
            n_max_excl = 26  # extend to N up to 25

        # --- Count target: log-uniform over 1..(n_max_excl - 1) ---
        N_target = max(1, min(
            n_max_excl - 1,
            int(math.exp(rng.uniform(math.log(1), math.log(n_max_excl)))),
        ))

        # --- IQR of valid patch positions (same as _sample_box_size) ---
        lat_t, lon_t, depth_t = scene.latitude, scene.longitude, scene.depth
        cos_lat = torch.cos(lat_t)
        pts_all = torch.stack([
            depth_t * cos_lat * torch.sin(lon_t),
            depth_t * torch.sin(lat_t),
            depth_t * cos_lat * torch.cos(lon_t),
        ], dim=-1)
        q25 = pts_all.quantile(0.25, dim=0)
        q75 = pts_all.quantile(0.75, dim=0)

        # --- Shared seed dims (also used per-OBB when per_obb_dims=False) ---
        max_dim_cap = max(0.25, min(3.26, 1.8 / math.sqrt(N_target)))

        def _draw_dims():
            dm = math.exp(rng.uniform(math.log(0.25), math.log(max_dim_cap)))
            da = math.exp(rng.uniform(math.log(0.07), math.log(dm)))
            db = math.exp(rng.uniform(math.log(0.07), math.log(dm)))
            triple = [dm, da, db]
            rng.shuffle(triple)
            return tuple(triple)

        seed_dims = _draw_dims()
        seed_r_sphere = math.sqrt(sum(v ** 2 for v in seed_dims)) / 2.0

        ref_idx = rng.randrange(scene.n_valid)
        placed_centers = []
        placed_rotations = []
        placed_dims = []   # per-OBB dims (== seed_dims for all when per_obb_dims=False)
        placed_radii = []  # per-OBB bounding-sphere radii
        all_synth_lat, all_synth_lon, all_synth_depth = [], [], []

        _n_total_default = self._per_box_budget(N_target)

        # L3: pre-pick which OBB indices will be canvas-collinear partners of
        # an earlier-placed center. Only meaningful for N_target >= 4.
        los_partner_idx = set()
        if los_pair_count > 0 and N_target >= 4:
            for _p in range(los_pair_count):
                pi = rng.randrange(N_target // 2 + 1, N_target)
                los_partner_idx.add(pi)

        # Colinear-all mode (shortcut-buster): independent of the counting
        # difficulty mix, force every OBB past the anchor to be a colinear
        # partner so all N centers share one ray from origin. Transitively
        # works because partner_i = anchor_c * alpha_i and all earlier
        # partners are already on the anchor's ray.
        colinear_on = (
            self._colinear_centers_prob > 0.0
            and N_target >= 2
            and rng.random() < self._colinear_centers_prob
        )
        if colinear_on:
            los_pair_count = max(los_pair_count, N_target - 1)
            los_partner_idx = set(range(1, N_target))

        for k_obb in range(N_target):
            # L4 (or shared seed dims).
            dims_k = _draw_dims() if per_obb_dims else seed_dims
            r_sphere_k = math.sqrt(sum(v ** 2 for v in dims_k)) / 2.0

            # Independent rotation per instance — objects of the same type are
            # typically oriented differently (chairs, bottles, etc.).
            ax = torch.randn(3)
            ax = ax / ax.norm()
            angle = rng.uniform(0, 2 * math.pi)
            K_mat = torch.tensor(
                [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]
            )
            R = torch.eye(3) + math.sin(angle) * K_mat + (1 - math.cos(angle)) * (K_mat @ K_mat)

            # L3: canvas-collinear partner. Place along world-origin ray of an
            # already-placed center; partner and anchor share (lat, lon) and
            # are distinguishable only by depth (per-patch xyz channel).
            is_partner = (k_obb in los_partner_idx) and len(placed_centers) > 0
            if is_partner:
                anchor_idx = rng.randrange(len(placed_centers))
                anchor_c = placed_centers[anchor_idx]
                r_anchor = placed_radii[anchor_idx]
                d_anchor = float(anchor_c.norm().item())
                # Ensure 3D non-overlap by construction: required ray-distance
                # between centers = touching-sphere bound under pack_factor,
                # plus jitter so partners aren't all at the minimum gap.
                sep_min = (pack_factor / 2.0) * (r_sphere_k + r_anchor)
                sep_target = sep_min + rng.uniform(0.10, 1.00)
                alpha_delta = sep_target / max(d_anchor, 1e-3)
                sign = -1.0 if rng.random() < 0.5 else 1.0
                alpha = 1.0 + sign * alpha_delta
                # Avoid alpha <= 0 (would flip direction across origin and
                # break canvas-collinearity).
                if alpha <= 0.05:
                    alpha = 0.05

            placed = False
            for _attempt in range(50):
                if is_partner:
                    center = anchor_c * alpha
                else:
                    center = torch.tensor([
                        rng.uniform(q25[0].item(), q75[0].item()),
                        rng.uniform(q25[1].item(), q75[1].item()),
                        rng.uniform(q25[2].item(), q75[2].item()),
                    ], dtype=torch.float32)

                # Bounding-sphere non-overlap check (per-OBB radii). pack=2.0
                # recovers the original "< 2*r_sphere" touching-sphere bound
                # bit-exact when all OBBs share seed_dims.
                if any(
                    (center - c).norm().item() < (pack_factor / 2.0) * (r_sphere_k + r_other)
                    for c, r_other in zip(placed_centers, placed_radii)
                ):
                    if is_partner:
                        # Partner is geometrically determined by anchor; no
                        # re-roll possible. Skip this OBB.
                        break
                    continue

                # L1: sparse markers — emit only K random OBB corners. Bypass
                # the max(8, ...) floor in _sample_obb_surface_points so the
                # marker cluster genuinely tests xyz-adjacency clustering
                # rather than 2D blob density.
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
                    if is_partner:
                        break
                    continue

                placed_centers.append(center)
                placed_rotations.append(R)
                placed_dims.append(dims_k)
                placed_radii.append(r_sphere_k)
                all_synth_lat.append(lats[valid])
                all_synth_lon.append(lons[valid])
                all_synth_depth.append(depths[valid])
                placed = True
                break

            if not placed:
                if is_partner:
                    # Partner failure: drop just this OBB and keep going so a
                    # single bad alpha doesn't truncate the whole sequence.
                    continue
                break  # stop early; actual count < target

        N_actual = len(placed_centers)
        if N_actual == 0:
            return None, None

        scene._count = N_actual
        # Expose per-box geometry for the visualizer (not used by training).
        scene._multi_box_centers = placed_centers
        scene._multi_box_dims = placed_dims
        scene._multi_box_rotations = placed_rotations
        # Visualizer-only: record the difficulty draw + lever values so the
        # .txt dump can label each sample easy/medium/hard and show which
        # levers fired. Training code ignores these attributes.
        scene._counting_diff_level = level
        scene._counting_diff_sparse_K = int(sparse_K)
        scene._counting_diff_pack_factor = float(pack_factor)
        scene._counting_diff_per_obb_dims = bool(per_obb_dims)
        scene._counting_diff_los_pair_count = int(los_pair_count)
        scene._counting_diff_n_max_excl = int(n_max_excl)
        scene._counting_diff_N_target = int(N_target)
        scene._counting_diff_flag_on = bool(self._counting_difficulty_mix)
        scene._counting_colinear_on = bool(colinear_on)
        scene._synthetic_spherical = torch.stack([
            torch.cat(all_synth_lat),
            torch.cat(all_synth_lon),
            torch.cat(all_synth_depth),
        ], dim=-1)  # [N_total_synth, 3]
        scene._synthetic_feature_source = ref_idx

        # Hide the reference patch and all same-frame patches within 20° angular
        # distance — prevents the model from counting the original canvas patch
        # as an additional instance or being confused by visually similar neighbors.
        ref_lat = scene.latitude[ref_idx]
        ref_lon = scene.longitude[ref_idx]
        ref_frame = scene.frame_index[ref_idx]
        dlat = scene.latitude - ref_lat
        dlon = scene.longitude - ref_lon
        ang_dist = torch.sqrt(dlat ** 2 + dlon ** 2)
        _HIDE_RADIUS_RAD = math.radians(20)
        hide_mask = (scene.frame_index == ref_frame) & (ang_dist < _HIDE_RADIUS_RAD)
        scene._hide_patch_indices = hide_mask.nonzero(as_tuple=True)[0].tolist()

        return [ref_idx], [int(scene.frame_index[ref_idx].item())]

    def _sample_object_class_counting_real(self, scene, rng):
        """Count target-class real-asset instances.

        Pastes N target-class assets + M random other-class distractors onto
        the canvas. Each asset carries its own per-patch features harvested
        from a real ScanNet scene, so the model has to use shape-class
        signal rather than count identical OBB markers.

        Total placed objects is always fixed to ``curriculum_fixed_counting_total``
        (default 9). N_target ~ Uniform{1..total}, N_distract = total -
        N_target. Severs the density-vs-count correlation so the model must
        classify instances by class rather than estimate visual density.

        Sets ``scene._count`` (target-class count, distractors excluded) and
        ``scene._counting_class_label`` for the QA builder.
        """
        bank = self._real_asset_bank
        inflation_frac = self._draw_real_asset_inflation_frac(rng)
        if bank is None or not bank.labels():
            return None, None

        labels = bank.labels()
        # Per-class uniform — chair (98 assets) and microwave (4) get equal
        # selection mass. Avoids dominant-class bias that per-asset uniform
        # would introduce.
        target_label = rng.choice(labels)
        other_labels = [l for l in labels if l != target_label]

        total = int(getattr(self.data_args, "curriculum_fixed_counting_total", 9))
        # Uniform on {1..total} inclusive — N=0 excluded so the model can't
        # default to "zero". Total visual mass is constant so density cannot
        # proxy for count; the model must classify instances by class.
        N_target = rng.randint(1, total)
        N_distract = total - N_target
        primary_min, primary_max = _scene_p5_p95_aabb(scene)
        mid = (primary_min + primary_max) * 0.5
        half = (primary_max - primary_min) * 0.5 * 1.5
        fallback_min = mid - half
        fallback_max = mid + half
        placement_aabb = (primary_min, primary_max)
        placement_fallback = (fallback_min, fallback_max)

        placed_centers: list = []
        placed_radii: list = []
        scene._real_asset_pastes = []

        # Build a shuffled placement order so partial-placement failures
        # (rare under the wider AABB) drop slots proportionally rather than
        # truncating only the trailing target/distractor pool.
        if not other_labels and N_distract > 0:
            N_distract = 0  # no other classes available
        slot_order = (["target"] * N_target) + (["distract"] * N_distract)
        rng.shuffle(slot_order)

        n_target_placed = 0
        n_distract_placed = 0
        for slot in slot_order:
            if slot == "target":
                asset = bank.sample(target_label, rng, inflation_frac=inflation_frac)
                paste_label = target_label
            else:
                d_label = rng.choice(other_labels)
                asset = bank.sample(d_label, rng, inflation_frac=inflation_frac)
                paste_label = d_label
            r_sphere = float(asset["bbox_dims"].max().item()) / 2.0 + 0.20
            center = _sample_collision_free_center(
                placed_centers, placed_radii, r_sphere, rng,
                aabb_min=placement_aabb[0], aabb_max=placement_aabb[1],
                fallback_aabb=placement_fallback,
            )
            if center is None:
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            spread = int(asset["frame_indices"].max().item()
                         - asset["frame_indices"].min().item())
            t_max = max(0, int(scene.n_images) - 1 - spread)
            t_start = rng.randint(0, t_max) if t_max > 0 else 0

            placed_centers.append(center)
            placed_radii.append(r_sphere)
            scene._real_asset_pastes.append({
                "asset": asset,
                "target_center": center,
                "yaw_rad": float(yaw),
                "t_start": int(t_start),
                "label": paste_label,
            })
            if slot == "target":
                n_target_placed += 1
            else:
                n_distract_placed += 1

        if n_target_placed == 0 and n_distract_placed == 0:
            return None, None
        # N_target >= 1 is enforced by randint(1, total), so n_target_placed==0
        # only happens if every placement attempt failed (very rare). Reject.
        if n_target_placed == 0:
            return None, None

        scene._count = n_target_placed
        scene._counting_class_label = target_label
        return [], []
