"""Data collator for supervised 3D scene QA training."""

import itertools
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from .dataset_utils import pad_and_cat


@dataclass
class FlattenedDataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    # to_flatten = ["input_ids_raw", "input_ids", "position_ids_raw", "position_ids", "attention_mask_raw", "attention_mask", "labels"]
    to_flatten = ["input_ids", "labels", "attention_mask",]
    data_packing: bool = False

    # Keys whose values are kept as Python lists (not stacked/padded).
    _list_keys = frozenset({
        "answer", "question", "question_type", "images", "scene_id",
        "_dataloader_retries",
        # Per-sample reproduction metadata (idx, seed, task, patch_indices, ...).
        # Geometric probing dataset stamps this so log lines can be reproduced.
        "_repro",
        # Per-sample visualizer extras (center_point, multi-box metadata, etc.).
        # Geometric probing dataset stamps this; training ignores it.
        "_debug",
        # Projected-path keys — variable-length per sample; the forward pass
        # will left-pad and stack them itself.
        "projected_input_ids", "projected_position_ids",
        "projected_attention_mask", "projected_labels",
        "projected_embeds", "projected_aux_layers", "projected_depth_bins",
        "projected_ray_dirs",
        "projected_inline_patch_indices_local",
        # Prompt-only keys for generation (excludes answer tokens)
        "projected_input_ids_prompt", "projected_position_ids_prompt",
        "projected_attention_mask_prompt", "rope_deltas_prompt",
        # Inline patch markers — per-sample variable length, used by the
        # geometric probing dataset to copy real features from specific patches
        # at marker positions.
        "inline_patch_indices",
        # Inline patch marker TOKEN positions (per-sample variable length). Kept
        # as a list so mixed / variable-marker-count batches don't try to stack
        # ragged tensors; forward slices [b] and .tolist()s. The toolcall_marker
        # branch always emits these (the toolcall_points matched branch does not).
        "inline_patch_positions",
        # toolcall_marker track: per-sample marker points (call frame, [K,3]) and
        # resolved stash feature rows ([K, N_layers, D]). model.forward PATH B
        # pastes one canvas token per marker after reproject_scene, then points
        # the inline placeholders at the appended rows.
        "marker_points",
        "marker_features",
        # toolcall_marker v2: inline text-marker + canvas-twin marker TOKENS.
        # Per-sample ragged: positions [M], features [M, N_layers, D], is_canvas
        # [M] bool, points [M, 3] (call frame). forward slices [b] and hands
        # them to prepare_batch's marker_tokens input.
        "marker_tok_positions",
        "marker_tok_features",
        "marker_tok_is_canvas",
        "marker_tok_points",
        # One canvas per observation: per-sample [K, 4] block table and the
        # block each marker token binds to. Ragged across samples.
        "canvas_blocks",
        "marker_tok_block",
        # Optional per-marker T-axis override (SpatialPretrainingDataset shuffles
        # marker T for spatial tasks to remove the source-frame shortcut).
        "inline_patch_t_indices",
        # Direct per-marker T values (integers). Used by the marker-stash
        # path where there is no real canvas patch to index into for T;
        # values are mapped through the same T-band scaling the canvas uses.
        "inline_patch_t_values",
        # Marker-stash overlay: for each sample, a list of canvas patch
        # indices whose features should be replaced with scene-agnostic
        # stash features (instead of whatever the visual tower would produce
        # for that position). Variable length per sample.
        "stash_overlay_indices",
        "stash_overlay_features",
        # Per-sample canvas-strip flag (True for tasks whose scene context
        # would leak the answer, e.g. route_plan_*).
        "canvas_obb_only",
        # patch_exists probe: canvas patch indices to zero after feature copy.
        "hide_source_patches",
        # box_size probe: synthetic patch geometry for scene extension.
        "synthetic_patch_spherical", "synthetic_patch_feature_source",
        # Multi-box distance probes (dist_box / rel_dist_box*): per-box feature
        # sources and point counts for slicing the flat synthetic_patch_spherical.
        "synthetic_patch_feature_sources_per_box", "synthetic_patch_box_sizes",
        # appearance_order_box: per-point T override for the synthetic box
        # body rows, so each box is visible across Uniform{T_min_k, ..., T_max}
        # frames instead of being pinned at T=0.
        "synthetic_patch_t_overrides",
        # box_floor_area family: per-point scene-patch index (variable length
        # per sample) for random-texture slab features.
        "synthetic_patch_per_point_sources",
        # Real-object asset paste rows for the live PATH B path (2026-08-16):
        # per-sample ragged tensors ([K, N_layers, C] features, [K, 3]
        # lat/lon/depth, [K] T) precomputed by the dataset because the
        # forward has neither the asset bank nor an rng. forward slices [b]
        # and concatenates onto the reprojected scene after strip + synthetic
        # append. Before this the slow path served real_* tasks an EMPTY
        # canvas (the append existed only on the stash fast path).
        "real_paste_embeds", "real_paste_spherical", "real_paste_frame_index",
    })

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        first_instance = instances[0]
        projection_done = first_instance.get("projection_done", False)

        # Use union of all keys so heterogeneous samples (e.g. a ConcatDataset
        # of SpatialPretrainingDataset + SceneQADataset) don't drop keys.
        all_keys: set = set()
        for ins in instances:
            all_keys.update(ins.keys())

        for key in all_keys:
            # Flag — just propagate the bool.
            if key == "projection_done":
                batch[key] = projection_done
                continue

            # 1. Handle text/metadata and projected-path per-sample lists.
            #    Only collect from instances that actually have this key.
            if key in self._list_keys:
                batch[key] = [ins[key] for ins in instances if key in ins]
                continue

            values = [ins[key] for ins in instances if key in ins]
            if not values:
                continue
            # 2. Handle specific flattened keys
            if key in self.to_flatten:

                if key in ["attention_mask"]:
                    if self.data_packing:
                        attention_mask = list(
                            itertools.chain(
                                *(
                                    instance["attention_mask"].squeeze(0).squeeze(0)
                                    for instance in instances
                                    if "attention_mask" in instance
                                )
                            )
                        )
                        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
                        batch[key] = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
                    else:
                        # Pad with 0 so padding positions are masked out (not attended to)
                        # NOT TRUNCATED TO model_max_length. This used to slice
                        # every flattened key to the cap, which silently decapitated
                        # any sample longer than it -- and the tail of a tool-loop
                        # episode is its final call and its answer, so an over-long
                        # chain trained on a question with the conclusion cut off,
                        # with nothing in the log to say so. A sample that does not
                        # fit should announce itself by failing, not by quietly
                        # becoming a different sample (owner 2026-09-22: "If it
                        # crashes because some trace is super long then it will but
                        # at least we'll know").
                        batch[key] = pad_and_cat(values, pad_value=0)
                        batch[key] = batch[key].squeeze(0)

                else:
                    if self.data_packing:
                        batch[key] = torch.cat(values, dim=2)
                    else:
                        batch[key] = pad_and_cat(values)
                        batch[key] = batch[key][0]

            else:
                # Standard stacking for keys not in to_flatten (like pixel_values)
                batch[key] = torch.stack(values)

        return batch
