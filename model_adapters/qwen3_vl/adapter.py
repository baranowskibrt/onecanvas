"""Qwen3-VL adapter: convert ReprojectedScene -> Qwen3-VL training sample.

This module handles all Qwen-specific concerns:
- Position scaling (lon/lat/frame -> MRoPE [3, N] format)
- Sequence splicing (replace dummy image region with projected tokens)
- Depth embedding preparation
- Inline-patch token insertion
"""

import math
from functools import partial

import torch

from ..base import VLMAdapterConfig, VLMAdapter


def _scale_positions(scene, config):
    """Map raw (lon, lat, frame_index) to Qwen MRoPE [3, N_valid] = (T, H, W).

    W = longitude and H = latitude are each mapped into the full
    [0, rope_pos_range] span (the shipped checkpoints were trained with
    use_isotropic_rope=False, i.e. no half-scale on H). T = normalized mean
    source-frame index in [0, temporal_max_range].
    """
    # `or 100.0`: back-compat coercion. Old checkpoints' resolved_config.json
    # recorded rope_pos_range=0.0 (the former field default, meaning "unset"),
    # and run_benchmarks copies it back in. 0 must mean 100, not zeroed positions.
    rope_pos_range = config.get("rope_pos_range", 100.0) or 100.0

    # W = longitude: [-pi, pi] -> [0, rope_pos_range]
    w_pos = (scene.longitude + math.pi) / (2 * math.pi) * rope_pos_range
    # H = latitude: [-pi/2, pi/2] -> [0, rope_pos_range]
    h_pos = (scene.latitude + math.pi / 2) / math.pi * rope_pos_range
    # T = normalized source-frame index -> [0, temporal_max_range], or, with
    # temporal_raw_frame_index, the frame index itself: no divisor, so a canvas
    # that gains a frame appends a T instead of moving every token's T.
    if config.get("temporal_raw_frame_index", False):
        t_pos = scene.frame_index.to(h_pos.dtype)
    else:
        max_range = config.get("temporal_max_range", 100.0)
        t_pos = scene.frame_index / max(scene.n_images - 1, 1) * max_range

    return torch.stack([t_pos, h_pos, w_pos], dim=0)  # [3, N_valid]


def _prepare_depth_embedding(scene, config):
    """Prepare depth embedding tensor from ReprojectedScene.

    Returns depth_bins tensor or None.
    """
    # Depth is fed to the shipped cartesian_fourier encoder as raw metric
    # depths (meters); it is disabled otherwise (mode left at the "off"
    # sentinel when use_depth_embedding is off).
    if config.get("depth_embed_mode", "off") != "cartesian_fourier":
        return None
    # Recover depth via the same log round-trip the reprojection path uses so
    # the encoder inputs match bit-for-bit (float32 precision).
    #
    # NO de_min floor here: the cartesian_fourier encoder is well-defined down
    # to depth 0 (it builds xyz = raw_depth * ray_dir, no division by depth),
    # so flooring near-anchor patches to de_min would corrupt geometry when the
    # canvas is centered on/near a surface. The internal (shipped) code path
    # returns unclamped here; a `.clamp(min=de_min)` was re-introduced by
    # accident in release commit 6bfc8b6 (the cf/binned branch collapse) and
    # broke bit-exact reproduction for patches with radial depth < de_min.
    depth_scaled = torch.log(scene.depth) * 10.0
    return torch.exp(depth_scaled / 10.0)


def _ray_dirs_from_lat_lon(latitude, longitude):
    """Compute unit ray direction in the (x_c, y_c, z_c) frame from spherical coords.

    Matches scene_reprojection.py:76-84:
        x_c = depth * cos(lat) * sin(lon)
        y_c = depth * sin(lat)
        z_c = depth * cos(lat) * cos(lon)
    Dividing by depth gives the unit direction. Result is unit-norm by construction.
    """
    cos_lat = torch.cos(latitude)
    return torch.stack([
        cos_lat * torch.sin(longitude),  # rx
        torch.sin(latitude),              # ry
        cos_lat * torch.cos(longitude),   # rz
    ], dim=-1).float()


def prepare_batch(
    scene,              # ReprojectedScene
    input_ids,          # [1, seq_len]
    attention_mask,     # [1, seq_len]
    labels,             # [1, seq_len]
    first_idx,          # int — first image_pad token
    last_idx,           # int — last image_pad token
    config,             # dict with adapter settings
    inline_patch_positions=None,      # list[int] — positions of inline patch markers in original input_ids
    inline_patch_indices=None,  # list[int] — global canvas patch index per inline marker
    inline_patch_t_indices=None, # list[int] — optional per-marker T-axis override: the MRoPE T value for each marker is read from this canvas-patch index instead of the marker's own source patch. Removes the source-frame shortcut for spatial probe tasks. Same length as inline_patch_indices.
    inline_patch_t_values=None,       # list[int] — optional DIRECT per-marker T values (0..N_virtual_frames-1). Takes precedence over inline_patch_t_indices. Used by the probe stash path where marker feature content comes from a scene-agnostic stash and there is no canvas patch with a meaningful source-frame T to read; the caller passes raw T integers which are mapped through the same T-band scaling the canvas uses.
    hide_source_patches=None,          # list[int] — canvas patch indices to zero (patch_exists probe)
    marker_tokens=None,                # dict — toolcall_marker track (v2): inline
                                       # text-marker + canvas-twin marker tokens with
                                       # EXPLICIT marker features. See the marker block.
):
    """Convert a ReprojectedScene into a Qwen3-VL training sample.

    Replaces the dummy image_pad region [first_idx, last_idx] with exactly
    N_valid projected tokens (+ optional inline patch markers).  No budget
    calculation needed — any dummy image size works.

    Inline patch markers: text-side references (at ``inline_patch_positions``)
    to specific canvas patches (``inline_patch_indices``), used by the
    spatial-pretraining curriculum. Each marker copies its source patch's visual
    features, depth, ray direction, and angle into a new appended row, with MRoPE
    position T = source frame index, H/W = source patch canvas position.
    """
    N_valid = scene.n_valid
    _img_tok_id = config.get("image_pad_token_id", 151655)

    # Inline patch markers are activated purely by the presence of patch indices.
    use_inline_patch = (
        inline_patch_indices is not None
        and len(inline_patch_indices) > 0
        and inline_patch_positions is not None
        and len(inline_patch_positions) == len(inline_patch_indices)
    )

    # --- Projected position IDs for image tokens ---
    img_position_ids = _scale_positions(scene, config)  # [3, N_valid]

    # --- Splice: replace dummy image_pad region with N_valid image tokens ---
    ids, attn, lbl = input_ids[0].long(), attention_mask[0].long(), labels[0].long()
    pre  = slice(0, first_idx)
    post = slice(last_idx + 1, None)
    # All fresh tensors below must live on the same device as ids so the later
    # torch.cat(parts_*) and new_pos_ids assignments (mixing with scene tensors,
    # which share ids' device in every caller) don't cross devices. On the CPU
    # dataloader path this is cpu; on the live inference pipeline it is cuda.
    dev = ids.device

    img_ids  = torch.full((N_valid,), _img_tok_id, dtype=ids.dtype, device=dev)
    img_attn = torch.ones(N_valid, dtype=attn.dtype, device=dev)
    img_lbl  = torch.full((N_valid,), -100, dtype=lbl.dtype, device=dev)

    # ---- Inline patch marker setup (text-side patch refs) ----
    n_inline = 0
    inline_t_cam_per_marker = None      # [n_inline] T value for each marker
    inline_patch_local_idx = None      # int tensor of marker positions in NEW input_ids
    inline_patch_sel = None             # [n_inline] global patch indices (patch-marker mode)
    # The splice replaces [first_idx..last_idx] with N_valid canvas tokens, so
    # any inline marker position p > last_idx shifts by:
    #     new_p = p - (last_idx - first_idx + 1) + N_valid
    inline_shift = N_valid - (last_idx - first_idx + 1)
    if use_inline_patch:
        # Sanitize patch indices: clamp to scene's available patch count.
        inline_patch_sel = torch.tensor(
            inline_patch_indices, dtype=torch.long).clamp(0, max(N_valid - 1, 0))
        n_inline = inline_patch_sel.shape[0]

        # T position for each marker: by default, the source frame index of
        # the referenced patch. img_position_ids was already computed via
        # _scale_positions, so we read the T row directly to get the same
        # temporal encoding (normalized, scaled, etc.) the canvas patches use.
        #
        # If the caller provides inline_patch_t_indices, read T from
        # THOSE canvas patches instead — used by GeometricProbingDataset to
        # decorrelate a marker's T-axis from its own source frame (spatial
        # tasks) while still keeping H/W coincident with the real source.
        #
        # inline_patch_t_values takes precedence when provided: the probe
        # stash path has no real canvas to index into (marker features come
        # from a scene-agnostic stash), so T arrives as raw integers in
        # [0, N_virtual_frames) and we apply the same T-band scaling the
        # canvas patches use.
        if (inline_patch_t_values is not None
                and len(inline_patch_t_values) == n_inline):
            _n_virtual = int(config.get("curriculum_virtual_num_frames", 32))
            _t_vals = torch.tensor(inline_patch_t_values, dtype=torch.float32)
            if config.get("temporal_raw_frame_index", False):
                inline_t_cam_per_marker = _t_vals   # same convention as the canvas
            else:
                _max_range = config.get("temporal_max_range", 100.0)
                inline_t_cam_per_marker = _t_vals / max(_n_virtual - 1, 1) * _max_range
        else:
            if (inline_patch_t_indices is not None
                    and len(inline_patch_t_indices) == n_inline):
                t_patch_sel = torch.tensor(
                    inline_patch_t_indices, dtype=torch.long,
                ).clamp(0, max(N_valid - 1, 0))
            else:
                t_patch_sel = inline_patch_sel
            inline_t_cam_per_marker = img_position_ids[0, t_patch_sel]  # [n_inline]

        inline_patch_local_idx = torch.tensor(
            [int(p) + inline_shift for p in inline_patch_positions], dtype=torch.long
        )

    # ---- Marker token setup (toolcall_marker v2: inline text-marker + canvas-twin) ----
    # marker_tokens carries ONE entry per marker TOKEN (a place op contributes TWO,
    # text marker then canvas twin, sharing one marker feature):
    #   positions: [M] placeholder positions in the ORIGINAL input_ids
    #   features:  [M, N_layers, D] EXPLICIT marker rows (full layer stack, so a twin
    #              threads the SAME multi-layer aux path a canvas patch does)
    #   is_canvas: [M] bool  (False = text marker, True = canvas twin)
    #   points:    [M, 3] the marker's call-frame point (canvas twins use it)
    # Text marker: natural text MRoPE + no geometry (ray 0 -> constant 3D-PE, no leak).
    # Canvas twin: W/H from the point's angles, T = last frame's T, metric 3D-PE from
    # depth*ray. Both bind to their shared marker by content; nothing binds through the
    # twin's canvas position ids (identity + recency ride the text marker). This is the
    # inline path made content-EXPLICIT: NOT the append-to-canvas paste (which would
    # leak future markers to earlier op lines under whole-episode teacher forcing).
    n_markers = 0
    mk_local_idx = mk_is_canvas = mk_feats = mk_depth = mk_ray = None
    mk_w = mk_h = mk_t_last = None
    use_markers = (marker_tokens is not None
                   and len(marker_tokens.get("positions") or []) > 0)
    if use_markers:
        from onecanvas.data.spatial_pretraining._common import (
            _real_asset_paste_matrix, _world_to_spherical)
        mk_positions = [int(p) for p in marker_tokens["positions"]]
        n_markers = len(mk_positions)
        mk_local_idx = torch.tensor(
            [p + inline_shift for p in mk_positions], dtype=torch.long, device=dev)
        mk_feats = marker_tokens["features"]
        if not torch.is_tensor(mk_feats):
            mk_feats = torch.as_tensor(mk_feats)
        mk_feats = mk_feats.to(dev)                       # [M, N_layers, D]
        mk_is_canvas = torch.as_tensor(
            [bool(c) for c in marker_tokens["is_canvas"]], dtype=torch.bool, device=dev)
        mk_points = marker_tokens["points"]
        if not torch.is_tensor(mk_points):
            mk_points = torch.as_tensor(mk_points, dtype=torch.float32)
        mk_points = mk_points.to(dtype=torch.float32, device=dev)   # [M, 3]
        # Canvas-twin geometry: SAME zero-yaw paste matrix + spherical as
        # marker_paste.append_marker_rows (loader already applied per-sample yaw).
        _Mpaste = _real_asset_paste_matrix(0.0).to(dev)
        _inter = mk_points @ _Mpaste.T                    # [M, 3] canvas-intermediate
        mk_lat, mk_lon, mk_depth_all, _mk_valid = _world_to_spherical(_inter)
        mk_ray = _ray_dirs_from_lat_lon(mk_lat, mk_lon)   # [M, 3] unit
        mk_ray[~mk_is_canvas] = 0.0                       # text markers: no direction
        mk_depth = mk_depth_all.clone()
        mk_depth[~mk_is_canvas] = 1.0                     # text: ray 0 -> xyz 0 anyway
        _rope_pos_range = config.get("rope_pos_range", 100.0) or 100.0
        mk_w = (mk_lon + math.pi) / (2 * math.pi) * _rope_pos_range
        mk_h = (mk_lat + math.pi / 2) / math.pi * _rope_pos_range
        # T = last frame's scaled T (the max canvas T); constant across twins.
        mk_t_last = (img_position_ids[0].max() if N_valid > 0
                     else torch.tensor(0.0, device=dev))

    # Assemble: pre-text | image | post-text-with-inline-patch-markers
    parts_ids  = [ids[pre], img_ids, ids[post]]
    parts_attn = [attn[pre], img_attn, attn[post]]
    parts_lbl  = [lbl[pre], img_lbl, lbl[post]]

    new_input_ids = torch.cat(parts_ids)
    new_attn_mask = torch.cat(parts_attn)
    new_labels    = torch.cat(parts_lbl)
    actual_len    = new_input_ids.shape[0]

    # --- Build position IDs from scratch ---
    #   [0 .. first_idx-1]            pre-image text (all 3 MRoPE dims identical)
    #   [first_idx .. +N_valid]       image tokens (projected spherical coords)
    #   [remainder]                   post-image text (sequential), with inline patch
    #                                  markers receiving a source-patch RoPE override
    new_pos_ids = torch.zeros(3, actual_len, dtype=torch.float32, device=dev)

    if first_idx > 0:
        pre_pos = torch.arange(first_idx, dtype=torch.float32, device=dev)
        new_pos_ids[:, :first_idx] = pre_pos.unsqueeze(0)

    new_pos_ids[:, first_idx:first_idx + N_valid] = img_position_ids

    vis_end = first_idx + N_valid

    if vis_end < actual_len:
        post_len = actual_len - vis_end
        post_pos = torch.arange(post_len, dtype=torch.float32, device=dev)    # [post_len]

        # Post-text starts right after the SCALAR max image position, shared
        # across all 3 MRoPE dims. Matches HF's get_rope_index, which uses
        # `llm_pos_ids_list[-1].max() + 1` for each post-image text segment.
        # The decode-step delta formula in model.py is also a single scalar
        # (`cache_pos - image_len_adjust + rope_deltas`) and must agree with
        # prefill across all dims; using per-dim post_starts here makes T/H
        # rows diverge from W in prefill but converge to W at decode, which
        # silently corrupts autoregressive generation while leaving training
        # loss intact.
        post_start = int(math.ceil(new_pos_ids[:, :vis_end].max().item())) + 1
        new_pos_ids[:, vis_end:] = float(post_start) + post_pos.unsqueeze(0)

        # Override inline patch markers' T-band with the source patch's T value,
        # and (when override_patch_rope) ALSO override H and W with the source
        # patch's canvas position so the marker is RoPE-coincident with its
        # source canvas patch. This lets the model read the marker's absolute
        # spherical (lat, lon) via the same relative offset any canvas patch
        # would use, instead of indirecting through content-similarity attention
        # to find the matching canvas patch first.
        #
        # inline_patch_override_rope=False disables the whole override:
        # markers keep their natural text-sequence position on all three RoPE
        # axes and must bind marker -> canvas patch via content-similarity
        # attention only. Ablation lever for checking whether the override
        # destroys first/second slot identity (two same-token markers collapse
        # to the same phase except for RoPE differences the model may not decode).
        override_patch_rope = config.get("inline_patch_override_rope", True)
        if use_inline_patch and n_inline > 0 and override_patch_rope:
            new_pos_ids[0, inline_patch_local_idx] = inline_t_cam_per_marker
            new_pos_ids[1, inline_patch_local_idx] = img_position_ids[1, inline_patch_sel]
            new_pos_ids[2, inline_patch_local_idx] = img_position_ids[2, inline_patch_sel]

        # Canvas TWINS get canvas MRoPE (W/H from the point's angles, T = last
        # frame's T); text MARKERS keep their natural post-text arange position.
        # The twin occupies a text-sequence slot (the arange counted it), so
        # tokens after it increment normally -- markers display canvas position
        # ids but do not disturb the text position counter (same convention the
        # inline override follows).
        if use_markers and bool(mk_is_canvas.any()):
            _cidx = mk_local_idx[mk_is_canvas]
            new_pos_ids[0, _cidx] = mk_t_last.to(new_pos_ids.dtype)
            new_pos_ids[1, _cidx] = mk_h[mk_is_canvas].to(new_pos_ids.dtype)
            new_pos_ids[2, _cidx] = mk_w[mk_is_canvas].to(new_pos_ids.dtype)

    rope_delta = int(math.ceil(new_pos_ids.max().item())) + 1 - actual_len

    # --- Depth embedding ---
    depth_bins = _prepare_depth_embedding(scene, config)

    # --- Ray directions (unit vectors in scene-center frame) for the depth embed ---
    # Always computed: cheap to derive from (lat, lon) and consumed by the
    # cartesian_fourier depth encoder (raw xyz = depth * ray_dir).
    ray_dirs = _ray_dirs_from_lat_lon(scene.latitude, scene.longitude)  # [N_valid, 3]

    # --- Extend embeds/aux_layers/depth/angles/ray_dirs in INPUT_IDS image_pad order ---
    # The order MUST match the splice + post-text image_pad order:
    #     canvas (N_valid) | inline patch markers (n_inline)
    # `masked_scatter` at model.forward() maps embed rows to input_ids image_pad
    # positions in order. Any deviation silently scrambles which content lands
    # at which token (and which token sees the depth/angle/ray contributions).
    embeds = scene.embeds
    aux_layers = scene.aux_layers
    C = embeds.shape[-1]

    # ---- Inline patch markers (post-text references) ----
    # Patch markers copy visual content (features, depth, angle, ray) from their
    # source canvas patches into new appended rows.
    if use_inline_patch and n_inline > 0:
        if config.get("curriculum_zero_visual", False):
            # Zero out visual features on marker tokens so the only signal
            # reaching the LLM is the depth/angle embedding added in forward().
            embeds = torch.cat(
                [embeds, torch.zeros(n_inline, C, dtype=embeds.dtype)], dim=0)
            aux_layers = [
                torch.cat([layer, torch.zeros(n_inline, C, dtype=layer.dtype)], dim=0)
                for layer in aux_layers
            ]
        else:
            embeds = torch.cat([embeds, scene.embeds[inline_patch_sel]], dim=0)
            aux_layers = [
                torch.cat([layer, layer[inline_patch_sel]], dim=0)
                for layer in aux_layers
            ]
        if depth_bins is not None:
            depth_bins = torch.cat([depth_bins, depth_bins[inline_patch_sel]], dim=0)
        ray_dirs = torch.cat([ray_dirs, ray_dirs[inline_patch_sel]], dim=0)

        # patch_exists probe: zero out source canvas patches AFTER their
        # features have been copied to the marker tokens above. The marker
        # retains the original features; the canvas patch becomes blank.
        if hide_source_patches is not None and len(hide_source_patches) > 0:
            hide_idx = torch.tensor(hide_source_patches, dtype=torch.long).clamp(0, N_valid - 1)
            embeds[hide_idx] = 0.0
            for layer in aux_layers:
                layer[hide_idx] = 0.0

    # ---- Marker tokens: append EXPLICIT marker rows (LAST, after any inline rows) ----
    # Row order MUST stay canvas (N_valid) | inline (n_inline) | markers (M) to match
    # the input_ids image_pad order masked_scatter consumes. This track never mixes
    # inline + markers, and marker positions arrive in text (ascending) order.
    if use_markers:
        n_layers_scene = 1 + len(aux_layers)
        if mk_feats.shape[1] != n_layers_scene:
            raise ValueError(
                f"marker features have {mk_feats.shape[1]} layers but scene expects "
                f"{n_layers_scene} (layer 0 + {len(aux_layers)} aux layers)")
        embeds = torch.cat([embeds, mk_feats[:, 0, :].to(embeds.dtype)], dim=0)
        aux_layers = [
            torch.cat([layer, mk_feats[:, li + 1, :].to(layer.dtype)], dim=0)
            for li, layer in enumerate(aux_layers)
        ]
        if depth_bins is not None:
            depth_bins = torch.cat(
                [depth_bins, mk_depth.to(depth_bins.dtype)], dim=0)
        ray_dirs = torch.cat([ray_dirs, mk_ray.to(ray_dirs.dtype)], dim=0)

    # Marker local indices for the identity embedding in forward(). Both patch
    # markers (n_inline) and toolcall markers (M) are non-scene reference tokens
    # sharing the canvas token id + (for twins) canvas MRoPE, so they get the
    # learned "I am a marker" constant. Markers occupy the rows after the canvas
    # (and after any inline patch rows).
    inline_patch_indices_local = None
    _local_parts = []
    if use_inline_patch and n_inline > 0:
        _local_parts.append(torch.arange(N_valid, N_valid + n_inline, dtype=torch.long))
    if use_markers:
        _base = N_valid + n_inline
        _local_parts.append(torch.arange(_base, _base + n_markers, dtype=torch.long))
    if _local_parts:
        inline_patch_indices_local = torch.cat(_local_parts)

    return {
        "input_ids":        new_input_ids,
        "position_ids":     new_pos_ids,
        "attention_mask":   new_attn_mask,
        "labels":           new_labels,
        "embeds":           embeds,
        "aux_layers":        aux_layers,
        "depth_bins":       depth_bins,
        "ray_dirs":         ray_dirs,
        "inline_patch_indices_local": inline_patch_indices_local,
        "rope_deltas":      torch.tensor(rope_delta, dtype=torch.long),
    }


def prepare_batch_blocks(
    scenes,             # list[ReprojectedScene], one per canvas block, in text order
    input_ids,          # [1, seq_len]
    attention_mask,     # [1, seq_len]
    labels,             # [1, seq_len]
    blocks,             # list[(first_idx, last_idx)] dummy image_pad runs, ascending
    config,             # dict with adapter settings
    marker_tokens=None, # dict as in prepare_batch, plus "block": [M] canvas block per token
):
    """Splice ONE CANVAS PER OBSERVATION into one sequence.

    The multi-observation counterpart of ``prepare_batch``, for a conversation
    whose turns each bring their own photo (a robot acting, observing, acting
    again). Every dummy image_pad run ``blocks[k]`` is replaced by exactly
    ``scenes[k].n_valid`` canvas tokens, and the text between the blocks keeps
    its order, so a token of turn k attends to canvas blocks 0..k and to none
    after it under the ordinary causal mask.

    POSITIONS FOLLOW ``prepare_batch`` BLOCK BY BLOCK. Text runs sequentially;
    a canvas block takes its own angular (T, H, W) positions; the text after a
    block resumes at the scalar maximum position so far plus one, which is the
    rule the decode-step delta formula assumes. With one block and no markers
    this reproduces ``prepare_batch`` exactly.

    TIME IS THE RUNNING FRAME INDEX. Block k's frames continue the raw frame
    index where block k-1 stopped, so a later photo reads as newer on the T
    axis and block 0 of a one-frame observation is T=0, the same as a
    single-canvas item. This needs ``temporal_raw_frame_index``; the normalised
    convention would rescale every earlier block whenever a photo is added, so
    it is refused rather than approximated.

    MARKERS BIND TO THEIR OWN BLOCK. A canvas twin takes its block's last T and
    the angles of its point; a text marker keeps its text position. Rows are
    emitted in the image_pad order of the new sequence (canvas 0, markers
    after it, canvas 1, ...), which is the order ``masked_scatter`` consumes.
    """
    if not config.get("temporal_raw_frame_index", False):
        raise ValueError(
            "prepare_batch_blocks needs temporal_raw_frame_index: under the "
            "normalised T convention adding a photo would move the T of every "
            "earlier block")
    blocks = [(int(a), int(b)) for a, b in blocks]
    if len(blocks) != len(scenes) or not blocks:
        raise ValueError(f"{len(blocks)} dummy blocks for {len(scenes)} scenes")
    for (a0, b0), (a1, _b1) in zip(blocks, blocks[1:]):
        if not (a0 <= b0 < a1):
            raise ValueError(f"canvas blocks overlap or are unordered: {blocks}")

    _img_tok_id = config.get("image_pad_token_id", 151655)
    ids, attn, lbl = input_ids[0].long(), attention_mask[0].long(), labels[0].long()
    dev = ids.device
    L = int(ids.shape[0])

    # ---- per-block canvas positions, T continuing across blocks ----
    block_pos = []
    t_offset = 0
    for scene in scenes:
        pos = _scale_positions(scene, config).to(dev)
        pos[0] = pos[0] + float(t_offset)
        block_pos.append(pos)
        t_offset += int(scene.n_images)

    # ---- markers ----
    M = 0
    mk_pos = mk_block = mk_is_canvas = mk_feats = None
    if marker_tokens is not None and len(marker_tokens.get("positions") or []) > 0:
        from onecanvas.data.spatial_pretraining._common import (
            _real_asset_paste_matrix, _world_to_spherical)
        mk_pos = [int(p) for p in marker_tokens["positions"]]
        M = len(mk_pos)
        mk_block = [int(v) for v in marker_tokens["block"]]
        if len(mk_block) != M:
            raise ValueError("marker_tokens['block'] must name one block per token")
        mk_feats = marker_tokens["features"]
        if not torch.is_tensor(mk_feats):
            mk_feats = torch.as_tensor(mk_feats)
        mk_feats = mk_feats.to(dev)
        mk_is_canvas = torch.as_tensor(
            [bool(c) for c in marker_tokens["is_canvas"]], dtype=torch.bool, device=dev)
        mk_points = marker_tokens["points"]
        if not torch.is_tensor(mk_points):
            mk_points = torch.as_tensor(mk_points, dtype=torch.float32)
        mk_points = mk_points.to(dtype=torch.float32, device=dev)
        _inter = mk_points @ _real_asset_paste_matrix(0.0).to(dev).T
        mk_lat, mk_lon, mk_depth_all, _ok = _world_to_spherical(_inter)
        mk_ray = _ray_dirs_from_lat_lon(mk_lat, mk_lon)
        mk_ray[~mk_is_canvas] = 0.0
        mk_depth = mk_depth_all.clone()
        mk_depth[~mk_is_canvas] = 1.0
        _rope = config.get("rope_pos_range", 100.0) or 100.0
        mk_w = (mk_lon + math.pi) / (2 * math.pi) * _rope
        mk_h = (mk_lat + math.pi / 2) / math.pi * _rope
        for p, k in zip(mk_pos, mk_block):
            if not (0 <= k < len(blocks)) or p <= blocks[k][1] or (
                    k + 1 < len(blocks) and p >= blocks[k + 1][0]):
                raise ValueError(
                    f"marker at {p} is bound to block {k} but does not lie "
                    f"between that block and the next: {blocks}")

    # ---- splice: text | canvas 0 | text | canvas 1 | ... | text ----
    parts_ids, parts_attn, parts_lbl = [], [], []
    shift_after = []            # cumulative length change after each block
    cursor, shift = 0, 0
    canvas_local = []
    for k, (first, last) in enumerate(blocks):
        n = int(scenes[k].n_valid)
        parts_ids += [ids[cursor:first],
                      torch.full((n,), _img_tok_id, dtype=ids.dtype, device=dev)]
        parts_attn += [attn[cursor:first],
                       torch.ones(n, dtype=attn.dtype, device=dev)]
        parts_lbl += [lbl[cursor:first],
                      torch.full((n,), -100, dtype=lbl.dtype, device=dev)]
        canvas_local.append((first + shift, first + shift + n))
        shift += n - (last - first + 1)
        shift_after.append(shift)
        cursor = last + 1
    parts_ids.append(ids[cursor:])
    parts_attn.append(attn[cursor:])
    parts_lbl.append(lbl[cursor:])
    new_input_ids = torch.cat(parts_ids)
    new_attn_mask = torch.cat(parts_attn)
    new_labels = torch.cat(parts_lbl)
    actual_len = int(new_input_ids.shape[0])

    def _local(p):
        """A text position of the original sequence, in the new sequence."""
        s = 0
        for k, (_first, last) in enumerate(blocks):
            if p > last:
                s = shift_after[k]
        return p + s

    # ---- position ids, segment by segment ----
    new_pos_ids = torch.zeros(3, actual_len, dtype=torch.float32, device=dev)
    text_start, next_text = 0, 0.0
    for k, (c0, c1) in enumerate(canvas_local):
        n_text = c0 - text_start
        if n_text > 0:
            new_pos_ids[:, text_start:c0] = next_text + torch.arange(
                n_text, dtype=torch.float32, device=dev).unsqueeze(0)
        new_pos_ids[:, c0:c1] = block_pos[k]
        next_text = float(math.ceil(new_pos_ids[:, :c1].max().item()) + 1)
        text_start = c1
    if text_start < actual_len:
        new_pos_ids[:, text_start:] = next_text + torch.arange(
            actual_len - text_start, dtype=torch.float32, device=dev).unsqueeze(0)
    if M and bool(mk_is_canvas.any()):
        for m in range(M):
            if not bool(mk_is_canvas[m]):
                continue
            k = mk_block[m]
            q = _local(mk_pos[m])
            t_last = (block_pos[k][0].max() if block_pos[k].shape[1] > 0
                      else torch.tensor(0.0, device=dev))
            new_pos_ids[0, q] = t_last
            new_pos_ids[1, q] = mk_h[m]
            new_pos_ids[2, q] = mk_w[m]
    rope_delta = int(math.ceil(new_pos_ids.max().item())) + 1 - actual_len

    # ---- rows in image_pad order: canvas k, then the markers after it ----
    order = [(blocks[k][0], "canvas", k) for k in range(len(blocks))]
    order += [(mk_pos[m], "marker", m) for m in range(M)]
    order.sort()
    first_scene = scenes[0]
    n_aux = len(first_scene.aux_layers)
    use_depth = config.get("depth_embed_mode", "off") == "cartesian_fourier"
    emb_parts, aux_parts, depth_parts, ray_parts = [], [[] for _ in range(n_aux)], [], []
    marker_rows = []
    row = 0
    for _p, kind, j in order:
        if kind == "canvas":
            scene = scenes[j]
            if len(scene.aux_layers) != n_aux:
                raise ValueError("canvas blocks disagree on the aux-layer count")
            emb_parts.append(scene.embeds)
            for li, layer in enumerate(scene.aux_layers):
                aux_parts[li].append(layer)
            if use_depth:
                depth_parts.append(_prepare_depth_embedding(scene, config))
            ray_parts.append(_ray_dirs_from_lat_lon(scene.latitude, scene.longitude))
            row += int(scene.n_valid)
        else:
            if mk_feats.shape[1] != 1 + n_aux:
                raise ValueError(
                    f"marker features have {mk_feats.shape[1]} layers but the "
                    f"canvas expects {1 + n_aux}")
            emb_parts.append(mk_feats[j:j + 1, 0, :].to(first_scene.embeds.dtype))
            for li in range(n_aux):
                aux_parts[li].append(
                    mk_feats[j:j + 1, li + 1, :].to(first_scene.aux_layers[li].dtype))
            if use_depth:
                depth_parts.append(mk_depth[j:j + 1])
            ray_parts.append(mk_ray[j:j + 1])
            marker_rows.append(row)
            row += 1
    embeds = torch.cat(emb_parts, dim=0)
    aux_layers = [torch.cat(parts, dim=0) for parts in aux_parts]
    depth_bins = (torch.cat([d.to(depth_parts[0].dtype) for d in depth_parts])
                  if use_depth else None)
    ray_dirs = torch.cat([r.float() for r in ray_parts], dim=0)

    n_pad = int((new_input_ids == _img_tok_id).sum().item())
    if n_pad != int(embeds.shape[0]):
        raise ValueError(
            f"{n_pad} image_pad tokens in the spliced sequence for "
            f"{int(embeds.shape[0])} rows; the scatter would misplace content")

    return {
        "input_ids":        new_input_ids,
        "position_ids":     new_pos_ids,
        "attention_mask":   new_attn_mask,
        "labels":           new_labels,
        "embeds":           embeds,
        "aux_layers":       aux_layers,
        "depth_bins":       depth_bins,
        "ray_dirs":         ray_dirs,
        "inline_patch_indices_local": (
            torch.tensor(marker_rows, dtype=torch.long) if marker_rows else None),
        "rope_deltas":      torch.tensor(rope_delta, dtype=torch.long),
        "canvas_blocks_local": canvas_local,
    }


def build_adapter(processor, data_args) -> VLMAdapter:
    """Build a Qwen3-VL adapter from a processor and data_args."""
    from ..qwen_mrope import get_rope_index_3
    from onecanvas.data.dataset_utils import preprocess_qwen_visual

    _tok = processor.tokenizer
    merge_size = getattr(processor.image_processor, "merge_size", 2)

    # Resolve token IDs from the tokenizer (safe across Qwen2-VL / Qwen3-VL).
    _img_pad = _tok.encode("<|image_pad|>", add_special_tokens=False)
    _vid_pad = _tok.encode("<|video_pad|>", add_special_tokens=False)
    _vis_start = _tok.encode("<|vision_start|>", add_special_tokens=False)
    _asst = _tok.encode("assistant", add_special_tokens=False)
    _im_end = _tok.encode("<|im_end|>", add_special_tokens=False)
    _obj_ref = _tok.encode("<|object_ref_start|>", add_special_tokens=False)

    _box_start = _tok.encode("<|box_start|>", add_special_tokens=False)
    _box_end = _tok.encode("<|box_end|>", add_special_tokens=False)

    config = VLMAdapterConfig(
        image_pad_token_id=_img_pad[0] if _img_pad else 151655,
        video_pad_token_id=_vid_pad[0] if _vid_pad else 151656,
        vision_start_token_id=_vis_start[0] if _vis_start else 151652,
        assistant_token_id=_asst[0] if _asst else 77091,
        im_end_token_id=_im_end[0] if _im_end else 151645,
        object_ref_start_token_id=_obj_ref[0] if _obj_ref else 151646,
        feature_prefix="qwen3_vl",
        rope_dims=3,
        spatial_merge_size=merge_size,
        chat_template_kwargs={"enable_thinking": False},
        box_start_token="<|box_start|>",
        box_end_token="<|box_end|>",
    )

    _get_rope = partial(
        get_rope_index_3,
        spatial_merge_size=merge_size,
        image_token_id=config.image_pad_token_id,
        video_token_id=config.video_pad_token_id,
        vision_start_token_id=config.vision_start_token_id,
    )

    _mask_labels = partial(
        preprocess_qwen_visual,
        asst_token_id=config.assistant_token_id,
        im_end_token_id=config.im_end_token_id,
    )

    return VLMAdapter(
        config=config,
        get_rope_index=_get_rope,
        prepare_batch=prepare_batch,
        mask_labels=_mask_labels,
    )
