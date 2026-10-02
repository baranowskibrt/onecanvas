import gc
import math
from typing import Optional, Union, Unpack

import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast, Qwen3VLModelOutputWithPast,
    TransformersKwargs)

from model_adapters.geometry_embedding import (
    DEPTH_VISUAL_STATS,  # re-exported for train.py's DepthMagnitudeCallback
    GeometryEmbeddingMixin,
)
from model_adapters.qwen3_vl.feature_extraction import split_image_features
from utils.tensor_ops import pad_and_stack


class Qwen3VL3DModel(GeometryEmbeddingMixin, Qwen3VLModel):
    # Depth + angle embedding methods (init_*, _encode_*, _stash_*, and the
    # _apply_geometry_embeddings dispatch) live on GeometryEmbeddingMixin.

    # ------------------------------------------------------------------
    # Inline patch marker embedding
    # ------------------------------------------------------------------
    def init_inline_patch_embedding(self, hidden_size):
        """Single learned [hidden_size] vector for inline patch markers.

        Added to every inline patch marker token so the model can
        distinguish "I am a reference to a canvas patch" from the canvas
        patch itself. Initialized at std=1.0 (much larger than other embeds)
        so the marker is immediately detectable.
        """
        self.inline_patch_constant_embed = torch.nn.Parameter(
            torch.zeros(hidden_size))
        torch.nn.init.normal_(self.inline_patch_constant_embed, mean=0.0, std=1.0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_deltas=None,
        labels=None,
        # Dataloader-projected path keys (set when projection was done on CPU)
        projection_done=None,
        projected_input_ids=None,
        projected_position_ids=None,
        projected_attention_mask=None,
        projected_labels=None,
        projected_embeds=None,
        projected_aux_layers=None,
        projected_depth_bins=None,
        projected_ray_dirs=None,
        projected_inline_patch_indices_local=None,
        image_len_adjust=None,
        _dataloader_retries=None,
        # PATH B keys (raw geometry + indices for live-features projection).
        # Consumed by the pre-PATH-A branch below; never forwarded to language_model.
        depths: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        image_dims: Optional[torch.Tensor] = None,
        image_token_first_idx: Optional[torch.Tensor] = None,
        image_token_last_idx: Optional[torch.Tensor] = None,
        input_seq_len: Optional[torch.Tensor] = None,
        # PATH B inline patch markers — list[torch.LongTensor], one per sample.
        # inline_patch_positions holds each marker's position in the sample's
        # ORIGINAL input_ids (before any model-side trimming); the paired
        # inline_patch_indices holds the global canvas patch indices the
        # live forward pass copies real visual features from at those positions.
        inline_patch_positions=None,
        inline_patch_indices=None,
        # Optional per-marker MRoPE T-axis override (probe shortcut removal).
        # When present, each marker's T value is taken from the canvas patch
        # at inline_patch_t_indices[b][k] instead of its own source
        # patch — decorrelating T from source-frame position.
        inline_patch_t_indices=None,
        # Direct per-marker T values (integers). Used by the marker-stash
        # path where there is no canvas patch carrying a meaningful T to read.
        # Takes precedence over inline_patch_t_indices in prepare_batch.
        inline_patch_t_values=None,
        # Marker-stash overlay: list[torch.LongTensor] of canvas patch indices
        # per sample, plus a parallel list[torch.Tensor] of stash features
        # [K, N_layers, D]. When any sample has non-empty overlay indices, the
        # visual tower call is skipped (all_features becomes zeros) and the
        # stash features are written onto scene.embeds / scene.aux_layers at
        # the specified indices before canvas_obb_only runs. This gives probe
        # samples scene-agnostic marker content for free.
        stash_overlay_indices=None,
        stash_overlay_features=None,
        # toolcall_marker track: per-sample marker points (call frame) and their
        # resolved stash feature rows. After reproject_scene we append ONE canvas
        # token per marker (LAST), then point the inline placeholders at those
        # rows. marker_points[b]: [K, 3] (centered/yawed z-up call frame, already
        # transformed by the dataloader); marker_features[b]: [K, N_layers, D].
        marker_points=None,
        marker_features=None,
        # toolcall_marker v2: inline text-marker + canvas-twin marker TOKENS
        # (supersedes the append-to-canvas paste above, which leaks future
        # markers to earlier op lines under whole-episode teacher forcing). Per
        # sample: marker_tok_positions[b] [M] placeholder positions in the
        # ORIGINAL input_ids, marker_tok_features[b] [M, N_layers, D] marker rows,
        # marker_tok_is_canvas[b] [M] bool (False=text marker, True=canvas twin),
        # marker_tok_points[b] [M, 3] call-frame points.
        marker_tok_positions=None,
        marker_tok_features=None,
        marker_tok_is_canvas=None,
        marker_tok_points=None,
        # Per-sample canvas-strip flag. When True, remove all real-scene
        # canvas patches except the marker-referenced ones (the synthetic
        # OBB points are still appended after). Set by the dataset for
        # tasks listed in curriculum_canvas_obb_only_tasks (route_plan_* by design).
        canvas_obb_only=None,
        # patch_exists probe: list[torch.LongTensor], canvas patch indices
        # to zero out after copying their features to the marker token.
        hide_source_patches=None,
        # box_size probe: synthetic patch data for scene extension in PATH B.
        synthetic_patch_spherical=None,
        synthetic_patch_feature_source=None,
        # Multi-box distance probes (dist_box / rel_dist_box*): per-box feature
        # sources and point counts for slicing synthetic_patch_spherical.
        synthetic_patch_feature_sources_per_box=None,
        synthetic_patch_box_sizes=None,
        # appearance_order_box: per-point T override for the synthetic box
        # body rows. list[torch.LongTensor], flat shape [sum(box_sizes_b)]
        # aligned to synthetic_patch_spherical rows. When set, the appended
        # scene.frame_index for synthetic rows uses these values instead of
        # zeros — each box becomes visible across a Uniform{T_min_k, ..., T_max}
        # spread so the model must aggregate and take the min T.
        synthetic_patch_t_overrides=None,
        # box_floor_area family: list[torch.LongTensor] of per-synthetic-point
        # scene-patch indices. When set, synthetic slab rows clone features
        # from these heterogeneous sources instead of one uniform ref_idx.
        # Indices reference the pre-strip scene, so features are captured
        # before any canvas_obb_only slicing.
        synthetic_patch_per_point_sources=None,
        # Real-object asset paste rows, dataset-precomputed (2026-08-16):
        # real_paste_embeds[b] [K, N_layers, C] the asset's OWN per-patch
        # features, real_paste_spherical[b] [K, 3] (lat, lon, depth),
        # real_paste_frame_index[b] [K]. PATH B appends them after strip +
        # synthetic append. Before this the live path had NO real-asset
        # append (it existed only on the dataset's stash fast path), so
        # every real_* sample evaluated through PATH B saw an empty canvas.
        real_paste_embeds=None,
        real_paste_spherical=None,
        real_paste_frame_index=None,
        # Panoramic augmentation (PATH B only — PATH A applies in dataloader).
        aug_center_override=None,  # [B, 3] or list of [3] tensors
        aug_yaw_angle=None,        # [B] tensor of yaw angles
        # ONE CANVAS PER OBSERVATION (multi-observation conversations). Per
        # sample: canvas_blocks[b] [K, 4] rows (first_idx, last_idx,
        # frame_start, frame_end) naming each dummy image_pad run in the
        # ORIGINAL input_ids and the sample's frames that make its canvas;
        # marker_tok_block[b] [M] the block each marker token binds to. Absent
        # or empty, every sample takes the single-canvas path unchanged.
        canvas_blocks=None,
        marker_tok_block=None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        # Surface dataloader retry counts to the profiling callback (if active).
        if _dataloader_retries is not None and self.training:
            _total = sum(_dataloader_retries) if isinstance(_dataloader_retries, (list, tuple)) else 0
            if _total > 0 and not getattr(self, "_retry_warning_printed", False):
                print(f"[data] dataloader_retries this batch: {_total}")
                self._retry_warning_printed = True
            # Store on model so ProfilingCallback can read it in on_step_end.
            self._last_batch_retries = _total

        prefill_stage = (
            (input_ids is not None and input_ids.shape[1] != 1) or
            (inputs_embeds is not None and inputs_embeds.shape[1] != 1) or
            # Path A: projection was done in the dataloader; treat as prefill only
            # when there is no KV cache yet.  HuggingFace generate() keeps
            # projection_done=True in model_kwargs for every decode step, so
            # without this guard path A would re-run on every generated token.
            (bool(projection_done) and past_key_values is None)
        )


        image_mask = None
        video_mask = None

        # ==================================================================
        # PRE-PATH-A — live visual encoding (use_precomputed_features=False).
        # When the dataloader returned a PATH B batch (raw pixel_values + raw
        # geometry, no precomputed projection), run the visual encoder here on
        # GPU and call reproject_scene + adapter.prepare_batch ourselves to
        # produce exactly the same projected_* lists PATH A consumes. Then
        # flip projection_done=True and fall through to PATH A unchanged.
        # ==================================================================
        if (
            prefill_stage
            and not projection_done
            and pixel_values is not None
            and image_grid_thw is not None
            and depths is not None
            # Defensive: the visual tower is always present on the 3D model, but
            # guard anyway so a stripped-down model degrades gracefully.
            and getattr(self, "visual", None) is not None
        ):
            from reprojection import reproject_scene
            from .adapter import prepare_batch as _qwen3vl_prepare_batch

            device = self.get_input_embeddings().weight.device
            cfg = getattr(self, "reprojection_config", None)
            if cfg is None:
                raise RuntimeError(
                    "Live-features path requires self.reprojection_config to be "
                    "set on the model. Train.py attaches it after "
                    "make_supervised_data_module(); make sure that wiring ran."
                )

            # Depth-resolution probe (eval only). Logged once so a run's own log
            # proves whether the knob was active — a silent default of 1 would
            # make a null result unreadable.
            _depth_downsample = int(cfg.get("depth_downsample", 1) or 1)
            if _depth_downsample > 1 and not getattr(self, "_depth_downsample_logged", False):
                self._depth_downsample_logged = True
                print(f"[depth_downsample] active: K={_depth_downsample} "
                      f"(depth grid coarsened to H_feat/K x W_feat/K, patch count unchanged)")

            # Collator stacks pixel_values to [1, N_total_patches, dim] and
            # image_grid_thw to [1, N_imgs, 3] for bs=1. Strip the leading
            # batch axis for get_image_features which wants the unbatched form.
            pv = pixel_values
            if pv.dim() == 3:
                pv = pv.reshape(-1, pv.shape[-1])
            gt = image_grid_thw
            if gt.dim() == 3:
                gt = gt.reshape(-1, 3)

            # Marker-stash path: when any sample has non-empty overlay indices,
            # skip the visual tower entirely and build a zero feature tensor of
            # the right shape. Stash features are written in per-sample after
            # reproject_scene. Works because for geometric-pretraining batches
            # every position the forward actually reads (markers + OBB feature
            # sources) gets overwritten with a stash feature; everything else
            # is stripped by canvas_obb_only.
            stash_active = (
                stash_overlay_indices is not None
                and any(
                    (
                        (idx is not None)
                        and (int(idx.numel()) if hasattr(idx, "numel") else len(idx)) > 0
                    )
                    for idx in stash_overlay_indices
                )
            )

            merge = self.visual.spatial_merge_size
            h_per = (gt[:, 1] // merge).tolist()
            w_per = (gt[:, 2] // merge).tolist()
            tokens_per_img = [h * w for h, w in zip(h_per, w_per)]
            H_feat, W_feat = h_per[0], w_per[0]
            assert all(h == H_feat and w == W_feat for h, w in zip(h_per, w_per)), (
                "live-features path requires homogeneous image shapes within a batch"
            )

            if stash_active:
                # Infer (N_layers, C) from the first non-empty stash block.
                _probe_feats = next(
                    f for f in stash_overlay_features
                    if f is not None and (
                        int(f.numel()) if hasattr(f, "numel") else len(f)
                    ) > 0
                )
                N_layers_stash = int(_probe_feats.shape[1])
                C = int(_probe_feats.shape[2])
                N_total = int(gt.shape[0])
                all_features = torch.zeros(
                    N_total, N_layers_stash, H_feat, W_feat, C,
                    dtype=torch.bfloat16,
                )
                if not getattr(self, "_stash_active_logged", False):
                    print(
                        f"[stash-forward] ViT skipped. all_features shape="
                        f"{tuple(all_features.shape)} dtype={all_features.dtype}"
                    )
                    self._stash_active_logged = True
            else:
                # Run visual encoder. Grad only when training AND the visual is
                # trainable (tune_mm_vision); matches the precomputed-path
                # behavior where features came from a frozen offline pass.
                # The self.training gate matters at inference: a plain
                # from_pretrained model (e.g. the merged release checkpoint)
                # has requires_grad=True everywhere, and enable_grad() would
                # override generate()'s outer no_grad — retaining all 27
                # vision-block activations and OOMing at high resolution.
                visual_needs_grad = self.training and any(
                    p.requires_grad for p in self.visual.parameters()
                )
                _ctx = torch.enable_grad() if visual_needs_grad else torch.no_grad()
                with _ctx:
                    image_outputs = self.get_image_features(pv, gt)
                # Mirrors scripts/precompute.py's feature reshaping.
                image_embeds_per_img, deepstack_features = split_image_features(image_outputs)
                C = image_embeds_per_img[0].shape[-1]

                base_layer = torch.stack(
                    [emb.view(H_feat, W_feat, C) for emb in image_embeds_per_img], dim=0,
                )  # [N_total, H, W, C]

                if deepstack_features is not None and len(deepstack_features) > 0:
                    layers = [base_layer]
                    for layer_tensor in deepstack_features:
                        chunks = torch.split(layer_tensor, tokens_per_img, dim=0)
                        layers.append(torch.stack(
                            [c.view(H_feat, W_feat, C) for c in chunks], dim=0,
                        ))
                    all_features = torch.stack(layers, dim=1)  # [N_total, N_layers, H, W, C]
                else:
                    all_features = base_layer.unsqueeze(1)     # [N_total, 1, H, W, C]

            # depths/poses/intrinsics/image_dims arrive as [B, N_per, ...] from
            # the collator's torch.stack on bs=1 (or homogeneous bs>1).
            B = depths.shape[0]
            N_per = depths.shape[1]
            assert B * N_per == all_features.shape[0], (
                f"image count mismatch: B={B} * N_per={N_per} != {all_features.shape[0]}"
            )

            # adapter.prepare_batch builds new tensors with CPU defaults and
            # torch.cat() requires same-device operands. The dataloader's
            # precomputed-features path runs everything on CPU; mirror that
            # here by moving features + geometry + text tokens to CPU before
            # calling reproject_scene / prepare_batch. PATH A then moves the
            # projected_* tensors back to GPU as needed (it already does
            # .to(device) on every consumer).
            #
            # NOTE: when visual_needs_grad=True (tune_mm_vision), .to("cpu")
            # is differentiable so gradients flow back into the visual encoder
            # via the CPU ops in prepare_batch. When frozen, the no_grad
            # context above already prevents grad tracking.
            # Cast geometry back to float32: HF Trainer + DeepSpeed cast every
            # floating-point batch input to the engine dtype (bf16) inside
            # _prepare_inputs, but reproject_scene's matmul expects fp32 poses
            # (the precomputed path never sees this because the dataloader
            # worker calls reproject_scene before Trainer touches the tensors).
            features_cpu      = all_features.to("cpu")
            depths_cpu        = depths.to("cpu").float()
            poses_cpu         = poses.to("cpu").float()
            intrinsics_cpu    = intrinsics.to("cpu").float()
            image_dims_cpu    = image_dims.to("cpu").float()
            input_ids_cpu     = input_ids.to("cpu")
            attention_mask_cpu = attention_mask.to("cpu")
            labels_cpu        = labels.to("cpu") if labels is not None else None

            projected_input_ids        = []
            projected_position_ids     = []
            projected_attention_mask   = []
            projected_labels           = []
            projected_embeds           = []
            projected_aux_layers        = []
            projected_depth_bins       = []
            projected_ray_dirs         = []
            projected_inline_patch_indices_local = []
            rope_deltas_list           = []

            for b in range(B):
                L_real = int(input_seq_len[b].item())
                first  = int(image_token_first_idx[b].item())
                last   = int(image_token_last_idx[b].item())

                feats_b = features_cpu[b * N_per : (b + 1) * N_per]

                # Per-sample panoramic augmentation (if provided by dataloader)
                _center_b = None
                if aug_center_override is not None:
                    if isinstance(aug_center_override, (list, tuple)) and len(aug_center_override) > b:
                        _center_b = aug_center_override[b]
                    elif hasattr(aug_center_override, 'shape') and aug_center_override.dim() == 2:
                        _center_b = aug_center_override[b]
                _yaw_b = None
                if aug_yaw_angle is not None:
                    if hasattr(aug_yaw_angle, 'shape'):
                        _yaw_b = float(aug_yaw_angle[b]) if aug_yaw_angle.numel() > 1 else float(aug_yaw_angle)
                    elif isinstance(aug_yaw_angle, (list, tuple)) and len(aug_yaw_angle) > b:
                        _yaw_b = float(aug_yaw_angle[b])

                # ---- one canvas per observation ----
                # A multi-observation sample splices one canvas block per
                # photo, all reprojected around the sample's one declared
                # origin, and none of the single-canvas extensions below
                # (stash overlay, canvas strip, synthetic or pasted rows,
                # inline patch markers) apply to it, so a sample that carries
                # both is refused rather than half-applied.
                _blocks_b = None
                if (canvas_blocks is not None and len(canvas_blocks) > b
                        and canvas_blocks[b] is not None
                        and len(canvas_blocks[b]) > 0):
                    _blocks_b = canvas_blocks[b]
                    _blocks_b = (_blocks_b.tolist() if hasattr(_blocks_b, "tolist")
                                 else [list(r) for r in _blocks_b])
                if _blocks_b is not None:
                    _extras = {
                        "stash_overlay_indices": stash_overlay_indices,
                        "inline_patch_indices": inline_patch_indices,
                        "synthetic_patch_spherical": synthetic_patch_spherical,
                        "real_paste_embeds": real_paste_embeds,
                        "hide_source_patches": hide_source_patches,
                    }
                    for _name, _v in _extras.items():
                        if _v is not None and len(_v) > b and _v[b] is not None and (
                                (int(_v[b].numel()) if hasattr(_v[b], "numel")
                                 else len(_v[b])) > 0):
                            raise ValueError(
                                f"sample {b} carries canvas_blocks and {_name}; "
                                f"the multi-canvas path does not apply "
                                f"single-canvas extensions")
                    _scenes = []
                    for _r in _blocks_b:
                        _f0, _f1 = int(_r[2]), int(_r[3])
                        _scenes.append(reproject_scene(
                            features=feats_b[_f0:_f1],
                            depths=depths_cpu[b][_f0:_f1],
                            poses=poses_cpu[b][_f0:_f1],
                            intrinsics=intrinsics_cpu[b][_f0:_f1],
                            image_dims=image_dims_cpu[b][_f0:_f1],
                            device="cpu",
                            center_override=_center_b,
                            yaw_angle=_yaw_b,
                            depth_downsample=_depth_downsample,
                        ))
                    ids_b = input_ids_cpu[b:b+1, :L_real]
                    mask_b = attention_mask_cpu[b:b+1, :L_real]
                    lbl_b = (labels_cpu[b:b+1, :L_real] if labels_cpu is not None
                             else torch.full((1, L_real), -100, dtype=torch.long))
                    _mtok_b = None
                    if (marker_tok_positions is not None
                            and len(marker_tok_positions) > b
                            and marker_tok_positions[b] is not None
                            and len(marker_tok_positions[b]) > 0):
                        def _as_list(x):
                            return x.tolist() if hasattr(x, "tolist") else list(x)
                        if marker_tok_block is None or len(marker_tok_block) <= b:
                            raise ValueError(
                                f"sample {b} has markers and canvas_blocks but "
                                f"no marker_tok_block to bind them")
                        _mtok_b = {
                            "positions": _as_list(marker_tok_positions[b]),
                            "features": marker_tok_features[b],
                            "is_canvas": _as_list(marker_tok_is_canvas[b]),
                            "points": marker_tok_points[b],
                            "block": _as_list(marker_tok_block[b]),
                        }
                    from .adapter import prepare_batch_blocks as _blocks_prepare
                    proj = _blocks_prepare(
                        scenes=_scenes, input_ids=ids_b, attention_mask=mask_b,
                        labels=lbl_b,
                        blocks=[(int(_r[0]), int(_r[1])) for _r in _blocks_b],
                        config=cfg, marker_tokens=_mtok_b)
                    projected_input_ids.append(proj["input_ids"])
                    projected_position_ids.append(proj["position_ids"])
                    projected_attention_mask.append(proj["attention_mask"])
                    projected_labels.append(proj["labels"])
                    projected_embeds.append(proj["embeds"])
                    projected_aux_layers.append(proj["aux_layers"])
                    projected_depth_bins.append(proj["depth_bins"])
                    projected_ray_dirs.append(proj.get("ray_dirs"))
                    projected_inline_patch_indices_local.append(
                        proj.get("inline_patch_indices_local"))
                    rope_deltas_list.append(proj["rope_deltas"])
                    continue

                scene = reproject_scene(
                    features=feats_b,
                    depths=depths_cpu[b],
                    poses=poses_cpu[b],
                    intrinsics=intrinsics_cpu[b],
                    image_dims=image_dims_cpu[b],
                    device="cpu",
                    center_override=_center_b,
                    yaw_angle=_yaw_b,
                    depth_downsample=_depth_downsample,
                )

                # Marker-stash overlay: when active, write scene-agnostic
                # stash features onto scene.embeds / scene.aux_layers at the
                # positions the downstream path will actually read (markers
                # + OBB feature sources + per-point sources). Everything
                # else stays zero and gets sliced away by canvas_obb_only.
                if stash_overlay_indices is not None and len(stash_overlay_indices) > b:
                    _ov_idx = stash_overlay_indices[b]
                    _ov_feats = stash_overlay_features[b]
                    _ov_n = (
                        int(_ov_idx.numel()) if hasattr(_ov_idx, "numel") else len(_ov_idx)
                    ) if _ov_idx is not None else 0
                    if _ov_n > 0 and _ov_feats is not None:
                        _idx_t = _ov_idx if torch.is_tensor(_ov_idx) else torch.tensor(
                            _ov_idx, dtype=torch.long
                        )
                        _idx_t = _idx_t.to(device=scene.embeds.device, dtype=torch.long)
                        _feats_t = _ov_feats.to(
                            device=scene.embeds.device, dtype=scene.embeds.dtype
                        )
                        # Layer 0 -> scene.embeds; layers 1..N-1 -> scene.aux_layers[l-1]
                        scene.embeds[_idx_t] = _feats_t[:, 0, :]
                        for l, layer in enumerate(scene.aux_layers):
                            if _feats_t.shape[1] > l + 1:
                                layer[_idx_t] = _feats_t[:, l + 1, :].to(layer.dtype)

                ids_b = input_ids_cpu[b:b+1, :L_real]
                mask_b = attention_mask_cpu[b:b+1, :L_real]
                if labels_cpu is not None:
                    lbl_b = labels_cpu[b:b+1, :L_real]
                else:
                    lbl_b = torch.full((1, L_real), -100, dtype=torch.long)

                # Per-sample inline patch markers (live-features path). The
                # dataloader already detected positions in the ORIGINAL input_ids;
                # here we just hand them to prepare_batch.
                if inline_patch_positions is not None and len(inline_patch_positions) > b:
                    _imp_b = inline_patch_positions[b]
                    _imp_b = _imp_b.tolist() if hasattr(_imp_b, "tolist") else list(_imp_b)
                else:
                    _imp_b = None
                if inline_patch_indices is not None and len(inline_patch_indices) > b:
                    _imp_patch_b = inline_patch_indices[b]
                    _imp_patch_b = _imp_patch_b.tolist() if hasattr(_imp_patch_b, "tolist") else list(_imp_patch_b)
                else:
                    _imp_patch_b = None
                # Direct per-marker T values (stash path uses these instead of
                # canvas-patch indexing — the stash has no canvas T to read).
                if inline_patch_t_values is not None and len(inline_patch_t_values) > b:
                    _imt_v_b = inline_patch_t_values[b]
                    if _imt_v_b is not None:
                        _imt_v_b = (
                            _imt_v_b.tolist() if hasattr(_imt_v_b, "tolist") else list(_imt_v_b)
                        )
                    if not _imt_v_b:
                        _imt_v_b = None
                else:
                    _imt_v_b = None
                if inline_patch_t_indices is not None and len(inline_patch_t_indices) > b:
                    _imt_b = inline_patch_t_indices[b]
                    if _imt_b is None:
                        _imt_b = None
                    else:
                        _imt_b = _imt_b.tolist() if hasattr(_imt_b, "tolist") else list(_imt_b)
                else:
                    _imt_b = None
                if hide_source_patches is not None and len(hide_source_patches) > b:
                    _hide_b = hide_source_patches[b]
                    _hide_b = _hide_b.tolist() if hasattr(_hide_b, "tolist") else list(_hide_b)
                else:
                    _hide_b = None

                # toolcall_marker v2 marker tokens (per sample). Built into a
                # single dict handed to prepare_batch; None when the sample has
                # no markers (every non-marker task).
                _mtok_b = None
                if (marker_tok_positions is not None
                        and len(marker_tok_positions) > b
                        and marker_tok_positions[b] is not None
                        and len(marker_tok_positions[b]) > 0):
                    def _as_list(x):
                        return x.tolist() if hasattr(x, "tolist") else list(x)
                    _mtok_b = {
                        "positions": _as_list(marker_tok_positions[b]),
                        "features": marker_tok_features[b],
                        "is_canvas": _as_list(marker_tok_is_canvas[b]),
                        "points": marker_tok_points[b],
                    }

                # Per-sample canvas-strip flag: True for tasks listed in
                # curriculum_canvas_obb_only_tasks (route_plan_*). Overrides the
                # global curriculum_single_patch_canvas cfg when True; cfg still
                # applies if the per-sample flag is not set.
                _strip_b = False
                if canvas_obb_only is not None and len(canvas_obb_only) > b:
                    _sb = canvas_obb_only[b]
                    _strip_b = bool(_sb.item() if hasattr(_sb, "item") else _sb)

                # Capture per-point feature sources BEFORE stripping: the
                # box_floor_area family samples random scene-patch indices
                # that reference the pre-strip scene, so we have to pull
                # their embeddings now or they'll be sliced away below.
                _per_pt_embeds_saved = None
                _per_pt_aux_layers_saved = None
                if (
                    synthetic_patch_per_point_sources is not None
                    and len(synthetic_patch_per_point_sources) > b
                ):
                    _pps = synthetic_patch_per_point_sources[b]
                    if _pps is not None and (
                        _pps.numel() if hasattr(_pps, "numel") else len(_pps)
                    ) > 0:
                        _pps = _pps.to(
                            device=scene.embeds.device, dtype=torch.long
                        )
                        _per_pt_embeds_saved = scene.embeds[_pps].clone()
                        _per_pt_aux_layers_saved = [
                            layer[_pps].clone() for layer in scene.aux_layers
                        ]

                # Single-patch canvas: keep only the marker-referenced
                # patch(es) so the canvas shrinks to 1 token per marker.
                # Also preserve marker-T-override patches (inline_patch_t_indices)
                # so appearance_order_box can point marker T at a different-T
                # patch without it being filtered out.
                # visibility_from_pose: the occluder box's feature-source patch
                # is referenced via synthetic_patch_feature_sources_per_box but
                # has no inline marker, so without this it would be sliced away
                # and the multi-box extension path below would index out of range.
                # Real-object paste samples (2026-08-16): the strip must ALSO
                # run when the sample carries real paste rows and no inline
                # markers. The real_* box families reference objects BY CLASS
                # NAME, so _imp_patch_b is empty, and gating the strip on it
                # left the WHOLE live scene underneath the pastes -- training's
                # fast path builds these canvases from an EMPTY keep-set plus
                # pastes. The path-loss probe measured the difference at 0.114
                # vs 0.498 first-turn NLL (stash vs slow, ckpt-40000). An
                # empty keep-set is legitimate here: the canvas becomes pastes
                # only, which IS the training distribution.
                _has_real_paste_b = (
                    real_paste_embeds is not None
                    and len(real_paste_embeds) > b
                    and real_paste_embeds[b] is not None
                    and real_paste_embeds[b].shape[0] > 0)
                remap = {}
                if (_strip_b or cfg.get("curriculum_single_patch_canvas", False)) and (
                        _imp_patch_b or _has_real_paste_b):
                    _keep_set = set(_imp_patch_b)
                    if _imt_b:
                        _keep_set.update(int(x) for x in _imt_b)
                    if (
                        synthetic_patch_feature_sources_per_box is not None
                        and len(synthetic_patch_feature_sources_per_box) > b
                    ):
                        _sbb_pre = synthetic_patch_feature_sources_per_box[b]
                        if _sbb_pre is not None and (
                            _sbb_pre.numel() if hasattr(_sbb_pre, "numel") else len(_sbb_pre)
                        ) > 0:
                            _keep_set.update(int(x) for x in _sbb_pre.tolist())
                    if (
                        synthetic_patch_feature_source is not None
                        and len(synthetic_patch_feature_source) > b
                    ):
                        _sfs = synthetic_patch_feature_source[b]
                        _sfs_val = int(_sfs.item() if hasattr(_sfs, "item") else _sfs)
                        if _sfs_val >= 0:
                            _keep_set.add(_sfs_val)
                    keep = sorted(_keep_set)
                    keep_t = torch.tensor(keep, dtype=torch.long)
                    scene.embeds = scene.embeds[keep_t]
                    scene.aux_layers = [layer[keep_t] for layer in scene.aux_layers]
                    scene.longitude = scene.longitude[keep_t]
                    scene.latitude = scene.latitude[keep_t]
                    scene.depth = scene.depth[keep_t]
                    scene.frame_index = scene.frame_index[keep_t]
                    scene.n_valid = len(keep)
                    remap = {old: new for new, old in enumerate(keep)}
                    _imp_patch_b = [remap[i] for i in _imp_patch_b]
                    if _imt_b:
                        _imt_b = [remap[int(i)] for i in _imt_b]
                    if _hide_b:
                        _hide_b = [remap[i] for i in _hide_b if i in remap]

                # --- Extend scene with synthetic patches (box_size / dist_box probing) ---
                _synth_b = None
                if synthetic_patch_spherical is not None and len(synthetic_patch_spherical) > b:
                    _synth_b = synthetic_patch_spherical[b]

                # Determine whether this is a multi-box sample (dist_box / rel_dist_box*)
                # or the legacy single-source sample (box_size / object_counting).
                _src_per_box_b = None
                _box_sizes_b = None
                if (
                    synthetic_patch_feature_sources_per_box is not None
                    and len(synthetic_patch_feature_sources_per_box) > b
                ):
                    _src_per_box_b = synthetic_patch_feature_sources_per_box[b]
                    if _src_per_box_b is not None and len(_src_per_box_b) == 0:
                        _src_per_box_b = None  # empty tensor → single-source path
                if (
                    synthetic_patch_box_sizes is not None
                    and len(synthetic_patch_box_sizes) > b
                ):
                    _box_sizes_b = synthetic_patch_box_sizes[b]
                    if _box_sizes_b is not None and len(_box_sizes_b) == 0:
                        _box_sizes_b = None

                # Move synthetic spherical data to the same device as scene tensors.
                if _synth_b is not None and _synth_b.shape[0] > 0:
                    _synth_b = _synth_b.to(scene.latitude.device)

                # Per-point T override for the synthetic rows (appearance_order_box).
                _t_ov_b = None
                if (
                    synthetic_patch_t_overrides is not None
                    and len(synthetic_patch_t_overrides) > b
                ):
                    _t_ov_b = synthetic_patch_t_overrides[b]
                    if _t_ov_b is not None and len(_t_ov_b) == 0:
                        _t_ov_b = None

                if _synth_b is not None and _synth_b.shape[0] > 0 and _src_per_box_b is not None:
                    # Multi-box path: each slice of synthetic points gets its own feature source.
                    offset = 0
                    all_synth_lat, all_synth_lon, all_synth_depth = [], [], []
                    all_embeds = []
                    all_aux_layers_rows = [[] for _ in scene.aux_layers]
                    for src_idx_raw, n_pts in zip(
                        _src_per_box_b.tolist(), _box_sizes_b.tolist()
                    ):
                        src_idx = remap.get(int(src_idx_raw), int(src_idx_raw))
                        pts_k = _synth_b[offset:offset + n_pts]
                        all_synth_lat.append(pts_k[:, 0])
                        all_synth_lon.append(pts_k[:, 1])
                        all_synth_depth.append(pts_k[:, 2])
                        all_embeds.append(
                            scene.embeds[src_idx].unsqueeze(0).expand(n_pts, -1).clone()
                        )
                        for li, layer in enumerate(scene.aux_layers):
                            all_aux_layers_rows[li].append(
                                layer[src_idx].unsqueeze(0).expand(n_pts, -1).clone()
                            )
                        offset += n_pts
                    n_new = offset
                    scene.latitude = torch.cat([scene.latitude] + all_synth_lat)
                    scene.longitude = torch.cat([scene.longitude] + all_synth_lon)
                    scene.depth = torch.cat([scene.depth] + all_synth_depth)
                    if _t_ov_b is not None and int(_t_ov_b.numel()) == n_new:
                        _new_t = _t_ov_b.to(
                            dtype=scene.frame_index.dtype,
                            device=scene.frame_index.device,
                        )
                    else:
                        _new_t = torch.zeros(
                            n_new,
                            dtype=scene.frame_index.dtype,
                            device=scene.frame_index.device,
                        )
                    scene.frame_index = torch.cat([scene.frame_index, _new_t])
                    scene.embeds = torch.cat([scene.embeds] + all_embeds, dim=0)
                    scene.aux_layers = [
                        torch.cat([layer] + all_aux_layers_rows[li], dim=0)
                        for li, layer in enumerate(scene.aux_layers)
                    ]
                    scene.n_valid += n_new

                elif _synth_b is not None and _synth_b.shape[0] > 0:
                    # Legacy single-source path (box_size / object_counting).
                    n_synth = _synth_b.shape[0]
                    ref_idx = int(synthetic_patch_feature_source[b].item())

                    # Remap ref_idx if patches were dropped
                    if ref_idx in remap:
                        ref_idx = remap[ref_idx]

                    # Extend geometry
                    scene.latitude = torch.cat([scene.latitude, _synth_b[:, 0]])
                    scene.longitude = torch.cat([scene.longitude, _synth_b[:, 1]])
                    scene.depth = torch.cat([scene.depth, _synth_b[:, 2]])
                    scene.frame_index = torch.cat([
                        scene.frame_index,
                        torch.zeros(n_synth, dtype=scene.frame_index.dtype,
                                    device=scene.frame_index.device),
                    ])

                    # Extend features. Two modes:
                    #   (a) per-point random sources (box_floor_area family):
                    #       use the pre-strip embeddings captured above, so
                    #       the slab has heterogeneous textures pulled from
                    #       random scene patches.
                    #   (b) single-source clone (box_size / object_counting):
                    #       broadcast ref_idx's embedding across all n_synth
                    #       points so the marker body reads as one object.
                    if (
                        _per_pt_embeds_saved is not None
                        and _per_pt_embeds_saved.shape[0] == n_synth
                    ):
                        scene.embeds = torch.cat(
                            [scene.embeds, _per_pt_embeds_saved], dim=0)
                        scene.aux_layers = [
                            torch.cat([layer, _per_pt_aux_layers_saved[li]], dim=0)
                            for li, layer in enumerate(scene.aux_layers)
                        ]
                    else:
                        ref_embeds = scene.embeds[ref_idx].unsqueeze(0).expand(n_synth, -1).clone()
                        scene.embeds = torch.cat([scene.embeds, ref_embeds], dim=0)
                        scene.aux_layers = [
                            torch.cat([layer, layer[ref_idx].unsqueeze(0).expand(n_synth, -1).clone()], dim=0)
                            for layer in scene.aux_layers
                        ]
                    scene.n_valid += n_synth

                # --- Real-object asset pastes (2026-08-16) ---
                # The dataset precomputed these rows (_real_paste_rows: the
                # asset's OWN features, spherical coords, T spread); this is
                # the same append _build_sample_stash_fast does on the fast
                # path, reduced to concatenation. Appended AFTER the strip
                # and the synthetic OBBs, mirroring the fast path's order,
                # so the paste rows survive canvas_obb_only automatically.
                _rp_emb = None
                if real_paste_embeds is not None and len(real_paste_embeds) > b:
                    _rp_emb = real_paste_embeds[b]
                if _rp_emb is not None and _rp_emb.shape[0] > 0:
                    if _rp_emb.shape[1] - 1 != len(scene.aux_layers):
                        raise RuntimeError(
                            f"real_paste_embeds carries {_rp_emb.shape[1]} layers "
                            f"but the scene has 1+{len(scene.aux_layers)}: the "
                            "asset bank and the feature config disagree.")
                    _rp_sph = real_paste_spherical[b].to(
                        device=scene.latitude.device, dtype=scene.latitude.dtype)
                    _rp_t = real_paste_frame_index[b].to(
                        device=scene.frame_index.device, dtype=scene.frame_index.dtype)
                    _rp_emb = _rp_emb.to(device=scene.embeds.device)
                    scene.latitude = torch.cat([scene.latitude, _rp_sph[:, 0]])
                    scene.longitude = torch.cat([scene.longitude, _rp_sph[:, 1]])
                    scene.depth = torch.cat([scene.depth, _rp_sph[:, 2]])
                    scene.frame_index = torch.cat([scene.frame_index, _rp_t])
                    scene.embeds = torch.cat(
                        [scene.embeds, _rp_emb[:, 0].to(scene.embeds.dtype)], dim=0)
                    scene.aux_layers = [
                        torch.cat([layer, _rp_emb[:, li + 1].to(layer.dtype)], dim=0)
                        for li, layer in enumerate(scene.aux_layers)
                    ]
                    scene.n_valid += int(_rp_emb.shape[0])

                # --- toolcall_marker v2: inline marker tokens ---
                # The v1 append-to-canvas paste (marker_paste.append_marker_rows)
                # is DELETED here: it put the twin in the shared canvas block,
                # which under whole-episode teacher forcing leaks FUTURE markers
                # to earlier op lines. v2 hands marker_tokens to prepare_batch,
                # which inlines the text marker + canvas twin at their <tool_response>
                # stream positions (causal order preserved). The legacy
                # marker_points/marker_features kwargs are accepted but no longer
                # drive a paste (kept for the rollout --full-prefill reference).

                proj = _qwen3vl_prepare_batch(
                    scene=scene,
                    input_ids=ids_b,
                    attention_mask=mask_b,
                    labels=lbl_b,
                    first_idx=first,
                    last_idx=last,
                    config=cfg,
                    inline_patch_positions=_imp_b,
                    inline_patch_indices=_imp_patch_b,
                    inline_patch_t_indices=_imt_b,
                    inline_patch_t_values=_imt_v_b,
                    hide_source_patches=_hide_b,
                    marker_tokens=_mtok_b,
                )

                projected_input_ids.append(proj["input_ids"])
                projected_position_ids.append(proj["position_ids"])
                projected_attention_mask.append(proj["attention_mask"])
                projected_labels.append(proj["labels"])
                projected_embeds.append(proj["embeds"])
                projected_aux_layers.append(proj["aux_layers"])
                projected_depth_bins.append(proj["depth_bins"])
                projected_ray_dirs.append(proj.get("ray_dirs"))
                projected_inline_patch_indices_local.append(proj.get("inline_patch_indices_local"))
                rope_deltas_list.append(proj["rope_deltas"])

            rope_deltas = rope_deltas_list  # PATH A line ~245 stacks list-of-tensors
            projection_done = True
            # If labels were not supplied (e.g. generation / eval), discard the
            # dummy -100 tensors built above so PATH A leaves labels=None and
            # the parent forward skips loss computation. Otherwise the parent
            # forward sees a [B, L] label tensor and tries to cross-entropy it
            # against the prefill's [B, 1] logits, raising a batch_size mismatch.
            if labels_cpu is None:
                projected_labels = None
            # Record the per-sample original prefill length (= the dummy-360
            # input_ids length that generate() saw, which is what cache_position
            # will count from). PATH A trims the sequence to L_actual = N_valid
            # + text tokens, but generate() does not know about that trim. We
            # fold the (L_360 - L_actual) overshoot into image_len_adjust below
            # so the decode-step delta formula
            #     delta = cache_position - image_len_adjust + rope_deltas
            # produces the correct RoPE position. Without this, decoded tokens
            # land ~(L_360 - L_actual) ≈ ~2000 positions ahead of where the
            # prefill ended, the model attends to a phantom prefill region, and
            # generation collapses into the text-prior.
            self._live_features_orig_lens = [int(input_seq_len[b].item()) for b in range(B)]
            # Fall through to PATH A below.

        # ==================================================================
        # PATH A — Projection already done in dataloader (CPU).
        # The batch carries pre-trimmed sequences and projected embeddings;
        # we just left-pad, stack, embed text tokens, and masked_scatter.
        # ==================================================================
        if prefill_stage and projection_done:
            device = self.get_input_embeddings().weight.device
            B = len(projected_input_ids)

            # --- Left-pad and stack per-sample tensors (same layout as the
            #     existing projection loop produces via pad_and_stack). ---
            position_ids   = pad_and_stack(
                [t.to(device) for t in projected_position_ids], dim=1)       # [3, B, max_L]
            attention_mask = pad_and_stack(
                [t.to(device) for t in projected_attention_mask], dim=0, pad_value=0)  # [B, max_L]
            new_seq_lens   = [ids.shape[-1] for ids in projected_input_ids]
            input_ids      = pad_and_stack(
                [t.to(device) for t in projected_input_ids], dim=0)          # [B, max_L]
            if projected_labels is not None and projected_labels[0] is not None:
                labels = pad_and_stack(
                    [t.to(device) for t in projected_labels], dim=0, pad_value=-100)
            else:
                labels = None

            # Prepend a contiguous text-position row so position_ids becomes
            # [4, B, max_L] = [text, T, H, W]. Qwen3VLTextModel splits row 0
            # off as `text_position_ids` and uses rows 1-3 for MRoPE rotary.
            # Without this row, FA2's `_is_packed_sequence` check (which fires
            # for batch_size==1 whenever positions aren't a contiguous arange)
            # treats every frame-0 image token as a packed-sequence boundary
            # and shreds attention. The cumsum(mask)-1 form keeps left-padded
            # positions at 0 for the multi-sample eval path; for bs=1 it's
            # exactly arange(L). Mirrors what qwen3_5/model.py already does.
            text_pos = attention_mask.long().cumsum(-1) - 1
            text_pos.masked_fill_(attention_mask == 0, 0)
            position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)  # [4, B, max_L]

            # Decode-step state
            if isinstance(rope_deltas, list):
                rope_deltas = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in rope_deltas])
            self.rope_deltas      = rope_deltas.to(device)             # [B]
            # In path A, prefill_input_len == actual_len (already trimmed), so the
            # decode delta formula   cache_pos - image_len_adjust + rope_deltas
            # must use image_len_adjust=0 (rope_deltas already encodes the full
            # positional correction).  Only the left-padding offset for shorter
            # batch members is added below.
            self.image_len_adjust = torch.zeros(B, dtype=torch.long)  # [B]
            self.prefill_input_len = input_ids.shape[1]  # padded length (= max trimmed len)

            max_new_seq_len = input_ids.shape[1]
            self.new_seq_lens   = new_seq_lens
            self.max_new_seq_len = max_new_seq_len
            for i in range(B):
                # Only add left-padding correction for shorter sequences in the batch.
                self.image_len_adjust[i] += max_new_seq_len - new_seq_lens[i]

            # Live-features path: generate() saw the full L_360 dummy input_ids
            # but PRE-PATH-A trimmed it to L_actual = new_seq_lens[i]. cache_position
            # during decode counts from L_360, so subtract (L_360 - L_actual) here
            # so the decode delta formula lands at L_actual + n_decoded.
            # PATH A (precomputed features) leaves _live_features_orig_lens unset
            # because the dataloader already produced trimmed projected_input_ids,
            # which prepare_model_inputs uses to size the dummy input_ids passed
            # to generate() — so cache_position already starts at L_actual there.
            _orig_lens = getattr(self, "_live_features_orig_lens", None)
            if _orig_lens is not None:
                for i in range(B):
                    self.image_len_adjust[i] += _orig_lens[i] - new_seq_lens[i]
                # generate() will see L_360-sized prefill, so prefill_input_len
                # (read by prepare_inputs_for_generation for the bs>1 mask rebuild)
                # has to match what generate() actually fed in, not the trimmed len.
                self.prefill_input_len = max(_orig_lens)
                self._live_features_orig_lens = None

            # Reset the PATH-B decode-step counter (used by
            # prepare_inputs_for_generation to size the bs=1 decode mask to the
            # true canvas-expanded KV length when generate() was fed a flat
            # PATH-B prompt whose length differs from the projected KV).
            self._pathb_decode_n = 0

            # generate() counts cache_position over the prompt it was given, and
            # the live-features path above changed the sequence length. SDPA
            # sizes its causal mask from cache_position, so the prefill must use
            # the projected length (FlashAttention ignores it, which is why only
            # the SDPA fallback failed). Decode keeps generate()'s count, which
            # image_len_adjust already accounts for.
            if cache_position is not None and cache_position.shape[0] != input_ids.shape[1]:
                past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_len, past_len + input_ids.shape[1], device=device)

            # --- Text embeddings ---
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # --- Concatenate projected image/aux-layer embeds across batch ---
            image_embeds = torch.cat(
                [e.to(device) for e in projected_embeds], dim=0)
            # Cast to match text embeddings dtype (e.g., BFloat16 during eval)
            image_embeds = image_embeds.to(inputs_embeds.dtype)

            image_embeds = self._apply_geometry_embeddings(
                image_embeds,
                projected_depth_bins,
                projected_ray_dirs,
                device,
            )

            # --- Inline patch marker identity embedding ---
            # When inline patch markers exist, add the learned constant vector so
            # the model can distinguish "I am a reference marker" from the ~N_valid
            # canvas patches that share the same token ID and MRoPE position.
            if (hasattr(self, "inline_patch_constant_embed")
                    and projected_inline_patch_indices_local is not None):
                pm_indices = []
                offset = 0
                for emb, pmil in zip(projected_embeds, projected_inline_patch_indices_local):
                    n_tok = emb.shape[0]
                    if pmil is not None and pmil.numel() > 0:
                        pm_indices.append(pmil.to(device, dtype=torch.long) + offset)
                    offset += n_tok
                if pm_indices:
                    pm_idx = torch.cat(pm_indices)
                    pm_emb = self.inline_patch_constant_embed.to(image_embeds.dtype)
                    image_embeds[pm_idx] = image_embeds[pm_idx] + pm_emb

            # Find the number of aux ViT layers from the first non-empty sample.
            # Depth-distance samples have an empty aux_layers list ([]); pad
            # them with zero tensors so the batch-cat below works uniformly.
            num_layers_ds = 0
            for ds in projected_aux_layers:
                if len(ds) > 0:
                    num_layers_ds = len(ds)
                    break

            if num_layers_ds > 0:
                D = projected_embeds[0].shape[-1]
                for i, ds in enumerate(projected_aux_layers):
                    if len(ds) == 0:
                        n_tok = projected_embeds[i].shape[0]
                        projected_aux_layers[i] = [
                            torch.zeros(n_tok, D) for _ in range(num_layers_ds)
                        ]

            # HF-boundary translation: HF Qwen3-VL consumes its DeepStack signal
            # via the `deepstack_visual_embeds` kwarg (flat list of per-layer
            # tensors, layer-major). Translate our model-agnostic
            # `aux_layers` (per-sample lists of per-layer tensors) into that
            # shape. Other adapters (Qwen3.5) ignore aux_layers
            # entirely and never reach this branch.
            deepstack_visual_embeds = [
                torch.cat([sample[l].to(device) for sample in projected_aux_layers], dim=0)
                for l in range(num_layers_ds)
            ]

            # --- Masked scatter ---
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Zero out left-padding positions
            max_emb_len = inputs_embeds.shape[1]
            pad_lens = torch.tensor(
                [max_emb_len - l for l in new_seq_lens], device=device)
            if pad_lens.max() > 0:
                seq_pos = torch.arange(max_emb_len, device=device).unsqueeze(0)
                zero_mask = seq_pos < pad_lens.unsqueeze(1)
                inputs_embeds[zero_mask] = 0

            visual_pos_masks = image_mask[..., 0]

        # ==================================================================
        # Decode step (single new token during generation).
        # ==================================================================
        else:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            batch_size, seq_length, _ = inputs_embeds.shape

            # Patch: If not set (first decode step), default to zeros
            if not hasattr(self, "image_len_adjust") or self.image_len_adjust is None:
                self.image_len_adjust = torch.zeros(batch_size, dtype=torch.long, device=inputs_embeds.device)
            if not hasattr(self, "rope_deltas") or self.rope_deltas is None:
                self.rope_deltas = torch.zeros(batch_size, dtype=torch.long, device=inputs_embeds.device)


            delta = (
                (cache_position[0] - self.image_len_adjust.to(inputs_embeds.device) + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.unsqueeze(-1)  # [batch] -> [batch, 1] to broadcast against [batch, seq_len]
            position_ids = position_ids.add(delta)
            # [4, B, 1] = [text, T, H, W]. Decode is a single token so the
            # text row matches T/H/W trivially, but we keep the format
            # consistent with prefill so HF's qwen3_vl modeling takes the
            # `shape[0] == 4` branch on every call.
            position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)
            visual_pos_masks = None
            deepstack_visual_embeds = None

            attention_mask = None

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        # Free DeepStack tensors (HF deepstack_visual_embeds) immediately after language model has consumed them.
        # gc.collect() is deliberately NOT called here: refcount drop from del
        # frees the tensors immediately, and a full-heap GC walk on every
        # forward pass costs ~1-2s per training step at 8B model scale.
        if deepstack_visual_embeds is not None:
            del deepstack_visual_embeds
            del image_embeds

        if prefill_stage and hasattr(self, 'new_seq_lens') and outputs.past_key_values is not None:
            kv = outputs.past_key_values
            pad_lens = [self.max_new_seq_len - vl for vl in self.new_seq_lens]
            if hasattr(kv, 'key_cache') and any(p > 0 for p in pad_lens):
                dev = kv.key_cache[0].device
                seq_len_kv = kv.key_cache[0].shape[2]
                # pos: [1, 1, seq, 1]  pad_lens_t: [batch, 1, 1, 1]
                pos = torch.arange(seq_len_kv, device=dev).view(1, 1, -1, 1)
                pad_lens_t = torch.tensor(pad_lens, device=dev).view(-1, 1, 1, 1)
                zero_mask = pos < pad_lens_t  # [batch, 1, seq, 1] — True at padding positions
                for lid in range(len(kv.key_cache)):
                    kv.key_cache[lid].masked_fill_(zero_mask, 0)
                    kv.value_cache[lid].masked_fill_(zero_mask, 0)

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        ), labels

class Qwen3VL3DForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VL3DModel(config)
    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # The 3D model computes its own RoPE position IDs in forward() based on
        # projected geometry.  Skip the standard get_rope_index() call which
        # crashes on multi-image image_grid_thw vs. single 360 input_ids.
        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            return position_ids
        seq_len = inputs_tensor.shape[1]
        batch_size = inputs_tensor.shape[0]
        return torch.arange(seq_len, device=inputs_tensor.device).unsqueeze(0).expand(batch_size, -1)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = Qwen3VLForConditionalGeneration.prepare_inputs_for_generation(self, input_ids, **kwargs)
        # Only apply correction during decode steps (input_ids is a single new token).
        # At prefill, input_ids has the full sequence length — we must not interfere.
        is_decode_step = (
            model_inputs.get("input_ids") is not None
            and model_inputs["input_ids"].shape[1] == 1
        ) or (
            model_inputs.get("inputs_embeds") is not None
            and model_inputs["inputs_embeds"].shape[1] == 1
        )
        # Custom right-aligned mask is only needed when the eval batch contains
        # multiple samples with different prefill lengths (left-padding leaves
        # zeros at the front of the KV cache that the decode step must skip).
        # With batch_size=1 — which is what training and benchmarks use everywhere
        # in this repo — there is no padding, and HF's default attention_mask is
        # already correct. Skipping the override for bs=1 keeps the FA2 backend
        # happy: HF's flash_attention_2 wrapper does not handle the right-aligned
        # zeros-on-the-left layout that this branch produces.
        # The decode attention mask must match the ACTUAL KV cache length. Two
        # cases break HF's default mask (which HF sizes to the input_ids
        # generate() saw, NOT the canvas-expanded KV):
        #   1. bs>1 left-padded eval batches (len(new_seq_lens) > 1), handled by
        #      the right-aligned rebuild below.
        #   2. A FLAT PATH-B (live-features) prompt whose length differs from the
        #      projected canvas KV: reproject_scene culls/expands the image
        #      placeholders, so the true KV (max_new_seq_len) != the prompt length
        #      generate() counted (prefill_input_len). HF's mask then references the
        #      wrong KV entries and the decode derails after a few tokens. The
        #      paper's PATH-A eval feeds a projected-length input, so
        #      prefill_input_len == max_new_seq_len and this branch does NOT fire
        #      for it. (Root cause of the decode-diag tool-call derailment,
        #      found 2026-07-18. The mismatch is usually flat > projected: the raw
        #      32-frame placeholder block is larger than the culled canvas.)
        _pil = getattr(self.model, "prefill_input_len", None)
        _mnsl = getattr(self.model, "max_new_seq_len", None)
        pathb_len_mismatch = (_pil is not None and _mnsl is not None and _pil != _mnsl)
        needs_custom_mask = (
            is_decode_step
            and hasattr(self.model, "max_new_seq_len")
            and model_inputs.get("cache_position") is not None
            and (len(getattr(self.model, "new_seq_lens", [])) > 1 or pathb_len_mismatch)
        )
        if needs_custom_mask:
            batch_size = input_ids.shape[0]
            if len(self.model.new_seq_lens) > 1:
                # bs>1 left-padded: valid prefill tokens are RIGHT-aligned in the KV
                # cache [kv_size-l : kv_size]; the new decode token sits at kv_size.
                # Contiguous per-sample region satisfies flash_attn_varlen_func.
                orig_cp = model_inputs["cache_position"]
                n_already = (orig_cp[0] - self.model.prefill_input_len).item()
                kv_size = self.model.max_new_seq_len + n_already
                total_size = kv_size + 1
                attn = torch.zeros(batch_size, total_size, dtype=torch.long, device=input_ids.device)
                for idx, l in enumerate(self.model.new_seq_lens):
                    attn[idx, kv_size - l : kv_size] = 1          # valid prefill (right-aligned)
                    attn[idx, kv_size:] = 1                        # already-generated + current token
            else:
                # bs=1 PATH-B length mismatch: attend to the full, contiguous KV.
                # Never trust the flat-prompt bookkeeping (prefill_input_len /
                # cache_position count the un-projected length) — read the real KV
                # length off the cache, falling back to projected-prefill + steps.
                pkv = model_inputs.get("past_key_values")
                kv_len = None
                if pkv is not None and hasattr(pkv, "get_seq_length"):
                    try:
                        kv_len = int(pkv.get_seq_length())
                    except Exception:
                        kv_len = None
                if kv_len is None:
                    kv_len = int(self.model.max_new_seq_len) + getattr(self.model, "_pathb_decode_n", 0)
                attn = torch.ones(batch_size, kv_len + 1, dtype=torch.long, device=input_ids.device)
            model_inputs["attention_mask"] = attn
        # Advance the PATH-B decode-step counter (bs=1 fallback KV sizing).
        if is_decode_step and pathb_len_mismatch:
            self.model._pathb_decode_n = getattr(self.model, "_pathb_decode_n", 0) + 1
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        outputs, labels = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            labels=labels,
            **kwargs,
        )

        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )
