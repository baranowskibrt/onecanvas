"""
Qwen3.5-9B 3D pipeline — mirrors qwen3_vl_3d.py but adapted for Qwen3_5.

Key differences from Qwen3VL3D:
  - Base classes: Qwen3_5Model / Qwen3_5ForConditionalGeneration
  - No DeepStack injection into LM layers (Qwen3.5 has no DeepStack mechanism)
  - Feature extraction: single-layer ViT output (no DeepStack indexes)
  - Image token ID: config.image_token_id (248056 for Qwen3.5-9B)
  - Output types: Qwen3_5ModelOutputWithPast / Qwen3_5CausalLMOutputWithPast
"""
import math
from typing import Optional, Union, Unpack

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast, Qwen3_5ForConditionalGeneration,
    Qwen3_5Model, Qwen3_5ModelOutputWithPast)
from transformers.utils import TransformersKwargs

from model_adapters.geometry_embedding import GeometryEmbeddingMixin
from utils.tensor_ops import pad_and_stack


class Qwen3_5_3DModel(GeometryEmbeddingMixin, Qwen3_5Model):
    """Qwen3.5 model extended with 3D panoramic (equirectangular) projection.

    Replaces image-placeholder tokens with projected 3D features.
    No DeepStack injection (Qwen3_5TextModel has no DeepStack layers).

    Depth + angle embedding methods (init_*, _encode_*, _stash_*, and the
    _apply_geometry_embeddings dispatch) live on GeometryEmbeddingMixin.
    """


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_deltas=None,
        labels=None,
        # Dataloader-projected path keys
        projection_done=None,
        projected_input_ids=None,
        projected_position_ids=None,
        projected_attention_mask=None,
        projected_labels=None,
        projected_embeds=None,
        projected_aux_layers=None,  # accepted for API compatibility; Qwen3.5 has no DeepStack equivalent so this is ignored
        projected_depth_bins=None,
        projected_ray_dirs=None,
        projected_inline_patch_indices_local=None,
        image_len_adjust=None,
        _dataloader_retries=None,
        # PATH B keys (raw geometry + indices for live-features projection)
        depths: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        image_dims: Optional[torch.Tensor] = None,
        image_token_first_idx: Optional[torch.Tensor] = None,
        image_token_last_idx: Optional[torch.Tensor] = None,
        input_seq_len: Optional[torch.Tensor] = None,
        inline_patch_positions=None,
        aug_center_override=None,
        aug_yaw_angle=None,
        n_source_images=None,
        **kwargs,
    ) -> Union[tuple, Qwen3_5ModelOutputWithPast]:

        prefill_stage = (
            (input_ids is not None and input_ids.shape[1] != 1) or
            (inputs_embeds is not None and inputs_embeds.shape[1] != 1) or
            (bool(projection_done) and past_key_values is None)
        )

        image_mask = None

        # ==================================================================
        # PRE-PATH-A — live visual encoding (use_precomputed_features=False).
        # Mirrors Qwen3-VL's pre-path-A block but for Qwen3.5's single-layer
        # vision tower (no DeepStack). When the dataloader returned a PATH B
        # batch (raw pixel_values + raw geometry), run self.visual here, call
        # reproject_scene + adapter.prepare_batch per sample, then flip
        # projection_done=True and fall through to PATH A unchanged.
        # ==================================================================
        if (
            prefill_stage
            and not projection_done
            and pixel_values is not None
            and image_grid_thw is not None
            and depths is not None
            and getattr(self, "visual", None) is not None
        ):
            from reprojection import reproject_scene

            cfg = getattr(self, "reprojection_config", None)
            if cfg is None:
                raise RuntimeError(
                    "Live-features path requires self.reprojection_config to be "
                    "set on the model. Train.py attaches it after "
                    "make_supervised_data_module(); make sure that wiring ran "
                    "(check the isinstance() chain in train.py around line 1296)."
                )

            # Depth-resolution probe (eval only); see the Qwen3-VL adapter.
            _depth_downsample = int(cfg.get("depth_downsample", 1) or 1)
            if _depth_downsample > 1 and not getattr(self, "_depth_downsample_logged", False):
                self._depth_downsample_logged = True
                print(f"[depth_downsample] active: K={_depth_downsample} "
                      f"(depth grid coarsened to H_feat/K x W_feat/K, patch count unchanged)")

            # Adapter.prepare_batch lives on the Qwen3.5 adapter, but it's
            # currently a partial re-export of Qwen3-VL's prepare_batch. Import
            # directly to avoid re-running build_adapter() per forward.
            from model_adapters.qwen3_vl.adapter import prepare_batch as _qwen3vl_prepare_batch

            pv = pixel_values
            if pv.dim() == 3:
                pv = pv.reshape(-1, pv.shape[-1])
            gt = image_grid_thw
            if gt.dim() == 3:
                gt = gt.reshape(-1, 3)

            merge = self.visual.spatial_merge_size
            h_per = (gt[:, 1] // merge).tolist()
            w_per = (gt[:, 2] // merge).tolist()
            tokens_per_img = [h * w for h, w in zip(h_per, w_per)]
            H_feat, W_feat = h_per[0], w_per[0]
            assert all(h == H_feat and w == W_feat for h, w in zip(h_per, w_per)), (
                "live-features path requires homogeneous image shapes within a batch"
            )

            # Run visual encoder. Grad only when training AND the visual is
            # trainable; matches the precomputed-path behavior. The
            # self.training gate keeps inference on a plain from_pretrained
            # model (requires_grad=True everywhere) from re-enabling grad
            # inside generate()'s no_grad and retaining all vision activations.
            visual_needs_grad = self.training and any(
                p.requires_grad for p in self.visual.parameters()
            )
            _ctx = torch.enable_grad() if visual_needs_grad else torch.no_grad()
            with _ctx:
                image_outputs = self.get_image_features(pv, gt)
            # HF Qwen3.5 returns BaseModelOutputWithPooling with .pooler_output
            # as a tuple of per-image [H_i*W_i, C] tensors (no DeepStack).
            image_embeds_per_img = image_outputs.pooler_output
            C = image_embeds_per_img[0].shape[-1]
            base_layer = torch.stack(
                [emb.view(H_feat, W_feat, C) for emb in image_embeds_per_img], dim=0,
            )  # [N_total, H, W, C]
            all_features = base_layer.unsqueeze(1)  # [N_total, 1, H, W, C]

            B = depths.shape[0]
            N_per = depths.shape[1]
            assert B * N_per == all_features.shape[0], (
                f"image count mismatch: B={B} * N_per={N_per} != {all_features.shape[0]}"
            )

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
            projected_aux_layers       = []
            projected_depth_bins       = []
            projected_ray_dirs         = []
            projected_inline_patch_indices_local = []
            rope_deltas_list           = []

            for b in range(B):
                L_real = int(input_seq_len[b].item())
                first  = int(image_token_first_idx[b].item())
                last   = int(image_token_last_idx[b].item())

                feats_b = features_cpu[b * N_per : (b + 1) * N_per]

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

                ids_b = input_ids_cpu[b:b+1, :L_real]
                mask_b = attention_mask_cpu[b:b+1, :L_real]
                if labels_cpu is not None:
                    lbl_b = labels_cpu[b:b+1, :L_real]
                else:
                    lbl_b = torch.full((1, L_real), -100, dtype=torch.long)

                if inline_patch_positions is not None and len(inline_patch_positions) > b:
                    _imp_b = inline_patch_positions[b]
                    _imp_b = _imp_b.tolist() if hasattr(_imp_b, "tolist") else list(_imp_b)
                else:
                    _imp_b = None

                proj = _qwen3vl_prepare_batch(
                    scene=scene,
                    input_ids=ids_b,
                    attention_mask=mask_b,
                    labels=lbl_b,
                    first_idx=first,
                    last_idx=last,
                    config=cfg,
                    inline_patch_positions=_imp_b,
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

            rope_deltas = rope_deltas_list
            projection_done = True
            if labels_cpu is None:
                projected_labels = None
            self._live_features_orig_lens = [int(input_seq_len[b].item()) for b in range(B)]
            # Fall through to PATH A below.

        # ==================================================================
        # PATH A — Projection already done in dataloader (CPU).
        # ==================================================================
        if prefill_stage and projection_done:
            self._decode_dbg_printed = False

            device = next(self.parameters()).device
            B = len(projected_input_ids)

            position_ids   = pad_and_stack(
                [t.to(device) for t in projected_position_ids], dim=1)
            attention_mask = pad_and_stack(
                [t.to(device) for t in projected_attention_mask], dim=0, pad_value=0)
            new_seq_lens   = [ids.shape[-1] for ids in projected_input_ids]
            input_ids      = pad_and_stack(
                [t.to(device) for t in projected_input_ids], dim=0)
            if projected_labels is not None and projected_labels[0] is not None:
                labels = pad_and_stack(
                    [t.to(device) for t in projected_labels], dim=0, pad_value=-100)
            else:
                labels = None

            # Prepend text position row: Qwen3.5 TextModel expects [4, B, seq]
            # where dim 0 = [text_positions, temporal, height, width]
            text_pos = attention_mask.long().cumsum(-1) - 1
            text_pos.masked_fill_(attention_mask == 0, 0)
            position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)  # [4, B, seq]

            # PRE-PATH-A produces a list (one rope_delta per sample); the
            # dataloader-precomputed PATH A already passes a tensor. Stack into
            # a [B] tensor here so the decode-step delta formula works either way.
            if isinstance(rope_deltas, list):
                rope_deltas = torch.stack([
                    t if isinstance(t, torch.Tensor) else torch.tensor(t)
                    for t in rope_deltas
                ])
            self.rope_deltas      = rope_deltas.to(device)
            self.image_len_adjust = torch.zeros(B, dtype=torch.long)
            self.prefill_input_len = input_ids.shape[1]

            max_new_seq_len = input_ids.shape[1]
            self.new_seq_lens    = new_seq_lens
            self.max_new_seq_len = max_new_seq_len
            for i in range(B):
                self.image_len_adjust[i] += max_new_seq_len - new_seq_lens[i]

            # Live-features correction: generate() saw the full L_360 dummy
            # input_ids, but PRE-PATH-A trimmed to L_actual. cache_position
            # during decode counts from L_360, so subtract (L_360 - L_actual)
            # here. PATH A from dataloader leaves _live_features_orig_lens
            # unset since projected_input_ids are already trimmed.
            _orig_lens = getattr(self, "_live_features_orig_lens", None)
            if _orig_lens is not None:
                for i in range(B):
                    self.image_len_adjust[i] += _orig_lens[i] - new_seq_lens[i]
                self.prefill_input_len = max(_orig_lens)
                self._live_features_orig_lens = None

            inputs_embeds = self.get_input_embeddings()(input_ids)
            image_embeds  = torch.cat([e.to(device) for e in projected_embeds], dim=0)
            # Cast to match text embeddings dtype: projected_embeds come from
            # the dataloader at fp32 (CPU geometry path), inputs_embeds is the
            # model dtype (bf16 under DeepSpeed). Without this cast,
            # masked_scatter raises a dtype mismatch on the first forward.
            image_embeds = image_embeds.to(inputs_embeds.dtype)

            # No DeepStack equivalent in Qwen3.5 — projected_aux_layers rows are empty lists and never consumed

            image_embeds = self._apply_geometry_embeddings(
                image_embeds,
                projected_depth_bins,
                projected_ray_dirs,
                device,
            )

            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            max_emb_len = inputs_embeds.shape[1]
            pad_lens = torch.tensor([max_emb_len - l for l in new_seq_lens], device=device)
            if pad_lens.max() > 0:
                seq_pos   = torch.arange(max_emb_len, device=device).unsqueeze(0)
                zero_mask = seq_pos < pad_lens.unsqueeze(1)
                inputs_embeds[zero_mask] = 0

            # Override cache_position to match projected sequence length
            cache_position = torch.arange(inputs_embeds.shape[1], device=device)

        # ==================================================================
        # Decode step (single new token).
        # ==================================================================
        else:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            batch_size, seq_length, _ = inputs_embeds.shape

            if not hasattr(self, "image_len_adjust") or self.image_len_adjust is None:
                self.image_len_adjust = torch.zeros(batch_size, dtype=torch.long,
                                                    device=inputs_embeds.device)
            if not hasattr(self, "rope_deltas") or self.rope_deltas is None:
                self.rope_deltas = torch.zeros(batch_size, dtype=torch.long,
                                               device=inputs_embeds.device)

            delta = (
                (cache_position[0] - self.image_len_adjust.to(inputs_embeds.device) + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.unsqueeze(-1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)  # [4, B, 1] for decode
            attention_mask = None  # KV-cache managed, safe to let attn run unmasked

        # ==================================================================
        # Language model forward (no DeepStack for Qwen3.5)
        # ==================================================================
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        # Zero out padding KV entries so they don't corrupt attention on decode.
        # Qwen3_5DynamicCache has None entries for linear-attention layers — skip those.
        if prefill_stage and hasattr(self, 'new_seq_lens') and outputs.past_key_values is not None:
            kv = outputs.past_key_values
            pad_lens = [self.max_new_seq_len - vl for vl in self.new_seq_lens]
            if hasattr(kv, 'key_cache') and any(p > 0 for p in pad_lens):
                # Find first full-attention (4-D) key cache to obtain device
                dev = next(
                    (kc.device for kc in kv.key_cache if kc is not None and kc.dim() == 4),
                    None
                )
                if dev is not None:
                    pad_lens_t = torch.tensor(pad_lens, device=dev).view(-1, 1, 1, 1)
                    for lid in range(len(kv.key_cache)):
                        kc = kv.key_cache[lid]
                        if kc is None or kc.dim() != 4:
                            continue  # skip linear-attention / uninitialized layers
                        seq_len_kv = kc.shape[2]
                        pos       = torch.arange(seq_len_kv, device=dev).view(1, 1, -1, 1)
                        zero_mask = pos < pad_lens_t
                        kc.masked_fill_(zero_mask, 0)
                        kv.value_cache[lid].masked_fill_(zero_mask, 0)

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        ), labels


class Qwen3_5_3DForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5_3DModel(config)

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # The 3D model computes its own RoPE position IDs in forward() based on
        # projected geometry. We skip the standard get_rope_index() call here
        # (which would crash on multi-image image_grid_thw vs. single 360 input_ids)
        # and return simple sequential position IDs as a no-op placeholder.
        # forward() will overwrite them with proper 3D (T/H/W) MRoPE coordinates.
        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            return position_ids
        seq_len = inputs_tensor.shape[1]
        batch_size = inputs_tensor.shape[0]
        return torch.arange(seq_len, device=inputs_tensor.device).unsqueeze(0).expand(batch_size, -1)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = Qwen3_5ForConditionalGeneration.prepare_inputs_for_generation(
            self, input_ids, **kwargs
        )
        is_decode_step = (
            model_inputs.get("input_ids") is not None
            and model_inputs["input_ids"].shape[1] == 1
        ) or (
            model_inputs.get("inputs_embeds") is not None
            and model_inputs["inputs_embeds"].shape[1] == 1
        )
        # Custom right-aligned mask only needed for multi-sample batches with
        # padding. bs=1 (training + benchmarks) uses HF's default mask, which
        # is also the only layout HF's flash_attention_2 wrapper accepts.
        needs_custom_mask = (
            is_decode_step
            and hasattr(self.model, "max_new_seq_len")
            and model_inputs.get("cache_position") is not None
            and len(getattr(self.model, "new_seq_lens", [])) > 1
        )
        if needs_custom_mask:
            orig_cp   = model_inputs["cache_position"]
            n_already = (orig_cp[0] - self.model.prefill_input_len).item()
            kv_size   = self.model.max_new_seq_len + n_already
            total_size = kv_size + 1

            batch_size = input_ids.shape[0]
            attn = torch.zeros(batch_size, total_size, dtype=torch.long, device=input_ids.device)
            for idx, l in enumerate(self.model.new_seq_lens):
                attn[idx, kv_size - l : kv_size] = 1
                attn[idx, kv_size:] = 1
            model_inputs["attention_mask"] = attn
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
    ) -> Union[tuple, Qwen3_5CausalLMOutputWithPast]:
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
            loss = self.loss_function(logits=logits, labels=labels,
                                      vocab_size=self.config.text_config.vocab_size)

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )
