"""Qwen3.5 adapter.

Qwen3.5 uses interleaved 3D MRoPE over (T, H, W) with mrope_section
[11, 11, 10]. The interleaving happens in frequency space inside HF's
``apply_interleaved_mrope`` (see transformers.models.qwen3_5.modeling_qwen3_5
around line 246), not in the position_ids construction — so this adapter
feeds the same [3, seq_len] position IDs that Qwen3-VL's get_rope_index_3
produces and lets HF's rope embedder re-arrange the frequency bands.

Ready for live-checkpoint validation: launch training with
``--model_name_or_path <path-containing-"qwen3.5"-or-"qwen3_5">`` and
verify the prefill forward runs without shape errors before claiming
the stub is end-to-end correct. If it crashes with a frequency-band
mismatch, write a dedicated get_rope_index_interleaved here that
respects mrope_section=[11, 11, 10] up-front.
"""

from functools import partial

from ..base import VLMAdapter, VLMAdapterConfig
from ..qwen3_vl.adapter import prepare_batch
from ..qwen_mrope import get_rope_index_3


def build_adapter(processor, data_args) -> VLMAdapter:
    """Build a Qwen3.5 adapter from a processor and data_args."""
    from onecanvas.data.dataset_utils import preprocess_qwen_visual

    _tok = processor.tokenizer
    merge_size = getattr(processor.image_processor, "merge_size", 2)

    _img_pad = _tok.encode("<|image_pad|>", add_special_tokens=False)
    _vid_pad = _tok.encode("<|video_pad|>", add_special_tokens=False)
    _vis_start = _tok.encode("<|vision_start|>", add_special_tokens=False)
    _asst = _tok.encode("assistant", add_special_tokens=False)
    _im_end = _tok.encode("<|im_end|>", add_special_tokens=False)
    _obj_ref = _tok.encode("<|object_ref_start|>", add_special_tokens=False)

    config = VLMAdapterConfig(
        image_pad_token_id=_img_pad[0] if _img_pad else 248056,
        video_pad_token_id=_vid_pad[0] if _vid_pad else 248057,
        vision_start_token_id=_vis_start[0] if _vis_start else 248053,
        assistant_token_id=_asst[0] if _asst else 74455,
        im_end_token_id=_im_end[0] if _im_end else 248046,
        object_ref_start_token_id=_obj_ref[0] if _obj_ref else 248047,
        feature_prefix="qwen3_5",
        rope_dims=3,
        spatial_merge_size=merge_size,
        chat_template_kwargs={"enable_thinking": False},
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
