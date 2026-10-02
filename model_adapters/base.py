"""VLM adapter interface: dataclasses + factory."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any


@dataclass
class VLMAdapterConfig:
    image_pad_token_id: int
    video_pad_token_id: int
    vision_start_token_id: int
    assistant_token_id: int
    im_end_token_id: int
    feature_prefix: str          # "qwen3_vl", "qwen3_5"
    rope_dims: int               # 3 (Qwen3-VL, Qwen3.5 interleaved)
    spatial_merge_size: int
    # Inline patch-marker placeholder token (e.g. "<|object_ref_start|>" in
    # the Qwen family). Used by the probing dataset and the cam-marker path
    # in data_processor_3d to insert marker positions into question text;
    # those positions are then swapped to image_pad_token_id before the
    # adapter splice. Token name is shared across Qwen3-VL and Qwen3.5 but
    # the IDs differ (151646 vs 248047).
    object_ref_start_token_id: int = 151646
    chat_template_kwargs: dict = field(default_factory=dict)
    # Grounding tokens (VLM-specific, used for 3D bbox output format)
    box_start_token: str = ""    # e.g. "<|box_start|>" for Qwen
    box_end_token: str = ""      # e.g. "<|box_end|>" for Qwen


@dataclass
class VLMAdapter:
    config: VLMAdapterConfig
    get_rope_index: Callable     # (input_ids, image_grid_thw, ...) -> (pos_ids, deltas)
    prepare_batch: Callable      # (scene, input_ids, pos_ids, ...) -> dict
    mask_labels: Callable        # (input_ids) -> (input_ids, labels)


def get_adapter(processor, data_args) -> VLMAdapter:
    """Build the right VLMAdapter based on data_args.model_type."""
    model_type = getattr(data_args, "model_type", "qwen3vl_3d")

    if "qwen3_5" in model_type or "qwen3.5" in model_type:
        from .qwen3_5.adapter import build_adapter
    elif "qwen3_vl" in model_type or "qwen3vl" in model_type:
        from .qwen3_vl.adapter import build_adapter
    else:
        raise ValueError(f"No adapter for model_type={model_type!r}")

    return build_adapter(processor, data_args)
