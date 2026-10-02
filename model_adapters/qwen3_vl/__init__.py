"""Qwen3-VL adapter: maps ReprojectedScene into Qwen3-VL's MRoPE format.

Lazy-imported: importing this package does not eagerly load the model
class (which pulls in torch/transformers/utils). Use
``from model_adapters.qwen3_vl.model import Qwen3VL3DModel`` for the
heavy import, or the top-level lazy accessor on ``model_adapters``.
"""

__all__ = [
    "prepare_batch",
    "build_adapter",
    "get_rope_index_3",
    "Qwen3VL3DModel",
    "Qwen3VL3DForConditionalGeneration",
]


def __getattr__(name):
    if name in ("prepare_batch", "build_adapter"):
        from .adapter import prepare_batch, build_adapter
        return {"prepare_batch": prepare_batch, "build_adapter": build_adapter}[name]
    if name == "get_rope_index_3":
        from ..qwen_mrope import get_rope_index_3
        return get_rope_index_3
    if name in ("Qwen3VL3DModel", "Qwen3VL3DForConditionalGeneration"):
        from .model import Qwen3VL3DModel, Qwen3VL3DForConditionalGeneration
        return {"Qwen3VL3DModel": Qwen3VL3DModel,
                "Qwen3VL3DForConditionalGeneration": Qwen3VL3DForConditionalGeneration}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
