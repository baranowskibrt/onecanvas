"""Model adapters: thin VLM-specific layers that consume ReprojectedScene.

Lazy-imported: ``import model_adapters`` only loads ``base`` (the factory
interface). Concrete model classes — Qwen3VL3DModel, Qwen3_5_3DModel — are
resolved on first attribute access via PEP-562 ``__getattr__``. Downstream
callers that only use one adapter never load the other.
"""

from .base import VLMAdapter, VLMAdapterConfig, get_adapter

__all__ = [
    "VLMAdapter", "VLMAdapterConfig", "get_adapter",
    "Qwen3VL3DModel", "Qwen3VL3DForConditionalGeneration",
    "Qwen3_5_3DModel", "Qwen3_5_3DForConditionalGeneration",
]

# name -> (submodule, attribute) — resolved lazily on first access
_LAZY_EXPORTS = {
    "Qwen3VL3DModel":                      ("qwen3_vl.model", "Qwen3VL3DModel"),
    "Qwen3VL3DForConditionalGeneration":   ("qwen3_vl.model", "Qwen3VL3DForConditionalGeneration"),
    "Qwen3_5_3DModel":                     ("qwen3_5.model",  "Qwen3_5_3DModel"),
    "Qwen3_5_3DForConditionalGeneration":  ("qwen3_5.model",  "Qwen3_5_3DForConditionalGeneration"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        from importlib import import_module
        submod_name, attr_name = _LAZY_EXPORTS[name]
        try:
            submod = import_module(f".{submod_name}", __name__)
        except ImportError as e:
            raise ImportError(
                f"Cannot load model_adapters.{submod_name}: {e}. "
                f"This adapter's dependencies may be missing."
            ) from e
        return getattr(submod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
