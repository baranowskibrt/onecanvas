"""Qwen3.5 adapter: maps ReprojectedScene into Qwen3.5's interleaved 3D MRoPE format.

Qwen3.5 uses 3D MRoPE over (T, H, W) — same axis count as Qwen3-VL —
but with mrope_section=[11, 11, 10] and frequency-space interleaving
applied by ``apply_interleaved_mrope`` inside HF's modeling_qwen3_5.
The interleaving is HF's responsibility; we feed standard [3, seq_len]
position IDs (the same Qwen3-VL produces) and the rope embedder
re-arranges the frequency bands internally.

Architectural differences vs Qwen3-VL:
  - No DeepStack analog (single-injection ViT-MLP-LLM stack)
  - No camera embedding / reference embedding (no FOV-aware metadata)
  - Different LM forward signature (Qwen3_5Model, not Qwen3_VLModel)

To launch a training run on Qwen3.5-9B:
  pass --model_name_or_path with "qwen3.5" or "qwen3_5" in the path
  (auto-detected by training/onecanvas/train/train.py:_is_qwen3_vl)

Lazy-imported: importing this package does not eagerly load the model
class. See ``model_adapters/__init__.py`` for the lazy-access pattern.
"""

__all__ = [
    "build_adapter",
    "Qwen3_5_3DModel",
    "Qwen3_5_3DForConditionalGeneration",
]


def __getattr__(name):
    if name == "build_adapter":
        from .adapter import build_adapter
        return build_adapter
    if name in ("Qwen3_5_3DModel", "Qwen3_5_3DForConditionalGeneration"):
        from .model import Qwen3_5_3DModel, Qwen3_5_3DForConditionalGeneration
        return {"Qwen3_5_3DModel": Qwen3_5_3DModel,
                "Qwen3_5_3DForConditionalGeneration": Qwen3_5_3DForConditionalGeneration}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
