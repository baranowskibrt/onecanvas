"""Attention-implementation resolution.

flash-attn is the default attention backend in the training / benchmark scripts
(``base.sh`` sets ``RUN_ATTN_IMPLEMENTATION=flash_attention_2``) but it is
deliberately NOT a declared dependency: its build routinely fails on machines
without a matching CUDA toolchain. Rather than crash inside
``from_pretrained`` when the wheel is absent, callers pass the requested
implementation through :func:`resolve_attn_implementation`, which downgrades to
PyTorch SDPA with a one-time warning.
"""
import importlib.util
import warnings


def resolve_attn_implementation(requested: str) -> str:
    """Return ``requested``, downgrading ``flash_attention_2`` to ``sdpa`` when
    the ``flash_attn`` package is not importable.

    Only ``flash_attention_2`` is guarded: ``sdpa`` and ``eager`` are always
    available in stock PyTorch. The warning fires once per process (default
    warning filter dedupes by call site)."""
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        warnings.warn(
            "attn_implementation='flash_attention_2' was requested but the "
            "flash_attn package is not installed; falling back to 'sdpa'. "
            "flash-attn is not a declared dependency (its build is fragile); "
            "install it manually for best throughput (see README > Install).",
            RuntimeWarning,
            stacklevel=2,
        )
        return "sdpa"
    return requested
