"""Shared model-setup helpers for Qwen3VL3D / Qwen3_5_3D.

Both ``training/onecanvas/train/train.py`` and ``training/run_benchmarks.py``
use these helpers so that processor configuration, embedding initialization,
and inner-model walking happen in exactly one place.
"""

from __future__ import annotations


__all__ = [
    "configure_processor",
    "get_inner_3d_model",
    "get_hidden_size",
    "init_3d_embeddings",
]


def configure_processor(processor) -> None:
    """Set the image-processor size knobs that this project uses everywhere.

    Both training and benchmarks need the same enormous longest_edge so that
    the processor never auto-resizes our pre-tiled panoramas.
    """
    processor.image_processor.size["longest_edge"] = 16777216 * 4
    processor.image_processor.size["shortest_edge"] = 65536


def get_inner_3d_model(model):
    """Walk wrapper layers to the inner Qwen*3DModel that owns the embedding
    init methods (``init_depth_embedding``, ``init_inline_patch_embedding``)
    and the projection-config attributes.

    Handles all four wrapping cases used in this repo:
      - plain ``Qwen3VL3DForConditionalGeneration``
      - plain ``Qwen3_5_3DForConditionalGeneration``
      - PEFT-wrapped versions of either

    The walk is driven by ``hasattr(candidate, "init_depth_embedding")``
    rather than a fixed depth, because PEFT/LoRA wrapping depth depends on
    whether the model has been merged and on which backbone class is in use.
    """
    candidate = model
    for _ in range(8):  # plenty for any wrapping configuration we use
        if hasattr(candidate, "init_depth_embedding"):
            return candidate
        # DeepSpeedEngine exposes the wrapped model as ``.module``; PEFT/HF
        # wrappers use ``.model``. Try both.
        if hasattr(candidate, "module") and not isinstance(candidate.module, type(candidate)):
            candidate = candidate.module
            continue
        if hasattr(candidate, "model"):
            candidate = candidate.model
            continue
        break
    raise ValueError(
        f"Could not find inner 3D model under {type(model).__name__}; "
        f"walked .model / .module attributes but no class with init_depth_embedding was reached."
    )


def get_hidden_size(model) -> int:
    """Resolve the language-model hidden size across model families:
      - Qwen3-VL: flat config with ``hidden_size``
      - Qwen3.5:  nested ``text_config.hidden_size``
    """
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    if hasattr(cfg, "text_config"):
        return cfg.text_config.hidden_size
    raise AttributeError(
        f"Could not resolve hidden_size from {type(cfg).__name__}: "
        f"none of hidden_size, text_config present."
    )


def init_3d_embeddings(model, data_args) -> None:
    """Initialize depth/angle/camera embedding modules on the inner 3D model
    based on flags in ``data_args`` (or any namespace with the same fields).

    Idempotent and safe to call regardless of model class:
      - Skips a module if its enable flag is off.
      - Skips ``init_inline_patch_embedding`` if the inner model class doesn't
        define it (Qwen3_5_3DModel does not).
      - Stores ``_depth_embed_min`` on the inner model when depth embedding is
        enabled (used by the depth encoder later).

    This is the union of the previous train.py and run_benchmarks.py blocks,
    with identical print output to what train.py used to emit.
    """
    inner = get_inner_3d_model(model)
    hidden_size = get_hidden_size(model)

    # ---------------- Depth embedding ----------------
    # Fallback TRUE, matching the DataArguments default (argument.py,
    # 2026-08-01): the model half must not go missing just because a caller
    # handed in a bare namespace.
    if getattr(data_args, "use_depth_embedding", True):
        de_mode = "cartesian_fourier"   # the only supported depth encoder
        de_num_freqs = getattr(data_args, "depth_embed_num_freqs", 16)
        de_min = getattr(data_args, "depth_embed_min", 0.3)
        de_mlp_hidden = getattr(data_args, "depth_embed_mlp_hidden", 512)
        de_cf_use_rmsnorm = getattr(
            data_args, "depth_embed_cartesian_fourier_use_rmsnorm", False)
        de_cf_per_channel_gate = getattr(
            data_args, "depth_embed_cartesian_fourier_per_channel_gate", False)
        de_cf_gate_init = getattr(
            data_args, "depth_embed_cartesian_fourier_gate_init", 1.0)
        de_cf_mlp_init_std = getattr(
            data_args, "depth_embed_cartesian_fourier_mlp_init_std", 0.02)

        inner.init_depth_embedding(
            hidden_size=hidden_size,
            mode=de_mode,
            num_freqs=de_num_freqs,
            mlp_hidden=de_mlp_hidden,
            cartesian_fourier_use_rmsnorm=de_cf_use_rmsnorm,
            cartesian_fourier_per_channel_gate=de_cf_per_channel_gate,
            cartesian_fourier_gate_init=de_cf_gate_init,
            cartesian_fourier_mlp_init_std=de_cf_mlp_init_std,
        )
        inner._depth_embed_min = de_min
        inner._probe_depth_scale = float(getattr(data_args, "curriculum_depth_scale", 1.0))
        inner._depth_ratio_penalty_beta = float(getattr(data_args, "depth_ratio_penalty_beta", 0.0))
        inner._depth_ratio_penalty_mode = str(getattr(data_args, "depth_ratio_penalty_mode", "hinge"))
        inner._depth_ratio_penalty_r0 = float(getattr(data_args, "depth_ratio_penalty_r0", 0.5))
        inner._depth_fixed_ratio = float(getattr(data_args, "depth_embed_fixed_ratio", 0.0))

        if getattr(data_args, "depth_embed_learned_scale", False):
            inner.init_depth_embed_learned_scale()
            print("[depth_embed] Learned scale enabled: exp(param), init=1.0")

        enc_dim = 3 * (1 + 2 * de_num_freqs) + 3
        opts = []
        if de_cf_use_rmsnorm:
            opts.append("RMSNorm")
        if de_cf_per_channel_gate:
            opts.append(f"per-channel-gate (init {de_cf_gate_init})")
        elif de_cf_gate_init != 1.0:
            opts.append(f"scalar-gate (init {de_cf_gate_init})")
        if inner._depth_fixed_ratio > 0:
            opts.append(f"fixed-ratio {inner._depth_fixed_ratio} (gate inert)")
        if de_cf_mlp_init_std != 0.02:
            # Report the REALIZED std, not the requested one. The flag is the
            # only lever that changes how fast the depth branch grows, so a run
            # whose log does not show it is void and must be readable as such
            # from the log alone.
            realized = float(
                inner.depth_cartesian_fourier_mlp[0].weight.detach().std())
            opts.append(
                f"mlp-init-std {de_cf_mlp_init_std} (realized {realized:.4f})")
        opt_str = f" [{', '.join(opts)}]" if opts else ""
        print(
            f"[depth_embed] Initialized cartesian_fourier depth embedding{opt_str}: "
            f"{de_num_freqs} freqs/axis, {enc_dim} input channels "
            f"(3×(1+2×{de_num_freqs}) xyz + 3 ray_dirs) → Linear({enc_dim}->hidden)"
        )

    # ---------------- Patch marker embedding ----------------
    # Default is False to match DataArguments.use_inline_patch_embedding. If
    # the default here was True, run_benchmarks.py (which may pass an argparse
    # Namespace without this attribute) would create a random module that
    # training never instantiated — silent asymmetry between the two paths.
    if getattr(data_args, "use_inline_patch_embedding", False):
        if hasattr(inner, "init_inline_patch_embedding"):
            inner.init_inline_patch_embedding(hidden_size=hidden_size)
            print(
                f"[inline_patch] Initialized constant patch-marker embedding: "
                f"single [{hidden_size}] learned vector (std=1.0) added to inline "
                f"patch marker tokens for identity discrimination."
            )
    else:
        print(
            "[inline_patch] SKIPPED patch-marker identity bias "
            "(use_inline_patch_embedding=False). Inline markers will carry ONLY "
            "their copied source-patch features; the model must discriminate "
            "markers from raw canvas patches via features + MRoPE alone."
        )
