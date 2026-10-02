"""Save / load helpers for the project's auxiliary 3D embedding modules.

Depth, angle, and camera-marker embeddings are not part of the PEFT adapter,
so they have to be persisted separately as ``.pt`` files alongside the LoRA
weights. The fully fine-tuned ``visual.merger`` (tune_mm_mlp=True) is also
persisted here for the same reason — PEFT only stores LoRA targets, so an
un-saved trained merger silently reverts to HF base weights at eval time and
makes training gen_eval disagree with run_benchmarks.

Both ``train.py`` (end-of-train save, best-checkpoint save, warm-start load)
and ``run_benchmarks.py`` (eval-time load) call into here so that the .pt
format and the wrapper-walking logic live in exactly one place.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

# NOTE: ``get_inner_3d_model`` lives in the training package (``onecanvas``).
# It is imported lazily inside the two functions below so that merely importing
# ``utils`` (or ``utils.embedding_io``) does not drag in the whole training /
# model stack. This keeps ``geometry`` / ``reprojection`` / ``inference`` usable
# standalone even though they re-export ``utils`` symbols.


__all__ = ["save_3d_embeddings", "load_3d_embeddings"]


_DEPTH_HANDOFF_VERSION = 1
_DEPTH_HANDOFF_CALIBRATION = {
    "method": "deterministic_synthetic_rays_v1",
    "seed": 0,
    "samples": 20000,
    "depth_min_m": 0.3,
    "depth_max_m": 6.0,
    # Measured for the maintained Qwen3-VL feature distribution. This is a
    # calibration reference, not a checkpoint-specific gate multiplier.
    "visual_norm": 20.3,
}


def _source_config(ckpt_dir: str) -> tuple[dict, str]:
    """Read the source run configuration required by a stage transition."""
    ckpt = Path(ckpt_dir)
    candidates = (ckpt / "resolved_config.json", ckpt.parent / "resolved_config.json")
    config_path = next((p for p in candidates if p.is_file()), None)
    if config_path is None:
        raise RuntimeError(
            "Cannot perform the free-running stage-2 depth handoff: the source "
            f"checkpoint {ckpt} has no resolved_config.json in the checkpoint or "
            "its parent directory. Restore that source run metadata (including the "
            "actual depth_embed_fixed_ratio) before starting the new stage."
        )
    with config_path.open() as f:
        return json.load(f), str(config_path)


def _source_fixed_ratio(config: dict, config_path: str) -> float:
    data = config.get("data")
    if not isinstance(data, dict) or "depth_embed_fixed_ratio" not in data:
        raise RuntimeError(
            "Cannot perform the free-running stage-2 depth handoff: source metadata "
            f"{config_path} does not record data.depth_embed_fixed_ratio. Add the "
            "actual source setting to resolved_config.json; refusing to load fixed-ratio "
            "weights as uncompensated free-running depth."
        )
    try:
        ratio = float(data["depth_embed_fixed_ratio"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid data.depth_embed_fixed_ratio={data['depth_embed_fixed_ratio']!r} "
            f"in source metadata {config_path}."
        ) from exc
    if not math.isfinite(ratio) or ratio < 0:
        raise RuntimeError(
            f"Invalid data.depth_embed_fixed_ratio={ratio!r} in source metadata "
            f"{config_path}; expected a finite non-negative value."
        )
    return ratio


def _calibrated_gate_scale(inner, source_ratio: float) -> tuple[float, float]:
    """Derive the one-time scalar from the loaded source embedding state."""
    required = (
        "_encode_depth_cartesian_fourier",
        "depth_cartesian_fourier_gate",
        "depth_cartesian_fourier_mlp",
        "_depth_fourier_freqs",
    )
    missing = [name for name in required if not hasattr(inner, name)]
    if missing:
        raise RuntimeError(
            "Cannot compensate fixed-ratio source depth: depth_embedding.pt/model "
            f"is missing {', '.join(missing)}. Use a complete compatible source "
            "checkpoint or keep stage 2 explicitly fixed-ratio."
        )

    # Generate on CPU so the sample is identical regardless of launch device,
    # then evaluate the already-loaded source module in bounded chunks.
    calibration = _DEPTH_HANDOFF_CALIBRATION
    generator = torch.Generator(device="cpu").manual_seed(calibration["seed"])
    count = calibration["samples"]
    rays = torch.nn.functional.normalize(
        torch.randn(count, 3, generator=generator, dtype=torch.float32), dim=-1)
    depths = calibration["depth_min_m"] + torch.rand(
        count, generator=generator, dtype=torch.float32
    ) * (calibration["depth_max_m"] - calibration["depth_min_m"])
    parameter = next(inner.depth_cartesian_fourier_mlp.parameters())
    total = 0.0
    with torch.no_grad():
        for start in range(0, count, 256):
            stop = min(start + 256, count)
            encoded = inner._encode_depth_cartesian_fourier(
                depths[start:stop].to(device=parameter.device, dtype=parameter.dtype),
                rays[start:stop].to(device=parameter.device, dtype=parameter.dtype),
            )
            total += encoded.detach().float().norm(dim=-1).sum().cpu().item()
    raw_ratio = (total / count) / calibration["visual_norm"]
    if not math.isfinite(raw_ratio) or raw_ratio <= 0:
        raise RuntimeError(
            f"Cannot compensate fixed-ratio source depth: derived raw ratio is {raw_ratio!r}. "
            "Inspect the source depth_embedding.pt for non-finite or zero weights."
        )
    scale = source_ratio / raw_ratio
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            f"Cannot compensate fixed-ratio source depth: derived gate scale is {scale!r} "
            f"from source ratio {source_ratio} and raw ratio {raw_ratio}."
        )
    return scale, raw_ratio


def _validate_calibration_metadata(inner, config: dict, config_path: str) -> None:
    data = config.get("data", {})
    expected_fields = {
        "depth_embed_num_freqs": int,
        "depth_embed_mlp_hidden": int,
        "depth_embed_cartesian_fourier_use_rmsnorm": bool,
        "depth_embed_cartesian_fourier_per_channel_gate": bool,
        "depth_embed_learned_scale": bool,
        "curriculum_depth_scale": float,
    }
    missing = [name for name in expected_fields if name not in data]
    if missing:
        raise RuntimeError(
            "Cannot compensate fixed-ratio source depth: source metadata "
            f"{config_path} is missing {', '.join(missing)}. Restore these actual "
            "source architecture settings; refusing to guess a calibration layout."
        )
    first_linear = inner.depth_cartesian_fourier_mlp[0]
    actual = {
        "depth_embed_num_freqs": int(inner._depth_fourier_freqs.numel()),
        "depth_embed_mlp_hidden": int(first_linear.out_features),
        "depth_embed_cartesian_fourier_use_rmsnorm": hasattr(
            inner, "depth_cartesian_fourier_norm"
        ),
        "depth_embed_cartesian_fourier_per_channel_gate": (
            inner.depth_cartesian_fourier_gate.numel() > 1
        ),
        "depth_embed_learned_scale": hasattr(inner, "depth_embed_log_scale"),
        "curriculum_depth_scale": float(
            getattr(inner, "_probe_depth_scale", 1.0)
        ),
    }
    mismatches = [
        f"{name}: source={data[name]!r}, loaded={actual[name]!r}"
        for name in expected_fields
        if expected_fields[name](data[name]) != actual[name]
    ]
    if mismatches:
        raise RuntimeError(
            "Cannot compensate fixed-ratio source depth because the requested stage-2 "
            "module does not match its recorded source architecture ("
            + "; ".join(mismatches)
            + "). Match the stage-2 depth embedding arguments to the source checkpoint."
        )
    if actual["depth_embed_learned_scale"] or actual["curriculum_depth_scale"] != 1.0:
        raise RuntimeError(
            "Cannot compensate fixed-ratio source depth with a post-normalization "
            "learned/probe scale. The maintained handoff requires "
            "depth_embed_learned_scale=False and curriculum_depth_scale=1.0; keep "
            "stage 2 explicitly fixed-ratio for this source."
        )


def _apply_stage2_depth_handoff(inner, ckpt_dir: str, target_fixed_ratio: float) -> dict:
    """Record an explicit pin, or convert a fixed source to free-running once."""
    if not math.isfinite(target_fixed_ratio) or target_fixed_ratio < 0:
        raise RuntimeError(
            f"Invalid stage-2 depth_embed_fixed_ratio={target_fixed_ratio!r}; expected "
            "zero for the free-running default or a positive explicit override."
        )
    config, config_path = _source_config(ckpt_dir)
    source_ratio = _source_fixed_ratio(config, config_path)
    previous = getattr(inner, "_depth_handoff_metadata", None)
    gate = getattr(inner, "depth_cartesian_fourier_gate", None)
    gate_before = None if gate is None else float(gate.detach().float().mean().cpu())
    metadata = {
        "version": _DEPTH_HANDOFF_VERSION,
        "source_fixed_ratio": source_ratio,
        "target_fixed_ratio": float(target_fixed_ratio),
        "source_config": config_path,
        "conversion_applied": False,
        "gate_scale": 1.0,
        "gate_before": gate_before,
        "gate_after": gate_before,
    }
    if previous is not None:
        metadata["previous_handoff"] = previous

    if target_fixed_ratio > 0:
        metadata["mode"] = "explicit_fixed_ratio_override"
    elif source_ratio > 0:
        _validate_calibration_metadata(inner, config, config_path)
        scale, raw_ratio = _calibrated_gate_scale(inner, source_ratio)
        if gate is None:
            raise RuntimeError(
                "Cannot compensate fixed-ratio source depth: the loaded model has no "
                "depth_cartesian_fourier_gate. Use a compatible checkpoint."
            )
        with torch.no_grad():
            gate.mul_(scale)
        metadata.update({
            "mode": "fixed_to_free",
            "conversion_applied": True,
            "gate_scale": scale,
            "source_raw_ratio": raw_ratio,
            "gate_after": float(gate.detach().float().mean().cpu()),
            "calibration": dict(_DEPTH_HANDOFF_CALIBRATION),
        })
    else:
        metadata["mode"] = "free_to_free"

    inner._depth_handoff_metadata = metadata
    print(
        "[depth_handoff] "
        f"mode={metadata['mode']} source_fixed_ratio={source_ratio} "
        f"target_fixed_ratio={target_fixed_ratio} conversion_applied="
        f"{metadata['conversion_applied']} gate_scale={metadata['gate_scale']:.8g}"
    )
    return metadata


def _trimmed(value):
    """Detach a tensor (or a state dict of tensors) onto its own storage.

    Under DeepSpeed ZeRO every parameter is a VIEW into one flat contiguous
    buffer, and ``torch.save`` serializes a view's whole backing storage. The
    3D position embedding is 2.2M parameters, 4.3 MB in bfloat16, but the
    stage-1 checkpoints saved before this fix are 1.4 GB: the geometry MLP's
    ``2.weight`` is a 4 MB view into a 1,401,036,832-byte adapter buffer, so
    each save wrote the entire LoRA alongside it. The file loads correctly
    either way, which is why it went unnoticed; it is 334x the download it
    needs to be. ``clone()`` gives the tensor its own exactly-sized storage.
    """
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _trimmed(v) for k, v in value.items()}
    return value


def save_3d_embeddings(model, output_dir: str, *, log_prefix: str = "") -> dict:
    """Save depth/angle/camera embedding state to ``{output_dir}/<name>.pt``.

    Silently skips a file if the corresponding module isn't initialized on
    the inner model. Returns a dict mapping kind → saved path for what was
    actually written (useful for tests / logging).

    The on-disk format matches what ``run_benchmarks.py`` and the existing
    train.py warm-start path expect:

      depth_embedding.pt:
        {"fourier_proj": state_dict, "fourier_freqs": tensor, "gate": tensor,
         "depth_handoff": optional metadata}
        OR
        {"embedding": state_dict, "gate": tensor}
    """
    from onecanvas.setup_3d import get_inner_3d_model

    inner = get_inner_3d_model(model)
    saved: dict[str, str] = {}
    prefix = f"{log_prefix} " if log_prefix else ""

    # ---------------- Depth ----------------
    de_state: dict = {}
    if hasattr(inner, "depth_fourier_proj"):
        de_state["fourier_proj"] = inner.depth_fourier_proj.state_dict()
        if hasattr(inner, "_depth_fourier_freqs"):
            de_state["fourier_freqs"] = inner._depth_fourier_freqs
    if hasattr(inner, "depth_loc_proj"):
        de_state["loc_proj"] = inner.depth_loc_proj.state_dict()
    if hasattr(inner, "depth_cartesian_proj"):
        de_state["cartesian_proj"] = inner.depth_cartesian_proj.state_dict()
    if hasattr(inner, "depth_cartesian_fourier_mlp"):
        de_state["cartesian_fourier_mlp"] = inner.depth_cartesian_fourier_mlp.state_dict()
        if hasattr(inner, "_depth_fourier_freqs"):
            de_state["cartesian_fourier_freqs"] = inner._depth_fourier_freqs
    if hasattr(inner, "depth_cartesian_fourier_norm"):
        de_state["cartesian_fourier_norm"] = inner.depth_cartesian_fourier_norm.state_dict()
    if hasattr(inner, "depth_cartesian_fourier_gate"):
        de_state["cartesian_fourier_gate"] = inner.depth_cartesian_fourier_gate.data
    if not de_state and hasattr(inner, "depth_embedding"):
        de_state["embedding"] = inner.depth_embedding.state_dict()
    if hasattr(inner, "depth_embed_log_scale"):
        de_state["learned_scale"] = inner.depth_embed_log_scale.data
    if hasattr(inner, "_depth_handoff_metadata"):
        de_state["depth_handoff"] = inner._depth_handoff_metadata
    if de_state:
        path = os.path.join(output_dir, "depth_embedding.pt")
        torch.save(_trimmed(de_state), path)
        print(f"{prefix}[depth_embed] Saved depth embedding to: {path}")
        saved["depth"] = path

    # ---------------- Visual merger (tune_mm_mlp) ----------------
    # PEFT's adapter_model.safetensors only holds LoRA targets, so a fully
    # fine-tuned visual.merger (tune_mm_mlp=True) is otherwise lost on save
    # and silently reverts to the base HF weights at eval time.
    merger = getattr(getattr(inner, "visual", None), "merger", None)
    if merger is not None and any(p.requires_grad for p in merger.parameters()):
        path = os.path.join(output_dir, "visual_merger.pt")
        torch.save(_trimmed(merger.state_dict()), path)
        print(f"{prefix}[visual_merger] Saved visual merger to: {path}")
        saved["visual_merger"] = path

    return saved


def load_3d_embeddings(
    model,
    ckpt_dir: str,
    *,
    load_depth: bool = True,
    stage2_target_fixed_ratio: float | None = None,
    require_stage2_handoff: bool = False,
) -> dict:
    """Load any of depth/angle/camera embedding .pt files that exist in
    ``ckpt_dir`` into the inner 3D model. Returns a dict mapping kind →
    loaded-path for diagnostics.

    No-ops silently when:
      - the .pt file doesn't exist, OR
      - the model has no matching module to load into.

    Handles every storage format used historically:
      - fourier_proj + fourier_freqs (current)
      - embedding (legacy nn.Embedding state dict)
      - raw nn.Embedding state dict at the top level (very old)
      - optional gate parameter and stage-transition metadata

    Set ``load_depth=False`` to skip the depth_embedding.pt branch and leave
    the depth module at its post-init state (cold start). Angle/camera/
    reference are still loaded.

    ``stage2_target_fixed_ratio`` marks an initial stage transition. Zero is
    the maintained free-running default: a fixed-ratio source is compensated
    once from its saved state before the control is disabled. A positive value
    is the explicit fixed-ratio stage-2 override. ``require_stage2_handoff`` is
    for resume and refuses legacy uncompensated free-running checkpoints.
    """
    from onecanvas.setup_3d import get_inner_3d_model

    inner = get_inner_3d_model(model)
    loaded: dict[str, str] = {}

    # ---------------- Depth ----------------
    de_file = os.path.join(ckpt_dir, "depth_embedding.pt")
    if not load_depth and os.path.exists(de_file):
        print(f"[depth_embed] Skipping load of {de_file} (load_depth=False, cold start)")
    if load_depth and not os.path.exists(de_file) and (
        stage2_target_fixed_ratio is not None or require_stage2_handoff
    ):
        raise RuntimeError(
            f"Stage-2 depth handoff requires {de_file}, but it is missing. Restore a "
            "complete source/checkpoint or set load_depth_embed_from_stage1=False for "
            "an intentional cold start."
        )
    if load_depth and os.path.exists(de_file):
        de_state = torch.load(de_file, map_location="cpu", weights_only=True)
        did_load = False
        if "fourier_proj" in de_state and hasattr(inner, "depth_fourier_proj"):
            inner.depth_fourier_proj.load_state_dict(de_state["fourier_proj"])
            if "fourier_freqs" in de_state and hasattr(inner, "_depth_fourier_freqs"):
                inner._depth_fourier_freqs.copy_(de_state["fourier_freqs"])
            did_load = True
        if "loc_proj" in de_state and hasattr(inner, "depth_loc_proj"):
            inner.depth_loc_proj.load_state_dict(de_state["loc_proj"])
            did_load = True
        if "cartesian_proj" in de_state and hasattr(inner, "depth_cartesian_proj"):
            inner.depth_cartesian_proj.load_state_dict(de_state["cartesian_proj"])
            did_load = True
        if "cartesian_fourier_mlp" in de_state and hasattr(inner, "depth_cartesian_fourier_mlp"):
            inner.depth_cartesian_fourier_mlp.load_state_dict(de_state["cartesian_fourier_mlp"])
            if "cartesian_fourier_freqs" in de_state and hasattr(inner, "_depth_fourier_freqs"):
                inner._depth_fourier_freqs.copy_(de_state["cartesian_fourier_freqs"])
            did_load = True
        if "cartesian_fourier_norm" in de_state and hasattr(inner, "depth_cartesian_fourier_norm"):
            inner.depth_cartesian_fourier_norm.load_state_dict(de_state["cartesian_fourier_norm"])
        if "cartesian_fourier_gate" in de_state and hasattr(inner, "depth_cartesian_fourier_gate"):
            saved_gate = de_state["cartesian_fourier_gate"]
            live_gate = inner.depth_cartesian_fourier_gate
            if saved_gate.shape != live_gate.shape:
                raise RuntimeError(
                    f"depth_cartesian_fourier_gate shape mismatch: saved {tuple(saved_gate.shape)} "
                    f"vs live {tuple(live_gate.shape)}. Make sure "
                    f"depth_embed_cartesian_fourier_per_channel_gate matches the stage-1 checkpoint."
                )
            live_gate.data.copy_(saved_gate)
        if not did_load and "embedding" in de_state and hasattr(inner, "depth_embedding"):
            inner.depth_embedding.load_state_dict(de_state["embedding"])
            did_load = True
        elif not did_load and hasattr(inner, "depth_embedding") and "weight" in de_state:
            # Very old format: bare nn.Embedding state dict
            inner.depth_embedding.load_state_dict(de_state)
            did_load = True
        if "learned_scale" in de_state and hasattr(inner, "depth_embed_log_scale"):
            inner.depth_embed_log_scale.data.copy_(de_state["learned_scale"])
            print(f"[depth_embed] Loaded learned scale: exp({de_state['learned_scale'].item():.4f}) = "
                  f"{torch.exp(de_state['learned_scale']).item():.4f}")
        if "depth_handoff" in de_state:
            handoff = de_state["depth_handoff"]
            if not isinstance(handoff, dict):
                raise RuntimeError(
                    f"Invalid depth_handoff metadata in {de_file}: expected a mapping."
                )
            inner._depth_handoff_metadata = handoff
        if did_load:
            print(f"[depth_embed] Loaded depth embedding from {de_file}")
            loaded["depth"] = de_file
        else:
            print(
                f"[depth_embed] Warning: {de_file} found but no matching depth module on inner model "
                f"({type(inner).__name__})"
            )

        if require_stage2_handoff:
            handoff = getattr(inner, "_depth_handoff_metadata", None)
            valid_handoff = (
                isinstance(handoff, dict)
                and float(handoff.get("target_fixed_ratio", -1.0)) == 0.0
                and (
                    float(handoff.get("source_fixed_ratio", -1.0)) == 0.0
                    or handoff.get("conversion_applied") is True
                )
            )
            if not valid_handoff:
                raise RuntimeError(
                    "Refusing to resume free-running stage 2 without checkpointed depth "
                    f"handoff metadata in {de_file}. A fixed-ratio source must record an "
                    "applied conversion. Resume from a checkpoint produced by the maintained "
                    "handoff, or use an explicit positive --depth_embed_fixed_ratio override "
                    "for a deliberately pinned run."
                )
            print(
                "[depth_handoff] resume restored converted gate; "
                f"not reapplying scale {handoff.get('gate_scale')}"
            )
        elif stage2_target_fixed_ratio is not None:
            _apply_stage2_depth_handoff(
                inner, ckpt_dir, float(stage2_target_fixed_ratio)
            )


    # ---------------- Visual merger (tune_mm_mlp) ----------------
    vm_file = os.path.join(ckpt_dir, "visual_merger.pt")
    merger = getattr(getattr(inner, "visual", None), "merger", None)
    if os.path.exists(vm_file) and merger is not None:
        vm_state = torch.load(vm_file, map_location="cpu")
        merger.load_state_dict(vm_state)
        print(f"[visual_merger] Loaded visual merger from {vm_file}")
        loaded["visual_merger"] = vm_file

    return loaded
