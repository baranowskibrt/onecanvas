"""Probe marker stash: auto-cached pool of real patch features used as
scene-agnostic marker content for probe training.

The stash is built once per (model, size) tuple by running the visual tower
on a diverse pool of held-out scene images. At training time, probe samples
harvest marker and synthetic-OBB feature vectors from this stash instead of
loading scene images and running the ViT per sample. This is the third
layer of the geometry-signal-only curriculum (stripped canvas + position-ID
geometry + scene-agnostic marker content).

Cache location: ``${XDG_CACHE_HOME:-~/.cache}/onecanvas_features/<feature_prefix>_obb_feature_stash_<size>_<sources>[_d<width>].pt``,
where the width suffix appears for any tower other than the family default
(Qwen3-VL-2B writes ``_d2048``).

Call ``maybe_build_obb_feature_stash(model, processor, data_args)`` from train.py
AFTER the model is fully loaded but BEFORE the datasets are instantiated.
The dataset's ``__init__`` then calls ``load_obb_feature_stash(...)`` to read the
cached tensor.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from copy import copy
from typing import Optional

import torch

from model_adapters.qwen3_vl.feature_extraction import split_image_features


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> str:
    return os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
        "onecanvas_features",
    )


def _sources_str(data_args) -> str:
    """The curriculum_scene_sources string the stash is harvested from. The stash
    content depends on it, so it is part of the cache identity."""
    return str(getattr(
        data_args, "curriculum_scene_sources",
        "vica_scannet_base,vica_arkit_base,vica_snpp_base",
    ))


def _sources_tag(sources: str) -> str:
    """Short stable fingerprint of the (order-insensitive) source set for the
    cache filename, so switching curriculum_scene_sources picks a different file
    instead of silently reusing a stale stash."""
    norm = ",".join(sorted(s.strip() for s in sources.split(",") if s.strip()))
    return hashlib.sha1(norm.encode()).hexdigest()[:8]


# Feature width of each family's default tower (Qwen3-VL-8B, Qwen3.5-9B). The
# stash is that tower's output, so a different model size needs its own file.
# Every stash cached before the width entered the name came from a default
# tower, so this width keeps the unsuffixed name and those files stay valid.
_DEFAULT_STASH_WIDTH = {"qwen3_vl": 4096, "qwen3_5": 4096}


def _cache_path(feature_prefix: str, size: int, sources: str,
                width: Optional[int] = None) -> str:
    tag = _sources_tag(sources)
    width_tag = ""
    if width is not None and int(width) != _DEFAULT_STASH_WIDTH.get(feature_prefix):
        width_tag = f"_d{int(width)}"
    return os.path.join(
        _cache_dir(),
        f"{feature_prefix}_obb_feature_stash_{size}_{tag}{width_tag}.pt")


def _model_feature_width(model) -> int:
    """Width of the image features the model's tower emits, which is the
    language model's hidden size for every supported family."""
    cfg = _resolve_inner_model(model).config
    return int(getattr(cfg, "text_config", cfg).hidden_size)


# ---------------------------------------------------------------------------
# DDP helpers (tolerate non-distributed launches)
# ---------------------------------------------------------------------------

def _is_rank_zero() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return int(os.environ.get("RANK", "0")) == 0


def _barrier() -> None:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_obb_feature_stash(data_args, feature_prefix: str, processor=None) -> torch.Tensor:
    """Load the cached marker stash tensor, building it on first use if missing.

    The stash is normally built at train.py startup by
    ``maybe_build_obb_feature_stash``. When it is absent (e.g. the README
    dataset quickstart, which never runs train.py) and a ``processor`` is
    available, it is built on the fly here so the dataset does not hard-crash.
    Returns a ``[N_PATCHES, N_LAYERS, D]`` bf16 tensor on CPU.
    """
    size = int(data_args.curriculum_obb_feature_stash_size)
    sources = _sources_str(data_args)
    # The startup build records the path it resolved from the loaded model.
    path = (getattr(data_args, "curriculum_obb_feature_stash_path", None)
            or _cache_path(feature_prefix, size, sources))
    if not os.path.exists(path):
        _build_stash_on_miss(data_args, feature_prefix, size, path, processor)
        path = getattr(data_args, "curriculum_obb_feature_stash_path", None) or path
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = blob.get("manifest", {})
    if m.get("feature_prefix") != feature_prefix:
        raise RuntimeError(
            f"[probe-stash] manifest feature_prefix mismatch: "
            f"{m.get('feature_prefix')!r} in cache vs {feature_prefix!r} expected. "
            f"Delete {path} and re-run to rebuild."
        )
    if m.get("curriculum_scene_sources") not in (None, sources):
        raise RuntimeError(
            f"[probe-stash] manifest curriculum_scene_sources mismatch: "
            f"{m.get('curriculum_scene_sources')!r} in cache vs {sources!r} expected. "
            f"Delete {path} and re-run to rebuild."
        )
    patches = blob["patches"]
    width = getattr(data_args, "curriculum_obb_feature_stash_width", None)
    if width is not None and int(patches.shape[2]) != int(width):
        raise RuntimeError(
            f"[probe-stash] {path} holds {patches.shape[2]}-wide features but the "
            f"model emits {width}-wide ones. The stash comes from a different "
            f"model size. Delete the file and re-run to rebuild."
        )
    print(
        f"[probe-stash] loaded {patches.shape[0]} patches "
        f"(layers={patches.shape[1]}, D={patches.shape[2]}, dtype={patches.dtype}) "
        f"from {path}"
    )
    return patches


def maybe_build_obb_feature_stash(model, processor, data_args) -> None:
    """Build the stash if the cache is missing. Rank 0 builds; others wait.

    No-op when ``curriculum_obb_feature_stash_enable`` is False.
    """
    if not getattr(data_args, "curriculum_obb_feature_stash_enable", False):
        return

    adapter = _resolve_adapter(processor, data_args)
    feature_prefix = adapter.config.feature_prefix
    size = int(data_args.curriculum_obb_feature_stash_size)
    sources = _sources_str(data_args)
    width = _model_feature_width(model) if model is not None else None
    path = _cache_path(feature_prefix, size, sources, width)
    # The dataset reads the stash from exactly this path and checks its width.
    data_args.curriculum_obb_feature_stash_path = path
    data_args.curriculum_obb_feature_stash_width = width

    # Only rank 0 checks existence and builds; EVERY rank then meets a single
    # barrier below. Branching each rank on its own os.path.exists is unsafe:
    # NFS attribute-cache lag can make ranks disagree about whether the cache
    # is present, and the old code returned early (no barrier) on the cache-hit
    # branch while other ranks blocked on _barrier() -> deadlock.
    if _is_rank_zero():
        if os.path.exists(path):
            print(f"[probe-stash] cache hit: {path}")
        else:
            print(f"[probe-stash] cache miss: building {size} patches for "
                  f"{feature_prefix!r} -> {path}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            t0 = time.time()
            patches = _build_stash(model, processor, data_args, target_patches=size)
            manifest = {
                "feature_prefix": feature_prefix,
                "curriculum_scene_sources": sources,
                "n_layers": int(patches.shape[1]),
                "d": int(patches.shape[2]),
                "build_time_s": round(time.time() - t0, 1),
            }
            tmp = path + ".tmp"
            torch.save({"patches": patches, "manifest": manifest}, tmp)
            os.replace(tmp, path)
            print(
                f"[probe-stash] saved {patches.shape[0]} patches "
                f"(shape={tuple(patches.shape)}, dtype={patches.dtype}) "
                f"in {time.time()-t0:.1f}s -> {path}"
            )
    _barrier()


# Base model to harvest the stash from when data_args carries no explicit
# model_name_or_path (build-on-miss from the dataset quickstart). Keyed by the
# adapter feature_prefix; matches the ModelArguments default.
_DEFAULT_STASH_BASE_MODEL = {"qwen3_vl": "Qwen/Qwen3-VL-8B-Instruct"}


def _build_stash_on_miss(data_args, feature_prefix, size, path, processor) -> None:
    """Build the stash when ``load_obb_feature_stash`` finds no cache.

    Loads the base VLM's visual tower on rank 0 and delegates to
    ``maybe_build_obb_feature_stash`` (which handles the build/save + the
    single cross-rank barrier). Raises a clear, actionable error when it cannot
    build (no processor / unknown base model) instead of a cryptic one.
    """
    if processor is None:
        raise RuntimeError(
            f"[probe-stash] cache not found at {path} and it cannot be auto-built: "
            f"load_obb_feature_stash was called without a processor. Pass the "
            f"dataset's processor, pre-build the stash "
            f"(maybe_build_obb_feature_stash at startup), or set "
            f"curriculum_obb_feature_stash_enable=False to use the per-sample "
            f"reprojection path."
        )
    model = None
    try:
        if _is_rank_zero():
            model_path = (
                getattr(data_args, "model_name_or_path", None)
                or _DEFAULT_STASH_BASE_MODEL.get(feature_prefix)
            )
            if not model_path:
                raise RuntimeError(
                    f"[probe-stash] cannot determine the base model to build the "
                    f"{feature_prefix!r} stash from. Set data_args.model_name_or_path "
                    f"to the model the processor was loaded from, or pre-build the "
                    f"stash, or set curriculum_obb_feature_stash_enable=False."
                )
            approx_mb = round(size * 32 / 1024)  # ~32 KB/patch (Qwen3-VL-8B)
            print(
                f"[probe-stash] cache miss at {path}: building {size} patches "
                f"(~{approx_mb} MB) once by loading {model_path!r} and running its "
                f"visual tower over held-out scenes. Set "
                f"curriculum_obb_feature_stash_enable=False to skip this."
            )
            model = _load_stash_build_model(model_path, feature_prefix, data_args)
        maybe_build_obb_feature_stash(model, processor, data_args)
    finally:
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _load_stash_build_model(model_path, feature_prefix, data_args):
    """Load the base 3D model (visual tower) for a build-on-miss harvest."""
    if feature_prefix == "qwen3_vl":
        from model_adapters.qwen3_vl.model import \
            Qwen3VL3DForConditionalGeneration as _Cls
    else:
        from model_adapters.qwen3_5.model import \
            Qwen3_5_3DForConditionalGeneration as _Cls
    from model_adapters.attention import resolve_attn_implementation
    attn = resolve_attn_implementation(
        getattr(data_args, "attn_implementation", "flash_attention_2"))
    model = _Cls.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation=attn)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_adapter(processor, data_args):
    """Build a fresh VLMAdapter (same pattern as SceneQADataset) to get the
    feature_prefix / config. The adapter is cheap — it's just metadata +
    processor-config tables; the model's visual tower lives separately."""
    from model_adapters import get_adapter
    return get_adapter(processor, data_args)


def _resolve_inner_model(model):
    """Find the object that owns get_image_features + .visual.

    With PEFT's LoRA wrap the chain is:
        PeftModel -> .base_model (LoraModel) -> .model (Qwen3VL3DForConditionalGeneration)
    Without PEFT it's the top-level model itself. Also tolerate an extra
    ``.model`` level for older wrappings.
    """
    def _has(obj):
        return (
            obj is not None
            and hasattr(obj, "get_image_features")
            and hasattr(obj, "visual")
        )

    candidates = [
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
    ]
    for c in candidates:
        if _has(c):
            return c
    raise RuntimeError(
        "[probe-stash] could not locate inner model with get_image_features + .visual. "
        "Checked model / model.model / model.model.model / model.base_model / "
        "model.base_model.model / model.base_model.model.model."
    )


def _build_stash(model, processor, data_args, target_patches: int) -> torch.Tensor:
    """Build the stash by running the visual tower on a diverse pool of scenes.

    Returns [target_patches, N_LAYERS, D] bf16 tensor on CPU.
    """
    inner = _resolve_inner_model(model)

    # The visual tower is typically on CPU at this point — DeepSpeed moves
    # weights to GPU later via the engine. Flash-attn has no CPU kernel, so
    # we move just the ViT to CUDA for the build and restore afterwards.
    visual = inner.visual
    _orig_visual_device = next(visual.parameters()).device
    _moved = False
    if torch.cuda.is_available() and _orig_visual_device.type != "cuda":
        print(
            f"[probe-stash] moving visual tower "
            f"{_orig_visual_device.type} -> cuda for stash build"
        )
        inner.visual = visual.to("cuda")
        _moved = True
    device = next(inner.visual.parameters()).device

    try:
        return _build_stash_inner(
            processor, data_args, target_patches, inner, device
        )
    finally:
        if _moved:
            print(
                f"[probe-stash] restoring visual tower cuda -> "
                f"{_orig_visual_device.type}"
            )
            inner.visual = inner.visual.to(_orig_visual_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _build_stash_inner(
    processor, data_args, target_patches: int, inner, device,
) -> torch.Tensor:
    from .data_processor_3d import SceneQADataset  # local import: avoids cycle

    # Build a private scene source over the configured probe scene sources.
    _da = copy(data_args)
    _da.dataset_use = str(getattr(
        data_args, "curriculum_scene_sources",
        "vica_scannet_base,vica_arkit_base,vica_snpp_base",
    ))
    _da.skip_asset_validation = True
    scene_src = SceneQADataset(
        processor, _da, data_split="train", sort=False,
    )
    # Small per-scene frame budget to keep build fast. Build-time only.
    scene_src.num_images = 8

    scenes = list(scene_src.list_data_dict)
    random.Random(42).shuffle(scenes)

    # Aim for ~3x target before random subsample, so we span many scenes.
    raw_target = target_patches * 3
    raw_chunks: list = []
    raw_count = 0
    scenes_used = 0
    last_report = 0

    for item in scenes:
        if raw_count >= raw_target:
            break
        try:
            assets = scene_src._load_scene_data(
                item["scene_id"], item["data_path"], sample_idx=0,
                pinned_frames=item.get("pinned_stems") or item.get("images"),
                scene_subdir=item.get("scene_subdir"),
                wants_aligned=False,
            )
        except Exception as e:
            print(f"[probe-stash] skip {item.get('scene_id')}: {e}")
            continue
        if assets is None or not assets.get("images"):
            continue

        images = assets["images"]
        # Process images through the tokenizer-free image path (image processor
        # only), matching how the curriculum dataset harvests patch features.
        processed = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device)
        image_grid_thw = processed["image_grid_thw"].to(device)

        with torch.no_grad():
            image_outputs = inner.get_image_features(pixel_values, image_grid_thw)

        # tuple of [H*W, C], plus the per-layer DeepStack list. Normalized
        # because 4.57.x returns a bare tuple and 5.x a dataclass.
        image_embeds_per_img, deepstack_features = split_image_features(image_outputs)

        # Concat all per-image patches along the patch dimension.
        base = torch.cat([emb for emb in image_embeds_per_img], dim=0)  # [N, C]

        if deepstack_features is not None and len(deepstack_features) > 0:
            layers = [base]
            layers.extend(list(deepstack_features))  # each: [N, C]
            all_features = torch.stack(layers, dim=1)  # [N, N_layers, C]
        else:
            all_features = base.unsqueeze(1)  # [N, 1, C]

        raw_chunks.append(all_features.to("cpu", dtype=torch.bfloat16))
        raw_count += all_features.shape[0]
        scenes_used += 1

        if raw_count - last_report >= 5000:
            print(
                f"[probe-stash] build progress: {raw_count} raw patches "
                f"from {scenes_used} scenes"
            )
            last_report = raw_count

    if raw_count == 0:
        raise RuntimeError(
            "[probe-stash] no patches harvested — scene source exhausted with no valid scenes."
        )

    raw = torch.cat(raw_chunks, dim=0)  # [N_raw, N_layers, C]
    print(
        f"[probe-stash] raw pool assembled: {raw.shape[0]} patches from "
        f"{scenes_used} scenes (shape={tuple(raw.shape)}, dtype={raw.dtype})"
    )

    # Subsample to target (random, no aggressive diversity filter — the
    # natural distribution is intentionally preserved so the training-time
    # marker content distribution matches what markers see at inference).
    if raw.shape[0] > target_patches:
        perm = torch.randperm(raw.shape[0], generator=torch.Generator().manual_seed(7))
        raw = raw[perm[:target_patches]]
    elif raw.shape[0] < target_patches:
        print(
            f"[probe-stash] WARNING: only {raw.shape[0]} patches harvested "
            f"vs target {target_patches}. Consider increasing scene_src.num_images "
            f"or widening curriculum_scene_sources."
        )

    return raw.contiguous()
