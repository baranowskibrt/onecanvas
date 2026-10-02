"""Generation-time monkey patches — Qwen3-VL specific.

Lives in ``model_adapters/qwen3_vl/`` because the two patches it installs
(silencing HF's kwarg validator, persisting custom kwargs across decode
steps) target Qwen3-VL's GenerationMixin behavior. Other VLM backbones
(Qwen3.5, etc.) have their own patch needs — those should
get a sibling ``model_adapters/<model>/patches.py`` rather than
extending this one.

Used identically by training (``training/onecanvas/train/train.py``) and
benchmarks (``training/run_benchmarks.py``). Both call
``apply_patches(model)`` after the final model object is built (post-PEFT).
"""

import types

import torch


__all__ = ["apply_patches"]


def _silent_validate(self, model_kwargs):
    """Replacement for HF GenerationMixin._validate_model_kwargs that allows
    arbitrary custom kwargs (the 3D model passes ``projection_done``,
    ``projected_input_ids`` and friends through generate())."""
    return []


def apply_patches(model):
    """Install generation-time patches on ``model``.

    Two things get patched:

    1. ``_validate_model_kwargs`` is silenced so the 3D model's custom
       ``projected_*`` kwargs survive ``generate()``. This needs to happen
       on BOTH the top-level wrapper AND the inner backbone, because when
       ``model`` is a PeftModel, ``peft_model.generate()`` delegates to
       ``backbone.generate()`` with ``self=backbone``, and inside that call
       the lookup of ``self._validate_model_kwargs`` resolves to the
       backbone's own (un-patched) method.

    2. ``prepare_inputs_for_generation`` is wrapped so that custom 3D keys
       (``image_embeds``, ``deepstack_image_embeds``, ``poses``, ``depths``)
       persist across decode steps, and so that an accidental 1-D
       ``attention_mask`` (which crashes HF's GenerationMixin) is silently
       upcast to a standard 2-D mask of ones.

    Idempotent: a flag on the top-level model and another on the backbone
    prevent double-patching when this function is called multiple times.
    """
    if getattr(model, "_scanqa_patch_applied", False):
        return

    # ---------------- _validate_model_kwargs ----------------
    model._validate_model_kwargs = types.MethodType(_silent_validate, model)

    # When `model` is a PeftModel, walk PeftModel -> LoraModel -> backbone
    # and patch the backbone's _validate_model_kwargs as well.
    backbone = model
    if hasattr(backbone, "base_model"):              # PeftModelForCausalLM -> LoraModel
        lora_model = backbone.base_model
        if hasattr(lora_model, "model"):             # LoraModel -> actual backbone
            backbone = lora_model.model
    if backbone is not model and not getattr(backbone, "_backbone_validate_patched", False):
        backbone._validate_model_kwargs = types.MethodType(_silent_validate, backbone)
        backbone._backbone_validate_patched = True

    # ---------------- prepare_inputs_for_generation ----------------
    original_prepare_inputs = model.prepare_inputs_for_generation

    def patched_prepare_inputs(self, input_ids, **kwargs):
        # Defensive: HF sometimes hands a 1-D "packed" mask when batching is
        # exotic. The 3D model expects a 2-D [B, L] mask. Replace with all-ones.
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None and attention_mask.ndim == 1:
            kwargs["attention_mask"] = torch.ones(
                (input_ids.shape[0], input_ids.shape[1]),
                device=input_ids.device,
                dtype=torch.long,
            )

        model_inputs = original_prepare_inputs(input_ids, **kwargs)

        # Persist custom 3D keys across decode steps. HF's default
        # _update_model_kwargs_for_generation only carries forward what
        # prepare_inputs_for_generation returned in model_inputs.
        for key in ("image_embeds", "deepstack_image_embeds", "poses", "depths"):
            if key in kwargs:
                model_inputs[key] = kwargs[key]

        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs, model)
    model._scanqa_patch_applied = True
