"""Qwen3-VL-specific live feature extraction.

Wraps HF Qwen3-VL's ``get_image_features`` so callers (the inference
pipeline, any future eval-time live-features path) don't have to inline
the per-image / per-layer reshape. Returns the model-agnostic
``[N_imgs, N_layers, H, W, C]`` tensor that ``reproject_scene``
consumes — Qwen3-VL specifics (DeepStack layer unpacking, the
hardcoded spatial-merge-size of 2, the
``image_grid_thw[:, 1:] // merge_size`` token-grid math) stay isolated
here so other adapters can ship their own ``feature_extraction.py``
without grepping pipelines.
"""

import torch


def split_image_features(image_outputs):
    """Normalize ``get_image_features()`` output across transformers versions.

    On the 4.57.x floor ``pyproject.toml`` declares, ``get_image_features`` has
    the signature ``(pixel_values, image_grid_thw)`` and returns a plain
    ``(image_embeds, deepstack_embeds)`` tuple, with ``image_embeds`` already
    split into per-image ``[tokens_per_img, C]`` tensors. On 5.x it returns a
    ``BaseModelOutputWithDeepstackFeatures`` exposing the same two things as
    ``.pooler_output`` and ``.deepstack_features``.

    Reading ``.pooler_output`` unconditionally (and passing ``return_dict=True``,
    which 4.57.x does not accept) made the live-ViT forward, the precompute
    script, and the stage-1 feature stash all die on the declared floor with
    ``'tuple' object has no attribute 'pooler_output'``. Callers must not pass
    ``return_dict``: omitting it works on both lines, and this normalizes
    whichever shape comes back.
    """
    if isinstance(image_outputs, (tuple, list)):
        embeds = image_outputs[0]
        deepstack = image_outputs[1] if len(image_outputs) > 1 else None
    else:
        embeds = image_outputs.pooler_output
        deepstack = getattr(image_outputs, "deepstack_features", None)
    return embeds, deepstack


def extract_features(model, pixel_values, image_grid_thw, merge_size: int = 2) -> torch.Tensor:
    """Run Qwen3-VL's visual encoder and stack outputs to ``[N_imgs, N_layers, H, W, C]``.

    HF Qwen3-VL's ``get_image_features`` returns
    ``(image_embeds, deepstack_features)`` where ``image_embeds`` is a
    tuple of per-image ``[tokens_per_img, C]`` tensors and
    ``deepstack_features`` is a list of per-layer ``[N_total_tokens, C]``
    tensors (one per DeepStack layer index). This helper reshapes both
    into a single stacked tensor with shape
    ``[N_imgs, N_layers, H_feat, W_feat, C]``, layer 0 first, with all
    spatial dims recovered from ``image_grid_thw``.

    The output tensor is what ``reproject_scene`` expects as its
    ``features`` argument — model-agnostic from that point on.
    """
    vision_output = model.get_image_features(pixel_values, image_grid_thw)
    image_embeds, deepstack_features = split_image_features(vision_output)

    h_feat = image_grid_thw[:, 1] // merge_size
    w_feat = image_grid_thw[:, 2] // merge_size
    tokens_per_img = (h_feat * w_feat).tolist()

    per_image_layers = [[] for _ in range(len(tokens_per_img))]
    for i, img_feat in enumerate(image_embeds):
        per_image_layers[i].append(img_feat.view(int(h_feat[i]), int(w_feat[i]), -1))
    for layer_tensor in deepstack_features:
        chunks = torch.split(layer_tensor, tokens_per_img, dim=0)
        for i, chunk in enumerate(chunks):
            per_image_layers[i].append(chunk.view(int(h_feat[i]), int(w_feat[i]), -1))

    return torch.stack([torch.stack(layers) for layers in per_image_layers])
