"""Shared generation-based evaluation loop.

Both ``training/onecanvas/train/train.py`` (``GenerationEvalCallback``) and
``training/run_benchmarks.py`` call into this module so that batch preparation,
generation, decoding, and result collection are defined in exactly one place.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Callable

import torch

from .bbox import clean_generated_text

__all__ = ["prepare_model_inputs", "run_generation_loop"]


# ── Keys that are metadata / ground-truth and must never reach model.generate() ──
# ``projection_done`` is intentionally NOT here: the model's forward() uses it
# to select the prefill code path.
_METADATA_KEYS = frozenset({
    "labels",
    "answer",
    "question",
    "question_type",
    "scene_id",
    "images",
    "projected_labels",
    # Prompt-only variants (used for swap, then discarded)
    "projected_input_ids_prompt",
    "projected_position_ids_prompt",
    "projected_attention_mask_prompt",
    "rope_deltas_prompt",
})


def prepare_model_inputs(
    batch: dict,
    device: torch.device,
) -> tuple[dict, int]:
    """Build the ``model_inputs`` dict from a dataloader batch.

    Handles:
    1. Filtering out metadata / ground-truth keys (``_METADATA_KEYS``)
    2. Swapping prompt-only projected variants when present
    3. Creating dummy ``input_ids`` / ``attention_mask`` for projected batches
    4. Moving tensors to *device*

    Returns ``(model_inputs, input_len)`` where *input_len* is the sequence
    length used later to slice newly generated tokens.
    """
    # --- prompt-version swap (must happen before filtering) ----------------
    # When the dataloader produced prompt-only variants (e.g. depth-distance
    # pretraining task), swap them into the canonical projected key names so
    # the rest of the pipeline is unaware of the distinction.
    if batch.get("projection_done") and "projected_input_ids_prompt" in batch:
        batch["projected_input_ids"] = batch["projected_input_ids_prompt"]
        batch["projected_position_ids"] = batch["projected_position_ids_prompt"]
        batch["projected_attention_mask"] = batch["projected_attention_mask_prompt"]
        if "rope_deltas_prompt" in batch:
            batch["rope_deltas"] = batch["rope_deltas_prompt"]

    # --- filter & move to device ------------------------------------------
    model_inputs: dict = {}
    for k, v in batch.items():
        if k in _METADATA_KEYS or k.startswith("_"):
            continue
        if isinstance(v, torch.Tensor):
            v = v.to(device)
        model_inputs[k] = v

    # --- projected batch: dummy input_ids / attention_mask -----------------
    if "projected_input_ids" in model_inputs:
        max_len = max(ids.shape[-1] for ids in model_inputs["projected_input_ids"])
        B = len(model_inputs["projected_input_ids"])
        model_inputs["input_ids"] = torch.zeros(B, max_len, dtype=torch.long, device=device)
        model_inputs["attention_mask"] = torch.ones(B, max_len, dtype=torch.long, device=device)

    input_len = model_inputs["input_ids"].shape[1]
    return model_inputs, input_len


def run_generation_loop(
    model,
    processor,
    dataloader,
    device: torch.device,
    *,
    max_new_tokens: int = 128,
    compute_da3_geometry_fn: Callable | None = None,
    max_batches: int | None = None,
    verbose: bool = True,
    log_prefix: str = "[Eval]",
) -> tuple[list[dict], list[str]]:
    """Run greedy generation over *dataloader* and return results.

    Parameters
    ----------
    model : nn.Module
        The model (plain or PeftModel-wrapped). Must already be in eval mode
        with ``use_cache=True``.
    processor : transformers.AutoProcessor
        Used for ``batch_decode``.
    dataloader : DataLoader
        Yields batch dicts produced by the project's data collator.
    device : torch.device
        Target device for tensors.
    max_new_tokens : int
        Maximum tokens to generate per sample.
    compute_da3_geometry_fn : callable, optional
        ``fn(image) -> (depth, pose, intrinsic)`` — called per-image when
        geometry is not precomputed. Only used by ``run_benchmarks.py``.
    max_batches : int, optional
        Stop after this many batches (for debug / smoke-test runs).
    verbose : bool
        Print per-sample progress.
    log_prefix : str
        Prefix for log lines (e.g. ``"[GenEval step 500]"``).

    Returns
    -------
    results : list[dict]
        Each dict has keys: ``prediction``, ``ground_truths``, ``scene_id``,
        ``question``, ``question_type``.
    skipped_scene_ids : list[str]
        Scene IDs skipped due to missing precomputed features.
    """
    results: list[dict] = []
    skipped_scene_ids: list[str] = []

    for i, batch in enumerate(dataloader):
        torch.cuda.empty_cache()
        start = time.time()

        # Optional on-the-fly geometry (standalone eval without precomputed)
        if compute_da3_geometry_fn is not None and "depths" not in batch:
            images_for_geom = batch.get("images")
            if images_for_geom is not None:
                all_depths, all_poses, all_intrinsics = [], [], []
                for j in range(len(images_for_geom)):
                    d, p, k_ = compute_da3_geometry_fn(images_for_geom[j])
                    all_depths.append(d)
                    all_poses.append(p)
                    all_intrinsics.append(k_)
                batch["depths"] = torch.stack(all_depths)
                batch["poses"] = torch.stack(all_poses)
                batch["intrinsics"] = torch.stack(all_intrinsics)

        model_inputs, input_len = prepare_model_inputs(batch, device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
            )

        raw_gen_ids = generated_ids[:, input_len:]
        responses = processor.batch_decode(
            raw_gen_ids,
            skip_special_tokens=False,
        )
        responses = [clean_generated_text(r) for r in responses]

        # --- cleanup GPU memory ------------------------------------------
        del generated_ids, model_inputs
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - start

        # --- collect results ---------------------------------------------
        actual_batch = len(responses)
        for b in range(actual_batch):
            gt_text = batch["answer"][b]
            gt_list = [gt_text] if isinstance(gt_text, str) else gt_text
            scene_id = batch.get("scene_id", ["unknown"])[b]
            question_text = batch.get("question", [""])[b]
            question_type = batch.get("question_type", [None])[b]

            _n_src = batch.get("n_source_images")
            if _n_src is not None:
                try:
                    n_source_images = int(_n_src[b].item() if hasattr(_n_src[b], "item") else _n_src[b])
                except Exception:
                    n_source_images = 0
            else:
                n_source_images = 0
            results.append({
                "prediction": responses[b],
                "ground_truths": gt_list,
                "scene_id": scene_id,
                "question": question_text,
                "question_type": question_type,
                "n_source_images": n_source_images,
            })

            if verbose:
                idx = len(results)
                print(
                    f"{log_prefix} [{idx}/{len(dataloader)}] "
                    f"Scene: {scene_id} | {elapsed / actual_batch:.2f}s/sample"
                )
                print(f"  Q: {question_text}")
                print(f"  GT: {gt_list[0]} | Pred: {responses[b]}")
                print("-" * 15)

        if max_batches is not None and (i + 1) >= max_batches:
            break

    return results, skipped_scene_ids
