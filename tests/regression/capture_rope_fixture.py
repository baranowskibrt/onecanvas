"""Capture a fixture of get_rope_index_3 output for regression testing.

Runs the function on a fixed dummy input and saves both the input and the
output to ``tests/regression/fixtures/rope_*.pt``. Used as the "before"
side of the qwen_mrope refactor regression check.

Pure CPU, integer-only ops — no GPU needed and bit-equality is achievable.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

import torch

# Canonical location of the MRoPE index builder.
from model_adapters.qwen_mrope import get_rope_index_3


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


def _dummy_inputs():
    """Construct a dummy input that exercises image_pad splicing.

    Mirrors the shape Qwen3-VL sees during training: one image with grid
    (T=1, H=16, W=16) embedded between text tokens. spatial_merge_size=2
    reduces to (1, 8, 8) = 64 image tokens.
    """
    vision_start_id = 151652
    image_pad_id = 151655
    text_pad_id = 198  # a benign text id, identity doesn't matter for the function
    n_image_tokens = 64

    input_ids = [text_pad_id] * 10 + [vision_start_id] + [image_pad_id] * n_image_tokens + [text_pad_id] * 5
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)
    return input_ids, attention_mask, image_grid_thw


def main():
    input_ids, attention_mask, image_grid_thw = _dummy_inputs()

    position_ids, mrope_deltas = get_rope_index_3(
        spatial_merge_size=2,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )

    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
            "spatial_merge_size": 2,
        },
        FIXTURE_DIR / "rope_inputs.pt",
    )
    torch.save(
        {"position_ids": position_ids, "mrope_deltas": mrope_deltas},
        FIXTURE_DIR / "rope_outputs.pt",
    )

    print(f"Captured rope fixture:")
    print(f"  inputs:  {FIXTURE_DIR / 'rope_inputs.pt'}")
    print(f"  outputs: {FIXTURE_DIR / 'rope_outputs.pt'}")
    print(f"  position_ids shape: {position_ids.shape}, dtype: {position_ids.dtype}")
    print(f"  mrope_deltas shape: {mrope_deltas.shape}")
    print(f"  position_ids sum: {position_ids.sum().item()}")


if __name__ == "__main__":
    main()
