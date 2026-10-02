"""Verify get_rope_index_3 still produces the captured fixture output.

Checks both public import paths — the canonical ``model_adapters.qwen_mrope``
and the ``model_adapters.qwen3_vl`` lazy re-export — and asserts they resolve
to the same function object, so the MRoPE index builder stays bit-identical
across any refactor.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

import torch

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture():
    inputs = torch.load(FIXTURE_DIR / "rope_inputs.pt", weights_only=True)
    outputs = torch.load(FIXTURE_DIR / "rope_outputs.pt", weights_only=True)
    return inputs, outputs


def _check_one(fn, label, inputs, expected):
    pos, deltas = fn(
        spatial_merge_size=inputs["spatial_merge_size"],
        input_ids=inputs["input_ids"],
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs["attention_mask"],
    )
    assert torch.equal(pos, expected["position_ids"]), (
        f"[{label}] position_ids mismatch.\n"
        f"  expected sum: {expected['position_ids'].sum().item()}\n"
        f"  got sum:      {pos.sum().item()}"
    )
    assert torch.equal(deltas, expected["mrope_deltas"]), (
        f"[{label}] mrope_deltas mismatch"
    )
    print(f"  [{label}] OK")


def main():
    inputs, expected = _load_fixture()
    print("Checking get_rope_index_3 against captured fixture...")

    # Canonical path.
    from model_adapters.qwen_mrope import get_rope_index_3 as canonical_fn
    _check_one(canonical_fn, "model_adapters.qwen_mrope", inputs, expected)

    # Public re-export path (lazy __getattr__ in qwen3_vl/__init__.py).
    from model_adapters.qwen3_vl import get_rope_index_3 as reexport_fn
    _check_one(reexport_fn, "model_adapters.qwen3_vl", inputs, expected)

    assert canonical_fn is reexport_fn, (
        "Canonical and re-export paths point to different function objects — "
        "the qwen3_vl re-export must `from ..qwen_mrope import ...`, not redefine."
    )
    print("  [identity] both paths resolve to the same function object")

    print("All rope checks passed.")


if __name__ == "__main__":
    main()
