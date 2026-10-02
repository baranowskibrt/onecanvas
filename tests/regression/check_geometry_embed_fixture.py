"""Verify the 3D position embedding still produces the captured output.

Re-runs the same seeded inputs through the ``cartesian_fourier`` encoder (the
only shipped mode) and asserts ``torch.equal`` against the captured fixture, so
any refactor of ``GeometryEmbeddingMixin`` is proven bit-identical. Exercises
both Qwen3VL3DModel and Qwen3_5_3DModel to confirm they share one geometry
implementation (the ``Qwen3_5`` copy once silently dropped the cartesian_fourier
dispatch — this guards against that regressing).
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

import torch

from capture_geometry_embed_fixture import run_capture


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _check(label, source_class, expected):
    print(f"Checking {label} against captured fixture...")
    got = run_capture(source_class)
    exp = expected["cartesian_fourier"]
    if not torch.equal(got, exp):
        max_abs = (got - exp).abs().max().item()
        raise AssertionError(
            f"[{label}] tensor mismatch.\n"
            f"  expected sum: {exp.sum().item()}\n"
            f"  got sum:      {got.sum().item()}\n"
            f"  max |delta|:  {max_abs}"
        )
    print(f"  [{label}] OK  (sum={got.sum().item():.6f})")


def main():
    fixture_path = FIXTURE_DIR / "geom_embed_outputs.pt"
    if not fixture_path.exists():
        raise SystemExit(
            f"Fixture missing at {fixture_path}. "
            "Run capture_geometry_embed_fixture.py first."
        )
    expected = torch.load(fixture_path, weights_only=True)

    from model_adapters.qwen3_vl.model import Qwen3VL3DModel
    _check("Qwen3VL3DModel", Qwen3VL3DModel, expected)

    from model_adapters.qwen3_5.model import Qwen3_5_3DModel
    _check("Qwen3_5_3DModel", Qwen3_5_3DModel, expected)


if __name__ == "__main__":
    main()
