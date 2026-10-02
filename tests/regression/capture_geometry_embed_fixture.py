"""Capture a fixture of the 3D position embedding's output.

Runs ``_apply_geometry_embeddings`` (the paper's additive 3D position
embedding, ``cartesian_fourier`` mode — the only shipped encoder) on
fixed-seed synthetic inputs and saves the output to
``tests/regression/fixtures/geom_embed_outputs.pt``.

Pairs with ``check_geometry_embed_fixture.py``: capture once on a known-good
commit, then re-run the check after any refactor of the embedding to prove
behaviour is bit-identical. Pure CPU, deterministic.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

import torch
import torch.nn as nn

from model_adapters.qwen3_vl.model import Qwen3VL3DModel


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# Geometry methods stolen onto a thin nn.Module so we exercise the embedding
# without instantiating the multi-billion-parameter language model.
_GEOMETRY_METHODS = [
    "init_depth_embedding",
    "init_depth_embed_learned_scale",
    "_encode_depth_cartesian_fourier",
    "_encode_cartesian_fourier",
    "_depth_ratio_penalty_value",
    "_stash_depth_visual_stats",
    "_add_depth_with_penalty",
    "_apply_geometry_embeddings",
]


def _make_testbed_class(source_class):
    attrs = {name: getattr(source_class, name) for name in _GEOMETRY_METHODS}
    return type("GeomTestbed", (nn.Module,), attrs)


def _seeded_inputs(hidden_size, dtype=torch.float32):
    """Deterministic per-patch inputs: features, raw depths (m), unit rays."""
    torch.manual_seed(123)
    N1, N2 = 50, 80
    image_embeds = torch.randn(N1 + N2, hidden_size, dtype=dtype)
    depths = [
        (torch.rand(N1) * 9.9 + 0.1).to(dtype),
        (torch.rand(N2) * 9.9 + 0.1).to(dtype),
    ]
    ray_dirs = []
    for n in (N1, N2):
        v = torch.randn(n, 3, dtype=dtype)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ray_dirs.append(v)
    return image_embeds, depths, ray_dirs


def run_capture(source_class=Qwen3VL3DModel, *, hidden_size=128, mlp_hidden=64,
                num_freqs=16):
    """Init a cartesian_fourier testbed, run ``_apply_geometry_embeddings`` once."""
    TestbedClass = _make_testbed_class(source_class)
    torch.manual_seed(42)
    tb = TestbedClass()
    tb.eval()  # disable the training-only penalty branch
    tb._probe_depth_scale = 1.0
    tb.init_depth_embedding(
        hidden_size=hidden_size,
        mode="cartesian_fourier",
        num_freqs=num_freqs,
        mlp_hidden=mlp_hidden,
    )
    image_embeds, depths, ray_dirs = _seeded_inputs(hidden_size)
    with torch.no_grad():
        out = tb._apply_geometry_embeddings(
            image_embeds, depths, ray_dirs, device=torch.device("cpu"),
        )
    return out.detach().clone()


def main():
    print("Capturing 3D position embedding fixture (cartesian_fourier)...")
    out = run_capture()
    print(f"  cartesian_fourier: shape={tuple(out.shape)} "
          f"sum={out.sum().item():.6f} abs.mean={out.abs().mean().item():.6f}")
    path = FIXTURE_DIR / "geom_embed_outputs.pt"
    torch.save({"cartesian_fourier": out}, path)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
