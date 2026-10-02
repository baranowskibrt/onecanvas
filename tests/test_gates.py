"""Pytest wrapper folding the release reproducibility gates into one command.

    pytest tests/               # fast, CPU-only regression checks

The regression checks shell out to their standalone scripts (each also runnable
on its own, e.g. ``python tests/regression/check_rope_fixture.py``) and assert a
clean exit. The lazy-import check in particular must run in a fresh process, so
subprocessing is load-bearing, not just convenient. ``test_obb_surface_points``
is collected directly by pytest from ``tests/test_obb_surface_points.py``.
"""
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REG = _ROOT / "tests" / "regression"


def _run(*args, env=None):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=_ROOT,
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        pytest.fail(
            f"gate failed (exit {r.returncode}): {' '.join(map(str, args))}\n"
            f"--- stdout ---\n{r.stdout[-2000:]}\n--- stderr ---\n{r.stderr[-2000:]}"
        )


def test_lazy_imports():
    """`import model_adapters` must not eagerly load a model-class module."""
    _run(_REG / "check_lazy_imports.py")


def test_rope_fixture():
    """get_rope_index_3 stays bit-identical across both public import paths."""
    _run(_REG / "check_rope_fixture.py")


def test_geometry_embed_fixture():
    """The 3D position embedding (cartesian_fourier) stays bit-identical."""
    _run(_REG / "check_geometry_embed_fixture.py")
