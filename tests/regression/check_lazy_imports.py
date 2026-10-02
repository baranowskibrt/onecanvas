"""Verify the lazy-import contract — `import model_adapters` must not eagerly
load any model-class module.

A user who only wants one backbone should be able to `import model_adapters`
and then `from model_adapters import Qwen3VL3DModel` without the other adapter's
module (`model_adapters.qwen3_5.model`) ever being imported. Accessing the class
should trigger only its own submodule via the PEP-562 `__getattr__`.
"""
import sys
from pathlib import Path

# The project's runtime convention: training/ must be on sys.path so the
# `onecanvas` package (training/onecanvas/) resolves. train.py and
# run_benchmarks.py do this themselves; tests do it here.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))


def main():
    print("Checking lazy-import contract of model_adapters/__init__.py...")

    assert "model_adapters" not in sys.modules, (
        "model_adapters already imported before this check — re-run in a fresh process"
    )

    import model_adapters  # noqa: F401

    # After bare import, no adapter-specific model module should be loaded.
    for forbidden in ("model_adapters.qwen3_vl.model",
                      "model_adapters.qwen3_5.model"):
        assert forbidden not in sys.modules, (
            f"{forbidden} was loaded eagerly by `import model_adapters` — "
            f"lazy imports are not in place. Currently loaded: "
            f"{[m for m in sys.modules if m.startswith('model_adapters')]}"
        )
    print(f"  bare import loaded only: "
          f"{[m for m in sys.modules if m.startswith('model_adapters')]}")

    # Accessing Qwen3VL3DModel should trigger qwen3_vl.model import (lazy
    # resolution), but NOT the qwen3_5 adapter.
    _ = model_adapters.Qwen3VL3DModel
    assert "model_adapters.qwen3_vl.model" in sys.modules, (
        "Qwen3VL3DModel access did not load qwen3_vl.model — lazy __getattr__ broken"
    )
    assert "model_adapters.qwen3_5.model" not in sys.modules, (
        "Accessing Qwen3VL3DModel triggered qwen3_5.model load — lazy isolation broken"
    )
    print("  Qwen3VL3DModel access loaded qwen3_vl.model only")

    # The model module must not drag in the evaluation stack (utils.metrics
    # imports nltk / pycocoevalcap / rouge_score at module top). It only needs
    # utils.tensor_ops.pad_and_stack, so loading the model must leave both
    # utils.metrics and nltk unimported — otherwise a fresh `pip install` of
    # just the model deps cannot even import a backbone.
    for forbidden in ("utils.metrics", "nltk", "pycocoevalcap", "rouge_score"):
        assert forbidden not in sys.modules, (
            f"{forbidden} was loaded by importing model_adapters.qwen3_vl.model — "
            f"the utils star-import chain has regressed (model classes must not "
            f"pull in the eval stack). Loaded utils modules: "
            f"{[m for m in sys.modules if m.startswith('utils')]}"
        )
    print("  qwen3_vl.model import did not pull in the eval stack (utils.metrics/nltk)")

    print("All lazy-import checks passed.")


if __name__ == "__main__":
    main()
