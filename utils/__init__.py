"""Utility package -- split from the monolithic ``utils.py``.

Backward-compatible re-exports resolve LAZILY (PEP 562 ``__getattr__``): a bare
``from utils import pad_and_stack`` imports only the submodule that defines the
symbol, so pulling in one tensor helper does NOT drag in the evaluation stack
(``nltk`` / ``pycocoevalcap`` / ``rouge_score``, imported at the top of
``utils.metrics``). Prefer importing from the concrete submodule directly
(``from utils.tensor_ops import pad_and_stack``); the lazy fallback exists only
so external code written against the old flat ``utils`` namespace keeps working.
"""
import importlib

# Submodules searched, in order, to resolve a bare ``from utils import X``.
# ``metrics`` is LAST so its heavy eval-only dependencies load only when an
# eval-only symbol (e.g. VSIBENCH_DISPLAY_NAMES) is actually requested.
_SUBMODULES = ("tensor_ops", "bbox", "io", "embedding_io", "eval_loop", "metrics")


def __getattr__(name):
    for _mod in _SUBMODULES:
        m = importlib.import_module(f"{__name__}.{_mod}")
        if hasattr(m, name):
            val = getattr(m, name)
            globals()[name] = val  # cache so __getattr__ fires once per symbol
            return val
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
