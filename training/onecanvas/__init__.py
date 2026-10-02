"""OneCanvas public API.

Convenience re-exports for the two headline use-cases:

  1. Standalone scene reprojection (model-agnostic)::

        from onecanvas import reproject_scene, ReprojectedScene, compute_scene_geometry

  2. The spatial-pretraining curriculum::

        from onecanvas import SpatialPretrainingDataset

Every symbol is imported LAZILY (PEP 562 ``__getattr__``) so ``import onecanvas``
stays cheap and does not require the heavy deps or ``ONECANVAS_DATA_ROOT`` to be
set unless you actually reference a symbol that needs them. In particular,
``SpatialPretrainingDataset`` pulls in the data package, which resolves dataset
roots at import time.
"""
from __future__ import annotations

__all__ = [
    "reproject_scene",
    "compute_scene_geometry",
    "ReprojectedScene",
    "SceneGeometry",
    "process_vision_and_generate",
    "answer_scene_question",
    "get_adapter",
    "VLMAdapter",
    "SpatialPretrainingDataset",
]

# public name -> (module to import, attribute on that module)
_LAZY = {
    "reproject_scene": ("reprojection", "reproject_scene"),
    "compute_scene_geometry": ("reprojection", "compute_scene_geometry"),
    "ReprojectedScene": ("reprojection", "ReprojectedScene"),
    "SceneGeometry": ("reprojection", "SceneGeometry"),
    "process_vision_and_generate": ("inference", "process_vision_and_generate"),
    "answer_scene_question": ("inference", "answer_scene_question"),
    "get_adapter": ("model_adapters", "get_adapter"),
    "VLMAdapter": ("model_adapters", "VLMAdapter"),
    "SpatialPretrainingDataset": (
        "onecanvas.data.spatial_pretraining", "SpatialPretrainingDataset"),
}


def __getattr__(name):
    try:
        mod_path, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'onecanvas' has no attribute {name!r}")
    import importlib
    val = getattr(importlib.import_module(mod_path), attr)
    globals()[name] = val  # cache so __getattr__ only fires once per symbol
    return val


def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
