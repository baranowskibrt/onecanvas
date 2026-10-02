"""Synthetic spatial-pretraining curriculum (live-features path).

Generates spatially grounded question/answer samples on the fly from real
ScanNet scene geometry, with no human annotation. Each sample loads a scene's
geometry and raw images, runs ``compute_scene_geometry`` (the geometry-only
sibling of ``reproject_scene``) to enumerate the valid 3D patches, places the
task's objects (real-asset pastes or synthetic OBBs) on the panoramic canvas,
builds the question and answer, and returns a sample dict that the model's
``forward()`` encodes and projects live (``use_precomputed_features=False``).

Tasks are organized into six families: metric measurement, egocentric
direction, N-turn navigation, observability (when/whether observed),
counting + arithmetic, and multi-target 3D-box readout. The exact task mix
per curriculum variant is defined in
``curriculum_task_mix.CURRICULA`` (the source of truth), sampled per family in the
``samplers_*.py`` modules, with question/answer construction in ``qa.py`` and
canvas patch layout in ``patches.py``. New or experimental tasks can be added
without editing the core sampler via the plugin seam in ``curriculum_task_registry``
(``register_probe_task`` + ``ONECANVAS_PROBE_TASK_PLUGINS``).

Patch indices picked here index into the same per-frame H_feat x W_feat grid
the live encoder produces, so they line up when the live forward path rebuilds
the scene with the real features.
"""

from .dataset import SpatialPretrainingDataset

__all__ = ["SpatialPretrainingDataset"]
