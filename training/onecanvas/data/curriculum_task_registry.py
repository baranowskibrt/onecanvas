"""External probe-task registry — the sanctioned seam for adding probe tasks
without editing onecanvas core.

Motivation
----------
The probe curriculum used to require editing core files for every new task
(a sampler ``elif`` in ``SpatialPretrainingDataset._sample_patches``, a Q&A
``elif`` in ``_build_question_and_answer``, plus a ``CURRICULA`` entry). That
made experiment churn land in release-bound core. This module lets a NEW task
live entirely in an external downstream package and register itself at import
time. The dataset's two dispatch chains
fall through to this registry for any task name they do not recognise.

Inert by default
----------------
Nothing here runs unless either (a) some module calls
``register_probe_task(...)``, or (b) ``ONECANVAS_PROBE_TASK_PLUGINS`` names
plugin modules to import. With an empty registry every public helper returns
empty, so the core fall-through behaves exactly as before (``raise
ValueError("Unknown probe task: ...")``). This keeps existing curricula and
running experiments untouched.

A task spec supplies two callables:

``sample(dataset, scene, rng)``
    Same contract as an internal ``_sample_<task>`` method. ``dataset`` is the
    live ``SpatialPretrainingDataset`` instance (``self``), so the plugin can
    reuse ``dataset._real_asset_bank``, ``dataset.data_args``, and existing
    placement/paste helpers. Returns whatever the internal samplers return
    (patch indices + marker metadata) and stashes any per-sample state on
    ``scene`` for the Q&A callable to read back.

``qa(dataset, scene, task)``
    Returns ``(question_str, answer_str)``, reading the scene attributes the
    sampler stashed.

Registering a task in a curriculum is unchanged: put its name in the
``--curriculum_task_types`` string (and ``--curriculum_canvas_obb_only_tasks`` if needed, or
set ``strip=True`` on the spec and it is unioned in automatically).
"""
from __future__ import annotations

import importlib
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class ProbeTaskSpec:
    """Everything the dataset needs to run an externally-defined probe task.

    name:
        Exact task string, as it appears in ``--curriculum_task_types``.
    sample:
        ``(dataset, scene, rng) -> sampler_return``. See module docstring.
    qa:
        ``(dataset, scene, task) -> (question, answer)``.
    strip:
        If True, the task is unioned into the dataset's strip set so its canvas
        is pruned to placed objects only (same policy as the route_plan_*
        tasks). External tasks default to True to match the locked
        stripped-canvas design.
    requires_real_assets:
        If True, the fall-through raises a clear error when the real-asset bank
        is not enabled (``--real_object_assets_enable True``), mirroring the
        guard on the internal ``*_real`` samplers.
    scene_ids:
        Optional task-specific scene pool. The curriculum dataset selects a
        scene from this pool after it selects the task. Other tasks keep the
        dataset's ordinary scene pool. This is for sparse tasks whose accepted
        scenes are known ahead of time. It avoids repeatedly loading the full
        pool until one eligible scene happens to appear.
    scene_groups:
        Optional source-balanced partition of ``scene_ids``. The dataset first
        samples a group using ``scene_group_weights`` and then samples a scene
        uniformly inside that group. This prevents a larger source inventory
        from silently changing an explicit dataset mixture.
    """

    name: str
    sample: Callable
    qa: Callable
    strip: bool = True
    requires_real_assets: bool = False
    scene_ids: Optional[Tuple[str, ...]] = None
    scene_groups: Optional[Tuple[Tuple[str, ...], ...]] = None
    scene_group_weights: Optional[Tuple[float, ...]] = None


_REGISTRY: dict[str, ProbeTaskSpec] = {}


def register_probe_task(spec: ProbeTaskSpec) -> ProbeTaskSpec:
    """Register (or overwrite) a probe task. Re-import is idempotent."""
    if not isinstance(spec, ProbeTaskSpec):
        raise TypeError(
            "register_probe_task expects a ProbeTaskSpec, got "
            f"{type(spec).__name__}"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get_probe_task(name: str) -> Optional[ProbeTaskSpec]:
    """Return the spec for ``name``, or None if not externally registered."""
    return _REGISTRY.get(name)


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def registered_task_names() -> Tuple[str, ...]:
    return tuple(_REGISTRY)


def registered_strip_tasks() -> Tuple[str, ...]:
    """Names of registered tasks that opt into canvas stripping."""
    return tuple(n for n, s in _REGISTRY.items() if s.strip)


_PLUGINS_LOADED: set[str] = set()


def load_plugins_from_env(var: str = "ONECANVAS_PROBE_TASK_PLUGINS") -> None:
    """Import every module named in ``$ONECANVAS_PROBE_TASK_PLUGINS`` (comma- or
    space-separated) so its ``register_probe_task`` calls run.

    No-op when the env var is unset/empty — this is what keeps the seam inert
    for normal training. Import failures surface (the env var is an explicit
    opt-in), but each module is imported at most once per process.
    """
    raw = os.environ.get(var, "").strip()
    if not raw:
        return
    modules = [m.strip() for m in raw.replace(",", " ").split() if m.strip()]
    for mod in modules:
        if mod in _PLUGINS_LOADED:
            continue
        try:
            importlib.import_module(mod)
            _PLUGINS_LOADED.add(mod)
        except Exception as exc:  # explicit opt-in: make the failure loud
            raise ImportError(
                f"Failed to import probe-task plugin module '{mod}' named in "
                f"${var}. Is the package installed and on PYTHONPATH?"
            ) from exc
    if modules and not _REGISTRY:
        warnings.warn(
            f"${var} listed {modules} but no probe tasks were registered; "
            "did the plugin module call register_probe_task()?",
            RuntimeWarning,
        )
