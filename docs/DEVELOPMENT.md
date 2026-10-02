# Developing OneCanvas

## Package layout

Dependencies point left to right: `geometry/` is the base, `reprojection/` builds
on it, and `model_adapters/`, `inference/`, and `training/` sit on top. `utils/`
and `debug/` are shared helpers used across the stack.

```
onecanvas/
├── geometry/                  # 3D lifting + equirectangular projection math (base layer)
│   ├── geometry_lifting.py    #   unproject 2D pixels + depth -> 3D world coords
│   ├── projection_maps.py     #   equirectangular projection maps
│   └── visibility.py          #   frustum / visibility tests
├── reprojection/              # model-agnostic scene reprojection
│   ├── types.py               #   SceneGeometry + ReprojectedScene dataclasses
│   └── scene_reprojection.py  #   compute_scene_geometry(), reproject_scene()
├── model_adapters/            # per-VLM model classes + adapters (the only VLM-specific code)
│   ├── base.py                #   VLMAdapter interface + get_adapter() dispatch on model_type
│   ├── qwen_mrope.py          #   canonical Qwen-family 3D MRoPE convention (shared)
│   ├── qwen3_vl/              #   Qwen3-VL-8B: model.py, adapter.py, feature_extraction.py, patches.py
│   └── qwen3_5/               #   Qwen3.5: model.py + adapter.py
├── inference/                 # high-level single-scene inference
│   └── pipeline.py            #   process_vision_and_generate()
├── training/                  # training + evaluation
│   ├── run_benchmarks.py      #   batch eval across checkpoints/datasets
│   ├── runs/                  #   paper-recipe stage scripts (stage 1 curriculum, stage 2 QA)
│   └── onecanvas/
│       ├── setup_3d.py        #   shared model-setup helpers
│       ├── train/             #   argument.py (all args) + train.py (entry point)
│       └── data/
│           ├── data_processor_3d.py     # scene dataset: batching, feature load, projection
│           ├── __init__.py              # dataset + path registry
│           ├── curriculum_task_registry.py   # plugin seam for curriculum tasks
│           └── spatial_pretraining/     # synthetic spatial curriculum (sampler package)
│               ├── dataset.py           #   SpatialPretrainingDataset
│               ├── qa.py, patches.py    #   question/answer + patch dispatch
│               └── samplers_*.py        #   per-family sampling (metric/direction/nav/...)
├── utils/                     # shared: metrics, bbox parsing, tensor ops, io, embeddings
├── debug/                     # visualization helpers (panorama overlays, point clouds)
├── scripts/                   # spbench_paper_table.py (SPBench score aggregation, used by run_benchmarks)
└── docs/                      # README figures + bad_scenes.md (DA3 pose-fallback scene audit)
```

## Where to add new code

| I want to add... | Put it in... |
|---|---|
| A new VLM backbone | `model_adapters/<name>/` (`model.py` + `adapter.py`), then follow the backbone checklist below (the factory alone is not enough) |
| New projection / lifting / geometry math | `geometry/` and `reprojection/` |
| A new evaluation metric | `utils/metrics.py` |
| A new benchmark or dataset | register a config dict in `training/onecanvas/data/__init__.py`. For eval, also add it to `ALL_BENCHMARKS` and the per-dataset defaults in `build_data_args()` in `training/run_benchmarks.py` |
| A new curriculum task | prefer the plugin seam (`register_probe_task` + `ONECANVAS_PROBE_TASK_PLUGINS`), or add a sampler in `training/onecanvas/data/spatial_pretraining/samplers_*.py` and register it |
| A reusable single-scene inference helper | `inference/pipeline.py` |
| A shared utility (parsing, tensor op, io) | `utils/` |

Rule of thumb: VLM-specific code lives only under `model_adapters/<vlm>/`. Anything
that produces or consumes a `ReprojectedScene` without knowing the model should stay
in `geometry/` or `reprojection/`. Experimental curriculum tasks go through the plugin
seam rather than editing the core sampler.

### Adding a VLM backbone

`get_adapter()` in `model_adapters/base.py` only routes the data-side adapter
(tokenization, MRoPE scaling, batch prep). The model class itself is selected
separately in each entry point, so a new backbone touches all of these:

1. `model_adapters/<name>/` with `model.py` (the 3D model classes) and
   `adapter.py` (a `build_adapter()` factory). List the package in
   `pyproject.toml`.
2. `get_adapter()` in `model_adapters/base.py`: add a `model_type` branch.
3. `training/onecanvas/train/train.py`: the `_is_qwen3_vl()` path-name check
   gates the model-class imports and the `model_type` fallback. Extend the
   check and both selection sites.
4. `training/run_benchmarks.py`: has its own copy of `_is_qwen3_vl()`, plus
   three gated sites (the `Model3DClass` selection, the live-features module
   selection, and the `model_type` fallback in `build_data_args()`).
5. Generation patches: `apply_patches` lives in
   `model_adapters/qwen3_vl/patches.py` but is backbone-generic (it lets the
   custom `projected_*` kwargs survive `generate()`) and is applied to every
   model. Reuse it unless the new backbone needs its own.
6. `inference/pipeline.py` imports `qwen3_vl` directly and only supports that
   backbone. Extend it or drive the new backbone through
   `training/run_benchmarks.py` instead.

Note the selection heuristic: `_is_qwen3_vl()` substring-matches the
checkpoint path name, and anything that does not match currently falls through
to the Qwen3.5 branch. A new backbone whose path name is not handled will
silently load the wrong model class. `model_adapters/qwen3_5/` was added on
top of `qwen3_vl` by exactly this checklist and is the worked example to diff
against.
