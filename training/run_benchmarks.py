#!/usr/bin/env python3
"""
Run evaluation benchmarks for Qwen3-VL-3D.

Usage examples:
  # Run all benchmarks (except spbench):
  python run_benchmarks.py --lora /path/to/lora_checkpoint/

  # Run a single benchmark:
  python run_benchmarks.py --datasets sqa3d --lora /path/to/checkpoint

  # Run specific benchmarks:
  python run_benchmarks.py --datasets sqa3d vsi_bench --lora /path/to/checkpoint

  # Base model (no LoRA):
  python run_benchmarks.py --datasets sqa3d --no-lora

  # Override output directory:
  python run_benchmarks.py --exp-name output/my_experiment --lora /path/to/checkpoint
"""

import argparse
import gc
import importlib
import os
import sys
import time
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, HfArgumentParser

# Ensure project root is on path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from peft import PeftModel
from onecanvas.data.data_processor_3d import make_supervised_data_module
from onecanvas.setup_3d import (configure_processor, get_inner_3d_model,
                                init_3d_embeddings)
from onecanvas.train.argument import DataArguments, ModelArguments, TrainingArguments
import utils
from utils.embedding_io import load_3d_embeddings
from model_adapters.qwen3_vl.patches import apply_patches
from inference import _compute_da3_geometry

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("ONECANVAS_FULLSPAN_SAMPLER", "1")


def _is_qwen3_vl(model_path: str) -> bool:
    lower = model_path.lower()
    return "qwen3-vl" in lower or "qwen3_vl" in lower

# ── All supported benchmarks (order matters for default runs) ────────────
ALL_BENCHMARKS = ["sqa3d", "vsi_bench"]
# Additional benchmarks can be added via --datasets: spbench, multi3drefer

# ── Dataset → MetricTracker name mapping (mirrors train.py) ─────
_TRACKER_NAME_MAP = {
    "vlm3r_vsibench": "vsi_bench",
    "vsibench": "vsi_bench",
    "vsi_bench": "vsi_bench",
    "sqa3d": "sqa3d",
    "spbench": "sp_bench",
    "multi3drefer": "multi3drefer",
}


def tracker_name_for(dataset_name: str) -> str | None:
    return _TRACKER_NAME_MAP.get(dataset_name.lower(), dataset_name.lower())


def parse_args():
    p = argparse.ArgumentParser(description="Run 3D-VL evaluation benchmarks", allow_abbrev=False)

    # Auto-config from checkpoint
    p.add_argument("--from-config", default=None, metavar="CHECKPOINT_DIR",
                    help="Load all settings from resolved_config.json in this checkpoint dir. "
                         "Also sets --lora to best_checkpoint/ if available. "
                         "CLI flags still override config values.")

    # Model
    p.add_argument("--model-path", default=None,
                    help="Base model HF id or path (default: from config or Qwen/Qwen3-VL-8B-Instruct)")
    p.add_argument("--lora", default=None,
                    help="Path to LoRA checkpoint (omit or use --no-lora for base)")
    p.add_argument("--no-lora", action="store_true",
                    help="Run base model without LoRA")
    p.add_argument("--stage1-lora", default=None,
                    help="Optional stage-1 LoRA to merge into base before applying --lora. "
                         "Reproduces training-time lora_checkpoint_merge=True. Auto-pulled "
                         "from resolved_config.json (training.lora_checkpoint_path) when "
                         "lora_checkpoint_merge is True and --from-config is used.")
    p.add_argument("--vanilla-qwen3vl", action="store_true", default=False,
                    help="Vanilla baseline: load stock Qwen3VLForConditionalGeneration "
                         "and feed raw multi-image inputs directly. Skips all 3D machinery.")
    p.add_argument("--attn-implementation", default="flash_attention_2",
                    choices=["sdpa", "flash_attention_2", "eager"],
                    help="HF attention backend.")

    # Datasets
    p.add_argument("--datasets", nargs="+", default=None,
                    help=f"Benchmarks to run. Default: {ALL_BENCHMARKS}")

    # 3D config
    p.add_argument("--rope-pos-range", type=float, default=0.0)

    # Data-processing flags that must match training config
    p.add_argument("--temporal-max-range", type=float, default=100.0)
    p.add_argument("--temporal-raw-frame-index",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Use source frame ordinal directly for MRoPE T. Default: the "
                        "checkpoint's recorded value, or off (normalized T, the "
                        "convention of checkpoints that record none).")
    p.add_argument("--use-depth-embedding", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth-embed-num-freqs", type=int, default=16)
    p.add_argument("--depth-embed-min", type=float, default=0.3)
    p.add_argument("--feature-set", default="balanced",
                    choices=["balanced", "aggregated"])
    p.add_argument("--metric-json-grounding-format", action="store_true", default=False,
                    help="GT bbox format: '[{\"bbox_3d\": [cx,cy,cz,sx,sy,sz], \"label\": ...}]'. "
                         "Must match training config for the eval checkpoint.")
    p.add_argument("--pano-grounding-format", action="store_true", default=False,
                    help="GT bbox format: pano-angular (u,v,depth,sx,sy,sz). "
                         "Mutually exclusive with --metric-json-grounding-format.")
    p.add_argument("--stratified-eval", action="store_true", default=False,
                    help="Use stratified sampling (matches gen_eval during training)")
    p.add_argument("--sqa3d-use-agent-pose", action=argparse.BooleanOptionalAction, default=True,
                    help="SQA3D: center panorama at the situated agent position and "
                         "rotate forward by the agent's yaw. Default ON (SQA3D questions "
                         "are phrased from the agent's viewpoint). Pass --no-sqa3d-use-agent-pose "
                         "for scene-center. Not overridden by --from-config.")
    p.add_argument("--sqa3d-agent-yaw-offset", type=float, default=0.0,
                    help="Extra yaw offset (radians) on top of the SQA3D agent rotation. "
                         "Use multiples of pi/2 to compensate for axis-convention mismatch.")
    p.add_argument("--sqa3d-canvas-center-mode", default="auto",
                    choices=["auto", "agent_pose", "scene_center", "random_camera",
                             "outside_bbox"],
                    help="Inference-time SQA3D canvas-origin ablation. "
                         "'auto' (default) defers to --sqa3d-use-agent-pose. "
                         "Explicit modes override: 'agent_pose' (situated origin+yaw), "
                         "'scene_center' (no override), "
                         "'random_camera' (one camera translation per sample, deterministic), "
                         "'outside_bbox' (scene XY-center + 1.5R along +X, no yaw). "
                         "Not overridden by --from-config.")
    p.add_argument("--spbench-use-camera-pose", action=argparse.BooleanOptionalAction, default=True,
                    help="SPBench-SI: center panorama at the single pinned camera's "
                         "position and rotate forward along its +Z axis. Default ON "
                         "(SPBench-SI questions are phrased 'from the camera's perspective'). "
                         "No-op for SPBench-MV (8 pinned frames) — gated on len(images)==1. "
                         "Not overridden by --from-config.")
    p.add_argument("--spbench-camera-yaw-offset", type=float, default=0.0,
                    help="Extra yaw offset (radians) on top of the SPBench pinned-camera "
                         "rotation. Use to compensate for axis-convention mismatch.")

    # Data
    p.add_argument("--num-images", type=int, default=32)
    p.add_argument("--image-resolution", default=None,
                    help="Resolution tag for resized images. Default: 640x480 for "
                         "vsi_bench/spbench, 320x240 for sqa3d.")
    p.add_argument("--limit", type=int, default=None,
                    help="Max samples per dataset (default: full test set)")
    p.add_argument("--precomputed-geometry", action="store_true", default=True)
    p.add_argument("--no-precomputed-geometry", dest="precomputed_geometry",
                    action="store_false")
    p.add_argument("--use-gt-all", action=argparse.BooleanOptionalAction, default=True,
                    help="Use GT poses/depth from per-scene files. "
                         "Pass --no-use-gt-all to use DA3-predicted poses from the .pt. "
                         "Loaded from --from-config unless explicitly passed.")
    p.add_argument("--scenes-file", default=None, dest="scene_filter_file",
                    help="File of scene ids (one per line); evaluate ONLY those scenes. "
                         "For re-measuring a fix on the subset it provably touches.")
    p.add_argument("--scannetpp-pose-frame", choices=["mesh", "arkit"], default=None,
                    help="World frame of ScanNet++ poses. Default: the checkpoint's recorded "
                         "value. The released model records 'arkit', the frame it trained on.")
    p.add_argument("--upright-arkit", action=argparse.BooleanOptionalAction, default=True,
                    help="Gravity-upright ARKitScenes frames before the vision tower. "
                         "ARKit stores every frame in the sensor's landscape buffer "
                         "regardless of phone roll, so gravity points sideways on 84 of "
                         "the 150 VSI ARKit scenes and the tower sees rooms on their "
                         "side. Rotates image, depth, intrinsics and pose together, so "
                         "the reconstructed 3D is unchanged and only the appearance "
                         "moves. No-op on ScanNet / ScanNet++. Default on.")
    p.add_argument("--force-turns", type=int, default=0, dest="upright_force_turns",
                    help="Causal check for --upright-arkit: force this many clockwise "
                         "quarter turns on every ARKit scene instead of the gravity-"
                         "derived one. The already-upright scenes should degrade.")
    p.add_argument("--predicted-geometry", default=None,
                    choices=["da3_eval32", "mapanything_eval32", "da3_eval32ctx256", "dvlt_eval32",
                             "da3_hires", "dvlt_hires", "da3_hires_scalecal", "da3_hires_scaleoracle",
                             "da3_upright32", "da3_hires32", "dvlt_ctx256"],
                    help="use a predictor's eval32 geometry (poses + "
                         "intrinsics + depth predicted from RGB on the exact GT-eval "
                         "frames) instead of GT/balanced_256. Pair with --no-use-gt-all. "
                         "da3_eval32ctx256 = same eval32 frames but predicted with "
                         "~256-frame context (falls back to plain eval32 per scene, "
                         "e.g. scannet which has no ctx256 stage). "
                         "da3_hires / dvlt_hires = the 2026-07-24 regeneration: full-res "
                         "input frames (ARKit vga_wide_640x480, matching what the eval "
                         "reads), ~286-frame context, and ARKit gravity-uprighted. da3_hires_scalecal = da3_hires plus a single per-sensor metric-scale constant (0.9669, median GT/DA3 depth over 876 non-eval ScanNet scenes) applied to ScanNet depth+pose translations; ARKit/ScanNet++ files are unchanged (their bias is <1%).")
    p.add_argument("--depth-downsample", type=int, default=1,
                    help="Coarsen the depth grid to (H_feat/K, W_feat/K) before the "
                         "per-patch depth lookup, then replicate back (eval-only input "
                         "perturbation). Every KxK block of canvas patches then shares "
                         "one depth sample. The patch count and the features are "
                         "unchanged, so this isolates 3D placement precision from "
                         "canvas token density (which is what --image-resolution moves). "
                         "Default 1 = off.")
    p.add_argument("--question-types", default=None,
                    help="Comma-separated question_type allowlist, applied when "
                         "annotations load (e.g. object_counting). Filtering happens "
                         "before the dataset is built, so DDP sharding and len(dataset) "
                         "stay correct. Default None = all question types.")
    p.add_argument("--canvas-yaw-offset", type=float, default=0.0,
                    help="Global canvas yaw offset in RADIANS (seam probe). "
                         "Rigidly rotates the whole canvas on top of the "
                         "dataset's own anchor yaw, so every ground-truth answer "
                         "stays correct and the ONLY thing that moves is where the "
                         "equirectangular longitude wrap falls relative to the scene. "
                         "Sweep it across runs and compare per-question accuracy to "
                         "measure the seam's cost on real data. Default 0.0 is a no-op "
                         "and reproduces the shipped eval exactly.")
    p.add_argument("--use-resized-images", action="store_true", default=True)
    p.add_argument("--no-resized-images", dest="use_resized_images",
                    action="store_false")

    # Eval
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--exp-name", default=None,
                    help="Output directory base. Default: output/<lora_dirname>")
    p.add_argument("--debug", action="store_true",
                    help="Stop after 1 sample per dataset")
    p.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader workers. Default: min(8, available cores). "
                         "One worker per core OOMs on high-core cluster nodes.")
    p.add_argument("--compute-val-loss", action="store_true",
                    help="Also compute teacher-forced validation loss")

    args = p.parse_args()

    # Downgrade flash_attention_2 -> sdpa when flash_attn isn't installed
    # (flash-attn is the default but not a declared dep).
    from model_adapters.attention import resolve_attn_implementation
    args.attn_implementation = resolve_attn_implementation(args.attn_implementation)

    # Auto-discover resolved_config.json next to (or one level up from) --lora.
    # Self-describing checkpoints win out over the CLI defaults; pass
    # --from-config explicitly to override the search path.
    if args.from_config is None and args.lora is not None:
        from pathlib import Path as _Path
        for _cand in (_Path(args.lora), _Path(args.lora).parent):
            if (_cand / "resolved_config.json").exists():
                args.from_config = str(_cand)
                print(f"[auto] found resolved_config.json in {_cand}; applying. "
                      f"Pass --from-config to override.")
                break

    # Self-contained merged checkpoint (the released form): full model weights
    # plus resolved_config.json and depth_embedding.pt in one dir, no adapter.
    # Detect it on --model-path so `--model-path <dir> --no-lora` picks up the
    # eval-time config without any LoRA plumbing.
    if args.from_config is None and args.lora is None and args.model_path:
        from pathlib import Path as _Path
        _mp = _Path(args.model_path)
        if (_mp.is_dir() and (_mp / "resolved_config.json").exists()
                and not (_mp / "adapter_config.json").exists()):
            args.from_config = str(_mp)
            args.no_lora = True
            print(f"[auto] {_mp} is a self-contained merged checkpoint; "
                  f"applying its resolved_config.json and skipping LoRA.")

    if args.from_config:
        _apply_from_config(args, p)
        args.attn_implementation = resolve_attn_implementation(args.attn_implementation)

    # Default model path if not set
    if args.model_path is None:
        args.model_path = "Qwen/Qwen3-VL-8B-Instruct"

    return args


def _apply_from_config(args, parser):
    """Load resolved_config.json and apply training-side settings to ``args``.

    Reads the typed-JSON format written by train.py — bools/ints/floats/null
    come through with their native Python types and are forwarded to argparse
    Namespace fields whose names match a ``DataArguments`` field. Only
    overrides values that the user did NOT explicitly pass on the CLI.

    Adding a new field to ``DataArguments`` requires zero changes here, as
    long as the corresponding CLI flag (or argparse field) shares the same
    name on ``args``.
    """
    import json
    import dataclasses

    ckpt_dir = Path(args.from_config)
    config_path = ckpt_dir / "resolved_config.json"
    if not config_path.exists():
        parser.error(f"resolved_config.json not found at {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    data = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    # Track which CLI flags the user explicitly passed (do not override those).
    _explicit = set()
    for action in parser._actions:
        # "--flag value" and "--flag=value" both count as set by the user.
        if action.option_strings and any(a == opt or a.startswith(opt + "=")
                                         for a in sys.argv for opt in action.option_strings):
            _explicit.add(action.dest)

    def _set(dest, value):
        if dest not in _explicit:
            setattr(args, dest, value)

    # Generic loop: any DataArguments / ModelArguments field with a matching
    # args attr gets the typed value from the config. Reads typed JSON only —
    # no string coercion — so legacy `str(v)` configs (pre-refactor) will not
    # work.
    # Eval-time overrides: these flags control pose-conditioned centering for
    # benchmarks whose questions are phrased from a specific reference frame
    # (SQA3D: situated agent; SPBench-SI: pinned camera). Scene-center gives
    # the wrong reference frame for both, so we keep --from-config from pulling
    # the training-side defaults (which are scene-center for historical reasons).
    # stratified_eval belongs to the training loop (gen_eval), not production eval;
    # loading it from config would force val split and override the test-split default.
    # dataset_use / val_sample_num / dataset_offset are training-loop concerns;
    # eval iterates per-benchmark via args.datasets and per-call --limit, so
    # never pull them from the training config.
    # scene_filter_file is a TRAINING scene allowlist. Inherited here it
    # restricts the benchmark to the rooms the checkpoint trained on, which is
    # the one thing a benchmark must not do, and it does it silently: a
    # tool-loop checkpoint carrying a 1401-room allowlist made VSI-Bench load
    # 1986 of its 5130 items, every one on a trained room, and the run then
    # reported an ordinary-looking score over them (jobs 2929352, 2929517).
    # Pass --scenes-file explicitly to evaluate a chosen subset.
    # upright_arkit is an evaluation input convention, not part of the model.
    # Training configs record False (the training data kept the sensor
    # orientation), and inheriting that silently evaluated VSI-Bench on sideways
    # ARKitScenes frames. The runner's own default (on) applies unless
    # --no-upright-arkit is passed.
    _skip_from_config = {
        "sqa3d_use_agent_pose", "sqa3d_canvas_center_mode", "spbench_use_camera_pose",
        "stratified_eval", "image_resolution",
        "dataset_use", "val_sample_num", "dataset_offset",
        "scene_filter_file", "upright_arkit", "upright_force_turns",
        "panoramic_eval_yaw_offset", "with_precomputed_geometry",
    }

    _data_field_names = {f.name for f in dataclasses.fields(DataArguments)}
    for fname in _data_field_names:
        if fname in _skip_from_config:
            continue
        if fname in data:
            _set(fname, data[fname])
    _model_field_names = {f.name for f in dataclasses.fields(ModelArguments)}
    for fname in _model_field_names:
        if fname in model_cfg:
            _set(fname, model_cfg[fname])

    # Special cases not driven by the loop:
    _set("model_path", model_cfg.get("model_name_or_path", "Qwen/Qwen3-VL-8B-Instruct"))
    if "lora" not in _explicit and not args.no_lora:
        best = ckpt_dir / "best_checkpoint"
        args.lora = str(best if best.exists() else ckpt_dir)
    training_cfg = cfg.get("training", {})
    if (
        "stage1_lora" not in _explicit
        and training_cfg.get("lora_checkpoint_merge")
        and training_cfg.get("lora_checkpoint_path")
    ):
        # Prefer the in-checkpoint snapshots when present — immune to upstream
        # stage-1 training rotating the original path out via save_total_limit.
        # lora_checkpoint_path may be a comma-separated chain (train.py merges
        # each link in order and snapshots them as stage1_lora, merge_lora_1,
        # merge_lora_2, ...). Reconstruct the FULL chain here: dropping a link
        # silently evaluates a model missing a whole training stage.
        _chain = [p.strip() for p in str(training_cfg["lora_checkpoint_path"]).split(",") if p.strip()]
        _resolved = []
        for _mi, _mpath in enumerate(_chain):
            _snap = ckpt_dir / ("stage1_lora" if _mi == 0 else f"merge_lora_{_mi}")
            _resolved.append(str(_snap) if _snap.exists() else _mpath)
        args.stage1_lora = ",".join(_resolved)
    # Field-name asymmetry between CLI and DataArguments:
    if "with_precomputed_geometry" in data:
        _set("precomputed_geometry", data["with_precomputed_geometry"])

    print(f"[from-config] Loaded settings from {config_path}")
    print(f"  model: {args.model_path}")
    print(f"  lora: {args.lora}")
    if args.stage1_lora:
        print(f"  stage1_lora (merge before lora): {args.stage1_lora}")
    print(f"  depth_embedding: {args.use_depth_embedding} (cartesian_fourier)")
    print(f"  image_resolution: {args.image_resolution}")


def load_model_and_processor(args):
    """Load base model, optionally apply LoRA, return (model, processor)."""
    print(f"Loading base model: {args.model_path}")

    if getattr(args, "vanilla_qwen3vl", False):
        from transformers import Qwen3VLForConditionalGeneration

        # Mirrors the wrapper in train.py: collator stacks pixel_values to
        # [1, P, dim] and image_grid_thw to [1, N, 3] for bs=1; stock forward
        # wants the unbatched form.
        class _VanillaQwen3VL(Qwen3VLForConditionalGeneration):
            def forward(self, pixel_values=None, image_grid_thw=None, **kw):
                if pixel_values is not None and pixel_values.dim() == 3:
                    pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
                if image_grid_thw is not None and image_grid_thw.dim() == 3:
                    image_grid_thw = image_grid_thw.reshape(-1, 3)
                return super().forward(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **kw,
                )

        print(f"  attn_implementation = {args.attn_implementation} (VANILLA BASELINE)")
        base_model = _VanillaQwen3VL.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=args.attn_implementation,
        )

        lora_path = None if args.no_lora else args.lora
        if lora_path is not None:
            print(f"Loading LoRA weights: {lora_path}")
            model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.bfloat16)
        else:
            print("Running base model (no LoRA)")
            model = base_model

        processor = AutoProcessor.from_pretrained(args.model_path)
        # No configure_processor: vanilla wants Qwen3-VL native processor sizes.
        model.processor = processor

        model.eval()
        model.config.use_cache = True
        return model, processor

    if _is_qwen3_vl(args.model_path):
        from model_adapters.qwen3_vl.model import Qwen3VL3DForConditionalGeneration as Model3DClass
    else:
        from model_adapters.qwen3_5.model import Qwen3_5_3DForConditionalGeneration as Model3DClass
    print(f"  attn_implementation = {args.attn_implementation}")
    # Fix safetensors/PEFT multi-rank load bug: safe_load_file(device="cuda")
    # hardcodes to cuda:0 instead of honoring torch.cuda.current_device(). On
    # non-rank-0 ranks the model lives on cuda:N (N>0) and the cuda:0 state
    # dict fails to apply, leaving LoRA B at zero-init. Monkey-patch PEFT's
    # infer_device() to return the indexed device for this process.
    import peft.utils.other as _peft_other
    import peft.peft_model as _peft_pm
    import peft.utils.save_and_load as _peft_sl
    def _infer_device_indexed():
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
    _peft_other.infer_device = _infer_device_indexed
    _peft_pm.infer_device = _infer_device_indexed
    _peft_sl.infer_device = _infer_device_indexed
    _per_rank_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cuda"
    print(f"[multi-rank-fix] infer_device patched to '{_per_rank_device}' (current_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'})")

    base_model = Model3DClass.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=_per_rank_device,
        attn_implementation=args.attn_implementation,
    )

    if getattr(args, "stage1_lora", None):
        # Comma-separated chain: merge each adapter in order, mirroring
        # train.py's lora_checkpoint_merge loop.
        _stage_paths = [p.strip() for p in str(args.stage1_lora).split(",") if p.strip()]
        for _si, _spath in enumerate(_stage_paths):
            print(f"[stage1-merge] Loading pre-merge LoRA {_si + 1}/{len(_stage_paths)}: {_spath}")
            stage1 = PeftModel.from_pretrained(base_model, _spath, torch_dtype=torch.bfloat16)
            base_model = stage1.merge_and_unload()
        print(f"[stage1-merge] Merged {len(_stage_paths)} adapter(s) into base; applying final LoRA on top")

    lora_path = None if args.no_lora else args.lora
    if lora_path is not None:
        if os.path.isdir(lora_path) and not os.path.exists(
                os.path.join(lora_path, "adapter_config.json")):
            raise SystemExit(
                f"--lora {lora_path} has no adapter_config.json. If this is a "
                f"merged self-contained checkpoint, pass it as "
                f"--model-path {lora_path} instead (without --lora).")
        print(f"Loading LoRA weights: {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.bfloat16)
        # Do NOT merge — keeping PeftModel matches training gen_eval behavior exactly.
        # merge_and_unload() introduces bfloat16 rounding that causes greedy-decode divergence.
        print("[LoRA] Loaded as PeftModel (no merge) -- matches training gen_eval behavior")
    else:
        print("Running base model (no LoRA)")
        model = base_model

    processor = AutoProcessor.from_pretrained(args.model_path)
    configure_processor(processor)
    model.processor = processor
    if hasattr(model, "model"):
        model.model.processor = processor

    model.eval()
    model.config.use_cache = True
    return model, processor


def load_da3_model():
    """Load DepthAnything3 (only needed when not using precomputed geometry)."""
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as e:
        raise ImportError(
            "depth_anything_3 is required for the pose/depth fallback (used when "
            "geometry is not precomputed, i.e. --no-use-gt-all without cached "
            ".pt features) but is not installed. Install Depth Anything 3 (the "
            "'depth_anything_3' package from its official release; see README > "
            "Install) or run with precomputed features instead."
        ) from e
    device = torch.device("cuda")
    # The metric model every precomputed *_metric.pt comes from. The non-nested
    # DA3-Large is not metric scaled (depth about 2.8x off on scene0675_01).
    da3_model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    return da3_model.to(device=device)


def _get_vl_model(model):
    """Return the Qwen3VL3DForConditionalGeneration regardless of PeftModel wrapping."""
    if isinstance(model, PeftModel):
        return model.base_model.model
    return model


def configure_model_3d(model, args, lora_path):
    """Set 3D projection attributes on the underlying model, init+load
    embeddings, install generation patches, and rebind 3D forward methods.

    The shared model-setup logic (embedding init, embedding load,
    generation-time patches) lives in ``onecanvas.setup_3d`` /
    ``utils.embedding_io`` / ``model_adapters.qwen3_vl.patches`` so that
    the same code path runs in training and benchmarks.
    """
    inner = get_inner_3d_model(model)
    vl = _get_vl_model(model)

    inner.debug = args.debug

    init_3d_embeddings(model, args)
    loaded = load_3d_embeddings(model, lora_path) if lora_path else {}
    if getattr(args, "use_depth_embedding", False) and "depth" not in loaded:
        raise SystemExit(
            "The 3D position embedding is enabled but no depth_embedding.pt was loaded "
            f"(looked in {lora_path!r}). Evaluating would use a randomly initialized "
            "geometry channel. Pass a local checkpoint directory as --model-path (merged "
            "model) or --lora (adapter), or --no-use-depth-embedding for a model without one.")
    apply_patches(model)

    # Rebind 3D forward methods on the underlying VL model (not the PEFT wrapper).
    if _is_qwen3_vl(args.model_path):
        import model_adapters.qwen3_vl.model as _mod
        inner.forward = types.MethodType(_mod.Qwen3VL3DModel.forward, inner)
        vl.forward = types.MethodType(_mod.Qwen3VL3DForConditionalGeneration.forward, vl)
    else:
        import model_adapters.qwen3_5.model as _mod
        inner.forward = types.MethodType(_mod.Qwen3_5_3DModel.forward, inner)
        vl.forward = types.MethodType(_mod.Qwen3_5_3DForConditionalGeneration.forward, vl)
    return model


def build_data_args(args, dataset_name):
    """Build DataArguments for a given dataset."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # --bf16 is only here to satisfy TrainingArguments; nothing downstream reads
    # it off this parse. On a CPU-only host its validator hard-errors, which
    # would block dataset-level tools that legitimately need no GPU (geometry
    # probes, annotation audits), so pair it with --use_cpu there.
    _cfg_args = [
        "--model_name_or_path", args.model_path,
        "--dataset_use", dataset_name,
        "--bf16", "True",
    ]
    if not torch.cuda.is_available():
        _cfg_args += ["--use_cpu", "True"]
    _, data_args, _ = parser.parse_args_into_dataclasses(args=_cfg_args)

    # Auto-infer model_type for adapter routing
    if not _is_qwen3_vl(args.model_path) and data_args.model_type == "qwen3vl_3d":
        data_args.model_type = "qwen3_5_3d"

    data_args.val_sample_num = args.limit
    data_args.num_images = args.num_images
    data_args.with_precomputed_geometry = args.precomputed_geometry
    data_args.dataset_offset = 0
    data_args.use_resized_images = args.use_resized_images
    # Per-dataset default resolution: 640x480 for vsi_bench/spbench, 320x240 for sqa3d
    _res = getattr(args, "image_resolution", None)
    if _res is None:
        _res = "320x240" if dataset_name in ("sqa3d", "sqa3d_agent_pose") else "640x480"
    data_args.image_resolution = _res
    data_args.stratified_eval = getattr(args, "stratified_eval", False)
    data_args.sqa3d_use_agent_pose = getattr(args, "sqa3d_use_agent_pose", False)
    data_args.sqa3d_agent_yaw_offset = getattr(args, "sqa3d_agent_yaw_offset", 0.0)
    data_args.sqa3d_canvas_center_mode = getattr(args, "sqa3d_canvas_center_mode", "auto")
    data_args.spbench_use_camera_pose = getattr(args, "spbench_use_camera_pose", False)
    data_args.spbench_camera_yaw_offset = getattr(args, "spbench_camera_yaw_offset", 0.0)

    data_args.rope_pos_range = getattr(args, "rope_pos_range", 0.0)
    data_args.temporal_max_range = getattr(args, "temporal_max_range", 100.0)
    # Set by --from-config off the checkpoint's resolved_config.json, so a
    # raw-T checkpoint is evaluated with raw T. No CLI flag on purpose: it is
    # a property of the trained weights, not an eval-time choice.
    data_args.temporal_raw_frame_index = bool(getattr(args, "temporal_raw_frame_index", None) or False)
    data_args.use_depth_embedding = getattr(args, "use_depth_embedding", True)
    data_args.depth_embed_num_freqs = getattr(args, "depth_embed_num_freqs", 16)
    data_args.depth_embed_min = getattr(args, "depth_embed_min", 0.3)
    data_args.feature_set = getattr(args, "feature_set", "balanced")
    data_args.metric_json_grounding_format = getattr(args, "metric_json_grounding_format", True)
    data_args.pano_grounding_format = getattr(args, "pano_grounding_format", False)
    data_args.use_gt_all = getattr(args, "use_gt_all", True)
    data_args.upright_arkit = bool(getattr(args, "upright_arkit", True))
    data_args.scene_filter_file = getattr(args, "scene_filter_file", None)
    data_args.upright_force_turns = int(getattr(args, "upright_force_turns", 0) or 0)
    data_args.predicted_geometry = getattr(args, "predicted_geometry", None)
    data_args.depth_downsample = int(getattr(args, "depth_downsample", 1) or 1)
    data_args.question_type_filter = getattr(args, "question_types", None)
    # Seam probe. Read from the CLI rather than the run config so
    # a sweep never rewrites resolved_config.json; 0.0 reproduces the shipped
    # eval exactly.
    data_args.panoramic_eval_yaw_offset = float(
        getattr(args, "canvas_yaw_offset", 0.0) or 0.0)

    # Vanilla baseline: short-circuit dataset to standard multi-image inputs.
    data_args.vanilla_qwen3vl = getattr(args, "vanilla_qwen3vl", False)

    # Catch-all: any DataArguments field whose value was loaded onto `args` by
    # _apply_from_config (from resolved_config.json) but isn't explicitly
    # propagated above gets copied through here. Lets new training-side fields
    # flow into eval without per-field plumbing. Skips None to preserve
    # dataclass defaults when argparse defaults to None.
    import dataclasses as _dc
    for _fname in (_f.name for _f in _dc.fields(DataArguments)):
        if hasattr(args, _fname):
            _val = getattr(args, _fname)
            if _val is not None:
                setattr(data_args, _fname, _val)

    return data_args


# Question counts of the complete test splits the published numbers are computed on.
FULL_TEST_QUESTIONS = {"sqa3d": 3519, "vsi_bench": 5130, "spbench": 1328}


def check_full_test_split(dataset_name, dataset, args):
    """Refuse a silently partial benchmark. Scenes whose files do not resolve are
    dropped while the dataset is built, and the score would then cover fewer
    questions than the published one. Runs that ask for a subset are exempt."""
    expected = FULL_TEST_QUESTIONS.get(str(dataset_name))
    subset = (getattr(args, "scene_filter_file", None) or getattr(args, "question_types", None)
              or getattr(args, "stratified_eval", False))
    if expected is None or subset or len(dataset) == expected:
        return
    raise SystemExit(
        f"{dataset_name}: the test split has {len(dataset)} questions, the complete split has "
        f"{expected}. Some scenes did not load (see the skip report above), so the score would "
        "not be comparable. Complete the data preparation in docs/DATA.md, or pass "
        "--scenes-file to evaluate a subset on purpose.")


def build_dataloaders(processor, data_args, args):
    """Build generation (and optionally loss) dataloaders."""
    data_args.with_answer = False
    if getattr(args, "stratified_eval", False):
        # Use val split with stratified sampling to reproduce gen_eval exactly.
        data_module = make_supervised_data_module(
            processor, data_args=data_args,
            build_train_dataset=False, build_eval_dataset=True,
        )
        dataset = data_module["eval_dataset"]
    else:
        # with_test=True means test_dataset is what gets evaluated, and neither
        # the train nor the val dataset is read past this line. Building them
        # anyway required the TRAIN and VAL annotations to be present AND
        # non-empty, so an eval failed for reasons entirely outside the split
        # it was evaluating: downloading only a benchmark's test split died on
        # FileNotFoundError for the train json, and --scenes-file naming
        # test-split scenes (what the flag is for) died on "Dataset is empty"
        # first for train, then for val. It also cost two full scene-index
        # builds on every eval. The `or` below cannot reach eval_dataset here,
        # since an empty test_dataset raises inside SceneQADataset first.
        data_module = make_supervised_data_module(
            processor, data_args=data_args, with_test=True,
            build_train_dataset=False, build_eval_dataset=False,
        )
        dataset = data_module.get("test_dataset") or data_module.get("eval_dataset")
        check_full_test_split(data_args.dataset_use, dataset, args)
        # Apply --limit (the data_processor cap only applies to val, so we truncate here for test)
        if args.limit is not None and len(dataset) > args.limit:
            dataset = torch.utils.data.Subset(dataset, range(args.limit))

    debug_mode = args.debug
    if debug_mode:
        num_workers = 0
    elif getattr(args, "num_workers", None) is not None:
        num_workers = args.num_workers
    else:
        # One worker per core OOMs a laptop / high-core node; cap at 8.
        num_workers = min(8, len(os.sched_getaffinity(0)))
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=False,
        **({"prefetch_factor": 2} if num_workers > 0 else {}),
    )
    dataloader = DataLoader(dataset, collate_fn=data_module["data_collator"], **loader_kwargs)

    loss_dataloader = None
    if args.compute_val_loss:
        data_args.with_answer = True
        loss_module = make_supervised_data_module(processor, data_args=data_args)
        loss_dataloader = DataLoader(
            loss_module.get("eval_dataset"),
            collate_fn=loss_module["data_collator"],
            **loader_kwargs,
        )
        data_args.with_answer = False

    return dataset, dataloader, loss_dataloader


def _attach_reprojection_config(model, dataset):
    """Wire reprojection_config onto the inner 3D model for live-features inference.

    Supports both Qwen3VL3DModel and Qwen3_5_3DModel — either class implements
    the PRE-PATH-A branch in forward() that consumes reprojection_config.
    """
    from model_adapters.qwen3_vl.model import Qwen3VL3DModel as _Qwen3VL3DModel
    # Qwen3.5 is the SECOND backbone, and this function needs its class only to
    # widen an isinstance tuple. Every other qwen3_5 import in this file sits
    # inside a qwen3_5 branch; this one is unconditional, so when the installed
    # transformers has no `models.qwen3_5` (4.57.3 does not, and it landed in
    # the shared `test` env on 2026-08-21 10:38) this single line took down
    # every Qwen3-VL eval and rollout with ModuleNotFoundError. An absent
    # optional backbone must not break the primary one.
    _live_features_classes = (_Qwen3VL3DModel,)
    try:
        from model_adapters.qwen3_5.model import Qwen3_5_3DModel as _Q35
    except ImportError as e:
        print(f"[reproj-config] Qwen3.5 backbone unavailable, Qwen3-VL only ({e})")
    else:
        _live_features_classes = (_Qwen3VL3DModel, _Q35)

    def _find_reproj_config(ds):
        if ds is None:
            return None
        # Unwrap Subset
        if isinstance(ds, torch.utils.data.Subset):
            return _find_reproj_config(ds.dataset)
        if hasattr(ds, "_reprojection_config"):
            return ds._reprojection_config()
        for child in getattr(ds, "datasets", []) or []:
            cfg = _find_reproj_config(child)
            if cfg is not None:
                return cfg
        return None

    reproj_cfg = _find_reproj_config(dataset)
    if reproj_cfg is None:
        raise RuntimeError(
            "[live-features] could not extract _reprojection_config from dataset. "
            "Live-features inference cannot proceed without it (the model would "
            "silently fall through to text-only forward and produce garbage)."
        )

    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
    ]
    for c in candidates:
        if isinstance(c, _live_features_classes):
            c.reprojection_config = reproj_cfg
            print(f"[live-features] reprojection_config attached to {type(c).__name__}")
            return

    raise RuntimeError(
        "[live-features] could not locate a 3D model (Qwen3VL3DModel / "
        "Qwen3_5_3DModel) in the model tree. Without it the live-features "
        "PRE-PATH-A branch will not run and inference will produce garbage."
    )


def run_eval_loop(model, processor, dataset, dataloader, loss_dataloader,
                  tracker, data_args, args, da3_model=None):
    """Core evaluation loop — uses the shared generation loop from utils.eval_loop."""
    # Build DA3 geometry callback if needed
    da3_fn = None
    if da3_model is not None and not data_args.with_precomputed_geometry:
        da3_fn = lambda img: _compute_da3_geometry(img, da3_model)

    # ── Generation (shared with training gen_eval) ──────────────────────
    results, skipped = utils.run_generation_loop(
        model, processor, dataloader, model.device,
        max_new_tokens=args.max_new_tokens,
        compute_da3_geometry_fn=da3_fn,
        max_batches=1 if args.debug else None,
        log_prefix=f"[{getattr(args, '_current_dataset', 'eval')}]",
    )

    if skipped:
        print(f"[Warning] Skipped {len(skipped)} samples due to missing precomputed features")

    # ── Feed results into MetricTracker ─────────────────────────────────
    for i, result in enumerate(results):
        tracker.update(
            count=i,
            scene_id=result["scene_id"],
            question=result["question"],
            prediction=result["prediction"],
            ground_truths=result["ground_truths"],
            question_type=result["question_type"],
            all_predictions=result.get("all_predictions"),
            n_source_images=result.get("n_source_images"),
        )

        if (i + 1) % 20 == 0:
            stats = {m: (sum(v) / len(v) if v else 0) for m, v in tracker.metrics_acc.items()}
            grnd = tracker.grounding_stats()
            print("\n" + "=" * 40)
            print(f"METRICS SUMMARY AT STEP {i + 1}")
            if grnd:
                # Grounding-dataset run: show IoU thresholds, not text metrics.
                print(f"Acc@0.25: {grnd['grnd_Acc@0.25']:.1%} | Acc@0.5: {grnd['grnd_Acc@0.5']:.1%} | "
                      f"Acc@0.1: {grnd['grnd_Acc@0.1']:.1%}")
                print(f"Mean IoU: {grnd['grnd_mean_IoU']:.4f} | Median IoU: {grnd['grnd_median_IoU']:.4f} | "
                      f"Parse: {grnd['grnd_parse_rate']:.0%}")
                if "grnd_center_dist_mean" in grnd:
                    print(f"Center dist: {grnd['grnd_center_dist_mean']:.2f}m | "
                          f"<1m: {grnd['grnd_center_within_1m']:.1%} | "
                          f"<2m: {grnd['grnd_center_within_2m']:.1%}")
            else:
                _em1_line = f"METEOR: {stats.get('METEOR', 0):.4f} | EM@1: {stats.get('EM@1', 0):.4f}"
                if tracker.dataset_name == "sqa3d":
                    _em1_line += f" | EM@R1: {stats.get('EM@R1', 0):.4f}"
                print(_em1_line)
                print(f"ROUGE-L: {stats.get('ROUGEL', 0):.4f} | ROUGE-1: {stats.get('ROUGE1', 0):.4f}")
            if tracker.per_type_scores:
                print("--- Per-Type ---")
                for qt, sl in sorted(tracker.per_type_scores.items()):
                    print(f"  {qt}: {sum(sl) / len(sl):.4f} (n={len(sl)})")
            print("=" * 40 + "\n")
            tracker.save(step=i + 1)

    # ── Optional teacher-forced val loss (separate pass) ────────────────
    if loss_dataloader is not None:
        from utils.eval_loop import _METADATA_KEYS
        # For loss computation, we NEED labels (the model uses them to compute loss).
        _loss_exclude = (_METADATA_KEYS - {"labels"})
        for loss_batch in tqdm(loss_dataloader, desc="Val loss"):
            _labels = loss_batch.get("labels")
            if _labels is None or not isinstance(_labels, torch.Tensor) or (_labels == -100).all():
                continue
            loss_inputs = {
                k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
                for k, v in loss_batch.items()
                if k not in _loss_exclude and not k.startswith("_")
            }
            with torch.no_grad():
                _loss_out = model(**loss_inputs)
            if _loss_out.loss is not None:
                tracker.update_val_loss(_loss_out.loss.item())

    tracker.save()
    if tracker.val_losses:
        avg_loss = sum(tracker.val_losses) / len(tracker.val_losses)
        print(f"\nFinal Avg Val Loss: {avg_loss:.4f} (over {len(tracker.val_losses)} batches)")
    return tracker


def main():
    args = parse_args()
    datasets = args.datasets or ALL_BENCHMARKS

    # Resolve output base
    if args.exp_name is None:
        if args.lora and not args.no_lora:
            lora_name = Path(args.lora).name
            if lora_name == "best_checkpoint":
                lora_name = Path(args.lora).parent.name
            args.exp_name = f"output/{lora_name}"
        elif args.model_path and os.path.isdir(args.model_path):
            args.exp_name = f"output/{Path(args.model_path).name}"
        else:
            args.exp_name = "output/base_model"

    print(f"Benchmarks to run: {datasets}")
    print(f"Output base: {args.exp_name}")

    # ── Load model once ──────────────────────────────────────────────
    model, processor = load_model_and_processor(args)

    lora_path = None if args.no_lora else args.lora
    # Merged self-contained checkpoints carry depth_embedding.pt next to the
    # model weights: restore the 3D embeddings from there when no adapter dir
    # is available to restore them from.
    if (lora_path is None and args.model_path and os.path.isdir(args.model_path)
            and os.path.exists(os.path.join(args.model_path, "depth_embedding.pt"))):
        lora_path = args.model_path
    if getattr(args, "vanilla_qwen3vl", False):
        apply_patches(model)
        target_model = model
    else:
        target_model = configure_model_3d(model, args, lora_path)

    # ── Move any newly-initialised modules (depth/angle/camera embed) to GPU ──
    _model_device = model.get_input_embeddings().weight.device
    model.to(_model_device)

    # ── Patch IMAGE_TOKEN_INDEX to match model ───────────────────────
    import onecanvas.data.data_processor_3d as _dp3d
    if hasattr(model.config, "image_token_id"):
        _dp3d.IMAGE_TOKEN_INDEX = model.config.image_token_id

    # ── Load DA3 if needed ───────────────────────────────────────────
    # Vanilla baseline has no canvas/depth pipeline, so DA3 is never needed.
    da3_model = None
    if not args.precomputed_geometry and not getattr(args, "vanilla_qwen3vl", False):
        da3_model = load_da3_model()

    # ── Run each benchmark ───────────────────────────────────────────
    all_results = {}
    for ds_name in datasets:
        print("\n" + "=" * 60)
        print(f"  BENCHMARK: {ds_name}")
        print("=" * 60)

        exp_dir = os.path.join(args.exp_name, ds_name)
        t_name = tracker_name_for(ds_name)

        data_args = build_data_args(args, ds_name)
        dataset, dataloader, loss_dataloader = build_dataloaders(processor, data_args, args)

        # Live-features path needs reprojection_config on the inner Qwen3VL3DModel.
        # Vanilla baseline has no canvas, no live-features hookup needed.
        if not getattr(args, "vanilla_qwen3vl", False):
            _attach_reprojection_config(model, dataset)

        print(f"Loaded {len(dataset)} samples for {ds_name}")

        tracker = utils.MetricTracker(benchmarking=True, exp_name=exp_dir, dataset_name=t_name,
                                      pano_grounding_format=bool(getattr(data_args, "pano_grounding_format", False)))

        args._current_dataset = ds_name
        run_eval_loop(
            model, processor, dataset, dataloader, loss_dataloader,
            tracker, data_args, args, da3_model=da3_model,
        )

        # Collect summary
        final_stats = {m: (sum(v) / len(v) if v else 0) for m, v in tracker.metrics_acc.items()}
        if tracker.per_type_scores:
            final_stats["per_type"] = {
                qt: sum(sl) / len(sl) for qt, sl in tracker.per_type_scores.items()
            }
        grnd = tracker.grounding_stats()
        if grnd:
            final_stats.update(grnd)
        mgrnd = tracker.multi_grounding_stats()
        if mgrnd:
            final_stats.update(mgrnd)
        if t_name == "vsi_bench" and tracker.per_type_scores:
            final_stats["official_overall"] = tracker._compute_vsibench_overall()
            _dir_subs = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"}
            _merged = {}
            for _k, _v in tracker.per_type_scores.items():
                if _k in _dir_subs:
                    _merged.setdefault("object_rel_direction", []).extend(_v)
                else:
                    _merged[_k] = _v
            final_stats["per_type_merged"] = {_k: sum(_v) / len(_v) for _k, _v in _merged.items()}
        all_results[ds_name] = final_stats

        print(f"\n--- {ds_name} done ---")
        if t_name == "vsi_bench" and "official_overall" in final_stats:
            from utils.metrics import VSIBENCH_DISPLAY_ORDER, VSIBENCH_DISPLAY_NAMES
            print(f"  Avg: {final_stats['official_overall']:.4f}")
            for _key in VSIBENCH_DISPLAY_ORDER:
                _sc = final_stats.get("per_type_merged", {}).get(_key)
                if _sc is not None:
                    print(f"    {VSIBENCH_DISPLAY_NAMES.get(_key, _key)}: {_sc:.4f}")
        elif "grnd_Acc@0.25" in final_stats:
            print(f"  Acc@0.25: {final_stats['grnd_Acc@0.25']:.1%}  Acc@0.5: {final_stats['grnd_Acc@0.5']:.1%}")
            print(f"  Acc@0.1:  {final_stats['grnd_Acc@0.1']:.1%}  Acc@0.05: {final_stats['grnd_Acc@0.05']:.1%}")
            print(f"  Mean IoU: {final_stats['grnd_mean_IoU']:.4f}  Median IoU: {final_stats['grnd_median_IoU']:.4f}")
            if "grnd_center_dist_mean" in final_stats:
                print(f"  Center dist: {final_stats['grnd_center_dist_mean']:.2f}m "
                      f"(med: {final_stats['grnd_center_dist_median']:.2f}m)")
                print(f"  Within 0.5m: {final_stats['grnd_center_within_0.5m']:.1%}  "
                      f"1m: {final_stats['grnd_center_within_1m']:.1%}  "
                      f"2m: {final_stats['grnd_center_within_2m']:.1%}")
            print(f"  Parsed: {int(final_stats['grnd_n'])}/{int(final_stats['grnd_total'])} "
                  f"({final_stats['grnd_parse_rate']:.0%})")
        else:
            for k, v in final_stats.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

        # Free dataloader memory
        del dataset, dataloader, loss_dataloader, tracker
        gc.collect()
        torch.cuda.empty_cache()

    # ── Final summary across all benchmarks ──────────────────────────
    print("\n" + "=" * 60)
    print("  ALL BENCHMARKS SUMMARY")
    print("=" * 60)
    for ds_name, stats in all_results.items():
        _t = tracker_name_for(ds_name)
        if _t == "vsi_bench" and "official_overall" in stats:
            from utils.metrics import VSIBENCH_DISPLAY_ORDER, VSIBENCH_DISPLAY_NAMES
            print(f"  {ds_name:20s}  Avg={stats['official_overall']:.4f}")
            for _key in VSIBENCH_DISPLAY_ORDER:
                _sc = stats.get("per_type_merged", {}).get(_key)
                if _sc is not None:
                    print(f"    {VSIBENCH_DISPLAY_NAMES.get(_key, _key)}: {_sc:.4f}")
        elif "grnd_Acc@0.25" in stats:
            print(f"  {ds_name:20s}  Acc@0.25={stats['grnd_Acc@0.25']:.4f}  "
                  f"Acc@0.5={stats['grnd_Acc@0.5']:.4f}  mIoU={stats['grnd_mean_IoU']:.4f}")
        else:
            em = stats.get("EM@1", 0)
            meteor = stats.get("METEOR", 0)
            print(f"  {ds_name:20s}  EM@1={em:.4f}  METEOR={meteor:.4f}")
            if "per_type" in stats:
                for qt, sc in stats["per_type"].items():
                    print(f"    {qt}: {sc:.4f}")
        if tracker_name_for(ds_name) == "sp_bench":
            _print_spbench_paper_table(args, ds_name)
    print("=" * 60)
    print("Done.")


def _print_spbench_paper_table(args, ds_name: str) -> None:
    """Cross-reference SPBench qa_results_final.json with the source jsonls to
    emit the paper-format SI/MV × NQ/MCQ breakdown. Overall = mean(SI-Avg, MV-Avg)."""
    try:
        from scripts.spbench_paper_table import analyze, _fmt
    except ImportError:
        import importlib.util
        _p = Path(__file__).resolve().parent.parent / "scripts" / "spbench_paper_table.py"
        _spec = importlib.util.spec_from_file_location("_spbench_table", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        analyze, _fmt = _mod.analyze, _mod._fmt
    out_dir = Path(args.exp_name) if args.exp_name else Path("output")
    qa_path = out_dir / ds_name / "qa_results_final.json"
    if not qa_path.exists():
        print(f"  [paper-table] {qa_path} not found; skipping")
        return
    r = analyze(str(qa_path))
    print(f"  [paper-table] matched={r['n_matched']}, unmatched={r['n_unmatched']}")
    print(f"                SPBench-SI              SPBench-MV              ")
    print(f"                NQ     MCQ    Avg.      NQ     MCQ    Avg.      Overall")
    print(f"    Ours:      {_fmt(r['SI_NQ'])}  {_fmt(r['SI_MCQ'])}  {_fmt(r['SI_Avg'])}    "
          f"{_fmt(r['MV_NQ'])}  {_fmt(r['MV_MCQ'])}  {_fmt(r['MV_Avg'])}    {_fmt(r['Overall'])}")


if __name__ == "__main__":
    main()
