"""Export a single self-contained OneCanvas checkpoint for release.

Replays the exact load chain run_benchmarks.py uses at eval time (base model,
merge the stage-1 LoRA chain, apply the stage-2 LoRA) but merges the final
adapter as well, then writes ONE directory holding everything the public code
needs:

  merged model safetensors + config.json     (save_pretrained)
  depth_embedding.pt [+ visual_merger.pt]    (copied from the stage-2 snapshot)
  processor / tokenizer files                (from the base model)
  resolved_config.json                       (regenerated: release-era data keys,
                                              lora_checkpoint_merge disabled)

The result loads with:

  python training/run_benchmarks.py --model-path <out_dir> --no-lora ...

Note on exactness: eval keeps the stage-2 adapter UNMERGED because
merge_and_unload() rounds in bfloat16 and can shift greedy decoding on rare
samples. A merged export accepts that rounding once; verify by re-running the
benchmarks on the exported model before publishing.

Example:

  python scripts/export_merged_checkpoint.py \
    --stage1-lora "$ONECANVAS_CHECKPOINT_DIR/stage1_curriculum/checkpoint-30000" \
    --lora "$ONECANVAS_CHECKPOINT_DIR/stage2_qa/best_checkpoint" \
    --out "$ONECANVAS_CHECKPOINT_DIR/OneCanvas-Qwen3-VL-8B"
"""

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct",
                   help="Base model HF id or path")
    p.add_argument("--stage1-lora", required=True,
                   help="Stage-1 adapter dir, or comma-separated chain merged in order "
                        "(mirrors training.lora_checkpoint_path)")
    p.add_argument("--lora", required=True,
                   help="Final-stage adapter dir; depth_embedding.pt is taken from here")
    p.add_argument("--run-config", default=None,
                   help="Dir holding the training resolved_config.json "
                        "(default: --lora dir, then its parent)")
    p.add_argument("--out", required=True, help="Output directory for the merged model")
    p.add_argument("--max-shard-size", default="4GB")
    return p.parse_args()


def find_run_config(args):
    candidates = [args.run_config] if args.run_config else [args.lora, str(Path(args.lora).parent)]
    for cand in candidates:
        if cand and (Path(cand) / "resolved_config.json").exists():
            return Path(cand) / "resolved_config.json"
    raise SystemExit(f"resolved_config.json not found in {candidates}; pass --run-config")


def regenerate_resolved_config(src_path, out_dir):
    """Rewrite the training config as a minimal eval-time config.

    Keeps only data keys that exist on the release DataArguments (old
    internal-era keys are dropped), keeps the model section, and replaces the
    training section: the exported model has no LoRA chain left to merge.
    """
    from onecanvas.train.argument import DataArguments, ModelArguments

    cfg = json.load(open(src_path))
    data_fields = {f.name for f in dataclasses.fields(DataArguments)}
    model_fields = {f.name for f in dataclasses.fields(ModelArguments)}
    data_in = cfg.get("data", {})
    dropped = sorted(set(data_in) - data_fields)
    out_cfg = {
        "model": {k: v for k, v in cfg.get("model", {}).items() if k in model_fields},
        "data": {k: v for k, v in data_in.items() if k in data_fields},
        "training": {"lora_checkpoint_merge": False, "lora_checkpoint_path": None},
    }
    with open(out_dir / "resolved_config.json", "w") as f:
        json.dump(out_cfg, f, indent=2)
    if dropped:
        print(f"[config] dropped {len(dropped)} non-release data keys: {', '.join(dropped)}")
    print(f"[config] wrote {out_dir / 'resolved_config.json'}")


def ensure_cross_version_rope_config(out_dir):
    """Make config.json loadable across the transformers range we support.

    transformers renamed the Qwen3-VL rope block from ``rope_scaling`` to
    ``rope_parameters`` in 5.x, and ``save_pretrained`` writes only whichever
    name the exporting version uses. A checkpoint exported under 5.x therefore
    dies under 4.57.x at ``config.rope_scaling.get("mrope_section")`` with
    ``'NoneType' object has no attribute 'get'``, and the reverse skew drops
    mrope_section on 5.x. pyproject allows ``transformers>=4.57``, so the
    artifact has to carry both spellings. 4.57.x also expects ``rope_theta``
    as a sibling field rather than inside the rope block; without it the model
    silently falls back to the class default instead of the trained value.
    """
    cfg_path = out_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    changed = []
    for scope in ("text_config", None):
        node = cfg.get(scope) if scope else cfg
        if not isinstance(node, dict):
            continue
        params, scaling = node.get("rope_parameters"), node.get("rope_scaling")
        if not params and not scaling:
            continue
        source = params or scaling
        theta = source.get("rope_theta", node.get("rope_theta"))
        where = scope or "top-level"

        if not scaling:
            node["rope_scaling"] = {k: v for k, v in source.items() if k != "rope_theta"}
            changed.append(f"{where}.rope_scaling")
        if not params:
            node["rope_parameters"] = dict(source)
            if theta is not None:
                node["rope_parameters"]["rope_theta"] = theta
            changed.append(f"{where}.rope_parameters")
        if theta is not None and node.get("rope_theta") != theta:
            node["rope_theta"] = theta
            changed.append(f"{where}.rope_theta")

    if changed:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"[config] cross-version rope keys added: {', '.join(changed)}")
    else:
        print("[config] rope keys already cross-version")


def ensure_cross_version_tokenizer_config(out_dir):
    """Normalize tokenizer_config.json to the spelling both transformers lines read.

    transformers 5.x renamed the special-token list from
    ``additional_special_tokens`` to ``extra_special_tokens``, and the two names
    hold different TYPES on 4.57.x: there ``extra_special_tokens`` is a dict of
    model-specific tokens, fed straight into
    ``_set_model_specific_special_tokens``. A checkpoint exported under 5.x
    therefore dies on the 4.57.x floor with ``'list' object has no attribute
    'keys'`` before any sample is read.

    Unlike the rope block, the two spellings cannot both be written: a list under
    the 5.x name is exactly what kills 4.57.x. We write the 4.x spelling, which
    is what ``Qwen/Qwen3-VL-8B-Instruct`` itself publishes and what 5.x read
    correctly when it produced this file. ``added_tokens_decoder`` is not
    restored because the fast tokenizer recovers it from tokenizer.json.
    """
    cfg_path = out_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        print("[tokenizer] no tokenizer_config.json to normalize")
        return
    with open(cfg_path) as f:
        cfg = json.load(f)

    changed = []
    if isinstance(cfg.get("extra_special_tokens"), list):
        cfg["additional_special_tokens"] = cfg.pop("extra_special_tokens")
        changed.append("extra_special_tokens(list) -> additional_special_tokens")
    for key in ("backend", "is_local"):
        # 5.x bookkeeping fields; 4.57.x forwards unknown keys into the
        # tokenizer constructor rather than ignoring them.
        if cfg.pop(key, None) is not None:
            changed.append(f"dropped 5.x-only '{key}'")

    if changed:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"[tokenizer] cross-version fixes: {', '.join(changed)}")
    else:
        print("[tokenizer] tokenizer_config.json already cross-version")


def main():
    args = parse_args()
    out_dir = Path(args.out)
    if "qwen3-vl" not in out_dir.name.lower() and "qwen3_vl" not in out_dir.name.lower():
        # _is_qwen3_vl() in train.py / run_benchmarks.py routes the model class
        # by substring-matching the path name; a miss silently loads Qwen3.5.
        raise SystemExit(f"Output dir name '{out_dir.name}' must contain 'Qwen3-VL' "
                         f"so the loaders route to the Qwen3-VL model class.")
    run_config = find_run_config(args)
    lora_dir = Path(args.lora)
    depth_pt = lora_dir / "depth_embedding.pt"
    if not depth_pt.exists():
        raise SystemExit(f"{depth_pt} not found: the final-stage snapshot must carry "
                         f"the 3D position embedding weights.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same class selection as run_benchmarks.load_model_and_processor.
    from model_adapters.qwen3_vl.model import Qwen3VL3DForConditionalGeneration as Model3DClass

    print(f"Loading base model on CPU: {args.model_path}")
    model = Model3DClass.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )

    stage_paths = [p.strip() for p in args.stage1_lora.split(",") if p.strip()]
    for i, spath in enumerate(stage_paths):
        print(f"[merge] stage-1 chain {i + 1}/{len(stage_paths)}: {spath}")
        model = PeftModel.from_pretrained(model, spath, torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()

    print(f"[merge] final adapter: {args.lora}")
    model = PeftModel.from_pretrained(model, args.lora, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    print(f"[save] writing merged model to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size=args.max_shard_size)

    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(args.model_path).save_pretrained(out_dir)
    print("[save] processor + tokenizer written")

    shutil.copy2(depth_pt, out_dir / "depth_embedding.pt")
    print(f"[save] copied {depth_pt}")
    merger_pt = lora_dir / "visual_merger.pt"
    if merger_pt.exists():
        shutil.copy2(merger_pt, out_dir / "visual_merger.pt")
        print(f"[save] copied {merger_pt}")

    regenerate_resolved_config(run_config, out_dir)
    ensure_cross_version_rope_config(out_dir)
    ensure_cross_version_tokenizer_config(out_dir)

    print("\nDone. Verify before publishing:")
    print(f"  python training/run_benchmarks.py --model-path {out_dir} --no-lora "
          f"--datasets vsi_bench sqa3d")


if __name__ == "__main__":
    main()
