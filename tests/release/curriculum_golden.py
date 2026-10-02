#!/usr/bin/env python3
"""Golden-fixture harness for the spatial-pretraining curriculum (reproducibility gate).

The curriculum sampler is fully seeded: rng = Random(base_seed + seed_offset +
idx) in SpatialPretrainingDataset.__getitem__, so for a fixed (curriculum, seed,
split, idx) the question text, GT answer, sampled patch/frame indices, MCQ
shuffle, AND the synthetic OBB surface-point cloud are BIT-EXACT. This harness
freezes that signal and re-checks it after any refactor / deletion so we can
prove "nothing changed on the seed".

It runs the portable configuration (synthetic_obb curriculum, OBB feature stash
disabled, real-asset bank disabled): PATH B returns the synthetic surface
points in ``synthetic_patch_spherical`` / ``synthetic_patch_box_sizes``, which
we hash directly. That makes the golden catch the ``_sample_obb_surface_points``
self-shadowing bug (corners-only collapse) — the marker COUNT is unchanged by
that bug, only the point VALUES are, so the point values are hashed.

Reproduction basis: the ``synthetic_obb`` curriculum is byte-identical to the
paper's main curriculum except one task
rename (``box_floor_area_nonrect_yaw_stripped`` -> ``..._obb_only``) at the same
positions, and the sampler geometry is the same code lineage, so this golden
reproduces the trained curriculum. The synthetic-point geometry is independent
of the stash branch (both paths sample from the same pre-branch ``geom``).

Usage:
    # write the golden (run once against the release baseline)
    python tests/release/curriculum_golden.py generate --curriculum synthetic_obb --n 48

    # verify nothing changed (run after every curriculum-touching commit)
    python tests/release/curriculum_golden.py check --curriculum synthetic_obb --n 48

Exit code 0 == byte-identical to the committed golden; nonzero == drift.
This is CPU-only (loads AutoProcessor + scene geometry, no model weights).
Requires ONECANVAS_DATA_ROOT to point at the dataset tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GOLDEN_DIR = os.path.join(_HERE, "fixtures")

# Deterministic, refactor-invariant repro fields (skipped if absent).
_REPRO_KEYS = ("task", "scene", "scene_id", "base_seed", "seed_offset",
               "patch_indices", "frame_indices", "mcq_order")


def _build_dataset(curriculum: str, seed: int, num_images: int,
                   split: str, model_path: str):
    from transformers import AutoProcessor
    from onecanvas.train.argument import DataArguments
    from onecanvas.data.spatial_pretraining import SpatialPretrainingDataset

    da = DataArguments(curriculum=curriculum)          # applies tasks/strip/distractors
    da.dataset_use = "geometric_probing"
    da.num_images = num_images
    da.projection_mode = "equirectangular"
    da.dataset_sampling_seed = seed
    da.curriculum_samples_per_scene = 1
    da.skip_asset_validation = True
    da.use_precomputed_features = False                # user policy: always False
    # Trained-curriculum config: OBB feature stash ON (matches the checkpoints;
    # the geometry is stash-independent, but the stash PATH A is the one whose
    # consecutive __getitem__ calls are exercised in training). Real-asset bank
    # OFF — synthetic_obb has no *_real tasks. The stash cache is built on first
    # use (maybe_build_obb_feature_stash); pre-warm it once before running.
    da.real_object_assets_enable = False

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return SpatialPretrainingDataset(processor, da, data_split=split)


def _geom_digest(sample: dict) -> str:
    """Hash the synthetic canvas geometry (values, not just counts).

    The self-shadowing bug leaves the synthetic OBB point COUNT unchanged (rows
    are still ``n_total``) but collapses the values to repeated corners, so the
    point VALUES must be hashed. Two surfaces are covered so the digest works in
    both dataset paths:

    - Stash PATH A (trained config): ``projected_position_ids`` — the integer
      MRoPE position IDs for every canvas token including the synthetic OBB
      body; repeated corners produce repeated position rows, so a value hash
      distinguishes corners-only from a dense surface cloud.
    - PATH B (stash off): ``synthetic_patch_spherical`` / ``_box_sizes``.

    Empty only when the task places no synthetic OBB body and no projection is
    returned."""
    import torch
    h = hashlib.sha256()
    touched = False
    pid = sample.get("projected_position_ids")
    if pid is not None:
        h.update(pid.detach().cpu().to(torch.int64).numpy().tobytes())
        touched = True
    sp = sample.get("synthetic_patch_spherical")
    if sp is not None:
        q = (sp.detach().cpu().to(torch.float64) * 1e4).round().to(torch.int64)
        h.update(q.numpy().tobytes())
        touched = True
    bs = sample.get("synthetic_patch_box_sizes")
    if bs is not None:
        h.update(bytes(str([int(x) for x in bs.tolist()]), "utf8"))
        touched = True
    return h.hexdigest()[:16] if touched else ""


def _extract(sample: dict) -> dict:
    """Deterministic, refactor-invariant projection of one __getitem__ result."""
    repro = dict(sample.get("_repro", {}) or {})
    rec = {
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "question_type": sample.get("question_type", ""),
        "repro": {k: repro[k] for k in _REPRO_KEYS if k in repro},
        "geom": _geom_digest(sample),
    }
    return rec


def _seed_all(s: int):
    """Pin the ambient global RNGs before each __getitem__.

    We deliberately do NOT pin torch's global RNG here: as of Phase 1.3
    SpatialPretrainingDataset.__getitem__ pins it itself (fork_rng +
    manual_seed(base + seed_offset + idx)) around the item, so the torch
    surface-point / randn / randperm draws are reproducible from core alone.
    Leaving torch un-pinned here makes this golden a real guard on that fix:
    revert the in-getitem pin and these records drift. python's global
    `random` and numpy are seeded only as cheap insurance against any global
    draw the sampler might grow; the sampler itself uses its own per-idx
    `random.Random(...)`, not the module singleton."""
    import random as _r
    import numpy as _np
    _r.seed(s)
    _np.random.seed(s % (2 ** 32))


def _records(curriculum: str, seed: int, num_images: int, split: str,
             model_path: str, n: int) -> list:
    ds = _build_dataset(curriculum, seed, num_images, split, model_path)
    # mirror the dataset's per-idx seed formula (train offset 0, val 999)
    seed_offset = 0 if split == "train" else 999
    out = []
    for idx in range(n):
        _seed_all(seed + seed_offset + idx)
        sample = ds[idx]
        rec = _extract(sample)
        rec["idx"] = idx
        rec["hash"] = hashlib.sha256(
            json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()[:16]
        out.append(rec)
    return out


def _golden_path(curriculum: str) -> str:
    return os.path.join(GOLDEN_DIR, f"curriculum_{curriculum}.json")


def _payload(args, records):
    return {
        "curriculum": args.curriculum, "seed": args.seed,
        "num_images": args.num_images, "split": args.split,
        "n": args.n, "model_path": args.model_path,
        "_generated_by": (
            "tests/release/curriculum_golden.py generate (release repo, post Phase 1.3: "
            "per-item torch RNG pin in __getitem__ + aug-seed decorrelation "
            "(_aug_seed = base + 0x5EED), on top of the _sample_obb_surface_points "
            "unshadow fix); synthetic_obb == the paper curriculum modulo the box_floor_area "
            "task rename. Only rel_dir_camera_* samples move vs. the pre-1.3 golden "
            "(camera-pose aug now draws from a decorrelated stream)."
        ),
        "records": records,
    }


def cmd_generate(args):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    records = _records(args.curriculum, args.seed, args.num_images,
                       args.split, args.model_path, args.n)
    with open(_golden_path(args.curriculum), "w") as f:
        json.dump(_payload(args, records), f, indent=2, default=str)
    print(f"[golden] wrote {len(records)} records -> {_golden_path(args.curriculum)}")
    return 0


def cmd_check(args):
    gp = _golden_path(args.curriculum)
    if not os.path.exists(gp):
        print(f"[FAIL] no golden at {gp}; run `generate` first against the baseline")
        return 2
    with open(gp) as f:
        golden = json.load(f)
    fresh = _records(args.curriculum, golden["seed"], golden["num_images"],
                     golden["split"], golden["model_path"], golden["n"])
    g = {r["idx"]: r["hash"] for r in golden["records"]}
    drift = [r["idx"] for r in fresh if g.get(r["idx"]) != r["hash"]]
    if drift:
        print(f"[FAIL] {len(drift)}/{len(fresh)} samples drifted from golden: "
              f"idx {drift[:10]}{'...' if len(drift) > 10 else ''}")
        bad = next(r for r in fresh if r["idx"] == drift[0])
        gold = next(r for r in golden["records"] if r["idx"] == drift[0])
        print(f"  idx {drift[0]} now : {json.dumps(bad, default=str)[:300]}")
        print(f"  idx {drift[0]} gold: {json.dumps(gold, default=str)[:300]}")
        return 1
    print(f"[OK] {len(fresh)}/{len(fresh)} samples byte-identical to golden ({args.curriculum})")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "check"):
        sp = sub.add_parser(name)
        sp.add_argument("--curriculum", default="synthetic_obb")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--num-images", dest="num_images", type=int, default=32)
        sp.add_argument("--split", default="train")
        sp.add_argument("--n", type=int, default=48)
        sp.add_argument("--model-path", dest="model_path",
                        default="Qwen/Qwen3-VL-8B-Instruct")
    args = p.parse_args()
    sys.exit({"generate": cmd_generate, "check": cmd_check}[args.cmd](args))


if __name__ == "__main__":
    main()
