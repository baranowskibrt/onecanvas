"""Audit GT pose sanity across all four datasets.

For each scene, read the GT pose translations (camera-to-world, post any
dataset-specific convention fixes that don't change |t|) and report:
    - n_poses, n_finite
    - max |t|, 95th-pctile |t|
    - per-axis translation span (max - min)
    - classification: OK / borderline / broken

This is the audit behind docs/bad_scenes.md (the "Drift" scenes it found are
the DA3_FALLBACK_SCENES in training/onecanvas/data/data_processor_3d.py).
The curated drop list docs/bad_scenes.json was derived from these per-dataset
outputs by hand; this script writes the raw per-dataset stats.

Dataset roots resolve through onecanvas.data (ONECANVAS_DATA_ROOT env var,
else a `datasets/` directory next to the repo root). CPU-only.

Usage:
    python scripts/audit_poses.py scannet
    python scripts/audit_poses.py scannetpp
    python scripts/audit_poses.py scannetpp_dslr
    python scripts/audit_poses.py arkitscenes [--workers 32] [--out path.json]

The machine-readable result defaults to docs/audit_<dataset>.json in the repo.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np

# Add repo root + training/ to sys.path so we can import onecanvas.data.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

from onecanvas.data import _DATA_PATH_MAP  # noqa: E402


SCANNET_ROOT = _DATA_PATH_MAP["scannet"]
SCANNETPP_ROOT = _DATA_PATH_MAP["scannetpp"]
ARKIT_ROOT = _DATA_PATH_MAP["arkitscenes"]


# --- per-scene pose-translation extraction -----------------------------------


def _scannet_translations(scene_id: str):
    scene_dir = Path(SCANNET_ROOT) / scene_id
    pose_dir = scene_dir / "pose"
    if not pose_dir.is_dir():
        return None
    ts = []
    for p in pose_dir.glob("*.txt"):
        try:
            m = np.loadtxt(str(p))
            if m.shape == (4, 4):
                ts.append(m[:3, 3])
        except Exception:
            continue
    return np.array(ts) if ts else None


def _scannetpp_iphone_translations(scene_id: str):
    jp = Path(SCANNETPP_ROOT) / scene_id / "iphone" / "pose_intrinsic_imu.json"
    if not jp.exists():
        return None
    try:
        data = json.loads(jp.read_text())
    except Exception:
        return None
    ts = []
    for entry in data.values():
        p = entry.get("pose")
        if p is None:
            continue
        m = np.asarray(p, dtype=np.float64)
        if m.shape == (4, 4):
            ts.append(m[:3, 3])
    return np.array(ts) if ts else None


def _scannetpp_dslr_translations(scene_id: str):
    jp = Path(SCANNETPP_ROOT) / scene_id / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    if not jp.exists():
        return None
    try:
        data = json.loads(jp.read_text())
    except Exception:
        return None
    ts = []
    for frame_list in (data.get("frames", []), data.get("test_frames", [])):
        for f in frame_list:
            m = np.asarray(f.get("transform_matrix"), dtype=np.float64)
            if m.shape == (4, 4):
                ts.append(m[:3, 3])
    return np.array(ts) if ts else None


def _arkit_translations(scene_tuple):
    split, scene_id = scene_tuple
    traj = Path(ARKIT_ROOT) / split / scene_id / "lowres_wide.traj"
    if not traj.exists():
        return None
    ts = []
    try:
        with open(traj) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 7:
                    continue
                ts.append([float(parts[4]), float(parts[5]), float(parts[6])])
    except Exception:
        return None
    return np.array(ts) if ts else None


# --- audit runner ------------------------------------------------------------


def _summarize(t):
    """Return dict of sanity stats given Nx3 translations."""
    if t is None or len(t) == 0:
        return {"n": 0}
    finite = np.isfinite(t).all(axis=1)
    t_fin = t[finite]
    if len(t_fin) == 0:
        return {"n": int(len(t)), "n_finite": 0}
    n_abs = np.linalg.norm(t_fin, axis=1)
    span = (t_fin.max(axis=0) - t_fin.min(axis=0)).max()
    return {
        "n": int(len(t)),
        "n_finite": int(len(t_fin)),
        "max_t": float(n_abs.max()),
        "p95_t": float(np.percentile(n_abs, 95)),
        "p50_t": float(np.percentile(n_abs, 50)),
        "span": float(span),
        "frac_over_50": float((n_abs > 50).mean()),
        "frac_over_500": float((n_abs > 500).mean()),
    }


def _classify(s):
    """OK / borderline / broken based on pose translation stats."""
    if s.get("n", 0) == 0:
        return "no_data"
    if s.get("n_finite", 0) == 0:
        return "all_nonfinite"
    if s["max_t"] > 500 or s["frac_over_500"] > 0.01:
        return "broken"
    if s["max_t"] > 50 or s["span"] > 30 or s["frac_over_50"] > 0.1:
        return "borderline"
    return "ok"


def run_dataset(dataset: str, n_workers=32, out_path=None):
    if dataset == "scannet":
        scenes = sorted(os.listdir(SCANNET_ROOT))
        scenes = [s for s in scenes if (Path(SCANNET_ROOT) / s).is_dir()]
        fn = _scannet_translations
    elif dataset == "scannetpp":
        scenes = sorted(os.listdir(SCANNETPP_ROOT))
        scenes = [s for s in scenes if (Path(SCANNETPP_ROOT) / s / "iphone").is_dir()]
        fn = _scannetpp_iphone_translations
    elif dataset == "scannetpp_dslr":
        scenes = sorted(os.listdir(SCANNETPP_ROOT))
        scenes = [s for s in scenes if (Path(SCANNETPP_ROOT) / s / "dslr").is_dir()]
        fn = _scannetpp_dslr_translations
    elif dataset == "arkitscenes":
        scenes = []
        for split in ("Training", "Validation"):
            sd = Path(ARKIT_ROOT) / split
            if sd.is_dir():
                for s in sorted(os.listdir(sd)):
                    if (sd / s).is_dir():
                        scenes.append((split, s))
        fn = _arkit_translations
    else:
        raise ValueError(dataset)

    print(f"# {dataset}: {len(scenes)} scenes, {n_workers} workers")

    with Pool(n_workers) as pool:
        results = list(pool.imap(fn, scenes, chunksize=4))

    buckets = {"ok": [], "borderline": [], "broken": [], "no_data": [], "all_nonfinite": []}
    for scene, ts in zip(scenes, results):
        s = _summarize(ts)
        label = _classify(s)
        buckets[label].append((scene, s))

    total = len(scenes)
    print(f"# summary: ok={len(buckets['ok'])} "
          f"borderline={len(buckets['borderline'])} "
          f"broken={len(buckets['broken'])} "
          f"no_data={len(buckets['no_data'])} "
          f"all_nonfinite={len(buckets['all_nonfinite'])} "
          f"(total={total})")

    for label in ("broken", "borderline", "all_nonfinite", "no_data"):
        items = buckets[label]
        if not items:
            continue
        print(f"\n## {label} ({len(items)})")
        items.sort(key=lambda x: -x[1].get("max_t", 0))
        for scene, s in items[:30]:
            name = scene if isinstance(scene, str) else f"{scene[0]}/{scene[1]}"
            if s.get("n", 0) == 0:
                print(f"  {name}: no data")
            elif s.get("n_finite", 0) == 0:
                print(f"  {name}: all non-finite ({s['n']} poses)")
            else:
                print(f"  {name}: n={s['n']} finite={s['n_finite']} "
                      f"max|t|={s['max_t']:.1f} p95={s['p95_t']:.1f} "
                      f"span={s['span']:.1f} "
                      f"frac>50m={s['frac_over_50']:.1%} frac>500m={s['frac_over_500']:.1%}")
        if len(items) > 30:
            print(f"  ... {len(items) - 30} more")

    # Also dump a machine-readable list for follow-up.
    if out_path is None:
        out_path = _REPO_ROOT / "docs" / f"audit_{dataset}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        k: [[s if isinstance(s, str) else list(s), st] for s, st in buckets[k]]
        for k in buckets
    }, indent=2))
    print(f"\n(full result dumped to {out_path})")


def main():
    parser = argparse.ArgumentParser(
        description="Audit GT pose sanity per dataset (see docs/bad_scenes.md).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset",
                        choices=["scannet", "scannetpp", "scannetpp_dslr", "arkitscenes"])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: docs/audit_<dataset>.json).")
    args = parser.parse_args()
    run_dataset(args.dataset, args.workers, args.out)


if __name__ == "__main__":
    main()
