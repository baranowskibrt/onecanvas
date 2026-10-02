#!/usr/bin/env python3
"""Print SPBench results in the paper-table format (SPBench-SI / SPBench-MV
with NQ/MCQ breakdown, plus Overall = mean of the two Avg. columns).

Cross-references qa_results_final.json with the source jsonls in
$ONECANVAS_DATA_ROOT/vlm_annotations/spbench/ to classify each sample
by:
  - split: SI (1 image in annotation) vs MV (8 images).
  - category: MCQ (options provided) vs NQ (free-form numerical).

Uses the 'official' metric already computed per sample (MRA for NQ,
letter-accuracy for MCQ). Overall = mean(SI-Avg, MV-Avg), matching the
paper's SPBench table within rounding noise (±0.05).

Usage:
  python scripts/spbench_paper_table.py <qa_results_final.json>
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

_DATA_ROOT = os.environ.get("ONECANVAS_DATA_ROOT", "")
if not _DATA_ROOT:
    # Mirror onecanvas.data's resolution: fall back to the sibling datasets/ dir
    # so this table printer works when ONECANVAS_DATA_ROOT is unset (e.g. eval
    # jobs that rely on the sibling fallback). Without this, SI_PATH became the
    # root-less '/vlm_annotations/...' and crashed the post-eval table.
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _sibling = os.path.join(os.path.dirname(_repo_root), "datasets")
    if os.path.isdir(_sibling):
        _DATA_ROOT = _sibling
SI_PATH = f"{_DATA_ROOT}/vlm_annotations/spbench/spbench_si.jsonl"
MV_PATH = f"{_DATA_ROOT}/vlm_annotations/spbench/spbench_mv.jsonl"


def _build_split_map() -> dict:
    split_map: dict[tuple[str, str, str], tuple[str, bool]] = {}
    for split, path in [("SI", SI_PATH), ("MV", MV_PATH)]:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                key = (d["scene_name"], d["question_type"], d["question"])
                split_map[key] = (split, d.get("options") is not None)
    return split_map


def _build_split_map_with_nimg() -> dict:
    """Key: (scene, qt, question, n_images). Disambiguates colliding keys where
    the same (scene, qt, question) text appears in both SI (1 image) and MV
    (8 images) splits (42 such keys in the source, 94 total samples).
    """
    split_map: dict[tuple[str, str, str, int], tuple[str, bool]] = {}
    for split, path in [("SI", SI_PATH), ("MV", MV_PATH)]:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                key = (d["scene_name"], d["question_type"], d["question"], len(d["images"]))
                split_map[key] = (split, d.get("options") is not None)
    return split_map


# MCQ/NQ is fully determined by question_type in SPBench, so we can classify
# purely from the qa_results item (no annotation cross-reference needed).
_MCQ_TYPES = {"object_rel_direction", "object_rel_distance"}


def analyze(qa_results_path: str) -> dict:
    split_map_nimg = _build_split_map_with_nimg()
    split_map_legacy = _build_split_map()  # (scene, qt, q) — fallback for pre-plumbing runs
    with open(qa_results_path) as f:
        results = json.load(f)

    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_qt_split: dict[tuple[str, str], list[float]] = defaultdict(list)
    unmatched: list[dict] = []
    n_via_nimg = 0
    n_via_legacy = 0
    for r in results:
        q_first = r["question"].split("\n", 1)[0].strip()
        n_img = r.get("n_source_images", 0) or 0
        split: str | None = None
        if n_img:
            key4 = (r["scene_id"], r["question_type"], q_first, n_img)
            if key4 in split_map_nimg:
                split, _ = split_map_nimg[key4]
                n_via_nimg += 1
        if split is None:
            key3 = (r["scene_id"], r["question_type"], q_first)
            if key3 in split_map_legacy:
                split, _ = split_map_legacy[key3]
                n_via_legacy += 1
        if split is None:
            unmatched.append(r)
            continue
        is_mcq = r["question_type"] in _MCQ_TYPES
        score = r["metrics"]["official"]
        cat = "MCQ" if is_mcq else "NQ"
        buckets[(split, cat)].append(score)
        buckets[(split, "AVG")].append(score)
        by_qt_split[(split, r["question_type"])].append(score)

    def pct(key: tuple[str, str]) -> float | None:
        v = buckets.get(key)
        return 100.0 * sum(v) / len(v) if v else None

    si_avg = pct(("SI", "AVG"))
    mv_avg = pct(("MV", "AVG"))
    overall = (si_avg + mv_avg) / 2 if (si_avg is not None and mv_avg is not None) else None

    return {
        "SI_NQ":  pct(("SI", "NQ")),
        "SI_MCQ": pct(("SI", "MCQ")),
        "SI_Avg": si_avg,
        "MV_NQ":  pct(("MV", "NQ")),
        "MV_MCQ": pct(("MV", "MCQ")),
        "MV_Avg": mv_avg,
        "Overall": overall,
        "per_qt_split": {f"{s}_{qt}": 100.0 * sum(v) / len(v) for (s, qt), v in by_qt_split.items()},
        "n_matched": sum(len(v) for k, v in buckets.items() if k[1] == "AVG"),
        "n_unmatched": len(unmatched),
        "n_via_nimg": n_via_nimg,
        "n_via_legacy": n_via_legacy,
    }


def _fmt(v: float | None) -> str:
    return f"{v:5.2f}" if v is not None else "  --  "


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("qa_results_path", help="Path to qa_results_final.json produced by run_benchmarks.py on --datasets spbench")
    args = p.parse_args()

    path = Path(args.qa_results_path)
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    r = analyze(str(path))
    print(f"\nSPBench paper-format metrics for {path}")
    print(f"  matched={r['n_matched']} (via n_images={r['n_via_nimg']}, legacy jsonl={r['n_via_legacy']}), unmatched={r['n_unmatched']}")
    print()
    print("              SPBench-SI              SPBench-MV              ")
    print("              NQ     MCQ    Avg.      NQ     MCQ    Avg.      Overall")
    print(f"Ours:        {_fmt(r['SI_NQ'])}  {_fmt(r['SI_MCQ'])}  {_fmt(r['SI_Avg'])}    "
          f"{_fmt(r['MV_NQ'])}  {_fmt(r['MV_MCQ'])}  {_fmt(r['MV_Avg'])}    {_fmt(r['Overall'])}")

    print("\nPer-(split, question_type) official x100:")
    for key in sorted(r["per_qt_split"]):
        print(f"  {key:40s}: {r['per_qt_split'][key]:5.2f}")


if __name__ == "__main__":
    main()
