"""Post-hoc VSI-Bench scorer for qa_results.json."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.metrics import (
    vsibench_score, VSIBENCH_DISPLAY_ORDER, VSIBENCH_DISPLAY_NAMES,
)

path = sys.argv[1]
with open(path) as f:
    results = json.load(f)

per_type = {}
for r in results:
    qt = r.get("question_type", "unknown")
    pred = r["prediction"]
    gts = r["ground_truths"]
    score = max(vsibench_score(pred, gt, qt) for gt in gts) if gts else 0.0
    per_type.setdefault(qt, []).append(score)

type_means = {qt: sum(s)/len(s) for qt, s in per_type.items()}
dir_subs = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"]
dir_means = [type_means[d] for d in dir_subs if d in type_means]
if dir_means:
    type_means["object_rel_direction"] = sum(dir_means) / len(dir_means)

print(f"\n[VSI-Bench] {path}  ({len(results)} samples)")
ordered = []
for qt in VSIBENCH_DISPLAY_ORDER:
    if qt in type_means:
        name = VSIBENCH_DISPLAY_NAMES.get(qt, qt)
        if qt == "object_rel_direction":
            n = sum(len(per_type[d]) for d in dir_subs if d in per_type)
        else:
            n = len(per_type.get(qt, []))
        print(f"  {name:<14s} {type_means[qt]*100:6.2f}  (n={n})")
        ordered.append(type_means[qt])
if ordered:
    print(f"  {'Avg':<14s} {sum(ordered)/len(ordered)*100:6.2f}")

unknown = [qt for qt in type_means if qt not in VSIBENCH_DISPLAY_ORDER and qt not in dir_subs]
if unknown:
    print("\n[other question_types seen]")
    for qt in unknown:
        print(f"  {qt:<32s} {type_means[qt]*100:6.2f}  (n={len(per_type[qt])})")
