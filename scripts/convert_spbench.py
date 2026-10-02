#!/usr/bin/env python3
"""Convert the SPBench parquet release to the OneCanvas eval jsonls.

Produces:
    <data_root>/vlm_annotations/spbench/spbench_si.jsonl   (1009 samples)
    <data_root>/vlm_annotations/spbench/spbench_mv.jsonl   (319 samples)

Consumed by the `spbench` registry entry (SPBENCH) in
training/onecanvas/data/__init__.py, whose annotation_path is the
vlm_annotations/spbench/ DIRECTORY (the loader globs *.jsonl under it, see
data_processor_3d.py). The si/mv file names are additionally hardcoded in
scripts/spbench_paper_table.py, so keep them exactly as above.

Upstream source: HuggingFace dataset `hongxingli/SPBench`
(https://huggingface.co/datasets/hongxingli/SPBench), files
SPBench-SI.parquet and SPBench-MV.parquet. Download e.g. with:

    huggingface-cli download hongxingli/SPBench --repo-type dataset \
        --local-dir <data_root>/vlm_annotations/spbench/

The upstream repo also ships SPBench-SI-images.zip / SPBench-MV-images.zip.
Those zips are UNUSED here: each item's "images" list holds frame stems like
"200.jpg" and the OneCanvas loader pins those frames out of the ScanNet
scene tree (<data_root>/scannet/scannet_preprocessed/<scene_name>/) by stem,
so the benchmark runs from the same ScanNet data as everything else.

The conversion is a faithful row-to-line dump: one JSON object per parquet
row, same fields, same order (id, dataset, scene_name, question_type,
question, ground_truth, options, images); numpy arrays become lists and a
missing options entry becomes null.

Example:
    python scripts/convert_spbench.py
    python scripts/convert_spbench.py --data-root /path/to/datasets
    python scripts/convert_spbench.py --spbench-dir /path/in --out-dir /path/out
"""

import argparse
import json
import os

import pandas as pd


def resolve_data_root(cli_value=None):
    """Mirror training/onecanvas/data/__init__.py: CLI > env > sibling datasets/."""
    if cli_value:
        return cli_value
    root = os.environ.get("ONECANVAS_DATA_ROOT")
    if root:
        return root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sibling = os.path.join(os.path.dirname(repo_root), "datasets")
    if os.path.isdir(sibling):
        return sibling
    raise SystemExit(
        "Cannot resolve dataset root: pass --data-root or export ONECANVAS_DATA_ROOT."
    )


def convert(parquet_path, out_path):
    df = pd.read_parquet(parquet_path)
    n = 0
    with open(out_path, "w") as out:
        for row in df.itertuples(index=False):
            rec = {
                "id": int(row.id),
                "dataset": row.dataset,
                "scene_name": row.scene_name,
                "question_type": row.question_type,
                "question": row.question,
                "ground_truth": row.ground_truth,
                "options": None if row.options is None else list(row.options),
                "images": list(row.images),
            }
            out.write(json.dumps(rec) + "\n")
            n += 1
    print(f"{parquet_path}: {n} items -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (default: $ONECANVAS_DATA_ROOT or sibling datasets/)")
    parser.add_argument("--spbench-dir", default=None,
                        help="Directory with SPBench-SI.parquet / SPBench-MV.parquet "
                             "(default: <data_root>/vlm_annotations/spbench/)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same as --spbench-dir)")
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    spbench_dir = args.spbench_dir or os.path.join(root, "vlm_annotations", "spbench")
    out_dir = args.out_dir or spbench_dir
    os.makedirs(out_dir, exist_ok=True)

    for parquet_name, jsonl_name in [
        ("SPBench-SI.parquet", "spbench_si.jsonl"),
        ("SPBench-MV.parquet", "spbench_mv.jsonl"),
    ]:
        parquet_path = os.path.join(spbench_dir, parquet_name)
        if not os.path.exists(parquet_path):
            raise SystemExit(
                f"Missing {parquet_path}. Download the hongxingli/SPBench "
                f"HuggingFace dataset into {spbench_dir} first."
            )
        convert(parquet_path, os.path.join(out_dir, jsonl_name))


if __name__ == "__main__":
    main()
