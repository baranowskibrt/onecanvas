#!/usr/bin/env python3
"""Guard --frozen-cache against writing empty annotation files and exiting 0.

Found by re-running every shipped converter from scratch. Pointing --out-dir at
a fresh directory produced this, with a zero exit status:

    ScanRefer val: 0 written, 9508 skipped -> .../scanrefer_val.jsonl
    Multi3DRefer train: 0 written, 43838 skipped -> .../multi3drefer_train.jsonl
    nr3d: 0 written, 41503 skipped -> .../nr3d.jsonl

Two faults stacked. The bbox cache path was derived from --out-dir even though
in --frozen-cache mode the cache is an INPUT, so redirecting the output silently
moved the cache lookup somewhere empty. Those paths coincide when --out-dir is
left alone, which is why it read as correct. And with no cache every lookup
missed, so every item counted as "skipped" and the converter wrote empty files
and reported success. A skip count is not an error, and downstream an empty
grounding split just looks like a small dataset.

Run with: pytest tests/release/test_grounding_frozen_cache.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "convert_grounding_datasets.py"
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import convert_grounding_datasets as cg  # noqa: E402


class _Args:
    """Only the fields Paths() reads."""
    data_root = None
    out_dir = None
    bbox_cache = None
    scannet_preprocessed = None
    annot_cache = None
    scanrefer_dir = None
    multi3drefer_dir = None
    referit3d_dir = None


def _paths(tmp_path, **over):
    a = _Args()
    a.data_root = str(tmp_path / "root")
    for k, v in over.items():
        setattr(a, k, v)
    return cg.Paths(a)


def test_bbox_cache_does_not_follow_out_dir(tmp_path):
    """Redirecting the output must not move the frozen cache lookup."""
    default = _paths(tmp_path)
    redirected = _paths(tmp_path, out_dir=str(tmp_path / "elsewhere"))
    assert redirected.bbox_cache == default.bbox_cache, (
        "the bbox cache is an input in --frozen-cache mode; deriving it from "
        "--out-dir makes a redirected run silently lose it"
    )
    assert redirected.out_dir != default.out_dir


def test_local_bbox_cache_wins_when_it_exists(tmp_path):
    """A cache in the data root is preferred over the shipped asset."""
    local = tmp_path / "root" / "vlm_annotations" / "scannet_object_bboxes.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}")
    p = _paths(tmp_path)
    assert p.bbox_cache == str(local)
    assert Path(p.bbox_cache).parent == Path(p.out_dir)


def test_falls_back_to_the_shipped_cache(tmp_path):
    """A fresh checkout has no local cache, which is the normal case.

    --frozen-cache is unusable without a cache and a rebuilt one does not
    reproduce the published splits, so the release has to carry one.
    """
    p = _paths(tmp_path)  # no local cache written
    shipped = _REPO_ROOT / "scripts" / "assets" / "scannet_object_bboxes.json.gz"
    assert shipped.exists(), "the release must ship the pinned bbox cache"
    assert p.bbox_cache == str(shipped)


def test_shipped_cache_is_readable_and_covers_the_published_scene_set(tmp_path):
    p = _paths(tmp_path)
    cache = cg.build_bbox_cache([], p, frozen=True)
    assert len(cache) == 793, "the pinned cache is the 793-scene one"


def test_frozen_mode_refuses_a_missing_cache(tmp_path):
    p = _paths(tmp_path, bbox_cache=str(tmp_path / "nope.json"))
    with pytest.raises(SystemExit) as e:
        cg.build_bbox_cache(["scene0000_00"], p, frozen=True)
    assert "missing" in str(e.value)


def test_frozen_mode_refuses_an_empty_cache(tmp_path):
    cache = tmp_path / "empty.json"
    cache.write_text("{}")
    p = _paths(tmp_path, bbox_cache=str(cache))
    with pytest.raises(SystemExit) as e:
        cg.build_bbox_cache(["scene0000_00"], p, frozen=True)
    assert "empty" in str(e.value)


def test_frozen_mode_accepts_a_populated_cache(tmp_path):
    cache = tmp_path / "ok.json"
    cache.write_text(json.dumps({"scene0000_00": {"0": [0, 0, 0, 1, 1, 1]}}))
    p = _paths(tmp_path, bbox_cache=str(cache))
    got = cg.build_bbox_cache(["scene0000_00"], p, frozen=True)
    assert got == {"scene0000_00": {"0": [0, 0, 0, 1, 1, 1]}}


def test_cli_exits_nonzero_and_writes_nothing(tmp_path):
    """End to end: the shape of the original failure, now fatal."""
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--frozen-cache",
         "--data-root", str(tmp_path / "root"),
         "--out-dir", str(out),
         "--bbox-cache", str(tmp_path / "absent.json")],
        capture_output=True, text=True,
    )
    assert r.returncode != 0, "a run that can only produce empty files exited 0"
    assert not list(out.rglob("*.jsonl")) if out.exists() else True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
