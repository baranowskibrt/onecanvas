#!/usr/bin/env python3
"""Guard the GT-only scene index, the path a fresh user actually takes.

Two bugs shipped here at once and both failed silently, producing
`skipped N/N samples` and then a misdirected `Dataset is empty. Check your
annotation paths.` rather than anything pointing at the cause:

1. `_build_scene_dir_index` reads `getattr(self, "_use_gt_all", False)`, and
   `self._use_gt_all` used to be assigned LATER in `__init__` than the
   `_load_annotations` loop that builds the index. The default always won, so
   the index always demanded a `da3_geometry_balanced_256*_metric.pt` even
   under `use_gt_all=True`. A tree built exactly as docs/DATA.md describes
   (GT depth and poses, no precompute) loaded ZERO samples.

2. Once that branch was live, it turned out to match only bare directory names
   while `_resolve_image_sources` also accepts the resolution-suffixed form, so
   50 ScanNet++ DSLR scenes shipping only `resized_undistorted_images_320x240`
   were still called missing.

Neither is visible on any development tree here, because those all carry DA3
.pt files left over from the predicted-geometry experiments. Only a tree built
from raw public downloads exposes them, which is why this test builds one.

Run with: pytest tests/release/test_gt_only_scene_index.py
"""
import inspect
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))

from onecanvas.data.data_processor_3d import SceneQADataset  # noqa: E402


class _IndexOnly:
    """Just enough object to call the unbound index builder."""

    def __init__(self, use_gt_all):
        self._use_gt_all = use_gt_all

    # staticmethod() re-wraps it; a bare assignment would rebind it as an
    # instance method and pass self as rel_path.
    _scene_id_from_rel_path = staticmethod(SceneQADataset._scene_id_from_rel_path)
    _build_scene_dir_index = SceneQADataset._build_scene_dir_index


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def gt_only_tree(tmp_path):
    """A tree holding frames and GT sensor data but NO predicted geometry."""
    root = tmp_path / "scannet_preprocessed"
    for scene in ("scene0231_00", "scene0426_00"):
        for sub in ("color", "color_320x240", "color_640x480", "depth", "pose"):
            _touch(root / scene / sub / "0.jpg")
    return root


def test_gt_only_tree_indexes_every_scene(gt_only_tree):
    """The whole point: no .pt on disk must still index under use_gt_all."""
    index = _IndexOnly(True)._build_scene_dir_index(str(gt_only_tree))
    assert set(index) == {"scene0231_00", "scene0426_00"}, (
        "a GT-only tree indexed nothing; the loader would drop every sample as "
        "missing_scene_dir and then report 'Dataset is empty'"
    )


def test_geometry_gated_branch_still_requires_a_pt(gt_only_tree):
    """The non-GT branch must keep gating on predicted geometry."""
    assert _IndexOnly(False)._build_scene_dir_index(str(gt_only_tree)) == {}


def test_resolution_suffixed_image_dirs_count(tmp_path):
    """A scene holding only color_<WxH>/ must still index.

    _resolve_image_sources accepts the suffixed directory, so the index has to
    as well or it reports scenes missing that the loader could have read.
    """
    root = tmp_path / "scannet_preprocessed"
    _touch(root / "scene0231_00" / "color_320x240" / "0.jpg")
    assert "scene0231_00" in _IndexOnly(True)._build_scene_dir_index(str(root))


def test_scannetpp_iphone_subtree_keys_on_the_scene_id(tmp_path):
    """ScanNet++ nests frames one level down under iphone/."""
    root = tmp_path / "scannetpp"
    _touch(root / "09c1414f1b" / "iphone" / "rgb_320x240" / "frame_000000.jpg")
    index = _IndexOnly(True)._build_scene_dir_index(str(root))
    assert "09c1414f1b" in index
    assert index["09c1414f1b"].endswith("iphone")


def test_dslr_only_subtree_is_not_keyed_by_scene_id(tmp_path):
    """Documents existing behaviour, deliberately NOT changed here.

    _scene_id_from_rel_path special-cases ``iphone`` but not ``dslr``, so a
    <scene>/dslr directory lands in the index under the literal key "dslr".
    That is pre-existing and identical on both branches, and it is harmless
    because the datasets that read DSLR frames pass ``scene_subdir="dslr"``,
    which _resolve_scene_dir handles before consulting this index at all.
    Changing it would make new scenes resolvable and move published numbers,
    so it stays as-is and is pinned here instead.
    """
    root = tmp_path / "scannetpp"
    _touch(root / "02455b3d20" / "dslr" / "resized_undistorted_images_320x240" / "a.JPG")
    index = _IndexOnly(True)._build_scene_dir_index(str(root))
    assert "02455b3d20" not in index
    assert "dslr" in index


def test_non_frame_siblings_do_not_qualify(tmp_path):
    """ARKitScenes lowres_wide_intrinsics/ is metadata, not frames."""
    root = tmp_path / "arkit"
    _touch(root / "40753679" / "lowres_wide_intrinsics" / "a.pincam")
    _touch(root / "40753679" / "lowres_depth" / "a.png")
    assert _IndexOnly(True)._build_scene_dir_index(str(root)) == {}


def test_use_gt_all_is_set_before_annotations_are_loaded():
    """The ordering bug itself: the flag must precede the index build.

    Checked on the source because reaching the loop needs a real processor.
    """
    src = inspect.getsource(SceneQADataset.__init__)
    assign = src.index("self._use_gt_all = ")
    load = src.index("self._load_annotations(")
    assert assign < load, (
        "self._use_gt_all is assigned after _load_annotations runs, so "
        "_build_scene_dir_index will read the getattr default and gate every "
        "scene on predicted geometry regardless of the config"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
