#!/usr/bin/env python3
"""Pin the ARKitScenes vga_wide frame selection and downsample.

Two steps here decide what the 640x480 and 320x240 ARKit evals actually read,
and only one of them is reproducible from the shipped code alone.

The downsample is: PIL LANCZOS at default save settings, verified byte-identical
against the published tree (96/96 frames over 12 scenes). PNG, so lossless, and
640x480 -> 320x240 is an exact 2x where an antialiased and a non-antialiased
filter visibly disagree. That one is pinned by test_downsample_recipe below.

The selection is not reproducible without help. `vga` keeps 256 evenly spaced
frames and deletes the rest IN PLACE, so the frame count that linspace divided
is destroyed by the very step that uses it. Nothing left on disk records it and
ARKitScenes' metadata.csv has no frame-count column, so the published subset
cannot be re-derived or even checked. Standing in the unpruned
lowres_wide_intrinsics list as a proxy for the original reproduces 0/10 scenes
(64 to 130 of 256 stems overlap), i.e. vga_wide is captured on its own
timestamps, not the lowres ones.

scripts/assets/arkitscenes_vga_frames.json.gz therefore ships the exact stems
for the 150 ARKitScenes scenes VSI-Bench evaluates. These tests check that the
manifest is used, that it wins over the frame count, and that a download which
cannot satisfy it fails instead of pruning to a third thing nobody can name.

Run with: pytest tests/release/test_arkit_frame_selection.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import download_arkitscenes as dl  # noqa: E402


def _fake_download(scene_dir, video_id, stems, size=(640, 480)):
    """A vga_wide/ as it looks straight off the official downloader."""
    img_dir = scene_dir / "vga_wide"
    intr_dir = scene_dir / "vga_wide_intrinsics"
    img_dir.mkdir(parents=True, exist_ok=True)
    intr_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for t in stems:
        arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{video_id}_{t}.png")
        (intr_dir / f"{video_id}_{t}.pincam").write_text("0 0 0 0 0 0\n")
    return img_dir, intr_dir


def test_manifest_ships_and_covers_the_vsi_bench_scenes():
    man = dl.load_frame_manifest()
    assert len(man) == 150, "VSI-Bench evaluates 150 ARKitScenes scenes"
    assert all(len(v) == 256 for v in man.values())
    assert dl.FRAME_MANIFEST.exists()


def test_manifest_selection_beats_the_frame_count(tmp_path):
    """The whole point: the kept set must not depend on how many frames came down.

    The synthetic download here holds the 256 manifest stems plus 40 extras, a
    count the original never had. linspace over 296 frames would keep a
    different subset. The manifest must keep exactly its own.
    """
    vid = "41069025"
    want = dl.load_frame_manifest()[vid]
    extras = [f"9{i:05d}.000" for i in range(40)]
    scene = tmp_path / vid
    _fake_download(scene, vid, list(want) + extras)

    dl.sample_scene(scene, vid, 256, (320, 240), manifest={vid: want})

    kept = sorted(p.stem for p in (scene / "vga_wide").glob("*.png"))
    assert kept == sorted(f"{vid}_{t}" for t in want)
    assert len(kept) == 256
    # intrinsics must be pruned in lockstep or the loader pairs frames wrongly
    assert sorted(p.stem for p in (scene / "vga_wide_intrinsics").glob("*.pincam")) == kept


def test_manifest_result_differs_from_the_linspace_fallback(tmp_path):
    """Guard that the manifest is doing real work, not agreeing by luck."""
    vid = "41069025"
    want = list(dl.load_frame_manifest()[vid])
    extras = [f"9{i:05d}.000" for i in range(40)]
    all_stems = sorted(want + extras)

    scene = tmp_path / "fallback" / vid
    _fake_download(scene, vid, all_stems)
    dl.sample_scene(scene, vid, 256, (320, 240), manifest=None)
    fallback = sorted(p.stem for p in (scene / "vga_wide").glob("*.png"))

    assert fallback != sorted(f"{vid}_{t}" for t in want), (
        "linspace over a different frame count happened to match the manifest; "
        "this test can no longer detect a selection regression"
    )


def test_missing_manifest_frames_refuse_to_prune(tmp_path):
    """A download that cannot satisfy the manifest must fail, not improvise."""
    vid = "41069025"
    want = list(dl.load_frame_manifest()[vid])
    scene = tmp_path / vid
    _fake_download(scene, vid, want[:-5])  # 5 frames short

    with pytest.raises(SystemExit) as e:
        dl.sample_scene(scene, vid, 256, (320, 240), manifest={vid: want})
    assert "missing 5" in str(e.value)
    # nothing was deleted
    assert len(list((scene / "vga_wide").glob("*.png"))) == 251


def test_unlisted_scene_falls_back_and_says_so(tmp_path, capsys):
    """Training scenes are not pinned. That must be visible, not silent."""
    vid = "40753679"
    scene = tmp_path / vid
    _fake_download(scene, vid, [f"{i:04d}.000" for i in range(300)])

    dl.sample_scene(scene, vid, 256, (320, 240), manifest={"41069025": []})

    assert "not in the frame manifest" in capsys.readouterr().out
    assert len(list((scene / "vga_wide").glob("*.png"))) == 256


def test_downsample_recipe(tmp_path):
    """vga_wide/ is PIL LANCZOS of the native frame, at default save settings.

    Verified byte-identical against the published tree; pinned here so a future
    Pillow or a well-meaning switch to INTER_AREA is caught.
    """
    src = tmp_path / "native.png"
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(src)
    expected = Image.open(src).resize((320, 240), Image.LANCZOS)

    dl.resize_image_inplace(src, (320, 240))
    got = Image.open(src)
    assert got.size == (320, 240)
    assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_downsample_is_a_no_op_when_already_at_size(tmp_path):
    """Re-running vga must not resize an already-downsampled frame again."""
    src = tmp_path / "small.png"
    rng = np.random.default_rng(4)
    Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(src)
    before = src.read_bytes()
    dl.resize_image_inplace(src, (320, 240))
    assert src.read_bytes() == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
