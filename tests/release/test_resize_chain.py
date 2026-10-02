#!/usr/bin/env python3
"""Pin the per-dataset resize chains that reproduce the published frames.

The release shipped one resize rule for every dataset (PIL LANCZOS from the
native frame directory, default JPEG quality). That is correct for ScanNet and
wrong for ScanNet++ iPhone, and it failed silently: a fresh user got frames that
looked fine and were not the ones the paper numbers were measured on. Measured
over 8 ScanNet++ scenes and 506 VSI-Bench questions, the wrong chain moved 99
of 506 predictions and the score by -0.0051.

What the published tree actually does, established byte-for-byte:

  ScanNet           color/ -> color_640x480/   (15/15 byte-identical)
                    color/ -> color_320x240/   (30/30 byte-identical)
  ScanNet++ iPhone  decoded video frame -> rgb/ and rgb_640x480/, one pass
                    rgb_640x480/ -> rgb_320x240/

The iPhone chain was first reconstructed WITHOUT a video to check against, from
two clues: in the published tree rgb/ and rgb_640x480/ frames are written 24 ms
apart by a single process, and a phase search over the exact 3x downscale picks
(dy=1, dx=1), the block centre, which is what cv2.resize samples at an integer
ratio. No PIL recipe reproduces it from rgb/, because rgb/ is a lossy JPEG of
that same frame.

A fresh ScanNet++ download on 2026-08-25 confirmed the reconstruction outright:
rebuilding 8 scenes with only these scripts produced rgb/, rgb_640x480/ and
rgb_320x240/ at 4000/4000 byte-identical each.

Run with: pytest tests/release/test_resize_chain.py
"""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import resize_frames  # noqa: E402


def _write_jpeg(path, size, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "JPEG", quality=95)


def test_scannetpp_iphone_320_chains_off_the_640_frames():
    assert resize_frames.RESIZE_SOURCE[("scannetpp_iphone", (320, 240))] == "rgb_640x480", (
        "rgb_320x240/ must be resized from rgb_640x480/. Resizing it from rgb/ "
        "reproduces nothing (mean pixel delta 2.1) and silently changes eval "
        "inputs at the size training reads."
    )


def test_scannet_is_not_chained():
    """ScanNet resizes both sizes from color/ and is already byte-exact."""
    for size in ((640, 480), (320, 240)):
        assert ("scannet", size) not in resize_frames.RESIZE_SOURCE
        assert ("scannet", size) not in resize_frames.NOT_BUILT_HERE


def test_iphone_640_is_refused_here():
    """This script must not offer to build a size it cannot reproduce."""
    assert ("scannetpp_iphone", (640, 480)) in resize_frames.NOT_BUILT_HERE


def test_chained_build_writes_the_right_directory_name(tmp_path):
    """Output naming follows the native dirname, not the chained source.

    A naive chain would write rgb_640x480_320x240/, which _resolve_image_sources
    does not match, so the loader would silently fall back to another size.
    """
    scene = tmp_path / "scannetpp" / "data" / "09c1414f1b" / "iphone"
    for i in range(3):
        _write_jpeg(scene / "rgb_640x480" / f"frame_{i:06d}.jpg", (640, 480), i)
    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "resize_frames.py"),
         "--dataset", "scannetpp_iphone", "--size", "320x240",
         "--data_root", str(tmp_path), "--quiet"],
        check=True, capture_output=True,
    )
    out = scene / "rgb_320x240"
    assert out.is_dir(), "chained build did not write rgb_320x240/"
    assert not (scene / "rgb_640x480_320x240").exists()
    assert sorted(p.name for p in out.iterdir()) == [
        f"frame_{i:06d}.jpg" for i in range(3)]
    assert Image.open(out / "frame_000000.jpg").size == (320, 240)


def test_chained_build_matches_a_direct_lanczos_of_the_640_frame(tmp_path):
    """The chain is PIL LANCZOS at default quality, byte-for-byte."""
    scene = tmp_path / "scannetpp" / "data" / "09c1414f1b" / "iphone"
    src = scene / "rgb_640x480" / "frame_000000.jpg"
    _write_jpeg(src, (640, 480), 7)
    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "resize_frames.py"),
         "--dataset", "scannetpp_iphone", "--size", "320x240",
         "--data_root", str(tmp_path), "--quiet"],
        check=True, capture_output=True,
    )
    expected = io.BytesIO()
    with Image.open(src) as im:
        im.resize((320, 240), Image.Resampling.LANCZOS).save(expected, "JPEG")
    got = (scene / "rgb_320x240" / "frame_000000.jpg").read_bytes()
    assert got == expected.getvalue()


def test_missing_chain_source_fails_loudly(tmp_path):
    """Skipping the 640x480 step must not look like success.

    The scene has rgb/ but no rgb_640x480/, i.e. the user ran the resize before
    the extraction. Writing nothing and exiting 0 is how the original bug hid.
    """
    scene = tmp_path / "scannetpp" / "data" / "09c1414f1b" / "iphone"
    _write_jpeg(scene / "rgb" / "frame_000000.jpg", (1920, 1440), 3)
    r = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "resize_frames.py"),
         "--dataset", "scannetpp_iphone", "--size", "320x240",
         "--data_root", str(tmp_path), "--quiet"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "rgb_640x480" in (r.stderr + r.stdout)
    assert not (scene / "rgb_320x240").exists()


def test_cv2_at_an_exact_3x_ratio_samples_the_block_centre():
    """Pin the operation the extractor relies on to match published frames.

    If a future OpenCV changes this mapping, rgb_640x480/ stops matching and
    this is the test that says so.
    """
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (1440, 1920, 3), dtype=np.uint8)
    got = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(got, frame[1::3, 1::3]), (
        "cv2.resize no longer point-samples the block centre at an exact 3x "
        "ratio; rgb_640x480/ will no longer reproduce the published frames"
    )


def test_extractor_writes_both_sizes_with_matching_stems(tmp_path):
    """rgb/ and rgb_640x480/ come out of one pass over the decoded video."""
    cv2 = pytest.importorskip("cv2")
    import json

    scene = tmp_path / "data" / "fakescene00" / "iphone"
    scene.mkdir(parents=True)
    vw = cv2.VideoWriter(str(scene / "rgb.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1440))
    rng = np.random.default_rng(0)
    for _ in range(10):
        vw.write(rng.integers(0, 255, (1440, 1920, 3), dtype=np.uint8))
    vw.release()
    (scene / "pose_intrinsic_imu.json").write_text(
        json.dumps({f"frame_{i:06d}": {} for i in range(10)}))

    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "extract_scannetpp_iphone_rgb.py"),
         "--data_root", str(tmp_path / "data"), "--scene", "fakescene00",
         "--num-frames", "4"],
        check=True, capture_output=True,
    )
    native = sorted(p.name for p in (scene / "rgb").iterdir())
    resized = sorted(p.name for p in (scene / "rgb_640x480").iterdir())
    assert native == resized, "stems must line up; the loader pairs them by stem"
    assert len(native) == 4
    assert Image.open(scene / "rgb" / native[0]).size == (1920, 1440)
    assert Image.open(scene / "rgb_640x480" / resized[0]).size == (640, 480)


def test_find_video_prefers_mkv_then_mp4(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_ex", _REPO_ROOT / "scripts" / "extract_scannetpp_iphone_rgb.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)

    scene = tmp_path / "09c1414f1b"
    (scene / "iphone").mkdir(parents=True)
    assert ex.find_video(scene) is None

    (scene / "iphone" / "rgb.mp4").write_bytes(b"")
    assert ex.find_video(scene).name == "rgb.mp4", "the legacy name must still work"

    (scene / "iphone" / "rgb.mkv").write_bytes(b"")
    assert ex.find_video(scene).name == "rgb.mkv", "current downloads ship .mkv"
    assert ex.VIDEO_NAMES[0] == "rgb.mkv"


def test_no_videos_found_is_fatal(tmp_path):
    """Zero matched scenes must not be a successful no-op.

    That is exactly how the .mp4 -> .mkv rename went unnoticed: a user
    downloads gigabytes, runs the documented command, is told it succeeded, and
    has an empty tree. Everything downstream then blames the annotations.
    """
    data_root = tmp_path / "scannetpp" / "data"
    (data_root / "09c1414f1b" / "iphone").mkdir(parents=True)
    r = subprocess.run(
        [sys.executable,
         str(_REPO_ROOT / "scripts" / "extract_scannetpp_iphone_rgb.py"),
         "--data_root", str(data_root)],
        capture_output=True, text=True,
    )
    assert r.returncode != 0, "a run that extracted nothing exited 0"
    out = r.stderr + r.stdout
    assert "rgb.mkv" in out and "rgb.mp4" in out, "the error must name both names"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
