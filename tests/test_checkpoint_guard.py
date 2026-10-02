"""Tests for checkpoint_guard against the 2026-08-05 ENOSPC failure signature:
a torn highest-numbered checkpoint (0-byte depth_embedding.pt, deepspeed shard
cut short) that auto-resume must skip in favor of the last complete one."""

import json
import os

import pytest

from onecanvas.train.checkpoint_guard import (SENTINEL_NAME,
                                              DiskSpaceGuardCallback,
                                              find_last_complete_checkpoint,
                                              finalize_checkpoint,
                                              is_checkpoint_complete)


def _make_ckpt(root, step, shard_bytes=1000, depth_bytes=100, sentinel=False):
    d = root / f"checkpoint-{step}"
    (d / "global_step" ).mkdir(parents=True)
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (d / "global_step" / "mp_rank_00_model_states.pt").write_bytes(b"x" * shard_bytes)
    (d / "depth_embedding.pt").write_bytes(b"y" * depth_bytes)
    if sentinel:
        (d / SENTINEL_NAME).write_text("{}")
    return d


def test_resume_skips_zero_byte_torn_checkpoint(tmp_path):
    _make_ckpt(tmp_path, 500)
    _make_ckpt(tmp_path, 1000)
    # the 2026-08-05 signature: highest checkpoint has a 0-byte embedding file
    _make_ckpt(tmp_path, 1500, depth_bytes=0)
    assert find_last_complete_checkpoint(tmp_path) == str(tmp_path / "checkpoint-1000")


def test_resume_skips_short_deepspeed_shard(tmp_path):
    _make_ckpt(tmp_path, 500)
    _make_ckpt(tmp_path, 1000)
    # shard cut short by ENOSPC: non-zero but smaller than siblings
    _make_ckpt(tmp_path, 1500, shard_bytes=200)
    assert find_last_complete_checkpoint(tmp_path) == str(tmp_path / "checkpoint-1000")


def test_sentinel_wins_over_structural_heuristics(tmp_path):
    _make_ckpt(tmp_path, 500, shard_bytes=1000)
    # smaller shard but sentinel present (e.g. legit config change): trusted
    _make_ckpt(tmp_path, 1000, shard_bytes=200, sentinel=True)
    assert find_last_complete_checkpoint(tmp_path) == str(tmp_path / "checkpoint-1000")


def test_all_torn_returns_none(tmp_path):
    _make_ckpt(tmp_path, 500, depth_bytes=0)
    assert find_last_complete_checkpoint(tmp_path) is None


def test_finalize_refuses_zero_byte_then_accepts(tmp_path):
    d = _make_ckpt(tmp_path, 500, depth_bytes=0)
    assert not finalize_checkpoint(d, step=500)
    assert not (d / SENTINEL_NAME).exists()
    (d / "depth_embedding.pt").write_bytes(b"y" * 10)
    assert finalize_checkpoint(d, step=500)
    assert (d / SENTINEL_NAME).exists()
    assert is_checkpoint_complete(d)


def test_prune_keeps_newest_complete(tmp_path):
    _make_ckpt(tmp_path, 500)
    _make_ckpt(tmp_path, 1000)
    keep = _make_ckpt(tmp_path, 1500, sentinel=True)
    cb = DiskSpaceGuardCallback()
    # need more space than the volume will ever report free -> prunes all but keep
    cb._prune_oldest(tmp_path, need=float("inf"))
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint-1500"]
    assert keep.exists()


class _Args:
    save_total_limit = None

    def __init__(self, out):
        self.output_dir = str(out)


class _State:
    def __init__(self, step):
        self.global_step = step


class _Control:
    should_save = True


def test_guard_skips_save_when_volume_short_and_no_fallback(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, 500, sentinel=True)

    import onecanvas.train.checkpoint_guard as g
    monkeypatch.setattr(g, "free_bytes", lambda path: 0)
    monkeypatch.delenv(g.FALLBACK_ENV, raising=False)

    control = DiskSpaceGuardCallback().on_step_end(_Args(tmp_path), _State(1000), _Control())
    assert control.should_save is False
    assert (tmp_path / "LOW_DISK_SAVES_SKIPPED").exists()
    # the checkpoint we would resume from must survive the prune attempt
    assert (tmp_path / "checkpoint-500").exists()


def test_guard_allows_save_when_space_ok(tmp_path):
    _make_ckpt(tmp_path, 500, sentinel=True)
    control = DiskSpaceGuardCallback().on_step_end(_Args(tmp_path), _State(1000), _Control())
    assert control.should_save is True


def test_guard_spills_to_fallback_volume(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    _make_ckpt(out, 500, sentinel=True)

    import onecanvas.train.checkpoint_guard as g
    # primary volume full, fallback roomy
    monkeypatch.setattr(
        g, "free_bytes",
        lambda path: 10**15 if str(path).startswith(str(fallback)) else 0)
    monkeypatch.setenv(g.FALLBACK_ENV, str(fallback))

    control = DiskSpaceGuardCallback().on_step_end(_Args(out), _State(1000), _Control())
    # the save proceeds, redirected through a symlink
    assert control.should_save is True
    link = out / "checkpoint-1000"
    assert link.is_symlink()
    target = fallback / "_spill" / "out" / "checkpoint-1000"
    assert link.resolve() == target.resolve()

    # a save writing through the symlink lands on the fallback and the
    # validated resume finder picks it up as the newest checkpoint
    (link / "trainer_state.json").write_text("{}")
    (link / "model.safetensors").write_bytes(b"z" * 100)
    assert (target / "trainer_state.json").exists()
    assert find_last_complete_checkpoint(out) == str(link)


def test_guard_picks_healthiest_fallback(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    small = tmp_path / "small"
    small.mkdir()
    big = tmp_path / "big"
    big.mkdir()

    import onecanvas.train.checkpoint_guard as g
    frees = {str(out): 0, str(small): 30 * (1 << 30), str(big): 500 * (1 << 30)}
    monkeypatch.setattr(
        g, "free_bytes",
        lambda path: next(v for k, v in frees.items() if str(path).startswith(k)))
    monkeypatch.setenv(g.FALLBACK_ENV, f"{small}:{big}")

    control = DiskSpaceGuardCallback().on_step_end(_Args(out), _State(200), _Control())
    assert control.should_save is True
    assert (out / "checkpoint-200").resolve() == (big / "_spill" / "out" / "checkpoint-200").resolve()


def test_cleanup_rotates_spilled_beyond_limit(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    fallback = tmp_path / "fallback"
    spill = fallback / "_spill" / "out"

    # two spilled + one real checkpoint, limit 2: the oldest (spilled) goes,
    # both its symlink and its data
    for step in (100, 200):
        d = spill / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text("{}")
        (out / f"checkpoint-{step}").symlink_to(d)
    _make_ckpt(out, 300, sentinel=True)

    args = _Args(out)
    args.save_total_limit = 2
    DiskSpaceGuardCallback().on_save(args, _State(300), _Control())

    assert not (out / "checkpoint-100").exists()
    assert not (spill / "checkpoint-100").exists()
    assert (out / "checkpoint-200").is_symlink()
    assert (spill / "checkpoint-200").exists()
    assert (out / "checkpoint-300").is_dir()


def test_launch_gate_fails_fresh_run_on_full_volume(tmp_path, monkeypatch):
    import onecanvas.train.checkpoint_guard as g
    monkeypatch.setattr(g, "free_bytes", lambda path: 0)
    monkeypatch.delenv(g.FALLBACK_ENV, raising=False)
    with pytest.raises(SystemExit):
        g.assert_free_space_at_launch(str(tmp_path))


def test_launch_gate_lets_resume_proceed_when_spill_has_room(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    _make_ckpt(out, 500, sentinel=True)

    import onecanvas.train.checkpoint_guard as g
    monkeypatch.setattr(
        g, "free_bytes",
        lambda path: 10**15 if str(path).startswith(str(fallback)) else 0)
    monkeypatch.setenv(g.FALLBACK_ENV, str(fallback))
    g.assert_free_space_at_launch(str(out))  # must not raise

    # same full volume but NO checkpoints -> fresh run must still hard-fail
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        g.assert_free_space_at_launch(str(empty))


def test_prune_removes_spilled_data_not_just_symlink(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    fallback = tmp_path / "fallback"
    d = fallback / "_spill" / "out" / "checkpoint-100"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text("{}")
    (out / "checkpoint-100").symlink_to(d)
    _make_ckpt(out, 200, sentinel=True)

    cb = DiskSpaceGuardCallback()
    cb._prune_oldest(out, need=float("inf"))
    assert not (out / "checkpoint-100").exists()
    assert not d.exists()
    assert (out / "checkpoint-200").exists()
