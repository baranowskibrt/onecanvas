"""Disk-space and checkpoint-integrity guard for training runs.

Motivated by a shared volume that filled to 0 bytes: three
runs crashed mid-checkpoint with ENOSPC, and one run left a torn checkpoint
(0-byte depth_embedding.pt, deepspeed shard 8 MB short) that auto-resume
would have picked because it was the highest-numbered.

Three layers, all cheap:

1. ``assert_free_space_at_launch`` — refuse to start a run on a nearly-full
   volume (tunable via ONECANVAS_MIN_FREE_GB, 0 disables).
2. ``DiskSpaceGuardCallback`` — before every checkpoint write, compare free
   space against the size of the last complete checkpoint (with headroom).
   If short, prune the oldest prunable checkpoints; if still short, SPILL the
   checkpoint to the healthiest volume in ONECANVAS_FALLBACK_CKPT_DIRS
   (colon-separated roots): the real directory is created there and a
   symlink is dropped at output_dir/checkpoint-N, so the trainer, resume,
   rotation, and every script that globs the output dir keep seeing ONE
   directory. Only if no fallback has room either is the save SKIPPED
   loudly instead of letting torch.save crash the run — the run keeps
   training and retries at the next save step.

   HF's rotate_checkpoints uses shutil.rmtree(ignore_errors=True), which is
   a silent no-op on a symlink (verified against transformers 5.2.0), so
   spilled checkpoints are never rotation-crashed; the guard's own on_save
   cleanup deletes spilled checkpoints that fall outside save_total_limit.
3. ``finalize_checkpoint`` / ``find_last_complete_checkpoint`` — a save that
   completes without exception and contains no zero-byte files gets a
   SAVE_COMPLETE.json sentinel; auto-resume only considers sentinel-bearing
   checkpoints (with a structural fallback for checkpoints predating the
   sentinel).
"""

import hashlib
import json
import logging
import os
import pathlib
import re
import shutil
import time

import torch
from transformers import TrainerCallback

SENTINEL_NAME = "SAVE_COMPLETE.json"
LOW_DISK_FLAG = "LOW_DISK_SAVES_SKIPPED"
FALLBACK_ENV = "ONECANVAS_FALLBACK_CKPT_DIRS"
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

# Floor for the very first save of a run, when there is no completed
# checkpoint to measure. Stage-1/2 checkpoints (LoRA + optimizer shards +
# deepspeed state) run single-digit GiB; 20 GiB covers them with margin.
_DEFAULT_NEED_BYTES = 20 * (1 << 30)
# A save needs room for the new checkpoint BEFORE rotation deletes the old
# one, plus slack for other writers on a shared volume.
_HEADROOM = 1.5


def _ckpt_step(path: pathlib.Path):
    m = _CKPT_RE.match(path.name)
    return int(m.group(1)) if m else None


def _iter_checkpoints(output_dir):
    out = pathlib.Path(output_dir)
    if not out.is_dir():
        return []
    ckpts = [p for p in out.iterdir() if p.is_dir() and _ckpt_step(p) is not None]
    return sorted(ckpts, key=_ckpt_step)


def _walk_files(ckpt_dir):
    for root, _dirs, files in os.walk(ckpt_dir):
        for f in files:
            p = os.path.join(root, f)
            yield os.path.relpath(p, ckpt_dir), p


def _zero_byte_files(ckpt_dir):
    bad = []
    for rel, p in _walk_files(ckpt_dir):
        if rel == SENTINEL_NAME:
            continue
        try:
            if os.path.getsize(p) == 0:
                bad.append(rel)
        except OSError:
            bad.append(rel)
    return bad


def checkpoint_size_bytes(ckpt_dir) -> int:
    total = 0
    for _rel, p in _walk_files(ckpt_dir):
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return total


# The LEARNED state of a checkpoint: the adapter, its shape, and the learned
# spatial/depth modules that are saved beside it. NOT the optimizer or the
# deepspeed shards -- those are resume machinery, they are an order of magnitude
# larger, and two checkpoints with identical learned weights are the same model
# whatever their Adam moments look like. Measured 2026-09-13 on a live r=64
# checkpoint: 666 MB of adapter plus 0.5 MB of depth state, sha256 in 2.0 s off
# the page cache it was just written into, against a 125-step save interval.
_LEARNED_STATE_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "depth_embedding.pt",
    "visual_merger.pt",
)


def _file_digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_provenance(ckpt_dir) -> dict:
    """Immutable source-model provenance for this checkpoint's lineage.

    Read from the run's own ``resolved_config.json`` rather than from launcher
    memory, and stored INSIDE the checkpoint, so it stays attached to the
    weights after the run directory has been re-entered, re-pinned or copied.
    """
    cfg = None
    for cand in (pathlib.Path(ckpt_dir) / "resolved_config.json",
                 pathlib.Path(ckpt_dir).parent / "resolved_config.json"):
        if cand.is_file():
            try:
                with open(cand) as f:
                    cfg = json.load(f)
                break
            except ValueError:
                continue
    if not isinstance(cfg, dict):
        return {"resolved_config": "unavailable"}
    md, tr = cfg.get("model") or {}, cfg.get("training") or {}
    return {
        "base_model": md.get("model_name_or_path"),
        "vanilla_qwen3vl": md.get("vanilla_qwen3vl"),
        "lora_checkpoint_path": tr.get("lora_checkpoint_path"),
        "lora_checkpoint_merge": tr.get("lora_checkpoint_merge"),
        "lora_r": tr.get("lora_r"),
        "lora_alpha": tr.get("lora_alpha"),
        "seed": tr.get("seed"),
    }


def content_identity(ckpt_dir, step=None) -> dict:
    """Content-based identity of a checkpoint's learned state.

    Hashes the adapter, its config and the learned spatial/depth modules, and
    binds that to the immutable source-model provenance. A SIZE is not an
    identity: every LoRA of a given rank over the same target modules has the
    same byte count, so two different trainings of the same recipe -- exactly
    the case where reusing one's evaluation for the other is wrong -- were
    indistinguishable under the previous size-based key.
    """
    ckpt_dir = pathlib.Path(ckpt_dir)
    files = {}
    for name in _LEARNED_STATE_FILES:
        p = ckpt_dir / name
        if p.is_file():
            files[name] = _file_digest(p)
    provenance = _source_provenance(ckpt_dir)
    payload = {"step": (int(step) if step is not None else None),
               "files": files, "provenance": provenance}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"step": payload["step"], "files": files, "provenance": provenance,
            "digest": digest, "id": digest[:16], "strong": bool(files)}


def _completeness_problems(ckpt_dir) -> list:
    """The completeness invariants a checkpoint must pass to be marked complete.

    One list, used by ``finalize_checkpoint`` when it writes the sentinel and
    again when a historical sentinel is read back, so "complete" means the same
    thing at both ends and a derived identity cannot be looser than a minted one.
    """
    problems = []
    if not os.path.isfile(os.path.join(str(ckpt_dir), "trainer_state.json")):
        problems.append("trainer_state.json missing")
    zeros = _zero_byte_files(str(ckpt_dir))
    if zeros:
        problems.append(f"zero-byte files: {zeros}")
    return problems


# Derived identities are memoized per directory: a checkpoint directory is
# written once and never rewritten in place, and the hash is seconds of I/O.
_DERIVED_IDENTITIES: dict = {}


def _derive_historical_identity(ckpt_dir, payload):
    """Identity for a checkpoint whose sentinel predates content identities.

    HISTORICAL SENTINELS ARE NOT WEAK SENTINELS. Before the identity existed
    the sentinel was ``{step, wall_time, size_bytes}``, written by a finalize
    that enforced exactly the invariants in ``_completeness_problems``. Those
    checkpoints ARE complete and their learned state is still on disk, so the
    identity is derived from the weights rather than declared missing.
    Declaring it missing is what made every result measured on a checkpoint
    older than the identity permanently unattributable, so a resume re-measured
    its baseline on every requeue and could never reuse the one before it.

    Deriving is not a bypass of the completeness check, it is a second run of
    it: the invariants are re-checked against the directory as it stands now,
    and the same learned-state files ``finalize_checkpoint`` would have hashed
    are the ones hashed here. Nothing is written, so the weights are untouched.
    A torn directory, an absent sentinel, or a CURRENT sentinel that recorded a
    weak identity all still yield nothing usable -- only the absence of the
    ``content_identity`` key means "this predates the identity", and a present
    key is an answer rather than a gap.
    """
    key = str(pathlib.Path(ckpt_dir).resolve())
    if key in _DERIVED_IDENTITIES:
        return _DERIVED_IDENTITIES[key]
    identity = None
    problems = _completeness_problems(ckpt_dir)
    if problems:
        print(f"[ckpt-guard] {ckpt_dir} carries a pre-identity sentinel but does "
              f"NOT pass the completeness check now ({'; '.join(problems)}), so "
              "no identity is derived for it and nothing measured on it is "
              "reusable.", flush=True)
    else:
        try:
            identity = content_identity(ckpt_dir, step=payload.get("step"))
        except OSError as e:
            print(f"[ckpt-guard] {ckpt_dir}: could not hash the learned state of "
                  f"a pre-identity checkpoint ({e}); no identity derived.",
                  flush=True)
            identity = None
        else:
            identity["derived_from"] = "historical_sentinel"
            print(f"[ckpt-guard] {ckpt_dir}: derived content identity "
                  f"{identity['id']} (strong={identity['strong']}) from a "
                  "pre-identity SAVE_COMPLETE sentinel; weights unchanged.",
                  flush=True)
    _DERIVED_IDENTITIES[key] = identity
    return identity


def read_content_identity(ckpt_dir):
    """The identity of a checkpoint that was marked complete, or None.

    Normally this is the digest ``finalize_checkpoint`` computed once, while
    the files were still in page cache, and cached in the sentinel, so a reader
    never re-hashes 666 MB and never sees an identity for a checkpoint that was
    never marked complete.

    A sentinel with no ``content_identity`` key at all was written before the
    identity existed; see ``_derive_historical_identity`` for why that case is
    derived from the weights instead of read as missing.
    """
    sentinel = pathlib.Path(ckpt_dir) / SENTINEL_NAME
    if not sentinel.is_file():
        return None
    try:
        with open(sentinel) as f:
            payload = json.load(f) or {}
    except ValueError:
        return None
    if "content_identity" in payload:
        return payload["content_identity"]
    return _derive_historical_identity(ckpt_dir, payload)


def finalize_checkpoint(ckpt_dir, step=None) -> bool:
    """Validate a just-written checkpoint and mark it complete.

    Call on rank 0 only, after everything belonging to the checkpoint
    (trainer state, deepspeed shards, 3D embedding .pt files) is on disk.
    Refuses the sentinel (and says so loudly) if any file is 0 bytes or
    trainer_state.json is missing.

    The content identity is computed HERE, once, while the files are still in
    page cache, and cached in the sentinel. Nothing else may mint one: an
    identity for a checkpoint that was never marked complete would be an
    identity for a possibly torn model.
    """
    ckpt_dir = str(ckpt_dir)
    problems = _completeness_problems(ckpt_dir)
    if problems:
        print(f"[ckpt-guard] REFUSING to mark {ckpt_dir} complete: "
              f"{'; '.join(problems)}. Auto-resume will skip it.")
        return False
    try:
        identity = content_identity(ckpt_dir, step=step)
    except OSError as e:
        print(f"[ckpt-guard] REFUSING to mark {ckpt_dir} complete: could not "
              f"hash its learned state ({e}). Auto-resume will skip it.")
        return False
    if not identity["files"]:
        # NOT a refusal: this function's contract is resume integrity, and a
        # full-finetune or otherwise unusual layout still resumes fine. But an
        # identity with nothing hashed under it must not be trusted for
        # evaluation reuse, so it says so rather than looking like a match.
        print(f"[ckpt-guard] {ckpt_dir}: no recognised learned-state file, so "
              "its content identity is weak and evaluation results measured "
              "on it are not reusable.")
    sentinel = {
        "step": step,
        "wall_time": time.time(),
        "size_bytes": checkpoint_size_bytes(ckpt_dir),
        "content_identity": identity,
    }
    tmp = os.path.join(ckpt_dir, SENTINEL_NAME + ".tmp")
    with open(tmp, "w") as f:
        json.dump(sentinel, f)
    os.replace(tmp, os.path.join(ckpt_dir, SENTINEL_NAME))
    return True


def is_checkpoint_complete(ckpt_dir, reference_sizes=None) -> bool:
    """Sentinel present, or (legacy, pre-sentinel) structurally sound.

    ``reference_sizes`` maps relative filename -> max size seen for that
    filename across sibling checkpoints; a legacy checkpoint whose copy of a
    non-.json file is SMALLER than the reference is treated as torn (this is
    what catches a deepspeed shard cut short by ENOSPC).
    """
    ckpt_dir = str(ckpt_dir)
    if os.path.isfile(os.path.join(ckpt_dir, SENTINEL_NAME)):
        return True
    if not os.path.isfile(os.path.join(ckpt_dir, "trainer_state.json")):
        return False
    if _zero_byte_files(ckpt_dir):
        return False
    if reference_sizes:
        for rel, p in _walk_files(ckpt_dir):
            if rel.endswith(".json"):
                continue
            ref = reference_sizes.get(rel)
            try:
                size = os.path.getsize(p)
            except OSError:
                return False
            if ref is not None and size < ref:
                return False
    return True


def _reference_sizes(ckpts):
    """Per-filename max size across checkpoints (constant-size files only
    matter: optimizer/model shards; .json files grow and are excluded)."""
    ref = {}
    for c in ckpts:
        for rel, p in _walk_files(c):
            if rel.endswith(".json") or rel == SENTINEL_NAME:
                continue
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            if size > ref.get(rel, -1):
                ref[rel] = size
    return ref


def find_last_complete_checkpoint(output_dir):
    """Highest-step checkpoint that passes the completeness check.

    Returns a str path or None. Logs every incomplete checkpoint it skips.
    """
    ckpts = _iter_checkpoints(output_dir)
    if not ckpts:
        return None
    ref = _reference_sizes(ckpts)
    for c in reversed(ckpts):
        if is_checkpoint_complete(c, reference_sizes=ref):
            return str(c)
        logging.warning("[ckpt-guard] skipping incomplete/torn checkpoint %s", c)
    return None


def free_bytes(path) -> int:
    return shutil.disk_usage(path).free


def assert_free_space_at_launch(output_dir):
    """Hard-fail launch when the output volume is nearly full.

    Threshold: ONECANVAS_MIN_FREE_GB (default 30). Set to 0 to disable.

    Exception: a RESUMING run (a complete checkpoint already in output_dir)
    with a roomy spill fallback configured may proceed on a full volume —
    reads come off it fine and every new save spills to the fallback.
    Moving such a run would orphan its checkpoints, so warning is correct.
    """
    min_free_gb = float(os.environ.get("ONECANVAS_MIN_FREE_GB", "30"))
    if min_free_gb <= 0:
        return
    free = free_bytes(output_dir)
    if free >= min_free_gb * (1 << 30):
        return
    need = min_free_gb * (1 << 30)
    roots = [r for r in os.environ.get(FALLBACK_ENV, "").split(":")
             if r and os.path.isdir(r)]
    if find_last_complete_checkpoint(output_dir) is not None \
            and any(free_bytes(r) >= need for r in roots):
        print(f"[ckpt-guard] WARNING: output volume for {output_dir} has only "
              f"{free / (1 << 30):.1f} GiB free, but this is a resume and a "
              f"spill fallback has room — proceeding; new checkpoints will "
              f"spill via {FALLBACK_ENV}.")
        return
    raise SystemExit(
        f"[ckpt-guard] output volume for {output_dir} has only "
        f"{free / (1 << 30):.1f} GiB free (< {min_free_gb:g} GiB). "
        f"Pick another volume (bash ~/scripts/check_node_storage.sh) or "
        f"set ONECANVAS_MIN_FREE_GB=0 to override."
    )


class DiskSpaceGuardCallback(TrainerCallback):
    """Spill or skip (never crash on) checkpoint saves when the volume is short.

    Register AFTER any callback that sets control.should_save so the guard
    sees the final decision. Rank 0 measures and decides; the decision is
    broadcast so all ranks agree (a split decision deadlocks deepspeed
    collectives during save).
    """

    def on_step_end(self, args, state, control, **kwargs):
        if not control.should_save:
            return control
        if _is_rank0():
            skip, spill_link = self._rank0_plan(args, state)
        else:
            skip, spill_link = False, None
        skip, spill_link = _broadcast_obj((skip, spill_link))
        if skip:
            control.should_save = False
        elif spill_link is not None and not _is_rank0():
            # NFS attribute caching can delay symlink visibility on other
            # nodes; make sure every rank sees the redirected dir before the
            # trainer's makedirs would create a REAL dir on the full volume.
            _wait_visible(spill_link)
        return control

    def on_save(self, args, state, control, **kwargs):
        # HF rotation silently skips symlinked (spilled) checkpoints, so
        # emulate save_total_limit for them here.
        if _is_rank0():
            self._cleanup_spilled(args)
        return control

    def _rank0_plan(self, args, state):
        """Returns (skip_save, spill_link_path)."""
        out = args.output_dir
        last = find_last_complete_checkpoint(out)
        need = int(_HEADROOM * checkpoint_size_bytes(last)) if last else _DEFAULT_NEED_BYTES
        free = free_bytes(out)
        if free >= need:
            return False, None
        freed = self._prune_oldest(out, need)
        free = free_bytes(out)
        if free >= need:
            print(f"[ckpt-guard] low disk: pruned {freed / (1 << 30):.1f} GiB of old "
                  f"checkpoints to make room ({free / (1 << 30):.1f} GiB free now)")
            return False, None
        link = self._spill(out, state, need)
        if link is not None:
            return False, link
        print(f"[ckpt-guard] *** SKIPPING CHECKPOINT SAVE: {free / (1 << 30):.1f} GiB "
              f"free on the volume of {out}, need ~{need / (1 << 30):.1f} GiB, and no "
              f"fallback volume ({FALLBACK_ENV}) has room either. Training continues; "
              f"the save retries at the next save step. Free space somewhere! ***")
        try:
            with open(os.path.join(out, LOW_DISK_FLAG), "a") as f:
                f.write(f"{time.time()}\n")
        except OSError:
            pass
        return True, None

    def _spill(self, output_dir, state, need):
        """Create checkpoint-N on the healthiest fallback volume and symlink
        it into output_dir. Returns the symlink path, or None if impossible."""
        roots = [r for r in os.environ.get(FALLBACK_ENV, "").split(":")
                 if r and os.path.isdir(r)]
        candidates = [(free_bytes(r), r) for r in roots]
        candidates = [(f, r) for f, r in candidates if f >= need]
        if not candidates or state is None:
            return None
        _, root = max(candidates)
        run_name = os.path.basename(os.path.normpath(output_dir))
        link = os.path.join(output_dir, f"checkpoint-{state.global_step}")
        target = os.path.join(root, "_spill", run_name,
                              f"checkpoint-{state.global_step}")
        if os.path.lexists(link):
            if os.path.islink(link):
                os.unlink(link)
            else:
                # a real (likely partial) dir already sits there; cannot
                # redirect this save
                print(f"[ckpt-guard] cannot spill: {link} already exists as a real dir")
                return None
        try:
            os.makedirs(target, exist_ok=True)
            os.symlink(target, link)
        except OSError as e:
            print(f"[ckpt-guard] spill to {target} failed: {e}")
            return None
        print(f"[ckpt-guard] *** low disk on the volume of {output_dir}: spilling "
              f"checkpoint-{state.global_step} to {target} (symlinked into the "
              f"output dir; resume and globs are unaffected) ***")
        return link

    def _prune_oldest(self, output_dir, need) -> int:
        """Delete oldest checkpoints (torn ones first are fine — they sort
        oldest anyway) while keeping the newest COMPLETE one, until enough
        space is free. Never touches best_checkpoint/. Returns bytes freed."""
        ckpts = _iter_checkpoints(output_dir)
        keep = find_last_complete_checkpoint(output_dir)
        freed = 0
        for c in ckpts:
            if keep is not None and str(c) == keep:
                continue
            if free_bytes(output_dir) >= need:
                break
            size = checkpoint_size_bytes(c)
            print(f"[ckpt-guard] low disk: deleting old checkpoint {c} "
                  f"({size / (1 << 30):.1f} GiB)")
            _remove_checkpoint(c)
            freed += size
        return freed

    def _cleanup_spilled(self, args):
        limit = getattr(args, "save_total_limit", None)
        if not limit or limit <= 0:
            return
        ckpts = _iter_checkpoints(args.output_dir)
        spilled_outside_limit = [c for c in ckpts[:-limit] if os.path.islink(c)]
        for c in spilled_outside_limit:
            print(f"[ckpt-guard] rotating spilled checkpoint {c} -> {os.path.realpath(c)}")
            _remove_checkpoint(c)


def _remove_checkpoint(path):
    """Delete a checkpoint dir, following (and removing) a spill symlink."""
    path = str(path)
    if os.path.islink(path):
        shutil.rmtree(os.path.realpath(path), ignore_errors=True)
        os.unlink(path)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _wait_visible(path, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.lexists(path):
            return
        time.sleep(0.2)
    print(f"[ckpt-guard] WARNING: spill symlink {path} not visible on this "
          f"rank after {timeout_s:.0f}s (NFS caching?); the save may land on "
          f"the full volume")


def _is_rank0() -> bool:
    return not (torch.distributed.is_available() and torch.distributed.is_initialized()) \
        or torch.distributed.get_rank() == 0


def _broadcast_obj(value):
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    obj = [value]
    torch.distributed.broadcast_object_list(obj, src=0)
    return obj[0]
