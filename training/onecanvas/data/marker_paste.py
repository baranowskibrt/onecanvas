"""Marker paste primitive for the tool-call marker-feedback loop.

Pastes exactly ONE canvas token per marker onto an already-reprojected scene,
so a model can SEE its own emitted 3D estimate in scene context. This is a
canvas capability (pasting patches onto a canvas), so it lives here in
onecanvas, not in the agentic repo (boundary rule). It is used by the
``toolcall_marker_points`` loader branch and the agentic rollout driver, both
of which build the marker set FRESH from a per-turn state dict, so a re-placed
marker's old token simply never exists in the rebuilt sequence (no masking, no
zeroing).

Per marker, exactly one appended token:
  1. Map the target point (in the sample's centered/yawed z-up call frame, the
     same frame the rendered call coordinates live in) to the canvas
     intermediate frame with the zero-yaw real-asset paste matrix. The loader
     has already applied the per-sample center/yaw, so this is a pure axis
     swap (z-up -> intermediate), matching how every real scene patch reached
     the canvas.
  2. ``_world_to_spherical`` for (lat, lon, depth).
  3. Feature = the marker id's fixed stash row (layer 0 + every aux layer). Row
     selection is a deterministic farthest-point sample over the stash's
     layer-0 vectors (``select_marker_rows``), so each id A-D gets an
     appearance maximally separated from typical scene content and from the
     other ids -- the model tells A from B by looking, not by matching
     coordinates against the transcript.
  4. Append the row field by field (embeds, each aux layer, lat, lon, depth,
     frame_index=0, bump n_valid), mirroring ``_append_real_object_assets``.
     Markers are appended LAST and are exempt from the real-asset patch cap.

Inline text-side reference tokens (the letter->marker binding) are the loader's
job: this primitive just returns, per marker, the canvas row it landed on, so
the loader can thread ``inline_patch_indices`` at that row.

Never imports agentic code.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional, Sequence, Tuple

import torch

from .spatial_pretraining._common import (
    _real_asset_paste_matrix,
    _world_to_spherical,
)

MARKER_POOL_VERSION = "onecanvas.marker_pool.v2"
MIN_MARKER_IDENTITIES = 128


def marker_identity_names(k: int = MIN_MARKER_IDENTITIES) -> Tuple[str, ...]:
    """Stable names for capacity checks, not a restriction on record ids."""
    legacy = ["A", "B", "C", "D"]
    return tuple((legacy + [f"marker_{i:03d}" for i in range(4, k)])[:k])


MARKER_IDS = marker_identity_names()
N_MARKER_IDS = len(MARKER_IDS)

# Canonical native tool schema for the toolcall_marker loop (v2). The loader
# passes this to tokenizer.apply_chat_template(tools=...) so the <tools> block
# renders in the system message. Source of truth lives HERE (release owns the
# token/canvas machinery); the agentic rollout driver imports it through the
# onecanvas_compat seam. Must stay in sync with the annotation builder's prose
# preamble (agentic build_toolcall_marker_annotations.py).
MARKER_TOOLS = [
    {"type": "function", "function": {
        "name": "place_marker",
        "description": ("Place (or re-place) a lettered marker at a 3D point on "
                        "the canvas. Calling it again with the same marker_id "
                        "moves that marker to the new position."),
        "parameters": {"type": "object", "properties": {
            "marker_id": {"type": "string", "description": "The marker letter, e.g. \"A\"."},
            "position": {"type": "array", "items": {"type": "number"},
                         "description": "The [x, y, z] position in metres."}},
            "required": ["marker_id", "position"]}}},
    {"type": "function", "function": {
        "name": "python",
        "description": ("Run a short numpy snippet that reads the placed markers "
                        "from the `markers` dict (marker_id -> 3D numpy array) and "
                        "prints the measurement needed to answer the question."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "The python source to execute."}},
            "required": ["code"]}}},
]

# --------------------------------------------------------------------------- #
# Delta correction grammar (added 2026-07-21, toolcall-marker branch).
#
# MARKER_TOOLS ABOVE IS DELIBERATELY UNTOUCHED. Every marker checkpoint ever
# trained or scored rendered exactly that two-tool list into its <tools> system
# block; appending a third entry to it would retroactively change the prompt of
# every one of them and invalidate every cross-checkpoint comparison on the
# track. The delta grammar is therefore a SEPARATE list, selected per episode.
#
# Why the op exists: with an ABSOLUTE correction (re-place at ground truth) the
# model earns perfect loss by ignoring the marker it is supposedly correcting and
# answering fresh -- and since the 2026-07-21 3D-PE harness fix a fresh answer is
# already ~2 cm, so an absolute correction turn carries essentially no gradient
# about the feedback. A RELATIVE target (gt - the model's own placement) cannot
# be produced without reading where the marker currently is.
ADJUST_MARKER_TOOL = {
    "type": "function", "function": {
        "name": "adjust_marker",
        "description": ("Nudge an already-placed marker by a relative offset. "
                        "The new position is the marker's current position plus "
                        "the delta. Use this to correct a marker you can see is "
                        "misplaced."),
        "parameters": {"type": "object", "properties": {
            "marker_id": {"type": "string",
                          "description": "The marker letter, e.g. \"A\"."},
            "delta": {"type": "array", "items": {"type": "number"},
                      "description": "The [dx, dy, dz] offset in metres to add "
                                     "to the marker's current position."}},
            "required": ["marker_id", "delta"]}}}

MARKER_TOOLS_DELTA = MARKER_TOOLS + [ADJUST_MARKER_TOOL]


def _expand_marker_pool(markers: torch.Tensor, target: int) -> torch.Tensor:
    """Extend a pinned pool without changing its original feature rows.

    Added rows are deterministic normalized blends of two tower-produced rows.
    They preserve layer count, feature width, dtype and per-layer norm scale.
    Every blend uses a distinct pair and coefficient.  The construction is
    versioned above and its bytes are included in the returned fingerprint.
    """
    if int(markers.shape[0]) >= target:
        return markers[:target]
    if int(markers.shape[0]) < 2:
        raise ValueError("marker pool needs at least two source features")
    out = [markers]
    n = int(markers.shape[0])
    made = n
    while made < target:
        i = made % n
        j = (made * 7 + 3) % n
        if j == i:
            j = (j + 1) % n
        alpha = 0.18 + 0.64 * (((made * 37) % 97) / 96.0)
        row = (1.0 - alpha) * markers[i].float() + alpha * markers[j].float()
        src_norm = markers[i].float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
        row = row / row.norm(dim=-1, keepdim=True).clamp_min(1e-8) * src_norm
        out.append(row.to(markers.dtype).unsqueeze(0))
        made += 1
    return torch.cat(out, dim=0)[:target]


def load_marker_pool(path: str, min_identities: int | None = None):
    """Load and version the pinned marker feature pool.

    The legacy artifact is sha1-checked first.  The second-iteration condition
    expands it to at least 128 distinct rows by default.  Set
    ``ONECANVAS_MARKER_POOL_IDENTITIES=16`` only for a named historical replay.
    scripts/build_toolmark_marker_pool.py) and assert its recorded sha1 matches
    the marker bytes. Returns (markers [K, N_layers, D] tensor, sha1 str).
    Marker features are tower outputs of rendered high-contrast patterns, never
    the OBB stash and never raw noise (work order section 3). The loader and
    the rollout driver both assert the SAME sha1 at startup so a silent pool
    swap is impossible."""
    art = torch.load(path, map_location="cpu")
    # "badges" is the pre-rename (2026-07-30) key of the old artifact still on
    # disk for in-flight jobs 2842471/2842466 -- old data, not old code. Drop
    # the fallback (and the old .pt) once those jobs are done.
    markers = art["markers"] if "markers" in art else art["badges"]
    recorded = art.get("sha1")
    actual = hashlib.sha1(markers.numpy().tobytes()).hexdigest()
    if recorded is not None and recorded != actual:
        raise ValueError(
            f"marker pool sha1 mismatch at {path}: recorded {recorded}, "
            f"computed {actual} (a silent pool swap breaks every marker's link)")
    target = int(os.environ.get(
        "ONECANVAS_MARKER_POOL_IDENTITIES",
        str(min_identities or MIN_MARKER_IDENTITIES)))
    if target <= 0:
        raise ValueError(f"marker identity capacity must be positive, got {target}")
    markers = _expand_marker_pool(markers, target)
    versioned = hashlib.sha1(
        MARKER_POOL_VERSION.encode() + markers.numpy().tobytes()).hexdigest()
    return markers, versioned


# --------------------------------------------------------------------------- #
# Deterministic per-id feature-row selection (farthest-point sampling)
# --------------------------------------------------------------------------- #
def select_marker_rows(stash: torch.Tensor, k: int = N_MARKER_IDS) -> List[int]:
    """Farthest-point-sample ``k`` stash rows on the layer-0 vectors.

    Deterministic given the stash file: seed at the row farthest from the stash
    mean, then greedily add the row maximizing the minimum L2 distance to the
    rows already chosen. Returns ``k`` int row indices, assigned to ids A..D in
    order. The stash keeps the raw ViT norm distribution on purpose (it is not
    standardized), so farthest-point rows are both far from typical scene
    content and mutually distinct -- exactly what makes A/B/C/D visually
    unambiguous.
    """
    x = stash[:, 0, :].float()                      # [N, D]
    n = x.shape[0]
    k = min(int(k), n)
    if k <= 0:
        return []
    mean = x.mean(dim=0, keepdim=True)
    first = int(((x - mean) ** 2).sum(dim=1).argmax())
    chosen = [first]
    min_d = ((x - x[first:first + 1]) ** 2).sum(dim=1)   # [N]
    min_d[first] = -1.0
    while len(chosen) < k:
        nxt = int(min_d.argmax())
        chosen.append(nxt)
        d = ((x - x[nxt:nxt + 1]) ** 2).sum(dim=1)
        min_d = torch.minimum(min_d, d)
        min_d[nxt] = -1.0
    return chosen


def stash_file_sha1(path: str, chunk: int = 1 << 20) -> str:
    """SHA1 of the stash file's bytes, for the train/rollout pinning assert (a
    silent stash swap changes every marker's appearance)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def marker_stash_rows(stash: torch.Tensor, stash_path: Optional[str] = None,
                      k: int = N_MARKER_IDS) -> List[int]:
    """``select_marker_rows`` with a JSON cache next to the stash file.

    The cache also records the stash sha1 and the k row indices -- the loader's
    stats JSON and the rollout dump header copy these for the startup pinning
    assert. Caching keeps per-sample cost to a dict lookup (FPS is O(N*k) and
    would otherwise recompute every sample).
    """
    if stash_path is None:
        return select_marker_rows(stash, k)
    cache = f"{stash_path}.marker_rows_{k}.json"
    if os.path.exists(cache):
        try:
            blob = json.load(open(cache))
            rows = [int(r) for r in blob.get("rows", [])]
            if blob.get("k") == k and len(rows) == k:
                return rows
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    rows = select_marker_rows(stash, k)
    try:
        tmp = cache + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"k": k, "rows": [int(r) for r in rows],
                       "stash_sha1": stash_file_sha1(stash_path)}, f)
        os.replace(tmp, cache)
    except OSError:
        pass
    return rows


def marker_stash_fingerprint(stash_path: str, rows: Sequence[int]) -> dict:
    """The pinning record embedded in the dataset stats JSON and the rollout
    dump header: the stash file sha1 and the chosen row indices."""
    return {"stash_sha1": stash_file_sha1(stash_path),
            "rows": [int(r) for r in rows]}


# --------------------------------------------------------------------------- #
# The paste
# --------------------------------------------------------------------------- #
def append_marker_rows(scene, points, features) -> List[int]:
    """Append one canvas token per (point, feature) pair to ``scene`` in place.

    This is the features-explicit core: the caller has already resolved each
    marker's stash feature row, so ``model.forward`` can paste without carrying
    the whole stash. ``append_marker_patches`` is the stash-resolving wrapper.

    Args:
        scene: a ``ReprojectedScene`` (already reprojected, per-sample
            center/yaw already applied).
        points: ``[K, 3]`` tensor or list of length-3 sequences, in the sample's
            centered/yawed z-up call frame (the frame the rendered call
            coordinates use).
        features: ``[K, N_layers, D]`` tensor, one feature row per point
            (layer 0 + every aux layer).

    Returns the canvas row indices of the appended tokens, IN ORDER (point k
    lands at row ``n_valid_before + k``). Markers are appended LAST and are
    exempt from any real-asset patch cap.
    """
    pts = points if torch.is_tensor(points) else torch.as_tensor(
        [[float(p[0]), float(p[1]), float(p[2])] for p in points],
        dtype=torch.float32)
    if pts.shape[0] == 0:
        return []
    n_layers = 1 + len(scene.aux_layers)
    if features.shape[1] != n_layers:
        raise ValueError(
            f"features have {features.shape[1]} layers but scene expects "
            f"{n_layers} (layer 0 + {len(scene.aux_layers)} aux layers)")

    dev = scene.latitude.device
    M = _real_asset_paste_matrix(0.0).to(dev)     # z-up call frame -> intermediate
    rows: List[int] = []
    for k in range(pts.shape[0]):
        p = pts[k].to(dtype=torch.float32, device=dev)
        inter = (M @ p).unsqueeze(0)              # [1, 3] canvas-intermediate
        lat, lon, depth, _valid = _world_to_spherical(inter)
        feat = features[k].to(device=dev)         # [N_layers, D]
        rows.append(int(scene.n_valid))

        scene.embeds = torch.cat(
            [scene.embeds, feat[0:1].to(scene.embeds.dtype)], dim=0)
        scene.aux_layers = [
            torch.cat([layer, feat[li + 1:li + 2].to(layer.dtype)], dim=0)
            for li, layer in enumerate(scene.aux_layers)
        ]
        scene.latitude = torch.cat(
            [scene.latitude, lat.to(scene.latitude.dtype)], dim=0)
        scene.longitude = torch.cat(
            [scene.longitude, lon.to(scene.longitude.dtype)], dim=0)
        scene.depth = torch.cat(
            [scene.depth, depth.to(scene.depth.dtype)], dim=0)
        scene.frame_index = torch.cat(
            [scene.frame_index,
             torch.zeros(1, dtype=scene.frame_index.dtype, device=dev)], dim=0)
        scene.n_valid = int(scene.n_valid) + 1
    return rows


def append_marker_patches(
    scene,
    markers: Sequence[Tuple[str, Sequence[float]]],
    stash: torch.Tensor,
    rows: Optional[Sequence[int]] = None,
) -> List[Tuple[str, int]]:
    """Append exactly one canvas token per marker to ``scene`` in place,
    resolving each marker id to its FPS stash row.

    Args:
        scene: a ``ReprojectedScene`` (already reprojected).
        markers: ordered list of ``(marker_id, point_call)``.
        stash: ``[N, N_layers, D]`` OBB feature stash.
        rows: optional precomputed FPS row indices for ids A..D; computed on the
            fly if None (callers should precompute + cache via
            ``marker_stash_rows``).

    Returns an ordered list of ``(marker_id, canvas_row_index)``, same order as
    ``markers``.
    """
    if not markers:
        return []
    if rows is None:
        rows = select_marker_rows(stash)
    n_layers = 1 + len(scene.aux_layers)
    if stash.shape[1] != n_layers:
        raise ValueError(
            f"stash has {stash.shape[1]} layers but scene expects {n_layers} "
            f"(layer 0 + {len(scene.aux_layers)} aux layers)")
    ids = [mid for mid, _ in markers]
    pts = [pt for _, pt in markers]
    if len(set(ids)) > len(rows):
        raise ValueError(
            f"{len(set(ids))} marker identities exceed {len(rows)} available rows")
    id_to_row = {mid: rows[i] for i, mid in enumerate(dict.fromkeys(ids))}
    feats = torch.stack([stash[id_to_row[mid]] for mid in ids], 0)
    row_idxs = append_marker_rows(scene, pts, feats)
    return list(zip(ids, row_idxs))
