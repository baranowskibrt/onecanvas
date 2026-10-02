"""Lazy-mmap whole-scene real-object asset bank with paste-time inflation.

Sister of RealObjectAssetBank. The legacy bank stores one .pt per OBB
with patches inside (tight bbox + 5 cm). This bank stores one .pt per
ScanNet scene with every valid patch + the OBB metadata for every
class-allowlisted annotation in that scene; the patch subset that
satisfies the OBB inclusion test is computed at sample time so paste-time
inflation becomes a free hyperparameter (no re-extraction per inflation
setting).

Each per-scene .pt (see scripts/extract_real_object_scene_assets.py):
    {
        "features":        [N_valid, N_layers, C] bf16,
        "world_xyz":       [N_valid, 3] float32,    # axis-aligned ScanNet world frame
        "frame_indices":   [N_valid] int16,
        "n_source_images": int,
        "source_scene":    str,
        "obbs": [{"target_id": int, "label": str,
                  "center": [3] float32, "dims": [3] float32,
                  "euler_zxy": [3] float32}, ...],
    }

A side-output ``scene_index.json`` at the bank root gives the per-class
lookup table:
    [{"label": str, "scene_path": str, "scene_id": str,
      "target_id": int, "obb_index": int}, ...]

Sample contract (matches RealObjectAssetBank for drop-in compatibility):
    {"features":        [K, N_layers, C] bf16,
     "xyz_offsets":     [K, 3] float32,    # OBB-local frame
     "frame_indices":   [K] int16,
     "n_source_images": int,
     "bbox_dims":       [3] float32,       # ALWAYS TIGHT, regardless of inflation_frac
     "obb_euler_zxy":   [3] float32,
     "label":           str,
     "source_scene":    str,
     "target_id":       int}

Critical invariant: bbox_dims is the TIGHT OBB. GT geometry (distance,
size, etc.) and the placement-collision sphere keep using tight dims.
Only ``xyz_offsets`` and ``features`` reflect the inflation_frac. So
the model has to localize the object inside the inflated patch before
measuring, but the supervision target stays correct.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch


def _euler_zxy_to_rotation_matrix_t(euler_zxy: torch.Tensor) -> torch.Tensor:
    """Torch port of utils.bbox._euler_zxy_to_rotation_matrix.

    EmbodiedScan stores angles matching pytorch3d.euler_angles_to_matrix(..., "ZXY").
    Returns a [3, 3] float32 tensor.
    """
    rx, ry, rz = float(euler_zxy[0]), float(euler_zxy[1]), float(euler_zxy[2])
    ca, sa = math.cos(rx), math.sin(rx)
    cb, sb = math.cos(ry), math.sin(ry)
    cc, sc = math.cos(rz), math.sin(rz)
    return torch.tensor([
        [ca * cc - sa * sb * sc, -sa * cb, ca * sc + sa * sb * cc],
        [sa * cc + ca * sb * sc,  ca * cb, sa * sc - ca * sb * cc],
        [-cb * sc,                sb,      cb * cc],
    ], dtype=torch.float32)


class RealObjectSceneBank:
    """Whole-scene asset bank with paste-time OBB inflation.

    LRU on per-scene mmap dicts caps the open-FD count. ``max_loaded_scenes``
    defaults to 128 because each scene serves ~10-20 OBBs so the working
    set churns slowly under shuffled per-class sampling.
    """

    def __init__(self, root, classes=None, max_loaded_scenes: int = 128):
        self.root = Path(root)
        index_path = self.root / "scene_index.json"
        idx = json.loads(index_path.read_text())
        self.by_label: dict = defaultdict(list)
        for entry in idx:
            if classes is None or entry["label"] in classes:
                self.by_label[entry["label"]].append(entry)
        self._lru: dict = {}
        self._lru_order: list = []
        self.max_loaded = int(max_loaded_scenes)

    def labels(self):
        return sorted(k for k, v in self.by_label.items() if v)

    def n_assets(self, label):
        return len(self.by_label.get(label, []))

    def _load(self, path: str) -> dict:
        if path in self._lru:
            return self._lru[path]
        d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        self._lru[path] = d
        self._lru_order.append(path)
        if len(self._lru_order) > self.max_loaded:
            evict = self._lru_order.pop(0)
            self._lru.pop(evict, None)
        return d

    def _build_asset(self, scene: dict, obb: dict, inflation_frac: float) -> dict:
        """Run the per-OBB inclusion test and pack a legacy-shaped asset dict."""
        center = obb["center"].to(torch.float32)
        dims_tight = obb["dims"].to(torch.float32)
        euler = obb["euler_zxy"].to(torch.float32)
        R = _euler_zxy_to_rotation_matrix_t(euler)

        world_xyz = scene["world_xyz"]                       # [N, 3] float32
        local_xyz = (world_xyz - center.unsqueeze(0)) @ R    # [N, 3]

        half_inflated = (dims_tight / 2.0) * (1.0 + float(inflation_frac))
        mask = (local_xyz.abs() <= half_inflated.unsqueeze(0)).all(dim=-1)
        # Materialize the long-index list once so the views below share it.
        keep = mask.nonzero(as_tuple=False).squeeze(-1)

        return {
            # Clones decouple the asset from the mmap'd scene tensor so the
            # consumer can free it independently and so torch ops downstream
            # don't trip over read-only mmap regions.
            "features":        scene["features"].index_select(0, keep).clone(),
            "xyz_offsets":     local_xyz.index_select(0, keep).clone(),
            "frame_indices":   scene["frame_indices"].index_select(0, keep).clone(),
            "n_source_images": int(scene["n_source_images"]),
            "bbox_dims":       dims_tight.clone(),     # TIGHT — invariant
            "obb_euler_zxy":   euler.clone(),
            "label":           obb["label"],
            "source_scene":    scene["source_scene"],
            "target_id":       int(obb["target_id"]),
        }

    def sample(self, label: str, rng: random.Random,
               inflation_frac: float = 0.0) -> dict:
        meta = rng.choice(self.by_label[label])
        scene = self._load(meta["scene_path"])
        obb = scene["obbs"][int(meta["obb_index"])]
        return self._build_asset(scene, obb, float(inflation_frac))

    def sample_random(self, rng: random.Random,
                      inflation_frac: float = 0.0):
        labels = self.labels()
        label = rng.choice(labels)
        return label, self.sample(label, rng, inflation_frac=inflation_frac)
