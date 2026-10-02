"""Lazy-mmap loader for the harvested real-object asset bank.

The bank is a directory of per-object .pt files plus a flat asset_index.json.
Each .pt is a self-contained dict (see scripts/extract_real_object_assets.py
for the canonical format):

    {
        "features":        [K, N_layers, C] bf16,
        "xyz_offsets":     [K, 3] float32,    # OBB-local frame
        "frame_indices":   [K] int16,         # raw source-scene frame index
        "n_source_images": int,
        "bbox_dims":       [3] float32,
        "obb_euler_zxy":   [3] float32,       # debug only
        "label":           str,
        "source_scene":    str,
        "target_id":       int,
    }

Loading uses torch.load(..., mmap=True, weights_only=False) so only the
patches actually accessed are faulted into RAM. Each loaded file holds an
FD until evicted, so an LRU on top of mmap caps the FD count.

Sampling is per-class uniform first, then uniform within the chosen class —
the bank is unbalanced (chair 98, microwave 4) so per-asset uniform would
let chair dominate.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch


class RealObjectAssetBank:
    def __init__(self, root, classes=None, max_loaded: int = 4096):
        self.root = Path(root)
        index_path = self.root / "asset_index.json"
        idx = json.loads(index_path.read_text())
        self.by_label: dict = defaultdict(list)
        for entry in idx:
            if classes is None or entry["label"] in classes:
                self.by_label[entry["label"]].append(entry)
        self._lru: dict = {}
        self._lru_order: list = []
        self.max_loaded = int(max_loaded)

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

    def sample(self, label: str, rng: random.Random,
               inflation_frac: float = 0.0) -> dict:
        # ``inflation_frac`` is a no-op for the legacy per-object bank
        # (patches were extracted with a fixed 5 cm boundary). Accepted so
        # the call sites can be shared with RealObjectSceneBank without
        # branching on bank type.
        del inflation_frac
        meta = rng.choice(self.by_label[label])
        return self._load(meta["path"])

    def sample_random(self, rng: random.Random,
                      inflation_frac: float = 0.0):
        labels = self.labels()
        label = rng.choice(labels)
        return label, self.sample(label, rng, inflation_frac=inflation_frac)
