# Bad-scene handling

Audit date: 2026-04-23. Machine-readable drop list: [bad_scenes.json](bad_scenes.json). Per-dataset raw stats: `audit_*.json`. Rerunnable via `python scripts/audit_poses.py <dataset>`.

## Summary

Audited GT camera poses across all four datasets (8 678 scenes total):

| Dataset | OK | Drift (pose tracker diverged) | Offset-only (render fine) | Missing data |
|---|---|---|---|---|
| ScanNet | 1540 / 1633 | 0 | 0 | 93 (test split, no GT released) |
| ScanNet++ iPhone | 947 / 1018 | **7** | 64 | 0 |
| ScanNet++ DSLR | 956 / 956 | 0 | 0 | 0 |
| ARKitScenes | 5040 / 5071 | 0 | 0 | 25 (24 empty stub dirs + 1 partial download) |

"Drift" = pose translation span on any axis exceeds 30 m (indoor scenes span less than 15 m). These are cases where ARKit / BundleFusion lost tracking during capture; translations blow up to kilometres or tens of kilometres over the sequence. "Offset-only" means large `|t|` but small span — the scene is in a weird world frame but internally consistent, so `get_scene_center` subtracts the mean pose translation and the pipeline renders correctly. Verified end-to-end on nine of them.

## DA3 fallback policy (active)

The seven drift scenes in ScanNet++ iPhone have broken GT poses but **DA3-predicted poses that are internally consistent** (max `|t|` 2–5 m, sensible spans). Six of the seven have a precomputed `da3_geometry_balanced_256_metric.pt` on disk; the seventh has no iPhone RGB extracted at all.

**Policy:** the six rescuable scenes force the DA3-predicted pose path even when training passes `--use_gt_all True`. Implementation: `DA3_FALLBACK_SCENES` in [training/onecanvas/data/data_processor_3d.py](../training/onecanvas/data/data_processor_3d.py), applied per-scene inside `_load_scene_data`. The training log prints one `[data] scene <id> is in DA3_FALLBACK_SCENES ...` line the first time each scene is sampled.

| Scene | Action | Why |
|---|---|---|
| 7dab70c8c8 | DA3 fallback | ARKit span 189 km; DA3 .pt present |
| d755b3d9d8 | DA3 fallback | ARKit span 72 km; DA3 .pt present |
| 120acffd90 | DA3 fallback | ARKit span 680 m; DA3 .pt present |
| cc0aa81452 | DA3 fallback | ARKit span 332 m; DA3 .pt present |
| 46001f434d | DA3 fallback | ARKit span 203 m; DA3 .pt present |
| ab4f373966 | DA3 fallback | ARKit span 158 m; DA3 .pt present |
| 02a980c994 | drop (not rescuable here) | ARKit drift AND no iPhone RGB AND no DA3 .pt. Not referenced in any training annotation, so no dataloader filter needed. |

Two further ScanNet++ iPhone scenes (`651dc6b4f1`, `d4d2019f5d`) have fine GT poses but also no iPhone RGB extracted and no DA3 .pt. Also not in any annotation; also safe to ignore.

## Paper framing

Competing methods on SQA3D / VSI-Bench / SPBench (SpaceMind, VLM-3R, Video-3D LLM, etc.) take video / point-cloud / image-only input and do not rely on ScanNet++ iPhone ARKit GT poses at all. Falling back to DA3 on the six flawed scenes is therefore a neutral design choice — not a concession. It brings us in line with the data-assumption baseline of the main competitors rather than giving us an advantage.

## ARKitScenes cleanup (optional)

- 24 empty scene directories for scenes *not in our download manifest* (`scenes_needed_with_fold.csv`): safe to `rm -rf`. List in [bad_scenes.json](bad_scenes.json).
- 1 partial download (`Training/47331480`) has everything except `lowres_wide.traj`; re-pull the traj file to use this scene. Currently dropped by the loader because poses cannot be parsed.
