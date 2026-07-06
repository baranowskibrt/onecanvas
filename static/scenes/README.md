# Interactive demo scenes

Each `<scene_id>/` folder feeds the interactive panoramic-reprojection demo on
the website (the three.js viewer in `static/js/onecanvas_demo.js`). A folder
holds:

- `scene.json` — agent center, sphere radius, the lifted patch-feature
  endpoints + per-camera colors, and the situated question poses.
- `cloud.bin` — the RGB scene point cloud (binary: `uint32` magic, `uint32` N,
  `float32[3N]` xyz in the scene's axis-aligned WORLD frame Z-up, `uint8[3N]`
  rgb).

`index.json` lists which scenes the viewer offers (the scene selector is
hidden when there is only one).

## Adding any scene

The viewer is data-driven, so adding a scene is just generating its assets:

```bash
# Needs the preprocessed ScanNet scene + cached qwen3-vl features + GT poses
# (same inputs as scripts/make_lifting_panorama_ply.py). CPU only.
python scripts/export_web_scene.py <scene_id> [<scene_id> ...]
```

This computes the 32-frame RGB cloud, the 5-camera lifted feature endpoints
(70 patches each, colored by source camera), the agent center + heading, and
writes `static/scenes/<scene_id>/{scene.json,cloud.bin}`, then registers the
scene in `index.json`.

Situated questions: poses are baked per scene in `SCENE_QUESTIONS` in
`scripts/export_web_scene.py` (currently the three SQA3D poses for
`scene0030_00`, matching the results video). A scene with no entry falls back
to a single "scene center" pose. To add questions for a new scene, add its
`(x, y, yaw_deg, question, answer)` tuples there (poses are in the same
axis-aligned world frame; `yaw` follows `heading = (cos yaw, sin yaw)`).

## Previewing locally

The viewer fetches assets and loads three.js as ES modules, so it must be
served over HTTP (not opened as a `file://` path):

```bash
cd resources/onecanvas_website && python3 -m http.server 8000
# then open http://localhost:8000/
```
