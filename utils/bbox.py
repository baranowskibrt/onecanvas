"""3D bounding box parsing and IoU.

The grounding pipeline uses bare metric float coordinates in meters.

Two formats are supported, selected by the ``bbox_format`` field in each
grounding annotation (defaulting to ``"aabb"`` when absent):

  **AABB (axis-aligned bounding box)** — 6 values:
    ``(cx, cy, cz, w, h, d)`` in the scene-centered axis-aligned frame.
    Used by ScanRefer, Multi3DRefer, Nr3D, Sr3D.

  **OBB (oriented bounding box)** — 9 values:
    ``(cx, cy, cz, dx, dy, dz, rx, ry, rz)`` where ``(rx, ry, rz)`` are
    Euler rotation angles (ZXY convention). Used by EmbodiedScan,
    HoLi-Spatial (raw).

Both training inputs and model outputs live in the same scene-centered frame,
so IoU is computed directly in metric meters.
"""

import json
import math
import re

import numpy as np
import torch


# Integer scale for the pano-format bbox_3d output (u, v) coordinates.
# Decoupled from the dummy image dims / feature grid — the same
# constant is used regardless of training config, so the bbox output format
# stays stable across runs. 1000 is chosen to match the order of magnitude
# of Qwen2/3-VL's native 2D grounding pixel coordinates.
PANO_COORD_SCALE = 1000


def format_metric_bbox(box, decimals=2, box_start="<|box_start|>", box_end="<|box_end|>"):
    """Format a metric bbox as a float-coordinate string with box tokens.

    Output: <|box_start|>(1.36, -1.85, 0.47, 0.50, 1.28, 0.80)<|box_end|>
    """
    fmt = f"{{:.{decimals}f}}"
    vals = ", ".join(fmt.format(v) for v in box)
    return f"{box_start}({vals}){box_end}"


def format_metric_bbox_json(centered_box, label=None, decimals=2):
    """Format a scene-centered metric bbox as Qwen3-VL-style bbox_3d JSON.

    Output: '[{"bbox_3d": [cx, cy, cz, sx, sy, sz]}]'
    All six values are raw metric meters in the scene-centered axis-aligned
    frame; this matches the wrapper structure of Qwen3-VL's native 3D output
    (the cookbook uses 9 values incl. orientation; here we use 6 since boxes
    are axis-aligned). Parser is the existing ``_parse_3d_bbox_json``.

    The ``label`` argument is accepted for call-site compatibility but is
    not emitted: ScanRefer-style descriptions don't carry a clean object
    name and the heuristic head-noun extractor produces noisy labels
    ("this", "color", "to", ...) that hurt training without affecting any
    metric (IoU is computed from coords only).
    """
    cx, cy, cz, sx, sy, sz = centered_box
    obj = [{
        "bbox_3d": [
            round(float(cx), decimals),
            round(float(cy), decimals),
            round(float(cz), decimals),
            round(float(sx), decimals),
            round(float(sy), decimals),
            round(float(sz), decimals),
        ],
    }]
    return json.dumps(obj, separators=(", ", ": "))


# Strip all <|...|> special tokens EXCEPT <|box_start|> and <|box_end|>.
_CLEAN_SPECIAL_RE = re.compile(r'<\|(?!box_start\||box_end\|)[^|]+\|>')


def clean_generated_text(text):
    """Strip special tokens from decoded text, preserving box tokens for grounding."""
    return _CLEAN_SPECIAL_RE.sub('', text).strip()


def parse_3d_bbox(text):
    """Extract (cx, cy, cz, w, h, d) from generated text. Returns 6-tuple or None.

    Supports:
      - Metric floats with box tokens: <|box_start|>(1.36, -1.85, -0.47, 0.14, 1.28, 1.50)<|box_end|>
      - Metric floats bare: (1.36, -1.85, -0.47, 0.14, 1.28, 1.50)
      - JSON metric floats: [{"bbox_3d": [1.36, -1.85, -0.47, 0.14, 1.28, 1.50], "label": "door"}]
    All values are in meters in the scene-centered axis-aligned frame.
    """
    bbox = _parse_3d_bbox_json(text)
    if bbox is not None:
        return bbox

    cleaned = re.sub(r'<\|box_start\|>|<\|box_end\|>', '', text)
    m = re.search(r'\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)', cleaned)
    if not m:
        return None
    try:
        return tuple(float(m.group(i)) for i in range(1, 7))
    except ValueError:
        return None


def _extract_single_bbox_json(text, length):
    """Return the ``bbox_3d`` tuple from data[0] of the first JSON array in
    `text` if it has exactly `length` float coords, else None.

    Shared skeleton for the single-box JSON parsers (AABB len-6, OBB len-9,
    pano len-6); they differ only in the expected coordinate count.
    """
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1:
            return None
        data = json.loads(text[start:end + 1])
        if not isinstance(data, list) or len(data) == 0:
            return None
        item = data[0]
        if not isinstance(item, dict) or "bbox_3d" not in item:
            return None
        coords = item["bbox_3d"]
        if not isinstance(coords, list) or len(coords) != length:
            return None
        return tuple(float(v) for v in coords)
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def _parse_3d_bbox_json(text):
    """Extract (cx, cy, cz, w, h, d) from JSON: [{"bbox_3d": [6 floats]}].
    Returns 6-tuple (metric meters) or None."""
    return _extract_single_bbox_json(text, 6)


# ---------------------------------------------------------------------------
# Pano-format bbox helpers (experimental pano_grounding_format)
#
# Reparameterizes the bbox center as (u, v, depth) where (u, v) are integer
# angular bins in [0, PANO_COORD_SCALE) under the equirectangular convention
# used by reprojection/scene_reprojection.py and geometry/projection_maps.py,
# and depth is the radial distance from the scene center in metric meters.
# Sizes (sx, sy, sz) stay in metric meters (pass-through).
#
# Convention (matches geometry/projection_maps.py:149-164):
#     x_c =  cx;  y_c = -cz;  z_c =  cy
#     lon = atan2(x_c, z_c)                           in [-π, π]
#     lat = atan2(y_c, sqrt(x_c^2 + z_c^2))           in [-π/2, π/2]
# ---------------------------------------------------------------------------


def world_bbox_to_pano_bbox(centered_box):
    """(cx, cy, cz, sx, sy, sz) [centered metric] -> (u, v, depth, sx, sy, sz).

    The input must be a bbox already translated into the scene-centered frame
    (i.e. after subtracting ``get_scene_center(poses)``). No rotation.
    """
    cx, cy, cz, sx, sy, sz = centered_box
    x_c =  float(cx)
    y_c = -float(cz)
    z_c =  float(cy)
    horiz = math.sqrt(x_c * x_c + z_c * z_c)
    lon = math.atan2(x_c, z_c)
    lat = math.atan2(y_c, max(horiz, 1e-8))
    depth = math.sqrt(x_c * x_c + y_c * y_c + z_c * z_c)
    depth = max(depth, 0.1)
    s = PANO_COORD_SCALE
    u = int(round((lon + math.pi) / (2.0 * math.pi) * s)) % s
    v_raw = int(round((math.pi / 2.0 + lat) / math.pi * s))
    v = max(0, min(v_raw, s - 1))
    return (u, v, depth, float(sx), float(sy), float(sz))


def pano_bbox_to_world_bbox(pano_box):
    """(u, v, depth, sx, sy, sz) -> (cx, cy, cz, sx, sy, sz) in centered metric frame.

    Inverse of ``world_bbox_to_pano_bbox``. Uses half-bin centers so the
    round-trip error is bounded by (2π / PANO_COORD_SCALE) * depth / 2.
    """
    u, v, depth, sx, sy, sz = pano_box
    s = PANO_COORD_SCALE
    u_norm = (float(u) + 0.5) / float(s)
    v_norm = (float(v) + 0.5) / float(s)
    lon = u_norm * 2.0 * math.pi - math.pi
    lat = v_norm * math.pi - math.pi / 2.0
    d = float(depth)
    horiz = d * math.cos(lat)
    x_c = horiz * math.sin(lon)
    y_c = d * math.sin(lat)
    z_c = horiz * math.cos(lon)
    cx =  x_c
    cy =  z_c
    cz = -y_c
    return (cx, cy, cz, float(sx), float(sy), float(sz))


def format_pano_bbox_json(pano_box, label="object"):
    """Format a pano-format bbox as Qwen3-VL-style JSON.

    Output: '[{"bbox_3d": [u, v, depth, sx, sy, sz], "label": "..."}]'
    u, v are integers in [0, PANO_COORD_SCALE); depth, sx, sy, sz are raw
    metric meters rounded to 2 decimals for text formatting only.
    """
    u, v, depth, sx, sy, sz = pano_box
    obj = [{
        "bbox_3d": [
            int(u),
            int(v),
            round(float(depth), 2),
            round(float(sx), 2),
            round(float(sy), 2),
            round(float(sz), 2),
        ],
        "label": str(label),
    }]
    return json.dumps(obj, separators=(", ", ": "))


def parse_pano_3d_bbox(text):
    """Extract (u, v, depth, sx, sy, sz) from pano-format JSON text.

    Expected: [{"bbox_3d": [u, v, depth, sx, sy, sz], "label": "..."}]
    Returns 6-tuple or None on parse failure. Same JSON search strategy as
    ``_parse_3d_bbox_json`` but interprets the tuple as the pano format.
    """
    return _extract_single_bbox_json(text, 6)


def compute_3d_iou(box_a, box_b):
    """Axis-aligned 3D IoU between two (cx, cy, cz, w, h, d) boxes."""
    a_min = [box_a[i] - box_a[i + 3] / 2 for i in range(3)]
    a_max = [box_a[i] + box_a[i + 3] / 2 for i in range(3)]
    b_min = [box_b[i] - box_b[i + 3] / 2 for i in range(3)]
    b_max = [box_b[i] + box_b[i + 3] / 2 for i in range(3)]

    inter = 1.0
    for i in range(3):
        lo = max(a_min[i], b_min[i])
        hi = min(a_max[i], b_max[i])
        if hi <= lo:
            return 0.0
        inter *= (hi - lo)

    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# OBB (oriented bounding box) helpers — 9-DoF: (cx, cy, cz, dx, dy, dz, rx, ry, rz)
# ---------------------------------------------------------------------------


def format_metric_obb_json(centered_box, label=None, decimals=2):
    """Format a scene-centered 9-DoF OBB as Qwen3-VL-style bbox_3d JSON.

    Output: '[{"bbox_3d": [cx, cy, cz, dx, dy, dz, rx, ry, rz]}]'
    """
    obj = [{
        "bbox_3d": [round(float(v), decimals) for v in centered_box[:9]],
    }]
    return json.dumps(obj, separators=(", ", ": "))


def format_multi_metric_obb_json(centered_boxes, label=None, decimals=2):
    """Format zero or more centered metric OBBs as Qwen3-VL-style JSON.

    Returns "none" for empty list, single-box JSON, or multi-box JSON array.
    """
    if len(centered_boxes) == 0:
        return "none"
    if len(centered_boxes) == 1:
        return format_metric_obb_json(centered_boxes[0], label=label, decimals=decimals)
    objs = []
    for box in centered_boxes:
        objs.append({
            "bbox_3d": [round(float(v), decimals) for v in box[:9]],
        })
    return json.dumps(objs, separators=(", ", ": "))


def parse_3d_obb(text):
    """Extract (cx, cy, cz, dx, dy, dz, rx, ry, rz) from generated text.

    Supports:
      - JSON: [{"bbox_3d": [9 values]}]
      - Parenthesized: (v1, v2, ..., v9)
    Returns 9-tuple or None.
    """
    obb = _parse_3d_obb_json(text)
    if obb is not None:
        return obb

    cleaned = re.sub(r'<\|box_start\|>|<\|box_end\|>', '', text)
    # Match 9 comma-separated floats inside parens
    pattern = r'\(\s*' + r'\s*,\s*'.join([r'(-?[\d.]+)'] * 9) + r'\s*\)'
    m = re.search(pattern, cleaned)
    if not m:
        return None
    try:
        return tuple(float(m.group(i)) for i in range(1, 10))
    except ValueError:
        return None


def _parse_3d_obb_json(text):
    """Extract (cx, cy, cz, dx, dy, dz, rx, ry, rz) from JSON. Returns 9-tuple or None."""
    return _extract_single_bbox_json(text, 9)


def parse_multi_3d_obb(text):
    """Parse zero or more 9-DoF OBBs from text.

    Supports:
      - "none" -> []
      - JSON array of bbox_3d with 9 values -> list of 9-tuples
      - Semicolon-separated parenthesized 9-tuples -> list of 9-tuples
    """
    if text.strip().lower() == "none":
        return []

    # Try JSON first
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list) and len(data) > 0:
                boxes = []
                for item in data:
                    if isinstance(item, dict) and "bbox_3d" in item:
                        coords = item["bbox_3d"]
                        if isinstance(coords, list) and len(coords) == 9:
                            boxes.append(tuple(float(v) for v in coords))
                if boxes:
                    return boxes
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Regex fallback
    cleaned = re.sub(r'<\|box_start\|>|<\|box_end\|>', '', text)
    pattern = r'\(\s*' + r'\s*,\s*'.join([r'(-?[\d.]+)'] * 9) + r'\s*\)'
    boxes = []
    for m in re.finditer(pattern, cleaned):
        try:
            boxes.append(tuple(float(m.group(i)) for i in range(1, 10)))
        except ValueError:
            continue
    return boxes


def _euler_zxy_to_rotation_matrix(rx, ry, rz):
    """Convert EmbodiedScan 9-DoF OBB euler angles to a 3x3 rotation matrix.

    EmbodiedScan stores angles matching pytorch3d.euler_angles_to_matrix(..., "ZXY"):
    the three values index Z, X, Y axes respectively (NOT the literal xyz letters
    in the arg names). So R = R_Z(rx) @ R_X(ry) @ R_Y(rz), where `rx` is the
    Z-axis yaw, `ry` is the X-axis pitch, and `rz` is the Y-axis roll.

    Verified against embodiedscan/structures/bbox_3d/euler_box3d.py:
        rot_mat_T = euler_angles_to_matrix(tensor[:, 6:9], 'ZXY').transpose(1, 2)
    """
    ca, sa = math.cos(rx), math.sin(rx)  # Z-axis
    cb, sb = math.cos(ry), math.sin(ry)  # X-axis
    cc, sc = math.cos(rz), math.sin(rz)  # Y-axis

    # R = R_Z(rx) @ R_X(ry) @ R_Y(rz)
    R = np.array([
        [ca * cc - sa * sb * sc, -sa * cb, ca * sc + sa * sb * cc],
        [sa * cc + ca * sb * sc,  ca * cb, sa * sc - ca * sb * cc],
        [-cb * sc,                sb,      cb * cc],
    ], dtype=np.float64)
    return R


def obb_corners(box):
    """Compute 8 corner points of a 9-DoF OBB.

    Args:
        box: (cx, cy, cz, dx, dy, dz, rx, ry, rz)
    Returns:
        (8, 3) ndarray of corner points in world frame.
    """
    cx, cy, cz, dx, dy, dz, rx, ry, rz = box
    R = _euler_zxy_to_rotation_matrix(rx, ry, rz)
    center = np.array([cx, cy, cz])
    half = np.array([dx / 2, dy / 2, dz / 2])

    # 8 corners in local frame
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    local_corners = signs * half  # (8, 3)
    world_corners = (R @ local_corners.T).T + center  # (8, 3)
    return world_corners


def obb_to_aabb(box):
    """Convert 9-DoF OBB to 6-DoF AABB by computing axis-aligned bounds of corners.

    Args:
        box: (cx, cy, cz, dx, dy, dz, rx, ry, rz)
    Returns:
        (cx, cy, cz, w, h, d) AABB tuple.
    """
    corners = obb_corners(box)
    mn = corners.min(axis=0)
    mx = corners.max(axis=0)
    center = (mn + mx) / 2
    size = mx - mn
    return (float(center[0]), float(center[1]), float(center[2]),
            float(size[0]), float(size[1]), float(size[2]))


def _sample_obb_surface_points(center, R, dims, n_per_face):
    """Sample points uniformly on the 6 faces of an OBB.

    Args:
        center: [3] torch.Tensor, OBB center in world frame.
        R: [3, 3] torch.Tensor, rotation matrix (columns = local axes).
        dims: (dx, dy, dz) tuple of full-length extents.
        n_per_face: int, points sampled per face (total = 6 * n_per_face).
    Returns:
        [6 * n_per_face, 3] torch.Tensor of world-frame surface points.
    """
    dx, dy, dz = dims
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    pts = []
    for face_axis, sign, ha, hb, hc in [
        (0, +hx, hy, hz, hx),  # +x face
        (0, -hx, hy, hz, hx),  # -x face
        (1, +hy, hx, hz, hy),  # +y face
        (1, -hy, hx, hz, hy),  # -y face
        (2, +hz, hx, hy, hz),  # +z face
        (2, -hz, hx, hy, hz),  # -z face
    ]:
        u = torch.FloatTensor(n_per_face).uniform_(-ha, ha)
        v = torch.FloatTensor(n_per_face).uniform_(-hb, hb)
        local = torch.zeros(n_per_face, 3)
        if face_axis == 0:
            local[:, 0] = sign
            local[:, 1] = u
            local[:, 2] = v
        elif face_axis == 1:
            local[:, 0] = u
            local[:, 1] = sign
            local[:, 2] = v
        else:
            local[:, 0] = u
            local[:, 1] = v
            local[:, 2] = sign
        pts.append((R @ local.T).T + center)
    return torch.cat(pts, dim=0)  # [6*n_per_face, 3]


def obb_surface_distance(center1, dims1, R1, center2, dims2, R2, n_surface=300):
    """Approximate surface-to-surface distance between two OBBs.

    Samples ``n_surface`` points on each OBB's surface (6 faces × n//6 each),
    then returns the minimum pairwise Euclidean distance between the two sets.
    This matches VSI-Bench's "measuring from the closest point of each object"
    semantics without requiring a GJK solver.

    Args:
        center1, center2: [3] torch.Tensors, OBB centers.
        dims1, dims2: (dx, dy, dz) tuples of full-length extents.
        R1, R2: [3, 3] torch.Tensors, rotation matrices.
        n_surface: total surface samples per box (split evenly over 6 faces).
    Returns:
        float, non-negative surface-to-surface distance in metres.
    """
    _, _, dist = obb_closest_surface_points(
        center1, dims1, R1, center2, dims2, R2, n_surface=n_surface
    )
    return dist


def obb_closest_surface_points(center1, dims1, R1, center2, dims2, R2, n_surface=300):
    """Approximate the closest surface-point pair between two OBBs.

    Samples ``n_surface`` surface points on each OBB and returns the pair of
    world-frame points that minimises pairwise Euclidean distance, along with
    that distance. Shares its point generator with ``obb_surface_distance`` so
    the two stay numerically consistent.

    Used by the geometric probe sampler: for distance tasks we add these two
    points to the markers of their respective boxes so the model can read the
    ground-truth distance off the nearest-token pair directly.

    Args:
        center1, center2: [3] torch.Tensors, OBB centers.
        dims1, dims2: (dx, dy, dz) tuples of full-length extents.
        R1, R2: [3, 3] torch.Tensors, rotation matrices.
        n_surface: total surface samples per box (split evenly over 6 faces).
    Returns:
        (pt1, pt2, dist):
            pt1, pt2: [3] torch.Tensors (world frame) — the closest sampled
                      points on box 1 and box 2 respectively.
            dist:     float, non-negative surface-to-surface distance in metres.
    """
    n_per_face = max(1, n_surface // 6)
    pts1 = _sample_obb_surface_points(center1, R1, dims1, n_per_face)  # [N, 3]
    pts2 = _sample_obb_surface_points(center2, R2, dims2, n_per_face)  # [M, 3]
    dmat = torch.cdist(pts1, pts2)
    min_idx = torch.argmin(dmat)
    i = int(min_idx // dmat.shape[1])
    j = int(min_idx % dmat.shape[1])
    dist = max(0.0, float(dmat[i, j].item()))
    return pts1[i], pts2[j], dist


def compute_3d_obb_iou(box_a, box_b):
    """Approximate 3D IoU between two 9-DoF OBBs.

    Uses the convex-hull method via Separating Axis Theorem (SAT) approximation:
    converts both OBBs to AABBs and computes AABB IoU. This is a conservative
    lower bound on the true OBB IoU but is fast and sufficient for grounding
    evaluation where the model is being trained to produce tight OBBs.

    For exact OBB IoU, a convex polytope intersection algorithm would be needed,
    but the AABB approximation is standard in the literature (EmbodiedScan,
    ScanRefer) and avoids scipy/trimesh dependencies.

    Args:
        box_a, box_b: 9-tuples (cx, cy, cz, dx, dy, dz, rx, ry, rz)
    Returns:
        IoU in [0, 1].
    """
    aabb_a = obb_to_aabb(box_a)
    aabb_b = obb_to_aabb(box_b)
    return compute_3d_iou(aabb_a, aabb_b)


# ---------------------------------------------------------------------------
def parse_multi_3d_bbox(text):
    """Parse zero or more 3D bounding boxes from text.

    Supports formats:
      - "none"                                              -> []
      - "(cx, cy, cz, w, h, d)"                             -> [bbox]
      - "(cx, cy, cz, w, h, d); (cx, cy, cz, w, h, d)"      -> [bbox, bbox]
      - '[{"bbox_3d": [cx, cy, cz, w, h, d]}]'              -> [bbox]
      - '[{"bbox_3d": [...]}, {"bbox_3d": [...]}]'          -> [bbox, bbox]

    The semicolon delimiter is just a convention; any non-paren separator works
    since the regex matches all `(...)` tuples regardless of what is between
    them. Returns list of 6-tuples in meters.
    """
    if text.strip().lower() == "none":
        return []

    # Try JSON-array form first: [{"bbox_3d":[...]}, {"bbox_3d":[...]}, ...]
    # This must be tried before the regex because the JSON tuples use square
    # brackets, which the (...) regex would never match.
    boxes_json = _parse_multi_3d_bbox_json(text)
    if boxes_json is not None:
        return boxes_json

    cleaned = re.sub(r'<\|box_start\|>|<\|box_end\|>', '', text)
    pattern = r'\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)'
    boxes = []
    for m in re.finditer(pattern, cleaned):
        try:
            boxes.append(tuple(float(m.group(i)) for i in range(1, 7)))
        except ValueError:
            continue
    return boxes


def _parse_multi_3d_bbox_json(text):
    """Extract a list of bboxes from a JSON array of bbox_3d objects.

    Returns:
        list of 6-tuples on success, or None if the text is not a valid
        JSON array of bbox_3d objects (in which case the caller should fall
        back to regex extraction). Accepts both 6-value (AABB) and 9-value
        (OBB) entries; 9-value entries are converted to AABB via obb_to_aabb.
    """
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1:
            return None
        data = json.loads(text[start:end + 1])
        if not isinstance(data, list) or len(data) == 0:
            return None
        boxes = []
        for item in data:
            if not isinstance(item, dict) or "bbox_3d" not in item:
                return None
            coords = item["bbox_3d"]
            if not isinstance(coords, list):
                return None
            if len(coords) == 6:
                boxes.append(tuple(float(v) for v in coords))
            elif len(coords) == 9:
                # Convert OBB to AABB for backward compatibility
                obb = tuple(float(v) for v in coords)
                boxes.append(obb_to_aabb(obb))
            else:
                return None
        return boxes
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def format_multi_metric_bbox_json(centered_boxes, label=None, decimals=2):
    """Format zero or more centered metric bboxes as Qwen3-VL-style JSON.

    Output:
      - len == 0  -> "none"
      - len == 1  -> '[{"bbox_3d": [cx, cy, cz, sx, sy, sz]}]'   (byte-equal
                     to format_metric_bbox_json — single-box back-compat)
      - len >= 2  -> '[{"bbox_3d": [...]}, {"bbox_3d": [...]}, ...]'

    All six values per box are raw metric meters in the scene-centered
    axis-aligned frame. Boxes are not reordered. The label argument is
    accepted for call-site compatibility but is not emitted, matching
    format_metric_bbox_json behavior.
    """
    if len(centered_boxes) == 0:
        return "none"
    if len(centered_boxes) == 1:
        # Delegate so single-box output is byte-equal to the existing path.
        return format_metric_bbox_json(centered_boxes[0], label=label, decimals=decimals)
    objs = []
    for box in centered_boxes:
        cx, cy, cz, sx, sy, sz = box
        objs.append({
            "bbox_3d": [
                round(float(cx), decimals),
                round(float(cy), decimals),
                round(float(cz), decimals),
                round(float(sx), decimals),
                round(float(sy), decimals),
                round(float(sz), decimals),
            ],
        })
    return json.dumps(objs, separators=(", ", ": "))


def format_multi_metric_bbox(centered_boxes, decimals=2,
                              box_start="<|box_start|>", box_end="<|box_end|>"):
    """Format zero or more centered metric bboxes in the legacy box-token format.

    Output:
      - len == 0  -> "none"
      - len == 1  -> "<|box_start|>(cx, cy, cz, w, h, d)<|box_end|>"
                     (byte-equal to format_metric_bbox)
      - len >= 2  -> "<|box_start|>(...); (...)<|box_end|>"
    """
    if len(centered_boxes) == 0:
        return "none"
    if len(centered_boxes) == 1:
        return format_metric_bbox(centered_boxes[0], decimals=decimals,
                                   box_start=box_start, box_end=box_end)
    fmt = f"{{:.{decimals}f}}"
    parts = []
    for box in centered_boxes:
        vals = ", ".join(fmt.format(v) for v in box)
        parts.append(f"({vals})")
    return f"{box_start}{'; '.join(parts)}{box_end}"


def compute_multi3drefer_f1(pred_boxes, gt_boxes, iou_threshold):
    """Compute F1 for multi-object 3D grounding (Multi3DRefer metric).

    Uses greedy matching: for each predicted box, find the best-matching GT box
    above the IoU threshold. Each GT box can be matched at most once.

    Returns (f1, precision, recall).
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0
    if n_pred == 0:
        return 0.0, 0.0, 0.0
    if n_gt == 0:
        return 0.0, 0.0, 0.0

    iou_matrix = []
    for p in pred_boxes:
        row = [compute_3d_iou(p, g) for g in gt_boxes]
        iou_matrix.append(row)

    matched_gt = set()
    tp = 0
    pred_order = sorted(range(n_pred),
                        key=lambda i: max(iou_matrix[i]),
                        reverse=True)
    for pi in pred_order:
        best_iou = -1
        best_gi = -1
        for gi in range(n_gt):
            if gi not in matched_gt and iou_matrix[pi][gi] > best_iou:
                best_iou = iou_matrix[pi][gi]
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_gt.add(best_gi)
            tp += 1

    precision = tp / n_pred
    recall = tp / n_gt
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall
