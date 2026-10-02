"""Geometry subpackage: 3D lifting, projection maps, visibility."""

from .geometry_lifting import (
    lift_to_3d_with_intrinsics,
    get_scene_center,
    compute_scene_aabb_from_depths,
)
from .projection_maps import (
    compute_equirectangular_mapping,
    world_to_spherical,
)
from .visibility import (
    project_world_to_camera,
    in_frustum,
    segment_aabb_intersect,
    is_visible,
    visibility_matrix,
    first_visible_frame,
)

__all__ = [
    "lift_to_3d_with_intrinsics",
    "get_scene_center",
    "compute_scene_aabb_from_depths",
    "compute_equirectangular_mapping",
    "world_to_spherical",
    "project_world_to_camera",
    "in_frustum",
    "segment_aabb_intersect",
    "is_visible",
    "visibility_matrix",
    "first_visible_frame",
]
