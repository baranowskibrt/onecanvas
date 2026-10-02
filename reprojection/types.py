"""Data types for the reprojection module."""

from dataclasses import dataclass

import torch


@dataclass
class SceneGeometry:
    """Model-agnostic geometric subset of a reprojected scene.

    Returned by ``compute_scene_geometry`` when only the geometric layout of
    valid patches is needed (e.g. by datasets that pick patches at sample
    time but defer feature extraction to the live visual encoder in
    ``model.forward()``). ``valid_indices`` is the global index of each valid
    patch into the flat ``N_imgs * H_feat * W_feat`` grid; ``reproject_scene``
    uses it to gather features without recomputing the validity mask.
    """

    longitude: torch.Tensor      # [N_valid] -- radians, [-pi, pi]
    latitude: torch.Tensor       # [N_valid] -- radians, [-pi/2, pi/2]
    depth: torch.Tensor          # [N_valid] -- metric depth in meters
    frame_index: torch.Tensor    # [N_valid] -- which source image (0..N_imgs-1)
    n_valid: int
    n_images: int
    center_point: torch.Tensor   # [3] -- scene center used for projection
    valid_indices: torch.Tensor  # [N_valid] -- global indices into the flat grid
    poses: torch.Tensor = None   # [N_imgs, 4, 4] -- camera-to-world
    intrinsics: torch.Tensor = None  # [N_imgs, 4] -- fx, fy, cx, cy (for exact-visibility probes)
    image_dims: torch.Tensor = None  # [N_imgs, 2] -- W, H (for exact-visibility probes)
    yaw_angle: float = None      # yaw augmentation angle (radians) applied to panoramic coords


@dataclass
class ReprojectedScene:
    """Model-agnostic output of scene reprojection.

    Contains flat arrays of valid tokens with their 3D coordinates.
    Position scaling and VLM-specific formatting happen downstream in
    the model adapter, not here.
    """

    embeds: torch.Tensor        # [N_valid, C] -- layer-0 features
    aux_layers: list            # list of [N_valid, C] tensors, one per extra ViT layer.
                                # Model-agnostic: Qwen3-VL feeds these as deepstack_visual_embeds
                                # at multiple LM layers (its native DeepStack architecture); other
                                # adapters can use only layer 0 (single-injection) or ignore.
    longitude: torch.Tensor     # [N_valid] -- radians, [-pi, pi]
    latitude: torch.Tensor      # [N_valid] -- radians, [-pi/2, pi/2]
    depth: torch.Tensor         # [N_valid] -- metric depth in meters
    frame_index: torch.Tensor   # [N_valid] -- which source image (0..N_imgs-1)
    n_valid: int
    n_images: int
    center_point: torch.Tensor  # [3] -- scene center used for projection
    poses: torch.Tensor = None  # [N_imgs, 4, 4] -- camera-to-world
    intrinsics: torch.Tensor = None  # [N_imgs, 4] -- fx, fy, cx, cy
    image_dims: torch.Tensor = None  # [N_imgs, 2] -- W, H (for exact-visibility probes)
    yaw_angle: float = None     # yaw augmentation angle (radians) applied to panoramic coords
    valid_indices: torch.Tensor = None  # [N_valid] -- global indices into the flat
                                # N_imgs*H_feat*W_feat patch grid, canvas-token order.
                                # Lets callers map a source pixel/patch to its canvas
                                # token index (inline-patch-marker placement).
