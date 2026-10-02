import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from reprojection import reproject_scene
from model_adapters.qwen3_vl.adapter import prepare_batch
from model_adapters.qwen3_vl.feature_extraction import extract_features
from utils.io import SuppressOutput, load_geometry_data
# NOTE: debug.visualizations imports open3d at module top; it is imported lazily
# inside the `if debug:` branch below so the normal inference path does not
# require open3d.


def load_onecanvas_model(checkpoint, *, device="cuda", attn_implementation="sdpa"):
    """Load a self-contained OneCanvas checkpoint for inference.

    The geometry embedding is saved separately from the Transformers shards.
    Construct its module from ``resolved_config.json`` before loading
    ``depth_embedding.pt`` so a missing or incompatible geometry state is a
    hard error instead of a silently random geometry channel.
    """
    from transformers import AutoProcessor
    from model_adapters.qwen3_vl.model import Qwen3VL3DForConditionalGeneration
    from model_adapters.qwen3_vl.patches import apply_patches
    from onecanvas.setup_3d import init_3d_embeddings
    from utils.embedding_io import load_3d_embeddings

    checkpoint = Path(checkpoint)
    config_path = checkpoint / "resolved_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"OneCanvas inference requires {config_path}; it defines the geometry module."
        )
    with config_path.open() as f:
        resolved = json.load(f)
    data_config = resolved.get("data")
    if not isinstance(data_config, dict):
        raise RuntimeError(f"{config_path} has no data configuration mapping.")

    model = Qwen3VL3DForConditionalGeneration.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).eval()
    init_3d_embeddings(model, SimpleNamespace(**data_config))
    # Canvas position-id conventions are properties of the trained weights. Carry
    # the checkpoint's recorded values onto the model, where the inference
    # adapter config reads them. A checkpoint that records no frame-index
    # convention was trained with the normalized one.
    inner = model.model.model if hasattr(model.model, "model") else model.model
    inner.rope_pos_range = float(data_config.get("rope_pos_range") or 100.0)
    inner.temporal_max_range = float(data_config.get("temporal_max_range") or 100.0)
    inner.temporal_raw_frame_index = bool(data_config.get("temporal_raw_frame_index", False))
    loaded = load_3d_embeddings(model, str(checkpoint))
    if "depth" not in loaded:
        raise RuntimeError(
            f"Required geometry state did not load from {checkpoint / 'depth_embedding.pt'}."
        )
    # Move after constructing the auxiliary module. Initializing it on an
    # already-moved model leaves its new parameters on CPU and also makes
    # ``model.device`` report CPU to the image-processing path.
    model.to(device)
    apply_patches(model)
    processor = AutoProcessor.from_pretrained(str(checkpoint))
    return model, processor


def _image_pad_token_id(processor):
    """Resolve the ``<|image_pad|>`` token id from the processor tokenizer.

    Same derivation the Qwen3-VL adapter uses to populate
    ``VLMAdapterConfig.image_pad_token_id``; avoids hardcoding the numeric id.
    """
    return processor.tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]


def _build_adapter_config(model, processor):
    """Build the adapter config dict from model attributes.

    Mirrors the config keys produced by data_processor_3d._reprojection_config().
    """
    m = model.model.model if hasattr(model.model, 'model') else model.model
    return {
        "rope_pos_range": getattr(m, "rope_pos_range", 100.0),
        "temporal_max_range": getattr(m, "temporal_max_range", 100.0),
        "temporal_raw_frame_index": getattr(m, "temporal_raw_frame_index", False),
        "depth_embed_mode": getattr(m, "_depth_embed_mode", "off"),
        "depth_embed_min": getattr(m, "_depth_embed_min", 0.3),
        "image_pad_token_id": _image_pad_token_id(processor),
    }


def _pixels_to_canvas_indices(scene, inline_patch_pixels, H_feat, W_feat, image_dims_t):
    """Map ``(frame_idx, px, py)`` source pixels to canvas-token indices.

    A canvas token is one valid reprojected patch; ``scene.valid_indices`` gives
    each token's position in the flat ``N_imgs*H_feat*W_feat`` patch grid, so a
    pixel maps to its feature patch, then to that patch's canvas-token index.
    The exact patch may have been dropped (invalid/zero depth), so this searches
    outward ring by ring for the nearest valid patch. Returns ``list[int]`` of
    indices in ``[0, n_valid)``, the form ``prepare_batch`` wants for
    ``inline_patch_indices``.
    """
    vi = scene.valid_indices
    HW = H_feat * W_feat
    out = []
    for (frame_i, px, py) in inline_patch_pixels:
        W_orig, H_orig = int(image_dims_t[frame_i][0]), int(image_dims_t[frame_i][1])
        fx0 = min(max(int(px * W_feat / W_orig), 0), W_feat - 1)
        fy0 = min(max(int(py * H_feat / H_orig), 0), H_feat - 1)
        canvas_idx = None
        for rad in range(max(H_feat, W_feat)):
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if max(abs(dy), abs(dx)) != rad:
                        continue  # only walk the new ring at this radius
                    yy, xx = fy0 + dy, fx0 + dx
                    if not (0 <= yy < H_feat and 0 <= xx < W_feat):
                        continue
                    flat = frame_i * HW + yy * W_feat + xx
                    m = (vi == flat).nonzero(as_tuple=True)[0]
                    if m.numel() > 0:
                        canvas_idx = int(m[0].item())
                        break
                if canvas_idx is not None:
                    break
            if canvas_idx is not None:
                break
        out.append(canvas_idx if canvas_idx is not None else 0)
    return out


def process_vision_and_generate(
    model,
    processor,
    question,
    image_paths=None,
    features=None,
    poses=None,
    depths=None,
    intrinsics=None,
    image_dims=None,
    max_tokens=250,
    debug=False,
    generate_point_cloud=True,
    da3_model=None,
    inline_patch_pixels=None,
    center_override=None,
    yaw_angle=None,
):
    """One-shot scene VQA: load vision, reproject onto the canvas, generate.

    The full single-scene pipeline behind the README inference quickstart:
    extract per-frame features (or accept precomputed ones), reproject them onto
    the panoramic canvas with ``reproject_scene``, and decode an answer to
    ``question``.

    ``question`` is fed to the model verbatim, exactly as training and the
    benchmark runner feed it. Write it the way the benchmarks do: a VSI-Bench
    multiple-choice question carries its options inline (``"...?\nA. chair\nB.
    table"``), and an SQA3D question starts with the situation (``"I am facing
    the window. What is on my left?"``).

    ``center_override`` (a world point) and ``yaw_angle`` (radians) place the
    canvas origin, as the benchmarks do for situated questions. Left unset, the
    canvas is centred on the mean camera position, the VSI-Bench setting.

    Geometry conventions match ``reproject_scene`` (one contract across the
    public API):

    - ``poses``: camera-to-world extrinsics, ``[N, 4, 4]`` (``[N, 3, 4]`` is
      accepted and squared up). NOT inverted internally.
    - ``depths``: MILLIMETRES (sensor-PNG convention); ``compute_scene_geometry``
      divides by 1000. numpy depth arrays are cast to uint16 to match the
      on-disk sensor-PNG format.
    - ``intrinsics``: ``[N, 4]`` ``(fx, fy, cx, cy)``; ``image_dims``: ``(W, H)``.

    If ``poses``/``depths`` are omitted, geometry is recovered from disk
    (``load_geometry_data``) or predicted by ``da3_model`` when supplied.
    Returns the decoded answer string. Exposed under the friendlier alias
    ``answer_scene_question``.
    """
    # 1. CENTRALIZED IMAGE LOADING
    images = None
    if image_paths is not None and (features is None or poses is None or depths is None or debug):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        if image_dims is None:
            image_dims = [torch.tensor(img.size) for img in images]

    # The question text as training and the benchmarks feed it
    # (SceneQADataset uses item["question"] as the whole user text). An added
    # instruction wrapper is a prompt the fine-tuned model never saw.
    prompt_text = question

    # -------------------------------------------------------------------------
    # GEOMETRY SETUP
    # -------------------------------------------------------------------------
    if poses is not None and depths is not None:
        # Poses are camera-to-world (same convention as reproject_scene and the
        # README); pass them through unchanged, only squaring [3,4] up to [4,4].
        # (Historically this inverted every pose, silently expecting
        # world-to-camera and diverging from the rest of the API.)
        poses_4x4 = []
        for p in poses:
            p_np = np.array(p)
            if p_np.shape == (3, 4):
                tmp = np.eye(4); tmp[:3, :] = p_np; p_np = tmp
            poses_4x4.append(p_np)

        # compute_scene_geometry indexes depths[i].to(device).float() (millimetres,
        # divided by 1000 internally), so depths must be tensors, matching the
        # disk-loading branch. Historically this wrapped them in uint16 PIL Images,
        # which have no .to() and truncated sub-mm precision.
        proj_depths = [
            torch.from_numpy(np.asarray(d).astype(np.float32)) if not torch.is_tensor(d) else d
            for d in depths
        ]
        proj_poses = torch.as_tensor(np.array(poses_4x4), dtype=torch.float32)
    else:
        if da3_model is not None:
            with SuppressOutput():
                proj_depths, proj_poses, intrinsics = _compute_da3_geometry(images, da3_model)
        else:
            proj_depths, proj_poses, intrinsics, _ = load_geometry_data(image_paths)

    # The public disk loader returns PIL depth images, NumPy pose matrices and
    # one shared intrinsics tuple. Normalize them to the tensor contract used
    # by ``reproject_scene`` and repeat shared intrinsics once per frame.
    if not torch.is_tensor(proj_depths):
        proj_depths = [
            d if torch.is_tensor(d) else torch.as_tensor(np.asarray(d).copy())
            for d in proj_depths
        ]
    if not torch.is_tensor(proj_poses):
        proj_poses = torch.as_tensor(np.asarray(proj_poses), dtype=torch.float32)
    if not torch.is_tensor(intrinsics):
        intrinsics = torch.as_tensor(np.asarray(intrinsics), dtype=torch.float32)
    if intrinsics.ndim == 1:
        intrinsics = intrinsics.unsqueeze(0).repeat(len(proj_depths), 1)

    # -------------------------------------------------------------------------
    # FEATURE SETUP
    # -------------------------------------------------------------------------
    if features is not None:
        feature_tensor = features if isinstance(features, torch.Tensor) else torch.stack(
            [torch.stack(img_layers) for img_layers in features]
        )
    else:
        messages = [{"role": "user", "content": [*[{"type": "image", "image": img} for img in images], {"type": "text", "text": prompt_text}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

        with torch.no_grad():
            # Qwen3-VL-specific: unpack HF get_image_features + DeepStack layers,
            # produce model-agnostic [N_imgs, N_layers, H, W, C]. Other adapters
            # will ship their own feature_extraction.extract_features().
            feature_tensor = extract_features(model, inputs.pixel_values, inputs.image_grid_thw)

    # -------------------------------------------------------------------------
    # PROJECTION & GENERATION
    # -------------------------------------------------------------------------
    with torch.no_grad():
        device = model.device

        proj_depths_t = proj_depths.to(device) if isinstance(proj_depths, torch.Tensor) else proj_depths
        image_dims_t = (torch.stack([torch.as_tensor(d) for d in image_dims]) if isinstance(image_dims, (list, tuple))
                        else torch.as_tensor(image_dims)).to(device)

        # Reproject scene using the same path as training
        scene = reproject_scene(
            features=feature_tensor,
            depths=proj_depths_t,
            poses=proj_poses,
            intrinsics=intrinsics,
            image_dims=image_dims_t,
            device=str(device),
            center_override=(None if center_override is None else
                             torch.as_tensor(np.asarray(center_override), dtype=torch.float32)),
            yaw_angle=None if yaw_angle is None else float(yaw_angle),
        )

        if debug:
            from debug.visualizations import run_debug_visualization  # imports open3d
            mock_values = np.full(np.array(images, dtype=np.float32).shape, 0.5)
            run_debug_visualization(mock_values, proj_depths_t.cpu(), proj_poses.cpu(), intrinsics.cpu(),
                                    scene.center_point.cpu(), generate_point_cloud)

        # Dummy image: only exists to make the tokenizer emit an image_pad region,
        # prepare_batch replaces it with the exact projected token count.
        dummy_img = Image.new('RGB', (64 * 32, 64 * 32), (0, 0, 0))
        msg_dummy = [{"role": "user", "content": [{"type": "image", "image": dummy_img}, {"type": "text", "text": prompt_text}]}]
        text_dummy = processor.apply_chat_template(msg_dummy, tokenize=False, add_generation_prompt=True)
        inputs_dummy = processor(text=[text_dummy], images=[dummy_img], return_tensors="pt").to(device)

        target_token = _image_pad_token_id(processor)
        input_ids_dummy = inputs_dummy.input_ids
        inline_patch_indices = None
        inline_patch_positions = None
        if inline_patch_pixels:
            # Inline-patch-marker path: the question text carries one
            # <|object_ref_start|> per requested pixel. Map each pixel to its
            # canvas-token index, then swap the marker tokens to <|image_pad|>
            # so the main canvas region is the first contiguous image_pad run and
            # every marker is an image_pad after a gap (matching the training
            # spatial-pretraining dataset's PATH-A splice).
            _, _, H_feat, W_feat, _ = feature_tensor.shape
            inline_patch_indices = _pixels_to_canvas_indices(
                scene, inline_patch_pixels, H_feat, W_feat, image_dims_t)
            obj_ref_id = processor.tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
            input_ids_dummy = input_ids_dummy.clone()
            input_ids_dummy[0, input_ids_dummy[0] == obj_ref_id] = target_token
            img_pos = (input_ids_dummy[0] == target_token).nonzero(as_tuple=True)[0]
            first_idx = int(img_pos[0].item())
            gaps = ((img_pos[1:] - img_pos[:-1]) > 1).nonzero(as_tuple=True)[0]
            if len(gaps) > 0:
                last_idx = int(img_pos[gaps[0]].item())
                inline_patch_positions = img_pos[gaps[0] + 1:].tolist()
            else:
                last_idx = int(img_pos[-1].item())
                inline_patch_positions = []
        else:
            indices = (input_ids_dummy[0] == target_token).nonzero(as_tuple=True)[0]
            first_idx, last_idx = indices[0].item(), indices[-1].item()

        # Build adapter config and prepare batch (same as training PATH A)
        config = _build_adapter_config(model, processor)
        proj = prepare_batch(
            scene=scene,
            input_ids=input_ids_dummy,
            attention_mask=inputs_dummy.attention_mask,
            labels=torch.full_like(input_ids_dummy, -100),
            first_idx=first_idx,
            last_idx=last_idx,
            config=config,
            inline_patch_positions=inline_patch_positions,
            inline_patch_indices=inline_patch_indices,
        )

        # Pass to model.generate() using PATH A format (matching run_benchmarks.py)
        max_len = proj["input_ids"].shape[0]
        output_ids = model.generate(
            input_ids=proj["input_ids"].unsqueeze(0).to(device),
            attention_mask=proj["attention_mask"].unsqueeze(0).to(device),
            projection_done=True,
            projected_input_ids=[proj["input_ids"].to(device)],
            projected_position_ids=[proj["position_ids"].to(device)],
            projected_attention_mask=[proj["attention_mask"].to(device)],
            projected_labels=None,
            projected_embeds=[proj["embeds"].to(device)],
            projected_aux_layers=[proj["aux_layers"]],
            projected_depth_bins=[proj["depth_bins"].to(device)] if proj["depth_bins"] is not None else None,
            projected_ray_dirs=[proj["ray_dirs"].to(device)] if proj.get("ray_dirs") is not None else None,
            rope_deltas=proj["rope_deltas"].unsqueeze(0).to(device),
            max_new_tokens=max_tokens,
            do_sample=False,
        )

        input_len = proj["input_ids"].shape[0]
        return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


def _compute_da3_geometry(images, da3_model):
    """Compute DA3 metric geometry with predicted poses.

    Returns ``(proj_depths, proj_poses, intrinsics)``. Poses are gravity- and
    yaw-aligned via the canonical transform below so the rendered panorama has
    a consistent up-direction across scenes.
    """
    from geometry import get_scene_center

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with SuppressOutput():
        prediction = da3_model.inference(images)

        # Convert depth to tensors and scale
        depth_tensors = torch.from_numpy(prediction.depth).to(device)
        proj_depths = (depth_tensors * 1000.0).to(torch.uint16)

        # Handle Intrinsics scaling
        model_h, model_w = depth_tensors.shape[1:]
        img_sizes = torch.tensor([img.size for img in images], device=device).float()
        sw = img_sizes[:, 0] / model_w
        sh = img_sizes[:, 1] / model_h

        K = torch.from_numpy(prediction.intrinsics).to(device)
        intrinsics = torch.stack([
            K[:, 0, 0] * sw,
            K[:, 1, 1] * sh,
            K[:, 0, 2] * sw,
            K[:, 1, 2] * sh
        ], dim=1)

        # Pose processing
        extrinsics = torch.from_numpy(prediction.extrinsics).to(device)
        batch_size = extrinsics.shape[0]

        # Create homogeneous coordinates
        bottom = torch.tensor([0, 0, 0, 1], device=device).reshape(1, 1, 4).expand(batch_size, 1, 4)
        raw_poses = torch.inverse(torch.cat([extrinsics, bottom], dim=1))

        # --- Gravity alignment: rotate so camera-down aligns to (0, 0, -1) ---
        avg_down = raw_poses[:, :3, 1].mean(dim=0)
        avg_down = avg_down / torch.norm(avg_down)
        target_down = torch.tensor([0.0, 0.0, -1.0], device=device)

        v = torch.linalg.cross(avg_down, target_down)
        c = torch.dot(avg_down, target_down)
        s = torch.norm(v)

        R_gravity = torch.eye(3, device=device)
        if s > 1e-6:
            kmat = torch.tensor([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], device=device)
            R_gravity = torch.eye(3, device=device) + kmat + (kmat @ kmat) * ((1 - c) / (s**2))

        raw_center = get_scene_center(raw_poses)

        # --- Canonical yaw: rotate mean camera forward to +Y (longitude=0) ---
        # Camera forward = -Z column of pose rotation in world frame
        avg_fwd = -(R_gravity @ raw_poses[:, :3, 2].mean(dim=0))
        # Project onto horizontal plane (zero out vertical = Z component after gravity align)
        avg_fwd[2] = 0.0
        fwd_norm = torch.norm(avg_fwd)
        R_yaw = torch.eye(3, device=device)
        if fwd_norm > 1e-6:
            avg_fwd = avg_fwd / fwd_norm
            # Target forward = +Y (maps to z_c = col-1 = "forward" in projection)
            target_fwd = torch.tensor([0.0, 1.0, 0.0], device=device)
            # Yaw angle around Z axis
            cos_yaw = torch.dot(avg_fwd, target_fwd)
            sin_yaw = avg_fwd[0] * target_fwd[1] - avg_fwd[1] * target_fwd[0]  # cross product Z component
            R_yaw = torch.tensor([
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw,  cos_yaw, 0.0],
                [0.0,      0.0,     1.0]
            ], device=device)

        R_canonical = R_yaw @ R_gravity

        # Apply gravity+yaw alignment to poses so the rendered panorama has a
        # consistent up-direction across scenes (gravity always at the same
        # latitude). Centering keeps the panorama origin at the mean camera
        # position; reproject_scene re-derives the same center via
        # get_scene_center, so the centering here is redundant but cheap.
        proj_poses = torch.eye(4, device=device).repeat(batch_size, 1, 1)
        proj_poses[:, :3, :3] = R_canonical @ raw_poses[:, :3, :3]
        proj_poses[:, :3, 3] = (R_canonical @ (raw_poses[:, :3, 3] - raw_center).T).T

    return proj_depths, proj_poses, intrinsics


# Friendlier public name for the one-shot scene-QA entry point.
answer_scene_question = process_vision_and_generate
