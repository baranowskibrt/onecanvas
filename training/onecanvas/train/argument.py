from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import transformers

from onecanvas.data.curriculum_task_mix import (
    CURRICULA, CURRENT_OBB_ONLY, CURRENT_TASKS, real_object_tasks,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-VL-8B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "Attention backend passed to from_pretrained: "
                          "'sdpa' (default), 'flash_attention_2', or 'eager'."},
    )
    vanilla_qwen3vl: bool = field(
        default=False,
        metadata={"help": "Baseline mode: load stock Qwen3VLForConditionalGeneration "
                          "and feed raw multi-image inputs directly. Skips all 3D "
                          "machinery (panoramic canvas, depth/angle/marker embeddings, "
                          "3D MRoPE, spatial pretraining). Use with vanilla run script."},
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    strict_data_loading: bool = field(
        default=False,
        metadata={"help": "Crash instead of skipping when scene assets (geometry, images) are unavailable. "
                          "Useful for catching NFS issues or missing preprocessing."}
    )
    max_pixels: int = field(default=28 * 28 * 576)
    with_precomputed_geometry: bool = field(default=True)
    border_crop_ratio: float = field(default=0.03)
    model_type: str = field(default="qwen3vl_3d")
    dataset_offset: int = field(default=0)
    val_sample_num: int = field(default=200)
    num_images: int = field(default=32)
    temporal_max_range: float = field(
        default=100.0,
        metadata={"help": "Maximum value for the normalized temporal (T) position. The mean "
                  "source-frame index is mapped from [0, num_images-1] to "
                  "[0, temporal_max_range]. 100 matches Qwen's typical position range."},
    )
    temporal_raw_frame_index: bool = field(
        default=True,
        metadata={"help": "T is the source-frame index itself, no division by the frame "
                  "count and no rescaling, so adding a frame to a canvas appends a new T "
                  "instead of moving every existing token's T. temporal_max_range is "
                  "ignored when this is on. This is Qwen3-VL's own video convention "
                  "(frame index on T); the [0, temporal_max_range] stretch is not."},
    )
    rope_pos_range: float = field(
        default=100.0,
        metadata={"help": "Scale of the angular MRoPE position IDs: longitude and "
                  "latitude are mapped into [0, rope_pos_range] (latitude carries "
                  "half scale, see _scale_positions in the qwen3_vl adapter). "
                  "0 is treated as 100 for backward compatibility with "
                  "resolved_config.json files written when the default was 0."},
    )
    projection_mode: str = field(
        default="equirectangular",
        metadata={"help": "Panoramic projection type. Only 'equirectangular' (360x180 "
                  "panorama) is supported. The reprojection path is "
                  "equirectangular-only."},
    )
    panoramic_augment_center: bool = field(
        default=False,
        metadata={"help": "Training-only augmentation: randomly pick one camera position "
                  "as the panoramic center instead of the mean of all cameras. "
                  "Changes the viewpoint each sample to reduce overfitting to fixed "
                  "panoramic layouts. Disabled for grounding samples (bbox alignment)."},
    )
    panoramic_augment_center_sigma: float = field(
        default=0.0,
        metadata={"help": "When > 0, sample the panoramic center from a Gaussian centered "
                  "on the camera mean with std = sigma * scene_radius (max camera "
                  "distance from mean). Overrides panoramic_augment_center. "
                  "sigma=1 means ~68%% of centers within the camera hull."},
    )
    panoramic_augment_center_uniform: bool = field(
        default=False,
        metadata={"help": "Training-only augmentation: sample the panoramic center "
                  "uniformly in the XYZ axis-aligned bbox of the camera translations. "
                  "Overrides panoramic_augment_center and panoramic_augment_center_sigma. "
                  "Reprojection has no z-buffering, so centers inside furniture / walls "
                  "are valid and encourage view-invariant spatial reasoning."},
    )
    panoramic_augment_center_uniform_scene: bool = field(
        default=False,
        metadata={"help": "Training-only augmentation: sample the panoramic center "
                  "uniformly in the AABB of unprojected depth points (full scene "
                  "extent, typically 1.5-2x larger than camera-translation AABB). "
                  "Overrides all other panoramic_augment_center_* flags."},
    )
    panoramic_augment_center_inflate: float = field(
        default=1.0,
        metadata={"help": "Inflation factor applied to the panorama-center sampling AABB "
                  "around its centroid. 1.0 = no change, 2.0 = 2x extent per axis. "
                  "Applies to both panoramic_augment_center_uniform and _uniform_scene."},
    )
    panoramic_augment_yaw: bool = field(
        default=False,
        metadata={"help": "Training-only augmentation: rotate the panoramic forward "
                  "direction by a random angle in [-pi, pi]. Changes which world "
                  "direction maps to longitude=0 each sample. Disabled for grounding "
                  "samples."},
    )
    sqa3d_use_agent_pose: bool = field(
        default=False,
        metadata={"help": "For SQA3D: place the panoramic center at the agent's "
                  "situation position and orient panorama forward along the agent's "
                  "situation rotation. Pulls position/rotation from the SQA3D "
                  "annotation JSON. Applies in both train and eval."},
    )
    sqa3d_agent_yaw_offset: float = field(
        default=0.0,
        metadata={"help": "Additional yaw offset (radians) applied on top of the "
                  "SQA3D agent rotation. Use to compensate for axis-convention "
                  "mismatches between SQA3D's agent-forward and this pipeline's "
                  "panorama-forward (e.g. pi/2 if agent faces +X but panorama "
                  "expects +Y, etc). Only used when sqa3d_use_agent_pose is set."},
    )
    sqa3d_canvas_center_mode: str = field(
        default="auto",
        metadata={"help": "Inference-time SQA3D canvas-origin ablation. "
                  "'auto' (default) defers to sqa3d_use_agent_pose. "
                  "'agent_pose' = situated agent position + yaw. "
                  "'scene_center' = no override (reproject_scene default). "
                  "'random_camera' = one camera translation per sample (deterministic, "
                  "seeded by sample index), no yaw rotation. "
                  "'outside_bbox' = scene XY-center pushed +X by 1.5 * scene XY-radius, "
                  "Z at camera-mean height, no yaw rotation. "
                  "Override modes apply only to SQA3D items (gated on agent_position)."},
    )
    spbench_use_camera_pose: bool = field(
        default=False,
        metadata={"help": "For SPBench-SI: center panorama at the single pinned "
                  "frame's camera-to-world position and orient forward along the "
                  "camera's +Z axis (OpenCV convention). Gated on items with exactly "
                  "one pinned image so SPBench-MV (8 frames) falls through to the "
                  "default scene-center projection."},
    )
    spbench_camera_yaw_offset: float = field(
        default=0.0,
        metadata={"help": "Additional yaw offset (radians) applied on top of the "
                  "SPBench pinned-camera rotation. Use to compensate for any "
                  "axis-convention mismatch empirically identified via the "
                  "[spbench-pose] debug log. Only used when spbench_use_camera_pose is set."},
    )
    panoramic_eval_yaw_offset: float = field(
        default=0.0,
        metadata={"help": "Global canvas yaw offset in radians, applied on top of "
                  "whatever yaw the dataset's anchor policy chose (0 for VSI-Bench, "
                  "agent pose for SQA3D, camera pose for SPBench). A rigid rotation "
                  "about the canvas vertical axis: every distance, size and relation "
                  "is unchanged, so every ground-truth answer stays correct. The only "
                  "thing it moves is where the equirectangular longitude wrap (the "
                  "seam) falls relative to the scene. Sweep it and compare per-question "
                  "accuracy to measure what the seam costs on real data. "
                  "Default 0.0 is a no-op."},
    )
    stratified_eval: bool = field(
        default=True,
        metadata={"help": (
            "When True (default), validation samples are drawn with equal representation "
            "per question_type (stratified equal sampling, seed=42). "
            "Set to False to use the legacy contiguous slice [offset:offset+val_sample_num]."
        )},
    )
    use_resized_images: bool = field(
        default=False,
        metadata={"help": "When True (default), use the pre-resized color_640x480 images "
                  "if the directory exists.  Set to False to always load from the "
                  "original 'color/' directory regardless of what is on disk."},
    )
    image_resolution: str = field(
        default="320x240",
        metadata={"help": "Resolution tag for resized images. Controls which "
                  "color_<res>/rgb_<res> subdirectory to load frames from. Nothing in "
                  "this release loads precomputed vision features, so this selects "
                  "images only. Examples: '640x480', '320x240'."},
    )
    image_resolution_fallback: str = field(
        default="error",
        metadata={"help": "What to do when the frames actually on disk are not the "
                  "configured image_resolution. 'error' (default) raises; 'warn' logs "
                  "every occurrence and counts them; 'off' disables the check. Default "
                  "is strict because a dangling vga_wide_640x480 link made every "
                  "VSI-Bench 640 run read ARKitScenes at 320x240 for two weeks with no "
                  "warning at all."},
    )
    observation_max_resolution: str = field(
        default="",
        metadata={"help": "Cap (e.g. '640x480') on the photos an annotation carries "
                  "itself (`frames` / `observations`, the robot rows). A larger "
                  "photo is shrunk to fit, preserving aspect: RGB by LANCZOS, depth "
                  "by nearest, intrinsics scaled. A smaller one is kept. Empty = "
                  "native size, which every earlier run read. The rollout policy "
                  "service reads it from the run's resolved config, so training "
                  "and rollouts see the same canvas."},
    )
    max_image_resolution: str = field(
        default="",
        metadata={"help": "Optional on-the-fly cap (e.g. '640x480'). If set and a loaded "
                  "RGB image exceeds this, it is PIL-resized to fit (preserving aspect). "
                  "Intrinsics are rescaled accordingly. Empty string = disabled."},
    )
    image_resolution_policy: str = field(
        default="exact",
        metadata={"help": "How image_resolution is interpreted. 'exact' (default) "
                  "reproduces every pre-existing run: the frames opened must BE that "
                  "size. 'cap' reads the largest source a scene actually has, "
                  "downscales anything above image_resolution, and keeps a smaller "
                  "native source at its own size instead of inventing pixels. 'cap' "
                  "requires max_image_resolution and reports the achieved size per "
                  "scene, because a dataset whose highest-fidelity install is below "
                  "the target (ARKitScenes vga_wide is 320x240 on disk) would "
                  "otherwise either refuse the whole source or be upsampled into a "
                  "4x token bill for no extra evidence."},
    )
    dataset_sampling_seed: int = field(
        default=42,
        metadata={"help": "Seed used for deterministic dataset-level subsampling when dataset specs "
                  "use sampling rates (e.g. vica_arkit%30)."},
    )
    curriculum_legacy_aug_seed: bool = field(
        default=False,
        metadata={"help": "Reproduce the shipped checkpoints' panoramic-augmentation stream: "
                  "seed the curriculum aug rng with dataset_sampling_seed itself (correlated "
                  "with the per-item sampler rng, so on the train split the aug center/yaw "
                  "replay the sampler's first draws: anchor camera pinned per task slot, "
                  "aug yaw equal to the sampler's next uniform draw in ~9 of 10 samples) "
                  "and apply augmentation on the eval split too. The shipped stage-1 "
                  "checkpoints (paper v11 and bottleneck64_dlr50_distract) trained with "
                  "this correlated stream. Leave False for new runs (decorrelated, "
                  "training-only augmentation)."},
    )
    use_depth_embedding: bool = field(
        default=True,
        metadata={"help": "Add a learned depth embedding to each image token's features. "
                  "Metric depth is fed to the cartesian_fourier encoder and the resulting "
                  "vector is added to the visual token before the LLM sees it. "
                  "This injects depth info through features instead of RoPE positions. "
                  "DEFAULT TRUE since 2026-08-01: every shipped model trains with "
                  "it on, and a False default silently amputated the metric "
                  "channel in three separate eval harnesses that built "
                  "DataArguments directly and never set the flag. With it off "
                  "the model keeps angular MRoPE, so it reads DIRECTION at a "
                  "cosine near 1.0 and reads DISTANCE not at all, which looks "
                  "like a model that has plateaued rather than a broken "
                  "harness. Set it False only for a deliberate no-3D-PE "
                  "ablation."},
    )
    depth_embed_min: float = field(
        default=0.3,
        metadata={"help": "Minimum depth (meters) fed to the depth encoder. "
                  "Depths below this are clamped up to it."},
    )
    depth_embed_num_freqs: int = field(
        default=16,
        metadata={"help": "Number of log-spaced frequencies per axis for the "
                  "cartesian_fourier depth encoder (sin + cos + raw per channel)."},
    )
    depth_embed_mlp_hidden: int = field(
        default=512,
        metadata={"help": "Hidden dim of the 2-layer MLP inside the cartesian_fourier "
                  "depth embedding (enc_dim -> mlp_hidden -> hidden_size). Lower values "
                  "(e.g. 64) bottleneck the depth path's expressivity. Used only by "
                  "depth_embed_mode='cartesian_fourier'."},
    )
    depth_embed_cartesian_fourier_use_rmsnorm: bool = field(
        default=False,
        metadata={"help": "Insert nn.RMSNorm(hidden_size) on the MLP output before the gate "
                  "in cartesian_fourier mode. Decouples output magnitude from MLP weight "
                  "growth (post-norm ‖.‖ ≈ √hidden_size regardless of MLP magnitude). "
                  "Pair with a small cartesian_fourier_gate_init (e.g. 0.025) to land "
                  "depth_emb at ~10% of visual norm. Used only by cartesian_fourier mode."},
    )
    depth_embed_cartesian_fourier_per_channel_gate: bool = field(
        default=False,
        metadata={"help": "Use a per-channel gate vector [hidden_size] instead of a scalar "
                  "for cartesian_fourier mode. Mathematically equivalent to a row-rescaling "
                  "of the second linear, but gives separate Adam state per output channel."},
    )
    depth_embed_cartesian_fourier_gate_init: float = field(
        default=1.0,
        metadata={"help": "Initial value of the cartesian_fourier gate (scalar or per-channel "
                  "vector). Default 1.0 matches the historical init. Use small values "
                  "(e.g. 0.025) when pairing with RMSNorm so depth_emb starts at a target "
                  "fraction of visual norm. DO NOT use this to raise the depth branch's "
                  "starting magnitude: Adam is invariant to a uniform gradient scaling, so "
                  "a large gate multiplies the whole ratio trajectory instead of raising "
                  "its floor, and a scalar cannot travel back (measured 2026-08-31, two "
                  "arms at 16.77 and 30.19 reached ratio 22.7 and 17.3 by step 9000 against "
                  "a reference 2.4). Use depth_embed_cartesian_fourier_mlp_init_std."},
    )
    depth_embed_cartesian_fourier_mlp_init_std: float = field(
        default=0.02,
        metadata={"help": "std of the normal init on BOTH cartesian_fourier MLP layers. "
                  "0.02 is the shipped value behind every published number, and it starts "
                  "the depth branch at ~0.062 of the visual norm, so it must grow ~16x in "
                  "output and ~3x in weight std before it carries anything. Healthy runs "
                  "converge to std ~0.060 (clampfix 0.0600/0.0594, the shipped checkpoint "
                  "0.0646/0.0619). Setting 0.06 starts the branch at that magnitude, giving "
                  "a step-0 ratio of 0.62 instead of 0.062. Unlike the gate this genuinely "
                  "damps growth, because Adam's step size is absolute so the same travel is "
                  "a smaller relative change from a larger start. It buys magnitude only, "
                  "not alignment: a trained MLP at std 0.060 outputs 58.7 where a random one "
                  "at the same std outputs 10.7."},
    )
    curriculum_samples_per_scene: int = field(
        default=8,
        metadata={"help": "SpatialPretrainingDataset: number of synthetic Q-A samples "
                  "generated per scene per epoch."},
    )
    curriculum_scene_retry_attempts: int = field(
        default=128,
        metadata={"help": "SpatialPretrainingDataset: how many random scenes ONE sample may "
                  "try before the draw raises. The task is drawn once and held across the "
                  "retries, so this budget has to cover 1/p for the LEAST-accepting task in "
                  "the mixture: at per-scene acceptance p a draw exhausts with probability "
                  "(1-p)^N, and an exhausted draw kills the run. p is set by the task's own "
                  "accept gates, so a mixture that tightens a gate has to raise this with "
                  "it. The 128 default suits gates that accept more than 20% of scenes."},
    )
    curriculum: str = field(
        default="",
        metadata={"help": "Named curriculum from onecanvas.data.curriculum_task_mix.CURRICULA "
                  "(e.g. 'synthetic_obb'). When set, overrides curriculum_task_types and "
                  "curriculum_canvas_obb_only_tasks with the curriculum's frozen task list and "
                  "obb-only list. Prefer this over passing inline --curriculum_task_types so the "
                  "training script stays decoupled from task-level details."},
    )
    curriculum_real_objects: bool = field(
        default=False,
        metadata={"help": "Run the selected curriculum with real objects instead of synthetic "
                  "boxes. Every task in curriculum_task_types and curriculum_canvas_obb_only_tasks "
                  "is replaced entry for entry by its real-asset twin "
                  "(onecanvas.data.curriculum_task_mix.REAL_OBJECT_SUBSTITUTIONS), which asks the "
                  "same question with the referent named by class instead of by an inline patch "
                  "marker; task weights are unchanged and a task without a twin is an error. "
                  "Requires --real_object_assets_enable True. Unless set explicitly, "
                  "real_object_assets_distractors default to (0, 3) and "
                  "curriculum_appearance_real_uniform_t is on, the real samplers' recipe from "
                  "scene_harvested. '--curriculum synthetic_obb --curriculum_real_objects True' "
                  "is the main-paper curriculum with real objects and nothing else changed."},
    )
    curriculum_task_types: str = field(
        default_factory=lambda: CURRENT_TASKS,
        metadata={"help": "SpatialPretrainingDataset: comma-separated list of task types "
                  "to sample. Default pulls from onecanvas.data.curriculum_task_mix.CURRENT_TASKS. "
                  "Prefer --curriculum <name> for a named curriculum; override this flag "
                  "only when pinning to a custom task mix."},
    )
    curriculum_zero_visual: bool = field(
        default=False,
        metadata={"help": "Zero out visual features on inline patch marker tokens so "
                  "the only signal is the depth/angle embedding. Isolates whether "
                  "the model can read geometric embeddings without visual noise."},
    )
    curriculum_single_patch_canvas: bool = field(
        default=False,
        metadata={"help": "Keep only the marker-referenced patch(es) in the canvas. "
                  "Reduces the canvas from ~100 tokens to 1 per marker, making the "
                  "depth embedding maximally salient for probing experiments. "
                  "Global toggle applied to every sample — for per-task control use "
                  "curriculum_canvas_obb_only_tasks instead."},
    )
    curriculum_canvas_obb_only_tasks: str = field(
        default_factory=lambda: CURRENT_OBB_ONLY,
        metadata={"help": "Comma-separated probe task names for which the canvas should "
                  "be stripped to only the marker-referenced patch(es) (+ synthetic OBB "
                  "points for box tasks). Per-sample equivalent of curriculum_single_patch_canvas. "
                  "Default pulls from onecanvas.data.curriculum_task_mix.CURRENT_OBB_ONLY. "
                  "Set to '' to disable all per-task stripping."},
    )
    curriculum_depth_scale: float = field(
        default=1.0,
        metadata={"help": "Multiplicative scale applied to depth embeddings. Use with "
                  "curriculum_zero_visual to boost the depth signal to normal hidden-state "
                  "magnitude (e.g. 50.0). Only affects probing experiments."},
    )
    depth_embed_learned_scale: bool = field(
        default=False,
        metadata={"help": "Add a learnable exp(param) scalar on the depth embedding. "
                  "Initialized at exp(0)=1.0 so it starts neutral; the model learns "
                  "how loud the depth signal should be. Logged as 'depth_embed_scale' "
                  "in the checkpoint. Replaces the fixed curriculum_depth_scale."},
    )
    depth_ratio_penalty_beta: float = field(
        default=0.0,
        metadata={"help": "Soft per-token penalty coefficient on ‖depth_emb‖ / ‖visual_emb‖ "
                  "(v_norm detached). Shape set by depth_ratio_penalty_mode. 0.0 disables "
                  "(default)."},
    )
    depth_ratio_penalty_mode: str = field(
        default="hinge",
        metadata={"help": "Shape of the depth/visual ratio penalty. 'hinge' (default): "
                  "beta * mean(relu(ratio - r0)^2), a soft budget that is free below r0 "
                  "and taxes excess above it (cannot drive depth below r0 or collapse it; "
                  "non-saturating restoring force pulls a too-loud depth back toward r0). "
                  "'l2': legacy beta * mean(ratio^2) with minimum at zero, kept only to "
                  "reproduce older ratiopen* runs."},
    )
    depth_ratio_penalty_r0: float = field(
        default=0.5,
        metadata={"help": "Free-budget threshold for the hinge penalty (ignored when "
                  "mode='l2'). Depth/visual ratio up to r0 is unpenalized; above r0 each "
                  "token costs beta*(ratio-r0)^2. Set near the healthy operating point "
                  "(~0.5 for QA). Raise for geometry-heavy stage-1 if probe accuracy needs "
                  "a louder depth budget."},
    )
    depth_embed_fixed_ratio: float = field(
        default=0.0,
        metadata={"help": "0 (default) = off. Otherwise the depth term is rescaled every "
                  "forward so that mean per-token ||depth|| equals this fraction of the "
                  "DETACHED mean per-token ||visual|| it is added to, i.e. depth/visual_ratio "
                  "is pinned at this value by construction and the branch learns direction "
                  "only. The rescale factor is in the graph on the depth side, so no gradient "
                  "asks for magnitude and the scalar gate cancels out (its gradient is exactly "
                  "zero; it stays at init). Not a penalty: there is no loss term and nothing "
                  "to tune. Why: on the synthetic curriculum the patch features carry no "
                  "signal, so nothing in the loss opposes the depth term growing past them. "
                  "Every mlp_init_std 0.06 stage 1 parked at ratio 8 to 13 and its stage 2 "
                  "collapsed to 45 VSI (2026-09-04); the healthy references settle at 0.8 to "
                  "2.1 on their own, and a hinge penalty pinning 0.6 cost nothing on the "
                  "curriculum. Recorded in resolved_config.json and applied at eval through "
                  "--from-config, since it is a property of the trained weights."},
    )
    curriculum_use_cot: bool = field(
        default=False,
        metadata={"help": "For dist_pp / dist_pc tasks, output chain-of-thought with "
                  "intermediate 3D coordinates before the final distance, e.g. "
                  "'p1=(1.23, 0.45, 3.10), p2=(2.40, 0.80, 1.90), dist=1.7'. "
                  "Gives 7x more supervision signal per sample."},
    )
    curriculum_shuffle_marker_t: bool = field(
        default=True,
        metadata={"help": "For spatial probe tasks (everything except appearance_order*, "
                  "same_frame, frame_order), override each inline patch marker's MRoPE "
                  "T-axis value with the T of a random canvas patch from a DIFFERENT "
                  "source frame. Removes the shortcut where the marker's source-frame "
                  "index (which correlates with physical position in a walking-through-"
                  "the-room video) leaks spatial info via T. Set False to reproduce the "
                  "pre-shuffle behavior for ablation."},
    )
    curriculum_dist_decimals: int = field(
        default=1,
        metadata={"help": "Number of decimal places for distance answers in dist_pp / "
                  "dist_pc / patch_depth tasks. Default 1 = '1.7', set to 2 = '1.73' "
                  "for finer-grained supervision."},
    )
    curriculum_max_body_patches_per_sample: int = field(
        default=100,
        metadata={"help": "Per-sample total budget for synthetic OBB body patches "
                  "(corners + face samples across all boxes). Split uniformly "
                  "across boxes and floored at 8 per box so every box always "
                  "contributes its 8 OBB corners; any remainder fills with "
                  "face samples."},
    )
    curriculum_counting_difficulty_mix: bool = field(
        default=False,
        metadata={"help": "Stochastic difficulty mix for the counting task family "
                  "(object_counting, object_counting_parity_box, "
                  "object_counting_mod3_box, rel_dir_count_side_box). Per call, "
                  "draws easy/medium/hard with weights 0.50/0.30/0.20 and applies "
                  "an escalating combination of: sparse OBB markers (2-4 corners "
                  "instead of full body, forcing 3D xyz-adjacency clustering); "
                  "tight 3D packing (relax 2*r_sphere non-overlap to 0.9-2.0*r); "
                  "canvas-collinear OBB pairs (origin-collinear so they overlap "
                  "on the equirectangular canvas, distinguishable only by depth, "
                  "object_counting only); per-OBB independent dims (kills the "
                  "'same size = same class' shortcut); extended N range up to 25 "
                  "(object_counting only)."},
    )
    curriculum_colinear_centers_prob: float = field(
        default=0.1,
        metadata={"help": "Probability per sample of forcing all task OBBs onto "
                  "a single ray from the panorama center (shared (lat, lon), "
                  "depth-only distinguishable). Applies to dist_box, "
                  "rel_dist_box, object_counting (+ parity/mod3 siblings), "
                  "appearance_order_box, and multi_box_grounding. Kills the "
                  "'canvas region -> answer' shortcut by collapsing per-marker "
                  "panoramic footprints onto the same bearing, so the model "
                  "must use depth / T / per-patch xyz rather than 2D canvas "
                  "position. Works under any canvas centering policy. Gate "
                  "short-circuits at prob=0.0 to skip the rng call (opt out "
                  "with --curriculum_colinear_centers_prob 0.0)."},
    )
    curriculum_num_images_range: str = field(
        default="",
        metadata={"help": "Comma-separated list of num_images values to sample from per "
                  "training example (e.g. '2,8,16,32'). Each __getitem__ call picks one "
                  "uniformly at random. Empty string = use the global num_images. "
                  "Eval always uses gen_eval_num_images."},
    )
    curriculum_resample_frames: bool = field(
        default=False,
        metadata={"help": "For geometric curriculum training, draw one source frame "
                  "from each temporal interval using a per-draw RNG stream seeded by "
                  "the training DataLoader. Repeated visits may differ. The recorded "
                  "frame_rng_seed replays the exact draw. Keys remain chronological "
                  "and RGB, depth, pose, intrinsics, and timestamps stay aligned. "
                  "Evaluation and pinned-frame inputs stay fixed."},
    )
    curriculum_min_frame_gap: float = field(
        default=0.0,
        metadata={"help": "Minimum fraction of total frames between two patches in "
                  "frame_order task. E.g. 0.25 means the two patches must be at "
                  "least 25%% of the frame range apart. Helps learning by ensuring "
                  "a salient T-axis difference. 0.0 = no constraint (original)."},
    )
    curriculum_appearance_radius: float = field(
        default=0.5,
        metadata={"help": "3D radius (meters) for local min-T in appearance_order probe. "
                  "For each sampled patch, the ground-truth first-appearance time is "
                  "the minimum frame_index among all valid patches within this radius."},
    )
    curriculum_appearance_spread_enable: bool = field(
        default=True,
        metadata={"help": "For appearance_order_box, override each synthetic surface "
                  "point's MRoPE T with a uniform random frame index in [T_min, "
                  "num_frames-1] (T_min = the box's ground-truth first-appearance "
                  "frame). Forces the model to aggregate across the box body and "
                  "find the minimum T rather than reading a single T off the marker. "
                  "Set False to recover the legacy single-T-per-box behaviour "
                  "(trivial scalar sort) for ablation."},
    )
    curriculum_appearance_n_boxes: int = field(
        default=4,
        metadata={"help": "Number of boxes / markers in appearance_order and "
                  "appearance_order_box probes. Assigned T_min values are "
                  "{k, k+1, ..., k+N-1} with k random. Increase for harder sorting "
                  "(5 → 120 permutations, 6 → 720). Default 4 preserves the legacy "
                  "4-way MCQ behaviour."},
    )
    curriculum_appearance_open_ended: bool = field(
        default=False,
        metadata={"help": "If True, appearance_order* answers are emitted as a "
                  "concatenated digit string (e.g. '2413' meaning p2<p4<p1<p3) "
                  "instead of a 4-way MCQ letter. Drops chance baseline from 25% "
                  "to 1/N! and forces the model to output the full permutation."},
    )
    curriculum_appearance_real_uniform_t: bool = field(
        default=False,
        metadata={"help": "appearance_order_real T design. False: each asset confined "
                  "to its own non-overlapping T bucket (n_imgs/N frames). True: "
                  "consecutive t_starts {k,...,k+N-1}, each asset's per-patch T uniform "
                  "over [t_start, n_imgs-1] with one patch forced to t_start. Mirrors "
                  "synthetic appearance_order_box: heavy T-overlap, only min(T) per "
                  "object differentiates the ordering. Auto-set True by the "
                  "scene_harvested curriculum."},
    )
    curriculum_real_asset_global_subsample: bool = field(
        default=False,
        metadata={"help": "If True, draw a single fraction f ~ LogUniform[f_min, 1.0] "
                  "per sample and scale every real-asset paste's cap_per_paste by f, "
                  "shared across all pastes in that sample. Decouples target-class "
                  "patch volume from instance count N (model can no longer use "
                  "'total chair-ness / per-instance density = N' as a counting "
                  "shortcut). Applies to counting / appearance_order_real / "
                  "grounding / etc. — any task that uses real-asset pastes."},
    )
    curriculum_real_asset_subsample_min: float = field(
        default=0.05,
        metadata={"help": "Minimum fraction for curriculum_real_asset_global_subsample's "
                  "LogUniform[f_min, 1.0] draw. f_min = 0.05 means the smallest "
                  "samples retain 5% of asset patches (5-10 per typical instance). "
                  "Lower = harder; floor of 1 patch per non-zero paste is enforced."},
    )
    curriculum_fixed_counting_total: int = field(
        default=9,
        metadata={"help": "object_class_counting_real / parity_real / mod3_real always "
                  "place exactly this many objects per sample, with "
                  "N_target ~ Uniform{1..total} and N_distract = total - N_target. "
                  "Severs density-vs-count correlation and drops N=0 so 'zero' "
                  "can't be used as a hedged-low default. Default 9."},
    )
    curriculum_scene_sources: str = field(
        default="vica_scannet_base,vica_arkit_base,vica_snpp_base",
        metadata={"help": "Comma-separated dataset names for geometric probing scene sources. "
                  "Default includes ScanNet, ScanNet++, and ARKitScenes to prevent overfitting."},
    )
    curriculum_obb_feature_stash_enable: bool = field(
        default=True,
        metadata={"help": "Use a pre-built patch-feature stash (drawn from a held-out scene "
                  "pool) for probe marker + synthetic-OBB feature content, skipping per-sample "
                  "image load and visual-tower forward. Cached per model under "
                  "~/.cache/onecanvas_features/<feature_prefix>_obb_feature_stash_<N>.pt; auto-built "
                  "on first use. Third layer of the geometry-signal-only curriculum "
                  "(stripped canvas + position-ID geometry + scene-agnostic marker content). "
                  "Default True: the stash fast-path is ~1.4x faster per step than the live "
                  "reprojection and is the recommended production config (iter-3 benchmark)."},
    )
    curriculum_obb_feature_stash_size: int = field(
        default=20000,
        metadata={"help": "Target number of patches in the marker stash. Each patch is "
                  "[N_layers, D] bf16 (~32 KB for Qwen3-VL-8B), so 20000 ≈ 640 MB in RAM. "
                  "Used in the cache filename; change triggers a rebuild."},
    )
    curriculum_num_distractors_min: int = field(
        default=-1,
        metadata={"help": "Minimum number of distractor OBBs sampled per probe sample. "
                  "Each distractor is a non-overlapping OBB with its own unique stash "
                  "feature (disjoint from task refs and from other distractors), so the "
                  "model cannot shortcut on 'the odd-one-out' or 'the coherent class'. "
                  "Excluded tasks: *floor_area*. Sentinel -1 means 'use curriculum default'. "
                  "Set to 0 explicitly to disable on a curriculum that "
                  "would otherwise turn distractors on."},
    )
    curriculum_num_distractors_max: int = field(
        default=-1,
        metadata={"help": "Maximum number of distractor OBBs sampled per probe sample. "
                  "Per-sample M ~ Uniform[min, max]. Sentinel -1 means 'use curriculum "
                  "default'; set to 0 explicitly to disable."},
    )
    curriculum_distractor_route_clone_landmark_p: float = field(
        default=0.0,
        metadata={"help": "Probability (per distractor) that a route_plan_*turn_box "
                  "distractor clones one of the route's landmark feature sources "
                  "instead of drawing a fresh stash patch. When >0 a distractor OBB "
                  "looks visually identical to a waypoint (same single canvas-patch "
                  "feature stamped on the body), so the model can't pick out "
                  "landmarks by feature uniqueness alone - it has to deduce which "
                  "OBB is which waypoint from inline marker IDs and OBB positions. "
                  "Applies only to route_plan_{1,2,3,4}turn_box. Default 0.0 keeps "
                  "the historic 'distractor features are always disjoint' policy."},
    )
    real_object_assets_enable: bool = field(
        default=False,
        metadata={"help": "Enable the real-object asset bank used by the *_real probe "
                  "tasks (object_class_counting_real / object_class_grounding_real / "
                  "object_class_appearance_order_real). Each pasted asset carries its "
                  "own per-patch features harvested from real ScanNet scenes via "
                  "scripts/extract_real_object_assets.py - unlike synthetic OBBs which "
                  "clone one feature across every body point, every real-asset patch "
                  "is a distinct feature, so the model must use shape-class signal "
                  "to count / locate / time the objects."},
    )
    real_object_assets_root: str = field(
        default="",
        metadata={"help": "Root directory of the harvested asset bank "
                  "(asset_index.json + per-class subdirectories of *.pt). "
                  "Required when --real_object_assets_enable True; otherwise "
                  "may be left empty. Override via the ONECANVAS_ASSETS_ROOT "
                  "env var if you prefer not to pass it on the CLI."},
    )
    real_object_assets_classes: str = field(
        default="",
        metadata={"help": "Comma-separated EmbodiedScan classes to load from the bank. "
                  "Empty string (default) loads every class present in asset_index.json. "
                  "Set explicitly to restrict to a subset."},
    )
    real_object_assets_max_patches_per_sample: int = field(
        default=300,
        metadata={"help": "Per-SAMPLE cap on total real-asset patches appended to the "
                  "compact scene tensor. Padded-batch compute pays for the longest sample "
                  "in the batch, so a single 633-patch route_plan sample makes the other "
                  "15 samples in an effective batch pay that cost too. The cap is split "
                  "evenly across all pastes attached to the sample (cap_per_paste = "
                  "max_per_sample // n_pastes); each paste then random-subsamples its "
                  "valid-projection patches down to that share. ~200 patches total still "
                  "carries plenty of class-identity signal (median asset has ~30 patches "
                  "anyway). Set <= 0 to disable."},
    )
    real_object_assets_distractors_min: int = field(
        default=-1,
        metadata={"help": "Minimum number of OTHER-class real-asset distractor pastes "
                  "appended to the *_real samplers that previously ran in a closed-world "
                  "configuration (every direction / metric / navigation / visibility / "
                  "appearance-order task). Sentinel -1 = use curriculum default; curriculum key "
                  "is 'real_distractors'. The three samplers that already had hardcoded distractors "
                  "(counting_real, size_real, rel_dir_count_side_real) are unaffected by this "
                  "knob; grounding_real draws from this range with a floor of one, because "
                  "'find every X' needs at least one object that is not an X."},
    )
    real_object_assets_distractors_max: int = field(
        default=-1,
        metadata={"help": "Maximum number of OTHER-class real-asset distractor pastes. "
                  "Per-sample M ~ Uniform[min, max]. Sentinel -1 = use curriculum default."},
    )
    real_object_scene_assets_root: str = field(
        default="",
        metadata={"help": "Path to the whole-scene real-object asset bank "
                  "(scripts/extract_real_object_scene_assets.py output). Empty = use "
                  "the legacy per-object bank at real_object_assets_root. The whole-scene "
                  "bank stores every valid patch per scene plus all OBB metadata, so "
                  "paste-time inflation (real_object_paste_inflation_max_frac) becomes "
                  "a free hyperparameter without re-extraction."},
    )
    real_object_paste_inflation_max_frac: float = field(
        default=0.0,
        metadata={"help": "Max paste-time OBB inflation as a fraction of the tight bbox "
                  "dims. 0.0 = tight (matches legacy bank up to its 5 cm boundary). "
                  "1.0 = double the bbox in each axis. Per-sample draw: inflation_frac ~ "
                  "Uniform[0, max]. Skipped (held at 0.0) for object_class_grounding_real "
                  "and object_class_appearance_order_real where the pasted region is the "
                  "supervision target. Requires real_object_scene_assets_root."},
    )
    inline_patch_override_rope: bool = field(
        default=False,
        metadata={"help": "When True, inline patch marker tokens have their MRoPE "
                  "(T,H,W) positions overwritten with the source canvas patch's positions, so "
                  "the marker is RoPE-coincident with its source patch. When False (default), "
                  "markers keep their natural text-sequence position on all three RoPE axes; "
                  "the model binds marker -> canvas patch via content-similarity attention "
                  "instead. Default flipped to False after the override was shown to collapse "
                  "slot identity across repeated same-token markers (two markers share MRoPE, "
                  "distinguished only via causal masking), which killed multi-marker probes "
                  "such as above_of while patch_exists_pair still hits 1.0 under False."},
    )
    use_inline_patch_embedding: bool = field(
        default=False,
        metadata={"help": "When True, inline-patch tokens get a single shared learned "
                  "[hidden_size] identity bias added on top of their copied source-patch features. "
                  "Default False: the identity bias is not load-bearing, so inline-patch tokens "
                  "carry ONLY the copied canvas features and binding is pure content-matching + "
                  "text-sequence MRoPE. Set True for the legacy identity-bias behavior."},
    )
    use_gt_depth: bool = field(
        default=False,
        metadata={"help": "Load ground-truth sensor depth from depth/{frame}.png instead of "
                  "DAv3-predicted depth stored in the geometry .pt file. Poses and "
                  "intrinsics still come from the geometry file. Only affects ScanNet "
                  "scenes that have a depth/ subdirectory."},
    )
    use_gt_all: bool = field(
        default=True,
        metadata={"help": "Bypass DA3 entirely: GT pose, K, sensor depth from each "
                  "dataset's per-scene files. ScanNet: pose/*.txt x axisAlignment, "
                  "<scene_id>.txt, depth/*.png. ARKitScenes: lowres_wide.traj (w2c, "
                  "OpenCV cam, z-up world), per-frame .pincam, lowres_depth/*.png. "
                  "ScanNet++: pose_intrinsic_imu.json (c2w, OpenCV cam, y-up world; "
                  "K is full-res so rescaled to 256x192), iphone/depth/*.png. Forces "
                  "--use_gt_depth. Scenes missing GT files silently fall back to DA3 "
                  "unless --gt_strict is set."},
    )
    scannetpp_pose_frame: str = field(
        default="mesh",
        metadata={"help": "World frame of ScanNet++ iPhone poses. 'mesh': the per-scene "
                  "registration to the annotated laser-scan mesh (scenes without one fall "
                  "back to predicted geometry). 'arkit': ARKit's frame turned z-up by a fixed "
                  "Rx(+90), for every scene. Checkpoints trained before the registration "
                  "existed, the released OneCanvas model among them, were trained on "
                  "'arkit'."},
    )
    upright_arkit: bool = field(
        default=False,
        metadata={"help": "Gravity-upright ARKitScenes frames before the vision "
                  "tower. ARKit writes every frame into the sensor's landscape "
                  "buffer regardless of phone roll, so gravity points sideways on "
                  "84 of the 150 VSI-Bench ARKit scenes; the poses encode the roll, "
                  "so the 3D is right and only the appearance is wrong. Rotates "
                  "image, depth, intrinsics and pose together by k clockwise "
                  "quarter turns (k from lowres_wide.traj gravity, NOT from "
                  "metadata.csv sky_direction, which is wrong on 20 of the 150), "
                  "which leaves the lifted world points unchanged. Structurally a "
                  "no-op on ScanNet / ScanNet++, which have no lowres_wide.traj. "
                  "WORKS IN TRAINING TOO, no re-extraction. This help text used "
                  "to say 'eval-side only: stage 2 reads cached per-frame "
                  "features, so using this in training means re-extracting "
                  "ARKit features'. That described the OLD precomputed-feature "
                  "pipeline and was already false when written: nothing on the "
                  "training path reads a qwen3_vl_features_*.pt any more "
                  "(data_processor_3d assigns self._feature_set and never "
                  "builds a path from it; scripts/precompute.py is the only "
                  "file that names those files, and it WRITES them). The "
                  "loader emits raw pixel_values and the visual tower runs "
                  "live in forward() on the PRE-PATH-A branch, so rotating at "
                  "load time is all it takes. Corrected 2026-08-27 after the "
                  "stale sentence nearly bought a 400 GB re-extraction job for "
                  "a one-flag change. "
                  "Default False reproduces every pre-existing number exactly."},
    )
    upright_force_turns: int = field(
        default=0,
        metadata={"help": "Causal check for --upright_arkit: force this many "
                  "clockwise quarter turns on EVERY ARKitScenes scene instead of "
                  "the gravity-derived k. Slice the already-upright (k=0) scenes "
                  "out of the results and they should degrade by roughly the "
                  "measured rotation deficit. 0 = use gravity (the real thing)."},
    )
    scene_filter_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file of scene ids (one per line) restricting "
                  "the eval to those scenes. Used to re-measure a fix on the subset "
                  "it provably touches instead of paying for the whole benchmark "
                  "(e.g. --upright_arkit changes nothing outside gravity-rotated "
                  "ARKit scenes). Default None = all scenes."},
    )
    gt_strict: bool = field(
        default=False,
        metadata={"help": "When --use_gt_all is set, refuse to fall back to DA3 for "
                  "scenes missing GT calib. The loader emits a (deduped) warning and "
                  "returns None so the caller can skip/retry with a different scene. "
                  "Use this for debug/visualization where mixing GT and DA3 scenes "
                  "silently would be misleading."},
    )
    predicted_geometry: Optional[str] = field(
        default=None,
        metadata={"help": "Predicted-geometry override (eval only). "
                  "'da3_eval32' or 'mapanything_eval32' prefer that predictor's "
                  "eval32 geometry .pt (poses + intrinsics + depth predicted from "
                  "RGB on the exact frames the GT eval selects) over the default "
                  "balanced_256 DA3 files. Run with --no-use-gt-all so nothing GT "
                  "leaks in. Default None = unchanged behavior."},
    )
    depth_downsample: int = field(
        default=1,
        metadata={"help": "Coarsen the depth grid before the per-patch lookup "
                  "(eval-only input perturbation). K>1 nearest-subsamples each "
                  "depth map to (H_feat/K, W_feat/K) and replicates it back, so "
                  "every KxK block of canvas patches shares one depth sample. "
                  "Patch count, features and validity grid are unchanged: this "
                  "degrades WHERE patches land in 3D without changing how many "
                  "there are (unlike input image resolution, which changes the "
                  "patch count). Default 1 = unchanged behavior."},
    )
    question_type_filter: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated question_type allowlist applied at "
                  "annotation-load time (e.g. 'object_counting'). Keeps eval "
                  "sharding and len(dataset) correct because filtering happens "
                  "before the dataset is built. Default None = keep everything."},
    )
    feature_set: str = field(
        default="balanced",
        metadata={"help": "Which precomputed feature/geometry set to use. "
                  "'balanced' uses balanced-256 geometry + balanced-64 features. "
                  "'aggregated' uses da3_50 geometry + resolution-specific aggregated features "
                  "(matches pre-refactor training runs)."},
    )
    pano_grounding_format: bool = field(
        default=False,
        metadata={"help": "Reparameterize 3D grounding GT and predictions as "
                  "'[{\"bbox_3d\": [u, v, depth, sx, sy, sz], \"label\": ...}]'. "
                  "(u, v) are integer pano-angular coordinates in "
                  "[0, utils.bbox.PANO_COORD_SCALE) (currently 1000), computed from "
                  "the equirectangular projection of the bbox center relative to "
                  "the scene center. depth, sx, sy, sz are raw metric meters. "
                  "Default False = current <|box_start|>(cx,cy,cz,w,h,d)<|box_end|> "
                  "format. Equirectangular projection only."},
    )
    metric_json_grounding_format: bool = field(
        default=True,
        metadata={"help": "Reparameterize 3D grounding GT and predictions as "
                  "'[{\"bbox_3d\": [cx, cy, cz, sx, sy, sz], \"label\": ...}]' "
                  "with all six values as raw metric meters in the scene-centered "
                  "axis-aligned frame. Same coordinates as the legacy "
                  "<|box_start|>(...)<|box_end|> format, but wrapped in Qwen3-VL's "
                  "native bbox_3d JSON structure to leverage the pretrained 3D "
                  "grounding output prior. Mutually exclusive with "
                  "pano_grounding_format."},
    )

    def __post_init__(self):
        if self.curriculum_real_objects:
            if not self.real_object_assets_enable:
                raise ValueError(
                    "--curriculum_real_objects True needs --real_object_assets_enable "
                    "True (and an asset bank root): the *_real tasks paste harvested "
                    "objects."
                )
            # Real-side defaults, the recipe scene_harvested declares. Applied
            # BEFORE the named-curriculum block so a curriculum without a
            # 'real_distractors' key does not resolve the -1 sentinel to 0
            # first. An explicit CLI value still wins.
            if self.real_object_assets_distractors_min == -1:
                self.real_object_assets_distractors_min = 0
            if self.real_object_assets_distractors_max == -1:
                self.real_object_assets_distractors_max = 3
            self.curriculum_appearance_real_uniform_t = True
        if self.curriculum:
            if self.curriculum not in CURRICULA:
                raise ValueError(
                    f"Unknown curriculum {self.curriculum!r}. "
                    f"Available: {sorted(CURRICULA)}"
                )
            curriculum = CURRICULA[self.curriculum]
            self.curriculum_task_types = curriculum["tasks"]
            self.curriculum_canvas_obb_only_tasks = curriculum["obb_only"]
            # Curriculum-level distractor defaults. The -1 sentinel means
            # "unset by CLI" — any explicit value (including 0 to disable)
            # takes priority over the curriculum default.
            _d = curriculum.get("distractors")
            if self.curriculum_num_distractors_min == -1:
                self.curriculum_num_distractors_min = int(_d[0]) if _d is not None else 0
            if self.curriculum_num_distractors_max == -1:
                self.curriculum_num_distractors_max = int(_d[1]) if _d is not None else 0
            _rd = curriculum.get("real_distractors")
            if self.real_object_assets_distractors_min == -1:
                self.real_object_assets_distractors_min = int(_rd[0]) if _rd is not None else 0
            if self.real_object_assets_distractors_max == -1:
                self.real_object_assets_distractors_max = int(_rd[1]) if _rd is not None else 0
            # appearance_order_real T design switch.
            if curriculum.get("appearance_uniform_t", False):
                self.curriculum_appearance_real_uniform_t = True
            # Global per-sample real-asset patch subsample.
            if curriculum.get("real_asset_global_subsample", False):
                self.curriculum_real_asset_global_subsample = True
        if self.curriculum_real_objects:
            # After the curriculum resolved its lists, so the switch applies
            # to a named curriculum and to an inline task list alike.
            self.curriculum_task_types = real_object_tasks(self.curriculum_task_types)
            self.curriculum_canvas_obb_only_tasks = real_object_tasks(
                self.curriculum_canvas_obb_only_tasks)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    continuation_stop_step: int = field(
        default=0,
        metadata={"help": "Stop a resumed continuation at this global step while "
                  "leaving max_steps unchanged so the restored learning-rate "
                  "schedule keeps its original horizon. Zero disables it."},
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=256)
    lora_alpha: int = field(default=512)
    lora_dropout: float = field(default=0.0)
    lora_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Path to a pre-trained LoRA adapter directory to warm-start from (e.g. stage 1 "
            "checkpoint). With lora_checkpoint_merge=True this may be a comma-separated chain "
            "of adapter dirs merged in order (e.g. 'stage1_ckpt,stage2_ckpt' for a stage-3 run); "
            "auxiliary 3D embeddings are loaded from the LAST entry."
        )},
    )
    freeze_3d_embeddings: bool = field(
        default=False,
        metadata={"help": (
            "If True, keep the auxiliary 3D embedding modules (depth/camera/angle/reference/"
            "patch-marker) frozen instead of unfreezing them alongside the LoRA. Use for "
            "stage-3 runs that must keep perception bit-identical to the merged base."
        )},
    )
    lora_checkpoint_merge: bool = field(
        default=False,
        metadata={"help": (
            "If True, load the stage-1 LoRA from lora_checkpoint_path, merge it into the base "
            "model weights (merge_and_unload), then wrap with a fresh LoRA using the current "
            "lora_r / lora_alpha. Use this to bake stage-1 skills into base and train a smaller "
            "target-task LoRA on top. Auxiliary 3D embeddings are still loaded from the checkpoint."
        )},
    )
    load_depth_embed_from_stage1: bool = field(
        default=True,
        metadata={"help": (
            "If False, skip loading depth_embedding.pt from lora_checkpoint_path so the depth "
            "embedding module starts from fresh init and is shaped by stage-2 gradients only. "
            "Angle/camera/reference embeddings are still loaded. Has no effect when "
            "lora_checkpoint_path is unset or when resuming a stage-2 checkpoint."
        )},
    )
    depth_lr_multiplier: float = field(
        default=1.0,
        metadata={"help": (
            "Per-param-group LR multiplier applied to the depth embedding modules "
            "(depth_embedding / depth_fourier / depth_loc_proj / depth_cartesian* / "
            "depth_embed_log_scale). 1.0 = no change. Use >1 to "
            "let a warm-started depth MLP move faster toward the stage-2 equilibrium "
            "without cold-starting it. Splits the standard decay/no-decay groups into "
            "(decay,depth) / (decay,non-depth) / (no_decay,depth) / (no_decay,non-depth) "
            "and applies the multiplier to the depth subgroups; LR scheduler scales all "
            "groups uniformly so the depth group follows the same cosine curve at a "
            "higher peak."
        )},
    )
    depth_no_weight_decay: bool = field(
        default=False,
        metadata={"help": (
            "Exclude the depth embedding params (depth_embedding / depth_fourier / "
            "depth_loc_proj / depth_cartesian* / depth_embed_log_scale) "
            "from weight decay. Default False reproduces every shipped checkpoint "
            "(depth params sit in the decay group at the global weight_decay). Set True to "
            "exclude them: weight decay is an unconditional pull toward zero weights "
            "(amplified ~depth_lr_multiplier-fold by the per-group LR) that can erode depth "
            "below the ratio penalty's free budget and re-trigger collapse. Note: True was a "
            "prior default that did NOT match the shipped checkpoints."
        )},
    )
    depth_warmstart_gate_scale: float = field(
        default=1.0,
        metadata={"help": (
            "Legacy reproduction-only manual warm-start scale. The maintained merged "
            "stage-2 handoff requires this to remain 1.0 and derives any fixed-to-free "
            "gate compensation from the source checkpoint state automatically."
        )},
    )

    ## Generation-based evaluation during training
    gen_eval_steps: int = field(
        default=500,
        metadata={"help": "Run generation-based eval (METEOR/EM/ROUGE/CIDEr) every this many training steps. 0 to disable."},
    )
    gen_eval_num_samples: int = field(
        default=100,
        metadata={"help": "Number of val samples for each periodic generation eval run."},
    )
    gen_eval_num_images: int = field(
        default=32,
        metadata={"help": "Number of images per scene used during generation eval runs."},
    )
    gen_eval_final_num_samples: int = field(
        default=500,
        metadata={"help": "Number of val samples for the final generation eval at end of training."},
    )
    gen_eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Dataset(s) to use for generation eval. Comma-separated for multi-dataset eval, "
            "e.g. 'sqa3d,vlm3r_vsibench'. Each dataset is evaluated separately and per-type "
            "metrics are logged to wandb under gen_eval/{dataset}/{metric}. "
            "Defaults to dataset_use if not set."
        )},
    )
    best_eval_metric: str = field(
        default="combined_official",
        metadata={"help": (
            "Metric key in the gen-eval combined_stats used to select best_checkpoint. "
            "Default 'combined_official' is the mean of official_overall across the "
            "non-probe eval datasets (geometric_probing is excluded as a diagnostic). "
            "For STAGE-1 probe runs the probe IS the objective, so set this to "
            "'geometric_probing/official_overall' — otherwise best_checkpoint is chosen "
            "purely on whatever benchmark (e.g. VSI-Bench) happens to be in the eval set, "
            "which is OOD/non-diagnostic for a probe-only model."
        )},
    )
