#!/bin/bash
# Stage 1 of the selected release recipe, synthetic spatial curriculum.
# Train a rank-256 adapter and a width-64 geometry MLP at 50x learning rate.
# Pin the geometry-to-visual norm ratio at 0.5 during this stage only.
# Launch on four allocated GPUs from the repository root.
# See docs/REPRODUCING.md for the configuration and stage-2 handoff.
set -euo pipefail

RUN_NAME="stage1_curriculum"
RUN_EXPECTED_GPUS=4
RUN_BATCH_SIZE=2
RUN_GRAD_CKPT=False
RUN_DEPTH_EMBED_CF_MLP_INIT_STD=0.06
RUN_DEPTH_EMBED_MLP_HIDDEN=64
RUN_DEPTH_LR_MULTIPLIER=50.0
export ONECANVAS_FULLSPAN_SAMPLER=1
RUN_EFFECTIVE_BATCH=16
RUN_LR=2e-5
RUN_DATASETS="geometric_probing@1.0"
RUN_GEN_EVAL_DATASET=geometric_probing
RUN_GEN_EVAL_STEPS=3000
RUN_GEN_EVAL_NUM_SAMPLES=500
RUN_VAL_SAMPLE_NUM=500
RUN_SAVE_STEPS=1000
RUN_MAX_STEPS=30000

# The released run's canvas conventions, pinned here rather than inherited
# from base.sh or the argument defaults, which may change.
RUN_TEMPORAL_RAW_FRAME_INDEX=False
RUN_EXTRA_ARGS="--depth_embed_fixed_ratio 0.5 \
                --scannetpp_pose_frame arkit \
                --depth_no_weight_decay False \
                --curriculum_legacy_aug_seed True \
                --curriculum_obb_feature_stash_enable True \
                --curriculum_obb_feature_stash_size 20000 \
                --curriculum synthetic_obb \
                --curriculum_samples_per_scene 16 \
                --curriculum_use_cot False \
                --curriculum_dist_decimals 1 \
                --curriculum_appearance_radius 0.5 \
                --curriculum_appearance_spread_enable True \
                --curriculum_shuffle_marker_t True \
                --curriculum_max_body_patches_per_sample 200 \
                --curriculum_num_distractors_min 6 \
                --curriculum_num_distractors_max 12 \
                --panoramic_augment_center True \
                --panoramic_augment_yaw True \
                --curriculum_scene_sources vica_scannet_base,vica_arkit_base,vica_snpp_base \
                --metric_json_grounding_format True"

source "$(dirname "$0")/base.sh"
