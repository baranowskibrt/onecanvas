#!/bin/bash
# Stage 2 from the 0.5-pinned stage-1 checkpoint.
# Merge stage 1; the shared loader derives and folds its geometry compensation
# once, then trains a fresh rank-64 adapter with the branch unconstrained.
# See docs/REPRODUCING.md for the measured configuration.
set -euo pipefail

RUN_NAME="stage2_qa"
RUN_STAGE1_CKPT="${STAGE1_CKPT:?Set STAGE1_CKPT to the stage-1 curriculum checkpoint dir}"
RUN_LORA_MERGE=True
RUN_LORA_R=64
RUN_LORA_ALPHA=128
RUN_EXPECTED_GPUS=8
RUN_EFFECTIVE_BATCH=32
RUN_MAX_STEPS=10000
RUN_SAVE_STEPS=500
RUN_GEN_EVAL_STEPS=500
RUN_GEN_EVAL_DATASET="vsi_bench"
RUN_LOAD_DEPTH_EMBED_FROM_STAGE1=True
RUN_DEPTH_EMBED_MLP_HIDDEN=64
RUN_DEPTH_LR_MULTIPLIER=50.0

RUN_DATASETS="sqa3d@1,sqa3d_agent_pose@1,vlm3r_vsibench@7.8,vica_arkit_base@0.74,vica_snpp_base@0.62,vica_scannet_base@0.58"

export ONECANVAS_FULLSPAN_SAMPLER=1

# The released run's canvas conventions, pinned here rather than inherited
# from base.sh or the argument defaults, which may change.
RUN_TEMPORAL_RAW_FRAME_INDEX=False
RUN_EXTRA_ARGS="--depth_embed_fixed_ratio 0.0 \
                --scannetpp_pose_frame arkit \
                --panoramic_augment_center True --panoramic_augment_yaw True \
                --use_gt_all True \
                --metric_json_grounding_format True"

source "$(dirname "$0")/base.sh"
