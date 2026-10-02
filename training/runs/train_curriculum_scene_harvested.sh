#!/bin/bash
# Stage-1 alternative: the scene-harvested real-asset spatial-pretraining
# curriculum, the appendix 'Scene-harvested objects' study (see
# onecanvas.data.curriculum_task_mix, key 'scene_harvested'). FIVE of the
# main-paper curriculum's six task families (no grounding family, see the
# registry docstring), adds an object-size task, weights the metric and
# navigation tasks differently, and is grounded on real EmbodiedScan object
# assets, which reintroduces class priors and scores slightly below the
# synthetic scheme (VSI avg 69.4 vs 70.1). For the main-paper curriculum with
# real objects and NOTHING else changed, run train_stage1_curriculum.sh with
# "--curriculum synthetic_obb --curriculum_real_objects True
# --real_object_assets_enable True --real_object_assets_root ..." instead
# (every synthetic task, grounding included, has a class-named twin). Requires the
# asset banks built by scripts/extract_real_object_assets.py and
# scripts/extract_real_object_scene_assets.py. LoRA r=256/alpha=512,
# dropout 0.15, from scratch, lr=2e-5, 30000 steps cosine.
#
# Launch (8 GPUs, torchrun is invoked by base.sh):
#   bash training/runs/train_curriculum_scene_harvested.sh

RUN_NAME="curriculum_scene_harvested"
RUN_EXPECTED_GPUS=8
RUN_BATCH_SIZE=2
RUN_EFFECTIVE_BATCH=16
RUN_GRAD_CKPT=False
RUN_LR=2e-5
RUN_LORA_R=256
RUN_LORA_ALPHA=512
RUN_LORA_DROPOUT=0.15
RUN_NUM_WORKERS=5
RUN_DATASETS="geometric_probing@1.0"
RUN_GEN_EVAL_DATASET=geometric_probing   # stage 1 is judged on the curriculum only
RUN_GEN_EVAL_STEPS=3000
RUN_GEN_EVAL_NUM_SAMPLES=500
RUN_VAL_SAMPLE_NUM=500
RUN_SAVE_STEPS=1000
RUN_SAVE_TOTAL_LIMIT=10
RUN_MAX_STEPS=30000

RUN_EXTRA_ARGS="--curriculum scene_harvested \
                --real_object_assets_enable True \
                --curriculum_counting_difficulty_mix True \
                --curriculum_colinear_centers_prob 0.1 \
                --curriculum_num_distractors_min 1 \
                --curriculum_num_distractors_max 4 \
                --curriculum_samples_per_scene 16 \
                --curriculum_use_cot False \
                --curriculum_dist_decimals 1 \
                --curriculum_appearance_radius 0.5 \
                --curriculum_appearance_spread_enable True \
                --curriculum_shuffle_marker_t True \
                --curriculum_max_body_patches_per_sample 100 \
                --panoramic_augment_center True \
                --panoramic_augment_yaw True \
                --curriculum_scene_sources vica_scannet_base,vica_arkit_base,vica_snpp_base \
                --metric_json_grounding_format True"

source "$(dirname "$0")/base.sh"
