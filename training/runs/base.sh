#!/bin/bash
# =============================================================================
# Shared run base — sourced by each run script.
# Each script sets its overrides, then sources this file to launch training.
# =============================================================================

set -euo pipefail

: "${RUN_NAME:?Must set RUN_NAME before sourcing base.sh}"
: "${RUN_EXTRA_ARGS:=}"

# --- Defaults (paper configuration) ---
: "${RUN_MODEL:=Qwen/Qwen3-VL-8B-Instruct}"
: "${RUN_LORA_R:=256}"
: "${RUN_LORA_ALPHA:=512}"
: "${RUN_LORA_DROPOUT:=0.05}"
: "${RUN_LR:=2e-5}"
: "${RUN_WEIGHT_DECAY:=0.01}"
: "${RUN_WARMUP:=0.03}"
# cosine is the recipe. constant_with_warmup is for a curve that keeps
# training as data is added and has no horizon; max_steps is then only a cap.
: "${RUN_LR_SCHEDULER:=cosine}"
: "${RUN_BATCH_SIZE:=1}"
: "${RUN_EFFECTIVE_BATCH:=16}"
: "${RUN_EXPECTED_GPUS:=8}"
: "${RUN_MAX_STEPS:=20000}"
: "${RUN_DATASETS:=sqa3d@2,sqa3d_agent_pose@1,vlm3r_vsibench@8,vica_arkit_base@1,vica_snpp_base@1,vica_scannet_base@1,geometric_probing@9}"
: "${RUN_GEN_EVAL_DATASET:=vsi_bench,sqa3d_agent_pose:200,geometric_probing:500}"
: "${RUN_NUM_IMAGES:=32}"
: "${RUN_MODEL_MAX_LENGTH:=12000}"
: "${RUN_IMAGE_RESOLUTION:=320x240}"
: "${RUN_MAX_IMAGE_RESOLUTION:=}"
# exact = every pre-existing run. cap = image_resolution is a ceiling: an
# oversized source is downscaled to it, a source already below it is read at
# its own size and reported. See DataArguments.image_resolution_policy.
: "${RUN_IMAGE_RESOLUTION_POLICY:=exact}"
: "${RUN_STAGE1_CKPT:=}"
# When a stage-1 checkpoint is given, True merges its LoRA into the base
# weights before training the fresh adapter (the paper's stage-2 recipe);
# False continues training the same adapter.
: "${RUN_LORA_MERGE:=False}"
: "${RUN_RESUME_FROM_CHECKPOINT:=}"
: "${RUN_SEED:=42}"
: "${RUN_ATTN_IMPLEMENTATION:=flash_attention_2}"

# Position encoding
: "${RUN_TEMPORAL_MAX_RANGE:=100.0}"
: "${RUN_TEMPORAL_RAW_FRAME_INDEX:=True}"

# Feature pipeline. Default False: visual encoder runs live in
# Qwen3VL3DModel.forward() and the .pt feature files are not loaded.

# Depth embedding
: "${RUN_USE_DEPTH_EMBEDDING:=True}"
: "${RUN_DEPTH_EMBED_NUM_FREQS:=16}"
: "${RUN_DEPTH_EMBED_MIN:=0.3}"
# Go-forward default: bottleneck-64 depth MLP (was 512 for the paper recipe).
: "${RUN_DEPTH_EMBED_MLP_HIDDEN:=64}"
: "${RUN_DEPTH_EMBED_CF_USE_RMSNORM:=False}"
: "${RUN_DEPTH_EMBED_CF_PER_CHANNEL_GATE:=False}"
: "${RUN_DEPTH_EMBED_CF_GATE_INIT:=1.0}"
: "${RUN_DEPTH_EMBED_CF_MLP_INIT_STD:=0.02}"
# Whether to warm-start depth_embedding.pt from RUN_STAGE1_CKPT. Set False
# for a cold-start of the depth module (shaped only by stage-2 gradients).
: "${RUN_LOAD_DEPTH_EMBED_FROM_STAGE1:=True}"
# Per-param-group LR multiplier for the 3D position embedding params. >1 lets
# a warm-started embedding MLP adapt faster than the LoRA. Go-forward default
# is 50 (the recipe behind the best checkpoints); the paper recipe used 1.0.
: "${RUN_DEPTH_LR_MULTIPLIER:=50.0}"

# Checkpointing
: "${RUN_SAVE_STEPS:=125}"
: "${RUN_SAVE_TOTAL_LIMIT:=3}"

# Eval
: "${RUN_GEN_EVAL_STEPS:=1000}"
: "${RUN_GEN_EVAL_NUM_SAMPLES:=1500}"
: "${RUN_GEN_EVAL_NUM_IMAGES:=32}"
: "${RUN_GEN_EVAL_FINAL_NUM_SAMPLES:=0}"
: "${RUN_VAL_SAMPLE_NUM:=100}"

# DataLoader prefetch factor. Drop to 1 on tight host-RAM budgets (e.g.
# 4-GPU holds with 5 workers/rank) to halve the in-flight batch buffer.
: "${RUN_DATALOADER_PREFETCH_FACTOR:=2}"

# --- Fixed across all runs ---
llm="${RUN_MODEL}"
entry_file="training/onecanvas/train/train.py"
: "${deepspeed:=./training/scripts/zero2.json}"

run_name="${RUN_NAME}"

# Candidate output roots, also used for mid-run checkpoint spillover by
# train/checkpoint_guard.py. Roots that don't exist are ignored; picked by
# free space at decision time.
export ONECANVAS_FALLBACK_CKPT_DIRS="${ONECANVAS_FALLBACK_CKPT_DIRS:-}"  # colon-separated, e.g. /data1/ckpt:/data2/ckpt

# Site hook: on a machine that provides ~/scripts/workspace.sh (a shared
# storage union), route outputs there when ONECANVAS_CHECKPOINT_DIR is unset.
# Everywhere else this block does nothing.
if [ -x "$HOME/scripts/workspace.sh" ]; then
  _site_root="$HOME/workspace"
  bash "$HOME/scripts/workspace.sh" mount >/dev/null 2>&1 || true
  if [ -z "${ONECANVAS_CHECKPOINT_DIR:-}" ] && mountpoint -q "$_site_root" 2>/dev/null \
     && ls "$_site_root" >/dev/null 2>&1; then
    ONECANVAS_CHECKPOINT_DIR="$_site_root/checkpoints/onecanvas_release_output"
    echo "[ckpt-guard] output root: ${ONECANVAS_CHECKPOINT_DIR} (site storage)"
  fi
fi

# Fallback: auto-allocate an output volume from ONECANVAS_FALLBACK_CKPT_DIRS
# (a full volume once killed three runs). Only reached when no explicit dir
# is set:
#   - ONECANVAS_CHECKPOINT_DIR unset: pick the fallback root with the most
#     free space.
#   - set, but this is a FRESH run (no existing run dir) and the volume has
#     under ONECANVAS_AUTOPICK_MIN_FREE_GB (default 200) free: re-pick,
#     loudly.
#   - set and the run dir already exists (a resume): ALWAYS respected, even
#     when full — moving a resume would orphan its checkpoints, and the
#     in-run spillover handles the space.
# ONECANVAS_AUTOPICK=0 disables re-picking entirely.
_pick_best_root() {
  local best="" best_free=0 r f
  local IFS=':'
  for r in ${ONECANVAS_FALLBACK_CKPT_DIRS}; do
    [ -d "$r" ] || continue
    f=$(df -P -B1G "$r" 2>/dev/null | awk 'NR==2{print $4}')
    [ -n "$f" ] && [ "$f" -gt "$best_free" ] && { best="$r"; best_free="$f"; }
  done
  echo "$best"
}
: "${ONECANVAS_AUTOPICK:=1}"
: "${ONECANVAS_AUTOPICK_MIN_FREE_GB:=200}"
if [ "${ONECANVAS_AUTOPICK}" = "1" ]; then
  if [ -z "${ONECANVAS_CHECKPOINT_DIR:-}" ]; then
    ONECANVAS_CHECKPOINT_DIR="$(_pick_best_root)"
    [ -z "${ONECANVAS_CHECKPOINT_DIR}" ] && ONECANVAS_CHECKPOINT_DIR=./output
    echo "[ckpt-guard] auto-allocated output volume: ${ONECANVAS_CHECKPOINT_DIR}"
  elif [ ! -d "${ONECANVAS_CHECKPOINT_DIR}/${run_name}" ]; then
    _free=$(df -P -B1G "${ONECANVAS_CHECKPOINT_DIR}" 2>/dev/null | awk 'NR==2{print $4}')
    if [ -n "${_free:-}" ] && [ "${_free}" -lt "${ONECANVAS_AUTOPICK_MIN_FREE_GB}" ]; then
      _best="$(_pick_best_root)"
      if [ -n "${_best}" ] && [ "${_best}" != "${ONECANVAS_CHECKPOINT_DIR}" ]; then
        echo "[ckpt-guard] ${ONECANVAS_CHECKPOINT_DIR} has only ${_free} GB free,"
        echo "[ckpt-guard] re-allocating this fresh run to ${_best} (set ONECANVAS_AUTOPICK=0 to keep the original)"
        ONECANVAS_CHECKPOINT_DIR="${_best}"
      fi
    fi
  fi
fi
export ONECANVAS_CHECKPOINT_DIR
output_dir="${ONECANVAS_CHECKPOINT_DIR:-./output}/${run_name}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_${run_name}"
export TRITON_HOME="/tmp/triton_home_${USER}_${run_name}"
# Pin wandb run id to RUN_NAME so resumes attach to the same run instead of
# starting a fresh one. `allow` creates the run on first launch and reattaches
# on every subsequent resume.
export WANDB_RUN_ID="${run_name}"
export WANDB_RESUME=allow
export WANDB_MODE="${WANDB_MODE:-online}"
# Colocate wandb's local files with the checkpoints instead of the checkout's
# cwd: 2026-08-06, two runs died with ENOSPC creating wandb/run-* on a full
# volume their checkpoints weren't even on. With WANDB_DIR on the output
# volume, train.py's launch-time free-space check gates both.
export WANDB_DIR="${WANDB_DIR:-${output_dir}}"
mkdir -p "${output_dir}"
# Generous wandb init timeout: slow shared filesystems + resume lookups.
export WANDB_INIT_TIMEOUT=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Cap glibc malloc arenas: many dataloader workers fragment the heap otherwise.
export MALLOC_ARENA_MAX=2
# Return freed memory to the OS promptly (long runs with large worker heaps).
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_THRESHOLD_=131072
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
# Set HF_HUB_OFFLINE=1 once the base checkpoint is fully cached to avoid
# hub lookups on every launch.
: "${HF_HUB_OFFLINE:=0}"
export HF_HUB_OFFLINE
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- GPU count ---
NNODES=${SLURM_NNODES:-1}
NPROC_PER_NODE=$(( RUN_EXPECTED_GPUS / NNODES ))
TOTAL_GPUS=${RUN_EXPECTED_GPUS}

# Respect CUDA_VISIBLE_DEVICES when set: nvidia-smi ignores it, so a run
# pinned to a subset of a node's GPUs would otherwise fail the count check.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _detected=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    _detected=$(nvidia-smi -L 2>/dev/null | grep -c "GPU") || _detected=0
fi
if [ "$_detected" -gt 0 ] && [ "$(( _detected * NNODES ))" -ne "$RUN_EXPECTED_GPUS" ]; then
    echo "ERROR: Expected ${RUN_EXPECTED_GPUS} GPU(s) but found $(( _detected * NNODES ))."
    exit 1
fi

# --- Effective batch size ---
RUN_GRAD_ACCUM=$(( RUN_EFFECTIVE_BATCH / (RUN_BATCH_SIZE * TOTAL_GPUS) ))
if [ "$RUN_GRAD_ACCUM" -lt 1 ]; then RUN_GRAD_ACCUM=1; fi
_actual_effective=$(( RUN_BATCH_SIZE * RUN_GRAD_ACCUM * TOTAL_GPUS ))
if [ "$_actual_effective" -ne "$RUN_EFFECTIVE_BATCH" ]; then
    echo "FATAL: effective_batch=${_actual_effective} != target ${RUN_EFFECTIVE_BATCH}."
    exit 1
fi

# --- Optional args ---
EXTRA_ARGS=""
if [ -n "${RUN_STAGE1_CKPT}" ]; then
    EXTRA_ARGS+=" --lora_checkpoint_path ${RUN_STAGE1_CKPT}"
    EXTRA_ARGS+=" --lora_checkpoint_merge ${RUN_LORA_MERGE}"
fi
if [ -n "${RUN_RESUME_FROM_CHECKPOINT}" ]; then
    EXTRA_ARGS+=" --resume_from_checkpoint ${RUN_RESUME_FROM_CHECKPOINT}"
fi

# Is the training mix probe-only, i.e. is this a stage-1 curriculum run?
_ds_names=$(echo "${RUN_DATASETS}" | tr ',' '\n' | sed 's/@.*//' | grep -v '^$' | sort -u | tr '\n' ',' | sed 's/,$//')
_probe_only=0
if [ "${_ds_names}" = "geometric_probing" ]; then _probe_only=1; fi

# Best-checkpoint selection metric. combined_official excludes geometric_probing
# (treated as a diagnostic), so for a probe-only stage-1 run it would otherwise
# pick best_checkpoint on whatever benchmark (e.g. VSI-Bench) is in the eval set,
# which is OOD/non-diagnostic for a probe-only model. Auto-select the probe
# metric when the training mix is probe-only; override with RUN_BEST_EVAL_METRIC.
if [ -z "${RUN_BEST_EVAL_METRIC:-}" ]; then
    if [ "${_probe_only}" = "1" ]; then
        RUN_BEST_EVAL_METRIC="geometric_probing/official_overall"
    else
        RUN_BEST_EVAL_METRIC="combined_official"
    fi
fi

# A PROBE-ONLY RUN EVALUATES ON THE PROBE AND NOTHING ELSE (2026-09-01).
# Stage 1 sees only synthetic boxes on an empty canvas, so every real benchmark
# is out of distribution for it and its curve DECLINES BY DESIGN, 42 to 23 over
# 30k steps in both July references. Selecting best_checkpoint on it was already
# prevented above, but the curve was still emitted, still logged to wandb under
# vsi_bench/, and therefore still readable as the run's result. It was read that
# way: on 2026-09-01 the runs board, which picks a headline benchmark itself for
# any run no group claims, headlined three live stage-1 arms on vsi_bench and
# drew that by-design decline while the metric they were actually training rose
# from 31.8 to 55.2. A number that must not be read should not be produced.
# THERE IS NO OVERRIDE. One existed (RUN_ALLOW_OOD_GEN_EVAL=1, "if you
# deliberately want the OOD curve") and was used four days later by the pinned
# stage-1 launchers, whose OOD curve then headlined the board a second time
# (owner, 2026-09-05: "remove all references of eval on stage 1 in vsibench
# ... I want you to never do it again"). An escape hatch for a number that must
# not be read is the number being read.
if [ "${_probe_only}" = "1" ]; then
    # `|| true` on each grep because a no-match exits 1, and under the
    # `set -euo pipefail` these launchers run with that aborts the assignment
    # BEFORE the FATAL below can print. An abort with no message is the silent
    # failure this whole guard exists to prevent.
    _eval_kept=$(echo "${RUN_GEN_EVAL_DATASET}" | tr ',' '\n' \
                 | { grep -E '^geometric_probing(:[0-9]+)?$' || true; } \
                 | tr '\n' ',' | sed 's/,$//')
    _eval_dropped=$(echo "${RUN_GEN_EVAL_DATASET}" | tr ',' '\n' \
                    | { grep -Ev '^(geometric_probing(:[0-9]+)?)?$' || true; } \
                    | tr '\n' ',' | sed 's/,$//')
    if [ -z "${_eval_kept}" ]; then
        echo "FATAL: probe-only training mix but gen_eval set '${RUN_GEN_EVAL_DATASET}'" >&2
        echo "       contains no geometric_probing, so this run would produce no" >&2
        echo "       readable score at all. Fix RUN_GEN_EVAL_DATASET." >&2
        exit 1
    fi
    if [ -n "${_eval_dropped}" ]; then
        echo "[base.sh] probe-only training mix, so DROPPING out-of-distribution"
        echo "          gen_eval dataset(s): ${_eval_dropped}"
    fi
    RUN_GEN_EVAL_DATASET="${_eval_kept}"
fi

echo "============================================="
echo "RUN: ${RUN_NAME}"
echo "  model=${llm}"
echo "  lora_r=${RUN_LORA_R}  lora_alpha=${RUN_LORA_ALPHA}"
echo "  lr=${RUN_LR}  warmup=${RUN_WARMUP}  schedule=${RUN_LR_SCHEDULER}"
echo "  datasets=${RUN_DATASETS}"
echo "  max_steps=${RUN_MAX_STEPS}"
echo "  num_images=${RUN_NUM_IMAGES}  resolution=${RUN_IMAGE_RESOLUTION} (policy=${RUN_IMAGE_RESOLUTION_POLICY}, cap=${RUN_MAX_IMAGE_RESOLUTION:-none})"
echo "  depth_embed=${RUN_USE_DEPTH_EMBEDDING} (${RUN_DEPTH_EMBED_NUM_FREQS} freqs)  warm_start_from_stage1=${RUN_LOAD_DEPTH_EMBED_FROM_STAGE1}  depth_lr_mult=${RUN_DEPTH_LR_MULTIPLIER}"
echo "  save_steps=${RUN_SAVE_STEPS}  save_total_limit=${RUN_SAVE_TOTAL_LIMIT}"
echo "  eval=${RUN_GEN_EVAL_DATASET}  eval_samples=${RUN_GEN_EVAL_NUM_SAMPLES}  best_metric=${RUN_BEST_EVAL_METRIC}"
echo "  output_dir=${output_dir}"
echo "  gpus=${TOTAL_GPUS}  batch=${RUN_BATCH_SIZE}  accum=${RUN_GRAD_ACCUM}  effective=${_actual_effective}"
echo "  attn_implementation=${RUN_ATTN_IMPLEMENTATION}"
_num_cpus=$(python3 -c "import os; print(len(os.sched_getaffinity(0)))")
_num_workers=${RUN_NUM_WORKERS:-4}
echo "  cpus=${_num_cpus}  gpus_per_node=${NPROC_PER_NODE}  workers_per_gpu=${_num_workers}  total_workers=$(( _num_workers * NPROC_PER_NODE ))"
echo "============================================="

# --- Torchrun ---
# Ensure repo-root CWD regardless of launcher; entry_file and other paths
# below are repo-relative. base.sh lives at <repo>/training/runs/.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [ "$NNODES" -gt 1 ]; then
    MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    TORCHRUN_ARGS="--nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} \
        --rdzv_backend=c10d --rdzv_endpoint=${MASTER_NODE}:29500 \
        --rdzv_id=${SLURM_JOB_ID:-$$}"
else
    TORCHRUN_ARGS="--nproc_per_node=${NPROC_PER_NODE} \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"
fi

torchrun ${TORCHRUN_ARGS} \
         ${entry_file} \
         --deepspeed ${deepspeed} \
         --model_name_or_path "${llm}" \
         --attn_implementation "${RUN_ATTN_IMPLEMENTATION}" \
         --dataset_use "${RUN_DATASETS}" \
         --output_dir "${output_dir}" \
         --run_name "${run_name}" \
         --bf16 True \
         --lora_enable True \
         --lora_r ${RUN_LORA_R} \
         --lora_alpha ${RUN_LORA_ALPHA} \
         --lora_dropout ${RUN_LORA_DROPOUT} \
         --max_steps ${RUN_MAX_STEPS} \
         --per_device_train_batch_size ${RUN_BATCH_SIZE} \
         --gradient_accumulation_steps ${RUN_GRAD_ACCUM} \
         --learning_rate ${RUN_LR} \
         --weight_decay ${RUN_WEIGHT_DECAY} \
         --warmup_ratio ${RUN_WARMUP} \
         --lr_scheduler_type "${RUN_LR_SCHEDULER}" \
         --model_max_length ${RUN_MODEL_MAX_LENGTH} \
         --temporal_max_range ${RUN_TEMPORAL_MAX_RANGE} \
         --temporal_raw_frame_index ${RUN_TEMPORAL_RAW_FRAME_INDEX} \
         --use_depth_embedding ${RUN_USE_DEPTH_EMBEDDING} \
         --depth_embed_num_freqs ${RUN_DEPTH_EMBED_NUM_FREQS} \
         --depth_embed_min ${RUN_DEPTH_EMBED_MIN} \
         --depth_embed_mlp_hidden ${RUN_DEPTH_EMBED_MLP_HIDDEN} \
         --depth_embed_cartesian_fourier_use_rmsnorm ${RUN_DEPTH_EMBED_CF_USE_RMSNORM} \
         --depth_embed_cartesian_fourier_per_channel_gate ${RUN_DEPTH_EMBED_CF_PER_CHANNEL_GATE} \
         --depth_embed_cartesian_fourier_gate_init ${RUN_DEPTH_EMBED_CF_GATE_INIT} \
         --depth_embed_cartesian_fourier_mlp_init_std ${RUN_DEPTH_EMBED_CF_MLP_INIT_STD} \
         --load_depth_embed_from_stage1 ${RUN_LOAD_DEPTH_EMBED_FROM_STAGE1} \
         --depth_lr_multiplier ${RUN_DEPTH_LR_MULTIPLIER} \
         --use_resized_images True \
         --image_resolution ${RUN_IMAGE_RESOLUTION} \
         --max_image_resolution "${RUN_MAX_IMAGE_RESOLUTION}" \
         --image_resolution_policy "${RUN_IMAGE_RESOLUTION_POLICY}" \
         --num_images ${RUN_NUM_IMAGES} \
         --gradient_checkpointing ${RUN_GRAD_CKPT:-True} \
         --logging_steps 5 \
         --save_steps ${RUN_SAVE_STEPS} \
         --save_total_limit ${RUN_SAVE_TOTAL_LIMIT} \
         --remove_unused_columns False \
         --seed ${RUN_SEED:-42} \
         --data_seed ${RUN_SEED:-42} \
         --report_to wandb \
         --eval_strategy "no" \
         --gen_eval_steps ${RUN_GEN_EVAL_STEPS} \
         --gen_eval_num_samples ${RUN_GEN_EVAL_NUM_SAMPLES} \
         --gen_eval_num_images ${RUN_GEN_EVAL_NUM_IMAGES} \
         --gen_eval_final_num_samples ${RUN_GEN_EVAL_FINAL_NUM_SAMPLES} \
         --gen_eval_dataset "${RUN_GEN_EVAL_DATASET}" \
         --best_eval_metric "${RUN_BEST_EVAL_METRIC}" \
         --val_sample_num ${RUN_VAL_SAMPLE_NUM} \
         --dataloader_pin_memory True \
         --dataloader_prefetch_factor ${RUN_DATALOADER_PREFETCH_FACTOR} \
         --dataloader_num_workers ${RUN_NUM_WORKERS:-4} \
         --projection_mode "equirectangular" \
         ${EXTRA_ARGS} ${RUN_EXTRA_ARGS}
