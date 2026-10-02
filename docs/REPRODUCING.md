# Reproducing the training procedure

This procedure produced the released model. Stage 2 trains from the
30,000-step stage-1 checkpoint, whose geometry-to-visual norm ratio is fixed at
0.5. The stage-2 geometry branch trains freely after a one-time scale
calibration at initialization. The released weights are stage-1 step 30,000
and stage-2 step 7,000, and both stages are downloadable on their own
(see the README's Checkpoints section).

## Configuration

| Setting | Stage 1 | Stage 2 |
|---|---|---|
| Backbone | Qwen3-VL-8B-Instruct | Stage-1 adapter merged into the base |
| LoRA rank / alpha | 256 / 512 | Fresh adapter, 64 / 128 |
| LoRA dropout | 0.05 | 0.05 |
| Effective batch | 16 | 32 |
| Measured GPU count | 4 | 8 |
| Base learning rate | 0.00002 | 0.00002 |
| Schedule | Cosine, 3% warmup | Cosine, 3% warmup |
| Geometry MLP hidden width | 64 | 64, loaded from stage 1 |
| Geometry learning-rate multiplier | 50 | 50 |
| Geometry MLP initialization std. | 0.06 | Loaded weights |
| Geometry-to-visual norm ratio | Fixed at 0.5 | Unconstrained |
| Training steps | 30,000 | Up to 10,000 |

Stage 1 uses the `synthetic_obb` curriculum, 16 samples per scene, 6 to 12
distractor boxes, and a 200-patch object budget. It uses a 20,000-entry feature stash, built on first use, and the legacy
augmentation seed convention of the measured run. These
settings are explicit in `training/runs/train_stage1_curriculum.sh`.

Stage 2 samples VLM-3R-VSIBench, SQA3D, and ViCA at approximately 66%, 17%,
and 17% by weight. Checkpoints are selected by periodic VSI-Bench generation
accuracy on a fixed, category-stratified 1,500-question sample from the benchmark
annotations. The released run stopped after about 9,000 steps and selected step 7,000,
preserved in `best_checkpoint/`. A new run should select its own checkpoint by
the same criterion.

## Environment

The measured run uses Python 3.11, PyTorch 2.7.0 with CUDA 12.8,
Transformers 5.2.0, PEFT 0.18.1, DeepSpeed 0.18.5, and FlashAttention 2.8.3.
The recorded package constraints are in `constraints-release.txt`.

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[train,eval,data]' -c constraints-release.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

FlashAttention needs a compatible compiler and CUDA toolkit. The constraints
record the measured environment. A fresh Python 3.11 installation of the
standalone export was validated through a real 32-frame ScanNet inference on
2026-09-22 with PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, Transformers
5.2.0, and PEFT 0.18.1. They are not a complete lock of system libraries.

Prepare the datasets using [DATA.md](DATA.md), then set both roots before
launching from the repository directory.

```bash
export ONECANVAS_DATA_ROOT=/path/to/datasets
export ONECANVAS_CHECKPOINT_DIR=/path/to/checkpoints
export WANDB_MODE=offline  # optional, keeps tracking local
bash training/runs/train_stage1_curriculum.sh
STAGE1_CKPT="$ONECANVAS_CHECKPOINT_DIR/stage1_curriculum/checkpoint-30000" \
  bash training/runs/train_stage2_qa.sh
```

The scripts launch with `torchrun`. Allocate the GPUs before invoking them.
Stage 1 was measured with four GPUs, batch two per GPU and accumulation two.
Stage 2 used eight GPUs, batch one per GPU and accumulation four. The launcher
computes accumulation from the effective batch. Changing GPU count changes
sample ordering and requires a fresh run when optimizer state is sharded.
For a continuation, pass `RUN_RESUME_FROM_CHECKPOINT` with the latest complete
checkpoint from that same stage. Stage 2 is a new adapter, not a stage-1 resume.

## Geometry-scale handoff

The per-input stage-1 normalization is not stored in the MLP weights. The
shared stage-2 loader reads the source fixed-ratio setting from its
`resolved_config.json`, estimates the raw geometry norm from the saved MLP on
20,000 deterministic synthetic rays with depths from 0.3 to 6 metres, and uses
the measured visual norm for this feature distribution. It derives the gate
factor from that source state and applies it exactly once. No launcher embeds a
checkpoint-specific factor. Missing source configuration or depth state is a
fatal, actionable load error rather than an uncompensated warm start.

This calibration sets the starting scale. It does not enforce a ratio of 0.5
on every real scene. Stage 2 uses `depth_embed_fixed_ratio=0.0` and learns the
scale freely. The converted gate and its handoff metadata are stored together
in `depth_embedding.pt`. Resume restores both and refuses to apply the
conversion again. The visual-norm calibration reference is specific to this
backbone and feature distribution. Set a positive
`--depth_embed_fixed_ratio` only when an explicitly pinned stage 2 is intended.
Zero is the free-running default.

## Evaluation and export

```bash
python training/run_benchmarks.py \
  --from-config "$ONECANVAS_CHECKPOINT_DIR/stage2_qa" \
  --lora "$ONECANVAS_CHECKPOINT_DIR/stage2_qa/best_checkpoint" \
  --datasets vsi_bench sqa3d spbench
```

The standard evaluation uses 640×480 inputs for VSI-Bench and SPBench and
320×240 for SQA3D. The headline metrics are VSI-Bench's mean of eight category
means, SQA3D strict EM@1, and SPBench's mean of SI and MV averages. Each SPBench
split average is the mean over its questions, with their observed numerical
and multiple-choice proportions.

To create a self-contained model directory, merge both adapters in order and
carry the final geometry weights with them.

```bash
python scripts/export_merged_checkpoint.py \
  --stage1-lora "$ONECANVAS_CHECKPOINT_DIR/stage1_curriculum/checkpoint-30000" \
  --lora "$ONECANVAS_CHECKPOINT_DIR/stage2_qa/best_checkpoint" \
  --out "$ONECANVAS_CHECKPOINT_DIR/OneCanvas-Qwen3-VL-8B"
```

With the evaluation command above, the released run's unmerged adapters score
65.33 SQA3D, 70.81 VSI-Bench and 74.32 SPBench on the complete test sets. These
are the numbers a new run of this procedure should approach. Merging rounds
weights in bfloat16 and can change generated answers, so the merged download is
measured separately, and the README lists its scores. Evaluate your own merged
export in the same way.

The launchers pin the released run's canvas conventions,
`RUN_TEMPORAL_RAW_FRAME_INDEX=False` (normalized frame index on MRoPE T) and
`--scannetpp_pose_frame arkit`, so newer argument defaults do not change the
recipe. A fresh run still differs from the measured one in four ways that
cannot be pinned:

- The stage-1 feature stash (20,000 real patch features) is built on first use
  from your own scene data, so its contents differ from the measured run's.
- A step-0 evaluation now runs before training, which advances the random state
  the stage-2 data sampler draws from, so the sample order differs.
- The few ScanNet++ scenes that need predicted geometry now get it at the right
  depth scale. The measured run dropped their patches.
- Each canvas token now reads its depth at the centre pixel of its patch. The
  measured run, and so the released weights, read the patch's top-left pixel,
  which put tokens a median 5.8 cm off their surface on the 320x240 training
  grid. The README scores are measured with the current rule.

Expect a new run to land near the measured scores, not on them.

The measured training and evaluation source is commit
`946aecd3af5200bee3604a4c20e25c638e45a7f7`. The current release contains later
changes and has not been retrained end to end. The paper's component ablations
and seed study were run on a different training run. The released run's two
stages used 568.5 allocated A6000 GPU-hours, including checkpoint selection and
the training tail, or 284.2 A100-equivalent GPU-hours under a 0.5
peak-throughput normalization. This excludes external benchmark evaluation and
other research experiments.
