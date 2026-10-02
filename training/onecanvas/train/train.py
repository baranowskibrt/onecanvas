# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import gc
import json
import logging
import math
import os
import pathlib
import sys
import time
from pathlib import Path

# Must be set before CUDA context is created (before `import torch` uses CUDA).
# Allows the allocator to grow existing segments instead of requiring a contiguous
# free block, preventing OOM when reserved-but-fragmented memory is available.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Suppress the "forked after parallelism" warning from HuggingFace tokenizers.
# DataLoader workers fork after the fast tokenizer has already been used in the
# main process; the tokenizer correctly disables its own parallelism, so this is
# safe — the warning is just noise.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datetime import timedelta

import torch
import transformers
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from onecanvas.data.data_processor_3d import make_supervised_data_module
from onecanvas.setup_3d import (configure_processor, get_inner_3d_model,
                                init_3d_embeddings)
from onecanvas.train.checkpoint_guard import (DiskSpaceGuardCallback,
                                              assert_free_space_at_launch,
                                              finalize_checkpoint,
                                              find_last_complete_checkpoint)
from onecanvas.train.argument import (DataArguments, ModelArguments,
                                      TrainingArguments)
from onecanvas.train import eval_contract
from transformers import AutoConfig, AutoProcessor, Trainer

import utils as eval_utils
from utils.embedding_io import load_3d_embeddings, save_3d_embeddings


def _record_depth_handoff_metadata(model, output_dir: str) -> None:
    """Mirror checkpointed handoff metadata into the run configuration."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return
    metadata = getattr(get_inner_3d_model(model), "_depth_handoff_metadata", None)
    if not isinstance(metadata, dict):
        return
    config_path = pathlib.Path(output_dir) / "resolved_config.json"
    with config_path.open() as f:
        config = json.load(f)
    config.setdefault("training", {})["depth_handoff"] = metadata
    tmp_path = config_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp_path, config_path)


def _is_qwen3_vl(model_path: str) -> bool:
    """Return True when *model_path* points to a Qwen3-VL checkpoint."""
    lower = model_path.lower()
    return "qwen3-vl" in lower or "qwen3_vl" in lower


local_rank = None
from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# Lightweight profiling: dataloader wait vs GPU compute time
# ---------------------------------------------------------------------------
class ProfilingCallback(TrainerCallback):
    """Measures and logs dataloader fetch time vs compute time to wandb.

    After each step, logs:
      - profiling/dataloader_ms:  time spent waiting for the next batch
      - profiling/compute_ms:     time spent in forward + backward + optimizer
      - profiling/dataloader_pct: dataloader wait as % of total step time
      - profiling/dataloader_retries: total retry count across batch samples
    """

    def __init__(self):
        self._step_start = None      # set when step begins (after data is ready)
        self._data_fetch_start = None  # set when step ends (before next fetch)
        self._last_dataloader_ms = 0.0
        self._cumulative_retries = 0

    def on_step_begin(self, args, state, control, **kwargs):
        now = time.monotonic()
        # Time since last step ended = time spent fetching data
        if self._data_fetch_start is not None:
            self._last_dataloader_ms = (now - self._data_fetch_start) * 1000
        self._step_start = now

    def on_step_end(self, args, state, control, model=None, **kwargs):
        now = time.monotonic()
        if self._step_start is not None:
            compute_ms = (now - self._step_start) * 1000
            dl_ms = self._last_dataloader_ms
            total = dl_ms + compute_ms
            dl_pct = (dl_ms / total * 100) if total > 0 else 0

            # Pick up retry count stashed by the model's forward()
            batch_retries = getattr(model, "_last_batch_retries", 0) if model else 0
            self._cumulative_retries += batch_retries

            # Buffer profiling metrics with commit=False so that WandbCallback's
            # subsequent wandb.log(train_metrics, step=N) merges them into the
            # same step.
            if state.global_step % args.logging_steps == 0:
                log_dict = {
                    "profiling/dataloader_ms": dl_ms,
                    "profiling/compute_ms": compute_ms,
                    "profiling/dataloader_pct": dl_pct,
                    "profiling/seconds_per_iter": total / 1000,
                }
                if self._cumulative_retries > 0:
                    log_dict["profiling/dataloader_retries_total"] = self._cumulative_retries
                    log_dict["profiling/dataloader_retries_per_step"] = (
                        self._cumulative_retries / max(state.global_step, 1)
                    )
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(log_dict, commit=False)
                except Exception:
                    pass

            # Check for excessive retries (likely NFS / data linking issue)
            if self._cumulative_retries > 100 and state.global_step <= 10:
                print(
                    f"\n{'!'*70}\n"
                    f"  HIGH DATALOADER RETRY COUNT: {self._cumulative_retries} retries total\n"
                    f"  This usually means dataset files are inaccessible from this node.\n"
                    f"  Check: NFS mounts, symlinks, dataloader_num_workers (try lowering to 4-8).\n"
                    f"{'!'*70}\n"
                )

        # Mark start of next data fetch. Must run on EVERY step, not in on_log
        # (which only fires every logging_steps), otherwise dl_ms ends up
        # measuring against a stale anchor and absorbs N-1 prior step durations.
        self._data_fetch_start = now


class DepthMagnitudeCallback(TrainerCallback):
    """Logs the depth-MLP output magnitude relative to visual every
    ``logging_steps`` to wandb. Reads ``DEPTH_VISUAL_STATS["last"]`` stashed
    by the model's forward pass (module-level dict because the model is
    wrapped in PEFT + DeepSpeed at training time, so attribute lookups on
    the wrapped model don't reliably reach the inner Qwen3VL3DModel).

    ALSO PRINTS to stdout, on a coarser stride than it logs. 2026-08-28: a
    stage-1 run trained a model that read direction and could not read
    identity, because the depth term grew to ~11x the norm of the visual
    features it is ADDED to. The ratio had been logged correctly the whole
    time and nobody could see it, because wandb was the ONLY sink: the job
    log had nothing, so run_report.py, the runs board and every post-hoc log
    read were all blind to it, and the cause was instead reconstructed from
    saved checkpoints days later. A diagnostic that can invalidate a run
    belongs in the run's own log.

    WHAT THE LINE MEANS. ``gate`` is the discriminator, not ``ratio``. On the
    two healthy reference runs the learned gate rises to ~1.0 by step 6000 and
    then TURNS OVER, falling to ~0.65-0.71 and flattening, which pulls the
    ratio back to ~1.5-2.5 and holds it there for the remaining 24k steps. On
    the run that failed, the gate never turned over: 0.98 -> 1.18 -> 1.28 ->
    1.34, still climbing when it was killed, with the ratio at 11.3. A ratio
    above 2.5 is NOT by itself a fault (both references touch 2.6-3.0 late).
    A gate still above 1.0 and rising at step 8000 is.
    """

    PRINT_EVERY = 500

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0:
            return
        try:
            from model_adapters.qwen3_vl.model import DEPTH_VISUAL_STATS
        except Exception:
            return
        stats = DEPTH_VISUAL_STATS.get("last")
        if not stats:
            return
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(dict(stats), commit=False)
        except Exception:
            pass
        if state.global_step % self.PRINT_EVERY == 0 and state.is_world_process_zero:
            gate = stats.get("depth/gate")
            print(f"[depth] step {state.global_step} "
                  f"ratio {stats.get('depth/visual_ratio', float('nan')):.3f} "
                  f"gate {gate if gate is None else f'{gate:.3f}'} "
                  f"|depth| {stats.get('depth/per_token_norm', float('nan')):.3f}",
                  flush=True)


class ContinuationStopCallback(TrainerCallback):
    """Bound a resume without changing the scheduler's original horizon."""

    def on_step_end(self, args, state, control, **kwargs):
        stop_step = int(getattr(args, "continuation_stop_step", 0))
        if stop_step > 0 and state.global_step >= stop_step:
            control.should_training_stop = True
            control.should_save = True
            if state.is_world_process_zero:
                print(f"[continuation] reached stop step {stop_step}", flush=True)
        return control


class GenerationEvalCallback(TrainerCallback):
    """Runs autoregressive generation-based evaluation (METEOR / EM@1 / ROUGE-L / CIDEr)
    every ``eval_gen_steps`` training steps and once at the very end of training.

    Results are saved in the same format as the eval notebook:
      {output_dir}/gen_eval/step_{N}/final_metrics.json
      {output_dir}/gen_eval/step_{N}/qa_results_final.json
      {output_dir}/gen_eval/step_{N}/intermediate_metrics/metrics_step_*.json
    """

    def __init__(
        self,
        processor,
        data_args,
        training_args,
        output_dir: str,
        eval_gen_steps: int = 500,
        num_samples: int = 100,
        eval_num_images: int = 20,
        final_num_samples: int = 500,
        best_metric: str = "combined_official",
        eval_dataset_use: str = None,
        lineage: dict = None,
    ):
        super().__init__()
        # WHICH WEIGHTS a recorded result was measured on. A step number alone
        # does not identify them: the same output dir re-entered with another
        # warm start would otherwise reuse the first lineage's numbers.
        self.lineage = dict(lineage or {})
        self.processor = processor
        self.data_args = data_args
        self.training_args = training_args
        self.output_dir = output_dir
        self.eval_gen_steps = eval_gen_steps
        self.num_samples = num_samples
        self.eval_num_images = eval_num_images
        self.final_num_samples = final_num_samples
        self.best_metric = best_metric
        self.eval_dataset_use = eval_dataset_use
        # Parse comma-separated dataset names; fall back to a single-element list
        # containing None so that existing single-dataset behaviour is preserved.
        # Supports per-dataset sample counts: "vsi_bench,scanrefer_val:100"
        if eval_dataset_use:
            self.eval_dataset_names = []
            self._per_dataset_num_samples = {}
            for entry in eval_dataset_use.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" in entry:
                    name, count = entry.rsplit(":", 1)
                    self.eval_dataset_names.append(name.strip())
                    self._per_dataset_num_samples[name.strip()] = int(count)
                else:
                    self.eval_dataset_names.append(entry)
        else:
            self.eval_dataset_names = [None]
            self._per_dataset_num_samples = {}
        self._best_score = -1.0
        self._best_step = None
        # Restore prior best across trainer restarts so that the first gen-eval
        # after resume cannot clobber a better pre-restart "best" with a worse
        # score (the in-memory _best_score would otherwise reset to -1.0 and any
        # post-resume score > -1.0 wins, overwriting best_checkpoint/).
        prior_best_path = Path(self.output_dir) / "best_gen_eval.json"
        if prior_best_path.exists():
            try:
                import json as _json
                with open(prior_best_path) as _f:
                    _prior = _json.load(_f)
                _prior_score = _prior.get(self.best_metric)
                _prior_step = _prior.get("best_step")
                if isinstance(_prior_score, (int, float)):
                    self._best_score = float(_prior_score)
                    self._best_step = _prior_step
                    print(
                        f"[GenEval] Restored prior best from {prior_best_path}: "
                        f"{self.best_metric}={self._best_score:.4f} @ step {self._best_step}"
                    )
            except Exception as _e:
                print(f"[GenEval] Warning: could not load prior best ({prior_best_path}): {_e}")
        self._dataloaders: dict = {}        # {dataset_name: DataLoader}  periodic
        self._final_dataloaders: dict = {}  # {dataset_name: DataLoader}  final
        # Population accounting for the evaluation ledger, refilled per eval.
        self._populations: dict = {}
        self._resolved_sizes: dict = {}     # {(name, requested): resolved items}
        self._last_eval_counts = (0, 0)     # (scored, skipped) of the last eval

    # ------------------------------------------------------------------
    # Evaluation contract (onecanvas/train/eval_contract.py)
    # ------------------------------------------------------------------
    CONTRACT_NAME = eval_contract.BUILTIN_EVALUATOR

    def _requested_populations(self, is_final: bool) -> dict:
        n = self.final_num_samples if is_final else self.num_samples
        return {
            (name if name is not None else ""):
                int(self._per_dataset_num_samples.get(name, n))
            for name in self.eval_dataset_names
        }

    def _is_this_run_s_evaluator(self) -> bool:
        """True when the built-in gen_eval is configured well enough to BE this
        run's evaluator. False when the run turned it off (cadence 0) or never
        named a dataset, in which case the contract is satisfied by an attached
        plugin evaluator instead and this callback must stay out of the way:
        it must not run a baseline, a schedule or a final eval it was not
        configured for."""
        return not self.evaluation_contract_spec().problems()

    # Every DataArguments field whose change would change the NUMBER, so that
    # reuse of a recorded result cannot cross it. Absence is recorded as
    # "<absent>" rather than dropped: a field that disappears is a different
    # configuration, and silently dropping it would make reuse looser.
    _INPUT_FIELDS = (
        "image_resolution", "max_image_resolution", "use_resized_images",
        "dataset_offset", "stratified_eval", "scene_filter_file",
        "upright_arkit", "use_gt_all", "projection_mode",
        "panoramic_augment_center", "panoramic_augment_yaw",
        "temporal_max_range", "temporal_raw_frame_index",
        "use_depth_embedding", "depth_embed_fixed_ratio",
    )
    _SCORING_FIELDS = (
        "pano_grounding_format", "metric_json_grounding_format",
        "vanilla_qwen3vl",
    )

    def evaluation_contract_spec(self):
        """What this callback will actually measure, read off the callback.

        ``eval_dataset_names == [None]`` means --gen_eval_dataset was never
        given, so the eval would silently draw from the TRAINING mix. That
        surfaces here as an unnamed dataset, which the contract refuses.

        ``inputs`` and ``scoring`` are what a recorded result is reused against.
        The 640-to-320 ARKit downgrade and the inherited training scene
        allowlist are both in here for the same reason: each produced an
        ordinary-looking number over a different population than the one the
        run was configured to read, and each would otherwise match a record
        made under the correct setting.
        """
        return eval_contract.EvaluatorSpec(
            name=self.CONTRACT_NAME,
            cadence_steps=int(self.eval_gen_steps),
            populations=self._requested_populations(is_final=False),
            final_population=(
                int(self.final_num_samples) if self.final_num_samples > 0 else None
            ),
            baseline=True,
            inputs=dict(
                {k: getattr(self.data_args, k, "<absent>") for k in self._INPUT_FIELDS},
                num_images=int(self.eval_num_images),
                model_max_length=int(
                    getattr(self.training_args, "model_max_length", 0) or 0
                ),
            ),
            scoring=dict(
                {k: getattr(self.data_args, k, "<absent>") for k in self._SCORING_FIELDS},
                best_metric=self.best_metric,
            ),
            headline={"metric": self.best_metric, "unit": "fraction",
                      "better": "higher"},
            details={"datasets": [n for n in self.eval_dataset_names]},
        )

    # ------------------------------------------------------------------
    # Lazy dataloader: one per (dataset_name, periodic|final) pair
    # When distributed, each rank gets a non-overlapping slice of the
    # dataset (no padding, no duplication) so metrics are exact.
    # ------------------------------------------------------------------
    def _get_dataloader(self, num_samples: int, dataset_name, cache_dict: dict):
        """Return a cached DataLoader for *dataset_name* with *num_samples* samples."""
        cache_key = dataset_name  # None is a valid dict key
        if cache_key in cache_dict:
            return cache_dict[cache_key]

        import copy
        gen_data_args = copy.copy(self.data_args)
        gen_data_args.with_answer = False
        gen_data_args.val_sample_num = num_samples
        gen_data_args.num_images = self.eval_num_images
        gen_data_args.dataset_offset = 0
        if dataset_name is not None:
            gen_data_args.dataset_use = dataset_name

        data_module = make_supervised_data_module(
            self.processor,
            data_args=gen_data_args,
            select_images_randomly=False,
            build_train_dataset=False,
            build_eval_dataset=True,
        )
        dataset = data_module["eval_dataset"]
        # A population of zero is an infrastructure failure, not an empty
        # result: the run was configured to score N items and resolved none,
        # so every later "score" would be a division over nothing.
        if dataset is None or len(dataset) == 0:
            raise eval_contract.EvaluationExecutionError(
                f"eval dataset {dataset_name!r} resolved to 0 items for a "
                f"requested population of {num_samples}. Check the dataset "
                "name, its data root, and any scene filter applied to it."
            )
        self._resolved_sizes[(dataset_name, int(num_samples))] = len(dataset)
        try:
            num_workers = max(1, int(self.training_args.dataloader_num_workers // 2))
        except AttributeError:
            num_workers = 2
            print("Worker error")

        # Split indices across ranks without padding or duplication.
        # Each rank gets a contiguous, non-overlapping slice of the dataset.
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            n = len(dataset)
            chunk = n // world_size
            start = rank * chunk
            end = start + chunk if rank < world_size - 1 else n
            dataset = torch.utils.data.Subset(dataset, list(range(start, end)))
        sampler = None

        loader = DataLoader(
            dataset,
            collate_fn=data_module["data_collator"],
            batch_size=1,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
        )
        cache_dict[cache_key] = loader
        label = dataset_name if dataset_name is not None else "default"
        rank_info = ""
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            rank_info = f" (rank {torch.distributed.get_rank()}: {len(dataset)} samples)"
        print(f"[GenEval] Dataloader ready for '{label}': {num_samples} total samples.{rank_info}")
        return loader

    # ------------------------------------------------------------------
    # Dataset-name normalisation for MetricTracker
    # ------------------------------------------------------------------
    @staticmethod
    def _dataset_to_tracker_name(dataset_name):
        """Map a training dataset key to the string understood by MetricTracker."""
        _MAP = {
            "vlm3r_vsibench":   "vsi_bench",
            "vsibench":         "vsi_bench",
            "vsi_bench":        "vsi_bench",
            "sqa3d":            "sqa3d",
            "sqa3d_agent_pose": "sqa3d",
        }
        if dataset_name is None:
            return None
        return _MAP.get(dataset_name.lower(), dataset_name.lower())

    # ------------------------------------------------------------------
    # Core eval loop — all ranks run generation on their shard, then
    # gather results to rank 0 for metric computation.
    # ------------------------------------------------------------------
    def _run_eval(self, model, global_step: int, dataset_name, dataloader,
                  phase: str = eval_contract.PHASE_SCHEDULED):
        tracker_name = self._dataset_to_tracker_name(dataset_name)
        label = dataset_name if dataset_name is not None else "default"
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        rank = torch.distributed.get_rank() if is_distributed else 0
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        # Save & override cache / use_cache for generation
        _orig_use_cache = getattr(model.config, "use_cache", False)
        model.config.use_cache = True

        # Free gradient buffers before generation
        model.zero_grad(set_to_none=True)
        for p in model.parameters():
            p.grad = None
        gc.collect()
        torch.cuda.empty_cache()

        model.eval()
        # The generation loop is the per-rank, COLLECTIVE-FREE part. A rank
        # that dies in here (OOM on a long canvas is the usual way) must be
        # agreed on BEFORE anyone enters the all_gather below, because a rank
        # already blocked in that gather cannot be rescued by a join issued
        # afterwards -- that is a different collective, and it can only
        # mismatch. So the error is carried out of the loop and joined at the
        # boundary. See eval_contract.phase_join.
        _local_error = None
        local_results, skipped_no_features = [], []
        try:
            local_results, skipped_no_features = eval_utils.run_generation_loop(
                model, self.processor, dataloader, model.device,
                log_prefix=f"[GenEval step {global_step}] [rank {rank}]",
            )
        except Exception as _e:  # noqa: BLE001 - re-raised on every rank below
            _local_error = _e
        finally:
            model.config.use_cache = _orig_use_cache
            if hasattr(model, 'for_training'):
                model.for_training()
            else:
                model.train()
                if getattr(self.training_args, 'gradient_checkpointing', False):
                    if hasattr(model, 'gradient_checkpointing_enable'):
                        model.gradient_checkpointing_enable()
                    else:
                        for module in model.modules():
                            if hasattr(module, "gradient_checkpointing"):
                                module.gradient_checkpointing = True
            m = model
            while hasattr(m, "model"):
                if hasattr(m, "_flag_for_generation"):
                    del m._flag_for_generation
                m = m.model
            if hasattr(m, "_flag_for_generation"):
                del m._flag_for_generation
            for _, module in model.named_modules():
                for attr in ("_cache", "_flex_attention_cache"):
                    if hasattr(module, attr):
                        try:
                            delattr(module, attr)
                        except Exception:
                            pass
            gc.collect()
            torch.cuda.empty_cache()
            _alloc = torch.cuda.memory_allocated() / 1024**3
            _res = torch.cuda.memory_reserved() / 1024**3
            print(f"[GenEval] Post-cleanup GPU (rank {rank}): {_alloc:.2f} GiB allocated, {_res:.2f} GiB reserved")

        # ------------------------------------------------------------------
        # Gather results from all ranks to rank 0 for metric computation
        # ------------------------------------------------------------------
        # THE AGREEMENT POINT, before the gather below. Raises on every rank
        # when any one of them failed in the loop above; the caller's handler
        # records it and stops the run.
        eval_contract.phase_join(
            _local_error, step=global_step, phase=phase,
            evaluator=self.CONTRACT_NAME,
            where=f"in the generation loop for {label!r}",
        )

        if is_distributed:
            # Gather results from all ranks via all_gather_object (uses pickle internally)
            all_results = [None] * world_size
            torch.distributed.all_gather_object(all_results, local_results)
            all_skipped = [None] * world_size
            torch.distributed.all_gather_object(all_skipped, skipped_no_features)
            if rank == 0:
                # Flatten gathered results (order: rank 0, rank 1, ...)
                local_results = []
                for rank_results in all_results:
                    local_results.extend(rank_results)
                skipped_no_features = []
                for rank_skipped in all_skipped:
                    skipped_no_features.extend(rank_skipped)
                print(f"[GenEval] Gathered {len(local_results)} results from {world_size} ranks")
        # ------------------------------------------------------------------

        # Only rank 0 computes metrics, saves, and returns stats
        self._last_eval_counts = (len(local_results), len(skipped_no_features))
        if rank != 0:
            return {}

        # Scoring nothing is an infrastructure failure. A model that answers
        # everything wrong still produces results; a loop that produced no
        # result at all produced no measurement.
        if not local_results:
            raise eval_contract.EvaluationExecutionError(
                f"eval of {label!r} at step {global_step} scored 0 items "
                f"({len(skipped_no_features)} dropped for missing features)."
            )

        exp_name = os.path.join(self.output_dir, "gen_eval", label, f"step_{global_step}")
        tracker = eval_utils.MetricTracker(
            benchmarking=True, exp_name=exp_name, dataset_name=tracker_name,
            pano_grounding_format=bool(getattr(self.data_args, "pano_grounding_format", False)),
        )

        for count, result in enumerate(local_results):
            tracker.update(
                count=count,
                scene_id=result["scene_id"],
                question=result["question"],
                prediction=result["prediction"],
                ground_truths=result["ground_truths"],
                question_type=result["question_type"],
            )
            if (count + 1) % 100 == 0:
                stats = {m: (sum(v) / len(v) if v else 0) for m, v in tracker.metrics_acc.items()}
                reg = tracker.regression_stats()
                grnd = tracker.grounding_stats()
                print("\n" + "=" * 40)
                print(f"METRICS SUMMARY AT SAMPLE {count + 1}")
                if grnd:
                    print(f"Acc@0.25: {grnd['grnd_Acc@0.25']:.1%} | Acc@0.1: {grnd['grnd_Acc@0.1']:.1%} | Acc@0.05: {grnd['grnd_Acc@0.05']:.1%}")
                    print(f"Mean IoU: {grnd['grnd_mean_IoU']:.4f} | Parse: {grnd['grnd_parse_rate']:.0%}")
                    if "grnd_center_dist_mean" in grnd:
                        print(f"Center dist: {grnd['grnd_center_dist_mean']:.2f}m | <1m: {grnd['grnd_center_within_1m']:.1%} | <2m: {grnd['grnd_center_within_2m']:.1%}")
                else:
                    _inter_official = None
                    if tracker.per_type_scores:
                        if tracker_name == "vsi_bench":
                            _inter_official = tracker._compute_vsibench_overall()
                        else:
                            _means = [sum(v) / len(v) for v in tracker.per_type_scores.values() if v]
                            _inter_official = sum(_means) / len(_means) if _means else None
                    if _inter_official is not None:
                        _all_q = tracker.all_questions_average()
                        print(f"Official Avg: {_inter_official:.4f}" + (f" | All-Q Avg: {_all_q:.4f}" if _all_q is not None else ""))
                    else:
                        print(f"METEOR: {stats.get('METEOR', 0):.4f} | EM@1: {stats.get('EM@1', 0):.4f}")
                    if reg:
                        print(f"MAE: {reg['reg_MAE']:.3f}m | r: {reg['reg_pearson_r']:.4f} | ≤1m: {reg['reg_within_1.0m']:.1%}")
                print("=" * 40 + "\n")
                tracker.save(step=count + 1)


        tracker.save()
        final_stats = {m: (sum(v) / len(v) if v else 0) for m, v in tracker.metrics_acc.items()}

        reg = tracker.regression_stats()
        if reg:
            final_stats.update(reg)

        if tracker.per_type_scores:
            for display_name, key, scores_list in tracker._ordered_type_items():
                final_stats[f"type/{display_name}"] = sum(scores_list) / len(scores_list) if scores_list else 0.0
            if tracker_name == "vsi_bench":
                final_stats["official_overall"] = tracker._compute_vsibench_overall()
            else:
                means = [sum(v) / len(v) for v in tracker.per_type_scores.values() if v]
                final_stats["official_overall"] = sum(means) / len(means) if means else 0.0

            all_q_avg = tracker.all_questions_average()
            if all_q_avg is not None:
                final_stats["all_questions_average"] = all_q_avg
                final_stats["all_questions_count"] = sum(len(v) for v in tracker.per_type_scores.values())

        grnd = tracker.grounding_stats()

        print("\n" + "=" * 40)
        print(f"[GenEval] FINAL METRICS @ training step {global_step} — {label}")
        if grnd:
            print(f"--- 3D Grounding ---")
            print(f"  Acc@0.25: {grnd['grnd_Acc@0.25']:.1%}  Acc@0.5: {grnd['grnd_Acc@0.5']:.1%}")
            print(f"  Acc@0.1:  {grnd['grnd_Acc@0.1']:.1%}  Acc@0.05: {grnd['grnd_Acc@0.05']:.1%}")
            print(f"  Mean IoU: {grnd['grnd_mean_IoU']:.4f}  Median IoU: {grnd['grnd_median_IoU']:.4f}")
            if "grnd_center_dist_mean" in grnd:
                print(f"  Center dist: {grnd['grnd_center_dist_mean']:.2f}m (med: {grnd['grnd_center_dist_median']:.2f}m)")
                print(f"  Within 0.5m: {grnd['grnd_center_within_0.5m']:.1%}  1m: {grnd['grnd_center_within_1m']:.1%}  2m: {grnd['grnd_center_within_2m']:.1%}")
            print(f"  Parsed: {grnd['grnd_n']}/{grnd['grnd_total']} ({grnd['grnd_parse_rate']:.0%})")
            final_stats.update(grnd)
            final_stats["official_overall"] = grnd["grnd_Acc@0.25"]
        else:
            if "official_overall" in final_stats:
                print(f"Official Overall: {final_stats['official_overall']:.4f}")
            if "all_questions_average" in final_stats:
                print(
                    f"All-Questions Avg (micro): {final_stats['all_questions_average']:.4f} "
                    f"(n={int(final_stats.get('all_questions_count', 0))})"
                )
            if reg:
                print(f"--- Regression ---")
                print(f"  MAE: {reg['reg_MAE']:.3f}m | MedAE: {reg['reg_MedAE']:.3f}m | Pearson r: {reg['reg_pearson_r']:.4f}")
                print(f"  ≤0.5m: {reg['reg_within_0.5m']:.1%} | ≤1.0m: {reg['reg_within_1.0m']:.1%} | ≤2.0m: {reg['reg_within_2.0m']:.1%}")
                print(f"  Pred μ={reg['reg_pred_mean']:.2f} σ={reg['reg_pred_std']:.2f} | GT μ={reg['reg_gt_mean']:.2f}")
                # Log learned depth embedding scale if present (canvas only)
                if not getattr(self.data_args, "vanilla_qwen3vl", False):
                    _inner = get_inner_3d_model(model)
                    if hasattr(_inner, "depth_embed_log_scale"):
                        _ls = _inner.depth_embed_log_scale.item()
                        print(f"  depth_embed_scale: exp({_ls:.4f}) = {math.exp(_ls):.4f}")
            if tracker.per_type_scores:
                print("--- Per-Type ---")
                for display_name, _, sl in tracker._ordered_type_items():
                    print(f"  {display_name.ljust(12)}: {sum(sl)/len(sl):.4f} (n={len(sl)})")
            print(f"METEOR: {final_stats.get('METEOR', 0):.4f} | EM@1: {final_stats.get('EM@1', 0):.4f}")
        print(f"Results saved to: {exp_name}")
        print("=" * 40 + "\n")

        return final_stats

    # ------------------------------------------------------------------
    # Multi-dataset eval orchestrator
    # ------------------------------------------------------------------
    def _run_all_evals(self, model, global_step: int, is_final: bool = False,
                       phase: str = None):
        """Run _run_eval for every dataset in self.eval_dataset_names.

        Logs structured wandb keys {dataset}/{metric} for all metrics
        including per-type (filtered: NLP metrics skipped for vsi_bench,
        regression metrics skipped for sqa3d).  Returns a combined stats dict
        with "combined_official" = mean of official_overall across datasets
        (used by _maybe_save_best).
        """
        cache_dict = self._final_dataloaders if is_final else self._dataloaders
        num_samples = self.final_num_samples if is_final else self.num_samples
        phase = phase or (
            eval_contract.PHASE_FINAL if is_final else eval_contract.PHASE_SCHEDULED
        )

        combined_stats = {}
        official_scores = []
        all_q_weighted_sum = 0.0
        all_q_total_count = 0
        self._populations = {}

        for dataset_name in self.eval_dataset_names:
            label = dataset_name if dataset_name is not None else "default"
            ds_num_samples = self._per_dataset_num_samples.get(dataset_name, num_samples)
            print(f"\n[GenEval] Evaluating dataset: '{label}' ({ds_num_samples} samples)")
            # An OOM, a dataset that resolves to nothing, or a scorer that
            # raises used to be a printed warning and a `continue`, which let a
            # run train to the end with no numbers at all. It is terminal now.
            # join_failure makes every rank take the same branch, so the run
            # dies as an evaluation failure rather than an NCCL timeout.
            local_error = None
            per_dataset_stats = {}
            try:
                loader = self._get_dataloader(ds_num_samples, dataset_name, cache_dict)
                per_dataset_stats = self._run_eval(
                    model, global_step, dataset_name=dataset_name,
                    dataloader=loader, phase=phase,
                )
            except Exception as e:  # noqa: BLE001 - re-raised below on all ranks
                local_error = e
                for p in model.parameters():
                    p.grad = None
                gc.collect()
                torch.cuda.empty_cache()
            joined = eval_contract.join_failure(local_error)
            if joined:
                eval_contract.fail_evaluation(
                    self.output_dir, step=global_step, phase=phase,
                    evaluator=self.CONTRACT_NAME,
                    cause=(local_error or RuntimeError(joined)),
                )

            scored, skipped = self._last_eval_counts
            self._populations[label] = {
                "requested": int(ds_num_samples),
                "resolved": int(
                    self._resolved_sizes.get((dataset_name, int(ds_num_samples)), 0)
                ),
                "scored": int(scored),
                "skipped_no_features": int(skipped),
            }

            # Filter metrics per dataset to avoid cluttering wandb
            _nlp_keys = {"METEOR", "ROUGE1", "ROUGE2", "ROUGEL", "CIDEr", "EM@1", "EM@R1"}
            _reg_keys = {k for k in per_dataset_stats if k.startswith("reg_")}
            for metric_key, value in per_dataset_stats.items():
                # VSI-Bench: skip NLP metrics (it's MCQ + numerical estimation)
                if dataset_name == "vsi_bench" and metric_key in _nlp_keys:
                    continue
                # SQA3D: skip regression metrics (it's QA, not numerical)
                if dataset_name in ("sqa3d", "sqa3d_agent_pose") and metric_key in _reg_keys:
                    continue
                combined_stats[f"{label}/{metric_key}"] = value

            # Geometric probe is a training-signal diagnostic, not a benchmark,
            # so exclude it from the combined best-checkpoint metric.
            if dataset_name != "geometric_probing":
                if "official_overall" in per_dataset_stats:
                    official_scores.append(per_dataset_stats["official_overall"])
                if "all_questions_average" in per_dataset_stats:
                    n_q = int(per_dataset_stats.get("all_questions_count", 0))
                    if n_q > 0:
                        all_q_weighted_sum += per_dataset_stats["all_questions_average"] * n_q
                        all_q_total_count += n_q

        # Cross-dataset aggregate for checkpoint selection (probe excluded above)
        if official_scores:
            combined_stats["combined_official"] = sum(official_scores) / len(official_scores)
        else:
            meteor_vals = [
                v for k, v in combined_stats.items()
                if k.endswith("/METEOR") and not k.startswith("geometric_probing/")
            ]
            combined_stats["combined_official"] = (
                sum(meteor_vals) / len(meteor_vals) if meteor_vals else 0.0
            )

        if all_q_total_count > 0:
            combined_stats["combined_all_questions_average"] = all_q_weighted_sum / all_q_total_count
            combined_stats["combined_all_questions_count"] = all_q_total_count

        # Log everything to wandb. Don't pass step= here: on_save fires after
        # the Trainer has already committed step N, so passing step=N triggers
        # the "step N < current step N+1" monotonicity error and the data is
        # silently dropped. Letting wandb auto-advance means the metrics appear
        # at N+1 in the UI, which is close enough.
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(combined_stats)
        except ImportError:
            pass

        # Durable record of this evaluation: metrics AND the population behind
        # them. The completion marker is written from this ledger, so a score
        # with no accounting cannot make a run read as finished.
        if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0:
            _spec = self.evaluation_contract_spec()
            eval_contract.record_evaluation(
                self.output_dir,
                step=global_step,
                phase=phase,
                evaluator=self.CONTRACT_NAME,
                populations=dict(self._populations),
                metrics=combined_stats,
                identity=_spec.identity(is_final=is_final),
                checkpoint=eval_contract.checkpoint_identity(
                    self.output_dir, step=global_step, lineage=self.lineage
                ),
                spec=_spec,
            )

        return combined_stats

    # ------------------------------------------------------------------
    # Trainer hooks
    # ------------------------------------------------------------------
    def _maybe_save_best(self, control, final_stats, global_step, model=None):
        """If the tracked metric improved, save to a dedicated best_checkpoint/ dir
        that is never pruned by save_total_limit, and flag Trainer to save too.

        Must be called on ALL ranks because model.save_pretrained() with DeepSpeed
        requires collective operations. Rank 0 decides whether to save, then
        broadcasts the decision so all ranks participate in save_pretrained().
        """
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        rank = torch.distributed.get_rank() if is_distributed else 0

        # Rank 0 decides; broadcast decision to all ranks
        should_save = False
        if rank == 0:
            score = final_stats.get(self.best_metric, 0.0)
            if score > self._best_score:
                self._best_score = score
                self._best_step = global_step
                should_save = True
                print(
                    f"\n[GenEval] ★ New best {self.best_metric}: {score:.4f} "
                    f"(step {global_step})"
                )
            else:
                print(
                    f"[GenEval] {self.best_metric} {score:.4f} did not improve "
                    f"(best: {self._best_score:.4f} @ step {self._best_step})."
                )

        if is_distributed:
            save_flag = [should_save]
            torch.distributed.broadcast_object_list(save_flag, src=0)
            should_save = save_flag[0]

        if not should_save:
            return control

        best_dir = os.path.join(self.output_dir, "best_checkpoint")

        # All ranks participate in save_pretrained (required by DeepSpeed)
        if model is not None:
            os.makedirs(best_dir, exist_ok=True)
            try:
                model.save_pretrained(best_dir)
                if rank == 0:
                    if not getattr(self.data_args, "vanilla_qwen3vl", False):
                        save_3d_embeddings(model, best_dir, log_prefix="[GenEval] ★")
                    print(f"[GenEval] ★ Best model saved to: {best_dir}")
            except Exception as e:
                if rank == 0:
                    print(f"[GenEval] Warning: could not save best model: {e}")

        # Only rank 0 writes metadata JSON
        if rank == 0:
            import json
            best_info = {
                "best_step": global_step,
                self.best_metric: final_stats.get(self.best_metric, 0.0),
                "all_metrics": final_stats,
            }
            with open(os.path.join(self.output_dir, "best_gen_eval.json"), "w") as f:
                json.dump(best_info, f, indent=4)
            with open(os.path.join(best_dir, "best_gen_eval.json"), "w") as f:
                json.dump(best_info, f, indent=4)

            print(
                f"[GenEval] ★ To load:  PeftModel.from_pretrained(base_model, \"{best_dir}\")\n"
            )
        control.should_save = True
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        """Run a pipeline sanity check before training starts."""
        rank = int(os.environ.get("RANK", 0))

        # MANDATORY BASELINE. Every curve has a zero point, and the baseline is
        # also what proves before the first optimizer step that the evaluation
        # population resolves at all: it builds the eval dataloaders and scores
        # them on the freshly-assembled model. On a warm start or LoRA merge it
        # doubles as the load check (stage-2 step 0 should read stage-1 final).
        #
        # On resume the baseline is required only when the APPLICABLE one is
        # missing from the ledger; a recorded baseline is reused, not redone.
        # The decision is taken on rank 0 and broadcast because all ranks must
        # enter _run_all_evals together (it runs all_gather collectives).
        # Best-checkpoint state is intentionally left untouched here.
        #
        # ONECANVAS_EVAL_AT_STEP0 used to gate this as an opt-in. It is now the
        # contract, and the variable is accepted and ignored.
        # Matched on the MEASUREMENT (population + declared inputs + scoring),
        # not on the sample count, and deliberately not on the checkpoint: the
        # baseline is the curve's zero point, so a resume at step 5000 reuses
        # the step-0 record rather than manufacturing a second one.
        _need_baseline = rank == 0 and self._is_this_run_s_evaluator() and eval_contract.find_recorded_evaluation(
            self.output_dir,
            phase=eval_contract.PHASE_BASELINE,
            evaluator=self.CONTRACT_NAME,
            identity=self.evaluation_contract_spec().identity(is_final=False),
        ) is None
        _need_baseline = eval_contract.broadcast_decision(_need_baseline)
        if _need_baseline:
            if rank == 0:
                print(
                    f"\n[GenEval] baseline evaluation at step "
                    f"{state.global_step} (evaluation contract)"
                )
            self._run_all_evals(
                kwargs.get("model"), state.global_step, is_final=False,
                phase=eval_contract.PHASE_BASELINE,
            )
        elif rank == 0 and self._is_this_run_s_evaluator():
            print(
                "[GenEval] baseline already recorded for this evaluation "
                "configuration; reusing it"
            )

        pano_fmt = bool(getattr(self.data_args, "pano_grounding_format", False))
        metric_json_fmt = bool(getattr(self.data_args, "metric_json_grounding_format", True))

        if pano_fmt:
            # Verify the pano-format `{"bbox_3d": [u, v, depth, sx, sy, sz], "label": ...}`
            # JSON survives tokenize -> decode -> parse. u, v are integers in
            # [0, PANO_COORD_SCALE); depth and sizes are metric meters.
            from utils.bbox import format_pano_bbox_json, parse_pano_3d_bbox
            expected = (500, 500, 2.5, 0.6, 1.22, 1.94)
            test_ans = format_pano_bbox_json(expected, label="cabinet")
            ids = self.processor.tokenizer.encode(test_ans, add_special_tokens=False)
            decoded = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
            cleaned = eval_utils.clean_generated_text(decoded)
            parsed = parse_pano_3d_bbox(cleaned)
            ok = (
                parsed is not None
                and len(parsed) == 6
                and abs(parsed[0] - expected[0]) <= 1
                and abs(parsed[1] - expected[1]) <= 1
                and all(abs(parsed[i] - expected[i]) < 1e-2 for i in range(2, 6))
            )
            if not ok:
                msg = (
                    f"[GenEval] SANITY CHECK FAILED: pano-format bbox_3d round-trip broken!\n"
                    f"  Input:    {test_ans!r}\n  Decoded:  {cleaned!r}\n"
                    f"  Parsed:   {parsed}\n  Expected: {expected}\n"
                    f"  Check that format_pano_bbox_json whitespace is tokenizer-friendly,"
                    f" and that parse_pano_3d_bbox reads the output."
                )
                if rank == 0:
                    print(msg)
                raise RuntimeError(msg)
            if rank == 0:
                print(f"[GenEval] Pipeline sanity check passed (pano bbox_3d round-trip OK)")
            return control

        if metric_json_fmt:
            # Verify the metric-JSON format `{"bbox_3d": [cx, cy, cz, sx, sy, sz], "label": ...}`
            # survives tokenize -> decode -> parse. All six values are signed metric meters
            # in the scene-centered axis-aligned frame. Parser is parse_3d_bbox (which
            # delegates to _parse_3d_bbox_json for 6-value bbox_3d entries).
            from utils.bbox import format_metric_bbox_json
            expected = (-1.22, -3.04, 0.61, 0.60, 1.22, 1.94)
            test_ans = format_metric_bbox_json(expected, label="cabinet")
            ids = self.processor.tokenizer.encode(test_ans, add_special_tokens=False)
            decoded = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
            cleaned = eval_utils.clean_generated_text(decoded)
            parsed = eval_utils.parse_3d_bbox(cleaned)
            ok = (
                parsed is not None
                and len(parsed) == 6
                and all(abs(parsed[i] - expected[i]) < 1e-2 for i in range(6))
            )
            if not ok:
                msg = (
                    f"[GenEval] SANITY CHECK FAILED: metric-JSON bbox_3d round-trip broken!\n"
                    f"  Input:    {test_ans!r}\n  Decoded:  {cleaned!r}\n"
                    f"  Parsed:   {parsed}\n  Expected: {expected}\n"
                    f"  Check format_metric_bbox_json whitespace and that parse_3d_bbox"
                    f" reads the JSON output."
                )
                if rank == 0:
                    print(msg)
                raise RuntimeError(msg)
            if rank == 0:
                print(f"[GenEval] Pipeline sanity check passed (metric-JSON bbox_3d round-trip OK)")
            return control

        # Verify box tokens + bare metric floats survive the tokenize -> decode -> parse
        # round-trip. Format is bare meters in scene-centered axis-aligned frame; see
        # docs/GROUNDING.md. Use signed sub-meter floats to exercise the regex too.
        expected = (-1.22, -3.04, 0.61, 0.60, 1.22, 1.94)
        test_ans = f"<|box_start|>({', '.join(f'{c:.2f}' for c in expected)})<|box_end|>"
        ids = self.processor.tokenizer.encode(test_ans, add_special_tokens=False)
        decoded = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
        cleaned = eval_utils.clean_generated_text(decoded)
        parsed = eval_utils.parse_3d_bbox(cleaned)
        ok = (
            parsed is not None
            and len(parsed) == 6
            and all(abs(p - e) < 1e-3 for p, e in zip(parsed, expected))
        )
        if not ok:
            msg = (
                f"[GenEval] SANITY CHECK FAILED: box token round-trip broken!\n"
                f"  Input:    {test_ans!r}\n  Decoded:  {cleaned!r}\n"
                f"  Parsed:   {parsed}\n  Expected: {expected}\n"
                f"  Grounding metrics will be wrong. Check that <|box_start|>/<|box_end|>"
                f" survive decode (skip_special_tokens=False), that clean_generated_text"
                f" preserves them, and that parse_3d_bbox handles signed metric floats."
            )
            if rank == 0:
                print(msg)
            raise RuntimeError(msg)
        if rank == 0:
            print(f"[GenEval] Pipeline sanity check passed (box token round-trip OK)")
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        # Save depth/camera embedding weights alongside each checkpoint so that
        # crash recovery and warm-starts always load a consistent geometry encoder.
        # save_3d_embeddings is a file write with no collective ops, so rank 0 only.
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        rank = torch.distributed.get_rank() if is_distributed else 0
        # Vanilla baseline has no 3D embeddings to save (stock Qwen3-VL).
        if rank == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt_dir):
                if model is not None and not getattr(self.data_args, "vanilla_qwen3vl", False):
                    save_3d_embeddings(model, ckpt_dir)
                # Mark the checkpoint complete NOW, before gen_eval below: a
                # crash during eval must not make auto-resume treat this
                # checkpoint as torn. finalize refuses the sentinel if any
                # file is 0 bytes (the 2026-08-05 ENOSPC signature).
                finalize_checkpoint(ckpt_dir, step=state.global_step)

        # Run gen_eval AFTER the checkpoint is saved, so a crash during eval
        # does not lose the checkpoint for this step.
        is_eval_step = self.eval_gen_steps > 0 and state.global_step % self.eval_gen_steps == 0
        if is_eval_step:
            print(f"\n[GenEval] Triggered at training step {state.global_step} ({self.num_samples} samples)")
            combined_stats = self._run_all_evals(model, state.global_step, is_final=False)
            control = self._maybe_save_best(
                control, combined_stats, state.global_step, model=model
            )
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # A COMPLETE CHECKPOINT AT EVERY EVALUATION BOUNDARY, whatever
        # save_steps is, plus one at the FINAL step. save_steps is resume
        # granularity and gen_eval_steps is curve resolution; they coincide
        # only by arithmetic accident, and max_steps is rarely a multiple of
        # either. Without the final one, on_train_end scores weights that no
        # checkpoint on disk identifies, so the result names no model and is
        # not reusable. See eval_contract.wants_checkpoint.
        if not self._is_this_run_s_evaluator():
            return control
        if eval_contract.wants_checkpoint(
            int(state.global_step),
            cadence=int(self.eval_gen_steps),
            max_steps=int(getattr(state, "max_steps", 0) or 0),
            stop_step=int(getattr(args, "continuation_stop_step", 0) or 0),
        ):
            control.should_save = True
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Final evaluation WHEN NEEDED, reusing a matching completed result.

        Needed means the last step carries no recorded result. A run whose
        schedule ended exactly on the cadence already scored this step in
        on_save, so scoring it again buys nothing; a run that stopped between
        cadence points would otherwise finish with its newest number thousands
        of steps stale, which is the case the completion marker refuses.
        """
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if not self._is_this_run_s_evaluator():
            return control
        is_final = self.final_num_samples > 0

        _needed = rank == 0 and eval_contract.find_recorded_evaluation(
            self.output_dir,
            step=state.global_step,
            evaluator=self.CONTRACT_NAME,
            identity=self.evaluation_contract_spec().identity(is_final=is_final),
            checkpoint_id=eval_contract.checkpoint_identity(
                self.output_dir, step=state.global_step, lineage=self.lineage
            )["id"],
        ) is None
        _needed = eval_contract.broadcast_decision(_needed)
        if not _needed:
            if rank == 0:
                print(
                    f"[GenEval] step {state.global_step} already has a matching "
                    "recorded evaluation; no final eval needed"
                )
            return control

        n = self.final_num_samples if is_final else self.num_samples
        print(f"\n[GenEval] Running FINAL generation eval (step {state.global_step}, {n} samples)...")
        combined_stats = self._run_all_evals(
            model, state.global_step, is_final=is_final,
            phase=eval_contract.PHASE_FINAL,
        )
        control = self._maybe_save_best(
            control, combined_stats, state.global_step, model=model
        )
        return control


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    # Qwen3.5: .visual and .language_model live under model.model (Qwen3_5Model),
    # not directly on the ForConditionalGeneration class like Qwen3-VL.
    inner = model.model if hasattr(model.model, "visual") else model

    if hasattr(inner, "visual"):
        if model_args.tune_mm_vision:
            for n, p in inner.visual.named_parameters():
                p.requires_grad = True
        else:
            for n, p in inner.visual.named_parameters():
                p.requires_grad = False

        if model_args.tune_mm_mlp:
            for n, p in inner.visual.merger.named_parameters():
                p.requires_grad = True
        else:
            for n, p in inner.visual.merger.named_parameters():
                p.requires_grad = False

    if hasattr(inner, "language_model"):
        if model_args.tune_mm_llm:
            for n, p in inner.language_model.named_parameters():
                p.requires_grad = True
            model.lm_head.requires_grad = True
        else:
            for n, p in inner.language_model.named_parameters():
                p.requires_grad = False
            model.lm_head.requires_grad = False


from torch.utils.data import WeightedRandomSampler


class FixedCompositionSampler(torch.utils.data.Sampler):
    """Every optimizer step holds exactly `probe_slots` curriculum rows.

    The index stream is cut into blocks of the effective batch. Accelerate
    deals batch j of the stream to rank j % world_size, and gradient
    accumulation groups consecutive batches, so one block is one optimizer
    step. In each block `probe_slots` random positions draw a curriculum row
    (the concatenated `geometric_probing` part), and the rest walk a shuffled
    permutation of the QA rows WITHOUT replacement, so one epoch of this
    sampler shows every QA row exactly once. The seed depends only on the run
    seed and the epoch, so every rank draws the same stream and a resume
    replays it.
    """

    def __init__(self, n_qa, n_probe, block, probe_slots, seed):
        if not 0 < probe_slots < block:
            raise ValueError(
                f"ONECANVAS_BATCH_PROBE_SLOTS={probe_slots} must be between 1 "
                f"and the effective batch {block} minus one")
        self.n_qa, self.n_probe = int(n_qa), int(n_probe)
        self.block, self.probe_slots = int(block), int(probe_slots)
        self.seed, self.epoch = int(seed), 0
        self.blocks = -(-self.n_qa // (self.block - self.probe_slots))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.blocks * self.block

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        per = self.block - self.probe_slots
        order = torch.randperm(self.n_qa, generator=g).tolist()
        # Pad the last block from a second permutation, never by repeating a
        # row of the first inside one epoch's ordering by accident.
        order += torch.randperm(self.n_qa, generator=g).tolist()[
            :self.blocks * per - self.n_qa]
        for b in range(self.blocks):
            qa = iter(order[b * per:(b + 1) * per])
            probe_at = set(torch.randperm(self.block, generator=g)[
                :self.probe_slots].tolist())
            for pos in range(self.block):
                if pos in probe_at:
                    yield self.n_qa + int(torch.randint(
                        self.n_probe, (1,), generator=g))
                else:
                    yield next(qa)


def _fixed_composition_sampler(args, dataset, probe_slots):
    from torch.utils.data import ConcatDataset
    from onecanvas.data.spatial_pretraining import SpatialPretrainingDataset
    parts = getattr(dataset, "datasets", None)
    if not (isinstance(dataset, ConcatDataset) and len(parts) == 2
            and isinstance(parts[1], SpatialPretrainingDataset)):
        raise ValueError(
            "ONECANVAS_BATCH_PROBE_SLOTS needs the QA rows concatenated with "
            "geometric_probing (dataset_use '<qa>@w,geometric_probing@w')")
    block = (args.per_device_train_batch_size * args.world_size
             * args.gradient_accumulation_steps)
    sampler = FixedCompositionSampler(len(parts[0]), len(parts[1]), block,
                                      probe_slots, args.seed)
    print(f"[data] fixed batch composition: {probe_slots} geometric_probing + "
          f"{block - probe_slots} QA rows per optimizer step of {block}; "
          f"{len(parts[0])} QA rows, one epoch = {sampler.blocks} steps, "
          f"QA drawn without replacement", flush=True)
    return sampler


class WeightedTrainer(Trainer):
    """Uses WeightedRandomSampler when the train dataset has sample_weights (@N syntax).
    Falls back to the default sampler for unweighted runs and plain ConcatDataset."""

    def save_model(self, output_dir=None, _internal_call=False):
        """Skip the full base-model state-dict clone that causes checkpoint-save OOMs.

        The default DeepSpeed path gathers all frozen base weights to CPU (~16 GB
        per rank, ~64 GB across 4 ranks for ZeRO-2) before PEFT filters them back
        down to the adapter. Replacing it with a direct PEFT save_pretrained cuts
        the spike to a few MB — same as what _maybe_save_best already does.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.is_deepspeed_enabled:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call)

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        probe_slots = int(os.environ.get("ONECANVAS_BATCH_PROBE_SLOTS") or 0)
        if probe_slots:
            return _fixed_composition_sampler(self.args, train_dataset, probe_slots)
        weights = getattr(train_dataset, "sample_weights", None)
        if not weights:
            return super()._get_train_sampler(train_dataset)
        t = torch.tensor(weights, dtype=torch.double)
        return WeightedRandomSampler(t, num_samples=len(t), replacement=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Adds the optional depth/visual ratio penalty stashed by the
        geometry-embedding mixin onto the standard loss. No-op when
        ``--depth_ratio_penalty_beta`` is 0 (the default)."""
        if num_items_in_batch is not None:
            out = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                       num_items_in_batch=num_items_in_batch)
        else:
            out = super().compute_loss(model, inputs, return_outputs=return_outputs)
        loss = out[0] if return_outputs else out
        # Cache the inner-3d-model reference on the trainer so we don't walk
        # the wrapper chain every step. The walk traverses both .model (PEFT/HF)
        # and .module (DeepSpeedEngine) — see get_inner_3d_model.
        if getattr(self, "_cached_inner_3d", None) is None:
            try:
                self._cached_inner_3d = get_inner_3d_model(model)
            except ValueError:
                self._cached_inner_3d = False  # mark resolved-but-none
        inner = self._cached_inner_3d
        if inner is not False:
            extra = getattr(inner, "_last_depth_ratio_penalty", None)
            if extra is not None:
                loss = loss + extra.to(loss.dtype).to(loss.device)
                inner._last_depth_ratio_penalty = None
        return (loss, out[1]) if return_outputs else loss

    def create_optimizer(self):
        """Optional per-group LR and weight-decay handling for depth params.

        Defers to the stock HF Trainer path only when ``depth_lr_multiplier``
        is 1.0 *and* ``depth_no_weight_decay`` is False. Otherwise splits the
        usual decay/no-decay groups into depth/non-depth subgroups so that:
          - the depth subgroups get ``depth_lr_multiplier`` x the base LR (so a
            warm-started depth MLP can rotate to the stage-2 equilibrium fast), and
          - depth params can be excluded from weight decay (``depth_no_weight_decay``).
        The latter matters because weight decay is an unconditional pull toward
        zero weights, amplified ~depth_lr_multiplier-fold by the per-group LR
        (AdamW couples decay to the group LR), and it is the only force that can
        erode depth below the ratio penalty's free budget and re-trigger collapse.
        Non-depth params are treated exactly as the stock path.
        """
        if self.optimizer is not None:
            return self.optimizer

        depth_lr_mult = float(getattr(self.args, "depth_lr_multiplier", 1.0))
        depth_no_wd = bool(getattr(self.args, "depth_no_weight_decay", False))
        if depth_lr_mult == 1.0 and not depth_no_wd:
            return super().create_optimizer()

        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)

        _DEPTH_KEYS = (
            "depth_embedding", "depth_fourier", "depth_loc_proj",
            "depth_cartesian", "depth_embed_log_scale",
        )
        def _is_depth(name: str) -> bool:
            return any(k in name for k in _DEPTH_KEYS)

        base_lr = self.args.learning_rate
        depth_lr = base_lr * depth_lr_mult
        depth_wd = 0.0 if depth_no_wd else self.args.weight_decay

        named = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
        groups = [
            {
                "params": [p for n, p in named if (n in decay_parameters) and not _is_depth(n)],
                "weight_decay": self.args.weight_decay,
                "lr": base_lr,
            },
            {
                "params": [p for n, p in named if (n not in decay_parameters) and not _is_depth(n)],
                "weight_decay": 0.0,
                "lr": base_lr,
            },
            {
                "params": [p for n, p in named if (n in decay_parameters) and _is_depth(n)],
                "weight_decay": depth_wd,
                "lr": depth_lr,
            },
            {
                "params": [p for n, p in named if (n not in decay_parameters) and _is_depth(n)],
                "weight_decay": 0.0,
                "lr": depth_lr,
            },
        ]
        groups = [g for g in groups if len(g["params"]) > 0]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)

        if int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0:
            n_depth = sum(1 for n, _ in named if _is_depth(n))
            n_base = sum(1 for n, _ in named if not _is_depth(n))
            print(f"[optimizer] depth_lr_multiplier={depth_lr_mult}  base_lr={base_lr:.2e}  depth_lr={depth_lr:.2e}")
            print(f"[optimizer] depth_no_weight_decay={depth_no_wd}  depth_wd={depth_wd}  (global weight_decay={self.args.weight_decay})")
            print(f"[optimizer] {n_depth} depth params (scaled LR), {n_base} non-depth params (base LR)")
        return self.optimizer


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --ddp_timeout reaches the DEFAULT process group only (accelerate passes it
    # to init_process_group). DeepSpeed's ZeRO groups come from
    # torch.distributed.new_group(ranks) with no timeout, which torch fills from
    # this module default (10 min for NCCL), not from the default group. The
    # gradient allreduce runs on one of those, so a stall longer than ten
    # minutes on one rank killed insitu_v8 at step 8144 with the default group
    # sitting at 7200 s. Set the module default so every later group inherits it.
    import torch.distributed.distributed_c10d as _c10d
    _c10d.default_pg_nccl_timeout = timedelta(seconds=training_args.ddp_timeout)

    # Downgrade flash_attention_2 -> sdpa when flash_attn isn't installed
    # (flash-attn is base.sh's default but not a declared dep).
    from model_adapters.attention import resolve_attn_implementation
    model_args.attn_implementation = resolve_attn_implementation(
        model_args.attn_implementation)

    # Mirror vanilla flag onto data_args so the dataset's __getitem__ can read
    # it via getattr(self.data_args, ...). HfArgumentParser parses the
    # --vanilla_qwen3vl CLI flag only into ModelArguments where it's defined.
    data_args.vanilla_qwen3vl = model_args.vanilla_qwen3vl

    transformers.set_seed(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    # Fail at launch, not 2000 steps in, when the output volume is nearly
    # full (2026-08-05: gimli hit 0 bytes and killed three runs mid-save).
    assert_free_space_at_launch(training_args.output_dir)

    # Dump all resolved args (including defaults) for reproducibility.
    # Writes typed JSON: bools/ints/floats/None come through with their
    # native Python types, so run_benchmarks.py --from-config can round-trip
    # them without manual string parsing.
    if local_rank in (0, -1):
        import dataclasses
        def _jsonable(v):
            if v is None or isinstance(v, (bool, int, float, str)):
                return v
            if isinstance(v, (list, tuple)):
                return [_jsonable(x) for x in v]
            if isinstance(v, dict):
                return {k: _jsonable(x) for k, x in v.items()}
            return str(v)  # last resort: Path, enums, etc.

        _all_args = {}
        for _name, _dc in [("model", model_args), ("data", data_args), ("training", training_args)]:
            _all_args[_name] = {k: _jsonable(v) for k, v in dataclasses.asdict(_dc).items()}
        with open(os.path.join(training_args.output_dir, "resolved_config.json"), "w") as f:
            json.dump(_all_args, f, indent=2)



    use_qwen3_vl = _is_qwen3_vl(model_args.model_name_or_path)

    # Vanilla baseline: stock HF Qwen3VL, no 3D machinery.
    if model_args.vanilla_qwen3vl:
        from transformers import Qwen3VLForConditionalGeneration

        # The collator stacks pixel_values to [1, P_total, dim] and
        # image_grid_thw to [1, N_imgs, 3] for bs=1. Stock Qwen3-VL forward
        # wants the unbatched form. Reshape on the way in, mirroring the
        # canvas model's PATH B handling (qwen3_vl/model.py L595-603).
        class _VanillaQwen3VL(Qwen3VLForConditionalGeneration):
            def forward(self, pixel_values=None, image_grid_thw=None, **kw):
                if pixel_values is not None and pixel_values.dim() == 3:
                    pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
                if image_grid_thw is not None and image_grid_thw.dim() == 3:
                    image_grid_thw = image_grid_thw.reshape(-1, 3)
                return super().forward(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **kw,
                )

        model = _VanillaQwen3VL.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
        print(f'the initialized model is {model_args.model_name_or_path} the class is {model.__class__.__name__} (VANILLA BASELINE)')
        print(f'  attn_implementation = {model_args.attn_implementation}')
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        model.processor = processor
        # No configure_processor (vanilla wants Qwen3-VL native sizes).
        # No init_3d_embeddings (no depth/angle/marker embeddings).
        # No model.model.processor / reprojection_config (no canvas).
    else:
        if use_qwen3_vl:
            from model_adapters.qwen3_vl.model import \
                Qwen3VL3DForConditionalGeneration as Model3DClass
        else:
            from model_adapters.qwen3_5.model import \
                Qwen3_5_3DForConditionalGeneration as Model3DClass

        # Auto-infer model_type for adapter routing if the user left the default
        if data_args.model_type == "qwen3vl_3d" and not use_qwen3_vl:
            data_args.model_type = "qwen3_5_3d"
            print(f"[auto] Set model_type={data_args.model_type!r} based on model path")

        _from_pretrained_kwargs = dict(
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            attn_implementation=model_args.attn_implementation,
        )
        model = Model3DClass.from_pretrained(
            model_args.model_name_or_path, **_from_pretrained_kwargs)
        print(f'the initialized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
        print(f'  attn_implementation = {model_args.attn_implementation}')
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        configure_processor(processor)
        model.processor = processor
        model.model.processor = processor

        # Reprojection knobs (rope_pos_range, temporal_max_range, depth_embed_*,
        # etc.) flow data_args -> data_processor_3d._reprojection_config
        # -> adapter.prepare_batch() for the precomputed-features path. The
        # live-features path (use_precomputed_features=False) needs the same dict
        # inside model.forward(), so we mirror it onto model.model.reprojection_config
        # after make_supervised_data_module() runs (see below).

        init_3d_embeddings(model, data_args)

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    _tok_kwargs = dict(
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # matches the left-padding used by the collator
    )
    _tok_kwargs["use_fast"] = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, **_tok_kwargs,
    )

    _merge_stage1 = False
    if training_args.lora_enable:
        from peft import (LoraConfig, PeftModel, TaskType, get_peft_model,
                          set_peft_model_state_dict)
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        _has_resume_ckpt = (
            training_args.resume_from_checkpoint
            or list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        )
        # Stage-1 merge must happen on BOTH initial launch AND resume — the
        # saved stage-2 LoRA was trained on top of the merged-stage1 base, so
        # loading it on top of raw base silently mis-assembles the model and
        # produces garbage outputs (loss ~2x normal, gen-eval collapses).
        _merge_stage1 = (
            training_args.lora_checkpoint_merge
            and training_args.lora_checkpoint_path
        )

        # Optional: load stage-1 LoRA, merge into base weights, then continue
        # with a fresh (possibly smaller) LoRA wrapping the merged base.
        # lora_checkpoint_path may be a comma-separated CHAIN of adapters
        # merged in order (stage-3 runs: 'stage1_ckpt,stage2_ckpt') — each
        # adapter was trained on top of the previous merge, so order matters.
        if _merge_stage1:
            _merge_paths = [p.strip() for p in
                            training_args.lora_checkpoint_path.split(",") if p.strip()]
            _is_rank_zero = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
            for _mi, _mpath in enumerate(_merge_paths):
                # Prefer the in-output snapshot — it's guaranteed to still exist
                # even if upstream rotated the original via save_total_limit.
                # First entry keeps the legacy name so existing stage-2 runs resume.
                _snap_name = "stage1_lora" if _mi == 0 else f"merge_lora_{_mi}"
                snapshot_dir = pathlib.Path(training_args.output_dir) / _snap_name
                ckpt_path = str(snapshot_dir) if snapshot_dir.exists() else _mpath
                # Snapshot the LoRA into our output dir on initial launch so the
                # resulting checkpoint stays self-contained and resumes work even
                # after upstream rotation.
                # Read rank from env directly: training_args.local_rank is unreliable
                # under torchrun+deepspeed in this transformers version, so all ranks
                # were entering the gate and racing on rmtree/copytree.
                if _is_rank_zero and not snapshot_dir.exists():
                    import shutil
                    tmp_dir = snapshot_dir.with_suffix(".tmp")
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir)
                    print(f"[lora_merge] Snapshotting adapter {_mi} to {snapshot_dir}")
                    # Copy ONLY the files the merge (from_pretrained) and the
                    # 3D-embed warm-start actually read. A full-checkpoint dir
                    # also holds ~8 GB of DeepSpeed optimizer + RNG state that a
                    # merge-then-fresh-adapter run never touches; copytree'ing
                    # all of it filled the page cache and OOM-killed the job near
                    # the cgroup mem limit (see docs/RELEASE_CLEANUP_PLAN.md).
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    _snap_files = [
                        "adapter_config.json", "adapter_model.safetensors",
                        "adapter_model.bin", "depth_embedding.pt",
                    ]
                    _copied = 0
                    for _fn in _snap_files:
                        _src = os.path.join(_mpath, _fn)
                        if os.path.exists(_src):
                            shutil.copy2(_src, tmp_dir / _fn)
                            _copied += 1
                    # The shared stage-transition loader needs the source's
                    # recorded ratio and depth architecture. Keep that metadata
                    # beside the self-contained adapter snapshot as well.
                    _source_configs = (
                        pathlib.Path(_mpath) / "resolved_config.json",
                        pathlib.Path(_mpath).parent / "resolved_config.json",
                    )
                    _source_config = next(
                        (_p for _p in _source_configs if _p.is_file()), None
                    )
                    if _source_config is not None:
                        shutil.copy2(_source_config, tmp_dir / "resolved_config.json")
                    if _copied == 0:
                        # Unexpected layout — fall back to a full copy so we
                        # never silently snapshot an empty adapter dir.
                        shutil.rmtree(tmp_dir)
                        shutil.copytree(_mpath, tmp_dir)
                    tmp_dir.rename(snapshot_dir)
                if (_is_rank_zero and snapshot_dir.exists()
                        and not (snapshot_dir / "resolved_config.json").exists()):
                    # Backfill snapshots made by older code. This is metadata
                    # only; adapter and depth weights remain untouched.
                    _source_config = next(
                        (_p for _p in (
                            pathlib.Path(_mpath) / "resolved_config.json",
                            pathlib.Path(_mpath).parent / "resolved_config.json",
                        ) if _p.is_file()),
                        None,
                    )
                    if _source_config is not None:
                        import shutil
                        shutil.copy2(
                            _source_config, snapshot_dir / "resolved_config.json"
                        )
                # Wait for rank 0 to finish the snapshot before other ranks proceed.
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
                print(f"[lora_merge] Loading adapter {_mi + 1}/{len(_merge_paths)} from {ckpt_path}")
                model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=False)
                model = model.merge_and_unload()
            print(f"[lora_merge] Merged {len(_merge_paths)} adapter(s) into base; "
                  "re-wrapping with new LoRA config")
            for p in model.parameters():
                p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            # Include MLP layers alongside attention for better spatial adaptation
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # Unfreeze depth/camera/angle embeddings -- new modules not covered by LoRA.
        # freeze_3d_embeddings=True keeps them frozen (stage-3 runs that must
        # keep perception bit-identical to the merged base); create_optimizer
        # filters on requires_grad, so frozen modules simply drop out of the
        # depth param groups.
        if training_args.freeze_3d_embeddings:
            print("[3d_embed] freeze_3d_embeddings=True — all auxiliary 3D embedding "
                  "modules stay frozen")
        else:
            for name, p in model.named_parameters():
                if "depth_embedding" in name or "depth_fourier" in name or "depth_loc_proj" in name or "depth_cartesian" in name or "depth_embed_log_scale" in name:
                    p.requires_grad = True
                    print(f"[depth_embed] Unfroze {name}")
                if "inline_patch_constant_embed" in name:
                    p.requires_grad = True
                    print(f"[inline_patch] Unfroze {name}")

        # Warm-start from a pre-trained LoRA checkpoint (e.g. stage 1).
        # Skip if resuming — the stage-1 weights are already baked into the
        # Trainer checkpoint and will be restored by Trainer.train().
        # In the merge path, stage-1 LoRA is already folded into base above,
        # so we only need to load the auxiliary 3D embeddings here.
        if training_args.lora_checkpoint_path and not _has_resume_ckpt:
            # For a merge chain, auxiliary 3D embeddings come from the LAST
            # adapter (the most recent training stage's final state).
            ckpt_path = [p.strip() for p in
                         training_args.lora_checkpoint_path.split(",") if p.strip()][-1]
            if not _merge_stage1:
                import safetensors.torch
                adapter_file = os.path.join(ckpt_path, "adapter_model.safetensors")
                if not os.path.exists(adapter_file):
                    adapter_file = os.path.join(ckpt_path, "adapter_model.bin")
                if os.path.exists(adapter_file):
                    if adapter_file.endswith(".safetensors"):
                        state_dict = safetensors.torch.load_file(adapter_file)
                    else:
                        state_dict = torch.load(adapter_file, map_location="cpu")
                    set_peft_model_state_dict(model, state_dict)
                    print(f"Loaded LoRA weights from {adapter_file}")
                else:
                    raise FileNotFoundError(
                        f"No adapter_model.safetensors or .bin found in {ckpt_path}"
                    )
            # Load auxiliary 3D embedding .pt files from the stage1 checkpoint
            # if present. Handles depth/angle/camera in one call. A merge into
            # free-running stage 2 also performs the maintained fixed-to-free
            # gate conversion here, from the source checkpoint state. Positive
            # depth_embed_fixed_ratio is the explicit pinned stage-2 override.
            _stage2_target_ratio = (
                float(data_args.depth_embed_fixed_ratio)
                if _merge_stage1 and training_args.load_depth_embed_from_stage1
                else None
            )
            _legacy_gate_scale = float(
                getattr(training_args, "depth_warmstart_gate_scale", 1.0)
            )
            if _stage2_target_ratio is not None and _legacy_gate_scale != 1.0:
                raise RuntimeError(
                    "depth_warmstart_gate_scale cannot be supplied on the maintained "
                    "stage-2 handoff. Remove the manual scale: fixed-ratio source "
                    "compensation is derived automatically from source checkpoint state."
                )
            load_3d_embeddings(
                model,
                ckpt_path,
                load_depth=training_args.load_depth_embed_from_stage1,
                stage2_target_fixed_ratio=_stage2_target_ratio,
            )
            _record_depth_handoff_metadata(model, training_args.output_dir)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            if hasattr(model, "visual"):
                model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    # Apply 3D generation patches after the final model object is set up.
    # Must come after get_peft_model() — PEFT replaces prepare_inputs_for_generation
    # and the patches need to be on the top-level wrapper that Trainer uses.
    # apply_patches is safe for vanilla too: silences kwargs validation and
    # forwards 3D-only keys that vanilla batches don't have.
    # qwen3_vl/patches.py also handles the qwen3_5 case (the patches are
    # generic to the HF generation protocol; the qwen3_5 model class
    # implements the same projected_* kwarg contract that the patches
    # forward through), and the vanilla qwen3-vl baseline.
    from model_adapters.qwen3_vl.patches import apply_patches
    apply_patches(model)

    if not model_args.vanilla_qwen3vl:
        model.model.model.debug = False

        # Probe marker stash: auto-built once per (model, size) and cached under
        # ~/.cache/onecanvas_features/. No-op when curriculum_obb_feature_stash_enable is
        # False. Must run BEFORE make_supervised_data_module so the probe dataset
        # can load the cached tensor at its __init__.
        from onecanvas.data.curriculum_obb_feature_stash import maybe_build_obb_feature_stash
        maybe_build_obb_feature_stash(model, processor, data_args)

    # DO NOT BUILD A VALIDATION DATASET NOTHING WILL READ. This gate was in
    # this file before the evaluation-contract work and came back out with it
    # (2026-09-14, job 2934131, dead in 2m30s on every rank). A probe-only
    # continuation sets ordinary eval off and gen_eval_steps to 0 and is scored
    # by the closed-loop plugin evaluator, which builds its own sources, so the
    # val split here serves nobody. It is not merely wasted work: the val-split
    # constructor REFUSES a curriculum task whose pool holds no val scene, and
    # the room families are deliberately pinned to nine accepted TRAIN rooms,
    # so building it kills the run before its first step.
    #
    # The contract is satisfied by the attached evaluator, not by this dataset.
    # When ordinary or generation evaluation IS configured, the val set is
    # built exactly as before.
    _ordinary_eval = str(training_args.eval_strategy).lower() not in {
        "no", "intervalstrategy.no"
    }
    _generation_eval = int(training_args.gen_eval_steps or 0) > 0
    data_module = make_supervised_data_module(
        processor, data_args=data_args,
        build_eval_dataset=(_ordinary_eval or _generation_eval))

    if not model_args.vanilla_qwen3vl:
        # Live-features path needs the reprojection config inside model.forward().
        # Pull it from the actual SceneQADataset instance so the dict matches
        # exactly what the precomputed path passes to adapter.prepare_batch().
        def _find_reproj_config(ds):
            if ds is None:
                return None
            if hasattr(ds, "_reprojection_config"):
                return ds._reprojection_config()
            # ConcatDataset of (SceneQADataset, SpatialPretrainingDataset) etc.
            for child in getattr(ds, "datasets", []) or []:
                cfg = _find_reproj_config(child)
                if cfg is not None:
                    return cfg
            return None

        _reproj_cfg = (
            _find_reproj_config(data_module.get("train_dataset"))
            or _find_reproj_config(data_module.get("eval_dataset"))
        )
        if _reproj_cfg is not None:
            # Walk to the inner 3D model. Without PEFT this is `model.model`
            # (e.g. Qwen3VL3DForConditionalGeneration -> Qwen3VL3DModel). With
            # PEFT it is `model.model.model` (PeftModel -> ForCondGen -> 3DModel).
            # Both Qwen3VL3DModel and Qwen3_5_3DModel implement the live-features
            # PRE-PATH-A branch.
            from model_adapters.qwen3_vl.model import Qwen3VL3DModel as _Qwen3VL3DModel
            # Qwen3.5 is the SECOND backbone and this needs its class only to
            # widen an isinstance tuple, so an installed transformers without
            # `models.qwen3_5` must not be fatal. 4.57.3 has no such module and
            # it landed in the shared `test` env on 2026-08-21; this one
            # unconditional line then killed a Qwen3-VL TRAINING run at startup
            # (job 2894084, 2026-08-26), after the identical line in
            # run_benchmarks.py had already killed every eval. Every other
            # qwen3_5 import in this tree sits inside a real backbone branch and
            # runs only when that backbone is the one selected; these two
            # isinstance-widening lines were the only unguarded pair.
            _live_features_classes = (_Qwen3VL3DModel,)
            try:
                from model_adapters.qwen3_5.model import Qwen3_5_3DModel as _Q35
            except ImportError as _e:
                print(f"[live-features] Qwen3.5 backbone unavailable, "
                      f"Qwen3-VL only ({_e})")
            else:
                _live_features_classes = (_Qwen3VL3DModel, _Q35)
            _candidates = [
                getattr(model, "model", None),
                getattr(getattr(model, "model", None), "model", None),
            ]
            _attached = False
            for _c in _candidates:
                if isinstance(_c, _live_features_classes):
                    _c.reprojection_config = _reproj_cfg
                    _attached = True
                    break
            if not _attached:
                # Hard error, not warning: the previous warning was misleading
                # ("will fail at runtime") because it didn't actually fail — the
                # model silently trained text-only without vision substitution,
                # wasting hours. Raise so the failure is loud and immediate.
                raise RuntimeError(
                    "[live-features] could not locate a 3D model "
                    "(Qwen3VL3DModel / Qwen3_5_3DModel) in the model tree to "
                    "attach reprojection_config. The model needs this config to "
                    "run reproject_scene in forward(); without it, the prefill "
                    "silently falls through to a text-only path and vision is "
                    "never substituted into inputs_embeds. Add the new model "
                    "class to _live_features_classes above."
                )

    # Generation-based eval callback: runs every training_args.gen_eval_steps steps
    # and once at the very end of training. Only active on rank 0.
    # The model lineage a recorded evaluation belongs to. Reusing a result
    # needs to know WHICH WEIGHTS produced it, and a step number does not say:
    # the same output dir re-entered with a different warm start, a different
    # merge chain or a different base model is a different model at the same
    # step. See eval_contract.checkpoint_identity.
    _eval_lineage = {
        "model_name_or_path": model_args.model_name_or_path,
        "lora_checkpoint_path": training_args.lora_checkpoint_path or "",
        "lora_checkpoint_merge": bool(training_args.lora_checkpoint_merge),
        "lora_r": int(training_args.lora_r),
        "lora_alpha": int(training_args.lora_alpha),
        "vanilla_qwen3vl": bool(model_args.vanilla_qwen3vl),
        "seed": int(training_args.seed),
    }

    gen_eval_cb = GenerationEvalCallback(
        processor=processor,
        data_args=data_args,
        training_args=training_args,
        output_dir=training_args.output_dir,
        eval_gen_steps=training_args.gen_eval_steps,
        num_samples=training_args.gen_eval_num_samples,
        eval_num_images=training_args.gen_eval_num_images,
        final_num_samples=training_args.gen_eval_final_num_samples,
        best_metric=training_args.best_eval_metric,  # default: mean of official_overall across non-probe eval datasets; stage-1 sets geometric_probing/official_overall
        eval_dataset_use=training_args.gen_eval_dataset,
        lineage=_eval_lineage,
    )

    # Trainer-plugin seam (mirrors ONECANVAS_PROBE_TASK_PLUGINS): an external
    # package can swap in a Trainer subclass without touching core. Env value:
    # "pkg.module:factory", where factory is callable(base_cls) -> Trainer
    # subclass. Defaults to WeightedTrainer, so this is inert unless set.
    _TrainerCls = WeightedTrainer
    _plugin = os.environ.get("ONECANVAS_TRAINER_PLUGIN", "").strip()
    if _plugin:
        import importlib
        _mod_name, _, _factory = _plugin.partition(":")
        _factory = _factory or "make_grpo_trainer_cls"
        _TrainerCls = getattr(importlib.import_module(_mod_name), _factory)(WeightedTrainer)
        logging.info("[trainer-plugin] using %s -> %s", _plugin, _TrainerCls.__name__)

    trainer = _TrainerCls(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        # DiskSpaceGuardCallback goes LAST so its on_step_end sees the final
        # control.should_save (DefaultFlowCallback and gen_eval_cb both set it).
        callbacks=[gen_eval_cb, ProfilingCallback(), DepthMagnitudeCallback(),
                   ContinuationStopCallback(),
                   DiskSpaceGuardCallback()],
        **data_module
    )

    resume_ckpt = None
    resume_ckpt_dir = None
    if training_args.resume_from_checkpoint:
        resume_ckpt = training_args.resume_from_checkpoint
        resume_ckpt_dir = training_args.resume_from_checkpoint
        logging.info("resuming from explicit checkpoint: %s", resume_ckpt)
    elif list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # Highest-step checkpoint that passes the completeness check — NOT
        # blindly the highest-numbered: an ENOSPC crash mid-save leaves a torn
        # highest checkpoint (2026-08-05 incident) and HF's own auto-pick
        # would load it. Pass the validated dir explicitly instead of True.
        resume_ckpt_dir = find_last_complete_checkpoint(training_args.output_dir)
        if resume_ckpt_dir is None:
            raise SystemExit(
                f"[ckpt-guard] checkpoints exist in {training_args.output_dir} "
                f"but none passed the completeness check. Refusing to silently "
                f"start from scratch; inspect/delete the torn checkpoints or "
                f"pass --resume_from_checkpoint explicitly."
            )
        resume_ckpt = resume_ckpt_dir
        logging.info("resuming from last complete checkpoint: %s", resume_ckpt_dir)

    # HF Trainer's auto-resume only restores the LoRA adapter + optimizer
    # state. The 3D embedding params (depth/angle/camera) are saved separately
    # by save_3d_embeddings; without this load they silently reset to base-model
    # state and the model wakes up with trained LoRA but UNTRAINED geometry.
    # Vanilla baseline has no 3D embeddings, skip.
    if resume_ckpt_dir is not None and not model_args.vanilla_qwen3vl:
        print(f"[resume] Loading 3D embeddings from {resume_ckpt_dir}")
        _require_stage2_handoff = bool(
            _merge_stage1
            and training_args.load_depth_embed_from_stage1
            and float(data_args.depth_embed_fixed_ratio) <= 0
        )
        load_3d_embeddings(
            model,
            resume_ckpt_dir,
            require_stage2_handoff=_require_stage2_handoff,
        )
        _record_depth_handoff_metadata(model, training_args.output_dir)

    # Optional: resume model weights but NOT optimizer / scheduler state.
    #
    # DeepSpeed ZeRO asserts the saved and rebuilt optimizers have the same
    # number of param groups. create_optimizer() splits depth params into their
    # own group whenever depth_lr_multiplier != 1.0 (or depth_no_weight_decay is
    # set), so resuming a checkpoint across a change to those flags dies with
    #   ValueError: loaded state dict has a different number of parameter groups
    # Set ONECANVAS_RESUME_SKIP_OPTIM=1 to continue such a run: the LoRA weights
    # still load (via DeepSpeed model_states) and the depth MLP still loads (via
    # load_3d_embeddings above), global_step is still restored from
    # trainer_state.json, and a fresh optimizer + scheduler is built at the new
    # LR. Cost is the discarded Adam moments and an LR-schedule re-warm.
    if os.environ.get("ONECANVAS_RESUME_SKIP_OPTIM", "0") == "1" and resume_ckpt:
        import glob as _glob
        import transformers.integrations.deepspeed as _ds_int
        import transformers.trainer as _hf_trainer

        def _load_ckpt_weights_only(deepspeed_engine, checkpoint_path, load_module_strict=True):
            if sorted(_glob.glob(f"{checkpoint_path}/global_step*")):
                load_path, _ = deepspeed_engine.load_checkpoint(
                    checkpoint_path,
                    load_module_strict=load_module_strict,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False,
                )
                if load_path is None:
                    raise ValueError(f"[deepspeed] failed to resume from checkpoint {checkpoint_path}")
            else:
                raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")

        _ds_int.deepspeed_load_checkpoint = _load_ckpt_weights_only
        if hasattr(_hf_trainer, "deepspeed_load_checkpoint"):
            _hf_trainer.deepspeed_load_checkpoint = _load_ckpt_weights_only
        if local_rank in (-1, 0):
            print("[resume] ONECANVAS_RESUME_SKIP_OPTIM=1 -> loading model weights only "
                  "(fresh optimizer + scheduler at the current LR; global_step still restored)")

    # EVALUATION LIFECYCLE FOR PLUGIN EVALUATORS. A trainer plugin attaches its
    # own callbacks in its __init__, so they exist by now. Any of them that
    # implements run_evaluation(model, step, phase) is DRIVEN from here through
    # the same lifecycle the built-in gets: baseline (fresh or at a resume that
    # never recorded one), the declared cadence after the checkpoint is saved,
    # a final evaluation when the last step has none, reuse of a matching
    # recorded result, one ledger entry per result, and a distributed failure
    # join. Declaring an evaluator without being driven leaves all six to the
    # plugin, which is what let a registered evaluator look integrated while
    # skipping four of them.
    _driven = [
        cb for cb in trainer.callback_handler.callbacks
        if eval_contract.is_driven_evaluator(cb)
        and cb.evaluation_contract_spec() is not None
    ]
    if _driven:
        trainer.add_callback(
            eval_contract.EvaluationLifecycleCallback(
                _driven, training_args.output_dir, _eval_lineage,
            )
        )

    # EVALUATION CONTRACT. Runs on every launch and every resume, immediately
    # before the optimization loop, and refuses a run that cannot score itself.
    # Reads the attached callbacks, so a plugin evaluator counts and a missing
    # launcher variable does not silently pass. See train/eval_contract.py.
    eval_contract.enforce_before_optimization(
        training_args.output_dir,
        trainer.callback_handler.callbacks,
        resuming=resume_ckpt is not None,
        is_rank_zero=(int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0),
        extra_context={
            "run_name": training_args.run_name,
            "dataset_use": data_args.dataset_use,
            "max_steps": int(training_args.max_steps),
            "trainer_plugin": os.environ.get("ONECANVAS_TRAINER_PLUGIN", ""),
            "resume_from": resume_ckpt or "",
        },
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Save auxiliary 3D embeddings (depth/angle/camera) alongside the LoRA
    # adapter. The previous end-of-train save was missing camera_embedding.pt
    # — that omission is fixed by routing through the shared helper.
    # Vanilla baseline has no 3D embeddings, skip.
    if not model_args.vanilla_qwen3vl:
        save_3d_embeddings(model, training_args.output_dir, log_prefix="[end-of-train]")
        # HF Trainer's final save (safe_save_model_for_hf_trainer above)
        # writes the LoRA adapter into checkpoint-{MAX_STEPS}/ without
        # firing the on_save callback, so the final checkpoint dir ends
        # up LoRA-only. Mirror the depth/angle embeddings into it here
        # so every checkpoint-N (including the final one) is a complete
        # warm-start source.
        _ckpts = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        if _ckpts:
            _final_ckpt = str(max(_ckpts, key=lambda p: int(p.name.split("-")[1])))
            save_3d_embeddings(model, _final_ckpt, log_prefix="[end-of-train final-ckpt]")

    processor.save_pretrained(training_args.output_dir)

    # Successful completion is recorded LAST, and only once the final
    # evaluation's metrics and population accounting are durably on disk.
    # Raises when they are not, so a run that trained every step and scored
    # nothing cannot be read as a finished result.
    # Rank 0 owns the ledger file, so only rank 0 can read it back reliably.
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0:
        eval_contract.record_completion(
            training_args.output_dir,
            step=int(trainer.state.global_step),
        )

    # Explicitly finalize wandb so the run is marked "finished" rather than
    # "crashed". Without this, torchrun's SIGTERM to non-rank-0 workers
    # races with wandb's atexit handler and wandb often marks the run as
    # crashed even though training completed cleanly.
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    train()
