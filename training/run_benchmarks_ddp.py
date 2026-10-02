#!/usr/bin/env python3
"""DDP launcher for run_benchmarks. Same flags as run_benchmarks.py.

Launch:
  torchrun --nproc_per_node=8 training/run_benchmarks_ddp.py \
      --from-config /path/to/checkpoint --datasets sqa3d vsi_bench

Each rank loads a full copy of the model on its own GPU, runs generation on
a non-overlapping slice of the test set, and rank 0 gathers results and
computes metrics. Mirrors training-time gen_eval (train.py:_run_eval).
"""

import datetime
import gc
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import utils
from inference import _compute_da3_geometry
from onecanvas.data.data_processor_3d import make_supervised_data_module
from training.run_benchmarks import (
    ALL_BENCHMARKS,
    _attach_reprojection_config,
    _print_spbench_paper_table,
    build_data_args,
    check_full_test_split,
    configure_model_3d,
    load_da3_model,
    load_model_and_processor,
    parse_args,
    tracker_name_for,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("ONECANVAS_FULLSPAN_SAMPLER", "1")


def setup_distributed():
    """Init torch.distributed from torchrun env vars. Returns (rank, world_size, local_rank).

    No-op when launched without torchrun (single-process fallback).
    """
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        # 2h collective timeout (default 10 min): the first cross-rank barrier
        # of a benchmark can wait on one rank's cold-cache feature loading,
        # which legitimately exceeds 10 min at 640x480 on a slow filesystem.
        torch.distributed.init_process_group(
            "nccl", timeout=datetime.timedelta(hours=2))
    return rank, world_size, local_rank


def build_sharded_dataloader(processor, data_args, args, rank, world_size):
    """Build the test/eval dataset, slice it for this rank, return (full_dataset, dataloader, full_n).

    `full_dataset` is returned (un-sliced) so the caller can pass it to
    _attach_reprojection_config, which walks the dataset to extract the
    reprojection config.
    """
    data_args.with_answer = False
    if getattr(args, "stratified_eval", False):
        dm = make_supervised_data_module(
            processor, data_args=data_args,
            build_train_dataset=False, build_eval_dataset=True,
        )
        full_dataset = dm["eval_dataset"]
    else:
        dm = make_supervised_data_module(processor, data_args=data_args, with_test=True,
                                         build_train_dataset=False, build_eval_dataset=False)
        full_dataset = dm.get("test_dataset") or dm.get("eval_dataset")
        check_full_test_split(data_args.dataset_use, full_dataset, args)
        if args.limit is not None and len(full_dataset) > args.limit:
            full_dataset = Subset(full_dataset, range(args.limit))

    full_n = len(full_dataset)
    if world_size > 1:
        chunk = full_n // world_size
        start = rank * chunk
        end = start + chunk if rank < world_size - 1 else full_n
        local_dataset = Subset(full_dataset, list(range(start, end)))
    else:
        local_dataset = full_dataset

    cpu_count = len(os.sched_getaffinity(0))
    if args.debug:
        num_workers = 0
    elif getattr(args, "num_workers", None) is not None:
        num_workers = args.num_workers
    else:
        # One worker per core per rank can exhaust host memory. Cap at 8.
        num_workers = min(8, max(1, cpu_count // max(1, world_size)))
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=False,
        **({"prefetch_factor": 2} if num_workers > 0 else {}),
    )
    dataloader = DataLoader(local_dataset, collate_fn=dm["data_collator"], **loader_kwargs)
    return full_dataset, local_dataset, dataloader, full_n


def gather_results(local_results, local_skipped, rank, world_size):
    """all_gather_object(local_results) -> flattened list on rank 0, [] elsewhere."""
    if world_size <= 1:
        return local_results, local_skipped
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_results)
    gathered_skipped = [None] * world_size
    torch.distributed.all_gather_object(gathered_skipped, local_skipped)
    if rank != 0:
        return [], []
    flat = [r for sub in gathered for r in (sub or [])]
    flat_skipped = [s for sub in gathered_skipped for s in (sub or [])]
    return flat, flat_skipped


def compute_and_print_summary(results, skipped, ds_name, t_name, exp_dir, data_args, args):
    """Rank-0-only: feed gathered results into MetricTracker, save, return final_stats dict."""
    if skipped:
        print(f"[Warning] Skipped {len(skipped)} samples due to missing precomputed features")

    tracker = utils.MetricTracker(
        benchmarking=True, exp_name=exp_dir, dataset_name=t_name,
        pano_grounding_format=bool(getattr(data_args, "pano_grounding_format", False)),
    )
    n_total = len(results)
    for i, r in enumerate(results):
        tracker.update(
            count=i,
            scene_id=r["scene_id"],
            question=r["question"],
            prediction=r["prediction"],
            ground_truths=r["ground_truths"],
            question_type=r["question_type"],
            all_predictions=r.get("all_predictions"),
            n_source_images=r.get("n_source_images"),
        )
        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            stats = {m: (sum(v) / len(v) if v else 0) for m, v in tracker.metrics_acc.items()}
            grnd = tracker.grounding_stats()
            print("\n" + "=" * 40)
            print(f"[{ds_name}] PROGRESS {i + 1}/{n_total}")
            if grnd:
                print(f"Acc@0.25: {grnd['grnd_Acc@0.25']:.1%} | Acc@0.5: {grnd['grnd_Acc@0.5']:.1%} | Acc@0.1: {grnd['grnd_Acc@0.1']:.1%}")
                print(f"Mean IoU: {grnd['grnd_mean_IoU']:.4f} | Parse: {grnd['grnd_parse_rate']:.0%}")
            else:
                line = f"METEOR: {stats.get('METEOR', 0):.4f} | EM@1: {stats.get('EM@1', 0):.4f}"
                if tracker.dataset_name == "sqa3d":
                    line += f" | EM@R1: {stats.get('EM@R1', 0):.4f}"
                print(line)
                print(f"ROUGE-L: {stats.get('ROUGEL', 0):.4f} | ROUGE-1: {stats.get('ROUGE1', 0):.4f}")
            print("=" * 40 + "\n")
            tracker.save(step=i + 1)

    tracker.save()

    final_stats = {m: (sum(v) / len(v) if v else 0) for m, v in tracker.metrics_acc.items()}
    if tracker.per_type_scores:
        final_stats["per_type"] = {
            qt: sum(sl) / len(sl) for qt, sl in tracker.per_type_scores.items()
        }
    grnd = tracker.grounding_stats()
    if grnd:
        final_stats.update(grnd)
    mgrnd = tracker.multi_grounding_stats()
    if mgrnd:
        final_stats.update(mgrnd)
    if t_name == "vsi_bench" and tracker.per_type_scores:
        final_stats["official_overall"] = tracker._compute_vsibench_overall()
        _dir_subs = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"}
        _merged = {}
        for k, v in tracker.per_type_scores.items():
            if k in _dir_subs:
                _merged.setdefault("object_rel_direction", []).extend(v)
            else:
                _merged[k] = v
        final_stats["per_type_merged"] = {k: sum(v) / len(v) for k, v in _merged.items()}
    return final_stats


def print_per_dataset_summary(ds_name, t_name, final_stats):
    print(f"\n--- {ds_name} done ---")
    if t_name == "vsi_bench" and "official_overall" in final_stats:
        from utils.metrics import VSIBENCH_DISPLAY_NAMES, VSIBENCH_DISPLAY_ORDER
        print(f"  Avg: {final_stats['official_overall']:.4f}")
        for key in VSIBENCH_DISPLAY_ORDER:
            sc = final_stats.get("per_type_merged", {}).get(key)
            if sc is not None:
                print(f"    {VSIBENCH_DISPLAY_NAMES.get(key, key)}: {sc:.4f}")
    elif "grnd_Acc@0.25" in final_stats:
        print(f"  Acc@0.25: {final_stats['grnd_Acc@0.25']:.1%}  Acc@0.5: {final_stats['grnd_Acc@0.5']:.1%}")
        print(f"  Mean IoU: {final_stats['grnd_mean_IoU']:.4f}")
    else:
        for k, v in final_stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_dist = world_size > 1
    is_main = rank == 0

    datasets = args.datasets or ALL_BENCHMARKS

    if args.exp_name is None:
        if args.lora and not args.no_lora:
            lora_name = Path(args.lora).name
            if lora_name == "best_checkpoint":
                lora_name = Path(args.lora).parent.name
            args.exp_name = f"output/{lora_name}"
        elif args.model_path and os.path.isdir(args.model_path):
            args.exp_name = f"output/{Path(args.model_path).name}"
        else:
            args.exp_name = "output/base_model"

    if is_main:
        print(f"[ddp] rank={rank}/{world_size} local_rank={local_rank}")
        print(f"Benchmarks to run: {datasets}")
        print(f"Output base: {args.exp_name}")

    if args.compute_val_loss and is_main:
        print("[ddp] WARNING: --compute-val-loss is not implemented in the DDP runner; skipping it")
    args.compute_val_loss = False

    # Each rank loads its own copy on its assigned GPU.
    # device_map="cuda" inside load_model_and_processor honors torch.cuda.current_device(),
    # which we set in setup_distributed().
    model, processor = load_model_and_processor(args)
    lora_path = None if args.no_lora else args.lora
    # Merged self-contained checkpoints carry depth_embedding.pt next to the
    # model weights: restore the 3D embeddings from there when no adapter dir
    # is available to restore them from. Mirrors run_benchmarks.main().
    if (lora_path is None and args.model_path and os.path.isdir(args.model_path)
            and os.path.exists(os.path.join(args.model_path, "depth_embedding.pt"))):
        lora_path = args.model_path
    if getattr(args, "vanilla_qwen3vl", False):
        from model_adapters.qwen3_vl.patches import apply_patches
        apply_patches(model)
    else:
        configure_model_3d(model, args, lora_path)
    _model_device = model.get_input_embeddings().weight.device
    model.to(_model_device)

    import onecanvas.data.data_processor_3d as _dp3d
    if hasattr(model.config, "image_token_id"):
        _dp3d.IMAGE_TOKEN_INDEX = model.config.image_token_id

    da3_model = None
    if not args.precomputed_geometry and not getattr(args, "vanilla_qwen3vl", False):
        da3_model = load_da3_model()

    all_results = {}
    for ds_name in datasets:
        if is_main:
            print("\n" + "=" * 60)
            print(f"  BENCHMARK: {ds_name}")
            print("=" * 60)

        exp_dir = os.path.join(args.exp_name, ds_name)
        t_name = tracker_name_for(ds_name)

        data_args = build_data_args(args, ds_name)
        full_dataset, local_dataset, dataloader, full_n = build_sharded_dataloader(
            processor, data_args, args, rank, world_size,
        )

        if not getattr(args, "vanilla_qwen3vl", False):
            _attach_reprojection_config(model, full_dataset)

        if is_main:
            print(f"Loaded {full_n} samples for {ds_name}")
            print(f"Per-rank slice: {len(local_dataset)} samples (this is rank 0)")
        else:
            print(f"[rank {rank}] {ds_name}: {len(local_dataset)} samples")

        # ── Generation on this rank's slice ──────────────────────────
        da3_fn = None
        if da3_model is not None and not data_args.with_precomputed_geometry:
            da3_fn = lambda img: _compute_da3_geometry(img, da3_model)

        local_results, local_skipped = utils.run_generation_loop(
            model, processor, dataloader, model.device,
            max_new_tokens=args.max_new_tokens,
            compute_da3_geometry_fn=da3_fn,
            max_batches=1 if args.debug else None,
            log_prefix=f"[{ds_name}][rank {rank}]",
        )

        # Free GPU memory for the slice before gather (gather is CPU-bound, pickled)
        gc.collect()
        torch.cuda.empty_cache()

        # ── Gather results to rank 0 ─────────────────────────────────
        results, skipped = gather_results(local_results, local_skipped, rank, world_size)
        if is_main and is_dist:
            print(f"[ddp] gathered {len(results)} results from {world_size} ranks for {ds_name}")

        # ── Rank-0 metrics + save ────────────────────────────────────
        if is_main:
            final_stats = compute_and_print_summary(
                results, skipped, ds_name, t_name, exp_dir, data_args, args,
            )
            all_results[ds_name] = final_stats
            print_per_dataset_summary(ds_name, t_name, final_stats)

        # Free dataloader memory across all ranks
        del full_dataset, local_dataset, dataloader, local_results, local_skipped, results, skipped
        gc.collect()
        torch.cuda.empty_cache()

        # Sync before next benchmark so rank 0's metric work doesn't leave
        # other ranks idle holding dataloader resources.
        if is_dist:
            torch.distributed.barrier()

    # ── Final summary across all benchmarks (rank 0) ─────────────────
    if is_main:
        print("\n" + "=" * 60)
        print("  ALL BENCHMARKS SUMMARY")
        print("=" * 60)
        for ds_name, stats in all_results.items():
            t = tracker_name_for(ds_name)
            if t == "vsi_bench" and "official_overall" in stats:
                from utils.metrics import VSIBENCH_DISPLAY_NAMES, VSIBENCH_DISPLAY_ORDER
                print(f"  {ds_name:20s}  Avg={stats['official_overall']:.4f}")
                for key in VSIBENCH_DISPLAY_ORDER:
                    sc = stats.get("per_type_merged", {}).get(key)
                    if sc is not None:
                        print(f"    {VSIBENCH_DISPLAY_NAMES.get(key, key)}: {sc:.4f}")
            elif "grnd_Acc@0.25" in stats:
                print(f"  {ds_name:20s}  Acc@0.25={stats['grnd_Acc@0.25']:.4f}  "
                      f"Acc@0.5={stats['grnd_Acc@0.5']:.4f}  mIoU={stats['grnd_mean_IoU']:.4f}")
            else:
                em = stats.get("EM@1", 0)
                meteor = stats.get("METEOR", 0)
                print(f"  {ds_name:20s}  EM@1={em:.4f}  METEOR={meteor:.4f}")
                if "per_type" in stats:
                    for qt, sc in stats["per_type"].items():
                        print(f"    {qt}: {sc:.4f}")
            if tracker_name_for(ds_name) == "sp_bench":
                _print_spbench_paper_table(args, ds_name)
        print("=" * 60)
        print("Done.")

    if is_dist:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Auto-launch with torchrun when not already inside torchrun.
    if "RANK" not in os.environ:
        ngpus = torch.cuda.device_count()
        if ngpus > 1:
            import random, subprocess
            port = 29500 + random.randint(0, 999)
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={ngpus}",
                f"--master_port={port}",
                __file__,
            ] + sys.argv[1:]
            sys.exit(subprocess.call(cmd))
    main()
