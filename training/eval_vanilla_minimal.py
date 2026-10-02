"""Minimal standalone eval for vanilla Qwen3-VL baseline.

Sidesteps run_benchmarks's custom infrastructure (apply_patches,
configure_model_3d, our reshape wrapper, etc.) and uses HF's plain
generate API directly. Useful when run_benchmarks_ddp hits the
2x-KV-cache bug with stock Qwen3-VL multi-image generate.

Usage:
    torchrun --nproc_per_node=8 --master_port=29600 \\
        training/eval_vanilla_minimal.py \\
        --lora /path/to/checkpoint \\
        --dataset vsi_bench --limit 1500 --stratified
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from onecanvas.train.argument import DataArguments
from onecanvas.data.data_processor_3d import SceneQADataset
import utils as eval_utils


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world_size, local_rank


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--lora", required=True)
    p.add_argument("--dataset", default="vsi_bench")
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--stratified", action="store_true")
    p.add_argument("--num-images", type=int, default=32)
    p.add_argument("--image-resolution", default="320x240")
    p.add_argument("--max-image-resolution", default="320x240")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--exp-name", default="output/vanilla_minimal_eval")
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    args = p.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"[rank {rank}] world_size={world_size}, loading {args.model_path}")

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{local_rank}",
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora, torch_dtype=torch.bfloat16, local_files_only=True)
    model.eval()
    model.config.use_cache = True

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)

    # Build dataset (val split, vanilla mode)
    da = DataArguments(
        dataset_use=args.dataset,
        image_resolution=args.image_resolution,
        max_image_resolution=args.max_image_resolution,
        use_resized_images=True,
        num_images=args.num_images,
        with_precomputed_geometry=True,
        val_sample_num=args.limit,
        stratified_eval=args.stratified,
    )
    da.vanilla_qwen3vl = True
    da.with_answer = False

    dataset = SceneQADataset(processor, data_args=da, data_split="val", sort=True)
    if is_main:
        print(f"Loaded {len(dataset)} samples for {args.dataset}")

    # Shard across ranks
    n = len(dataset)
    if world_size > 1:
        chunk = n // world_size
        start = rank * chunk
        end = start + chunk if rank < world_size - 1 else n
        indices = list(range(start, end))
    else:
        indices = list(range(n))

    print(f"[rank {rank}] processing {len(indices)} samples")

    results = []
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        # Build a clean batch on device
        device = next(model.parameters()).device
        batch = {
            "input_ids": sample["input_ids"].squeeze(0).to(device),
            "attention_mask": sample["attention_mask"].squeeze(0).to(device),
            "pixel_values": sample["pixel_values"].to(device, dtype=torch.bfloat16),
            "image_grid_thw": sample["image_grid_thw"].to(device),
        }
        input_len = batch["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
            )
        new_ids = generated[:, input_len:]
        prediction = processor.batch_decode(new_ids, skip_special_tokens=True)[0]

        results.append({
            "scene_id": sample["scene_id"],
            "question": sample["question"],
            "question_type": sample.get("question_type", "unknown"),
            "ground_truths": [sample["answer"]] if isinstance(sample["answer"], str) else sample["answer"],
            "prediction": prediction,
        })
        if is_main and (i + 1) % 25 == 0:
            print(f"[rank 0] {i+1}/{len(indices)} ({time.time()-t0:.2f}s/it)")

    # Gather results to rank 0
    if world_size > 1:
        all_results = [None] * world_size
        dist.gather_object(results, all_results if is_main else None, dst=0)
        if is_main:
            results = [r for sub in all_results for r in sub]

    if is_main:
        os.makedirs(args.exp_name, exist_ok=True)
        out_path = os.path.join(args.exp_name, "qa_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[done] {len(results)} predictions saved to {out_path}")

        from utils.metrics import (
            vsibench_score,
            VSIBENCH_DISPLAY_ORDER,
            VSIBENCH_DISPLAY_NAMES,
        )

        # Per-subtask scoring. Direction subtypes (easy/medium/hard) merge into
        # object_rel_direction by averaging *task* means (not pooling samples).
        per_type = {}
        for r in results:
            qt = r.get("question_type", "unknown")
            pred = r["prediction"]
            gts = r["ground_truths"]
            score = max(vsibench_score(pred, gt, qt) for gt in gts) if gts else 0.0
            per_type.setdefault(qt, []).append(score)

        type_means = {qt: sum(s)/len(s) for qt, s in per_type.items()}
        dir_subs = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"]
        dir_means = [type_means[d] for d in dir_subs if d in type_means]
        if dir_means:
            type_means["object_rel_direction"] = sum(dir_means) / len(dir_means)

        print("\n[VSI-Bench per-subtask]")
        ordered = []
        for qt in VSIBENCH_DISPLAY_ORDER:
            if qt in type_means:
                name = VSIBENCH_DISPLAY_NAMES.get(qt, qt)
                n = len(per_type.get(qt, [])) or sum(len(per_type[d]) for d in dir_subs if d in per_type)
                print(f"  {name:<14s} {type_means[qt]*100:6.2f}  (n={n})")
                ordered.append(type_means[qt])
        if ordered:
            print(f"  {'Avg':<14s} {sum(ordered)/len(ordered)*100:6.2f}")

        unknown = [qt for qt in type_means if qt not in VSIBENCH_DISPLAY_ORDER and qt not in dir_subs]
        if unknown:
            print("\n[other question_types seen]")
            for qt in unknown:
                print(f"  {qt:<32s} {type_means[qt]*100:6.2f}  (n={len(per_type[qt])})")


if __name__ == "__main__":
    main()
