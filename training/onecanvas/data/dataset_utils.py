"""Dataset utility functions: I/O, tensor helpers, tokenisation, and sampling."""

import errno
import json
import os
import random
import time
from typing import Dict

import torch

IGNORE_INDEX = -100


def pad_and_cat(tensor_list, pad_value=1):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", pad_value)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _load_pt_mmap(path):
    """Load a .pt file using memory mapping when supported (PyTorch >= 2.1).
    Memory mapping avoids loading the entire file into RAM upfront, which is
    beneficial when only a small subset of keys is accessed from a large dict.

    Retries briefly on ESTALE (NFS stale file handle) which can occur when a
    scene file is replaced underneath us, e.g. during a scene migration. The
    retry re-resolves the symlink so the new location is picked up.
    """
    last_estale = None
    for attempt in range(4):
        try:
            try:
                return torch.load(path, map_location="cpu", mmap=True)
            except TypeError:
                # Older PyTorch versions do not support the mmap kwarg
                return torch.load(path, map_location="cpu")
        except OSError as e:
            if e.errno == errno.ESTALE:
                last_estale = e
                time.sleep(0.5 * (attempt + 1))
                continue
            file_size = os.path.getsize(path) if os.path.exists(path) else -1
            print(f"\n{'='*80}\n"
                  f"ERROR: OSError loading feature file (errno={e.errno})\n"
                  f"  Path: {path}\n"
                  f"  Size: {file_size:,} bytes\n"
                  f"  Error: {e}\n"
                  f"{'='*80}\n")
            raise
        except Exception as e:
            # Log corrupted file with full details
            file_size = os.path.getsize(path) if os.path.exists(path) else -1
            print(f"\n{'='*80}\n"
                  f"ERROR: Failed to load feature file\n"
                  f"  Path: {path}\n"
                  f"  Size: {file_size:,} bytes\n"
                  f"  Error type: {type(e).__name__}\n"
                  f"  Error message: {str(e)}\n"
                  f"  This precomputed feature file is likely corrupted.\n"
                  f"  Delete it (rm \"{path}\") and re-extract features for this "
                  f"scene, or run with live vision (features are recomputed on "
                  f"the fly when the precomputed cache is absent).\n"
                  f"{'='*80}\n")
            raise
    # All retries exhausted on ESTALE
    raise last_estale


def rank0_print(*args):
    if int(os.environ.get("LOCAL_RANK", -1)) in (0, -1):
        print(*args)


def stack_tensor_list(items):
    """Helper to stack list of items into tensor, or return None."""
    if items and len(items) > 0:
        return torch.stack([torch.as_tensor(item) for item in items])
    return None


def prepare_depths(depths):
    """Helper to prepare depth tensors, handling None values.

    Resizes all depth maps to a common (H, W) so they can be stacked.
    This is needed when using GT depths which may vary in resolution per frame.
    """
    if not depths or not any(d is not None for d in depths):
        return []
    # Find target size from first valid depth
    target_h, target_w = None, None
    for d in depths:
        if d is not None and d.ndim == 2:
            target_h, target_w = d.shape
            break
    if target_h is None:
        return [d if d is not None else torch.zeros(1) for d in depths]
    result = []
    for d in depths:
        if d is None:
            result.append(torch.zeros(target_h, target_w))
        elif d.ndim == 2 and (d.shape[0] != target_h or d.shape[1] != target_w):
            result.append(
                torch.nn.functional.interpolate(
                    d.unsqueeze(0).unsqueeze(0).float(),
                    size=(target_h, target_w),
                    mode="nearest",
                ).squeeze(0).squeeze(0)
            )
        else:
            result.append(d)
    return result


def preprocess_qwen_visual(
    input_ids,
    asst_token_id: int = 77091,
    im_end_token_id: int = 151645,
) -> Dict:
    """Set labels for the assistant turn(s) in a tokenised chat sequence.

    Args:
        input_ids: 1-D or 2-D token tensor (shape [1, L] or list).
        asst_token_id: Token ID for the literal word ``assistant`` as it
            appears in the chat template (e.g. 77091 for Qwen2-VL,
            74455 for Qwen3.5).
        im_end_token_id: Token ID for ``<|im_end|>`` (e.g. 151645 for
            Qwen2-VL, 248046 for Qwen3.5).
    """
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == asst_token_id:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != im_end_token_id:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    return input_ids, labels


def _stratified_sample(items, n, seed=42):
    """Return n items with equal representation per question_type.

    Algorithm:
      1. Group items by question_type.
      2. Allocate floor(n / k) slots per type; distribute the remainder to the
         largest types first.
      3. If a type has fewer items than its allocation, take all of them and
         redistribute the slack to the remaining types.
      4. Shuffle within each group before selection (deterministic via seed).
      5. Final list is shuffled so types are interleaved in the DataLoader.
    """
    from collections import defaultdict
    rng = random.Random(seed)

    groups = defaultdict(list)
    for item in items:
        groups[item.get("question_type", "unknown")].append(item)

    # Shuffle within each group
    for g in groups.values():
        rng.shuffle(g)

    k = len(groups)
    if k == 0:
        return items[:n]

    # Allocate slots iteratively to handle groups smaller than their fair share
    allocations = {qt: n // k for qt in groups}
    remainder = n - sum(allocations.values())
    # Give remainder slots to the largest groups
    for qt in sorted(groups, key=lambda q: len(groups[q]), reverse=True):
        if remainder <= 0:
            break
        allocations[qt] += 1
        remainder -= 1

    selected, slack = [], 0
    for qt, alloc in allocations.items():
        available = groups[qt]
        take = min(alloc, len(available))
        selected.extend(available[:take])
        slack += alloc - take  # slots that couldn't be filled

    # Fill slack from items not yet selected, preserving determinism
    if slack > 0:
        selected_ids = {id(x) for x in selected}
        pool = [x for x in items if id(x) not in selected_ids]
        rng.shuffle(pool)
        selected.extend(pool[:slack])

    rng.shuffle(selected)

    # Print type distribution for visibility
    from collections import defaultdict as _dd
    type_counts = _dd(int)
    for item in selected:
        type_counts[item.get("question_type", "unknown")] += 1
    print(f"[stratified_eval] Sampled {len(selected)}/{len(items)} items: "
          + ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items())))

    return selected
