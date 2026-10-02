"""Tensor sequence utilities for model forward passes."""

import torch


def pad_and_stack(tensor_list, dim=1, pad_value=1):
    """Left-pad tensors to equal length and stack along dim.

    Left-padding keeps valid tokens at the END of each sequence, so the valid
    region stays contiguous with the new decode token -- required by
    flash_attn_varlen_func's contiguous attention-mask assumption.
    """
    max_length = max(t.shape[-1] for t in tensor_list)
    padded = []
    for t in tensor_list:
        pad_length = max_length - t.shape[-1]
        padded.append(torch.nn.functional.pad(t, (pad_length, 0), "constant", pad_value))
    return torch.stack(padded, dim=dim)
