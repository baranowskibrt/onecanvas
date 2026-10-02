"""Unit tests for utils.eval_loop.prepare_model_inputs().

Run with: python -m pytest utils/test_eval_loop.py -v
"""

import torch
import pytest
from utils.eval_loop import prepare_model_inputs, _METADATA_KEYS


def _make_batch(*, with_projection=True, with_prompt_variants=False):
    """Build a synthetic batch dict that mimics the data collator output."""
    batch = {
        # ── Metadata (should be excluded) ──
        "labels": torch.ones(1, 10, dtype=torch.long),
        "answer": ["the kitchen"],
        "question": ["Where is the table?"],
        "question_type": ["What"],
        "scene_id": ["scene0001_00"],
        "images": [torch.randn(3, 240, 320)],
        "projected_labels": [torch.ones(50, dtype=torch.long)],
        # Private key (should be excluded)
        "_dataloader_retries": 3,
    }

    if with_projection:
        # ── Projected keys (should pass through) ──
        batch["projection_done"] = True
        batch["projected_input_ids"] = [torch.ones(80, dtype=torch.long)]
        batch["projected_position_ids"] = [torch.ones(3, 80, dtype=torch.long)]
        batch["projected_attention_mask"] = [torch.ones(80, dtype=torch.long)]
        batch["projected_embeds"] = [torch.randn(80, 128)]
        batch["rope_deltas"] = torch.tensor([-5])

        if with_prompt_variants:
            # Prompt-only variants (should be swapped then excluded)
            batch["projected_input_ids_prompt"] = [torch.zeros(60, dtype=torch.long)]
            batch["projected_position_ids_prompt"] = [torch.zeros(3, 60, dtype=torch.long)]
            batch["projected_attention_mask_prompt"] = [torch.zeros(60, dtype=torch.long)]
            batch["rope_deltas_prompt"] = torch.tensor([-3])
    else:
        # Non-projected batch
        batch["input_ids"] = torch.ones(1, 20, dtype=torch.long)
        batch["attention_mask"] = torch.ones(1, 20, dtype=torch.long)

    return batch


class TestPrepareModelInputs:
    """Tests for prepare_model_inputs()."""

    def test_metadata_excluded(self):
        """All metadata keys must be filtered out."""
        batch = _make_batch(with_projection=True)
        model_inputs, _ = prepare_model_inputs(batch, torch.device("cpu"))

        for key in _METADATA_KEYS:
            assert key not in model_inputs, f"Metadata key '{key}' leaked into model_inputs"

    def test_private_keys_excluded(self):
        """Keys starting with '_' must be filtered out."""
        batch = _make_batch(with_projection=True)
        model_inputs, _ = prepare_model_inputs(batch, torch.device("cpu"))

        underscore_keys = [k for k in model_inputs if k.startswith("_")]
        assert underscore_keys == [], f"Private keys leaked: {underscore_keys}"

    def test_projection_done_preserved(self):
        """projection_done is a valid model parameter and must pass through."""
        batch = _make_batch(with_projection=True)
        model_inputs, _ = prepare_model_inputs(batch, torch.device("cpu"))

        assert "projection_done" in model_inputs
        assert model_inputs["projection_done"] is True

    def test_projected_keys_preserved(self):
        """Projected keys must pass through to model_inputs."""
        batch = _make_batch(with_projection=True)
        model_inputs, _ = prepare_model_inputs(batch, torch.device("cpu"))

        assert "projected_input_ids" in model_inputs
        assert "projected_position_ids" in model_inputs
        assert "projected_attention_mask" in model_inputs
        assert "projected_embeds" in model_inputs
        assert "rope_deltas" in model_inputs

    def test_dummy_input_ids_for_projected(self):
        """Projected batches should get dummy input_ids (zeros) and attention_mask (ones)."""
        batch = _make_batch(with_projection=True)
        model_inputs, input_len = prepare_model_inputs(batch, torch.device("cpu"))

        assert "input_ids" in model_inputs
        assert "attention_mask" in model_inputs
        assert (model_inputs["input_ids"] == 0).all()
        assert (model_inputs["attention_mask"] == 1).all()
        assert input_len == 80  # matches projected_input_ids length

    def test_prompt_variant_swap(self):
        """When prompt variants exist, they should replace the canonical projected keys."""
        batch = _make_batch(with_projection=True, with_prompt_variants=True)
        model_inputs, input_len = prepare_model_inputs(batch, torch.device("cpu"))

        # After swap, projected_input_ids should be the prompt version (60 tokens, zeros)
        assert model_inputs["projected_input_ids"][0].shape[0] == 60
        assert (model_inputs["projected_input_ids"][0] == 0).all()
        assert input_len == 60

        # rope_deltas should be the prompt version
        assert model_inputs["rope_deltas"].item() == -3

        # Prompt variant keys themselves must be excluded
        assert "projected_input_ids_prompt" not in model_inputs
        assert "projected_position_ids_prompt" not in model_inputs
        assert "projected_attention_mask_prompt" not in model_inputs
        assert "rope_deltas_prompt" not in model_inputs

    def test_non_projected_batch(self):
        """Non-projected batches should pass input_ids through directly."""
        batch = _make_batch(with_projection=False)
        model_inputs, input_len = prepare_model_inputs(batch, torch.device("cpu"))

        assert "input_ids" in model_inputs
        assert input_len == 20
        # Metadata should still be excluded
        assert "answer" not in model_inputs
        assert "question" not in model_inputs

    def test_no_string_values_leak(self):
        """No string or list-of-string values should appear in model_inputs."""
        batch = _make_batch(with_projection=True, with_prompt_variants=True)
        model_inputs, _ = prepare_model_inputs(batch, torch.device("cpu"))

        for k, v in model_inputs.items():
            if isinstance(v, str):
                pytest.fail(f"String value leaked: {k}={v!r}")
            if isinstance(v, list) and v and isinstance(v[0], str):
                pytest.fail(f"List[str] value leaked: {k}={v!r}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
