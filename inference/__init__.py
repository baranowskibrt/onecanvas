"""Inference pipeline: end-to-end vision processing and generation."""

from .pipeline import (
    load_onecanvas_model,
    process_vision_and_generate,
    answer_scene_question,
    _compute_da3_geometry,
)

__all__ = [
    "load_onecanvas_model",
    "process_vision_and_generate",
    "answer_scene_question",
]
