"""Setup utilities for Claude Memory Palace."""

from .detect_gpu import detect_gpu
from .model_recommendations import get_recommended_models
from .first_run import run_setup_wizard

__all__ = ["detect_gpu", "get_recommended_models", "run_setup_wizard"]
