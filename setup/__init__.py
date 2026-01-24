"""Setup utilities for Claude Memory Palace."""

from .detect_gpu import detect_gpu
from .model_recommendations import get_recommended_models
from .first_run import run_setup_wizard
from .configure_claude import main as configure_claude_desktop, configure_claude_desktop as configure_claude_desktop_func

__all__ = ["detect_gpu", "get_recommended_models", "run_setup_wizard", "configure_claude_desktop"]
