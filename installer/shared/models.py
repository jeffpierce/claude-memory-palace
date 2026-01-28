"""
Model selection and download for Claude Memory Palace installer.

Selects appropriate models based on GPU capabilities and manages Ollama pulls.
"""

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Callable

from .detect import GPUInfo


@dataclass
class ModelRecommendation:
    """Recommended models for the user's hardware."""
    embedding_model: str
    embedding_desc: str
    embedding_size: str  # Human-readable download size
    llm_model: Optional[str]
    llm_desc: Optional[str]
    llm_size: Optional[str]
    classification_model: Optional[str]
    classification_desc: Optional[str]
    classification_size: Optional[str]


def get_model_recommendation(gpu: GPUInfo) -> ModelRecommendation:
    """
    Get model recommendations based on detected GPU.
    
    Embedding: Always nomic-embed-text (small, fast, runs anywhere).
    Classification: Small model for edge type classification.
    LLM: Scales with available VRAM (optional — for local synthesis).
    """
    # Embedding is always the same — nomic is tiny and great
    embedding_model = "nomic-embed-text"
    embedding_desc = "Nomic Embed Text — compact, high-quality embeddings"
    embedding_size = "~270MB"

    # Classification model — small is fine, just returns one word
    classification_model = "qwen3:1.7b"
    classification_desc = "Qwen3 1.7B — fast edge classification"
    classification_size = "~1GB"

    # LLM scales with hardware
    # qwen3:1.7b is ALWAYS installed — it handles synthesis, classification,
    # and memory extraction. Without it, there's no local intelligence.
    # Larger models are optional upgrades for better quality.
    vram = gpu.vram_gb

    if vram >= 16:
        llm_model = "qwen3:14b"
        llm_desc = "Qwen3 14B — best quality for memory extraction"
        llm_size = "~9GB"
    elif vram >= 10:
        llm_model = "qwen3:8b"
        llm_desc = "Qwen3 8B — good quality for memory extraction"
        llm_size = "~5GB"
    elif vram >= 6:
        llm_model = "qwen3:4b"
        llm_desc = "Qwen3 4B — decent memory extraction"
        llm_size = "~2.5GB"
    else:
        # ≤2GB VRAM, CPU-only, or no GPU — qwen3:1.7b handles everything
        # Runs fine on CPU, just slower. ~1GB download.
        llm_model = "qwen3:1.7b"
        llm_desc = "Qwen3 1.7B — synthesis, classification, extraction"
        llm_size = "~1GB"
        # Same model covers classification — no separate pull needed
        classification_model = None
        classification_desc = None
        classification_size = None

    return ModelRecommendation(
        embedding_model=embedding_model,
        embedding_desc=embedding_desc,
        embedding_size=embedding_size,
        llm_model=llm_model,
        llm_desc=llm_desc,
        llm_size=llm_size,
        classification_model=classification_model,
        classification_desc=classification_desc,
        classification_size=classification_size,
    )


def pull_model(
    model_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: int = 1800,
) -> bool:
    """
    Pull an Ollama model.
    
    Args:
        model_name: Name of the model to pull
        progress_callback: Optional callback for status messages
        timeout: Timeout in seconds (default 30 minutes)
    
    Returns:
        True if successful, False otherwise
    """
    if progress_callback:
        progress_callback(f"Downloading {model_name}...")

    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(["ollama", "pull", model_name], **kwargs)

        if result.returncode == 0:
            if progress_callback:
                progress_callback(f"✓ {model_name} downloaded successfully")
            return True
        else:
            if progress_callback:
                error = result.stderr[:200] if result.stderr else "Unknown error"
                progress_callback(f"✗ Failed to download {model_name}: {error}")
            return False

    except subprocess.TimeoutExpired:
        if progress_callback:
            progress_callback(f"✗ Download timed out for {model_name}")
        return False
    except FileNotFoundError:
        if progress_callback:
            progress_callback("✗ Ollama not found — is it installed?")
        return False
    except Exception as e:
        if progress_callback:
            progress_callback(f"✗ Error downloading {model_name}: {str(e)[:100]}")
        return False


def check_model_installed(model_name: str) -> bool:
    """Check if an Ollama model is already installed."""
    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": 10,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(["ollama", "list"], **kwargs)
        if result.returncode == 0:
            return model_name.lower() in result.stdout.lower()
    except Exception:
        pass
    return False
