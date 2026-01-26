"""
Embedding operations for Claude Memory Palace.

Provides functions for generating embeddings and computing similarity using Ollama.
Uses aggressive VRAM management (keep_alive: 0) to allow model swapping.
"""

import math
import requests
from typing import List, Optional

from .config import (
    get_ollama_url,
    get_embedding_model,
    PREFERRED_EMBEDDING_MODELS,
)


# Module-level cache for detected embedding model
_detected_embedding_model: Optional[str] = None


def _detect_embedding_model() -> Optional[str]:
    """
    Auto-detect an available embedding model from Ollama.

    Queries Ollama for available models and returns the first one
    from our preferred list that's available.

    Returns:
        Model name if found, None if Ollama unavailable or no suitable model
    """
    global _detected_embedding_model

    if _detected_embedding_model is not None:
        return _detected_embedding_model

    ollama_url = get_ollama_url()

    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        available_models = {m.get("name", "") for m in data.get("models", [])}

        # Find first preferred model that's available
        for preferred in PREFERRED_EMBEDDING_MODELS:
            if preferred in available_models:
                _detected_embedding_model = preferred
                return preferred

            # Also check without tag suffix
            base_name = preferred.split(":")[0]
            for available in available_models:
                if available.startswith(base_name):
                    _detected_embedding_model = available
                    return available

        # No preferred model found, return first embedding-like model
        for model in available_models:
            if "embed" in model.lower():
                _detected_embedding_model = model
                return model

        return None

    except requests.exceptions.RequestException:
        return None


def get_active_embedding_model() -> Optional[str]:
    """
    Get the embedding model to use, either configured or auto-detected.

    Returns:
        Model name to use, or None if none available
    """
    # Check if explicitly configured
    configured = get_embedding_model()
    if configured:
        return configured

    # Otherwise auto-detect
    return _detect_embedding_model()


def get_embedding(text: str, model: Optional[str] = None) -> Optional[List[float]]:
    """
    Get embedding vector for text using Ollama.

    Args:
        text: Text to embed
        model: Model to use (uses config/auto-detected if not specified)

    Returns:
        List of floats representing the embedding, or None if Ollama unavailable
    """
    if not text or not text.strip():
        return None

    # Determine which model to use
    if model is None:
        model = get_active_embedding_model()

    if model is None:
        # No model available
        return None

    ollama_url = get_ollama_url()

    try:
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={
                "model": model,
                "prompt": text,
                "keep_alive": "0"  # Unload model immediately - aggressive VRAM strategy
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding")
    except requests.exceptions.RequestException:
        # Ollama unavailable or error - fail gracefully
        return None
    except (KeyError, ValueError):
        # Malformed response
        return None


def cosine_similarity(a, b) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector (list or numpy array)
        b: Second vector (list or numpy array)

    Returns:
        Similarity score between -1 and 1
    """
    # Handle None cases
    if a is None or b is None:
        return 0.0
    
    # Convert to list if numpy array
    try:
        if hasattr(a, 'tolist'):
            a = a.tolist()
        if hasattr(b, 'tolist'):
            b = b.tolist()
    except Exception:
        pass
    
    # Check length match
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def is_ollama_available() -> bool:
    """
    Check if Ollama is running and accessible.

    Returns:
        True if Ollama is accessible, False otherwise
    """
    ollama_url = get_ollama_url()

    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def clear_model_cache() -> None:
    """Clear the detected model cache, forcing re-detection on next call."""
    global _detected_embedding_model
    _detected_embedding_model = None
