"""
LLM generation operations for Claude Memory Palace.

Provides functions for generating text using Ollama LLM models.
Uses aggressive VRAM management (keep_alive: 0) to allow model swapping.
"""

import requests
from typing import Optional

from .config import (
    get_ollama_url,
    get_llm_model,
    PREFERRED_LLM_MODELS,
)


# Module-level cache for detected LLM model
_detected_llm_model: Optional[str] = None


def _detect_llm_model() -> Optional[str]:
    """
    Auto-detect an available LLM model from Ollama.

    Queries Ollama for available models and returns the first one
    from our preferred list that's available.

    Returns:
        Model name if found, None if Ollama unavailable or no suitable model
    """
    global _detected_llm_model

    if _detected_llm_model is not None:
        return _detected_llm_model

    ollama_url = get_ollama_url()

    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        available_models = {m.get("name", "") for m in data.get("models", [])}

        # Find first preferred model that's available
        for preferred in PREFERRED_LLM_MODELS:
            if preferred in available_models:
                _detected_llm_model = preferred
                return preferred

            # Also check without tag suffix
            base_name = preferred.split(":")[0]
            for available in available_models:
                if available.startswith(base_name):
                    _detected_llm_model = available
                    return available

        # No preferred model found, skip embedding models and return first available
        for model in available_models:
            # Skip embedding-specific models
            if "embed" in model.lower():
                continue
            _detected_llm_model = model
            return model

        return None

    except requests.exceptions.RequestException:
        return None


def get_active_llm_model() -> Optional[str]:
    """
    Get the LLM model to use, either configured or auto-detected.

    Returns:
        Model name to use, or None if none available
    """
    # Check if explicitly configured
    configured = get_llm_model()
    if configured:
        return configured

    # Otherwise auto-detect
    return _detect_llm_model()


def generate_with_llm(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None
) -> Optional[str]:
    """
    Generate text using Ollama LLM.

    Args:
        prompt: The prompt to send to the LLM
        system: Optional system message to set model behavior
        model: Model to use (uses config/auto-detected if not specified)

    Returns:
        Generated text response, or None if Ollama unavailable or error
    """
    # Determine which model to use
    if model is None:
        model = get_active_llm_model()

    if model is None:
        # No model available
        return None

    ollama_url = get_ollama_url()

    try:
        request_body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": True,  # Enable Qwen3 thinking/reasoning mode
            "keep_alive": "0",  # Unload model immediately - aggressive VRAM strategy
            "options": {
                "num_ctx": 65536,   # 64K context window - full Qwen capacity
                "flash_attn": True  # Flash attention - ~2x KV cache efficiency
            }
        }
        if system:
            request_body["system"] = system

        response = requests.post(
            f"{ollama_url}/api/generate",
            json=request_body,
            timeout=180  # Transcripts can be long, thinking takes time
        )
        response.raise_for_status()
        data = response.json()
        # With think:true, response has "thinking" (reasoning) and "response" (answer)
        # We only return the final answer, but thinking trace is in data["thinking"]
        return data.get("response")
    except requests.exceptions.RequestException as e:
        # Ollama unavailable or error - fail gracefully
        print(f"LLM generation failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        # Malformed response
        print(f"LLM response parsing failed: {e}")
        return None


def is_llm_available() -> bool:
    """
    Check if an LLM model is available for generation.

    Returns:
        True if a model is available, False otherwise
    """
    return get_active_llm_model() is not None


def clear_model_cache() -> None:
    """Clear the detected model cache, forcing re-detection on next call."""
    global _detected_llm_model
    _detected_llm_model = None
