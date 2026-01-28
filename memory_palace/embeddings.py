"""
Embedding operations for Claude Memory Palace.

Provides functions for generating embeddings and computing similarity using Ollama.
Uses aggressive VRAM management (keep_alive: 0) to allow model swapping.

Reliability guarantees:
- Retries with exponential backoff on transient failures (cold model load, timeout)
- Truncates oversized input to fit model context window rather than silently failing
- Surfaces Ollama error responses explicitly instead of swallowing them
- Logs all failures for diagnostics
"""

import logging
import math
import time
import requests
from typing import List, Optional

from .config import (
    get_ollama_url,
    get_embedding_model,
    PREFERRED_EMBEDDING_MODELS,
)

logger = logging.getLogger(__name__)

# nomic-embed-text has 8192 token context. Tokenization ratio varies:
# - English prose: ~1.2-1.5 chars/token (~5500-6800 chars)
# - Code/technical: ~1.0-1.2 chars/token (~6800-8192 chars)
# - Worst case (single chars): 1 char = 1 token (8192 chars)
# We use 6000 chars as a safe default that handles all content types
# with margin for model overhead tokens (BOS, EOS, etc.).
DEFAULT_MAX_EMBEDDING_CHARS = 6000

# Retry configuration for transient failures (cold model load, etc.)
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


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


def _truncate_for_embedding(text: str, max_chars: int = DEFAULT_MAX_EMBEDDING_CHARS) -> str:
    """
    Truncate text to fit within the embedding model's context window.

    Preserves the beginning of the text (subject/type prefix) and truncates
    the end, which is typically the body content. Adds a marker so we know
    truncation happened.

    Args:
        text: Text to truncate
        max_chars: Maximum character length

    Returns:
        Text truncated to fit, with marker if truncation occurred
    """
    if len(text) <= max_chars:
        return text

    # Reserve space for truncation marker
    marker = "\n[TRUNCATED FOR EMBEDDING]"
    truncated = text[:max_chars - len(marker)] + marker
    logger.info(
        "Truncated embedding text from %d to %d chars (limit: %d)",
        len(text), len(truncated), max_chars
    )
    return truncated


def get_embedding(text: str, model: Optional[str] = None) -> Optional[List[float]]:
    """
    Get embedding vector for text using Ollama.

    Reliability features:
    - Truncates oversized input to fit model context window
    - Retries with exponential backoff on transient failures
    - Checks Ollama error responses explicitly (not just HTTP status)
    - Logs all failure modes for diagnostics

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
        logger.warning("No embedding model available (Ollama not running or no model installed)")
        return None

    # Truncate to fit model context window
    text = _truncate_for_embedding(text)

    ollama_url = get_ollama_url()
    last_error = None

    for attempt in range(EMBEDDING_MAX_RETRIES):
        try:
            # First attempt uses standard timeout; retries use longer timeout
            # to account for cold model loading
            timeout = 30 if attempt == 0 else 60

            response = requests.post(
                f"{ollama_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text,
                    "keep_alive": "0"  # Unload model immediately - aggressive VRAM strategy
                },
                timeout=timeout
            )

            # Parse response body BEFORE raise_for_status — Ollama sometimes
            # returns error details in the body of a 500 response
            try:
                data = response.json()
            except ValueError:
                data = {}

            # Check for Ollama-level errors (can come as 200 or 500 with error field)
            if "error" in data:
                error_msg = data["error"]
                logger.error(
                    "Ollama embedding error (attempt %d/%d): %s",
                    attempt + 1, EMBEDDING_MAX_RETRIES, error_msg
                )
                # Context length errors won't be fixed by retry — this shouldn't
                # happen anymore due to truncation, but handle it defensively
                if "context length" in error_msg.lower():
                    logger.error(
                        "Input exceeds context length even after truncation "
                        "(%d chars). This is a bug — please report it.",
                        len(text)
                    )
                    return None
                last_error = error_msg
                # Other Ollama errors might be transient, retry
                if attempt < EMBEDDING_MAX_RETRIES - 1:
                    delay = EMBEDDING_RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                continue

            embedding = data.get("embedding")
            if embedding and len(embedding) > 0:
                if attempt > 0:
                    logger.info(
                        "Embedding succeeded on attempt %d/%d",
                        attempt + 1, EMBEDDING_MAX_RETRIES
                    )
                return embedding
            else:
                logger.warning(
                    "Ollama returned empty embedding (attempt %d/%d). "
                    "Response keys: %s",
                    attempt + 1, EMBEDDING_MAX_RETRIES, list(data.keys())
                )
                last_error = "empty embedding returned"

        except requests.exceptions.ConnectionError as e:
            last_error = f"connection error: {e}"
            logger.warning(
                "Ollama connection failed (attempt %d/%d): %s",
                attempt + 1, EMBEDDING_MAX_RETRIES, e
            )
        except requests.exceptions.Timeout:
            last_error = "timeout"
            logger.warning(
                "Ollama embedding timed out (attempt %d/%d, timeout=%ds)",
                attempt + 1, EMBEDDING_MAX_RETRIES, timeout
            )
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            logger.warning(
                "Ollama request failed (attempt %d/%d): %s",
                attempt + 1, EMBEDDING_MAX_RETRIES, e
            )
        except (KeyError, ValueError) as e:
            last_error = f"malformed response: {e}"
            logger.warning(
                "Malformed Ollama response (attempt %d/%d): %s",
                attempt + 1, EMBEDDING_MAX_RETRIES, e
            )

        # Exponential backoff before retry
        if attempt < EMBEDDING_MAX_RETRIES - 1:
            delay = EMBEDDING_RETRY_BASE_DELAY * (2 ** attempt)
            logger.info("Retrying embedding in %.1fs...", delay)
            time.sleep(delay)

    logger.error(
        "Embedding failed after %d attempts. Last error: %s. Text length: %d chars",
        EMBEDDING_MAX_RETRIES, last_error, len(text)
    )
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
