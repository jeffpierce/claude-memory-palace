"""Model recommendations based on available VRAM."""

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    size_gb: float
    description: str
    ollama_name: str


# Embedding models ordered by quality (best last)
EMBEDDING_MODELS = [
    ModelInfo(
        name="nomic-embed-text",
        size_gb=0.3,
        description="Lightweight, good quality (768-dim)",
        ollama_name="nomic-embed-text"
    ),
    ModelInfo(
        name="snowflake-arctic-embed-l",
        size_gb=1.0,
        description="Strong performance, efficient (1024-dim)",
        ollama_name="snowflake-arctic-embed:335m"
    ),
    ModelInfo(
        name="sfr-embedding-mistral",
        size_gb=14.0,
        description="Best quality, MTEB #2 (4096-dim, 32K context)",
        ollama_name="sfr-embedding-mistral:f16"
    ),
]

# LLM models for reflection/extraction
LLM_MODELS = [
    ModelInfo(
        name="qwen2.5:3b",
        size_gb=2.0,
        description="Minimal, basic extraction capability",
        ollama_name="qwen2.5:3b"
    ),
    ModelInfo(
        name="qwen2.5:7b",
        size_gb=4.5,
        description="Good balance of speed and quality",
        ollama_name="qwen2.5:7b"
    ),
    ModelInfo(
        name="qwen3:8b",
        size_gb=5.0,
        description="Newer architecture, better reasoning",
        ollama_name="qwen3:8b"
    ),
    ModelInfo(
        name="qwen3:14b",
        size_gb=9.0,
        description="Best extraction quality, needs headroom",
        ollama_name="qwen3:14b"
    ),
]


# VRAM tier configurations
# Each tier specifies what fits comfortably with some headroom
VRAM_TIERS: Dict[int, Dict[str, str]] = {
    4: {
        "embed": "nomic-embed-text",
        "llm": "qwen2.5:3b",
        "notes": "Tight fit. Consider CPU fallback for LLM.",
    },
    6: {
        "embed": "nomic-embed-text",
        "llm": "qwen2.5:7b",
        "notes": "Works but may need model swapping.",
    },
    8: {
        "embed": "snowflake-arctic-embed-l",
        "llm": "qwen2.5:7b",
        "notes": "Comfortable for both models.",
    },
    10: {
        "embed": "snowflake-arctic-embed-l",
        "llm": "qwen3:8b",
        "notes": "Good headroom for concurrent use.",
    },
    12: {
        "embed": "snowflake-arctic-embed-l",
        "llm": "qwen3:14b",
        "notes": "Can run larger LLM but not simultaneously.",
    },
    16: {
        "embed": "sfr-embedding-mistral",
        "llm": "qwen3:14b",
        "notes": "Premium models. Run one at a time (both ~14GB).",
    },
    24: {
        "embed": "sfr-embedding-mistral",
        "llm": "qwen3:14b",
        "notes": "Comfortable headroom. Could run both with careful management.",
    },
}


def get_vram_tier(vram_gb: float) -> int:
    """Get the appropriate VRAM tier for given VRAM amount."""
    tiers = sorted(VRAM_TIERS.keys())
    for tier in reversed(tiers):
        if vram_gb >= tier:
            return tier
    return tiers[0]  # Return minimum tier if less than 4GB


def get_recommended_models(vram_gb: float) -> Tuple[str, str]:
    """
    Get recommended embedding and LLM models based on available VRAM.

    Args:
        vram_gb: Available VRAM in gigabytes

    Returns:
        Tuple of (embedding_model_name, llm_model_name) - Ollama model names

    Examples:
        >>> get_recommended_models(8.0)
        ('snowflake-arctic-embed:335m', 'qwen2.5:7b')

        >>> get_recommended_models(16.0)
        ('sfr-embedding-mistral:f16', 'qwen3:14b')
    """
    tier = get_vram_tier(vram_gb)
    config = VRAM_TIERS[tier]

    # Find the ModelInfo objects for the recommended models
    embed_model = next(m for m in EMBEDDING_MODELS if m.name == config["embed"])
    llm_model = next(m for m in LLM_MODELS if m.name == config["llm"])

    return (embed_model.ollama_name, llm_model.ollama_name)


def get_model_details(vram_gb: float) -> Dict:
    """
    Get detailed model recommendations with explanations.

    Args:
        vram_gb: Available VRAM in gigabytes

    Returns:
        Dictionary with model recommendations and details
    """
    tier = get_vram_tier(vram_gb)
    config = VRAM_TIERS[tier]

    embed_model = next(m for m in EMBEDDING_MODELS if m.name == config["embed"])
    llm_model = next(m for m in LLM_MODELS if m.name == config["llm"])

    return {
        "vram_detected": vram_gb,
        "tier_used": tier,
        "embedding": {
            "name": embed_model.name,
            "ollama_name": embed_model.ollama_name,
            "size_gb": embed_model.size_gb,
            "description": embed_model.description,
        },
        "llm": {
            "name": llm_model.name,
            "ollama_name": llm_model.ollama_name,
            "size_gb": llm_model.size_gb,
            "description": llm_model.description,
        },
        "notes": config["notes"],
        "total_size_gb": embed_model.size_gb + llm_model.size_gb,
    }


def list_all_models() -> Dict[str, List[Dict]]:
    """List all available models with their details."""
    return {
        "embedding_models": [
            {
                "name": m.name,
                "ollama_name": m.ollama_name,
                "size_gb": m.size_gb,
                "description": m.description,
            }
            for m in EMBEDDING_MODELS
        ],
        "llm_models": [
            {
                "name": m.name,
                "ollama_name": m.ollama_name,
                "size_gb": m.size_gb,
                "description": m.description,
            }
            for m in LLM_MODELS
        ],
    }


if __name__ == "__main__":
    import json

    # Test different VRAM amounts
    test_vrams = [4, 6, 8, 10, 12, 16, 24]

    print("Model Recommendations by VRAM:\n")
    for vram in test_vrams:
        embed, llm = get_recommended_models(vram)
        details = get_model_details(vram)
        print(f"{vram}GB VRAM:")
        print(f"  Embedding: {embed} ({details['embedding']['size_gb']}GB)")
        print(f"  LLM: {llm} ({details['llm']['size_gb']}GB)")
        print(f"  Notes: {details['notes']}")
        print()
