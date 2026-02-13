"""
Configuration for Memory Palace v2.

Supports both PostgreSQL (primary) and SQLite (legacy/migration).
PostgreSQL is required for full 2.0 functionality (knowledge graph, native vector search).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


# Default data directory: ~/.memory-palace/
DEFAULT_DATA_DIR = Path.home() / ".memory-palace"
CONFIG_FILE_NAME = "config.json"

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    # Database configuration
    "database": {
        "type": "sqlite",  # "sqlite" (default, zero-install) or "postgres" (upgrade path)
        "url": None,  # PostgreSQL connection URL, or None to use default SQLite path
        # Default: postgresql://localhost:5432/memory_palace (if type=postgres and url=None)
        # Default: sqlite:///~/.memory-palace/memories.db (if type=sqlite and url=None)
    },
    # Ollama configuration
    "ollama_url": "http://localhost:11434",
    "embedding_model": None,  # Auto-detected from Ollama
    "embedding_dimension": 768,  # Default for nomic-embed-text
    "llm_model": None,  # Auto-detected from Ollama
    # Synthesis configuration
    "synthesis": {
        "enabled": True,  # Set False to disable local LLM synthesis (AWS/GPU-free mode)
        # When disabled, memory_recall always returns raw memories regardless of
        # synthesize parameter. The calling agent handles synthesis via its own
        # reasoning or sub-agents. Useful when:
        # - No local GPU available (AWS deployment)
        # - GPU is busy (gaming, ComfyUI, etc.)
        # - You want the AI assistant to do the reasoning instead of local Qwen
    },
    # Auto-linking configuration (creates edges at remember() time)
    "auto_link": {
        "enabled": True,  # Set False to disable automatic edge creation
        "link_threshold": 0.75,  # Auto-create edges at this similarity and above
        "suggest_threshold": 0.675,  # Surface suggestions between this and link_threshold
        "max_suggestions": 10,  # Cap on suggested (non-auto) links returned
        "same_project_only": True,  # Only link to memories in the same project
        "classify_edges": True,  # Use LLM to classify edge types (vs all relates_to)
        "classification_model": None,  # Auto-detected; prefers small models for speed
    },
    # MCP output format
    "toon_output": True,  # Use TOON encoding for MCP responses (saves ~30% tokens)
    # Instance configuration
    # Instances represent different AI contexts (e.g., "prime", "code", "desktop").
    # Configure your actual instances in ~/.memory-palace/config.json — this default
    # is only a fallback. The MEMORY_PALACE_INSTANCE_ID env var auto-adds to the list.
    "instances": ["default"],
}

# Preferred models for auto-detection (in order of preference)
# nomic-embed-text is preferred: 768d fits pgvector HNSW limits, runs well on CPU
PREFERRED_EMBEDDING_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    # sfr-embedding-mistral is high quality but 4096d exceeds pgvector 0.6.0 index limits
    # Uncomment if using pgvector 0.7+ or brute-force search is acceptable
    # "avr/sfr-embedding-mistral:f16",
    # "sfr-embedding-mistral:f16",
    # "sfr-embedding-mistral",
]

PREFERRED_LLM_MODELS = [
    "qwen3:14b",
    "qwen3:8b", 
    "qwen3:4b",
    "llama3.2",
    "llama3.1",
    "mistral",
]

# Small models preferred for edge classification (CPU-friendly, fast inference)
# These only need to return a single word from a constrained set
PREFERRED_CLASSIFICATION_MODELS = [
    "qwen3:1.7b",
    "qwen3:0.6b",
    "gemma3:1b",
    "llama3.2:1b",
    "phi3:mini",
]

# Model dimensions (for pgvector column sizing)
# Note: pgvector 0.6.0 HNSW/IVFFlat indexes max at 2000 dimensions
MODEL_DIMENSIONS = {
    "nomic-embed-text": 768,  # Recommended: fits index limits, fast on CPU
    "mxbai-embed-large": 1024,
    "sfr-embedding-mistral": 4096,  # Exceeds index limits in pgvector <0.7
    "avr/sfr-embedding-mistral:f16": 4096,
    "sfr-embedding-mistral:f16": 4096,
}

# Module-level config cache
_config_cache: Optional[Dict[str, Any]] = None


def get_config_path() -> Path:
    """Get the path to the config file."""
    data_dir = Path(os.environ.get("MEMORY_PALACE_DATA_DIR", DEFAULT_DATA_DIR))
    return data_dir / CONFIG_FILE_NAME


def load_config() -> Dict[str, Any]:
    """
    Load configuration from JSON file, with defaults for missing values.

    Environment variables can override config file values:
    - MEMORY_PALACE_DATA_DIR: Override data directory
    - MEMORY_PALACE_DATABASE_URL: Override database URL
    - OLLAMA_HOST: Override ollama_url
    - MEMORY_PALACE_EMBEDDING_MODEL: Override embedding_model
    - MEMORY_PALACE_LLM_MODEL: Override llm_model
    - MEMORY_PALACE_INSTANCE_ID: Override default instance

    Returns:
        Dict containing configuration values
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    config = _deep_copy_config(DEFAULT_CONFIG)
    config_path = get_config_path()

    # Load from file if it exists
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                _deep_merge(config, file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    # Environment variable overrides
    if os.environ.get("MEMORY_PALACE_DATABASE_URL"):
        config["database"]["url"] = os.environ["MEMORY_PALACE_DATABASE_URL"]
        # Infer type from URL
        url = config["database"]["url"]
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            config["database"]["type"] = "postgres"
        elif url.startswith("sqlite://"):
            config["database"]["type"] = "sqlite"

    if os.environ.get("OLLAMA_HOST"):
        config["ollama_url"] = os.environ["OLLAMA_HOST"]

    if os.environ.get("MEMORY_PALACE_EMBEDDING_MODEL"):
        config["embedding_model"] = os.environ["MEMORY_PALACE_EMBEDDING_MODEL"]

    if os.environ.get("MEMORY_PALACE_LLM_MODEL"):
        config["llm_model"] = os.environ["MEMORY_PALACE_LLM_MODEL"]

    if os.environ.get("MEMORY_PALACE_INSTANCE_ID"):
        default_instance = os.environ["MEMORY_PALACE_INSTANCE_ID"]
        if default_instance not in config["instances"]:
            config["instances"].append(default_instance)

    _config_cache = config
    return config


def _deep_copy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy a config dict."""
    result = {}
    for k, v in config.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_config(v)
        elif isinstance(v, list):
            result[k] = v.copy()
        else:
            result[k] = v
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Deep merge override into base, modifying base in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def get_database_url() -> str:
    """
    Get the database URL for SQLAlchemy.

    Returns:
        Database URL string
    """
    config = load_config()
    db_config = config.get("database", {})
    db_type = db_config.get("type", "postgres")
    db_url = db_config.get("url")

    if db_url:
        return db_url

    # Default URLs
    if db_type == "postgres":
        return "postgresql://localhost:5432/memory_palace"
    else:
        # SQLite fallback
        data_dir = ensure_data_dir()
        return f"sqlite:///{data_dir}/memories.db"


def get_database_type() -> str:
    """
    Get the database type ("postgres" or "sqlite").

    Returns:
        Database type string
    """
    config = load_config()
    db_config = config.get("database", {})
    return db_config.get("type", "postgres")


def is_postgres() -> bool:
    """Check if using PostgreSQL."""
    return get_database_type() == "postgres"


def is_sqlite() -> bool:
    """Check if using SQLite."""
    return get_database_type() == "sqlite"


def get_embedding_dimension() -> int:
    """
    Get the embedding dimension for the configured model.

    Returns:
        Embedding dimension (default 4096 for sfr-embedding-mistral)
    """
    config = load_config()
    
    # Check if explicitly configured
    if config.get("embedding_dimension"):
        return config["embedding_dimension"]
    
    # Infer from model name
    model = config.get("embedding_model")
    if model:
        for model_prefix, dim in MODEL_DIMENSIONS.items():
            if model.startswith(model_prefix):
                return dim
    
    # Default
    return 4096


def save_config(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dict to save. If None, saves current config.
    """
    global _config_cache

    if config is None:
        config = load_config()

    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    _config_cache = config


def clear_config_cache() -> None:
    """Clear the config cache, forcing reload on next access."""
    global _config_cache
    _config_cache = None


def get_ollama_url() -> str:
    """Get the Ollama base URL from config."""
    return load_config().get("ollama_url", DEFAULT_CONFIG["ollama_url"])


def get_embedding_model() -> Optional[str]:
    """Get the configured embedding model, or None for auto-detection."""
    return load_config().get("embedding_model")


def get_llm_model() -> Optional[str]:
    """Get the configured LLM model, or None for auto-detection."""
    return load_config().get("llm_model")


def get_instances() -> List[str]:
    """
    Get the list of configured instance IDs.

    Instances are configured in ~/.memory-palace/config.json under the
    "instances" key. The MEMORY_PALACE_INSTANCE_ID env var automatically
    appends to this list if not already present (see load_config()).

    Returns:
        List of valid instance ID strings (e.g., ["prime", "code", "desktop", ...])
    """
    return load_config().get("instances", DEFAULT_CONFIG["instances"])


def is_synthesis_enabled() -> bool:
    """
    Check if local LLM synthesis is enabled.
    
    When False, memory_recall should always return raw memories and let
    the calling agent handle synthesis. Useful for:
    - AWS deployment (no local GPU)
    - GPU-busy scenarios (gaming, image generation)
    - Preferring the calling model's reasoning over local synthesis
    
    Returns:
        True if local synthesis is enabled, False to always return raw
    
    Usage in recall():
        from memory_palace.config_v2 import is_synthesis_enabled
        
        def recall(..., synthesize: bool = True):
            # Override synthesize if config disables it
            if not is_synthesis_enabled():
                synthesize = False
            ...
    """
    config = load_config()
    synthesis_config = config.get("synthesis", {})
    return synthesis_config.get("enabled", True)


def get_auto_link_config() -> Dict[str, Any]:
    """
    Get auto-linking configuration.

    Returns:
        Dict with auto_link settings:
        - enabled: bool (default True)
        - link_threshold: float (default 0.75) — auto-create edges at this similarity+
        - suggest_threshold: float (default 0.675) — surface suggestions between this and link_threshold
        - max_suggestions: int (default 10) — cap on suggested (non-auto) links
        - same_project_only: bool (default True)
        - classify_edges: bool (default True)
        - classification_model: str or None
    """
    config = load_config()
    auto_link = config.get("auto_link", {})
    return {
        "enabled": auto_link.get("enabled", True),
        "link_threshold": auto_link.get("link_threshold", 0.75),
        "suggest_threshold": auto_link.get("suggest_threshold", 0.675),
        "max_suggestions": auto_link.get("max_suggestions", 10),
        "same_project_only": auto_link.get("same_project_only", True),
        "classify_edges": auto_link.get("classify_edges", True),
        "classification_model": auto_link.get("classification_model", None),
    }


def is_toon_output_enabled() -> bool:
    """
    Check if TOON output encoding is enabled for MCP responses.

    When True (default), MCP tool responses are TOON-encoded to save tokens.
    When False, raw JSON/dict responses are returned.

    Returns:
        True if TOON encoding is enabled, False for raw JSON
    """
    config = load_config()
    return config.get("toon_output", True)


def ensure_data_dir() -> Path:
    """Create data directory if it doesn't exist."""
    data_dir = Path(os.environ.get("MEMORY_PALACE_DATA_DIR", DEFAULT_DATA_DIR))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# Legacy compatibility - will be removed in future versions
def get_legacy_database_url() -> str:
    """Get SQLite database URL for migration purposes."""
    data_dir = ensure_data_dir()
    return f"sqlite:///{data_dir}/memories.db"
