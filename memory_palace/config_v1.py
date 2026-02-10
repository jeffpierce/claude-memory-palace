"""
Configuration for Memory Palace.

Handles paths, defaults, JSON config file, and environment-based overrides.
Configuration is loaded from ~/.memory-palace/config.json with sensible defaults.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default data directory: ~/.memory-palace/
DEFAULT_DATA_DIR = Path.home() / ".memory-palace"
CONFIG_FILE_NAME = "config.json"

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "ollama_url": "http://localhost:11434",
    "embedding_model": None,  # Auto-detected from Ollama
    "llm_model": None,  # Auto-detected from Ollama
    "instances": ["default"],  # User-configurable instance list
    "db_path": "~/.memory-palace/memories.db",
}

# Preferred models for auto-detection (in order of preference)
PREFERRED_EMBEDDING_MODELS = [
    "avr/sfr-embedding-mistral:f16",
    "sfr-embedding-mistral:f16",
    "sfr-embedding-mistral",
    "nomic-embed-text",
    "mxbai-embed-large",
]

PREFERRED_LLM_MODELS = [
    "qwen3:14b",
    "qwen3:8b",
    "qwen3:4b",
    "llama3.2",
    "llama3.1",
    "mistral",
]

# Module-level config cache
_config_cache: Optional[Dict[str, Any]] = None


def get_config_path() -> Path:
    """Get the path to the config file."""
    data_dir = Path(os.environ.get("MEMORY_PALACE_DATA_DIR", DEFAULT_DATA_DIR))
    return data_dir / CONFIG_FILE_NAME


def get_db_path() -> Path:
    """
    Get the database path, expanding ~ and creating directory if needed.

    Returns:
        Path: Absolute path to the database file
    """
    config = load_config()
    db_path = Path(config.get("db_path", DEFAULT_CONFIG["db_path"])).expanduser()

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return db_path


def load_config() -> Dict[str, Any]:
    """
    Load configuration from JSON file, with defaults for missing values.

    Environment variables can override config file values:
    - MEMORY_PALACE_DATA_DIR: Override data directory
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

    config = DEFAULT_CONFIG.copy()
    config_path = get_config_path()

    # Load from file if it exists
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                config.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            # Log but continue with defaults
            print(f"Warning: Could not load config from {config_path}: {e}")

    # Environment variable overrides
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

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Update cache
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
    """Get the list of configured instance IDs."""
    return load_config().get("instances", DEFAULT_CONFIG["instances"])


def ensure_data_dir() -> Path:
    """Create data directory if it doesn't exist."""
    data_dir = Path(os.environ.get("MEMORY_PALACE_DATA_DIR", DEFAULT_DATA_DIR))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# Legacy compatibility exports
DATA_DIR = Path(os.environ.get("MEMORY_PALACE_DATA_DIR", DEFAULT_DATA_DIR))
DATABASE_PATH = DATA_DIR / "memories.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
DEFAULT_INSTANCE_ID = os.environ.get("MEMORY_PALACE_INSTANCE_ID", "unknown")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("MEMORY_PALACE_EMBEDDING_MODEL", "sfr-embedding-mistral:f16")
