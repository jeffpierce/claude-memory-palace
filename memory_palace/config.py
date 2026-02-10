"""
Configuration for Memory Palace.

v2: Re-exports from config_v2 for PostgreSQL + pgvector support.
For legacy SQLite config, see config_v1.py.
"""

# Re-export everything from v2
from memory_palace.config_v2 import (
    # Config paths
    DEFAULT_DATA_DIR,
    CONFIG_FILE_NAME,
    DEFAULT_CONFIG,
    get_config_path,
    
    # Model lists
    PREFERRED_EMBEDDING_MODELS,
    PREFERRED_LLM_MODELS,
    PREFERRED_CLASSIFICATION_MODELS,
    MODEL_DIMENSIONS,
    
    # Config loading
    load_config,
    save_config,
    clear_config_cache,
    
    # Database config
    get_database_url,
    get_database_type,
    is_postgres,
    is_sqlite,
    
    # Model config
    get_embedding_dimension,
    get_ollama_url,
    get_embedding_model,
    get_llm_model,
    
    # Instance config
    get_instances,
    
    # Synthesis config
    is_synthesis_enabled,
    
    # Auto-link config
    get_auto_link_config,

    # TOON output config
    is_toon_output_enabled,

    # Utilities
    ensure_data_dir,
    get_legacy_database_url,
)

# Legacy compatibility aliases
DATABASE_URL = get_database_url()
DATA_DIR = DEFAULT_DATA_DIR
OLLAMA_HOST = get_ollama_url()
EMBEDDING_MODEL = get_embedding_model() or "nomic-embed-text"
DEFAULT_INSTANCE_ID = "unknown"  # Now configured per-call

# Legacy path helper
def get_db_path():
    """Legacy compatibility - returns path component of database URL."""
    from pathlib import Path
    url = get_database_url()
    if url.startswith("sqlite:///"):
        return Path(url.replace("sqlite:///", ""))
    return Path(DEFAULT_DATA_DIR) / "memories.db"

DATABASE_PATH = get_db_path()
