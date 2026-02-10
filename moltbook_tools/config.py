"""
Configuration for Moltbook gateway.

Config file: ~/.moltbook/config.json
Credentials searched in order:
1. ~/.moltbook/config.json -> api_key
2. ~/.config/moltbook/credentials.json -> api_key
3. MOLTBOOK_API_KEY environment variable
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


_DEFAULTS = {
    "api_base_url": "https://www.moltbook.com/api/v1",
    "rate_limits": {
        "post_cooldown_seconds": 1800,
        "comment_cooldown_seconds": 20,
    },
    "word_limits": {
        "post_max_words": 1000,
        "comment_max_words": 300,
    },
    "qc_token_ttl_minutes": 30,
    "similarity_threshold": 0.85,
    "similarity_lookback_hours": 72,
}


@dataclass
class GatewayConfig:
    """Gateway configuration."""
    api_base_url: str = "https://www.moltbook.com/api/v1"
    api_key: Optional[str] = None
    post_cooldown_seconds: int = 1800
    comment_cooldown_seconds: int = 20
    post_max_words: int = 1000
    comment_max_words: int = 300
    qc_token_ttl_minutes: int = 30
    similarity_threshold: float = 0.85
    similarity_lookback_hours: int = 72


def _load_json_file(path: Path) -> dict:
    """Load a JSON file, return empty dict if missing or invalid."""
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _find_api_key(config_data: dict) -> Optional[str]:
    """
    Search for API key in priority order:
    1. config.json api_key field
    2. ~/.config/moltbook/credentials.json
    3. MOLTBOOK_API_KEY env var
    """
    # 1. Config file
    if config_data.get("api_key"):
        return config_data["api_key"]

    # 2. Moltbook's recommended credential location
    creds_path = Path.home() / ".config" / "moltbook" / "credentials.json"
    creds = _load_json_file(creds_path)
    if creds.get("api_key"):
        return creds["api_key"]

    # 3. Environment variable
    env_key = os.environ.get("MOLTBOOK_API_KEY")
    if env_key:
        return env_key

    return None


def load_config() -> GatewayConfig:
    """
    Load gateway configuration.

    Merges defaults with config file values.
    Searches for API key across multiple sources.
    """
    config_path = Path.home() / ".moltbook" / "config.json"
    data = _load_json_file(config_path)

    # Merge with defaults
    merged = {**_DEFAULTS, **data}
    rate_limits = {**_DEFAULTS["rate_limits"], **merged.get("rate_limits", {})}
    word_limits = {**_DEFAULTS["word_limits"], **merged.get("word_limits", {})}

    api_key = _find_api_key(data)

    return GatewayConfig(
        api_base_url=merged.get("api_base_url", _DEFAULTS["api_base_url"]),
        api_key=api_key,
        post_cooldown_seconds=rate_limits["post_cooldown_seconds"],
        comment_cooldown_seconds=rate_limits["comment_cooldown_seconds"],
        post_max_words=word_limits["post_max_words"],
        comment_max_words=word_limits["comment_max_words"],
        qc_token_ttl_minutes=merged.get("qc_token_ttl_minutes", _DEFAULTS["qc_token_ttl_minutes"]),
        similarity_threshold=merged.get("similarity_threshold", _DEFAULTS["similarity_threshold"]),
        similarity_lookback_hours=merged.get("similarity_lookback_hours", _DEFAULTS["similarity_lookback_hours"]),
    )
