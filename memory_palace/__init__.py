"""
Claude Memory Palace - Persistent memory system for Claude instances.

Provides semantic search over memories and handoff messaging between
Claude instances. Instances are configured in ~/.memory-palace/config.json.
"""

__version__ = "0.1.0"
__author__ = "Jeff Pierce"
__email__ = "jeff.r.pierce@gmail.com"

from memory_palace.models import Memory, HandoffMessage
from memory_palace.database import get_engine, get_session, init_db

__all__ = [
    "__version__",
    "Memory",
    "HandoffMessage",
    "get_engine",
    "get_session",
    "init_db",
]
