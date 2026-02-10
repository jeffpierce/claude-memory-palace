"""
MCP Tools for Claude Memory Palace v2.0.

Each tool is in its own module for maintainability.
This module registers the 12 v2.0 tools (down from 25 in v1.0).
"""

# Core memory operations
from .remember import register_remember
from .recall import register_recall
from .get_memory import register_get_memory
from .archive import register_archive

# Knowledge graph
from .link import register_link
from .unlink import register_unlink

# Messaging
from .message import register_message

# Code indexing
from .code_remember import register_code_remember

# Maintenance
from .audit import register_audit
from .reembed import register_reembed
from .memory_stats import register_memory_stats

# Processing
from .reflect import register_reflect


def register_all_tools(mcp):
    """Register all Memory Palace v2.0 tools with the MCP server."""
    # Core memory operations
    register_remember(mcp)
    register_recall(mcp)
    register_get_memory(mcp)
    register_archive(mcp)

    # Knowledge graph
    register_link(mcp)
    register_unlink(mcp)

    # Messaging
    register_message(mcp)

    # Code indexing
    register_code_remember(mcp)

    # Maintenance
    register_audit(mcp)
    register_reembed(mcp)
    register_memory_stats(mcp)

    # Processing
    register_reflect(mcp)


__all__ = ["register_all_tools"]
