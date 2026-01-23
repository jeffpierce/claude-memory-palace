"""
MCP Tools for Claude Memory Palace.

Each tool is in its own module for maintainability.
"""

from .remember import register_remember
from .recall import register_recall
from .forget import register_forget
from .memory_stats import register_memory_stats
from .backfill_embeddings import register_backfill_embeddings
from .send_handoff import register_send_handoff
from .get_handoffs import register_get_handoffs
from .mark_handoff_read import register_mark_handoff_read
from .reflect import register_reflect
from .jsonl_to_toon import register_jsonl_to_toon
from .get_memory import register_get_memory


def register_all_tools(mcp):
    """Register all Memory Palace tools with the MCP server."""
    register_remember(mcp)
    register_recall(mcp)
    register_forget(mcp)
    register_get_memory(mcp)
    register_memory_stats(mcp)
    register_backfill_embeddings(mcp)
    register_send_handoff(mcp)
    register_get_handoffs(mcp)
    register_mark_handoff_read(mcp)
    register_reflect(mcp)
    register_jsonl_to_toon(mcp)


__all__ = ["register_all_tools"]
