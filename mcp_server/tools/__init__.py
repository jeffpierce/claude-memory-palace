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
from .link import register_link
from .unlink import register_unlink
from .related import register_related
from .supersede import register_supersede
from .graph import register_graph
from .code_remember import register_code_remember
from .code_recall import register_code_recall
from .audit import register_audit
from .batch_archive import register_batch_archive
from .reembed import register_reembed


def register_all_tools(mcp):
    """Register all Memory Palace tools with the MCP server."""
    # Core memory operations
    register_remember(mcp)
    register_recall(mcp)
    register_forget(mcp)
    register_get_memory(mcp)
    register_memory_stats(mcp)
    register_backfill_embeddings(mcp)
    # Handoff messaging
    register_send_handoff(mcp)
    register_get_handoffs(mcp)
    register_mark_handoff_read(mcp)
    # Reflection/processing
    register_reflect(mcp)
    register_jsonl_to_toon(mcp)
    # Knowledge graph
    register_link(mcp)
    register_unlink(mcp)
    register_related(mcp)
    register_supersede(mcp)
    register_graph(mcp)
    # Code retrieval
    register_code_remember(mcp)
    register_code_recall(mcp)
    # Maintenance
    register_audit(mcp)
    register_batch_archive(mcp)
    register_reembed(mcp)


__all__ = ["register_all_tools"]
