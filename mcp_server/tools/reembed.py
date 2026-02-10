"""Reembed tool for Claude Memory Palace MCP server."""
from typing import Any, Optional, List

from memory_palace.services.maintenance_service import reembed_memories
from mcp_server.toon_wrapper import toon_response


def register_reembed(mcp):
    """Register the reembed tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_reembed(
        older_than_days: Optional[int] = None,
        memory_ids: Optional[List[int]] = None,
        project: Optional[str] = None,
        all_memories: bool = False,
        missing_only: bool = False,
        batch_size: int = 50,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """
        Regenerate embeddings for memories. Use missing_only=True to backfill
        memories without embeddings (replaces memory_backfill_embeddings).

        SAFETY: dry_run=True by default.

        Args:
            older_than_days: Re-embed embeddings older than N days
            memory_ids: Explicit list of memory IDs to re-embed
            project: Filter by project
            all_memories: Re-embed everything (use with caution)
            missing_only: Only embed memories with NULL embeddings (backfill mode)
            batch_size: Processing batch size (default 50)
            dry_run: Preview only (default True)

        Returns:
            Preview or execution results with counts
        """
        return reembed_memories(
            older_than_days=older_than_days,
            memory_ids=memory_ids,
            project=project,
            all_memories=all_memories,
            missing_only=missing_only,
            batch_size=batch_size,
            dry_run=dry_run,
        )
