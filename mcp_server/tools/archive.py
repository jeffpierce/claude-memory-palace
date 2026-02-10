"""Archive tool for Memory Palace MCP server."""
from typing import Any, List, Optional

from memory_palace.services.memory_service import archive_memory
from mcp_server.toon_wrapper import toon_response


def register_archive(mcp):
    """Register the archive tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_archive(
        memory_ids: Optional[List[int]] = None,
        older_than_days: Optional[int] = None,
        max_access_count: Optional[int] = None,
        memory_type: Optional[str] = None,
        project: Optional[str] = None,
        centrality_protection: bool = True,
        min_centrality_threshold: int = 5,
        dry_run: bool = True,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Archive memories (soft delete). Replaces memory_forget and memory_batch_archive.

        Supports both explicit ID lists and filter-based archival.
        Foundational memories are always protected from archival.

        SAFETY: dry_run=True by default — returns preview of what would be archived.

        Args:
            memory_ids: Explicit list of memory IDs to archive
            older_than_days: Archive memories older than N days
            max_access_count: Archive memories with access_count <= N
            memory_type: Filter by memory type
            project: Filter by project
            centrality_protection: Protect high-centrality memories (default True)
            min_centrality_threshold: In-degree count for protection (default 5)
            dry_run: Preview only (default True)
            reason: Archival reason for audit trail

        Returns:
            Preview (dry_run=True): {would_archive, memories, protected, note}
            Execute (dry_run=False): {archived, memories, protected}
        """
        return archive_memory(
            memory_ids=memory_ids,
            older_than_days=older_than_days,
            max_access_count=max_access_count,
            memory_type=memory_type,
            project=project,
            centrality_protection=centrality_protection,
            min_centrality_threshold=min_centrality_threshold,
            dry_run=dry_run,
            reason=reason,
        )
