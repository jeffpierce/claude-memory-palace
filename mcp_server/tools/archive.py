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
        database: Optional[str] = None,
    ) -> dict[str, Any]:
        # Archive (soft delete). ID list or filters. Foundational always protected.
        """
        Archive memories. Foundational always protected.

        dry_run: True by default (preview).
        centrality_protection: Protect high-centrality (default True, threshold=5 in-degree).
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
            database=database,
        )
