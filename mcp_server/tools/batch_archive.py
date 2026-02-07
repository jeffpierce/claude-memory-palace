"""
Batch archive tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional, List

from memory_palace.services.maintenance_service import batch_archive_memories
from mcp_server.toon_wrapper import toon_response


def register_batch_archive(mcp):
    """Register the batch archive tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_batch_archive(
        older_than_days: Optional[int] = None,
        max_access_count: Optional[int] = None,
        memory_type: Optional[str] = None,
        project: Optional[str] = None,
        memory_ids: Optional[List[int]] = None,
        centrality_protection: bool = True,
        min_centrality_threshold: int = 5,
        dry_run: bool = True,
        reason: Optional[str] = None,
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Archive multiple memories matching criteria.

        SAFETY: dry_run=True by default. Returns preview of what would be archived.
        Set dry_run=False to execute.

        CENTRALITY PROTECTION (THE KEY INSIGHT):
        High-centrality memories (many incoming edges) are automatically protected
        from archival. The graph structure itself identifies foundational memories
        that other memories depend on. No manual importance tagging needed.

        Example: Memory #167 (Sandy's identity) has 128+ connections. The topology
        says "this is load-bearing" - archiving it would orphan 128 edges and break
        semantic connections across the palace.

        Use case patterns:
        - Daily content cleanup: older_than_days=90, memory_type="daily_content"
        - Low-value cleanup: max_access_count=1, older_than_days=30
        - Project cleanup: project="old-project", older_than_days=180
        - Explicit list: memory_ids=[1, 2, 3]

        Args:
            older_than_days: Age filter (memories older than N days)
            max_access_count: Low-access filter (access count <= N)
            memory_type: Type filter (e.g., "daily_content", "episode")
            project: Project filter
            memory_ids: Explicit ID list (overrides other filters)
            centrality_protection: Protect high-centrality memories (default True)
            min_centrality_threshold: In-degree count that grants protection (default 5)
            dry_run: Preview only (default True for safety)
            reason: Archival reason for audit trail

        Returns:
            If dry_run=True:
                {
                    "would_archive": N,
                    "memories": [{"id": X, "subject": "...", "age_days": N}, ...],
                    "protected": [{"id": X, "in_degree": N, "reason": "..."}, ...],
                    "note": "DRY RUN - ..."
                }
            If dry_run=False:
                {
                    "archived": N,
                    "memories": [...],
                    "protected": [...]
                }
        """
        return batch_archive_memories(
            older_than_days=older_than_days,
            max_access_count=max_access_count,
            memory_type=memory_type,
            project=project,
            memory_ids=memory_ids,
            centrality_protection=centrality_protection,
            min_centrality_threshold=min_centrality_threshold,
            dry_run=dry_run,
            reason=reason
        )
