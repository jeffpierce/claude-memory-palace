"""
Reembed tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional, List

from memory_palace.services.maintenance_service import reembed_memories


def register_reembed(mcp):
    """Register the reembed tool with the MCP server."""

    @mcp.tool()
    async def memory_reembed(
        older_than_days: Optional[int] = None,
        memory_ids: Optional[List[int]] = None,
        project: Optional[str] = None,
        all_memories: bool = False,
        batch_size: int = 50,
        dry_run: bool = True
    ) -> dict[str, Any]:
        """
        Regenerate embeddings for memories.

        Use when:
        - Embedding model changes (e.g., upgrading to a better model)
        - Embeddings seem to be returning poor results
        - After bulk import without embeddings
        - Memory content was updated and embedding is stale

        SAFETY: dry_run=True by default. Returns preview with time estimate.

        Args:
            older_than_days: Re-embed embeddings older than N days
            memory_ids: Explicit list of memory IDs to re-embed
            project: Filter by project
            all_memories: Nuclear option - re-embed everything (use with caution)
            batch_size: Batch size for processing (default 50)
            dry_run: Preview only (default True for safety)

        Returns:
            If dry_run=True:
                {
                    "would_reembed": N,
                    "estimated_time_seconds": N,
                    "memories": [{"id": X, "subject": "...", "type": "..."}, ...],
                    "note": "DRY RUN - ..."
                }
            If dry_run=False:
                {
                    "reembedded": N,
                    "failed": N,
                    "total": N,
                    "failed_ids": [...]  # if any failures
                }
        """
        return reembed_memories(
            older_than_days=older_than_days,
            memory_ids=memory_ids,
            project=project,
            all_memories=all_memories,
            batch_size=batch_size,
            dry_run=dry_run
        )
