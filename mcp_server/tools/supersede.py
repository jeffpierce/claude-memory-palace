"""
Supersede tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import supersede_memory


def register_supersede(mcp):
    """Register the memory_supersede tool with the MCP server."""

    @mcp.tool()
    async def memory_supersede(
        new_memory_id: int,
        old_memory_id: int,
        archive_old: bool = True,
        created_by: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Mark a new memory as superseding an old one.

        This is a convenience wrapper for the common pattern of:
        1. Creating a 'supersedes' edge from new -> old
        2. Archiving the old memory (optional but default)

        Use this when information has been updated/corrected and you want
        to maintain the history while making it clear which version is current.

        The old memory is archived by default but remains in the database.
        You can find superseded memories via memory_graph or memory_related.

        Args:
            new_memory_id: ID of the newer/updated memory (the current truth)
            old_memory_id: ID of the older/outdated memory (being replaced)
            archive_old: Whether to archive the old memory (default True)
            created_by: Instance ID that created this supersession

        Returns:
            Dict with edge ID and archive status
        """
        return supersede_memory(
            new_memory_id=new_memory_id,
            old_memory_id=old_memory_id,
            archive_old=archive_old,
            created_by=created_by
        )
