"""
Link tool for Memory Palace MCP server.
"""
from typing import Any, Dict, Optional

from memory_palace.services import link_memories
from mcp_server.toon_wrapper import toon_response


def register_link(mcp):
    """Register the memory_link tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_link(
        source_id: int,
        target_id: int,
        relation_type: str,
        strength: float = 1.0,
        bidirectional: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
        archive_old: bool = False
    ) -> dict[str, Any]:
        """
        Create a relationship edge between two memories.

        Standard types: supersedes, relates_to, derived_from, contradicts, exemplifies, refines (custom allowed).

        When archive_old=True with relation_type="supersedes", archives the target memory.
        This replaces the old memory_supersede tool.

        Args:
            source_id: ID of the source memory
            target_id: ID of the target memory (edge points TO this)
            relation_type: Type of relationship (standard or custom)
            strength: Edge weight 0.0-1.0 for weighted traversal (default 1.0)
            bidirectional: If True, edge works in both directions (default False)
            metadata: Optional extra data to store with the edge (JSON object)
            created_by: Instance ID creating this edge (e.g., "clawdbot", "desktop")
            archive_old: If True AND relation_type="supersedes", archives the target (default False)

        Returns:
            Dict with edge ID and confirmation message
        """
        return link_memories(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            bidirectional=bidirectional,
            metadata=metadata,
            created_by=created_by,
            archive_old=archive_old
        )
