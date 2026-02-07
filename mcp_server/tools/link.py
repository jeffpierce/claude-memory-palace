"""
Link tool for Claude Memory Palace MCP server.
"""
from typing import Any, Dict, List, Optional

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
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Create a relationship edge between two memories.

        Use this to build the knowledge graph by connecting related memories.
        The edge goes from source -> target (directional by default).

        STANDARD RELATIONSHIP TYPES:
        - relates_to: General association (often set bidirectional=True)
        - derived_from: This memory came from processing that one
        - contradicts: Memories make conflicting claims (usually bidirectional). Auto-classifier uses this for any detected conflict.
        - exemplifies: This is an example of that concept
        - refines: Adds detail/nuance to another memory
        - supersedes: Newer memory replaces older (use memory_supersede for convenience)
          ⚠️ HUMAN-ONLY: Never set supersedes automatically. The auto-classifier will
          flag conflicts as "contradicts" — only promote to "supersedes" when the user
          explicitly confirms that the newer memory replaces the older one.

        Custom types are allowed - use descriptive names like "caused_by", "leads_to", etc.

        Args:
            source_id: ID of the source memory
            target_id: ID of the target memory (edge points TO this)
            relation_type: Type of relationship (see above, or custom)
            strength: Edge weight 0.0-1.0 for weighted traversal (default 1.0)
            bidirectional: If True, edge works in both directions (default False)
            metadata: Optional extra data to store with the edge (JSON object)
            created_by: Instance ID creating this edge (e.g., "clawdbot", "desktop")

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
            created_by=created_by
        )
