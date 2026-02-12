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
        # Create edge between memories. Replaces memory_supersede.
        """
        Create relationship edge.

        Standard types: supersedes, relates_to, derived_from, contradicts, exemplifies, refines (custom OK).
        archive_old: True + relation_type="supersedes" archives target (default False).
        strength: 0.0-1.0 (default 1.0).
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
