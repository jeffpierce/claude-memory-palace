"""
Related tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import get_related_memories
from mcp_server.toon_wrapper import toon_response


def register_related(mcp):
    """Register the memory_related tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_related(
        memory_id: int,
        relation_type: Optional[str] = None,
        direction: str = "both",
        include_memory_content: bool = True,
        detail_level: str = "summary",
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Get memories related to a given memory via edges.

        This is for exploring immediate connections (1 hop).
        For deeper traversal, use memory_graph instead.

        Args:
            memory_id: ID of the memory to find relations for
            relation_type: Filter by specific relation type (optional)
            direction: "outgoing" (this->others), "incoming" (others->this), 
                      or "both" (default "both")
            include_memory_content: Include memory details in response (default True)
            detail_level: "summary" or "verbose" for memory content

        Returns:
            Dict with outgoing and/or incoming edges and related memories
        """
        return get_related_memories(
            memory_id=memory_id,
            relation_type=relation_type,
            direction=direction,
            include_memory_content=include_memory_content,
            detail_level=detail_level
        )
