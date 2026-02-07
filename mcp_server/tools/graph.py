"""
Graph traversal tool for Claude Memory Palace MCP server.
"""
from typing import Any, List, Optional

from memory_palace.services import traverse_graph, get_relationship_types
from mcp_server.toon_wrapper import toon_response


def register_graph(mcp):
    """Register the memory_graph tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_graph(
        start_id: int,
        max_depth: int = 2,
        relation_types: Optional[List[str]] = None,
        direction: str = "outgoing",
        min_strength: float = 0.0,
        include_archived: bool = False,
        detail_level: str = "summary",
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Traverse the memory knowledge graph from a starting point.

        Performs breadth-first traversal following edges up to max_depth.
        Useful for exploring how memories connect to each other.

        Example uses:
        - Find all memories derived from a key insight
        - Trace the evolution of an idea through supersession chains
        - Discover memory clusters around a topic

        Args:
            start_id: ID of the memory to start from
            max_depth: Maximum traversal depth 1-5 (default 2)
            relation_types: List of relation types to follow (optional - all if None)
                           e.g., ["supersedes", "relates_to"]
            direction: "outgoing" (follow edges forward), "incoming" (follow edges backward),
                      or "both" (default "outgoing")
            min_strength: Minimum edge strength to follow, 0.0-1.0 (default 0.0 = all)
            include_archived: Include archived memories in results (default False)
            detail_level: "summary" or "verbose" for memory content

        Returns:
            Dict with nodes (memories) and edges discovered during traversal.
            Each node includes _depth indicating hops from start.
        """
        return traverse_graph(
            start_id=start_id,
            max_depth=max_depth,
            relation_types=relation_types,
            direction=direction,
            min_strength=min_strength,
            include_archived=include_archived,
            detail_level=detail_level
        )

    @mcp.tool()
    @toon_response
    async def memory_relationship_types(toon: Optional[bool] = None) -> dict[str, Any]:
        """
        Get information about available relationship types.

        Returns the standard relationship types with descriptions,
        plus any custom types currently in use in the database.

        Use this to understand what relationship types are available
        before creating edges with memory_link.

        Returns:
            Dict with standard_types (with descriptions) and custom_types_in_use
        """
        return get_relationship_types()
