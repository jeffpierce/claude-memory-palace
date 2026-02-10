"""Get memory by ID tool for Memory Palace MCP server."""
from typing import Any, List, Optional, Union

from memory_palace.services import get_memory_by_id, get_memories_by_ids
from mcp_server.toon_wrapper import toon_response


def register_get_memory(mcp):
    """Register the get_memory tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_get(
        memory_ids: Union[int, List[int]],
        detail_level: str = "verbose",
        synthesize: bool = False,
        include_graph: bool = True,
        graph_depth: int = 1,
        traverse: bool = False,
        max_depth: int = 3,
        direction: Optional[str] = None,
        relation_types: Optional[List[str]] = None,
        min_strength: Optional[float] = None
    ) -> dict[str, Any]:
        """
        Retrieve one or more memories by their IDs with optional graph traversal.

        Use this when you have specific memory IDs to retrieve, such as from
        handoff messages that reference specific memories (e.g., "Memory 151").

        Args:
            memory_ids: Single memory ID (int) or list of memory IDs to retrieve
            detail_level: "summary" for condensed, "verbose" for full content (default: verbose)
            synthesize: If True, use LLM to synthesize multiple memories into natural language summary.
                       Skipped for single memory (pointless). Default: False (returns raw memory objects).
            include_graph: Include graph context for all retrieved memories (default True)
            graph_depth: How many hops to follow in graph context (1-3, default 1).
                        Use 2 for bootstrap/startup to see the broader neighborhood.
            traverse: If True, do BFS traversal instead of context mode (replaces memory_graph tool)
            max_depth: Max depth for BFS traverse (1-5, only used if traverse=True)
            direction: "outgoing", "incoming", or None for both (applies to graph context and traversal)
            relation_types: Filter edges by type (e.g., ["related_to", "supersedes"])
            min_strength: Filter edges by minimum strength (0.0-1.0)

        Returns:
            For single ID: {"memory": dict, "graph_context": dict (optional)} or {"error": str} if not found
            For multiple IDs (synthesize=False): {"memories": list[dict], "count": int, "not_found": list[int], "graph_context": dict (optional)}
            For multiple IDs (synthesize=True): {"summary": str, "count": int, "memory_ids": list[int], "not_found": list[int], "graph_context": dict (optional)}
            graph_context format: {"nodes": {id: subject}, "edges": [{source, target, type, strength}]}
        """
        # Normalize to list
        if isinstance(memory_ids, int):
            ids = [memory_ids]
            single_mode = True
        else:
            ids = memory_ids
            single_mode = False

        # Single memory: use simple fetch
        if single_mode:
            result = get_memory_by_id(
                ids[0],
                detail_level=detail_level,
                include_graph=include_graph,
                graph_depth=graph_depth,
                traverse=traverse,
                max_depth=max_depth,
                direction=direction,
                relation_types=relation_types,
                min_strength=min_strength
            )
            if result:
                return result  # Already has {"memory": ..., "graph_context": ...} format
            else:
                return {"error": f"Memory {ids[0]} not found"}

        # Multiple memories: use batch fetch with optional synthesis
        # Note: batch fetch only supports basic graph context, not full traversal
        # Advanced graph params (traverse, direction, relation_types, min_strength) are ignored for batch mode
        return get_memories_by_ids(
            ids,
            detail_level=detail_level,
            synthesize=synthesize,
            include_graph=include_graph,
            graph_depth=graph_depth
        )
