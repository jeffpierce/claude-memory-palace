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
        min_strength: Optional[float] = None,
        graph_mode: str = "summary"
    ) -> dict[str, Any]:
        # Fetch memories by ID with optional graph context.
        # Use when you have specific memory IDs (e.g., from handoff messages: "Memory 151").
        """
        Fetch memories by ID with optional graph context.

        synthesize: LLM-synthesize multiple memories (skipped for single). Default False.
        graph_depth: 1-3 hops. Use 2 for bootstrap.
        traverse: BFS walk instead of context mode.
        direction: "outgoing", "incoming", None=both.

        graph_mode "summary" (default): Nodes are flattened strings: "subject | N connections | avg S | >type,<type,<>type"
        Direction indicators (>,<,<>) invisible to users — translate to plain language when explaining ("links to", "linked from").
        graph_mode "full": {"nodes": {id: subject}, "edges": [{source, target, type, strength}]}
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
                min_strength=min_strength,
                graph_mode=graph_mode
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
            graph_depth=graph_depth,
            graph_mode=graph_mode
        )
