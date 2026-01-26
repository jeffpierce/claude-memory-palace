"""
Get memory by ID tool for Claude Memory Palace MCP server.
"""
from typing import Any, List, Optional, Union

from memory_palace.services import get_memory_by_id, get_memories_by_ids


def register_get_memory(mcp):
    """Register the get_memory tool with the MCP server."""

    @mcp.tool()
    async def memory_get(
        memory_ids: Union[int, List[int]],
        detail_level: str = "verbose",
        synthesize: bool = False
    ) -> dict[str, Any]:
        """
        Retrieve one or more memories by their IDs.

        PROACTIVE USE:
        - When memory IDs are mentioned (in handoffs, session notes, or conversation), FETCH THEM
        - Don't ask the user what's in a memory - retrieve it yourself
        - When a memory recall returns IDs that seem relevant, use this to get full content
        - At session start, if foundational memories are mentioned, load them immediately

        Use this when you have specific memory IDs to retrieve, such as from
        handoff messages that reference specific memories (e.g., "Memory 151").

        Args:
            memory_ids: Single memory ID (int) or list of memory IDs to retrieve
            detail_level: "summary" for condensed, "verbose" for full content (default: verbose)
            synthesize: If True, use LLM to synthesize multiple memories into natural language summary.
                       Skipped for single memory (pointless). Default: False (returns raw memory objects).

        Returns:
            For single ID: {"memory": dict} or {"error": str} if not found
            For multiple IDs (synthesize=False): {"memories": list[dict], "count": int, "not_found": list[int]}
            For multiple IDs (synthesize=True): {"summary": str, "count": int, "memory_ids": list[int], "not_found": list[int]}
        """
        # Normalize to list
        if isinstance(memory_ids, int):
            ids = [memory_ids]
            single_mode = True
        else:
            ids = memory_ids
            single_mode = False

        # Single memory: use simple fetch (synthesis doesn't apply)
        if single_mode:
            result = get_memory_by_id(ids[0], detail_level=detail_level)
            if result:
                return {"memory": result}
            else:
                return {"error": f"Memory {ids[0]} not found"}

        # Multiple memories: use batch fetch with optional synthesis
        return get_memories_by_ids(ids, detail_level=detail_level, synthesize=synthesize)
