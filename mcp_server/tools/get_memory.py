"""
Get memory by ID tool for Claude Memory Palace MCP server.
"""
from typing import Any, List, Optional, Union

from memory_palace.services import get_memory_by_id


def register_get_memory(mcp):
    """Register the get_memory tool with the MCP server."""

    @mcp.tool()
    async def memory_get(
        memory_ids: Union[int, List[int]],
        detail_level: str = "verbose"
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

        Returns:
            For single ID: {"memory": dict} or {"error": str} if not found
            For multiple IDs: {"memories": list[dict], "not_found": list[int]}
        """
        # Normalize to list
        if isinstance(memory_ids, int):
            ids = [memory_ids]
            single_mode = True
        else:
            ids = memory_ids
            single_mode = False

        memories = []
        not_found = []

        for mid in ids:
            result = get_memory_by_id(mid, detail_level=detail_level)
            if result:
                memories.append(result)
            else:
                not_found.append(mid)

        # Return format depends on whether single or multiple IDs requested
        if single_mode:
            if memories:
                return {"memory": memories[0]}
            else:
                return {"error": f"Memory {ids[0]} not found"}
        else:
            result = {"memories": memories, "count": len(memories)}
            if not_found:
                result["not_found"] = not_found
            return result
