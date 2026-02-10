"""Recent memories tool for Memory Palace MCP server."""
from typing import Any, List, Optional, Union

from memory_palace.services import get_recent_memories
from mcp_server.toon_wrapper import toon_response


def register_recent(mcp):
    """Register the memory_recent tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_recent(
        limit: int = 20,
        verbose: bool = False,
        project: Optional[Union[str, List[str]]] = None,
        memory_type: Optional[str] = None,
        instance_id: Optional[str] = None,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        """
        Get the last X memories, newest first.

        Default returns title-card format (id, subject, type, project, date).
        Set verbose=True for full memory details.

        Args:
            limit: Number of memories to return (default 20, max 200)
            verbose: False = title only (compact). True = full details.
            project: Filter by project (string or list). Optional.
            memory_type: Filter by type, supports wildcards like "code_*". Optional.
            instance_id: Filter by instance. Optional.
            include_archived: Include archived memories (default False)

        Returns:
            {"memories": list, "count": int, "total_available": int}
        """
        return get_recent_memories(
            limit=limit,
            verbose=verbose,
            project=project,
            memory_type=memory_type,
            instance_id=instance_id,
            include_archived=include_archived,
        )
