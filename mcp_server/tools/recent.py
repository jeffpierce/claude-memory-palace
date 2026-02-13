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
        database: Optional[str] = None,
    ) -> dict[str, Any]:
        # Last X memories, newest first. Default: title-card (id, subject, type, project, date).
        """
        Last X memories, newest first.

        verbose: False=title-card (default), True=full details.
        memory_type: Supports wildcards like "code_*".
        limit: Max 200.
        """
        return get_recent_memories(
            limit=limit,
            verbose=verbose,
            project=project,
            memory_type=memory_type,
            instance_id=instance_id,
            include_archived=include_archived,
            database=database,
        )
