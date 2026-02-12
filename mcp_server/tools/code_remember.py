"""Code remember tool for Memory Palace MCP server."""
from typing import Any

from memory_palace.services.code_service import code_remember
from mcp_server.toon_wrapper import toon_response


def register_code_remember(mcp):
    """Register the code_remember tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def code_remember_tool(
        code_path: str,
        project: str,
        instance_id: str,
        force: bool = False,
    ) -> dict[str, Any]:
        # Index source file for NL search. Creates prose (embedded) + raw code (stored).
        """
        Index source file. Creates prose (embedded) + raw code.

        force: Re-index if already indexed (default False).
        """
        return code_remember(
            code_path=code_path,
            project=project,
            instance_id=instance_id,
            force=force,
        )
