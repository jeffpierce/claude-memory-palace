"""Code remember tool for Claude Memory Palace MCP server."""
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
        """
        Index a source file for natural language search. Creates linked prose
        description (embedded) and raw code (stored, not embedded).

        Args:
            code_path: Absolute path to the source file
            project: Project this code belongs to
            instance_id: Which instance is indexing
            force: Re-index even if already indexed (default False)

        Returns:
            {prose_id, code_id, subject, language, patterns} or {error}
        """
        return code_remember(
            code_path=code_path,
            project=project,
            instance_id=instance_id,
            force=force,
        )
