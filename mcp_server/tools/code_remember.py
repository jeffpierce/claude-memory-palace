"""
Code remember tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

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
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Index a source file into the memory palace for natural language search.

        Creates two linked memories:
        - A prose description (embedded for semantic search)
        - The actual source code (stored but NOT embedded)

        The prose acts as a semantic index - queries like "how do embeddings work"
        find relevant prose, then graph traversal retrieves the actual code.

        WHEN TO USE:
        - Index important source files you want to query later
        - After significant code changes, re-index with force=True
        - Index config files, scripts, anything you might search for

        Args:
            code_path: Absolute path to the source file to index
            project: Project this code belongs to (e.g., "memory-palace", "wordleap")
            instance_id: Which instance is indexing (e.g., "code", "desktop")
            force: Re-index even if already indexed (default False)

        Returns:
            On success: {prose_id, code_id, subject, language, patterns}
            If already indexed: {already_indexed: True, prose_id, code_id, subject}
            On error: {error: str}
        """
        return code_remember(
            code_path=code_path,
            project=project,
            instance_id=instance_id,
            force=force
        )
