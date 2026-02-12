"""Reflect tool for Memory Palace MCP server."""
from typing import Any, Optional

from memory_palace.services import reflect
from mcp_server.toon_wrapper import toon_response


def register_reflect(mcp):
    """Register the reflect tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_reflect(
        instance_id: str,
        transcript_path: str,
        session_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        # Extract memories from transcript using LLM.
        """
        Extract memories from transcript (JSONL or TOON format).

        dry_run: Report without writing (default False).
        """
        return reflect(
            instance_id=instance_id,
            transcript_path=transcript_path,
            session_id=session_id,
            dry_run=dry_run,
        )
