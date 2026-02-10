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
        """
        Extract memories from a conversation transcript using LLM.

        Args:
            instance_id: Which instance is reflecting
            transcript_path: Path to transcript file (JSONL or TOON format)
            session_id: Optional session ID to link memories to source
            dry_run: Report what would be stored without writing (default False)

        Returns:
            {extracted_count, embedded_count, types_breakdown}
        """
        return reflect(
            instance_id=instance_id,
            transcript_path=transcript_path,
            session_id=session_id,
            dry_run=dry_run,
        )
