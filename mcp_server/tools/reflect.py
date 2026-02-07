"""
Reflect tool for Claude Memory Palace MCP server.
"""
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
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Process a conversation transcript and extract memories worth keeping.

        Uses LLM (via Ollama) for intelligent extraction. Extracts facts, insights,
        decisions, blockers, gotchas, and other valuable information from transcripts.

        Args:
            instance_id: Which instance is doing the reflection (e.g., "desktop", "code", "web")
            transcript_path: Path to the transcript file to analyze (JSONL or TOON format)
            session_id: Optional session ID to link memories back to source
            dry_run: If true, only report what would be stored without writing to database

        Returns:
            Dict with extracted count, embedded count, and types breakdown
        """
        return reflect(
            instance_id=instance_id,
            transcript_path=transcript_path,
            session_id=session_id,
            dry_run=dry_run
        )
