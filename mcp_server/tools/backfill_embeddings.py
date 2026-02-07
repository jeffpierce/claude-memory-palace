"""
Backfill embeddings tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import backfill_embeddings
from mcp_server.toon_wrapper import toon_response


def register_backfill_embeddings(mcp):
    """Register the backfill_embeddings tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_backfill_embeddings(toon: Optional[bool] = None) -> dict[str, Any]:
        """
        Generate embeddings for all memories that don't have them.

        Useful for:
        - Backfilling existing memories after enabling semantic search
        - Retrying after Ollama was unavailable
        - Recovering from partial failures

        Returns:
            Dictionary with counts: total, generated, failed, and any failed IDs
        """
        return backfill_embeddings()
