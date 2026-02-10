"""
Moltbook QC tool for Claude Memory Palace MCP server.

Wraps qc.create_approval() as an MCP tool.
QC agents call this after reviewing content to issue approval tokens.
"""
from typing import Any, Optional

from mcp_server.toon_wrapper import toon_response


def register_moltbook_qc(mcp):
    """Register the moltbook_qc tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def moltbook_qc(
        content: str,
        notes: str,
        verdict: str = "pass",
        toon: Optional[bool] = None,
    ) -> dict[str, Any]:
        """
        Create a QC approval token for reviewed Moltbook content.

        Called by the QC agent after reviewing content against the posting rubric.
        The token is required by moltbook_submit — no token, no post.

        Tokens are:
        - Single-use (consumed when submission succeeds)
        - Time-limited (30 minutes by default)
        - Content-bound (hash must match what's submitted)

        If you modify content after QC approval, you need a new token.

        Args:
            content: The reviewed content (will be hashed for matching)
            notes: QC agent's reasoning (e.g., "Passed 6-point rubric, under word limit")
            verdict: "pass" or "fail" (default "pass")

        Returns:
            {"token": "uuid", "content_hash": "sha256", "verdict": "pass", "expires_at": "iso"}
        """
        from moltbook_tools.qc import create_approval

        return create_approval(
            content=content,
            notes=notes,
            verdict=verdict,
        )
