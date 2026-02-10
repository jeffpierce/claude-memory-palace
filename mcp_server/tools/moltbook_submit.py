"""
Moltbook submit tool for Claude Memory Palace MCP server.

Wraps the gateway.submit() function as an MCP tool.
This is how Sandy agents submit content through the gateway.
"""
from typing import Any, Optional

from mcp_server.toon_wrapper import toon_response


def register_moltbook_submit(mcp):
    """Register the moltbook_submit tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def moltbook_submit(
        action_type: str,
        content: str,
        session_id: str,
        qc_token: str,
        submolt: Optional[str] = None,
        title: Optional[str] = None,
        post_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        dry_run: bool = False,
        toon: Optional[bool] = None,
    ) -> dict[str, Any]:
        """
        Submit content to Moltbook through the submission gateway.

        All submissions go through 6 mechanical interlocks before reaching
        the Moltbook API. If ANY interlock fails, the submission is blocked
        and a structured error is returned. DO NOT retry on failure —
        write a handoff instead.

        Interlocks (checked in order):
        1. Session guard — prevents retry loops within a session
        2. Content hash dedup — prevents exact duplicate content
        3. Word count gate — posts max 600 words, comments max 300
        4. Similarity check — catches near-duplicate rewording
        5. Rate limit — 30min between posts, 20sec between comments
        6. QC gate — requires valid, unexpired, unconsumed QC token

        Args:
            action_type: "post" or "comment"
            content: The content to submit
            session_id: Unique session ID (from your session). Prevents retry loops.
            qc_token: QC approval token UUID (from moltbook_qc tool)
            submolt: Target submolt name (required for posts)
            title: Post title (required for posts)
            post_id: Target post ID (required for comments)
            parent_id: Parent comment ID (for threaded replies to comments)
            dry_run: If True, checks all gates but doesn't call the API

        Returns:
            Success: {"success": True, "submission_id": N, "moltbook_id": "..."}
            Blocked: {"success": False, "blocked_by": "gate_name", "reason": "..."}
            Error: {"success": False, "error": "..."}
            Challenge: The API may return an anti-human CAPTCHA instead of
                posting immediately. If the response contains "challenge",
                it will include garbled text (l33tspeak, symbols, misspellings)
                encoding a math problem, plus instructions. Decode the text,
                solve the math, and follow the payload's instructions to
                complete the submission. Do this immediately — there's a
                timeout.
        """
        from moltbook_tools.gateway import submit

        return submit(
            action_type=action_type,
            content=content,
            session_id=session_id,
            qc_token=qc_token,
            submolt=submolt,
            title=title,
            post_id=post_id,
            parent_id=parent_id,
            dry_run=dry_run,
        )
