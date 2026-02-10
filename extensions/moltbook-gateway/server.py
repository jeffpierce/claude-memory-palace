"""
Moltbook Gateway MCP Extension Server

Standalone MCP server that exposes moltbook submission tools.
This is a thin wrapper around the moltbook_tools package.
"""
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("moltbook-gateway")


@mcp.tool()
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
    3. Word count gate — posts max 1000 words, comments max 300
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


@mcp.tool()
async def moltbook_qc(
    content: str,
    notes: str,
    verdict: str = "pass",
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


def main():
    """Entry point for running the MCP server via stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
