"""Remember tool for Memory Palace MCP server."""
from typing import Any, List, Optional, Union

from memory_palace.services import remember
from mcp_server.toon_wrapper import toon_response


def register_remember(mcp):
    """Register the remember tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_remember(
        instance_id: str,
        memory_type: str,
        content: str,
        subject: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        foundational: bool = False,
        project: Union[str, List[str]] = "life",
        source_type: str = "explicit",
        source_context: Optional[str] = None,
        source_session_id: Optional[str] = None,
        supersedes_id: Optional[int] = None,
        auto_link: Optional[bool] = None
    ) -> dict[str, Any]:
        # Store new memory. Auto-links similar memories (>=0.75 creates edge, 0.675-0.75 suggests).
        # For explicit relationships, use supersedes_id or memory_link.
        """
        Store new memory in palace.

        AUTO-LINKING: >=0.75 similarity auto-creates edges (LLM-typed, in links_created). 0.675-0.75 returns suggested_links for review.
        Use supersedes_id or memory_link for explicit relationships.

        memory_type: Open-ended. Common: fact, preference, event, insight, architecture, gotcha, solution, design_decision.
        foundational: Never archived (default False).
        supersedes_id: Create supersedes edge + archive target. Only when user confirms.

        Returns: {id, subject, embedded, links_created, suggested_links}
        """
        return remember(
            instance_id=instance_id,
            memory_type=memory_type,
            content=content,
            subject=subject,
            keywords=keywords,
            tags=tags,
            foundational=foundational,
            project=project,
            source_type=source_type,
            source_context=source_context,
            source_session_id=source_session_id,
            supersedes_id=supersedes_id,
            auto_link=auto_link
        )
