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
        """
        Store a new memory in the memory palace.

        AUTO-LINKING (two tiers):
        - **Auto-linked** (>= 0.75 similarity): Edges created automatically with LLM-classified types.
          Returned in `links_created`.
        - **Suggested** (0.675–0.75 similarity): Surfaced for human review, no edges created.
          Returned in `suggested_links` — present these to the user if relevant.

        For explicit supersession or other typed relationships, use supersedes_id or memory_link.

        Args:
            instance_id: Which instance is storing this (e.g., "desktop", "code", "web")
            memory_type: Type of memory (open-ended - use existing types or create new ones like: fact, preference, event, context, insight, relationship, architecture, gotcha, blocker, solution, workaround, design_decision)
            content: The actual memory content
            subject: What/who this memory is about (optional but recommended)
            keywords: List of keywords for searchability
            tags: Freeform organizational tags (separate from keywords)
            foundational: True if this is a foundational/core memory that should never be archived (default False)
            project: Project this memory belongs to. Can be a string (single project) or list of strings (multi-project). Default "life".
            source_type: How this memory was created (conversation, explicit, inferred, observation)
            source_context: Snippet of original context
            source_session_id: Link back to conversation session
            supersedes_id: If set, create a 'supersedes' edge to this memory and archive it. Only use when the user explicitly confirms supersession.
            auto_link: Override config to enable/disable similarity-based auto-linking (None = use config)

        Returns:
            Dict with id, subject, embedded status, links_created (auto edges), and
            suggested_links (sub-threshold candidates for human review)
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
