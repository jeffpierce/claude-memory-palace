"""
Recall tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import recall


def register_recall(mcp):
    """Register the recall tool with the MCP server."""

    @mcp.tool()
    async def memory_recall(
        query: str,
        instance_id: Optional[str] = None,
        project: Optional[str] = None,
        memory_type: Optional[str] = None,
        subject: Optional[str] = None,
        min_importance: Optional[int] = None,
        include_archived: bool = False,
        limit: int = 20,
        detail_level: str = "summary",
        synthesize: bool = True,
        weight_similarity: Optional[float] = None,
        weight_access: Optional[float] = None,
        weight_centrality: Optional[float] = None
    ) -> dict[str, Any]:
        """
        Search memories using semantic search (with keyword fallback).

        PROACTIVE USE - USE THIS TOOL AS A REFLEX, NOT AN OPTION:
        - When you lack context on a topic, SEARCH MEMORY BEFORE asking the user
        - When you're uncertain about prior decisions, history, or preferences, SEARCH FIRST
        - When you think "have we discussed this?" or "what was the decision?" - that's a memory search
        - When you would otherwise say "I don't know our history on this" - SEARCH INSTEAD
        - Treat this like web search: you wouldn't ask the user to Google something for you
        - The user built this memory system so you DON'T have to ask them for context you already have
        - NOT using this when context might exist is a failure mode - you're ignoring your own memory

        Uses centrality-weighted ranking combining:
        - Semantic similarity (embedding cosine similarity)
        - Access frequency (log-transformed to prevent Matthew effect)
        - Graph centrality (in-degree from knowledge graph edges)

        Args:
            query: Search query - uses semantic similarity when Ollama is available, falls back to keyword matching
            instance_id: Filter by instance (optional)
            project: Filter by project (optional, e.g., "memory-palace", "wordleap", "life")
            memory_type: Filter by type (e.g., fact, preference, event, context, insight, relationship, architecture, gotcha, blocker, solution, workaround, design_decision, or any custom type)
            subject: Filter by subject
            min_importance: Only return memories with importance >= this (1-10)
            include_archived: Include archived memories (default false)
            limit: Maximum memories to return (default 20)
            detail_level: "summary" for condensed, "verbose" for full content (only applies when synthesize=True)
            synthesize: If True (default), use local LLM to synthesize results. If False, return raw memory objects with full content for cloud AI to process.
            weight_similarity: Override weight for semantic similarity (default 0.7, range 0-1)
            weight_access: Override weight for access count (default 0.15, range 0-1)
            weight_centrality: Override weight for graph centrality (default 0.15, range 0-1)

        Returns:
            Dictionary with format depending on synthesize parameter:
            - synthesize=True: {"summary": str, "count": int, "search_method": str, "memory_ids": list}
            - synthesize=False: {"memories": list[dict], "count": int, "search_method": str}
              Raw mode always returns verbose content with similarity_score when available.
        """
        # Set weights as environment variables if provided (for this call only)
        import os
        old_env = {}
        try:
            if weight_similarity is not None:
                old_env['MEMORY_PALACE_WEIGHT_SIMILARITY'] = os.environ.get('MEMORY_PALACE_WEIGHT_SIMILARITY')
                os.environ['MEMORY_PALACE_WEIGHT_SIMILARITY'] = str(weight_similarity)
            if weight_access is not None:
                old_env['MEMORY_PALACE_WEIGHT_ACCESS'] = os.environ.get('MEMORY_PALACE_WEIGHT_ACCESS')
                os.environ['MEMORY_PALACE_WEIGHT_ACCESS'] = str(weight_access)
            if weight_centrality is not None:
                old_env['MEMORY_PALACE_WEIGHT_CENTRALITY'] = os.environ.get('MEMORY_PALACE_WEIGHT_CENTRALITY')
                os.environ['MEMORY_PALACE_WEIGHT_CENTRALITY'] = str(weight_centrality)

            return recall(
                query=query,
                instance_id=instance_id,
                project=project,
                memory_type=memory_type,
                subject=subject,
                min_importance=min_importance,
                include_archived=include_archived,
                limit=limit,
                detail_level=detail_level,
                synthesize=synthesize
            )
        finally:
            # Restore old environment
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
