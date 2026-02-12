"""Recall tool for Memory Palace MCP server."""
from typing import Any, List, Optional, Union

from memory_palace.services import recall
from mcp_server.toon_wrapper import toon_response


def register_recall(mcp):
    """Register the recall tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_recall(
        query: str,
        instance_id: Optional[str] = None,
        project: Optional[Union[str, List[str]]] = None,
        memory_type: Optional[str] = None,
        subject: Optional[str] = None,
        min_foundational: Optional[bool] = None,
        include_archived: bool = False,
        limit: int = 20,
        detail_level: str = "summary",
        synthesize: bool = True,
        include_graph: bool = True,
        graph_top_n: int = 5,
        graph_depth: int = 1,
        graph_mode: str = "summary"
    ) -> dict[str, Any]:
        """
        Search memories using semantic search (with keyword fallback).

        Uses centrality-weighted ranking combining semantic similarity, access frequency, and graph centrality.

        Args:
            query: Search query - uses semantic similarity when Ollama is available, falls back to keyword matching
            instance_id: Filter by instance (optional)
            project: Filter by project (optional, e.g., "memory-palace", "wordleap", "life"). Can be a string or list of strings.
            memory_type: Filter by type (e.g., fact, preference, event, context, insight, relationship, architecture, gotcha, blocker, solution, workaround, design_decision, or any custom type). Supports wildcards like "code_*" for pattern matching.
            subject: Filter by subject
            min_foundational: Only return foundational memories if True (optional)
            include_archived: Include archived memories (default false)
            limit: Maximum memories to return (default 20)
            detail_level: "summary" for condensed, "verbose" for full content (only applies when synthesize=True)
            synthesize: If True (default), use local LLM to synthesize results. If False, return raw memory objects with full content for cloud AI to process.
            include_graph: Include graph context for top N results (default True)
            graph_top_n: Number of top results to fetch graph context for (default 5)
            graph_depth: How many hops to follow in graph context (1-3, default 1)
            graph_mode: "summary" for per-node stats, "full" for raw edge list (default "summary")

        Returns:
            Dictionary with format depending on synthesize parameter:
            - synthesize=True: {"summary": str, "count": int, "search_method": str, "memory_ids": list, "graph_context": dict (optional)}
            - synthesize=False: {"memories": list[dict], "count": int, "search_method": str, "graph_context": dict (optional)}
              Raw mode always returns verbose content with similarity_score when available.
              If graph_mode == "full": graph_context format: {"nodes": {id: subject}, "edges": [{source, target, type, strength}]}
              If graph_mode == "summary": graph_context format: {"nodes": {id: {subject, connections, avg_strength, edge_types}}, "total_edges": int, "seed_ids": list}
        """
        return recall(
            query=query,
            instance_id=instance_id,
            project=project,
            memory_type=memory_type,
            subject=subject,
            min_foundational=min_foundational,
            include_archived=include_archived,
            limit=limit,
            detail_level=detail_level,
            synthesize=synthesize,
            include_graph=include_graph,
            graph_top_n=graph_top_n,
            graph_depth=graph_depth,
            graph_mode=graph_mode
        )
