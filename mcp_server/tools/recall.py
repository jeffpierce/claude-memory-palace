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
        # Semantic search (keyword fallback). Centrality-weighted ranking.
        """
        Semantic search (keyword fallback). Centrality-weighted ranking.

        query: Semantic when Ollama available, else keyword.
        memory_type: Supports wildcards like "code_*".
        synthesize: True=LLM synthesis (default), False=raw objects for cloud AI.
        graph_top_n: Fetch graph context for top N results (default 5).

        See memory_get for graph_mode format details.
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
