"""
Code recall tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services.code_service import code_recall
from mcp_server.toon_wrapper import toon_response


def register_code_recall(mcp):
    """Register the code_recall tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def code_recall_tool(
        query: str,
        project: Optional[str] = None,
        synthesize: bool = True,
        limit: int = 5,
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Search indexed code using natural language.

        Finds relevant code by:
        1. Semantic search on prose descriptions (finds conceptually similar code)
        2. Graph traversal to retrieve actual source code
        3. Optionally LLM synthesis to answer your question directly

        WHEN TO USE:
        - "How does embedding generation work?"
        - "Show me the retry logic"
        - "Where is the database connection handled?"
        - "What design patterns are used in the services?"
        - Any natural language question about indexed code

        Args:
            query: Natural language search query
            project: Filter by project (optional, e.g., "memory-palace")
            synthesize: If True (default), LLM answers using code context.
                       If False, returns raw {prose, code} pairs for you to analyze.
            limit: Maximum number of code files to return (default 5)

        Returns:
            If synthesize=True:
                {answer: str, sources: [{file, subject, relevance}], count: int}
            If synthesize=False:
                {matches: [{prose, code, similarity}], count: int}
        """
        return code_recall(
            query=query,
            project=project,
            synthesize=synthesize,
            limit=limit
        )
