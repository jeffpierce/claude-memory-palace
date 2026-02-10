"""Audit tool for Claude Memory Palace MCP server."""
from typing import Any, Dict, List, Optional

from memory_palace.services.maintenance_service import audit_palace
from mcp_server.toon_wrapper import toon_response


def register_audit(mcp):
    """Register the audit tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_audit(
        checks: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        limit_per_category: int = 20,
    ) -> dict[str, Any]:
        """
        Audit palace health. Checks for duplicates, stale memories, orphan edges,
        missing embeddings, and contradictions. Foundational memories are never stale.

        Args:
            checks: Which checks to run (default all). Valid: duplicates, stale,
                    orphan_edges, missing_embeddings, contradictions
            thresholds: Override thresholds dict, e.g. {"duplicate_similarity": 0.92,
                       "stale_days": 90, "stale_max_access": 2,
                       "stale_min_centrality": 3}
            project: Filter by project
            limit_per_category: Max results per issue type (default 20)

        Returns:
            Dict with findings per category and summary counts
        """
        return audit_palace(
            checks=checks,
            thresholds=thresholds,
            project=project,
            limit_per_category=limit_per_category,
        )
