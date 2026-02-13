"""Audit tool for Memory Palace MCP server."""
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
        database: Optional[str] = None,
    ) -> dict[str, Any]:
        # Check palace health. Foundational never stale.
        """
        Audit palace health. Foundational never stale.

        checks: duplicates, stale, orphan_edges, missing_embeddings, contradictions (default all).
        thresholds: Override defaults, e.g. {"duplicate_similarity": 0.92, "stale_days": 90}.
        """
        return audit_palace(
            checks=checks,
            thresholds=thresholds,
            project=project,
            limit_per_category=limit_per_category,
            database=database,
        )
