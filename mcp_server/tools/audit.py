"""
Audit tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services.maintenance_service import audit_palace


def register_audit(mcp):
    """Register the audit tool with the MCP server."""

    @mcp.tool()
    async def memory_audit(
        check_duplicates: bool = True,
        check_stale: bool = True,
        check_orphan_edges: bool = True,
        check_embeddings: bool = True,
        check_contradictions: bool = True,
        stale_days: int = 90,
        stale_access_threshold: int = 2,
        stale_centrality_threshold: int = 3,
        duplicate_threshold: float = 0.92,
        project: Optional[str] = None,
        limit_per_category: int = 20
    ) -> dict[str, Any]:
        """
        Audit palace health and return actionable findings.

        Checks for:
        - Duplicates: Near-duplicate memories (high semantic similarity)
        - Stale: Old + low access + low centrality (safe to archive)
        - Orphan edges: Edges pointing to archived memories
        - Missing embeddings: Memories that failed to embed
        - Contradictions: Conflicting memories flagged for review

        CENTRALITY PROTECTION:
        Stale detection uses in-degree (graph centrality) as a protection signal.
        Memories with many incoming edges are considered foundational and won't
        be flagged as stale, even if old and rarely accessed. This prevents
        accidental removal of hub memories that other memories depend on.

        Args:
            check_duplicates: Find near-duplicates (>0.9 similarity)
            check_stale: Old + low access + low centrality
            check_orphan_edges: Edges pointing to archived memories
            check_embeddings: Memories missing embeddings
            check_contradictions: Find 'contradicts' edges for review
            stale_days: Age threshold for "stale" (default 90)
            stale_access_threshold: Access count threshold (default 2)
            stale_centrality_threshold: In-degree below this = not protected (default 3)
            duplicate_threshold: Similarity threshold for duplicates (default 0.92)
            project: Filter by project (optional)
            limit_per_category: Cap results per issue type (default 20)

        Returns:
            {
                "duplicates": [{"memory_id": X, "similar_to": Y, "similarity": 0.95}, ...],
                "stale": [{"memory_id": X, "age_days": 120, "access_count": 1, "in_degree": 0}, ...],
                "orphan_edges": [{"edge_id": X, "source": Y, "target": Z, "reason": "..."}, ...],
                "missing_embeddings": [memory_id, ...],
                "contradictions": [{"memory_id": X, "contradicts": Y}, ...],
                "summary": {
                    "total_issues": N,
                    "duplicates_found": N,
                    "stale_found": N,
                    ...
                }
            }
        """
        return audit_palace(
            check_duplicates=check_duplicates,
            check_stale=check_stale,
            check_orphan_edges=check_orphan_edges,
            check_embeddings=check_embeddings,
            check_contradictions=check_contradictions,
            stale_days=stale_days,
            stale_access_threshold=stale_access_threshold,
            stale_centrality_threshold=stale_centrality_threshold,
            duplicate_threshold=duplicate_threshold,
            project=project,
            limit_per_category=limit_per_category
        )
