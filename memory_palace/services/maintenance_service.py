"""
Maintenance service for Claude Memory Palace.

Provides health checks, bulk archival, consolidation, and re-embedding
operations for palace maintenance.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
import logging

from sqlalchemy import func, or_

from memory_palace.models import Memory, MemoryEdge
from memory_palace.database import get_session
from memory_palace.embeddings import cosine_similarity, get_embedding

logger = logging.getLogger(__name__)


def _compute_in_degree(db, memory_id: int) -> int:
    """
    Compute in-degree (number of incoming edges) for a memory.

    Args:
        db: Database session
        memory_id: Memory ID to compute centrality for

    Returns:
        Count of incoming edges
    """
    return db.query(func.count(MemoryEdge.id)).filter(
        MemoryEdge.target_id == memory_id
    ).scalar() or 0


def _find_duplicates(
    db,
    threshold: float = 0.92,
    project: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Find near-duplicate memories based on embedding similarity.

    Args:
        db: Database session
        threshold: Similarity threshold for duplicates (default 0.92)
        project: Filter by project
        limit: Maximum duplicates to return

    Returns:
        List of duplicate pairs with similarity scores
    """
    duplicates = []

    # Get all active memories with embeddings
    query = db.query(Memory).filter(
        Memory.is_archived == False,
        Memory.embedding.isnot(None)
    )

    if project:
        query = query.filter(Memory.project == project)

    memories = query.all()

    # Compare all pairs (O(n²) but we cap results)
    for i, mem1 in enumerate(memories):
        if len(duplicates) >= limit:
            break

        for mem2 in memories[i+1:]:
            if len(duplicates) >= limit:
                break

            similarity = cosine_similarity(mem1.embedding, mem2.embedding)
            if similarity >= threshold:
                duplicates.append({
                    "memory_id": mem1.id,
                    "similar_to": mem2.id,
                    "similarity": round(similarity, 4),
                    "subject_1": mem1.subject or "(no subject)",
                    "subject_2": mem2.subject or "(no subject)",
                    "type_1": mem1.memory_type,
                    "type_2": mem2.memory_type
                })

    return duplicates


def _find_stale_memories(
    db,
    stale_days: int = 90,
    stale_access_threshold: int = 2,
    stale_centrality_threshold: int = 3,
    project: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Find stale memories (old, low access, low centrality).

    Foundational memories are NEVER considered stale.

    Args:
        db: Database session
        stale_days: Age threshold in days
        stale_access_threshold: Access count below this is "low"
        stale_centrality_threshold: In-degree below this is "low"
        project: Filter by project
        limit: Maximum stale memories to return

    Returns:
        List of stale memories with diagnostic info
    """
    cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=stale_days)

    # Get candidate memories (exclude foundational)
    query = db.query(Memory).filter(
        Memory.is_archived == False,
        Memory.foundational == False,  # Foundational memories never stale
        Memory.created_at < cutoff_date,
        Memory.access_count <= stale_access_threshold
    )

    if project:
        query = query.filter(Memory.project == project)

    candidates = query.all()

    stale = []
    for memory in candidates:
        if len(stale) >= limit:
            break

        in_degree = _compute_in_degree(db, memory.id)

        # Memory is stale only if centrality is also low
        if in_degree < stale_centrality_threshold:
            age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - memory.created_at).days
            stale.append({
                "memory_id": memory.id,
                "subject": memory.subject or "(no subject)",
                "type": memory.memory_type,
                "age_days": age_days,
                "access_count": memory.access_count,
                "in_degree": in_degree
            })

    return stale


def _find_orphan_edges(
    db,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Find edges pointing to archived memories.

    Args:
        db: Database session
        limit: Maximum orphan edges to return

    Returns:
        List of orphaned edges
    """
    orphans = []

    # Find edges where target is archived
    edges = db.query(MemoryEdge).join(
        Memory, MemoryEdge.target_id == Memory.id
    ).filter(
        Memory.is_archived == True
    ).limit(limit).all()

    for edge in edges:
        source = db.query(Memory).filter(Memory.id == edge.source_id).first()
        target = db.query(Memory).filter(Memory.id == edge.target_id).first()

        orphans.append({
            "edge_id": edge.id,
            "source": edge.source_id,
            "target": edge.target_id,
            "relation_type": edge.relation_type,
            "reason": f"target #{edge.target_id} is archived",
            "source_subject": source.subject if source else "(not found)",
            "target_subject": target.subject if target else "(not found)"
        })

    return orphans


def _find_missing_embeddings(
    db,
    project: Optional[str] = None,
    limit: int = 20
) -> List[int]:
    """
    Find memories missing embeddings.

    Args:
        db: Database session
        project: Filter by project
        limit: Maximum to return

    Returns:
        List of memory IDs
    """
    query = db.query(Memory.id).filter(
        Memory.is_archived == False,
        Memory.embedding.is_(None)
    )

    if project:
        query = query.filter(Memory.project == project)

    return [mid for (mid,) in query.limit(limit).all()]


def _find_contradictions(
    db,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Find memories with 'contradicts' edges that need resolution.

    Args:
        db: Database session
        limit: Maximum contradictions to return

    Returns:
        List of contradiction pairs
    """
    contradictions = []

    edges = db.query(MemoryEdge).filter(
        MemoryEdge.relation_type == "contradicts"
    ).limit(limit).all()

    for edge in edges:
        source = db.query(Memory).filter(Memory.id == edge.source_id).first()
        target = db.query(Memory).filter(Memory.id == edge.target_id).first()

        if source and target:
            contradictions.append({
                "memory_id": edge.source_id,
                "contradicts": edge.target_id,
                "needs_resolution": True,
                "subject_1": source.subject or "(no subject)",
                "subject_2": target.subject or "(no subject)",
                "edge_id": edge.id
            })

    return contradictions


def _find_unlinked_memories(
    db,
    project: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Find memories with no edges (isolated nodes).

    Args:
        db: Database session
        project: Filter by project
        limit: Maximum to return

    Returns:
        List of unlinked memories
    """
    # Find memories with no outgoing or incoming edges
    query = db.query(Memory).filter(
        Memory.is_archived == False
    )

    if project:
        query = query.filter(Memory.project == project)

    candidates = query.all()
    unlinked = []

    for memory in candidates:
        if len(unlinked) >= limit:
            break

        # Check if memory has any edges
        has_edges = db.query(MemoryEdge).filter(
            or_(
                MemoryEdge.source_id == memory.id,
                MemoryEdge.target_id == memory.id
            )
        ).first()

        if not has_edges:
            age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - memory.created_at).days
            unlinked.append({
                "memory_id": memory.id,
                "subject": memory.subject or "(no subject)",
                "type": memory.memory_type,
                "age_days": age_days,
                "access_count": memory.access_count
            })

    return unlinked


def audit_palace(
    checks: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    limit_per_category: int = 20
) -> Dict[str, Any]:
    """
    Audit palace health and return actionable findings.

    Args:
        checks: List of check names to run (None = all).
                Valid: "duplicates", "stale", "orphan_edges", "missing_embeddings",
                       "contradictions", "unlinked"
        thresholds: Override thresholds, e.g. {"duplicate_similarity": 0.95,
                    "stale_days": 90, "stale_max_access": 5}
        project: Filter by project
        limit_per_category: Cap results per issue type

    Returns:
        Dict with findings by category and summary
    """
    db = get_session()
    try:
        # Default to all checks if none specified
        if checks is None:
            checks = ["duplicates", "stale", "orphan_edges", "missing_embeddings",
                     "contradictions", "unlinked"]

        # Default thresholds
        default_thresholds = {
            "duplicate_similarity": 0.92,
            "stale_days": 90,
            "stale_max_access": 2,
            "stale_min_centrality": 3
        }

        # Merge user-provided thresholds
        if thresholds:
            default_thresholds.update(thresholds)

        result = {}

        # Check duplicates
        if "duplicates" in checks:
            duplicates = _find_duplicates(
                db,
                threshold=default_thresholds["duplicate_similarity"],
                project=project,
                limit=limit_per_category
            )
            result["duplicates"] = duplicates

        # Check stale memories
        if "stale" in checks:
            stale = _find_stale_memories(
                db,
                stale_days=default_thresholds["stale_days"],
                stale_access_threshold=default_thresholds["stale_max_access"],
                stale_centrality_threshold=default_thresholds["stale_min_centrality"],
                project=project,
                limit=limit_per_category
            )
            result["stale"] = stale

        # Check orphan edges
        if "orphan_edges" in checks:
            orphans = _find_orphan_edges(db, limit=limit_per_category)
            result["orphan_edges"] = orphans

        # Check missing embeddings
        if "missing_embeddings" in checks:
            missing = _find_missing_embeddings(db, project=project, limit=limit_per_category)
            result["missing_embeddings"] = missing

        # Check contradictions
        if "contradictions" in checks:
            contradictions = _find_contradictions(db, limit=limit_per_category)
            result["contradictions"] = contradictions

        # Check unlinked memories
        if "unlinked" in checks:
            unlinked = _find_unlinked_memories(db, project=project, limit=limit_per_category)
            result["unlinked"] = unlinked

        # Build summary
        total_issues = 0
        summary = {}

        if "duplicates" in checks:
            count = len(result.get("duplicates", []))
            summary["duplicates_found"] = count
            total_issues += count

        if "stale" in checks:
            count = len(result.get("stale", []))
            summary["stale_found"] = count
            total_issues += count

        if "orphan_edges" in checks:
            count = len(result.get("orphan_edges", []))
            summary["orphan_edges_found"] = count
            total_issues += count

        if "missing_embeddings" in checks:
            count = len(result.get("missing_embeddings", []))
            summary["missing_embeddings_found"] = count
            total_issues += count

        if "contradictions" in checks:
            count = len(result.get("contradictions", []))
            summary["contradictions_found"] = count
            total_issues += count

        if "unlinked" in checks:
            count = len(result.get("unlinked", []))
            summary["unlinked_found"] = count
            total_issues += count

        summary["total_issues"] = total_issues
        result["summary"] = summary

        return result
    finally:
        db.close()


def batch_archive_memories(
    older_than_days: Optional[int] = None,
    max_access_count: Optional[int] = None,
    memory_type: Optional[str] = None,
    project: Optional[str] = None,
    memory_ids: Optional[List[int]] = None,
    centrality_protection: bool = True,
    min_centrality_threshold: int = 5,
    dry_run: bool = True,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Archive multiple memories matching criteria.

    DEPRECATED: Use memory_service.archive_memory() instead.
    This is a thin wrapper for backward compatibility.

    Safety: dry_run=True by default. Returns preview of what would be archived.

    Args:
        older_than_days: Age filter
        max_access_count: Low-access filter
        memory_type: Type filter
        project: Project filter
        memory_ids: Explicit ID list (overrides other filters)
        centrality_protection: Protect high-centrality memories (deprecated, foundational protection is automatic)
        min_centrality_threshold: In-degree count that grants protection (deprecated)
        dry_run: Preview only (default safe)
        reason: Archival reason for audit trail

    Returns:
        Dict with preview or execution results
    """
    # Delegate to the new unified archive_memory function
    from memory_palace.services.memory_service import archive_memory

    return archive_memory(
        memory_ids=memory_ids,
        older_than_days=older_than_days,
        max_access_count=max_access_count,
        project=project,
        memory_type=memory_type,
        centrality_protection=centrality_protection,
        min_centrality_threshold=min_centrality_threshold,
        dry_run=dry_run,
        reason=reason
    )


def reembed_memories(
    older_than_days: Optional[int] = None,
    memory_ids: Optional[List[int]] = None,
    project: Optional[str] = None,
    all_memories: bool = False,
    missing_only: bool = False,
    batch_size: int = 50,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Regenerate embeddings for memories.

    Use when:
    - Embedding model changes
    - Embeddings seem to be returning poor results
    - After bulk import
    - Backfilling missing embeddings (missing_only=True)

    Args:
        older_than_days: Re-embed old embeddings
        memory_ids: Explicit list
        project: Filter by project
        all_memories: Nuclear option - re-embed everything
        missing_only: Only re-embed memories with NULL/empty embeddings (replaces backfill_embeddings)
        batch_size: Batch size for processing
        dry_run: Preview only (default safe)

    Returns:
        Dict with preview or execution results
    """
    db = get_session()
    try:
        # Build query
        if missing_only:
            # Only memories with missing embeddings
            query = db.query(Memory).filter(
                Memory.is_archived == False,
                Memory.embedding.is_(None)
            )
            if project:
                query = query.filter(Memory.project == project)
        elif memory_ids:
            query = db.query(Memory).filter(Memory.id.in_(memory_ids))
        elif all_memories:
            query = db.query(Memory).filter(Memory.is_archived == False)
        else:
            query = db.query(Memory).filter(Memory.is_archived == False)

            if older_than_days:
                cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=older_than_days)
                query = query.filter(Memory.created_at < cutoff_date)

            if project:
                query = query.filter(Memory.project == project)

        memories = query.all()

        if not memories:
            return {"error": "No memories match the criteria"}

        # Estimate time (rough: 200ms per embedding)
        estimated_seconds = len(memories) * 0.2

        if dry_run:
            return {
                "would_reembed": len(memories),
                "estimated_time_seconds": int(estimated_seconds),
                "memories": [
                    {
                        "id": m.id,
                        "subject": m.subject or "(no subject)",
                        "type": m.memory_type
                    }
                    for m in memories[:20]  # Show first 20
                ],
                "note": "DRY RUN - no embeddings regenerated. Set dry_run=False to execute."
            }

        # Execute re-embedding
        success = 0
        failed = 0
        failed_ids = []

        for memory in memories:
            embedding_text = memory.embedding_text()
            embedding = get_embedding(embedding_text)

            if embedding:
                memory.embedding = embedding
                success += 1
            else:
                failed += 1
                failed_ids.append(memory.id)

        db.commit()

        result = {
            "reembedded": success,
            "failed": failed,
            "total": len(memories)
        }

        if failed_ids:
            result["failed_ids"] = failed_ids[:20]  # Show first 20

        return result
    finally:
        db.close()
