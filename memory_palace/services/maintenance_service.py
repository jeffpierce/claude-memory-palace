"""
Maintenance service for Claude Memory Palace.

Provides health checks, bulk archival, consolidation, and re-embedding
operations for palace maintenance.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging

from sqlalchemy import func, or_

from memory_palace.models import Memory, MemoryEdge
from memory_palace.database import get_session
from memory_palace.embeddings import cosine_similarity, get_embedding
from memory_palace.services.memory_service import forget

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
    cutoff_date = datetime.utcnow() - timedelta(days=stale_days)

    # Get candidate memories
    query = db.query(Memory).filter(
        Memory.is_archived == False,
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
            age_days = (datetime.utcnow() - memory.created_at).days
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


def audit_palace(
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
) -> Dict[str, Any]:
    """
    Audit palace health and return actionable findings.

    Args:
        check_duplicates: Find near-duplicates (>0.9 similarity)
        check_stale: Old + low access + low centrality
        check_orphan_edges: Edges pointing to archived memories
        check_embeddings: Memories missing embeddings
        check_contradictions: Find 'contradicts' edges for review
        stale_days: Age threshold for "stale"
        stale_access_threshold: Access count threshold
        stale_centrality_threshold: In-degree below this = not protected
        duplicate_threshold: Similarity threshold for duplicates
        project: Filter by project
        limit_per_category: Cap results per issue type

    Returns:
        Dict with findings by category and summary
    """
    db = get_session()
    try:
        result = {}

        # Check duplicates
        if check_duplicates:
            duplicates = _find_duplicates(
                db,
                threshold=duplicate_threshold,
                project=project,
                limit=limit_per_category
            )
            result["duplicates"] = duplicates

        # Check stale memories
        if check_stale:
            stale = _find_stale_memories(
                db,
                stale_days=stale_days,
                stale_access_threshold=stale_access_threshold,
                stale_centrality_threshold=stale_centrality_threshold,
                project=project,
                limit=limit_per_category
            )
            result["stale"] = stale

        # Check orphan edges
        if check_orphan_edges:
            orphans = _find_orphan_edges(db, limit=limit_per_category)
            result["orphan_edges"] = orphans

        # Check missing embeddings
        if check_embeddings:
            missing = _find_missing_embeddings(db, project=project, limit=limit_per_category)
            result["missing_embeddings"] = missing

        # Check contradictions
        if check_contradictions:
            contradictions = _find_contradictions(db, limit=limit_per_category)
            result["contradictions"] = contradictions

        # Build summary
        total_issues = 0
        summary = {}

        if check_duplicates:
            count = len(result.get("duplicates", []))
            summary["duplicates_found"] = count
            total_issues += count

        if check_stale:
            count = len(result.get("stale", []))
            summary["stale_found"] = count
            total_issues += count

        if check_orphan_edges:
            count = len(result.get("orphan_edges", []))
            summary["orphan_edges_found"] = count
            total_issues += count

        if check_embeddings:
            count = len(result.get("missing_embeddings", []))
            summary["missing_embeddings_found"] = count
            total_issues += count

        if check_contradictions:
            count = len(result.get("contradictions", []))
            summary["contradictions_found"] = count
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

    Safety: dry_run=True by default. Returns preview of what would be archived.

    Args:
        older_than_days: Age filter
        max_access_count: Low-access filter
        memory_type: Type filter
        project: Project filter
        memory_ids: Explicit ID list (overrides other filters)
        centrality_protection: Protect high-centrality memories
        min_centrality_threshold: In-degree count that grants protection
        dry_run: Preview only (default safe)
        reason: Archival reason for audit trail

    Returns:
        Dict with preview or execution results
    """
    db = get_session()
    try:
        # Build query
        if memory_ids:
            # Explicit ID list
            candidates = db.query(Memory).filter(
                Memory.id.in_(memory_ids),
                Memory.is_archived == False
            ).all()
        else:
            # Build filter query
            query = db.query(Memory).filter(Memory.is_archived == False)

            if older_than_days:
                cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
                query = query.filter(Memory.created_at < cutoff_date)

            if max_access_count is not None:
                query = query.filter(Memory.access_count <= max_access_count)

            if memory_type:
                query = query.filter(Memory.memory_type == memory_type)

            if project:
                query = query.filter(Memory.project == project)

            candidates = query.all()

        # Apply centrality protection
        to_archive = []
        protected = []

        for memory in candidates:
            if centrality_protection:
                in_degree = _compute_in_degree(db, memory.id)
                if in_degree >= min_centrality_threshold:
                    age_days = (datetime.utcnow() - memory.created_at).days
                    protected.append({
                        "id": memory.id,
                        "subject": memory.subject or "(no subject)",
                        "in_degree": in_degree,
                        "age_days": age_days,
                        "reason": f"protected by centrality (in-degree={in_degree})"
                    })
                    continue

            age_days = (datetime.utcnow() - memory.created_at).days
            to_archive.append({
                "id": memory.id,
                "subject": memory.subject or "(no subject)",
                "type": memory.memory_type,
                "age_days": age_days,
                "access_count": memory.access_count
            })

        # Execute archival if not dry run
        if not dry_run:
            archived_count = 0
            for mem_info in to_archive:
                result = forget(mem_info["id"], reason=reason)
                if "error" not in result:
                    archived_count += 1

            return {
                "archived": archived_count,
                "memories": to_archive,
                "protected": protected
            }
        else:
            # Dry run - preview only
            return {
                "would_archive": len(to_archive),
                "memories": to_archive,
                "protected": protected,
                "note": "DRY RUN - no memories were archived. Set dry_run=False to execute."
            }
    finally:
        db.close()


def reembed_memories(
    older_than_days: Optional[int] = None,
    memory_ids: Optional[List[int]] = None,
    project: Optional[str] = None,
    all_memories: bool = False,
    batch_size: int = 50,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Regenerate embeddings for memories.

    Use when:
    - Embedding model changes
    - Embeddings seem to be returning poor results
    - After bulk import

    Args:
        older_than_days: Re-embed old embeddings
        memory_ids: Explicit list
        project: Filter by project
        all_memories: Nuclear option - re-embed everything
        batch_size: Batch size for processing
        dry_run: Preview only (default safe)

    Returns:
        Dict with preview or execution results
    """
    db = get_session()
    try:
        # Build query
        if memory_ids:
            query = db.query(Memory).filter(Memory.id.in_(memory_ids))
        elif all_memories:
            query = db.query(Memory).filter(Memory.is_archived == False)
        else:
            query = db.query(Memory).filter(Memory.is_archived == False)

            if older_than_days:
                cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
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
