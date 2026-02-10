"""
Knowledge graph service for Claude Memory Palace.

Provides functions for creating, traversing, and managing relationships
between memories via the memory_edges table.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set

from sqlalchemy import or_, and_
from sqlalchemy.orm import joinedload

from memory_palace.models import Memory, MemoryEdge, RELATIONSHIP_TYPES
from memory_palace.database import get_session


def link_memories(
    source_id: int,
    target_id: int,
    relation_type: str,
    strength: float = 1.0,
    bidirectional: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    created_by: Optional[str] = None,
    archive_old: bool = False
) -> Dict[str, Any]:
    """
    Create a relationship edge between two memories.

    Args:
        source_id: ID of the source memory
        target_id: ID of the target memory
        relation_type: Type of relationship (supersedes, relates_to, derived_from,
                       contradicts, exemplifies, refines, or custom)
        strength: Edge weight 0.0-1.0 for weighted traversal (default 1.0)
        bidirectional: If True, edge works in both directions (default False)
        metadata: Optional extra data to store with the edge
        created_by: Instance ID that created this edge
        archive_old: If True AND relation_type="supersedes", automatically archive the target memory

    Returns:
        Dict with edge ID and confirmation, or error
    """
    db = get_session()
    try:
        # Validate memories exist
        source = db.query(Memory).filter(Memory.id == source_id).first()
        target = db.query(Memory).filter(Memory.id == target_id).first()

        if not source:
            return {"error": f"Source memory {source_id} not found"}
        if not target:
            return {"error": f"Target memory {target_id} not found"}

        # Check for self-loop
        if source_id == target_id:
            return {"error": "Cannot create edge from memory to itself"}

        # Check if edge already exists
        existing = db.query(MemoryEdge).filter(
            MemoryEdge.source_id == source_id,
            MemoryEdge.target_id == target_id,
            MemoryEdge.relation_type == relation_type
        ).first()

        if existing:
            return {
                "error": f"Edge already exists (id={existing.id})",
                "existing_edge_id": existing.id
            }

        # Clamp strength to valid range
        strength = max(0.0, min(1.0, strength))

        # Create the edge
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            bidirectional=bidirectional,
            edge_metadata=metadata or {},
            created_by=created_by
        )
        db.add(edge)

        # Handle supersession archival if requested
        archived_old = False
        if archive_old and relation_type == "supersedes":
            if not target.is_archived:
                target.is_archived = True
                if target.source_context:
                    target.source_context += f"\n[SUPERSEDED by #{source_id}]"
                else:
                    target.source_context = f"[SUPERSEDED by #{source_id}]"
                archived_old = True

        db.commit()
        db.refresh(edge)

        # Build response
        direction = "<->" if bidirectional else "->"
        source_label = f"#{source_id}"
        if source.subject:
            source_label += f" ({source.subject})"
        target_label = f"#{target_id}"
        if target.subject:
            target_label += f" ({target.subject})"

        result = {
            "id": edge.id,
            "message": f"Created edge: {source_label} {direction}[{relation_type}]{direction} {target_label}",
            "relation_type": relation_type,
            "bidirectional": bidirectional
        }

        if archived_old:
            result["old_memory_archived"] = True

        return result
    finally:
        db.close()


def unlink_memories(
    source_id: int,
    target_id: int,
    relation_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Remove relationship edge(s) between two memories.

    Args:
        source_id: ID of the source memory
        target_id: ID of the target memory
        relation_type: Specific relation to remove (optional - if None, removes all edges between the pair)

    Returns:
        Dict with count of removed edges
    """
    db = get_session()
    try:
        # Build query for edges to delete
        query = db.query(MemoryEdge).filter(
            MemoryEdge.source_id == source_id,
            MemoryEdge.target_id == target_id
        )

        if relation_type:
            query = query.filter(MemoryEdge.relation_type == relation_type)

        edges = query.all()

        if not edges:
            return {"error": f"No edges found from {source_id} to {target_id}" +
                    (f" with type '{relation_type}'" if relation_type else "")}

        count = len(edges)
        for edge in edges:
            db.delete(edge)
        db.commit()

        return {
            "removed": count,
            "message": f"Removed {count} edge(s) from #{source_id} to #{target_id}"
        }
    finally:
        db.close()


def get_related_memories(
    memory_id: int,
    relation_type: Optional[str] = None,
    direction: str = "both",
    include_memory_content: bool = True,
    detail_level: str = "summary"
) -> Dict[str, Any]:
    """
    Get memories related to a given memory via edges.

    Args:
        memory_id: ID of the memory to find relations for
        relation_type: Filter by specific relation type (optional)
        direction: "outgoing", "incoming", or "both" (default "both")
        include_memory_content: Include memory details in response (default True)
        detail_level: "summary" or "verbose" for memory content

    Returns:
        Dict with outgoing and/or incoming edges and related memories
    """
    db = get_session()
    try:
        # Verify memory exists
        memory = db.query(Memory).filter(Memory.id == memory_id).first()
        if not memory:
            return {"error": f"Memory {memory_id} not found"}

        result = {
            "memory_id": memory_id,
            "subject": memory.subject
        }

        # Get outgoing edges (this memory -> others)
        if direction in ("outgoing", "both"):
            out_query = db.query(MemoryEdge).filter(MemoryEdge.source_id == memory_id)
            if relation_type:
                out_query = out_query.filter(MemoryEdge.relation_type == relation_type)
            outgoing = out_query.all()

            out_list = []
            for edge in outgoing:
                edge_dict = edge.to_dict()
                if include_memory_content:
                    target = db.query(Memory).filter(Memory.id == edge.target_id).first()
                    if target:
                        edge_dict["target_memory"] = target.to_dict(detail_level=detail_level)
                out_list.append(edge_dict)
            result["outgoing"] = out_list

        # Get incoming edges (others -> this memory)
        if direction in ("incoming", "both"):
            in_query = db.query(MemoryEdge).filter(MemoryEdge.target_id == memory_id)
            if relation_type:
                in_query = in_query.filter(MemoryEdge.relation_type == relation_type)
            incoming = in_query.all()

            in_list = []
            for edge in incoming:
                edge_dict = edge.to_dict()
                if include_memory_content:
                    source = db.query(Memory).filter(Memory.id == edge.source_id).first()
                    if source:
                        edge_dict["source_memory"] = source.to_dict(detail_level=detail_level)
                in_list.append(edge_dict)
            result["incoming"] = in_list

        # Also include bidirectional edges that point TO this memory
        # (they logically apply in both directions)
        if direction in ("outgoing", "both"):
            bidir_query = db.query(MemoryEdge).filter(
                MemoryEdge.target_id == memory_id,
                MemoryEdge.bidirectional == True
            )
            if relation_type:
                bidir_query = bidir_query.filter(MemoryEdge.relation_type == relation_type)
            bidirectional = bidir_query.all()

            if bidirectional:
                bidir_list = []
                for edge in bidirectional:
                    edge_dict = edge.to_dict()
                    edge_dict["_note"] = "bidirectional edge (stored as incoming, applies as outgoing too)"
                    if include_memory_content:
                        source = db.query(Memory).filter(Memory.id == edge.source_id).first()
                        if source:
                            edge_dict["related_memory"] = source.to_dict(detail_level=detail_level)
                    bidir_list.append(edge_dict)
                result["bidirectional_incoming"] = bidir_list

        return result
    finally:
        db.close()


def supersede_memory(
    new_memory_id: int,
    old_memory_id: int,
    archive_old: bool = True,
    created_by: Optional[str] = None
) -> Dict[str, Any]:
    """
    Mark a new memory as superseding an old one.

    DEPRECATED: Use link_memories() with archive_old=True instead.
    This is a convenience wrapper that will be removed in a future version.

    This wrapper:
    1. Creates a 'supersedes' edge from new -> old
    2. Optionally archives the old memory

    Args:
        new_memory_id: ID of the newer/updated memory
        old_memory_id: ID of the older/outdated memory
        archive_old: Whether to archive the old memory (default True)
        created_by: Instance ID that created this supersession

    Returns:
        Dict with edge ID and archive status
    """
    # Delegate to link_memories with archive_old parameter
    return link_memories(
        source_id=new_memory_id,
        target_id=old_memory_id,
        relation_type="supersedes",
        strength=1.0,
        bidirectional=False,
        metadata={"superseded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()},
        created_by=created_by,
        archive_old=archive_old
    )


def traverse_graph(
    start_id: int,
    max_depth: int = 2,
    relation_types: Optional[List[str]] = None,
    direction: str = "outgoing",
    min_strength: float = 0.0,
    include_archived: bool = False,
    detail_level: str = "summary"
) -> Dict[str, Any]:
    """
    Traverse the memory graph from a starting point.

    Performs breadth-first traversal following edges up to max_depth.

    Args:
        start_id: ID of the memory to start from
        max_depth: Maximum traversal depth (default 2, max 5)
        relation_types: List of relation types to follow (optional - all if None)
        direction: "outgoing", "incoming", or "both" (default "outgoing")
        min_strength: Minimum edge strength to follow (default 0.0)
        include_archived: Include archived memories in results (default False)
        detail_level: "summary" or "verbose" for memory content

    Returns:
        Dict with nodes (memories) and edges discovered during traversal
    """
    db = get_session()
    try:
        # Clamp max_depth
        max_depth = max(1, min(5, max_depth))

        # Verify start memory exists
        start = db.query(Memory).filter(Memory.id == start_id).first()
        if not start:
            return {"error": f"Start memory {start_id} not found"}

        # Track visited nodes and edges
        visited_ids: Set[int] = {start_id}
        discovered_edges: List[Dict] = []
        nodes_by_depth: Dict[int, List[Dict]] = {0: [start.to_dict(detail_level=detail_level)]}

        # BFS traversal
        current_frontier = {start_id}

        for depth in range(1, max_depth + 1):
            next_frontier: Set[int] = set()

            for node_id in current_frontier:
                # Build edge query based on direction
                if direction == "outgoing":
                    edge_query = db.query(MemoryEdge).filter(MemoryEdge.source_id == node_id)
                elif direction == "incoming":
                    edge_query = db.query(MemoryEdge).filter(MemoryEdge.target_id == node_id)
                else:  # both
                    edge_query = db.query(MemoryEdge).filter(
                        or_(
                            MemoryEdge.source_id == node_id,
                            MemoryEdge.target_id == node_id
                        )
                    )

                # Filter by relation types if specified
                if relation_types:
                    edge_query = edge_query.filter(MemoryEdge.relation_type.in_(relation_types))

                # Filter by strength
                if min_strength > 0:
                    edge_query = edge_query.filter(MemoryEdge.strength >= min_strength)

                edges = edge_query.all()

                for edge in edges:
                    # Determine the "other" node
                    if edge.source_id == node_id:
                        other_id = edge.target_id
                    else:
                        other_id = edge.source_id
                        # Skip if edge isn't bidirectional and we're going "backwards"
                        if direction == "outgoing" and not edge.bidirectional:
                            continue

                    # Skip if already visited
                    if other_id in visited_ids:
                        # Still record the edge for completeness
                        edge_dict = edge.to_dict()
                        edge_dict["_depth"] = depth
                        edge_dict["_note"] = "connects to already-visited node"
                        discovered_edges.append(edge_dict)
                        continue

                    # Get the other memory
                    other = db.query(Memory).filter(Memory.id == other_id).first()
                    if not other:
                        continue

                    # Skip archived unless requested
                    if other.is_archived and not include_archived:
                        continue

                    # Add to results
                    visited_ids.add(other_id)
                    next_frontier.add(other_id)

                    edge_dict = edge.to_dict()
                    edge_dict["_depth"] = depth
                    discovered_edges.append(edge_dict)

                    if depth not in nodes_by_depth:
                        nodes_by_depth[depth] = []
                    nodes_by_depth[depth].append(other.to_dict(detail_level=detail_level))

            current_frontier = next_frontier
            if not current_frontier:
                break

        # Flatten nodes for response
        all_nodes = []
        for d in sorted(nodes_by_depth.keys()):
            for node in nodes_by_depth[d]:
                node["_depth"] = d
                all_nodes.append(node)

        return {
            "start_id": start_id,
            "max_depth": max_depth,
            "direction": direction,
            "nodes_found": len(all_nodes),
            "edges_found": len(discovered_edges),
            "nodes": all_nodes,
            "edges": discovered_edges
        }
    finally:
        db.close()


def get_relationship_types() -> Dict[str, Any]:
    """
    Get information about available relationship types.

    Returns:
        Dict with standard types and their descriptions, plus custom types in use
    """
    db = get_session()
    try:
        # Standard types with descriptions
        standard_types = {
            "supersedes": "Newer memory replaces older (directional)",
            "relates_to": "General association (often bidirectional)",
            "derived_from": "This memory came from processing that one (directional)",
            "contradicts": "Memories are in tension (bidirectional)",
            "exemplifies": "This is an example of that concept (directional)",
            "refines": "Adds detail/nuance to another memory (directional)"
        }

        # Find any custom types in use
        all_types = db.query(MemoryEdge.relation_type).distinct().all()
        all_types = [t[0] for t in all_types]

        custom_types = [t for t in all_types if t not in RELATIONSHIP_TYPES]

        return {
            "standard_types": standard_types,
            "custom_types_in_use": custom_types,
            "note": "Custom types are allowed - use standard types when they fit, create new ones when needed"
        }
    finally:
        db.close()
