"""
Memory service for Memory Palace.

Provides functions for storing, recalling, archiving, and managing memories.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import math

from sqlalchemy import func, or_, String

from memory_palace.models import Memory, MemoryEdge, _normalize_projects, _project_contains, _projects_overlap
from memory_palace.database import get_session
from memory_palace.embeddings import get_embedding, cosine_similarity
from memory_palace.config_v2 import get_auto_link_config, get_instances, is_postgres
from memory_palace.llm import classify_edge_type, classify_edge_types_batch


# Valid source types for memories
VALID_SOURCE_TYPES = ["conversation", "explicit", "inferred", "observation"]

# Centrality-weighted retrieval configuration
# These control how much each factor influences ranking:
# score = (cosine_similarity × α) + (log(access_count + 1) × β) + (centrality × γ)
DEFAULT_WEIGHT_SIMILARITY = 0.7  # Semantic relevance (primary signal)
DEFAULT_WEIGHT_ACCESS = 0.15     # Popularity/usefulness (secondary)
DEFAULT_WEIGHT_CENTRALITY = 0.15 # Graph importance (secondary)

def _get_retrieval_weights() -> Tuple[float, float, float]:
    """
    Get the retrieval weighting factors from environment variables.

    Returns:
        Tuple of (alpha, beta, gamma) weights
    """
    alpha = float(os.environ.get("MEMORY_PALACE_WEIGHT_SIMILARITY", DEFAULT_WEIGHT_SIMILARITY))
    beta = float(os.environ.get("MEMORY_PALACE_WEIGHT_ACCESS", DEFAULT_WEIGHT_ACCESS))
    gamma = float(os.environ.get("MEMORY_PALACE_WEIGHT_CENTRALITY", DEFAULT_WEIGHT_CENTRALITY))

    # Normalize to ensure they sum to ~1.0 (for interpretability)
    total = alpha + beta + gamma
    if total > 0:
        alpha /= total
        beta /= total
        gamma /= total

    return alpha, beta, gamma


def _get_graph_context_for_memories(
    db,
    memories: List[Memory],
    max_depth: int = 1,
    direction: Optional[str] = None,
    relation_types: Optional[List[str]] = None,
    min_strength: Optional[float] = None,
    graph_mode: str = "summary"
) -> Dict[str, Any]:
    """
    Fetch deduplicated graph context for a list of memories using BFS frontier expansion.

    Uses adjacency list representation: nodes emitted once, edges reference IDs only.
    Supports depth-1 (immediate connections) or depth-2+ (follow edges from discovered nodes).

    Args:
        db: Database session
        memories: List of Memory objects to fetch graph context for
        max_depth: How many hops to follow (1 = immediate edges, 2 = edges of edges, etc.)
        direction: "outgoing", "incoming", or None for both
        relation_types: List of relation types to follow (optional - all if None)
        min_strength: Minimum edge strength to follow (optional)
        graph_mode: "summary" for per-node stats, "full" for raw edge list (default "summary")

    Returns:
        If graph_mode == "full": {"nodes": {id: subject, ...}, "edges": [{source, target, type, strength}, ...]}
        If graph_mode == "summary": {"nodes": {id: {subject, connections, avg_strength, edge_types}, ...}, "total_edges": int, "seed_ids": list}
            edge_types include direction indicators: > (outgoing), < (incoming), <> (bidirectional)
        Returns {} if no edges found.
    """
    if not memories:
        return {}

    # Clamp depth to sensible range
    max_depth = max(1, min(max_depth, 5))

    nodes: Dict[str, str] = {}  # "id" -> subject (dedup registry, string keys for TOON compat)
    edges: List[Dict[str, Any]] = []  # flat edge list
    seen_edges: set = set()  # (source_id, target_id, relation_type) for dedup

    # Seed nodes from the requested memories
    frontier_ids = set(m.id for m in memories)
    for m in memories:
        nodes[str(m.id)] = m.subject or "(no subject)"

    for depth in range(max_depth):
        if not frontier_ids:
            break  # No new nodes to explore

        # Batch fetch outgoing + incoming edges for current frontier based on direction
        if direction == "outgoing":
            outgoing = db.query(MemoryEdge).filter(MemoryEdge.source_id.in_(frontier_ids))
            if relation_types:
                outgoing = outgoing.filter(MemoryEdge.relation_type.in_(relation_types))
            if min_strength is not None:
                outgoing = outgoing.filter(MemoryEdge.strength >= min_strength)
            outgoing = outgoing.all()
            incoming = []
        elif direction == "incoming":
            incoming = db.query(MemoryEdge).filter(MemoryEdge.target_id.in_(frontier_ids))
            if relation_types:
                incoming = incoming.filter(MemoryEdge.relation_type.in_(relation_types))
            if min_strength is not None:
                incoming = incoming.filter(MemoryEdge.strength >= min_strength)
            incoming = incoming.all()
            outgoing = []
        else:  # both or None
            outgoing = db.query(MemoryEdge).filter(MemoryEdge.source_id.in_(frontier_ids))
            if relation_types:
                outgoing = outgoing.filter(MemoryEdge.relation_type.in_(relation_types))
            if min_strength is not None:
                outgoing = outgoing.filter(MemoryEdge.strength >= min_strength)
            outgoing = outgoing.all()

            incoming = db.query(MemoryEdge).filter(MemoryEdge.target_id.in_(frontier_ids))
            if relation_types:
                incoming = incoming.filter(MemoryEdge.relation_type.in_(relation_types))
            if min_strength is not None:
                incoming = incoming.filter(MemoryEdge.strength >= min_strength)
            incoming = incoming.all()

        next_frontier: set = set()
        new_referenced_ids: set = set()

        for e in outgoing + incoming:
            edge_key = (e.source_id, e.target_id, e.relation_type)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append({
                "source": e.source_id,
                "target": e.target_id,
                "type": e.relation_type,
                "strength": round(e.strength, 4)
            })
            # Track new nodes we haven't seen
            for nid in (e.source_id, e.target_id):
                if str(nid) not in nodes:
                    new_referenced_ids.add(nid)
                    next_frontier.add(nid)

        # Batch fetch subjects for newly discovered nodes
        if new_referenced_ids:
            refs = db.query(Memory.id, Memory.subject).filter(
                Memory.id.in_(new_referenced_ids)
            ).all()
            for ref in refs:
                nodes[str(ref.id)] = ref.subject or "(no subject)"

        frontier_ids = next_frontier

    # Return based on graph_mode
    if not edges:
        return {}

    if graph_mode == "full":
        return {"nodes": nodes, "edges": edges}

    # graph_mode == "summary": aggregate per-node stats
    seed_ids = [m.id for m in memories]
    node_stats: Dict[str, Dict[str, Any]] = {}
    for node_id, subject in nodes.items():
        # Count edges involving this node
        node_edges = [e for e in edges if str(e["source"]) == node_id or str(e["target"]) == node_id]
        if not node_edges:
            # Include seed nodes even with no edges
            if int(node_id) in seed_ids:
                node_stats[node_id] = {
                    "subject": subject,
                    "connections": 0,
                    "avg_strength": 0,
                    "edge_types": []
                }
            continue
        strengths = [e["strength"] for e in node_edges]

        # Track edge types with direction indicators
        # > = outgoing (this node is source)
        # < = incoming (this node is target)
        # <> = bidirectional (appears as both source and target for this type)
        edge_type_directions: Dict[str, set] = {}
        for e in node_edges:
            edge_type = e["type"]
            if edge_type not in edge_type_directions:
                edge_type_directions[edge_type] = set()

            if str(e["source"]) == node_id:
                edge_type_directions[edge_type].add(">")
            if str(e["target"]) == node_id:
                edge_type_directions[edge_type].add("<")

        # Format edge types with direction indicators
        edge_types_with_dir = []
        for edge_type in sorted(edge_type_directions.keys()):
            directions = edge_type_directions[edge_type]
            if directions == {">", "<"}:
                prefix = "<>"
            elif ">" in directions:
                prefix = ">"
            else:  # "<" in directions
                prefix = "<"
            edge_types_with_dir.append(f"{prefix}{edge_type}")

        node_stats[node_id] = {
            "subject": subject,
            "connections": len(node_edges),
            "avg_strength": round(sum(strengths) / len(strengths), 4),
            "edge_types": edge_types_with_dir
        }

    # Cap summary to top N nodes by connection count (seeds always included)
    MAX_SUMMARY_NODES = 30
    if len(node_stats) > MAX_SUMMARY_NODES:
        seed_id_strs = {str(sid) for sid in seed_ids}
        # Separate seeds (always kept) from non-seeds
        seed_entries = {k: v for k, v in node_stats.items() if k in seed_id_strs}
        non_seed_entries = [(k, v) for k, v in node_stats.items() if k not in seed_id_strs]
        # Sort non-seeds by connections descending
        non_seed_entries.sort(key=lambda x: x[1]["connections"], reverse=True)
        # Take top N minus however many seeds we have
        remaining_slots = MAX_SUMMARY_NODES - len(seed_entries)
        top_non_seeds = dict(non_seed_entries[:remaining_slots])
        omitted = len(node_stats) - len(seed_entries) - len(top_non_seeds)
        node_stats = {**seed_entries, **top_non_seeds}
        flat = _flatten_node_stats(node_stats)
        return {"nodes": flat, "total_edges": len(edges), "seed_ids": seed_ids, "omitted_nodes": omitted}
    if not node_stats:
        return {}
    flat = _flatten_node_stats(node_stats)
    return {"nodes": flat, "total_edges": len(edges), "seed_ids": seed_ids}


def _flatten_node_stats(node_stats: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Flatten per-node stat dicts into pipe-delimited strings for compact TOON output.

    Converts: {"subject": "Foo", "connections": 5, "avg_strength": 0.87, "edge_types": [">a", "<b"]}
    Into:     "Foo | 5 connections | avg 0.87 | >a,<b"

    Edge types include direction indicators:
      > = outgoing (this node is source)
      < = incoming (this node is target)
      <> = bidirectional (this node appears as both source and target for this type)
    """
    flat: Dict[str, str] = {}
    for node_id, stats in node_stats.items():
        types_str = ",".join(stats["edge_types"]) if stats["edge_types"] else "none"
        flat[node_id] = (
            f"{stats['subject']} | {stats['connections']} connections "
            f"| avg {stats['avg_strength']:.2f} | {types_str}"
        )
    return flat


def _compute_in_degree_centrality(db, memory_ids: List[int]) -> Dict[int, float]:
    """
    Compute normalized in-degree centrality for a list of memories.

    In-degree = number of incoming edges (how many other memories point to this one)
    Normalized = in_degree / max_in_degree (so scores are 0-1)

    Args:
        db: Database session
        memory_ids: List of memory IDs to compute centrality for

    Returns:
        Dict mapping memory_id -> normalized_centrality (0.0-1.0)
    """
    if not memory_ids:
        return {}

    # Count incoming edges for each memory in one query
    # This is O(1) with proper indexing on target_id
    centrality_query = db.query(
        MemoryEdge.target_id,
        func.count(MemoryEdge.id).label("in_degree")
    ).filter(
        MemoryEdge.target_id.in_(memory_ids)
    ).group_by(MemoryEdge.target_id).all()

    # Build raw centrality map
    centrality_raw = {mid: 0 for mid in memory_ids}
    for target_id, in_degree in centrality_query:
        centrality_raw[target_id] = in_degree

    # Normalize by max (so most central node has score 1.0)
    max_degree = max(centrality_raw.values()) if centrality_raw.values() else 0

    if max_degree == 0:
        # No edges - all memories have centrality 0
        return {mid: 0.0 for mid in memory_ids}

    # Normalize to 0-1 range
    return {mid: degree / max_degree for mid, degree in centrality_raw.items()}


def _find_similar_memories(
    db,
    embedding: List[float],
    exclude_id: int,
    projects: Optional[List[str]] = None,
    threshold: float = 0.675,
) -> List[Tuple[int, float]]:
    """
    Find memories similar to the given embedding.

    Args:
        db: Database session
        embedding: The embedding vector to compare against
        exclude_id: Memory ID to exclude (the new memory itself)
        projects: If set, only match memories in these projects
        threshold: Minimum cosine similarity to include

    Returns:
        List of (memory_id, similarity_score) tuples, sorted by similarity descending.
        No hard cap — caller is responsible for tiering by confidence.
    """
    # Build query for candidate memories
    query = db.query(Memory).filter(
        Memory.id != exclude_id,
        Memory.is_archived == False,
        Memory.embedding.isnot(None)
    )

    if projects:
        query = query.filter(_projects_overlap(projects))

    # Fetch candidates and score them
    candidates = query.all()
    scored = []

    for memory in candidates:
        similarity = cosine_similarity(embedding, memory.embedding)
        if similarity >= threshold:
            scored.append((memory.id, similarity))

    # Sort by similarity (highest first)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def remember(
    instance_id: str,
    memory_type: str,
    content: str,
    subject: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    foundational: bool = False,
    project: Union[str, List[str]] = "life",
    source_type: str = "explicit",
    source_context: Optional[str] = None,
    source_session_id: Optional[str] = None,
    supersedes_id: Optional[int] = None,
    auto_link: Optional[bool] = None,
    database: Optional[str] = None
) -> Dict[str, Any]:
    """
    Store a new memory in the memory palace.

    Args:
        instance_id: Which instance is storing this (must be a configured instance in
            ~/.memory-palace/config.json). Unconfigured instances trigger a warning
            but are still accepted for backward compatibility.
        memory_type: Type of memory (open-ended - use existing types or create new ones)
        content: The actual memory content
        subject: What/who this memory is about (optional but recommended)
        keywords: List of keywords for searchability
        tags: Freeform organizational tags (separate from keywords)
        foundational: True if this is a foundational/core memory (default False)
        project: Project this memory belongs to (default "life" for non-project memories). Can be a string or list of strings.
        source_type: How this memory was created (conversation, explicit, inferred, observation)
        source_context: Snippet of original context
        source_session_id: Link back to conversation session
        supersedes_id: If set, create a 'supersedes' edge to this memory ID and archive it
        auto_link: Override config to enable/disable auto-linking (None = use config)

    Returns:
        Dict with id, subject, embedded status, and links_created (if any)
    """
    # Warn (not error) if instance_id isn't in configured list
    instance_warning = None
    configured_instances = get_instances()
    if instance_id not in configured_instances:
        instance_warning = (
            f"Instance '{instance_id}' is not in configured instances {configured_instances}. "
            f"Memory stored anyway, but consider adding it to ~/.memory-palace/config.json"
        )

    db = get_session(database)
    try:
        # memory_type is open-ended - use existing types when they fit, create new ones when needed
        # Types are included in semantic vector calculation
        if source_type not in VALID_SOURCE_TYPES:
            return {"error": f"Invalid source_type. Must be one of: {VALID_SOURCE_TYPES}"}

        # Normalize project parameter to list
        projects = _normalize_projects(project)

        memory = Memory(
            instance_id=instance_id,
            projects=projects,
            memory_type=memory_type,
            content=content,
            subject=subject,
            keywords=keywords,
            tags=tags,
            foundational=foundational,
            source_type=source_type,
            source_context=source_context,
            source_session_id=source_session_id
        )
        db.add(memory)
        db.commit()
        db.refresh(memory)

        # Generate embedding for semantic search
        embedding_text = memory.embedding_text()
        embedding = get_embedding(embedding_text)
        embedding_status = "generated"
        if embedding:
            memory.embedding = embedding
            db.commit()
        else:
            embedding_status = "failed"
            # Tag the memory so it's easy to find un-embedded memories
            if memory.tags is None:
                memory.tags = []
            if "embedding_failed" not in memory.tags:
                memory.tags = list(memory.tags) + ["embedding_failed"]
                db.commit()

        # Track created links
        links_created = []

        # Handle explicit supersession
        if supersedes_id is not None:
            old_memory = db.query(Memory).filter(Memory.id == supersedes_id).first()
            if old_memory:
                # Create supersedes edge
                edge = MemoryEdge(
                    source_id=memory.id,
                    target_id=supersedes_id,
                    relation_type="supersedes",
                    strength=1.0,
                    bidirectional=False,
                    edge_metadata={"superseded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()},
                    created_by=instance_id
                )
                db.add(edge)

                # Archive the old memory
                if not old_memory.is_archived:
                    old_memory.is_archived = True
                    if old_memory.source_context:
                        old_memory.source_context += f"\n[SUPERSEDED by #{memory.id}]"
                    else:
                        old_memory.source_context = f"[SUPERSEDED by #{memory.id}]"

                db.commit()
                links_created.append({
                    "target": supersedes_id,
                    "target_subject": old_memory.subject or "(no subject)",
                    "type": "supersedes",
                    "archived_old": True
                })

        # Auto-link by similarity if enabled and we have an embedding
        auto_link_config = get_auto_link_config()
        should_auto_link = auto_link if auto_link is not None else auto_link_config["enabled"]

        # Track suggested (sub-threshold) links separately
        suggested_links = []

        if should_auto_link and embedding:
            link_threshold = auto_link_config["link_threshold"]
            suggest_threshold = auto_link_config["suggest_threshold"]
            max_suggestions = auto_link_config["max_suggestions"]

            # Find all similar memories down to the suggest threshold
            # Scoped to same project per auto_link config
            similar_projects = projects if auto_link_config["same_project_only"] else None
            similar = _find_similar_memories(
                db,
                embedding,
                memory.id,
                projects=similar_projects,
                threshold=suggest_threshold,
            )

            # Split into auto-link tier (>= link_threshold) and suggest tier
            auto_tier = [(tid, score) for tid, score in similar if score >= link_threshold]
            suggest_tier = [(tid, score) for tid, score in similar
                           if suggest_threshold <= score < link_threshold][:max_suggestions]

            # Determine if we should classify edge types
            should_classify = auto_link_config.get("classify_edges", True)

            # Pre-fetch target subjects for both tiers in one query
            all_target_ids = [tid for tid, _ in auto_tier + suggest_tier if tid != supersedes_id]
            target_subjects = {}
            if all_target_ids:
                targets = db.query(Memory.id, Memory.subject).filter(
                    Memory.id.in_(all_target_ids)
                ).all()
                target_subjects = {t.id: t.subject for t in targets}

            # Filter auto_tier targets (exclude supersedes duplicates)
            auto_targets = [
                (tid, score) for tid, score in auto_tier
                if tid != supersedes_id
            ]

            # Batch-classify all auto-link edge types in ONE LLM call
            # This replaces N sequential calls with 1 call that has full
            # cross-pair context for more coherent classifications.
            edge_type_map: dict = {}
            if should_classify and auto_targets:
                classify_pairs = [
                    (tid, target_subjects.get(tid, "(no subject)"))
                    for tid, _ in auto_targets
                    if tid in target_subjects
                ]
                if classify_pairs:
                    edge_type_map = classify_edge_types_batch(
                        new_subject=memory.subject or "(no subject)",
                        targets=classify_pairs,
                    )

            # Create edges using batch classification results
            for target_id, score in auto_targets:
                edge_type = edge_type_map.get(target_id, "relates_to")

                # Bidirectional only for symmetric relationships
                is_bidirectional = edge_type in ("relates_to", "contradicts")

                edge = MemoryEdge(
                    source_id=memory.id,
                    target_id=target_id,
                    relation_type=edge_type,
                    strength=score,
                    bidirectional=is_bidirectional,
                    edge_metadata={
                        "auto_linked": True,
                        "method": "embedding_similarity",
                        "classified": should_classify and target_id in edge_type_map,
                    },
                    created_by=instance_id
                )
                db.add(edge)
                links_created.append({
                    "target": target_id,
                    "target_subject": target_subjects.get(target_id, "(no subject)"),
                    "type": edge_type,
                    "score": round(score, 4)
                })

            # Suggest tier: surface for human review, no edges created
            for target_id, score in suggest_tier:
                if target_id == supersedes_id:
                    continue
                suggested_links.append({
                    "target": target_id,
                    "target_subject": target_subjects.get(target_id, "(no subject)"),
                    "score": round(score, 4)
                })

            if auto_tier:
                db.commit()

        # Build response
        embedded = embedding_status == "generated"
        result = {
            "id": memory.id,
            "subject": subject,
            "embedded": embedded,
        }

        # Surface embedding failures prominently so callers can't miss them
        if not embedded:
            result["warning"] = (
                "EMBEDDING FAILED: Memory stored but NOT semantically searchable. "
                "It will not appear in memory_recall results until embedding is backfilled. "
                "Tagged with 'embedding_failed' for easy discovery."
            )

        if instance_warning:
            result["instance_warning"] = instance_warning

        if links_created:
            result["links_created"] = links_created

        if suggested_links:
            result["suggested_links"] = suggested_links

        return result
    finally:
        db.close()


def _synthesize_memories_with_llm(
    memories: List[Any],
    query: Optional[str] = None,
    similarity_scores: Optional[Dict[int, float]] = None
) -> Optional[str]:
    """
    Use LLM to synthesize memories into a natural language summary.

    Args:
        memories: List of Memory objects to synthesize
        query: The original search query for context (optional - uses generic prompt if None)
        similarity_scores: Optional dict mapping memory.id -> similarity score (0.0-1.0)

    Returns:
        Natural language synthesis, or None if LLM unavailable
    """
    from memory_palace.llm import generate_with_llm, is_llm_available

    if not is_llm_available():
        return None

    if not memories:
        return "No memories found." if not query else "No memories found matching your query."

    # Default query for direct ID fetches (no search context)
    if not query:
        query = "Summarize these memories"

    # Check if all scores are below confidence threshold
    # (only applies if we have scores - keyword fallback won't have them)
    has_scores = similarity_scores and len(similarity_scores) > 0
    all_low_confidence = False
    if has_scores:
        scores_list = [s for s in similarity_scores.values() if s >= 0]  # Exclude -1.0 (no embedding)
        if scores_list and all(s < 0.5 for s in scores_list):
            all_low_confidence = True

    # Build FULL representation for the LLM - no truncation, let Qwen see everything
    memory_texts = []
    for m in memories:
        parts = []

        # Add similarity score if available (semantic search only)
        if has_scores and m.id in similarity_scores:
            score = similarity_scores[m.id]
            if score >= 0:  # Don't show -1.0 (no embedding marker)
                parts.append(f"[similarity: {score:.2f}]")

        # Add metadata and FULL content - no truncation
        parts.append(f"[type: {m.memory_type}]")
        parts.append(f"[id: {m.id}]")
        if m.subject:
            parts.append(f"[subject: {m.subject}]")
        parts.append(f"\n{m.content}")  # Full content, no truncation
        memory_texts.append(" ".join(parts))

    memories_block = "\n\n---\n\n".join(memory_texts)

    # System prompt: focused extraction for small models
    system = """You are a memory recall assistant. Your job is to answer the query using ONLY the information in the provided memories.

RULES:
- Answer the query directly using facts from the memories
- Include specific details (dates, names, numbers) when present
- If multiple memories are relevant, combine their information
- If memories don't answer the query well, say so briefly
- ONLY report what the memories contain - do NOT add information, speculation, or analysis beyond what's stored
- Do NOT invent details, fill gaps with assumptions, or extrapolate beyond the data
- Do NOT add recommendations, next steps, or research agendas

RELEVANCE:
- Focus on memories with similarity scores > 0.6
- Ignore low-relevance memories (< 0.5) unless they contain directly useful info
- Don't waste space explaining why irrelevant memories are irrelevant

FORMAT:
- Be concise. A few paragraphs is usually enough.
- Use bullet points for lists of facts
- Skip headers unless the answer genuinely covers multiple distinct topics"""

    # Add warning to prompt if all scores are low confidence
    confidence_note = ""
    if all_low_confidence:
        confidence_note = "\n\n**NOTE:** All similarity scores are below 0.5, indicating weak semantic relevance. Evaluate carefully whether these memories actually address the query, or if they're tangential matches.\n"

    prompt = f"""Query: {query}

Found {len(memories)} memories to analyze:{confidence_note}

{memories_block}

Answer the query using these memories. Be direct and factual:"""

    return generate_with_llm(prompt, system=system)


def _format_memories_as_text(memories: List[Any]) -> str:
    """
    Format memories as simple text list (fallback when LLM unavailable).

    Args:
        memories: List of Memory objects

    Returns:
        Simple text list of memories
    """
    if not memories:
        return "No memories found."

    lines = []
    for m in memories:
        subject_part = f" ({m.subject})" if m.subject else ""
        preview = m.content[:100] + "..." if len(m.content) > 100 else m.content
        lines.append(f"- [{m.memory_type}]{subject_part}: {preview}")

    return "\n".join(lines)


def recall(
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
    graph_mode: str = "summary",
    database: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search memories using semantic search (with keyword fallback).

    Args:
        query: Search query (used for semantic similarity or keyword matching)
        instance_id: Filter by instance (optional)
        project: Filter by project (optional, e.g., "memory-palace", "wordleap", "life")
                Can be a string or list of strings. None = no project filter.
        memory_type: Filter by type (optional). Supports wildcards like "code_*" for
                    pattern matching (replaces separate code_recall tool).
        subject: Filter by subject (optional)
        min_foundational: Only return foundational memories if True (optional)
        include_archived: Include archived memories (default False)
        limit: Maximum memories to return (default 20)
        detail_level: "summary" for condensed, "verbose" for full content (only applies when synthesize=True)
        synthesize: If True (default), use LLM to synthesize. If False, return raw memory objects with full content.
        include_graph: Include graph context for top N results (default True)
        graph_top_n: Number of top results to fetch graph context for (default 5).
            NOTE: This limits graph context for performance - searches may return many results,
            but only the top N are enriched with relationship edges.
        graph_depth: How many hops to follow in graph context (1-3, default 1)

    Returns:
        Dictionary with one of three formats:
        - synthesize=True + LLM available: {"summary": str, "count": int, "search_method": str, "memory_ids": list, "graph_context": dict (optional)}
        - synthesize=True + LLM unavailable: {"summary": str (text list), "count": int, "search_method": str, "memory_ids": list, "graph_context": dict (optional)}
        - synthesize=False: {"memories": list[dict], "count": int, "search_method": str, "graph_context": dict (optional)}
          Note: Raw mode always returns verbose content regardless of detail_level parameter.
          Note: graph_context format: {"nodes": {id: subject}, "edges": [{source, target, type, strength}]}
    """
    db = get_session(database)
    try:
        # Get weights from config (not passable per-call anymore)
        alpha, beta, gamma = _get_retrieval_weights()

        # Clamp graph_top_n to sensible range
        graph_top_n = max(0, min(graph_top_n, limit))

        # Build base query with filters (no keyword search yet)
        base_query = db.query(Memory)

        # Filter out archived unless requested
        if not include_archived:
            base_query = base_query.filter(Memory.is_archived == False)

        # Filter by instance if specified
        if instance_id:
            base_query = base_query.filter(Memory.instance_id == instance_id)

        # Filter by project if specified (supports list or string)
        if project is not None:
            if isinstance(project, list):
                base_query = base_query.filter(_projects_overlap(project))
            else:
                base_query = base_query.filter(_project_contains(project))

        # Filter by memory_type (supports wildcards like "code_*")
        if memory_type:
            if "*" in memory_type:
                # Convert wildcard to SQL LIKE pattern
                like_pattern = memory_type.replace("*", "%")
                base_query = base_query.filter(Memory.memory_type.like(like_pattern))
            else:
                base_query = base_query.filter(Memory.memory_type == memory_type)

        if subject:
            base_query = base_query.filter(Memory.subject.ilike(f"%{subject}%"))

        if min_foundational is not None and min_foundational:
            base_query = base_query.filter(Memory.foundational == True)

        # Try semantic search first
        search_method = "semantic"

        # Format query for SFR-Embedding-Mistral (instruction format)
        formatted_query = f"Instruct: Given a memory search query, retrieve relevant memories.\nQuery: {query}"
        query_embedding = get_embedding(formatted_query)

        if query_embedding:
            # Semantic search - use pgvector native ANN search on PostgreSQL,
            # fall back to Python-side cosine similarity on SQLite

            if is_postgres():
                # PostgreSQL + pgvector: Use native <=> operator with HNSW index
                # This is O(log N) approximate nearest neighbor search
                search_method = "semantic (pgvector, centrality-weighted)"

                # Fetch more candidates than limit since we'll re-rank
                # Fetch 3x to ensure we have enough after centrality weighting
                candidate_limit = limit * 3

                # Filter to only memories with embeddings, order by cosine distance
                vector_query = base_query.filter(
                    Memory.embedding.isnot(None)
                ).order_by(
                    Memory.embedding.cosine_distance(query_embedding)
                ).limit(candidate_limit)

                candidates = vector_query.all()

                # Compute similarity scores for the candidates
                similarity_scores = {}
                for memory in candidates:
                    sim = cosine_similarity(query_embedding, memory.embedding)
                    similarity_scores[memory.id] = sim
            else:
                # SQLite: No pgvector, use Python-side cosine similarity
                # This is O(N) but SQLite has no vector index anyway
                search_method = "semantic (centrality-weighted)"
                all_memories = base_query.all()

                # Score each memory by cosine similarity
                candidates = []
                similarity_scores = {}
                for memory in all_memories:
                    if memory.embedding is not None:
                        similarity = cosine_similarity(query_embedding, memory.embedding)
                        similarity_scores[memory.id] = similarity
                        candidates.append(memory)
                    else:
                        # No embedding - give a low similarity score so it appears at the end
                        similarity_scores[memory.id] = -1.0
                        candidates.append(memory)

            # Centrality-weighted re-ranking
            # Compute normalized access counts (log transform to prevent Matthew effect)
            access_counts = {m.id: m.access_count for m in candidates}
            max_access = max(access_counts.values()) if access_counts.values() else 0

            if max_access > 0:
                # Normalize log(access_count + 1) to 0-1 range
                max_log_access = math.log(max_access + 1)
                normalized_access = {
                    mid: math.log(count + 1) / max_log_access
                    for mid, count in access_counts.items()
                }
            else:
                normalized_access = {mid: 0.0 for mid in access_counts.keys()}

            # Compute normalized in-degree centrality
            candidate_ids = [m.id for m in candidates]
            normalized_centrality = _compute_in_degree_centrality(db, candidate_ids)

            # Compute combined scores
            combined_scores = []
            for memory in candidates:
                mid = memory.id

                # Normalize similarity score (already 0-1, but handle -1 for missing embeddings)
                sim_score = max(0.0, similarity_scores.get(mid, 0.0))
                access_score = normalized_access.get(mid, 0.0)
                centrality_score = normalized_centrality.get(mid, 0.0)

                # Weighted combination
                combined = (alpha * sim_score) + (beta * access_score) + (gamma * centrality_score)

                combined_scores.append((memory, combined))

            # Sort by combined score (highest first)
            combined_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top N
            memories = [m for m, score in combined_scores[:limit]]

            # Keep similarity_scores for synthesis context
            # (LLM needs to know semantic relevance, not combined score)
        else:
            # Fallback to keyword search (improved: AND together all words)
            search_method = "keyword (centrality-weighted fallback)"

            if query:
                # Split query into words and AND them together
                words = query.strip().split()
                for word in words:
                    word_pattern = f"%{word}%"
                    base_query = base_query.filter(
                        or_(
                            Memory.content.ilike(word_pattern),
                            Memory.subject.ilike(word_pattern),
                            Memory.keywords.cast(String).ilike(word_pattern)
                        )
                    )

            # Get all keyword matches (no limit yet - we'll re-rank)
            candidates = base_query.all()

            # Without semantic similarity, we use foundational + access + centrality
            # Compute normalized access counts (log transform)
            access_counts = {m.id: m.access_count for m in candidates}
            max_access = max(access_counts.values()) if access_counts.values() else 0

            if max_access > 0:
                max_log_access = math.log(max_access + 1)
                normalized_access = {
                    mid: math.log(count + 1) / max_log_access
                    for mid, count in access_counts.items()
                }
            else:
                normalized_access = {mid: 0.0 for mid in access_counts.keys()}

            # Compute normalized in-degree centrality
            candidate_ids = [m.id for m in candidates]
            normalized_centrality = _compute_in_degree_centrality(db, candidate_ids)

            # Compute normalized foundational (binary: 1 or 0)
            normalized_foundational = {m.id: (1.0 if m.foundational else 0.0) for m in candidates}

            # Compute combined scores
            # In keyword mode: score = (foundational × 0.4) + (access × beta) + (centrality × gamma)
            # Foundational gets a boost since we have no semantic signal
            combined_scores = []
            for memory in candidates:
                mid = memory.id

                foundational_score = normalized_foundational.get(mid, 0.0)
                access_score = normalized_access.get(mid, 0.0)
                centrality_score = normalized_centrality.get(mid, 0.0)

                # Weighted combination (foundational gets 40%, rest split by beta/gamma ratio)
                combined = (0.4 * foundational_score) + (beta * access_score) + (gamma * centrality_score)

                combined_scores.append((memory, combined))

            # Sort by combined score (highest first)
            combined_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top N
            memories = [m for m, score in combined_scores[:limit]]
            similarity_scores = {}

        # Update access tracking for retrieved memories
        for memory in memories:
            memory.last_accessed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            memory.access_count += 1
        db.commit()

        # Fetch graph context if requested
        graph_context = _get_graph_context_for_memories(
            db, memories[:graph_top_n], max_depth=graph_depth, graph_mode=graph_mode
        ) if include_graph and memories else {}

        # Return raw memories if synthesize=False
        # Force verbose detail when returning raw - cloud AI needs full content
        if not synthesize:
            result_memories = []
            for m in memories:
                mem_dict = m.to_dict(detail_level="verbose")
                if m.id in similarity_scores:
                    mem_dict["similarity_score"] = round(similarity_scores[m.id], 4)
                result_memories.append(mem_dict)
            result = {
                "memories": result_memories,
                "count": len(memories),
                "search_method": search_method
            }
            if graph_context:
                result["graph_context"] = graph_context
            return result

        # Try LLM synthesis first, fall back to text list
        # Pass similarity scores if we have them (semantic search only)
        synthesis = _synthesize_memories_with_llm(
            memories,
            query,
            similarity_scores if similarity_scores else None
        )

        if synthesis:
            # LLM synthesis available - return natural language response
            result = {
                "summary": synthesis,
                "count": len(memories),
                "search_method": search_method,
                "memory_ids": [m.id for m in memories]
            }
            if graph_context:
                result["graph_context"] = graph_context
            return result
        else:
            # Fallback to simple text list
            text_list = _format_memories_as_text(memories)
            result = {
                "summary": text_list,
                "count": len(memories),
                "search_method": search_method + " (no LLM)",
                "memory_ids": [m.id for m in memories]
            }
            if graph_context:
                result["graph_context"] = graph_context
            return result
    finally:
        db.close()


def archive_memory(
    memory_ids: Optional[List[int]] = None,
    older_than_days: Optional[int] = None,
    max_access_count: Optional[int] = None,
    project: Optional[str] = None,
    memory_type: Optional[str] = None,
    centrality_protection: bool = True,
    min_centrality_threshold: int = 5,
    dry_run: bool = True,
    reason: Optional[str] = None,
    database: Optional[str] = None
) -> Dict[str, Any]:
    """
    Archive memories (soft delete). Replaces both old forget() and batch_archive_memories().

    Safety: dry_run=True by default. Returns preview of what would be archived.

    Args:
        memory_ids: List of memory IDs to archive (if provided, other filters ignored)
        older_than_days: Age filter (archive memories older than N days)
        max_access_count: Archive memories with access_count <= N
        project: Project filter
        memory_type: Type filter
        centrality_protection: If True, protect high-centrality memories from archival (default True)
        min_centrality_threshold: In-degree count for centrality protection (default 5)
        dry_run: Preview only (default True for safety)
        reason: Optional reason for archiving (added to source_context)

    Returns:
        Dict with archived_count, skipped_count (foundational), warnings (high centrality), details
    """
    db = get_session(database)
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
                cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=older_than_days)
                query = query.filter(Memory.created_at < cutoff_date)

            if max_access_count is not None:
                query = query.filter(Memory.access_count <= max_access_count)

            if project:
                if isinstance(project, list):
                    query = query.filter(_projects_overlap(project))
                else:
                    query = query.filter(_project_contains(project))

            if memory_type:
                query = query.filter(Memory.memory_type == memory_type)

            candidates = query.all()

        # Apply foundational protection and centrality protection
        to_archive = []
        skipped_foundational = []
        skipped_centrality = []
        warnings = []

        for memory in candidates:
            # Skip foundational memories
            if memory.foundational:
                skipped_foundational.append({
                    "id": memory.id,
                    "subject": memory.subject or "(no subject)",
                    "reason": "foundational memory - cannot archive"
                })
                continue

            # Check centrality
            in_degree = db.query(func.count(MemoryEdge.id)).filter(
                MemoryEdge.target_id == memory.id
            ).scalar() or 0

            # Apply centrality protection if enabled
            if centrality_protection and in_degree >= min_centrality_threshold:
                age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - memory.created_at).days
                skipped_centrality.append({
                    "id": memory.id,
                    "subject": memory.subject or "(no subject)",
                    "in_degree": in_degree,
                    "age_days": age_days,
                    "reason": f"protected by centrality (in-degree={in_degree} >= {min_centrality_threshold})"
                })
                continue

            # Warn if centrality is notable (but lower than protection threshold)
            if in_degree >= 3:
                age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - memory.created_at).days
                warnings.append({
                    "id": memory.id,
                    "subject": memory.subject or "(no subject)",
                    "in_degree": in_degree,
                    "age_days": age_days,
                    "warning": f"Notable centrality (in-degree={in_degree})"
                })

            age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - memory.created_at).days
            to_archive.append({
                "id": memory.id,
                "subject": memory.subject or "(no subject)",
                "type": memory.memory_type,
                "age_days": age_days,
                "access_count": memory.access_count,
                "in_degree": in_degree
            })

        # Execute archival if not dry run
        if not dry_run:
            archived_count = 0
            for mem_info in to_archive:
                memory = db.query(Memory).filter(Memory.id == mem_info["id"]).first()
                if memory:
                    memory.is_archived = True
                    if reason:
                        if memory.source_context:
                            memory.source_context = f"{memory.source_context}\n[ARCHIVED: {reason}]"
                        else:
                            memory.source_context = f"[ARCHIVED: {reason}]"
                    archived_count += 1

            db.commit()

            return {
                "archived_count": archived_count,
                "skipped_foundational_count": len(skipped_foundational),
                "skipped_centrality_count": len(skipped_centrality),
                "warning_count": len(warnings),
                "details": {
                    "archived": to_archive,
                    "skipped_foundational": skipped_foundational,
                    "skipped_centrality": skipped_centrality,
                    "warnings": warnings
                }
            }
        else:
            # Dry run - preview only
            return {
                "would_archive": len(to_archive),
                "would_skip_foundational": len(skipped_foundational),
                "would_skip_centrality": len(skipped_centrality),
                "warning_count": len(warnings),
                "details": {
                    "would_archive": to_archive,
                    "skipped_foundational": skipped_foundational,
                    "skipped_centrality": skipped_centrality,
                    "warnings": warnings
                },
                "note": "DRY RUN - no memories were archived. Set dry_run=False to execute."
            }
    finally:
        db.close()


# Legacy compatibility wrapper
def forget(
    memory_id: int,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Archive a memory (soft delete). Legacy wrapper around archive_memory().

    Args:
        memory_id: ID of the memory to archive
        reason: Optional reason for archiving

    Returns:
        Compact confirmation string
    """
    result = archive_memory(
        memory_ids=[memory_id],
        dry_run=False,
        reason=reason
    )

    if result.get("archived_count", 0) > 0:
        mem_info = result["details"]["archived"][0]
        subject_info = f" ({mem_info['subject']})" if mem_info['subject'] != "(no subject)" else ""
        return {"message": f"Archived memory {memory_id}{subject_info}"}
    elif result.get("skipped_foundational_count", 0) > 0:
        return {"error": f"Memory {memory_id} is foundational and cannot be archived"}
    elif result.get("skipped_centrality_count", 0) > 0:
        return {"error": f"Memory {memory_id} is protected by centrality and cannot be archived"}
    else:
        return {"error": f"Memory {memory_id} not found or already archived"}


def get_memory_stats(database: Optional[str] = None) -> Dict[str, Any]:
    """
    Get overview statistics of the memory system.

    Returns stats on total memories, by type, by instance, most accessed, etc.
    """
    db = get_session(database)
    try:
        # Total counts
        total = db.query(func.count(Memory.id)).scalar()
        total_active = db.query(func.count(Memory.id)).filter(Memory.is_archived == False).scalar()
        total_archived = db.query(func.count(Memory.id)).filter(Memory.is_archived == True).scalar()

        # By type
        by_type = {}
        type_counts = db.query(
            Memory.memory_type,
            func.count(Memory.id)
        ).filter(Memory.is_archived == False).group_by(Memory.memory_type).all()
        for memory_type, count in type_counts:
            by_type[memory_type] = count

        # By instance
        by_instance = {}
        instance_counts = db.query(
            Memory.instance_id,
            func.count(Memory.id)
        ).filter(Memory.is_archived == False).group_by(Memory.instance_id).all()
        for instance, count in instance_counts:
            by_instance[instance] = count

        # By project (explode arrays Python-side)
        by_project = {}
        active_memories = db.query(Memory).filter(Memory.is_archived == False).all()
        for memory in active_memories:
            for proj in (memory.projects or ["life"]):
                by_project[proj] = by_project.get(proj, 0) + 1

        # Most accessed (top 5)
        most_accessed = db.query(Memory).filter(
            Memory.is_archived == False
        ).order_by(Memory.access_count.desc()).limit(5).all()

        # Recently added (top 5)
        recent = db.query(Memory).filter(
            Memory.is_archived == False
        ).order_by(Memory.created_at.desc()).limit(5).all()

        # Foundational count
        foundational_count = db.query(func.count(Memory.id)).filter(
            Memory.is_archived == False,
            Memory.foundational == True
        ).scalar()

        # Trimmed response: most_accessed and recently_added as compact ID + subject pairs
        return {
            "total_memories": total,
            "active_memories": total_active,
            "archived_memories": total_archived,
            "foundational_memories": foundational_count,
            "by_type": by_type,
            "by_instance": by_instance,
            "by_project": by_project,
            "most_accessed": [f"#{m.id}: {m.subject or '(no subject)'}" for m in most_accessed],
            "recently_added": [f"#{m.id}: {m.subject or '(no subject)'}" for m in recent]
        }
    finally:
        db.close()


def backfill_embeddings(database: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate embeddings for all memories that don't have them.

    Useful for:
    - Backfilling existing memories after enabling embeddings
    - Retrying after Ollama was unavailable
    - Recovering from partial failures

    Returns:
        Dictionary with counts of success/failures
    """
    db = get_session(database)
    try:
        # Find all memories without embeddings (including archived for completeness)
        memories_without_embeddings = db.query(Memory).filter(
            Memory.embedding.is_(None)
        ).all()

        total = len(memories_without_embeddings)
        if total == 0:
            return {
                "success": True,
                "message": "All memories already have embeddings",
                "total": 0,
                "generated": 0,
                "failed": 0
            }

        generated = 0
        failed = 0
        failed_ids = []

        for memory in memories_without_embeddings:
            # Generate embedding using model's embedding_text method
            embedding_text = memory.embedding_text()
            embedding = get_embedding(embedding_text)

            if embedding:
                memory.embedding = embedding
                # Clear the embedding_failed tag if present
                if memory.tags and "embedding_failed" in memory.tags:
                    memory.tags = [t for t in memory.tags if t != "embedding_failed"]
                generated += 1
            else:
                failed += 1
                failed_ids.append(memory.id)

        db.commit()

        result = {
            "success": True,
            "message": f"Backfill complete: {generated}/{total} embeddings generated",
            "total": total,
            "generated": generated,
            "failed": failed
        }

        if failed_ids:
            result["failed_memory_ids"] = failed_ids[:20]  # Limit to first 20
            if failed > 20:
                result["note"] = f"Showing first 20 of {failed} failed IDs"

        return result
    finally:
        db.close()


def get_memory_by_id(
    memory_id: int,
    detail_level: str = "verbose",
    include_graph: bool = True,
    graph_depth: int = 0,
    traverse: bool = False,
    max_depth: int = 3,
    direction: Optional[str] = None,
    relation_types: Optional[List[str]] = None,
    min_strength: Optional[float] = None,
    graph_mode: str = "summary",
    database: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a single memory by ID with optional graph traversal.

    Absorbs functionality from graph_service.traverse_graph() and get_related_memories().

    Args:
        memory_id: ID of the memory to retrieve
        detail_level: "summary" for condensed, "verbose" for full content
        include_graph: Include graph context (default True)
        graph_depth: How many hops to follow in graph context (0 = no graph, 1-3 = context mode)
        traverse: If True, do BFS traversal instead of context mode
        max_depth: Max depth for BFS traverse (1-5, only used if traverse=True)
        direction: "outgoing", "incoming", or None for both (only used if traverse=True or include_graph=True)
        relation_types: Filter edges by type (only used if traverse=True or include_graph=True)
        min_strength: Filter edges by minimum strength (only used if traverse=True or include_graph=True)
        graph_mode: "summary" for per-node stats, "full" for raw edge list (default "summary")

    Returns:
        {"memory": dict, "graph_context": dict (optional)} if found, None if not found
        If graph_mode == "full": graph_context format: {"nodes": {id: subject}, "edges": [{source, target, type, strength}]}
        If graph_mode == "summary": graph_context format: {"nodes": {id: {subject, connections, avg_strength, edge_types}}, "total_edges": int, "seed_ids": list}
    """
    db = get_session(database)
    try:
        memory = db.query(Memory).filter(Memory.id == memory_id).first()
        if not memory:
            return None

        # Update access tracking
        memory.last_accessed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        memory.access_count += 1
        db.commit()

        # Build result with memory dict
        result = {"memory": memory.to_dict(detail_level=detail_level)}

        # Foundational memories are hubs — clamp to depth 1 to avoid graph explosion
        if memory.foundational and graph_depth > 1:
            graph_depth = 1

        # Fetch graph context if requested
        if include_graph and graph_depth > 0:
            if traverse:
                # BFS traversal mode
                from memory_palace.services.graph_service import traverse_graph
                traversal = traverse_graph(
                    start_id=memory_id,
                    max_depth=max_depth,
                    relation_types=relation_types,
                    direction=direction or "outgoing",
                    min_strength=min_strength or 0.0,
                    include_archived=False,
                    detail_level=detail_level
                )
                result["traversal"] = traversal
            else:
                # Context mode (adjacency list)
                graph_context = _get_graph_context_for_memories(
                    db, [memory],
                    max_depth=graph_depth,
                    direction=direction,
                    relation_types=relation_types,
                    min_strength=min_strength,
                    graph_mode=graph_mode
                )
                if graph_context:
                    result["graph_context"] = graph_context

        return result
    finally:
        db.close()


def get_memories_by_ids(
    memory_ids: List[int],
    detail_level: str = "verbose",
    synthesize: bool = False,
    include_graph: bool = True,
    graph_depth: int = 1,
    graph_mode: str = "summary",
    database: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get multiple memories by ID, with optional LLM synthesis.

    Args:
        memory_ids: List of memory IDs to retrieve
        detail_level: "summary" for condensed, "verbose" for full content (only applies when synthesize=False)
        synthesize: If True, use LLM to synthesize memories into natural language summary
        include_graph: Include graph context for ALL found memories (default True).
            NOTE: Unlike recall(), this fetches graph context for ALL found memories, not just
            a top-N subset. This is intentional - these are targeted fetches where the user
            wants full context for specific memories.
        graph_depth: How many hops to follow in graph context (1-3, default 1)
        graph_mode: "summary" for per-node stats, "full" for raw edge list (default "summary")

    Returns:
        If synthesize=False: {"memories": list[dict], "count": int, "not_found": list[int], "graph_context": dict (optional)}
        If synthesize=True: {"summary": str, "count": int, "memory_ids": list[int], "not_found": list[int], "graph_context": dict (optional)}
        If graph_mode == "full": graph_context format: {"nodes": {id: subject}, "edges": [{source, target, type, strength}]}
        If graph_mode == "summary": graph_context format: {"nodes": {id: {subject, connections, avg_strength, edge_types}}, "total_edges": int, "seed_ids": list}
    """
    db = get_session(database)
    try:
        # Fetch all memories in one query
        memories = db.query(Memory).filter(Memory.id.in_(memory_ids)).all()

        # Track which IDs were found
        found_ids = {m.id for m in memories}
        not_found = [mid for mid in memory_ids if mid not in found_ids]

        # Update access tracking for all found memories
        for memory in memories:
            memory.last_accessed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            memory.access_count += 1
        db.commit()

        # Fetch graph context if requested (for ALL found memories)
        graph_context = _get_graph_context_for_memories(db, memories, max_depth=graph_depth, graph_mode=graph_mode) if include_graph and memories else {}

        # Skip synthesis for single memory (pointless) or empty results
        if synthesize and len(memories) > 1:
            synthesis = _synthesize_memories_with_llm(memories)
            if synthesis:
                result = {
                    "summary": synthesis,
                    "count": len(memories),
                    "memory_ids": [m.id for m in memories]
                }
                if not_found:
                    result["not_found"] = not_found
                if graph_context:
                    result["graph_context"] = graph_context
                return result
            # Fall through to raw return if LLM unavailable

        # Return raw memory dicts
        result = {
            "memories": [m.to_dict(detail_level=detail_level) for m in memories],
            "count": len(memories)
        }
        if not_found:
            result["not_found"] = not_found
        if graph_context:
            result["graph_context"] = graph_context
        return result
    finally:
        db.close()


def get_recent_memories(
    limit: int = 20,
    verbose: bool = False,
    project: Optional[Union[str, List[str]]] = None,
    memory_type: Optional[str] = None,
    instance_id: Optional[str] = None,
    include_archived: bool = False,
    database: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the most recent memories, ordered by creation time descending.

    Default (verbose=False) returns title-card format: id, subject, memory_type,
    project, created_at. Verbose mode returns full to_dict() output.

    Args:
        limit: Number of memories to return (default 20, max 200)
        verbose: If False (default), return title-only compact list.
                 If True, return full memory details via to_dict(verbose).
        project: Filter by project — string for single, list for union (optional)
        memory_type: Filter by memory type, supports wildcards like "code_*" (optional)
        instance_id: Filter by instance (optional)
        include_archived: Include archived memories (default False)

    Returns:
        {"memories": list[dict], "count": int, "total_available": int}
    """
    limit = min(limit, 200)  # Cap at 200
    db = get_session(database)
    try:
        query = db.query(Memory)

        if not include_archived:
            query = query.filter(Memory.is_archived == False)

        if project is not None:
            if isinstance(project, list):
                query = query.filter(_projects_overlap(project))
            else:
                query = query.filter(_project_contains(project))

        if memory_type is not None:
            if "*" in memory_type:
                pattern = memory_type.replace("*", "%")
                query = query.filter(Memory.memory_type.like(pattern))
            else:
                query = query.filter(Memory.memory_type == memory_type)

        if instance_id is not None:
            query = query.filter(Memory.instance_id == instance_id)

        total_available = query.count()
        memories = query.order_by(Memory.created_at.desc()).limit(limit).all()

        if verbose:
            memory_list = [m.to_dict(detail_level="verbose") for m in memories]
        else:
            memory_list = [
                {
                    "id": m.id,
                    "subject": m.subject,
                    "memory_type": m.memory_type,
                    "project": m.projects,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "foundational": m.foundational,
                }
                for m in memories
            ]

        return {
            "memories": memory_list,
            "count": len(memory_list),
            "total_available": total_available,
        }
    finally:
        db.close()


def update_memory(
    memory_id: int,
    content: Optional[str] = None,
    subject: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    foundational: Optional[bool] = None,
    memory_type: Optional[str] = None,
    regenerate_embedding: bool = True,
    database: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update an existing memory's content or metadata.

    Args:
        memory_id: ID of the memory to update
        content: New content (if changing)
        subject: New subject (if changing)
        keywords: New keywords (if changing)
        foundational: New foundational status (if changing)
        memory_type: New type (if changing)
        regenerate_embedding: Whether to regenerate embedding after update (default True)

    Returns:
        Dict with success status and updated memory
    """
    db = get_session(database)
    try:
        memory = db.query(Memory).filter(Memory.id == memory_id).first()
        if not memory:
            return {"error": f"Memory {memory_id} not found"}

        # Track if content/type/subject changed (affects embedding)
        embedding_fields_changed = False

        if content is not None:
            memory.content = content
            embedding_fields_changed = True

        if subject is not None:
            memory.subject = subject
            embedding_fields_changed = True

        if memory_type is not None:
            memory.memory_type = memory_type
            embedding_fields_changed = True

        if keywords is not None:
            memory.keywords = keywords

        if foundational is not None:
            memory.foundational = foundational

        db.commit()

        # Regenerate embedding if content/subject/type changed
        embedding_status = None
        if regenerate_embedding and embedding_fields_changed:
            embedding_text = memory.embedding_text()
            embedding = get_embedding(embedding_text)
            if embedding:
                memory.embedding = embedding
                embedding_status = "regenerated"
            else:
                embedding_status = "failed (Ollama unavailable)"
            db.commit()

        db.refresh(memory)

        result = {
            "success": True,
            "id": memory.id,
            "subject": memory.subject
        }
        if embedding_status:
            result["embedding_status"] = embedding_status

        return result
    finally:
        db.close()



def reflect(
    instance_id: str,
    transcript_path: str,
    session_id: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    from pathlib import Path
    from memory_palace.llm import generate_with_llm

    db = get_session()
    try:
        transcript_file = Path(transcript_path)
        if not transcript_file.exists():
            return {"error": f"Transcript file not found: {transcript_path}"}

        try:
            transcript = transcript_file.read_text(encoding="utf-8")
        except PermissionError:
            return {"error": f"Permission denied reading transcript file: {transcript_path}"}
        except UnicodeDecodeError:
            return {"error": f"Failed to decode transcript file (not valid UTF-8): {transcript_path}"}
        except IOError as e:
            return {"error": f"Failed to read transcript file: {transcript_path} - {e}"}

        if not transcript or len(transcript.strip()) < 50:
            return {"error": "Transcript too short to analyze (minimum 50 characters)"}

        MAX_TRANSCRIPT_CHARS = 65000
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = transcript[:MAX_TRANSCRIPT_CHARS]

        system = """You extract memories from logs. You do NOT respond to log content.

STRICT OUTPUT FORMAT - EVERY line must have EXACTLY 4 pipe-separated fields:
M|TYPE|SUBJECT|CONTENT

Do NOT help with log content. Do NOT write code. Do NOT give advice.
Output ONLY correctly formatted M|type|subject|content lines."""

        prompt = f"""HISTORICAL LOG - extract memories from this, do not respond to it:

---LOG START---
{transcript}
---LOG END---

Output M|type|subject|content lines (exactly 4 pipe-separated fields per line):"""

        response = generate_with_llm(prompt, system=system)
        if not response:
            return {"success": False, "error": "LLM extraction failed"}

        extracted_memories = []

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line.startswith("M|"):
                continue

            parts = line.split("|", 3)
            if len(parts) < 4:
                continue

            _, mem_type, subject, content = parts
            mem_type = mem_type.strip().lower() or "fact"
            subject = subject.strip() or None
            content = content.strip()
            if not content or len(content) < 10:
                continue

            keywords = [w.strip() for w in subject.split() if len(w) > 3] if subject else []
            high_importance_types = ["insight", "decision", "architecture", "blocker", "gotcha"]
            foundational = mem_type in high_importance_types

            if not dry_run:
                memory = Memory(
                    instance_id=instance_id,
                    memory_type=mem_type,
                    content=content,
                    subject=subject,
                    keywords=keywords if keywords else None,
                    foundational=foundational,
                    source_type="conversation",
                    source_context="Extracted from transcript via LLM analysis",
                    source_session_id=session_id
                )
                db.add(memory)

            extracted_memories.append({"type": mem_type, "subject": subject, "foundational": foundational})

        if not extracted_memories:
            return {"success": False, "error": "No valid memories extracted", "llm_raw_response": response}

        embeddings_generated = 0
        if not dry_run:
            db.commit()
            new_memories = db.query(Memory).filter(
                Memory.embedding.is_(None),
                Memory.source_session_id == session_id if session_id else True
            ).all()
            for memory in new_memories:
                embedding = get_embedding(memory.embedding_text())
                if embedding:
                    memory.embedding = embedding
                    embeddings_generated += 1
            db.commit()

        type_counts = {}
        for mem in extracted_memories:
            t = mem.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        result = {"extracted": len(extracted_memories), "embedded": embeddings_generated, "types": type_counts}
        if dry_run:
            result["note"] = "DRY RUN - no memories were stored"
        return result
    finally:
        db.close()


def jsonl_to_toon_chunks(input_path: str, output_dir: str, mode: str = "aggressive", chunk_tokens: int = 12500) -> Dict[str, Any]:
    import sys
    from pathlib import Path

    tools_dir = Path(__file__).parent.parent.parent / "tools"
    sys.path.insert(0, str(tools_dir))

    try:
        from toon_converter import convert_jsonl_to_toon_chunks as do_convert
        return do_convert(input_path, output_dir, mode, chunk_tokens)
    except ImportError as e:
        return {"error": f"Failed to import converter: {e}"}
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Conversion failed: {e}"}
