"""
Memory service for Claude Memory Palace.

Provides functions for storing, recalling, archiving, and managing memories.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from sqlalchemy import func, or_, String

from memory_palace.models import Memory, MemoryEdge
from memory_palace.database import get_session
from memory_palace.embeddings import get_embedding, cosine_similarity
from memory_palace.config_v2 import get_auto_link_config
from memory_palace.llm import classify_edge_type, classify_edge_types_batch


# Valid source types for memories
VALID_SOURCE_TYPES = ["conversation", "explicit", "inferred", "observation"]


def _find_similar_memories(
    db,
    embedding: List[float],
    exclude_id: int,
    project: Optional[str] = None,
    threshold: float = 0.50,
) -> List[Tuple[int, float]]:
    """
    Find memories similar to the given embedding.
    
    Args:
        db: Database session
        embedding: The embedding vector to compare against
        exclude_id: Memory ID to exclude (the new memory itself)
        project: If set, only match memories in this project
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
    
    if project:
        query = query.filter(Memory.project == project)
    
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
    importance: int = 5,
    project: str = "life",
    source_type: str = "explicit",
    source_context: Optional[str] = None,
    source_session_id: Optional[str] = None,
    supersedes_id: Optional[int] = None,
    auto_link: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Store a new memory in the memory palace.

    Args:
        instance_id: Which instance is storing this (e.g., "desktop", "code", "web")
        memory_type: Type of memory (open-ended - use existing types or create new ones)
        content: The actual memory content
        subject: What/who this memory is about (optional but recommended)
        keywords: List of keywords for searchability
        tags: Freeform organizational tags (separate from keywords)
        importance: 1-10, higher = more important (default 5)
        project: Project this memory belongs to (default "life" for non-project memories)
        source_type: How this memory was created (conversation, explicit, inferred, observation)
        source_context: Snippet of original context
        source_session_id: Link back to conversation session
        supersedes_id: If set, create a 'supersedes' edge to this memory ID and archive it
        auto_link: Override config to enable/disable auto-linking (None = use config)

    Returns:
        Dict with id, subject, embedded status, and links_created (if any)
    """
    db = get_session()
    try:
        # memory_type is open-ended - use existing types when they fit, create new ones when needed
        # Types are included in semantic vector calculation
        if source_type not in VALID_SOURCE_TYPES:
            return {"error": f"Invalid source_type. Must be one of: {VALID_SOURCE_TYPES}"}

        # Clamp importance to valid range
        importance = max(1, min(10, importance))

        memory = Memory(
            instance_id=instance_id,
            project=project,
            memory_type=memory_type,
            content=content,
            subject=subject,
            keywords=keywords,
            tags=tags,
            importance=importance,
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
                    edge_metadata={"superseded_at": datetime.utcnow().isoformat()},
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
            similar_project = project if auto_link_config["same_project_only"] else None
            similar = _find_similar_memories(
                db,
                embedding,
                memory.id,
                project=similar_project,
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
    project: Optional[str] = None,
    memory_type: Optional[str] = None,
    subject: Optional[str] = None,
    min_importance: Optional[int] = None,
    include_archived: bool = False,
    limit: int = 20,
    detail_level: str = "summary",
    synthesize: bool = True
) -> Dict[str, Any]:
    """
    Search memories using semantic search (with keyword fallback).

    Args:
        query: Search query (used for semantic similarity or keyword matching)
        instance_id: Filter by instance (optional)
        project: Filter by project (optional, e.g., "memory-palace", "wordleap", "life")
        memory_type: Filter by type (optional)
        subject: Filter by subject (optional)
        min_importance: Only return memories with importance >= this
        include_archived: Include archived memories (default False)
        limit: Maximum memories to return (default 20)
        detail_level: "summary" for condensed, "verbose" for full content (only applies when synthesize=True)
        synthesize: If True (default), use LLM to synthesize. If False, return raw memory objects with full content.

    Returns:
        Dictionary with one of three formats:
        - synthesize=True + LLM available: {"summary": str, "count": int, "search_method": str, "memory_ids": list}
        - synthesize=True + LLM unavailable: {"summary": str (text list), "count": int, "search_method": str, "memory_ids": list}
        - synthesize=False: {"memories": list[dict], "count": int, "search_method": str}
          Note: Raw mode always returns verbose content regardless of detail_level parameter.
    """
    db = get_session()
    try:
        # Build base query with filters (no keyword search yet)
        base_query = db.query(Memory)

        # Filter out archived unless requested
        if not include_archived:
            base_query = base_query.filter(Memory.is_archived == False)

        # Filter by instance if specified
        if instance_id:
            base_query = base_query.filter(Memory.instance_id == instance_id)

        # Filter by project if specified
        if project:
            base_query = base_query.filter(Memory.project == project)

        if memory_type:
            base_query = base_query.filter(Memory.memory_type == memory_type)

        if subject:
            base_query = base_query.filter(Memory.subject.ilike(f"%{subject}%"))

        if min_importance:
            base_query = base_query.filter(Memory.importance >= min_importance)

        # Try semantic search first
        search_method = "semantic"

        # Format query for SFR-Embedding-Mistral (instruction format)
        formatted_query = f"Instruct: Given a memory search query, retrieve relevant memories.\nQuery: {query}"
        query_embedding = get_embedding(formatted_query)

        if query_embedding:
            # Semantic search: fetch all matching memories and rank by similarity
            all_memories = base_query.all()

            # Score each memory by cosine similarity
            scored_memories = []
            for memory in all_memories:
                if memory.embedding is not None:
                    similarity = cosine_similarity(query_embedding, memory.embedding)
                    scored_memories.append((memory, similarity))
                else:
                    # No embedding - give a low similarity score so it appears at the end
                    scored_memories.append((memory, -1.0))

            # Sort by similarity (highest first)
            scored_memories.sort(key=lambda x: x[1], reverse=True)

            # Take top N
            memories = [m for m, score in scored_memories[:limit]]
            similarity_scores = {m.id: score for m, score in scored_memories[:limit]}
        else:
            # Fallback to keyword search (improved: AND together all words)
            search_method = "keyword (fallback)"

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

            # Order by importance DESC, then access_count DESC, then created_at DESC
            base_query = base_query.order_by(
                Memory.importance.desc(),
                Memory.access_count.desc(),
                Memory.created_at.desc()
            ).limit(limit)

            memories = base_query.all()
            similarity_scores = {}

        # Update access tracking for retrieved memories
        for memory in memories:
            memory.last_accessed_at = datetime.utcnow()
            memory.access_count += 1
        db.commit()

        # Return raw memories if synthesize=False
        # Force verbose detail when returning raw - cloud AI needs full content
        if not synthesize:
            result_memories = []
            for m in memories:
                mem_dict = m.to_dict(detail_level="verbose")
                if m.id in similarity_scores:
                    mem_dict["similarity_score"] = round(similarity_scores[m.id], 4)
                result_memories.append(mem_dict)
            return {
                "memories": result_memories,
                "count": len(memories),
                "search_method": search_method
            }

        # Try LLM synthesis first, fall back to text list
        # Pass similarity scores if we have them (semantic search only)
        synthesis = _synthesize_memories_with_llm(
            memories,
            query,
            similarity_scores if similarity_scores else None
        )

        if synthesis:
            # LLM synthesis available - return natural language response
            return {
                "summary": synthesis,
                "count": len(memories),
                "search_method": search_method,
                "memory_ids": [m.id for m in memories]
            }
        else:
            # Fallback to simple text list
            text_list = _format_memories_as_text(memories)
            return {
                "summary": text_list,
                "count": len(memories),
                "search_method": search_method + " (no LLM)",
                "memory_ids": [m.id for m in memories]
            }
    finally:
        db.close()


def forget(
    memory_id: int,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Archive a memory (soft delete).

    Args:
        memory_id: ID of the memory to archive
        reason: Optional reason for archiving

    Returns:
        Compact confirmation string
    """
    db = get_session()
    try:
        memory = db.query(Memory).filter(Memory.id == memory_id).first()
        if not memory:
            return {"error": f"Memory {memory_id} not found"}

        subject_info = f" ({memory.subject})" if memory.subject else ""
        memory.is_archived = True
        if reason and memory.source_context:
            memory.source_context = f"{memory.source_context}\n[ARCHIVED: {reason}]"
        elif reason:
            memory.source_context = f"[ARCHIVED: {reason}]"

        db.commit()

        # Compact response
        return {"message": f"Archived memory {memory_id}{subject_info}"}
    finally:
        db.close()


def get_memory_stats() -> Dict[str, Any]:
    """
    Get overview statistics of the memory system.

    Returns stats on total memories, by type, by instance, most accessed, etc.
    """
    db = get_session()
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

        # By project
        by_project = {}
        project_counts = db.query(
            Memory.project,
            func.count(Memory.id)
        ).filter(Memory.is_archived == False).group_by(Memory.project).all()
        for project, count in project_counts:
            by_project[project or "life"] = count

        # Most accessed (top 5)
        most_accessed = db.query(Memory).filter(
            Memory.is_archived == False
        ).order_by(Memory.access_count.desc()).limit(5).all()

        # Recently added (top 5)
        recent = db.query(Memory).filter(
            Memory.is_archived == False
        ).order_by(Memory.created_at.desc()).limit(5).all()

        # Average importance
        avg_importance = db.query(func.avg(Memory.importance)).filter(
            Memory.is_archived == False
        ).scalar()

        # Trimmed response: most_accessed and recently_added as compact ID + subject pairs
        return {
            "total_memories": total,
            "active_memories": total_active,
            "archived_memories": total_archived,
            "by_type": by_type,
            "by_instance": by_instance,
            "by_project": by_project,
            "average_importance": round(avg_importance, 2) if avg_importance else 0,
            "most_accessed": [f"#{m.id}: {m.subject or '(no subject)'}" for m in most_accessed],
            "recently_added": [f"#{m.id}: {m.subject or '(no subject)'}" for m in recent]
        }
    finally:
        db.close()


def backfill_embeddings() -> Dict[str, Any]:
    """
    Generate embeddings for all memories that don't have them.

    Useful for:
    - Backfilling existing memories after enabling embeddings
    - Retrying after Ollama was unavailable
    - Recovering from partial failures

    Returns:
        Dictionary with counts of success/failures
    """
    db = get_session()
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


def get_memory_by_id(memory_id: int, detail_level: str = "verbose") -> Optional[Dict[str, Any]]:
    """
    Get a single memory by ID.

    Args:
        memory_id: ID of the memory to retrieve
        detail_level: "summary" for condensed, "verbose" for full content

    Returns:
        Memory dict if found, None if not found
    """
    db = get_session()
    try:
        memory = db.query(Memory).filter(Memory.id == memory_id).first()
        if not memory:
            return None

        # Update access tracking
        memory.last_accessed_at = datetime.utcnow()
        memory.access_count += 1
        db.commit()

        return memory.to_dict(detail_level=detail_level)
    finally:
        db.close()


def get_memories_by_ids(
    memory_ids: List[int],
    detail_level: str = "verbose",
    synthesize: bool = False
) -> Dict[str, Any]:
    """
    Get multiple memories by ID, with optional LLM synthesis.

    Args:
        memory_ids: List of memory IDs to retrieve
        detail_level: "summary" for condensed, "verbose" for full content (only applies when synthesize=False)
        synthesize: If True, use LLM to synthesize memories into natural language summary

    Returns:
        If synthesize=False: {"memories": list[dict], "count": int, "not_found": list[int]}
        If synthesize=True: {"summary": str, "count": int, "memory_ids": list[int], "not_found": list[int]}
    """
    db = get_session()
    try:
        # Fetch all memories in one query
        memories = db.query(Memory).filter(Memory.id.in_(memory_ids)).all()
        
        # Track which IDs were found
        found_ids = {m.id for m in memories}
        not_found = [mid for mid in memory_ids if mid not in found_ids]
        
        # Update access tracking for all found memories
        for memory in memories:
            memory.last_accessed_at = datetime.utcnow()
            memory.access_count += 1
        db.commit()

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
                return result
            # Fall through to raw return if LLM unavailable

        # Return raw memory dicts
        result = {
            "memories": [m.to_dict(detail_level=detail_level) for m in memories],
            "count": len(memories)
        }
        if not_found:
            result["not_found"] = not_found
        return result
    finally:
        db.close()


def update_memory(
    memory_id: int,
    content: Optional[str] = None,
    subject: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    importance: Optional[int] = None,
    memory_type: Optional[str] = None,
    regenerate_embedding: bool = True
) -> Dict[str, Any]:
    """
    Update an existing memory's content or metadata.

    Args:
        memory_id: ID of the memory to update
        content: New content (if changing)
        subject: New subject (if changing)
        keywords: New keywords (if changing)
        importance: New importance (if changing)
        memory_type: New type (if changing)
        regenerate_embedding: Whether to regenerate embedding after update (default True)

    Returns:
        Dict with success status and updated memory
    """
    db = get_session()
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

        if importance is not None:
            memory.importance = max(1, min(10, importance))

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
            importance = 7 if mem_type in high_importance_types else 5

            if not dry_run:
                memory = Memory(
                    instance_id=instance_id,
                    memory_type=mem_type,
                    content=content,
                    subject=subject,
                    keywords=keywords if keywords else None,
                    importance=importance,
                    source_type="conversation",
                    source_context="Extracted from transcript via LLM analysis",
                    source_session_id=session_id
                )
                db.add(memory)

            extracted_memories.append({"type": mem_type, "subject": subject, "importance": importance})

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
