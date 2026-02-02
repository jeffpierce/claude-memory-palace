"""
Code service for Claude Memory Palace.

Provides functions for indexing and retrieving source code via natural language.
Uses a two-layer architecture:
- Prose descriptions (embedded for semantic search)
- Raw code content (stored but NOT embedded)
- Connected by source_of edges

The prose layer acts as the semantic index; code is retrieved via graph traversal.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from memory_palace.models import Memory, MemoryEdge
from memory_palace.database import get_session
from memory_palace.services.memory_service import remember, recall
from memory_palace.services.graph_service import link_memories, get_related_memories, supersede_memory
from memory_palace.llm import generate_with_llm, is_llm_available

# Add tools directory to path for transpiler import
_tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(_tools_dir))

from code_transpiler import transpile_code_to_prose, transpile_file
import re


def _normalize_path(path: str) -> str:
    """
    Normalize a file path to a canonical form for cross-platform matching.

    Handles:
    - WSL paths: /mnt/c/Users/... → C:/Users/...
    - Windows backslashes: C:\\Users\\... → C:/Users/...
    - Drive letter case: c:/... → C:/...

    Returns a canonical path with forward slashes and uppercase drive letter.
    """
    path = str(path)

    # Convert WSL /mnt/X/ paths to X:/
    wsl_match = re.match(r'^/mnt/([a-zA-Z])/(.*)', path)
    if wsl_match:
        drive = wsl_match.group(1).upper()
        rest = wsl_match.group(2)
        path = f"{drive}:/{rest}"

    # Convert backslashes to forward slashes
    path = path.replace('\\', '/')

    # Uppercase drive letter if present (C:/ not c:/)
    if len(path) >= 2 and path[1] == ':':
        path = path[0].upper() + path[1:]

    return path


def _get_path_variants(normalized_path: str) -> List[str]:
    """
    Generate path variants for matching against stored paths.

    Given a normalized path like C:/Users/jeffr/..., generates:
    - C:/Users/jeffr/... (normalized)
    - /mnt/c/Users/jeffr/... (WSL)
    - C:\\Users\\jeffr\\... (Windows backslash)
    """
    variants = [normalized_path]

    # If it's a Windows-style path, generate WSL variant
    if len(normalized_path) >= 2 and normalized_path[1] == ':':
        drive = normalized_path[0].lower()
        rest = normalized_path[3:]  # Skip "C:/"
        wsl_path = f"/mnt/{drive}/{rest}"
        variants.append(wsl_path)

    # Add backslash variant
    variants.append(normalized_path.replace('/', '\\'))

    return variants


def code_remember(
    code_path: str,
    project: str,
    instance_id: str,
    force: bool = False
) -> Dict[str, Any]:
    """
    Index a source file into the memory palace.
    
    Creates two memories:
    - A prose description (memory_type: code_description) - embedded for semantic search
    - The raw code content (memory_type: code) - stored but NOT embedded
    
    Connected by a source_of edge (prose -> code).
    
    Args:
        code_path: Path to the source file to index
        project: Project this code belongs to (e.g., "memory-palace")
        instance_id: Which instance is indexing (e.g., "code", "desktop")
        force: Re-index even if already indexed (default False)
    
    Returns:
        Dict with prose_id, code_id, subject on success, or error dict
    """
    db = get_session()
    try:
        # Normalize path for consistent cross-platform matching
        code_path = str(Path(code_path).resolve())
        canonical_path = _normalize_path(code_path)
        path_variants = _get_path_variants(canonical_path)

        # Check for existing index using any path variant (handles WSL vs Windows)
        from sqlalchemy import or_
        existing_prose = db.query(Memory).filter(
            Memory.project == project,
            Memory.memory_type == "code_description",
            Memory.is_archived == False,
            or_(*[Memory.source_context.contains(variant) for variant in path_variants])
        ).first()

        existing_code_id = None
        if existing_prose:
            # Find the linked code memory
            related = get_related_memories(
                existing_prose.id,
                relation_type="source_of",
                direction="outgoing"
            )
            existing_code_id = related["memories"][0]["id"] if related.get("memories") else None

            if not force:
                # Already indexed and not forcing re-index
                return {
                    "already_indexed": True,
                    "prose_id": existing_prose.id,
                    "code_id": existing_code_id,
                    "subject": existing_prose.subject,
                    "message": "File already indexed. Use force=True to re-index."
                }
            # If force=True, continue to re-index (old IDs captured for supersession)
        
        # Read and transpile the file
        transpile_result = transpile_file(code_path)
        
        if "error" in transpile_result:
            return {"error": f"Transpilation failed: {transpile_result['error']}"}
        
        # Read raw code content
        try:
            code_content = Path(code_path).read_text(encoding="utf-8")
        except Exception as e:
            return {"error": f"Failed to read code file: {e}"}
        
        # Build keywords from transpiler output (use canonical path for consistency)
        keywords = [canonical_path, transpile_result["language"]]
        keywords.extend(transpile_result.get("dependencies", []))
        keywords.extend(transpile_result.get("patterns", []))
        # Filter out empty strings and deduplicate
        keywords = list(set(k for k in keywords if k))

        # Store prose description via remember() - gets embedding
        # Use canonical_path in source_context for cross-platform matching
        prose_result = remember(
            instance_id=instance_id,
            memory_type="code_description",
            content=transpile_result["prose"],
            subject=transpile_result["subject"],
            keywords=keywords,
            importance=5,
            project=project,
            source_type="explicit",
            source_context=f"Indexed from: {canonical_path}"
        )
        
        if "error" in prose_result:
            return {"error": f"Failed to store prose description: {prose_result['error']}"}
        
        prose_id = prose_result["id"]
        
        # Store raw code via DIRECT Memory creation (no embedding)
        # Use canonical_path for cross-platform consistency
        file_name = Path(code_path).name
        code_memory = Memory(
            instance_id=instance_id,
            project=project,
            memory_type="code",
            content=code_content,
            subject=f"Source: {file_name}",
            keywords=[canonical_path, file_name],
            importance=5,
            source_type="explicit",
            source_context=f"Raw code from: {canonical_path}"
            # NOTE: No embedding field set - this memory is NOT embedded
        )
        db.add(code_memory)
        db.commit()
        db.refresh(code_memory)
        
        code_id = code_memory.id
        
        # Create source_of edge: prose -> code
        link_result = link_memories(
            source_id=prose_id,
            target_id=code_id,
            relation_type="source_of",
            strength=1.0,
            bidirectional=False,
            created_by=instance_id
        )
        
        if "error" in link_result:
            # Edge creation failed, but memories exist - log but don't fail
            return {
                "prose_id": prose_id,
                "code_id": code_id,
                "subject": transpile_result["subject"],
                "warning": f"Memories created but edge failed: {link_result['error']}"
            }

        # Supersede old pair if this was a re-index (force=True with existing)
        superseded_prose_id = None
        if existing_prose:
            supersede_memory(prose_id, existing_prose.id, archive_old=True, created_by=instance_id)
            superseded_prose_id = existing_prose.id
            if existing_code_id:
                supersede_memory(code_id, existing_code_id, archive_old=True, created_by=instance_id)

        return {
            "prose_id": prose_id,
            "code_id": code_id,
            "subject": transpile_result["subject"],
            "language": transpile_result["language"],
            "patterns": transpile_result.get("patterns", []),
            "superseded_prose_id": superseded_prose_id  # None if fresh index
        }
        
    finally:
        db.close()


def code_recall(
    query: str,
    project: Optional[str] = None,
    synthesize: bool = True,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search indexed code using natural language.
    
    Finds relevant code by:
    1. Semantic search on prose descriptions (memory_type: code_description)
    2. Following source_of edges to retrieve actual code
    3. Optionally synthesizing an answer using the code context
    
    Args:
        query: Natural language search query (e.g., "how do embeddings work")
        project: Filter by project (optional)
        synthesize: If True, LLM answers using code context. If False, return raw matches.
        limit: Maximum number of code files to return (default 5)
    
    Returns:
        If synthesize=True: {answer: str, sources: list, count: int}
        If synthesize=False: {matches: list of {prose, code} dicts, count: int}
    """
    # Search prose descriptions (semantic search on embedded content)
    recall_result = recall(
        query=query,
        memory_type="code_description",
        project=project,
        synthesize=False,  # We want raw memories to traverse
        limit=limit
    )
    
    if "error" in recall_result:
        return {"error": f"Recall failed: {recall_result['error']}"}
    
    prose_memories = recall_result.get("memories", [])
    
    if not prose_memories:
        return {
            "matches" if not synthesize else "answer": [] if not synthesize else "No indexed code found matching your query.",
            "count": 0,
            "search_method": recall_result.get("search_method", "unknown")
        }
    
    # For each prose memory, retrieve the linked code via source_of edge
    matches = []
    for prose in prose_memories:
        prose_id = prose["id"]
        
        # Get related code memory
        related = get_related_memories(
            prose_id,
            relation_type="source_of",
            direction="outgoing"
        )
        
        code_memories = related.get("memories", [])
        code = code_memories[0] if code_memories else None
        
        matches.append({
            "prose": prose,
            "code": code,
            "similarity": prose.get("similarity_score", None)
        })
    
    # If not synthesizing, return raw matches
    if not synthesize:
        return {
            "matches": matches,
            "count": len(matches),
            "search_method": recall_result.get("search_method", "unknown")
        }
    
    # Synthesize answer using LLM
    if not is_llm_available():
        # Fallback: return structured data without synthesis
        return {
            "answer": "LLM unavailable for synthesis. Returning raw matches.",
            "sources": [
                {
                    "file": m["prose"].get("source_context", "").replace("Indexed from: ", ""),
                    "subject": m["prose"].get("subject"),
                    "relevance": m.get("similarity")
                }
                for m in matches
            ],
            "count": len(matches),
            "llm_available": False
        }
    
    # Build context for synthesis
    code_blocks = []
    for i, match in enumerate(matches, 1):
        prose = match["prose"]
        code = match["code"]
        
        source_file = prose.get("source_context", "").replace("Indexed from: ", "")
        similarity = match.get("similarity")
        sim_str = f" (relevance: {similarity:.2f})" if similarity else ""
        
        block = f"### File {i}: {source_file}{sim_str}\n\n"
        block += f"**Description:** {prose.get('content', '(no description)')[:500]}\n\n"
        
        if code:
            code_content = code.get("content", "(code not found)")
            # Truncate very long code for synthesis context
            if len(code_content) > 8000:
                code_content = code_content[:8000] + "\n... (truncated)"
            block += f"```\n{code_content}\n```"
        else:
            block += "(code content not found)"
        
        code_blocks.append(block)
    
    context = "\n\n---\n\n".join(code_blocks)
    
    system = """You are a code expert answering questions about a codebase.
Answer using ONLY the code and descriptions provided.
Be specific - reference function names, line numbers, patterns.
If the code doesn't answer the question well, say so.
Keep your answer focused and technical."""

    prompt = f"""Question: {query}

Here are the most relevant code files:

{context}

Answer the question based on this code:"""

    answer = generate_with_llm(prompt, system=system)
    
    if not answer:
        return {
            "answer": "LLM synthesis failed. See sources for raw code.",
            "sources": [
                {
                    "file": m["prose"].get("source_context", "").replace("Indexed from: ", ""),
                    "subject": m["prose"].get("subject"),
                    "relevance": m.get("similarity")
                }
                for m in matches
            ],
            "count": len(matches),
            "synthesis_failed": True
        }
    
    return {
        "answer": answer,
        "sources": [
            {
                "file": m["prose"].get("source_context", "").replace("Indexed from: ", ""),
                "subject": m["prose"].get("subject"),
                "relevance": m.get("similarity")
            }
            for m in matches
        ],
        "count": len(matches),
        "search_method": recall_result.get("search_method", "unknown")
    }
