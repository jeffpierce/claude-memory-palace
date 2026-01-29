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
from memory_palace.services.graph_service import link_memories, get_related_memories
from memory_palace.llm import generate_with_llm, is_llm_available

# Add tools directory to path for transpiler import
_tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(_tools_dir))

from code_transpiler import transpile_code_to_prose, transpile_file


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
        # Normalize path for consistent matching
        code_path = str(Path(code_path).resolve())
        
        # Check if already indexed (unless force=True)
        if not force:
            existing = db.query(Memory).filter(
                Memory.project == project,
                Memory.memory_type == "code_description",
                Memory.source_context.contains(code_path)
            ).first()
            
            if existing:
                # Find the linked code memory
                related = get_related_memories(
                    existing.id,
                    relation_type="source_of",
                    direction="outgoing"
                )
                code_id = related["memories"][0]["id"] if related.get("memories") else None
                
                return {
                    "already_indexed": True,
                    "prose_id": existing.id,
                    "code_id": code_id,
                    "subject": existing.subject,
                    "message": f"File already indexed. Use force=True to re-index."
                }
        
        # Read and transpile the file
        transpile_result = transpile_file(code_path)
        
        if "error" in transpile_result:
            return {"error": f"Transpilation failed: {transpile_result['error']}"}
        
        # Read raw code content
        try:
            code_content = Path(code_path).read_text(encoding="utf-8")
        except Exception as e:
            return {"error": f"Failed to read code file: {e}"}
        
        # Build keywords from transpiler output
        keywords = [code_path, transpile_result["language"]]
        keywords.extend(transpile_result.get("dependencies", []))
        keywords.extend(transpile_result.get("patterns", []))
        # Filter out empty strings and deduplicate
        keywords = list(set(k for k in keywords if k))
        
        # Store prose description via remember() - gets embedding
        prose_result = remember(
            instance_id=instance_id,
            memory_type="code_description",
            content=transpile_result["prose"],
            subject=transpile_result["subject"],
            keywords=keywords,
            importance=5,
            project=project,
            source_type="explicit",
            source_context=f"Indexed from: {code_path}"
        )
        
        if "error" in prose_result:
            return {"error": f"Failed to store prose description: {prose_result['error']}"}
        
        prose_id = prose_result["id"]
        
        # Store raw code via DIRECT Memory creation (no embedding)
        file_name = Path(code_path).name
        code_memory = Memory(
            instance_id=instance_id,
            project=project,
            memory_type="code",
            content=code_content,
            subject=f"Source: {file_name}",
            keywords=[code_path, file_name],
            importance=5,
            source_type="explicit",
            source_context=f"Raw code from: {code_path}"
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
        
        return {
            "prose_id": prose_id,
            "code_id": code_id,
            "subject": transpile_result["subject"],
            "language": transpile_result["language"],
            "patterns": transpile_result.get("patterns", [])
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
