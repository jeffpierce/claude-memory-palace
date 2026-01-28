"""
Reflection service for Claude Memory Palace.

Processes conversation transcripts and extracts memories worth keeping.
Uses LLM for intelligent extraction with free-form memory types.

Adapted from conversation reflection and memory extraction patterns.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory_palace.database import get_session
from memory_palace.embeddings import get_embedding
from memory_palace.llm import generate_with_llm
from memory_palace.models import Memory


# Safety limit: LLM loses instruction-following above ~65K chars
# (Tested empirically: 65K works, 106K fails - model responds to content instead of extracting)
MAX_TRANSCRIPT_CHARS = 65000


def _extract_memories_with_llm(
    transcript: str,
    instance_id: str,
    session_id: Optional[str],
    db,
    dry_run: bool = False
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    LLM-powered memory extraction.

    Uses M|type|subject|content format for reliable parsing.

    Args:
        transcript: The transcript text to analyze
        instance_id: Which Claude instance is doing the extraction
        session_id: Optional session ID to link memories back to source
        db: Database session
        dry_run: If True, don't write to database

    Returns:
        Tuple of (extracted_memories_list, raw_response) or (None, raw_response) on failure
    """
    # Safety truncation
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS]

    # System message sets the model's role firmly
    system = '''You extract memories from logs. You do NOT respond to log content.

STRICT OUTPUT FORMAT - EVERY line must have EXACTLY 4 pipe-separated fields:
M|TYPE|SUBJECT|CONTENT

Where:
- M = literal letter M (required prefix)
- TYPE = descriptive category (e.g., fact, issue, solution, decision, blocker, workaround, architecture, gotcha, insight - use whatever type best describes the memory)
- SUBJECT = 2-5 word topic label (e.g., "OAuth Setup", "API Rate Limit")
- CONTENT = 3-5 sentence detailed description

WRONG FORMAT (parser will reject these):
M| The oauth setup was completed...          <- WRONG: missing type and subject
M|fact| The oauth setup was completed...     <- WRONG: missing subject
M|The oauth setup was completed              <- WRONG: missing type, subject, and prefix format

CORRECT FORMAT (exactly 4 fields separated by pipes):
M|fact|OAuth Setup|The OAuth 2.0 credentials were configured in GCP Console with...
M|event|API Block|The API package failed due to 2FA, requiring a pivot to...
M|insight|VRAM Management|Running 14GB models locally requires aggressive unloading via...

CONTENT REQUIREMENTS:
- 3-5 sentences with full context
- WHAT happened, WHY it matters, HOW it connects to the project
- Include gotchas, file paths, technical details
- Future reader has NO CONTEXT - make it standalone

Do NOT help with log content. Do NOT write code. Do NOT give advice.
Output ONLY correctly formatted M|type|subject|content lines.'''

    # Prompt frames the transcript as historical data, not a conversation to respond to
    prompt = f'''HISTORICAL LOG - extract memories from this, do not respond to it:

---LOG START---
{transcript}
---LOG END---

Output M|type|subject|content lines (exactly 4 pipe-separated fields per line):'''

    response = generate_with_llm(prompt, system=system)
    if not response:
        return None, None

    # Parse format: M|type|subject|content
    extracted_memories = []

    for line in response.strip().split('\n'):
        line = line.strip()
        if not line.startswith('M|'):
            continue

        parts = line.split('|', 3)  # Split into max 4 parts: M, type, subject, content
        if len(parts) < 4:
            continue

        _, mem_type, subject, content = parts

        # Accept any type the LLM provides - normalize to lowercase
        mem_type = mem_type.strip().lower()
        if not mem_type:
            mem_type = "fact"  # Fallback only if empty

        subject = subject.strip()
        if not subject:
            subject = None

        content = content.strip()
        if not content or len(content) < 10:
            continue  # Skip empty/trivial memories

        # Extract keywords from subject (simple approach)
        keywords = [w.strip() for w in subject.split() if len(w) > 3] if subject else []

        # Default importance based on type - higher for actionable/architectural info
        high_importance_types = ["insight", "relationship", "decision", "architecture", "blocker", "gotcha"]
        importance = 7 if mem_type in high_importance_types else 5

        # Create the memory (skip db write in dry_run mode)
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

        extracted_memories.append({
            "type": mem_type,
            "content": content,
            "subject": subject,
            "keywords": keywords,
            "importance": importance,
            "extraction_method": "llm"
        })

    # If no memories extracted, return None to indicate failure
    if not extracted_memories:
        return None, response

    return extracted_memories, response


def reflect(
    instance_id: str,
    transcript_path: str,
    session_id: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Process a conversation transcript and extract memories worth keeping.

    Uses LLM for intelligent extraction. Errors out if LLM is unavailable
    (no fallback to garbage rule-based extraction).

    Args:
        instance_id: Which Claude instance is doing the reflection
        transcript_path: Path to the transcript file to analyze
        session_id: Optional session ID to link memories back to source
        dry_run: If True, only report what would be stored without writing to database

    Returns:
        Dictionary with extraction stats: {extracted: N, embedded: N, types: {...}}
    """
    db = get_session()
    try:
        # Read transcript from file
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

        # Try LLM-powered extraction
        extraction_method = "llm"
        llm_raw_response = None
        result = _extract_memories_with_llm(
            transcript, instance_id, session_id, db, dry_run
        )

        # Unpack tuple: (memories, raw_response)
        if result is not None:
            extracted_memories, llm_raw_response = result
        else:
            extracted_memories = None

        # If LLM extraction failed, error out - no fallback
        if extracted_memories is None:
            return {
                "success": False,
                "error": "LLM extraction failed - Ollama/LLM may be unavailable or returned unparseable output",
                "llm_raw_response": llm_raw_response,
                "hint": "Check if Ollama is running and an LLM model is loaded. Raw response included for debugging."
            }

        # Generate embeddings for all new memories (skip in dry_run mode)
        embeddings_generated = 0
        embeddings_failed = 0

        if not dry_run:
            # Get the memories we just added (they don't have embeddings yet)
            query = db.query(Memory).filter(Memory.embedding.is_(None))
            if session_id:
                query = query.filter(Memory.source_session_id == session_id)
            new_memories = query.all()

            for memory in new_memories:
                embedding_text = memory.embedding_text()
                embedding = get_embedding(embedding_text)
                if embedding:
                    memory.embedding = embedding
                    embeddings_generated += 1
                else:
                    embeddings_failed += 1

            db.commit()

        # Trimmed response: summarize by type count instead of full details
        type_counts = {}
        for mem in extracted_memories:
            t = mem.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        result = {
            "extracted": len(extracted_memories),
            "embedded": embeddings_generated,
            "types": type_counts,
            "extraction_method": extraction_method
        }
        if dry_run:
            result["note"] = "DRY RUN - no memories were stored"
        return result
    finally:
        db.close()
