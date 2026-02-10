# Palace Maintenance: Design Document

## The Problem

Currently, palace maintenance is manual and doesn't scale:
- `memory_archive` archives one memory at a time
- `memory_stats` reports counts but not problems
- `memory_reembed` handles missing embeddings
- No detection of issues, no batch operations, no consolidation

For a user like Santa News (daily YouTube automation), this means:
- 365 "covered this story" memories per year
- No way to batch-archive old content
- No way to consolidate "January news" into a summary
- Personality memories mixed with ephemeral content memories

## Current State

### What We Have
| Tool | Purpose | Limitation |
|------|---------|------------|
| `memory_archive` | Archive single memory | Manual, one-by-one |
| `memory_link` (with `relation_type="supersedes"`, `archive_old=True`) | Replace old with new | Manual, requires knowing both IDs |
| `memory_stats` | Counts and top-5 lists | Reports facts, not problems |
| `memory_reembed` | Generate missing embeddings | Reactive, not proactive |

### What's Missing

**Detection/Audit:**
- Find near-duplicate memories (high semantic similarity)
- Find contradicting memories (via `contradicts` edges or semantic conflict)
- Find stale memories (old + low access + no edges)
- Find orphaned edges (pointing to archived memories)
- Find memories missing embeddings (before backfill)

**Batch Operations:**
- Archive by criteria (age, access count, type, project)
- Re-embed all memories (when model changes)
- Bulk update metadata (fix type classifications)

**Consolidation:**
- Merge related memories into summary
- Create "supersedes" chain from daily → weekly → monthly summaries
- Archive originals after consolidation

**Health Dashboard:**
- "Problems to fix" report
- Memory age distribution
- Graph health (orphans, broken edges)
- Embedding coverage and staleness

---

## Proposed Design

### 1. `memory_audit` - Health Check Tool

```python
async def memory_audit(
    check_duplicates: bool = True,      # Find near-duplicates (>0.9 similarity)
    check_stale: bool = True,           # Old + low access + low centrality
    check_orphan_edges: bool = True,    # Edges pointing to archived memories
    check_embeddings: bool = True,      # Memories missing embeddings
    check_contradictions: bool = True,  # Find 'contradicts' edges for review
    stale_days: int = 90,               # Age threshold for "stale"
    stale_access_threshold: int = 2,    # Access count threshold
    stale_centrality_threshold: int = 3,# In-degree below this = not protected
    duplicate_threshold: float = 0.92,  # Similarity threshold for duplicates
    project: str = None,                # Filter by project
    limit_per_category: int = 20        # Cap results per issue type
) -> dict:
    """
    Audit palace health and return actionable findings.

    Returns:
        {
            "duplicates": [{"memory_id": X, "similar_to": Y, "similarity": 0.95}, ...],
            "stale": [{"memory_id": X, "age_days": 120, "access_count": 1, "in_degree": 0}, ...],
            "orphan_edges": [{"edge_id": X, "source": Y, "target": Z, "reason": "target archived"}, ...],
            "missing_embeddings": [memory_id, ...],
            "contradictions": [{"memory_id": X, "contradicts": Y, "needs_resolution": true}, ...],
            "summary": {
                "total_issues": N,
                "duplicates_found": N,
                "stale_found": N,
                ...
            }
        }
    """
```

### 2. `memory_archive` - Bulk Archival

```python
async def memory_archive(
    # Filter criteria (all optional, combined with AND)
    older_than_days: int = None,        # Age filter
    max_access_count: int = None,       # Low-access filter
    memory_type: str = None,            # Type filter
    project: str = None,                # Project filter
    memory_ids: list[int] = None,       # Explicit ID list

    # Centrality protection (THE KEY INSIGHT)
    centrality_protection: bool = True,       # Protect high-centrality memories
    min_centrality_threshold: int = 5,        # In-degree count that grants protection

    # Safety
    dry_run: bool = True,               # Preview only (default safe)
    require_confirmation: bool = True,  # Return preview, require second call

    # Metadata
    reason: str = None                  # Archival reason for audit trail
) -> dict:
    """
    Archive multiple memories matching criteria.

    Safety: dry_run=True by default. Returns preview of what would be archived.
    Second call with dry_run=False and same criteria executes.

    Returns:
        {
            "would_archive": N,  # or "archived": N if dry_run=False
            "memories": [{"id": X, "subject": "...", "age_days": N}, ...],
            "confirmation_token": "abc123"  # For require_confirmation flow
        }
    """
```

### 3. `memory_consolidate` - Merge Related Memories

```python
async def memory_consolidate(
    memory_ids: list[int],              # Memories to consolidate
    new_subject: str,                   # Subject for consolidated memory
    new_type: str = None,               # Type (defaults to most common from inputs)
    archive_originals: bool = True,     # Archive inputs after consolidation

    # LLM synthesis
    synthesis_prompt: str = None,       # Custom prompt for consolidation
    preserve_key_details: bool = True,  # Instruct LLM to keep important specifics

    dry_run: bool = True
) -> dict:
    """
    Consolidate multiple memories into one summary memory.

    Uses LLM to synthesize content from all input memories.
    Creates supersedes edges from new memory to all originals.
    Archives originals by default.

    Use case: "Consolidate all January 2026 news coverage into monthly summary"

    Returns:
        {
            "consolidated_memory_id": X,
            "content_preview": "...",
            "archived_ids": [...],
            "supersedes_edges_created": N
        }
    """
```

### 4. `memory_reembed` - Refresh Embeddings

```python
async def memory_reembed(
    # Filter criteria
    older_than_days: int = None,        # Re-embed old embeddings
    memory_ids: list[int] = None,       # Explicit list
    project: str = None,
    all: bool = False,                  # Nuclear option

    # Progress
    batch_size: int = 50,
    dry_run: bool = True
) -> dict:
    """
    Regenerate embeddings for memories.

    Use when:
    - Embedding model changes
    - Embeddings seem to be returning poor results
    - After bulk import

    Returns:
        {
            "would_reembed": N,  # or "reembedded": N
            "estimated_time_seconds": N,
            "memories": [{"id": X, "subject": "..."}, ...]
        }
    """
```

---

## Centrality as Protection (Key Insight)

**The graph already knows what's foundational.**

We built centrality-weighted retrieval with the Clawddar collaboration:
```
score = (similarity × 0.7) + (log(access + 1) × 0.15) + (centrality × 0.15)
```

That same centrality signal works for maintenance protection:

| Centrality | Meaning | Maintenance Action |
|------------|---------|-------------------|
| High (10+ in-degree) | Many memories link to this | **PROTECTED** - never auto-archive |
| Medium (3-9 in-degree) | Some dependencies | Warn before archiving |
| Low (0-2 in-degree) | Standalone/ephemeral | Safe to archive if stale |

### Why This Works

Memory 167 (Sandy's identity) has 128+ connections. The graph structure itself says "this is load-bearing." Archiving it would:
- Orphan 128 edges
- Break semantic connections across the palace
- Lose context that other memories depend on

**No manual importance tagging needed** - the topology IS the importance.

### Santa News Example

```
"Santa's personality guide"     → 50+ in-degree (every episode links to it)
"Preferred news sources"        → 30+ in-degree (daily content references it)
"Editorial style decisions"     → 20+ in-degree (affects all content)

"Covered Ukraine news Jan 15"   → 0 in-degree (nothing references it)
"Covered tech news Jan 16"      → 0 in-degree (ephemeral, standalone)
"Covered sports Jan 17"         → 1 in-degree (maybe one follow-up)
```

Batch archive with `centrality_protection=True` would:
- Skip personality/sources/style (high centrality)
- Archive the daily coverage memories (low centrality, old, ephemeral)

**The user doesn't have to manually tag importance.** They just need to LINK memories properly. The graph does the rest.

### Virtuous Cycle

This creates an incentive for good graph hygiene:
1. **Link your memories** → foundational ones accumulate in-degree
2. **In-degree = protection** → important memories survive maintenance
3. **Low-link memories age out** → palace stays lean
4. **The graph self-organizes** → topology reflects true importance

Auto-linking already helps: when you store a memory, it finds similar ones and creates edges. Over time, hub memories naturally emerge. The maintenance system respects what the graph learned.

### Implementation Detail

For `memory_audit` stale detection:
```python
# Memory is stale ONLY if:
# - Older than threshold AND
# - Access count below threshold AND
# - In-degree (centrality) below protection threshold

is_stale = (
    age_days > stale_days and
    access_count < stale_access_threshold and
    in_degree < centrality_threshold  # NEW: graph protection
)
```

For `memory_archive`:
```python
# Exclude from archival if centrality protected
if centrality_protection and memory.in_degree >= min_centrality_threshold:
    protected_memories.append(memory)
    continue  # Skip this memory
```

---

## Maintenance Patterns

### Daily Content Pattern (Santa News)

For workflows generating daily ephemeral content:

```
1. Use memory_type="daily_content" or "episode" for ephemeral memories
2. Weekly: `memory_consolidate` daily → weekly summary
3. Monthly: `memory_consolidate` weekly → monthly summary
4. Quarterly: `memory_archive` older_than_days=90, memory_type="daily_content"
```

This creates a natural compression:
- 365 daily memories → 52 weekly → 12 monthly
- Originals archived but recoverable
- Supersedes chain preserves provenance

### Identity vs Content Split

Recommend users distinguish:
- **Identity memories** (importance >= 7): personality, preferences, style guides
- **Content memories** (importance <= 5): what was done, ephemeral facts
- **Decision memories** (importance 6-7): choices made, could inform future

Batch archive targets content memories. Identity stays forever.

### Contradiction Resolution Flow

When `memory_audit` finds contradictions:
1. Surface both memories to user/AI
2. User decides which is authoritative
3. `memory_link` (with `relation_type="supersedes"`, `archive_old=True`) new over old (if new correct)
4. Or `memory_archive` the incorrect one

The `contradicts` edge type is a FLAG, not a resolution. Human reviews.

---

## Implementation Priority

### Phase 1: Audit (Detection)
- `memory_audit` with all check types
- Surfaces problems, doesn't fix them
- Low risk, high visibility

### Phase 2: Batch Archive
- `memory_archive` with dry_run safety
- Most requested maintenance operation
- Enables "clean up old stuff" workflows

### Phase 3: Consolidation
- `memory_consolidate` with LLM synthesis
- Enables compression patterns
- More complex, needs good prompts

### Phase 4: Re-embed
- `memory_reembed` for model migration
- Only needed when embedding model changes
- Lower priority until model upgrade

---

## Open Questions

1. **Decay**: Should importance auto-decay over time? Or is explicit archival better?
   - Pro decay: automatic, less maintenance
   - Con decay: might lose important old memories
   - Current lean: explicit archival, no decay

2. **Scheduled maintenance**: Should there be a cron-able "maintenance pass"?
   - Run audit, auto-archive obvious stale, report findings
   - Risk: autonomous archival could lose things
   - Maybe: audit + report only, human approves actions

3. **Edge cleanup**: When memory archived, what happens to edges?
   - Current: edges remain (orphaned)
   - Option A: cascade delete edges
   - Option B: mark edges as "to_archived" for audit
   - Current lean: Option B (preserve provenance)

4. **Consolidation quality**: How good is LLM at summarizing memories?
   - Need testing with real consolidation scenarios
   - May need few-shot examples in prompt
   - Consider: keep original content in `source_context` field

---

## Success Criteria

Palace maintenance is successful when:

1. **Santa News can run for a year** without palace bloat becoming unusable
2. **Audit takes < 30 seconds** on a 10k memory palace
3. **Batch archive is safe** - dry_run default, clear preview, reversible
4. **Consolidation preserves meaning** - LLM summary captures key facts
5. **Users understand the patterns** - docs explain when to use what

---

*Design document created: 2026-02-02*
*Branch: feature/palace-maintenance*
*Author: Sandy (Code instance)*
