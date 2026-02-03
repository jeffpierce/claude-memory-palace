# Testing Memory Palace Maintenance Tools

This guide shows how to test and use the three new maintenance tools:
- `memory_audit` - Health checks and issue detection
- `memory_batch_archive` - Bulk archival with centrality protection
- `memory_reembed` - Regenerate embeddings

## Quick Start

### Option 1: Direct Python Testing

Run the test script:

```bash
cd /c/Users/jeffr/projects/efaas/claude-memory-palace
python examples/test_maintenance.py
```

This runs all three tools in safe mode (dry_run=True) and shows what they would do.

### Option 2: Via MCP (Claude Code)

The tools are registered as MCP tools and available in any Claude Code session.

## Testing Each Tool

### 1. memory_audit - Palace Health Check

**What it does:** Scans the palace for issues and returns actionable findings.

**Test via Python:**

```python
from memory_palace.services import audit_palace

result = audit_palace(
    check_duplicates=True,
    check_stale=True,
    check_orphan_edges=True,
    check_embeddings=True,
    check_contradictions=True,
    stale_days=90,
    stale_centrality_threshold=3,
    limit_per_category=20
)

print(f"Total issues: {result['summary']['total_issues']}")
print(f"Duplicates: {result['summary']['duplicates_found']}")
print(f"Stale: {result['summary']['stale_found']}")
```

**Test via MCP (Claude Code):**

Ask Claude:
> "Run a memory audit. Check for duplicates, stale memories, and missing embeddings."

**Expected Output:**

```json
{
  "duplicates": [
    {
      "memory_id": 42,
      "similar_to": 128,
      "similarity": 0.94,
      "subject_1": "Sandy's sarcastic personality",
      "subject_2": "Sandy personality traits"
    }
  ],
  "stale": [
    {
      "memory_id": 156,
      "subject": "Old test memory",
      "age_days": 120,
      "access_count": 0,
      "in_degree": 0
    }
  ],
  "summary": {
    "total_issues": 15,
    "duplicates_found": 3,
    "stale_found": 8,
    ...
  }
}
```

**Key Insight: Centrality Protection**

Notice the `in_degree` field in stale memories. Memories with high in-degree (many incoming edges) are NOT flagged as stale, even if old and rarely accessed. The graph structure protects foundational memories automatically.

Example:
- Memory #167 (Sandy's identity): 128 in-degree → NEVER flagged as stale
- Memory #156 (old test): 0 in-degree → Flagged as stale after 90 days

### 2. memory_batch_archive - Bulk Archival

**What it does:** Archives multiple memories matching criteria, with centrality protection.

**SAFETY:** Defaults to `dry_run=True`. You must explicitly set `dry_run=False` to execute.

**Test via Python (Dry Run):**

```python
from memory_palace.services import batch_archive_memories

# Preview what would be archived
result = batch_archive_memories(
    older_than_days=180,
    max_access_count=2,
    centrality_protection=True,
    min_centrality_threshold=5,
    dry_run=True  # Safe preview
)

print(f"Would archive: {result['would_archive']} memories")
print(f"Protected: {len(result['protected'])} memories")
```

**Test via MCP (Claude Code):**

Ask Claude:
> "Show me what would happen if we archived all memories older than 180 days with low access. Use dry run mode."

**Expected Output:**

```json
{
  "would_archive": 15,
  "memories": [
    {
      "id": 156,
      "subject": "Old daily summary",
      "type": "daily_content",
      "age_days": 200,
      "access_count": 1
    }
  ],
  "protected": [
    {
      "id": 167,
      "subject": "Sandy's identity",
      "in_degree": 128,
      "reason": "protected by centrality (in-degree=128)"
    }
  ],
  "note": "DRY RUN - no memories were archived. Set dry_run=False to execute."
}
```

**Execute (After Review):**

```python
# Actually archive after reviewing dry run
result = batch_archive_memories(
    older_than_days=180,
    max_access_count=2,
    dry_run=False,  # Execute for real
    reason="Quarterly cleanup - old low-value content"
)

print(f"Archived: {result['archived']} memories")
```

**Common Use Cases:**

1. **Daily content cleanup:**
   ```python
   batch_archive_memories(
       older_than_days=90,
       memory_type="daily_content",
       dry_run=False
   )
   ```

2. **Project cleanup:**
   ```python
   batch_archive_memories(
       project="old-prototype",
       older_than_days=180,
       dry_run=False
   )
   ```

3. **Explicit list:**
   ```python
   batch_archive_memories(
       memory_ids=[156, 157, 158],
       dry_run=False,
       reason="Manual cleanup"
   )
   ```

### 3. memory_reembed - Regenerate Embeddings

**What it does:** Re-generates embeddings for memories (e.g., after model upgrade).

**SAFETY:** Defaults to `dry_run=True`.

**Test via Python (Dry Run):**

```python
from memory_palace.services import reembed_memories

# Preview
result = reembed_memories(
    older_than_days=365,
    dry_run=True
)

print(f"Would re-embed: {result['would_reembed']} memories")
print(f"Estimated time: {result['estimated_time_seconds']} seconds")
```

**Test via MCP (Claude Code):**

Ask Claude:
> "How many memories would need re-embedding if we updated all embeddings older than 1 year?"

**Execute (After Review):**

```python
result = reembed_memories(
    older_than_days=365,
    dry_run=False
)

print(f"Re-embedded: {result['reembedded']} memories")
print(f"Failed: {result['failed']}")
```

**Common Use Cases:**

1. **Model upgrade:**
   ```python
   reembed_memories(all_memories=True, dry_run=False)
   ```

2. **Stale embeddings:**
   ```python
   reembed_memories(older_than_days=180, dry_run=False)
   ```

3. **Specific memories:**
   ```python
   reembed_memories(memory_ids=[42, 128, 167], dry_run=False)
   ```

## Real-World Workflow

### Scenario: Santa News Daily Cleanup

Santa News generates daily episode summaries. After 90 days, these become noise.

**Step 1: Audit**
```python
result = audit_palace(
    check_stale=True,
    stale_days=90,
    project="santa-news"
)

print(f"Found {result['summary']['stale_found']} stale episode summaries")
```

**Step 2: Preview Archival**
```python
result = batch_archive_memories(
    older_than_days=90,
    memory_type="daily_episode",
    project="santa-news",
    centrality_protection=True,  # Protects personality memories
    dry_run=True
)

# Review the preview
print(f"Would archive: {result['would_archive']}")
print(f"Protected: {len(result['protected'])}")  # Personality guide safe
```

**Step 3: Execute**
```python
result = batch_archive_memories(
    older_than_days=90,
    memory_type="daily_episode",
    project="santa-news",
    dry_run=False,
    reason="Quarterly cleanup - old episode summaries"
)

print(f"Archived {result['archived']} old episodes")
```

Result: 90+ old episode summaries archived, but personality guide (100+ in-degree) remains untouched.

## Centrality Protection: The Key Insight

The graph already knows what's foundational.

### Example: Sandy's Identity Memory

```
Memory #167: "Sandy's identity and personality"
- In-degree: 128 (128 other memories reference this)
- Age: 30 days
- Access count: 5
```

**Without centrality protection:**
- Old (30 days) + low access (5) = WOULD BE FLAGGED as stale
- Could be accidentally archived

**With centrality protection:**
- In-degree 128 > threshold 5 = PROTECTED
- Never flagged as stale, regardless of age/access

**Why This Works:**

The topology IS the importance. 128 connections means:
- Foundational memory
- Many others depend on it
- Archiving would orphan 128 edges
- Loss would break semantic connections

No manual tagging needed. Link your memories properly, and the graph protects what matters.

## Troubleshooting

### Audit finds no issues

Good! Your palace is healthy. Run audit periodically (monthly) to catch issues early.

### Batch archive protects everything

Your memories are well-connected (high centrality). Consider:
- Lower `min_centrality_threshold` (default 5 → try 3)
- Use explicit `memory_ids` for specific cleanup
- Check if criteria are too broad (tighten `older_than_days`)

### Reembed fails for some memories

Check `failed_ids` in result. Common causes:
- Ollama not running
- Embedding model not available
- Memory content too large (rare)

Run `memory_backfill_embeddings` to retry failures.

## Next Steps

1. Run `examples/test_maintenance.py` to see tools in action
2. Review audit findings for your palace
3. Plan quarterly cleanup workflow for your use case
4. Consider integrating audit into your regular workflow

## Safety Reminders

- All destructive operations default to `dry_run=True`
- Review dry run output before executing
- Centrality protection is ON by default
- Archived memories are soft-deleted (recoverable via include_archived=True)
- Audit is always safe (read-only)
