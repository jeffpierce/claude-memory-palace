# Testing Maintenance Tools via MCP

Since the Python test script requires database access (and may have missing dependencies like psycopg2), the easiest way to test the maintenance tools is via MCP in Claude Code.

## Prerequisites

1. The MCP server should be running (Claude Code will start it automatically)
2. PostgreSQL should be running (if using postgres backend)
3. Ollama should be running for embedding operations

## Test Commands

### 1. Test memory_audit

Ask Claude:

```
Run a memory audit. Check for duplicates, stale memories, orphan edges, missing embeddings, and contradictions. Limit results to 5 per category for readability.
```

Expected response format:
```json
{
  "summary": {
    "total_issues": N,
    "duplicates_found": N,
    "stale_found": N,
    "orphan_edges_found": N,
    "missing_embeddings_found": N,
    "contradictions_found": N
  },
  "duplicates": [...],
  "stale": [...],
  ...
}
```

### 2. Test memory_batch_archive (dry run)

Ask Claude:

```
Show me what would happen if we batch archived all memories older than 180 days with access count <= 2. Use dry run mode and enable centrality protection with threshold 5.
```

Expected response format:
```json
{
  "would_archive": N,
  "memories": [
    {
      "id": X,
      "subject": "...",
      "type": "...",
      "age_days": N,
      "access_count": N
    }
  ],
  "protected": [
    {
      "id": X,
      "subject": "...",
      "in_degree": N,
      "reason": "protected by centrality (in-degree=N)"
    }
  ],
  "note": "DRY RUN - no memories were archived. Set dry_run=False to execute."
}
```

### 3. Test memory_batch_archive (execute)

After reviewing the dry run results:

```
Execute the batch archive for memories older than 180 days with access count <= 2. Set dry_run=False and add reason "Testing maintenance tools".
```

### 4. Test memory_reembed (dry run)

Ask Claude:

```
How many memories would need re-embedding if we regenerated embeddings for all memories older than 365 days? Use dry run mode.
```

Expected response format:
```json
{
  "would_reembed": N,
  "estimated_time_seconds": N,
  "memories": [
    {
      "id": X,
      "subject": "...",
      "type": "..."
    }
  ],
  "note": "DRY RUN - no embeddings regenerated. Set dry_run=False to execute."
}
```

## Example Session

Here's a complete test session you can run:

```
User: Run a full palace audit. Check everything, limit to 5 results per category.

Claude: [Calls memory_audit tool, shows results]

User: Great. Now show me what would be archived if we cleaned up memories older than 180 days with low access (<=2). Dry run mode.

Claude: [Calls memory_batch_archive with dry_run=True, shows preview]

User: I see 15 would be archived and 3 are protected by centrality. That looks good. How long would it take to re-embed all memories older than 1 year?

Claude: [Calls memory_reembed with dry_run=True, shows estimate]

User: That's reasonable. Let's execute the batch archive for the old low-access memories. Set dry_run=False and add reason "Quarterly cleanup".

Claude: [Calls memory_batch_archive with dry_run=False, executes archival]
```

## What to Look For

### Audit Results

1. **Duplicates**: High similarity scores (>0.92) between memories with similar subjects
2. **Stale memories**: Old, low access, LOW centrality (in-degree < 3)
3. **Protected memories**: Old but high centrality (in-degree >= 5) - these WON'T be flagged as stale
4. **Orphan edges**: Edges pointing to archived memories (cleanup opportunity)
5. **Missing embeddings**: Memories that failed to embed (run backfill_embeddings)

### Centrality Protection in Action

When you see stale memory results, notice the `in_degree` field:
- Memory with in-degree 0-2: Standalone, safe to archive if old/unused
- Memory with in-degree 5+: Hub memory, protected even if old/unused
- Memory with in-degree 50+: Foundational, never flagged as stale

Example from real palace:
```
Memory #167: "Sandy's identity and personality"
  in-degree: 128
  age_days: 30
  access_count: 5

→ NOT flagged as stale despite low access
→ Protected by centrality (128 memories reference this)
```

### Batch Archive Safety

The dry run preview shows:
1. Memories that WOULD be archived (based on your criteria)
2. Memories PROTECTED by centrality (high in-degree)
3. Count of each group

Review this carefully before executing with `dry_run=False`.

## Troubleshooting

### "No issues found" in audit

Good news! Your palace is healthy. Consider:
- Lowering thresholds (e.g., duplicate_threshold from 0.92 to 0.88)
- Checking older time ranges (stale_days from 90 to 180)
- Running audit on specific projects

### Everything is protected in batch archive

Your memories are well-connected. Options:
- Lower `min_centrality_threshold` (from 5 to 3)
- Use explicit `memory_ids` list for targeted cleanup
- Review if your linking strategy is too aggressive

### Database connection errors

Ensure:
1. PostgreSQL is running: Check with Task Manager or `services.msc`
2. Database exists: `memory_palace` database should exist
3. Credentials match: Check `~/.memory-palace/config.json`

If using SQLite instead, update config:
```json
{
  "database": {
    "type": "sqlite",
    "url": null
  }
}
```

### Ollama errors in reembed

Ensure:
1. Ollama is running: Check Task Manager
2. Embedding model is pulled: `ollama list` should show nomic-embed-text or similar
3. Model is configured: Check `~/.memory-palace/config.json`

## Next Steps

1. Run the audit to understand your palace health
2. Identify maintenance patterns for your use case
3. Schedule regular audits (monthly recommended)
4. Document your cleanup criteria for future sessions

## Advanced: Automated Maintenance

Consider creating a maintenance workflow:

1. **Monthly audit**: Check palace health
2. **Quarterly cleanup**: Batch archive old low-value content
3. **Annual re-embed**: After model upgrades
4. **Ongoing**: Monitor contradictions and resolve conflicts

You can script this workflow using the MCP tools from Claude Code or build a scheduled task.
