# Memory Palace Maintenance - Quick Reference

## TL;DR

Three new tools for palace maintenance:

| Tool | Purpose | Default Safety |
|------|---------|----------------|
| `memory_audit` | Find issues | Read-only (safe) |
| `memory_archive` | Bulk archival | `dry_run=True` |
| `memory_reembed` | Regenerate embeddings | `dry_run=True` |

**Key insight:** High-centrality memories (many incoming edges) are automatically protected. The graph knows what's important.

## Commands

### Quick Health Check

```
Run a memory audit.
```

### Find What to Clean Up

```
Show me stale memories older than 90 days with low access.
```

### Preview Bulk Archive

```
What would happen if we archived memories older than 180 days with access count <= 2?
Use dry run mode and centrality protection.
```

### Execute Archive (After Review)

```
Execute batch archive for memories older than 180 days with access count <= 2.
Set dry_run=False, add reason "Quarterly cleanup".
```

### Check Embedding Coverage

```
How many memories are missing embeddings?
```

### Regenerate Embeddings

```
Re-embed all memories older than 1 year. Start with dry run.
```

## Centrality Protection

**What it is:** Memories with many incoming edges (in-degree) are protected from archival.

**Why it matters:** The graph structure identifies foundational memories. No manual importance tagging needed.

**Example:**
```
Memory #167: "Sandy's identity"
  - In-degree: 128 (128 other memories reference this)
  - Result: PROTECTED (never flagged as stale)

Memory #156: "Old test note"
  - In-degree: 0 (no references)
  - Result: Safe to archive if old/unused
```

**Default threshold:** In-degree >= 5 grants protection

## Common Workflows

### Monthly: Check Palace Health
```
Run memory_audit with default settings
→ Review duplicates, stale memories, orphan edges
```

### Quarterly: Clean Up Old Content
```
1. memory_audit (check_stale=True)
2. memory_archive (dry_run=True) - preview
3. Review protected vs. to-archive
4. memory_archive (dry_run=False) - execute
```

### Annually: Upgrade Embeddings
```
1. memory_reembed (all_memories=True, dry_run=True) - estimate
2. memory_reembed (all_memories=True, dry_run=False) - execute
```

### Daily Content Pattern (e.g., Santa News)
```
Weekly: memory_audit (project="santa-news", stale_days=90)
Quarterly: batch_archive (memory_type="daily_episode", older_than_days=90)
Result: Personality guide (high centrality) protected, old episodes archived
```

## Safety Checklist

Before executing any batch operation:

1. ✅ Run with `dry_run=True` first
2. ✅ Review preview (would_archive count, protected list)
3. ✅ Verify centrality protection is ON (default)
4. ✅ Add a reason field for audit trail
5. ✅ Confirm criteria are correct (dates, counts, types)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No issues found in audit | Palace is healthy! Or try lower thresholds |
| Everything protected in batch archive | Lower `min_centrality_threshold` (5 → 3) |
| Reembed fails | Check Ollama running, embedding model available |
| Database errors | Check PostgreSQL/SQLite running, config correct |

## Parameter Quick Reference

### memory_audit
```
check_duplicates=True          # Find near-duplicates
check_stale=True               # Find old low-value memories
check_orphan_edges=True        # Find broken edges
check_embeddings=True          # Find missing embeddings
check_contradictions=True      # Find conflicts
stale_days=90                  # Age threshold
stale_centrality_threshold=3   # Protection threshold
duplicate_threshold=0.92       # Similarity threshold
limit_per_category=20          # Results cap
```

### memory_archive
```
older_than_days=180           # Age filter
max_access_count=2            # Low-access filter
memory_type="daily_content"   # Type filter
project="santa-news"          # Project filter
memory_ids=[1,2,3]            # Explicit list
centrality_protection=True    # Auto-protect hubs
min_centrality_threshold=5    # Protection in-degree
dry_run=True                  # Safety default
reason="Quarterly cleanup"    # Audit trail
```

### memory_reembed
```
older_than_days=365          # Age filter
memory_ids=[1,2,3]           # Explicit list
project="santa-news"         # Project filter
all_memories=True            # Nuclear option
dry_run=True                 # Safety default
```

## Red Flags

🚨 **DON'T:**
- Execute batch operations without dry run preview
- Disable centrality protection unless you know why
- Archive explicitly by ID without reviewing memory first
- Re-embed all memories without time estimate

✅ **DO:**
- Always preview with dry_run=True first
- Review protected memory list (are these really important?)
- Add reason field to batch operations
- Monitor audit summary counts over time

## Emergency Recovery

If you accidentally archived something important:

1. **Find the memory:** `memory_recall(query="subject", include_archived=True)`
2. **Identify ID:** Note the memory ID from results
3. **Un-archive manually:** Update database `is_archived=False` OR
4. **Re-create from content:** Store new memory with same content

Archived memories are SOFT DELETED. They're recoverable.

## Performance Notes

- Audit on 1k memories: ~2-5 seconds
- Audit on 10k memories: Not yet tested
- Duplicate detection: O(n²) but capped by limit
- Centrality computation: O(1) per memory with indexes
- Re-embedding: ~200ms per memory (sequential)

## When to Run What

| Frequency | Operation | Purpose |
|-----------|-----------|---------|
| Weekly | Audit (project-specific) | Catch issues early |
| Monthly | Audit (full palace) | Health dashboard |
| Quarterly | Batch archive (old content) | Keep palace lean |
| Annually | Re-embed (model upgrade) | Improve search quality |
| As needed | Consolidate (future) | Compress daily → summaries |

## Integration Ideas

1. **Morning briefing:** Include audit summary in daily report
2. **Scheduled cleanup:** Cron job for quarterly batch archive
3. **Pre-commit hook:** Audit before major changes
4. **Dashboard widget:** Show palace health stats
5. **Notification:** Alert on high duplicate/stale counts

## Further Reading

- Design spec: `docs/MAINTENANCE.md`
- Testing guide: `docs/TESTING_MAINTENANCE.md`
- MCP testing: `examples/test_maintenance_mcp.md`
- Implementation: `MAINTENANCE_IMPLEMENTATION.md`
