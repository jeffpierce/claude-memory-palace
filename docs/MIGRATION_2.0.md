# Migrating to Memory Palace 2.0

This guide walks you through upgrading from Memory Palace 1.x to 2.0. The migration changes the database schema, replaces several tools, and adds new features. Your existing memories are preserved.

## Before You Start

**Back up your database.** The migration modifies tables in place.

- **SQLite:** Copy `~/.memory-palace/memories.db` somewhere safe
- **PostgreSQL:** Run `pg_dump memory_palace > backup.sql`

Check your current version:

```bash
pip show memory-palace
```

If the version is `1.x`, you're in the right place.

## What Changed

### Schema

| 1.x | 2.0 | Notes |
|-----|-----|-------|
| `importance` (Integer 1-10) | `foundational` (Boolean) | Memories with importance >= 8 become foundational |
| `handoff_messages` table | `messages` table | Renamed, plus pubsub columns added |
| No knowledge graph | `memory_edges` table | Typed, weighted, directional edges between memories |
| `project` (single string) | `projects` (array) | Memories can belong to multiple projects |
| No `tags` column | `tags` (array) | Freeform organizational tags |

### Tools

Several tools were consolidated or replaced. If you reference tool names in agent prompts or system instructions, update them.

**Replaced tools:**

| 1.x Tool | 2.0 Replacement | Migration Notes |
|----------|-----------------|-----------------|
| `memory_forget` | `memory_archive` | Now has `dry_run=True` safety default |
| `memory_batch_archive` | `memory_archive` | Unified into single tool with filter-based archival |
| `memory_supersede` | `memory_link` | Use `relation_type="supersedes"` and `archive_old=True` |
| `memory_graph` | `memory_get` | Use `traverse=True` for BFS graph traversal |
| `memory_related` | `memory_get` | Graph context is now automatic on all retrievals |
| `handoff_send` | `message` | Use `action="send"` |
| `handoff_get` | `message` | Use `action="get"` |
| `handoff_mark_read` | `message` | Use `action="mark_read"` |

**New tools (no 1.x equivalent):**

| Tool | Description |
|------|-------------|
| `memory_recent` | Get the last N memories in title-card format |
| `memory_link` | Create typed, weighted edges between memories |
| `memory_unlink` | Remove edges |
| `code_remember_tool` | Index source files for natural language code search |
| `memory_audit` | Health checks: duplicates, stale memories, orphan edges |
| `memory_reembed` | Regenerate embeddings (backfill, refresh, model change) |
| `memory_reflect` | Extract memories from conversation transcripts |

**Changed tools:**

| Tool | What Changed |
|------|-------------|
| `memory_remember` | Added `foundational`, `tags`, `auto_link`, and `project` now accepts a list |
| `memory_recall` | Added centrality-weighted ranking and automatic graph context |
| `memory_get` | Added graph traversal (`traverse`), graph context (`include_graph`), multi-ID fetch |
| `memory_archive` | Added `dry_run` safety, filter-based archival, centrality protection |
| `memory_stats` | Enhanced with per-project breakdown, explodes multi-project counts |

### Configuration

The config file at `~/.memory-palace/config.json` has structural changes:

**1.x config:**
```json
{
  "db_path": "~/.memory-palace/memories.db",
  "ollama_url": "http://localhost:11434",
  "embedding_model": "nomic-embed-text",
  "llm_model": "llama3.2",
  "instances": ["support", "engineering"]
}
```

**2.0 config:**
```json
{
  "database": {
    "type": "sqlite",
    "url": null
  },
  "ollama_url": "http://localhost:11434",
  "embedding_model": null,
  "llm_model": null,
  "synthesis": {
    "enabled": true
  },
  "auto_link": {
    "enabled": true,
    "link_threshold": 0.75,
    "suggest_threshold": 0.675
  },
  "toon_output": true,
  "instances": ["support", "engineering"],
  "notify_command": null
}
```

Key differences:

- `db_path` is gone. Use `database.type` (`"sqlite"` or `"postgres"`) and `database.url` instead. When `url` is null, defaults are used (`~/.memory-palace/memories.db` for SQLite, `localhost:5432/memory_palace` for PostgreSQL).
- `embedding_model` and `llm_model` default to `null` for auto-detection. The setup wizard picks the best models for your hardware. You can still hardcode them.
- `synthesis.enabled` controls whether the local LLM synthesizes recall results. Set to `false` if you want your AI assistant to handle synthesis instead, or if you're running without a GPU.
- `auto_link` configures automatic edge creation when storing new memories.
- `toon_output` enables token-efficient TOON encoding for MCP responses.
- `notify_command` is an optional shell command template executed after sending messages.

## Migration Steps

### Step 1: Update the Code

```bash
cd memory-palace
git pull
pip install -e .
```

Or if installing from a release:

```bash
pip install --upgrade memory-palace
```

### Step 2: Run the Setup Wizard

The wizard detects your GPU and recommends models:

```bash
python -m setup.first_run
```

This creates or updates `~/.memory-palace/config.json` with the new structure. If you have an existing config, the wizard merges your settings — it won't overwrite values you've already set.

### Step 3: Run Schema Migrations

Two migrations run in sequence. Both are idempotent — safe to run multiple times.

**Migration 1: Core schema (v2 → v3)**

Adds `foundational` column, removes `importance`, renames `handoff_messages` to `messages`, adds pubsub columns.

```bash
memory-palace-migrate
```

Or with an explicit database URL:

```bash
memory-palace-migrate --database-url "sqlite:///path/to/memories.db"
memory-palace-migrate --database-url "postgresql://user:pass@localhost:5432/memory_palace"
```

What this does:
1. Adds `foundational` column to `memories`
2. Sets `foundational = true` for all memories where `importance >= 8`
3. Drops the `importance` column
4. Renames `handoff_messages` → `messages`
5. Adds pubsub columns: `channel`, `delivery_status`, `delivered_at`, `expires_at`, `priority`
6. Creates new indexes

**Migration 2: Multi-project support (v3 → v3.1)**

Converts the single `project` string to a `projects` array.

```bash
memory-palace-migrate-v3-1
```

What this does:
1. Adds `projects` column (ARRAY on PostgreSQL, JSON on SQLite)
2. Populates `projects` from existing `project` values: `"life"` → `["life"]`
3. Drops the old `project` column
4. Creates new indexes

### Step 4: Update Your Config

If you still have a 1.x style config, update it:

```bash
# Edit ~/.memory-palace/config.json
```

Replace `"db_path": "..."` with the `database` object. See the config comparison above.

Or just delete your config and re-run the setup wizard — it'll rebuild it:

```bash
rm ~/.memory-palace/config.json
python -m setup.first_run
```

### Step 5: Re-embed (Optional but Recommended)

If you're changing embedding models (e.g., from a 1.x model to the auto-detected recommendation), re-embed your memories:

```bash
# Preview what would be re-embedded
# (via MCP tool — run from your AI assistant)
memory_reembed(all_memories=True, dry_run=True)

# Execute
memory_reembed(all_memories=True, dry_run=False)
```

If you're keeping the same embedding model, you can skip this. Existing embeddings remain valid.

### Step 6: Verify

From your AI assistant, run:

```
memory_stats()
```

Check that:
- Memory count matches what you had before migration
- Projects show correctly (each memory should be in at least one project)
- No unexpected archived memories

Run an audit to catch any issues:

```
memory_audit()
```

This checks for duplicates, stale memories, orphan edges, missing embeddings, and contradictions.

## Upgrading to PostgreSQL (Optional)

2.0 supports both SQLite and PostgreSQL. SQLite works fine for personal use. PostgreSQL adds native vector search (pgvector), concurrent access, and partial indexes.

For comprehensive PostgreSQL setup including named databases and LISTEN/NOTIFY configuration, see [POSTGRES.md](POSTGRES.md).

### Prerequisites

1. Install PostgreSQL and pgvector
2. Create the database: `createdb memory_palace`
3. Enable pgvector: `psql memory_palace -c "CREATE EXTENSION vector;"`

### Create Target Tables

Start the MCP server once with your PostgreSQL config — it creates tables automatically on first run:

```json
{
  "database": {
    "type": "postgres",
    "url": "postgresql://localhost:5432/memory_palace"
  }
}
```

### Migrate Data

Use the built-in migration tool:

```bash
# Preview what would be migrated
memory-palace-sqlite-to-pg --dry-run

# Run the migration
memory-palace-sqlite-to-pg \
  --source "sqlite:///~/.memory-palace/memories.db" \
  --target "postgresql://localhost:5432/memory_palace"
```

If `--source` and `--target` are omitted, the tool uses your config defaults. If the target already has data, use `--force` to proceed (existing rows are skipped via `ON CONFLICT DO NOTHING`).

The tool converts all data types automatically:
- JSON text embeddings → pgvector Vector
- JSON text arrays (keywords, tags, projects) → PostgreSQL ARRAY
- Integer booleans (0/1) → PostgreSQL BOOLEAN
- JSON metadata → JSONB

### Verify

The migration tool prints a verification summary comparing row counts. You can also run `memory_stats()` and `memory_audit()` from your AI assistant to confirm everything transferred cleanly.

## Updating Agent Prompts

If your agent system prompts or persona files reference specific tool names, update them:

```
# Before (1.x)
Use handoff_send to send messages to other instances.
Use memory_forget to archive old memories.
Use memory_supersede to replace outdated memories.

# After (2.0)
Use message(action="send") to send messages to other instances.
Use memory_archive to archive old memories (dry_run=True by default).
Use memory_link(relation_type="supersedes", archive_old=True) to replace outdated memories.
```

The `project` parameter still accepts a single string for backward compatibility. To use multi-project, pass a list: `project=["life", "my-project"]`.

## Troubleshooting

**"no such table: memories"**

The migration expects existing tables. If you have a fresh database, just start the MCP server — it creates tables from the 2.0 schema automatically.

**"column importance does not exist"**

Migration 1 already ran. This is safe to ignore — the migration is idempotent.

**"column project does not exist"**

Migration 2 already ran. Same as above.

**"Both handoff_messages and messages tables exist"**

This happens if the MCP server created the new `messages` table before you ran the migration. The migration handles this — it merges data from `handoff_messages` into `messages` and drops the old table.

**SQLite: "Cannot drop column"**

SQLite versions before 3.35 don't support `ALTER TABLE DROP COLUMN`. The migration detects your SQLite version and skips the drop if unsupported. The old column remains in the table but is harmless — the ORM ignores unmapped columns.

**Embeddings not working after migration**

If you changed embedding models, existing embeddings are incompatible. Run `memory_reembed(all_memories=True, dry_run=False)` to regenerate them with the new model.

## What's Next: v3.0

Memory Palace continues to evolve. Key additions since 2.0:

**Named Databases** — Domain partitions (life, work, per-project) with auto-derivation, runtime management tools, and auto-creation of PostgreSQL databases on first access. See [POSTGRES.md](POSTGRES.md).

**OpenClaw Native Plugin** — All 13 palace tools registered directly with the OpenClaw gateway, eliminating MCP protocol overhead. Includes a persistent Python bridge subprocess, real-time pubsub wake via PostgreSQL LISTEN/NOTIFY, and automatic session discovery. See [OPENCLAW.md](OPENCLAW.md).

**Database Manager Extension** — Four new tools (`memory_list_databases`, `memory_register_database`, `memory_set_default_database`, `memory_current_database`) for runtime database management. Enable via the `db_manager` extension.
