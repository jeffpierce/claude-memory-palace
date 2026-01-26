# Memory Palace 2.0 Migration Plan

## Overview

Major version bump: SQLite → PostgreSQL with pgvector, plus knowledge graph support.

**Not backwards compatible.** Requires data migration.

## Goals

1. **PostgreSQL with pgvector** - Native vector operations, proper concurrent access
2. **Knowledge graph** - Relational edges between memories
3. **Project scoping** - Memories organized by project (default: "life")
4. **Tags** - Freeform organizational tags (separate from keywords)
5. **Schema cleanup** - Proper types (arrays, booleans, vectors)

## Current Schema (1.x - SQLite)

```python
class Memory:
    id = Integer, PK
    created_at = DateTime
    updated_at = DateTime
    instance_id = String(50)
    memory_type = String(50)
    subject = String(255)
    content = Text
    keywords = JSON  # stored as JSON list
    importance = Integer
    source_type = String(50)
    source_context = Text
    source_session_id = String(100)
    embedding = JSON  # stored as JSON list of floats
    last_accessed_at = DateTime
    access_count = Integer
    expires_at = DateTime
    is_archived = Integer  # SQLite bool workaround

class HandoffMessage:
    id = Integer, PK
    created_at = DateTime
    from_instance = String(50)
    to_instance = String(50)
    message_type = String(50)
    subject = String(255)
    content = Text
    read_at = DateTime
    read_by = String(50)
```

## Target Schema (2.0 - PostgreSQL)

```sql
-- Core memories table
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    instance_id TEXT NOT NULL,
    project TEXT NOT NULL DEFAULT 'life',
    memory_type TEXT NOT NULL,
    subject TEXT,
    content TEXT NOT NULL,
    keywords TEXT[],
    tags TEXT[],
    importance INTEGER DEFAULT 5 CHECK (importance BETWEEN 1 AND 10),
    embedding vector(4096),  -- sfr-embedding-mistral dimension
    source_type TEXT,
    source_context TEXT,
    source_session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    last_accessed_at TIMESTAMPTZ,
    access_count INTEGER DEFAULT 0,
    expires_at TIMESTAMPTZ,
    is_archived BOOLEAN DEFAULT false
);

-- Knowledge graph edges
CREATE TABLE memory_edges (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relationship TEXT NOT NULL,
    strength FLOAT DEFAULT 1.0 CHECK (strength BETWEEN 0 AND 1),
    bidirectional BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    created_by TEXT,
    
    CONSTRAINT no_self_loops CHECK (source_id != target_id),
    UNIQUE(source_id, target_id, relationship)
);

-- Handoff messages (mostly unchanged)
CREATE TABLE handoff_messages (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    from_instance TEXT NOT NULL,
    to_instance TEXT NOT NULL,
    message_type TEXT NOT NULL,
    subject TEXT,
    content TEXT NOT NULL,
    read_at TIMESTAMPTZ,
    read_by TEXT
);

-- Indexes
CREATE INDEX idx_memories_instance ON memories(instance_id);
CREATE INDEX idx_memories_instance_project ON memories(instance_id, project);
CREATE INDEX idx_memories_project ON memories(project);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_importance ON memories(importance DESC);
CREATE INDEX idx_memories_created ON memories(created_at DESC);
CREATE INDEX idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_memories_keywords ON memories USING gin(keywords);
CREATE INDEX idx_memories_tags ON memories USING gin(tags);

CREATE INDEX idx_edges_source ON memory_edges(source_id);
CREATE INDEX idx_edges_target ON memory_edges(target_id);
CREATE INDEX idx_edges_relationship ON memory_edges(relationship);
CREATE INDEX idx_edges_source_rel ON memory_edges(source_id, relationship);

CREATE INDEX idx_handoff_to ON handoff_messages(to_instance);
CREATE INDEX idx_handoff_from ON handoff_messages(from_instance);
CREATE INDEX idx_handoff_unread ON handoff_messages(to_instance) WHERE read_at IS NULL;
```

## Relationship Types

Supported edge relationships:
- `supersedes` - Newer memory replaces older (directional)
- `relates_to` - General association (often bidirectional)
- `derived_from` - Memory came from processing another (directional)
- `contradicts` - Memories are in tension (bidirectional)
- `exemplifies` - Example of a concept (directional)
- `refines` - Adds detail/nuance (directional)

## New MCP Tools

| Tool | Description |
|------|-------------|
| `memory_link` | Create relationship between memories |
| `memory_unlink` | Remove relationship |
| `memory_related` | Get memories within N hops |
| `memory_supersede` | Create supersedes edge + optionally archive old |
| `memory_graph` | Return subgraph around a memory |

## Migration Steps

### Phase 1: Infrastructure
- [ ] Add psycopg2/asyncpg to dependencies
- [ ] Add pgvector to dependencies  
- [ ] Update config to support postgres connection string
- [ ] Create database.py abstraction that supports both SQLite and Postgres
- [ ] Add Alembic for migrations (or raw SQL migration scripts)

### Phase 2: Schema
- [ ] Create new SQLAlchemy models for Postgres
- [ ] Create memory_edges model
- [ ] Add project and tags columns
- [ ] Update embedding to use pgvector type

### Phase 3: Migration Script
- [ ] Export SQLite data to intermediate format (JSON/CSV)
- [ ] Transform data (keywords JSON → array, embedding JSON → vector, etc.)
- [ ] Backfill project column (default 'life', infer where possible)
- [ ] Import to Postgres
- [ ] Verify counts and spot-check data

### Phase 4: Tool Updates
- [ ] Update recall.py for pgvector similarity search
- [ ] Update remember.py for new schema
- [ ] Implement memory_link, memory_unlink
- [ ] Implement memory_related (graph traversal)
- [ ] Implement memory_supersede
- [ ] Implement memory_graph
- [ ] Update get_memory.py for new fields

### Phase 5: Testing
- [ ] Unit tests for new tools
- [ ] Integration test for full workflow
- [ ] Performance test for vector search
- [ ] Performance test for graph traversal

### Phase 6: Documentation
- [ ] Update README
- [ ] Update tool documentation
- [ ] Add Postgres setup instructions
- [ ] Document migration process for existing users

## Configuration Changes

```json
{
  "database": {
    "type": "postgres",  // or "sqlite" for legacy
    "url": "postgresql://user:pass@localhost:5432/memory_palace"
  },
  "ollama_url": "http://localhost:11434",
  "embedding_model": "sfr-embedding-mistral:f16",
  "llm_model": "qwen3:14b",
  "instances": ["desktop", "code", "clawdbot"]
}
```

## Open Questions

1. **Embedding dimension** - sfr-embedding-mistral is 4096d. Confirm this is correct.
2. **Keep SQLite support?** - Or fully deprecate for 2.0?
3. **Alembic vs raw SQL** - For migration management
4. **Connection pooling** - PgBouncer or just rely on psycopg pool?

## Notes

- Branch: `2.0`
- This is a breaking change - existing SQLite databases need explicit migration
- Consider providing a migration CLI command: `memory-palace migrate --from-sqlite`
