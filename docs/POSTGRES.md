# PostgreSQL Setup & Operations

This guide covers PostgreSQL setup, configuration, and operations for Memory Palace. PostgreSQL provides production-grade features for multi-agent systems, including concurrent access, efficient vector search, and real-time messaging.

## When to Use PostgreSQL vs SQLite

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Setup | Zero config, file-based | Requires server install |
| Concurrent agents | 1-10 (write lock) | 10-10,000+ (MVCC) |
| Vector search | Brute-force cosine | pgvector with HNSW indexes |
| Real-time messaging | Polling only | LISTEN/NOTIFY push |
| Named databases | Derived file paths | True database isolation |
| Best for | Personal use, single agent | Teams, multi-agent, production |

**Use PostgreSQL when:**
- Running multiple concurrent agent instances
- Working in team environments
- Building production systems
- Need real-time inter-instance messaging
- Memory sets exceed 10,000+ entries
- Require advanced vector search performance

**Use SQLite when:**
- Personal, single-agent use
- Simple setup requirements
- Portable, file-based storage needed
- Memory sets under 10,000 entries

## Installation

### Prerequisites

- PostgreSQL 15+ (14 works but 15+ recommended)
- pgvector extension

### Quick Setup

**Ubuntu/Debian:**
```bash
sudo apt install postgresql-15 postgresql-15-pgvector
```

**macOS:**
```bash
brew install postgresql@15 pgvector
```

**Windows:**
- Download PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
- Install pgvector from [pgvector releases](https://github.com/pgvector/pgvector/releases)

### Database Creation

```bash
# Create database
createdb memory_palace

# Enable pgvector extension
psql memory_palace -c "CREATE EXTENSION vector;"
```

### Verify Installation

```bash
# Connect to database
psql memory_palace

# Check pgvector
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## Configuration

Memory Palace supports both legacy single-database configuration and the newer named databases pattern (v2.1+).

### Legacy Single Database

Still supported for backward compatibility:

```json
{
  "database": {
    "type": "postgres",
    "url": "postgresql://user:pass@localhost:5432/memory_palace"
  }
}
```

### Named Databases (v2.1+)

Named databases provide domain partitioning for different memory contexts:

```json
{
  "database": {
    "type": "postgres",
    "url": "postgresql://user:pass@localhost:5432/memory_palace"
  },
  "databases": {
    "default": {
      "type": "postgres",
      "url": "postgresql://user:pass@localhost:5432/memory_palace"
    },
    "life": {
      "type": "postgres",
      "url": "postgresql://user:pass@localhost:5432/memory_palace_life"
    },
    "work": {
      "type": "postgres",
      "url": "postgresql://user:pass@localhost:5432/memory_palace_work"
    }
  },
  "default_database": "default"
}
```

### Environment Variable Override

Set `MEMORY_PALACE_DATABASE_URL` to override the `database.url` configuration:

```bash
export MEMORY_PALACE_DATABASE_URL="postgresql://user:pass@localhost:5432/memory_palace"
```

## Named Databases Deep Dive

### What Are Named Databases?

Named databases provide domain partitions - separate databases for different memory contexts (life, work, project-specific). Each database has its own isolated `memories`, `edges`, and `messages` tables.

### Configuration Structure

- **`databases` dict**: Maps logical names to `{type, url}` configuration objects
- **`default_database`**: Selects which database name is used when no `database=` parameter is provided on tool calls
- **Fallback**: If the `databases` key is absent, the system falls back to the legacy `database` key for backward compatibility

### Auto-Derivation

When you request a database name that isn't explicitly configured, the system auto-derives a URL from the default database:

**PostgreSQL:**
- Replaces the database name in the path
- Prefixes with `memory_palace_` if needed
- Example: `postgresql://localhost/memory_palace` + name `life` → `postgresql://localhost/memory_palace_life`

**SQLite:**
- Appends to the filename
- Example: `sqlite:///path/memories.db` + name `life` → `sqlite:///path/memories_life.db`

Auto-derived URLs are registered in-memory for future lookups.

Source: `_derive_database_url()` in `memory_palace/config_v2.py`

### Runtime Management Tools

Enable the database manager extension:

```json
{
  "extensions": ["mcp_server.extensions.db_manager"]
}
```

Available tools:

- **`memory_list_databases`**: List all configured databases with connection status and table counts
- **`memory_register_database(name, url?)`**: Register a database at runtime (not persisted). If no URL provided, auto-derives from default.
- **`memory_set_default_database(name)`**: Change which database is used by default (runtime only, not persisted)
- **`memory_current_database`**: Show current default database, URL, connection status, and table counts

### Auto-Creation

When an engine is requested for a named database, the system automatically creates the PostgreSQL database if it doesn't exist:

1. Connects to the `postgres` maintenance database
2. Checks `pg_database` for existence
3. Creates the database if missing
4. Initializes tables and pgvector extension

SQLite databases auto-create by nature when first accessed.

Source: `ensure_database_exists()` in `memory_palace/database_v3.py`

## Migration from SQLite

Use the `memory-palace-sqlite-to-pg` CLI tool to migrate existing SQLite databases to PostgreSQL:

```bash
memory-palace-sqlite-to-pg \
  --source "sqlite:///~/.memory-palace/memories.db" \
  --target "postgresql://localhost:5432/memory_palace"
```

### Data Type Conversions

The migration tool handles the following conversions:

- **JSON text embeddings** → pgvector Vector
- **JSON text arrays** → PostgreSQL ARRAY
- **Integer booleans** → PostgreSQL BOOLEAN
- **JSON metadata** → JSONB

See [MIGRATION_2.0.md](MIGRATION_2.0.md) for the full migration guide.

## LISTEN/NOTIFY Mechanics

PostgreSQL's LISTEN/NOTIFY provides real-time pub/sub messaging for inter-instance communication.

### Channel Naming Convention

Channels follow the pattern: `memory_palace_msg_{instance_id}`

Examples:
- `memory_palace_msg_prime`
- `memory_palace_msg_crashtest`
- `memory_palace_msg_sandy`

### Payload Format

Messages are sent as JSON payloads:

```json
{
  "from_instance": "prime",
  "to_instance": "crashtest",
  "message_type": "handoff",
  "subject": "Code review complete",
  "message_id": 42,
  "priority": 5
}
```

### MCP Server Mode

In MCP server mode:
- Message service calls `pg_notify()` after writing messages to the database
- Receiving instances must poll for messages (no push available in MCP)
- The `instance_routes` config enables HTTP wake via OpenClaw gateway hooks as a push alternative

### OpenClaw Native Plugin Mode

In OpenClaw native plugin mode, true push messaging is available:

1. **Bridge subprocess startup**: Listener thread starts if PostgreSQL and instanceId are configured
2. **Listener thread**: Holds raw psycopg2 connection, runs `LISTEN memory_palace_msg_{instanceId}`
3. **Polling**: Checks `raw_conn.poll()` every 100ms
4. **Event emission**: Emits `{"event": "new_message", ...}` on stdout when notifications arrive
5. **Plugin dispatch**: Plugin receives event, dispatches to correct agent session via `enqueueSystemEvent`

**Dynamic subscription:**
- `_subscribe` and `_unsubscribe` management methods add/remove LISTEN channels at runtime
- Auto-subscribe on session discovery: when a new instance_id appears in tool calls, bridge auto-subscribes to its channel

See [OPENCLAW.md](OPENCLAW.md) for the full wake chain documentation.

## Connection Management

Memory Palace uses SQLAlchemy for connection pooling and management.

### Connection Pool Configuration

- **Pool type**: QueuePool
- **pool_size**: 5 connections
- **max_overflow**: 10 additional connections
- **pool_pre_ping**: True (validates connections before use)

### Engine Registry

- One engine per named database
- Engines created lazily on first access
- Session factories bound to corresponding engines

### Engine Management

- **`reset_engine(db_name)`**: Dispose a specific database engine
- **`reset_engine()`**: Dispose ALL engines (no arguments)

### Extension Auto-Creation

The pgvector extension is automatically created on first connection via SQLAlchemy event listeners:

```python
@event.listens_for(Engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.close()
```

## Troubleshooting

### Connection Issues

**Check PostgreSQL is running:**
```bash
pg_isready
```

**Test connection:**
```bash
psql -h localhost -U user -d memory_palace
```

**Check connection limits:**
```sql
SHOW max_connections;
SELECT count(*) FROM pg_stat_activity;
```

### pgvector Not Available

**Install pgvector:**
```bash
# Ubuntu/Debian
sudo apt install postgresql-15-pgvector

# macOS
brew install pgvector
```

**Enable extension:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Performance Issues

**Add HNSW index for vector search:**
```sql
CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops);
```

**Check query performance:**
```sql
EXPLAIN ANALYZE SELECT * FROM memories
ORDER BY embedding <=> '[...]' LIMIT 10;
```

### Database Auto-Creation Fails

Ensure the PostgreSQL user has CREATEDB privilege:

```sql
ALTER USER your_user CREATEDB;
```

## See Also

- [OPENCLAW.md](OPENCLAW.md) - OpenClaw integration and wake chain mechanics
- [MIGRATION_2.0.md](MIGRATION_2.0.md) - Full migration guide from SQLite
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/) - Connection pooling details
- [pgvector Documentation](https://github.com/pgvector/pgvector) - Vector extension details
