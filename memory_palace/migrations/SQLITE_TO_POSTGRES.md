# SQLite to PostgreSQL Migration Guide

This guide walks you through migrating your Memory Palace data from SQLite to PostgreSQL.

## Why Migrate?

PostgreSQL offers several advantages over SQLite for Memory Palace:

- **Native vector search** with pgvector (faster semantic search)
- **Concurrent access** (multiple MCP clients can connect simultaneously)
- **Better performance** for large palaces (10k+ memories)
- **Graph query optimization** for complex relationship traversal
- **Real-time notifications** with LISTEN/NOTIFY for pubsub messaging

## Prerequisites

### 1. PostgreSQL Installation

Install PostgreSQL 14+ with the pgvector extension:

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-14-pgvector

# macOS (Homebrew)
brew install postgresql@14 pgvector

# Windows
# Download from https://www.postgresql.org/download/windows/
# Then install pgvector from https://github.com/pgvector/pgvector
```

### 2. Create Database

```bash
# Create the database
createdb memory_palace

# Or via psql
psql -c "CREATE DATABASE memory_palace;"
```

### 3. Install pgvector Extension

```bash
psql -d memory_palace -c "CREATE EXTENSION vector;"
```

### 4. Initialize Tables

The easiest way is to start the MCP server once with PostgreSQL configured:

```bash
# Update your config first (see below)
# Then start the server to create tables
memory-palace-mcp
```

Or manually run:

```bash
python -c "from memory_palace.database_v3 import init_db; init_db()"
```

## Migration Process

### Step 1: Update Configuration

Edit `~/.memory-palace/config.json`:

```json
{
  "database": {
    "type": "postgres",
    "url": "postgresql://localhost:5432/memory_palace"
  }
}
```

Or use environment variables:

```bash
export MEMORY_PALACE_DATABASE_URL="postgresql://localhost:5432/memory_palace"
```

### Step 2: Dry Run (Recommended)

First, run a dry run to see what will be migrated:

```bash
memory-palace-sqlite-to-pg --dry-run
```

This will show you:
- How many rows will be migrated from each table
- Any warnings or errors
- No data will be written

### Step 3: Run Migration

```bash
memory-palace-sqlite-to-pg
```

The script will:
1. Verify both databases are accessible
2. Check that target tables exist
3. Migrate all data in batches
4. Verify row counts match

### Step 4: Verify

After migration, verify your data:

```bash
# Check memory count
psql -d memory_palace -c "SELECT COUNT(*) FROM memories;"

# Check if embeddings migrated correctly
psql -d memory_palace -c "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL;"

# Verify relationships
psql -d memory_palace -c "SELECT COUNT(*) FROM memory_edges;"
```

## Advanced Options

### Custom Database URLs

If your databases aren't in the default locations:

```bash
memory-palace-sqlite-to-pg \
  --source "sqlite:///path/to/old/memories.db" \
  --target "postgresql://user:pass@host:5432/dbname"
```

### Force Migration

If the target database already has data and you want to proceed anyway:

```bash
memory-palace-sqlite-to-pg --force
```

**Warning:** This uses `ON CONFLICT DO NOTHING`, so existing rows won't be overwritten but gaps in IDs may occur.

### Batch Size

For very large databases, adjust the batch size:

```bash
memory-palace-sqlite-to-pg --batch-size 500
```

## What Gets Migrated

### memories Table

- **IDs preserved** — Maintains foreign key relationships
- **Embeddings** — JSON text arrays → pgvector Vector type
- **Arrays** — keywords/tags/projects JSON → PostgreSQL ARRAY(Text)
- **Booleans** — SQLite integers (0/1) → PostgreSQL BOOLEAN

### memory_edges Table

- **IDs preserved** — Maintains graph structure
- **Metadata** — JSON text → PostgreSQL JSONB
- **All relationships** — Source/target IDs remain valid

### messages Table

- **All messages** — Direct copy with type conversions
- **Pubsub data** — Channels, priorities, delivery status

## Troubleshooting

### pgvector Extension Missing

```
Error: pgvector extension not installed
```

**Solution:**

```bash
psql -d memory_palace -c "CREATE EXTENSION vector;"
```

### Target Tables Don't Exist

```
Error: Target database missing tables: ['memories', 'memory_edges', 'messages']
```

**Solution:** Run the MCP server once to create tables:

```bash
memory-palace-mcp
# Press Ctrl+C after it starts
```

### Row Count Mismatch

```
memories      source=1234 target=1230 ✗
```

**Possible causes:**
- Migration interrupted mid-batch
- Foreign key constraints (rare with current schema)
- Existing data conflicts

**Solution:** Check PostgreSQL logs and re-run migration with `--force`

### Database Connection Failed

```
Error: could not connect to server
```

**Solution:** Verify PostgreSQL is running:

```bash
# Check status
pg_ctl status

# Start if needed
pg_ctl start
```

## Performance Notes

- Migration speed: ~1000 rows/second on typical hardware
- For 10k memories: ~10-15 seconds total
- For 100k memories: ~2-3 minutes total
- Embeddings slow down migration slightly (JSON parsing + vector conversion)

## Rolling Back

If you need to go back to SQLite:

1. Stop the MCP server
2. Update `~/.memory-palace/config.json`:
   ```json
   {
     "database": {
       "type": "sqlite",
       "url": null
     }
   }
   ```
3. Restart MCP server — it will use the original SQLite database

Your SQLite database is **never modified** by the migration script.

## Post-Migration

After successful migration:

1. **Test semantic search** — Verify pgvector is working
2. **Check graph queries** — Verify relationships are intact
3. **Monitor performance** — PostgreSQL should be faster for large palaces
4. **Backup PostgreSQL** — Set up regular backups with `pg_dump`

Example backup:

```bash
pg_dump memory_palace > memory_palace_backup.sql
```

## Getting Help

If you encounter issues:

1. Run with `--dry-run` to see what's happening
2. Check PostgreSQL logs: `tail -f /var/log/postgresql/postgresql-14-main.log`
3. Open an issue on GitHub with the error message and migration output
