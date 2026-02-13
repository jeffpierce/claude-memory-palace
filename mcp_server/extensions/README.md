# Memory Palace Extensions

Extensions are Python modules that register additional MCP tools on the Memory Palace server.

## How Extensions Work

1. Extensions are loaded at server startup via `load_extensions(server)` in `mcp_server/server.py`
2. Each extension module must provide a `register(mcp)` function
3. The `register` function receives the FastMCP server instance and registers its tools
4. Extensions are configured in `~/.memory-palace/config.json` under the `"extensions"` key

## Configuration

Add extensions to your config file:

```json
{
  "extensions": [
    "mcp_server.extensions.switch_db",
    "mcp_server.extensions.my_custom_extension"
  ]
}
```

## Creating an Extension

1. Create a Python module in `mcp_server/extensions/`
2. Define a `register(mcp)` function
3. Inside `register`, use `@mcp.tool()` decorator to register tools
4. Add the module path to your config's `"extensions"` list

### Example Extension

```python
# mcp_server/extensions/hello_world.py
"""
Example extension that adds a hello_world tool.
"""
from typing import Any
from mcp_server.toon_wrapper import toon_response


def register(mcp):
    """Register hello_world tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def hello_world(name: str = "World") -> dict[str, Any]:
        """
        Say hello to someone.

        Args:
            name: Name to greet

        Returns:
            Greeting message
        """
        return {
            "message": f"Hello, {name}!",
            "extension": "hello_world"
        }
```

Then add to config:

```json
{
  "extensions": [
    "mcp_server.extensions.hello_world"
  ]
}
```

## Built-in Extensions

### db_manager (Recommended)

Config-aware database management for named databases. Replaces `switch_db` with full multi-database support.

**Tools:**

| Tool | Description |
|------|-------------|
| `memory_list_databases()` | List all configured databases with connection status, URLs (masked), and table counts |
| `memory_register_database(name, url?)` | Register a new database at runtime. Auto-derives URL from default if not provided |
| `memory_set_default_database(name)` | Change which database is used when no `database=` param is given |
| `memory_current_database()` | Show current default database, masked URL, connection status, and table counts |

**Configuration:**

```json
{
  "extensions": ["mcp_server.extensions.db_manager"],
  "databases": {
    "default": {"type": "postgres", "url": "postgresql://localhost:5432/memory_palace"},
    "life":    {"type": "postgres", "url": "postgresql://localhost:5432/memory_palace_life"},
    "work":    {"type": "postgres", "url": "postgresql://localhost:5432/memory_palace_work"}
  },
  "default_database": "default"
}
```

**Usage:**

```python
# List all databases
result = await memory_list_databases()

# Register a new database (auto-derives URL from default)
result = await memory_register_database(name="experiments")

# Switch default target
result = await memory_set_default_database(name="work")

# Check current database
result = await memory_current_database()
```

**Important:** All runtime changes (register, set default) are **not persisted** to config. This is by design — edit `~/.memory-palace/config.json` for permanent changes.

See [docs/POSTGRES.md](../../docs/POSTGRES.md) for full named database documentation.

### switch_db (Legacy)

Simple runtime database switching by URL manipulation. Predates named databases.

**Tools:**
- `memory_switch_db(database_name: str)` - Switch to a different PostgreSQL database by replacing the database name in the URL
- `memory_current_db()` - Get current database name and connection info

**Usage:**

```python
# Switch to a different database
result = await memory_switch_db("memory_palace2")

# Check current database
result = await memory_current_db()
```

**Note:** For new setups, prefer `db_manager` which understands named database config. `switch_db` still works for simple single-database configs where you just need to swap the DB name in the URL.

## Extension Loading

Extensions load at server startup. If an extension fails to load:

1. The error is logged to stderr
2. The server continues loading other extensions
3. The server does not crash

This allows graceful degradation if an extension has issues.

## Best Practices

1. **Keep extensions focused** - One extension per feature/domain
2. **Use @toon_response** - Leverage TOON encoding for efficient responses
3. **Follow naming conventions** - Use `memory_*` prefix for tool names
4. **Document tools** - Include clear docstrings
5. **Handle errors gracefully** - Don't crash the server
6. **Test independently** - Extensions should work standalone
7. **Version compatibility** - Document which Memory Palace version your extension requires

## Testing

See `test_extensions.py` in the project root for an example of testing extensions.

```bash
python test_extensions.py
```

## Loading Order

1. Core tools are registered via `register_all_tools(server)`
2. Extensions are loaded via `load_extensions(server)`
3. Database is initialized via `init_db()`
4. Server starts serving requests

This ensures extensions can rely on the database being available.
