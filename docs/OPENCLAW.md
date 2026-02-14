# OpenClaw Native Plugin Guide

## Overview

Memory Palace registers 13 palace tools as first-class OpenClaw agent tools through a native plugin architecture. This eliminates MCP protocol overhead by executing tools directly via a persistent Python bridge subprocess.

The plugin implementation is located in `openclaw_plugin/`.

**Key Benefits:**
- Direct tool execution without MCP serialization overhead
- Real-time pubsub message delivery via PostgreSQL LISTEN/NOTIFY
- Automatic session discovery and targeted agent wake
- Persistent bridge subprocess for low-latency calls

## Architecture

```
OpenClaw Gateway
    │
    ▼
TS Plugin (index.ts)
    │ NDJSON over stdin/stdout
    ▼
Python Bridge (bridge.py)
    │
    ▼
Memory Palace Services
    │
    ▼
PostgreSQL / SQLite
```

The plugin spawns a Python bridge subprocess on initialization. Communication happens via newline-delimited JSON (NDJSON) over stdin/stdout. The bridge maintains a persistent connection to the database and handles LISTEN/NOTIFY for real-time message delivery.

## Installation

### Build the Plugin

```bash
cd openclaw_plugin
npm install
npm run build
```

### Register the Plugin

Add the plugin to your OpenClaw configuration using the manifest at `openclaw_plugin/openclaw.plugin.json`.

### Plugin Configuration

The following options are available in `pluginConfig`:

- **`bridgeCommand`** (string, default: `"memory-palace-bridge"`)
  Command to spawn the bridge subprocess. Uses this as the executable name if no `pythonPath` is specified.

- **`pythonPath`** (string, optional)
  Explicit Python interpreter path. When set, spawns the bridge as `pythonPath -m memory_palace.bridge` instead of using `bridgeCommand`.

- **`instanceId`** (string, optional)
  Instance ID for auto-subscribe to DM channel. When set, the bridge automatically subscribes to `memory_palace_msg_{instanceId}` on startup for immediate message delivery.

**Example configuration:**

```json
{
  "pluginConfig": {
    "pythonPath": "/usr/bin/python3",
    "instanceId": "primary"
  }
}
```

## Bridge Protocol

The plugin communicates with the Python bridge using newline-delimited JSON (NDJSON) over stdin/stdout.

### Startup Handshake

After initialization, the bridge emits a ready message on stdout:

```json
{"ready": true, "version": "2.0.1", "tools": 13}
```

The plugin waits for this handshake with a default 10-second timeout before accepting tool calls. If the handshake is not received, the plugin fails initialization.

### Request/Response Format

**Request (plugin to bridge):**

```json
{"id": "req-1", "method": "memory_recall", "params": {"query": "auth"}}
```

**Success response (bridge to plugin):**

```json
{"id": "req-1", "result": {"memories": [...], "count": 5}}
```

**Error response (bridge to plugin):**

```json
{"id": "req-1", "error": {"message": "Unknown method: foo", "code": "METHOD_NOT_FOUND"}}
```

### Server Events

The bridge can emit events without an `id` field. These are asynchronous notifications from the database:

```json
{"event": "new_message", "data": {"from": "engineering", "to": "support", "type": "handoff", "subject": "Deployment complete", "id": 42, "priority": 5}}
```

### Management Methods

Management methods are prefixed with underscore and control bridge behavior:

**Shutdown:**

```json
{"id": "req-2", "method": "_shutdown"}
```

Gracefully stops the bridge subprocess.

**Subscribe to channel:**

```json
{"id": "req-3", "method": "_subscribe", "params": {"channel": "memory_palace_msg_support"}}
```

Issues a PostgreSQL `LISTEN` on the specified channel.

**Unsubscribe from channel:**

```json
{"id": "req-4", "method": "_unsubscribe", "params": {"channel": "memory_palace_msg_support"}}
```

Issues a PostgreSQL `UNLISTEN` for the specified channel.

### Error Codes

- **`PARSE_ERROR`** - Invalid JSON received on stdin
- **`METHOD_NOT_FOUND`** - Unknown method name
- **`INTERNAL_ERROR`** - Handler threw an exception
- **`SUBSCRIBE_ERROR`** - LISTEN operation failed
- **`UNSUBSCRIBE_ERROR`** - UNLISTEN operation failed

### Dispatch Table

The bridge exposes 13 Memory Palace methods:

1. `memory_set`
2. `memory_recall`
3. `memory_get`
4. `memory_recent`
5. `memory_archive`
6. `memory_link`
7. `memory_unlink`
8. `message`
9. `code_remember_tool`
10. `memory_audit`
11. `memory_reembed`
12. `memory_stats`
13. `memory_reflect`

## Real-Time Pubsub Wake Chain

The full chain for PostgreSQL real-time message delivery:

1. **Instance A sends message** - Calls `message(action="send")` which writes to database and executes `pg_notify("memory_palace_msg_B", payload)`

2. **Bridge receives notification** - Listener thread polls raw connection every 100ms and picks up NOTIFY

3. **Bridge emits event** - Outputs `{"event": "new_message", "data": {...}}` on stdout

4. **Plugin receives event** - Looks up target instance in session registry

5. **Plugin injects system event** - Calls `enqueueSystemEvent(text, {sessionKey})` to queue notification for target agent

6. **Plugin wakes agent** - Calls `requestHeartbeatNow({reason: "exec-event", coalesceMs: 250})` to trigger immediate processing

7. **Agent processes event** - Picks up the system event on next heartbeat cycle

**Critical Note:** Step 6 uses the "exec-event" reason hack. See [PATCH_NOTES.md](../openclaw_plugin/PATCH_NOTES.md) for details. Without the extensionAPI.js patch, steps 5-6 succeed silently but the agent never wakes up. The events go into a void queue that is never processed.

## Session Auto-Discovery

The plugin uses the `before_tool_call` hook to automatically discover which agent sessions are using palace tools:

- When any agent calls a palace tool, the hook extracts `instance_id` from params and `sessionKey` from the call context
- Mappings are stored in an in-memory `sessionRegistry` (Map<instanceId, {sessionKey, agentId, lastSeen}>)
- New registrations automatically subscribe the bridge to that instance's PostgreSQL LISTEN channel
- This enables targeted dispatch: when a message arrives for instance X, the plugin knows which session to inject the system event into

**Important:** The registry only populates when agents actually use palace tools. If an agent has never called a palace tool, it will not be in the registry and cannot receive messages.

## extensionAPI.js Patch

**Required for pubsub wake to function.**

OpenClaw's build process creates isolated copies of `enqueueSystemEvent` and `requestHeartbeatNow` in `extensionAPI.js`. These copies don't share closures with the heartbeat handler, so events go into a void queue and never wake the agent.

The patch rewires `createPluginRuntime()` to import these functions from the loader chunk instead of using the isolated copies.

**Full procedure and maintenance checklist:** [PATCH_NOTES.md](../openclaw_plugin/PATCH_NOTES.md)

**After any OpenClaw update:** Check if `extensionAPI.js` was regenerated and reapply the patch if necessary.

## Troubleshooting

### "Bridge did not send ready handshake"

**Cause:** Python not found, incorrect `bridgeCommand`, or bridge crashed on startup.

**Solution:** Check stderr output from the bridge process. Verify Python is installed and the bridge command is correct. If using `pythonPath`, ensure it points to a valid Python 3.8+ interpreter with Memory Palace dependencies installed.

### Agent doesn't wake on messages

**Cause:** extensionAPI.js patch not applied or was overwritten by an OpenClaw update.

**Solution:** Verify the patch is applied using the procedure in [PATCH_NOTES.md](../openclaw_plugin/PATCH_NOTES.md). Reapply if necessary.

### "Instance X not in session registry"

**Cause:** The target agent hasn't called any palace tool yet. The hook only fires on tool usage.

**Solution:** Have the agent call any palace tool (e.g., `memory_recent`) to register its session. After registration, it will receive messages.

### Bridge keeps reconnecting

**Cause:** Python dependencies missing or database connectivity issues.

**Solution:**
- Verify all Memory Palace dependencies are installed: `pip install -e .`
- Check database connection settings and credentials
- Verify PostgreSQL is running (for LISTEN/NOTIFY support)
- Review database connection configuration in Memory Palace settings

### Tool calls timeout after 30s

**Cause:** Bridge subprocess may be stuck or deadlocked.

**Solution:** Check stderr output for Python exceptions or database errors. Look for long-running queries or connection pool exhaustion. Restart the OpenClaw gateway to respawn the bridge.

### PostgreSQL NOTIFY not working

**Cause:** Using SQLite or connection pooling interfering with LISTEN.

**Solution:** LISTEN/NOTIFY requires PostgreSQL with a dedicated listener connection. See [POSTGRES.md](./POSTGRES.md) for setup details. SQLite does not support real-time pubsub and will fall back to polling.

## Additional Resources

- **[PATCH_NOTES.md](../openclaw_plugin/PATCH_NOTES.md)** - extensionAPI.js patch procedure and maintenance
- **[POSTGRES.md](./POSTGRES.md)** - PostgreSQL LISTEN/NOTIFY configuration details
- **`openclaw_plugin/index.ts`** - Plugin implementation source
- **`memory_palace/bridge.py`** - Python bridge implementation
