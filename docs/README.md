# Memory Palace - Documentation

A persistent memory system for AI instances, enabling semantic search across conversations, facts, and insights with intelligent auto-linking and knowledge graph capabilities.

## Quick Start

### Prerequisites

1. **Python 3.10+** - Required for the MCP server
2. **Ollama** - For local embedding and LLM models
   - Download from: https://ollama.ai/download
3. **NVIDIA GPU** - Recommended for acceptable performance (4GB+ VRAM)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jeffpierce/memory-palace.git
   cd memory-palace
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

   Or use the installer scripts which handle everything:
   - **Windows:** `install.bat` or `install.ps1`
   - **macOS/Linux:** `./install.sh`

4. **Run first-time setup:**
   ```bash
   python -m setup.first_run
   ```

   This will:
   - Detect your GPU and VRAM
   - Recommend appropriate models
   - Download required Ollama models

### Configure Claude Desktop

Add the following to your Claude Desktop MCP configuration:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "memory-palace": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:\\path\\to\\memory-palace",
      "env": {
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

Adjust paths for your system.

## Usage

Memory Palace provides 13 core MCP tools organized into five categories, plus extension tools for database management:

### Core Memory Tools

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a new memory with automatic linking |
| `memory_recall` | Search memories using semantic search with centrality-weighted ranking |
| `memory_get` | Retrieve memories by ID with optional graph traversal |
| `memory_recent` | Get the last X memories (default 20, max 200) |
| `memory_archive` | Archive memories with protection for foundational/high-centrality nodes |

#### memory_remember - Store Memories

Stores a new memory with intelligent auto-linking based on semantic similarity.

**Auto-linking behavior (two tiers):**
- **Auto-linked** (≥0.75 similarity): Edges created automatically with LLM-classified types. Returned in `links_created`.
- **Suggested** (0.675-0.75 similarity): Surfaced for human review, no edges created. Returned in `suggested_links`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `instance_id` | string | Which AI instance is storing this (e.g., "support", "engineering", "analytics") |
| `memory_type` | string | Type of memory (see standard types below) |
| `content` | string | The actual memory content |
| `subject` | string | What/who this memory is about (optional but recommended) |
| `keywords` | list | List of keywords for searchability (optional) |
| `tags` | list | Freeform organizational tags (optional) |
| `foundational` | bool | True if core/foundational (never archived, default: false) |
| `project` | string/list | Project(s) this memory belongs to (default: "life") |
| `source_type` | string | How created (conversation, explicit, inferred, observation) |
| `source_context` | string | Snippet of original context (optional) |
| `source_session_id` | string | Link to conversation session (optional) |
| `supersedes_id` | int | If set, creates 'supersedes' edge and archives target (optional) |
| `auto_link` | bool | Override config to enable/disable auto-linking (optional) |

**Standard memory types:**
- `fact` - Objective information
- `preference` - User preferences
- `event` - Things that happened
- `context` - Situational context
- `insight` - Derived understanding
- `relationship` - Connections between things
- `architecture` - System design information
- `gotcha` - Things to watch out for
- `blocker` - Blocking issues
- `solution` - Solutions to problems
- `workaround` - Temporary fixes
- `design_decision` - Why something was done a certain way

Custom types are allowed - use descriptive names that fit your needs.

**Example:**
```python
memory_remember(
    instance_id="engineering",
    memory_type="architecture",
    content="The authentication system uses JWT tokens with 24-hour expiry",
    subject="authentication system",
    keywords=["jwt", "auth", "tokens"],
    project="my-app",
    foundational=True
)
```

#### memory_recall - Semantic Search

Search memories using semantic similarity with centrality-weighted ranking.

**Centrality-weighted ranking formula:**
Combines semantic similarity, access frequency, and graph centrality (in-degree count) to surface the most relevant and important memories.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query (semantic when Ollama available, keyword fallback) |
| `instance_id` | string | Filter by instance (optional) |
| `project` | string/list | Filter by project (optional) |
| `memory_type` | string | Filter by type, supports wildcards (e.g., "code_*") |
| `subject` | string | Filter by subject (optional) |
| `min_foundational` | bool | Only return foundational memories (optional) |
| `include_archived` | bool | Include archived memories (default: false) |
| `limit` | int | Maximum memories to return (default: 20) |
| `detail_level` | string | "summary" or "verbose" (applies when synthesize=True) |
| `synthesize` | bool | Use local LLM to synthesize results (default: true) |
| `include_graph` | bool | Include graph context for top N results (default: true) |
| `graph_top_n` | int | Number of top results to fetch graph context for (default: 5) |
| `graph_depth` | int | How many hops to follow (1-3, default: 1) |

**Returns (synthesize=True):**
```json
{
  "summary": "Natural language summary of results",
  "count": 5,
  "search_method": "semantic",
  "memory_ids": [42, 17, 89, 103, 56],
  "graph_context": {"nodes": {...}, "edges": [...]}
}
```

**Returns (synthesize=False):**
```json
{
  "memories": [
    {
      "id": 42,
      "subject": "authentication system",
      "content": "...",
      "similarity_score": 0.89,
      ...
    }
  ],
  "count": 5,
  "search_method": "semantic",
  "graph_context": {"nodes": {...}, "edges": [...]}
}
```

**Example:**
```python
# Quick natural language summary
memory_recall(query="How does authentication work?")

# Get raw memories with high similarity for detailed analysis
memory_recall(query="authentication approach", synthesize=False, limit=10)

# Search within a project
memory_recall(query="API endpoints", project="my-app", synthesize=False)
```

#### memory_get - Retrieve by ID

Retrieve one or more memories by their IDs with optional graph traversal.

**Use this when:**
- You have specific memory IDs from handoff messages (e.g., "Memory 151")
- You need to fetch exact memories without search
- You want to traverse the knowledge graph from known nodes

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_ids` | int/list | Single memory ID or list of IDs to retrieve |
| `detail_level` | string | "summary" or "verbose" (default: "verbose") |
| `synthesize` | bool | Use LLM to synthesize (skipped for single memory, default: false) |
| `include_graph` | bool | Include graph context for all memories (default: true) |
| `graph_depth` | int | Hops to follow (1-3, default: 1, use 2 for bootstrap) |
| `traverse` | bool | Do BFS traversal instead of context (default: false) |
| `max_depth` | int | Max depth for BFS traverse (1-5, only if traverse=True) |
| `direction` | string | "outgoing", "incoming", or None for both (optional) |
| `relation_types` | list | Filter edges by type (optional) |
| `min_strength` | float | Filter edges by minimum strength 0.0-1.0 (optional) |

**Graph context vs. Traversal:**
- `include_graph=True`: Shows immediate connections (asymmetric depth-1 neighborhood)
- `traverse=True`: Performs breadth-first search traversal (replaces old `memory_graph` tool)

**Key difference from memory_recall:**
`memory_recall` limits graph context to top N results (performance), while `memory_get` includes graph context for ALL requested memories (intentional targeted fetches).

**Example:**
```python
# Single memory with graph context
memory_get(memory_ids=42)

# Multiple memories, raw, with graph
memory_get(memory_ids=[167, 168, 169], synthesize=False)

# Traverse from a starting point
memory_get(memory_ids=42, traverse=True, max_depth=2, direction="outgoing")

# Bootstrap/startup with broader context
memory_get(memory_ids=[1, 2, 3], graph_depth=2)
```

#### memory_recent - Recent Memories

Get the last X memories in title-card format.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | int | Number of memories to return (default: 20, max: 200) |
| `verbose` | bool | Include full details (default: false, title-card format) |
| `project` | string | Filter by project (optional) |
| `memory_type` | string | Filter by type (optional) |
| `instance_id` | string | Filter by instance (optional) |
| `include_archived` | bool | Include archived memories (default: false) |

**Example:**
```python
# Get last 20 memories (title-card format)
memory_recent()

# Get last 50 with full details
memory_recent(limit=50, verbose=True)

# Recent memories for a specific project
memory_recent(limit=30, project="my-app")
```

#### memory_archive - Archive Memories

Archive memories (soft delete) with protection for foundational and high-centrality nodes.

**Replaces:** `memory_forget` from v1.x

**SAFETY:** `dry_run=True` by default - returns preview of what would be archived.

Supports both explicit ID lists and filter-based archival. Foundational memories are always protected.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_ids` | list | Explicit list of memory IDs to archive (optional) |
| `older_than_days` | int | Archive memories older than N days (optional) |
| `max_access_count` | int | Archive memories with access_count ≤ N (optional) |
| `memory_type` | string | Filter by memory type (optional) |
| `project` | string | Filter by project (optional) |
| `centrality_protection` | bool | Protect high-centrality memories (default: true) |
| `min_centrality_threshold` | int | In-degree count for protection (default: 5) |
| `dry_run` | bool | Preview only (default: true) |
| `reason` | string | Archival reason for audit trail (optional) |

**Returns (dry_run=True):**
```json
{
  "would_archive": 15,
  "memories": [...],
  "protected": 3,
  "note": "Set dry_run=False to execute"
}
```

**Returns (dry_run=False):**
```json
{
  "archived": 15,
  "memories": [...],
  "protected": 3
}
```

**Example:**
```python
# Preview what would be archived
memory_archive(older_than_days=90, max_access_count=2)

# Archive specific memories (still needs dry_run=False)
memory_archive(memory_ids=[42, 17], reason="Outdated information")

# Execute archival
memory_archive(older_than_days=90, max_access_count=2, dry_run=False, reason="Cleanup")
```

### Knowledge Graph Tools

Build typed relationships between memories. v2.0 simplifies the graph API to just two tools: `memory_link` and `memory_unlink`. Graph traversal is now built into `memory_get` via `traverse=True`.

| Tool | Description |
|------|-------------|
| `memory_link` | Create typed edges between memories |
| `memory_unlink` | Remove edges between memories |

**Removed in v2.0:** `memory_related`, `memory_graph`, `memory_supersede`, `memory_relationship_types` (functionality integrated into `memory_get` and `memory_link`)

#### memory_link - Create Edges

Create a relationship edge between two memories.

**Standard relationship types:**
- `supersedes` - Newer memory replaces older (use `archive_old=True` to archive target)
- `relates_to` - General association
- `derived_from` - This memory came from that one
- `contradicts` - Memories make conflicting claims
- `exemplifies` - This is an example of that concept
- `refines` - Adds detail/nuance to another memory

Custom types are allowed - use descriptive names like `caused_by`, `blocks`, `spawned_by`, etc.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_id` | int | ID of the source memory |
| `target_id` | int | ID of the target memory (edge points TO this) |
| `relation_type` | string | Type of relationship (standard or custom) |
| `strength` | float | Edge weight 0.0-1.0 for weighted traversal (default: 1.0) |
| `bidirectional` | bool | Edge works in both directions (default: false) |
| `metadata` | dict | Extra data to store with edge (optional) |
| `created_by` | string | Instance ID creating this edge (optional) |
| `archive_old` | bool | If true AND relation_type="supersedes", archives target (default: false) |

**Replaces supersede workflow:**
```python
# Old way (v1.x):
# memory_supersede(new_id=42, old_id=17)

# New way (v2.0):
memory_link(source_id=42, target_id=17, relation_type="supersedes", archive_old=True)
```

**Example:**
```python
# Link related memories
memory_link(source_id=42, target_id=17, relation_type="derived_from")

# Create bidirectional association
memory_link(source_id=42, target_id=89, relation_type="relates_to", bidirectional=True)

# Supersede and archive old memory
memory_link(source_id=150, target_id=103, relation_type="supersedes", archive_old=True)
```

#### memory_unlink - Remove Edges

Remove relationship edge(s) between two memories.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_id` | int | ID of the source memory |
| `target_id` | int | ID of the target memory |
| `relation_type` | string | Specific relation to remove (optional - removes ALL if None) |

**Example:**
```python
# Remove specific edge
memory_unlink(source_id=42, target_id=17, relation_type="derived_from")

# Remove ALL edges from 42 to 17
memory_unlink(source_id=42, target_id=17)
```

**Finding related memories in v2.0:**

Use `memory_get` with `include_graph=True` (default behavior):

```python
# Get memory with its immediate connections
result = memory_get(memory_ids=42)
# Returns graph_context with incoming/outgoing edges

# Traverse the graph (replaces old memory_graph)
result = memory_get(memory_ids=42, traverse=True, max_depth=2)
```

### Messaging Tools

Inter-instance messaging with pub/sub support. Replaces the old separate `handoff_send`, `handoff_get`, `handoff_mark_read` tools with a unified action-based interface.

| Tool | Description |
|------|-------------|
| `message` | Unified messaging with actions: send, get, mark_read, mark_unread, subscribe, unsubscribe |

#### message - Unified Messaging

**Actions:**
- `send` - Send message (requires: from_instance, to_instance, content)
- `get` - Get messages (requires: instance_id)
- `mark_read` - Mark read (requires: message_id, instance_id)
- `mark_unread` - Mark unread (requires: message_id)
- `subscribe` - Subscribe to channel (requires: instance_id, channel)
- `unsubscribe` - Unsubscribe (requires: instance_id, channel)

**Message types:**
- `handoff` - Passing context to another instance
- `status` - Status updates
- `question` - Questions for another instance
- `fyi` - For your information
- `context` - Contextual information
- `event` - Event notifications
- `message` - General message (default)

**Delivery:**
- **PostgreSQL:** Uses NOTIFY for real-time delivery
- **SQLite:** Uses polling

**Parameters (vary by action):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | string | One of: send, get, mark_read, mark_unread, subscribe, unsubscribe |
| `instance_id` | string | Instance performing the action (for get/mark_read/subscribe) |
| `from_instance` | string | Sender (for send) |
| `to_instance` | string | Recipient or "all" for broadcast (for send) |
| `content` | string | Message content (for send) |
| `message_type` | string | Type of message (default: "message") |
| `subject` | string | Optional short summary (for send) |
| `channel` | string | Channel name (for send/get/subscribe/unsubscribe) |
| `priority` | int | 0-10, higher = more urgent (for send, default: 0) |
| `unread_only` | bool | Only unread messages (for get, default: true) |
| `limit` | int | Max messages to return (for get, default: 50) |
| `message_id` | int | Message ID (for mark_read/mark_unread) |

**Example:**
```python
# Send a handoff message
message(
    action="send",
    from_instance="engineering",
    to_instance="support",
    content="Completed indexing 42 files. Check memories 167-209.",
    message_type="handoff",
    subject="Code indexing complete",
    priority=5
)

# Get unread messages
message(action="get", instance_id="support")

# Subscribe to a channel
message(action="subscribe", instance_id="support", channel="eng-updates")

# Broadcast to all instances
message(
    action="send",
    from_instance="support",
    to_instance="all",
    content="System maintenance in 1 hour",
    message_type="status",
    channel="system",
    priority=8
)

# Mark message as read
message(action="mark_read", message_id=42, instance_id="support")
```

### Code Indexing Tools

Index source files for natural language search over your codebase. The system creates two linked memories per file:

1. **Prose memory** - LLM-generated description of the code (embedded for semantic search)
2. **Code memory** - The actual source code (stored but NOT embedded, linked via knowledge graph)

This separation is intentional: embedding raw code produces poor semantic matches, but embedding a prose description enables queries like "how does retry logic work?" to find relevant files.

| Tool | Description |
|------|-------------|
| `code_remember_tool` | Index a source file into the palace |

**Note:** `code_recall_tool` from v1.x has been removed. Use `memory_recall` with `project` filter instead.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `code_path` | string | Absolute path to the source file |
| `project` | string | Project name (e.g., "memory-palace") |
| `instance_id` | string | Which instance is indexing (e.g., "engineering") |
| `force` | bool | Re-index even if already indexed (default: false) |

**Example workflow:**

```python
# Index important files
code_remember_tool(
    code_path="/project/src/database.py",
    project="my-app",
    instance_id="engineering"
)

code_remember_tool(
    code_path="/project/src/api/endpoints.py",
    project="my-app",
    instance_id="engineering"
)

# Later, query naturally using memory_recall
memory_recall(query="How does database connection pooling work?", project="my-app")

# Get raw results for manual analysis
memory_recall(query="Show me the retry logic", project="my-app", synthesize=False)
```

### Maintenance Tools

System health and maintenance operations.

| Tool | Description |
|------|-------------|
| `memory_audit` | Health checks for the palace |
| `memory_reembed` | Regenerate embeddings for memories |
| `memory_stats` | Overview statistics |

#### memory_audit - Health Checks

Audit palace health. Checks for duplicates, stale memories, orphan edges, missing embeddings, contradictions, and cross-project auto-links.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `checks` | list | Which checks to run (default: all) |
| `thresholds` | dict | Override thresholds (optional) |
| `project` | string | Filter by project (optional) |
| `limit_per_category` | int | Max results per issue type (default: 20) |

**Valid checks:**
- `duplicates` - Find highly similar memories (potential duplicates)
- `stale` - Find low-access, old, low-centrality memories (foundational exempt)
- `orphan_edges` - Find edges pointing to non-existent memories
- `missing_embeddings` - Find memories without embeddings
- `contradictions` - Find memories that may contradict each other
- `cross_project_auto_links` - Find auto-links spanning projects

**Default thresholds:**
```json
{
  "duplicate_similarity": 0.95,
  "stale_days": 90,
  "stale_max_access": 2,
  "stale_min_centrality": 3
}
```

**Example:**
```python
# Full audit
memory_audit()

# Check only for duplicates and stale memories
memory_audit(checks=["duplicates", "stale"])

# Custom thresholds
memory_audit(
    checks=["duplicates"],
    thresholds={"duplicate_similarity": 0.92}
)

# Project-specific audit
memory_audit(project="my-app")
```

#### memory_reembed - Regenerate Embeddings

Regenerate embeddings for memories. Use `missing_only=True` to backfill memories without embeddings.

**Replaces:** `memory_backfill_embeddings` from v1.x

**SAFETY:** `dry_run=True` by default.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `older_than_days` | int | Re-embed embeddings older than N days (optional) |
| `memory_ids` | list | Explicit list of memory IDs to re-embed (optional) |
| `project` | string | Filter by project (optional) |
| `all_memories` | bool | Re-embed everything (use with caution, default: false) |
| `missing_only` | bool | Only embed memories with NULL embeddings - backfill mode (default: false) |
| `batch_size` | int | Processing batch size (default: 50) |
| `dry_run` | bool | Preview only (default: true) |

**Example:**
```python
# Backfill missing embeddings (preview)
memory_reembed(missing_only=True)

# Execute backfill
memory_reembed(missing_only=True, dry_run=False)

# Re-embed old embeddings
memory_reembed(older_than_days=180, dry_run=False)

# Re-embed specific memories
memory_reembed(memory_ids=[42, 17, 89], dry_run=False)
```

#### memory_stats - System Statistics

Get overview statistics about the memory palace.

**Returns:**
```json
{
  "total_memories": 542,
  "by_type": {"fact": 150, "preference": 42, ...},
  "by_instance": {"engineering": 200, "support": 342},
  "by_project": {"life": 300, "my-app": 200, ...},
  "foundational_count": 15,
  "archived_count": 23,
  "most_accessed": [...],
  "recently_added": [...]
}
```

**Example:**
```python
memory_stats()
```

### Processing Tools

Extract memories from conversation transcripts.

| Tool | Description |
|------|-------------|
| `memory_reflect` | Extract memories from conversation transcripts |

#### memory_reflect - Extract from Transcripts

Extract memories from a conversation transcript using LLM.

**Supported formats:**
- JSONL (JSON Lines)
- TOON (chunked encoding)

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `instance_id` | string | Which instance is reflecting |
| `transcript_path` | string | Path to transcript file |
| `session_id` | string | Optional session ID to link memories to source |
| `dry_run` | bool | Report what would be stored without writing (default: false) |

**Returns:**
```json
{
  "extracted_count": 15,
  "embedded_count": 15,
  "types_breakdown": {"fact": 5, "preference": 3, "insight": 7}
}
```

**Example:**
```python
# Extract from JSONL transcript
memory_reflect(
    instance_id="support",
    transcript_path="/path/to/conversation.jsonl",
    session_id="2024-01-15-morning"
)

# Preview without storing
memory_reflect(
    instance_id="support",
    transcript_path="/path/to/conversation.toon",
    dry_run=True
)
```

**Note:** Converting JSONL to TOON format is done via a CLI utility, not an MCP tool:
```bash
python tools/dump_memories_toon.py input.jsonl output.toon
```

### Named Database Tools (Extension)

Multi-database management for domain partitions (life, work, per-project). Requires the `db_manager` extension.

| Tool | Description |
|------|-------------|
| `memory_list_databases` | List all configured databases with connection status and table counts |
| `memory_register_database` | Register a new database at runtime (auto-derives URL if not provided) |
| `memory_set_default_database` | Change which database is used when no `database=` param is given |
| `memory_current_database` | Show current default database, URL, connection status |

**Enable:** Add `"mcp_server.extensions.db_manager"` to the `"extensions"` list in config.

See [POSTGRES.md](POSTGRES.md) for full named database setup and [mcp_server/extensions/README.md](../mcp_server/extensions/README.md) for extension details.

### Result Enhancement Features

Both `memory_recall` and `memory_get` support enhanced result parameters:

#### Synthesis Parameter

- **`synthesize=false`**: Returns raw memory objects with full content. Best when you need exact wording or are processing with a cloud AI that can handle the full context.

- **`synthesize=true`**: Runs memories through the local LLM (Qwen) to produce a natural language summary. Reduces token usage but takes longer (~1-2 min for large result sets).

For `memory_get`, synthesis is skipped for single memories (pointless to summarize one thing).

#### Graph Context Parameters

- **`include_graph=bool`** (default `true`): Include graph context (immediate incoming/outgoing edges) for retrieved memories. Helps understand how memories connect without needing separate graph traversal calls.

- **`graph_top_n=int`** (default `5`, only for `memory_recall`): Number of top-ranked results to fetch graph context for. Clamped to the query limit. Prevents returning massive amounts of graph data for searches with many results.

- **`graph_depth=int`** (default `1`): How many hops to follow in graph context (1-3). Use 2 for bootstrap/startup scenarios.

**Key difference:** `memory_recall` limits graph context to top N results (performance consideration for broad searches), while `memory_get` includes graph context for ALL requested memories (intentional targeted fetches where full context is desired).

**Graph context format:**
```json
{
  "graph_context": {
    "nodes": {
      "42": "authentication system",
      "17": "JWT token handling",
      "89": "session management"
    },
    "edges": [
      {
        "source": 42,
        "target": 17,
        "type": "relates_to",
        "strength": 1.0
      }
    ]
  }
}
```

#### TOON Encoding

When `synthesize=False`, results are returned in TOON (Thoughtful Object Observation Notation) encoding for efficient token usage while preserving full content fidelity.

### Example Usage Patterns

**Storing a memory:**
```
"Remember that the API endpoint changed from /v1/users to /v2/users on 2024-01-15"
```

**Recalling memories:**
```
"What do you remember about API changes?"
```

**Retrieving specific memories by ID:**
```python
# Raw with graph context (default)
memory_get(memory_ids=[167, 168, 169])

# Synthesized summary with graph context
memory_get(memory_ids=[167, 168, 169], synthesize=True)

# Without graph context (faster, less context)
memory_get(memory_ids=[167, 168, 169], include_graph=False)

# Traverse from a starting point
memory_get(memory_ids=42, traverse=True, max_depth=2)
```

**Searching with graph context awareness:**
```python
# Default: returns top 5 results with their graph context
memory_recall(query="API authentication approach")

# Get more graph context (top 10 results)
memory_recall(query="API authentication", limit=20, graph_top_n=10)

# Disable graph context for speed
memory_recall(query="API authentication", include_graph=False)

# Get raw memories for detailed analysis
memory_recall(query="authentication", synthesize=False)
```

**Multi-project memories:**
```python
# Store a memory in multiple projects
memory_remember(
    instance_id="engineering",
    memory_type="architecture",
    content="All services use standard retry with exponential backoff",
    project=["my-app", "shared-patterns"],
    foundational=True
)

# Search across projects
memory_recall(query="retry patterns", project=["my-app", "shared-patterns"])
```

**Reflecting on transcripts:**
```
"Reflect on today's conversation and extract any important memories"
```

**Handoff between instances:**
```python
# Desktop → Code
message(
    action="send",
    from_instance="support",
    to_instance="engineering",
    content="Please index the authentication module files",
    message_type="handoff",
    priority=5
)

# Code checks messages
message(action="get", instance_id="engineering")

# Code completes work and responds
message(
    action="send",
    from_instance="engineering",
    to_instance="support",
    content="Indexed 12 files. See memories 200-212 for authentication patterns.",
    message_type="handoff"
)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_PALACE_DATA_DIR` | Data directory | `~/.memory-palace` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `MEMORY_PALACE_EMBEDDING_MODEL` | Embedding model name | Auto-detected |
| `MEMORY_PALACE_LLM_MODEL` | LLM model for reflection | Auto-detected |
| `MEMORY_PALACE_INSTANCE_ID` | Default instance ID | `unknown` |
| `MEMORY_PALACE_DATABASE_URL` | Database connection URL (overrides config) | None |
| `MEMORY_PALACE_NOTIFY_COMMAND` | Post-send notification command template | None |
| `MEMORY_PALACE_INSTANCE_ROUTES` | Instance route map (JSON string) for push notifications | `{}` |

### Config File

Configuration is loaded from `~/.memory-palace/config.json`:

```json
{
  "database": {
    "type": "postgres",
    "url": "postgresql://localhost:5432/memory_palace"
  },
  "databases": {
    "default": {"type": "postgres", "url": "postgresql://localhost:5432/memory_palace"},
    "life":    {"type": "postgres", "url": "postgresql://localhost:5432/memory_palace_life"}
  },
  "default_database": "default",
  "ollama_url": "http://localhost:11434",
  "embedding_model": null,
  "llm_model": null,
  "extensions": ["mcp_server.extensions.db_manager"],
  "instances": ["support", "engineering", "analytics"],
  "notify_command": null,
  "instance_routes": {}
}
```

The `databases` and `default_database` keys are optional. If absent, the system uses the legacy single `database` key. See [POSTGRES.md](POSTGRES.md) for named database setup.

Environment variables override config file values.

### Instance Routes (Push Notifications)

The `instance_routes` config enables real-time push notifications when messages are sent between instances. Each route maps an instance ID to an OpenClaw gateway URL and auth token:

```json
{
  "instance_routes": {
    "support": {
      "gateway": "http://localhost:18789",
      "token": "your-gateway-token-here"
    },
    "engineering": {
      "gateway": "http://localhost:18790",
      "token": "your-other-token-here"
    }
  }
}
```

**How it works:**
- When a message is sent, the palace looks up the recipient's route
- If found, it POSTs to `{gateway}/hooks/wake` with a wake text and mode
- Priority >= 5 uses `"mode": "now"` (immediate wake), lower uses `"next-heartbeat"`
- Broadcast messages (`to_instance: "all"`) wake all routed instances except the sender
- Wake failures are logged but never break message delivery

**Prerequisites:** Each target OpenClaw gateway must have external hooks enabled:
```json
{
  "hooks": {
    "enabled": true,
    "token": "shared-secret"
  }
}
```

The existing `notify_command` (shell hook) continues to work as a fallback or alongside instance routes. Both mechanisms can fire independently.

### Model Configuration

See [models.md](models.md) for detailed model selection guide.

## OpenClaw Integration

Memory Palace can also run as a **native OpenClaw plugin**, registering all 13 tools directly with the OpenClaw gateway. This eliminates MCP protocol overhead and enables real-time pubsub wake — when a message arrives, the agent wakes automatically instead of waiting for the next poll.

See [OPENCLAW.md](OPENCLAW.md) for the full plugin guide including installation, bridge protocol, and pubsub wake chain.

## Troubleshooting

### "Ollama not found"

Ensure Ollama is installed and in your PATH:
```bash
ollama --version
```

### "Model not found"

Download the required model:
```bash
ollama pull nomic-embed-text
ollama pull qwen3:1.7b
```

### "CUDA out of memory"

Your VRAM is insufficient for the configured models. Options:
1. Use smaller models (see [models.md](models.md))
2. Ensure only one model runs at a time
3. Close other GPU-intensive applications

### Slow embedding/generation

If using CPU inference, performance will be significantly slower. Consider:
1. Using an NVIDIA GPU
2. Using smaller models
3. Batching operations during off-hours

### Database Migration

If upgrading from v1.x to v2.0, the migration will run automatically on first launch. The migration:
- Preserves all existing memories and relationships
- Adds new v2.0 features (multi-project support, channels, etc.)
- Is safe and reversible (backups recommended)

## Architecture

```
memory-palace/
├── mcp_server/              # MCP server package
│   ├── server.py            # Server entry point
│   ├── toon_wrapper.py      # TOON response encoding
│   ├── tools/               # 13 core tool implementations
│   └── extensions/          # Extension tools (db_manager, switch_db)
├── memory_palace/           # Core library
│   ├── models_v3.py         # SQLAlchemy models (Memory, MemoryEdge, Message)
│   ├── database_v3.py       # Named engine registry (SQLite / PostgreSQL)
│   ├── bridge.py            # OpenClaw bridge subprocess (NDJSON protocol)
│   ├── embeddings.py        # Ollama embedding client
│   ├── llm.py               # LLM integration (synthesis, classification)
│   ├── config_v2.py         # Configuration with named databases + auto-link
│   ├── services/            # Business logic layer
│   │   ├── memory_service.py      # remember, recall, archive, stats
│   │   ├── graph_service.py       # link, unlink, traverse
│   │   ├── message_service.py     # pub/sub messaging
│   │   ├── maintenance_service.py # audit, reembed, cleanup
│   │   ├── code_service.py        # code indexing + retrieval
│   │   └── reflection_service.py  # transcript extraction
│   └── migrations/          # Schema migration scripts
├── openclaw_plugin/         # Native OpenClaw plugin (TypeScript)
│   ├── src/index.ts         # Plugin registration + session registry + wake dispatch
│   ├── src/bridge.ts        # PalaceBridge class (NDJSON over stdin/stdout)
│   └── src/tools/           # 13 tool definitions mapped to bridge methods
├── setup/                   # Setup wizard
├── extensions/              # Optional extensions (Moltbook gateway, TOON converter)
├── docs/                    # Documentation
└── tests/                   # Test suite
```

## Support

For issues and feature requests, please open a GitHub issue.
