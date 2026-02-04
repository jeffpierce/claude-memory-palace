# Claude Memory Palace - Documentation

A persistent memory system for Claude instances, enabling semantic search across conversations, facts, and insights.

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

Once configured, Claude will have access to the following memory tools:

### Core Tools

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a new memory |
| `memory_recall` | Search memories using semantic search (supports `synthesize` and graph context) |
| `memory_forget` | Archive a memory (soft delete) |
| `memory_get` | Retrieve memories by ID (supports `synthesize` and graph context) |
| `memory_stats` | Get overview of memory system |

#### Result Enhancement Parameters

Both `memory_recall` and `memory_get` support enhanced result parameters:

##### Synthesis Parameter

- **`synthesize=false`** (default for `memory_get`): Returns raw memory objects with full content. Best when you need exact wording or are processing with a cloud AI that can handle the full context.

- **`synthesize=true`** (default for `memory_recall`): Runs memories through the local LLM (Qwen) to produce a natural language summary. Reduces token usage but takes longer (~1-2 min for large memories).

For `memory_get`, synthesis is skipped for single memories (pointless to summarize one thing).

##### Graph Context Parameters

- **`include_graph=bool`** (default `true`): Include depth-1 graph context (immediate incoming/outgoing edges) for retrieved memories. Helps understand how memories connect without needing separate graph traversal calls.

- **`graph_top_n=int`** (default `5`, only for `memory_recall`): Number of top-ranked results to fetch graph context for. Clamped to the query limit. This prevents returning massive amounts of graph data for searches with many results.

**Key difference:** `memory_recall` limits graph context to top N results (performance consideration for broad searches), while `memory_get` includes graph context for ALL requested memories (intentional targeted fetches where full context is desired).

**Graph context format:**
```json
{
  "graph_context": {
    "memory_id": {
      "outgoing": [
        {
          "target_id": 42,
          "target_subject": "Related Memory",
          "relation_type": "relates_to",
          "strength": 1.0
        }
      ],
      "incoming": [
        {
          "source_id": 17,
          "source_subject": "Source Memory",
          "relation_type": "derived_from",
          "strength": 1.0
        }
      ]
    }
  }
}
```

### Reflection Tools

| Tool | Description |
|------|-------------|
| `memory_reflect` | Extract memories from conversation transcripts |
| `memory_backfill_embeddings` | Generate embeddings for memories that don't have them |
| `convert_jsonl_to_toon` | Convert JSONL transcripts to chunked TOON format |

### Code Retrieval Tools

Index source files for natural language search over your codebase. The system creates two linked memories per file:

1. **Prose memory** - LLM-generated description of the code (embedded for semantic search)
2. **Code memory** - The actual source code (stored but NOT embedded, linked via knowledge graph)

This separation is intentional: embedding raw code produces poor semantic matches, but embedding a prose description of what the code does enables queries like "how does retry logic work?" to find relevant files.

| Tool | Description |
|------|-------------|
| `code_remember_tool` | Index a source file into the palace |
| `code_recall_tool` | Search indexed code using natural language |

**`code_remember_tool` parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `code_path` | string | Absolute path to the source file |
| `project` | string | Project name (e.g., "memory-palace") |
| `instance_id` | string | Which instance is indexing (e.g., "code") |
| `force` | bool | Re-index even if already indexed (default: false) |

**`code_recall_tool` parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Natural language search query |
| `project` | string | Filter by project (optional) |
| `synthesize` | bool | If true, LLM answers directly. If false, returns raw matches. (default: true) |
| `limit` | int | Maximum files to return (default: 5) |

**Example workflow:**

```python
# Index important files
code_remember_tool(code_path="/project/src/database.py", project="my-app", instance_id="code")
code_remember_tool(code_path="/project/src/api/endpoints.py", project="my-app", instance_id="code")

# Later, query naturally
code_recall_tool(query="How does database connection pooling work?")
# Returns: synthesized answer with source citations

code_recall_tool(query="Show me the retry logic", synthesize=False)
# Returns: raw {prose, code, similarity} pairs for manual analysis
```

### Handoff Tools

| Tool | Description |
|------|-------------|
| `handoff_send` | Send message to another Claude instance |
| `handoff_get` | Check for messages from other instances |
| `handoff_mark_read` | Mark a handoff message as read |

### Knowledge Graph Tools

Build typed relationships between memories. Enables graph traversal, supersession tracking, and centrality-weighted retrieval.

| Tool | Description |
|------|-------------|
| `memory_link` | Create an edge between two memories |
| `memory_unlink` | Remove an edge between memories |
| `memory_related` | Get memories connected to a given memory (1 hop) |
| `memory_graph` | Traverse the knowledge graph from a starting point |
| `memory_supersede` | Mark a new memory as replacing an old one |
| `memory_relationship_types` | List available relationship types |

**Standard relationship types:**

| Type | Description |
|------|-------------|
| `relates_to` | General association (often bidirectional) |
| `derived_from` | This memory came from processing that one |
| `contradicts` | Memories make conflicting claims |
| `exemplifies` | This is an example of that concept |
| `refines` | Adds detail/nuance to another memory |
| `supersedes` | Newer memory replaces older (archives the old one) |

Custom types are allowed - use descriptive names like `caused_by`, `blocks`, `spawned_by`, etc.

**Example:**

```python
# Link related memories
memory_link(source_id=42, target_id=17, relation_type="derived_from")

# Find what's connected
memory_related(memory_id=42)  # Returns incoming and outgoing edges

# Traverse the graph
memory_graph(start_id=42, max_depth=2)  # BFS traversal up to 2 hops
```

### Example Usage

**Storing a memory:**
```
"Remember that the API endpoint changed from /v1/users to /v2/users on 2024-01-15"
```

**Recalling memories:**
```
"What do you remember about API changes?"
```

**Retrieving specific memories by ID:**
```
# Raw with graph context (default)
memory_get(memory_ids=[167, 168, 169], synthesize=False)
# Returns: {memories: [...], graph_context: {...}}

# Synthesized with graph context
memory_get(memory_ids=[167, 168, 169], synthesize=True)
# Returns: {summary: "...", memory_ids: [...], graph_context: {...}}

# Without graph context (faster, less context)
memory_get(memory_ids=[167, 168, 169], include_graph=False)
```

**Searching with graph context awareness:**
```
# Default: returns top 5 results with their graph context
memory_recall(query="API authentication approach")

# Get more graph context (top 10 results)
memory_recall(query="API authentication", limit=20, graph_top_n=10)

# Disable graph context for speed
memory_recall(query="API authentication", include_graph=False)
```

**Reflecting on transcripts:**
```
"Reflect on today's conversation and extract any important memories"
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

### Config File

Configuration is loaded from `~/.memory-palace/config.json`:

```json
{
  "ollama_url": "http://localhost:11434",
  "embedding_model": null,
  "llm_model": null,
  "db_path": "~/.memory-palace/memories.db",
  "instances": ["desktop", "code", "web"]
}
```

Environment variables override config file values.

### Model Configuration

See [models.md](models.md) for detailed model selection guide.

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
ollama pull qwen2.5:7b
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

## Architecture

```
memory-palace/
├── mcp_server/
│   ├── server.py          # MCP server entry point
│   └── tools/             # Tool implementations
│       ├── remember.py
│       ├── recall.py
│       ├── forget.py
│       ├── reflect.py
│       └── ...
├── memory_palace/
│   ├── config.py          # Configuration handling
│   ├── models.py          # SQLAlchemy models
│   ├── database.py        # Database connection
│   ├── embeddings.py      # Ollama embedding client
│   └── llm.py             # LLM integration
├── setup/
│   └── first_run.py       # Setup wizard
├── install.sh             # macOS/Linux installer
├── install.bat            # Windows installer (cmd)
├── install.ps1            # Windows installer (PowerShell)
└── docs/
    ├── README.md          # This file
    └── models.md          # Model guide
```

## Support

For issues and feature requests, please open a GitHub issue.
