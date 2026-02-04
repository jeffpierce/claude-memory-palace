# Memory Palace

Persistent semantic memory for Claude instances. Store facts, decisions, insights, and context across conversations. Search by meaning, not just keywords.

## Features

- **Semantic Search** - Find memories by meaning using local embedding models
- **Memory Types** - Organize memories as facts, decisions, architecture, gotchas, solutions, and more
- **Transcript Reflection** - Automatically extract memories from conversation logs
- **Multi-Instance Support** - Share memories across Claude Desktop, Claude Code, and web
- **Local Processing** - All embeddings and extraction run locally via Ollama
- **MCP Integration** - Works natively with Claude Desktop's MCP protocol

## Installation

See [docs/README.md](docs/README.md) for detailed installation instructions.

**Quick start:**

```bash
# Clone repository
git clone https://github.com/jeffpierce/memory-palace.git
cd memory-palace

# Install (choose one method)
pip install -e .                    # Editable install from pyproject.toml
# OR use the installer scripts:
# Windows: install.bat or install.ps1
# macOS/Linux: ./install.sh

# Run setup wizard (detects GPU, downloads models)
python -m setup.first_run
```

## Requirements

- Python 3.10+
- Ollama (https://ollama.ai)
- NVIDIA GPU with 4GB+ VRAM (recommended)

## Model Selection

Claude Memory Palace automatically selects models based on your available VRAM:

| VRAM | Embedding | LLM | Quality |
|------|-----------|-----|---------|
| 4-6GB | nomic-embed-text | qwen2.5:7b | Good |
| 8-12GB | snowflake-arctic-embed | qwen3:8b | Better |
| 16GB+ | sfr-embedding-mistral | qwen3:14b | Best |

See [docs/models.md](docs/models.md) for detailed model information.

## Usage

Once configured with Claude Desktop, use natural language:

```
"Remember that the database migration requires downtime"
"What do we know about the API rate limits?"
"Reflect on this conversation and save important decisions"
```

## Tools

### Memory Tools

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a new memory |
| `memory_recall` | Semantic search across memories (supports `synthesize` param and graph context) |
| `memory_forget` | Archive a memory |
| `memory_reflect` | Extract memories from transcripts |
| `memory_stats` | Memory system overview |
| `memory_get` | Retrieve memories by ID (supports `synthesize` param and graph context) |
| `memory_backfill_embeddings` | Generate embeddings for memories without them |

#### Result Enhancement Features

Both `memory_recall` and `memory_get` support:

- **`synthesize=true/false`** - Control whether results are returned as raw objects or processed through the local LLM for natural language summaries.

- **`include_graph=true/false`** - Include depth-1 graph context (incoming/outgoing edges) in results. Default: `true`
  - `memory_recall`: Graph context included for top N results (controlled by `graph_top_n`, default 5)
  - `memory_get`: Graph context included for ALL retrieved memories (intentional targeted fetches)

Graph context returns edge information showing how memories connect:
```json
{
  "graph_context": {
    "memory_id": {
      "outgoing": [{"target_id": 42, "target_subject": "...", "relation_type": "relates_to", "strength": 1.0}],
      "incoming": [{"source_id": 17, "source_subject": "...", "relation_type": "derived_from", "strength": 1.0}]
    }
  }
}
```

### Code Retrieval Tools

Index source files for natural language search. The system creates two linked memories per file: a prose description (embedded for semantic search) and the actual source code (stored but not embedded). Queries hit the prose, then graph traversal retrieves the code.

| Tool | Description |
|------|-------------|
| `code_remember_tool` | Index a source file into the palace |
| `code_recall_tool` | Search indexed code using natural language |

**Example usage:**

```python
# Index a file
code_remember_tool(
    code_path="/path/to/database.py",
    project="my-project",
    instance_id="code"
)

# Query later
code_recall_tool(query="How does the database connection work?")
code_recall_tool(query="Show me the retry logic", synthesize=False)
```

### Handoff Tools

| Tool | Description |
|------|-------------|
| `handoff_send` | Send message to another Claude instance |
| `handoff_get` | Check for messages from other instances |
| `handoff_mark_read` | Mark a handoff message as read |

### Knowledge Graph Tools

Build relationships between memories for richer retrieval and traversal.

| Tool | Description |
|------|-------------|
| `memory_link` | Create an edge between two memories |
| `memory_unlink` | Remove an edge between memories |
| `memory_related` | Get memories connected to a given memory (1 hop) |
| `memory_graph` | Traverse the knowledge graph from a starting point |
| `memory_supersede` | Mark a new memory as replacing an old one |
| `memory_relationship_types` | List available relationship types |

**Standard relationship types:** `relates_to`, `derived_from`, `contradicts`, `exemplifies`, `refines`, `supersedes`

## Architecture

```
memory-palace/
├── mcp_server/              # MCP server package
│   ├── server.py            # Server entry point
│   └── tools/               # Tool implementations
├── memory_palace/           # Core library
│   ├── config.py            # Configuration handling
│   ├── database.py          # SQLAlchemy database
│   ├── models.py            # Data models
│   ├── embeddings.py        # Ollama embedding client
│   └── llm.py               # LLM integration
├── setup/                   # Setup utilities
│   └── first_run.py         # Setup wizard
└── docs/                    # Documentation
```

## Configuration

Configuration is loaded from `~/.memory-palace/config.json` with environment variable overrides.

**Environment variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_PALACE_DATA_DIR` | Data directory | `~/.memory-palace` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `MEMORY_PALACE_EMBEDDING_MODEL` | Embedding model | Auto-detected |
| `MEMORY_PALACE_LLM_MODEL` | LLM for reflection | Auto-detected |
| `MEMORY_PALACE_INSTANCE_ID` | Default instance ID | `unknown` |

**Config file (`~/.memory-palace/config.json`):**

```json
{
  "ollama_url": "http://localhost:11434",
  "embedding_model": null,
  "llm_model": null,
  "db_path": "~/.memory-palace/memories.db",
  "instances": ["desktop", "code", "web"]
}
```

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
