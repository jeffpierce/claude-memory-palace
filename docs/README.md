# Memory Palace - Documentation

A persistent memory system for any MCP-compatible AI, enabling semantic search and knowledge graph traversal across conversations, facts, and insights.

## Quick Start

### Prerequisites

1. **Python 3.10+** - Required for the MCP server
2. **Ollama** - For local embedding and LLM models
   - Download from: https://ollama.ai/download
3. **~1.3GB disk space** for default models (GPU optional — CPU works, just slower)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jeffpierce/claude-memory-palace.git
   cd claude-memory-palace
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
   - Detect your hardware
   - Download default models (~1.3GB total)
   - Optionally recommend upgraded models if a GPU is detected

### Configure Your MCP Client

Add the following to your MCP client's configuration file:

```json
{
  "mcpServers": {
    "memory-palace": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/claude-memory-palace",
      "env": {
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

**Config file locations by client:**

| Client | Windows | macOS / Linux |
|--------|---------|---------------|
| Claude Desktop | `%APPDATA%\Claude\claude_desktop_config.json` | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Cursor | Settings → MCP Servers | Settings → MCP Servers |
| Claude Code | `~/.claude/claude_desktop_config.json` | `~/.claude/claude_desktop_config.json` |

The installer scripts auto-detect installed clients and configure them automatically.

## Usage

Once configured, your AI will have access to the following memory tools:

### Core Tools

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a new memory |
| `memory_recall` | Search memories using semantic search (supports `synthesize` param) |
| `memory_forget` | Archive a memory (soft delete) |
| `memory_get` | Retrieve memories by ID (supports `synthesize` param) |
| `memory_stats` | Get overview of memory system |

#### Synthesis Parameter

Both `memory_recall` and `memory_get` support a `synthesize` parameter:

- **`synthesize=false`** (default for `memory_get`): Returns raw memory objects with full content. Best when you need exact wording or are processing with a cloud AI that can handle the full context.

- **`synthesize=true`** (default for `memory_recall`): Runs memories through the local LLM (Qwen) to produce a natural language summary. Reduces token usage but takes longer (~1-2 min for large memories).

For `memory_get`, synthesis is skipped for single memories (pointless to summarize one thing).

### Reflection Tools

| Tool | Description |
|------|-------------|
| `memory_reflect` | Extract memories from conversation transcripts |
| `memory_backfill_embeddings` | Generate embeddings for memories that don't have them |
| `convert_jsonl_to_toon` | Convert JSONL transcripts to chunked TOON format |

### Handoff Tools

| Tool | Description |
|------|-------------|
| `handoff_send` | Send message to another Claude instance |
| `handoff_get` | Check for messages from other instances |
| `handoff_mark_read` | Mark a handoff message as read |

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
# Raw (full content, fast)
memory_get(memory_ids=[167, 168, 169], synthesize=False)

# Synthesized (natural language summary, slower)
memory_get(memory_ids=[167, 168, 169], synthesize=True)
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

The defaults (nomic-embed-text + qwen3:1.7b) work everywhere, including CPU-only machines. If you have a dedicated GPU and want better quality, see [models.md](models.md) for optional upgrades.

## Troubleshooting

### "Ollama not found"

Ensure Ollama is installed and in your PATH:
```bash
ollama --version
```

### "Model not found"

Download the default models manually:
```bash
ollama pull nomic-embed-text
ollama pull qwen3:1.7b
```

### "CUDA out of memory"

If you upgraded to larger models and hit VRAM limits:
1. Switch back to defaults (nomic-embed-text + qwen3:1.7b) — they always work
2. Ensure only one model runs at a time (Ollama swaps automatically)
3. Close other GPU-intensive applications
4. See [models.md](models.md) for VRAM requirements per model

### Slow performance on CPU

The default models are chosen to be usable on CPU. If you're experiencing slow performance:
1. Stick with the defaults — qwen3:1.7b is responsive even on CPU
2. Larger models (8B+) are significantly slower without a GPU
3. Embedding is faster than LLM inference; recall should still feel snappy

## Architecture

```
claude-memory-palace/
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

## Further Reading

- [Architecture & Vision](architecture.md) — Why Memory Palace exists, what problems it solves, and the scaling path from personal SQLite to enterprise PostgreSQL clusters
- [Use Cases](use-cases.md) — Real-world examples: personal memory, team knowledge sharing, agent swarm coordination, sovereign enterprise deployment
- [Model Guide](models.md) — Default models and optional GPU upgrades

## Support

For issues and feature requests, please open a GitHub issue.
