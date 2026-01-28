# Claude Memory Palace

Persistent semantic memory for AI — any model, any provider, your hardware, your data.

## Why

Every AI session starts as a blank slate. Context windows are finite. Sessions end, knowledge dies. Vendor-specific memory features lock you into one provider.

Memory Palace fixes this by separating memory from the model:

- **Cross-session** — AI remembers across conversations
- **Cross-model** — Switch from Claude to GPT to local Qwen; same memories, zero migration
- **Cross-provider** — Cancel any subscription, keep all your context
- **No cloud dependency** — Runs entirely on your hardware with local models
- **No vendor lock-in** — Open protocol (MCP), standard database (SQLite/PostgreSQL), your data
- **Data sovereignty** — `SELECT * FROM memories` whenever you want
- **Agent coordination** — Handoff system enables decentralized swarms without a controller bottleneck

The context window is working memory. Memory Palace is long-term storage. That's how brains work.

> Read the full [Architecture & Vision](docs/architecture.md) document for the technical deep-dive, or see [Use Cases](docs/use-cases.md) for real-world examples from personal use to enterprise agent fleets.

## Features

- **Semantic Search** — Find memories by meaning using local embedding models
- **Memory Types** — Organize memories as facts, decisions, architecture, gotchas, solutions, and more
- **Transcript Reflection** — Automatically extract memories from conversation logs
- **Multi-Instance Support** — Share memories across Claude Desktop, CLI, web, and other AI tools
- **Handoff Messages** — AI instances can send messages to each other through the memory store
- **Local Processing** — All embeddings and extraction run locally via Ollama
- **MCP Integration** — Works with any MCP-compatible AI client

## Scaling

```
SQLite (personal)  →  PostgreSQL (team)  →  PostgreSQL cluster (enterprise)
     Same MCP API          Same MCP API             Same MCP API
```

| Tier | Backend | Agents | Use Case |
|------|---------|--------|----------|
| Personal | SQLite | 1–10 | Individual dev, local AI instances |
| Team | PostgreSQL + pgvector | 10–100 | Shared team knowledge |
| Enterprise | PostgreSQL cluster | 500–10,000+ | Agent swarm orchestration |

See [Architecture](docs/architecture.md) for the full scaling path and [Use Cases](docs/use-cases.md) for examples at each tier.

## Quick Start

```bash
# Clone repository
git clone https://github.com/jeffpierce/claude-memory-palace.git
cd claude-memory-palace

# Install
pip install -e .

# Run setup wizard (detects GPU, downloads models)
python -m setup.first_run
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) (for local embeddings and LLM)
- NVIDIA GPU with 4GB+ VRAM (recommended; CPU fallback available)

## Model Selection

Models are auto-detected based on available VRAM:

| VRAM | Embedding | LLM | Quality |
|------|-----------|-----|---------|
| 4–6GB | nomic-embed-text | qwen2.5:7b | Good |
| 8–12GB | snowflake-arctic-embed | qwen3:8b | Better |
| 16GB+ | sfr-embedding-mistral | qwen3:14b | Best |

See [Model Selection Guide](docs/models.md) for details.

## Tools

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a new memory |
| `memory_recall` | Semantic search across memories |
| `memory_forget` | Archive a memory (soft delete) |
| `memory_reflect` | Extract memories from conversation transcripts |
| `memory_stats` | Memory system overview |
| `memory_get` | Retrieve specific memories by ID |
| `memory_backfill_embeddings` | Regenerate embeddings (e.g., after model change) |
| `handoff_send` | Send message to another AI instance |
| `handoff_get` | Check for messages from other instances |
| `handoff_mark_read` | Mark a handoff message as read |

Both `memory_recall` and `memory_get` support `synthesize=true/false` to control whether results are returned raw or summarized by the local LLM.

## Usage

Once configured with any MCP-compatible client:

```
"Remember that the database migration requires downtime"
"What do we know about the API rate limits?"
"Reflect on this conversation and save important decisions"
```

## Configuration

### Claude Desktop

Add to your MCP configuration:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_PALACE_DATA_DIR` | Data directory | `~/.memory-palace` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `MEMORY_PALACE_EMBEDDING_MODEL` | Embedding model | Auto-detected |
| `MEMORY_PALACE_LLM_MODEL` | LLM for reflection | Auto-detected |
| `MEMORY_PALACE_INSTANCE_ID` | Default instance ID | `unknown` |

### Config File

`~/.memory-palace/config.json`:

```json
{
  "ollama_url": "http://localhost:11434",
  "embedding_model": null,
  "llm_model": null,
  "db_path": "~/.memory-palace/memories.db",
  "instances": ["desktop", "code", "web"]
}
```

## Architecture

```
claude-memory-palace/
├── mcp_server/              # MCP server (protocol layer)
│   ├── server.py            # Server entry point
│   └── tools/               # Tool implementations
├── memory_palace/           # Core library
│   ├── config.py            # Configuration
│   ├── database.py          # SQLAlchemy database
│   ├── models.py            # Data models
│   ├── embeddings.py        # Ollama embedding client
│   └── llm.py               # LLM integration
├── setup/                   # Setup utilities
│   └── first_run.py         # Setup wizard
├── docs/
│   ├── README.md            # Detailed installation & usage
│   ├── architecture.md      # Architecture & vision
│   ├── use-cases.md         # Real-world use case examples
│   └── models.md            # Model selection guide
└── installer/               # Platform-specific installers
```

## Documentation

- [Installation & Usage](docs/README.md) — Detailed setup guide
- [Architecture & Vision](docs/architecture.md) — Why this exists and where it's going
- [Use Cases](docs/use-cases.md) — Personal, team, and enterprise examples
- [Model Selection](docs/models.md) — GPU/VRAM guide for choosing models

## License

MIT License — see [LICENSE](LICENSE) for details.
