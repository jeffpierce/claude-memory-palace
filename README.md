# Memory Palace

Persistent semantic memory for AI — any model, any provider, your hardware, your data.

## Why

Every AI session starts as a blank slate. Context windows are finite. Sessions end, knowledge dies. Vendor-specific memory features lock you into one provider.

Memory Palace fixes this by separating memory from the model:

- **Cross-session** — AI remembers across conversations
- **Cross-model** — Switch from Claude to GPT to local Llama; same memories, zero migration
- **Cross-provider** — Cancel any subscription, keep all your context
- **No cloud dependency** — Runs entirely on your hardware with local models
- **No vendor lock-in** — Open protocol (MCP), standard database (SQLite/PostgreSQL), your data
- **Data sovereignty** — `SELECT * FROM memories` whenever you want
- **Knowledge graph** — Memories aren't just searchable, they're connected. Traverse relationships between decisions, components, and concepts.
- **Agent coordination** — Handoff system enables decentralized swarms without a controller bottleneck

The context window is working memory. Memory Palace is long-term storage. That's how brains work.

> Read the full [Architecture & Vision](docs/architecture.md) document for the technical deep-dive, or see [Use Cases](docs/use-cases.md) for real-world examples from personal use to enterprise agent fleets.

## Features

- **Semantic Search** — Find memories by meaning using local embedding models
- **Knowledge Graph** — Typed relationships between memories with graph traversal (`memory_graph`, `memory_link`, `memory_related`)
- **Memory Types** — Organize memories as facts, decisions, architecture, gotchas, solutions, and more
- **Transcript Reflection** — Automatically extract memories from conversation logs
- **Multi-Instance Support** — Share memories across any MCP-compatible AI tools
- **Handoff Messages** — AI instances can send messages to each other through the memory store
- **Local Processing** — All embeddings and extraction run locally via Ollama
- **MCP Integration** — Works with any MCP-compatible client (Claude Desktop, Cursor, Windsurf, Claude Code, custom agents, etc.)

## Scaling

Both backends ship today. SQLite for zero-config personal use, PostgreSQL for everything bigger. Switch with a one-line config change — no code changes, no data migration tools.

| Tier | Backend | Agents | Use Case | Status |
|------|---------|--------|----------|--------|
| Personal | SQLite | 1–10 | Individual dev, local AI instances | ✅ Shipping |
| Team | PostgreSQL + pgvector | 10–100 | Shared team knowledge | ✅ Shipping |
| Enterprise | PostgreSQL cluster | 500–10,000+ | Agent swarm orchestration | ✅ Shipping |

See [Architecture](docs/architecture.md) for the full scaling path and [Use Cases](docs/use-cases.md) for examples at each tier.

## Quick Start

### Using Installers (Recommended)

Platform-specific installers handle everything — Python, Ollama, models, and MCP client configuration:

| Platform | Command |
|----------|---------|
| **macOS / Linux** | `./install.sh` |
| **Windows (PowerShell)** | `.\install.ps1` |
| **Windows (Command Prompt)** | `install.bat` |

The installer will:
1. Detect your hardware (GPU, VRAM)
2. Install Ollama and download default models (~1.3GB)
3. Auto-configure any detected MCP clients (Claude Desktop, Cursor, Windsurf, etc.)

### Manual Install

```bash
# Clone repository
git clone https://github.com/jeffpierce/claude-memory-palace.git
cd claude-memory-palace

# Create virtual environment and install
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .

# Run setup wizard
python -m setup.first_run
```

Then add to your MCP client's configuration:

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
| Cursor | `%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json` | `~/.config/Cursor/...` |
| Windsurf | `%APPDATA%\Windsurf\...` | `~/.config/Windsurf/...` |
| Claude Code | `~/.claude/claude_desktop_config.json` | `~/.claude/claude_desktop_config.json` |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) (for local embeddings and LLM)
- ~1.3GB disk space for default models (GPU optional — CPU works, just slower)

## Models

Memory Palace works out of the box with minimal hardware. The defaults are deliberately small — they run on laptops, old desktops, even CPU-only machines:

| Component | Default Model | Size | Notes |
|-----------|--------------|------|-------|
| Embeddings | nomic-embed-text | ~300MB | Semantic search vectors |
| LLM | qwen3:1.7b | ~1GB | Synthesis, classification, extraction |

**That's it.** ~1.3GB total download. No GPU required (just slower on CPU). The setup wizard installs these automatically.

If you have a dedicated GPU and want better extraction quality, see the [Model Upgrade Guide](docs/models.md) for optional upgrades based on your available VRAM.

## Tools

### Core Memory

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a new memory |
| `memory_recall` | Semantic search across memories |
| `memory_forget` | Archive a memory (soft delete) |
| `memory_reflect` | Extract memories from conversation transcripts |
| `memory_stats` | Memory system overview |
| `memory_get` | Retrieve specific memories by ID |
| `memory_backfill_embeddings` | Regenerate embeddings (e.g., after model change) |

### Knowledge Graph

| Tool | Description |
|------|-------------|
| `memory_link` | Create a typed relationship between two memories |
| `memory_unlink` | Remove a relationship between memories |
| `memory_related` | Get immediate connections (1 hop) from a memory |
| `memory_graph` | Traverse the knowledge graph (breadth-first, configurable depth) |
| `memory_relationship_types` | List available relationship types |

### Instance Coordination

| Tool | Description |
|------|-------------|
| `handoff_send` | Send message to another AI instance |
| `handoff_get` | Check for messages from other instances |
| `handoff_mark_read` | Mark a handoff message as read |

Both `memory_recall` and `memory_get` support `synthesize=true/false` to control whether results are returned raw or summarized by the local LLM.

## Usage

Once configured with any MCP-compatible client:

```
"Remember that the database migration requires downtime"
"What do we know about the API rate limits?"
"How does the payment service connect to the event bus?"
"Reflect on this conversation and save important decisions"
```

## Configuration

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

For PostgreSQL, set the database config:

```json
{
  "database": {
    "type": "postgres",
    "url": "postgresql://user:pass@localhost/memory_palace"
  }
}
```

## Documentation

- [Installation & Usage](docs/README.md) — Detailed setup and troubleshooting
- [Architecture & Vision](docs/architecture.md) — Why this exists, knowledge graph, scaling path
- [Use Cases](docs/use-cases.md) — Personal, team, and enterprise examples
- [Model Guide](docs/models.md) — Default models and optional GPU upgrades

## License

MIT License — see [LICENSE](LICENSE) for details.
