# Claude Memory Palace

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
git clone https://github.com/jeffpierce/claude-memory-palace.git
cd claude-memory-palace

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

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a new memory |
| `memory_recall` | Semantic search across memories |
| `memory_forget` | Archive a memory |
| `memory_reflect` | Extract memories from transcripts |
| `memory_stats` | Memory system overview |
| `memory_get` | Retrieve memories by ID |
| `memory_backfill_embeddings` | Generate embeddings for memories without them |
| `handoff_send` | Send message to another Claude instance |
| `handoff_get` | Check for messages from other instances |
| `handoff_mark_read` | Mark a handoff message as read |

## Architecture

```
claude-memory-palace/
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
