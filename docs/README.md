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
      "cwd": "C:\\path\\to\\claude-memory-palace",
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
| `memory_recall` | Search memories using semantic search |
| `memory_forget` | Archive a memory (soft delete) |
| `memory_get` | Retrieve memories by ID |
| `memory_stats` | Get overview of memory system |

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

## Support

For issues and feature requests, please open a GitHub issue.
