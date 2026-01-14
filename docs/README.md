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
   git clone https://github.com/your-repo/claude-memory-palace.git
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
   pip install -r requirements.txt
   ```

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
      "args": ["-m", "memory_palace.mcp_server"],
      "cwd": "C:\\path\\to\\claude-memory-palace",
      "env": {
        "MEMORY_PALACE_DB": "C:\\path\\to\\your\\memories.db"
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
| `sandy_remember` | Store a new memory |
| `sandy_recall` | Search memories using semantic search |
| `sandy_forget` | Archive a memory (soft delete) |
| `sandy_memory_stats` | Get overview of memory system |

### Reflection Tools

| Tool | Description |
|------|-------------|
| `sandy_reflect` | Extract memories from conversation transcripts |
| `backfill_embeddings` | Generate embeddings for memories that don't have them |

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
| `MEMORY_PALACE_DB` | Path to SQLite database | `./memories.db` |
| `OLLAMA_HOST` | Ollama server address | `http://localhost:11434` |
| `EMBED_MODEL` | Embedding model name | Auto-detected |
| `LLM_MODEL` | LLM model for reflection | Auto-detected |

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
├── memory_palace/
│   ├── mcp_server.py      # MCP server entry point
│   ├── tools.py           # Tool implementations
│   ├── models.py          # SQLAlchemy models
│   ├── database.py        # Database connection
│   └── embeddings.py      # Ollama embedding client
├── setup/
│   ├── detect_gpu.py      # GPU detection
│   ├── model_recommendations.py  # Model selection
│   └── first_run.py       # Setup wizard
└── docs/
    ├── README.md          # This file
    └── models.md          # Model guide
```

## Support

For issues and feature requests, please open a GitHub issue.
