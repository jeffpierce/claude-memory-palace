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
git clone https://github.com/your-repo/claude-memory-palace.git
cd claude-memory-palace

# Install dependencies
pip install -r requirements.txt

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
| `sandy_remember` | Store a new memory |
| `sandy_recall` | Semantic search across memories |
| `sandy_forget` | Archive a memory |
| `sandy_reflect` | Extract memories from transcripts |
| `sandy_memory_stats` | Memory system overview |

## Architecture

```
claude-memory-palace/
├── memory_palace/       # Core library
│   ├── mcp_server.py    # MCP protocol server
│   ├── tools.py         # Tool implementations
│   └── embeddings.py    # Ollama integration
├── setup/               # Setup utilities
│   ├── detect_gpu.py    # GPU detection
│   └── first_run.py     # Setup wizard
└── docs/                # Documentation
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_PALACE_DB` | Database path | `./memories.db` |
| `OLLAMA_HOST` | Ollama server | `localhost:11434` |
| `EMBED_MODEL` | Embedding model | Auto-detected |
| `LLM_MODEL` | LLM for reflection | Auto-detected |

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
