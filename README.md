# Memory Palace

Persistent semantic memory for AI agents. Store facts, decisions, insights, and context across conversations. Search by meaning, not just keywords. Build a knowledge graph of connected memories. Share across models, instances, and providers.

## The Problem

Every AI session starts as a blank slate. Context windows are finite. Sessions end, knowledge dies.

Current solutions are all vendor-locked: ChatGPT's memory only works with OpenAI. Anthropic's projects only work with Claude. Switch providers and you start over. Your accumulated context — decisions, preferences, project history — belongs to the vendor, not to you.

Meanwhile, the industry races to build bigger context windows. 128K. 200K. 1M tokens. But you don't solve human amnesia by giving someone a bigger whiteboard. Memory doesn't belong inside the model — it belongs alongside it.

Memory Palace is a persistent semantic memory layer that any MCP-compatible AI can access. It separates memory from the model, the same way databases separated data from applications decades ago. The context window becomes working memory. Memory Palace is long-term storage. That's how actual brains work.

## Quick Start

```bash
# Clone and install
git clone https://github.com/jeffpierce/memory-palace.git
cd memory-palace
pip install -e .

# Run setup wizard (detects GPU, downloads models)
python -m setup.first_run
```

Platform-specific installers are also available: `install.bat` / `install.ps1` (Windows), `./install.sh` (macOS/Linux).

See [docs/README.md](docs/README.md) for detailed installation and configuration instructions.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) (local model serving)
- NVIDIA GPU with 4GB+ VRAM (recommended, not required)

## Model Selection

Models are auto-detected by the setup wizard. Defaults are chosen to run everywhere:

| VRAM | Embedding | LLM | Quality |
|------|-----------|-----|---------|
| Any (CPU ok) | nomic-embed-text | qwen3:1.7b | Good — runs on anything |
| 6-8GB | nomic-embed-text | qwen3:8b | Better reasoning |
| 12GB+ | snowflake-arctic-embed | qwen3:14b | Best extraction quality |

See [docs/models.md](docs/models.md) for the full model guide with VRAM budgets and upgrade paths.

## Features

- **Semantic Search** — Find memories by meaning using local embedding models via Ollama
- **Knowledge Graph** — Typed, weighted, directional edges between memories with automatic graph context in retrieval
- **Centrality-Weighted Ranking** — Retrieval scores combine semantic similarity, access frequency, and graph centrality
- **Auto-Linking** — New memories automatically link to similar existing ones (configurable thresholds)
- **Multi-Project Support** — Memories can belong to multiple projects simultaneously
- **Foundational Memories** — Core memories protected from archival and decay
- **Code Indexing** — Index source files as prose descriptions for natural language code search
- **Inter-Instance Messaging** — Unified pub/sub messaging between AI instances with channels, priorities, and push notifications via OpenClaw gateway wake
- **Transcript Reflection** — Automatically extract memories from conversation logs
- **Multi-Backend** — SQLite for personal use, PostgreSQL + pgvector for teams
- **Local Processing** — All embeddings, extraction, and synthesis run locally via Ollama
- **MCP Integration** — Works natively with any MCP-compatible client (Claude Desktop, Claude Code, etc.)
- **TOON Encoding** — Token-efficient structured responses that reduce context window usage

## How Does It Compare?

The MCP memory space is active. Here's how Memory Palace stacks up against the most capable alternatives:

| Feature | Memory Palace | [Mem0](https://mem0.ai/openmemory) | [Cognee](https://github.com/topoteretes/cognee) | [Memento](https://github.com/gannonh/memento-mcp) | [Zep/Graphiti](https://getzep.com) | [doobidoo](https://github.com/doobidoo/mcp-memory-service) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Persistent memory | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Semantic search | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Knowledge graph (typed edges) | ✅ | Partial | ✅ | ✅ | ✅ | ❌ |
| Centrality-weighted ranking | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multi-instance messaging | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Lifecycle management (audit, dedup, contradictions) | ✅ | Partial | Partial | ❌ | Partial | Partial |
| Semantic code search (prose-based) | ✅ | ❌ | AST-based | ❌ | ❌ | ❌ |
| Transcript extraction | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Fully local (no cloud LLMs) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| MCP native | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### What's actually different

**Semantic code search that isn't grep or AST parsing.** `code_remember` uses a local LLM to generate a prose description of what a source file does, embeds *that* prose, and stores the raw code separately. When you search "how does authentication work", you're matching against natural language descriptions, not token sequences or syntax trees. The only other tool using this technique is [Greptile](https://www.greptile.com/) (cloud SaaS, $30/dev/mo). Everyone else — Cursor, Sourcegraph, Cognee, GitLab — embeds raw code chunks or parses ASTs.

**Multi-instance messaging built into the memory layer.** Different AI instances (desktop app, code editor, web) can send typed messages to each other through the palace — handoffs, status updates, questions, context sharing. No other MCP memory server has this. Agent orchestration frameworks (A2A, Agent-MCP) exist but they're not memory systems.

**Everything runs locally.** Embeddings, search synthesis, relationship classification, transcript extraction — all via Ollama on your hardware. Most competitors require cloud LLM calls for at least some operations.

## Tools (13)

### Core Memory

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a new memory with optional auto-linking to similar memories |
| `memory_recall` | Semantic search with centrality-weighted ranking and graph context |
| `memory_get` | Retrieve memories by ID with optional graph traversal (BFS) |
| `memory_recent` | Get the last X memories — title-card format by default, verbose on request |
| `memory_archive` | Archive memories with foundational/centrality protection (soft delete) |

### Knowledge Graph

| Tool | Description |
|------|-------------|
| `memory_link` | Create a typed, weighted, optionally bidirectional edge between memories |
| `memory_unlink` | Remove edges between memories |

Standard relationship types: `relates_to`, `derived_from`, `contradicts`, `exemplifies`, `refines`, `supersedes`. Custom types are also supported.

### Messaging

| Tool | Description |
|------|-------------|
| `message` | Unified inter-instance messaging — send, get, mark read/unread, subscribe to channels |

Replaces the old `handoff_send` / `handoff_get` / `handoff_mark_read` tools with a single action-based interface supporting channels, priorities (0-10), and pub/sub patterns.

### Code Indexing

| Tool | Description |
|------|-------------|
| `code_remember_tool` | Index a source file into the palace (creates linked prose + code memories) |

Queries hit the prose description via semantic search, then graph traversal retrieves the actual source code. This separation produces far better search results than embedding raw code.

### Maintenance

| Tool | Description |
|------|-------------|
| `memory_audit` | Health checks — duplicates, stale memories, orphan edges, contradictions, missing embeddings |
| `memory_reembed` | Regenerate embeddings (backfill missing, refresh stale, re-embed after model change) |
| `memory_stats` | Overview statistics — counts by type, instance, project, most accessed, recently added |

### Processing

| Tool | Description |
|------|-------------|
| `memory_reflect` | Extract memories from conversation transcripts (JSONL or TOON format) |

## Key Concepts

### Graph Context in Retrieval

Both `memory_recall` and `memory_get` automatically include depth-1 graph context (incoming/outgoing edges) by default. This shows how memories connect without separate graph traversal calls.

- **`memory_recall`** — Graph context for top N results (default 5, configurable via `graph_top_n`)
- **`memory_get`** — Graph context for ALL requested memories (targeted fetches get full context)

### Auto-Linking

When storing a new memory, the system automatically finds similar existing memories and creates typed edges:

- **Auto-linked** (>= 0.75 similarity) — Edges created automatically with LLM-classified relationship types
- **Suggested** (0.675-0.75 similarity) — Surfaced for human review, no edges created

Configurable per-instance. Can be scoped to same-project only.

### Multi-Project Memories

Memories can belong to one or more projects simultaneously. Queries can filter by single project (contains) or multiple projects (union). Stats explode multi-project memories across each project for accurate counts.

### Centrality-Weighted Ranking

Recall results are ranked by a weighted combination of:

```
score = (semantic_similarity x 0.7) + (log(access_count + 1) x 0.15) + (in_degree_centrality x 0.15)
```

Frequently accessed, well-connected memories rank higher than isolated ones at the same similarity score.

## Configuration

Configuration loads from `~/.memory-palace/config.json` with environment variable overrides.

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_PALACE_DATA_DIR` | Data directory | `~/.memory-palace` |
| `MEMORY_PALACE_DATABASE_URL` | Database connection URL (overrides config file) | None |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `MEMORY_PALACE_EMBEDDING_MODEL` | Embedding model | Auto-detected |
| `MEMORY_PALACE_LLM_MODEL` | LLM for synthesis/extraction | Auto-detected |
| `MEMORY_PALACE_INSTANCE_ID` | Default instance ID | `unknown` |
| `MEMORY_PALACE_NOTIFY_COMMAND` | Post-send notification command template | None |
| `MEMORY_PALACE_INSTANCE_ROUTES` | Instance route map (JSON string) for push notifications | `{}` |

```json
{
  "database": {
    "type": "sqlite",
    "url": null
  },
  "ollama_url": "http://localhost:11434",
  "embedding_model": null,
  "llm_model": null,
  "synthesis": {
    "enabled": true
  },
  "auto_link": {
    "enabled": true,
    "link_threshold": 0.75,
    "suggest_threshold": 0.675
  },
  "toon_output": true,
  "instances": ["desktop", "code", "web"],
  "notify_command": null,
  "instance_routes": {
    "prime": {
      "gateway": "http://localhost:18789",
      "token": "your-gateway-token-here"
    }
  }
}
```

For PostgreSQL, set `database.type` to `"postgres"` and provide a connection URL. See [docs/architecture.md](docs/architecture.md) for backend details.

## Architecture

```
memory-palace/
├── mcp_server/              # MCP server package
│   ├── server.py            # Server entry point
│   ├── toon_wrapper.py      # TOON response encoding
│   └── tools/               # 13 tool implementations
├── memory_palace/           # Core library
│   ├── models_v3.py         # SQLAlchemy models (Memory, MemoryEdge, Message)
│   ├── database.py          # Database connection (SQLite / PostgreSQL)
│   ├── embeddings.py        # Ollama embedding client
│   ├── llm.py               # LLM integration (synthesis, classification)
│   ├── config_v2.py         # Configuration with auto-link settings
│   ├── services/            # Business logic layer
│   │   ├── memory_service.py      # remember, recall, archive, stats
│   │   ├── graph_service.py       # link, unlink, traverse
│   │   ├── message_service.py     # pub/sub messaging
│   │   ├── maintenance_service.py # audit, reembed, cleanup
│   │   ├── code_service.py        # code indexing + retrieval
│   │   └── reflection_service.py  # transcript extraction
│   └── migrations/          # Schema migration scripts
├── setup/                   # Setup wizard
├── extensions/              # Optional extensions (Moltbook gateway, TOON converter)
├── examples/                # Integration examples and walkthroughs
├── docs/                    # Documentation
└── tests/                   # Test suite
```

See [docs/architecture.md](docs/architecture.md) for the full design vision, knowledge graph details, and scaling roadmap.

## Extensions

Memory Palace includes optional extensions that operate as standalone tools or additional MCP servers:

| Extension | Description | Type |
|-----------|-------------|------|
| [moltbook-gateway](extensions/moltbook-gateway/) | Standalone MCP server for Moltbook submission with 6 mechanical interlocks | MCP Server |
| [toon-converter](extensions/toon-converter/) | CLI + optional MCP server for converting JSONL to TOON format | CLI / MCP Server |

Extensions are independent from the core Memory Palace server and can be used separately.

## Examples

| Example | Description |
|---------|-------------|
| [agent-prompt.md](examples/agent-prompt.md) | Template for adding memory instructions to agent system prompts |
| [soul-file.md](examples/soul-file.md) | Template for integrating memory into character/persona files |
| [centrality_weighted_search.py](examples/centrality_weighted_search.py) | Python example of centrality-weighted search |
| [test_graph_context_mcp.md](examples/test_graph_context_mcp.md) | Walkthrough for testing graph context via MCP |
| [test_maintenance_mcp.md](examples/test_maintenance_mcp.md) | Walkthrough for testing maintenance tools via MCP |

## Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Detailed installation, configuration, and usage guide |
| [docs/architecture.md](docs/architecture.md) | Design vision, knowledge graph, backends, scaling roadmap |
| [docs/models.md](docs/models.md) | Model selection guide with VRAM budgets |
| [docs/use-cases.md](docs/use-cases.md) | Real-world use cases from personal to enterprise |
| [docs/centrality-weighted-retrieval.md](docs/centrality-weighted-retrieval.md) | Centrality ranking deep-dive |
| [docs/QUICKSTART_CENTRALITY.md](docs/QUICKSTART_CENTRALITY.md) | Centrality-weighted retrieval quickstart |
| [docs/MAINTENANCE.md](docs/MAINTENANCE.md) | Maintenance design document |
| [docs/MAINTENANCE_QUICKREF.md](docs/MAINTENANCE_QUICKREF.md) | Maintenance quick reference |
| [docs/TESTING_MAINTENANCE.md](docs/TESTING_MAINTENANCE.md) | Testing maintenance tools guide |
| [docs/MIGRATION_2.0.md](docs/MIGRATION_2.0.md) | v1.0 to v2.0 migration guide |

## License

[MIT](LICENSE)
