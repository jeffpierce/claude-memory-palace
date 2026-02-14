# Architecture & Vision

## The Problem

Every AI session starts as a blank slate. Context windows are finite. Sessions end, knowledge dies. Each AI instance is an island.

Current solutions are all vendor-locked: ChatGPT's memory only works with OpenAI. Anthropic's projects only work with Claude. Switch providers and you start over. Your accumulated context — decisions, preferences, project history — belongs to the vendor, not to you.

Meanwhile, the industry races to build bigger context windows. 128K. 200K. 1M tokens. But a bigger scratchpad isn't memory. You don't solve human amnesia by giving someone a bigger whiteboard.

## The Solution

Memory Palace takes a different approach: **memory doesn't belong inside the model — it belongs alongside it.**

```
┌─────────────────────────────────────────────────┐
│                  Your AI Stack                   │
│                                                  │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│   │ Claude  │  │  Gemini │  │  Local  │  ...    │
│   │         │  │         │  │  Qwen  │         │
│   └────┬────┘  └────┬────┘  └────┬────┘        │
│        │            │            │               │
│        └────────────┼────────────┘               │
│                     │                            │
│              ┌──────┴──────┐                     │
│              │  MCP (open  │                     │
│              │  protocol)  │                     │
│              └──────┬──────┘                     │
│                     │                            │
│         ┌───────────┴───────────┐                │
│         │   Memory Palace       │                │
│         │   ┌───────────────┐   │                │
│         │   │ SQLite/Postgres│   │                │
│         │   │ + Embeddings  │   │                │
│         │   └───────────────┘   │                │
│         └───────────────────────┘                │
│                YOUR HARDWARE                     │
└─────────────────────────────────────────────────┘
```

Memory Palace is a persistent semantic memory layer that any MCP-compatible AI can access. It separates memory from the model, the same way databases separated data from applications decades ago.

The context window becomes **working memory** — the scratchpad for the current task. Memory Palace is **long-term storage** — the accumulated knowledge that persists across sessions, models, and providers.

That's how actual brains work. Short-term processing buffer plus long-term retrieval.

## What This Solves

### 1. Cross-Session Continuity

AI doesn't forget anymore. Sessions end, memory stays. Start a new conversation and recall what happened last week, last month, or last year.

### 2. Cross-Model Portability

Switch from Claude to Gemini to local Qwen? Same memories. Zero migration. **The model is replaceable, the memory isn't.**

### 3. Cross-Subscription Independence

Cancel Anthropic, sign up for OpenAI, spin up local Ollama — doesn't matter. Your memory layer doesn't care who's thinking, only who's remembering.

### 4. Zero Cloud Dependency

It runs on YOUR hardware. SQLite + local embeddings via Ollama. No one's training on your memories. No one's monetizing your context. No API keys to a memory service that'll sunset in 18 months.

### 5. No Vendor Lock-In

ChatGPT's memory locks you to OpenAI. Anthropic's project knowledge locks you to Claude. Gemini's context locks you to Google. Memory Palace? It's *yours*. The protocol is open. The data is local. Walk away from any provider whenever you want.

### 6. Multi-Instance Coordination

The handoff system means AI instances aren't just individually persistent — they can communicate. Your desktop AI can leave a note for your CLI agent. Your coding assistant can pass context to your chat assistant. That's not just memory — it's organizational infrastructure.

### 7. Data Sovereignty

Your memories, your conversations, your context — it's in a SQLite file on YOUR machine. Full stop. `SELECT * FROM memories` whenever you want. Export it. Back it up. Audit it. Try doing that with any cloud AI's memory system.

## The Knowledge Graph: Connected Memory

**Status:** ✅ Shipping

Semantic search finds memories by meaning. But memories don't exist in isolation — they relate to each other. A decision connects to the architecture it shaped, which connects to the incident that informed it, which connects to the policy that prevents recurrence.

Memory Palace includes a built-in knowledge graph with typed, directional, weighted edges:

```
┌─────────────────┐  relates_to  ┌─────────────────┐
│ Auth Decision    │─────────────→│ JWT Architecture │
│ (decision)       │              │ (architecture)   │
└────────┬────────┘              └────────┬────────┘
         │                                │
    exemplifies                      caused_by
         │                                │
         ▼                                ▼
┌─────────────────┐              ┌─────────────────┐
│ Token Expiry     │              │ Session Hijack   │
│ Incident         │              │ Incident         │
│ (event)          │              │ (event)          │
└─────────────────┘              └─────────────────┘
```

### Three Levels of Memory

1. **Storage** (flat files) — things exist
2. **Search** (embeddings) — things are findable by meaning
3. **Understanding** (knowledge graph) — things are *connected*

### Why This Matters for Code

A codebase with 500 files doesn't fit in any context window. But a graph traversal at depth 2–3 from any starting node gives you exactly the relevant context — nothing more, nothing less:

```
memory_graph(start_id=PaymentService, max_depth=2)

→ PaymentService
  ├── uses → OutboxPattern (architecture)
  │   └── publishes_to → EventBus (architecture)
  ├── caused_by → DuplicateChargeIncident (event)
  │   └── informed → NeverCallEventBusDirectly (decision)
  └── depends_on → UserService (architecture)
      └── authenticates_via → JWTAuth (architecture)
```

The AI doesn't need to ingest 500 files. It traverses the graph, pulling only what's connected to the question being asked. Small context windows become a non-issue when you have a map of how everything relates.

**Known limitations:**
- Traversing from hub nodes (identity/foundational memories) at depth 2+ can return megabytes
- Needs result limits and degree-aware traversal strategies (not yet implemented)
- Embedding model truncates files >8192 tokens

### Graph Tools

| Tool | Description | Status |
|------|-------------|--------|
| `memory_link` | Create a typed, weighted, optionally bidirectional edge between two memories | ✅ Shipping |
| `memory_unlink` | Remove edges between memories | ✅ Shipping |

**Note:** Graph traversal is now built into `memory_get` via `traverse=True`, `graph_depth`, and `direction` parameters. The `memory_link` tool with `archive_old=True` and `relation_type="supersedes"` replaces the previous `memory_supersede` tool.

Edges include metadata explaining *why* the connection exists, strength weights for traversal filtering, and directional semantics for accurate graph queries.

### Automatic Graph Context in Retrieval

**Status:** ✅ Shipping

Both `memory_recall` and `memory_get` automatically include depth-1 graph context (immediate incoming/outgoing edges) by default. This eliminates the need for separate graph traversal calls in most cases.

**Design rationale for asymmetric behavior:**

- **`memory_recall`** limits graph context to the top N results (default 5, configurable via `graph_top_n`). Semantic searches can return many results, and fetching graph context for all of them would be expensive and likely unhelpful. The user cares most about understanding connections for the highest-ranked matches.

- **`memory_get`** includes graph context for ALL requested memories. When fetching by ID, it's an intentional targeted operation — the user explicitly wants those specific memories and their full context. No limiting needed.

This asymmetry optimizes for the common case: broad searches get focused graph context, targeted fetches get complete graph context.

**Graph context format:**
```json
{
  "graph_context": {
    "42": {
      "outgoing": [
        {"target_id": 17, "target_subject": "JWT Architecture", "relation_type": "relates_to", "strength": 1.0}
      ],
      "incoming": [
        {"source_id": 99, "source_subject": "Auth Decision", "relation_type": "derived_from", "strength": 1.0}
      ]
    }
  }
}
```

Both parameters are controllable:
- `include_graph=false` disables graph context entirely (for speed)
- `graph_top_n` (recall only) controls how many results get graph context

### Centrality-Weighted Retrieval

**Status:** ✅ Shipping

Memory Palace doesn't just store connections — it uses them to improve search results. The retrieval system combines semantic similarity with access patterns and graph structure:

```
score = (semantic_similarity × 0.7) + (log(access_count + 1) × 0.15) + (in_degree_centrality × 0.15)
```

**Why this matters:** Frequently accessed memories and well-connected hub nodes rank higher than isolated matches. A memory referenced by 10 other memories is more foundational than a one-off note with similar wording. The graph structure becomes a signal of importance.

This is the "understanding" layer in action — the graph doesn't just store relationships, it actively shapes what you retrieve.

## The Handoff System: Decentralized Agent Coordination

**Status:** ✅ Shipping (push-based for PostgreSQL, polling for SQLite)

### The Old Way: Hub-and-Spoke

Traditional agentic swarm architectures use a controller:

```
       ┌─────────────────────┐
       │  Controller AI      │
       │  (big context,      │
       │   expensive,        │
       │   bottleneck)       │
       └──┬──────┬──────┬────┘
          │      │      │
       ┌──┴──┐┌──┴──┐┌──┴──┐
       │ W-A ││ W-B ││ W-C │
       └─────┘└─────┘└─────┘
```

Everything funnels through the controller. Controller's context fills up. Controller becomes the single point of failure. Controller is the most expensive token burn in the whole system.

Hub-and-spoke doesn't scale. We've known this since distributed systems 101.

### The New Way: Shared Memory Bus

Memory Palace + handoffs turns agent coordination into a decentralized message bus:

```
  ┌─────────┐         ┌─────────┐
  │ Agent A │         │ Agent B │
  └────┬────┘         └────┬────┘
       │                   │
       │  memory_set       │  memory_recall
       │  message(send)    │  message(get)
       │                   │
  ┌────┴───────────────────┴────┐
  │       Memory Palace         │
  │    (persistent memory +     │
  │     message bus)            │
  └────┬───────────────────┬────┘
       │                   │
       │  memory_recall    │  memory_set
       │  message(get)     │  message(send)
       │                   │
  ┌────┴────┐         ┌────┴────┐
  │ Agent C │         │ Agent D │
  └─────────┘         └─────────┘
```

**No controller.** Each agent reads and writes to shared memory. Each agent can leave targeted handoff messages for specific other agents. They coordinate through the data store, not through a supervisor.

Each worker can be a *different model*. Cheap local model for routine tasks, frontier models for complex reasoning, specialized fine-tuned model for domain work — all sharing the same memory, all passing messages through the same bus. No single model needs to hold the whole picture.

**Current implementation:** The unified `message` tool handles all inter-instance messaging with `action="send"`, `action="get"`, `action="mark_read"`, and pubsub operations (`subscribe`/`unsubscribe`).

**Delivery paths:**

| Path | Backend | Mechanism | Latency |
|------|---------|-----------|---------|
| MCP + SQLite | SQLite | Polling on `message(action="get")` | Next poll |
| MCP + Postgres | PostgreSQL | HTTP wake via `instance_routes` config | ~1s (wake + heartbeat) |
| OpenClaw + Postgres | PostgreSQL | LISTEN/NOTIFY → bridge event → `enqueueSystemEvent` → heartbeat wake | ~250ms |

The OpenClaw path achieves near-real-time delivery: PostgreSQL NOTIFY fires on message send, the bridge's listener thread picks it up within 100ms, and the plugin injects a system event that wakes the agent on the next heartbeat cycle. See [OPENCLAW.md](OPENCLAW.md) for the full wake chain and [POSTGRES.md](POSTGRES.md) for LISTEN/NOTIFY mechanics.

## Client Paths

Memory Palace supports two client integration paths:

```
┌──────────────┐         ┌──────────────┐
│ MCP Clients  │         │   OpenClaw   │
│ (Claude,     │         │   Gateway    │
│  Cursor,     │         │              │
│  etc.)       │         └──────┬───────┘
└──────┬───────┘                │
       │                  TS Plugin (index.ts)
  MCP Protocol            NDJSON stdin/stdout
       │                        │
       ▼                  Python Bridge (bridge.py)
┌──────────────┐                │
│  MCP Server  │                │
│  (server.py) │                │
└──────┬───────┘                │
       │                        │
       └────────┬───────────────┘
                │
         ┌──────┴──────┐
         │   Services  │
         │   Layer     │
         └──────┬──────┘
                │
         ┌──────┴──────┐
         │  Database   │
         └─────────────┘
```

**MCP path:** Standard MCP protocol. Works with any MCP-compatible client. Messaging uses polling or HTTP wake via `instance_routes`.

**OpenClaw path:** Native plugin with zero MCP overhead. 13 tools registered directly with the gateway. Persistent bridge subprocess. Real-time pubsub wake via PostgreSQL LISTEN/NOTIFY. See [OPENCLAW.md](OPENCLAW.md) for the full plugin guide.

Both paths share the same services layer and database — they're just different entry points.

## Backends

Memory Palace currently ships with two backends:

```
SQLite (personal)     PostgreSQL (team/enterprise)
  Zero config            Concurrent access
  Single file            pgvector search
  No dependencies        LISTEN/NOTIFY pubsub
       └──── Same API ────┘
```

| Tier | Backend | Concurrent Agents | Use Case | Status |
|------|---------|-------------------|----------|--------|
| Personal | SQLite | 1–10 | Individual developer, local AI instances | ✅ Shipping |
| Team | PostgreSQL + pgvector | 10–100 | Dev team sharing AI memory | ✅ Shipping |
| Department | PostgreSQL + read replicas | 100–500 | Cross-team knowledge sharing | 📋 Planned |
| Enterprise | PostgreSQL cluster | 500–10,000+ | Full agent swarm orchestration | 📋 Planned |

**Legend:**
- ✅ Shipping — Built, tested, in daily use
- 🔧 Code complete — Implementation exists, needs production validation  
- 📋 Planned — Architecture defined, implementation not started

SQLite is the default for zero-config setup — no database server needed, just a file. PostgreSQL is a config change away, no code changes required.

### Named Databases (Domain Partitions)

**Status:** Shipping

Named databases let you partition memory into separate domains — `life`, `work`, per-project, etc. Each gets its own PostgreSQL database (or SQLite file) with independent memories, edges, and messages.

```json
{
  "databases": {
    "default": {"type": "postgres", "url": "postgresql://localhost/memory_palace"},
    "life":    {"type": "postgres", "url": "postgresql://localhost/memory_palace_life"},
    "work":    {"type": "postgres", "url": "postgresql://localhost/memory_palace_work"}
  },
  "default_database": "default"
}
```

Key design decisions:
- **Auto-derivation:** Request a name that isn't configured, and the system derives a URL from the default database (appends `memory_palace_` prefix for Postgres, `_name` suffix for SQLite)
- **Auto-creation:** PostgreSQL databases are created automatically on first access
- **Named engine registry:** `database_v3.py` manages one engine + session factory per named database, created lazily
- **Runtime tools:** The `db_manager` extension provides list/register/switch/status tools (runtime only, not persisted)
- **Backward compatible:** If no `databases` key in config, falls back to legacy single `database` key

See [POSTGRES.md](POSTGRES.md) for full setup guide.

### Why PostgreSQL for Scale

SQLite is perfect for single-user local use. It's fast, zero-config, and file-based. But SQLite has a write lock — one writer at a time. That's fine for one person. It's not fine for 100+ concurrent agents.

PostgreSQL with pgvector provides:

- **MVCC (Multi-Version Concurrency Control)** — Every agent reads and writes without blocking others
- **pgvector** — Native vector similarity search with indexing at database scale
- **Connection pooling** — SQLAlchemy's QueuePool handles connection reuse (external pooling like PgBouncer would help at higher scale)
- **Replication** — Read replicas for recall-heavy workloads (architecture defined, not yet implemented)

Switching from SQLite to PostgreSQL is a one-line config change:

```json
{
  "database": {
    "type": "postgres",
    "url": "postgresql://user:pass@localhost/memory_palace"
  }
}
```

No client changes. No data migration tool needed. The MCP API is identical.

### Roadmap: Enterprise Features

The following are architected but **not yet implemented**:

- **Schema-based tenant isolation** — Each department gets its own PostgreSQL schema for data isolation
- **PgBouncer integration** — External connection pooling for thousands of concurrent agents
- **Read replicas** — Separate read scaling from write path

### Air-Gapped & Sovereign Deployment

Because Memory Palace runs entirely on local infrastructure with local models (Ollama), it can be deployed in air-gapped environments. No cloud APIs required. No data leaves the network.

This is critical for:
- Government and defense applications
- Healthcare (HIPAA compliance)
- Financial services (data residency requirements)
- Any organization with strict data sovereignty policies

## Code Retrieval: Natural Language → Implementation

**Status:** ✅ Shipping

Embedding raw source code produces terrible semantic search results. `def calculate_payment(...)` embedded as tokens doesn't match the query "how do we handle refunds?". The solution: dual-memory indexing.

### The Pattern

Each indexed source file creates **two linked memories**:

1. **Prose description** (embedded) — Natural language summary of what the code does, key patterns, and gotchas
2. **Raw code** (stored, not embedded) — The actual implementation

```
Query: "how do we handle duplicate payments?"
  ↓ (semantic search)
Prose: "PaymentService uses outbox pattern to prevent duplicate charges..."
  ↓ (graph traversal via memory_get)
Code: [Full PaymentService.ts implementation]
```

### Why This Works

- **Search hits prose:** "duplicate charges" matches "prevent duplicate charges" in natural language
- **Graph retrieves code:** Once you find the right file via prose, traversal pulls the implementation
- **Small context budget:** You don't embed 500 files. You embed 500 descriptions and pull code on-demand

This is how humans navigate large codebases: you ask someone "where's the payment logic?", they tell you the file, *then* you read the code. Memory Palace does the same.

## Multi-Project Support

**Status:** ✅ Shipping

Memories can belong to **multiple projects simultaneously**. This enables cross-project knowledge sharing while maintaining project-scoped queries.

### Implementation

- **Storage:** PostgreSQL uses native `ARRAY` columns, SQLite uses JSON arrays
- **Query helpers:** `_project_contains(project)` for single-project filtering, `_projects_overlap([...])` for union queries
- **Auto-link scoping:** New memories auto-link to similar memories, optionally restricted to same-project only

### Use Cases

- Shared infrastructure decisions (relevant to `backend`, `frontend`, `mobile`)
- Cross-cutting concerns (auth, logging, observability)
- Team onboarding memories (tagged with multiple product areas)
- Common gotchas (applies to all projects using the same framework)

Assign `project="life"` for personal memories unrelated to code. Most coding memories get explicit project tags.

## Foundational Memories

**Status:** ✅ Shipping

Core memories can be flagged `foundational=True` to mark them as permanent. Foundational memories:

- **Never archived** by automated cleanup, even if old or rarely accessed
- **Never flagged as stale** by audit tools
- **Protected from bulk operations** unless explicitly targeted by ID

Use for: identity information, core principles, critical architectural decisions, or any memory that should persist indefinitely regardless of usage patterns.

## Auto-Linking: Self-Organizing Memory

**Status:** ✅ Shipping

New memories automatically find similar existing memories and create typed relationship edges. This happens during `memory_set` with no manual intervention.

### Two-Tier System

- **Auto-linked (≥0.75 similarity):** Edges created automatically with LLM-classified relationship types (`relates_to`, `refines`, `contradicts`, etc.)
- **Suggested (0.675–0.75 similarity):** Surfaced for human review, no edges created automatically

Auto-linking builds the knowledge graph organically. The more you store, the more connected it becomes — without manual curation.

## Design Principles

1. **Open Protocol** — MCP is a standard. Any compliant client works. No proprietary lock-in.
2. **Local-First** — All processing happens on your hardware by default. Cloud is optional, not required.
3. **Data Ownership** — Your memories are in a standard database you can query, export, backup, and audit.
4. **Backend Agnostic** — The MCP API stays the same whether you're running SQLite or a PostgreSQL cluster.
5. **Model Agnostic** — Any AI that speaks MCP gets persistent memory. Switch models freely.
