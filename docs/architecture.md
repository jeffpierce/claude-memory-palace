# Architecture & Vision

## The Problem

Every AI session starts as a blank slate. Context windows are finite. Sessions end, knowledge dies. Each AI instance is an island.

Current solutions are all vendor-locked: ChatGPT's memory only works with OpenAI. Claude's projects only work with Anthropic. Switch providers and you start over. Your accumulated context — decisions, preferences, project history — belongs to the vendor, not to you.

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

ChatGPT's memory locks you to OpenAI. Claude's project knowledge locks you to Anthropic. Gemini's context locks you to Google. Memory Palace? It's *yours*. The protocol is open. The data is local. Walk away from any provider whenever you want.

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
| `memory_related` | Get immediate connections (1 hop) from a memory | ✅ Shipping |
| `memory_graph` | Breadth-first traversal to configurable depth | ✅ Shipping |
| `memory_relationship_types` | List standard relationship types | ✅ Shipping |

Edges include metadata explaining *why* the connection exists, strength weights for traversal filtering, and directional semantics for accurate graph queries.

## The Handoff System: Decentralized Agent Coordination

**Status:** ✅ Shipping (polling-based)

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
       │  memory_remember  │  memory_recall
       │  handoff_send     │  handoff_get
       │                   │
  ┌────┴───────────────────┴────┐
  │       Memory Palace         │
  │    (persistent memory +     │
  │     message bus)            │
  └────┬───────────────────┬────┘
       │                   │
       │  memory_recall    │  memory_remember
       │  handoff_get      │  handoff_send
       │                   │
  ┌────┴────┐         ┌────┴────┐
  │ Agent C │         │ Agent D │
  └─────────┘         └─────────┘
```

**No controller.** Each agent reads and writes to shared memory. Each agent can leave targeted handoff messages for specific other agents. They coordinate through the data store, not through a supervisor.

Each worker can be a *different model*. Cheap local model for routine tasks, Claude for complex reasoning, specialized fine-tuned model for domain work — all sharing the same memory, all passing messages through the same bus. No single model needs to hold the whole picture.

**Current implementation:** Agents poll for handoffs via `handoff_get`. Push notifications via LISTEN/NOTIFY are planned but not yet implemented.

## Backends

Memory Palace currently ships with two backends:

```
SQLite (personal)     PostgreSQL (team/enterprise)
  Zero config            Concurrent access
  Single file            pgvector search
  No dependencies        Scales with infra
       └──── Same MCP API ────┘
```

| Tier | Backend | Concurrent Agents | Use Case | Status |
|------|---------|-------------------|----------|--------|
| Personal | SQLite | 1–10 | Individual developer, local AI instances | ✅ Shipping |
| Team | PostgreSQL + pgvector | 10–100 | Dev team sharing AI memory | 🔧 Code complete |
| Department | PostgreSQL + read replicas | 100–500 | Cross-team knowledge sharing | 📋 Planned |
| Enterprise | PostgreSQL cluster | 500–10,000+ | Full agent swarm orchestration | 📋 Planned |

**Legend:**
- ✅ Shipping — Built, tested, in daily use
- 🔧 Code complete — Implementation exists, needs production validation  
- 📋 Planned — Architecture defined, implementation not started

SQLite is the default for zero-config setup — no database server needed, just a file. PostgreSQL is a config change away, no code changes required.

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
- **LISTEN/NOTIFY** — Push-based handoff delivery instead of polling
- **Read replicas** — Separate read scaling from write path

### Air-Gapped & Sovereign Deployment

Because Memory Palace runs entirely on local infrastructure with local models (Ollama), it can be deployed in air-gapped environments. No cloud APIs required. No data leaves the network.

This is critical for:
- Government and defense applications
- Healthcare (HIPAA compliance)
- Financial services (data residency requirements)
- Any organization with strict data sovereignty policies

## Design Principles

1. **Open Protocol** — MCP is a standard. Any compliant client works. No proprietary lock-in.
2. **Local-First** — All processing happens on your hardware by default. Cloud is optional, not required.
3. **Data Ownership** — Your memories are in a standard database you can query, export, backup, and audit.
4. **Backend Agnostic** — The MCP API stays the same whether you're running SQLite or a PostgreSQL cluster.
5. **Model Agnostic** — Any AI that speaks MCP gets persistent memory. Switch models freely.
