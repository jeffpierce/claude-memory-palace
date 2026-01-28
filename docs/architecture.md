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

ChatGPT's memory locks you to OpenAI. Claude's projects lock you to Anthropic. Memory Palace? It's *yours*. The protocol is open. The data is local. Walk away from any provider whenever you want.

### 6. Multi-Instance Coordination

The handoff system means AI instances aren't just individually persistent — they can communicate. Desktop Claude can leave a note for CLI Claude. Your coding agent can pass context to your chat agent. That's not just memory — it's organizational infrastructure.

### 7. Data Sovereignty

Your memories, your conversations, your context — it's in a SQLite file on YOUR machine. Full stop. `SELECT * FROM memories` whenever you want. Export it. Back it up. Audit it. Try doing that with any cloud AI's memory system.

## The Handoff System: Decentralized Agent Coordination

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

## Scaling Path

The MCP interface is the abstraction layer. Swap the backend without changing any client code:

```
SQLite (personal)  →  Postgres (team)  →  Postgres cluster (enterprise)
  Same MCP API           Same MCP API          Same MCP API
```

| Tier | Backend | Concurrent Agents | Use Case |
|------|---------|-------------------|----------|
| Personal | SQLite | 1–10 | Individual developer, local AI instances |
| Team | PostgreSQL + pgvector | 10–100 | Dev team sharing AI memory |
| Department | PostgreSQL + read replicas | 100–500 | Cross-team knowledge sharing |
| Enterprise | PostgreSQL cluster | 500–10,000+ | Full agent swarm orchestration |

### Why PostgreSQL for Scale

SQLite is perfect for single-user local use. It's fast, zero-config, and file-based. But SQLite has a write lock — one writer at a time. That's fine for one person. It's not fine for 1,500 concurrent agents.

PostgreSQL with pgvector provides:

- **MVCC (Multi-Version Concurrency Control)** — Every agent reads and writes without blocking others
- **pgvector** — Native vector similarity search with indexing at database scale
- **Connection pooling** — PgBouncer maps thousands of agent connections to a manageable pool
- **LISTEN/NOTIFY** — Agents can receive push notifications for handoffs instead of polling
- **Replication** — Read replicas for recall-heavy workloads (most agents read more than write)

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
