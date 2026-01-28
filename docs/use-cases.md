# Use Cases

Real-world scenarios where Memory Palace solves problems that bigger context windows can't.

---

## Personal: Cross-Session AI Memory

**Status:** ✅ Shipping

**Problem:** You've been working with Claude on a project for weeks. Every new session, you re-explain your architecture, your preferences, your decisions. Context windows don't carry over.

**Solution:** Memory Palace stores decisions, architecture notes, and preferences as persistent memories. Start a new session, recall what you need, and pick up where you left off.

```
Session 1: "Remember that we chose PostgreSQL over MongoDB for the user service 
            because we need strong consistency for financial transactions."

Session 47: "What database did we pick for the user service and why?"
→ Returns the decision with full context, instantly.
```

---

## Personal: Model Migration

**Status:** ✅ Shipping

**Problem:** You've built up months of context in ChatGPT's memory. Now Claude Opus ships and you want to switch. Your options: start over, or manually recreate everything.

**Solution:** Memory Palace doesn't care which model you use. Switch providers, keep all your memories.

```
Monday:     Claude stores 200 memories about your project
Tuesday:    OpenAI releases GPT-5, you want to try it
Wednesday:  GPT-5 connects to the same Memory Palace, recalls everything Claude stored
Thursday:   You switch back to Claude. Nothing was lost.
```

Your institutional knowledge is never held hostage by a subscription.

---

## Developer: Multi-Tool Workflow

**Status:** ✅ Shipping

**Problem:** You use a desktop AI for planning, a CLI coding agent for implementation, and a web-based AI for documentation. Each tool is a separate context silo.

**Solution:** All three connect to Memory Palace. Planning decisions made in the desktop tool are available to the coding agent during implementation. Architecture insights from coding sessions are available when writing docs.

```
Desktop AI:    "The API uses JWT auth with RS256 signing, 15-min expiry, 
                refresh tokens in HTTP-only cookies."

CLI Agent:     memory_recall("API authentication approach")
               → Gets the full auth decision, implements accordingly

Web AI:        memory_recall("authentication architecture")
               → Gets both the decision AND implementation details for docs
```

---

## Developer: Codebase Onboarding

**Status:** ✅ Shipping (with caveats — see Known Limitations)

**Problem:** A new team member (or a new AI session) needs to understand a large codebase. Reading everything into context is expensive and hits limits.

**Solution:** Store architectural decisions, gotchas, and tribal knowledge in Memory Palace. New sessions recall what they need on demand instead of ingesting everything upfront.

```
memory_remember: "The payment service uses an outbox pattern for event publishing. 
                  Never call the event bus directly — always write to the outbox 
                  table and let the relay pick it up. Direct calls caused duplicate 
                  charges in prod (incident INC-2847, March 2025)."

Any future session: memory_recall("how does payment event publishing work")
→ Gets the pattern, the reason, and the incident that proved why it matters.
```

**Known Limitation:** The default embedding model (nomic-embed-text) has an 8192 token context window. Files exceeding this are truncated before embedding, which can lose information from long files. Mitigation: chunk large files or use summary-based ingestion for oversized sources.

---

## Team: Shared Knowledge Base

**Status:** 🔧 Code complete, needs production validation

**Problem:** Your team of 5 developers each uses AI assistants. Knowledge is scattered across individual conversations. When Alice figures out why the deploy pipeline fails on ARM, Bob's AI doesn't know.

**Solution:** Team Memory Palace (PostgreSQL backend). All team members' AI instances read from and write to the same memory store.

```
Alice's AI:  memory_remember("ARM deploy fix: the base Docker image must use 
             --platform=linux/arm64 explicitly. The multi-arch manifest doesn't 
             resolve correctly behind our corporate proxy.")

Bob's AI:    memory_recall("ARM deployment issues")
             → Gets Alice's fix immediately, no Slack thread archaeology required.
```

**Current state:** PostgreSQL + pgvector backend code exists and handles concurrent connections via SQLAlchemy's QueuePool. Not yet validated with multiple concurrent users in production.

---

## Agent Swarm: Research Pipeline

**Status:** ✅ Architecture works, demonstrated in practice

**Problem:** You want to research a topic using multiple AI agents — one for web search, one for analysis, one for synthesis. Traditional approach: a controller agent manages all three, its context fills up, it becomes the bottleneck.

**Solution:** Each agent reads and writes to Memory Palace independently. No controller needed.

```
Agent: Search
  Role: Find sources
  Loop:
    - Search for relevant papers/articles
    - memory_remember each finding with source, key claims, relevance score
    - handoff_send to "analysis" when batch is ready

Agent: Analysis  
  Role: Evaluate claims
  Loop:
    - handoff_get from "search"
    - memory_recall recent findings
    - Cross-reference claims, identify contradictions
    - memory_remember analysis results
    - handoff_send to "synthesis" when analysis batch is complete

Agent: Synthesis
  Role: Produce final output
  Loop:
    - handoff_get from "analysis"  
    - memory_recall all findings + analysis
    - Generate cohesive research summary
    - memory_remember final synthesis
```

No single agent holds everything in context. The memory store IS the shared state. Each agent can be a different model optimized for its task.

---

## Agent Swarm: Code Review Pipeline

**Status:** 📋 Architecture defined, not yet implemented

**Problem:** You want automated code review across a large PR — security analysis, performance review, style checking, and a final summary. One model can't hold the full diff in context.

**Solution:** Specialized agents review different aspects, all writing findings to shared memory.

```
Agent: Security Reviewer (could be a security-tuned model)
  - Reviews diff for vulnerabilities
  - memory_remember each finding: type, severity, file, line, recommendation

Agent: Performance Reviewer (could be a different model)  
  - Reviews diff for N+1 queries, missing indexes, unnecessary allocations
  - memory_remember each finding with impact estimate

Agent: Style Reviewer (could be a cheap/fast local model)
  - Checks naming conventions, documentation, test coverage
  - memory_remember each finding

Agent: Summary Writer (could be Claude for synthesis quality)
  - memory_recall all findings from the review
  - Produces unified review with prioritized findings
  - Posts review comment to PR
```

Cost-effective: expensive models only where they add value. Fast local models handle routine checks. All coordinated through shared memory, no controller.

---

## Agent Swarm: Continuous Monitoring

**Status:** 📋 Architecture defined, not yet implemented

**Problem:** You want AI agents monitoring multiple data sources — logs, metrics, social media, news — and alerting when patterns emerge across sources.

**Solution:** Each monitor agent writes observations to Memory Palace. A pattern-detection agent periodically recalls recent observations and looks for cross-source correlations.

```
Agent: Log Monitor
  - Watches application logs
  - memory_remember anomalies: "Error rate for /api/payments spiked 3x at 14:23 UTC"

Agent: Social Monitor  
  - Watches Twitter/Reddit for brand mentions
  - memory_remember sentiment shifts: "Negative sentiment spike on Reddit about checkout flow"

Agent: Metrics Monitor
  - Watches Grafana/Datadog dashboards
  - memory_remember threshold breaches: "P99 latency for payment service exceeded 2s"

Agent: Pattern Detector (runs every 15 min)
  - memory_recall recent observations across all sources
  - Correlates: log errors + social complaints + metric spikes = likely incident
  - handoff_send to "incident-response" with correlated evidence
```

---

## Developer: Codebase Knowledge Graph

**Status:** ✅ Shipping (with caveats)

**Problem:** Your AI coding assistant needs to understand how your codebase fits together — which services depend on which, what decisions shaped the architecture, what incidents informed current patterns. Ingesting the entire codebase into a context window is expensive, slow, and hits token limits.

**Solution:** Store architectural relationships in Memory Palace's knowledge graph. When an agent needs context about a component, it traverses the graph at depth 2–3 and gets exactly the relevant connected knowledge.

```
# Store architectural knowledge
memory_remember: "PaymentService uses the outbox pattern for event publishing"
memory_remember: "EventBus consumes from the outbox relay"
memory_remember: "Direct EventBus calls caused duplicate charges (INC-2847)"
memory_remember: "Policy: Never call EventBus directly, always use outbox"

# Link them
memory_link(PaymentService → OutboxPattern, "uses")
memory_link(OutboxPattern → EventBus, "publishes_to")  
memory_link(DuplicateChargeIncident → DirectCallPolicy, "informed")

# Later, any agent asks about payments:
memory_graph(start_id=PaymentService, max_depth=2)
→ Returns the service, its patterns, the event bus, the incident, 
  and the policy — exactly what's needed, nothing extra.
```

**Known limitations:**
- Embedding model (nomic-embed-text) truncates files >8192 tokens
- Graph traversal from hub nodes (identity/foundational memories) can return megabytes at depth 2+
- Need traversal strategies and result limits for large graphs (not yet implemented)

---

## Enterprise: Sovereign AI Memory

**Status:** ✅ Core infrastructure works

**Problem:** Your organization handles classified or regulated data. You need AI assistants with persistent memory, but no data can leave your network. Cloud AI memory features are a non-starter.

**Solution:** Memory Palace runs entirely on-premises. SQLite or PostgreSQL on your servers. Ollama with local models for embeddings. No cloud APIs. No data exfiltration. Full audit trail in a standard database.

```
Infrastructure:
  - PostgreSQL on internal servers (pgvector extension)
  - Ollama on GPU servers (embedding + LLM models downloaded once)
  - MCP server behind corporate firewall
  - AI clients connect via internal network only

Compliance:
  - All data stays on-premises ✓
  - Full audit trail (SQL queryable) ✓
  - No third-party API calls ✓
  - Standard database backups ✓
  - Role-based access via PostgreSQL permissions ✓
```

---

## Enterprise: Multi-Department Agent Fleet

**Status:** 📋 Architecture defined, not yet implemented

**Problem:** You have 1,500 AI agents across engineering, support, sales, and operations. Each department has different knowledge. Agents need shared context within their department and selective cross-department access.

**Planned Solution:** PostgreSQL with schema-based isolation. Each department gets its own schema. Cross-department queries use read-only views.

```
PostgreSQL Schemas:
  engineering.*    — Architecture decisions, incident postmortems, runbooks
  support.*        — Customer issues, resolution patterns, escalation history
  sales.*          — Product knowledge, competitive analysis, deal context
  operations.*     — Process documentation, compliance requirements
  shared.*         — Cross-department read-only views

Agent Fleet:
  500 engineering agents  → read/write engineering.*, read shared.*
  400 support agents      → read/write support.*, read shared.*
  300 sales agents        → read/write sales.*, read shared.*
  300 ops agents          → read/write operations.*, read shared.*
```

**What this needs (not yet built):**
- Schema-based tenant isolation in the MCP server
- PgBouncer or equivalent connection pooling for high concurrency
- LISTEN/NOTIFY integration for real-time handoff delivery
- Access control layer mapping agents to schemas

---

## Summary

| Scale | Backend | Agents | Key Feature | Status |
|-------|---------|--------|-------------|--------|
| Personal | SQLite | 1–10 | Zero config, local file | ✅ Shipping |
| Team | PostgreSQL | 10–100 | Shared knowledge, concurrent access | 🔧 Code complete |
| Department | PostgreSQL + replicas | 100–500 | Read scaling, department isolation | 📋 Planned |
| Enterprise | PostgreSQL cluster | 500–10,000+ | Full fleet coordination, compliance | 📋 Planned |

**Legend:**
- ✅ Shipping — Built, tested, in daily use
- 🔧 Code complete — Implementation exists, needs production validation
- 📋 Planned — Architecture defined, implementation not started

The MCP API is identical at every scale. The only thing that changes is infrastructure — when that infrastructure gets built.
