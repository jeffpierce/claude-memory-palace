# Centrality-Weighted Retrieval

## Overview

Memory retrieval now uses a **weighted scoring formula** that combines three signals to rank memories:

```
score = (cosine_similarity × α) + (log(access_count + 1) × β) + (centrality × γ)
```

Where:
- **α (alpha)**: Weight for semantic similarity (default: 0.7)
- **β (beta)**: Weight for access frequency (default: 0.15)
- **γ (gamma)**: Weight for graph centrality (default: 0.15)

All weights are automatically normalized to sum to 1.0 for interpretability.

## Motivation

Pure semantic search (cosine similarity only) doesn't capture:
- **Memory importance within the knowledge graph** — some memories are "hubs" referenced by many others
- **Practical usefulness** — memories accessed frequently are demonstrably valuable

Centrality-weighted retrieval surfaces memories that are both semantically relevant AND structurally important.

## Components

### 1. Semantic Similarity (α = 0.7)

**Primary ranking signal.** Uses embedding cosine similarity to find semantically related memories.

- **PostgreSQL + pgvector**: Native approximate nearest neighbor (ANN) via HNSW index
- **SQLite**: Python-side cosine similarity (still fast for typical memory counts)

### 2. Access Frequency (β = 0.15)

**Log-transformed to prevent Matthew effect.** Memories accessed frequently are likely useful, but we don't want to create a rich-get-richer dynamic.

- Formula: `log(access_count + 1) / log(max_access_count + 1)`
- Normalized to [0, 1] range
- Log transform flattens the curve — a memory with 100 accesses isn't 100x more valuable than one with 1 access

### 3. Graph Centrality (γ = 0.15)

**In-degree count (number of incoming edges).** Memories referenced by many others are structurally important "hub" nodes.

- Formula: `in_degree / max_in_degree`
- Normalized to [0, 1] range
- Computed efficiently via SQL GROUP BY (O(1) with proper indexing)

**Why in-degree (not PageRank)?**
- PageRank is O(n³) worst case — too expensive for real-time retrieval
- In-degree is O(1) with indexed lookups
- In-degree is "good enough" for v1 — we can add precomputed PageRank later if needed

## Configuration

### Environment Variables

Set weights globally via environment variables:

```bash
export MEMORY_PALACE_WEIGHT_SIMILARITY=0.6   # α (semantic relevance)
export MEMORY_PALACE_WEIGHT_ACCESS=0.2       # β (access frequency)
export MEMORY_PALACE_WEIGHT_CENTRALITY=0.2   # γ (graph centrality)
```

Weights are automatically normalized to sum to 1.0.

### Per-Query Overrides (MCP Tool)

The `memory_recall` MCP tool accepts optional weight parameters:

```python
memory_recall(
    query="how do embeddings work?",
    weight_similarity=0.5,    # Reduce semantic weight
    weight_access=0.25,       # Boost access weight
    weight_centrality=0.25    # Boost centrality weight
)
```

Use this to experiment with different weightings for specific queries.

### Graph Context Integration

Both `memory_recall` and `memory_get` automatically include depth-1 graph context (immediate incoming/outgoing edges) in their responses. This enhances centrality-weighted retrieval by not just ranking hub memories higher, but also showing you *why* they're hubs.

**Parameters:**

- `include_graph` (bool, default `true`): Include graph context in results
- `graph_top_n` (int, default 5, `memory_recall` only): Limit graph context to top N results

**Example:**

```python
# Default: top 5 results get graph context
result = memory_recall(query="authentication patterns", limit=20)

# Access graph context
for memory_id, context in result['graph_context'].items():
    print(f"Memory {memory_id}:")
    print(f"  Referenced by: {len(context['incoming'])} memories")
    print(f"  References: {len(context['outgoing'])} memories")
```

**Why this matters:** A memory with high centrality (many incoming edges) might rank highly even if it's not the closest semantic match. The graph context shows you the connections that made it important, helping you understand the knowledge graph topology.

## Keyword Fallback Mode

When semantic search is unavailable (no embeddings), the system falls back to keyword search with adjusted weighting:

```
score = (importance × 0.4) + (access × β_normalized) + (centrality × γ_normalized)
```

- **importance** replaces similarity (since we have no semantic signal)
- **β and γ** are renormalized to sum to 0.6 (remaining weight after importance)

## Implementation Notes

### Candidate Fetching

To ensure proper re-ranking, we fetch **3x the requested limit** of candidates before applying centrality weighting:

```python
candidate_limit = limit * 3  # Fetch extra candidates for re-ranking
```

This ensures the top results after re-ranking are actually the best by the combined metric, not just "top by similarity, then filtered."

### Normalization

All three components are normalized to [0, 1] before combining:

- **Similarity**: Already 0-1 (cosine similarity)
- **Access**: `log(access_count + 1) / log(max_access_count + 1)`
- **Centrality**: `in_degree / max_in_degree`

This ensures weights (α, β, γ) have proportional effect regardless of absolute magnitudes.

### Performance

- **In-degree computation**: O(1) with indexed `target_id` lookups
- **Log transform**: O(n) where n = number of candidates
- **Re-ranking**: O(n log n) for sort, but n is small (3 × limit, typically 60 memories)

Total overhead is negligible compared to embedding generation.

## Testing

Run the test suite:

```bash
python test_centrality_retrieval.py
```

Tests:
1. Weight configuration (env vars)
2. In-degree centrality computation
3. Semantic recall with centrality weighting
4. Keyword fallback with centrality weighting

## Future Enhancements

### Precomputed PageRank

For large knowledge graphs (>10K memories), we could:
1. Precompute PageRank scores periodically (e.g., nightly)
2. Store as `Memory.pagerank` column
3. Use in place of in-degree for more accurate centrality

### Temporal Decay

Add time-based weighting to favor recent memories:

```
score += (recency × δ)
```

Where `recency = 1 - (age_days / max_age_days)`.

### Edge Type Weighting

Not all edges are equal. `supersedes` edges indicate replacement, while `relates_to` edges indicate connection:

```python
edge_weights = {
    "supersedes": 0.5,    # Lower weight (old memory less important)
    "relates_to": 1.0,    # Full weight
    "contradicts": 0.8,   # Slightly lower (tension)
    "refines": 1.2,       # Slightly higher (adds detail)
}
```

Compute weighted in-degree using edge type multipliers.

## References

- **Moltbook community feedback**: Original feature request from Clawddar
- **Matthew effect**: Rich-get-richer phenomenon (Merton, 1968)
- **PageRank**: Brin & Page, 1998 — computationally expensive, deferred to v2

## Credits

Implementation in Claude Code (Sonnet 4.5) based on Clawddar's Moltbook proposal.
