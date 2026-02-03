# Quick Start: Centrality-Weighted Retrieval

## What Is This?

Memory retrieval now uses a **smarter ranking algorithm** that considers:
- **Semantic similarity** — How related is the memory to your query?
- **Access frequency** — Is this memory frequently useful?
- **Graph centrality** — Is this memory a "hub" referenced by many others?

The result: More relevant memories surface, especially "important" ones that might not be the closest semantic match.

## Try It Now

No configuration needed — it works out of the box with sensible defaults!

```python
from memory_palace.services.memory_service import recall

# Basic search (uses default weights automatically)
result = recall(
    query="how do embeddings work?",
    limit=10,
    synthesize=True  # Get a natural language summary
)

print(result['summary'])
```

## Adjust the Weighting

Want to favor frequently-accessed memories? Set environment variables:

```bash
export MEMORY_PALACE_WEIGHT_SIMILARITY=0.5   # Reduce semantic weight
export MEMORY_PALACE_WEIGHT_ACCESS=0.3       # Boost access weight
export MEMORY_PALACE_WEIGHT_CENTRALITY=0.2   # Boost centrality weight
```

Or override per-query via the MCP tool:

```python
memory_recall(
    query="how do embeddings work?",
    weight_similarity=0.5,
    weight_access=0.3,
    weight_centrality=0.2
)
```

## Understanding the Weights

The three weights control how much each factor influences ranking:

| Weight | Factor | Default | What It Measures |
|--------|--------|---------|------------------|
| α (alpha) | `weight_similarity` | 0.7 | Semantic relevance (embedding cosine similarity) |
| β (beta) | `weight_access` | 0.15 | How often this memory is accessed |
| γ (gamma) | `weight_centrality` | 0.15 | How many other memories reference this one |

**Tip:** Weights are automatically normalized to sum to 1.0, so you can think of them as percentages.

## Common Configurations

### Balanced (Default)
Best for general use — prioritizes semantic relevance but considers popularity and importance.

```bash
export MEMORY_PALACE_WEIGHT_SIMILARITY=0.7
export MEMORY_PALACE_WEIGHT_ACCESS=0.15
export MEMORY_PALACE_WEIGHT_CENTRALITY=0.15
```

### Favor Frequently-Used Memories
Good when you want "tried and true" memories that have proven useful.

```bash
export MEMORY_PALACE_WEIGHT_SIMILARITY=0.5
export MEMORY_PALACE_WEIGHT_ACCESS=0.3
export MEMORY_PALACE_WEIGHT_CENTRALITY=0.2
```

### Favor Graph Hubs
Good when you want "important" memories that tie together many concepts.

```bash
export MEMORY_PALACE_WEIGHT_SIMILARITY=0.5
export MEMORY_PALACE_WEIGHT_ACCESS=0.1
export MEMORY_PALACE_WEIGHT_CENTRALITY=0.4
```

### Pure Semantic (Disable Centrality Weighting)
Falls back to classic similarity-only ranking.

```bash
export MEMORY_PALACE_WEIGHT_SIMILARITY=1.0
export MEMORY_PALACE_WEIGHT_ACCESS=0.0
export MEMORY_PALACE_WEIGHT_CENTRALITY=0.0
```

## Examples

See `examples/centrality_weighted_search.py` for runnable examples showing how ranking changes with different weights.

## How It Works (High Level)

1. **Fetch candidates** — Get top semantic matches (3x your requested limit)
2. **Compute signals:**
   - Similarity: Embedding cosine similarity (0-1)
   - Access: Log-transformed access count (0-1)
   - Centrality: Number of incoming edges, normalized (0-1)
3. **Combine with weights:**
   - `score = (similarity × α) + (access × β) + (centrality × γ)`
4. **Re-rank** — Sort by combined score
5. **Return top N** — Your requested limit

## Performance

Overhead is negligible (<1ms per query):
- In-degree computation is O(1) with indexed lookups
- Re-ranking is fast (only ~60 candidates)

You won't notice any slowdown.

## When to Use Which Weights

### High Similarity Weight (α = 0.7+)
Use when:
- You want semantically close matches
- Query is specific and precise
- You trust your embedding model

### High Access Weight (β = 0.3+)
Use when:
- You want "battle-tested" memories
- You don't care about novelty
- You want memories that have proven useful

### High Centrality Weight (γ = 0.3+)
Use when:
- You want "hub" memories that connect many concepts
- You're exploring a knowledge graph
- You want memories that are structurally important

## Debugging

To see what ranking was used, check the `search_method` field:

```python
result = recall(query="test", synthesize=False)
print(result['search_method'])
# Output: "semantic (centrality-weighted)" or "semantic (pgvector, centrality-weighted)"
```

If you see "keyword (centrality-weighted fallback)", it means embeddings weren't available and keyword search was used.

## Learn More

- **Full documentation:** `docs/centrality-weighted-retrieval.md`
- **Implementation details:** `CENTRALITY_RETRIEVAL_SUMMARY.md`
- **Examples:** `examples/centrality_weighted_search.py`
- **Tests:** `test_centrality_retrieval.py`

## Questions?

- How is centrality computed? **In-degree count (number of incoming edges)**
- Does this require schema changes? **No — uses existing columns**
- Is it backward compatible? **Yes — default weights preserve similar behavior**
- Can I disable it? **Yes — set all weights to favor similarity (1.0, 0.0, 0.0)**
