# Testing Graph Context via MCP

This document shows how to test the graph context feature using the MCP tools directly.

## Prerequisites

1. Memory Palace configured and running via MCP
2. Some memories stored with relationships between them
3. Claude Desktop or another MCP client

## Example Workflow

### 1. Create Some Connected Memories

First, let's create a few memories with relationships:

```
Remember that we chose PostgreSQL over MongoDB for the user service because we need strong consistency for financial transactions.
```

```
Remember that the authentication service uses JWT tokens with RS256 signing.
```

```
Remember that we had an incident (INC-2847) where direct EventBus calls caused duplicate charges.
```

Now link them together:

```
Create a relationship from the JWT authentication memory to the PostgreSQL choice memory with relation type "relates_to" because both are part of the same service architecture.
```

### 2. Test memory_recall with Graph Context

**Default behavior (graph context enabled):**

```
Search for memories about "authentication" with limit 5.
```

Expected response format:
```json
{
  "memories": [...],
  "count": 3,
  "search_method": "semantic (centrality-weighted)",
  "graph_context": {
    "42": {
      "outgoing": [
        {
          "target_id": 17,
          "target_subject": "PostgreSQL choice for user service",
          "relation_type": "relates_to",
          "strength": 1.0
        }
      ],
      "incoming": []
    }
  }
}
```

**With graph context disabled:**

```
Search for memories about "authentication" with limit 5 and include_graph set to false.
```

Response will NOT include the `graph_context` key.

**With custom graph_top_n:**

```
Search for memories about "authentication" with limit 20 and graph_top_n set to 10.
```

Only the top 10 results will have graph context, even though 20 results were returned.

### 3. Test memory_get with Graph Context

**Retrieve specific memories by ID:**

```
Retrieve memories with IDs [42, 17, 99] with include_graph enabled.
```

Expected response:
```json
{
  "memories": [
    {"id": 42, "subject": "JWT authentication", ...},
    {"id": 17, "subject": "PostgreSQL choice", ...},
    {"id": 99, "subject": "Duplicate charge incident", ...}
  ],
  "count": 3,
  "graph_context": {
    "42": {
      "outgoing": [...],
      "incoming": [...]
    },
    "17": {
      "outgoing": [...],
      "incoming": [...]
    },
    "99": {
      "outgoing": [...],
      "incoming": [...]
    }
  }
}
```

Note: ALL requested memories get graph context (not limited like recall).

**With graph context disabled:**

```
Retrieve memories with IDs [42, 17] with include_graph set to false.
```

Response will NOT include the `graph_context` key.

## Understanding the Output

### Graph Context Structure

Each memory ID in the `graph_context` dict has:

```json
{
  "outgoing": [
    {
      "target_id": 17,
      "target_subject": "Target Memory Subject",
      "relation_type": "relates_to",
      "strength": 1.0
    }
  ],
  "incoming": [
    {
      "source_id": 99,
      "source_subject": "Source Memory Subject",
      "relation_type": "derived_from",
      "strength": 1.0
    }
  ]
}
```

- **outgoing**: Edges FROM this memory TO others (this memory references...)
- **incoming**: Edges FROM others TO this memory (this memory is referenced by...)

### Interpreting Centrality

If a memory has many incoming edges, it's a "hub" in the knowledge graph:

```json
{
  "incoming": [
    {"source_id": 1, ...},
    {"source_id": 5, ...},
    {"source_id": 12, ...},
    {"source_id": 23, ...},
    {"source_id": 45, ...}
  ]
}
```

This memory is referenced by 5 others, making it structurally important. This is why centrality-weighted retrieval might rank it highly even if the semantic similarity is lower.

## Performance Testing

### Test with Large Result Sets

```
Search for memories about "decision" with limit 50 and graph_top_n set to 50.
```

Observe response time. Graph context fetching should add minimal overhead (<50ms for 50 memories with typical edge counts).

### Test with No Edges

Create a standalone memory with no relationships:

```
Remember that we tested the graph context feature on 2026-02-03.
```

Then retrieve it:

```
Retrieve memory about "graph context feature test".
```

Expected: `graph_context` will have empty `incoming` and `outgoing` arrays.

## Common Use Cases

### 1. Understanding Why a Memory Ranked Highly

When centrality-weighted retrieval surfaces a memory you didn't expect:

1. Check the similarity score (might be lower)
2. Check the graph context incoming edges
3. High centrality (many incoming edges) explains the high rank

### 2. Exploring Connected Knowledge

When you find a relevant memory and want to see what's connected:

1. Use `memory_recall` to find the memory
2. Graph context immediately shows connections
3. No need for separate `memory_related` call

### 3. Building Knowledge Clusters

When building a new feature and want to see related architecture:

1. Search for a key architectural memory
2. Graph context shows what it connects to
3. Follow the outgoing edges to related decisions

## Troubleshooting

### Graph Context is Empty

Possible causes:
1. Memory has no relationships (expected)
2. `include_graph=false` was set
3. For recall: Memory was outside `graph_top_n` limit

### Graph Context is Huge

If a memory has 100+ edges, the response will be large. Solutions:
1. Reduce `graph_top_n` for recall
2. Use `memory_graph` with depth/relation filters for more control
3. Disable graph context if not needed for this query

### Memory IDs Don't Match

Graph context uses string keys (JSON requirement):
- Memory object: `"id": 42`
- Graph context key: `"42": {...}`

Convert to string when looking up: `graph_context[str(memory_id)]`

## Further Reading

- Full documentation: `docs/architecture.md` (graph context section)
- Implementation details: `GRAPH_CONTEXT_IMPLEMENTATION.md`
- Python examples: `examples/centrality_weighted_search.py`
