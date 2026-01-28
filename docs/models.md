# Model Guide

Memory Palace uses two types of local models via [Ollama](https://ollama.ai):

1. **Embedding Model** — Converts text to vectors for semantic search
2. **LLM Model** — Extracts memories from transcripts, synthesizes recall results

## Defaults: Runs on Anything

The default models are chosen to work everywhere — laptops, old desktops, CPU-only machines, even a Raspberry Pi if you're patient:

| Component | Default Model | Download Size | RAM/VRAM | Purpose |
|-----------|--------------|---------------|----------|---------|
| Embedding | nomic-embed-text | ~300MB | ~300MB | Semantic vector search |
| LLM | qwen3:1.7b | ~1GB | ~1.5GB | Synthesis, classification, extraction |

**Total: ~1.3GB download.** No GPU required. These are installed automatically by the setup wizard.

The defaults are opinionated: they prioritize accessibility over maximum quality. Memory Palace should work for everyone, not just people with expensive GPUs. You can always upgrade later without losing any data.

## Upgrading Models (Optional)

If you have a dedicated NVIDIA GPU and want better extraction quality, you can swap in larger models. The embedding model rarely needs upgrading — nomic-embed-text is excellent for its size. The LLM is where more VRAM pays off.

### LLM Upgrades

The LLM handles memory extraction from transcripts and synthesis of recall results. Bigger models catch more nuance and produce better summaries.

| VRAM Available | Recommended LLM | Download | Improvement |
|---------------|-----------------|----------|-------------|
| Default (any) | qwen3:1.7b | ~1GB | Baseline — works everywhere |
| 6GB+ | qwen3:4b | ~2.5GB | Better instruction following |
| 8GB+ | qwen3:8b | ~5GB | Noticeably better reasoning |
| 12GB+ | qwen3:14b | ~9GB | Best extraction quality |

To upgrade:

```bash
# Pull the larger model
ollama pull qwen3:8b

# Set via environment variable
export MEMORY_PALACE_LLM_MODEL=qwen3:8b

# Or set in config file (~/.memory-palace/config.json)
# "llm_model": "qwen3:8b"
```

### Embedding Upgrades

The embedding model affects semantic search quality — how well "what was our auth decision?" finds a memory about "JWT tokens with RS256 signing." nomic-embed-text is genuinely good here; upgrade only if you want the absolute best retrieval accuracy.

| VRAM Available | Recommended Embedding | Download | Dimensions | Notes |
|---------------|----------------------|----------|------------|-------|
| Default (any) | nomic-embed-text | ~300MB | 768 | Great quality-to-size ratio |
| 8GB+ | snowflake-arctic-embed:335m | ~1GB | 1024 | Better retrieval accuracy |
| 16GB+ | sfr-embedding-mistral:f16 | ~14GB | 4096 | MTEB #2, best available |

**⚠️ Changing embedding models requires re-embedding all existing memories.** Different models produce incompatible vectors. After switching, run:

```bash
# Re-generate all embeddings with the new model
# Use the memory_backfill_embeddings tool, or:
export MEMORY_PALACE_EMBEDDING_MODEL=snowflake-arctic-embed:335m
# Then call memory_backfill_embeddings via your MCP client
```

### Combined VRAM Budget

Ollama loads one model at a time by default, swapping as needed. If you want to understand total VRAM requirements:

| Setup | Embedding | LLM | Peak VRAM | Notes |
|-------|-----------|-----|-----------|-------|
| Default | nomic-embed-text (300MB) | qwen3:1.7b (1.5GB) | ~1.5GB | Runs on anything |
| Mid-range | nomic-embed-text (300MB) | qwen3:8b (5GB) | ~5GB | Best bang for buck |
| High-end | snowflake-arctic (1GB) | qwen3:14b (9GB) | ~9GB | Serious quality uplift |
| Premium | sfr-embedding-mistral (14GB) | qwen3:14b (9GB) | ~14GB | One model at a time |

Models swap automatically — you don't need VRAM for both simultaneously unless you're running very high-throughput workloads.

## CPU Fallback

No GPU? No problem. All models run on CPU via Ollama. Performance differences:

- **Embedding** — ~2-5x slower on CPU. Still fast enough for interactive use.
- **LLM (1.7B)** — Responsive on modern CPUs. A few seconds per synthesis.
- **LLM (8B+)** — Noticeably slower on CPU (~10-30x). Consider sticking with 1.7B if CPU-only.

The default models were chosen specifically because they're usable on CPU. If you're CPU-only, the defaults are your best option.

## Changing Models

To switch models after initial setup:

1. **Pull the new model:**
   ```bash
   ollama pull model-name
   ```

2. **Update configuration** (choose one):
   ```bash
   # Environment variable
   export MEMORY_PALACE_LLM_MODEL=qwen3:8b
   export MEMORY_PALACE_EMBEDDING_MODEL=snowflake-arctic-embed:335m

   # Or config file (~/.memory-palace/config.json)
   {
     "llm_model": "qwen3:8b",
     "embedding_model": "snowflake-arctic-embed:335m"
   }
   ```

3. **If you changed the embedding model**, re-embed existing memories using the `memory_backfill_embeddings` tool.

Environment variables override config file values.
