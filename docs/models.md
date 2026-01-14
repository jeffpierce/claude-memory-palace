# Model Selection Guide

Claude Memory Palace uses two types of models:
1. **Embedding Models** - Convert text to vectors for semantic search
2. **LLM Models** - Extract memories from conversation transcripts

## Quick Reference

| VRAM | Embedding Model | LLM Model | Notes |
|------|-----------------|-----------|-------|
| 4GB | nomic-embed-text | qwen2.5:3b | Tight fit, consider CPU |
| 6GB | nomic-embed-text | qwen2.5:7b | Works with model swapping |
| 8GB | snowflake-arctic-embed | qwen2.5:7b | Comfortable |
| 10GB | snowflake-arctic-embed | qwen3:8b | Good headroom |
| 12GB | snowflake-arctic-embed | qwen3:14b | Non-simultaneous use |
| 16GB+ | sfr-embedding-mistral | qwen3:14b | Premium, one at a time |
| 24GB+ | sfr-embedding-mistral | qwen3:14b | Can overlap carefully |

## Embedding Models

Embedding models convert text into high-dimensional vectors, enabling semantic similarity search (finding memories by meaning, not just keywords).

### nomic-embed-text (Recommended for 4-6GB VRAM)

```bash
ollama pull nomic-embed-text
```

- **Size:** ~300MB
- **Dimensions:** 768
- **Context:** 8,192 tokens
- **Quality:** Good for most use cases
- **Speed:** Very fast

Best for systems with limited VRAM or when running alongside other applications.

### snowflake-arctic-embed-l (Recommended for 8-12GB VRAM)

```bash
ollama pull snowflake-arctic-embed:335m
```

- **Size:** ~1GB
- **Dimensions:** 1024
- **Context:** 512 tokens
- **Quality:** Excellent retrieval performance
- **Speed:** Fast

Good balance of quality and resource usage. Recommended for most users with dedicated GPU.

### sfr-embedding-mistral (Premium - 16GB+ VRAM)

```bash
ollama pull sfr-embedding-mistral:f16
```

- **Size:** ~14GB
- **Dimensions:** 4096
- **Context:** 32,768 tokens
- **Quality:** MTEB #2, best available
- **Speed:** Slower but worth it

The gold standard for embedding quality. Use if you have the VRAM and want the best possible semantic search. The full-precision F16 version is recommended over quantized versions.

## LLM Models

LLM models are used by `sandy_reflect` to intelligently extract memories from conversation transcripts. They analyze conversations and identify facts, decisions, insights, and other memorable content.

### qwen2.5:3b (Minimal - 4GB VRAM)

```bash
ollama pull qwen2.5:3b
```

- **Size:** ~2GB
- **Context:** 32K tokens
- **Quality:** Basic extraction capability
- **Speed:** Fast

Use only if VRAM is severely constrained. Extraction quality is noticeably lower.

### qwen2.5:7b (Balanced - 6-8GB VRAM)

```bash
ollama pull qwen2.5:7b
```

- **Size:** ~4.5GB
- **Context:** 32K tokens
- **Quality:** Good extraction, reliable format following
- **Speed:** Good

Recommended for most users. Good balance of quality and resource usage.

### qwen3:8b (Upgraded - 10GB VRAM)

```bash
ollama pull qwen3:8b
```

- **Size:** ~5GB
- **Context:** 32K tokens
- **Quality:** Better reasoning, improved extraction
- **Speed:** Good

Newer architecture with improved instruction following. Worth the upgrade if you have the VRAM.

### qwen3:14b (Premium - 12GB+ VRAM)

```bash
ollama pull qwen3:14b
```

- **Size:** ~9GB loaded (14GB on disk)
- **Context:** 32K tokens
- **Quality:** Best extraction quality
- **Speed:** Slower but thorough

Best available extraction quality. Catches nuanced information and produces well-typed memories. Requires dedicated GPU with 12GB+ VRAM.

## VRAM Management

### Single Model at a Time

On systems with limited VRAM (under 24GB), run only one model at a time:
1. Load embedding model for storing/searching memories
2. Unload embedding model (Ollama does this automatically)
3. Load LLM for reflection
4. Unload LLM when done

Ollama handles model loading/unloading automatically based on usage.

### Memory Headroom

Always leave some VRAM headroom:
- 14GB model on 16GB card = works but tight
- 14GB model on 24GB card = comfortable
- 9GB + 5GB models on 16GB = cannot run simultaneously

### CPU Fallback

If VRAM is insufficient, Ollama will use CPU. This is significantly slower:
- Embedding: ~10x slower on CPU
- LLM generation: ~20-50x slower on CPU

Consider smaller models if you frequently hit VRAM limits.

## Changing Models

To switch models after initial setup:

1. **Pull new model:**
   ```bash
   ollama pull new-model-name
   ```

2. **Update environment variables:**
   ```bash
   # Windows
   set EMBED_MODEL=new-embed-model
   set LLM_MODEL=new-llm-model

   # macOS/Linux
   export EMBED_MODEL=new-embed-model
   export LLM_MODEL=new-llm-model
   ```

3. **Re-embed existing memories (if changing embedding model):**
   Use `backfill_embeddings` tool to regenerate embeddings with the new model.

   Note: Different embedding models produce incompatible vectors. After switching, old embeddings won't match new queries properly until re-embedded.

## Model Comparison Testing

To test which models work best for your use case:

1. Create test memories with known content
2. Query using different phrasings
3. Compare recall accuracy
4. Balance quality against speed

The premium models (sfr-embedding-mistral, qwen3:14b) consistently outperform smaller models, but smaller models may be sufficient for simpler use cases.
