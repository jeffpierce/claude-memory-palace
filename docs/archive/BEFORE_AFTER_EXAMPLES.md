# Memory Recall: Before vs After Examples

## Example 1: High Confidence Query (Well-Matched Results)

### Query
"What are the context window limits for Qwen3?"

### Memories Retrieved
1. [gotcha] (Ollama context window) - Similarity: 0.82
   - "Qwen3 14B degrades at 27K tokens, works reliably at 16K tokens"
2. [architecture] (TOON chunking) - Similarity: 0.75
   - "Default chunk size is 12500 tokens with safety truncation at 65K chars"

### Before Improvements
```
Qwen3 14B works with context windows up to 27K tokens but performs best at 16K tokens.
The TOON chunking system uses 12,500 token chunks with a safety truncation at 65K characters
to ensure reliable processing.
```
**Problem:** Synthesized both memories equally, even though #2 is less relevant.

### After Improvements
```
[0.82] [gotcha] (Ollama context window) Qwen3 14B degrades at 27K tokens...
[0.75] [architecture] (TOON chunking) Default chunk size is 12500 tokens...

The Qwen3 14B model degrades at 27K tokens and functions reliably at 16K tokens in the
Ollama context window. Another memory mentions a default chunk size of 12,500 tokens with
safety truncation at 65K characters, but this appears tangentially related and does not
directly specify Qwen3's overall context window limits.
```
**Better:** Qwen sees scores, acknowledges the second memory is tangential.

---

## Example 2: Low Confidence Query (Weak Results)

### Query
"How does LinkedIn authentication work?"

### Memories Retrieved
1. [event] (Installing Ollama) - Similarity: 0.35
   - "Jeff installed Ollama on 2026-01-06 for embedding generation"
2. [tool] (SFR-Embedding-Mistral) - Similarity: 0.42
   - "14GB embedding model with 32K context window"

### Before Improvements
```
LinkedIn authentication involves various technologies. The system uses tools like
Ollama and SFR-Embedding-Mistral for processing. The SFR-Embedding-Mistral is a
14GB model with 32K context window that was installed on 2026-01-06.
```
**Problem:** Forced synthesis from irrelevant results, created nonsense connection.

### After Improvements
```
NOTE: All similarity scores are below 0.5, indicating weak relevance.
Evaluate carefully whether these memories actually address the query.

[0.35] [event] (Installing Ollama) Jeff installed Ollama on 2026-01-06...
[0.42] [tool] (SFR-Embedding-Mistral) 14GB embedding model with 32K context...

The memories provided are tangentially related to AI embedding technologies but do not
directly address how LinkedIn authentication works. They mention tools like Ollama and
SFR-Embedding-Mistral, which are unrelated to authentication mechanisms or LinkedIn's
specific implementation. No relevant details about LinkedIn's authentication process are
present in the given memories.
```
**Better:** Qwen sees low scores + warning, correctly admits results don't answer the question.

---

## Example 3: Keyword Fallback (No Scores Available)

### Query
"How should Sandy interact with Jeff?"

### Memories Retrieved (Keyword Match)
1. [insight] (Sandy persona)
   - "Sandy is peer and coworker, not assistant - asks before doing things FOR Jeff"
2. [architecture] (FastAPI structure)
   - "EFaaS uses FastAPI with SQLAlchemy models for Item, TrackedEntity, Action"

### Before Improvements
```
Sandy should interact with Jeff using the FastAPI architecture. The system uses
SQLAlchemy models for Item, TrackedEntity, and Action, and Sandy operates as a peer
and coworker who asks before taking action on Jeff's behalf.
```
**Problem:** Mixed architectural detail into persona guidance.

### After Improvements
```
(No scores shown - keyword fallback)

[insight] (Sandy persona) Sandy is peer and coworker, not assistant...
[architecture] (FastAPI structure) EFaaS uses FastAPI with SQLAlchemy models...

The first memory suggests Sandy should approach Jeff as a peer and coworker, asking
for input before taking initiative on his behalf. The second memory describes a technical
architecture involving FastAPI and SQLAlchemy, which is unrelated to interpersonal
interaction guidance. These memories are tangentially related but do not directly
address how Sandy should interact with Jeff.
```
**Better:** Even without scores, Qwen evaluates relevance and separates the useful memory from the noise.

---

## Example 4: Mixed Results (Some Relevant, Some Not)

### Query
"Why can't we use LinkedIn's API?"

### Memories Retrieved
1. [blocker] (LinkedIn 2FA) - Similarity: 0.78
   - "LinkedIn API auth blocked by 2FA, pivoting to browser automation"
2. [decision] (Chrome MCP) - Similarity: 0.71
   - "Using Chrome MCP for LinkedIn scraping instead of Voyager API"
3. [event] (Recent meeting) - Similarity: -1.0 (no embedding)
   - "Had a meeting about project timeline"

### Before Improvements
```
LinkedIn API authentication is blocked by two-factor authentication requirements.
The project has pivoted to using Chrome MCP for LinkedIn scraping instead of the
Voyager API. There was also a recent meeting about the project timeline.
```
**Problem:** Included the irrelevant meeting memory, no indication of confidence.

### After Improvements
```
[0.78] [blocker] (LinkedIn 2FA) LinkedIn API auth blocked by 2FA...
[0.71] [decision] (Chrome MCP) Using Chrome MCP for LinkedIn scraping...
(Note: -1.0 score not shown for memory #3)

The use of LinkedIn's API is currently blocked due to authentication restrictions
imposed by LinkedIn's two-factor authentication (2FA) system, prompting a pivot to
browser automation tools like Chrome MCP for scraping tasks instead. Direct API access
via methods such as the Voyager API has been abandoned in favor of these workarounds,
though this approach introduces its own challenges. The mention of a recent meeting
about project timelines is tangentially related but does not directly address the
technical barriers to API usage.
```
**Better:** High-confidence memories synthesized well, low-confidence memory acknowledged as tangential, -1.0 score hidden.

---

## Key Improvements Demonstrated

### 1. Score-Aware Formatting
- **Before:** Qwen couldn't see confidence levels
- **After:** `[0.82]` scores shown inline, Qwen can weight accordingly

### 2. Relevance Evaluation
- **Before:** Forced synthesis even from weak matches
- **After:** "these memories are tangentially related but don't directly address..."

### 3. Confidence Threshold
- **Before:** No special handling for low-confidence results
- **After:** Warning note when all scores < 0.5

### 4. Graceful Degradation
- **Before:** Hallucinated connections between irrelevant memories
- **After:** Honest admission when memories don't answer the question

---

## Impact Summary

| Scenario | Before | After |
|----------|--------|-------|
| High confidence match | Good synthesis, but no score visibility | Great synthesis with confidence awareness |
| Low confidence match | Forced synthesis, hallucination | Honest "doesn't answer the question" response |
| Keyword fallback | No special handling | Works correctly without scores |
| Mixed relevance | All memories treated equally | High-confidence prioritized, low acknowledged as tangential |
| No embedding (-1.0) | Included without context | Hidden from Qwen, doesn't pollute synthesis |

---

## Testing These Examples

Run `test_recall_improvements.py` to see these scenarios in action with real Qwen3 synthesis.

```bash
cd claude-memory-palace
python test_recall_improvements.py
```

The test script demonstrates all four improvements with concrete examples.
