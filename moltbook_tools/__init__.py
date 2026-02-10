"""
Moltbook Submission Gateway — mechanical interlocks for safe content posting.

All submissions to Moltbook go through this gateway. It enforces:
1. Session guard (no retry loops)
2. Content hash dedup (no exact duplicates)
3. Word count gate (platform-appropriate length)
4. Similarity check (no near-duplicates)
5. Rate limiting (matches Moltbook server limits)
6. QC gate (must have approval token)

The gateway holds the API credentials. Agents don't get the key directly.
"""

__version__ = "1.0.0"
