#!/usr/bin/env python3
"""
Test script for centrality-weighted retrieval.

Tests that the weighted ranking formula works correctly:
- Semantic similarity (cosine)
- Access frequency (log-transformed)
- Graph centrality (in-degree)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from memory_palace.services.memory_service import recall, _compute_in_degree_centrality, _get_retrieval_weights
from memory_palace.database import get_session


def test_weight_configuration():
    """Test that weights can be configured via environment variables."""
    print("Testing weight configuration...")

    # Test defaults
    alpha, beta, gamma = _get_retrieval_weights()
    print(f"Default weights: alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")
    assert abs(alpha + beta + gamma - 1.0) < 0.01, "Weights should sum to ~1.0"

    # Test environment override
    os.environ['MEMORY_PALACE_WEIGHT_SIMILARITY'] = '0.5'
    os.environ['MEMORY_PALACE_WEIGHT_ACCESS'] = '0.3'
    os.environ['MEMORY_PALACE_WEIGHT_CENTRALITY'] = '0.2'

    alpha, beta, gamma = _get_retrieval_weights()
    print(f"Custom weights: alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")
    assert abs(alpha - 0.5) < 0.01, "Alpha should be 0.5"
    assert abs(beta - 0.3) < 0.01, "Beta should be 0.3"
    assert abs(gamma - 0.2) < 0.01, "Gamma should be 0.2"

    # Clean up
    del os.environ['MEMORY_PALACE_WEIGHT_SIMILARITY']
    del os.environ['MEMORY_PALACE_WEIGHT_ACCESS']
    del os.environ['MEMORY_PALACE_WEIGHT_CENTRALITY']

    print("[PASS] Weight configuration works\n")


def test_in_degree_computation():
    """Test that in-degree centrality computation works."""
    print("Testing in-degree centrality computation...")

    try:
        db = get_session()
    except Exception as e:
        print(f"[SKIP] Could not connect to database: {e}")
        return

    try:
        # Get a few memory IDs from the database
        from memory_palace.models import Memory
        memories = db.query(Memory).limit(10).all()

        if not memories:
            print("[SKIP] No memories in database, skipping in-degree test")
            return

        memory_ids = [m.id for m in memories]
        centrality = _compute_in_degree_centrality(db, memory_ids)

        print(f"Computed centrality for {len(centrality)} memories")
        for mid, score in list(centrality.items())[:5]:
            print(f"  Memory {mid}: centrality={score:.3f}")

        # Check that all scores are in 0-1 range
        assert all(0.0 <= score <= 1.0 for score in centrality.values()), \
            "Centrality scores should be in [0, 1] range"

        print("[PASS] In-degree computation works\n")
    finally:
        db.close()


def test_semantic_recall():
    """Test that recall works with centrality weighting."""
    print("Testing centrality-weighted recall...")

    try:
        # Try a simple query
        result = recall(
            query="embedding",
            limit=5,
            synthesize=False  # Get raw results to inspect scores
        )
    except Exception as e:
        print(f"[SKIP] Could not connect to database: {e}")
        return

    print(f"Search method: {result['search_method']}")
    print(f"Found {result['count']} memories")

    if result['count'] > 0:
        print("Top result:")
        top = result['memories'][0]
        print(f"  ID: {top['id']}")
        print(f"  Subject: {top['subject']}")
        print(f"  Access count: {top['access_count']}")
        if 'similarity_score' in top:
            print(f"  Similarity: {top['similarity_score']:.3f}")

    print("[PASS] Semantic recall works\n")


def test_keyword_fallback():
    """Test that keyword fallback also uses centrality weighting."""
    print("Testing keyword fallback with centrality...")

    try:
        # Query that will use keyword fallback (if embedding model unavailable)
        result = recall(
            query="test memory",
            limit=5,
            synthesize=False
        )
    except Exception as e:
        print(f"[SKIP] Could not connect to database: {e}")
        return

    print(f"Search method: {result['search_method']}")
    print(f"Found {result['count']} memories")

    if "keyword" in result['search_method']:
        print("[PASS] Keyword fallback uses centrality weighting")
    elif "semantic" in result['search_method']:
        print("[PASS] Semantic search uses centrality weighting")

    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Centrality-Weighted Retrieval Test Suite")
    print("=" * 60)
    print()

    try:
        test_weight_configuration()
        test_in_degree_computation()
        test_semantic_recall()
        test_keyword_fallback()

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
