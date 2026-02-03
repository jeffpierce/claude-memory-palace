#!/usr/bin/env python3
"""
Example: Centrality-Weighted Memory Search

Demonstrates how to use the centrality-weighted retrieval feature with different
weight configurations to explore how ranking changes.
"""

import os
from memory_palace.services.memory_service import recall


def example_default_weights():
    """Example: Search with default weights (balanced)."""
    print("=" * 60)
    print("Example 1: Default Weights (Balanced)")
    print("=" * 60)
    print("Weights: similarity=0.7, access=0.15, centrality=0.15")
    print()

    result = recall(
        query="embedding generation",
        limit=5,
        synthesize=False
    )

    print(f"Search method: {result['search_method']}")
    print(f"Found {result['count']} memories\n")

    for i, mem in enumerate(result['memories'][:5], 1):
        print(f"{i}. [{mem['memory_type']}] {mem['subject']}")
        print(f"   Access count: {mem['access_count']}")
        if 'similarity_score' in mem:
            print(f"   Similarity: {mem['similarity_score']:.3f}")
        print()


def example_favor_access():
    """Example: Favor frequently-accessed memories."""
    print("=" * 60)
    print("Example 2: Favor Frequently-Accessed Memories")
    print("=" * 60)
    print("Weights: similarity=0.5, access=0.3, centrality=0.2")
    print()

    # Set environment variables for this search
    os.environ['MEMORY_PALACE_WEIGHT_SIMILARITY'] = '0.5'
    os.environ['MEMORY_PALACE_WEIGHT_ACCESS'] = '0.3'
    os.environ['MEMORY_PALACE_WEIGHT_CENTRALITY'] = '0.2'

    result = recall(
        query="embedding generation",
        limit=5,
        synthesize=False
    )

    # Clean up environment
    del os.environ['MEMORY_PALACE_WEIGHT_SIMILARITY']
    del os.environ['MEMORY_PALACE_WEIGHT_ACCESS']
    del os.environ['MEMORY_PALACE_WEIGHT_CENTRALITY']

    print(f"Search method: {result['search_method']}")
    print(f"Found {result['count']} memories\n")

    for i, mem in enumerate(result['memories'][:5], 1):
        print(f"{i}. [{mem['memory_type']}] {mem['subject']}")
        print(f"   Access count: {mem['access_count']}")
        if 'similarity_score' in mem:
            print(f"   Similarity: {mem['similarity_score']:.3f}")
        print()


def example_favor_centrality():
    """Example: Favor graph hub memories."""
    print("=" * 60)
    print("Example 3: Favor Graph Hub Memories")
    print("=" * 60)
    print("Weights: similarity=0.5, access=0.1, centrality=0.4")
    print()

    os.environ['MEMORY_PALACE_WEIGHT_SIMILARITY'] = '0.5'
    os.environ['MEMORY_PALACE_WEIGHT_ACCESS'] = '0.1'
    os.environ['MEMORY_PALACE_WEIGHT_CENTRALITY'] = '0.4'

    result = recall(
        query="embedding generation",
        limit=5,
        synthesize=False
    )

    # Clean up environment
    del os.environ['MEMORY_PALACE_WEIGHT_SIMILARITY']
    del os.environ['MEMORY_PALACE_WEIGHT_ACCESS']
    del os.environ['MEMORY_PALACE_WEIGHT_CENTRALITY']

    print(f"Search method: {result['search_method']}")
    print(f"Found {result['count']} memories\n")

    for i, mem in enumerate(result['memories'][:5], 1):
        print(f"{i}. [{mem['memory_type']}] {mem['subject']}")
        print(f"   Access count: {mem['access_count']}")
        if 'similarity_score' in mem:
            print(f"   Similarity: {mem['similarity_score']:.3f}")
        print()


def example_pure_semantic():
    """Example: Pure semantic search (disable access & centrality)."""
    print("=" * 60)
    print("Example 4: Pure Semantic Search")
    print("=" * 60)
    print("Weights: similarity=1.0, access=0.0, centrality=0.0")
    print()

    os.environ['MEMORY_PALACE_WEIGHT_SIMILARITY'] = '1.0'
    os.environ['MEMORY_PALACE_WEIGHT_ACCESS'] = '0.0'
    os.environ['MEMORY_PALACE_WEIGHT_CENTRALITY'] = '0.0'

    result = recall(
        query="embedding generation",
        limit=5,
        synthesize=False
    )

    # Clean up environment
    del os.environ['MEMORY_PALACE_WEIGHT_SIMILARITY']
    del os.environ['MEMORY_PALACE_WEIGHT_ACCESS']
    del os.environ['MEMORY_PALACE_WEIGHT_CENTRALITY']

    print(f"Search method: {result['search_method']}")
    print(f"Found {result['count']} memories\n")

    for i, mem in enumerate(result['memories'][:5], 1):
        print(f"{i}. [{mem['memory_type']}] {mem['subject']}")
        print(f"   Access count: {mem['access_count']}")
        if 'similarity_score' in mem:
            print(f"   Similarity: {mem['similarity_score']:.3f}")
        print()


if __name__ == "__main__":
    print()
    print("Centrality-Weighted Memory Search Examples")
    print()

    try:
        example_default_weights()
        example_favor_access()
        example_favor_centrality()
        example_pure_semantic()

        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: These examples require a configured memory database.")
        print("If you see connection errors, check your database configuration.")
