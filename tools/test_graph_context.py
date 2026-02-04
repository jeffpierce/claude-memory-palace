#!/usr/bin/env python3
"""
Test script for graph context in memory_recall.

This verifies that the new include_graph parameter works correctly
and returns the expected structure.
"""
import os
import sys
from pathlib import Path

# Force SQLite mode for testing and disable Ollama (to avoid embedding generation issues)
os.environ["MEMORY_PALACE_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["OLLAMA_HOST"] = "http://localhost:99999"  # Unreachable - disables embeddings

# Add parent directory to path for imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from memory_palace.database import init_db
from memory_palace.services import recall, remember, link_memories

def test_graph_context():
    """Test that graph context is included in recall results."""
    print("Testing graph context in memory_recall...\n")

    # Initialize database
    init_db()

    # Create test memories
    print("Creating test memories...")
    mem1 = remember(
        instance_id="test",
        memory_type="test",
        content="This is the root concept for testing graph context",
        subject="Root Concept",
        project="test"
    )
    print(f"Created memory {mem1['id']}: {mem1['subject']}")

    mem2 = remember(
        instance_id="test",
        memory_type="test",
        content="This exemplifies the root concept",
        subject="Example 1",
        project="test"
    )
    print(f"Created memory {mem2['id']}: {mem2['subject']}")

    mem3 = remember(
        instance_id="test",
        memory_type="test",
        content="This also relates to the root concept",
        subject="Example 2",
        project="test"
    )
    print(f"Created memory {mem3['id']}: {mem3['subject']}")

    # Create edges
    print("\nCreating edges...")
    link_memories(
        source_id=mem2['id'],
        target_id=mem1['id'],
        relation_type="exemplifies",
        created_by="test"
    )
    print(f"Created edge: {mem2['id']} --[exemplifies]--> {mem1['id']}")

    link_memories(
        source_id=mem3['id'],
        target_id=mem1['id'],
        relation_type="relates_to",
        bidirectional=True,
        created_by="test"
    )
    print(f"Created edge: {mem3['id']} <--[relates_to]--> {mem1['id']}")

    # Test recall with graph context (default: include_graph=True)
    print("\n--- Test 1: Recall with graph context (default) ---")
    result = recall(
        query="root concept",
        project="test",
        synthesize=False,
        limit=10
    )

    print(f"Found {result['count']} memories")
    print(f"Search method: {result['search_method']}")

    if 'graph_context' in result:
        print(f"\nGraph context included for {len(result['graph_context'])} memories:")
        for memory_id, context in result['graph_context'].items():
            print(f"\nMemory #{memory_id}:")
            if 'outgoing' in context:
                print(f"  Outgoing edges: {len(context['outgoing'])}")
                for edge in context['outgoing']:
                    print(f"    -> #{edge['target_id']} ({edge['target_subject']}) [{edge['relation_type']}]")
            if 'incoming' in context:
                print(f"  Incoming edges: {len(context['incoming'])}")
                for edge in context['incoming']:
                    print(f"    <- #{edge['source_id']} ({edge['source_subject']}) [{edge['relation_type']}]")
    else:
        print("ERROR: No graph_context in result!")
        return False

    # Test recall without graph context
    print("\n--- Test 2: Recall without graph context ---")
    result_no_graph = recall(
        query="root concept",
        project="test",
        synthesize=False,
        include_graph=False,
        limit=10
    )

    if 'graph_context' in result_no_graph:
        print("ERROR: graph_context present when include_graph=False!")
        return False
    else:
        print("[OK] No graph_context when include_graph=False (expected)")

    # Test with synthesize=True
    print("\n--- Test 3: Recall with synthesize=True and graph context ---")
    result_synthesized = recall(
        query="root concept",
        project="test",
        synthesize=True,
        limit=10
    )

    if 'graph_context' in result_synthesized:
        print(f"[OK] Graph context included with synthesize=True")
        print(f"  Memory IDs: {result_synthesized.get('memory_ids', [])}")
        print(f"  Graph context keys: {list(result_synthesized['graph_context'].keys())}")
    else:
        print("Note: No graph context (memories may have no edges)")

    # Test graph_top_n parameter
    print("\n--- Test 4: Test graph_top_n parameter ---")
    result_top_2 = recall(
        query="concept",
        project="test",
        synthesize=False,
        include_graph=True,
        graph_top_n=2,
        limit=10
    )

    if 'graph_context' in result_top_2:
        print(f"[OK] Graph context with graph_top_n=2: {len(result_top_2['graph_context'])} memories")
        if len(result_top_2['graph_context']) <= 2:
            print("  [OK] Correctly limited to top 2 results")
        else:
            print(f"  ERROR: Expected <= 2, got {len(result_top_2['graph_context'])}")
            return False

    print("\n[PASS] All tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_graph_context()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
