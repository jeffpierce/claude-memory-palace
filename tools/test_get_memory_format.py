#!/usr/bin/env python3
"""
Test script to verify get_memory_by_id response format change.
"""
import os
import sys
from pathlib import Path

# Force SQLite mode for testing and disable Ollama
os.environ["MEMORY_PALACE_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["OLLAMA_HOST"] = "http://localhost:99999"

# Add parent directory to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from memory_palace.database import init_db
from memory_palace.services import remember, get_memory_by_id, link_memories

def test_get_memory_format():
    """Test that get_memory_by_id returns the correct format."""
    print("Testing get_memory_by_id response format...\n")

    # Initialize database
    init_db()

    # Create a test memory
    print("Creating test memory...")
    mem1 = remember(
        instance_id="test",
        memory_type="test",
        content="Test memory for format verification",
        subject="Format Test",
        project="test"
    )
    print(f"Created memory {mem1['id']}: {mem1['subject']}")

    # Test 1: Single memory fetch without graph context
    print("\n--- Test 1: Single memory without graph context ---")
    result = get_memory_by_id(mem1['id'], include_graph=False)

    if result is None:
        print("ERROR: get_memory_by_id returned None!")
        return False

    # Check structure
    if 'memory' not in result:
        print("ERROR: 'memory' key not in result!")
        print(f"Result keys: {list(result.keys())}")
        return False

    if 'graph_context' in result:
        print("ERROR: 'graph_context' present when include_graph=False!")
        return False

    print("[OK] Format is correct: {'memory': {...}}")
    print(f"Memory ID: {result['memory']['id']}")
    print(f"Memory subject: {result['memory']['subject']}")

    # Test 2: Single memory with graph context (but no edges)
    print("\n--- Test 2: Single memory with graph context enabled (no edges) ---")
    result_with_graph = get_memory_by_id(mem1['id'], include_graph=True)

    if 'memory' not in result_with_graph:
        print("ERROR: 'memory' key not in result!")
        return False

    # Should not have graph_context if there are no edges
    if 'graph_context' in result_with_graph:
        print("ERROR: 'graph_context' present when memory has no edges!")
        return False

    print("[OK] No graph_context when memory has no edges (expected)")

    # Test 3: Create edges and verify graph context
    print("\n--- Test 3: Memory with actual edges ---")
    mem2 = remember(
        instance_id="test",
        memory_type="test",
        content="Related memory",
        subject="Related",
        project="test"
    )
    print(f"Created memory {mem2['id']}: {mem2['subject']}")

    link_memories(
        source_id=mem1['id'],
        target_id=mem2['id'],
        relation_type="relates_to",
        created_by="test"
    )
    print(f"Created edge: {mem1['id']} --[relates_to]--> {mem2['id']}")

    result_with_edges = get_memory_by_id(mem1['id'], include_graph=True)

    if 'memory' not in result_with_edges:
        print("ERROR: 'memory' key not in result!")
        return False

    if 'graph_context' not in result_with_edges:
        print("ERROR: 'graph_context' missing when memory has edges!")
        return False

    # Verify graph_context is at top level, not nested in memory
    if 'graph_context' in result_with_edges['memory']:
        print("ERROR: graph_context is nested inside memory dict (should be sibling)!")
        return False

    print("[OK] graph_context is at top level (sibling to 'memory')")
    print(f"Graph context keys: {list(result_with_edges['graph_context'].keys())}")

    # Verify structure
    mem_id_str = str(mem1['id'])
    if mem_id_str not in result_with_edges['graph_context']:
        print(f"ERROR: Memory ID {mem_id_str} not in graph_context!")
        return False

    edges = result_with_edges['graph_context'][mem_id_str]
    if 'outgoing' not in edges:
        print("ERROR: 'outgoing' key missing from graph context!")
        return False

    print(f"[OK] Graph context structure correct: outgoing edges = {len(edges['outgoing'])}")

    print("\n[PASS] All format tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_get_memory_format()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
