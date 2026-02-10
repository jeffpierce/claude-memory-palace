"""
Test script for maintenance tools.

Demonstrates how to use memory_audit, memory_archive, and memory_reembed
via the service layer (bypassing MCP for direct testing).
"""

from memory_palace.services import (
    audit_palace,
    batch_archive_memories,
    reembed_memories,
    remember
)


def test_audit():
    """Test the audit functionality."""
    print("=" * 60)
    print("TESTING: memory_audit")
    print("=" * 60)

    result = audit_palace(
        check_duplicates=True,
        check_stale=True,
        check_orphan_edges=True,
        check_embeddings=True,
        check_contradictions=True,
        stale_days=90,
        limit_per_category=5  # Keep output small for testing
    )

    print(f"\nAudit Summary:")
    print(f"  Total issues found: {result['summary']['total_issues']}")
    print(f"  - Duplicates: {result['summary']['duplicates_found']}")
    print(f"  - Stale memories: {result['summary']['stale_found']}")
    print(f"  - Orphan edges: {result['summary']['orphan_edges_found']}")
    print(f"  - Missing embeddings: {result['summary']['missing_embeddings_found']}")
    print(f"  - Contradictions: {result['summary']['contradictions_found']}")

    if result.get('duplicates'):
        print("\nSample duplicates:")
        for dup in result['duplicates'][:3]:
            print(f"  - Memory #{dup['memory_id']} similar to #{dup['similar_to']} "
                  f"(similarity: {dup['similarity']})")
            print(f"    Subjects: '{dup['subject_1']}' vs '{dup['subject_2']}'")

    if result.get('stale'):
        print("\nSample stale memories:")
        for stale in result['stale'][:3]:
            print(f"  - Memory #{stale['memory_id']}: {stale['subject']}")
            print(f"    Age: {stale['age_days']} days, Access: {stale['access_count']}, "
                  f"In-degree: {stale['in_degree']}")

    return result


def test_batch_archive_dry_run():
    """Test batch archive in dry run mode."""
    print("\n" + "=" * 60)
    print("TESTING: memory_archive (DRY RUN)")
    print("=" * 60)

    # Test with aggressive criteria to see what would be archived
    result = batch_archive_memories(
        older_than_days=180,
        max_access_count=2,
        centrality_protection=True,
        min_centrality_threshold=5,
        dry_run=True
    )

    print(f"\nDry Run Results:")
    print(f"  Would archive: {result['would_archive']} memories")
    print(f"  Protected: {len(result.get('protected', []))} memories")

    if result.get('memories'):
        print("\nSample memories that would be archived:")
        for mem in result['memories'][:3]:
            print(f"  - Memory #{mem['id']}: {mem['subject']}")
            print(f"    Type: {mem['type']}, Age: {mem['age_days']} days, "
                  f"Access: {mem['access_count']}")

    if result.get('protected'):
        print("\nSample protected memories (high centrality):")
        for prot in result['protected'][:3]:
            print(f"  - Memory #{prot['id']}: {prot['subject']}")
            print(f"    {prot['reason']}")

    return result


def test_reembed_dry_run():
    """Test reembed in dry run mode."""
    print("\n" + "=" * 60)
    print("TESTING: memory_reembed (DRY RUN)")
    print("=" * 60)

    result = reembed_memories(
        older_than_days=365,  # Very old embeddings
        dry_run=True
    )

    if 'error' in result:
        print(f"\nNo memories match criteria: {result['error']}")
    else:
        print(f"\nDry Run Results:")
        print(f"  Would re-embed: {result['would_reembed']} memories")
        print(f"  Estimated time: {result['estimated_time_seconds']} seconds")

        if result.get('memories'):
            print("\nSample memories that would be re-embedded:")
            for mem in result['memories'][:3]:
                print(f"  - Memory #{mem['id']}: {mem['subject']}")
                print(f"    Type: {mem['type']}")

    return result


def main():
    """Run all maintenance tests."""
    print("\n")
    print("+" + "=" * 58 + "+")
    print("|  Memory Palace Maintenance Tools - Test Suite           |")
    print("+" + "=" * 58 + "+")

    try:
        # Test 1: Audit
        audit_result = test_audit()

        # Test 2: Batch archive (dry run)
        archive_result = test_batch_archive_dry_run()

        # Test 3: Reembed (dry run)
        reembed_result = test_reembed_dry_run()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

        print("\nNext steps:")
        print("1. Review the audit findings above")
        print("2. To actually archive memories, call batch_archive with dry_run=False")
        print("3. To actually re-embed, call reembed_memories with dry_run=False")
        print("\nReminder: All destructive operations default to dry_run=True for safety.")

    except Exception as e:
        print(f"\n[ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
