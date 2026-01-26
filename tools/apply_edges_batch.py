#!/usr/bin/env python3
"""
Batch apply edges to the memory palace knowledge graph.

Usage:
    python apply_edges_batch.py edges.json [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_palace.services import link_memories


def main():
    parser = argparse.ArgumentParser(description="Batch apply edges")
    parser.add_argument("edges_file", help="JSON file with edge proposals")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually create edges")
    parser.add_argument("--created-by", default="clawdbot", help="Instance ID for created_by field")
    args = parser.parse_args()
    
    # Load edges
    edges = json.loads(Path(args.edges_file).read_text())
    
    print(f"Loaded {len(edges)} edge proposals")
    
    created = 0
    skipped = 0
    errors = 0
    
    for edge in edges:
        source_id = edge["source_id"]
        target_id = edge["target_id"]
        relation_type = edge["relation_type"]
        bidirectional = edge.get("bidirectional", False)
        confidence = edge.get("confidence", "unknown")
        reasoning = edge.get("reasoning", "")
        
        direction = "<->" if bidirectional else "->"
        print(f"  #{source_id} {direction}[{relation_type}]{direction} #{target_id} ({confidence})")
        
        if args.dry_run:
            print(f"    [DRY RUN] Would create edge")
            created += 1
            continue
        
        result = link_memories(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            bidirectional=bidirectional,
            metadata={"reasoning": reasoning, "confidence": confidence},
            created_by=args.created_by
        )
        
        if "error" in result:
            if "already exists" in result["error"]:
                print(f"    [SKIP] {result['error']}")
                skipped += 1
            else:
                print(f"    [ERROR] {result['error']}")
                errors += 1
        else:
            print(f"    [OK] Created edge #{result['id']}")
            created += 1
    
    print(f"\nSummary: {created} created, {skipped} skipped (already exist), {errors} errors")


if __name__ == "__main__":
    main()
