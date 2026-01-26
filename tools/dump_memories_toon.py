#!/usr/bin/env python3
"""
Dump all memories in TOON-style token-optimized format for LLM analysis.

Output format (designed for minimal tokens while preserving semantics):

M#167|identity|memory-palace|10|Sandy's History and Origins
K:sandy history,identity,personhood,memory palace
---
[full content here]
===

Usage:
    python dump_memories_toon.py > memories.toon
    python dump_memories_toon.py --min-importance 7 > important_memories.toon
    python dump_memories_toon.py --project life > life_memories.toon
    python dump_memories_toon.py --ids 167,168,169 > specific.toon
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_palace.database import get_session
from memory_palace.models import Memory


def dump_memory_toon(memory: Memory) -> str:
    """Convert a single memory to TOON format."""
    lines = []
    
    # Header line: M#id|type|project|importance|subject
    subject = memory.subject or "(no subject)"
    lines.append(f"M#{memory.id}|{memory.memory_type}|{memory.project}|{memory.importance}|{subject}")
    
    # Keywords line (if any)
    if memory.keywords:
        lines.append(f"K:{','.join(memory.keywords)}")
    
    # Tags line (if any and different from keywords)
    if memory.tags:
        lines.append(f"T:{','.join(memory.tags)}")
    
    # Content separator and content
    lines.append("---")
    lines.append(memory.content)
    lines.append("===")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Dump memories in TOON format")
    parser.add_argument("--min-importance", type=int, help="Minimum importance filter")
    parser.add_argument("--project", type=str, help="Filter by project")
    parser.add_argument("--type", type=str, help="Filter by memory type")
    parser.add_argument("--ids", type=str, help="Comma-separated list of specific IDs")
    parser.add_argument("--include-archived", action="store_true", help="Include archived memories")
    parser.add_argument("--output", "-o", type=str, help="Output file (default: stdout)")
    args = parser.parse_args()
    
    db = get_session()
    try:
        query = db.query(Memory)
        
        # Apply filters
        if not args.include_archived:
            query = query.filter(Memory.is_archived == False)
        
        if args.min_importance:
            query = query.filter(Memory.importance >= args.min_importance)
        
        if args.project:
            query = query.filter(Memory.project == args.project)
        
        if args.type:
            query = query.filter(Memory.memory_type == args.type)
        
        if args.ids:
            id_list = [int(x.strip()) for x in args.ids.split(",")]
            query = query.filter(Memory.id.in_(id_list))
        
        # Order by ID for consistent output
        query = query.order_by(Memory.id)
        
        memories = query.all()
        
        # Build output
        output_lines = [
            f"# MEMORY DUMP — {len(memories)} memories",
            f"# Format: M#id|type|project|importance|subject",
            f"# K:keywords  T:tags  ---content===",
            ""
        ]
        
        for memory in memories:
            output_lines.append(dump_memory_toon(memory))
            output_lines.append("")  # Blank line between memories
        
        output_text = "\n".join(output_lines)
        
        # Write output
        if args.output:
            Path(args.output).write_text(output_text, encoding="utf-8")
            print(f"Wrote {len(memories)} memories to {args.output}", file=sys.stderr)
        else:
            print(output_text)
            
    finally:
        db.close()


if __name__ == "__main__":
    main()
