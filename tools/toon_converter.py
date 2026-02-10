#!/usr/bin/env python3
"""
JSONL to TOON (Token-Optimized Notation) Converter

Converts AI transcript JSONL files to a compact text format optimized for token efficiency.
Achieves 95%+ compression ratio while preserving conversation semantics.

Two modes:
- conservative: Keep timestamps, summarize thinking, indicate tool calls with names
- aggressive: Maximum compression, role indicators only, strip all metadata

Usage:
    python toon_converter.py input.jsonl output.toon --mode conservative
    python toon_converter.py input.jsonl output.toon --mode aggressive
    python toon_converter.py input.jsonl --chunks output_dir/ --chunk-tokens 12500
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def parse_timestamp(ts_str: str) -> Optional[str]:
    """Parse ISO timestamp to compact HH:MM:SS format."""
    if not ts_str:
        return None
    try:
        # Handle ISO format with Z or timezone
        ts_str = ts_str.replace('Z', '+00:00')
        if '.' in ts_str:
            # Truncate microseconds if too long
            parts = ts_str.split('.')
            frac_and_tz = parts[1]
            # Find where the timezone starts
            for i, c in enumerate(frac_and_tz):
                if c in ('+', '-'):
                    frac = frac_and_tz[:min(i, 6)]
                    tz = frac_and_tz[i:]
                    ts_str = f"{parts[0]}.{frac}{tz}"
                    break
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return None


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_content_conservative(content_item: dict) -> Optional[str]:
    """
    Extract content from a message content item in conservative mode.

    Preserves:
    - Full text content
    - [THINKING: ...] - summarized thinking blocks (truncated to 200 chars)
    - [TOOL: name] - tool call indicators with names
    - [RESULT:id]: preview - tool results with short preview
    """
    content_type = content_item.get("type", "")

    if content_type == "text":
        return content_item.get("text", "")

    elif content_type == "thinking":
        thinking_text = content_item.get("thinking", "")
        if thinking_text:
            return f"[THINKING: {truncate_text(thinking_text, 200)}]"
        return None

    elif content_type == "tool_use":
        tool_name = content_item.get("name", "unknown")
        return f"[TOOL: {tool_name}]"

    elif content_type == "tool_result":
        tool_id = content_item.get("tool_use_id", "")[:8] if content_item.get("tool_use_id") else ""
        # Check for nested content
        nested = content_item.get("content", [])
        if isinstance(nested, list) and nested:
            first_text = next((c.get("text", "")[:100] for c in nested if c.get("type") == "text"), "")
            if first_text:
                return f"[RESULT:{tool_id}]: {truncate_text(first_text, 150)}"
        return f"[RESULT:{tool_id}]"

    return None


def extract_content_aggressive(content_item: dict) -> Optional[str]:
    """
    Extract content from a message content item in aggressive mode.

    Maximum compression:
    - Full text content preserved
    - Thinking blocks stripped entirely
    - Tool calls reduced to [TOOL:name]
    - Tool results stripped (redundant with tool use indicator)
    """
    content_type = content_item.get("type", "")

    if content_type == "text":
        return content_item.get("text", "")

    elif content_type == "thinking":
        return None  # Strip thinking entirely

    elif content_type == "tool_use":
        tool_name = content_item.get("name", "unknown")
        return f"[TOOL:{tool_name}]"

    elif content_type == "tool_result":
        return None  # Strip tool results (redundant with tool use)

    return None


def get_role_conservative(record_type: str, role: str = "") -> str:
    """Get role identifier for conservative mode."""
    if record_type == "user":
        return "USER"
    elif record_type == "assistant":
        return "ASSISTANT"
    elif record_type == "summary":
        return "SUMMARY"
    elif role:
        return role.upper()
    return record_type.upper()


def get_role_aggressive(record_type: str, role: str = "") -> str:
    """Get role identifier for aggressive mode."""
    if record_type == "user" or role == "user":
        return "U"
    elif record_type == "assistant" or role == "assistant":
        return "A"
    elif record_type == "summary":
        return "S"
    elif record_type == "system":
        return "SYS"
    return record_type[0].upper() if record_type else "?"


def process_record_conservative(record: dict) -> Optional[str]:
    """
    Process a single JSONL record in conservative mode.

    Format: [HH:MM:SS] ROLE: content
    """
    record_type = record.get("type", "")

    # Skip metadata-only records
    if record_type in ("file-history-snapshot",):
        return None

    # Get timestamp
    timestamp = parse_timestamp(record.get("timestamp", ""))
    ts_prefix = f"[{timestamp}] " if timestamp else ""

    # Get role
    message = record.get("message", {})
    role = message.get("role", "")
    role_str = get_role_conservative(record_type, role)

    # Extract content
    content_parts = []
    content = message.get("content", [])

    if isinstance(content, str):
        content_parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                extracted = extract_content_conservative(item)
                if extracted:
                    content_parts.append(extracted)
            elif isinstance(item, str):
                content_parts.append(item)

    if not content_parts:
        return None

    # Combine content
    combined = "\n".join(content_parts)

    return f"{ts_prefix}{role_str}: {combined}"


def process_record_aggressive(record: dict) -> Optional[str]:
    """
    Process a single JSONL record in aggressive mode.

    Format: R: content (where R is single-char role indicator)
    Whitespace collapsed for maximum compression.
    """
    record_type = record.get("type", "")

    # Skip metadata-only records
    if record_type in ("file-history-snapshot",):
        return None

    # Get role
    message = record.get("message", {})
    role = message.get("role", "")
    role_str = get_role_aggressive(record_type, role)

    # Extract content
    content_parts = []
    content = message.get("content", [])

    if isinstance(content, str):
        content_parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                extracted = extract_content_aggressive(item)
                if extracted:
                    content_parts.append(extracted)
            elif isinstance(item, str):
                content_parts.append(item)

    if not content_parts:
        return None

    # Combine content - strip excessive whitespace in aggressive mode
    combined = " ".join(content_parts)
    # Collapse multiple spaces/newlines
    combined = " ".join(combined.split())

    return f"{role_str}: {combined}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def convert_jsonl_to_toon(input_path: str, output_path: str, mode: str = "conservative") -> dict:
    """
    Convert JSONL transcript to TOON format (single file output).

    Args:
        input_path: Path to source JSONL file
        output_path: Path for output TOON file
        mode: 'conservative' or 'aggressive'

    Returns:
        Stats dict with original_size, converted_size, compression_ratio, estimated_tokens.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    original_size = input_path.stat().st_size

    # Select processor based on mode
    if mode == "conservative":
        processor = process_record_conservative
    elif mode == "aggressive":
        processor = process_record_aggressive
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'conservative' or 'aggressive'.")

    output_lines = []
    records_processed = 0
    records_skipped = 0
    errors = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                result = processor(record)
                if result:
                    output_lines.append(result)
                    records_processed += 1
                else:
                    records_skipped += 1
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 3:
                    print(f"Warning: JSON decode error on line {line_num}: {e}", file=sys.stderr)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)

    # Write output - conservative uses double newlines, aggressive uses single
    output_content = "\n\n".join(output_lines) if mode == "conservative" else "\n".join(output_lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_content)

    converted_size = output_path.stat().st_size
    compression_ratio = (1 - converted_size / original_size) * 100 if original_size > 0 else 0
    estimated_tokens = converted_size // 4  # Rough estimate: ~4 bytes per token

    return {
        "original_size": original_size,
        "converted_size": converted_size,
        "compression_ratio": compression_ratio,
        "estimated_tokens": estimated_tokens,
        "records_processed": records_processed,
        "records_skipped": records_skipped,
        "errors": errors,
    }


def convert_jsonl_to_toon_chunks(
    input_path: str,
    output_dir: str,
    mode: str = "aggressive",
    chunk_tokens: int = 12500
) -> dict:
    """
    Convert JSONL transcript to chunked TOON files.

    Splits on USER record boundaries when approaching chunk_tokens limit.
    This ensures clean conversation breaks for downstream processing.

    Args:
        input_path: Path to source JSONL file
        output_dir: Directory to write chunk files
        mode: 'conservative' or 'aggressive'
        chunk_tokens: Target tokens per chunk (default 12500 - leaves headroom
                      for LLM reasoning + output tokens in downstream processing)

    Returns:
        Dict with chunk_count, chunk_files, total_tokens, compression_ratio, stats per chunk
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    original_size = input_path.stat().st_size

    # Select processor based on mode
    if mode == "conservative":
        processor = process_record_conservative
        separator = "\n\n"
    elif mode == "aggressive":
        processor = process_record_aggressive
        separator = "\n"
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'conservative' or 'aggressive'.")

    # Parse all records first, tracking which are USER boundaries
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record_type = record.get("type", "")
                processed = processor(record)
                if processed:
                    records.append({
                        "content": processed,
                        "tokens": estimate_tokens(processed),
                        "is_user": record_type == "user"
                    })
            except (json.JSONDecodeError, Exception):
                pass  # Skip malformed records

    if not records:
        return {
            "error": "No valid records found in input file",
            "chunk_count": 0,
            "chunk_files": [],
            "original_size": original_size
        }

    # Build chunks, splitting on USER boundaries when near limit
    chunks = []
    current_chunk = []
    current_tokens = 0

    for i, record in enumerate(records):
        # Check if we should start a new chunk:
        # - We're at or past the token limit
        # - This is a USER record (conversation boundary)
        # - We have content in current chunk
        if (current_tokens >= chunk_tokens and
            record["is_user"] and
            current_chunk):
            # Save current chunk
            chunks.append({
                "records": current_chunk,
                "tokens": current_tokens
            })
            current_chunk = []
            current_tokens = 0

        current_chunk.append(record["content"])
        current_tokens += record["tokens"]

    # Don't forget the last chunk
    if current_chunk:
        chunks.append({
            "records": current_chunk,
            "tokens": current_tokens
        })

    # Write chunk files
    chunk_files = []
    chunk_stats = []
    total_converted_size = 0

    base_name = input_path.stem

    for i, chunk in enumerate(chunks, 1):
        chunk_filename = f"{base_name}_chunk_{i:03d}.toon"
        chunk_path = output_dir / chunk_filename

        content = separator.join(chunk["records"])

        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(content)

        chunk_size = chunk_path.stat().st_size
        total_converted_size += chunk_size

        chunk_files.append(str(chunk_path))
        chunk_stats.append({
            "file": chunk_filename,
            "tokens": chunk["tokens"],
            "size": chunk_size,
            "records": len(chunk["records"])
        })

    compression_ratio = (1 - total_converted_size / original_size) * 100 if original_size > 0 else 0

    return {
        "chunk_count": len(chunks),
        "chunk_files": chunk_files,
        "chunk_stats": chunk_stats,
        "total_tokens": sum(c["tokens"] for c in chunks),
        "original_size": original_size,
        "converted_size": total_converted_size,
        "compression_ratio": compression_ratio,
        "mode": mode,
        "target_chunk_tokens": chunk_tokens
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert AI transcript JSONL to TOON (Token-Optimized Notation) format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  conservative  Keep timestamps (HH:MM:SS), summarize thinking blocks,
                indicate tool calls with names. Readable format.

  aggressive    Maximum compression. Role indicators only (U:/A:/S:),
                strip ALL metadata, timestamps, thinking content.
                Tool calls become just [TOOL:name].

Examples:
  # Single file conversion
  python toon_converter.py transcript.jsonl output.toon
  python toon_converter.py transcript.jsonl output.toon --mode aggressive

  # Chunked conversion (for large files)
  python toon_converter.py transcript.jsonl --chunks ./chunks/
  python toon_converter.py transcript.jsonl --chunks ./chunks/ --chunk-tokens 10000 --mode conservative
"""
    )

    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("output", nargs="?", help="Output TOON file path (for single file mode)")
    parser.add_argument(
        "--mode",
        choices=["conservative", "aggressive"],
        default="aggressive",
        help="Conversion mode (default: aggressive)"
    )
    parser.add_argument(
        "--chunks",
        metavar="DIR",
        help="Output directory for chunked conversion (enables chunk mode)"
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=12500,
        help="Target tokens per chunk (default: 12500)"
    )

    args = parser.parse_args()

    try:
        if args.chunks:
            # Chunked mode
            stats = convert_jsonl_to_toon_chunks(
                args.input,
                args.chunks,
                args.mode,
                args.chunk_tokens
            )

            if stats.get("error"):
                print(f"Error: {stats['error']}", file=sys.stderr)
                sys.exit(1)

            print(f"\nChunked conversion complete ({args.mode} mode)")
            print(f"{'=' * 50}")
            print(f"Original size:     {format_size(stats['original_size'])}")
            print(f"Converted size:    {format_size(stats['converted_size'])}")
            print(f"Compression ratio: {stats['compression_ratio']:.1f}%")
            print(f"Total tokens:      ~{stats['total_tokens']:,}")
            print(f"Chunks created:    {stats['chunk_count']}")
            print(f"Target per chunk:  {stats['target_chunk_tokens']:,} tokens")
            print(f"{'=' * 50}")
            print("\nChunk details:")
            for cs in stats['chunk_stats']:
                print(f"  {cs['file']}: {cs['tokens']:,} tokens, {cs['records']} records")
            print(f"\nOutput written to: {args.chunks}")

        else:
            # Single file mode
            if not args.output:
                print("Error: output path required for single file mode", file=sys.stderr)
                print("Use --chunks DIR for chunked mode", file=sys.stderr)
                sys.exit(1)

            stats = convert_jsonl_to_toon(args.input, args.output, args.mode)

            print(f"\nConversion complete ({args.mode} mode)")
            print(f"{'=' * 40}")
            print(f"Original size:     {format_size(stats['original_size'])}")
            print(f"Converted size:    {format_size(stats['converted_size'])}")
            print(f"Compression ratio: {stats['compression_ratio']:.1f}%")
            print(f"Estimated tokens:  ~{stats['estimated_tokens']:,}")
            print(f"{'=' * 40}")
            print(f"Records processed: {stats['records_processed']}")
            print(f"Records skipped:   {stats['records_skipped']}")
            if stats['errors']:
                print(f"Errors:            {stats['errors']}")
            print(f"\nOutput written to: {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
