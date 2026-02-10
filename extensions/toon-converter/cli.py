"""
CLI entry point for TOON converter.

Usage:
    python -m extensions.toon-converter.cli input.jsonl output.toon
    python -m extensions.toon-converter.cli input.jsonl --output-dir ./converted/
    python -m extensions.toon-converter.cli input_dir/ --output-dir output_dir/
"""
import argparse
import sys
from pathlib import Path

from .converter import convert_file, convert_directory, format_size


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL files to TOON (Token-Optimized Notation) format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file with explicit output path
  python -m extensions.toon-converter.cli input.jsonl output.toon

  # Convert single file (auto-generates output.toon name)
  python -m extensions.toon-converter.cli input.jsonl

  # Convert with output directory specified
  python -m extensions.toon-converter.cli input.jsonl --output-dir ./converted/

  # Batch convert directory
  python -m extensions.toon-converter.cli input_dir/ --output-dir output_dir/

  # Use conservative mode (more readable, less compression)
  python -m extensions.toon-converter.cli input.jsonl --mode conservative

Modes:
  aggressive    Maximum compression (default)
                - Single-char role indicators (U:/A:/S:)
                - Strip timestamps, thinking, tool results
                - Collapse whitespace

  conservative  More readable, still good compression
                - Keep timestamps (HH:MM:SS)
                - Summarize thinking blocks
                - Show tool names and results
"""
    )

    parser.add_argument(
        "input",
        help="Input JSONL file or directory"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output TOON file path (for single file) or directory (for batch)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (alternative to positional output argument)"
    )
    parser.add_argument(
        "--mode",
        choices=["conservative", "aggressive"],
        default="aggressive",
        help="Conversion mode (default: aggressive)"
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="File pattern for directory conversion (default: *.jsonl)"
    )

    args = parser.parse_args()

    # Determine input type (file or directory)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output_arg = args.output_dir if args.output_dir else args.output

    try:
        if input_path.is_file():
            # Single file conversion
            result = convert_file(
                str(input_path),
                output_arg,
                args.mode
            )

            if "error" in result:
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)

            # Print stats
            print(f"\nConversion complete ({args.mode} mode)")
            print("=" * 60)
            print(f"Input:             {result['input_file']}")
            print(f"Output:            {result['output_file']}")
            print(f"Original size:     {format_size(result['original_size'])}")
            print(f"Converted size:    {format_size(result['converted_size'])}")
            print(f"Compression ratio: {result['compression_ratio']:.1f}%")
            print(f"Estimated tokens:  ~{result['estimated_tokens']:,}")
            print("=" * 60)
            print(f"Records processed: {result['records_processed']}")
            print(f"Records skipped:   {result['records_skipped']}")
            if result.get('errors', 0) > 0:
                print(f"Errors:            {result['errors']}")
            print()

        elif input_path.is_dir():
            # Directory conversion
            if not output_arg:
                print("Error: Output directory required for batch conversion", file=sys.stderr)
                print("Use: python -m extensions.toon-converter.cli input_dir/ output_dir/", file=sys.stderr)
                sys.exit(1)

            result = convert_directory(
                str(input_path),
                output_arg,
                args.mode,
                args.pattern
            )

            # Print batch stats
            print(f"\nBatch conversion complete ({args.mode} mode)")
            print("=" * 60)
            print(f"Input directory:   {result['input_dir']}")
            print(f"Output directory:  {result['output_dir']}")
            print(f"Files processed:   {result['files_processed']}")
            print(f"Files failed:      {result['files_failed']}")
            print(f"Total original:    {format_size(result['total_original_size'])}")
            print(f"Total converted:   {format_size(result['total_converted_size'])}")
            print(f"Avg compression:   {result['avg_compression_ratio']:.1f}%")
            print("=" * 60)

            # Show per-file results
            if result['file_results']:
                print("\nPer-file results:")
                for file_result in result['file_results']:
                    status = file_result['status']
                    input_file = Path(file_result['input_file']).name

                    if status == "success":
                        ratio = file_result['compression_ratio']
                        size = format_size(file_result['converted_size'])
                        print(f"  [OK] {input_file}: {ratio:.1f}% compression, {size}")
                    else:
                        error = file_result.get('error', 'Unknown error')
                        print(f"  [FAIL] {input_file}: {error}")
            print()

        else:
            print(f"Error: Input path is neither a file nor directory: {input_path}", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nConversion interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: Unexpected error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
