"""
Core conversion logic for JSONL to TOON format.

This module wraps the existing tools/toon_converter.py implementation,
providing a clean API for both CLI and MCP server usage.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import the existing converter implementation
tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

try:
    from toon_converter import convert_jsonl_to_toon as _convert_single
except ImportError as e:
    raise ImportError(
        f"Failed to import toon_converter from tools/: {e}\n"
        "Make sure tools/toon_converter.py exists."
    )


def convert_file(
    input_path: str,
    output_path: Optional[str] = None,
    mode: str = "aggressive"
) -> Dict[str, Any]:
    """
    Convert a JSONL file to TOON format.

    Args:
        input_path: Path to source JSONL file
        output_path: Path for output TOON file (if None, uses same name with .toon extension)
        mode: Conversion mode - "conservative" or "aggressive" (default: "aggressive")

    Returns:
        Dict with conversion stats:
            - original_size: Size of input file in bytes
            - converted_size: Size of output file in bytes
            - compression_ratio: Compression percentage (0-100)
            - estimated_tokens: Rough token count for converted file
            - records_processed: Number of records successfully converted
            - records_skipped: Number of records skipped
            - errors: Number of errors encountered
            - input_file: Path to input file
            - output_file: Path to output file

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If mode is invalid
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Auto-generate output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix('.toon')
    else:
        output_path = Path(output_path)

    # Validate mode
    if mode not in ("conservative", "aggressive"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'conservative' or 'aggressive'.")

    # Perform conversion using existing implementation
    try:
        stats = _convert_single(str(input_path), str(output_path), mode)

        # Add file paths to result
        stats['input_file'] = str(input_path)
        stats['output_file'] = str(output_path)

        return stats
    except Exception as e:
        return {
            "error": str(e),
            "input_file": str(input_path),
            "output_file": str(output_path) if output_path else None
        }


def convert_directory(
    input_dir: str,
    output_dir: str,
    mode: str = "aggressive",
    pattern: str = "*.jsonl"
) -> Dict[str, Any]:
    """
    Convert all JSONL files in a directory to TOON format.

    Args:
        input_dir: Directory containing JSONL files
        output_dir: Directory for output TOON files
        mode: Conversion mode - "conservative" or "aggressive" (default: "aggressive")
        pattern: Glob pattern for finding JSONL files (default: "*.jsonl")

    Returns:
        Dict with batch conversion stats:
            - files_processed: Number of files successfully converted
            - files_failed: Number of files that failed
            - total_original_size: Sum of all input file sizes
            - total_converted_size: Sum of all output file sizes
            - avg_compression_ratio: Average compression percentage
            - file_results: List of per-file results

    Raises:
        FileNotFoundError: If input directory doesn't exist
        ValueError: If mode is invalid
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    input_files = list(input_dir.glob(pattern))

    if not input_files:
        return {
            "files_processed": 0,
            "files_failed": 0,
            "total_original_size": 0,
            "total_converted_size": 0,
            "avg_compression_ratio": 0.0,
            "file_results": [],
            "message": f"No files matching pattern '{pattern}' found in {input_dir}"
        }

    # Process each file
    file_results: List[Dict[str, Any]] = []
    files_processed = 0
    files_failed = 0
    total_original_size = 0
    total_converted_size = 0
    compression_ratios: List[float] = []

    for input_file in input_files:
        # Generate output filename
        output_file = output_dir / input_file.with_suffix('.toon').name

        try:
            result = convert_file(str(input_file), str(output_file), mode)

            if "error" in result:
                files_failed += 1
                result["status"] = "failed"
            else:
                files_processed += 1
                result["status"] = "success"
                total_original_size += result['original_size']
                total_converted_size += result['converted_size']
                compression_ratios.append(result['compression_ratio'])

            file_results.append(result)

        except Exception as e:
            files_failed += 1
            file_results.append({
                "status": "failed",
                "error": str(e),
                "input_file": str(input_file),
                "output_file": str(output_file)
            })

    # Calculate averages
    avg_compression_ratio = (
        sum(compression_ratios) / len(compression_ratios)
        if compression_ratios else 0.0
    )

    return {
        "files_processed": files_processed,
        "files_failed": files_failed,
        "total_original_size": total_original_size,
        "total_converted_size": total_converted_size,
        "avg_compression_ratio": avg_compression_ratio,
        "file_results": file_results,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mode": mode
    }


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
