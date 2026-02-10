#!/usr/bin/env python3
"""
Optional MCP server for TOON converter extension.

Exposes the JSONL-to-TOON converter as an MCP tool for agents.
This is a standalone server - extensions use plain MCP, not TOON encoding.

Usage:
    python extensions/toon-converter/server.py
"""
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: mcp package not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

try:
    # When run as a module or from the extensions directory
    from .converter import convert_file, format_size
except ImportError:
    # When run as a standalone script
    from converter import convert_file, format_size


# Initialize FastMCP server
mcp = FastMCP("toon-converter")


@mcp.tool()
def convert_jsonl_to_toon(
    input_path: str,
    output_path: Optional[str] = None,
    mode: str = "aggressive"
) -> dict:
    """
    Convert a JSONL file to TOON (Token-Optimized Notation) format.

    TOON is a compact text format optimized for token efficiency, achieving
    95%+ compression while preserving conversation semantics.

    Args:
        input_path: Path to source JSONL file
        output_path: Path for output TOON file (if not provided, uses same name with .toon extension)
        mode: Conversion mode:
            - "aggressive": Maximum compression (default) - single-char roles, no metadata
            - "conservative": More readable - keep timestamps, summarize thinking

    Returns:
        Dict with conversion stats: original_size, converted_size, compression_ratio,
        estimated_tokens, records_processed, records_skipped, errors, output_file
    """
    try:
        result = convert_file(input_path, output_path, mode)

        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "input_file": result.get("input_file"),
                "output_file": result.get("output_file")
            }

        # Format sizes for readability
        result["original_size_formatted"] = format_size(result["original_size"])
        result["converted_size_formatted"] = format_size(result["converted_size"])
        result["success"] = True

        return result

    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"File not found: {e}"
        }
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid parameter: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Conversion failed: {e}"
        }


def main():
    """Run the MCP server."""
    # FastMCP handles stdio communication automatically
    mcp.run()


if __name__ == "__main__":
    main()
