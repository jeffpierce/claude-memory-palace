# TOON Converter Extension

Standalone CLI tool and optional MCP extension for converting JSONL files to TOON (Token-Optimized Notation) format.

## Overview

TOON (Token-Optimized Notation) is a compact text format optimized for token efficiency, achieving 95%+ compression while preserving conversation semantics. This extension extracts the JSONL-to-TOON conversion functionality from the core Memory Palace MCP server into a standalone tool.

## CLI Usage

The converter works as a standalone Python tool without requiring MCP:

```bash
# Convert a single JSONL file to TOON
python -m extensions.toon-converter input.jsonl output.toon

# Specify output directory (auto-generates filename)
python -m extensions.toon-converter input.jsonl --output-dir ./converted/

# Batch convert a directory
python -m extensions.toon-converter input_dir/ --output-dir output_dir/

# Choose compression mode
python -m extensions.toon-converter input.jsonl output.toon --mode conservative
python -m extensions.toon-converter input.jsonl output.toon --mode aggressive
```

## Conversion Modes

- **conservative**: Keep timestamps (HH:MM:SS), summarize thinking blocks, indicate tool calls with names. More readable.
- **aggressive**: Maximum compression. Role indicators only (U:/A:/S:), strip all metadata, timestamps, and thinking content. (default)

## MCP Server (Optional)

The extension can also run as an MCP server for agents that want to convert files programmatically:

```bash
# Via stdio (standard MCP mode)
python extensions/toon-converter/server.py
```

### MCP Tool

- **convert_jsonl_to_toon** - Convert a JSONL file to TOON format
  - `input_path` (required): Path to source JSONL file
  - `output_path` (optional): Path for output file (defaults to same name with .toon extension)
  - `mode` (optional): "conservative" or "aggressive" (default: "aggressive")

Returns stats: lines processed, original size, converted size, compression ratio.

## Design

This extension follows the Memory Palace v2.0 extension pattern:

1. Core conversion logic in `converter.py`
2. CLI entry point in `cli.py`
3. Optional MCP server wrapper in `server.py`
4. Minimal dependencies - just Python + existing toon_converter from tools/

The converter is a thin wrapper around the existing `tools/toon_converter.py` implementation, exposing it as a reusable module.

## Dependencies

- Python 3.10+
- No additional dependencies for CLI mode
- `mcp>=1.0` for MCP server mode (optional)
