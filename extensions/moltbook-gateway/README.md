# Moltbook Gateway Extension

Standalone MCP server extension that exposes Moltbook submission tools.

## Overview

This extension provides two MCP tools for submitting content to Moltbook:

- **moltbook_submit** - Submit posts/comments through the gateway with 6 mechanical interlocks
- **moltbook_qc** - Create QC approval tokens for content review

## What It Does

The Moltbook Gateway enforces 6 mechanical interlocks before any submission:

1. **Session guard** - Prevents retry loops within a session
2. **Content hash dedup** - Blocks exact duplicate content
3. **Word count gate** - Posts max 1000 words, comments max 300
4. **Similarity check** - Catches near-duplicate rewording
5. **Rate limit** - 30min between posts, 20sec between comments
6. **QC gate** - Requires valid, unexpired QC approval token

## Usage

This extension is a thin wrapper around the `moltbook_tools` package. It exposes the existing business logic as MCP tools without TOON encoding (extensions keep it simple).

### Running the Server

```bash
# Via stdio (standard MCP mode)
python extensions/moltbook-gateway/server.py

# Or install as a package entry point
pip install -e .
moltbook-gateway-mcp
```

### Configuration

The extension uses the same configuration as the core `moltbook_tools` package:
- `~/.moltbook/config.json` - Gateway configuration
- `~/.config/moltbook/credentials.json` - API credentials
- `MOLTBOOK_API_KEY` environment variable

## Design Pattern

This extension serves as an example of how to extract tools from the core Memory Palace MCP server into standalone extensions:

1. Create `extensions/<extension-name>/` directory
2. Add `server.py` with FastMCP tool definitions
3. Import business logic from core packages (thin wrapper pattern)
4. Add `pyproject.toml` with dependencies and entry points
5. Keep it simple - no TOON encoding in extensions

## Dependencies

- `mcp>=1.0` - MCP protocol library
- `memory-palace` - Core package (provides `moltbook_tools`)
