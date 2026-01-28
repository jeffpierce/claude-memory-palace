# Installer Architecture

## Goal

Zero-friction installation of Claude Memory Palace for non-technical users.
The only prerequisite knowledge: "what AI tools do you already use?"

## Database

**Default: SQLite** — zero-install, single file (`~/.memory-palace/memories.db`).
Full knowledge graph, embeddings, edges, traversal — all work on SQLite.

**Upgrade path: PostgreSQL + pgvector** — for enterprise/scale deployments.
Native vector indexes, ARRAY types, JSONB. Not required for individual users.

## Installer Targets

| Target | Format | UI | Primary Audience |
|--------|--------|-----|-----------------|
| **Windows GUI** | `.exe` (PyInstaller) | tkinter | General Windows users |
| **PowerShell** | `install.ps1` | Terminal prompts | Windows power users, sysadmins |
| **macOS GUI** | `.app` (py2app) | tkinter | Mac users |
| **Linux GUI** | AppImage or similar | tkinter | Steam Deck, Desktop Linux |
| **install.sh** | Shell script | Terminal prompts | Linux/Mac/WSL, developers |

## Shared Core

All installers share the same logic, implemented in `installer/shared/`:

### Detection
- **Platform**: Linux, macOS, Windows, WSL (with cross-platform offer)
- **Python**: Version check, auto-install offer
- **Ollama**: Installed? Running? Models present?
- **GPU**: NVIDIA (nvidia-smi), Apple Silicon (sysctl), AMD (rocm-smi)
- **AI Clients**: Which MCP-compatible tools are installed?

### AI Client Discovery & Configuration

All clients use the same `{"mcpServers": {...}}` format (except Continue):

| Client | Config Path (macOS) | Config Path (Windows) | Config Path (Linux) |
|--------|--------------------|-----------------------|--------------------|
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` | `%APPDATA%/Claude/claude_desktop_config.json` | `~/.config/Claude/claude_desktop_config.json` |
| Claude Code | `~/.claude.json` | `~/.claude.json` | `~/.claude.json` |
| Cursor | `~/.cursor/mcp.json` | `~/.cursor/mcp.json` | `~/.cursor/mcp.json` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` | `~/.codeium/windsurf/mcp_config.json` | `~/.codeium/windsurf/mcp_config.json` |
| VS Code + Continue | `~/.continue/config.json` | `~/.continue/config.json` | `~/.continue/config.json` |

**Detection heuristic**: If the config directory or parent app exists, the client is installed.
We only offer to configure clients we detect (but allow manual selection too).

### Installation Steps
1. Check/install Python 3.10+
2. Check/install Ollama
3. Create venv, pip install package
4. Pull embedding model (nomic-embed-text — ~270MB, runs on anything)
5. Optionally pull LLM model (for local synthesis — based on GPU)
6. Configure selected AI clients (merge into existing configs, backup first)
7. Verify: start MCP server, test embedding, confirm client config

### Model Selection

Embedding model is always `nomic-embed-text` — small, fast, runs on CPU.

LLM model (optional, for local transcript processing) scales with hardware:

| VRAM | LLM Model | Notes |
|------|-----------|-------|
| 16GB+ | qwen3:14b | Best quality |
| 10GB+ | qwen3:8b | Good quality |
| 6GB+ | qwen3:4b | Decent |
| <6GB / CPU | qwen3:1.7b | Basic, but works |
| None desired | Skip | Cloud AI handles synthesis |

## WSL Special Case

When `install.sh` detects WSL:
1. Install normally in WSL (Linux paths)
2. Detect Windows-side AI clients (via `/mnt/c/Users/*/AppData/...`)
3. Offer: "Noticed you're in WSL. Would you also like any Windows desktop services configured?"
4. If yes, write configs to Windows-side paths with WSL-aware command paths

## Config Merge Strategy

**Never overwrite existing configs.** Always:
1. Read existing config
2. Backup to `.json.backup` (with counter for multiple backups)
3. Merge `memory-palace` entry into `mcpServers`
4. Write merged config
5. Report what changed

If `memory-palace` already exists in config, ask before updating.

## File Structure

```
install.sh                       # Universal shell installer (Linux/Mac/WSL)
install.ps1                      # PowerShell installer (Windows)
installer/
├── ARCHITECTURE.md              # This file
├── gui/
│   ├── __init__.py
│   └── app.py                   # Cross-platform tkinter GUI (all 6 screens)
├── shared/
│   ├── __init__.py
│   ├── detect.py                # Platform, GPU, Python, Ollama detection
│   ├── clients.py               # AI client discovery and configuration
│   ├── models.py                # Model selection and download
│   └── install_core.py          # Package installation (venv, pip, clone)
├── windows/
│   ├── run_gui.py               # Thin launcher → gui/app.py
│   └── build.py                 # PyInstaller → MemoryPalaceSetup.exe
├── macos/
│   ├── run_gui.py               # Thin launcher → gui/app.py
│   └── build.py                 # py2app → Memory Palace Setup.app + .dmg
└── linux/
    ├── run_gui.py               # Thin launcher → gui/app.py (Steam Deck aware)
    └── build.py                 # AppImage → MemoryPalaceSetup-x86_64.AppImage
```

## Building

```bash
# Windows (run on Windows or cross-compile)
cd installer/windows && python build.py

# macOS (must run on macOS)
cd installer/macos && python build.py

# Linux AppImage (must run on Linux, WSL works)
cd installer/linux && python build.py
```

## Build Requirements

| Platform | Tool | Install |
|----------|------|---------|
| Windows | PyInstaller | `pip install pyinstaller` |
| macOS | py2app | `pip install py2app` |
| Linux | appimagetool | Auto-downloaded by build.py |
