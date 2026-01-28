#!/usr/bin/env bash
# ============================================================================
# Claude Memory Palace — Universal Installer
# Supports: Linux, macOS, WSL (with optional Windows-side configuration)
#
# Usage: curl -fsSL https://raw.githubusercontent.com/yourusername/memory-palace/main/install.sh | bash
#   or:  ./install.sh
# ============================================================================

set -euo pipefail

# --- Colors & Formatting ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# --- Constants ---
REPO_URL="https://github.com/clawdbot/memory-palace.git"
BRANCH="main"
INSTALL_DIR="$HOME/memory-palace"
EMBEDDING_MODEL="nomic-embed-text"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

# --- State ---
OS_TYPE=""          # linux, macos, windows
IS_WSL=false
WSL_VERSION=0
WIN_USER_DIR=""     # /mnt/c/Users/<name>
ARCH=""
DISTRO=""
IS_STEAM_DECK=false
PYTHON_CMD=""
HAS_OLLAMA=false
HAS_GPU=false
GPU_VENDOR=""       # nvidia, apple, amd
GPU_NAME=""
VRAM_GB=0
SELECTED_CLIENTS=()
LLM_MODEL=""

# ============================================================================
# Utilities
# ============================================================================

info()    { echo -e "${BLUE}ℹ${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC} $*"; }
error()   { echo -e "${RED}✗${NC} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}$*${NC}\n"; }

prompt_yn() {
    local question="$1"
    local default="${2:-y}"
    local yn_hint
    if [[ "$default" == "y" ]]; then
        yn_hint="[Y/n]"
    else
        yn_hint="[y/N]"
    fi
    
    echo -en "${BOLD}$question${NC} $yn_hint "
    read -r answer
    answer="${answer:-$default}"
    [[ "$answer" =~ ^[Yy] ]]
}

prompt_choice() {
    # Display a numbered menu from an array, return the selected index
    local prompt="$1"
    shift
    local options=("$@")
    
    echo -e "\n${BOLD}$prompt${NC}"
    for i in "${!options[@]}"; do
        echo -e "  ${CYAN}$((i+1))${NC}) ${options[$i]}"
    done
    echo -en "\nChoice: "
    read -r choice
    echo "$((choice - 1))"
}

# ============================================================================
# Detection
# ============================================================================

detect_platform() {
    header "🔍 Detecting your system..."
    
    ARCH="$(uname -m)"
    
    case "$(uname -s)" in
        Darwin)
            OS_TYPE="macos"
            info "Platform: macOS ($ARCH)"
            ;;
        Linux)
            OS_TYPE="linux"
            
            # Check for WSL
            if grep -qi "microsoft\|wsl" /proc/version 2>/dev/null; then
                IS_WSL=true
                if grep -qi "wsl2" /proc/version 2>/dev/null; then
                    WSL_VERSION=2
                else
                    WSL_VERSION=1
                fi
                info "Platform: WSL${WSL_VERSION} (Linux on Windows)"
                
                # Find Windows user directory
                if [[ -d "/mnt/c/Users" ]]; then
                    for dir in /mnt/c/Users/*/; do
                        local name
                        name="$(basename "$dir")"
                        case "$name" in
                            Public|Default|"Default User"|"All Users") continue ;;
                        esac
                        if [[ -d "${dir}AppData" ]]; then
                            WIN_USER_DIR="$dir"
                            info "Windows user: $name"
                            break
                        fi
                    done
                fi
            else
                info "Platform: Linux ($ARCH)"
            fi
            
            # Detect distro
            if [[ -f /etc/os-release ]]; then
                # shellcheck disable=SC1091
                source /etc/os-release
                DISTRO="${ID:-unknown}"
                if [[ "$DISTRO" == "steamos" ]]; then
                    IS_STEAM_DECK=true
                    info "Detected: Steam Deck 🎮"
                else
                    info "Distro: ${PRETTY_NAME:-$DISTRO}"
                fi
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            OS_TYPE="windows"
            info "Platform: Windows (Git Bash / MSYS2)"
            warn "For best results on Windows, use install.ps1 or the GUI installer"
            ;;
        *)
            error "Unknown platform: $(uname -s)"
            exit 1
            ;;
    esac
}

detect_python() {
    info "Checking Python..."
    
    for cmd in python3 python py; do
        if command -v "$cmd" &>/dev/null; then
            local version
            version="$($cmd --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)"
            if [[ -n "$version" ]]; then
                local major minor
                major="$(echo "$version" | cut -d. -f1)"
                minor="$(echo "$version" | cut -d. -f2)"
                if (( major >= MIN_PYTHON_MAJOR && minor >= MIN_PYTHON_MINOR )); then
                    PYTHON_CMD="$cmd"
                    success "Python $version ($cmd)"
                    return
                fi
            fi
        fi
    done
    
    warn "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ not found"
}

detect_ollama() {
    info "Checking Ollama..."
    
    if command -v ollama &>/dev/null; then
        local version
        version="$(ollama --version 2>&1 | grep -oP '\d+\.\d+\.?\d*' | head -1)"
        HAS_OLLAMA=true
        success "Ollama ${version:-installed}"
        
        # Check if running
        if ollama list &>/dev/null 2>&1; then
            local models
            models="$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')"
            if [[ -n "$models" ]]; then
                info "Installed models: $(echo "$models" | tr '\n' ', ' | sed 's/,$//')"
            fi
        else
            warn "Ollama installed but not running — will need to start it"
        fi
    else
        warn "Ollama not installed"
    fi
}

detect_gpu() {
    info "Checking GPU..."
    
    # NVIDIA
    if command -v nvidia-smi &>/dev/null; then
        local gpu_info
        gpu_info="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null)"
        if [[ -n "$gpu_info" ]]; then
            HAS_GPU=true
            GPU_VENDOR="nvidia"
            GPU_NAME="$(echo "$gpu_info" | cut -d, -f1 | xargs)"
            local vram_mb
            vram_mb="$(echo "$gpu_info" | cut -d, -f2 | xargs)"
            VRAM_GB=$((vram_mb / 1024))
            success "$GPU_NAME (${VRAM_GB}GB VRAM)"
            return
        fi
    fi
    
    # Apple Silicon
    if [[ "$OS_TYPE" == "macos" ]]; then
        local cpu_brand
        cpu_brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
        if echo "$cpu_brand" | grep -qi "apple"; then
            HAS_GPU=true
            GPU_VENDOR="apple"
            GPU_NAME="$cpu_brand"
            # Approximate unified memory available for GPU
            local mem_bytes
            mem_bytes="$(sysctl -n hw.memsize 2>/dev/null)"
            if [[ -n "$mem_bytes" ]]; then
                VRAM_GB=$(( mem_bytes / 1073741824 * 3 / 4 ))  # 75% of total
            else
                VRAM_GB=8
            fi
            success "$GPU_NAME (~${VRAM_GB}GB unified memory)"
            return
        fi
    fi
    
    # AMD ROCm
    if command -v rocm-smi &>/dev/null; then
        HAS_GPU=true
        GPU_VENDOR="amd"
        GPU_NAME="AMD GPU (ROCm)"
        info "AMD GPU detected (ROCm) — VRAM detection limited"
        VRAM_GB=8  # Conservative default
        return
    fi
    
    info "No dedicated GPU detected — CPU mode is fine for embeddings"
}

select_llm_model() {
    # Determine LLM model based on VRAM
    if (( VRAM_GB >= 16 )); then
        LLM_MODEL="qwen3:14b"
    elif (( VRAM_GB >= 10 )); then
        LLM_MODEL="qwen3:8b"
    elif (( VRAM_GB >= 6 )); then
        LLM_MODEL="qwen3:4b"
    elif (( VRAM_GB >= 2 )); then
        LLM_MODEL="qwen3:1.7b"
    else
        LLM_MODEL=""  # Skip — cloud AI handles it
    fi
}

# ============================================================================
# AI Client Discovery
# ============================================================================

# Associative arrays for client config paths
declare -A CLIENT_NAMES=(
    [claude-desktop]="Claude Desktop"
    [claude-code]="Claude Code"
    [cursor]="Cursor"
    [windsurf]="Windsurf"
    [continue]="VS Code + Continue"
    [claude-desktop-win]="Claude Desktop (Windows)"
    [cursor-win]="Cursor (Windows)"
    [windsurf-win]="Windsurf (Windows)"
)

get_client_config_path() {
    local client_id="$1"
    
    case "$client_id" in
        claude-desktop)
            case "$OS_TYPE" in
                macos)  echo "$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
                linux)  echo "$HOME/.config/Claude/claude_desktop_config.json" ;;
                *)      echo "" ;;
            esac
            ;;
        claude-code)
            echo "$HOME/.claude.json"
            ;;
        cursor)
            echo "$HOME/.cursor/mcp.json"
            ;;
        windsurf)
            echo "$HOME/.codeium/windsurf/mcp_config.json"
            ;;
        continue)
            echo "$HOME/.continue/config.json"
            ;;
        # Windows-side paths (from WSL)
        claude-desktop-win)
            [[ -n "$WIN_USER_DIR" ]] && echo "${WIN_USER_DIR}AppData/Roaming/Claude/claude_desktop_config.json" || echo ""
            ;;
        cursor-win)
            [[ -n "$WIN_USER_DIR" ]] && echo "${WIN_USER_DIR}.cursor/mcp.json" || echo ""
            ;;
        windsurf-win)
            [[ -n "$WIN_USER_DIR" ]] && echo "${WIN_USER_DIR}.codeium/windsurf/mcp_config.json" || echo ""
            ;;
    esac
}

is_client_installed() {
    local client_id="$1"
    local config_path
    config_path="$(get_client_config_path "$client_id")"
    
    # Check if config file or its parent directory exists
    if [[ -n "$config_path" ]]; then
        if [[ -f "$config_path" ]] || [[ -d "$(dirname "$config_path")" ]]; then
            return 0
        fi
    fi
    
    # Check for CLI tools
    case "$client_id" in
        claude-code)
            command -v claude &>/dev/null && return 0
            ;;
        cursor)
            # Check for Cursor app
            if [[ "$OS_TYPE" == "macos" ]] && [[ -d "/Applications/Cursor.app" ]]; then
                return 0
            fi
            command -v cursor &>/dev/null && return 0
            ;;
        windsurf)
            if [[ "$OS_TYPE" == "macos" ]] && [[ -d "/Applications/Windsurf.app" ]]; then
                return 0
            fi
            command -v windsurf &>/dev/null && return 0
            ;;
    esac
    
    return 1
}

is_already_configured() {
    local config_path="$1"
    [[ -f "$config_path" ]] && grep -q '"memory-palace"' "$config_path" 2>/dev/null
}

discover_clients() {
    header "🤖 Which AI tools do you use?"
    
    local detected=()
    local all_clients=(claude-desktop claude-code cursor windsurf continue)
    
    # Add Windows-side clients if WSL
    if $IS_WSL && [[ -n "$WIN_USER_DIR" ]]; then
        all_clients+=(claude-desktop-win cursor-win windsurf-win)
    fi
    
    # Detect installed clients
    for client_id in "${all_clients[@]}"; do
        if is_client_installed "$client_id"; then
            local config_path
            config_path="$(get_client_config_path "$client_id")"
            local status=""
            if [[ -n "$config_path" ]] && is_already_configured "$config_path"; then
                status=" ${DIM}(already configured)${NC}"
            fi
            detected+=("$client_id")
            success "Found: ${CLIENT_NAMES[$client_id]}${status}"
        fi
    done
    
    if [[ ${#detected[@]} -eq 0 ]]; then
        warn "No AI clients detected automatically."
        echo ""
        echo "Memory Palace works with any MCP-compatible client."
        echo "You can configure your client manually after installation."
        echo ""
        return
    fi
    
    # WSL cross-platform notice
    if $IS_WSL; then
        local has_win_clients=false
        for client_id in "${detected[@]}"; do
            if [[ "$client_id" == *"-win" ]]; then
                has_win_clients=true
                break
            fi
        done
        if $has_win_clients; then
            echo ""
            info "Noticed you're in WSL. Would you also like any Windows desktop services configured?"
            info "Windows-side clients are listed above with '(Windows)' suffix."
        fi
    fi
    
    # Ask which to configure
    echo ""
    echo -e "${BOLD}Select clients to configure:${NC}"
    echo -e "${DIM}(Enter numbers separated by spaces, or 'a' for all, 'n' for none)${NC}"
    echo ""
    
    for i in "${!detected[@]}"; do
        local client_id="${detected[$i]}"
        local config_path
        config_path="$(get_client_config_path "$client_id")"
        local status=""
        if [[ -n "$config_path" ]] && is_already_configured "$config_path"; then
            status=" ${DIM}[configured]${NC}"
        fi
        echo -e "  ${CYAN}$((i+1))${NC}) ${CLIENT_NAMES[$client_id]}${status}"
    done
    
    echo ""
    echo -en "Selection: "
    read -r selection
    
    case "$selection" in
        [aA]|[aA]ll)
            SELECTED_CLIENTS=("${detected[@]}")
            ;;
        [nN]|[nN]one|"")
            SELECTED_CLIENTS=()
            ;;
        *)
            for num in $selection; do
                local idx=$((num - 1))
                if (( idx >= 0 && idx < ${#detected[@]} )); then
                    SELECTED_CLIENTS+=("${detected[$idx]}")
                fi
            done
            ;;
    esac
    
    if [[ ${#SELECTED_CLIENTS[@]} -gt 0 ]]; then
        echo ""
        info "Will configure: $(printf '%s, ' "${SELECTED_CLIENTS[@]}" | sed 's/, $//')"
    else
        info "No clients selected — you can configure manually later"
    fi
}

# ============================================================================
# Installation
# ============================================================================

install_python() {
    header "📦 Installing Python..."
    
    case "$OS_TYPE" in
        macos)
            if command -v brew &>/dev/null; then
                info "Installing via Homebrew..."
                brew install python@3.12
            else
                error "Please install Python 3.10+ from https://www.python.org/downloads/"
                error "Or install Homebrew first: https://brew.sh"
                exit 1
            fi
            ;;
        linux)
            if command -v apt-get &>/dev/null; then
                info "Installing via apt..."
                sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
            elif command -v dnf &>/dev/null; then
                info "Installing via dnf..."
                sudo dnf install -y python3 python3-pip
            elif command -v pacman &>/dev/null; then
                info "Installing via pacman..."
                sudo pacman -S --noconfirm python python-pip
            else
                error "Please install Python 3.10+ using your package manager"
                exit 1
            fi
            ;;
    esac
    
    # Re-detect
    detect_python
    if [[ -z "$PYTHON_CMD" ]]; then
        error "Python installation failed. Please install manually."
        exit 1
    fi
}

install_ollama() {
    header "📦 Installing Ollama..."
    
    case "$OS_TYPE" in
        macos)
            if command -v brew &>/dev/null; then
                info "Installing via Homebrew..."
                brew install ollama
            else
                info "Opening download page..."
                open "https://ollama.com/download"
                echo "Please install Ollama and re-run this script."
                exit 1
            fi
            ;;
        linux)
            info "Running official installer..."
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
    esac
    
    HAS_OLLAMA=true
    success "Ollama installed"
    
    # Start Ollama in the background if it's not running
    if ! ollama list &>/dev/null 2>&1; then
        info "Starting Ollama service..."
        if command -v systemctl &>/dev/null; then
            sudo systemctl start ollama 2>/dev/null || ollama serve &>/dev/null &
        else
            ollama serve &>/dev/null &
        fi
        sleep 2
    fi
}

download_repo() {
    header "📥 Downloading Memory Palace..."
    
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        info "Existing installation found — updating..."
        cd "$INSTALL_DIR"
        git pull origin "$BRANCH"
    else
        if command -v git &>/dev/null; then
            git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
        else
            # Fallback: download tarball
            info "git not found — downloading archive..."
            mkdir -p "$INSTALL_DIR"
            curl -fsSL "https://github.com/clawdbot/memory-palace/archive/refs/heads/${BRANCH}.tar.gz" \
                | tar -xz --strip-components=1 -C "$INSTALL_DIR"
        fi
    fi
    
    success "Memory Palace downloaded to $INSTALL_DIR"
}

setup_venv() {
    header "🐍 Setting up Python environment..."
    
    local venv_dir="$INSTALL_DIR/venv"
    
    info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$venv_dir"
    
    local pip_cmd="$venv_dir/bin/pip"
    
    info "Installing Memory Palace package..."
    "$pip_cmd" install -e "$INSTALL_DIR" --quiet
    
    success "Package installed"
    
    # Verify
    local python_cmd="$venv_dir/bin/python"
    if "$python_cmd" -c "import memory_palace; import mcp_server; print('OK')" 2>/dev/null | grep -q "OK"; then
        success "Installation verified"
    else
        warn "Package import check failed — installation may have issues"
    fi
}

pull_models() {
    header "🧠 Downloading AI models..."
    
    # Embedding model (always)
    if ollama list 2>/dev/null | grep -qi "nomic-embed-text"; then
        success "Embedding model already installed"
    else
        info "Downloading embedding model ($EMBEDDING_MODEL)..."
        info "${DIM}This is ~270MB — should be quick${NC}"
        ollama pull "$EMBEDDING_MODEL"
        success "Embedding model ready"
    fi
    
    # Base LLM model (required — handles synthesis, classification, extraction)
    # qwen3:1.7b is always installed as the baseline
    local BASE_LLM="qwen3:1.7b"
    echo ""
    if ollama list 2>/dev/null | grep -qi "qwen3:1.7b"; then
        success "Base LLM model (qwen3:1.7b) already installed"
    else
        info "Downloading base LLM model (qwen3:1.7b)..."
        info "${DIM}~1GB — handles synthesis, classification, and extraction${NC}"
        ollama pull "$BASE_LLM"
        success "Base LLM model ready"
    fi
    
    # Upgraded LLM model (optional, for better quality on capable hardware)
    select_llm_model
    
    if [[ -n "$LLM_MODEL" ]] && [[ "$LLM_MODEL" != "$BASE_LLM" ]]; then
        echo ""
        if ollama list 2>/dev/null | grep -qi "${LLM_MODEL%%:*}"; then
            success "Upgraded LLM model already installed"
        else
            echo -e "Recommended upgrade for your hardware: ${BOLD}$LLM_MODEL${NC}"
            echo -e "${DIM}Better quality for memory extraction and synthesis${NC}"
            echo ""
            if prompt_yn "Download $LLM_MODEL?" "y"; then
                info "Downloading $LLM_MODEL (this may take a while)..."
                ollama pull "$LLM_MODEL"
                success "Upgraded LLM model ready"
            else
                info "Skipped — you can upgrade later with: ollama pull $LLM_MODEL"
            fi
        fi
    fi
}

# ============================================================================
# Client Configuration
# ============================================================================

build_mcp_entry() {
    local client_id="$1"
    local python_path="$INSTALL_DIR/venv/bin/python"
    local cwd="$INSTALL_DIR"
    
    # For Windows-side clients from WSL, translate paths
    if [[ "$client_id" == *"-win" ]]; then
        local install_str="$INSTALL_DIR"
        if [[ "$install_str" == /mnt/* ]]; then
            # /mnt/c/Users/X/memory-palace → C:\Users\X\memory-palace
            local drive="${install_str:5:1}"
            local rest="${install_str:7}"
            local win_path="${drive^^}:\\${rest//\//\\}"
            python_path="${win_path}\\venv\\Scripts\\python.exe"
            cwd="$win_path"
        else
            # WSL filesystem — use wsl.exe wrapper
            local distro="${WSL_DISTRO_NAME:-Ubuntu}"
            cat <<EOF
{
    "command": "wsl.exe",
    "args": ["-d", "$distro", "--", "$python_path", "-m", "mcp_server.server"],
    "cwd": "$cwd"
}
EOF
            return
        fi
    fi
    
    cat <<EOF
{
    "command": "$python_path",
    "args": ["-m", "mcp_server.server"],
    "cwd": "$cwd"
}
EOF
}

configure_client() {
    local client_id="$1"
    local config_path
    config_path="$(get_client_config_path "$client_id")"
    
    if [[ -z "$config_path" ]]; then
        warn "No config path for ${CLIENT_NAMES[$client_id]}"
        return 1
    fi
    
    # Special handling for Continue (different format)
    if [[ "$client_id" == "continue" ]]; then
        configure_continue "$config_path"
        return $?
    fi
    
    # Standard MCP config format
    local mcp_entry
    mcp_entry="$(build_mcp_entry "$client_id")"
    
    # Ensure directory exists
    mkdir -p "$(dirname "$config_path")"
    
    # Backup existing
    if [[ -f "$config_path" ]]; then
        local backup="${config_path}.backup"
        local counter=1
        while [[ -f "$backup" ]]; do
            backup="${config_path}.backup.${counter}"
            ((counter++))
        done
        cp "$config_path" "$backup"
        info "Backed up to: $(basename "$backup")"
    fi
    
    # Merge config using Python (jq might not be available everywhere)
    local merge_script
    merge_script=$(cat <<'PYEOF'
import json, sys

config_path = sys.argv[1]
mcp_entry = json.loads(sys.argv[2])

# Load existing config
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    config = {}

# Merge
if 'mcpServers' not in config:
    config['mcpServers'] = {}
config['mcpServers']['memory-palace'] = mcp_entry

# Write
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('OK')
PYEOF
)
    
    local result
    result="$($PYTHON_CMD -c "$merge_script" "$config_path" "$mcp_entry" 2>&1)"
    
    if [[ "$result" == "OK" ]]; then
        success "Configured ${CLIENT_NAMES[$client_id]}"
        return 0
    else
        error "Failed to configure ${CLIENT_NAMES[$client_id]}: $result"
        return 1
    fi
}

configure_continue() {
    local config_path="$1"
    
    mkdir -p "$(dirname "$config_path")"
    
    # Backup
    if [[ -f "$config_path" ]]; then
        cp "$config_path" "${config_path}.backup"
    fi
    
    local python_path="$INSTALL_DIR/venv/bin/python"
    
    local merge_script
    merge_script=$(cat <<'PYEOF'
import json, sys

config_path = sys.argv[1]
python_path = sys.argv[2]
install_dir = sys.argv[3]

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    config = {}

if 'experimental' not in config:
    config['experimental'] = {}
if 'modelContextProtocolServers' not in config['experimental']:
    config['experimental']['modelContextProtocolServers'] = []

servers = config['experimental']['modelContextProtocolServers']
servers = [s for s in servers if s.get('name') != 'memory-palace']
servers.append({
    'name': 'memory-palace',
    'command': python_path,
    'args': ['-m', 'mcp_server.server'],
    'cwd': install_dir
})
config['experimental']['modelContextProtocolServers'] = servers

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('OK')
PYEOF
)
    
    local result
    result="$($PYTHON_CMD -c "$merge_script" "$config_path" "$python_path" "$INSTALL_DIR" 2>&1)"
    
    if [[ "$result" == "OK" ]]; then
        success "Configured VS Code + Continue"
        return 0
    else
        error "Failed to configure Continue: $result"
        return 1
    fi
}

configure_selected_clients() {
    if [[ ${#SELECTED_CLIENTS[@]} -eq 0 ]]; then
        return
    fi
    
    header "⚙️  Configuring AI clients..."
    
    for client_id in "${SELECTED_CLIENTS[@]}"; do
        configure_client "$client_id"
    done
}

# ============================================================================
# Summary
# ============================================================================

show_summary() {
    header "🎉 Memory Palace is ready!"
    
    echo -e "${GREEN}Installation:${NC} $INSTALL_DIR"
    echo -e "${GREEN}Embedding:${NC}    $EMBEDDING_MODEL"
    if [[ -n "$LLM_MODEL" ]]; then
        echo -e "${GREEN}LLM:${NC}          $LLM_MODEL"
    fi
    
    if [[ ${#SELECTED_CLIENTS[@]} -gt 0 ]]; then
        echo ""
        echo -e "${GREEN}Configured clients:${NC}"
        for client_id in "${SELECTED_CLIENTS[@]}"; do
            echo -e "  ✓ ${CLIENT_NAMES[$client_id]}"
        done
    fi
    
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    
    if [[ ${#SELECTED_CLIENTS[@]} -gt 0 ]]; then
        echo "  1. Restart your AI client(s) to activate Memory Palace"
        echo "  2. Test it out:"
        echo -e "     ${DIM}Tell Claude: \"Remember that I like coffee\"${NC}"
        echo -e "     ${DIM}In a new chat: \"What do I like to drink?\"${NC}"
    else
        echo "  1. Configure your MCP client to use Memory Palace"
        echo "     Add this to your MCP config:"
        echo ""
        echo -e "     ${DIM}\"memory-palace\": {${NC}"
        echo -e "     ${DIM}  \"command\": \"$INSTALL_DIR/venv/bin/python\",${NC}"
        echo -e "     ${DIM}  \"args\": [\"-m\", \"mcp_server.server\"],${NC}"
        echo -e "     ${DIM}  \"cwd\": \"$INSTALL_DIR\"${NC}"
        echo -e "     ${DIM}}${NC}"
    fi
    
    echo ""
    echo -e "${DIM}To update later: cd $INSTALL_DIR && git pull && venv/bin/pip install -e .${NC}"
    echo ""
}

# ============================================================================
# Main Flow
# ============================================================================

main() {
    echo ""
    echo -e "${BOLD}${CYAN}╔════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║     Claude Memory Palace Installer     ║${NC}"
    echo -e "${BOLD}${CYAN}╚════════════════════════════════════════╝${NC}"
    echo ""
    
    # --- Detection ---
    detect_platform
    detect_python
    detect_ollama
    detect_gpu
    
    # --- Prerequisites ---
    if [[ -z "$PYTHON_CMD" ]]; then
        if prompt_yn "Python 3.10+ is required. Install it now?" "y"; then
            install_python
        else
            error "Python 3.10+ is required. Please install it and re-run."
            exit 1
        fi
    fi
    
    if ! $HAS_OLLAMA; then
        if prompt_yn "Ollama is required for embeddings. Install it now?" "y"; then
            install_ollama
        else
            error "Ollama is required. Install from https://ollama.com and re-run."
            exit 1
        fi
    fi
    
    # --- Client Discovery ---
    discover_clients
    
    # --- Installation ---
    download_repo
    setup_venv
    pull_models
    
    # --- Client Configuration ---
    configure_selected_clients
    
    # --- Done! ---
    show_summary
}

# Run it
main "$@"
