#!/bin/bash
#
# Claude Memory Palace - Linux/macOS Installer
#
# Installs dependencies with explicit user permission for each step.
# Never auto-installs without consent.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

header() { echo -e "\n${CYAN}=== $1 ===${NC}"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warning() { echo -e "${YELLOW}[!!]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
info() { echo -e "[INFO] $1"; }

ask_permission() {
    local question="$1"
    local default="${2:-N}"
    local prompt

    if [ "$default" = "Y" ]; then
        prompt="(Y/n)"
    else
        prompt="(y/N)"
    fi

    read -p "$question $prompt " response

    if [ -z "$response" ]; then
        [ "$default" = "Y" ] && return 0 || return 1
    fi

    [[ "$response" =~ ^[Yy] ]] && return 0 || return 1
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)  OS="linux" ;;
        Darwin*) OS="macos" ;;
        *)       OS="unknown" ;;
    esac
}

echo -e "${MAGENTA}"
cat << 'EOF'

  ____  _                 _        __  __
 / ___|| | __ _ _   _  __| | ___  |  \/  | ___ _ __ ___   ___  _ __ _   _
| |    | |/ _` | | | |/ _` |/ _ \ | |\/| |/ _ \ '_ ` _ \ / _ \| '__| | | |
| |___ | | (_| | |_| | (_| |  __/ | |  | |  __/ | | | | | (_) | |  | |_| |
 \____||_|\__,_|\__,_|\__,_|\___| |_|  |_|\___|_| |_| |_|\___/|_|   \__, |
                                                                    |___/
         P A L A C E   -   I N S T A L L E R

EOF
echo -e "${NC}"

echo "This installer will check dependencies and ask permission before installing anything."
echo ""

detect_os
info "Detected OS: $OS"

# Track what's missing
MISSING=()
GPU_VRAM_MB=0
GPU_NAME=""

# =============================================================================
# CHECK PYTHON
# =============================================================================
header "Checking Python"

PYTHON_CMD=""
PYTHON_VERSION=""

for cmd in python3 python; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" --version 2>&1 | grep -oP 'Python \K[0-9]+\.[0-9]+\.[0-9]+' || echo "")
        if [ -n "$version" ]; then
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                PYTHON_CMD="$cmd"
                PYTHON_VERSION="$version"
                break
            fi
        fi
    fi
done

if [ -n "$PYTHON_CMD" ]; then
    success "Python $PYTHON_VERSION found ($PYTHON_CMD)"
else
    warning "Python 3.10+ not found"
    MISSING+=("Python")
    info "Install Python 3.10+ using your package manager:"
    if [ "$OS" = "macos" ]; then
        info "  brew install python@3.11"
    else
        info "  sudo apt install python3.11  # Debian/Ubuntu"
        info "  sudo dnf install python3.11  # Fedora"
    fi
fi

# =============================================================================
# CHECK PIP
# =============================================================================
header "Checking pip"

PIP_CMD=""

if [ -n "$PYTHON_CMD" ]; then
    for cmd in pip3 pip "$PYTHON_CMD -m pip"; do
        if [ "$cmd" = "$PYTHON_CMD -m pip" ]; then
            if $PYTHON_CMD -m pip --version &> /dev/null; then
                PIP_CMD="$PYTHON_CMD -m pip"
                break
            fi
        elif command -v "$cmd" &> /dev/null; then
            PIP_CMD="$cmd"
            break
        fi
    done
fi

if [ -n "$PIP_CMD" ]; then
    success "pip found ($PIP_CMD)"
else
    warning "pip not found"
    info "Install pip:"
    info "  $PYTHON_CMD -m ensurepip --upgrade"
fi

# =============================================================================
# CHECK OLLAMA
# =============================================================================
header "Checking Ollama"

OLLAMA_FOUND=false

if command -v ollama &> /dev/null; then
    OLLAMA_FOUND=true
    success "Ollama found: $(which ollama)"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        success "Ollama server is running"
    else
        warning "Ollama is installed but not running"
        info "Start with: ollama serve"
    fi
else
    warning "Ollama not found"
    MISSING+=("Ollama")
    info "Ollama is required for embeddings and LLM inference."
    echo ""

    if [ "$OS" = "macos" ]; then
        info "Install Ollama on macOS:"
        info "  brew install ollama"
        info "  OR download from: https://ollama.com/download"
        echo ""
        if ask_permission "Install Ollama via Homebrew?"; then
            if command -v brew &> /dev/null; then
                brew install ollama
                OLLAMA_FOUND=true
                success "Ollama installed via Homebrew"
            else
                error "Homebrew not found. Install from https://brew.sh first."
            fi
        fi
    else
        info "Install Ollama on Linux:"
        info "  curl -fsSL https://ollama.com/install.sh | sh"
        echo ""
        if ask_permission "Install Ollama via official installer script?"; then
            curl -fsSL https://ollama.com/install.sh | sh
            OLLAMA_FOUND=true
            success "Ollama installed"
        fi
    fi
fi

# =============================================================================
# CHECK GPU
# =============================================================================
header "Checking GPU"

if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")

    if [ -n "$gpu_info" ] && [[ ! "$gpu_info" =~ "failed" ]]; then
        # Parse GPU info (format: "GPU Name, 12345 MiB")
        GPU_NAME=$(echo "$gpu_info" | cut -d',' -f1 | xargs)
        GPU_VRAM_MB=$(echo "$gpu_info" | grep -oP '\d+(?=\s*MiB)' | head -1)

        if [ -n "$GPU_VRAM_MB" ]; then
            GPU_VRAM_GB=$(echo "scale=1; $GPU_VRAM_MB / 1024" | bc)
            success "NVIDIA GPU: $GPU_NAME ($GPU_VRAM_GB GB VRAM)"

            # Recommendations based on VRAM
            if [ "$GPU_VRAM_MB" -ge 16000 ]; then
                info "Excellent! You can run the largest recommended models"
            elif [ "$GPU_VRAM_MB" -ge 8000 ]; then
                info "Good. You can run all recommended models"
            else
                warning "Limited VRAM. Consider using smaller models or CPU inference."
            fi
        fi
    fi
else
    # Check for AMD GPU on Linux
    if [ "$OS" = "linux" ] && command -v rocm-smi &> /dev/null; then
        info "AMD GPU detected (ROCm). Ollama supports AMD GPUs."
    # Check for Apple Silicon
    elif [ "$OS" = "macos" ]; then
        chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
        if [[ "$chip" =~ "Apple" ]]; then
            # Get unified memory
            mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
            mem_gb=$((mem_bytes / 1024 / 1024 / 1024))
            success "Apple Silicon detected with ${mem_gb}GB unified memory"
            info "Ollama will use Metal acceleration"
        fi
    else
        warning "No NVIDIA GPU detected (or nvidia-smi not available)"
        info "The memory palace will use CPU inference (slower but functional)"
    fi
fi

# =============================================================================
# INSTALL PYTHON DEPENDENCIES
# =============================================================================
header "Python Dependencies"

if [ -z "$PYTHON_CMD" ]; then
    error "Cannot install Python dependencies without Python installed."
    info "Please install Python 3.10+ and re-run this installer."
elif [ -z "$PIP_CMD" ]; then
    error "Cannot install Python dependencies without pip."
    info "Please install pip and re-run this installer."
else
    PYPROJECT="$SCRIPT_DIR/pyproject.toml"

    if [ -f "$PYPROJECT" ]; then
        info "Found pyproject.toml - will install package in development mode"
        echo ""
        echo -e "${YELLOW}This will install the following:${NC}"
        echo "  - memory-palace package (editable install)"
        echo "  - Required dependencies from pyproject.toml"
        echo ""

        if ask_permission "Install Python dependencies with pip?"; then
            info "Installing dependencies..."
            if $PIP_CMD install -e "$SCRIPT_DIR"; then
                success "Dependencies installed successfully"
            else
                error "Failed to install dependencies"
                info "You can try manually: $PIP_CMD install -e \"$SCRIPT_DIR\""
            fi
        else
            info "Skipped pip install. Run manually when ready:"
            info "  $PIP_CMD install -e \"$SCRIPT_DIR\""
        fi
    else
        warning "pyproject.toml not found in $SCRIPT_DIR"
        info "Cannot determine dependencies to install"
    fi
fi

# =============================================================================
# DATABASE MIGRATION (for upgrades)
# =============================================================================
header "Upgrade Check"

if [ -f "$HOME/.memory-palace/memories.db" ]; then
    info "Found existing Memory Palace database"
    info "Running upgrade check..."

    if [ -n "$PYTHON_CMD" ]; then
        info "Running v2 -> v3 migration (idempotent)..."
        if $PYTHON_CMD -m memory_palace.migrations.v2_to_v3; then
            success "v2 -> v3 migration check complete"
        else
            warning "v2 -> v3 migration failed"
            info "You can retry manually: $PYTHON_CMD -m memory_palace.migrations.v2_to_v3"
        fi

        info "Running v3 -> v3.1 migration (idempotent)..."
        if $PYTHON_CMD -m memory_palace.migrations.v3_to_v3_1; then
            success "v3 -> v3.1 migration check complete"
        else
            warning "v3 -> v3.1 migration failed"
            info "You can retry manually: $PYTHON_CMD -m memory_palace.migrations.v3_to_v3_1"
        fi
    else
        warning "Cannot run migrations without Python"
    fi
else
    info "No existing database found — fresh install"
fi

# =============================================================================
# FIRST-TIME SETUP
# =============================================================================
header "First-Time Setup"

SETUP_SCRIPT="$SCRIPT_DIR/setup/first_run.py"

if [ -f "$SETUP_SCRIPT" ]; then
    info "Found first-time setup script"
    echo ""
    echo -e "${YELLOW}The setup wizard will:${NC}"
    echo "  - Detect your GPU"
    echo "  - Recommend models based on your hardware"
    echo "  - Download recommended Ollama models"
    echo ""

    if [ -n "$PYTHON_CMD" ] && ask_permission "Run first-time setup?"; then
        info "Running first-time setup..."
        if $PYTHON_CMD "$SETUP_SCRIPT"; then
            success "First-time setup complete"
        else
            error "First-time setup failed"
            info "You can run it manually: $PYTHON_CMD \"$SETUP_SCRIPT\""
        fi
    else
        info "Skipped first-time setup. Run manually when ready:"
        info "  $PYTHON_CMD \"$SETUP_SCRIPT\""
    fi
else
    warning "First-time setup script not found at $SETUP_SCRIPT"
    info "You may need to run setup manually after installation"
fi

# =============================================================================
# SUMMARY
# =============================================================================
header "Installation Summary"

if [ ${#MISSING[@]} -eq 0 ]; then
    success "All required dependencies are available!"
else
    warning "Missing dependencies: ${MISSING[*]}"
    info "Please install the missing dependencies and re-run this installer."
fi

if [ -n "$GPU_NAME" ]; then
    info "GPU: $GPU_NAME with ${GPU_VRAM_GB:-unknown} GB VRAM"
else
    info "GPU: None detected (will use CPU)"
fi

echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  1. Ensure Ollama is running: ollama serve"
echo "  2. Run setup wizard: python -m setup.first_run"
echo "  3. Configure your AI client to use the MCP server (see docs/README.md)"
echo ""
echo "Documentation: $SCRIPT_DIR/docs/"
echo ""
