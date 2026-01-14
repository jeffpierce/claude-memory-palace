#Requires -Version 5.1
<#
.SYNOPSIS
    Claude Memory Palace - Windows PowerShell Installer
.DESCRIPTION
    Installs dependencies for Claude Memory Palace with explicit user permission for each step.
    Never auto-installs without consent.
.NOTES
    Run with: powershell -ExecutionPolicy Bypass -File install.ps1
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Colors for output
function Write-Header { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "[!!] $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor White }

function Ask-Permission {
    param(
        [string]$Question,
        [string]$Default = "N"
    )
    $prompt = if ($Default -eq "Y") { "(Y/n)" } else { "(y/N)" }
    $response = Read-Host "$Question $prompt"
    if ([string]::IsNullOrWhiteSpace($response)) {
        return $Default -eq "Y"
    }
    return $response -match "^[Yy]"
}

Write-Host @"

  ____  _                 _        __  __
 / ___|| | __ _ _   _  __| | ___  |  \/  | ___ _ __ ___   ___  _ __ _   _
| |    | |/ _` | | | |/ _` |/ _ \ | |\/| |/ _ \ '_ ` _ \ / _ \| '__| | | |
| |___ | | (_| | |_| | (_| |  __/ | |  | |  __/ | | | | | (_) | |  | |_| |
 \____||_|\__,_|\__,_|\__,_|\___| |_|  |_|\___|_| |_| |_|\___/|_|   \__, |
                                                                    |___/
         P A L A C E   -   I N S T A L L E R

"@ -ForegroundColor Magenta

Write-Host "This installer will check dependencies and ask permission before installing anything.`n"

# Track what's missing
$missing = @()
$gpuInfo = $null

# =============================================================================
# CHECK PYTHON
# =============================================================================
Write-Header "Checking Python"

$pythonCmd = $null
$pythonVersion = $null

# Try python first, then python3
foreach ($cmd in @("python", "python3")) {
    try {
        $output = & $cmd --version 2>&1
        if ($output -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 10) {
                $pythonCmd = $cmd
                $pythonVersion = "$major.$minor.$($Matches[3])"
                break
            }
        }
    } catch {
        # Command not found, continue
    }
}

if ($pythonCmd) {
    Write-Success "Python $pythonVersion found ($pythonCmd)"
} else {
    Write-Warning "Python 3.10+ not found"
    $missing += "Python"

    # Check if winget is available
    $wingetAvailable = $false
    try {
        $null = & winget --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $wingetAvailable = $true
        }
    } catch {
        # winget not available
    }

    if ($wingetAvailable) {
        Write-Info "You can install Python via winget (recommended) or manually:"
        Write-Info "  - winget: winget install Python.Python.3.12"
        Write-Info "  - Python.org: https://www.python.org/downloads/"
        Write-Host ""
        if (Ask-Permission "Install Python 3.12 using winget?") {
            Write-Info "Installing Python 3.12 via winget..."
            try {
                & winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Python installed successfully!"
                    Write-Info "Please close and re-open this terminal, then re-run this installer."
                    Write-Info "(New PATH entries require a fresh terminal session)"
                } else {
                    Write-Error "winget install failed with exit code $LASTEXITCODE"
                    Write-Info "You can try manually: winget install Python.Python.3.12"
                }
            } catch {
                Write-Error "Failed to run winget: $_"
                Write-Info "You can try manually: winget install Python.Python.3.12"
            }
            exit 0
        }
    } else {
        Write-Info "You can install Python from:"
        Write-Info "  - Python.org: https://www.python.org/downloads/"
        Write-Info "  - Or install winget first, then: winget install Python.Python.3.12"
        Write-Host ""
        if (Ask-Permission "Open Python.org download page in browser?") {
            Start-Process "https://www.python.org/downloads/"
            Write-Info "Please install Python and re-run this installer."
            exit 0
        }
    }
}

# =============================================================================
# CHECK OLLAMA
# =============================================================================
Write-Header "Checking Ollama"

$ollamaFound = $false
$ollamaPath = $null

# Check common locations
$ollamaPaths = @(
    "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
    "$env:ProgramFiles\Ollama\ollama.exe",
    "C:\Program Files\Ollama\ollama.exe"
)

foreach ($path in $ollamaPaths) {
    if (Test-Path $path) {
        $ollamaPath = $path
        $ollamaFound = $true
        break
    }
}

# Also check if ollama is in PATH
if (-not $ollamaFound) {
    try {
        $null = & ollama --version 2>&1
        $ollamaFound = $true
        $ollamaPath = "ollama (in PATH)"
    } catch {
        # Not in PATH
    }
}

if ($ollamaFound) {
    Write-Success "Ollama found: $ollamaPath"

    # Check if Ollama is running
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2
        Write-Success "Ollama server is running"
    } catch {
        Write-Warning "Ollama is installed but not running"
        Write-Info "Start Ollama before using the memory palace"
    }
} else {
    Write-Warning "Ollama not found"
    $missing += "Ollama"
    Write-Info "Ollama is required for embeddings and LLM inference."
    Write-Info "Download from: https://ollama.com/download"
    Write-Host ""
    if (Ask-Permission "Open Ollama download page in browser?") {
        Start-Process "https://ollama.com/download"
        Write-Info "Please install Ollama and re-run this installer."
    }
}

# =============================================================================
# CHECK GPU
# =============================================================================
Write-Header "Checking GPU"

try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0 -and $nvidiaSmi -notmatch "NVIDIA-SMI has failed") {
        $lines = $nvidiaSmi -split "`n" | Where-Object { $_ -match "\S" }
        foreach ($line in $lines) {
            if ($line -match "^(.+),\s*(\d+)\s*MiB") {
                $gpuName = $Matches[1].Trim()
                $vramMB = [int]$Matches[2]
                $vramGB = [math]::Round($vramMB / 1024, 1)

                Write-Success "NVIDIA GPU: $gpuName ($vramGB GB VRAM)"
                $gpuInfo = @{
                    Name = $gpuName
                    VRAM_MB = $vramMB
                    VRAM_GB = $vramGB
                }

                # Recommendations based on VRAM
                if ($vramGB -ge 16) {
                    Write-Info "Excellent! You can run large embedding models (SFR-Embedding-Mistral F16)"
                } elseif ($vramGB -ge 8) {
                    Write-Info "Good. You can run medium embedding models (nomic-embed-text)"
                } else {
                    Write-Warning "Limited VRAM. Consider using smaller models or CPU inference."
                }
            }
        }
    } else {
        throw "nvidia-smi failed"
    }
} catch {
    Write-Warning "No NVIDIA GPU detected (or nvidia-smi not available)"
    Write-Info "The memory palace will use CPU inference (slower but functional)"
}

# =============================================================================
# CHECK PIP AND INSTALL DEPENDENCIES
# =============================================================================
Write-Header "Python Dependencies"

if (-not $pythonCmd) {
    Write-Error "Cannot install Python dependencies without Python installed."
    Write-Info "Please install Python 3.10+ and re-run this installer."
} else {
    # Check if pip is available
    $pipCmd = $null
    foreach ($cmd in @("pip", "pip3", "$pythonCmd -m pip")) {
        try {
            if ($cmd -match "-m pip") {
                $output = & $pythonCmd -m pip --version 2>&1
            } else {
                $output = & $cmd --version 2>&1
            }
            if ($output -match "pip") {
                $pipCmd = $cmd
                break
            }
        } catch {
            # Continue
        }
    }

    if (-not $pipCmd) {
        Write-Warning "pip not found"
        Write-Info "Try: $pythonCmd -m ensurepip --upgrade"
    } else {
        Write-Success "pip found"

        # Check for pyproject.toml
        $pyprojectPath = Join-Path $ScriptDir "pyproject.toml"
        if (Test-Path $pyprojectPath) {
            Write-Info "Found pyproject.toml - will install package in development mode"
            Write-Host ""
            Write-Host "This will install the following:" -ForegroundColor Yellow
            Write-Host "  - claude-memory-palace package (editable install)"
            Write-Host "  - Required dependencies from pyproject.toml"
            Write-Host ""

            if (Ask-Permission "Install Python dependencies with pip?") {
                Write-Info "Installing dependencies..."
                try {
                    & $pythonCmd -m pip install -e $ScriptDir
                    Write-Success "Dependencies installed successfully"
                } catch {
                    Write-Error "Failed to install dependencies: $_"
                    Write-Info "You can try manually: $pythonCmd -m pip install -e `"$ScriptDir`""
                }
            } else {
                Write-Info "Skipped pip install. Run manually when ready:"
                Write-Info "  $pythonCmd -m pip install -e `"$ScriptDir`""
            }
        } else {
            Write-Warning "pyproject.toml not found in $ScriptDir"
            Write-Info "Cannot determine dependencies to install"
        }
    }
}

# =============================================================================
# FIRST-TIME SETUP
# =============================================================================
Write-Header "First-Time Setup"

$setupScript = Join-Path $ScriptDir "setup" "first_time_setup.py"
if (Test-Path $setupScript) {
    Write-Info "Found first-time setup script"
    Write-Host ""
    Write-Host "First-time setup will:" -ForegroundColor Yellow
    Write-Host "  - Create configuration directory (~/.claude-memory-palace/)"
    Write-Host "  - Initialize the SQLite database"
    Write-Host "  - Pull required Ollama models (if Ollama is running)"
    Write-Host ""

    if ($pythonCmd -and (Ask-Permission "Run first-time setup?")) {
        Write-Info "Running first-time setup..."
        try {
            & $pythonCmd $setupScript
            Write-Success "First-time setup complete"
        } catch {
            Write-Error "First-time setup failed: $_"
            Write-Info "You can run it manually: $pythonCmd `"$setupScript`""
        }
    } else {
        Write-Info "Skipped first-time setup. Run manually when ready:"
        Write-Info "  $pythonCmd `"$setupScript`""
    }
} else {
    Write-Warning "First-time setup script not found at $setupScript"
    Write-Info "You may need to run setup manually after installation"
}

# =============================================================================
# SUMMARY
# =============================================================================
Write-Header "Installation Summary"

if ($missing.Count -eq 0) {
    Write-Success "All required dependencies are available!"
} else {
    Write-Warning "Missing dependencies: $($missing -join ', ')"
    Write-Info "Please install the missing dependencies and re-run this installer."
}

if ($gpuInfo) {
    Write-Info "GPU: $($gpuInfo.Name) with $($gpuInfo.VRAM_GB) GB VRAM"
} else {
    Write-Info "GPU: None detected (will use CPU)"
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Ensure Ollama is running: ollama serve"
Write-Host "  2. Pull embedding model: ollama pull sfr-embedding-mistral:f16"
Write-Host "  3. Configure Claude Code to use the MCP server"
Write-Host ""
Write-Host "Documentation: $ScriptDir\docs\"
Write-Host ""
