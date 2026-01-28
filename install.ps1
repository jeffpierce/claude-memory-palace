# ============================================================================
# Claude Memory Palace — Windows PowerShell Installer
# 
# Usage: irm https://raw.githubusercontent.com/yourusername/memory-palace/main/install.ps1 | iex
#   or:  .\install.ps1
# ============================================================================

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# --- Constants ---
$REPO_URL = "https://github.com/clawdbot/memory-palace.git"
$BRANCH = "main"
$INSTALL_DIR = Join-Path $env:USERPROFILE "memory-palace"
$EMBEDDING_MODEL = "nomic-embed-text"
$MIN_PYTHON_MAJOR = 3
$MIN_PYTHON_MINOR = 10

# --- State ---
$script:PythonCmd = ""
$script:HasOllama = $false
$script:HasGPU = $false
$script:GPUName = ""
$script:VramGB = 0
$script:LLMModel = ""
$script:SelectedClients = @()

# ============================================================================
# Utilities
# ============================================================================

function Write-Info    { param($m) Write-Host "i " -ForegroundColor Blue -NoNewline; Write-Host $m }
function Write-Success { param($m) Write-Host "✓ " -ForegroundColor Green -NoNewline; Write-Host $m }
function Write-Warn    { param($m) Write-Host "⚠ " -ForegroundColor Yellow -NoNewline; Write-Host $m }
function Write-Err     { param($m) Write-Host "✗ " -ForegroundColor Red -NoNewline; Write-Host $m }
function Write-Header  { param($m) Write-Host "`n$m`n" -ForegroundColor Cyan }

function Prompt-YN {
    param([string]$Question, [string]$Default = "y")
    $hint = if ($Default -eq "y") { "[Y/n]" } else { "[y/N]" }
    $answer = Read-Host "$Question $hint"
    if ([string]::IsNullOrWhiteSpace($answer)) { $answer = $Default }
    return $answer -match "^[Yy]"
}

# ============================================================================
# Detection
# ============================================================================

function Detect-Python {
    Write-Info "Checking Python..."
    
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $output = & $cmd --version 2>&1
            if ($output -match "Python (\d+)\.(\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -ge $MIN_PYTHON_MAJOR -and $minor -ge $MIN_PYTHON_MINOR) {
                    $script:PythonCmd = $cmd
                    Write-Success "Python $($Matches[0]) ($cmd)"
                    return
                }
            }
        } catch { }
    }
    
    Write-Warn "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ not found"
}

function Detect-Ollama {
    Write-Info "Checking Ollama..."
    
    try {
        $output = & ollama --version 2>&1
        $script:HasOllama = $true
        if ($output -match "(\d+\.\d+\.?\d*)") {
            Write-Success "Ollama $($Matches[1])"
        } else {
            Write-Success "Ollama installed"
        }
        
        # Check models
        try {
            $models = & ollama list 2>&1
            Write-Info "Models: $($models | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] } | Where-Object { $_ } | Join-String -Separator ', ')"
        } catch { }
    } catch {
        Write-Warn "Ollama not installed"
    }
}

function Detect-GPU {
    Write-Info "Checking GPU..."
    
    try {
        $output = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>&1
        if ($LASTEXITCODE -eq 0 -and $output) {
            $parts = $output -split ","
            $script:HasGPU = $true
            $script:GPUName = $parts[0].Trim()
            $script:VramGB = [math]::Floor([int]$parts[1].Trim() / 1024)
            Write-Success "$($script:GPUName) ($($script:VramGB)GB VRAM)"
            return
        }
    } catch { }
    
    Write-Info "No dedicated GPU detected — CPU mode is fine for embeddings"
}

function Select-LLMModel {
    if ($script:VramGB -ge 16) { $script:LLMModel = "qwen3:14b" }
    elseif ($script:VramGB -ge 10) { $script:LLMModel = "qwen3:8b" }
    elseif ($script:VramGB -ge 6) { $script:LLMModel = "qwen3:4b" }
    elseif ($script:VramGB -ge 2) { $script:LLMModel = "qwen3:1.7b" }
    else { $script:LLMModel = "" }
}

# ============================================================================
# AI Client Discovery
# ============================================================================

$CLIENT_CONFIGS = @{
    "claude-desktop" = @{
        Name = "Claude Desktop"
        Path = Join-Path $env:APPDATA "Claude\claude_desktop_config.json"
    }
    "claude-code" = @{
        Name = "Claude Code"
        Path = Join-Path $env:USERPROFILE ".claude.json"
    }
    "cursor" = @{
        Name = "Cursor"
        Path = Join-Path $env:USERPROFILE ".cursor\mcp.json"
    }
    "windsurf" = @{
        Name = "Windsurf"
        Path = Join-Path $env:USERPROFILE ".codeium\windsurf\mcp_config.json"
    }
    "continue" = @{
        Name = "VS Code + Continue"
        Path = Join-Path $env:USERPROFILE ".continue\config.json"
    }
}

function Is-ClientInstalled {
    param([string]$ClientId)
    $cfg = $CLIENT_CONFIGS[$ClientId]
    if (-not $cfg) { return $false }
    
    $configPath = $cfg.Path
    if (Test-Path $configPath) { return $true }
    if (Test-Path (Split-Path $configPath -Parent)) { return $true }
    
    # App-specific checks
    switch ($ClientId) {
        "claude-code" { return (Get-Command "claude" -ErrorAction SilentlyContinue) -ne $null }
        "cursor" {
            $localAppData = $env:LOCALAPPDATA
            return (Test-Path (Join-Path $localAppData "Programs\Cursor")) -or 
                   (Test-Path (Join-Path $localAppData "Cursor"))
        }
        "windsurf" {
            $localAppData = $env:LOCALAPPDATA
            return (Test-Path (Join-Path $localAppData "Programs\Windsurf")) -or
                   (Test-Path (Join-Path $localAppData "Windsurf"))
        }
    }
    return $false
}

function Is-AlreadyConfigured {
    param([string]$ConfigPath)
    if (-not (Test-Path $ConfigPath)) { return $false }
    $content = Get-Content $ConfigPath -Raw -ErrorAction SilentlyContinue
    return $content -and $content.Contains('"memory-palace"')
}

function Discover-Clients {
    Write-Header "🤖 Which AI tools do you use?"
    
    $detected = @()
    
    foreach ($clientId in $CLIENT_CONFIGS.Keys) {
        if (Is-ClientInstalled $clientId) {
            $cfg = $CLIENT_CONFIGS[$clientId]
            $status = ""
            if (Is-AlreadyConfigured $cfg.Path) {
                $status = " (already configured)"
            }
            $detected += $clientId
            Write-Success "Found: $($cfg.Name)$status"
        }
    }
    
    if ($detected.Count -eq 0) {
        Write-Warn "No AI clients detected automatically."
        Write-Host ""
        Write-Host "Memory Palace works with any MCP-compatible client."
        Write-Host "You can configure your client manually after installation."
        return
    }
    
    Write-Host ""
    Write-Host "Select clients to configure:" -ForegroundColor White
    Write-Host "(Enter numbers separated by spaces, or 'a' for all, 'n' for none)" -ForegroundColor DarkGray
    Write-Host ""
    
    for ($i = 0; $i -lt $detected.Count; $i++) {
        $clientId = $detected[$i]
        $cfg = $CLIENT_CONFIGS[$clientId]
        $status = ""
        if (Is-AlreadyConfigured $cfg.Path) { $status = " [configured]" }
        Write-Host "  $($i + 1)) $($cfg.Name)$status" -ForegroundColor Cyan
    }
    
    Write-Host ""
    $selection = Read-Host "Selection"
    
    switch -Regex ($selection) {
        "^[aA]" { $script:SelectedClients = $detected }
        "^[nN]|^$" { $script:SelectedClients = @() }
        default {
            foreach ($num in ($selection -split '\s+')) {
                $idx = [int]$num - 1
                if ($idx -ge 0 -and $idx -lt $detected.Count) {
                    $script:SelectedClients += $detected[$idx]
                }
            }
        }
    }
    
    if ($script:SelectedClients.Count -gt 0) {
        Write-Host ""
        Write-Info "Will configure: $($script:SelectedClients | ForEach-Object { $CLIENT_CONFIGS[$_].Name } | Join-String -Separator ', ')"
    }
}

# ============================================================================
# Installation
# ============================================================================

function Install-Python {
    Write-Header "📦 Installing Python..."
    
    try {
        $hasWinget = (Get-Command "winget" -ErrorAction SilentlyContinue) -ne $null
        if ($hasWinget) {
            Write-Info "Installing via winget..."
            winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
            
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + 
                        [System.Environment]::GetEnvironmentVariable("Path", "User")
            
            Detect-Python
        } else {
            Write-Err "Please install Python 3.10+ from https://www.python.org/downloads/"
            Write-Err "Or install winget first (included in modern Windows 10/11)"
            exit 1
        }
    } catch {
        Write-Err "Python installation failed: $_"
        exit 1
    }
}

function Install-Ollama {
    Write-Header "📦 Installing Ollama..."
    
    try {
        $hasWinget = (Get-Command "winget" -ErrorAction SilentlyContinue) -ne $null
        if ($hasWinget) {
            Write-Info "Installing via winget..."
            winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
            
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + 
                        [System.Environment]::GetEnvironmentVariable("Path", "User")
            
            $script:HasOllama = $true
            Write-Success "Ollama installed"
            
            # Start Ollama
            Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 3
        } else {
            Write-Info "Opening download page..."
            Start-Process "https://ollama.com/download"
            Write-Host "Please install Ollama and re-run this script."
            exit 1
        }
    } catch {
        Write-Err "Ollama installation failed. Please install from https://ollama.com"
        exit 1
    }
}

function Download-Repo {
    Write-Header "📥 Downloading Memory Palace..."
    
    if (Test-Path (Join-Path $INSTALL_DIR ".git")) {
        Write-Info "Existing installation found — updating..."
        Push-Location $INSTALL_DIR
        git pull origin $BRANCH
        Pop-Location
    } else {
        $hasGit = (Get-Command "git" -ErrorAction SilentlyContinue) -ne $null
        if ($hasGit) {
            git clone --branch $BRANCH $REPO_URL $INSTALL_DIR
        } else {
            # Fallback: download zip
            Write-Info "git not found — downloading archive..."
            $zipUrl = "https://github.com/clawdbot/memory-palace/archive/refs/heads/${BRANCH}.zip"
            $zipFile = Join-Path $env:TEMP "memory-palace.zip"
            
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipFile
            Expand-Archive -Path $zipFile -DestinationPath $env:TEMP -Force
            
            # Move extracted folder
            $extracted = Join-Path $env:TEMP "memory-palace-$BRANCH"
            if (Test-Path $INSTALL_DIR) { Remove-Item $INSTALL_DIR -Recurse -Force }
            Move-Item $extracted $INSTALL_DIR
            Remove-Item $zipFile -ErrorAction SilentlyContinue
        }
    }
    
    Write-Success "Memory Palace downloaded to $INSTALL_DIR"
}

function Setup-Venv {
    Write-Header "🐍 Setting up Python environment..."
    
    $venvDir = Join-Path $INSTALL_DIR "venv"
    
    Write-Info "Creating virtual environment..."
    & $script:PythonCmd -m venv $venvDir
    
    $pipCmd = Join-Path $venvDir "Scripts\pip.exe"
    
    Write-Info "Installing Memory Palace package..."
    & $pipCmd install -e $INSTALL_DIR --quiet
    
    Write-Success "Package installed"
    
    # Verify
    $pythonCmd = Join-Path $venvDir "Scripts\python.exe"
    $verify = & $pythonCmd -c "import memory_palace; import mcp_server; print('OK')" 2>&1
    if ($verify -match "OK") {
        Write-Success "Installation verified"
    } else {
        Write-Warn "Package import check failed — installation may have issues"
    }
}

function Pull-Models {
    Write-Header "🧠 Downloading AI models..."
    
    # Embedding model
    $modelList = & ollama list 2>&1
    if ($modelList -match "nomic-embed-text") {
        Write-Success "Embedding model already installed"
    } else {
        Write-Info "Downloading embedding model ($EMBEDDING_MODEL)..."
        Write-Host "  This is ~270MB — should be quick" -ForegroundColor DarkGray
        & ollama pull $EMBEDDING_MODEL
        Write-Success "Embedding model ready"
    }
    
    # Base LLM model (required — handles synthesis, classification, extraction)
    $baseLLM = "qwen3:1.7b"
    Write-Host ""
    if ($modelList -match "qwen3:1.7b") {
        Write-Success "Base LLM model (qwen3:1.7b) already installed"
    } else {
        Write-Info "Downloading base LLM model (qwen3:1.7b)..."
        Write-Host "  ~1GB — handles synthesis, classification, and extraction" -ForegroundColor DarkGray
        & ollama pull $baseLLM
        Write-Success "Base LLM model ready"
    }

    # Upgraded LLM model (optional, for better quality on capable hardware)
    Select-LLMModel
    
    if ($script:LLMModel -and $script:LLMModel -ne $baseLLM) {
        $modelBase = ($script:LLMModel -split ":")[0]
        if ($modelList -match $modelBase) {
            Write-Success "Upgraded LLM model already installed"
        } else {
            Write-Host ""
            Write-Host "Recommended upgrade for your hardware: $($script:LLMModel)" -ForegroundColor White
            Write-Host "Better quality for memory extraction and synthesis" -ForegroundColor DarkGray
            Write-Host ""
            if (Prompt-YN "Download $($script:LLMModel)?") {
                Write-Info "Downloading $($script:LLMModel) (this may take a while)..."
                & ollama pull $script:LLMModel
                Write-Success "Upgraded LLM model ready"
            } else {
                Write-Info "Skipped — upgrade later with: ollama pull $($script:LLMModel)"
            }
        }
    }
}

function Configure-Client {
    param([string]$ClientId)
    
    $cfg = $CLIENT_CONFIGS[$ClientId]
    if (-not $cfg) { Write-Warn "Unknown client: $ClientId"; return }
    
    $configPath = $cfg.Path
    $pythonPath = Join-Path $INSTALL_DIR "venv\Scripts\python.exe"
    
    # Ensure directory exists
    $configDir = Split-Path $configPath -Parent
    if (-not (Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }
    
    # Backup existing
    if (Test-Path $configPath) {
        $backup = "${configPath}.backup"
        $counter = 1
        while (Test-Path $backup) {
            $backup = "${configPath}.backup.${counter}"
            $counter++
        }
        Copy-Item $configPath $backup
        Write-Info "Backed up to: $(Split-Path $backup -Leaf)"
    }
    
    # Load or create config
    $config = @{}
    if (Test-Path $configPath) {
        try {
            $config = Get-Content $configPath -Raw | ConvertFrom-Json -AsHashtable
        } catch {
            $config = @{}
        }
    }
    
    $mcpEntry = @{
        command = $pythonPath
        args = @("-m", "mcp_server.server")
        cwd = $INSTALL_DIR
    }
    
    if ($ClientId -eq "continue") {
        # Continue uses different format
        if (-not $config.ContainsKey("experimental")) {
            $config["experimental"] = @{}
        }
        if (-not $config["experimental"].ContainsKey("modelContextProtocolServers")) {
            $config["experimental"]["modelContextProtocolServers"] = @()
        }
        
        $servers = @($config["experimental"]["modelContextProtocolServers"] | Where-Object { $_.name -ne "memory-palace" })
        $servers += @{
            name = "memory-palace"
            command = $pythonPath
            args = @("-m", "mcp_server.server")
            cwd = $INSTALL_DIR
        }
        $config["experimental"]["modelContextProtocolServers"] = $servers
    } else {
        # Standard MCP format
        if (-not $config.ContainsKey("mcpServers")) {
            $config["mcpServers"] = @{}
        }
        $config["mcpServers"]["memory-palace"] = $mcpEntry
    }
    
    # Write config
    $config | ConvertTo-Json -Depth 10 | Set-Content $configPath -Encoding UTF8
    Write-Success "Configured $($cfg.Name)"
}

function Configure-SelectedClients {
    if ($script:SelectedClients.Count -eq 0) { return }
    
    Write-Header "⚙️  Configuring AI clients..."
    
    foreach ($clientId in $script:SelectedClients) {
        Configure-Client $clientId
    }
}

# ============================================================================
# Summary
# ============================================================================

function Show-Summary {
    Write-Header "🎉 Memory Palace is ready!"
    
    Write-Host "Installation: $INSTALL_DIR" -ForegroundColor Green
    Write-Host "Embedding:    $EMBEDDING_MODEL" -ForegroundColor Green
    if ($script:LLMModel) {
        Write-Host "LLM:          $($script:LLMModel)" -ForegroundColor Green
    }
    
    if ($script:SelectedClients.Count -gt 0) {
        Write-Host ""
        Write-Host "Configured clients:" -ForegroundColor Green
        foreach ($clientId in $script:SelectedClients) {
            Write-Host "  ✓ $($CLIENT_CONFIGS[$clientId].Name)"
        }
    }
    
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor White
    
    if ($script:SelectedClients.Count -gt 0) {
        Write-Host "  1. Restart your AI client(s) to activate Memory Palace"
        Write-Host "  2. Test it out:"
        Write-Host '     Tell Claude: "Remember that I like coffee"' -ForegroundColor DarkGray
        Write-Host '     In a new chat: "What do I like to drink?"' -ForegroundColor DarkGray
    } else {
        $pythonPath = Join-Path $INSTALL_DIR "venv\Scripts\python.exe"
        Write-Host "  1. Configure your MCP client to use Memory Palace"
        Write-Host "     Add this to your MCP config:"
        Write-Host ""
        Write-Host "     `"memory-palace`": {" -ForegroundColor DarkGray
        Write-Host "       `"command`": `"$pythonPath`"," -ForegroundColor DarkGray
        Write-Host "       `"args`": [`"-m`", `"mcp_server.server`"]," -ForegroundColor DarkGray
        Write-Host "       `"cwd`": `"$INSTALL_DIR`"" -ForegroundColor DarkGray
        Write-Host "     }" -ForegroundColor DarkGray
    }
    
    Write-Host ""
    Write-Host "To update later: cd $INSTALL_DIR; git pull; .\venv\Scripts\pip.exe install -e ." -ForegroundColor DarkGray
    Write-Host ""
}

# ============================================================================
# Main Flow
# ============================================================================

function Main {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║     Claude Memory Palace Installer     ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    
    # --- Detection ---
    Write-Header "🔍 Detecting your system..."
    Write-Info "Platform: Windows ($([System.Environment]::OSVersion.Version))"
    
    Detect-Python
    Detect-Ollama
    Detect-GPU
    
    # --- Prerequisites ---
    if (-not $script:PythonCmd) {
        if (Prompt-YN "Python 3.10+ is required. Install it now?") {
            Install-Python
        } else {
            Write-Err "Python 3.10+ is required. Please install and re-run."
            exit 1
        }
    }
    
    if (-not $script:HasOllama) {
        if (Prompt-YN "Ollama is required for embeddings. Install it now?") {
            Install-Ollama
        } else {
            Write-Err "Ollama is required. Install from https://ollama.com and re-run."
            exit 1
        }
    }
    
    # --- Client Discovery ---
    Discover-Clients
    
    # --- Installation ---
    Download-Repo
    Setup-Venv
    Pull-Models
    
    # --- Client Configuration ---
    Configure-SelectedClients
    
    # --- Done! ---
    Show-Summary
}

# Run it
Main
