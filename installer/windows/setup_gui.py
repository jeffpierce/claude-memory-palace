#!/usr/bin/env python3
"""
Claude Memory Palace - Windows GUI Installer
A tkinter-based installer for non-developer users.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import sys
import os
import re
import webbrowser
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class DependencyChecker:
    """Check system dependencies for Memory Palace."""

    @staticmethod
    def check_python() -> Tuple[bool, str]:
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        if version.major >= 3 and version.minor >= 10:
            return True, f"Python {version_str}"
        return False, f"Python {version_str} (need 3.10+)"

    @staticmethod
    def check_winget() -> bool:
        """Check if winget is available."""
        try:
            result = subprocess.run(
                ["winget", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    @staticmethod
    def install_python_with_winget() -> Tuple[bool, str]:
        """Attempt to install Python 3.12 via winget."""
        try:
            result = subprocess.run(
                ["winget", "install", "Python.Python.3.12",
                 "--accept-package-agreements", "--accept-source-agreements"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            if result.returncode == 0:
                return True, "Python 3.12 installed successfully"
            return False, f"Installation failed: {result.stderr[:100]}"
        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    @staticmethod
    def check_ollama() -> Tuple[bool, str]:
        """Check if Ollama is installed and running."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            if result.returncode == 0:
                version = result.stdout.strip() or result.stderr.strip()
                # Try to extract version number
                match = re.search(r'(\d+\.\d+\.?\d*)', version)
                if match:
                    return True, f"Ollama {match.group(1)}"
                return True, "Ollama installed"
            return False, "Ollama not found"
        except FileNotFoundError:
            return False, "Ollama not installed"
        except subprocess.TimeoutExpired:
            return False, "Ollama not responding"
        except Exception as e:
            return False, f"Error: {str(e)[:30]}"

    @staticmethod
    def check_nvidia_gpu() -> Tuple[bool, str, int]:
        """Check for NVIDIA GPU and return VRAM in GB."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    # Parse "GeForce RTX 3080, 10240" format
                    parts = output.split(",")
                    if len(parts) >= 2:
                        gpu_name = parts[0].strip()
                        vram_mb = int(parts[1].strip())
                        vram_gb = vram_mb // 1024
                        return True, f"{gpu_name} ({vram_gb}GB)", vram_gb
                    return True, output, 0
            return False, "No NVIDIA GPU detected", 0
        except FileNotFoundError:
            return False, "nvidia-smi not found", 0
        except Exception as e:
            return False, f"Error: {str(e)[:30]}", 0

    @staticmethod
    def check_ollama_model(model_name: str) -> Tuple[bool, str]:
        """Check if a specific Ollama model is installed."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            if result.returncode == 0:
                # Check if model name appears in output
                if model_name.lower() in result.stdout.lower():
                    return True, f"{model_name} installed"
                return False, f"{model_name} not found"
            return False, "Could not list models"
        except Exception as e:
            return False, f"Error: {str(e)[:30]}"

    @staticmethod
    def get_model_recommendations(vram_gb: int) -> Dict[str, str]:
        """Get recommended models based on available VRAM."""
        if vram_gb >= 16:
            return {
                "embedding": "sfr-embedding-mistral:f16",
                "embedding_desc": "SFR-Embedding-Mistral F16 (14GB) - Best quality",
                "llm": "qwen3:14b",
                "llm_desc": "Qwen3 14B (14GB) - Best for memory extraction"
            }
        elif vram_gb >= 10:
            return {
                "embedding": "snowflake-arctic-embed:l",
                "embedding_desc": "Snowflake Arctic Embed L (1.5GB) - Good quality",
                "llm": "qwen2.5:7b",
                "llm_desc": "Qwen2.5 7B (7GB) - Good for memory extraction"
            }
        elif vram_gb >= 6:
            return {
                "embedding": "snowflake-arctic-embed:m",
                "embedding_desc": "Snowflake Arctic Embed M (700MB) - Moderate quality",
                "llm": "qwen2.5:3b",
                "llm_desc": "Qwen2.5 3B (3GB) - Basic memory extraction"
            }
        else:
            return {
                "embedding": "snowflake-arctic-embed:s",
                "embedding_desc": "Snowflake Arctic Embed S (300MB) - Basic quality",
                "llm": "qwen2.5:1.5b",
                "llm_desc": "Qwen2.5 1.5B (1.5GB) - Minimal"
            }


class InstallerGUI:
    """Main installer GUI application."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Claude Memory Palace Setup")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")

        # State
        self.current_screen = 0
        self.checker = DependencyChecker()
        self.dependency_results = {}
        self.vram_gb = 0
        self.model_recommendations = {}

        # Installation options
        self.install_ollama = tk.BooleanVar(value=False)
        self.install_package = tk.BooleanVar(value=True)
        self.install_embedding = tk.BooleanVar(value=True)
        self.install_llm = tk.BooleanVar(value=False)
        self.configure_claude = tk.BooleanVar(value=True)

        # Track if auto-configure succeeded
        self.configure_result = None

        # Store package directory for later use
        installer_dir = os.path.dirname(os.path.abspath(__file__))
        self.package_dir = os.path.dirname(os.path.dirname(installer_dir))

        # Create main container
        self.container = ttk.Frame(self.root, padding="20")
        self.container.pack(fill=tk.BOTH, expand=True)

        # Show first screen
        self.show_welcome()

    def clear_container(self):
        """Clear all widgets from container."""
        for widget in self.container.winfo_children():
            widget.destroy()

    def show_welcome(self):
        """Screen 1: Welcome screen."""
        self.clear_container()
        self.current_screen = 1

        # Title
        title = ttk.Label(
            self.container,
            text="Claude Memory Palace Setup",
            font=("Segoe UI", 24, "bold")
        )
        title.pack(pady=(30, 20))

        # Description
        desc_frame = ttk.Frame(self.container)
        desc_frame.pack(fill=tk.X, padx=40, pady=20)

        description = """Welcome to the Claude Memory Palace installer.

This will set up a persistent memory system for your Claude instances,
enabling semantic search across conversations and memory handoff
between different Claude instances.

The installer will check your system for:
  - Python 3.10 or higher
  - Ollama (local LLM runner)
  - NVIDIA GPU for acceleration
  - Required AI models

Click Next to check your system."""

        desc_label = ttk.Label(
            desc_frame,
            text=description,
            font=("Segoe UI", 11),
            justify=tk.LEFT,
            wraplength=500
        )
        desc_label.pack(anchor=tk.W)

        # Spacer
        ttk.Frame(self.container).pack(fill=tk.BOTH, expand=True)

        # Navigation
        nav_frame = ttk.Frame(self.container)
        nav_frame.pack(fill=tk.X, pady=20)

        ttk.Button(
            nav_frame,
            text="Cancel",
            command=self.root.destroy,
            width=15
        ).pack(side=tk.LEFT)

        ttk.Button(
            nav_frame,
            text="Next",
            command=self.show_dependency_check,
            width=15
        ).pack(side=tk.RIGHT)

    def show_dependency_check(self):
        """Screen 2: Dependency check screen."""
        self.clear_container()
        self.current_screen = 2

        # Title
        title = ttk.Label(
            self.container,
            text="Checking System Requirements",
            font=("Segoe UI", 18, "bold")
        )
        title.pack(pady=(20, 30))

        # Checklist frame
        check_frame = ttk.LabelFrame(self.container, text="System Status", padding="20")
        check_frame.pack(fill=tk.X, padx=40)

        # Status items
        self.status_labels = {}
        checks = [
            ("python", "Python Version"),
            ("ollama", "Ollama"),
            ("gpu", "NVIDIA GPU"),
            ("embedding", "Embedding Model"),
            ("llm", "LLM Model (optional)")
        ]

        for key, label in checks:
            row = ttk.Frame(check_frame)
            row.pack(fill=tk.X, pady=5)

            status_label = ttk.Label(row, text="...", width=3, font=("Segoe UI", 12))
            status_label.pack(side=tk.LEFT)

            name_label = ttk.Label(row, text=label, font=("Segoe UI", 11), width=20, anchor=tk.W)
            name_label.pack(side=tk.LEFT, padx=(10, 0))

            detail_label = ttk.Label(row, text="Checking...", font=("Segoe UI", 10), foreground="gray")
            detail_label.pack(side=tk.LEFT, padx=(10, 0))

            self.status_labels[key] = (status_label, detail_label)

        # Spacer
        ttk.Frame(self.container).pack(fill=tk.BOTH, expand=True)

        # Navigation
        nav_frame = ttk.Frame(self.container)
        nav_frame.pack(fill=tk.X, pady=20)

        ttk.Button(
            nav_frame,
            text="Back",
            command=self.show_welcome,
            width=15
        ).pack(side=tk.LEFT)

        self.next_btn = ttk.Button(
            nav_frame,
            text="Next",
            command=self.show_installation_options,
            width=15,
            state=tk.DISABLED
        )
        self.next_btn.pack(side=tk.RIGHT)

        # Run checks in background
        threading.Thread(target=self.run_dependency_checks, daemon=True).start()

    def update_status(self, key: str, success: bool, detail: str):
        """Update a status item."""
        if key in self.status_labels:
            status_label, detail_label = self.status_labels[key]
            if success:
                status_label.config(text="[OK]", foreground="green")
            else:
                status_label.config(text="[X]", foreground="red")
            detail_label.config(text=detail, foreground="black" if success else "gray")

    def show_python_upgrade_dialog(self, current_version: str):
        """Show dialog with Python upgrade options."""
        has_winget = self.checker.check_winget()

        if has_winget:
            message = (
                f"Python {current_version} is installed but version 3.10+ is required.\n\n"
                "Would you like to install Python 3.12 using winget?\n\n"
                "Command: winget install Python.Python.3.12"
            )
            if messagebox.askyesno("Python Upgrade Required", message):
                # Show progress in main window
                self.log_python_install("Installing Python 3.12 via winget...")
                self.log_python_install("This may take a few minutes...")

                def do_install():
                    success, msg = self.checker.install_python_with_winget()
                    if success:
                        self.root.after(0, lambda: messagebox.showinfo(
                            "Success",
                            "Python 3.12 installed successfully!\n\n"
                            "Please restart this installer to continue."
                        ))
                        self.root.after(0, self.root.destroy)
                    else:
                        self.root.after(0, lambda: messagebox.showerror(
                            "Installation Failed",
                            f"{msg}\n\n"
                            "Please install Python manually:\n"
                            "winget install Python.Python.3.12"
                        ))

                threading.Thread(target=do_install, daemon=True).start()
        else:
            message = (
                f"Python {current_version} is installed but version 3.10+ is required.\n\n"
                "Please install Python 3.12 manually:\n\n"
                "Option 1 (recommended): winget install Python.Python.3.12\n"
                "Option 2: Download from https://www.python.org/downloads/\n\n"
                "After installing, restart this installer."
            )
            messagebox.showwarning("Python Upgrade Required", message)

    def log_python_install(self, message: str):
        """Log message during Python install (reuses status labels if available)."""
        if hasattr(self, 'status_labels') and "python" in self.status_labels:
            status_label, detail_label = self.status_labels["python"]
            detail_label.config(text=message, foreground="blue")

    def run_dependency_checks(self):
        """Run all dependency checks."""
        # Python
        success, detail = self.checker.check_python()
        self.dependency_results["python"] = (success, detail)
        self.root.after(0, lambda: self.update_status("python", success, detail))

        # If Python version is too old, offer to upgrade
        if not success:
            version = sys.version_info
            current = f"{version.major}.{version.minor}"
            self.root.after(100, lambda: self.show_python_upgrade_dialog(current))

        # Ollama
        success, detail = self.checker.check_ollama()
        self.dependency_results["ollama"] = (success, detail)
        self.root.after(0, lambda: self.update_status("ollama", success, detail))

        # GPU
        success, detail, vram = self.checker.check_nvidia_gpu()
        self.dependency_results["gpu"] = (success, detail)
        self.vram_gb = vram
        self.model_recommendations = self.checker.get_model_recommendations(vram)
        self.root.after(0, lambda: self.update_status("gpu", success, detail))

        # Embedding model
        if self.model_recommendations:
            embedding_model = self.model_recommendations.get("embedding", "snowflake-arctic-embed:l")
            success, detail = self.checker.check_ollama_model(embedding_model)
            self.dependency_results["embedding"] = (success, detail)
            self.root.after(0, lambda: self.update_status("embedding", success, detail))
        else:
            self.root.after(0, lambda: self.update_status("embedding", False, "Need GPU info first"))

        # LLM model (optional)
        if self.model_recommendations:
            llm_model = self.model_recommendations.get("llm", "qwen2.5:7b")
            success, detail = self.checker.check_ollama_model(llm_model)
            self.dependency_results["llm"] = (success, detail)
            self.root.after(0, lambda: self.update_status("llm", success, detail))
        else:
            self.root.after(0, lambda: self.update_status("llm", False, "Need GPU info first"))

        # Enable next button
        self.root.after(0, lambda: self.next_btn.config(state=tk.NORMAL))

    def show_installation_options(self):
        """Screen 3: Installation options screen."""
        self.clear_container()
        self.current_screen = 3

        # Title
        title = ttk.Label(
            self.container,
            text="Installation Options",
            font=("Segoe UI", 18, "bold")
        )
        title.pack(pady=(20, 20))

        # Options frame
        options_frame = ttk.LabelFrame(self.container, text="Select Components to Install", padding="20")
        options_frame.pack(fill=tk.X, padx=40)

        # Ollama option (if not installed)
        ollama_installed = self.dependency_results.get("ollama", (False, ""))[0]
        if not ollama_installed:
            self.install_ollama.set(True)
            row = ttk.Frame(options_frame)
            row.pack(fill=tk.X, pady=5)
            ttk.Checkbutton(
                row,
                text="Open Ollama download page",
                variable=self.install_ollama
            ).pack(anchor=tk.W)
            ttk.Label(
                row,
                text="(Required - Ollama is not installed)",
                font=("Segoe UI", 9),
                foreground="red"
            ).pack(anchor=tk.W, padx=(25, 0))

        # Package installation
        row = ttk.Frame(options_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            row,
            text="Install Memory Palace package",
            variable=self.install_package
        ).pack(anchor=tk.W)
        ttk.Label(
            row,
            text="(Core package for MCP server)",
            font=("Segoe UI", 9),
            foreground="gray"
        ).pack(anchor=tk.W, padx=(25, 0))

        # Embedding model
        embedding_installed = self.dependency_results.get("embedding", (False, ""))[0]
        if not embedding_installed:
            self.install_embedding.set(True)
        row = ttk.Frame(options_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            row,
            text="Download embedding model",
            variable=self.install_embedding
        ).pack(anchor=tk.W)
        embedding_desc = self.model_recommendations.get("embedding_desc", "Unknown")
        ttk.Label(
            row,
            text=f"({embedding_desc})",
            font=("Segoe UI", 9),
            foreground="gray" if embedding_installed else "blue"
        ).pack(anchor=tk.W, padx=(25, 0))

        # LLM model (optional)
        llm_installed = self.dependency_results.get("llm", (False, ""))[0]
        row = ttk.Frame(options_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            row,
            text="Download LLM model (optional)",
            variable=self.install_llm
        ).pack(anchor=tk.W)
        llm_desc = self.model_recommendations.get("llm_desc", "Unknown")
        ttk.Label(
            row,
            text=f"({llm_desc})",
            font=("Segoe UI", 9),
            foreground="gray" if llm_installed else "blue"
        ).pack(anchor=tk.W, padx=(25, 0))
        ttk.Label(
            row,
            text="For memory extraction from transcripts - can be added later",
            font=("Segoe UI", 9, "italic"),
            foreground="gray"
        ).pack(anchor=tk.W, padx=(25, 0))

        # Configure Claude Desktop option
        row = ttk.Frame(options_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            row,
            text="Configure Claude Desktop automatically",
            variable=self.configure_claude
        ).pack(anchor=tk.W)
        ttk.Label(
            row,
            text="(Adds memory-palace to Claude Desktop's MCP servers)",
            font=("Segoe UI", 9),
            foreground="gray"
        ).pack(anchor=tk.W, padx=(25, 0))

        # VRAM info
        if self.vram_gb > 0:
            info_frame = ttk.Frame(self.container)
            info_frame.pack(fill=tk.X, padx=40, pady=20)
            ttk.Label(
                info_frame,
                text=f"Detected {self.vram_gb}GB VRAM - models selected for your GPU",
                font=("Segoe UI", 10),
                foreground="green"
            ).pack(anchor=tk.W)

        # Spacer
        ttk.Frame(self.container).pack(fill=tk.BOTH, expand=True)

        # Navigation
        nav_frame = ttk.Frame(self.container)
        nav_frame.pack(fill=tk.X, pady=20)

        ttk.Button(
            nav_frame,
            text="Back",
            command=self.show_dependency_check,
            width=15
        ).pack(side=tk.LEFT)

        ttk.Button(
            nav_frame,
            text="Cancel",
            command=self.root.destroy,
            width=15
        ).pack(side=tk.LEFT, padx=(10, 0))

        ttk.Button(
            nav_frame,
            text="Install Selected",
            command=self.show_progress,
            width=15
        ).pack(side=tk.RIGHT)

    def show_progress(self):
        """Screen 4: Progress screen."""
        self.clear_container()
        self.current_screen = 4

        # Title
        title = ttk.Label(
            self.container,
            text="Installing Components",
            font=("Segoe UI", 18, "bold")
        )
        title.pack(pady=(20, 30))

        # Progress bar
        self.progress = ttk.Progressbar(
            self.container,
            length=500,
            mode="determinate"
        )
        self.progress.pack(pady=20)

        # Status text
        self.status_text = tk.Text(
            self.container,
            height=12,
            width=65,
            font=("Consolas", 10),
            state=tk.DISABLED
        )
        self.status_text.pack(pady=20)

        # Spacer
        ttk.Frame(self.container).pack(fill=tk.BOTH, expand=True)

        # Start installation
        threading.Thread(target=self.run_installation, daemon=True).start()

    def log_status(self, message: str):
        """Add a message to the status text."""
        def update():
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.see(tk.END)
            self.status_text.config(state=tk.DISABLED)
        self.root.after(0, update)

    def set_progress(self, value: int):
        """Set progress bar value (0-100)."""
        self.root.after(0, lambda: self.progress.configure(value=value))

    def run_installation(self):
        """Run the installation process."""
        steps_total = sum([
            self.install_ollama.get(),
            self.install_package.get(),
            self.install_embedding.get(),
            self.install_llm.get(),
            self.configure_claude.get()
        ])
        step_size = 100 // max(steps_total, 1)
        current_progress = 0

        success = True

        try:
            # Step 1: Ollama
            if self.install_ollama.get():
                self.log_status("Opening Ollama download page...")
                webbrowser.open("https://ollama.com/download")
                self.log_status("Please install Ollama, then restart this installer.")
                current_progress += step_size
                self.set_progress(current_progress)

                # Show message and wait
                self.root.after(0, lambda: messagebox.showinfo(
                    "Install Ollama",
                    "The Ollama download page has been opened.\n\n"
                    "Please download and install Ollama, then restart this installer."
                ))
                self.root.after(0, self.root.destroy)
                return

            # Step 2: Create venv and install package
            if self.install_package.get():
                # Create venv
                venv_dir = Path(self.package_dir) / "venv"
                self.log_status(f"Creating virtual environment at {venv_dir}...")

                result = subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )

                if result.returncode != 0:
                    self.log_status(f"Warning: venv creation issue: {result.stderr[:200]}")
                else:
                    self.log_status("Virtual environment created.")

                # Determine venv pip path
                if sys.platform == "win32":
                    venv_pip = venv_dir / "Scripts" / "pip.exe"
                else:
                    venv_pip = venv_dir / "bin" / "pip"

                self.log_status("Installing Memory Palace package into venv...")

                result = subprocess.run(
                    [str(venv_pip), "install", "-e", self.package_dir],
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )

                if result.returncode == 0:
                    self.log_status("Package installed successfully.")
                else:
                    self.log_status(f"Warning: {result.stderr[:200]}")
                    success = False

                current_progress += step_size
                self.set_progress(current_progress)

            # Step 3: Embedding model
            if self.install_embedding.get():
                model = self.model_recommendations.get("embedding", "snowflake-arctic-embed:l")
                self.log_status(f"Downloading embedding model: {model}")
                self.log_status("This may take several minutes...")

                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minute timeout
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )

                if result.returncode == 0:
                    self.log_status(f"Embedding model {model} installed.")
                else:
                    self.log_status(f"Error downloading model: {result.stderr[:200]}")
                    success = False

                current_progress += step_size
                self.set_progress(current_progress)

            # Step 4: LLM model
            if self.install_llm.get():
                model = self.model_recommendations.get("llm", "qwen2.5:7b")
                self.log_status(f"Downloading LLM model: {model}")
                self.log_status("This may take several minutes...")

                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minute timeout
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )

                if result.returncode == 0:
                    self.log_status(f"LLM model {model} installed.")
                else:
                    self.log_status(f"Error downloading model: {result.stderr[:200]}")
                    success = False

                current_progress += step_size
                self.set_progress(current_progress)

            # Step 5: Configure Claude Desktop
            if self.configure_claude.get():
                self.log_status("Configuring Claude Desktop...")
                try:
                    from setup.configure_claude import configure_claude_desktop
                    self.configure_result = configure_claude_desktop(
                        interactive=False,
                        cwd_override=self.package_dir
                    )
                    if self.configure_result.success:
                        self.log_status("Claude Desktop configured successfully!")
                        if self.configure_result.backup_path:
                            self.log_status(f"  (Backup: {self.configure_result.backup_path})")
                    else:
                        self.log_status(f"Configure warning: {self.configure_result.message}")
                except ImportError as e:
                    self.log_status(f"Could not import configure module: {e}")
                    self.configure_result = None
                except Exception as e:
                    self.log_status(f"Configure error: {str(e)[:100]}")
                    self.configure_result = None

                current_progress += step_size
                self.set_progress(current_progress)

            self.set_progress(100)
            self.log_status("")
            self.log_status("Installation complete!")

        except Exception as e:
            self.log_status(f"Error: {str(e)}")
            success = False

        # Show completion screen after a short delay
        self.root.after(1500, lambda: self.show_complete(success))

    def show_complete(self, success: bool):
        """Screen 5: Completion screen."""
        self.clear_container()
        self.current_screen = 5

        # Check if auto-configure succeeded
        auto_configured = (
            self.configure_result is not None and 
            self.configure_result.success and
            "configured successfully" in self.configure_result.message
        )

        if success:
            # Title
            title = ttk.Label(
                self.container,
                text="Installation Complete",
                font=("Segoe UI", 18, "bold"),
                foreground="green"
            )
            title.pack(pady=(20, 20))

            # Success message
            msg_frame = ttk.Frame(self.container)
            msg_frame.pack(fill=tk.X, padx=40, pady=20)

            ttk.Label(
                msg_frame,
                text="Claude Memory Palace has been installed successfully!",
                font=("Segoe UI", 12),
                foreground="green"
            ).pack(anchor=tk.W)

            if auto_configured:
                # Auto-configure succeeded - simplified instructions
                ttk.Label(
                    msg_frame,
                    text="Claude Desktop has been configured automatically!",
                    font=("Segoe UI", 11),
                    foreground="green"
                ).pack(anchor=tk.W, pady=(10, 0))

                instructions_frame = ttk.LabelFrame(
                    self.container,
                    text="Final Step",
                    padding="15"
                )
                instructions_frame.pack(fill=tk.X, padx=40, pady=10)

                instructions = """Please restart Claude Desktop to activate Memory Palace.

After restarting, test by asking Claude:

  "Remember that I like coffee"

Then in a new conversation:

  "What do I like to drink?"

Claude should remember your preference!
"""

                instructions_text = tk.Text(
                    instructions_frame,
                    height=10,
                    width=60,
                    font=("Segoe UI", 10),
                    wrap=tk.WORD
                )
                instructions_text.insert(tk.END, instructions)
                instructions_text.config(state=tk.DISABLED)
                instructions_text.pack(fill=tk.X)

            else:
                # Manual configuration needed - show venv python path
                venv_python = Path(self.package_dir) / "venv" / "Scripts" / "python.exe"
                instructions_frame = ttk.LabelFrame(
                    self.container,
                    text="Next Steps: Configure Claude Desktop",
                    padding="15"
                )
                instructions_frame.pack(fill=tk.X, padx=40, pady=10)

                instructions = f"""1. Open Claude Desktop settings (Settings > Developer > MCP Servers)

2. Add the following configuration:

{{
  "mcpServers": {{
    "memory-palace": {{
      "command": "{venv_python}",
      "args": ["-m", "mcp_server.server"],
      "cwd": "{self.package_dir}"
    }}
  }}
}}

3. Restart Claude Desktop

4. Test by asking Claude: "Remember that I like coffee"
   Then in a new conversation: "What do I like to drink?"
"""

                instructions_text = tk.Text(
                    instructions_frame,
                    height=14,
                    width=60,
                    font=("Consolas", 9),
                    wrap=tk.WORD
                )
                instructions_text.insert(tk.END, instructions)
                instructions_text.config(state=tk.DISABLED)
                instructions_text.pack(fill=tk.X)

                # Copy button
                def copy_config():
                    config = f"""{{
  "mcpServers": {{
    "memory-palace": {{
      "command": "{venv_python}",
      "args": ["-m", "mcp_server.server"],
      "cwd": "{self.package_dir}"
    }}
  }}
}}"""
                    self.root.clipboard_clear()
                    self.root.clipboard_append(config)
                    messagebox.showinfo("Copied", "Configuration copied to clipboard!")

                ttk.Button(
                    instructions_frame,
                    text="Copy Config to Clipboard",
                    command=copy_config
                ).pack(pady=(10, 0))

        else:
            # Error title
            title = ttk.Label(
                self.container,
                text="Installation Had Errors",
                font=("Segoe UI", 18, "bold"),
                foreground="orange"
            )
            title.pack(pady=(20, 20))

            ttk.Label(
                self.container,
                text="Some components may not have installed correctly.\n"
                     "Please check the log above for details.",
                font=("Segoe UI", 11),
                justify=tk.CENTER
            ).pack(pady=20)

        # Spacer
        ttk.Frame(self.container).pack(fill=tk.BOTH, expand=True)

        # Close button
        nav_frame = ttk.Frame(self.container)
        nav_frame.pack(fill=tk.X, pady=20)

        ttk.Button(
            nav_frame,
            text="Close",
            command=self.root.destroy,
            width=15
        ).pack(side=tk.RIGHT)

    def run(self):
        """Run the installer."""
        self.root.mainloop()


def main():
    """Entry point."""
    app = InstallerGUI()
    app.run()


if __name__ == "__main__":
    main()
