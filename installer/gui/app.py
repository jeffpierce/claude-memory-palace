"""
Claude Memory Palace — Cross-Platform GUI Installer

Thin tkinter UI over the shared installer core.
Platform-agnostic: Windows, macOS, Linux (including Steam Deck).
"""

import sys
import os
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import List, Optional

# Add parent directory to path so shared modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.detect import detect_all, SystemInfo
from shared.clients import discover_clients, configure_clients, ClientInfo
from shared.models import get_model_recommendation, pull_model, check_model_installed, ModelRecommendation
from shared.install_core import (
    get_default_install_dir,
    find_python,
    create_venv,
    install_package,
    install_ollama,
    clone_or_update_repo,
    verify_installation,
    get_venv_python,
)


class InstallerApp:
    """Main installer GUI application."""

    # Screen indices
    WELCOME = 0
    DETECTING = 1
    CLIENTS = 2
    OPTIONS = 3
    PROGRESS = 4
    COMPLETE = 5

    def __init__(self, title: str = "Claude Memory Palace Setup"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("650x550")
        self.root.resizable(False, False)
        self._center_window()

        # State
        self.system_info: Optional[SystemInfo] = None
        self.clients: List[ClientInfo] = []
        self.model_rec: Optional[ModelRecommendation] = None
        self.install_dir = Path.home() / "memory-palace"

        # User selections
        self.selected_client_vars: dict[str, tk.BooleanVar] = {}
        self.install_llm_var = tk.BooleanVar(value=False)

        # Container
        self.container = ttk.Frame(self.root, padding="20")
        self.container.pack(fill=tk.BOTH, expand=True)

        # Start
        self._show_welcome()

    def _center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

    def _clear(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    # --- Navigation helpers ---

    def _nav_frame(self, back=None, next_cmd=None, next_label="Next", cancel=True):
        """Create a standard navigation bar at the bottom."""
        nav = ttk.Frame(self.container)
        nav.pack(fill=tk.X, side=tk.BOTTOM, pady=(20, 0))

        if back:
            ttk.Button(nav, text="Back", command=back, width=12).pack(side=tk.LEFT)

        if cancel:
            ttk.Button(nav, text="Cancel", command=self.root.destroy, width=12).pack(
                side=tk.LEFT, padx=(10, 0)
            )

        if next_cmd:
            btn = ttk.Button(nav, text=next_label, command=next_cmd, width=15)
            btn.pack(side=tk.RIGHT)
            return btn

        return None

    def _title(self, text: str, size: int = 20):
        ttk.Label(
            self.container, text=text, font=("Segoe UI", size, "bold")
        ).pack(pady=(10, 15))

    def _spacer(self):
        ttk.Frame(self.container).pack(fill=tk.BOTH, expand=True)

    # ========================================================================
    # Screen 1: Welcome
    # ========================================================================

    def _show_welcome(self):
        self._clear()
        self._title("Claude Memory Palace", size=24)

        desc = ttk.Frame(self.container)
        desc.pack(fill=tk.X, padx=30, pady=10)

        ttk.Label(
            desc,
            text=(
                "Welcome! This installer sets up a persistent memory system\n"
                "for your AI tools — Claude Desktop, Cursor, Windsurf, and more.\n\n"
                "Your memories stay local. Nothing leaves your machine unless\n"
                "you choose to share it.\n\n"
                "The only question we'll ask: which AI tools do you already use?\n"
                "We'll handle the rest."
            ),
            font=("Segoe UI", 11),
            justify=tk.LEFT,
            wraplength=550,
        ).pack(anchor=tk.W)

        self._spacer()
        self._nav_frame(next_cmd=self._show_detecting, next_label="Get Started")

    # ========================================================================
    # Screen 2: Detection (auto-advance)
    # ========================================================================

    def _show_detecting(self):
        self._clear()
        self._title("Checking Your System")

        # Status list
        self.detect_frame = ttk.LabelFrame(
            self.container, text="System Check", padding="15"
        )
        self.detect_frame.pack(fill=tk.X, padx=30, pady=10)

        self.detect_labels = {}
        checks = [
            ("platform", "Platform"),
            ("python", "Python"),
            ("ollama", "Ollama"),
            ("gpu", "GPU"),
            ("clients", "AI Clients"),
        ]
        for key, label in checks:
            row = ttk.Frame(self.detect_frame)
            row.pack(fill=tk.X, pady=3)

            status = ttk.Label(row, text="…", width=3, font=("Segoe UI", 11))
            status.pack(side=tk.LEFT)

            ttk.Label(row, text=label, font=("Segoe UI", 11), width=12, anchor=tk.W).pack(
                side=tk.LEFT, padx=(8, 0)
            )

            detail = ttk.Label(row, text="Checking...", font=("Segoe UI", 10), foreground="gray")
            detail.pack(side=tk.LEFT, padx=(8, 0))

            self.detect_labels[key] = (status, detail)

        self._spacer()

        # No nav buttons — auto-advance after detection
        threading.Thread(target=self._run_detection, daemon=True).start()

    def _update_detect(self, key: str, ok: bool, detail: str):
        if key in self.detect_labels:
            s, d = self.detect_labels[key]
            s.config(text="✓" if ok else "✗", foreground="green" if ok else "red")
            d.config(text=detail, foreground="black" if ok else "gray")

    def _run_detection(self):
        # System detection
        self.system_info = detect_all()
        si = self.system_info

        # Platform
        plat_str = si.platform.os.capitalize()
        if si.platform.is_wsl:
            plat_str = f"WSL{si.platform.wsl_version} (Linux on Windows)"
        if si.platform.is_steam_deck:
            plat_str = "Steam Deck 🎮"
        self.root.after(0, lambda: self._update_detect("platform", True, plat_str))

        # Python
        self.root.after(
            0,
            lambda: self._update_detect(
                "python",
                si.python.meets_minimum,
                f"Python {si.python.version}" if si.python.available else "Not found",
            ),
        )

        # Ollama
        if si.ollama.installed:
            detail = f"Ollama {si.ollama.version}" if si.ollama.version else "Installed"
            if si.ollama.models:
                detail += f" ({len(si.ollama.models)} models)"
        else:
            detail = "Not installed"
        self.root.after(
            0, lambda: self._update_detect("ollama", si.ollama.installed, detail)
        )

        # GPU
        self.root.after(
            0,
            lambda: self._update_detect(
                "gpu", si.gpu.available, si.gpu.detail or "None detected (CPU mode)"
            ),
        )

        # Model recommendations
        self.model_rec = get_model_recommendation(si.gpu)

        # AI Clients
        self.clients = discover_clients(si.platform)
        installed = [c for c in self.clients if c.installed]
        if installed:
            names = ", ".join(c.name for c in installed[:4])
            if len(installed) > 4:
                names += f" +{len(installed) - 4} more"
            client_detail = names
        else:
            client_detail = "None detected"
        self.root.after(
            0,
            lambda: self._update_detect("clients", len(installed) > 0, client_detail),
        )

        # Auto-advance after a brief pause
        self.root.after(800, self._post_detection)

    def _post_detection(self):
        si = self.system_info

        # Handle missing prerequisites
        if not si.python.meets_minimum:
            if messagebox.askyesno(
                "Python Required",
                "Python 3.10+ is required but not found.\n\n"
                "Would you like to install it now?",
            ):
                threading.Thread(
                    target=self._install_python_then_continue, daemon=True
                ).start()
                return
            else:
                messagebox.showerror(
                    "Cannot Continue",
                    "Please install Python 3.10+ and re-run this installer.",
                )
                self.root.destroy()
                return

        if not si.ollama.installed:
            if messagebox.askyesno(
                "Ollama Required",
                "Ollama is required for AI embeddings.\n\n"
                "Would you like to install it now?",
            ):
                threading.Thread(
                    target=self._install_ollama_then_continue, daemon=True
                ).start()
                return
            else:
                messagebox.showerror(
                    "Cannot Continue",
                    "Please install Ollama from https://ollama.com and re-run.",
                )
                self.root.destroy()
                return

        # All good — go to client selection
        self._show_clients()

    def _install_python_then_continue(self):
        result = install_ollama(self.system_info.platform)  # Placeholder — need Python install
        # For now, direct them
        self.root.after(
            0,
            lambda: messagebox.showinfo(
                "Install Python",
                "Please install Python 3.10+ from:\nhttps://www.python.org/downloads/\n\n"
                "Then re-run this installer.",
            ),
        )
        self.root.after(0, self.root.destroy)

    def _install_ollama_then_continue(self):
        def progress(msg):
            self.root.after(
                0,
                lambda: self._update_detect("ollama", False, msg),
            )

        result = install_ollama(self.system_info.platform, progress=progress)
        if result.success:
            self.system_info.ollama.installed = True
            self.root.after(
                0, lambda: self._update_detect("ollama", True, "Installed")
            )
            self.root.after(500, self._show_clients)
        else:
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Installation Failed",
                    f"{result.message}\n\n"
                    "Please install Ollama from https://ollama.com and re-run.",
                ),
            )

    # ========================================================================
    # Screen 3: Client Selection
    # ========================================================================

    def _show_clients(self):
        self._clear()
        self._title("Which AI tools do you use?")

        installed_clients = [c for c in self.clients if c.installed]
        not_installed = [c for c in self.clients if not c.installed]

        if installed_clients:
            found_frame = ttk.LabelFrame(
                self.container, text="Detected on your system", padding="10"
            )
            found_frame.pack(fill=tk.X, padx=30, pady=(5, 10))

            for client in installed_clients:
                var = tk.BooleanVar(value=not client.already_configured)
                self.selected_client_vars[client.id] = var

                row = ttk.Frame(found_frame)
                row.pack(fill=tk.X, pady=2)

                cb = ttk.Checkbutton(row, text=client.name, variable=var)
                cb.pack(side=tk.LEFT)

                if client.already_configured:
                    ttk.Label(
                        row,
                        text="(already configured)",
                        font=("Segoe UI", 9),
                        foreground="green",
                    ).pack(side=tk.LEFT, padx=(8, 0))
                else:
                    ttk.Label(
                        row,
                        text=client.description,
                        font=("Segoe UI", 9),
                        foreground="gray",
                    ).pack(side=tk.LEFT, padx=(8, 0))

        # WSL notice
        if self.system_info.platform.is_wsl:
            win_clients = [c for c in installed_clients if c.id.endswith("-windows")]
            if win_clients:
                notice = ttk.Frame(self.container)
                notice.pack(fill=tk.X, padx=30, pady=5)
                ttk.Label(
                    notice,
                    text="💡 Noticed you're in WSL — Windows-side clients are listed above.",
                    font=("Segoe UI", 10),
                    foreground="blue",
                    wraplength=550,
                ).pack(anchor=tk.W)

        # Not-detected section (collapsed, for manual selection)
        if not_installed:
            other_frame = ttk.LabelFrame(
                self.container, text="Not detected (select manually if installed)", padding="10"
            )
            other_frame.pack(fill=tk.X, padx=30, pady=(5, 10))

            for client in not_installed:
                var = tk.BooleanVar(value=False)
                self.selected_client_vars[client.id] = var

                row = ttk.Frame(other_frame)
                row.pack(fill=tk.X, pady=2)

                ttk.Checkbutton(row, text=client.name, variable=var).pack(side=tk.LEFT)
                ttk.Label(
                    row,
                    text=client.description,
                    font=("Segoe UI", 9),
                    foreground="gray",
                ).pack(side=tk.LEFT, padx=(8, 0))

        if not installed_clients and not not_installed:
            ttk.Label(
                self.container,
                text="No MCP-compatible clients detected.\nYou can configure manually after installation.",
                font=("Segoe UI", 11),
                justify=tk.CENTER,
            ).pack(pady=20)

        self._spacer()
        self._nav_frame(
            back=self._show_detecting,
            next_cmd=self._show_options,
        )

    # ========================================================================
    # Screen 4: Options (models, install dir)
    # ========================================================================

    def _show_options(self):
        self._clear()
        self._title("Installation Options")

        opts_frame = ttk.LabelFrame(self.container, text="Components", padding="15")
        opts_frame.pack(fill=tk.X, padx=30, pady=10)

        # Embedding model (always)
        row = ttk.Frame(opts_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="✓", foreground="green", font=("Segoe UI", 11)).pack(side=tk.LEFT)
        ttk.Label(row, text="Embedding model", font=("Segoe UI", 11)).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Label(
            row,
            text=f"{self.model_rec.embedding_desc} ({self.model_rec.embedding_size})",
            font=("Segoe UI", 9),
            foreground="gray",
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Base LLM — always installed (qwen3:1.7b handles synthesis + classification)
        row = ttk.Frame(opts_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="✓", foreground="green", font=("Segoe UI", 11)).pack(side=tk.LEFT)
        ttk.Label(row, text="Base LLM (qwen3:1.7b)", font=("Segoe UI", 11)).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Label(
            row,
            text="Synthesis, classification, extraction (~1GB)",
            font=("Segoe UI", 9),
            foreground="gray",
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Upgraded LLM model (optional — only shown if hardware supports something bigger)
        if self.model_rec.llm_model and self.model_rec.llm_model != "qwen3:1.7b":
            row = ttk.Frame(opts_frame)
            row.pack(fill=tk.X, pady=3)
            self.install_llm_var.set(False)
            ttk.Checkbutton(
                row, text="Upgraded LLM (optional)", variable=self.install_llm_var
            ).pack(side=tk.LEFT)
            ttk.Label(
                row,
                text=f"{self.model_rec.llm_desc} ({self.model_rec.llm_size})",
                font=("Segoe UI", 9),
                foreground="gray",
            ).pack(side=tk.LEFT, padx=(8, 0))

            ttk.Label(
                opts_frame,
                text="Higher quality extraction and synthesis. Can be added later.",
                font=("Segoe UI", 9, "italic"),
                foreground="gray",
                wraplength=520,
            ).pack(anchor=tk.W, padx=(25, 0), pady=(0, 5))

        # Install directory
        dir_frame = ttk.LabelFrame(self.container, text="Install Location", padding="10")
        dir_frame.pack(fill=tk.X, padx=30, pady=10)

        ttk.Label(
            dir_frame,
            text=str(self.install_dir),
            font=("Consolas", 10),
        ).pack(anchor=tk.W)

        # GPU info
        if self.system_info.gpu.available:
            info_frame = ttk.Frame(self.container)
            info_frame.pack(fill=tk.X, padx=30, pady=5)
            ttk.Label(
                info_frame,
                text=f"🖥️ {self.system_info.gpu.detail} — models selected for your hardware",
                font=("Segoe UI", 10),
                foreground="green",
            ).pack(anchor=tk.W)
        else:
            info_frame = ttk.Frame(self.container)
            info_frame.pack(fill=tk.X, padx=30, pady=5)
            ttk.Label(
                info_frame,
                text="💻 CPU mode — embeddings work great, LLM is optional",
                font=("Segoe UI", 10),
                foreground="gray",
            ).pack(anchor=tk.W)

        # Summary of selected clients
        selected = [
            cid for cid, var in self.selected_client_vars.items() if var.get()
        ]
        if selected:
            client_names = []
            for c in self.clients:
                if c.id in selected:
                    client_names.append(c.name)
            summary_frame = ttk.Frame(self.container)
            summary_frame.pack(fill=tk.X, padx=30, pady=5)
            ttk.Label(
                summary_frame,
                text=f"Will configure: {', '.join(client_names)}",
                font=("Segoe UI", 10),
                foreground="blue",
            ).pack(anchor=tk.W)

        self._spacer()
        self._nav_frame(
            back=self._show_clients,
            next_cmd=self._show_progress,
            next_label="Install",
        )

    # ========================================================================
    # Screen 5: Progress
    # ========================================================================

    def _show_progress(self):
        self._clear()
        self._title("Installing...")

        self.progress_bar = ttk.Progressbar(
            self.container, length=550, mode="determinate"
        )
        self.progress_bar.pack(padx=30, pady=(10, 15))

        self.log_text = tk.Text(
            self.container,
            height=14,
            width=72,
            font=("Consolas", 10),
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.log_text.pack(padx=30, pady=5)

        self._spacer()
        # No nav during install

        threading.Thread(target=self._run_install, daemon=True).start()

    def _log(self, msg: str):
        def update():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(0, update)

    def _set_progress(self, val: int):
        self.root.after(0, lambda: self.progress_bar.configure(value=val))

    def _run_install(self):
        steps = 6  # download, venv, package, embedding, base LLM, configure
        if (self.install_llm_var.get() and self.model_rec.llm_model
                and self.model_rec.llm_model != "qwen3:1.7b"):
            steps += 1  # upgraded LLM
        step_size = 100 // steps
        progress = 0
        success = True

        try:
            # 1. Download / clone repo
            self._log("📥 Downloading Memory Palace...")
            result = clone_or_update_repo(
                self.install_dir,
                progress=lambda m: self._log(f"  {m}"),
            )
            if not result.success:
                self._log(f"  ✗ {result.message}")
                if result.detail:
                    self._log(f"    {result.detail}")
                success = False
                self._finish_install(False)
                return
            progress += step_size
            self._set_progress(progress)

            # 2. Create venv
            self._log("🐍 Creating Python environment...")
            result = create_venv(
                self.install_dir,
                progress=lambda m: self._log(f"  {m}"),
            )
            if not result.success:
                self._log(f"  ✗ {result.message}")
                success = False
                self._finish_install(False)
                return
            progress += step_size
            self._set_progress(progress)

            # 3. Install package
            self._log("📦 Installing Memory Palace package...")
            result = install_package(
                self.install_dir,
                progress=lambda m: self._log(f"  {m}"),
            )
            if not result.success:
                self._log(f"  ✗ {result.message}")
                success = False
                self._finish_install(False)
                return
            progress += step_size
            self._set_progress(progress)

            # 4. Pull embedding model
            self._log(f"🧠 Downloading embedding model ({self.model_rec.embedding_model})...")
            if check_model_installed(self.model_rec.embedding_model):
                self._log("  ✓ Already installed")
            else:
                self._log(f"  This is {self.model_rec.embedding_size} — should be quick")
                ok = pull_model(
                    self.model_rec.embedding_model,
                    progress_callback=lambda m: self._log(f"  {m}"),
                )
                if not ok:
                    self._log("  ⚠ Embedding model download failed — you can retry later")
                    # Non-fatal: continue
            progress += step_size
            self._set_progress(progress)

            # 5. Base LLM model (always — handles synthesis + classification)
            base_llm = "qwen3:1.7b"
            self._log(f"🧠 Downloading base LLM ({base_llm})...")
            if check_model_installed(base_llm):
                self._log("  ✓ Already installed")
            else:
                self._log("  ~1GB — synthesis, classification, extraction")
                ok = pull_model(
                    base_llm,
                    progress_callback=lambda m: self._log(f"  {m}"),
                )
                if not ok:
                    self._log("  ⚠ Base LLM download failed — retry later: ollama pull qwen3:1.7b")
            progress += step_size
            self._set_progress(progress)

            # 6. Optional upgraded LLM model
            if (self.install_llm_var.get() and self.model_rec.llm_model
                    and self.model_rec.llm_model != base_llm):
                self._log(f"🧠 Downloading upgraded LLM ({self.model_rec.llm_model})...")
                if check_model_installed(self.model_rec.llm_model):
                    self._log("  ✓ Already installed")
                else:
                    self._log(f"  This is {self.model_rec.llm_size} — may take a while")
                    ok = pull_model(
                        self.model_rec.llm_model,
                        progress_callback=lambda m: self._log(f"  {m}"),
                    )
                    if not ok:
                        self._log("  ⚠ Upgraded LLM download failed — you can add it later")
                progress += step_size
                self._set_progress(progress)

            # 6. Configure clients
            selected_ids = [
                cid for cid, var in self.selected_client_vars.items() if var.get()
            ]
            if selected_ids:
                self._log("⚙️  Configuring AI clients...")
                results = configure_clients(
                    self.clients,
                    selected_ids,
                    self.install_dir,
                    self.system_info.platform,
                )
                for r in results:
                    if r.success:
                        self._log(f"  ✓ {r.message}")
                        if r.backup_path:
                            self._log(f"    Backup: {r.backup_path.name}")
                    else:
                        self._log(f"  ⚠ {r.message}")
            progress += step_size
            self._set_progress(progress)

            # 7. Verify
            self._log("🔍 Verifying installation...")
            result = verify_installation(
                self.install_dir,
                progress=lambda m: self._log(f"  {m}"),
            )
            if not result.success:
                self._log(f"  ⚠ {result.message} (non-fatal)")

            self._set_progress(100)
            self._log("")
            self._log("✅ Installation complete!")

        except Exception as e:
            self._log(f"❌ Unexpected error: {str(e)}")
            success = False

        self.root.after(1000, lambda: self._show_complete(success))

    def _finish_install(self, success: bool):
        self._set_progress(100)
        self.root.after(1000, lambda: self._show_complete(success))

    # ========================================================================
    # Screen 6: Complete
    # ========================================================================

    def _show_complete(self, success: bool):
        self._clear()

        selected_ids = [
            cid for cid, var in self.selected_client_vars.items() if var.get()
        ]
        configured_names = [
            c.name for c in self.clients if c.id in selected_ids
        ]

        if success:
            self._title("🎉 Memory Palace is Ready!")

            info_frame = ttk.Frame(self.container)
            info_frame.pack(fill=tk.X, padx=30, pady=10)

            ttk.Label(
                info_frame,
                text=f"Installed to: {self.install_dir}",
                font=("Segoe UI", 10),
                foreground="green",
            ).pack(anchor=tk.W)

            if configured_names:
                ttk.Label(
                    info_frame,
                    text=f"Configured: {', '.join(configured_names)}",
                    font=("Segoe UI", 10),
                    foreground="green",
                ).pack(anchor=tk.W, pady=(5, 0))

            # Next steps
            steps_frame = ttk.LabelFrame(
                self.container, text="Next Steps", padding="15"
            )
            steps_frame.pack(fill=tk.X, padx=30, pady=15)

            if configured_names:
                steps_text = (
                    "1. Restart your AI client(s) to activate Memory Palace\n\n"
                    "2. Test it out:\n"
                    '   Tell Claude: "Remember that I like coffee"\n'
                    '   In a new chat: "What do I like to drink?"\n\n'
                    "   Claude should remember! 🧠"
                )
            else:
                python_path = get_venv_python(self.install_dir)
                steps_text = (
                    "1. Add Memory Palace to your MCP client's config:\n\n"
                    '   "memory-palace": {\n'
                    f'     "command": "{python_path}",\n'
                    '     "args": ["-m", "mcp_server.server"],\n'
                    f'     "cwd": "{self.install_dir}"\n'
                    "   }\n\n"
                    "2. Restart your AI client\n\n"
                    "3. Test: Tell Claude to remember something!"
                )

            text = tk.Text(
                steps_frame,
                height=10,
                width=65,
                font=("Segoe UI", 10),
                wrap=tk.WORD,
                borderwidth=0,
                background=steps_frame.cget("background") if hasattr(steps_frame, "cget") else "white",
            )
            text.insert(tk.END, steps_text)
            text.config(state=tk.DISABLED)
            text.pack(fill=tk.X)

        else:
            self._title("Installation Had Issues")

            ttk.Label(
                self.container,
                text=(
                    "Some steps may not have completed successfully.\n"
                    "Check the log above for details.\n\n"
                    "You can re-run this installer to retry."
                ),
                font=("Segoe UI", 11),
                justify=tk.CENTER,
                foreground="orange",
            ).pack(pady=20)

        self._spacer()

        nav = ttk.Frame(self.container)
        nav.pack(fill=tk.X, side=tk.BOTTOM, pady=(20, 0))

        ttk.Button(nav, text="Close", command=self.root.destroy, width=12).pack(
            side=tk.RIGHT
        )

    # ========================================================================
    # Run
    # ========================================================================

    def run(self):
        self.root.mainloop()


def main():
    """Entry point for the GUI installer."""
    app = InstallerApp()
    app.run()


if __name__ == "__main__":
    main()
