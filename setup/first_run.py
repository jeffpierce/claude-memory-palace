"""Interactive first-time setup wizard for Memory Palace."""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

from .detect_gpu import detect_gpu, get_gpu_info_detailed
from .model_recommendations import (
    get_recommended_models,
    get_model_details,
    EMBEDDING_MODELS,
    LLM_MODELS,
)


def check_ollama_installed() -> Tuple[bool, Optional[str]]:
    """
    Check if Ollama is installed and get version.

    Returns:
        Tuple of (is_installed, version_string)
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return True, version
        return False, None
    except FileNotFoundError:
        return False, None
    except subprocess.TimeoutExpired:
        return False, None


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except:
        return False


def get_installed_models() -> list:
    """Get list of currently installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return []

        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:  # Just header or empty
            return []

        models = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except:
        return []


def pull_model(model_name: str) -> bool:
    """
    Pull an Ollama model.

    Args:
        model_name: Name of the model to pull

    Returns:
        True if successful, False otherwise
    """
    print(f"\nDownloading {model_name}...")
    print("(This may take a while depending on your connection)")

    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            timeout=3600  # 1 hour timeout for large models
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Download timed out for {model_name}")
        return False
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        return False


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt user for yes/no response."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        response = input(prompt + suffix).strip().lower()
        if response == "":
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'")


def print_header(text: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60 + "\n")


def run_setup_wizard() -> dict:
    """
    Run the interactive setup wizard.

    Returns:
        Dictionary with setup results and configuration
    """
    results = {
        "gpu": None,
        "vram_gb": None,
        "ollama_installed": False,
        "ollama_version": None,
        "embed_model": None,
        "llm_model": None,
        "models_downloaded": [],
        "success": False,
    }

    print_header("Memory Palace - First Time Setup")

    print("This wizard will help you configure Memory Palace")
    print("by detecting your hardware and setting up the required models.\n")

    # Step 1: GPU Detection
    print_header("Step 1: GPU Detection")

    gpu_info = detect_gpu()
    if gpu_info:
        gpu_name, vram_gb = gpu_info
        results["gpu"] = gpu_name
        results["vram_gb"] = vram_gb
        print(f"Detected: {gpu_name}")
        print(f"VRAM: {vram_gb} GB")

        detailed = get_gpu_info_detailed()
        if "vram_free_gb" in detailed:
            print(f"Free VRAM: {detailed['vram_free_gb']} GB")
    else:
        print("No NVIDIA GPU detected.")
        print("\nMemory Palace requires an NVIDIA GPU with CUDA support")
        print("for running local embedding and LLM models.")
        print("\nYou can still use the system with CPU-only inference,")
        print("but performance will be significantly slower.")

        if not prompt_yes_no("Continue with CPU-only setup?", default=False):
            print("\nSetup cancelled. Please install NVIDIA drivers or use a system with a GPU.")
            return results

        # Default to minimal models for CPU
        vram_gb = 4.0
        results["vram_gb"] = vram_gb
        print("\nUsing minimal model configuration for CPU inference.")

    # Step 2: Model Recommendations
    print_header("Step 2: Model Selection")

    details = get_model_details(results["vram_gb"])
    embed_model = details["embedding"]["ollama_name"]
    llm_model = details["llm"]["ollama_name"]

    print(f"Based on your hardware ({results['vram_gb']}GB VRAM), we recommend:\n")
    print(f"Embedding Model: {embed_model}")
    print(f"  - {details['embedding']['description']}")
    print(f"  - Size: {details['embedding']['size_gb']} GB\n")
    print(f"LLM Model: {llm_model}")
    print(f"  - {details['llm']['description']}")
    print(f"  - Size: {details['llm']['size_gb']} GB\n")
    print(f"Note: {details['notes']}")

    if not prompt_yes_no("\nUse recommended models?", default=True):
        print("\nYou can manually configure models later in the config file.")
        print("See docs/models.md for available options.")
        return results

    results["embed_model"] = embed_model
    results["llm_model"] = llm_model

    # Step 3: Ollama Check
    print_header("Step 3: Ollama Installation")

    installed, version = check_ollama_installed()
    results["ollama_installed"] = installed
    results["ollama_version"] = version

    if installed:
        print(f"Ollama is installed: {version}")

        if not check_ollama_running():
            print("\nOllama server is not running.")
            print("Please start Ollama before downloading models.")
            print("\nOn Windows: Run 'ollama serve' in a terminal")
            print("On macOS/Linux: Ollama should auto-start, or run 'ollama serve'")

            if not prompt_yes_no("Is Ollama running now?", default=False):
                print("\nPlease start Ollama and re-run setup.")
                return results
    else:
        print("Ollama is not installed.\n")
        print("Please install Ollama from: https://ollama.ai/download")
        print("\nAfter installing, start Ollama and re-run this setup.")
        return results

    # Step 4: Download Models
    print_header("Step 4: Download Models")

    installed_models = get_installed_models()
    print(f"Currently installed models: {', '.join(installed_models) if installed_models else 'None'}")

    models_to_download = []

    if embed_model not in installed_models:
        models_to_download.append(("embedding", embed_model, details["embedding"]["size_gb"]))
    else:
        print(f"\nEmbedding model {embed_model} is already installed.")
        results["models_downloaded"].append(embed_model)

    if llm_model not in installed_models:
        models_to_download.append(("llm", llm_model, details["llm"]["size_gb"]))
    else:
        print(f"LLM model {llm_model} is already installed.")
        results["models_downloaded"].append(llm_model)

    if models_to_download:
        total_size = sum(m[2] for m in models_to_download)
        print(f"\nModels to download: {len(models_to_download)}")
        print(f"Total download size: ~{total_size:.1f} GB")

        for model_type, model_name, size in models_to_download:
            print(f"  - {model_name} ({size} GB)")

        if prompt_yes_no("\nDownload these models now?", default=True):
            for model_type, model_name, size in models_to_download:
                if pull_model(model_name):
                    print(f"Successfully downloaded {model_name}")
                    results["models_downloaded"].append(model_name)
                else:
                    print(f"Failed to download {model_name}")
                    print("You can manually download it later with: ollama pull " + model_name)
        else:
            print("\nYou can download models later with:")
            for _, model_name, _ in models_to_download:
                print(f"  ollama pull {model_name}")

    # Step 5: Configuration Summary
    print_header("Setup Complete")

    if len(results["models_downloaded"]) == 2:
        results["success"] = True
        print("Memory Palace is ready to use!\n")
        print("Configuration:")
        print(f"  Embedding Model: {embed_model}")
        print(f"  LLM Model: {llm_model}")
        print(f"  GPU: {results['gpu'] or 'CPU-only'}")
        print("\nNext steps:")
        print("1. Configure MCP in Claude Desktop (see docs/README.md)")
        print("2. Start using memory tools in your AI conversations")
    else:
        print("Setup partially complete.\n")
        print("Please download the remaining models and re-run setup to verify.")

    return results


if __name__ == "__main__":
    results = run_setup_wizard()
    print("\n" + "=" * 60)
    print("Setup Results:")
    import json
    print(json.dumps(results, indent=2))
