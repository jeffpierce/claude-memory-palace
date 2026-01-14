"""GPU detection utilities for Claude Memory Palace."""

import subprocess
import re
from typing import Optional, Tuple


def detect_gpu() -> Optional[Tuple[str, float]]:
    """
    Detect NVIDIA GPU using nvidia-smi.

    Returns:
        Tuple of (gpu_name, vram_gb) if NVIDIA GPU found, None otherwise.

    Examples:
        >>> detect_gpu()
        ('NVIDIA GeForce RTX 5070 Ti', 16.0)

        >>> detect_gpu()  # No NVIDIA GPU
        None
    """
    try:
        # Run nvidia-smi to get GPU info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return None

        output = result.stdout.strip()
        if not output:
            return None

        # Parse first GPU (in case of multi-GPU systems)
        first_line = output.split("\n")[0]
        parts = first_line.split(", ")

        if len(parts) != 2:
            return None

        gpu_name = parts[0].strip()
        vram_mb = float(parts[1].strip())
        vram_gb = vram_mb / 1024.0

        return (gpu_name, round(vram_gb, 1))

    except FileNotFoundError:
        # nvidia-smi not found - no NVIDIA GPU or drivers not installed
        return None
    except subprocess.TimeoutExpired:
        return None
    except (ValueError, IndexError):
        return None


def get_gpu_info_detailed() -> dict:
    """
    Get detailed GPU information for diagnostics.

    Returns:
        Dictionary with GPU details or error information.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,driver_version,cuda_version",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return {"error": "nvidia-smi failed", "stderr": result.stderr}

        output = result.stdout.strip()
        if not output:
            return {"error": "No GPU detected"}

        first_line = output.split("\n")[0]
        parts = [p.strip() for p in first_line.split(", ")]

        if len(parts) >= 6:
            return {
                "name": parts[0],
                "vram_total_mb": float(parts[1]),
                "vram_free_mb": float(parts[2]),
                "vram_used_mb": float(parts[3]),
                "driver_version": parts[4],
                "cuda_version": parts[5] if parts[5] != "[N/A]" else None,
                "vram_total_gb": round(float(parts[1]) / 1024, 1),
                "vram_free_gb": round(float(parts[2]) / 1024, 1),
            }
        return {"error": "Unexpected nvidia-smi output format"}

    except FileNotFoundError:
        return {"error": "nvidia-smi not found - NVIDIA drivers may not be installed"}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Quick test
    gpu = detect_gpu()
    if gpu:
        print(f"Detected GPU: {gpu[0]} with {gpu[1]} GB VRAM")
    else:
        print("No NVIDIA GPU detected")

    print("\nDetailed info:")
    import json
    print(json.dumps(get_gpu_info_detailed(), indent=2))
