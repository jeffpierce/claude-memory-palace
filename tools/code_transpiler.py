#!/usr/bin/env python3
"""
Code to Prose Transpiler

Converts source code files to prose descriptions optimized for semantic search.
The prose captures WHAT the code does and WHY, enabling natural language queries
to find relevant code via embedding similarity.

Part of the Code Retrieval Layer for Claude Memory Palace.

Usage:
    python code_transpiler.py path/to/file.py
    python code_transpiler.py path/to/file.py --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_palace.llm import generate_with_llm, is_llm_available, get_active_llm_model


# Maximum file size: 120k chars (~30k tokens for Qwen3's 32k context)
MAX_FILE_CHARS = 120_000

# Preferred models for code transpilation
# Order matters - first available will be used
# qwen3:14b is reliable for JSON output; mistral escapes underscores as markdown
TRANSPILER_PREFERRED_MODELS = [
    "qwen3:14b",
    "qwen3:8b",
    "qwen3:4b",
    "mistral:7b",
    "llama3.2",
]


def _get_transpiler_model() -> str:
    """
    Get the best available model for code transpilation.
    
    Code analysis needs more capability than simple text generation,
    so we prefer larger models even if a smaller one is configured.
    Falls back to the default configured model if none preferred are available.
    """
    import requests
    from memory_palace.config import get_ollama_url
    
    try:
        response = requests.get(f"{get_ollama_url()}/api/tags", timeout=5)
        response.raise_for_status()
        available = {m.get("name", "") for m in response.json().get("models", [])}
        
        for preferred in TRANSPILER_PREFERRED_MODELS:
            if preferred in available:
                return preferred
            # Check without tag suffix
            base = preferred.split(":")[0]
            for avail in available:
                if avail.startswith(base + ":"):
                    return avail
    except:
        pass
    
    # Fall back to configured model
    return get_active_llm_model()


def transpile_code_to_prose(code_content: str, file_path: str) -> Dict[str, Any]:
    """
    Convert source code to a prose description for semantic search.
    
    Args:
        code_content: The actual source code content
        file_path: Path to the file (used for context, language detection)
    
    Returns:
        Dictionary with:
            subject: 2-5 word topic
            prose: 3-5 paragraph description
            language: detected programming language
            dependencies: list of imports/requires
            patterns: design patterns identified
        
        Or on error:
            error: error message
    """
    # Validate file size
    if len(code_content) > MAX_FILE_CHARS:
        return {
            "error": f"File too large: {len(code_content):,} chars exceeds {MAX_FILE_CHARS:,} char limit (~30k tokens)"
        }
    
    if not code_content.strip():
        return {"error": "Empty file"}
    
    # Check LLM availability
    if not is_llm_available():
        return {"error": "LLM unavailable (Ollama not running?)"}
    
    # Extract file extension for language hint
    ext = Path(file_path).suffix.lower()
    
    system = """You are a code documentation expert. Your job is to write prose descriptions of source code that will be used for semantic search.

OUTPUT FORMAT - You MUST output valid JSON with exactly these fields:
{
  "subject": "2-5 word topic describing the file's main purpose",
  "prose": "3-5 paragraphs describing what the code does",
  "language": "programming language",
  "dependencies": ["list", "of", "imports"],
  "patterns": ["design", "patterns", "used"]
}

PROSE WRITING RULES:
- Write for someone searching with natural language (e.g., "how do embeddings work")
- Include function and class names so exact name searches work
- Describe WHAT the code does and WHY, not line-by-line HOW
- Make it standalone - the reader has NO CONTEXT about the codebase
- Mention error handling, edge cases, and important behaviors
- If the code has configuration or important constants, mention them

DEPENDENCIES:
- List actual imports/requires from the code
- Include standard library AND third-party dependencies

PATTERNS:
- Identify design patterns (singleton, factory, repository, etc.)
- Note architectural patterns (service layer, MVC, etc.)
- Include algorithmic patterns if relevant (memoization, retry logic, etc.)

Output ONLY the JSON object, no markdown fences, no explanation."""

    prompt = f"""Analyze this source code and generate a prose description for semantic search.

File: {file_path}
Extension: {ext}

---CODE START---
{code_content}
---CODE END---

Output the JSON object:"""

    # Use larger model for code analysis (needs more capability)
    model = _get_transpiler_model()
    response = generate_with_llm(prompt, system=system, model=model)
    
    if not response:
        return {"error": "LLM generation failed (no response)"}
    
    # Parse JSON response
    try:
        # Clean up response - remove markdown fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove opening fence
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        
        result = json.loads(cleaned)
        
        # Validate required fields
        required_fields = ["subject", "prose", "language", "dependencies", "patterns"]
        missing = [f for f in required_fields if f not in result]
        if missing:
            return {"error": f"LLM response missing fields: {missing}", "raw_response": response}
        
        # Ensure lists are actually lists
        if not isinstance(result.get("dependencies"), list):
            result["dependencies"] = []
        if not isinstance(result.get("patterns"), list):
            result["patterns"] = []
        
        return result
        
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse LLM JSON response: {e}", "raw_response": response}


def transpile_file(file_path: str) -> Dict[str, Any]:
    """
    Convenience function: read file and transpile.
    
    Args:
        file_path: Path to source file
    
    Returns:
        Transpilation result or error dict
    """
    path = Path(file_path)
    
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    
    if not path.is_file():
        return {"error": f"Not a file: {file_path}"}
    
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": f"Failed to decode file (not valid UTF-8): {file_path}"}
    except PermissionError:
        return {"error": f"Permission denied: {file_path}"}
    except IOError as e:
        return {"error": f"Failed to read file: {e}"}
    
    return transpile_code_to_prose(content, file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert source code to prose descriptions for semantic search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python code_transpiler.py src/memory_service.py
    python code_transpiler.py src/memory_service.py --json
    
The prose description is optimized for semantic search - it captures what the
code does in natural language so that queries like "how do embeddings work"
can find relevant code files.
"""
    )
    
    parser.add_argument("file", help="Path to source file to transpile")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )
    
    args = parser.parse_args()
    
    result = transpile_file(args.file)
    
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        if "raw_response" in result:
            print(f"\nRaw LLM response:\n{result['raw_response']}", file=sys.stderr)
        sys.exit(1)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Subject: {result['subject']}")
        print(f"Language: {result['language']}")
        print(f"\nDependencies: {', '.join(result['dependencies']) if result['dependencies'] else '(none)'}")
        print(f"Patterns: {', '.join(result['patterns']) if result['patterns'] else '(none)'}")
        print(f"\n{'=' * 60}")
        print(f"\n{result['prose']}")


if __name__ == "__main__":
    main()
