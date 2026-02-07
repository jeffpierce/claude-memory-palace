"""
JSONL to TOON converter tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import jsonl_to_toon_chunks
from mcp_server.toon_wrapper import toon_response


def register_jsonl_to_toon(mcp):
    """Register the jsonl_to_toon tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def convert_jsonl_to_toon(
        input_path: str,
        output_dir: str,
        mode: str = "aggressive",
        chunk_tokens: int = 12500,
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Convert Claude JSONL transcript to chunked TOON files.

        TOON (Token-Optimized Notation) is a compact format optimized for token efficiency.
        Achieves 95%+ compression while preserving conversation semantics.
        Splits on conversation boundaries for feeding to memory_reflect.

        Args:
            input_path: Path to source JSONL transcript file
            output_dir: Directory to write chunk files (created if doesn't exist)
            mode: Conversion mode:
                - "conservative": Keep timestamps, summarize thinking
                - "aggressive": Maximum compression (default)
            chunk_tokens: Target tokens per chunk (default 12500 - leaves headroom for LLM reasoning)

        Returns:
            Dict with chunk_count, chunk_files, total_tokens, compression_ratio
        """
        return jsonl_to_toon_chunks(
            input_path=input_path,
            output_dir=output_dir,
            mode=mode,
            chunk_tokens=chunk_tokens
        )
