"""
TOON encoding wrapper for MCP tool responses.

Provides a decorator that automatically TOON-encodes tool responses when enabled.
Returns TOON-encoded strings as MCP TextContent so the SDK doesn't reject them.
"""
from functools import wraps
from typing import Any, Callable, Dict

from mcp.types import TextContent
from memory_palace.config import is_toon_output_enabled

# Lazy import toons to avoid import errors if not installed
_toon_encode = None


def _get_toon_encoder():
    """Lazy-load the TOON encoder."""
    global _toon_encode
    if _toon_encode is None:
        try:
            from toons import dumps as toon_encode
            _toon_encode = toon_encode
        except ImportError:
            # If toons not installed, return identity function
            _toon_encode = lambda x: x
    return _toon_encode


def toon_response(func: Callable) -> Callable:
    """
    Decorator that TOON-encodes tool responses when enabled.

    The tool function should accept a `toon` boolean parameter (defaults to config value).
    If `toon=True`, the response is TOON-encoded. If `toon=False`, raw dict is returned.

    Usage:
        @mcp.tool()
        @toon_response
        async def my_tool(arg1: str, toon: bool = None) -> dict[str, Any]:
            # ... tool logic ...
            return {"result": "data"}
    """
    # Build wrapper with the ORIGINAL function's signature (minus return type)
    # so MCP SDK gets proper parameter schema but no DictModel output validator.
    # We can't use @wraps because inspect.signature() follows __wrapped__
    # back to the original function which still has -> dict[str, Any].
    import inspect
    sig = inspect.signature(func)
    new_sig = sig.replace(return_annotation=inspect.Parameter.empty)

    async def wrapper(*args, **kwargs):
        # Get the toon parameter, defaulting to config if not provided
        toon = kwargs.get('toon')
        if toon is None:
            toon = is_toon_output_enabled()
            kwargs['toon'] = toon

        # Call the original function
        result = await func(*args, **kwargs)

        # If toon encoding is enabled and result is a dict, encode it
        if toon and isinstance(result, dict):
            encoder = _get_toon_encoder()
            try:
                encoded = encoder(result)
                # Return as TextContent so MCP SDK accepts a string response
                return [TextContent(type="text", text=encoded)]
            except Exception as e:
                # If TOON encoding fails, fall back to raw dict
                print(f"Warning: TOON encoding failed: {e}")
                return result

        return result

    # Preserve function identity for MCP registration
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    # Copy annotations WITHOUT return type
    wrapper.__annotations__ = {
        k: v for k, v in func.__annotations__.items() if k != 'return'
    }
    # Set the signature explicitly (params preserved, return stripped)
    wrapper.__signature__ = new_sig

    return wrapper
