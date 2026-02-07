"""
TOON encoding wrapper for MCP tool responses.

Provides a decorator that automatically TOON-encodes tool responses when enabled.
Returns TOON-encoded strings as MCP TextContent so the SDK doesn't reject them.
"""
from functools import wraps
from typing import Any, Callable, Dict

from mcp.types import TextContent
from memory_palace.config import is_toon_output_enabled

# Lazy import toons — fail loud if missing when enabled (12-factor: no silent degradation)
_toon_encode = None


def _get_toon_encoder():
    """Lazy-load the TOON encoder. Raises ImportError if toons is not installed."""
    global _toon_encode
    if _toon_encode is None:
        from toons import dumps as toon_encode
        _toon_encode = toon_encode
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
        # No try/except — if TOON encoding fails, that's a bug. Fail loud. (12-factor)
        if toon and isinstance(result, dict):
            encoder = _get_toon_encoder()
            encoded = encoder(result)
            return [TextContent(type="text", text=encoded)]

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
