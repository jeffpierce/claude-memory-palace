"""
TOON encoding wrapper for MCP tool responses.

Provides a decorator that automatically TOON-encodes tool responses when enabled.
Returns TOON-encoded strings as MCP TextContent so the SDK doesn't reject them.
"""
from functools import wraps
from typing import Any, Callable, Dict

from memory_palace.config import is_toon_output_enabled

# Lazy import — MCP SDK is a runtime dep, not needed for tests that only
# exercise config/notification logic without actually serving MCP tools.
_TextContent = None


def _get_text_content_class():
    """Lazy-load mcp.types.TextContent. Fails loud at runtime if missing."""
    global _TextContent
    if _TextContent is None:
        from mcp.types import TextContent
        _TextContent = TextContent
    return _TextContent

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

    The decorator intercepts the `toon` parameter from kwargs (if provided by caller)
    and uses config default if not provided. The tool function does NOT need to accept
    a `toon` parameter - it's handled entirely by this decorator.

    If `toon=True`, the response is TOON-encoded. If `toon=False`, raw dict is returned.

    Usage:
        @mcp.tool()
        @toon_response
        async def my_tool(arg1: str) -> dict[str, Any]:
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
        toon = kwargs.pop('toon', None)  # Remove from kwargs if present
        if toon is None:
            toon = is_toon_output_enabled()

        # Call the original function (without toon in kwargs)
        result = await func(*args, **kwargs)

        # If toon encoding is enabled and result is a dict, encode it
        # No try/except — if TOON encoding fails, that's a bug. Fail loud. (12-factor)
        if toon and isinstance(result, dict):
            encoder = _get_toon_encoder()
            encoded = encoder(result)
            TC = _get_text_content_class()
            return [TC(type="text", text=encoded)]

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
