# Plan: OpenClaw Push Notifications (feat/openclaw-push)

## Problem

Inter-instance messaging is pull-only. When Instance A sends a message to Instance B, B only sees it on next boot or heartbeat poll. No real-time push.

## Current State

- `notify_command` exists in config as a single shell command template (string or null)
- Template vars available: `{to_instance}`, `{from_instance}`, `{channel}`, `{priority}`, `{message_type}`, `{subject}`, `{message_id}`
- `_execute_notify_hook()` in `mcp_server/tools/message.py` fires after successful send — shell exec, fire-and-forget, 5s timeout
- Works but **cannot route to different gateways per instance** — single command template doesn't know that `prime` is on port 18789 and `crashtest` is on port 18790

## OpenClaw Wake API (Resolved)

**Endpoint:** `POST {gateway_url}/hooks/wake`
**Payload:** `{ "text": "...", "mode": "now" | "next-heartbeat" }`
**Auth:** One of:
- `Authorization: Bearer <token>`
- `x-openclaw-token: <token>`
- `?token=<token>`

**Prerequisite:** External hooks must be enabled on each target gateway:
```json
{
  "hooks": {
    "enabled": true,
    "token": "shared-secret"
  }
}
```

**Note:** `/hooks/wake` wakes all heartbeat-enabled agents on that gateway. There's no session targeting on the wake endpoint — it's a broadcast to the gateway. This is actually fine for our use case: each container runs one primary agent, so waking the gateway wakes the right agent. For multi-agent gateways (like Prime's with main/sarah/anthony), the wake text should indicate who the message is for so the agent can check.

## Solution

Add `instance_routes` mapping in the palace config. Each instance maps to a gateway URL + auth token. The notify hook looks up the route for `to_instance` and POSTs to that gateway's wake endpoint.

## Config Schema

New top-level key in `~/.memory-palace/config.json`:

```json
{
  "instance_routes": {
    "prime": {
      "gateway": "http://localhost:18789",
      "token": "96529d3cd83f1e0fcdcc2949e67aa10dad7013c1e83f116f"
    },
    "crashtest": {
      "gateway": "http://localhost:18790",
      "token": "crashtest-gateway-token-here"
    }
  }
}
```

Minimal config — just gateway URL and auth token per instance. No session key needed because `/hooks/wake` is a gateway-level broadcast.

The old `notify_command` string continues to work as a fallback for non-OpenClaw setups. If `instance_routes` has a matching route, it takes priority. If not, falls back to `notify_command`. Both can coexist.

## Implementation Tasks

### Task 1: Config layer (`memory_palace/config_v2.py`)

Add to `DEFAULT_CONFIG`:
```python
"instance_routes": {},  # Map instance_id -> {"gateway": "url", "token": "secret"}
```

Add helpers:
```python
def get_instance_routes() -> Dict[str, Dict[str, str]]:
    """Get all instance routes."""
    return load_config().get("instance_routes", {})

def get_instance_route(instance_id: str) -> Optional[Dict[str, str]]:
    """Get route for a specific instance, or None."""
    return get_instance_routes().get(instance_id)
```

Env var override: `MEMORY_PALACE_INSTANCE_ROUTES` (JSON string) for container deployments where you can't easily edit config.json.

### Task 2: HTTP wake function (`mcp_server/tools/message.py`)

Add `_execute_openclaw_wake()`:
```python
def _execute_openclaw_wake(
    route: Dict[str, str],
    from_instance: str,
    to_instance: str,
    message_type: str,
    subject: Optional[str],
    message_id: int,
    priority: int,
) -> None:
    """
    Fire-and-forget HTTP wake to an OpenClaw gateway.
    Never raises — logs warnings on failure.
    """
    try:
        import urllib.request
        import json

        gateway_url = route["gateway"].rstrip("/")
        token = route.get("token", "")

        wake_text = f"Palace message from {from_instance}: {subject or message_type} (msg #{message_id})"

        payload = json.dumps({
            "text": wake_text,
            "mode": "now" if priority >= 5 else "next-heartbeat",
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{gateway_url}/hooks/wake",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )

        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"Warning: OpenClaw wake failed for {to_instance}: {e}", file=sys.stderr)
```

Uses `urllib` (stdlib) — no new dependencies.

Priority-based mode: priority >= 5 wakes immediately, lower priority waits for next heartbeat.

### Task 3: Update post-send logic (`mcp_server/tools/message.py`)

In the `message()` tool's send action, after successful send:

```python
# 1. Try instance_routes (HTTP wake) — preferred
route = get_instance_route(to_instance)
if route:
    _execute_openclaw_wake(route, from_instance, to_instance, ...)
elif to_instance == "all":
    # Broadcast: wake ALL routed instances
    for inst_id, inst_route in get_instance_routes().items():
        if inst_id != from_instance:  # Don't wake yourself
            _execute_openclaw_wake(inst_route, from_instance, inst_id, ...)

# 2. Fallback to notify_command (shell exec) — backwards compat
notify_cmd = get_notify_command()
if notify_cmd is not None:
    _execute_notify_hook(command_template=notify_cmd, ...)
```

Both mechanisms can fire (route-based wake AND shell hook) — they serve different purposes and the shell hook might do other things (logging, etc).

### Task 4: Documentation

- Update `README.md` messaging feature description
- Add `instance_routes` to `docs/README.md` config section
- Add example config showing multi-gateway routing

### Task 5: Tests

- `tests/test_push_notify.py`:
  - Config loading with instance_routes
  - Route lookup (found, not found, "all" broadcast)
  - Wake HTTP call (mock urllib)
  - Priority-based mode selection
  - Fallback to notify_command when no route

## File Changes

| File | Change | Lines Est. |
|------|--------|-----------|
| `memory_palace/config_v2.py` | Add `instance_routes` default + getters + env override | ~25 |
| `mcp_server/tools/message.py` | Add HTTP wake function, update post-send logic, add import | ~60 |
| `README.md` | Update messaging bullet point | ~5 |
| `docs/README.md` | Add instance_routes config section | ~30 |
| `tests/test_push_notify.py` | New test file | ~80 |

**Total: ~200 lines of new/changed code**

## Execution: Single Sonnet Agent

Work is serial (config → wake function → integration → docs → tests). Spawn one Sonnet agent to implement all tasks sequentially on the `feat/openclaw-push` branch.
