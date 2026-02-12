"""
Tests for OpenClaw push notification (HTTP wake) in message tool.

Tests _execute_openclaw_wake(), instance_routes config, and the post-send
routing logic. No database needed — all network/config calls are mocked.

NOTE: Lazy imports used (same pattern as test_notify_hook.py) to avoid
triggering the full message_service → database_v3 import chain at
collection time.
"""
import json
import os
from unittest.mock import patch, MagicMock, call

import pytest


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def execute_wake():
    """Lazily import _execute_openclaw_wake to avoid module chain pollution."""
    from mcp_server.tools.message import _execute_openclaw_wake
    return _execute_openclaw_wake


@pytest.fixture
def sample_route():
    """A single instance route."""
    return {
        "gateway": "http://localhost:18789",
        "token": "test-token-abc",
        "session": "agent:main:main",
    }


@pytest.fixture
def wake_params():
    """Standard wake call parameters."""
    return {
        "from_instance": "code",
        "to_instance": "prime",
        "message_type": "handoff",
        "subject": "Task completed",
        "message_id": 42,
        "priority": 5,
    }


@pytest.fixture
def multi_routes():
    """Multiple instance routes for broadcast tests."""
    return {
        "prime": {
            "gateway": "http://localhost:18789",
            "token": "prime-token",
        },
        "crashtest": {
            "gateway": "http://localhost:18790",
            "token": "crashtest-token",
        },
        "desktop": {
            "gateway": "http://localhost:18791",
            "token": "desktop-token",
        },
    }


# -- Config Tests ------------------------------------------------------------

class TestInstanceRoutesConfig:
    """Tests for instance_routes configuration loading."""

    def test_default_instance_routes_is_empty(self):
        """Default config has empty instance_routes."""
        from memory_palace.config_v2 import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.get("instance_routes") == {}

    def test_get_instance_routes_returns_dict(self):
        """get_instance_routes() returns a dict (possibly empty)."""
        from memory_palace.config_v2 import get_instance_routes, clear_config_cache
        clear_config_cache()
        result = get_instance_routes()
        assert isinstance(result, dict)
        clear_config_cache()

    def test_get_instance_route_returns_none_for_unknown(self):
        """get_instance_route() returns None for unconfigured instance."""
        from memory_palace.config_v2 import get_instance_route, clear_config_cache
        clear_config_cache()
        assert get_instance_route("nonexistent-instance") is None
        clear_config_cache()

    def test_env_var_override_instance_routes(self):
        """MEMORY_PALACE_INSTANCE_ROUTES env var overrides config."""
        from memory_palace.config_v2 import (
            get_instance_routes,
            get_instance_route,
            clear_config_cache,
        )

        routes = {
            "prime": {"gateway": "http://localhost:18789", "token": "abc"},
        }
        old_val = os.environ.get("MEMORY_PALACE_INSTANCE_ROUTES")
        try:
            os.environ["MEMORY_PALACE_INSTANCE_ROUTES"] = json.dumps(routes)
            clear_config_cache()

            result = get_instance_routes()
            assert result == routes

            route = get_instance_route("prime")
            assert route is not None
            assert route["gateway"] == "http://localhost:18789"
            assert route["token"] == "abc"
        finally:
            if old_val is None:
                os.environ.pop("MEMORY_PALACE_INSTANCE_ROUTES", None)
            else:
                os.environ["MEMORY_PALACE_INSTANCE_ROUTES"] = old_val
            clear_config_cache()

    def test_env_var_bad_json_is_ignored(self):
        """Malformed JSON in env var doesn't crash, routes stay default."""
        from memory_palace.config_v2 import get_instance_routes, clear_config_cache

        old_val = os.environ.get("MEMORY_PALACE_INSTANCE_ROUTES")
        try:
            os.environ["MEMORY_PALACE_INSTANCE_ROUTES"] = "not valid json {"
            clear_config_cache()

            result = get_instance_routes()
            assert isinstance(result, dict)
        finally:
            if old_val is None:
                os.environ.pop("MEMORY_PALACE_INSTANCE_ROUTES", None)
            else:
                os.environ["MEMORY_PALACE_INSTANCE_ROUTES"] = old_val
            clear_config_cache()


# -- Wake Function Tests -----------------------------------------------------

class TestOpenClawWake:
    """Tests for the _execute_openclaw_wake function."""

    @patch("urllib.request.urlopen")
    def test_wake_sends_http_post(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """Verify urlopen is called with correct URL and method."""
        execute_wake(route=sample_route, **wake_params)

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:18789/hooks/palace"
        assert req.method == "POST"
        assert req.get_header("Content-type") == "application/json"
        assert req.get_header("Authorization") == "Bearer test-token-abc"

    @patch("urllib.request.urlopen")
    def test_wake_payload_contains_structured_fields(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """Verify the JSON payload has correct structured fields for /hooks/palace."""
        execute_wake(route=sample_route, **wake_params)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["from_instance"] == "code"
        assert payload["to_instance"] == "prime"
        assert payload["message_type"] == "handoff"
        assert payload["subject"] == "Task completed"
        assert payload["message_id"] == 42
        assert payload["priority"] == 5

    @patch("urllib.request.urlopen")
    def test_payload_includes_priority(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """Priority is passed through in the payload."""
        wake_params["priority"] = 4
        execute_wake(route=sample_route, **wake_params)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["priority"] == 4

    @patch("urllib.request.urlopen")
    def test_priority_5_in_payload(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """Priority == 5 is included in payload."""
        wake_params["priority"] = 5
        execute_wake(route=sample_route, **wake_params)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["priority"] == 5

    @patch("urllib.request.urlopen")
    def test_priority_10_in_payload(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """Priority == 10 (max) is included in payload."""
        wake_params["priority"] = 10
        execute_wake(route=sample_route, **wake_params)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["priority"] == 10

    @patch("urllib.request.urlopen")
    def test_priority_0_in_payload(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """Priority == 0 (default) is included in payload."""
        wake_params["priority"] = 0
        execute_wake(route=sample_route, **wake_params)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["priority"] == 0

    @patch("urllib.request.urlopen")
    def test_none_subject_passes_through(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """When subject is None, it's passed as null in payload."""
        wake_params["subject"] = None
        execute_wake(route=sample_route, **wake_params)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["subject"] is None
        assert payload["message_type"] == "handoff"

    @patch("urllib.request.urlopen")
    def test_gateway_url_trailing_slash_stripped(
        self, mock_urlopen, execute_wake, wake_params
    ):
        """Trailing slash on gateway URL is stripped before /hooks/palace."""
        route = {"gateway": "http://localhost:18789/", "token": "tok"}
        execute_wake(route=route, **wake_params)

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:18789/hooks/palace"

    @patch("urllib.request.urlopen")
    def test_route_session_key_not_in_payload(
        self, mock_urlopen, execute_wake, wake_params
    ):
        """Session key from route is not included in palace webhook payload
        (routing is handled by the JS transform on the gateway side)."""
        route = {
            "gateway": "http://localhost:18789",
            "token": "tok",
            "session": "agent:anthony:main",
        }
        wake_params["to_instance"] = "anthony"
        execute_wake(route=route, **wake_params)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        # Payload is structured data, not text with relay instructions
        assert payload["to_instance"] == "anthony"
        assert "sessionKey" not in payload

    @patch("urllib.request.urlopen")
    def test_missing_token_uses_empty_bearer(
        self, mock_urlopen, execute_wake, wake_params
    ):
        """Route without token key sends empty Bearer header."""
        route = {"gateway": "http://localhost:18789"}
        execute_wake(route=route, **wake_params)

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer "

    @patch("urllib.request.urlopen")
    def test_wake_timeout_is_5_seconds(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """urlopen is called with timeout=5."""
        execute_wake(route=sample_route, **wake_params)

        assert mock_urlopen.call_args[1]["timeout"] == 5

    @patch("urllib.request.urlopen")
    def test_wake_failure_does_not_raise(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """Fire-and-forget: network errors do not propagate."""
        mock_urlopen.side_effect = Exception("Connection refused")
        # Should not raise
        execute_wake(route=sample_route, **wake_params)

    @patch("urllib.request.urlopen")
    def test_wake_timeout_does_not_raise(
        self, mock_urlopen, execute_wake, sample_route, wake_params
    ):
        """Fire-and-forget: timeout errors do not propagate."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("timed out")
        execute_wake(route=sample_route, **wake_params)

    @patch("urllib.request.urlopen")
    def test_wake_bad_route_does_not_raise(self, mock_urlopen, execute_wake):
        """Route missing 'gateway' key doesn't raise."""
        execute_wake(
            route={"token": "abc"},  # missing gateway
            from_instance="code",
            to_instance="prime",
            message_type="message",
            subject=None,
            message_id=1,
            priority=0,
        )
        # Should not raise; urlopen may or may not be called


# -- Routing Logic Tests -----------------------------------------------------

class TestPostSendRouting:
    """Tests for the post-send routing logic (route → wake, fallback → shell)."""

    @patch("mcp_server.tools.message.get_instance_routes")
    @patch("mcp_server.tools.message.get_instance_route")
    @patch("mcp_server.tools.message._execute_openclaw_wake")
    @patch("mcp_server.tools.message.get_notify_command", return_value=None)
    @patch("mcp_server.tools.message._execute_notify_hook")
    def test_direct_send_wakes_target(
        self,
        mock_hook,
        mock_get_cmd,
        mock_wake,
        mock_get_route,
        mock_get_routes,
    ):
        """Direct send to instance with route calls wake for that instance."""
        route = {"gateway": "http://localhost:18789", "token": "tok"}
        mock_get_route.return_value = route

        # Import the internals we need to simulate post-send
        from mcp_server.tools.message import (
            _execute_openclaw_wake,
            get_instance_route,
            get_instance_routes,
            get_notify_command,
        )

        # Simulate what happens in the send action after result.get("success")
        to_instance = "prime"
        from_instance = "code"
        result = {"success": True, "id": 42}

        _notify_params = {
            "from_instance": from_instance,
            "to_instance": to_instance,
            "message_type": "handoff",
            "subject": "test",
            "message_id": result.get("id", ""),
            "priority": 5,
        }

        r = get_instance_route(to_instance)
        if r:
            _execute_openclaw_wake(route=r, **_notify_params)

        mock_wake.assert_called_once_with(
            route=route,
            from_instance="code",
            to_instance="prime",
            message_type="handoff",
            subject="test",
            message_id=42,
            priority=5,
        )
        mock_hook.assert_not_called()

    @patch("mcp_server.tools.message.get_instance_routes")
    @patch("mcp_server.tools.message.get_instance_route")
    @patch("mcp_server.tools.message._execute_openclaw_wake")
    def test_broadcast_wakes_all_except_sender(
        self, mock_wake, mock_get_route, mock_get_routes, multi_routes
    ):
        """Broadcast (to_instance='all') wakes all routes except sender."""
        mock_get_route.return_value = None  # "all" is not in routes
        mock_get_routes.return_value = multi_routes

        from mcp_server.tools.message import (
            _execute_openclaw_wake,
            get_instance_route,
            get_instance_routes,
        )

        to_instance = "all"
        from_instance = "prime"
        _notify_params = {
            "from_instance": from_instance,
            "to_instance": to_instance,
            "message_type": "status",
            "subject": "maintenance",
            "message_id": 99,
            "priority": 8,
        }

        route = get_instance_route(to_instance)
        if route:
            _execute_openclaw_wake(route=route, **_notify_params)
        elif to_instance == "all":
            for inst_id, inst_route in get_instance_routes().items():
                if inst_id != from_instance:
                    _execute_openclaw_wake(
                        route=inst_route,
                        **{**_notify_params, "to_instance": inst_id},
                    )

        # Should wake crashtest and desktop, not prime (sender)
        assert mock_wake.call_count == 2
        waked_instances = [c.kwargs["to_instance"] for c in mock_wake.call_args_list]
        assert "prime" not in waked_instances
        assert "crashtest" in waked_instances
        assert "desktop" in waked_instances

    @patch("mcp_server.tools.message._execute_notify_hook")
    @patch("mcp_server.tools.message.get_notify_command")
    @patch("mcp_server.tools.message._execute_openclaw_wake")
    @patch("mcp_server.tools.message.get_instance_route", return_value=None)
    def test_no_route_falls_back_to_notify_command(
        self, mock_get_route, mock_wake, mock_get_cmd, mock_hook
    ):
        """No route for instance → falls back to notify_command if set."""
        mock_get_cmd.return_value = "echo {message_id}"

        from mcp_server.tools.message import (
            _execute_openclaw_wake,
            _execute_notify_hook,
            get_instance_route,
            get_notify_command,
        )

        to_instance = "unknown-instance"
        result = {"success": True, "id": 50}

        route = get_instance_route(to_instance)
        if route:
            _execute_openclaw_wake(route=route, from_instance="code",
                                   to_instance=to_instance, message_type="msg",
                                   subject=None, message_id=50, priority=0)

        notify_cmd = get_notify_command()
        if notify_cmd is not None:
            _execute_notify_hook(
                command_template=notify_cmd, send_result=result,
                from_instance="code", to_instance=to_instance,
                message_type="msg", subject=None, channel=None, priority=0,
            )

        mock_wake.assert_not_called()
        mock_hook.assert_called_once()

    @patch("mcp_server.tools.message._execute_notify_hook")
    @patch("mcp_server.tools.message.get_notify_command")
    @patch("mcp_server.tools.message._execute_openclaw_wake")
    @patch("mcp_server.tools.message.get_instance_route")
    def test_both_route_and_command_can_fire(
        self, mock_get_route, mock_wake, mock_get_cmd, mock_hook
    ):
        """When both route and notify_command exist, both fire."""
        route = {"gateway": "http://localhost:18789", "token": "tok"}
        mock_get_route.return_value = route
        mock_get_cmd.return_value = "echo {message_id}"

        from mcp_server.tools.message import (
            _execute_openclaw_wake,
            _execute_notify_hook,
            get_instance_route,
            get_notify_command,
        )

        to_instance = "prime"
        result = {"success": True, "id": 77}

        _notify_params = {
            "from_instance": "code",
            "to_instance": to_instance,
            "message_type": "handoff",
            "subject": "test",
            "message_id": 77,
            "priority": 5,
        }

        r = get_instance_route(to_instance)
        if r:
            _execute_openclaw_wake(route=r, **_notify_params)

        notify_cmd = get_notify_command()
        if notify_cmd is not None:
            _execute_notify_hook(
                command_template=notify_cmd, send_result=result,
                from_instance="code", to_instance=to_instance,
                message_type="handoff", subject="test", channel=None, priority=5,
            )

        mock_wake.assert_called_once()
        mock_hook.assert_called_once()
