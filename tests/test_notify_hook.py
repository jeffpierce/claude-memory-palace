"""
Tests for post-send notification hook in message tool.

Tests _execute_notify_hook() as a standalone function — no database needed.

NOTE: We use lazy imports (inside fixtures/tests) to avoid triggering the
full message_service → database_v3 import chain at collection time.
test_v3_core_services.py does aggressive module deletion/reimport for its
early patching, and any test file that eagerly imports database_v3 or
models_v3 will cause module-identity conflicts in the full suite.
"""
import os
import subprocess
from unittest.mock import patch

import pytest


@pytest.fixture
def execute_hook():
    """Lazily import _execute_notify_hook to avoid module chain pollution."""
    from mcp_server.tools.message import _execute_notify_hook
    return _execute_notify_hook


@pytest.fixture
def mock_send_result():
    """Mock successful send_message result."""
    return {"success": True, "id": 42}


@pytest.fixture
def send_params():
    """Standard send parameters for testing."""
    return {
        "from_instance": "code",
        "to_instance": "desktop",
        "message_type": "handoff",
        "subject": "Task completed",
        "channel": "general",
        "priority": 5,
    }


class TestNotifyHook:
    """Tests for the _execute_notify_hook function."""

    @patch("subprocess.run")
    def test_notification_fires_on_send(
        self, mock_run, execute_hook, mock_send_result, send_params
    ):
        """Verify subprocess.run is called with the templated command."""
        execute_hook(
            command_template="echo Message {message_id} sent",
            send_result=mock_send_result,
            **send_params,
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["shell"] is True
        assert call_args[1]["timeout"] == 5
        assert call_args[1]["capture_output"] is True

    @patch("subprocess.run")
    def test_template_variables_substituted(
        self, mock_run, execute_hook, mock_send_result, send_params
    ):
        """Verify all 7 template variables are correctly substituted."""
        execute_hook(
            command_template=(
                "notify --from {from_instance} --to {to_instance} "
                "--type {message_type} --subject {subject} --channel {channel} "
                "--priority {priority} --id {message_id}"
            ),
            send_result=mock_send_result,
            **send_params,
        )

        actual_command = mock_run.call_args[0][0]

        # shlex.quote only adds quotes when value contains spaces/special chars
        assert "code" in actual_command
        assert "desktop" in actual_command
        assert "handoff" in actual_command
        assert "'Task completed'" in actual_command  # space → quoted
        assert "general" in actual_command
        assert "5" in actual_command
        assert "42" in actual_command

    @patch("subprocess.run")
    def test_template_variables_are_shell_escaped(
        self, mock_run, execute_hook, mock_send_result, send_params
    ):
        """Verify values are shell-escaped to prevent command injection."""
        dangerous_params = send_params.copy()
        dangerous_params["subject"] = "; rm -rf /"

        execute_hook(
            command_template="echo {subject}",
            send_result=mock_send_result,
            **dangerous_params,
        )

        actual_command = mock_run.call_args[0][0]
        # shlex.quote wraps dangerous content: "; rm -rf /" → "'; rm -rf /'"
        assert "'; rm -rf /'" in actual_command

    @patch("subprocess.run")
    def test_notification_failure_does_not_raise(
        self, mock_run, execute_hook, mock_send_result, send_params
    ):
        """Fire-and-forget: exception in subprocess does not propagate."""
        mock_run.side_effect = Exception("Command failed")

        execute_hook(
            command_template="failing-command {message_id}",
            send_result=mock_send_result,
            **send_params,
        )

        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_notification_timeout_does_not_raise(
        self, mock_run, execute_hook, mock_send_result, send_params
    ):
        """Fire-and-forget: subprocess timeout does not propagate."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)

        execute_hook(
            command_template="slow-command {message_id}",
            send_result=mock_send_result,
            **send_params,
        )

        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_none_subject_and_channel_become_empty_strings(
        self, mock_run, execute_hook, mock_send_result
    ):
        """None values for optional params are converted to empty strings."""
        execute_hook(
            command_template="notify --subject {subject} --channel {channel}",
            send_result=mock_send_result,
            from_instance="code",
            to_instance="desktop",
            message_type="message",
            subject=None,
            channel=None,
            priority=0,
        )

        actual_command = mock_run.call_args[0][0]
        # shlex.quote('') → "''"
        assert "''" in actual_command

    @patch("subprocess.run")
    def test_message_id_extracted_from_send_result(
        self, mock_run, execute_hook, send_params
    ):
        """message_id comes from result['id'], not a parameter."""
        execute_hook(
            command_template="echo {message_id}",
            send_result={"success": True, "id": 999},
            **send_params,
        )

        actual_command = mock_run.call_args[0][0]
        assert "999" in actual_command

    @patch("subprocess.run")
    def test_missing_message_id_uses_empty_string(
        self, mock_run, execute_hook, send_params
    ):
        """If result has no 'id' key, message_id defaults to empty string."""
        execute_hook(
            command_template="echo {message_id}",
            send_result={"success": True},  # No 'id' key
            **send_params,
        )

        actual_command = mock_run.call_args[0][0]
        assert "''" in actual_command


class TestNotifyConfig:
    """Tests for notify_command config integration."""

    def test_env_var_overrides_config(self):
        """Verify MEMORY_PALACE_NOTIFY_COMMAND env var takes precedence."""
        from memory_palace.config_v2 import get_notify_command, clear_config_cache

        clear_config_cache()
        old_val = os.environ.get("MEMORY_PALACE_NOTIFY_COMMAND")
        try:
            os.environ["MEMORY_PALACE_NOTIFY_COMMAND"] = "env-command {message_id}"
            clear_config_cache()
            assert get_notify_command() == "env-command {message_id}"
        finally:
            if old_val is None:
                os.environ.pop("MEMORY_PALACE_NOTIFY_COMMAND", None)
            else:
                os.environ["MEMORY_PALACE_NOTIFY_COMMAND"] = old_val
            clear_config_cache()

    def test_default_notify_command_is_none(self):
        """Without env var override, default notify_command is None."""
        from memory_palace.config_v2 import get_notify_command, clear_config_cache, DEFAULT_CONFIG

        # Verify the DEFAULT_CONFIG has None for notify_command
        assert DEFAULT_CONFIG.get("notify_command") is None
