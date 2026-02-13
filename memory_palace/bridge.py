"""
Memory Palace OpenClaw Native Tool Plugin Bridge

This module implements a long-running persistent subprocess that serves as a bridge
between OpenClaw and Memory Palace services. It reads JSON commands from stdin,
dispatches to service functions, and returns JSON responses on stdout.

Protocol:
  - Startup: Emit {"ready": true, "version": "2.0.1", "tools": N}
  - Request: {"id": "req-001", "method": "memory_remember", "params": {...}}
  - Response: {"id": "req-001", "result": {...}}
  - Error: {"id": "req-001", "error": {"message": "...", "code": "ERROR_CODE"}}
  - Server events (pubsub): {"event": "new_message", "data": {...}} (no id field)

The bridge supports:
  1. Memory operations (remember, recall, get, recent, archive, link, unlink)
  2. Messaging (send, get, mark_read, mark_unread, subscribe, unsubscribe)
  3. Maintenance (audit, reembed, stats, reflect)
  4. Code indexing (code_remember_tool)
  5. Real-time pubsub via Postgres LISTEN/NOTIFY
"""

import json
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from memory_palace.config import is_postgres
from memory_palace.database import get_engine, init_db
from memory_palace.services import (
    remember,
    recall,
    get_memory_by_id,
    get_memories_by_ids,
    get_recent_memories,
    archive_memory,
    link_memories,
    unlink_memories,
    send_message,
    get_messages,
    mark_message_read,
    mark_message_unread,
    subscribe,
    unsubscribe,
    code_remember,
    audit_palace,
    reembed_memories,
    get_memory_stats,
    reflect,
)
from memory_palace.services.message_service import _get_channel_name


# Global state
_shutdown_event: threading.Event = threading.Event()
_stdout_lock: threading.Lock = threading.Lock()
_listener_lock: threading.Lock = threading.Lock()
_listener_conn: Optional[Any] = None


def _write_response(obj: Dict[str, Any]) -> None:
    """
    Write a JSON response to stdout, thread-safe.

    Acquires lock, writes JSON line, flushes, releases lock.
    """
    with _stdout_lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def _handle_memory_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom handler for memory_get that normalizes memory_ids parameter.

    Dispatches to get_memory_by_id (single) or get_memories_by_ids (batch)
    based on whether memory_ids is an int or list.

    Args:
        params: Request parameters including memory_ids

    Returns:
        Memory object(s) or error dict
    """
    memory_ids = params.pop("memory_ids")
    if isinstance(memory_ids, int):
        ids = [memory_ids]
        single_mode = True
    else:
        ids = memory_ids
        single_mode = False

    if single_mode:
        result = get_memory_by_id(ids[0], **params)
        if result:
            return result
        else:
            return {"error": f"Memory {ids[0]} not found"}
    else:
        return get_memories_by_ids(ids, **params)


def _handle_message(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom handler for message dispatch based on action parameter.

    Dispatches to send_message, get_messages, mark_message_read, etc.

    Args:
        params: Request parameters including action

    Returns:
        Operation result or error dict
    """
    action = params.pop("action")

    if action == "send":
        from_instance = params.get("from_instance")
        to_instance = params.get("to_instance")
        content = params.get("content")
        if not from_instance or not to_instance or not content:
            return {
                "error": "send requires from_instance, to_instance, and content"
            }

        result = send_message(
            from_instance=from_instance,
            to_instance=to_instance,
            content=content,
            message_type=params.get("message_type", "message"),
            subject=params.get("subject"),
            channel=params.get("channel"),
            priority=params.get("priority", 0),
        )

        return result

    elif action == "get":
        instance_id = params.get("instance_id")
        if not instance_id:
            return {"error": "get requires instance_id"}
        return get_messages(
            instance_id=instance_id,
            unread_only=params.get("unread_only", True),
            channel=params.get("channel"),
            message_type=params.get("message_type")
            if params.get("message_type") != "message"
            else None,
            limit=params.get("limit", 50),
        )

    elif action == "mark_read":
        message_id = params.get("message_id")
        instance_id = params.get("instance_id")
        if message_id is None or not instance_id:
            return {"error": "mark_read requires message_id and instance_id"}
        return mark_message_read(message_id=message_id, instance_id=instance_id)

    elif action == "mark_unread":
        message_id = params.get("message_id")
        if message_id is None:
            return {"error": "mark_unread requires message_id"}
        return mark_message_unread(message_id=message_id)

    elif action == "subscribe":
        instance_id = params.get("instance_id")
        channel = params.get("channel")
        if not instance_id or not channel:
            return {"error": "subscribe requires instance_id and channel"}
        return subscribe(instance_id=instance_id, channel=channel)

    elif action == "unsubscribe":
        instance_id = params.get("instance_id")
        channel = params.get("channel")
        if not instance_id or not channel:
            return {"error": "unsubscribe requires instance_id and channel"}
        return unsubscribe(instance_id=instance_id, channel=channel)

    else:
        return {
            "error": f"Unknown action: {action}. Valid: send, get, mark_read, mark_unread, subscribe, unsubscribe"
        }


# Dispatch table mapping method names to handlers
DISPATCH: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "memory_remember": lambda params: remember(**params),
    "memory_recall": lambda params: recall(**params),
    "memory_get": _handle_memory_get,
    "memory_recent": lambda params: get_recent_memories(**params),
    "memory_archive": lambda params: archive_memory(**params),
    "memory_link": lambda params: link_memories(**params),
    "memory_unlink": lambda params: unlink_memories(**params),
    "message": _handle_message,
    "code_remember_tool": lambda params: code_remember(**params),
    "memory_audit": lambda params: audit_palace(**params),
    "memory_reembed": lambda params: reembed_memories(**params),
    "memory_stats": lambda params: get_memory_stats(),
    "memory_reflect": lambda params: reflect(**params),
}


def _start_listener_loop(
    instance_id: str, write_fn: Callable[[Dict[str, Any]], None]
) -> None:
    """
    Start Postgres LISTEN thread for real-time pubsub.

    Listens on memory_palace_msg_{instance_id} channel and emits
    server-initiated events on new messages.

    Args:
        instance_id: The instance ID to listen for
        write_fn: Function to write events (thread-safe)
    """
    global _listener_conn

    try:
        engine = get_engine()
        raw_conn = engine.raw_connection()
        _listener_conn = raw_conn

        cursor = raw_conn.cursor()
        channel = f"memory_palace_msg_{instance_id}"
        cursor.execute(f"LISTEN {channel}")
        raw_conn.commit()

        # Polling loop with cross-platform compatibility
        # Windows select.select doesn't work reliably with psycopg2 connections
        while not _shutdown_event.is_set():
            # Collect events under lock, emit AFTER releasing to avoid
            # ABBA deadlock with _stdout_lock (write_fn acquires _stdout_lock)
            pending_events = []
            with _listener_lock:
                raw_conn.poll()
                while raw_conn.notifies:
                    notify = raw_conn.notifies.pop(0)
                    try:
                        payload = json.loads(notify.payload)
                        event = {
                            "event": "new_message",
                            "data": {
                                "from": payload.get("from_instance"),
                                "to": payload.get("to_instance"),
                                "type": payload.get("message_type"),
                                "subject": payload.get("subject"),
                                "id": payload.get("message_id"),
                                "priority": payload.get("priority", 0),
                            },
                        }
                        pending_events.append(event)
                    except (json.JSONDecodeError, Exception) as e:
                        print(
                            f"Warning: Failed to parse NOTIFY payload: {e}",
                            file=sys.stderr,
                        )

            # Emit events outside the lock — write_fn acquires _stdout_lock
            for event in pending_events:
                write_fn(event)

            # Sleep 100ms between polls to avoid busy-waiting
            time.sleep(0.1)

        cursor.close()
        raw_conn.close()

    except Exception as e:
        print(f"Error in listener loop: {e}", file=sys.stderr)


def main() -> None:
    """
    Main bridge entry point.

    Initializes database, starts listener thread, and enters request loop.
    Reads JSON commands from stdin, dispatches to handlers, writes responses
    to stdout. Supports shutdown via _shutdown method or EOF.
    """
    global _shutdown_event, _listener_conn

    # Parse instance ID from environment or config
    instance_id = os.environ.get("MEMORY_PALACE_INSTANCE_ID")

    # Initialize database
    init_db()

    # Initialize shutdown event
    _shutdown_event = threading.Event()

    # Start pubsub listener if Postgres and instance_id is set
    listener_thread: Optional[threading.Thread] = None
    if is_postgres() and instance_id:
        listener_thread = threading.Thread(
            target=_start_listener_loop,
            args=(instance_id, _write_response),
            daemon=True,
        )
        listener_thread.start()

    # Emit ready handshake
    _write_response(
        {"ready": True, "version": "2.0.1", "tools": len(DISPATCH)}
    )

    # Main request loop - read from stdin
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                _write_response(
                    {
                        "error": {
                            "message": f"Invalid JSON: {e}",
                            "code": "PARSE_ERROR",
                        }
                    }
                )
                continue

            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})

            # Auto-parse stringified params (OpenClaw may serialize them)
            if isinstance(params, str):
                sys.stderr.write(
                    f"[bridge] WARNING: params is str for method={method}: {repr(params)[:300]}\n"
                )
                sys.stderr.flush()
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Log params type for debugging
            sys.stderr.write(
                f"[bridge] dispatch method={method} params_type={type(params).__name__} keys={list(params.keys()) if isinstance(params, dict) else 'N/A'}\n"
            )
            sys.stderr.flush()

            # Bridge management methods (prefixed with _)
            if method == "_shutdown":
                _shutdown_event.set()
                _write_response(
                    {"id": req_id, "result": {"message": "Shutting down"}}
                )
                break

            elif method == "_subscribe":
                # Dynamic channel subscription
                channel = params.get("channel")
                if channel and _listener_conn:
                    try:
                        with _listener_lock:
                            cursor = _listener_conn.cursor()
                            cursor.execute(f"LISTEN {channel}")
                            _listener_conn.commit()
                            cursor.close()
                        _write_response(
                            {
                                "id": req_id,
                                "result": {
                                    "message": f"Subscribed to {channel}"
                                },
                            }
                        )
                    except Exception as e:
                        _write_response(
                            {
                                "id": req_id,
                                "error": {
                                    "message": str(e),
                                    "code": "SUBSCRIBE_ERROR",
                                },
                            }
                        )
                else:
                    _write_response(
                        {
                            "id": req_id,
                            "result": {
                                "message": f"Subscribed to {channel} (no listener active)"
                            },
                        }
                    )
                continue

            elif method == "_unsubscribe":
                # Dynamic channel unsubscription
                channel = params.get("channel")
                if channel and _listener_conn:
                    try:
                        with _listener_lock:
                            cursor = _listener_conn.cursor()
                            cursor.execute(f"UNLISTEN {channel}")
                            _listener_conn.commit()
                            cursor.close()
                        _write_response(
                            {
                                "id": req_id,
                                "result": {
                                    "message": f"Unsubscribed from {channel}"
                                },
                            }
                        )
                    except Exception as e:
                        _write_response(
                            {
                                "id": req_id,
                                "error": {
                                    "message": str(e),
                                    "code": "UNSUBSCRIBE_ERROR",
                                },
                            }
                        )
                else:
                    _write_response(
                        {
                            "id": req_id,
                            "result": {
                                "message": f"Unsubscribed from {channel} (no listener active)"
                            },
                        }
                    )
                continue

            # Look up dispatch handler
            handler = DISPATCH.get(method)
            if not handler:
                _write_response(
                    {
                        "id": req_id,
                        "error": {
                            "message": f"Unknown method: {method}",
                            "code": "METHOD_NOT_FOUND",
                        },
                    }
                )
                continue

            # For message method, capture action and channel before handler modifies params
            message_action = None
            message_channel = None
            if method == "message":
                message_action = params.get("action")
                message_channel = params.get("channel")

            # Execute handler and write response
            try:
                result = handler(params)
                _write_response({"id": req_id, "result": result})

                # After handler completes, sync LISTEN/UNLISTEN on bridge's _listener_conn
                # This ensures the bridge polls the correct channels for subscribe/unsubscribe actions
                if method == "message" and _listener_conn is not None:
                    if message_action == "subscribe" and message_channel:
                        try:
                            with _listener_lock:
                                channel_name = _get_channel_name(message_channel, None)
                                cursor = _listener_conn.cursor()
                                cursor.execute(f"LISTEN {channel_name}")
                                _listener_conn.commit()
                                cursor.close()
                            sys.stderr.write(
                                f"[bridge] LISTEN on {channel_name} via _listener_conn\n"
                            )
                            sys.stderr.flush()
                        except Exception as e:
                            sys.stderr.write(
                                f"[bridge] Failed to LISTEN on {message_channel}: {e}\n"
                            )
                            sys.stderr.flush()
                    elif message_action == "unsubscribe" and message_channel:
                        try:
                            with _listener_lock:
                                channel_name = _get_channel_name(message_channel, None)
                                cursor = _listener_conn.cursor()
                                cursor.execute(f"UNLISTEN {channel_name}")
                                _listener_conn.commit()
                                cursor.close()
                            sys.stderr.write(
                                f"[bridge] UNLISTEN on {channel_name} via _listener_conn\n"
                            )
                            sys.stderr.flush()
                        except Exception as e:
                            sys.stderr.write(
                                f"[bridge] Failed to UNLISTEN on {message_channel}: {e}\n"
                            )
                            sys.stderr.flush()

            except Exception as e:
                sys.stderr.write(
                    f"[bridge] method={method} params_type={type(params).__name__} "
                    f"params_repr={repr(params)[:500]} error={e}\n"
                )
                sys.stderr.flush()
                _write_response(
                    {
                        "id": req_id,
                        "error": {"message": str(e), "code": "INTERNAL_ERROR"},
                    }
                )

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        _shutdown_event.set()
        if listener_thread:
            listener_thread.join(timeout=3)


if __name__ == "__main__":
    main()
