"""
CLI entry point for Moltbook submission gateway.

Usage:
    python -m moltbook_tools submit post --submolt X --title Y --content Z --session-id S --qc-token T
    python -m moltbook_tools submit comment --post-id P --content Z --session-id S --qc-token T
    python -m moltbook_tools qc approve --content Z [--notes N]
    python -m moltbook_tools qc check --content Z
    python -m moltbook_tools status
    python -m moltbook_tools log [--last N] [--action-type post|comment]

All output is JSON. Exit codes: 0=success, 1=blocked, 2=error.
"""
import argparse
import json
import sys
from typing import Any, Dict


def _output(data: Dict[str, Any], exit_code: int = 0):
    """Print JSON output and exit."""
    print(json.dumps(data, indent=2, default=str))
    sys.exit(exit_code)


def cmd_submit(args):
    """Handle submit subcommand."""
    from moltbook_tools.gateway import submit

    # Read content from file if --content-file specified
    content = args.content
    if args.content_file:
        try:
            with open(args.content_file, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            _output({"success": False, "error": f"Cannot read content file: {e}"}, exit_code=2)

    if not content:
        _output({"success": False, "error": "No content provided. Use --content or --content-file."}, exit_code=2)

    result = submit(
        action_type=args.action_type,
        content=content,
        session_id=args.session_id,
        qc_token=args.qc_token,
        submolt=args.submolt,
        title=args.title,
        post_id=args.post_id,
        parent_id=args.parent_id,
        dry_run=args.dry_run,
    )

    if result.get("success"):
        _output(result, exit_code=0)
    elif result.get("blocked_by"):
        _output(result, exit_code=1)
    else:
        _output(result, exit_code=2)


def cmd_qc(args):
    """Handle qc subcommand."""
    from moltbook_tools.qc import create_approval, check_status

    if args.qc_action == "approve":
        content = args.content
        if args.content_file:
            try:
                with open(args.content_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError as e:
                _output({"error": f"Cannot read content file: {e}"}, exit_code=2)

        if not content:
            _output({"error": "No content provided. Use --content or --content-file."}, exit_code=2)

        result = create_approval(
            content=content,
            notes=args.notes or "",
            verdict=args.verdict or "pass",
        )
        if result.get("error"):
            _output(result, exit_code=2)
        else:
            _output(result, exit_code=0)

    elif args.qc_action == "check":
        content = args.content
        if args.content_file:
            try:
                with open(args.content_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError as e:
                _output({"error": f"Cannot read content file: {e}"}, exit_code=2)

        if not content:
            _output({"error": "No content provided. Use --content or --content-file."}, exit_code=2)

        result = check_status(content=content)
        _output(result, exit_code=0)


def cmd_status(args):
    """Handle status subcommand."""
    from moltbook_tools.gateway import get_status
    _output(get_status())


def cmd_log(args):
    """Handle log subcommand."""
    from moltbook_tools.gateway import get_log
    _output(get_log(last=args.last, action_type=args.action_type))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="moltbook-gateway",
        description="Moltbook Submission Gateway — mechanical interlocks for safe posting",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── submit ──
    submit_parser = subparsers.add_parser("submit", help="Submit content through the gateway")
    submit_parser.add_argument("action_type", choices=["post", "comment"],
                               help="Type of submission")
    submit_parser.add_argument("--content", help="Content to submit")
    submit_parser.add_argument("--content-file", help="Read content from file")
    submit_parser.add_argument("--session-id", required=True,
                               help="Session ID (prevents retry loops)")
    submit_parser.add_argument("--qc-token", required=True,
                               help="QC approval token UUID")
    submit_parser.add_argument("--submolt", help="Target submolt (required for posts)")
    submit_parser.add_argument("--title", help="Post title (required for posts)")
    submit_parser.add_argument("--post-id", help="Target post ID (required for comments)")
    submit_parser.add_argument("--parent-id", help="Parent comment ID (threaded replies)")
    submit_parser.add_argument("--dry-run", action="store_true",
                               help="Check all gates without calling API")
    submit_parser.set_defaults(func=cmd_submit)

    # ── qc ──
    qc_parser = subparsers.add_parser("qc", help="QC token management")
    qc_subparsers = qc_parser.add_subparsers(dest="qc_action", help="QC actions")

    # qc approve
    qc_approve = qc_subparsers.add_parser("approve", help="Create QC approval token")
    qc_approve.add_argument("--content", help="Reviewed content")
    qc_approve.add_argument("--content-file", help="Read content from file")
    qc_approve.add_argument("--notes", help="QC agent reasoning")
    qc_approve.add_argument("--verdict", choices=["pass", "fail"], default="pass",
                             help="QC verdict (default: pass)")

    # qc check
    qc_check = qc_subparsers.add_parser("check", help="Check QC status for content")
    qc_check.add_argument("--content", help="Content to check")
    qc_check.add_argument("--content-file", help="Read content from file")

    qc_parser.set_defaults(func=cmd_qc)

    # ── status ──
    status_parser = subparsers.add_parser("status", help="View gateway status")
    status_parser.set_defaults(func=cmd_status)

    # ── log ──
    log_parser = subparsers.add_parser("log", help="View submission log")
    log_parser.add_argument("--last", type=int, default=10,
                             help="Number of entries (default: 10)")
    log_parser.add_argument("--action-type", choices=["post", "comment"],
                             help="Filter by action type")
    log_parser.set_defaults(func=cmd_log)

    # Parse and dispatch
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(2)

    if not hasattr(args, "func"):
        # Subcommand exists but no sub-action (e.g., "qc" without "approve" or "check")
        if args.command == "qc":
            qc_parser.print_help()
        sys.exit(2)

    try:
        args.func(args)
    except Exception as e:
        _output({"success": False, "error": str(e)}, exit_code=2)


if __name__ == "__main__":
    main()
