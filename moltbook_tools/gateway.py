"""
Moltbook Submission Gateway — the single entry point for all Moltbook API calls.

All 6 interlocks are enforced here. The gateway holds the API credentials.
Agents call submit() and get back success or a structured error.
"""
import hashlib
import unicodedata
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

import requests
from sqlalchemy import desc

from moltbook_tools.config import load_config, GatewayConfig
from moltbook_tools.database import get_session, init_db, session_scope
from moltbook_tools.models import SubmissionLog, QCApproval


def _normalize_content(text: str) -> str:
    """Normalize content for hashing and similarity comparison."""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())  # collapse whitespace
    return text.strip()


def _hash_content(text: str) -> str:
    """SHA-256 hash of normalized content."""
    normalized = _normalize_content(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _utcnow():
    """Return current UTC time as a naive datetime (SQLite doesn't store tz info)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Interlock checks ────────────────────────────────────────────────

def _check_session_guard(session, session_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
    """Gate 1: Block duplicate content within same session."""
    existing = (
        session.query(SubmissionLog)
        .filter(
            SubmissionLog.session_id == session_id,
            SubmissionLog.content_hash == content_hash,
        )
        .first()
    )
    if existing:
        return {
            "blocked_by": "session_guard",
            "reason": f"Content already submitted in this session (submission #{existing.id}). "
                      "Do NOT retry. Write a handoff instead.",
        }
    return None


def _check_content_dedup(session, content_hash: str) -> Optional[Dict[str, Any]]:
    """Gate 2: Block exact duplicate content (any session)."""
    existing = (
        session.query(SubmissionLog)
        .filter(
            SubmissionLog.content_hash == content_hash,
            SubmissionLog.status == "submitted",
        )
        .first()
    )
    if existing:
        return {
            "blocked_by": "content_dedup",
            "reason": f"Identical content was already posted (submission #{existing.id}, "
                      f"{existing.created_at.isoformat() if existing.created_at else 'unknown'}).",
        }
    return None


def _check_word_count(content: str, action_type: str, config: GatewayConfig) -> Optional[Dict[str, Any]]:
    """Gate 3: Block content exceeding word limits."""
    word_count = len(content.split())
    max_words = config.post_max_words if action_type == "post" else config.comment_max_words

    if word_count > max_words:
        return {
            "blocked_by": "word_count",
            "reason": f"{action_type.title()} has {word_count} words (max {max_words}). "
                      "Trim it down before resubmitting.",
        }
    return None


def _check_similarity(session, content_normalized: str, config: GatewayConfig) -> Optional[Dict[str, Any]]:
    """Gate 4: Block near-duplicate content (catches rewording)."""
    cutoff = _utcnow() - timedelta(hours=config.similarity_lookback_hours)
    recent = (
        session.query(SubmissionLog)
        .filter(
            SubmissionLog.status == "submitted",
            SubmissionLog.created_at >= cutoff,
        )
        .all()
    )

    for entry in recent:
        ratio = SequenceMatcher(None, content_normalized, entry.content_normalized).ratio()
        if ratio >= config.similarity_threshold:
            return {
                "blocked_by": "similarity",
                "reason": f"Content is {ratio:.0%} similar to submission #{entry.id} "
                          f"(threshold: {config.similarity_threshold:.0%}). "
                          "This looks like a retry with minor edits.",
            }
    return None


def _check_rate_limit(session, action_type: str, config: GatewayConfig) -> Optional[Dict[str, Any]]:
    """Gate 5: Block if within cooldown period."""
    cooldown = (
        config.post_cooldown_seconds if action_type == "post"
        else config.comment_cooldown_seconds
    )
    cutoff = _utcnow() - timedelta(seconds=cooldown)

    recent = (
        session.query(SubmissionLog)
        .filter(
            SubmissionLog.action_type == action_type,
            SubmissionLog.status == "submitted",
            SubmissionLog.created_at >= cutoff,
        )
        .order_by(desc(SubmissionLog.created_at))
        .first()
    )

    if recent:
        elapsed = (_utcnow() - recent.created_at).total_seconds()
        remaining = int(cooldown - elapsed)
        return {
            "blocked_by": "rate_limit",
            "reason": f"Rate limit: {remaining}s remaining before next {action_type}. "
                      f"Last {action_type} was {int(elapsed)}s ago (cooldown: {cooldown}s).",
        }
    return None


def _check_qc_gate(session, qc_token: str, content_hash: str) -> Optional[Dict[str, Any]]:
    """Gate 6: Require valid QC approval token."""
    if not qc_token:
        return {
            "blocked_by": "qc_gate",
            "reason": "No QC token provided. Content must be reviewed and approved by QC before submission.",
        }

    approval = (
        session.query(QCApproval)
        .filter(QCApproval.token == qc_token)
        .first()
    )

    if not approval:
        return {
            "blocked_by": "qc_gate",
            "reason": f"QC token '{qc_token}' not found. Was it created by the QC agent?",
        }

    if approval.verdict != "pass":
        return {
            "blocked_by": "qc_gate",
            "reason": f"QC verdict was '{approval.verdict}', not 'pass'. Content was rejected by QC.",
        }

    if approval.consumed_at is not None:
        return {
            "blocked_by": "qc_gate",
            "reason": f"QC token already consumed at {approval.consumed_at.isoformat()}. Tokens are single-use.",
        }

    now = _utcnow()
    if approval.expires_at <= now:
        return {
            "blocked_by": "qc_gate",
            "reason": f"QC token expired at {approval.expires_at.isoformat()}. Request a new QC review.",
        }

    if approval.content_hash != content_hash:
        return {
            "blocked_by": "qc_gate",
            "reason": "QC token was issued for different content. The content was modified after QC review.",
        }

    return None


# ── Main submit function ────────────────────────────────────────────

def submit(
    action_type: str,
    content: str,
    session_id: str,
    qc_token: str,
    submolt: Optional[str] = None,
    title: Optional[str] = None,
    post_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Submit content to Moltbook through the gateway.

    Runs all 6 interlocks in order (fail-fast). If all pass,
    makes the API call and logs the result.

    Args:
        action_type: "post" or "comment"
        content: The content to submit
        session_id: Unique session identifier (prevents retry loops)
        qc_token: QC approval token UUID
        submolt: Target submolt (required for posts)
        title: Post title (required for posts)
        post_id: Target post ID (required for comments)
        parent_id: Parent comment ID (for threaded replies)
        dry_run: If True, check all gates but don't call API

    Returns:
        Success: {"success": True, "submission_id": N, "moltbook_id": "..."}
        Blocked: {"success": False, "blocked_by": "...", "reason": "..."}
        Error: {"success": False, "error": "..."}
    """
    # Validate action_type
    if action_type not in ("post", "comment"):
        return {"success": False, "error": f"Invalid action_type: '{action_type}'. Must be 'post' or 'comment'."}

    # Validate required fields
    if action_type == "post":
        if not submolt:
            return {"success": False, "error": "submolt is required for posts."}
        if not title:
            return {"success": False, "error": "title is required for posts."}
    elif action_type == "comment":
        if not post_id:
            return {"success": False, "error": "post_id is required for comments."}

    config = load_config()
    content_normalized = _normalize_content(content)
    content_hash = hashlib.sha256(content_normalized.encode("utf-8")).hexdigest()

    # Ensure DB tables exist
    init_db()

    session = get_session()
    try:
        # ── Gate 1: Session guard ──
        block = _check_session_guard(session, session_id, content_hash)
        if block:
            _log_blocked(session, session_id, action_type, content_hash,
                         content_normalized, block["blocked_by"],
                         submolt=submolt, title=title, post_id=post_id,
                         parent_id=parent_id, qc_token=qc_token)
            return {"success": False, **block}

        # ── Gate 2: Content hash dedup ──
        block = _check_content_dedup(session, content_hash)
        if block:
            _log_blocked(session, session_id, action_type, content_hash,
                         content_normalized, block["blocked_by"],
                         submolt=submolt, title=title, post_id=post_id,
                         parent_id=parent_id, qc_token=qc_token)
            return {"success": False, **block}

        # ── Gate 3: Word count ──
        block = _check_word_count(content, action_type, config)
        if block:
            _log_blocked(session, session_id, action_type, content_hash,
                         content_normalized, block["blocked_by"],
                         submolt=submolt, title=title, post_id=post_id,
                         parent_id=parent_id, qc_token=qc_token)
            return {"success": False, **block}

        # ── Gate 4: Similarity check ──
        block = _check_similarity(session, content_normalized, config)
        if block:
            _log_blocked(session, session_id, action_type, content_hash,
                         content_normalized, block["blocked_by"],
                         submolt=submolt, title=title, post_id=post_id,
                         parent_id=parent_id, qc_token=qc_token)
            return {"success": False, **block}

        # ── Gate 5: Rate limit ──
        block = _check_rate_limit(session, action_type, config)
        if block:
            _log_blocked(session, session_id, action_type, content_hash,
                         content_normalized, block["blocked_by"],
                         submolt=submolt, title=title, post_id=post_id,
                         parent_id=parent_id, qc_token=qc_token)
            return {"success": False, **block}

        # ── Gate 6: QC gate ──
        block = _check_qc_gate(session, qc_token, content_hash)
        if block:
            _log_blocked(session, session_id, action_type, content_hash,
                         content_normalized, block["blocked_by"],
                         submolt=submolt, title=title, post_id=post_id,
                         parent_id=parent_id, qc_token=qc_token)
            return {"success": False, **block}

        # ── All gates passed ──
        if dry_run:
            return {"success": True, "dry_run": True, "gates_passed": True}

        # Check API key
        if not config.api_key:
            return {
                "success": False,
                "error": "No API key found. Check ~/.moltbook/config.json, "
                         "~/.config/moltbook/credentials.json, or MOLTBOOK_API_KEY env var.",
            }

        # Build API request
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

        if action_type == "post":
            url = f"{config.api_base_url}/posts"
            payload = {"submolt": submolt, "title": title, "content": content}
        else:
            url = f"{config.api_base_url}/posts/{post_id}/comments"
            payload = {"content": content}
            if parent_id:
                payload["parent_id"] = parent_id

        # Make API call
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
        except requests.RequestException as e:
            _log_submission(
                session, session_id, action_type, content_hash,
                content_normalized, "failed",
                submolt=submolt, title=title, post_id=post_id,
                parent_id=parent_id, qc_token=qc_token,
            )
            return {"success": False, "error": f"API request failed: {e}"}

        # Parse response
        api_data = {}
        try:
            api_data = response.json()
        except ValueError:
            pass

        if response.status_code == 429:
            # Rate limited by server
            _log_submission(
                session, session_id, action_type, content_hash,
                content_normalized, "failed",
                api_response_code=response.status_code,
                submolt=submolt, title=title, post_id=post_id,
                parent_id=parent_id, qc_token=qc_token,
            )
            retry_info = ""
            if "retry_after_minutes" in api_data:
                retry_info = f" Retry after {api_data['retry_after_minutes']} minutes."
            elif "retry_after_seconds" in api_data:
                retry_info = f" Retry after {api_data['retry_after_seconds']} seconds."
            return {
                "success": False,
                "error": f"Server rate limited (429).{retry_info}",
                "api_response_code": 429,
            }

        if not api_data.get("success"):
            _log_submission(
                session, session_id, action_type, content_hash,
                content_normalized, "failed",
                api_response_code=response.status_code,
                submolt=submolt, title=title, post_id=post_id,
                parent_id=parent_id, qc_token=qc_token,
            )
            error_msg = api_data.get("error", f"HTTP {response.status_code}")
            hint = api_data.get("hint", "")
            return {
                "success": False,
                "error": f"API error: {error_msg}" + (f" Hint: {hint}" if hint else ""),
                "api_response_code": response.status_code,
            }

        # Success — extract moltbook ID from response
        moltbook_id = None
        if "data" in api_data and isinstance(api_data["data"], dict):
            moltbook_id = api_data["data"].get("id") or api_data["data"].get("post_id") or api_data["data"].get("comment_id")

        # Log successful submission
        log_entry = _log_submission(
            session, session_id, action_type, content_hash,
            content_normalized, "submitted",
            api_response_code=response.status_code,
            moltbook_post_id=str(moltbook_id) if moltbook_id else None,
            submolt=submolt, title=title, post_id=post_id,
            parent_id=parent_id, qc_token=qc_token,
        )

        # Consume QC token
        approval = session.query(QCApproval).filter(QCApproval.token == qc_token).first()
        if approval:
            approval.consumed_at = _utcnow()
            session.commit()

        return {
            "success": True,
            "submission_id": log_entry.id,
            "moltbook_id": moltbook_id,
            "action_type": action_type,
        }

    finally:
        session.close()


def _log_blocked(session, session_id, action_type, content_hash,
                 content_normalized, blocked_by, **kwargs):
    """Log a blocked submission attempt."""
    entry = SubmissionLog(
        session_id=session_id,
        action_type=action_type,
        content_hash=content_hash,
        content_normalized=content_normalized,
        status="blocked",
        blocked_by=blocked_by,
        submolt=kwargs.get("submolt"),
        title=kwargs.get("title"),
        post_id=kwargs.get("post_id"),
        parent_id=kwargs.get("parent_id"),
        qc_token=kwargs.get("qc_token"),
    )
    session.add(entry)
    session.commit()


def _log_submission(session, session_id, action_type, content_hash,
                    content_normalized, status, **kwargs):
    """Log a submission (success or failure)."""
    entry = SubmissionLog(
        session_id=session_id,
        action_type=action_type,
        content_hash=content_hash,
        content_normalized=content_normalized,
        status=status,
        api_response_code=kwargs.get("api_response_code"),
        moltbook_post_id=kwargs.get("moltbook_post_id"),
        submolt=kwargs.get("submolt"),
        title=kwargs.get("title"),
        post_id=kwargs.get("post_id"),
        parent_id=kwargs.get("parent_id"),
        qc_token=kwargs.get("qc_token"),
    )
    session.add(entry)
    session.commit()
    return entry


# ── Status and log queries ───────────────────────────────────────────

def get_status() -> Dict[str, Any]:
    """Get gateway status: config, rate limits, recent activity."""
    config = load_config()
    init_db()

    session = get_session()
    try:
        now = _utcnow()

        # Find last post and comment
        last_post = (
            session.query(SubmissionLog)
            .filter(SubmissionLog.action_type == "post", SubmissionLog.status == "submitted")
            .order_by(desc(SubmissionLog.created_at))
            .first()
        )
        last_comment = (
            session.query(SubmissionLog)
            .filter(SubmissionLog.action_type == "comment", SubmissionLog.status == "submitted")
            .order_by(desc(SubmissionLog.created_at))
            .first()
        )

        # Calculate cooldowns
        def _cooldown_info(last_entry, cooldown_seconds):
            if not last_entry:
                return {"available": True, "seconds_remaining": 0}
            elapsed = (now - last_entry.created_at).total_seconds()
            remaining = max(0, cooldown_seconds - elapsed)
            return {
                "available": remaining == 0,
                "seconds_remaining": int(remaining),
                "last_submission": last_entry.created_at.isoformat(),
            }

        # Count recent submissions
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_count = (
            session.query(SubmissionLog)
            .filter(SubmissionLog.status == "submitted", SubmissionLog.created_at >= today_start)
            .count()
        )
        blocked_count = (
            session.query(SubmissionLog)
            .filter(SubmissionLog.status == "blocked", SubmissionLog.created_at >= today_start)
            .count()
        )

        return {
            "config_loaded": True,
            "api_key_configured": config.api_key is not None,
            "api_base_url": config.api_base_url,
            "rate_limits": {
                "post": _cooldown_info(last_post, config.post_cooldown_seconds),
                "comment": _cooldown_info(last_comment, config.comment_cooldown_seconds),
            },
            "today": {
                "submitted": today_count,
                "blocked": blocked_count,
            },
        }
    finally:
        session.close()


def get_log(last: int = 10, action_type: Optional[str] = None) -> Dict[str, Any]:
    """Get recent submission log entries."""
    init_db()
    session = get_session()
    try:
        query = session.query(SubmissionLog).order_by(desc(SubmissionLog.created_at))
        if action_type:
            query = query.filter(SubmissionLog.action_type == action_type)
        entries = query.limit(last).all()
        return {
            "entries": [e.to_dict() for e in entries],
            "count": len(entries),
        }
    finally:
        session.close()
