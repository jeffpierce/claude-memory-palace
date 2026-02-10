"""
QC (Quality Control) token management for Moltbook gateway.

QC agents review content and issue approval tokens. The gateway requires
a valid token before any submission. Tokens are:
- Single-use (consumed on successful submission)
- Time-limited (TTL from config, default 30 minutes)
- Content-bound (hash must match submitted content)
"""
import hashlib
import unicodedata
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from moltbook_tools.config import load_config
from moltbook_tools.database import get_session, init_db
from moltbook_tools.models import QCApproval


def _utcnow():
    """Return current UTC time as a naive datetime (SQLite doesn't store tz info)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_content(text: str) -> str:
    """Normalize content for hashing (same as gateway.py)."""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()


def _hash_content(text: str) -> str:
    """SHA-256 hash of normalized content."""
    normalized = _normalize_content(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def create_approval(
    content: str,
    notes: str = "",
    verdict: str = "pass",
) -> Dict[str, Any]:
    """
    Create a QC approval token for reviewed content.

    Called by the QC agent after reviewing content against the rubric.

    Args:
        content: The reviewed content (will be hashed for matching)
        notes: QC agent's reasoning/notes
        verdict: "pass" or "fail"

    Returns:
        {"token": "uuid", "content_hash": "sha256", "verdict": "pass", "expires_at": "iso"}
    """
    if verdict not in ("pass", "fail"):
        return {"error": f"Invalid verdict: '{verdict}'. Must be 'pass' or 'fail'."}

    config = load_config()
    content_hash = _hash_content(content)
    token = str(uuid.uuid4())
    now = _utcnow()
    expires_at = now + timedelta(minutes=config.qc_token_ttl_minutes)

    init_db()
    session = get_session()
    try:
        approval = QCApproval(
            content_hash=content_hash,
            token=token,
            verdict=verdict,
            notes=notes,
            expires_at=expires_at,
        )
        session.add(approval)
        session.commit()

        return {
            "token": token,
            "content_hash": content_hash,
            "verdict": verdict,
            "expires_at": expires_at.isoformat(),
            "ttl_minutes": config.qc_token_ttl_minutes,
        }
    finally:
        session.close()


def check_status(content: str) -> Dict[str, Any]:
    """
    Check QC approval status for content.

    Args:
        content: The content to check

    Returns:
        Dict with approval status, token info if exists
    """
    content_hash = _hash_content(content)
    init_db()

    session = get_session()
    try:
        approvals = (
            session.query(QCApproval)
            .filter(QCApproval.content_hash == content_hash)
            .order_by(QCApproval.created_at.desc())
            .all()
        )

        if not approvals:
            return {"has_approval": False, "content_hash": content_hash}

        latest = approvals[0]
        now = _utcnow()

        return {
            "has_approval": True,
            "content_hash": content_hash,
            "token": latest.token,
            "verdict": latest.verdict,
            "expired": latest.expires_at <= now,
            "consumed": latest.consumed_at is not None,
            "usable": (
                latest.verdict == "pass"
                and latest.consumed_at is None
                and latest.expires_at > now
            ),
            "expires_at": latest.expires_at.isoformat(),
            "notes": latest.notes,
            "total_approvals": len(approvals),
        }
    finally:
        session.close()
