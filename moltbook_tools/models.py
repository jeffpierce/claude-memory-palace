from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

def _utcnow():
    """Return current UTC time as a naive datetime (SQLite doesn't store tz info)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

class SubmissionLog(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=_utcnow)
    session_id = Column(String, nullable=False, index=True)
    action_type = Column(String, nullable=False)  # "post" | "comment"
    content_hash = Column(String(64), nullable=False, index=True)
    content_normalized = Column(Text, nullable=False)
    submolt = Column(String, nullable=True)
    post_id = Column(String, nullable=True)
    parent_id = Column(String, nullable=True)
    title = Column(String, nullable=True)
    status = Column(String, default="submitted")  # submitted | failed | blocked
    blocked_by = Column(String, nullable=True)
    qc_token = Column(String, nullable=True)
    api_response_code = Column(Integer, nullable=True)
    moltbook_post_id = Column(String, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "session_id": self.session_id,
            "action_type": self.action_type,
            "content_hash": self.content_hash,
            "submolt": self.submolt,
            "post_id": self.post_id,
            "parent_id": self.parent_id,
            "title": self.title,
            "status": self.status,
            "blocked_by": self.blocked_by,
            "qc_token": self.qc_token,
            "api_response_code": self.api_response_code,
            "moltbook_post_id": self.moltbook_post_id,
        }


class QCApproval(Base):
    __tablename__ = "qc_approvals"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=_utcnow)
    content_hash = Column(String(64), nullable=False, index=True)
    token = Column(String(36), nullable=False, unique=True)  # UUID4
    verdict = Column(String, nullable=False)  # "pass" | "fail"
    notes = Column(String, nullable=True)
    consumed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "content_hash": self.content_hash,
            "token": self.token,
            "verdict": self.verdict,
            "notes": self.notes,
            "consumed_at": self.consumed_at.isoformat() if self.consumed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
