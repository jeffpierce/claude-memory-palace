"""
Tests for Moltbook Submission Gateway.

Tests all 6 interlocks and the gateway's end-to-end behavior using
in-memory SQLite and mocked HTTP calls.
"""
import hashlib
import re
import unicodedata
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import moltbook_tools.database as db_module
import moltbook_tools.gateway as gateway_module
from moltbook_tools.config import GatewayConfig
from moltbook_tools.database import init_db
from moltbook_tools.models import Base, SubmissionLog, QCApproval
from moltbook_tools.gateway import submit, _normalize_content


# ── Helpers ──────────────────────────────────────────────────────────

def _test_config(**overrides) -> GatewayConfig:
    """Return a GatewayConfig with test defaults. Override any field."""
    defaults = dict(
        api_base_url="https://test.moltbook.com/api/v1",
        api_key="test-api-key-12345",
        post_cooldown_seconds=1800,
        comment_cooldown_seconds=20,
        post_max_words=1000,
        comment_max_words=300,
        qc_token_ttl_minutes=30,
        similarity_threshold=0.85,
        similarity_lookback_hours=72,
    )
    defaults.update(overrides)
    return GatewayConfig(**defaults)


def _hash_content(text: str) -> str:
    """Hash content the same way the gateway does."""
    normalized = _normalize_content(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _utcnow():
    """Return current UTC time as a naive datetime (matches gateway behavior)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _make_qc_token(session, content: str, verdict: str = "pass",
                   ttl_minutes: int = 30, consumed: bool = False,
                   expired: bool = False) -> str:
    """Create a valid QC approval token in the DB and return the token string."""
    content_hash = _hash_content(content)
    token = str(uuid.uuid4())
    now = _utcnow()

    if expired:
        expires_at = now - timedelta(minutes=1)
    else:
        expires_at = now + timedelta(minutes=ttl_minutes)

    approval = QCApproval(
        content_hash=content_hash,
        token=token,
        verdict=verdict,
        notes="test approval",
        expires_at=expires_at,
        consumed_at=now if consumed else None,
    )
    session.add(approval)
    session.commit()
    return token


def _insert_submission(session, content: str, action_type: str = "post",
                       session_id: str = "old-session",
                       status: str = "submitted",
                       created_at: datetime = None) -> SubmissionLog:
    """Insert a submission log entry for testing."""
    normalized = _normalize_content(content)
    content_hash = _hash_content(content)
    entry = SubmissionLog(
        session_id=session_id,
        action_type=action_type,
        content_hash=content_hash,
        content_normalized=normalized,
        status=status,
        submolt="test",
        title="Test",
    )
    if created_at:
        entry.created_at = created_at
    session.add(entry)
    session.commit()
    return entry


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def gateway_db():
    """Set up an in-memory SQLite database for each test."""
    # Reset module-level singletons
    db_module.reset_engine()

    # Create in-memory engine
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    db_module._engine = engine
    db_module._SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )

    # Create tables
    init_db(engine)

    yield engine

    # Teardown
    db_module.reset_engine()


@pytest.fixture
def db_session():
    """Get a database session for direct DB manipulation in tests."""
    session = db_module.get_session()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def mock_config():
    """Mock load_config to return test config for every test."""
    with patch.object(gateway_module, "load_config", return_value=_test_config()):
        yield


def _mock_api_success(post_id="mb-post-123"):
    """Return a mock response for successful API calls."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {"id": post_id},
    }
    return mock_resp


def _mock_api_failure(status_code=500, error="Internal Server Error"):
    """Return a mock response for failed API calls."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "success": False,
        "error": error,
    }
    return mock_resp


# ── Test Classes ─────────────────────────────────────────────────────

class TestSessionGuard:
    """Gate 1: Session guard prevents retry loops within a session."""

    def test_blocks_same_session_same_content(self, db_session):
        """Same session + same content = blocked on second attempt."""
        content = "This is test content for session guard."
        sid = "session-guard-test-1"
        token = _make_qc_token(db_session, content)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result1 = submit(
                action_type="post", content=content, session_id=sid,
                qc_token=token, submolt="test", title="Test",
            )
        assert result1["success"] is True

        # Second attempt with same session and content (need new QC token since first was consumed)
        token2 = _make_qc_token(db_session, content)
        result2 = submit(
            action_type="post", content=content, session_id=sid,
            qc_token=token2, submolt="test", title="Test",
        )
        assert result2["success"] is False
        assert result2["blocked_by"] == "session_guard"

    def test_allows_different_session_same_content(self, db_session):
        """Different session + same content = allowed (dedup checks status=submitted separately)."""
        content = "Content for cross-session test."
        token1 = _make_qc_token(db_session, content)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result1 = submit(
                action_type="post", content=content, session_id="session-A",
                qc_token=token1, submolt="test", title="Test",
            )
        assert result1["success"] is True

        # Different session, same content -- session guard passes, but content_dedup catches it
        token2 = _make_qc_token(db_session, content)
        result2 = submit(
            action_type="post", content=content, session_id="session-B",
            qc_token=token2, submolt="test", title="Test",
        )
        # Session guard passes (different session), but content_dedup should block
        assert result2["success"] is False
        assert result2["blocked_by"] == "content_dedup"

    def test_allows_same_session_different_content(self, db_session):
        """Same session + different content = allowed."""
        content1 = "First piece of unique content for this session."
        content2 = "Completely different second piece of content here."
        sid = "session-same-diff-content"
        token1 = _make_qc_token(db_session, content1)
        token2 = _make_qc_token(db_session, content2)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result1 = submit(
                action_type="post", content=content1, session_id=sid,
                qc_token=token1, submolt="test", title="Test 1",
            )
        assert result1["success"] is True

        # Need to bypass rate limit -- backdate the first submission
        entry = db_session.query(SubmissionLog).filter(
            SubmissionLog.session_id == sid
        ).first()
        entry.created_at = _utcnow() - timedelta(hours=2)
        db_session.commit()

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result2 = submit(
                action_type="post", content=content2, session_id=sid,
                qc_token=token2, submolt="test", title="Test 2",
            )
        assert result2["success"] is True


class TestContentHashDedup:
    """Gate 2: Content hash dedup prevents exact duplicate submissions."""

    def test_blocks_exact_duplicate(self, db_session):
        """Exact same content submitted twice (different sessions) = blocked by dedup."""
        content = "Exact duplicate content for dedup test."
        _insert_submission(db_session, content, status="submitted",
                           created_at=_utcnow() - timedelta(hours=2))

        token = _make_qc_token(db_session, content)
        result = submit(
            action_type="post", content=content, session_id="new-session-dedup",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "content_dedup"

    def test_blocks_regardless_of_session(self, db_session):
        """Dedup is global -- blocks even from a fresh session."""
        content = "Global dedup content test."
        _insert_submission(db_session, content, session_id="session-X",
                           status="submitted",
                           created_at=_utcnow() - timedelta(hours=2))

        token = _make_qc_token(db_session, content)
        result = submit(
            action_type="post", content=content, session_id="session-Y",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "content_dedup"


class TestWordCount:
    """Gate 3: Word count limits."""

    def test_blocks_post_over_1000_words(self, db_session):
        """Post with 1001 words = blocked."""
        content = " ".join(["word"] * 1001)
        token = _make_qc_token(db_session, content)
        result = submit(
            action_type="post", content=content, session_id="wc-post-over",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "word_count"

    def test_allows_post_at_1000_words(self, db_session):
        """Post with exactly 1000 words = allowed."""
        content = " ".join(["word"] * 1000)
        token = _make_qc_token(db_session, content)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result = submit(
                action_type="post", content=content, session_id="wc-post-exact",
                qc_token=token, submolt="test", title="Test",
            )
        assert result["success"] is True

    def test_blocks_comment_over_300_words(self, db_session):
        """Comment with 301 words = blocked."""
        content = " ".join(["word"] * 301)
        token = _make_qc_token(db_session, content)
        result = submit(
            action_type="comment", content=content, session_id="wc-comment-over",
            qc_token=token, post_id="post-123",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "word_count"

    def test_allows_comment_at_300_words(self, db_session):
        """Comment with exactly 300 words = allowed."""
        content = " ".join(["word"] * 300)
        token = _make_qc_token(db_session, content)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result = submit(
                action_type="comment", content=content, session_id="wc-comment-exact",
                qc_token=token, post_id="post-123",
            )
        assert result["success"] is True


class TestSimilarity:
    """Gate 4: Similarity check catches near-duplicate rewording."""

    def test_blocks_similar_content(self, db_session):
        """Content that is very similar to a recent submission = blocked."""
        original = "The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon"
        similar = "The quick brown fox jumps over the lazy dog near the river bank on a sunny morning"

        _insert_submission(db_session, original, status="submitted",
                           created_at=_utcnow() - timedelta(hours=1))

        token = _make_qc_token(db_session, similar)
        result = submit(
            action_type="post", content=similar, session_id="sim-block",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "similarity"

    def test_allows_sufficiently_different_content(self, db_session):
        """Content that is different enough from recent submissions = allowed."""
        original = "The quick brown fox jumps over the lazy dog near the river bank."
        different = "A completely unrelated discussion about quantum computing and neural networks in 2026."

        _insert_submission(db_session, original, status="submitted",
                           created_at=_utcnow() - timedelta(hours=1))

        token = _make_qc_token(db_session, different)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result = submit(
                action_type="post", content=different, session_id="sim-allow",
                qc_token=token, submolt="test", title="Test",
            )
        assert result["success"] is True

    def test_only_checks_within_lookback_window(self, db_session):
        """Submissions older than lookback window are ignored."""
        old_content = "This is old content that should not trigger similarity."
        # Same content -- would definitely match, but it's old
        new_content = "This is old content that should not trigger similarity."

        _insert_submission(db_session, old_content, status="submitted",
                           created_at=_utcnow() - timedelta(hours=100))  # Beyond 72h lookback

        # Content dedup will catch exact match, so use a slightly different version
        new_content = "This is very old content that should not trigger any similarity check."
        token = _make_qc_token(db_session, new_content)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result = submit(
                action_type="post", content=new_content, session_id="sim-lookback",
                qc_token=token, submolt="test", title="Test",
            )
        # Should pass similarity (old entry outside window) but may still be fine
        # The key assertion: not blocked by similarity
        assert result.get("blocked_by") != "similarity"


class TestRateLimit:
    """Gate 5: Rate limiting."""

    def test_blocks_post_within_30_minutes(self, db_session):
        """Post submitted within 30 minutes of last post = blocked."""
        old_content = "Previous post within rate limit window."
        _insert_submission(db_session, old_content, action_type="post",
                           status="submitted",
                           created_at=_utcnow() - timedelta(minutes=10))

        new_content = "Brand new post that should be rate limited."
        token = _make_qc_token(db_session, new_content)
        result = submit(
            action_type="post", content=new_content, session_id="rate-post-block",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "rate_limit"

    def test_allows_post_after_30_minutes(self, db_session):
        """Post submitted after 30+ minutes since last post = allowed."""
        old_content = "Previous post outside rate limit window."
        _insert_submission(db_session, old_content, action_type="post",
                           status="submitted",
                           created_at=_utcnow() - timedelta(minutes=35))

        new_content = "New post after cooldown period has elapsed."
        token = _make_qc_token(db_session, new_content)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result = submit(
                action_type="post", content=new_content, session_id="rate-post-allow",
                qc_token=token, submolt="test", title="Test",
            )
        assert result["success"] is True

    def test_blocks_comment_within_20_seconds(self, db_session):
        """Comment submitted within 20 seconds of last comment = blocked."""
        old_content = "Previous comment within rate limit."
        _insert_submission(db_session, old_content, action_type="comment",
                           status="submitted",
                           created_at=_utcnow() - timedelta(seconds=5))

        new_content = "New comment that should be rate limited by comment cooldown."
        token = _make_qc_token(db_session, new_content)
        result = submit(
            action_type="comment", content=new_content, session_id="rate-comment-block",
            qc_token=token, post_id="post-123",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "rate_limit"

    def test_allows_comment_after_20_seconds(self, db_session):
        """Comment submitted after 20+ seconds since last comment = allowed."""
        old_content = "Previous comment outside cooldown."
        _insert_submission(db_session, old_content, action_type="comment",
                           status="submitted",
                           created_at=_utcnow() - timedelta(seconds=25))

        new_content = "New comment after the cooldown period has elapsed."
        token = _make_qc_token(db_session, new_content)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result = submit(
                action_type="comment", content=new_content,
                session_id="rate-comment-allow",
                qc_token=token, post_id="post-123",
            )
        assert result["success"] is True

    def test_post_and_comment_have_independent_limits(self, db_session):
        """A recent post should not block a comment, and vice versa."""
        post_content = "Recent post content."
        _insert_submission(db_session, post_content, action_type="post",
                           status="submitted",
                           created_at=_utcnow() - timedelta(minutes=5))

        comment_content = "A comment that should not be rate limited by the post cooldown."
        token = _make_qc_token(db_session, comment_content)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result = submit(
                action_type="comment", content=comment_content,
                session_id="rate-independent",
                qc_token=token, post_id="post-123",
            )
        # Comment rate limit is 20s, we haven't commented recently, so this should pass
        assert result["success"] is True


class TestQCGate:
    """Gate 6: QC approval token validation."""

    def test_blocks_without_token(self, db_session):
        """No QC token = blocked."""
        content = "Content without QC token."
        result = submit(
            action_type="post", content=content, session_id="qc-no-token",
            qc_token="", submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "qc_gate"

    def test_blocks_expired_token(self, db_session):
        """Expired QC token = blocked."""
        content = "Content with expired QC token."
        token = _make_qc_token(db_session, content, expired=True)
        result = submit(
            action_type="post", content=content, session_id="qc-expired",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "qc_gate"
        assert "expired" in result["reason"].lower()

    def test_blocks_consumed_token(self, db_session):
        """Already-consumed QC token = blocked."""
        content = "Content with consumed QC token."
        token = _make_qc_token(db_session, content, consumed=True)
        result = submit(
            action_type="post", content=content, session_id="qc-consumed",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "qc_gate"
        assert "consumed" in result["reason"].lower()

    def test_blocks_wrong_content_hash(self, db_session):
        """QC token for different content = blocked."""
        approved_content = "This is the content that was approved by QC."
        actual_content = "This is totally different content being submitted."
        token = _make_qc_token(db_session, approved_content)
        result = submit(
            action_type="post", content=actual_content, session_id="qc-wrong-hash",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "qc_gate"
        assert "different content" in result["reason"].lower() or "modified" in result["reason"].lower()

    def test_blocks_failed_verdict(self, db_session):
        """QC verdict 'fail' = blocked."""
        content = "Content that failed QC review."
        token = _make_qc_token(db_session, content, verdict="fail")
        result = submit(
            action_type="post", content=content, session_id="qc-failed",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "qc_gate"
        assert "fail" in result["reason"].lower()

    def test_allows_valid_token(self, db_session):
        """Valid, unexpired, unconsumed QC token with matching hash = allowed."""
        content = "Content with a perfectly valid QC token."
        token = _make_qc_token(db_session, content, verdict="pass")

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result = submit(
                action_type="post", content=content, session_id="qc-valid",
                qc_token=token, submolt="test", title="Test",
            )
        assert result["success"] is True

    def test_consumes_token_on_success(self, db_session):
        """Successful submission marks the QC token as consumed."""
        content = "Content whose token will be consumed."
        token = _make_qc_token(db_session, content, verdict="pass")

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success()):
            result = submit(
                action_type="post", content=content, session_id="qc-consume",
                qc_token=token, submolt="test", title="Test",
            )
        assert result["success"] is True

        # Verify token is consumed in DB
        approval = db_session.query(QCApproval).filter(
            QCApproval.token == token
        ).first()
        assert approval.consumed_at is not None


class TestGateway:
    """End-to-end gateway tests."""

    def test_happy_path_post(self, db_session):
        """All gates pass, API succeeds -> success with submission_id and moltbook_id."""
        content = "A great post about the state of the world."
        token = _make_qc_token(db_session, content)

        with patch("moltbook_tools.gateway.requests.post", return_value=_mock_api_success("mb-999")):
            result = submit(
                action_type="post", content=content, session_id="happy-post",
                qc_token=token, submolt="test-submolt", title="Great Post",
            )
        assert result["success"] is True
        assert "submission_id" in result
        assert result["moltbook_id"] == "mb-999"
        assert result["action_type"] == "post"

    def test_happy_path_comment(self, db_session):
        """All gates pass for a comment -> success."""
        content = "An insightful comment."
        token = _make_qc_token(db_session, content)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "data": {"comment_id": "mc-456"},
        }

        with patch("moltbook_tools.gateway.requests.post", return_value=mock_resp):
            result = submit(
                action_type="comment", content=content, session_id="happy-comment",
                qc_token=token, post_id="post-789",
            )
        assert result["success"] is True
        assert result["moltbook_id"] == "mc-456"
        assert result["action_type"] == "comment"

    def test_fail_fast_order(self, db_session):
        """Gates are checked in order: cheapest first. Session guard before QC gate."""
        content = "Content for fail-fast order test."
        sid = "fail-fast-session"

        # Insert a prior submission to trigger session guard
        _insert_submission(db_session, content, session_id=sid, status="blocked")

        # Even with no QC token, session guard should fire first
        result = submit(
            action_type="post", content=content, session_id=sid,
            qc_token="", submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "session_guard"

    def test_dry_run_doesnt_call_api(self, db_session):
        """Dry run checks all gates but does NOT call the API."""
        content = "Content for dry run test."
        token = _make_qc_token(db_session, content)

        with patch("moltbook_tools.gateway.requests.post") as mock_post:
            result = submit(
                action_type="post", content=content, session_id="dry-run",
                qc_token=token, submolt="test", title="Test",
                dry_run=True,
            )
            mock_post.assert_not_called()

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["gates_passed"] is True

    def test_api_failure_logged(self, db_session):
        """API returning error -> logged as failed."""
        content = "Content where the API will fail."
        token = _make_qc_token(db_session, content)

        with patch("moltbook_tools.gateway.requests.post",
                    return_value=_mock_api_failure(500, "Server Error")):
            result = submit(
                action_type="post", content=content, session_id="api-fail",
                qc_token=token, submolt="test", title="Test",
            )
        assert result["success"] is False
        assert "error" in result
        assert result["api_response_code"] == 500

        # Verify it was logged
        entry = db_session.query(SubmissionLog).filter(
            SubmissionLog.session_id == "api-fail",
            SubmissionLog.status == "failed",
        ).first()
        assert entry is not None
        assert entry.api_response_code == 500

    def test_logs_blocked_submissions(self, db_session):
        """Blocked submissions are logged with blocked_by field."""
        content = " ".join(["word"] * 1001)  # Over word limit
        token = _make_qc_token(db_session, content)

        result = submit(
            action_type="post", content=content, session_id="log-blocked",
            qc_token=token, submolt="test", title="Test",
        )
        assert result["success"] is False
        assert result["blocked_by"] == "word_count"

        # Verify it was logged in the DB
        entry = db_session.query(SubmissionLog).filter(
            SubmissionLog.session_id == "log-blocked",
            SubmissionLog.status == "blocked",
        ).first()
        assert entry is not None
        assert entry.blocked_by == "word_count"
