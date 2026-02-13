"""
Tests for Maintenance Service and Extension Loading.

Tests all maintenance operations (audit, reembed, batch archive) and
verifies extension modules can be imported properly.
"""
import pytest
import sys
import json
import importlib
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from memory_palace.models_v3 import Base, Memory, MemoryEdge
from memory_palace.models import _project_contains
import memory_palace.database_v3 as db_module
from memory_palace.services.maintenance_service import (
    audit_palace,
    _find_stale_memories,
    _find_orphan_edges,
    _find_missing_embeddings,
    _find_contradictions,
    _find_unlinked_memories,
    _find_cross_project_auto_links,
    cleanup_cross_project_auto_links,
    reembed_memories,
    batch_archive_memories,
)

FAKE_EMBEDDING = [0.1] * 768
FAKE_EMBEDDING_JSON = json.dumps(FAKE_EMBEDDING)  # For SQLite storage


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def test_db():
    """Set up an in-memory SQLite database for each test."""
    # Reset module-level singletons
    old_engines = db_module._engines.copy()
    old_sessions = db_module._session_factories.copy()

    # Create in-memory engine
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    db_module._engines["default"] = engine
    db_module._session_factories["default"] = Session

    yield engine, Session

    # Teardown
    db_module._engines.clear()
    db_module._engines.update(old_engines)
    db_module._session_factories.clear()
    db_module._session_factories.update(old_sessions)
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Get a database session for direct DB manipulation in tests."""
    session = db_module.get_session()
    yield session
    session.close()


@pytest.fixture
def mock_get_embedding():
    """Mock the embedding function to return fake embeddings as JSON string for SQLite."""
    with patch("memory_palace.services.maintenance_service.get_embedding", return_value=FAKE_EMBEDDING_JSON):
        yield


@pytest.fixture
def mock_cosine_similarity():
    """Mock cosine similarity for predictable duplicate detection."""
    def fake_similarity(emb1, emb2):
        # Return 0.95 if both are FAKE_EMBEDDING_JSON, else 0.3
        # Need to handle both JSON string and parsed list
        def normalize(emb):
            if isinstance(emb, str):
                return emb
            elif isinstance(emb, list):
                return json.dumps(emb)
            return emb

        e1 = normalize(emb1)
        e2 = normalize(emb2)

        if e1 == FAKE_EMBEDDING_JSON and e2 == FAKE_EMBEDDING_JSON:
            return 0.95
        return 0.3

    with patch("memory_palace.services.maintenance_service.cosine_similarity", side_effect=fake_similarity):
        yield


# ── Helpers ──────────────────────────────────────────────────────────


def _create_test_memory(session, **kwargs):
    """Helper to create a memory with defaults."""
    # Convert project (singular) to projects (plural) if provided
    if "project" in kwargs:
        project_val = kwargs.pop("project")
        if isinstance(project_val, str):
            kwargs["projects"] = [project_val]
        else:
            kwargs["projects"] = project_val

    defaults = {
        "instance_id": "test",
        "projects": ["life"],  # Use projects (plural) for new schema
        "memory_type": "fact",
        "content": "Test content",
        "created_at": datetime.now(timezone.utc),
        "access_count": 0,
        "is_archived": False,
        "foundational": False,
        "embedding": FAKE_EMBEDDING_JSON,  # Use JSON string for SQLite
    }
    defaults.update(kwargs)

    # Handle embedding parameter - convert list to JSON string for SQLite
    if "embedding" in defaults and isinstance(defaults["embedding"], list):
        defaults["embedding"] = json.dumps(defaults["embedding"])

    memory = Memory(**defaults)
    session.add(memory)
    session.commit()
    session.refresh(memory)
    return memory


def _create_test_edge(session, source_id, target_id, relation_type="relates_to", strength=1.0, edge_metadata=None):
    """Helper to create a memory edge."""
    edge = MemoryEdge(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        strength=strength,
        edge_metadata=edge_metadata,
    )
    session.add(edge)
    session.commit()
    session.refresh(edge)
    return edge


# ── Test Classes ─────────────────────────────────────────────────────


class TestAuditPalace:
    """Tests for audit_palace() - palace health audit."""

    def test_runs_all_checks_by_default(self, db_session):
        """When checks=None, all checks should run."""
        result = audit_palace(checks=None)

        # Should have all check categories
        assert "duplicates" in result
        assert "stale" in result
        assert "orphan_edges" in result
        assert "missing_embeddings" in result
        assert "contradictions" in result
        assert "unlinked" in result
        assert "cross_project_auto_links" in result
        assert "summary" in result

        # Summary should have all counts
        summary = result["summary"]
        assert "duplicates_found" in summary
        assert "stale_found" in summary
        assert "orphan_edges_found" in summary
        assert "missing_embeddings_found" in summary
        assert "contradictions_found" in summary
        assert "unlinked_found" in summary
        assert "cross_project_auto_links_found" in summary
        assert "total_issues" in summary

    def test_runs_only_specified_checks(self, db_session):
        """When checks list provided, only those checks should run."""
        result = audit_palace(checks=["duplicates", "stale"])

        # Should have only requested checks
        assert "duplicates" in result
        assert "stale" in result
        assert "summary" in result

        # Should NOT have other checks
        assert "orphan_edges" not in result
        assert "missing_embeddings" not in result
        assert "contradictions" not in result
        assert "unlinked" not in result

        # Summary should only count requested checks
        summary = result["summary"]
        assert "duplicates_found" in summary
        assert "stale_found" in summary
        assert "orphan_edges_found" not in summary

    def test_returns_summary_with_total_issue_counts(self, db_session):
        """Summary should aggregate all issue counts."""
        # Create some issues
        old_date = datetime.now(timezone.utc) - timedelta(days=100)

        # Stale memory
        _create_test_memory(
            db_session,
            subject="stale",
            created_at=old_date,
            access_count=0,
            foundational=False,
        )

        # Missing embedding
        _create_test_memory(
            db_session,
            subject="no embedding",
            embedding=None,
        )

        result = audit_palace()
        summary = result["summary"]

        assert summary["total_issues"] >= 0
        # Total should be sum of all individual counts
        expected_total = (
            summary.get("duplicates_found", 0) +
            summary.get("stale_found", 0) +
            summary.get("orphan_edges_found", 0) +
            summary.get("missing_embeddings_found", 0) +
            summary.get("contradictions_found", 0) +
            summary.get("unlinked_found", 0) +
            summary.get("cross_project_auto_links_found", 0)
        )
        assert summary["total_issues"] == expected_total

    def test_custom_thresholds_override_defaults(self, db_session, mock_cosine_similarity):
        """Custom thresholds should override defaults."""
        # Create two memories with same embedding
        _create_test_memory(db_session, subject="mem1")
        _create_test_memory(db_session, subject="mem2")

        # With default threshold (0.92), our 0.95 similarity should trigger
        result1 = audit_palace(checks=["duplicates"])
        assert len(result1["duplicates"]) == 1

        # With higher threshold (0.98), our 0.95 should not trigger
        result2 = audit_palace(
            checks=["duplicates"],
            thresholds={"duplicate_similarity": 0.98}
        )
        assert len(result2["duplicates"]) == 0

    def test_project_filter_applied_to_relevant_checks(self, db_session):
        """Project filter should be applied to checks that support it."""
        # Create memories in different projects
        _create_test_memory(db_session, subject="project A", project="project-a")
        _create_test_memory(
            db_session,
            subject="project B",
            project="project-b",
            embedding=None,
        )

        # Filter by project-b
        result = audit_palace(
            checks=["missing_embeddings"],
            project="project-b"
        )

        # Should only find missing embedding in project-b
        assert len(result["missing_embeddings"]) == 1


class TestFindStaleMemories:
    """Tests for _find_stale_memories() - stale memory detection."""

    def test_finds_memories_older_than_n_days_with_low_access_and_centrality(self, db_session):
        """Should find memories that are old, low access, AND low centrality."""
        old_date = datetime.now(timezone.utc) - timedelta(days=100)

        # Truly stale: old + low access + low centrality
        stale_mem = _create_test_memory(
            db_session,
            subject="stale memory",
            created_at=old_date,
            access_count=1,
            foundational=False,
        )

        result = _find_stale_memories(db_session, stale_days=90, stale_access_threshold=2)

        assert len(result) == 1
        assert result[0]["memory_id"] == stale_mem.id
        assert result[0]["subject"] == "stale memory"
        assert result[0]["age_days"] >= 90
        assert result[0]["access_count"] == 1
        assert result[0]["in_degree"] == 0

    def test_never_considers_foundational_memories_as_stale(self, db_session):
        """Foundational memories should NEVER be considered stale."""
        old_date = datetime.now(timezone.utc) - timedelta(days=200)

        # Old, low access, low centrality, BUT foundational
        _create_test_memory(
            db_session,
            subject="foundational but old",
            created_at=old_date,
            access_count=0,
            foundational=True,  # This should protect it
        )

        result = _find_stale_memories(db_session, stale_days=90, stale_access_threshold=2)

        # Should find nothing - foundational is immune
        assert len(result) == 0

    def test_respects_centrality_threshold(self, db_session):
        """Memories with high centrality should not be considered stale."""
        old_date = datetime.now(timezone.utc) - timedelta(days=100)

        # Old + low access but HIGH centrality
        high_centrality_mem = _create_test_memory(
            db_session,
            subject="high centrality",
            created_at=old_date,
            access_count=1,
            foundational=False,
        )

        # Create edges pointing TO this memory (increasing in-degree)
        for i in range(5):
            source = _create_test_memory(db_session, subject=f"source {i}")
            _create_test_edge(db_session, source.id, high_centrality_mem.id)

        result = _find_stale_memories(
            db_session,
            stale_days=90,
            stale_access_threshold=2,
            stale_centrality_threshold=3,
        )

        # Should not find it - in-degree of 5 exceeds threshold of 3
        assert len(result) == 0

    def test_limits_results(self, db_session):
        """Should respect limit parameter."""
        old_date = datetime.now(timezone.utc) - timedelta(days=100)

        # Create 10 stale memories
        for i in range(10):
            _create_test_memory(
                db_session,
                subject=f"stale {i}",
                created_at=old_date,
                access_count=0,
                foundational=False,
            )

        result = _find_stale_memories(db_session, stale_days=90, limit=5)

        # Should only return 5
        assert len(result) == 5


class TestFindOrphanEdges:
    """Tests for _find_orphan_edges() - orphan edge detection."""

    def test_finds_edges_pointing_to_archived_memories(self, db_session):
        """Should find edges where target is archived."""
        source = _create_test_memory(db_session, subject="source")
        target = _create_test_memory(db_session, subject="target", is_archived=True)
        edge = _create_test_edge(db_session, source.id, target.id, "relates_to")

        result = _find_orphan_edges(db_session)

        assert len(result) == 1
        assert result[0]["edge_id"] == edge.id
        assert result[0]["source"] == source.id
        assert result[0]["target"] == target.id
        assert "archived" in result[0]["reason"].lower()

    def test_includes_source_and_target_subjects(self, db_session):
        """Should include subject information for debugging."""
        source = _create_test_memory(db_session, subject="Active Memory")
        target = _create_test_memory(db_session, subject="Archived Memory", is_archived=True)
        _create_test_edge(db_session, source.id, target.id)

        result = _find_orphan_edges(db_session)

        assert result[0]["source_subject"] == "Active Memory"
        assert result[0]["target_subject"] == "Archived Memory"


class TestFindMissingEmbeddings:
    """Tests for _find_missing_embeddings() - missing embedding detection."""

    def test_finds_active_memories_with_null_embedding(self, db_session):
        """Should find active memories with NULL embedding."""
        mem_with_emb = _create_test_memory(db_session, subject="has embedding")
        mem_without = _create_test_memory(db_session, subject="no embedding", embedding=None)

        result = _find_missing_embeddings(db_session)

        assert len(result) == 1
        assert mem_without.id in result
        assert mem_with_emb.id not in result

    def test_ignores_archived_memories(self, db_session):
        """Should not include archived memories."""
        _create_test_memory(
            db_session,
            subject="archived, no embedding",
            embedding=None,
            is_archived=True,
        )

        result = _find_missing_embeddings(db_session)

        # Should find nothing - archived memories are excluded
        assert len(result) == 0

    def test_respects_project_filter(self, db_session):
        """Should filter by project when specified."""
        _create_test_memory(db_session, subject="project A", project="proj-a", embedding=None)
        _create_test_memory(db_session, subject="project B", project="proj-b", embedding=None)

        result = _find_missing_embeddings(db_session, project="proj-a")

        # Should only find 1 from proj-a
        assert len(result) == 1


class TestFindContradictions:
    """Tests for _find_contradictions() - contradiction detection."""

    def test_finds_edges_with_contradicts_relation_type(self, db_session):
        """Should find edges with relation_type='contradicts'."""
        mem1 = _create_test_memory(db_session, subject="Memory 1")
        mem2 = _create_test_memory(db_session, subject="Memory 2")
        edge = _create_test_edge(db_session, mem1.id, mem2.id, "contradicts")

        result = _find_contradictions(db_session)

        assert len(result) == 1
        assert result[0]["memory_id"] == mem1.id
        assert result[0]["contradicts"] == mem2.id
        assert result[0]["needs_resolution"] is True
        assert result[0]["edge_id"] == edge.id

    def test_includes_subjects_for_both_memories(self, db_session):
        """Should include subject info for debugging."""
        mem1 = _create_test_memory(db_session, subject="Claim A")
        mem2 = _create_test_memory(db_session, subject="Claim B")
        _create_test_edge(db_session, mem1.id, mem2.id, "contradicts")

        result = _find_contradictions(db_session)

        assert result[0]["subject_1"] == "Claim A"
        assert result[0]["subject_2"] == "Claim B"

    def test_ignores_other_relation_types(self, db_session):
        """Should only find contradicts edges, not other types."""
        mem1 = _create_test_memory(db_session, subject="Memory 1")
        mem2 = _create_test_memory(db_session, subject="Memory 2")
        _create_test_edge(db_session, mem1.id, mem2.id, "relates_to")
        _create_test_edge(db_session, mem1.id, mem2.id, "supports")

        result = _find_contradictions(db_session)

        # Should find nothing - no contradicts edges
        assert len(result) == 0


class TestFindUnlinkedMemories:
    """Tests for _find_unlinked_memories() - isolated node detection."""

    def test_finds_memories_with_no_edges(self, db_session):
        """Should find memories with no incoming or outgoing edges."""
        unlinked = _create_test_memory(db_session, subject="isolated")
        linked = _create_test_memory(db_session, subject="linked")
        other = _create_test_memory(db_session, subject="other")
        _create_test_edge(db_session, linked.id, other.id)

        result = _find_unlinked_memories(db_session)

        # Should only find the unlinked one
        assert len(result) == 1
        assert result[0]["memory_id"] == unlinked.id
        assert result[0]["subject"] == "isolated"

    def test_excludes_memories_with_outgoing_edges(self, db_session):
        """Memories with outgoing edges should not be considered unlinked."""
        mem1 = _create_test_memory(db_session, subject="source")
        mem2 = _create_test_memory(db_session, subject="target")
        _create_test_edge(db_session, mem1.id, mem2.id)

        result = _find_unlinked_memories(db_session)

        # mem1 has outgoing edge, mem2 has incoming edge - neither unlinked
        assert len(result) == 0

    def test_excludes_memories_with_incoming_edges(self, db_session):
        """Memories with incoming edges should not be considered unlinked."""
        mem1 = _create_test_memory(db_session, subject="source")
        mem2 = _create_test_memory(db_session, subject="target")
        _create_test_edge(db_session, mem1.id, mem2.id)

        result = _find_unlinked_memories(db_session)

        # Both have edges - neither unlinked
        assert len(result) == 0

    def test_respects_project_filter(self, db_session):
        """Should filter by project when specified."""
        _create_test_memory(db_session, subject="proj A isolated", project="proj-a")
        _create_test_memory(db_session, subject="proj B isolated", project="proj-b")

        result = _find_unlinked_memories(db_session, project="proj-a")

        # Should only find 1 from proj-a
        assert len(result) == 1


class TestReembedMemories:
    """Tests for reembed_memories() - re-embedding operation."""

    def test_dry_run_returns_preview(self, db_session):
        """dry_run=True should return preview without modifying anything."""
        mem = _create_test_memory(db_session, subject="test", embedding=None)

        result = reembed_memories(memory_ids=[mem.id], dry_run=True)

        assert "would_reembed" in result
        assert result["would_reembed"] == 1
        assert "note" in result
        assert "DRY RUN" in result["note"]

        # Memory should still have NULL embedding
        db_session.refresh(mem)
        assert mem.embedding is None

    def test_missing_only_filters_to_null_embeddings(self, db_session, mock_get_embedding):
        """missing_only=True should only process memories with NULL embedding."""
        mem_with = _create_test_memory(db_session, subject="has embedding")
        mem_without = _create_test_memory(db_session, subject="no embedding", embedding=None)

        result = reembed_memories(missing_only=True, dry_run=False)

        assert result["reembedded"] == 1
        assert result["failed"] == 0

        # Check that only mem_without was updated
        db_session.refresh(mem_without)
        assert mem_without.embedding == FAKE_EMBEDDING_JSON

    def test_processes_explicit_memory_ids_list(self, db_session, mock_get_embedding):
        """Should process explicit memory_ids list."""
        mem1 = _create_test_memory(db_session, subject="mem1")
        mem2 = _create_test_memory(db_session, subject="mem2")
        mem3 = _create_test_memory(db_session, subject="mem3")

        # Only reembed mem1 and mem3
        result = reembed_memories(memory_ids=[mem1.id, mem3.id], dry_run=False)

        assert result["reembedded"] == 2
        assert result["total"] == 2

    def test_all_memories_true_reembeds_everything(self, db_session, mock_get_embedding):
        """all_memories=True should re-embed all active memories."""
        for i in range(3):
            _create_test_memory(db_session, subject=f"mem {i}")

        result = reembed_memories(all_memories=True, dry_run=False)

        assert result["reembedded"] == 3

    def test_older_than_days_filters_by_age(self, db_session, mock_get_embedding):
        """older_than_days should filter by created_at."""
        old_date = datetime.now(timezone.utc) - timedelta(days=100)

        _create_test_memory(db_session, subject="old", created_at=old_date)
        _create_test_memory(db_session, subject="new")

        result = reembed_memories(older_than_days=50, dry_run=False)

        # Should only reembed the old one
        assert result["reembedded"] == 1

    def test_returns_success_and_failed_counts(self, db_session):
        """Should return success and failed counts."""
        mem1 = _create_test_memory(db_session, subject="mem1")
        mem2 = _create_test_memory(db_session, subject="mem2")

        # Mock get_embedding to fail for one memory
        def mock_embedding(text):
            if "mem1" in text:
                return FAKE_EMBEDDING_JSON
            return None  # Fail for mem2

        with patch("memory_palace.services.maintenance_service.get_embedding", side_effect=mock_embedding):
            result = reembed_memories(memory_ids=[mem1.id, mem2.id], dry_run=False)

        assert result["reembedded"] == 1
        assert result["failed"] == 1
        assert result["total"] == 2
        assert "failed_ids" in result


class TestBatchArchiveWrapper:
    """Tests for batch_archive_memories() - deprecated wrapper."""

    def test_delegates_to_archive_memory(self, db_session):
        """Should delegate to memory_service.archive_memory()."""
        mem = _create_test_memory(db_session, subject="test")

        result = batch_archive_memories(memory_ids=[mem.id], dry_run=True)

        # Should return preview with expected keys
        assert "would_archive" in result or "archived_count" in result

    def test_passes_all_parameters_through(self, db_session):
        """Should pass all parameters to archive_memory(), including centrality protection."""
        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        mem = _create_test_memory(db_session, subject="test", created_at=old_date, access_count=1)

        # Call with all parameters
        result = batch_archive_memories(
            older_than_days=50,
            max_access_count=5,
            memory_type="fact",
            project="life",
            centrality_protection=True,
            min_centrality_threshold=5,
            dry_run=True,
            reason="test reason",
        )

        # Should not error - parameters accepted
        assert "would_archive" in result or "archived_count" in result

    def test_centrality_protection_flows_through(self, db_session):
        """centrality_protection parameter must flow through to archive_memory()."""
        old_date = datetime.now(timezone.utc) - timedelta(days=100)

        # Create high-centrality memory
        high_centrality = _create_test_memory(
            db_session,
            subject="high centrality",
            created_at=old_date,
            access_count=1,
        )

        # Create edges pointing to it
        for i in range(6):
            source = _create_test_memory(db_session, subject=f"source {i}")
            _create_test_edge(db_session, source.id, high_centrality.id)

        # With centrality_protection=True, should skip it
        result = batch_archive_memories(
            older_than_days=50,
            max_access_count=5,
            centrality_protection=True,
            min_centrality_threshold=5,
            dry_run=True,
        )

        # Should be skipped due to centrality (dry_run returns would_skip_centrality)
        assert "would_skip_centrality" in result, "dry_run response should include would_skip_centrality"
        assert result["would_skip_centrality"] == 1

        # Verify the skipped memory is the high-centrality one
        assert "details" in result
        assert "skipped_centrality" in result["details"]
        assert len(result["details"]["skipped_centrality"]) == 1
        assert result["details"]["skipped_centrality"][0]["id"] == high_centrality.id

    def test_max_access_count_flows_through(self, db_session):
        """max_access_count parameter must flow through correctly."""
        old_date = datetime.now(timezone.utc) - timedelta(days=100)

        low_access = _create_test_memory(
            db_session,
            subject="low access",
            created_at=old_date,
            access_count=2,
        )
        high_access = _create_test_memory(
            db_session,
            subject="high access",
            created_at=old_date,
            access_count=10,
        )

        result = batch_archive_memories(
            older_than_days=50,
            max_access_count=5,
            dry_run=True,
        )

        # Should only include low_access (access_count=2 <= 5)
        assert "would_archive" in result, "dry_run response must include would_archive"
        assert result["would_archive"] == 1, f"Expected 1 memory to archive, got {result['would_archive']}"

        assert "details" in result, "dry_run response must include details"
        assert "would_archive" in result["details"], "details must include would_archive list"
        archived_ids = [m["id"] for m in result["details"]["would_archive"]]
        assert low_access.id in archived_ids, "low_access memory should be in archive list"
        assert high_access.id not in archived_ids, "high_access memory should NOT be in archive list"


class TestMoltbookExtension:
    """Tests for Moltbook Gateway extension."""

    @pytest.mark.parametrize("attr_name", ["mcp", "moltbook_submit", "moltbook_qc"])
    def test_server_exports_callable(self, attr_name):
        """Smoke test: moltbook server module exports expected callables."""
        ext_path = str(Path(__file__).parent.parent / "extensions" / "moltbook-gateway")
        sys.path.insert(0, ext_path)
        try:
            import server as moltbook_server
            assert hasattr(moltbook_server, attr_name), f"server missing {attr_name}"
            # mcp is a FastMCP instance, others are functions
            if attr_name != "mcp":
                assert callable(getattr(moltbook_server, attr_name)), f"server.{attr_name} not callable"
        except ImportError as e:
            # moltbook_tools or mcp SDK may not be installed in test env
            if "moltbook_tools" in str(e) or "mcp" in str(e):
                pytest.skip(f"dependency not installed: {e}")
            raise
        finally:
            # Clean up sys.path
            if ext_path in sys.path:
                sys.path.remove(ext_path)


class TestToonConverterExtension:
    """Tests for TOON Converter extension."""

    @pytest.mark.parametrize("module_name,attr_name", [
        ("converter", "convert_file"),
        ("converter", "format_size"),
    ])
    def test_converter_exports_callable(self, module_name, attr_name):
        """Smoke test: converter module exports expected functions."""
        ext_path = str(Path(__file__).parent.parent / "extensions" / "toon-converter")
        sys.path.insert(0, ext_path)
        try:
            mod = importlib.import_module(module_name)
            assert hasattr(mod, attr_name), f"{module_name} missing {attr_name}"
            assert callable(getattr(mod, attr_name)), f"{module_name}.{attr_name} not callable"
        finally:
            if ext_path in sys.path:
                sys.path.remove(ext_path)

    def test_cli_module_has_argparse_setup(self):
        """CLI module should have proper argparse setup."""
        # CLI uses relative imports which don't work when imported directly
        # Just verify the file exists and has the expected structure
        cli_path = Path(__file__).parent.parent / "extensions" / "toon-converter" / "cli.py"
        assert cli_path.exists(), "CLI module should exist"

        # Read and verify it has main function
        content = cli_path.read_text()
        assert "def main():" in content
        assert "argparse" in content

    def test_server_module_defines_mcp_tool(self):
        """Server module should define the MCP tool."""
        # Load toon-converter server separately to avoid moltbook-gateway imports
        ext_path = str(Path(__file__).parent.parent / "extensions" / "toon-converter")

        # Check we're not accidentally loading moltbook server
        spec = importlib.util.spec_from_file_location(
            "toon_server",
            Path(ext_path) / "server.py"
        )
        toon_server = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(toon_server)
            assert hasattr(toon_server, "mcp")
            assert hasattr(toon_server, "convert_jsonl_to_toon")
            assert callable(toon_server.convert_jsonl_to_toon)
        except (ImportError, SystemExit) as e:
            # MCP SDK or converter deps may not be installed in test env
            pytest.skip(f"server dependencies not available: {e}")


class TestCrossProjectAutoLinks:
    """Tests for cross-project auto-link detection and cleanup."""

    def test_find_cross_project_auto_links_finds_spanning_edges(self, db_session):
        """Should find auto-linked edges that span different projects."""
        # Create memories in different projects
        mem_a = _create_test_memory(db_session, subject="Memory A", project="project-a")
        mem_b = _create_test_memory(db_session, subject="Memory B", project="project-b")

        # Create an auto-linked edge between them
        edge = _create_test_edge(
            db_session,
            mem_a.id,
            mem_b.id,
            relation_type="relates_to",
            edge_metadata={"auto_linked": True}
        )

        result = _find_cross_project_auto_links(db_session)

        # Should find the cross-project auto-link
        assert len(result) == 1
        assert result[0]["edge_id"] == edge.id
        assert result[0]["source_id"] == mem_a.id
        assert result[0]["target_id"] == mem_b.id
        assert result[0]["source_subject"] == "Memory A"
        assert result[0]["target_subject"] == "Memory B"
        assert "project-a" in result[0]["source_projects"]
        assert "project-b" in result[0]["target_projects"]

    def test_find_cross_project_auto_links_ignores_same_project(self, db_session):
        """Should NOT find auto-linked edges within the same project."""
        # Create memories in the same project
        mem_a = _create_test_memory(db_session, subject="Memory A", project="life")
        mem_b = _create_test_memory(db_session, subject="Memory B", project="life")

        # Create an auto-linked edge between them
        _create_test_edge(
            db_session,
            mem_a.id,
            mem_b.id,
            relation_type="relates_to",
            edge_metadata={"auto_linked": True}
        )

        result = _find_cross_project_auto_links(db_session)

        # Should NOT find it - same project
        assert len(result) == 0

    def test_find_cross_project_auto_links_ignores_manual_links(self, db_session):
        """Should NOT find manually created edges (no auto_linked metadata)."""
        # Create memories in different projects
        mem_a = _create_test_memory(db_session, subject="Memory A", project="project-a")
        mem_b = _create_test_memory(db_session, subject="Memory B", project="project-b")

        # Create a manual edge (no auto_linked metadata)
        _create_test_edge(
            db_session,
            mem_a.id,
            mem_b.id,
            relation_type="relates_to"
        )

        result = _find_cross_project_auto_links(db_session)

        # Should NOT find it - not auto-linked
        assert len(result) == 0

    def test_cleanup_dry_run_returns_preview(self, db_session):
        """dry_run=True should return preview without deleting."""
        # Create memories in different projects
        mem_a = _create_test_memory(db_session, subject="Memory A", project="project-a")
        mem_b = _create_test_memory(db_session, subject="Memory B", project="project-b")

        # Create an auto-linked edge
        edge = _create_test_edge(
            db_session,
            mem_a.id,
            mem_b.id,
            relation_type="relates_to",
            edge_metadata={"auto_linked": True}
        )

        result = cleanup_cross_project_auto_links(dry_run=True)

        # Should return preview
        assert "would_remove" in result
        assert result["would_remove"] == 1
        assert "note" in result
        assert "DRY RUN" in result["note"]

        # Edge should still exist
        db_session.refresh(edge)
        assert edge is not None

    def test_cleanup_executes_deletion(self, db_session):
        """dry_run=False should actually remove the edges."""
        # Create memories in different projects
        mem_a = _create_test_memory(db_session, subject="Memory A", project="project-a")
        mem_b = _create_test_memory(db_session, subject="Memory B", project="project-b")

        # Create an auto-linked edge
        edge = _create_test_edge(
            db_session,
            mem_a.id,
            mem_b.id,
            relation_type="relates_to",
            edge_metadata={"auto_linked": True}
        )

        edge_id = edge.id

        result = cleanup_cross_project_auto_links(dry_run=False)

        # Should report removal
        assert result["removed"] == 1
        assert "log_path" in result

        # Edge should be deleted
        deleted_edge = db_session.query(MemoryEdge).filter(MemoryEdge.id == edge_id).first()
        assert deleted_edge is None

    def test_audit_includes_cross_project_check(self, db_session):
        """audit_palace with cross_project_auto_links check should return results."""
        # Create memories in different projects
        mem_a = _create_test_memory(db_session, subject="Memory A", project="project-a")
        mem_b = _create_test_memory(db_session, subject="Memory B", project="project-b")

        # Create an auto-linked edge
        _create_test_edge(
            db_session,
            mem_a.id,
            mem_b.id,
            relation_type="relates_to",
            edge_metadata={"auto_linked": True}
        )

        result = audit_palace(checks=["cross_project_auto_links"])

        # Should have cross_project_auto_links in results
        assert "cross_project_auto_links" in result
        assert len(result["cross_project_auto_links"]) == 1
        assert result["summary"]["cross_project_auto_links_found"] == 1


class TestMultiProjectSupport:
    """Tests for multi-project memory handling in maintenance functions."""

    def test_reembed_filters_by_project_with_array_column(self, db_session, mock_get_embedding):
        """reembed_memories with project filter should work with projects array."""
        # Create memories in different projects
        mem_life = _create_test_memory(
            db_session,
            subject="life memory",
            project="life",
            embedding=None
        )
        mem_proj = _create_test_memory(
            db_session,
            subject="project memory",
            project="test-project",
            embedding=None
        )

        result = reembed_memories(project="life", dry_run=False)

        # Should only reembed the life memory
        assert result["reembedded"] == 1

        # Verify which memory was updated
        db_session.refresh(mem_life)
        db_session.refresh(mem_proj)
        assert mem_life.embedding == FAKE_EMBEDDING_JSON
        assert mem_proj.embedding is None

    def test_audit_with_project_filter_uses_contains(self, db_session):
        """audit_palace with project filter should find memories containing that project."""
        # Create memories with different projects
        mem_life = _create_test_memory(
            db_session,
            subject="life memory",
            project="life",
            embedding=None
        )
        mem_proj = _create_test_memory(
            db_session,
            subject="project memory",
            project="test-project",
            embedding=None
        )

        result = audit_palace(
            checks=["missing_embeddings"],
            project="life"
        )

        # Should only find missing embedding in life project
        assert len(result["missing_embeddings"]) == 1
        assert mem_life.id in result["missing_embeddings"]
        assert mem_proj.id not in result["missing_embeddings"]

    def test_multi_project_memory_found_by_any_project_filter(self, db_session):
        """Memory with multiple projects should be found by filtering on any of them."""
        # Create a memory that belongs to multiple projects
        # Note: The Memory model should handle list conversion automatically
        mem_multi = _create_test_memory(
            db_session,
            subject="multi-project memory",
            project=["life", "work"],  # Pass as list
            embedding=None
        )

        # Filter by "life" - should find it
        result_life = audit_palace(
            checks=["missing_embeddings"],
            project="life"
        )
        assert mem_multi.id in result_life["missing_embeddings"]

        # Filter by "work" - should also find it
        result_work = audit_palace(
            checks=["missing_embeddings"],
            project="work"
        )
        assert mem_multi.id in result_work["missing_embeddings"]


class TestToolRegistration:
    """Tests for tool registration in mcp_server/tools/__init__.py."""

    @pytest.mark.parametrize("module_path,attr_name", [
        ("mcp_server.tools", "register_remember"),
        ("mcp_server.tools", "register_recall"),
        ("mcp_server.tools", "register_get_memory"),
        ("mcp_server.tools", "register_recent"),
        ("mcp_server.tools", "register_archive"),
        ("mcp_server.tools", "register_link"),
        ("mcp_server.tools", "register_unlink"),
        ("mcp_server.tools", "register_message"),
        ("mcp_server.tools", "register_code_remember"),
        ("mcp_server.tools", "register_audit"),
        ("mcp_server.tools", "register_reembed"),
        ("mcp_server.tools", "register_memory_stats"),
        ("mcp_server.tools", "register_reflect"),
    ])
    def test_module_exports_callable(self, module_path, attr_name):
        """Smoke test: verify all expected modules and functions exist."""
        import importlib
        mod = importlib.import_module(module_path)
        assert hasattr(mod, attr_name), f"{module_path} missing {attr_name}"
        assert callable(getattr(mod, attr_name)), f"{module_path}.{attr_name} not callable"

    def test_register_all_tools(self):
        """register_all_tools calls all 13 registration functions."""
        mock_mcp = MagicMock()
        from mcp_server.tools import register_all_tools

        # Should not raise any errors
        register_all_tools(mock_mcp)

        # Verify mcp.tool was called for each of the 13 tools
        # (Each registration function decorates with @mcp.tool())
        assert mock_mcp.tool.call_count == 13, f"Expected 13 tool registrations, got {mock_mcp.tool.call_count}"
