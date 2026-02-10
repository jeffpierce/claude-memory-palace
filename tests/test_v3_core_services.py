"""
Tests for Memory Palace v2.0 Core Services (Memory Service and Graph Service).

Tests the remember(), recall(), get_memory_by_id(), archive_memory(), forget(),
link_memories(), and unlink_memories() functions with in-memory SQLite and mocked
external dependencies.
"""
import sys
import importlib
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from typing import List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# ── Early Patching ───────────────────────────────────────────────────


# CRITICAL: We need to patch is_postgres() before any models are imported
# because models_v3 sets _USE_PG_TYPES at module import time.

# First, ensure config_v2 is imported and patched
import memory_palace.config_v2
with patch.object(memory_palace.config_v2, 'is_postgres', return_value=False):
    # Reload models_v3 if it's already loaded (to pick up the new is_postgres value)
    if 'memory_palace.models_v3' in sys.modules:
        del sys.modules['memory_palace.models_v3']
    if 'memory_palace.models' in sys.modules:
        del sys.modules['memory_palace.models']
    if 'memory_palace.database_v3' in sys.modules:
        del sys.modules['memory_palace.database_v3']
    if 'memory_palace.database' in sys.modules:
        del sys.modules['memory_palace.database']

    # Now import with SQLite mode
    import memory_palace.database_v3 as db_module
    from memory_palace.models_v3 import Base, Memory, MemoryEdge


# ── Fixtures ─────────────────────────────────────────────────────────


# Fake embeddings for deterministic testing
FAKE_EMBEDDING_A = [0.1] * 768  # 768-dim like nomic-embed-text
FAKE_EMBEDDING_B = [0.2] * 768
FAKE_EMBEDDING_C = [0.9] * 768  # Very different


def _fake_cosine(a: List[float], b: List[float]) -> float:
    """Return high similarity for identical embeddings, low for different."""
    if a == b:
        return 0.95
    # Simple heuristic based on first element difference
    diff = abs(a[0] - b[0])
    if diff < 0.2:
        return 0.85  # Similar
    elif diff < 0.5:
        return 0.60  # Moderately similar
    else:
        return 0.30  # Different


@pytest.fixture(autouse=True)
def test_db():
    """Set up in-memory SQLite database for each test."""
    # Ensure is_postgres returns False for the database setup
    with patch("memory_palace.config_v2.is_postgres", return_value=False):
        # Reset module-level singletons
        db_module.reset_engine()

        # Create in-memory engine with StaticPool so all connections share state
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)

        # Monkey-patch the database module's engine and session factory
        old_engine = db_module._engine
        old_session = db_module._SessionLocal
        db_module._engine = engine
        db_module._SessionLocal = Session

        yield engine, Session

        # Cleanup
        db_module._engine = old_engine
        db_module._SessionLocal = old_session
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock embedding generation and similarity calculation."""
    import json

    def get_embedding_mock(text: str):
        """Return deterministic fake embedding based on text content."""
        # Use text hash to decide which embedding to return
        if "similar" in text.lower() or "alpha" in text.lower():
            embedding = FAKE_EMBEDDING_A.copy()
        elif "related" in text.lower() or "beta" in text.lower():
            embedding = FAKE_EMBEDDING_B.copy()
        elif "different" in text.lower() or "gamma" in text.lower():
            embedding = FAKE_EMBEDDING_C.copy()
        else:
            # Default to EMBEDDING_A for generic content
            embedding = FAKE_EMBEDDING_A.copy()

        # For SQLite, embeddings are stored as JSON strings in Text columns
        # Serialize to JSON string to match production behavior
        return json.dumps(embedding)

    def cosine_similarity_mock(a, b):
        """Deserialize JSON strings and compute similarity."""
        import json
        # Handle both list and JSON string inputs
        if isinstance(a, str):
            a = json.loads(a)
        if isinstance(b, str):
            b = json.loads(b)
        return _fake_cosine(a, b)

    with patch("memory_palace.services.memory_service.get_embedding", side_effect=get_embedding_mock):
        with patch("memory_palace.services.memory_service.cosine_similarity", side_effect=cosine_similarity_mock):
            yield


@pytest.fixture(autouse=True)
def mock_config():
    """Mock configuration functions."""
    with patch("memory_palace.services.memory_service.get_instances", return_value=["test", "code", "desktop"]):
        with patch("memory_palace.services.memory_service.get_auto_link_config", return_value={
            "enabled": False,  # Disable auto-link by default, enable in specific tests
            "link_threshold": 0.75,
            "suggest_threshold": 0.675,
            "max_suggestions": 10,
            "same_project_only": True,
            "classify_edges": False,
        }):
            with patch("memory_palace.services.memory_service.is_postgres", return_value=False):
                yield


@pytest.fixture(autouse=True)
def mock_llm():
    """Mock LLM synthesis for recall()."""
    with patch("memory_palace.services.memory_service._synthesize_memories_with_llm", return_value="Mock synthesis"):
        yield


# ── Test Classes ─────────────────────────────────────────────────────


class TestRemember:
    """Tests for remember() — storing memories."""

    def test_stores_memory_and_returns_id(self):
        """Stores a memory and returns id, subject, embedded status."""
        from memory_palace.services.memory_service import remember

        result = remember(
            instance_id="test",
            memory_type="fact",
            content="The capital of France is Paris.",
            subject="France capital"
        )

        assert "id" in result
        assert result["subject"] == "France capital"
        assert result["embedded"] is True

    def test_sets_foundational_true_when_passed(self):
        """Sets foundational=True when passed."""
        from memory_palace.services.memory_service import remember
        from memory_palace.database import get_session

        result = remember(
            instance_id="test",
            memory_type="core_principle",
            content="Always validate user input.",
            foundational=True
        )

        # Verify in database
        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == result["id"]).first()
            assert memory.foundational is True
        finally:
            db.close()

    def test_rejects_invalid_source_type(self):
        """Rejects invalid source_type."""
        from memory_palace.services.memory_service import remember

        result = remember(
            instance_id="test",
            memory_type="fact",
            content="Test content",
            source_type="invalid_type"
        )

        assert "error" in result
        assert "Invalid source_type" in result["error"]

    def test_warns_unconfigured_instance_id(self):
        """Warns (but allows) unconfigured instance_id."""
        from memory_palace.services.memory_service import remember

        result = remember(
            instance_id="unconfigured_instance",
            memory_type="fact",
            content="Test content"
        )

        assert "id" in result  # Still succeeds
        assert "instance_warning" in result
        assert "unconfigured_instance" in result["instance_warning"]

    def test_creates_supersedes_edge_and_archives_old(self):
        """Creates supersedes edge + archives old memory when supersedes_id is set."""
        from memory_palace.services.memory_service import remember
        from memory_palace.database import get_session

        # Create old memory
        old_result = remember(
            instance_id="test",
            memory_type="fact",
            content="Old version of the fact.",
            subject="Old fact"
        )
        old_id = old_result["id"]

        # Create new memory that supersedes old one
        new_result = remember(
            instance_id="test",
            memory_type="fact",
            content="New updated version of the fact.",
            subject="New fact",
            supersedes_id=old_id
        )

        assert "links_created" in new_result
        assert len(new_result["links_created"]) == 1
        assert new_result["links_created"][0]["type"] == "supersedes"
        assert new_result["links_created"][0]["target"] == old_id
        assert new_result["links_created"][0]["archived_old"] is True

        # Verify old memory is archived
        db = get_session()
        try:
            old_memory = db.query(Memory).filter(Memory.id == old_id).first()
            assert old_memory.is_archived is True
            assert f"SUPERSEDED by #{new_result['id']}" in old_memory.source_context
        finally:
            db.close()

    def test_auto_links_when_similar_memories_exist(self):
        """Auto-links when similar memories exist (project-scoped)."""
        from memory_palace.services.memory_service import remember
        from memory_palace.database import get_session

        # Enable auto-link by patching the config for this test
        with patch("memory_palace.services.memory_service.get_auto_link_config", return_value={
            "enabled": True,
            "link_threshold": 0.75,
            "suggest_threshold": 0.675,
            "max_suggestions": 10,
            "same_project_only": True,
            "classify_edges": False,
        }):
            # Create first memory with "similar" keyword to get FAKE_EMBEDDING_A
            first = remember(
                instance_id="test",
                memory_type="fact",
                content="This is similar content alpha.",
                subject="Similar fact alpha",
                project="test-project"
            )

            # Create second similar memory with "similar" keyword
            second = remember(
                instance_id="test",
                memory_type="fact",
                content="This is similar content alpha too.",
                subject="Similar fact alpha two",
                project="test-project",
                auto_link=True  # Explicitly enable
            )

            # Should have auto-created link
            assert "links_created" in second
            assert len(second["links_created"]) > 0
            assert second["links_created"][0]["target"] == first["id"]

            # Verify edge exists in database
            db = get_session()
            try:
                edge = db.query(MemoryEdge).filter(
                    MemoryEdge.source_id == second["id"],
                    MemoryEdge.target_id == first["id"]
                ).first()
                assert edge is not None
            finally:
                db.close()

    def test_does_not_auto_link_across_projects(self):
        """Does NOT auto-link across projects when same_project_only=True."""
        from memory_palace.services.memory_service import remember
        from memory_palace.database import get_session

        # Enable auto-link with same_project_only
        with patch("memory_palace.services.memory_service.get_auto_link_config", return_value={
            "enabled": True,
            "link_threshold": 0.75,
            "suggest_threshold": 0.675,
            "max_suggestions": 10,
            "same_project_only": True,
            "classify_edges": False,
        }):
            # Create memory in project A
            first = remember(
                instance_id="test",
                memory_type="fact",
                content="This is similar content alpha.",
                subject="Similar fact alpha",
                project="project-A"
            )

            # Create similar memory in project B
            second = remember(
                instance_id="test",
                memory_type="fact",
                content="This is similar content alpha too.",
                subject="Similar fact alpha two",
                project="project-B",
                auto_link=True
            )

            # Should NOT have auto-created links (different projects)
            assert "links_created" not in second or len(second.get("links_created", [])) == 0

            # Verify no edge exists
            db = get_session()
            try:
                edge = db.query(MemoryEdge).filter(
                    MemoryEdge.source_id == second["id"],
                    MemoryEdge.target_id == first["id"]
                ).first()
                assert edge is None
            finally:
                db.close()


class TestRecall:
    """Tests for recall() — searching memories."""

    def test_returns_results_with_semantic_search(self):
        """Returns results with semantic search when embedding available."""
        from memory_palace.services.memory_service import remember, recall

        # Create memories with different embeddings
        remember(
            instance_id="test",
            memory_type="fact",
            content="This is similar alpha content.",
            subject="Alpha"
        )
        remember(
            instance_id="test",
            memory_type="fact",
            content="This is different gamma content.",
            subject="Gamma"
        )

        # Search for "similar" should match alpha better
        result = recall(query="similar alpha", synthesize=False)

        assert result["count"] > 0
        assert result["search_method"].startswith("semantic")
        assert len(result["memories"]) > 0

    def test_falls_back_to_keyword_search(self):
        """Falls back to keyword search when embedding fails."""
        from memory_palace.services.memory_service import remember, recall

        # Mock embedding failure
        with patch("memory_palace.services.memory_service.get_embedding", return_value=None):
            remember(
                instance_id="test",
                memory_type="fact",
                content="This content has no embedding.",
                subject="No embedding"
            )

            result = recall(query="content", synthesize=False)

            assert "keyword" in result["search_method"].lower()

    def test_filters_by_project_single_string(self):
        """Filters by project (single string)."""
        from memory_palace.services.memory_service import remember, recall

        remember(
            instance_id="test",
            memory_type="fact",
            content="Content in project A.",
            project="project-A"
        )
        remember(
            instance_id="test",
            memory_type="fact",
            content="Content in project B.",
            project="project-B"
        )

        result = recall(query="content", project="project-A", synthesize=False)

        assert result["count"] == 1
        assert result["memories"][0]["project"] == "project-A"

    def test_filters_by_project_list(self):
        """Filters by project (list of strings)."""
        from memory_palace.services.memory_service import remember, recall

        remember(instance_id="test", memory_type="fact", content="A", project="project-A")
        remember(instance_id="test", memory_type="fact", content="B", project="project-B")
        remember(instance_id="test", memory_type="fact", content="C", project="project-C")

        result = recall(query="content", project=["project-A", "project-B"], synthesize=False)

        assert result["count"] == 2
        projects = {m["project"] for m in result["memories"]}
        assert projects == {"project-A", "project-B"}

    def test_filters_by_memory_type_with_wildcard(self):
        """Filters by memory_type with wildcard (e.g., 'code_*' → SQL LIKE 'code_%')."""
        from memory_palace.services.memory_service import remember, recall

        remember(instance_id="test", memory_type="code_function", content="Function A")
        remember(instance_id="test", memory_type="code_class", content="Class B")
        remember(instance_id="test", memory_type="fact", content="Fact C")

        result = recall(query="content", memory_type="code_*", synthesize=False)

        assert result["count"] == 2
        types = {m["memory_type"] for m in result["memories"]}
        assert types == {"code_function", "code_class"}

    def test_filters_by_min_foundational(self):
        """Filters by min_foundational=True."""
        from memory_palace.services.memory_service import remember, recall

        remember(instance_id="test", memory_type="fact", content="Regular", foundational=False)
        remember(instance_id="test", memory_type="core", content="Foundational", foundational=True)

        result = recall(query="content", min_foundational=True, synthesize=False)

        assert result["count"] == 1
        assert result["memories"][0]["foundational"] is True

    def test_includes_graph_context_in_results(self):
        """Includes graph context in results."""
        from memory_palace.services.memory_service import remember, recall
        from memory_palace.services.graph_service import link_memories

        # Create two memories and link them
        m1 = remember(instance_id="test", memory_type="fact", content="Memory 1", subject="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="Memory 2", subject="M2")
        link_memories(m1["id"], m2["id"], "relates_to")

        result = recall(query="Memory", include_graph=True, synthesize=False)

        assert "graph_context" in result
        assert "edges" in result["graph_context"]
        assert len(result["graph_context"]["edges"]) > 0

    def test_updates_access_count(self):
        """Updates access_count on retrieved memories."""
        from memory_palace.services.memory_service import remember, recall
        from memory_palace.database import get_session

        m = remember(instance_id="test", memory_type="fact", content="Test", subject="Test")

        # Initial access_count should be 0
        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m["id"]).first()
            initial_count = memory.access_count
        finally:
            db.close()

        # Recall should increment access_count
        recall(query="Test", synthesize=False)

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m["id"]).first()
            assert memory.access_count == initial_count + 1
        finally:
            db.close()

    def test_returns_raw_memories_when_synthesize_false(self):
        """Returns raw memories when synthesize=False."""
        from memory_palace.services.memory_service import remember, recall

        remember(instance_id="test", memory_type="fact", content="Raw test")

        result = recall(query="Raw", synthesize=False)

        assert "memories" in result
        assert "summary" not in result
        assert isinstance(result["memories"], list)


class TestGetMemoryById:
    """Tests for get_memory_by_id() — single memory retrieval."""

    def test_returns_memory_dict_with_graph_context(self):
        """Returns memory dict with graph context."""
        from memory_palace.services.memory_service import remember, get_memory_by_id
        from memory_palace.services.graph_service import link_memories

        m1 = remember(instance_id="test", memory_type="fact", content="Memory 1", subject="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="Memory 2", subject="M2")
        link_memories(m1["id"], m2["id"], "relates_to")

        result = get_memory_by_id(m1["id"], include_graph=True, graph_depth=1)

        assert "memory" in result
        assert result["memory"]["id"] == m1["id"]
        assert "graph_context" in result

    def test_updates_access_tracking(self):
        """Updates access tracking."""
        from memory_palace.services.memory_service import remember, get_memory_by_id
        from memory_palace.database import get_session

        m = remember(instance_id="test", memory_type="fact", content="Test")

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m["id"]).first()
            initial_count = memory.access_count
            initial_accessed = memory.last_accessed_at
        finally:
            db.close()

        get_memory_by_id(m["id"])

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m["id"]).first()
            assert memory.access_count == initial_count + 1
            assert memory.last_accessed_at is not None
            if initial_accessed:
                assert memory.last_accessed_at > initial_accessed
        finally:
            db.close()

    def test_returns_none_for_nonexistent_id(self):
        """Returns None for non-existent ID."""
        from memory_palace.services.memory_service import get_memory_by_id

        result = get_memory_by_id(99999)

        assert result is None

    def test_graph_depth_parameter_works(self):
        """Graph depth parameter works (0 = no graph, 1+ = context)."""
        from memory_palace.services.memory_service import remember, get_memory_by_id
        from memory_palace.services.graph_service import link_memories

        m1 = remember(instance_id="test", memory_type="fact", content="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="M2")
        link_memories(m1["id"], m2["id"], "relates_to")

        # graph_depth=0 should not include graph
        result_no_graph = get_memory_by_id(m1["id"], include_graph=True, graph_depth=0)
        assert "graph_context" not in result_no_graph

        # graph_depth=1 should include graph
        result_with_graph = get_memory_by_id(m1["id"], include_graph=True, graph_depth=1)
        assert "graph_context" in result_with_graph


class TestArchiveMemory:
    """Tests for archive_memory() — unified archival."""

    def test_archives_by_explicit_id_list(self):
        """Archives by explicit ID list."""
        from memory_palace.services.memory_service import remember, archive_memory
        from memory_palace.database import get_session

        m1 = remember(instance_id="test", memory_type="fact", content="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="M2")

        result = archive_memory(memory_ids=[m1["id"], m2["id"]], dry_run=False)

        assert result["archived_count"] == 2

        db = get_session()
        try:
            mem1 = db.query(Memory).filter(Memory.id == m1["id"]).first()
            mem2 = db.query(Memory).filter(Memory.id == m2["id"]).first()
            assert mem1.is_archived is True
            assert mem2.is_archived is True
        finally:
            db.close()

    def test_archives_by_filter_criteria(self):
        """Archives by filter criteria (older_than_days, max_access_count, project, memory_type)."""
        from memory_palace.services.memory_service import remember, archive_memory
        from memory_palace.database import get_session

        # Create old memory
        m_old = remember(instance_id="test", memory_type="fact", content="Old")
        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m_old["id"]).first()
            memory.created_at = datetime.now(timezone.utc) - timedelta(days=100)
            db.commit()
        finally:
            db.close()

        # Create recent memory
        m_new = remember(instance_id="test", memory_type="fact", content="New")

        result = archive_memory(older_than_days=50, dry_run=False)

        assert result["archived_count"] == 1
        assert result["details"]["archived"][0]["id"] == m_old["id"]

    def test_skips_foundational_memories(self):
        """SKIPS foundational memories (never archives them)."""
        from memory_palace.services.memory_service import remember, archive_memory
        from memory_palace.database import get_session

        m_foundational = remember(
            instance_id="test",
            memory_type="core",
            content="Foundational",
            foundational=True
        )

        result = archive_memory(memory_ids=[m_foundational["id"]], dry_run=False)

        assert result["archived_count"] == 0
        assert result["skipped_foundational_count"] == 1

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m_foundational["id"]).first()
            assert memory.is_archived is False
        finally:
            db.close()

    def test_skips_high_centrality_memories(self):
        """SKIPS high-centrality memories when centrality_protection=True."""
        from memory_palace.services.memory_service import remember, archive_memory
        from memory_palace.services.graph_service import link_memories
        from memory_palace.database import get_session

        # Create target memory
        target = remember(instance_id="test", memory_type="fact", content="Target")

        # Create 5 memories that link TO target (giving it in-degree of 5)
        for i in range(5):
            source = remember(instance_id="test", memory_type="fact", content=f"Source {i}")
            link_memories(source["id"], target["id"], "relates_to")

        result = archive_memory(
            memory_ids=[target["id"]],
            centrality_protection=True,
            min_centrality_threshold=5,
            dry_run=False
        )

        assert result["archived_count"] == 0
        assert result["skipped_centrality_count"] == 1

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == target["id"]).first()
            assert memory.is_archived is False
        finally:
            db.close()

    def test_dry_run_returns_preview_without_archiving(self):
        """dry_run=True returns preview without archiving."""
        from memory_palace.services.memory_service import remember, archive_memory
        from memory_palace.database import get_session

        m = remember(instance_id="test", memory_type="fact", content="Test")

        result = archive_memory(memory_ids=[m["id"]], dry_run=True)

        assert "would_archive" in result
        assert result["would_archive"] == 1
        assert "note" in result
        assert "DRY RUN" in result["note"]

        # Verify NOT archived in database
        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m["id"]).first()
            assert memory.is_archived is False
        finally:
            db.close()

    def test_dry_run_false_actually_archives(self):
        """dry_run=False actually archives."""
        from memory_palace.services.memory_service import remember, archive_memory
        from memory_palace.database import get_session

        m = remember(instance_id="test", memory_type="fact", content="Test")

        result = archive_memory(memory_ids=[m["id"]], dry_run=False)

        assert result["archived_count"] == 1

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m["id"]).first()
            assert memory.is_archived is True
        finally:
            db.close()

    def test_adds_reason_to_source_context(self):
        """Adds reason to source_context when provided."""
        from memory_palace.services.memory_service import remember, archive_memory
        from memory_palace.database import get_session

        m = remember(instance_id="test", memory_type="fact", content="Test")

        archive_memory(
            memory_ids=[m["id"]],
            reason="No longer relevant",
            dry_run=False
        )

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m["id"]).first()
            assert "ARCHIVED: No longer relevant" in memory.source_context
        finally:
            db.close()


class TestForget:
    """Tests for forget() — legacy wrapper."""

    def test_delegates_to_archive_memory_correctly(self):
        """Delegates to archive_memory correctly."""
        from memory_palace.services.memory_service import remember, forget
        from memory_palace.database import get_session

        m = remember(instance_id="test", memory_type="fact", content="Test")

        result = forget(m["id"], reason="Test forget")

        assert "message" in result
        assert f"Archived memory {m['id']}" in result["message"]

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == m["id"]).first()
            assert memory.is_archived is True
        finally:
            db.close()

    def test_returns_error_for_foundational_memories(self):
        """Returns error for foundational memories."""
        from memory_palace.services.memory_service import remember, forget

        m = remember(
            instance_id="test",
            memory_type="core",
            content="Foundational",
            foundational=True
        )

        result = forget(m["id"])

        assert "error" in result
        assert "foundational" in result["error"].lower()

    def test_returns_error_for_centrality_protected_memories(self):
        """Returns error for centrality-protected memories."""
        from memory_palace.services.memory_service import remember, forget
        from memory_palace.services.graph_service import link_memories

        target = remember(instance_id="test", memory_type="fact", content="Target")

        # Give it high centrality (5+ incoming edges)
        for i in range(5):
            source = remember(instance_id="test", memory_type="fact", content=f"Source {i}")
            link_memories(source["id"], target["id"], "relates_to")

        result = forget(target["id"])

        assert "error" in result
        assert "centrality" in result["error"].lower()


class TestLinkMemories:
    """Tests for link_memories() — create edges."""

    def test_creates_edge_between_two_memories(self):
        """Creates edge between two memories."""
        from memory_palace.services.memory_service import remember
        from memory_palace.services.graph_service import link_memories
        from memory_palace.database import get_session

        m1 = remember(instance_id="test", memory_type="fact", content="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="M2")

        result = link_memories(m1["id"], m2["id"], "relates_to")

        assert "id" in result
        assert result["relation_type"] == "relates_to"

        db = get_session()
        try:
            edge = db.query(MemoryEdge).filter(MemoryEdge.id == result["id"]).first()
            assert edge is not None
            assert edge.source_id == m1["id"]
            assert edge.target_id == m2["id"]
        finally:
            db.close()

    def test_rejects_self_loops(self):
        """Rejects self-loops."""
        from memory_palace.services.memory_service import remember
        from memory_palace.services.graph_service import link_memories

        m = remember(instance_id="test", memory_type="fact", content="M")

        result = link_memories(m["id"], m["id"], "relates_to")

        assert "error" in result
        assert "itself" in result["error"].lower()

    def test_rejects_duplicate_edges(self):
        """Rejects duplicate edges."""
        from memory_palace.services.memory_service import remember
        from memory_palace.services.graph_service import link_memories

        m1 = remember(instance_id="test", memory_type="fact", content="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="M2")

        # Create first edge
        link_memories(m1["id"], m2["id"], "relates_to")

        # Try to create duplicate
        result = link_memories(m1["id"], m2["id"], "relates_to")

        assert "error" in result
        assert "already exists" in result["error"].lower()

    def test_clamps_strength_to_0_1(self):
        """Clamps strength to 0-1."""
        from memory_palace.services.memory_service import remember
        from memory_palace.services.graph_service import link_memories
        from memory_palace.database import get_session

        m1 = remember(instance_id="test", memory_type="fact", content="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="M2")

        # Try invalid strength values
        result1 = link_memories(m1["id"], m2["id"], "test1", strength=-0.5)
        result2 = link_memories(m1["id"], m2["id"], "test2", strength=1.5)

        db = get_session()
        try:
            edge1 = db.query(MemoryEdge).filter(MemoryEdge.id == result1["id"]).first()
            edge2 = db.query(MemoryEdge).filter(MemoryEdge.id == result2["id"]).first()
            assert edge1.strength == 0.0  # Clamped to 0
            assert edge2.strength == 1.0  # Clamped to 1
        finally:
            db.close()

    def test_archive_old_with_supersedes_archives_target(self):
        """archive_old=True with relation_type='supersedes' archives target."""
        from memory_palace.services.memory_service import remember
        from memory_palace.services.graph_service import link_memories
        from memory_palace.database import get_session

        old = remember(instance_id="test", memory_type="fact", content="Old version")
        new = remember(instance_id="test", memory_type="fact", content="New version")

        result = link_memories(
            new["id"],
            old["id"],
            "supersedes",
            archive_old=True
        )

        assert result.get("old_memory_archived") is True

        db = get_session()
        try:
            old_memory = db.query(Memory).filter(Memory.id == old["id"]).first()
            assert old_memory.is_archived is True
        finally:
            db.close()

    def test_archive_old_with_non_supersedes_does_not_archive(self):
        """archive_old=True with non-supersedes type does NOT archive."""
        from memory_palace.services.memory_service import remember
        from memory_palace.services.graph_service import link_memories
        from memory_palace.database import get_session

        m1 = remember(instance_id="test", memory_type="fact", content="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="M2")

        result = link_memories(
            m1["id"],
            m2["id"],
            "relates_to",
            archive_old=True  # Should be ignored for non-supersedes
        )

        assert "old_memory_archived" not in result or result.get("old_memory_archived") is False

        db = get_session()
        try:
            m2_memory = db.query(Memory).filter(Memory.id == m2["id"]).first()
            assert m2_memory.is_archived is False
        finally:
            db.close()


class TestUnlinkMemories:
    """Tests for unlink_memories() — remove edges."""

    def test_removes_specific_edge_by_relation_type(self):
        """Removes specific edge by relation_type."""
        from memory_palace.services.memory_service import remember
        from memory_palace.services.graph_service import link_memories, unlink_memories
        from memory_palace.database import get_session

        m1 = remember(instance_id="test", memory_type="fact", content="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="M2")

        link_memories(m1["id"], m2["id"], "relates_to")
        link_memories(m1["id"], m2["id"], "derived_from")

        result = unlink_memories(m1["id"], m2["id"], "relates_to")

        assert result["removed"] == 1

        db = get_session()
        try:
            # relates_to edge should be gone
            edge1 = db.query(MemoryEdge).filter(
                MemoryEdge.source_id == m1["id"],
                MemoryEdge.target_id == m2["id"],
                MemoryEdge.relation_type == "relates_to"
            ).first()
            assert edge1 is None

            # derived_from edge should still exist
            edge2 = db.query(MemoryEdge).filter(
                MemoryEdge.source_id == m1["id"],
                MemoryEdge.target_id == m2["id"],
                MemoryEdge.relation_type == "derived_from"
            ).first()
            assert edge2 is not None
        finally:
            db.close()

    def test_removes_all_edges_when_relation_type_none(self):
        """Removes all edges between pair when relation_type=None."""
        from memory_palace.services.memory_service import remember
        from memory_palace.services.graph_service import link_memories, unlink_memories
        from memory_palace.database import get_session

        m1 = remember(instance_id="test", memory_type="fact", content="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="M2")

        link_memories(m1["id"], m2["id"], "relates_to")
        link_memories(m1["id"], m2["id"], "derived_from")

        result = unlink_memories(m1["id"], m2["id"], relation_type=None)

        assert result["removed"] == 2

        db = get_session()
        try:
            edges = db.query(MemoryEdge).filter(
                MemoryEdge.source_id == m1["id"],
                MemoryEdge.target_id == m2["id"]
            ).all()
            assert len(edges) == 0
        finally:
            db.close()

    def test_returns_error_when_no_edges_found(self):
        """Returns error when no edges found."""
        from memory_palace.services.memory_service import remember
        from memory_palace.services.graph_service import unlink_memories

        m1 = remember(instance_id="test", memory_type="fact", content="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="M2")

        result = unlink_memories(m1["id"], m2["id"])

        assert "error" in result
        assert "No edges found" in result["error"]
