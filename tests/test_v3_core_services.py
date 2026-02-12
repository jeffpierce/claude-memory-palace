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


# ── Known Embedding Vectors with Geometric Properties ────────────────

# Create 768-dimensional vectors with known cosine similarities
# Pad to 768 dimensions (nomic-embed-text size) with zeros
def _make_vector(coords: List[float]) -> List[float]:
    """Pad a small coordinate list to 768 dims with zeros."""
    padded = coords + [0.0] * (768 - len(coords))
    return padded

# Unit vectors pointing in different directions (for cosine similarity testing)
VEC_NORTH = _make_vector([1.0, 0.0, 0.0])  # (1, 0, 0, ...)
VEC_SOUTH = _make_vector([-1.0, 0.0, 0.0])  # (-1, 0, 0, ...) — cos sim = -1
VEC_EAST = _make_vector([0.0, 1.0, 0.0])  # (0, 1, 0, ...) — cos sim with NORTH = 0
VEC_NORTHEAST = _make_vector([0.707, 0.707, 0.0])  # 45° from NORTH — cos sim ≈ 0.707
VEC_NORTH_CLOSE = _make_vector([0.95, 0.31, 0.0])  # Close to NORTH — cos sim ≈ 0.95
VEC_UNRELATED = _make_vector([0.0, 0.0, 1.0])  # (0, 0, 1, ...) — orthogonal to all above


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
    """
    Mock embedding generation with known geometric vectors.

    Does NOT mock cosine_similarity — we use the REAL implementation from embeddings.py
    so that the actual similarity scoring, centrality weighting, and ranking logic are tested.
    """
    import json

    # Mapping of content patterns to known vectors
    VECTOR_MAP = {
        "north": VEC_NORTH,
        "south": VEC_SOUTH,
        "east": VEC_EAST,
        "northeast": VEC_NORTHEAST,
        "north_close": VEC_NORTH_CLOSE,
        "unrelated": VEC_UNRELATED,
    }

    def get_embedding_mock(text: str):
        """Return known vector based on text content markers."""
        text_lower = text.lower()

        # Check for vector markers in the text
        for marker, vector in VECTOR_MAP.items():
            if marker in text_lower:
                # Return as JSON string (SQLite storage format)
                return json.dumps(vector)

        # Default to VEC_NORTH for unmarked content
        return json.dumps(VEC_NORTH)

    # Mock get_embedding BUT NOT cosine_similarity — let the real math run
    with patch("memory_palace.services.memory_service.get_embedding", side_effect=get_embedding_mock):
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
            # Create first memory with VEC_NORTH_CLOSE
            first = remember(
                instance_id="test",
                memory_type="fact",
                content="This is north_close content.",
                subject="Similar fact",
                project="test-project"
            )

            # Create second memory with VEC_NORTH (cos sim ≈ 0.95 > 0.75 threshold)
            second = remember(
                instance_id="test",
                memory_type="fact",
                content="This is north content too.",
                subject="Similar fact two",
                project="test-project",
                auto_link=True  # Explicitly enable
            )

            # Should have auto-created link (similarity ≈ 0.95 > threshold 0.75)
            assert "links_created" in second, "Should auto-link similar memories"
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
            # Create memory in project A with VEC_NORTH_CLOSE
            first = remember(
                instance_id="test",
                memory_type="fact",
                content="This is north_close content.",
                subject="Similar fact",
                project="project-A"
            )

            # Create similar memory in project B with VEC_NORTH (high similarity, different project)
            second = remember(
                instance_id="test",
                memory_type="fact",
                content="This is north content too.",
                subject="Similar fact two",
                project="project-B",
                auto_link=True
            )

            # Should NOT have auto-created links (different projects)
            assert "links_created" not in second or len(second.get("links_created", [])) == 0, \
                "Should NOT auto-link across projects when same_project_only=True"

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
        """Returns results with semantic search and ranks by actual cosine similarity."""
        from memory_palace.services.memory_service import remember, recall

        # Create memories with known embeddings
        # VEC_NORTH and VEC_SOUTH are opposite (cos sim = -1)
        # VEC_NORTH_CLOSE is close to VEC_NORTH (cos sim ≈ 0.95)
        m_north = remember(
            instance_id="test",
            memory_type="fact",
            content="Content pointing north.",
            subject="North Memory"
        )
        m_south = remember(
            instance_id="test",
            memory_type="fact",
            content="Content pointing south.",
            subject="South Memory"
        )
        m_close = remember(
            instance_id="test",
            memory_type="fact",
            content="Content pointing north_close.",
            subject="Close to North"
        )

        # Query with VEC_NORTH — should rank north_close > north > south
        result = recall(query="north", synthesize=False, limit=10)

        assert result["count"] == 3
        assert result["search_method"].startswith("semantic")
        assert len(result["memories"]) == 3

        # Check ranking order: north_close (cos≈0.95) should be first, south (cos≈-1) should be last
        # Actual ranking uses centrality-weighted scoring, but with all memories having equal
        # access_count and centrality, similarity should dominate
        ids_in_order = [m["id"] for m in result["memories"]]

        # m_close should rank higher than m_south
        assert ids_in_order.index(m_close["id"]) < ids_in_order.index(m_south["id"])

        # Check that similarity scores are included and make sense
        memories_by_id = {m["id"]: m for m in result["memories"]}

        # north_close should have high similarity (≈0.95 with VEC_NORTH query)
        close_sim = memories_by_id[m_close["id"]].get("similarity_score")
        assert close_sim is not None
        assert close_sim > 0.9, f"Expected north_close similarity > 0.9, got {close_sim}"

        # south should have negative similarity (≈-1 with VEC_NORTH query)
        south_sim = memories_by_id[m_south["id"]].get("similarity_score")
        assert south_sim is not None
        assert south_sim < 0.0, f"Expected south similarity < 0, got {south_sim}"

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
        """Filters by project (single string) with semantic search."""
        from memory_palace.services.memory_service import remember, recall

        # Store memories in different projects, both with VEC_NORTH
        m_a = remember(
            instance_id="test",
            memory_type="fact",
            content="Content pointing north in project A.",
            subject="Project A Memory",
            project="project-A"
        )
        m_b = remember(
            instance_id="test",
            memory_type="fact",
            content="Content pointing north in project B.",
            subject="Project B Memory",
            project="project-B"
        )

        # Query with VEC_NORTH, filtered to project-A
        result = recall(query="north", project="project-A", synthesize=False)

        assert result["count"] == 1, f"Expected 1 memory, got {result['count']}"
        assert result["memories"][0]["project"] == ["project-A"]
        assert result["memories"][0]["id"] == m_a["id"]

    def test_filters_by_project_list(self):
        """Filters by project (list of strings) with semantic search."""
        from memory_palace.services.memory_service import remember, recall

        # All memories have VEC_NORTH for consistent similarity
        m_a = remember(instance_id="test", memory_type="fact", content="north in A", project="project-A")
        m_b = remember(instance_id="test", memory_type="fact", content="north in B", project="project-B")
        m_c = remember(instance_id="test", memory_type="fact", content="north in C", project="project-C")

        result = recall(query="north", project=["project-A", "project-B"], synthesize=False)

        assert result["count"] == 2, f"Expected 2 memories, got {result['count']}"

        # Verify correct projects are included
        returned_ids = {m["id"] for m in result["memories"]}
        assert returned_ids == {m_a["id"], m_b["id"]}

        # Check that projects are correct
        projects = set()
        for m in result["memories"]:
            for p in m["project"]:
                projects.add(p)
        assert projects == {"project-A", "project-B"}

    def test_filters_by_memory_type_with_wildcard(self):
        """Filters by memory_type with wildcard and ranks by similarity."""
        from memory_palace.services.memory_service import remember, recall

        # Store memories with different types but similar vectors
        m_func = remember(instance_id="test", memory_type="code_function", content="north Function A")
        m_class = remember(instance_id="test", memory_type="code_class", content="north_close Class B")
        m_fact = remember(instance_id="test", memory_type="fact", content="north Fact C")

        # Query with wildcard filter for code_*
        result = recall(query="north", memory_type="code_*", synthesize=False)

        assert result["count"] == 2, f"Expected 2 memories, got {result['count']}"

        # Verify correct types are returned
        returned_ids = {m["id"] for m in result["memories"]}
        assert returned_ids == {m_func["id"], m_class["id"]}

        types = {m["memory_type"] for m in result["memories"]}
        assert types == {"code_function", "code_class"}

        # Verify ranking: m_func (north, cos=1.0 with query) should rank higher than m_class (north_close, cos≈0.95)
        ids_in_order = [m["id"] for m in result["memories"]]
        assert ids_in_order[0] == m_func["id"], "m_func should rank first (exact match with query)"

    def test_filters_by_min_foundational(self):
        """Filters by min_foundational=True with semantic search."""
        from memory_palace.services.memory_service import remember, recall

        # Both memories have similar vectors (north) but different foundational status
        m_regular = remember(instance_id="test", memory_type="fact", content="north Regular", foundational=False)
        m_found = remember(instance_id="test", memory_type="core", content="north Foundational", foundational=True)

        # Query for foundational only
        result = recall(query="north", min_foundational=True, synthesize=False)

        assert result["count"] == 1, f"Expected 1 foundational memory, got {result['count']}"
        assert result["memories"][0]["foundational"] is True
        assert result["memories"][0]["id"] == m_found["id"]

    def test_includes_graph_context_in_results(self):
        """Includes graph context in results with semantic search (summary mode by default)."""
        from memory_palace.services.memory_service import remember, recall
        from memory_palace.services.graph_service import link_memories

        # Create two memories with known vectors and link them
        m1 = remember(instance_id="test", memory_type="fact", content="north Memory 1", subject="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="north Memory 2", subject="M2")
        link_result = link_memories(m1["id"], m2["id"], "relates_to")

        assert "id" in link_result, "link_memories should create an edge"

        # Query with VEC_NORTH to find both memories
        result = recall(query="north", include_graph=True, synthesize=False)

        assert result["count"] >= 2, f"Expected at least 2 memories, got {result['count']}"
        assert "graph_context" in result, "Should include graph_context when include_graph=True"

        # With summary mode (default), should have nodes with stats and total_edges
        assert "nodes" in result["graph_context"]
        assert "total_edges" in result["graph_context"]
        assert result["graph_context"]["total_edges"] > 0, "Should have at least one edge"

        # Verify both nodes are in the graph context with connection info
        nodes = result["graph_context"]["nodes"]
        assert str(m1["id"]) in nodes or str(m2["id"]) in nodes

        # At least one node should have connections > 0
        has_connections = any(node["connections"] > 0 for node in nodes.values())
        assert has_connections, "At least one node should have connections"

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
        """Returns raw memories when synthesize=False with full content and similarity scores."""
        from memory_palace.services.memory_service import remember, recall

        # Create a memory with known vector
        m = remember(
            instance_id="test",
            memory_type="fact",
            content="north Raw test content",
            subject="Raw Test Memory"
        )

        # Query with VEC_NORTH to get exact match
        result = recall(query="north", synthesize=False)

        # Check structure
        assert "memories" in result, "Should return 'memories' key when synthesize=False"
        assert "summary" not in result, "Should NOT return 'summary' when synthesize=False"
        assert isinstance(result["memories"], list)
        assert len(result["memories"]) > 0

        # Check content of returned memory
        returned_mem = result["memories"][0]
        assert returned_mem["id"] == m["id"]
        assert returned_mem["subject"] == "Raw Test Memory"
        assert "content" in returned_mem, "Should return full content in verbose mode"
        assert returned_mem["content"] == "north Raw test content"
        assert "similarity_score" in returned_mem, "Should include similarity_score"
        assert returned_mem["similarity_score"] > 0.99, "Exact match should have similarity ≈ 1.0"


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


class TestMultiProject:
    """Tests for multi-project support."""

    def test_remember_with_list_project(self):
        """remember() accepts list of projects."""
        from memory_palace.services.memory_service import remember
        from memory_palace.database import get_session

        result = remember(
            instance_id="test",
            memory_type="fact",
            content="Multi-project memory",
            subject="Multi",
            project=["life", "palace"]
        )

        assert "id" in result

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == result["id"]).first()
            assert memory.projects == ["life", "palace"]
        finally:
            db.close()

    def test_remember_normalizes_string_to_list(self):
        """remember() normalizes string project to single-element list."""
        from memory_palace.services.memory_service import remember
        from memory_palace.database import get_session

        result = remember(
            instance_id="test",
            memory_type="fact",
            content="Single project memory",
            project="test-project"
        )

        db = get_session()
        try:
            memory = db.query(Memory).filter(Memory.id == result["id"]).first()
            assert memory.projects == ["test-project"]
        finally:
            db.close()

    def test_recall_single_project_matches_multi_project_memory(self):
        """recall(project="life") finds memories that include "life" in their projects array."""
        from memory_palace.services.memory_service import remember, recall

        mem = remember(
            instance_id="test",
            memory_type="fact",
            content="This memory is in both life and palace projects.",
            subject="Dual project",
            project=["life", "palace"]
        )

        result = recall(query="dual project", project="life", synthesize=False)

        # Should find exactly the one memory we created
        assert result["count"] == 1
        assert result["memories"][0]["id"] == mem["id"]
        assert "life" in result["memories"][0]["project"]

    def test_recall_list_project_matches_union(self):
        """recall(project=["life", "palace"]) returns memories from either project."""
        from memory_palace.services.memory_service import remember, recall

        # All three memories use VEC_NORTH for consistent similarity
        m_life = remember(
            instance_id="test",
            memory_type="fact",
            content="Life only memory north.",
            subject="Life only",
            project="life"
        )
        m_palace = remember(
            instance_id="test",
            memory_type="fact",
            content="Palace only memory north.",
            subject="Palace only",
            project="palace"
        )
        m_other = remember(
            instance_id="test",
            memory_type="fact",
            content="Unrelated project north.",
            subject="Other project",
            project="other"
        )

        result = recall(query="north", project=["life", "palace"], synthesize=False)
        assert result["count"] == 2, f"Expected 2 memories from life/palace, got {result['count']}"

        # Verify only life and palace memories are returned
        returned_ids = {m["id"] for m in result["memories"]}
        assert returned_ids == {m_life["id"], m_palace["id"]}

    def test_stats_explodes_multi_project(self):
        """get_memory_stats() counts multi-project memories in each project."""
        from memory_palace.services.memory_service import remember, get_memory_stats

        remember(
            instance_id="test",
            memory_type="fact",
            content="Dual membership memory.",
            project=["alpha", "beta"]
        )

        stats = get_memory_stats()
        # Memory should count toward BOTH projects
        assert stats["by_project"].get("alpha", 0) >= 1
        assert stats["by_project"].get("beta", 0) >= 1

    def test_auto_link_checks_project_overlap(self):
        """Auto-link with same_project_only uses overlap, not exact match."""
        from memory_palace.services.memory_service import remember

        with patch("memory_palace.services.memory_service.get_auto_link_config", return_value={
            "enabled": True,
            "link_threshold": 0.75,
            "suggest_threshold": 0.675,
            "max_suggestions": 10,
            "same_project_only": True,
            "classify_edges": False,
        }):
            # Memory in ["life", "palace"] with VEC_NORTH_CLOSE
            first = remember(
                instance_id="test",
                memory_type="fact",
                content="This is north_close content.",
                subject="Similar fact",
                project=["life", "palace"]
            )

            # Memory in ["palace", "tools"] with VEC_NORTH — overlaps on "palace", high similarity
            second = remember(
                instance_id="test",
                memory_type="fact",
                content="This is north content too.",
                subject="Similar fact two",
                project=["palace", "tools"],
                auto_link=True
            )

            # Should auto-link because projects overlap on "palace" and similarity > threshold
            assert "links_created" in second, "Should auto-link when projects overlap"
            assert len(second["links_created"]) > 0


class TestGraphModeSummary:
    """Tests for graph_mode parameter in _get_graph_context_for_memories()."""

    def test_summary_mode_returns_node_stats(self):
        """Summary mode returns node stats with subject, connections, avg_strength, edge_types."""
        from memory_palace.services.memory_service import remember, _get_graph_context_for_memories
        from memory_palace.services.graph_service import link_memories
        from memory_palace.database import get_session

        # Create two memories with an edge between them
        m1 = remember(instance_id="test", memory_type="fact", content="Memory 1", subject="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="Memory 2", subject="M2")
        link_memories(m1["id"], m2["id"], "relates_to", strength=0.9)

        db = get_session()
        try:
            # Get memory objects
            mem1 = db.query(Memory).filter(Memory.id == m1["id"]).first()

            # Call with summary mode
            result = _get_graph_context_for_memories(db, [mem1], graph_mode="summary")

            # Assert structure
            assert "nodes" in result
            assert "total_edges" in result
            assert "seed_ids" in result

            # Assert omitted_nodes is NOT present when under 30 nodes
            assert "omitted_nodes" not in result, "omitted_nodes should only appear when capping happens"

            # Assert seed_ids contains m1
            assert m1["id"] in result["seed_ids"]

            # Assert node entries have required fields
            for node_id, node_data in result["nodes"].items():
                assert "subject" in node_data
                assert "connections" in node_data
                assert "avg_strength" in node_data
                assert "edge_types" in node_data

            # Check that m1 has the expected edge
            m1_node = result["nodes"][str(m1["id"])]
            assert m1_node["connections"] == 1
            assert m1_node["avg_strength"] == 0.9
            assert "relates_to" in m1_node["edge_types"]
        finally:
            db.close()

    def test_full_mode_returns_existing_format(self):
        """Full mode returns existing format with nodes (id->subject) and edges list."""
        from memory_palace.services.memory_service import remember, _get_graph_context_for_memories
        from memory_palace.services.graph_service import link_memories
        from memory_palace.database import get_session

        # Create two memories with an edge between them
        m1 = remember(instance_id="test", memory_type="fact", content="Memory 1", subject="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="Memory 2", subject="M2")
        link_memories(m1["id"], m2["id"], "relates_to", strength=0.8)

        db = get_session()
        try:
            # Get memory objects
            mem1 = db.query(Memory).filter(Memory.id == m1["id"]).first()

            # Call with full mode
            result = _get_graph_context_for_memories(db, [mem1], graph_mode="full")

            # Assert structure
            assert "nodes" in result
            assert "edges" in result
            assert "total_edges" not in result  # Should not have summary fields

            # Assert nodes is dict of id->subject string
            assert isinstance(result["nodes"], dict)
            assert str(m1["id"]) in result["nodes"]
            assert isinstance(result["nodes"][str(m1["id"])], str)
            assert result["nodes"][str(m1["id"])] == "M1"

            # Assert edges is list of dicts with source/target/type/strength
            assert isinstance(result["edges"], list)
            assert len(result["edges"]) > 0

            edge = result["edges"][0]
            assert "source" in edge
            assert "target" in edge
            assert "type" in edge
            assert "strength" in edge
            assert edge["source"] == m1["id"]
            assert edge["target"] == m2["id"]
            assert edge["type"] == "relates_to"
            assert edge["strength"] == 0.8
        finally:
            db.close()

    def test_summary_mode_is_default(self):
        """Summary mode is the default when graph_mode is not specified."""
        from memory_palace.services.memory_service import remember, _get_graph_context_for_memories
        from memory_palace.services.graph_service import link_memories
        from memory_palace.database import get_session

        # Create two memories with an edge between them
        m1 = remember(instance_id="test", memory_type="fact", content="Memory 1", subject="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="Memory 2", subject="M2")
        link_memories(m1["id"], m2["id"], "relates_to")

        db = get_session()
        try:
            # Get memory objects
            mem1 = db.query(Memory).filter(Memory.id == m1["id"]).first()

            # Call without specifying graph_mode (should default to "summary")
            result = _get_graph_context_for_memories(db, [mem1])

            # Assert it has summary format (total_edges key)
            assert "total_edges" in result, "Should default to summary mode"
            assert "nodes" in result
            assert "seed_ids" in result
        finally:
            db.close()

    def test_summary_no_edges_returns_empty(self):
        """Summary mode returns empty dict when memory has no edges."""
        from memory_palace.services.memory_service import remember, _get_graph_context_for_memories
        from memory_palace.database import get_session

        # Create a memory with no edges
        m1 = remember(instance_id="test", memory_type="fact", content="Isolated Memory", subject="Isolated")

        db = get_session()
        try:
            # Get memory object
            mem1 = db.query(Memory).filter(Memory.id == m1["id"]).first()

            # Call with summary mode
            result = _get_graph_context_for_memories(db, [mem1], graph_mode="summary")

            # Assert returns empty dict
            assert result == {}
        finally:
            db.close()

    def test_summary_multiple_edge_types(self):
        """Summary mode edge_types list contains all unique edge types."""
        from memory_palace.services.memory_service import remember, _get_graph_context_for_memories
        from memory_palace.services.graph_service import link_memories
        from memory_palace.database import get_session

        # Create three memories with different edge types
        m1 = remember(instance_id="test", memory_type="fact", content="Memory 1", subject="M1")
        m2 = remember(instance_id="test", memory_type="fact", content="Memory 2", subject="M2")
        m3 = remember(instance_id="test", memory_type="fact", content="Memory 3", subject="M3")

        # Create edges of different types
        link_memories(m1["id"], m2["id"], "relates_to", strength=0.8)
        link_memories(m1["id"], m3["id"], "derived_from", strength=0.7)
        link_memories(m2["id"], m1["id"], "refines", strength=0.9)

        db = get_session()
        try:
            # Get memory object
            mem1 = db.query(Memory).filter(Memory.id == m1["id"]).first()

            # Call with summary mode
            result = _get_graph_context_for_memories(db, [mem1], graph_mode="summary")

            # Assert m1's edge_types contains all three types
            m1_node = result["nodes"][str(m1["id"])]
            edge_types = m1_node["edge_types"]
            assert "relates_to" in edge_types
            assert "derived_from" in edge_types
            assert "refines" in edge_types
            assert len(edge_types) == 3

            # Verify connections count
            assert m1_node["connections"] == 3

            # Verify avg_strength is calculated correctly
            expected_avg = round((0.8 + 0.7 + 0.9) / 3, 4)
            assert m1_node["avg_strength"] == expected_avg
        finally:
            db.close()


class TestFoundationalGraphClamp:
    """Tests for foundational memory graph depth clamping."""

    def test_foundational_memory_clamps_depth(self):
        """Foundational memories clamp graph_depth to 1 even if requested higher."""
        from memory_palace.services.memory_service import remember, get_memory_by_id

        # Create a foundational memory
        m = remember(
            instance_id="test",
            memory_type="core",
            content="Foundational core concept",
            subject="Core concept",
            foundational=True
        )

        # Call get_memory_by_id with graph_depth=2 (which would normally explore 2 hops)
        # This should not error — the depth should be clamped internally
        result = get_memory_by_id(m["id"], include_graph=True, graph_depth=2)

        # Happy path test: just verify it doesn't error and returns the memory
        assert result is not None
        assert "memory" in result
        assert result["memory"]["id"] == m["id"]
        assert result["memory"]["foundational"] is True
