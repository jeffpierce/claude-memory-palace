"""
Pytest configuration for tests.

Sets up SQLite mode for all tests BEFORE any models are imported,
unless a PostgreSQL test URL is provided (migration tests).
"""
import os

# Force SQLite mode - MUST be before any memory_palace imports
# Use DATABASE_URL which is actually supported by load_config()
# Skip override when running PostgreSQL migration tests (CI sets both env vars)
if not os.environ.get("MEMORY_PALACE_TEST_POSTGRES_URL"):
    os.environ["MEMORY_PALACE_DATABASE_URL"] = "sqlite:///:memory:"
