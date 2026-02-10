"""
Pytest configuration for tests.

Sets up SQLite mode for all tests BEFORE any models are imported.
"""
import os

# Force SQLite mode - MUST be before any memory_palace imports
# Use DATABASE_URL which is actually supported by load_config()
os.environ["MEMORY_PALACE_DATABASE_URL"] = "sqlite:///:memory:"
