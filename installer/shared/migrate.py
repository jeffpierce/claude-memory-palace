"""
Memory Palace — Upgrade Detection & Migration Orchestration

Stdlib-only module (no SQLAlchemy imports). Runs before pip install,
using Python's built-in sqlite3 and json modules.

Detects existing v1/v2 palaces, migrates config files, backs up
databases, and invokes schema migration scripts via subprocess.
"""

import json
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


@dataclass
class PalaceInfo:
    """Snapshot of an existing Memory Palace installation."""
    exists: bool = False
    data_dir: Optional[Path] = None
    db_path: Optional[Path] = None
    config_path: Optional[Path] = None
    config_version: Optional[str] = None    # "v1", "v2", None
    schema_version: Optional[str] = None    # "v2", "v3", "v3.1", None
    needs_config_migration: bool = False
    needs_schema_migration: bool = False
    migration_chain: List[str] = field(default_factory=list)  # ["v2_to_v3", "v3_to_v3_1"]
    is_postgres: bool = False


@dataclass
class MigrationResult:
    """Aggregate result of the full migration pipeline."""
    config_migrated: bool = False
    config_backup_path: Optional[Path] = None
    db_backed_up: bool = False
    db_backup_path: Optional[Path] = None
    schema_migrated: bool = False
    migrations_run: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def had_work(self) -> bool:
        return self.config_migrated or self.schema_migrated or self.db_backed_up


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_existing_palace() -> PalaceInfo:
    """
    Probe ~/.memory-palace/ for an existing installation.

    Returns a PalaceInfo describing what was found and what needs migration.
    """
    info = PalaceInfo()
    data_dir = Path.home() / ".memory-palace"

    if not data_dir.exists():
        return info

    info.exists = True
    info.data_dir = data_dir
    info.config_path = data_dir / "config.json"

    # --- Config version ---
    if info.config_path.exists():
        try:
            with open(info.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "db_path" in config and "database" not in config:
                info.config_version = "v1"
                info.needs_config_migration = True
            elif "database" in config:
                info.config_version = "v2"
                # Check for PostgreSQL
                db_cfg = config.get("database", {})
                if db_cfg.get("type") == "postgres" or (
                    db_cfg.get("url") and "postgresql" in str(db_cfg.get("url", ""))
                ):
                    info.is_postgres = True
            else:
                # Has config but unrecognised shape — treat as v2
                info.config_version = "v2"
        except (json.JSONDecodeError, OSError):
            # Corrupt config — flag it but don't block
            info.config_version = None

    # --- Database / schema version ---
    if info.is_postgres:
        # Can't inspect PG without SQLAlchemy — defer to post-install.
        # Any memories.db sitting around is a stale artifact, ignore it.
        info.needs_schema_migration = True
        info.migration_chain = ["v2_to_v3", "v3_to_v3_1"]
    else:
        # SQLite: inspect the actual file
        db_path = data_dir / "memories.db"
        if db_path.exists():
            info.db_path = db_path
            info.schema_version = _detect_schema_version(db_path)

            if info.schema_version == "v2":
                info.needs_schema_migration = True
                info.migration_chain = ["v2_to_v3", "v3_to_v3_1"]
            elif info.schema_version == "v3":
                info.needs_schema_migration = True
                info.migration_chain = ["v3_to_v3_1"]
            elif info.schema_version == "v3.1":
                info.needs_schema_migration = False
            # None / empty = no memories table = nothing to migrate

    return info


def _detect_schema_version(db_path: Path) -> Optional[str]:
    """
    Inspect a SQLite database to determine its schema version.

    Returns "v2", "v3", "v3.1", or None (no memories table).
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if memories table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        if not cursor.fetchone():
            conn.close()
            return None

        # Get column names
        cursor.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        # v2: has importance, no foundational
        if "importance" in columns and "foundational" not in columns:
            return "v2"

        # v3: has foundational + project (singular string), no projects (array)
        if "foundational" in columns and "project" in columns and "projects" not in columns:
            return "v3"

        # v3.1: has projects (array), no project
        if "projects" in columns:
            return "v3.1"

        # Edge case: has both project and projects (mid-migration)
        if "project" in columns and "projects" in columns:
            return "v3"

        # Has foundational but neither project nor projects — odd, treat as v3.1
        if "foundational" in columns:
            return "v3.1"

        return None

    except sqlite3.Error:
        return None


# ---------------------------------------------------------------------------
# Config migration
# ---------------------------------------------------------------------------

def migrate_config(palace: PalaceInfo, progress: Optional[Callable[[str], None]] = None) -> MigrationResult:
    """
    Migrate a v1 config.json to v2 format.

    Preserves user settings (ollama_url, embedding_model, etc.).
    Backs up original first.
    """
    result = MigrationResult()
    log = progress or (lambda m: None)

    if not palace.needs_config_migration:
        return result

    if not palace.config_path or not palace.config_path.exists():
        result.errors.append("Config file not found for migration")
        return result

    try:
        # Read current config
        with open(palace.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Backup first
        backup_path = _backup_file(palace.config_path)
        if backup_path:
            result.config_backup_path = backup_path
            log(f"  Backed up config to {backup_path.name}")

        # Transform v1 -> v2
        config.pop("db_path", None)
        config.setdefault("database", {"type": "sqlite", "url": None})
        config.setdefault("synthesis", {"enabled": True})
        config.setdefault("auto_link", {
            "enabled": True,
            "link_threshold": 0.75,
            "suggest_threshold": 0.675,
        })
        config.setdefault("toon_output", True)

        # Write updated config
        with open(palace.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        result.config_migrated = True
        log("  Config migrated v1 -> v2")

    except Exception as e:
        result.errors.append(f"Config migration failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Database backup
# ---------------------------------------------------------------------------

def backup_database(palace: PalaceInfo, progress: Optional[Callable[[str], None]] = None) -> MigrationResult:
    """
    Back up the SQLite database before running schema migrations.

    PostgreSQL: logs advisory to back up manually.
    No DB: no-op.
    """
    result = MigrationResult()
    log = progress or (lambda m: None)

    if palace.is_postgres:
        log("  PostgreSQL detected — please back up your database manually")
        return result

    if not palace.db_path or not palace.db_path.exists():
        return result

    try:
        backup_path = _backup_file(palace.db_path)
        if backup_path:
            result.db_backed_up = True
            result.db_backup_path = backup_path
            log(f"  Database backed up to {backup_path.name}")
    except Exception as e:
        result.errors.append(f"Database backup failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Schema migrations (via subprocess)
# ---------------------------------------------------------------------------

def run_migrations(
    install_dir: Path,
    palace: PalaceInfo,
    progress: Optional[Callable[[str], None]] = None,
) -> MigrationResult:
    """
    Run the schema migration chain via subprocess using the venv Python.

    Migrations import memory_palace.config_v2 + sqlalchemy, which are
    only available inside the venv created by the installer.
    """
    result = MigrationResult()
    log = progress or (lambda m: None)

    if not palace.migration_chain:
        return result

    venv_python = _get_venv_python(install_dir)
    if not venv_python.exists():
        result.errors.append(f"Venv python not found at {venv_python}")
        return result

    for migration in palace.migration_chain:
        log(f"  Running {migration}...")
        try:
            proc = subprocess.run(
                [str(venv_python), "-m", f"memory_palace.migrations.{migration}"],
                capture_output=True,
                text=True,
                cwd=str(install_dir),
                timeout=120,
            )
            if proc.returncode == 0:
                result.migrations_run.append(migration)
                log(f"  {migration} completed successfully")
            else:
                error_msg = (proc.stderr or proc.stdout or "Unknown error").strip()
                # Check if the migration reported "already migrated" in stdout
                if "already migrated" in (proc.stdout or "").lower():
                    result.migrations_run.append(migration)
                    log(f"  {migration} — already up to date")
                else:
                    result.errors.append(f"{migration} failed: {error_msg}")
                    log(f"  {migration} failed: {error_msg}")
                    # Don't continue the chain if a migration fails
                    break
        except subprocess.TimeoutExpired:
            result.errors.append(f"{migration} timed out after 120s")
            log(f"  {migration} timed out")
            break
        except Exception as e:
            result.errors.append(f"{migration} error: {e}")
            log(f"  {migration} error: {e}")
            break

    if result.migrations_run and not result.errors:
        result.schema_migrated = True

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def detect_and_migrate(
    install_dir: Path,
    palace: PalaceInfo,
    progress: Optional[Callable[[str], None]] = None,
) -> MigrationResult:
    """
    Top-level orchestrator: detect -> backup config -> migrate config ->
    backup DB -> run schema migrations.

    Returns a MigrationResult with the aggregate state for the GUI.
    """
    log = progress or (lambda m: None)
    final = MigrationResult()

    # 1. Config migration
    if palace.needs_config_migration:
        log("Migrating configuration file...")
        cfg_result = migrate_config(palace, progress)
        final.config_migrated = cfg_result.config_migrated
        final.config_backup_path = cfg_result.config_backup_path
        final.errors.extend(cfg_result.errors)

    # 2. Database backup
    if palace.needs_schema_migration:
        log("Backing up database...")
        db_result = backup_database(palace, progress)
        final.db_backed_up = db_result.db_backed_up
        final.db_backup_path = db_result.db_backup_path
        final.errors.extend(db_result.errors)

    # 3. Schema migrations (only if backup succeeded or is a no-op)
    if palace.needs_schema_migration and not final.errors:
        log("Running schema migrations...")
        mig_result = run_migrations(install_dir, palace, progress)
        final.schema_migrated = mig_result.schema_migrated
        final.migrations_run = mig_result.migrations_run
        final.errors.extend(mig_result.errors)

    return final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backup_file(path: Path) -> Optional[Path]:
    """
    Create a backup using incrementing counter pattern.

    foo.db -> foo.db.backup, foo.db.backup.1, foo.db.backup.2, ...
    """
    if not path.exists():
        return None

    backup_path = path.parent / f"{path.name}.backup"
    counter = 1
    while backup_path.exists():
        backup_path = path.parent / f"{path.name}.backup.{counter}"
        counter += 1

    shutil.copy2(path, backup_path)
    return backup_path


def _get_venv_python(install_dir: Path) -> Path:
    """Get the path to python in the virtual environment."""
    venv_dir = install_dir / "venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


# ---------------------------------------------------------------------------
# Display helpers for the GUI
# ---------------------------------------------------------------------------

def detection_display(palace: PalaceInfo) -> tuple:
    """
    Return (ok: bool, detail: str) for the detection screen row.
    """
    if not palace.exists:
        return True, "Fresh install"

    if palace.is_postgres:
        # Can't inspect PG schema pre-install — migrations are idempotent
        return True, "PostgreSQL \u2014 will check after install"

    if palace.needs_schema_migration:
        version = palace.schema_version or "unknown"
        return True, f"Found {version} schema \u2014 will upgrade"

    if palace.schema_version == "v3.1":
        return True, "Up to date (v3.1)"

    return True, "Existing palace found"
