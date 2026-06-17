from pathlib import Path

from victor.core.database import (
    ProjectDatabaseManager,
    get_project_database,
    reset_project_database,
)


def _assert_project_connection_pragmas(db: ProjectDatabaseManager) -> None:
    conn = db.get_connection()

    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()
    synchronous = conn.execute("PRAGMA synchronous").fetchone()
    foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()
    wal_autocheckpoint = conn.execute("PRAGMA wal_autocheckpoint").fetchone()

    assert journal_mode is not None and journal_mode[0] == "wal"
    assert synchronous is not None and synchronous[0] == 1
    assert foreign_keys is not None and foreign_keys[0] == 1
    assert wal_autocheckpoint is not None
    assert wal_autocheckpoint[0] == ProjectDatabaseManager._WAL_AUTOCHECKPOINT_PAGES


def test_project_database_reapplies_connection_pragmas_after_reopen(
    tmp_path: Path,
) -> None:
    db = get_project_database(tmp_path)

    try:
        _assert_project_connection_pragmas(db)

        db.close()

        _assert_project_connection_pragmas(db)
    finally:
        reset_project_database(tmp_path)
