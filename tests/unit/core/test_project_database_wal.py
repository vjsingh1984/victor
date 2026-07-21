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
    busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()

    assert journal_mode is not None and journal_mode[0] == "wal"
    assert synchronous is not None and synchronous[0] == 1
    assert foreign_keys is not None and foreign_keys[0] == 1
    assert wal_autocheckpoint is not None
    assert wal_autocheckpoint[0] == ProjectDatabaseManager._WAL_AUTOCHECKPOINT_PAGES
    # P6: deliberately 5000ms (NOT the 30s connect timeout) — with the undo-log
    # write made non-fatal, a short bounded wait beats a long stall.
    assert busy_timeout is not None and busy_timeout[0] == 5000


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


def test_project_database_uses_incremental_auto_vacuum(tmp_path: Path) -> None:
    db = get_project_database(tmp_path)
    try:
        # 0=NONE, 1=FULL, 2=INCREMENTAL — a fresh project DB must reclaim incrementally.
        mode = db.get_connection().execute("PRAGMA auto_vacuum").fetchone()
        assert mode is not None and mode[0] == 2
    finally:
        reset_project_database(tmp_path)


def test_maintain_checkpoints_and_reclaims_space(tmp_path: Path) -> None:
    db = get_project_database(tmp_path)
    try:
        # Use the same raw connection maintain() uses so visibility is unambiguous.
        conn = db._get_raw_connection()
        conn.execute("CREATE TABLE bulk (id INTEGER PRIMARY KEY, blob TEXT)")
        conn.executemany(
            "INSERT INTO bulk (blob) VALUES (?)",
            [("x" * 2000,) for _ in range(5000)],
        )
        conn.commit()
        # Materialize the WAL into the main file so the baseline reflects the real data size.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        full_size = db.db_path.stat().st_size
        assert full_size > 1_000_000  # ~10 MB of data is now in the main file

        conn.execute("DELETE FROM bulk")
        conn.commit()

        stats = db.maintain()

        assert stats["path"] == str(db.db_path)
        # VACUUM reclaimed the freed pages: the file shrank well below the populated size.
        assert stats["size_after"] < full_size
        # WAL is truncated by the checkpoint — no oversized -wal left behind.
        wal = db.db_path.with_name(db.db_path.name + "-wal")
        assert (not wal.exists()) or wal.stat().st_size < full_size
    finally:
        reset_project_database(tmp_path)
