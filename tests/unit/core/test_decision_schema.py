# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FEP-0012 Phase 2b: the decision-learning schema (decision_log,
decision_outcome global; local_classifier_delta project) is registered, migrated
to v8, and yields valid, queryable tables."""

import sqlite3

import pytest

from victor.core.schema import (
    CURRENT_SCHEMA_VERSION,
    Schema,
    Tables,
    get_migration_sql,
)


def test_table_constants_defined():
    assert Tables.DECISION_LOG == "decision_log"
    assert Tables.DECISION_OUTCOME == "decision_outcome"
    assert Tables.LOCAL_CLASSIFIER_DELTA == "local_classifier_delta"


def test_schema_version_bumped_to_8():
    assert CURRENT_SCHEMA_VERSION == 8


def test_global_tables_registered_in_get_all_schemas():
    stmts = Schema.get_all_schemas()
    assert Schema.DECISION_LOG in stmts
    assert Schema.DECISION_OUTCOME in stmts
    # The delta is a PROJECT table — must NOT be in the global list.
    assert Schema.LOCAL_CLASSIFIER_DELTA not in stmts


def test_delta_registered_in_get_project_schemas():
    proj = Schema.get_project_schemas()
    assert Schema.LOCAL_CLASSIFIER_DELTA in proj
    # Global decision tables must NOT be in the project list.
    assert Schema.DECISION_LOG not in proj
    assert Schema.DECISION_OUTCOME not in proj


def test_v8_migration_creates_all_three_tables():
    sqls = get_migration_sql(7, 8)
    # Three single CREATE TABLE statements (indexes are applied separately).
    assert Schema.DECISION_LOG in sqls
    assert Schema.DECISION_OUTCOME in sqls
    assert Schema.LOCAL_CLASSIFIER_DELTA in sqls


def _apply(conn: sqlite3.Connection, statements):
    for s in statements:
        conn.execute(s)
    conn.commit()


def test_global_tables_create_and_query():
    conn = sqlite3.connect(":memory:")
    _apply(conn, [Schema.SYS_METADATA])
    _apply(conn, Schema.get_all_schemas())
    for idx in Schema.get_all_indexes():
        conn.executescript(idx)
    conn.commit()

    # decision_log round-trip with the FEP-0012 correlation fields.
    conn.execute(
        f"""INSERT INTO {Tables.DECISION_LOG}
            (decision_id, decision_type, session_id, turn_id, source,
             confidence, model_version, feature_spec_version, feature_digest,
             context, result)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            "d1",
            "task_type_classification",
            "sess-1",
            "turn-1",
            "local_classifier",
            0.8,
            "edge_v1",
            "1",
            "abc",
            "{}",
            "debug",
        ),
    )
    row = conn.execute(
        f"SELECT session_id, turn_id, model_version FROM {Tables.DECISION_LOG} "
        f"WHERE decision_id='d1'"
    ).fetchone()
    assert row == ("sess-1", "turn-1", "edge_v1")

    # decision_outcome round-trip with attributed_reward.
    conn.execute(
        f"""INSERT INTO {Tables.DECISION_OUTCOME}
            (decision_id, session_id, success, quality_score, attributed_reward,
             credit_method)
            VALUES (?,?,?,?,?,?)""",
        ("d1", "sess-1", 1, 0.9, 0.7, "gae"),
    )
    row = conn.execute(
        f"SELECT attributed_reward FROM {Tables.DECISION_OUTCOME} WHERE decision_id='d1'"
    ).fetchone()
    assert row[0] == pytest.approx(0.7)
    conn.close()


def test_project_delta_table_creates_and_query():
    conn = sqlite3.connect(":memory:")
    _apply(conn, Schema.get_project_schemas())
    for idx in Schema.get_project_indexes():
        conn.executescript(idx)
    conn.commit()

    conn.execute(
        f"""INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}
            (decision_type, feature_hash, weight, samples, sum_reward)
            VALUES (?,?,?,?,?)""",
        ("task_type_classification", 12345, 0.42, 7, 5.5),
    )
    # Composite PK dedups on (decision_type, feature_hash).
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"""INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}
                (decision_type, feature_hash, weight) VALUES (?, ?, ?)""",
            ("task_type_classification", 12345, 0.9),
        )
    row = conn.execute(
        f"SELECT weight, samples FROM {Tables.LOCAL_CLASSIFIER_DELTA} "
        f"WHERE decision_type='task_type_classification'"
    ).fetchone()
    assert row[0] == pytest.approx(0.42) and row[1] == 7
    conn.close()
