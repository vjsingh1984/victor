"""One-shot migration utility: copies per-learner private tables into unified RL tables.

Each learner previously created its own private tables in victor.db. The unified schema
uses rl_q_value, rl_transition, rl_param, and rl_task_stat — each partitioned by
learner_id — as the single source of truth.

Migration is idempotent: completion is recorded in sys_metadata so subsequent runs
are no-ops. Old tables are preserved (not dropped) for safety; a follow-up PR removes
them after production stability is confirmed.
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
import logging
import sqlite3
from typing import Callable

from victor.core.schema import Tables

logger = logging.getLogger(__name__)

# Prefix used to track per-learner migration status in sys_metadata
_MIGRATION_KEY_PREFIX = "rl_unified_migration_v1:"

# Helpers for safe table-existence checks
_TABLE_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return bool(conn.execute(_TABLE_EXISTS_SQL, (table_name,)).fetchone())


def _has_column(conn: sqlite3.Connection, table_name: str, column: str) -> bool:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    return column in cols


class RLTableMigrator:
    """Idempotent one-shot migrator from per-learner private tables to unified schema."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ensure_unified_tables()

    def _ensure_unified_tables(self) -> None:
        """Create unified RL tables if they don't exist (safe for direct learner tests)."""
        from victor.core.schema import Schema

        try:
            for stmt in (
                Schema.SYS_METADATA,  # needed for migration completion tracking
                Schema.RL_Q_VALUE,
                Schema.RL_Q_VALUE_INDEXES,
                Schema.RL_TRANSITION,
                Schema.RL_TRANSITION_INDEXES,
                Schema.RL_PARAM,
                Schema.RL_TASK_STAT,
            ):
                self._conn.execute(stmt)
            self._conn.commit()
        except Exception as exc:
            logger.debug(f"RL migration: unified table creation skipped: {exc}")

    def _migration_key(self, learner_id: str) -> str:
        return _MIGRATION_KEY_PREFIX + learner_id

    def is_migrated(self, learner_id: str) -> bool:
        try:
            row = self._conn.execute(
                f"SELECT value FROM {Tables.SYS_METADATA} WHERE key = ?",
                (self._migration_key(learner_id),),
            ).fetchone()
            return row is not None and row[0] == "done"
        except Exception:
            return False

    def _mark_migrated(self, learner_id: str) -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO {Tables.SYS_METADATA} (key, value, updated_at) "
            f"VALUES (?, 'done', datetime('now'))",
            (self._migration_key(learner_id),),
        )
        self._conn.commit()

    def run_if_needed(
        self, learner_id: str, migrate_fn: Callable[[sqlite3.Connection], None]
    ) -> None:
        """Run migrate_fn once for learner_id; subsequent calls are no-ops."""
        if self.is_migrated(learner_id):
            return
        try:
            migrate_fn(self._conn)
            self._mark_migrated(learner_id)
            logger.info(f"RL migration: {learner_id} -> unified tables complete")
        except Exception as exc:
            logger.warning(f"RL migration: {learner_id} failed (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # Per-learner migration functions
    # ------------------------------------------------------------------

    @staticmethod
    def migrate_mode_transition(conn: sqlite3.Connection) -> None:
        # RL_MODE_Q -> rl_q_value
        if _table_exists(conn, Tables.RL_MODE_Q):
            rows = conn.execute(
                f"SELECT state_key, action_key, q_value, visit_count FROM {Tables.RL_MODE_Q}"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_Q_VALUE} "
                f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
                f"VALUES ('mode_transition', ?, ?, ?, ?, datetime('now'))",
                [(r[0], r[1], r[2], r[3]) for r in rows],
            )
        # RL_MODE_TASK -> rl_task_stat (3 stat rows per task_type)
        if _table_exists(conn, Tables.RL_MODE_TASK):
            rows = conn.execute(
                f"SELECT task_type, optimal_tool_budget, avg_quality_score, avg_completion_rate, sample_count "
                f"FROM {Tables.RL_MODE_TASK}"
            ).fetchall()
            stats = []
            for task_type, budget, quality, completion, count in rows:
                stats += [
                    (
                        "mode_transition",
                        task_type,
                        "optimal_tool_budget",
                        float(budget or 10),
                        count,
                    ),
                    (
                        "mode_transition",
                        task_type,
                        "avg_quality_score",
                        float(quality or 0.5),
                        count,
                    ),
                    (
                        "mode_transition",
                        task_type,
                        "avg_completion_rate",
                        float(completion or 0.5),
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                stats,
            )
        # RL_MODE_HISTORY -> rl_transition
        if _table_exists(conn, Tables.RL_MODE_HISTORY):
            cols = {
                r[1]
                for r in conn.execute(f"PRAGMA table_info({Tables.RL_MODE_HISTORY})").fetchall()
            }
            extra = "profile_name, trigger" if "profile_name" in cols and "trigger" in cols else ""
            rows = conn.execute(
                "SELECT from_mode, to_mode, action_key, reward, success, quality_score"
                + (", profile_name, trigger" if extra else "")
                + f" FROM {Tables.RL_MODE_HISTORY}"
            ).fetchall()
            transitions = []
            for row in rows:
                from_mode, to_mode, action_key, reward, success, quality = row[:6]
                meta = {"success": success, "quality_score": quality}
                if len(row) > 6:
                    meta["profile_name"] = row[6]
                    meta["trigger"] = row[7]
                transitions.append(
                    (
                        "mode_transition",
                        from_mode,
                        to_mode,
                        action_key,
                        reward,
                        json_dumps(meta),
                    )
                )
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TRANSITION} "
                f"(learner_id, from_state, to_state, action, reward, metadata, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                transitions,
            )
        conn.commit()

    @staticmethod
    def migrate_model_selector(conn: sqlite3.Connection) -> None:
        # RL_MODEL_Q -> rl_q_value (global: action_key="select")
        if _table_exists(conn, Tables.RL_MODEL_Q):
            rows = conn.execute(
                f"SELECT provider, q_value, selection_count FROM {Tables.RL_MODEL_Q}"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_Q_VALUE} "
                f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
                f"VALUES ('model_selector', ?, 'select', ?, ?, datetime('now'))",
                [(r[0], r[1], r[2]) for r in rows],
            )
        # RL_MODEL_TASK -> rl_q_value (task-specific: state_key=provider, action_key=task_type)
        if _table_exists(conn, Tables.RL_MODEL_TASK):
            rows = conn.execute(
                f"SELECT provider, task_type, q_value, selection_count FROM {Tables.RL_MODEL_TASK}"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_Q_VALUE} "
                f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
                f"VALUES ('model_selector', ?, ?, ?, ?, datetime('now'))",
                [(r[0], r[1], r[2], r[3]) for r in rows],
            )
        # RL_MODEL_STATE -> rl_param (key-value scalar state)
        if _table_exists(conn, Tables.RL_MODEL_STATE):
            rows = conn.execute(
                f"SELECT key, value FROM {Tables.RL_MODEL_STATE} WHERE key IN ('epsilon', 'total_selections')"
            ).fetchall()
            for key, value in rows:
                try:
                    numeric = float(value)
                except (ValueError, TypeError):
                    numeric = None
                conn.execute(
                    f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                    f"(learner_id, param_key, param_value, context, updated_at) "
                    f"VALUES ('model_selector', ?, ?, NULL, datetime('now'))",
                    (key, numeric),
                )
        # rl_model_threshold -> rl_param (JSON blob via value_text)
        if _table_exists(conn, "rl_model_threshold"):
            rows = conn.execute(
                "SELECT decision_type, observations FROM rl_model_threshold"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                f"(learner_id, param_key, param_value, value_text, context, updated_at) "
                f"VALUES ('model_selector', ?, NULL, ?, NULL, datetime('now'))",
                [(f"threshold:{r[0]}", r[1]) for r in rows],
            )
        conn.commit()

    @staticmethod
    def migrate_tool_selector(conn: sqlite3.Connection) -> None:
        # RL_TOOL_Q -> rl_q_value (global: action_key="select")
        if _table_exists(conn, Tables.RL_TOOL_Q):
            rows = conn.execute(
                f"SELECT tool_name, q_value, selection_count, success_count FROM {Tables.RL_TOOL_Q}"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_Q_VALUE} "
                f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
                f"VALUES ('tool_selector', ?, 'select', ?, ?, datetime('now'))",
                [(r[0], r[1], r[2]) for r in rows],
            )
            # success_count -> rl_task_stat
            stats = [("tool_selector", r[0], "success_count", float(r[3] or 0), r[2]) for r in rows]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                stats,
            )
        # RL_TOOL_TASK -> rl_q_value (state_key=tool_name, action_key=task_type)
        if _table_exists(conn, Tables.RL_TOOL_TASK):
            rows = conn.execute(
                f"SELECT tool_name, task_type, q_value, selection_count FROM {Tables.RL_TOOL_TASK}"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_Q_VALUE} "
                f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
                f"VALUES ('tool_selector', ?, ?, ?, ?, datetime('now'))",
                [(r[0], r[1], r[2], r[3]) for r in rows],
            )
        # RL_TOOL_OUTCOME -> rl_transition
        if _table_exists(conn, Tables.RL_TOOL_OUTCOME):
            rows = conn.execute(
                f"SELECT tool_name, task_type, success, quality_score, reward, metadata FROM {Tables.RL_TOOL_OUTCOME}"
            ).fetchall()
            transitions = []
            for tool_name, task_type, success, quality, reward, meta in rows:
                extra = {}
                if meta:
                    try:
                        extra = json_loads(meta)
                    except Exception:
                        pass
                extra.update({"success": success, "quality_score": quality})
                transitions.append(
                    (
                        "tool_selector",
                        task_type,
                        tool_name,
                        "execute",
                        reward,
                        json_dumps(extra),
                    )
                )
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TRANSITION} "
                f"(learner_id, from_state, to_state, action, reward, metadata, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                transitions,
            )
        conn.commit()

    @staticmethod
    def migrate_cache_eviction(conn: sqlite3.Connection) -> None:
        # RL_CACHE_Q -> rl_q_value
        if _table_exists(conn, Tables.RL_CACHE_Q):
            rows = conn.execute(
                f"SELECT state_key, action, q_value, visit_count FROM {Tables.RL_CACHE_Q}"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_Q_VALUE} "
                f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
                f"VALUES ('cache_eviction', ?, ?, ?, ?, datetime('now'))",
                [(r[0], r[1], r[2], r[3]) for r in rows],
            )
        # RL_CACHE_TOOL -> rl_task_stat (3 rows per tool_name)
        if _table_exists(conn, Tables.RL_CACHE_TOOL):
            rows = conn.execute(
                f"SELECT tool_name, estimated_value, hit_count, miss_count FROM {Tables.RL_CACHE_TOOL}"
            ).fetchall()
            stats = []
            for tool_name, est_val, hit_count, miss_count in rows:
                total = (hit_count or 0) + (miss_count or 0)
                stats += [
                    (
                        "cache_eviction",
                        tool_name,
                        "estimated_value",
                        float(est_val or 0.5),
                        total,
                    ),
                    (
                        "cache_eviction",
                        tool_name,
                        "hit_count",
                        float(hit_count or 0),
                        total,
                    ),
                    (
                        "cache_eviction",
                        tool_name,
                        "miss_count",
                        float(miss_count or 0),
                        total,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                stats,
            )
        # RL_CACHE_HISTORY -> rl_transition
        if _table_exists(conn, Tables.RL_CACHE_HISTORY):
            rows = conn.execute(
                f"SELECT state_key, action, tool_name, reward, hit_after FROM {Tables.RL_CACHE_HISTORY}"
            ).fetchall()
            transitions = [
                (
                    "cache_eviction",
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    json_dumps({"hit_after": r[4]}),
                )
                for r in rows
            ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TRANSITION} "
                f"(learner_id, from_state, to_state, action, reward, metadata, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                transitions,
            )
        conn.commit()

    @staticmethod
    def migrate_grounding_threshold(conn: sqlite3.Connection) -> None:
        # RL_GROUNDING_PARAM -> rl_param (alpha + beta as separate rows per context_key:threshold)
        if _table_exists(conn, Tables.RL_GROUNDING_PARAM):
            rows = conn.execute(
                f"SELECT context_key, threshold, alpha, beta, sample_count FROM {Tables.RL_GROUNDING_PARAM}"
            ).fetchall()
            params = []
            for ctx, thresh, alpha, beta, count in rows:
                key_suffix = f"{ctx}:{thresh}"
                params += [
                    (
                        "grounding_threshold",
                        f"alpha:{key_suffix}",
                        float(alpha or 1.0),
                        None,
                        count,
                    ),
                    (
                        "grounding_threshold",
                        f"beta:{key_suffix}",
                        float(beta or 1.0),
                        None,
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                f"(learner_id, param_key, param_value, context, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                params,
            )
        # RL_GROUNDING_STAT -> rl_task_stat (4 confusion matrix rows per provider)
        if _table_exists(conn, Tables.RL_GROUNDING_STAT):
            rows = conn.execute(
                f"SELECT provider, true_positives, true_negatives, false_positives, false_negatives "
                f"FROM {Tables.RL_GROUNDING_STAT}"
            ).fetchall()
            stats = []
            for provider, tp, tn, fp, fn in rows:
                total = (tp or 0) + (tn or 0) + (fp or 0) + (fn or 0)
                stats += [
                    (
                        "grounding_threshold",
                        provider,
                        "true_positives",
                        float(tp or 0),
                        total,
                    ),
                    (
                        "grounding_threshold",
                        provider,
                        "true_negatives",
                        float(tn or 0),
                        total,
                    ),
                    (
                        "grounding_threshold",
                        provider,
                        "false_positives",
                        float(fp or 0),
                        total,
                    ),
                    (
                        "grounding_threshold",
                        provider,
                        "false_negatives",
                        float(fn or 0),
                        total,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                stats,
            )
        # RL_GROUNDING_HISTORY -> rl_transition
        if _table_exists(conn, Tables.RL_GROUNDING_HISTORY):
            rows = conn.execute(
                f"SELECT context_key, result_type, threshold, reward FROM {Tables.RL_GROUNDING_HISTORY}"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TRANSITION} "
                f"(learner_id, from_state, to_state, action, reward, metadata, created_at) "
                f"VALUES ('grounding_threshold', ?, ?, ?, ?, '{{}}', datetime('now'))",
                [(r[0], r[1], str(r[2]), r[3]) for r in rows],
            )
        conn.commit()

    @staticmethod
    def migrate_semantic_threshold(conn: sqlite3.Connection) -> None:
        # RL_SEMANTIC_STAT -> rl_task_stat (multiple stat rows per context_key)
        if _table_exists(conn, Tables.RL_SEMANTIC_STAT):
            rows = conn.execute(
                f"SELECT context_key, total_searches, zero_result_count, low_quality_count, "
                f"avg_results_count, avg_threshold, recommended_threshold "
                f"FROM {Tables.RL_SEMANTIC_STAT}"
            ).fetchall()
            stats = []
            for ctx, total, zero, low_q, avg_res, avg_thresh, rec_thresh in rows:
                count = total or 0
                stats += [
                    (
                        "semantic_threshold",
                        ctx,
                        "avg_threshold",
                        float(avg_thresh or 0),
                        count,
                    ),
                    (
                        "semantic_threshold",
                        ctx,
                        "recommended_threshold",
                        float(rec_thresh or 0),
                        count,
                    ),
                    (
                        "semantic_threshold",
                        ctx,
                        "zero_result_count",
                        float(zero or 0),
                        count,
                    ),
                    (
                        "semantic_threshold",
                        ctx,
                        "low_quality_count",
                        float(low_q or 0),
                        count,
                    ),
                    (
                        "semantic_threshold",
                        ctx,
                        "avg_results_count",
                        float(avg_res or 0),
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                stats,
            )
        conn.commit()

    @staticmethod
    def migrate_continuation_patience(conn: sqlite3.Connection) -> None:
        # RL_PATIENCE_STAT -> rl_param (current_patience) + rl_task_stat (counts)
        if _table_exists(conn, Tables.RL_PATIENCE_STAT):
            rows = conn.execute(
                f"SELECT context_key, provider, model, task_type, current_patience, "
                f"total_sessions, false_positives, true_positives, missed_stuck_loops "
                f"FROM {Tables.RL_PATIENCE_STAT}"
            ).fetchall()
            params = []
            stats = []
            for (
                ctx,
                provider,
                model,
                task_type,
                patience,
                total,
                fp,
                tp,
                missed,
            ) in rows:
                count = total or 0
                params.append(
                    (
                        "continuation_patience",
                        "current_patience",
                        float(patience or 5),
                        ctx,
                        count,
                    )
                )
                stats += [
                    (
                        "continuation_patience",
                        ctx,
                        "total_sessions",
                        float(total or 0),
                        count,
                    ),
                    (
                        "continuation_patience",
                        ctx,
                        "false_positives",
                        float(fp or 0),
                        count,
                    ),
                    (
                        "continuation_patience",
                        ctx,
                        "true_positives",
                        float(tp or 0),
                        count,
                    ),
                    (
                        "continuation_patience",
                        ctx,
                        "missed_stuck_loops",
                        float(missed or 0),
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                f"(learner_id, param_key, param_value, context, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                params,
            )
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                stats,
            )
        conn.commit()

    @staticmethod
    def migrate_continuation_prompts(conn: sqlite3.Connection) -> None:
        # RL_PROMPT_STAT -> rl_param (current_max_prompts) + rl_task_stat (counts)
        if _table_exists(conn, Tables.RL_PROMPT_STAT):
            rows = conn.execute(
                f"SELECT context_key, provider, model, task_type, current_max_prompts, "
                f"total_sessions, successful_sessions, stuck_loop_count, forced_completion_count "
                f"FROM {Tables.RL_PROMPT_STAT}"
            ).fetchall()
            params = []
            stats = []
            for (
                ctx,
                provider,
                model,
                task_type,
                max_p,
                total,
                success,
                stuck,
                forced,
            ) in rows:
                count = total or 0
                params.append(
                    (
                        "continuation_prompts",
                        "current_max_prompts",
                        float(max_p or 3),
                        ctx,
                        count,
                    )
                )
                stats += [
                    (
                        "continuation_prompts",
                        ctx,
                        "total_sessions",
                        float(total or 0),
                        count,
                    ),
                    (
                        "continuation_prompts",
                        ctx,
                        "successful_sessions",
                        float(success or 0),
                        count,
                    ),
                    (
                        "continuation_prompts",
                        ctx,
                        "stuck_loop_count",
                        float(stuck or 0),
                        count,
                    ),
                    (
                        "continuation_prompts",
                        ctx,
                        "forced_completion_count",
                        float(forced or 0),
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                f"(learner_id, param_key, param_value, context, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                params,
            )
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                stats,
            )
        conn.commit()

    @staticmethod
    def migrate_quality_weights(conn: sqlite3.Connection) -> None:
        # RL_QUALITY_WEIGHT -> rl_param (weight + velocity per task_type:dimension)
        if _table_exists(conn, Tables.RL_QUALITY_WEIGHT):
            rows = conn.execute(
                f"SELECT task_type, dimension, weight, velocity, sample_count FROM {Tables.RL_QUALITY_WEIGHT}"
            ).fetchall()
            params = []
            for task_type, dimension, weight, velocity, count in rows:
                params += [
                    (
                        "quality_weights",
                        f"weight:{dimension}",
                        float(weight or 0.5),
                        task_type,
                        count,
                    ),
                    (
                        "quality_weights",
                        f"velocity:{dimension}",
                        float(velocity or 0.0),
                        task_type,
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                f"(learner_id, param_key, param_value, context, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                params,
            )
        # rl_user_weight_preference -> rl_param (user-scoped weights)
        if _table_exists(conn, "rl_user_weight_preference"):
            rows = conn.execute(
                "SELECT user_id, task_type, dimension, weight FROM rl_user_weight_preference"
            ).fetchall()
            params = [
                (
                    "quality_weights",
                    f"user_weight:{r[0]}:{r[1]}:{r[2]}",
                    float(r[3] or 0.5),
                    None,
                    0,
                )
                for r in rows
            ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                f"(learner_id, param_key, param_value, context, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                params,
            )
        # RL_QUALITY_HISTORY -> rl_transition
        if _table_exists(conn, Tables.RL_QUALITY_HISTORY):
            rows = conn.execute(
                f"SELECT task_type, dimension_scores, overall_success, weights_used FROM {Tables.RL_QUALITY_HISTORY}"
            ).fetchall()
            transitions = [
                (
                    "quality_weights",
                    r[0],
                    None,
                    None,
                    r[2],
                    json_dumps({"dimension_scores": r[1], "weights_used": r[3]}),
                )
                for r in rows
            ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TRANSITION} "
                f"(learner_id, from_state, to_state, action, reward, metadata, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                transitions,
            )
        conn.commit()

    @staticmethod
    def migrate_context_pruning(conn: sqlite3.Connection) -> None:
        # RL_CONTEXT_PRUNING -> rl_q_value
        if _table_exists(conn, Tables.RL_CONTEXT_PRUNING):
            rows = conn.execute(
                f"SELECT state_key, action, q_value, visit_count FROM {Tables.RL_CONTEXT_PRUNING}"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_Q_VALUE} "
                f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
                f"VALUES ('context_pruning', ?, ?, ?, ?, datetime('now'))",
                [(r[0], r[1], r[2], r[3]) for r in rows],
            )
        # RL_CONTEXT_PRUNING_stats -> rl_task_stat
        stats_table = f"{Tables.RL_CONTEXT_PRUNING}_stats"
        if _table_exists(conn, stats_table):
            rows = conn.execute(
                f"SELECT provider_type, total_decisions, total_tokens_saved, avg_success_rate "
                f"FROM {stats_table}"
            ).fetchall()
            stats = []
            for provider_type, total, tokens_saved, avg_success in rows:
                count = total or 0
                stats += [
                    (
                        "context_pruning",
                        provider_type,
                        "total_decisions",
                        float(total or 0),
                        count,
                    ),
                    (
                        "context_pruning",
                        provider_type,
                        "total_tokens_saved",
                        float(tokens_saved or 0),
                        count,
                    ),
                    (
                        "context_pruning",
                        provider_type,
                        "avg_success_rate",
                        float(avg_success or 0),
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                stats,
            )
        conn.commit()

    @staticmethod
    def migrate_workflow_execution(conn: sqlite3.Connection) -> None:
        # AGENT_WORKFLOW_Q -> rl_q_value (state_key=workflow_name, action_key=task_type)
        if _table_exists(conn, Tables.AGENT_WORKFLOW_Q):
            rows = conn.execute(
                f"SELECT workflow_name, task_type, q_value, execution_count FROM {Tables.AGENT_WORKFLOW_Q}"
            ).fetchall()
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_Q_VALUE} "
                f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
                f"VALUES ('workflow_execution', ?, ?, ?, ?, datetime('now'))",
                [(r[0], r[1], r[2], r[3]) for r in rows],
            )
        # AGENT_WORKFLOW_RUN -> rl_transition
        if _table_exists(conn, Tables.AGENT_WORKFLOW_RUN):
            rows = conn.execute(
                f"SELECT workflow_name, task_type, success, duration_seconds, quality_score, vertical, mode "
                f"FROM {Tables.AGENT_WORKFLOW_RUN}"
            ).fetchall()
            transitions = [
                (
                    "workflow_execution",
                    r[1],
                    None,
                    r[0],
                    None,
                    json_dumps(
                        {
                            "success": r[2],
                            "duration": r[3],
                            "quality_score": r[4],
                            "vertical": r[5],
                            "mode": r[6],
                        }
                    ),
                )
                for r in rows
            ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TRANSITION} "
                f"(learner_id, from_state, to_state, action, reward, metadata, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                transitions,
            )
        conn.commit()

    @staticmethod
    def migrate_prompt_template(conn: sqlite3.Connection) -> None:
        # AGENT_PROMPT_STYLE -> rl_param (alpha + beta per style, context=task_type:provider)
        if _table_exists(conn, Tables.AGENT_PROMPT_STYLE):
            rows = conn.execute(
                f"SELECT task_type, provider, style, alpha, beta, sample_count FROM {Tables.AGENT_PROMPT_STYLE}"
            ).fetchall()
            params = []
            for task_type, provider, style, alpha, beta, count in rows:
                ctx = f"{task_type}:{provider}"
                params += [
                    (
                        "prompt_template",
                        f"style_alpha:{style}",
                        float(alpha or 1.0),
                        ctx,
                        count,
                    ),
                    (
                        "prompt_template",
                        f"style_beta:{style}",
                        float(beta or 1.0),
                        ctx,
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                f"(learner_id, param_key, param_value, context, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                params,
            )
        # AGENT_PROMPT_ELEMENT -> rl_param (alpha + beta per element)
        if _table_exists(conn, Tables.AGENT_PROMPT_ELEMENT):
            rows = conn.execute(
                f"SELECT task_type, provider, element, alpha, beta, sample_count FROM {Tables.AGENT_PROMPT_ELEMENT}"
            ).fetchall()
            params = []
            for task_type, provider, element, alpha, beta, count in rows:
                ctx = f"{task_type}:{provider}"
                params += [
                    (
                        "prompt_template",
                        f"elem_alpha:{element}",
                        float(alpha or 1.0),
                        ctx,
                        count,
                    ),
                    (
                        "prompt_template",
                        f"elem_beta:{element}",
                        float(beta or 1.0),
                        ctx,
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                f"(learner_id, param_key, param_value, context, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                params,
            )
        # agent_enrichment_stats -> rl_param (alpha + beta + JSON blob)
        if _table_exists(conn, "agent_enrichment_stats"):
            rows = conn.execute(
                "SELECT vertical, enrichment_type, task_type, alpha, beta, sample_count "
                "FROM agent_enrichment_stats"
            ).fetchall()
            params = []
            for vertical, enr_type, task_type, alpha, beta, count in rows:
                ctx = f"{vertical}:{task_type}"
                params += [
                    (
                        "prompt_template",
                        f"enrichment_alpha:{enr_type}",
                        float(alpha or 1.0),
                        ctx,
                        count,
                    ),
                    (
                        "prompt_template",
                        f"enrichment_beta:{enr_type}",
                        float(beta or 1.0),
                        ctx,
                        count,
                    ),
                ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_PARAM} "
                f"(learner_id, param_key, param_value, context, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, datetime('now'))",
                params,
            )
        # AGENT_PROMPT_HISTORY -> rl_transition
        if _table_exists(conn, Tables.AGENT_PROMPT_HISTORY):
            rows = conn.execute(
                f"SELECT task_type, provider, model, template_used, success FROM {Tables.AGENT_PROMPT_HISTORY}"
            ).fetchall()
            transitions = [
                (
                    "prompt_template",
                    r[0],
                    None,
                    r[3],
                    None,
                    json_dumps({"provider": r[1], "model": r[2], "success": r[4]}),
                )
                for r in rows
            ]
            conn.executemany(
                f"INSERT OR IGNORE INTO {Tables.RL_TRANSITION} "
                f"(learner_id, from_state, to_state, action, reward, metadata, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                transitions,
            )
        conn.commit()
