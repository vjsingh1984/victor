# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Database schema constants and definitions for Victor.

This module provides centralized table name constants and schema definitions
to ensure consistency across the codebase. All database operations should
reference these constants instead of hardcoding table names.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    TABLE NAMING CONVENTION                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Domain Prefix:                                                      │
    │  ├── rl_     → Reinforcement Learning tables                        │
    │  ├── agent_  → Agent execution & teams                              │
    │  ├── ui_     → User interface (sessions, preferences)               │
    │  └── sys_    → System metadata                                      │
    │                                                                      │
    │  Naming Pattern: {domain}_{entity}                                   │
    │  Examples: rl_outcome, agent_team_run, ui_session                   │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    from victor.core.schema import Tables, Schema

    # Reference table names
    cursor.execute(f"SELECT * FROM {Tables.RL_OUTCOME}")

    # Get full schema SQL
    schema_sql = Schema.RL_OUTCOME
"""

from __future__ import annotations


class Tables:
    """Centralized table name constants.

    All table names are defined here to ensure consistency and enable
    easy refactoring. Use these constants instead of hardcoding table names.

    Naming Convention:
    - Domain prefix: rl_, agent_, ui_, sys_
    - Singular nouns: rl_outcome not rl_outcomes
    - Max 3 words: concise and descriptive
    """

    # ===========================================
    # SYSTEM DOMAIN (sys_)
    # ===========================================
    SYS_METADATA = "sys_metadata"
    SYS_SCHEMA_VERSION = "sys_schema_version"

    # ===========================================
    # REINFORCEMENT LEARNING DOMAIN (rl_)
    # ===========================================

    # Core RL tables (consolidated)
    RL_LEARNER = "rl_learner"  # Learner registry/configuration
    RL_OUTCOME = "rl_outcome"  # All learner outcomes (central fact table)
    RL_METRIC = "rl_metric"  # Telemetry and monitoring
    RL_Q_VALUE = "rl_q_value"  # Unified Q-values (partitioned by learner_id)
    RL_TRANSITION = "rl_transition"  # Unified state transitions (partitioned by learner_id)
    RL_PARAM = "rl_param"  # Unified parameters (partitioned by learner_id)
    RL_TASK_STAT = "rl_task_stat"  # Unified task statistics (partitioned by learner_id)

    # Mode transition learner
    RL_MODE_Q = "rl_mode_q"  # Mode transition Q-values
    RL_MODE_HISTORY = "rl_mode_history"  # Mode transition history
    RL_MODE_TASK = "rl_mode_task"  # Mode task statistics

    # Model selector learner
    RL_MODEL_Q = "rl_model_q"  # Model selector Q-values
    RL_MODEL_TASK = "rl_model_task"  # Model task Q-values
    RL_MODEL_STATE = "rl_model_state"  # Model selector state

    # Tool selector learner
    RL_TOOL_Q = "rl_tool_q"  # Tool selector Q-values
    RL_TOOL_TASK = "rl_tool_task"  # Tool task Q-values
    RL_TOOL_OUTCOME = "rl_tool_outcome"  # Tool outcomes

    # Cache eviction learner
    RL_CACHE_Q = "rl_cache_q"  # Cache eviction Q-values
    RL_CACHE_TOOL = "rl_cache_tool"  # Cache tool values
    RL_CACHE_HISTORY = "rl_cache_history"  # Cache eviction history

    # Grounding threshold learner
    RL_GROUNDING_PARAM = "rl_grounding_param"  # Grounding parameters
    RL_GROUNDING_STAT = "rl_grounding_stat"  # Grounding statistics
    RL_GROUNDING_HISTORY = "rl_grounding_history"  # Grounding history

    # Other learner stats
    RL_SEMANTIC_STAT = "rl_semantic_stat"  # Semantic threshold stats
    RL_PATIENCE_STAT = "rl_patience_stat"  # Continuation patience stats
    RL_PROMPT_STAT = "rl_prompt_stat"  # Continuation prompt stats
    RL_QUALITY_WEIGHT = "rl_quality_weight"  # Quality weights
    RL_QUALITY_HISTORY = "rl_quality_history"  # Quality weight history

    # Context pruning (token optimization)
    RL_CONTEXT_PRUNING = "rl_context_pruning"  # Context pruning Q-values

    # Cross-vertical learning
    RL_PATTERN = "rl_pattern"  # Cross-vertical patterns
    RL_PATTERN_USE = "rl_pattern_use"  # Pattern application tracking

    # ===========================================
    # AGENT DOMAIN (agent_)
    # ===========================================

    # Team execution
    AGENT_TEAM_CONFIG = "agent_team_config"  # Team composition Q-values
    AGENT_TEAM_RUN = "agent_team_run"  # Team execution records

    # Workflow execution
    AGENT_WORKFLOW_RUN = "agent_workflow_run"  # Workflow execution records
    AGENT_WORKFLOW_Q = "agent_workflow_q"  # Workflow Q-values

    # Prompt templates
    AGENT_PROMPT_STYLE = "agent_prompt_style"  # Prompt style definitions
    AGENT_PROMPT_ELEMENT = "agent_prompt_element"  # Prompt components
    AGENT_PROMPT_HISTORY = "agent_prompt_history"  # Prompt history

    # Curriculum & Policy
    AGENT_CURRICULUM_STAGE = "agent_curriculum_stage"  # Learning curriculum
    AGENT_CURRICULUM_METRIC = "agent_curriculum_metric"  # Curriculum performance
    AGENT_CURRICULUM_HISTORY = "agent_curriculum_history"  # Curriculum history
    AGENT_POLICY_SNAPSHOT = "agent_policy_snapshot"  # Policy state snapshots

    # ===========================================
    # GRAPH DOMAIN (graph_) - Project-level tables
    # ===========================================

    # Core graph tables (stored in project.db)
    GRAPH_NODE = "graph_node"  # Symbol nodes (functions, classes, etc.)
    GRAPH_EDGE = "graph_edge"  # References, calls, imports between nodes
    GRAPH_FILE_MTIME = "graph_file_mtime"  # File modification times for staleness

    # ===========================================
    # UI DOMAIN (ui_)
    # ===========================================
    UI_SESSION = "ui_session"  # TUI session persistence
    UI_FAILED_CALL = "ui_failed_call"  # Failed tool call signatures

    # ===========================================
    # LEARNER IDENTIFIERS
    # ===========================================


class LearnerID:
    """Learner identifiers for partitioning unified tables.

    When using consolidated tables like rl_q_value, use these IDs
    to partition data by learner.
    """

    MODE_TRANSITION = "mode_transition"
    MODEL_SELECTOR = "model_selector"
    TOOL_SELECTOR = "tool_selector"
    CACHE_EVICTION = "cache_eviction"
    GROUNDING_THRESHOLD = "grounding_threshold"
    QUALITY_WEIGHT = "quality_weight"
    CONTINUATION_PATIENCE = "continuation_patience"
    CONTINUATION_PROMPT = "continuation_prompt"
    SEMANTIC_THRESHOLD = "semantic_threshold"
    PROMPT_TEMPLATE = "prompt_template"
    CROSS_VERTICAL = "cross_vertical"
    WORKFLOW_EXECUTION = "workflow_execution"
    TEAM_COMPOSITION = "team_composition"

    @classmethod
    def all(cls) -> list[str]:
        """Get all learner IDs."""
        return [
            cls.MODE_TRANSITION,
            cls.MODEL_SELECTOR,
            cls.TOOL_SELECTOR,
            cls.CACHE_EVICTION,
            cls.GROUNDING_THRESHOLD,
            cls.QUALITY_WEIGHT,
            cls.CONTINUATION_PATIENCE,
            cls.CONTINUATION_PROMPT,
            cls.SEMANTIC_THRESHOLD,
            cls.PROMPT_TEMPLATE,
            cls.CROSS_VERTICAL,
            cls.WORKFLOW_EXECUTION,
            cls.TEAM_COMPOSITION,
        ]


class Schema:
    """SQL schema definitions for all tables.

    Use these to create tables with consistent structure.
    """

    # ===========================================
    # SYSTEM TABLES
    # ===========================================

    SYS_METADATA = f"""
        CREATE TABLE IF NOT EXISTS {Tables.SYS_METADATA} (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """

    SYS_SCHEMA_VERSION = f"""
        CREATE TABLE IF NOT EXISTS {Tables.SYS_SCHEMA_VERSION} (
            version INTEGER PRIMARY KEY,
            applied_at TEXT DEFAULT (datetime('now'))
        )
    """

    # ===========================================
    # RL TABLES
    # ===========================================

    RL_LEARNER = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_LEARNER} (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            config TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT
        )
    """

    RL_OUTCOME = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_OUTCOME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            task_type TEXT,
            vertical TEXT DEFAULT 'coding',
            success INTEGER,
            quality_score REAL,
            metadata TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    RL_OUTCOME_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_rl_outcome_learner
            ON {Tables.RL_OUTCOME}(learner_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_rl_outcome_context
            ON {Tables.RL_OUTCOME}(provider, model, task_type);
    """

    RL_Q_VALUE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_Q_VALUE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            state_key TEXT NOT NULL,
            action_key TEXT NOT NULL,
            q_value REAL DEFAULT 0.5,
            visit_count INTEGER DEFAULT 0,
            last_updated TEXT DEFAULT (datetime('now')),
            UNIQUE(learner_id, state_key, action_key)
        )
    """

    RL_Q_VALUE_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_rl_q_learner
            ON {Tables.RL_Q_VALUE}(learner_id);
    """

    RL_TRANSITION = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_TRANSITION} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            from_state TEXT,
            to_state TEXT,
            action TEXT,
            reward REAL,
            metadata TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    RL_TRANSITION_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_rl_trans_learner
            ON {Tables.RL_TRANSITION}(learner_id, created_at);
    """

    RL_PARAM = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_PARAM} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            param_key TEXT NOT NULL,
            param_value REAL,
            context TEXT,
            sample_count INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.5,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(learner_id, param_key, context)
        )
    """

    RL_TASK_STAT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_TASK_STAT} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            stat_key TEXT NOT NULL,
            stat_value REAL,
            sample_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(learner_id, task_type, stat_key)
        )
    """

    RL_PATTERN = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_PATTERN} (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            pattern_name TEXT,
            avg_quality REAL,
            confidence REAL,
            source_verticals TEXT,
            recommendation TEXT,
            sample_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT
        )
    """

    RL_PATTERN_USE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_PATTERN_USE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id TEXT NOT NULL,
            target_vertical TEXT,
            success INTEGER,
            quality_score REAL,
            applied_at TEXT DEFAULT (datetime('now'))
        )
    """

    RL_METRIC = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_METRIC} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT,
            metric_type TEXT NOT NULL,
            metric_value REAL,
            metadata TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    # ===========================================
    # AGENT TABLES
    # ===========================================

    AGENT_TEAM_CONFIG = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_TEAM_CONFIG} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key TEXT UNIQUE NOT NULL,
            formation TEXT NOT NULL,
            role_counts TEXT NOT NULL,
            task_category TEXT,
            execution_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            avg_quality REAL DEFAULT 0.5,
            avg_duration REAL DEFAULT 0,
            q_value REAL DEFAULT 0.5,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_TEAM_RUN = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_TEAM_RUN} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            task_category TEXT,
            formation TEXT,
            role_counts TEXT,
            member_count INTEGER,
            budget_used INTEGER,
            tools_used INTEGER,
            success INTEGER,
            quality_score REAL,
            duration_seconds REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_WORKFLOW_RUN = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_WORKFLOW_RUN} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_name TEXT NOT NULL,
            task_type TEXT,
            success INTEGER,
            duration_seconds REAL,
            quality_score REAL,
            vertical TEXT,
            mode TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_PROMPT_STYLE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_PROMPT_STYLE} (
            id TEXT PRIMARY KEY,
            style_name TEXT NOT NULL,
            description TEXT,
            template TEXT,
            success_rate REAL DEFAULT 0.5,
            usage_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_PROMPT_ELEMENT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_PROMPT_ELEMENT} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style_id TEXT,
            element_type TEXT,
            content TEXT,
            position INTEGER
        )
    """

    AGENT_CURRICULUM_STAGE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_CURRICULUM_STAGE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage_name TEXT NOT NULL,
            difficulty REAL,
            prerequisites TEXT,
            completion_criteria TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_CURRICULUM_METRIC = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_CURRICULUM_METRIC} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage_id INTEGER,
            metric_name TEXT,
            metric_value REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_POLICY_SNAPSHOT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_POLICY_SNAPSHOT} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT,
            snapshot_name TEXT,
            policy_data TEXT,
            performance_score REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    # ===========================================
    # UI TABLES
    # ===========================================

    UI_SESSION = f"""
        CREATE TABLE IF NOT EXISTS {Tables.UI_SESSION} (
            id TEXT PRIMARY KEY,
            name TEXT,
            provider TEXT,
            model TEXT,
            profile TEXT,
            data TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT
        )
    """

    UI_SESSION_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_ui_session_updated
            ON {Tables.UI_SESSION}(updated_at DESC);
    """

    UI_FAILED_CALL = f"""
        CREATE TABLE IF NOT EXISTS {Tables.UI_FAILED_CALL} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            args_hash TEXT NOT NULL,
            args_json TEXT,
            error_message TEXT,
            failure_count INTEGER DEFAULT 1,
            first_seen REAL,
            last_seen REAL,
            expires_at REAL,
            UNIQUE(tool_name, args_hash)
        )
    """

    UI_FAILED_CALL_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_ui_failed_lookup
            ON {Tables.UI_FAILED_CALL}(tool_name, args_hash);
        CREATE INDEX IF NOT EXISTS idx_ui_failed_expires
            ON {Tables.UI_FAILED_CALL}(expires_at);
    """

    # ===========================================
    # GRAPH TABLES (project-level)
    # ===========================================

    GRAPH_NODE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_NODE} (
            node_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            file TEXT NOT NULL,
            line INTEGER,
            end_line INTEGER,
            lang TEXT,
            signature TEXT,
            docstring TEXT,
            parent_id TEXT,
            embedding_ref TEXT,
            metadata TEXT,
            FOREIGN KEY (parent_id) REFERENCES {Tables.GRAPH_NODE}(node_id)
        )
    """

    GRAPH_EDGE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_EDGE} (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL,
            metadata TEXT,
            PRIMARY KEY (src, dst, type)
        )
    """

    GRAPH_FILE_MTIME = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_FILE_MTIME} (
            file TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            indexed_at REAL NOT NULL
        )
    """

    GRAPH_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_graph_node_type_name
            ON {Tables.GRAPH_NODE}(type, name);
        CREATE INDEX IF NOT EXISTS idx_graph_node_file
            ON {Tables.GRAPH_NODE}(file);
        CREATE INDEX IF NOT EXISTS idx_graph_node_parent
            ON {Tables.GRAPH_NODE}(parent_id);
        CREATE INDEX IF NOT EXISTS idx_graph_edge_src_type
            ON {Tables.GRAPH_EDGE}(src, type);
        CREATE INDEX IF NOT EXISTS idx_graph_edge_dst_type
            ON {Tables.GRAPH_EDGE}(dst, type);
        CREATE INDEX IF NOT EXISTS idx_graph_file_mtime
            ON {Tables.GRAPH_FILE_MTIME}(mtime);
    """

    # ===========================================
    # CONVERSATION TABLES (project-level)
    # ===========================================

    CONV_MESSAGE = """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    CONV_SESSION = """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT,
            provider TEXT,
            model TEXT,
            profile TEXT,
            data TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT
        )
    """

    CONV_CONTEXT_SIZE = """
        CREATE TABLE IF NOT EXISTS context_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            token_count INTEGER,
            message_count INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    CONV_CONTEXT_SUMMARY = """
        CREATE TABLE IF NOT EXISTS context_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            summary TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    CONV_INDEXES = """
        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_context_sizes_session
            ON context_sizes(session_id);
    """

    # ===========================================
    # MODE LEARNING TABLES (project-level)
    # ===========================================

    MODE_Q_VALUE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_MODE_Q} (
            state_key TEXT NOT NULL,
            action_key TEXT NOT NULL,
            q_value REAL NOT NULL DEFAULT 0.0,
            visit_count INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL,
            PRIMARY KEY (state_key, action_key)
        )
    """

    MODE_TASK_STAT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_MODE_TASK} (
            task_type TEXT PRIMARY KEY,
            optimal_tool_budget INTEGER DEFAULT 10,
            avg_quality_score REAL DEFAULT 0.5,
            avg_completion_rate REAL DEFAULT 0.5,
            sample_count INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL
        )
    """

    MODE_TRANSITION_HISTORY = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_MODE_HISTORY} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_name TEXT NOT NULL,
            from_mode TEXT NOT NULL,
            to_mode TEXT NOT NULL,
            trigger TEXT NOT NULL,
            state_key TEXT NOT NULL,
            action_key TEXT NOT NULL,
            reward REAL,
            timestamp TEXT NOT NULL
        )
    """

    MODE_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_{Tables.RL_MODE_Q}_state
            ON {Tables.RL_MODE_Q}(state_key);
        CREATE INDEX IF NOT EXISTS idx_{Tables.RL_MODE_HISTORY}_profile
            ON {Tables.RL_MODE_HISTORY}(profile_name, timestamp);
    """

    # ===========================================
    # PROFILE LEARNING TABLES (project-level)
    # ===========================================

    PROFILE_INTERACTION = """
        CREATE TABLE IF NOT EXISTS interaction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_type TEXT NOT NULL,
            context TEXT,
            response_style TEXT,
            success INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    PROFILE_METRIC = """
        CREATE TABLE IF NOT EXISTS profile_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            context TEXT,
            sample_count INTEGER DEFAULT 1,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(metric_name, context)
        )
    """

    PROFILE_INDEXES = """
        CREATE INDEX IF NOT EXISTS idx_interaction_type
            ON interaction_history(interaction_type);
    """

    # ===========================================
    # CHANGES TABLES (project-level)
    # ===========================================

    CHANGES_GROUP = """
        CREATE TABLE IF NOT EXISTS change_groups (
            id TEXT PRIMARY KEY,
            description TEXT,
            commit_hash TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            finalized_at TEXT
        )
    """

    CHANGES_FILE = """
        CREATE TABLE IF NOT EXISTS file_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            change_type TEXT NOT NULL,
            old_content TEXT,
            new_content TEXT,
            diff TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (group_id) REFERENCES change_groups(id)
        )
    """

    CHANGES_INDEXES = """
        CREATE INDEX IF NOT EXISTS idx_file_changes_group
            ON file_changes(group_id);
        CREATE INDEX IF NOT EXISTS idx_file_changes_path
            ON file_changes(file_path);
    """

    @classmethod
    def get_all_schemas(cls) -> list[str]:
        """Get all schema definitions in creation order."""
        return [
            # System
            cls.SYS_METADATA,
            cls.SYS_SCHEMA_VERSION,
            # RL
            cls.RL_LEARNER,
            cls.RL_OUTCOME,
            cls.RL_Q_VALUE,
            cls.RL_TRANSITION,
            cls.RL_PARAM,
            cls.RL_TASK_STAT,
            cls.RL_PATTERN,
            cls.RL_PATTERN_USE,
            cls.RL_METRIC,
            # Agent
            cls.AGENT_TEAM_CONFIG,
            cls.AGENT_TEAM_RUN,
            cls.AGENT_WORKFLOW_RUN,
            cls.AGENT_PROMPT_STYLE,
            cls.AGENT_PROMPT_ELEMENT,
            cls.AGENT_CURRICULUM_STAGE,
            cls.AGENT_CURRICULUM_METRIC,
            cls.AGENT_POLICY_SNAPSHOT,
            # UI
            cls.UI_SESSION,
            cls.UI_FAILED_CALL,
        ]

    @classmethod
    def get_all_indexes(cls) -> list[str]:
        """Get all index definitions."""
        return [
            cls.RL_OUTCOME_INDEXES,
            cls.RL_Q_VALUE_INDEXES,
            cls.RL_TRANSITION_INDEXES,
            cls.UI_SESSION_INDEXES,
            cls.UI_FAILED_CALL_INDEXES,
        ]

    @classmethod
    def get_project_schemas(cls) -> list[str]:
        """Get schema definitions for project-level tables (graph, etc.)."""
        return [
            # Graph
            cls.GRAPH_NODE,
            cls.GRAPH_EDGE,
            cls.GRAPH_FILE_MTIME,
            # Conversation
            cls.CONV_MESSAGE,
            cls.CONV_SESSION,
            cls.CONV_CONTEXT_SIZE,
            cls.CONV_CONTEXT_SUMMARY,
            # Mode learning
            cls.MODE_Q_VALUE,
            cls.MODE_TASK_STAT,
            cls.MODE_TRANSITION_HISTORY,
            # Profile learning
            cls.PROFILE_INTERACTION,
            cls.PROFILE_METRIC,
            # Changes
            cls.CHANGES_GROUP,
            cls.CHANGES_FILE,
        ]

    @classmethod
    def get_project_indexes(cls) -> list[str]:
        """Get index definitions for project-level tables."""
        return [
            cls.GRAPH_INDEXES,
            cls.CONV_INDEXES,
            cls.MODE_INDEXES,
            cls.PROFILE_INDEXES,
            cls.CHANGES_INDEXES,
        ]


# Schema version for migrations
CURRENT_SCHEMA_VERSION = 2


def get_migration_sql(from_version: int, to_version: int) -> list[str]:
    """Get SQL statements needed to migrate between schema versions.

    Args:
        from_version: Current schema version
        to_version: Target schema version

    Returns:
        List of SQL statements to execute
    """
    migrations: dict[int, list[str]] = {
        # Version 1 -> 2: Rename tables to new naming convention
        2: [
            # Rename system tables
            f"ALTER TABLE _db_metadata RENAME TO {Tables.SYS_METADATA}",
            # Create new RL tables
            Schema.RL_LEARNER,
            Schema.RL_Q_VALUE,
            Schema.RL_TRANSITION,
            Schema.RL_PARAM,
            Schema.RL_TASK_STAT,
            Schema.RL_METRIC,
            # Rename existing tables
            f"ALTER TABLE rl_outcomes RENAME TO {Tables.RL_OUTCOME}",
            f"ALTER TABLE cross_vertical_patterns RENAME TO {Tables.RL_PATTERN}",
            f"ALTER TABLE cross_vertical_applications RENAME TO {Tables.RL_PATTERN_USE}",
            # Rename agent tables
            f"ALTER TABLE team_composition_stats RENAME TO {Tables.AGENT_TEAM_CONFIG}",
            f"ALTER TABLE team_execution_history RENAME TO {Tables.AGENT_TEAM_RUN}",
            f"ALTER TABLE workflow_executions RENAME TO {Tables.AGENT_WORKFLOW_RUN}",
            f"ALTER TABLE prompt_template_styles RENAME TO {Tables.AGENT_PROMPT_STYLE}",
            f"ALTER TABLE prompt_template_elements RENAME TO {Tables.AGENT_PROMPT_ELEMENT}",
            f"ALTER TABLE curriculum_stages RENAME TO {Tables.AGENT_CURRICULUM_STAGE}",
            f"ALTER TABLE curriculum_metrics RENAME TO {Tables.AGENT_CURRICULUM_METRIC}",
            f"ALTER TABLE policy_checkpoints RENAME TO {Tables.AGENT_POLICY_SNAPSHOT}",
            # Rename UI tables
            f"ALTER TABLE sessions RENAME TO {Tables.UI_SESSION}",
            f"ALTER TABLE failed_signatures RENAME TO {Tables.UI_FAILED_CALL}",
        ],
    }

    sql_statements = []
    for version in range(from_version + 1, to_version + 1):
        if version in migrations:
            sql_statements.extend(migrations[version])

    return sql_statements


__all__ = [
    "Tables",
    "LearnerID",
    "Schema",
    "CURRENT_SCHEMA_VERSION",
    "get_migration_sql",
]
