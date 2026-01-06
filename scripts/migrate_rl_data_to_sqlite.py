#!/usr/bin/env python3
"""Migrate legacy RL JSON data to unified SQLite database.

This script migrates data from the old JSON-based RL storage to the new
framework-level SQLite database at ~/.victor/graph/graph.db.

Legacy JSON files:
- ~/.victor/rl_data/continuation_rl.json → continuation_prompts_stats
- ~/.victor/rl_data/semantic_threshold_rl.json → semantic_threshold_stats
- ~/.victor/rl_q_tables.json → model_selector_q_values, model_selector_task_q_values

Usage:
    python scripts/migrate_rl_data_to_sqlite.py
    python scripts/migrate_rl_data_to_sqlite.py --dry-run
    python scripts/migrate_rl_data_to_sqlite.py --backup-dir /path/to/backup
"""

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RLDataMigrator:
    """Migrate legacy RL JSON data to SQLite."""

    def __init__(self, db_path: Optional[Path] = None, backup_dir: Optional[Path] = None):
        """Initialize migrator.

        Args:
            db_path: Path to SQLite database (defaults to ~/.victor/graph/graph.db)
            backup_dir: Path to backup directory (defaults to ~/.victor/rl_data/backups/)
        """
        self.db_path = db_path or Path.home() / ".victor" / "graph" / "graph.db"
        self.backup_dir = backup_dir or Path.home() / ".victor" / "rl_data" / "backups"
        self.rl_data_dir = Path.home() / ".victor" / "rl_data"

        # Legacy JSON file paths
        self.continuation_json = self.rl_data_dir / "continuation_rl.json"
        self.semantic_json = self.rl_data_dir / "semantic_threshold_rl.json"
        self.model_selector_json = Path.home() / ".victor" / "rl_q_tables.json"

        self.stats = {
            "continuation_prompts": 0,
            "semantic_threshold": 0,
            "model_selector_global": 0,
            "model_selector_task": 0,
        }

    def backup_json_files(self) -> bool:
        """Create backups of existing JSON files.

        Returns:
            True if backups created successfully
        """
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        backed_up = []
        for json_file in [self.continuation_json, self.semantic_json, self.model_selector_json]:
            if json_file.exists():
                backup_path = self.backup_dir / f"{json_file.name}.{timestamp}.bak"
                shutil.copy2(json_file, backup_path)
                backed_up.append(f"{json_file.name} → {backup_path}")

        if backed_up:
            print(f"\n✅ Backed up {len(backed_up)} JSON files:")
            for backup in backed_up:
                print(f"   {backup}")
            return True

        print("\n⚠️  No JSON files found to backup")
        return False

    def migrate_continuation_prompts(self, dry_run: bool = False) -> int:
        """Migrate continuation_rl.json to continuation_prompts_stats table.

        Args:
            dry_run: If True, only show what would be migrated

        Returns:
            Number of entries migrated
        """
        if not self.continuation_json.exists():
            print(f"\n⚠️  {self.continuation_json} not found, skipping")
            return 0

        print("\n📊 Migrating continuation prompts data...")

        try:
            with open(self.continuation_json) as f:
                data = json.load(f)

            stats_data = data.get("stats", {})
            if not stats_data:
                print("   No stats data found")
                return 0

            print(f"   Found {len(stats_data)} context entries")

            if dry_run:
                for key, stats in list(stats_data.items())[:3]:
                    print(f"   - {key}: {stats.get('total_sessions', 0)} sessions")
                if len(stats_data) > 3:
                    print(f"   - ... and {len(stats_data) - 3} more")
                return len(stats_data)

            # Connect to database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Migrate each context
            for context_key, stats in stats_data.items():
                # Parse context key: provider:model:task_type
                parts = context_key.split(":", 2)
                if len(parts) != 3:
                    print(f"   ⚠️  Skipping invalid key: {context_key}")
                    continue

                provider, model, task_type = parts

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO continuation_prompts_stats
                    (context_key, provider, model, task_type, total_sessions,
                     successful_sessions, stuck_loop_count, forced_completion_count,
                     avg_quality_score, avg_prompts_used, current_max_prompts,
                     recommended_max_prompts, quality_sum, prompts_sum, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        context_key,
                        provider,
                        model,
                        task_type,
                        stats.get("total_sessions", 0),
                        stats.get("successful_sessions", 0),
                        stats.get("stuck_loop_count", 0),
                        stats.get("forced_completion_count", 0),
                        stats.get("avg_quality_score", 0.0),
                        stats.get("avg_prompts_used", 0.0),
                        stats.get("current_max_prompts", 6),
                        stats.get("recommended_max_prompts"),
                        stats.get("quality_sum", 0.0),
                        stats.get("prompts_sum", 0.0),
                        stats.get("last_updated", datetime.now().isoformat()),
                    ),
                )

            conn.commit()
            conn.close()

            self.stats["continuation_prompts"] = len(stats_data)
            print(f"   ✅ Migrated {len(stats_data)} continuation prompts entries")
            return len(stats_data)

        except Exception as e:
            print(f"   ❌ Error migrating continuation prompts: {e}")
            return 0

    def migrate_semantic_threshold(self, dry_run: bool = False) -> int:
        """Migrate semantic_threshold_rl.json to semantic_threshold_stats table.

        Args:
            dry_run: If True, only show what would be migrated

        Returns:
            Number of entries migrated
        """
        if not self.semantic_json.exists():
            print(f"\n⚠️  {self.semantic_json} not found, skipping")
            return 0

        print("\n📊 Migrating semantic threshold data...")

        try:
            with open(self.semantic_json) as f:
                data = json.load(f)

            stats_data = data.get("stats", {})
            if not stats_data:
                print("   No stats data found")
                return 0

            print(f"   Found {len(stats_data)} context entries")

            if dry_run:
                for key, stats in list(stats_data.items())[:3]:
                    print(f"   - {key}: {stats.get('total_searches', 0)} searches")
                if len(stats_data) > 3:
                    print(f"   - ... and {len(stats_data) - 3} more")
                return len(stats_data)

            # Connect to database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Migrate each context
            for context_key, stats in stats_data.items():
                # Parse context key: embedding_model:task_type:tool_name
                parts = context_key.split(":", 2)
                if len(parts) != 3:
                    print(f"   ⚠️  Skipping invalid key: {context_key}")
                    continue

                embedding_model, task_type, tool_name = parts

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO semantic_threshold_stats
                    (context_key, embedding_model, task_type, tool_name,
                     total_searches, zero_result_count, low_quality_count,
                     avg_results_count, avg_threshold, recommended_threshold,
                     results_sum, threshold_sum, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        context_key,
                        embedding_model,
                        task_type,
                        tool_name,
                        stats.get("total_searches", 0),
                        stats.get("zero_result_count", 0),
                        stats.get("low_quality_count", 0),
                        stats.get("avg_results_count", 0.0),
                        stats.get("avg_threshold", 0.5),
                        stats.get("recommended_threshold"),
                        stats.get("_results_sum", 0.0),
                        stats.get("_threshold_sum", 0.0),
                        stats.get("last_updated", datetime.now().isoformat()),
                    ),
                )

            conn.commit()
            conn.close()

            self.stats["semantic_threshold"] = len(stats_data)
            print(f"   ✅ Migrated {len(stats_data)} semantic threshold entries")
            return len(stats_data)

        except Exception as e:
            print(f"   ❌ Error migrating semantic threshold: {e}")
            return 0

    def migrate_model_selector(self, dry_run: bool = False) -> int:
        """Migrate rl_q_tables.json to model_selector tables.

        Args:
            dry_run: If True, only show what would be migrated

        Returns:
            Number of entries migrated
        """
        if not self.model_selector_json.exists():
            print(f"\n⚠️  {self.model_selector_json} not found, skipping")
            return 0

        print("\n📊 Migrating model selector Q-tables...")

        try:
            with open(self.model_selector_json) as f:
                data = json.load(f)

            q_table = data.get("q_table", {})
            selection_counts = data.get("selection_counts", {})
            q_table_by_task = data.get("q_table_by_task", {})
            task_selection_counts = data.get("task_selection_counts", {})
            epsilon = data.get("epsilon", 0.3)
            total_selections = data.get("total_selections", 0)

            print(f"   Found {len(q_table)} global Q-values")
            print(
                f"   Found {sum(len(tasks) for tasks in q_table_by_task.values())} task-specific Q-values"
            )

            if dry_run:
                for provider, q_value in list(q_table.items())[:3]:
                    count = selection_counts.get(provider, 0)
                    print(f"   - {provider}: Q={q_value:.3f}, n={count}")
                if len(q_table) > 3:
                    print(f"   - ... and {len(q_table) - 3} more")
                return len(q_table)

            # Connect to database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Migrate global Q-values
            for provider, q_value in q_table.items():
                count = selection_counts.get(provider, 0)
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO model_selector_q_values
                    (provider, q_value, selection_count, last_updated)
                    VALUES (?, ?, ?, ?)
                    """,
                    (provider, q_value, count, datetime.now().isoformat()),
                )

            # Migrate task-specific Q-values
            task_count = 0
            for provider, tasks in q_table_by_task.items():
                for task_type, q_value in tasks.items():
                    count = 0
                    if provider in task_selection_counts:
                        count = task_selection_counts[provider].get(task_type, 0)

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO model_selector_task_q_values
                        (provider, task_type, q_value, selection_count, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (provider, task_type, q_value, count, datetime.now().isoformat()),
                    )
                    task_count += 1

            # Migrate state
            cursor.execute(
                "INSERT OR REPLACE INTO model_selector_state (key, value) VALUES ('epsilon', ?)",
                (str(epsilon),),
            )
            cursor.execute(
                "INSERT OR REPLACE INTO model_selector_state (key, value) VALUES ('total_selections', ?)",
                (str(total_selections),),
            )

            conn.commit()
            conn.close()

            self.stats["model_selector_global"] = len(q_table)
            self.stats["model_selector_task"] = task_count
            print(f"   ✅ Migrated {len(q_table)} global Q-values")
            print(f"   ✅ Migrated {task_count} task-specific Q-values")
            print(f"   ✅ Migrated epsilon={epsilon:.3f}, total_selections={total_selections}")
            return len(q_table) + task_count

        except Exception as e:
            print(f"   ❌ Error migrating model selector: {e}")
            return 0

    def verify_migration(self) -> bool:
        """Verify migrated data in SQLite database.

        Returns:
            True if verification successful
        """
        print("\n🔍 Verifying migration...")

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Check each table
            tables = [
                ("continuation_prompts_stats", self.stats["continuation_prompts"]),
                ("semantic_threshold_stats", self.stats["semantic_threshold"]),
                ("model_selector_q_values", self.stats["model_selector_global"]),
                ("model_selector_task_q_values", self.stats["model_selector_task"]),
            ]

            all_verified = True
            for table_name, expected_count in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                actual_count = cursor.fetchone()[0]

                if actual_count == expected_count:
                    print(f"   ✅ {table_name}: {actual_count} entries")
                else:
                    print(f"   ⚠️  {table_name}: expected {expected_count}, found {actual_count}")
                    all_verified = False

            conn.close()

            return all_verified

        except Exception as e:
            print(f"   ❌ Error verifying migration: {e}")
            return False

    def run(self, dry_run: bool = False) -> bool:
        """Run full migration.

        Args:
            dry_run: If True, only show what would be migrated

        Returns:
            True if migration successful
        """
        print("=" * 80)
        print("RL Data Migration: JSON → SQLite")
        print("=" * 80)
        print(f"\nTarget database: {self.db_path}")
        print(f"Backup directory: {self.backup_dir}")

        if dry_run:
            print("\n⚠️  DRY RUN MODE - No changes will be made")

        # Create backup first (only if not dry run)
        if not dry_run:
            self.backup_json_files()

        # Migrate each learner
        self.migrate_continuation_prompts(dry_run)
        self.migrate_semantic_threshold(dry_run)
        self.migrate_model_selector(dry_run)

        if dry_run:
            print("\n" + "=" * 80)
            print("DRY RUN COMPLETE - No changes made")
            print("=" * 80)
            return True

        # Verify migration
        verified = self.verify_migration()

        # Summary
        print("\n" + "=" * 80)
        print("MIGRATION SUMMARY")
        print("=" * 80)
        print(f"\n✅ Continuation Prompts: {self.stats['continuation_prompts']} entries")
        print(f"✅ Semantic Threshold: {self.stats['semantic_threshold']} entries")
        print(f"✅ Model Selector (Global): {self.stats['model_selector_global']} entries")
        print(f"✅ Model Selector (Task): {self.stats['model_selector_task']} entries")
        print(f"\nTotal: {sum(self.stats.values())} entries migrated")

        if verified:
            print("\n✅ Migration verified successfully!")
        else:
            print("\n⚠️  Migration verification had issues (see above)")

        print("\n" + "=" * 80)
        return verified


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy RL JSON data to unified SQLite database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Path to SQLite database (default: ~/.victor/graph/graph.db)",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Path to backup directory (default: ~/.victor/rl_data/backups/)",
    )
    args = parser.parse_args()

    migrator = RLDataMigrator(db_path=args.db_path, backup_dir=args.backup_dir)
    success = migrator.run(dry_run=args.dry_run)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
