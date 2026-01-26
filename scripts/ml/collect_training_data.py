#!/usr/bin/env python3
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

"""Collect training data from workflow execution logs.

This script extracts features and labels from historical workflow executions
to create training data for ML-based formation selection.

Usage:
    python scripts/ml/collect_training_data.py \
        --input-dir logs/workflows/ \
        --output-data data/historical_executions.json \
        --min-samples 100

Output format:
    [
        {
            "task_features": {
                "task_id": "...",
                "complexity": 0.8,
                "urgency": 0.5,
                ...
            },
            "formation": "parallel",
            "success": true,
            "duration_seconds": 15.3,
            "efficiency_score": 0.85,
            "timestamp": "2025-01-15T10:30:00Z"
        },
        ...
    ]
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.workflows.ml_formation_selector import TaskFeatures, TrainingExample

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """Collect training data from workflow execution logs.

    This class parses workflow execution logs, extracts features from tasks,
    and creates labeled training examples for ML model training.

    Attributes:
        min_samples: Minimum number of samples required
        filter_success_only: Only include successful executions
    """

    def __init__(self, min_samples: int = 50, filter_success_only: bool = False):
        """Initialize training data collector.

        Args:
            min_samples: Minimum samples to collect
            filter_success_only: Only include successful executions
        """
        self.min_samples = min_samples
        self.filter_success_only = filter_success_only
        self.examples: List[TrainingExample] = []

    def collect_from_directory(self, input_dir: str) -> List[TrainingExample]:
        """Collect training data from log files in directory.

        Args:
            input_dir: Directory containing log files

        Returns:
            List of training examples
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return []

        # Find all log files
        log_files = list(input_path.glob("**/*.json")) + list(input_path.glob("**/*.log"))

        logger.info(f"Found {len(log_files)} log files in {input_dir}")

        for log_file in log_files:
            try:
                if log_file.suffix == ".json":
                    self._parse_json_log(log_file)
                else:
                    self._parse_text_log(log_file)
            except Exception as e:
                logger.warning(f"Failed to parse {log_file}: {e}")

        logger.info(f"Collected {len(self.examples)} training examples")

        if len(self.examples) < self.min_samples:
            logger.warning(
                f"Only collected {len(self.examples)} examples, "
                f"below minimum of {self.min_samples}"
            )

        return self.examples

    def _parse_json_log(self, log_file: Path) -> None:
        """Parse JSON-formatted log file.

        Args:
            log_file: Path to JSON log file
        """
        with open(log_file, "r") as f:
            data = json.load(f)

        # Handle both single execution and list of executions
        executions = data if isinstance(data, list) else [data]

        for execution in executions:
            example = self._extract_example_from_dict(execution)
            if example:
                self.examples.append(example)

    def _parse_text_log(self, log_file: Path) -> None:
        """Parse text-formatted log file.

        Args:
            log_file: Path to text log file
        """
        # Extract structured data from text logs
        # This is a simplified implementation
        with open(log_file, "r") as f:
            content = f.read()

        # Look for execution patterns
        # In production, you'd use more sophisticated parsing
        import re

        formation_pattern = r"formation[:\s]+(\w+)"
        duration_pattern = r"duration[:\s]+([\d.]+)"

        formations = re.findall(formation_pattern, content, re.IGNORECASE)
        durations = re.findall(duration_pattern, content, re.IGNORECASE)

        # Create synthetic examples (placeholder)
        for formation, duration in zip(formations, durations):
            if formation in ["sequential", "parallel", "hierarchical", "pipeline", "consensus"]:
                example = TrainingExample(
                    task_features=TaskFeatures(
                        task_id=f"synthetic_{log_file.stem}_{len(self.examples)}",
                        complexity=0.5,
                        urgency=0.5,
                    ),
                    formation=formation,
                    success=True,
                    duration_seconds=float(duration),
                    efficiency_score=0.7,
                )
                self.examples.append(example)

    def _extract_example_from_dict(self, execution: Dict[str, Any]) -> Optional[TrainingExample]:
        """Extract training example from execution dictionary.

        Args:
            execution: Execution data dictionary

        Returns:
            Training example or None
        """
        try:
            # Extract formation
            formation = execution.get("formation") or execution.get("team_formation")
            if not formation or formation not in [
                "sequential",
                "parallel",
                "hierarchical",
                "pipeline",
                "consensus",
            ]:
                return None

            # Filter failed executions if requested
            success = execution.get("success", True)
            if self.filter_success_only and not success:
                return None

            # Extract task features
            task_data = execution.get("task", {})
            task_features = TaskFeatures(
                task_id=task_data.get("task_id", execution.get("execution_id", "unknown")),
                complexity=task_data.get("complexity", 0.5),
                urgency=task_data.get("urgency", 0.5),
                uncertainty=task_data.get("uncertainty", 0.5),
                dependencies=task_data.get("dependencies", 0.5),
                resource_constraints=task_data.get("resource_constraints", 0.5),
                word_count=task_data.get("word_count", 0),
                node_count=execution.get("node_count", 0),
                agent_count=execution.get("agent_count", 0),
                deadline_proximity=task_data.get("deadline_proximity", 0.0),
                priority_level=task_data.get("priority_level", 0.5),
                novelty_score=task_data.get("novelty_score", 0.5),
                ambiguity_score=task_data.get("ambiguity_score", 0.5),
                tool_budget=execution.get("tool_budget"),
                time_limit_seconds=execution.get("time_limit"),
            )

            # Extract metrics
            duration_seconds = execution.get("duration_seconds", 0.0)

            # Calculate efficiency score
            efficiency_score = execution.get("efficiency_score")
            if efficiency_score is None:
                # Calculate from success and duration
                efficiency_score = (1.0 if success else 0.0) / (duration_seconds + 1.0) * 10.0
                efficiency_score = min(efficiency_score, 1.0)

            # Parse timestamp
            timestamp_str = execution.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            return TrainingExample(
                task_features=task_features,
                formation=formation,
                success=success,
                duration_seconds=duration_seconds,
                efficiency_score=efficiency_score,
                timestamp=timestamp,
            )

        except Exception as e:
            logger.warning(f"Failed to extract example from execution: {e}")
            return None

    def save_examples(self, output_path: str) -> None:
        """Save training examples to JSON file.

        Args:
            output_path: Path to output file
        """
        if not self.examples:
            logger.warning("No examples to save")
            return

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = [example.to_dict() for example in self.examples]

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(data)} training examples to {output_path}")

        # Print statistics
        self._print_statistics()

    def _print_statistics(self) -> None:
        """Print collection statistics."""
        formations = {}
        success_count = 0
        total_duration = 0.0

        for example in self.examples:
            formations[example.formation] = formations.get(example.formation, 0) + 1
            if example.success:
                success_count += 1
            total_duration += example.duration_seconds

        logger.info("=" * 60)
        logger.info("Training Data Statistics:")
        logger.info(f"  Total examples: {len(self.examples)}")
        logger.info(f"  Successful: {success_count} ({success_count/len(self.examples)*100:.1f}%)")
        logger.info(f"  Average duration: {total_duration/len(self.examples):.2f}s")
        logger.info("  Formation distribution:")
        for formation, count in sorted(formations.items()):
            logger.info(f"    {formation}: {count} ({count/len(self.examples)*100:.1f}%)")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect training data from workflow execution logs"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing log files")
    parser.add_argument(
        "--output-data",
        default="data/historical_executions.json",
        help="Output path for training data JSON",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum number of samples to collect",
    )
    parser.add_argument(
        "--filter-success-only",
        action="store_true",
        help="Only include successful executions",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Collect training data
    collector = TrainingDataCollector(
        min_samples=args.min_samples, filter_success_only=args.filter_success_only
    )

    collector.collect_from_directory(args.input_dir)
    collector.save_examples(args.output_data)


if __name__ == "__main__":
    main()
