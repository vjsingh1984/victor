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

"""Train ML models for formation selection.

This script trains machine learning models to predict optimal team formations
from historical execution data.

Usage:
    python scripts/ml/train_model.py \
        --training-data data/historical_executions.json \
        --algorithm random_forest \
        --output-model models/formation_selector/rf_model.pkl

Supported algorithms:
    - random_forest: Random Forest Classifier (default, robust)
    - gradient_boosting: Gradient Boosting Classifier (higher accuracy)
    - neural_network: Neural Network (best for complex patterns)

Output:
    - Trained model pickle file
    - Training metrics JSON
    - Feature importance plot (if matplotlib available)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.workflows.ml_formation_selector import ModelTrainer, ModelMetrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train ML models for formation selection")
    parser.add_argument(
        "--training-data", required=True, help="Path to training data JSON file"
    )
    parser.add_argument(
        "--algorithm",
        choices=["random_forest", "gradient_boosting", "neural_network"],
        default="random_forest",
        help="ML algorithm to use",
    )
    parser.add_argument(
        "--output-model",
        default="models/formation_selector/model.pkl",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--output-metrics",
        default="models/formation_selector/metrics.json",
        help="Path to save training metrics",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data for testing"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    training_data_path = Path(args.training_data)
    if not training_data_path.exists():
        logger.error(f"Training data file not found: {args.training_data}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Formation Selection Model Training")
    logger.info("=" * 60)
    logger.info(f"Training data: {args.training_data}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random state: {args.random_state}")
    logger.info("=" * 60)

    # Create trainer
    trainer = ModelTrainer(
        algorithm=args.algorithm, test_size=args.test_size, random_state=args.random_state
    )

    # Train model
    logger.info("Starting model training...")
    metrics = trainer.train(str(training_data_path))

    # Print metrics
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics.accuracy:.4f}")
    logger.info(f"Precision: {metrics.precision:.4f}")
    logger.info(f"Recall: {metrics.recall:.4f}")
    logger.info(f"F1 Score: {metrics.f1_score:.4f}")
    logger.info(f"Training time: {metrics.training_time_seconds:.2f}s")
    logger.info(f"Inference time: {metrics.inference_time_seconds*1000:.2f}ms")
    logger.info("Formation distribution:")
    for formation, count in metrics.formation_distribution.items():
        logger.info(f"  {formation}: {count}")
    logger.info("=" * 60)

    # Save model
    logger.info(f"Saving model to {args.output_model}")
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_model)

    # Save metrics
    logger.info(f"Saving metrics to {args.output_metrics}")
    with open(args.output_metrics, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    # Print feature importance if available
    if args.algorithm in ["random_forest", "gradient_boosting"]:
        try:
            importances = trainer.model.feature_importances_
            feature_names = [
                "complexity",
                "urgency",
                "uncertainty",
                "dependencies",
                "resource_constraints",
                "word_count",
                "node_count",
                "agent_count",
                "deadline_proximity",
                "priority_level",
                "novelty_score",
                "ambiguity_score",
            ]

            logger.info("=" * 60)
            logger.info("Feature Importance:")
            logger.info("=" * 60)
            for name, importance in sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True
            ):
                logger.info(f"  {name}: {importance:.4f}")
            logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

    logger.info("✓ Training complete!")


if __name__ == "__main__":
    main()
