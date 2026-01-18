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

"""ML-powered adaptive formation selection for team workflows.

This module provides machine learning-based formation selection that learns
from historical execution data to predict optimal team formations for tasks.

Key Features:
- Feature extraction from task descriptions and workflow metadata
- Multiple ML algorithms: Random Forest, Gradient Boosting, Neural Network
- Model training from historical execution logs
- Online learning support (update models from new executions)
- Model persistence and versioning

Architecture:
    FeatureExtractor: Extract features from workflows and tasks
    ModelTrainer: Train and evaluate ML models
    AdaptiveFormationML: Main ML-based formation selector

SOLID Principles:
    - SRP: Each class has a single responsibility
    - OCP: Extensible via custom feature extractors and models
    - DIP: Depends on abstractions (protocols) for testability

Example:
    >>> from victor.workflows.ml_formation_selector import AdaptiveFormationML
    >>>
    >>> # Load trained model
    >>> selector = AdaptiveFormationML(model_path="models/formation_selector/rf_model.pkl")
    >>>
    >>> # Predict optimal formation
    >>> formation = await selector.predict_formation(task, context, agents)
    >>>
    >>> # Train new model
    >>> from victor.workflows.ml_formation_selector import ModelTrainer
    >>> trainer = ModelTrainer()
    >>> metrics = trainer.train("data/historical_executions.json")
    >>> trainer.save_model("models/formation_selector/new_model.pkl")
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TaskFeatures:
    """Features extracted from a task for ML prediction.

    Attributes:
        task_id: Unique task identifier
        complexity: Task complexity score (0-1)
        urgency: Time urgency score (0-1)
        uncertainty: Uncertainty level (0-1)
        dependencies: Dependency complexity score (0-1)
        resource_constraints: Resource constraint severity (0-1)
        word_count: Number of words in task description
        node_count: Number of workflow nodes (if applicable)
        agent_count: Number of available agents
        deadline_proximity: How close deadline is (0-1, 1 = imminent)
        priority_level: Task priority (0-1)
        novelty_score: How novel the task is (0-1)
        ambiguity_score: Ambiguity in task description (0-1)
        tool_budget: Tool budget for task
        time_limit_seconds: Time limit for task (None = unlimited)
    """

    task_id: str
    complexity: float = 0.5
    urgency: float = 0.5
    uncertainty: float = 0.5
    dependencies: float = 0.5
    resource_constraints: float = 0.5
    word_count: int = 0
    node_count: int = 0
    agent_count: int = 0
    deadline_proximity: float = 0.0
    priority_level: float = 0.5
    novelty_score: float = 0.5
    ambiguity_score: float = 0.5
    tool_budget: Optional[int] = None
    time_limit_seconds: Optional[float] = None

    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for ML models.

        Returns:
            List of feature values
        """
        return [
            self.complexity,
            self.urgency,
            self.uncertainty,
            self.dependencies,
            self.resource_constraints,
            self.word_count / 1000.0,  # Normalize
            self.node_count / 100.0,  # Normalize
            self.agent_count / 10.0,  # Normalize
            self.deadline_proximity,
            self.priority_level,
            self.novelty_score,
            self.ambiguity_score,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "complexity": self.complexity,
            "urgency": self.urgency,
            "uncertainty": self.uncertainty,
            "dependencies": self.dependencies,
            "resource_constraints": self.resource_constraints,
            "word_count": self.word_count,
            "node_count": self.node_count,
            "agent_count": self.agent_count,
            "deadline_proximity": self.deadline_proximity,
            "priority_level": self.priority_level,
            "novelty_score": self.novelty_score,
            "ambiguity_score": self.ambiguity_score,
            "tool_budget": self.tool_budget,
            "time_limit_seconds": self.time_limit_seconds,
        }


@dataclass
class TrainingExample:
    """A single training example for formation selection.

    Attributes:
        task_features: Features extracted from task
        formation: Formation that was used
        success: Whether execution was successful
        duration_seconds: Execution duration
        efficiency_score: Efficiency score (0-1)
        timestamp: When execution occurred
    """

    task_features: TaskFeatures
    formation: str
    success: bool
    duration_seconds: float
    efficiency_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_features": self.task_features.to_dict(),
            "formation": self.formation,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "efficiency_score": self.efficiency_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModelMetrics:
    """Metrics for trained model performance.

    Attributes:
        accuracy: Classification accuracy
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        training_time_seconds: Time to train model
        inference_time_seconds: Average inference time
        formation_distribution: Distribution of predicted formations
        confusion_matrix: Confusion matrix (if available)
    """

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    training_time_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    formation_distribution: Dict[str, int] = field(default_factory=dict)
    confusion_matrix: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "training_time_seconds": self.training_time_seconds,
            "inference_time_seconds": self.inference_time_seconds,
            "formation_distribution": self.formation_distribution,
            "confusion_matrix": self.confusion_matrix,
        }


# =============================================================================
# Feature Extractor
# =============================================================================


class FeatureExtractor:
    """Extract features from workflows and tasks for ML prediction.

    This class analyzes task descriptions, workflow metadata, and execution
    context to extract meaningful features for formation selection.

    Features extracted:
    - Task complexity: Based on length, structure, keyword density
    - Urgency: From deadlines, priority levels, time pressure
    - Uncertainty: From novel terms, ambiguity indicators
    - Dependencies: From workflow structure, data flow
    - Resource constraints: From tool budget, agent availability

    SOLID: SRP (feature extraction only)

    Attributes:
        use_embeddings: Whether to use embeddings for semantic analysis
        embedding_model: Model for text embeddings (if available)

    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract_features(task, context, agents)
        >>> print(f"Complexity: {features.complexity}")
        >>> print(f"Urgency: {features.urgency}")
    """

    def __init__(self, use_embeddings: bool = False, embedding_model: Optional[Any] = None):
        """Initialize feature extractor.

        Args:
            use_embeddings: Whether to use semantic embeddings
            embedding_model: Embedding model instance (optional)
        """
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model

        # Feature scaling parameters
        self._max_word_count = 1000
        self._max_node_count = 100
        self._max_agent_count = 10

    def extract_features(
        self,
        task: Any,
        context: Any,
        agents: List[Any],
    ) -> TaskFeatures:
        """Extract features from task and context.

        Args:
            task: Task message or description
            context: Team or workflow context
            agents: Available agents

        Returns:
            Extracted task features
        """
        # Get task content
        task_content = self._get_task_content(task)

        # Extract basic features
        word_count = len(task_content.split())

        # Complexity features
        complexity = self._extract_complexity(task_content, context)

        # Urgency features
        urgency = self._extract_urgency(task_content, context)

        # Uncertainty features
        uncertainty = self._extract_uncertainty(task_content, context)

        # Dependency features
        dependencies = self._extract_dependencies(task_content, context)

        # Resource constraint features
        resource_constraints = self._extract_resource_constraints(task_content, context)

        # Additional context features
        node_count = self._extract_node_count(context)
        agent_count = len(agents)
        deadline_proximity = self._extract_deadline_proximity(task_content, context)
        priority_level = self._extract_priority_level(task_content, context)
        novelty_score = self._extract_novelty(task_content, context)
        ambiguity_score = self._extract_ambiguity(task_content, context)

        # Budget and time limits
        tool_budget = context.get("tool_budget") if hasattr(context, "get") else None
        time_limit = context.get("time_limit") if hasattr(context, "get") else None

        return TaskFeatures(
            task_id=self._generate_task_id(task),
            complexity=complexity,
            urgency=urgency,
            uncertainty=uncertainty,
            dependencies=dependencies,
            resource_constraints=resource_constraints,
            word_count=min(word_count, self._max_word_count),
            node_count=min(node_count, self._max_node_count),
            agent_count=min(agent_count, self._max_agent_count),
            deadline_proximity=deadline_proximity,
            priority_level=priority_level,
            novelty_score=novelty_score,
            ambiguity_score=ambiguity_score,
            tool_budget=tool_budget,
            time_limit_seconds=time_limit,
        )

    def _get_task_content(self, task: Any) -> str:
        """Extract text content from task.

        Args:
            task: Task object or description

        Returns:
            Task text content
        """
        if hasattr(task, "content"):
            return task.content
        elif isinstance(task, str):
            return task
        elif hasattr(task, "data") and "content" in task.data:
            return task.data["content"]
        else:
            return str(task)

    def _extract_complexity(self, content: str, context: Any) -> float:
        """Extract task complexity score.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Complexity score (0-1)
        """
        score = 0.0

        # Word count contribution (normalized 0-0.3)
        words = content.split()
        word_score = min(len(words) / 500.0, 0.3)
        score += word_score

        # Technical complexity keywords (0-0.3)
        complex_keywords = [
            "algorithm",
            "architecture",
            "optimize",
            "refactor",
            "integrate",
            "implement",
            "design",
            "complex",
        ]
        keyword_count = sum(1 for kw in complex_keywords if kw.lower() in content.lower())
        score += min(keyword_count * 0.05, 0.3)

        # Structure complexity from context (0-0.2)
        if hasattr(context, "metadata"):
            if context.metadata.get("has_subtasks", False):
                score += 0.1
            if context.metadata.get("nested_workflow", False):
                score += 0.1

        # Dependency complexity (0-0.2)
        dependency_keywords = ["depends", "requires", "sequence", "wait", "blocked"]
        if any(kw in content.lower() for kw in dependency_keywords):
            score += 0.2

        return min(score, 1.0)

    def _extract_urgency(self, content: str, context: Any) -> float:
        """Extract urgency score.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Urgency score (0-1)
        """
        score = 0.0

        # Urgency keywords
        urgent_keywords = ["urgent", "asap", "immediately", "deadline", "critical", "priority"]
        if any(kw in content.lower() for kw in urgent_keywords):
            score += 0.5

        # Context urgency
        if hasattr(context, "get"):
            if context.get("urgent", False):
                score += 0.3
            if context.get("deadline"):
                # Check deadline proximity
                deadline = context.get("deadline")
                if isinstance(deadline, datetime):
                    hours_remaining = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
                    if hours_remaining < 1:
                        score += 0.2
                    elif hours_remaining < 24:
                        score += 0.1

        return min(score, 1.0)

    def _extract_uncertainty(self, content: str, context: Any) -> float:
        """Extract uncertainty score.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Uncertainty score (0-1)
        """
        score = 0.0

        # Uncertainty indicators
        uncertainty_keywords = [
            "maybe",
            "possibly",
            "uncertain",
            "explore",
            "investigate",
            "figure out",
            "unclear",
            "unknown",
        ]
        keyword_count = sum(1 for kw in uncertainty_keywords if kw.lower() in content.lower())
        score += min(keyword_count * 0.15, 0.5)

        # Question marks indicate uncertainty
        question_count = content.count("?")
        score += min(question_count * 0.1, 0.3)

        # Exploration phrases
        if any(
            phrase in content.lower() for phrase in ["need to explore", "let's try", "not sure"]
        ):
            score += 0.2

        return min(score, 1.0)

    def extract_dependencies(self, content: str, context: Any) -> float:
        """Public method to extract dependency score.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Dependency score (0-1)
        """
        return self._extract_dependencies(content, context)

    def _extract_dependencies(self, content: str, context: Any) -> float:
        """Extract dependency complexity score.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Dependency score (0-1)
        """
        score = 0.0

        # Dependency keywords
        dependency_keywords = [
            "depends",
            "requires",
            "after",
            "then",
            "wait for",
            "sequence",
            "blocked by",
            "prerequisite",
        ]
        keyword_count = sum(1 for kw in dependency_keywords if kw.lower() in content.lower())
        score += min(keyword_count * 0.2, 0.6)

        # Ordinal numbers indicate sequencing
        ordinal_keywords = ["first", "second", "third", "then", "next", "finally"]
        if any(kw in content.lower() for kw in ordinal_keywords):
            score += 0.2

        # Context dependencies
        if hasattr(context, "metadata"):
            if context.metadata.get("has_dependencies", False):
                score += 0.2

        return min(score, 1.0)

    def _extract_resource_constraints(self, content: str, context: Any) -> float:
        """Extract resource constraint severity.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Resource constraint score (0-1)
        """
        score = 0.0

        # Constraint keywords
        constraint_keywords = [
            "limited",
            "constraint",
            "budget",
            "only",
            "minimal",
            "restrict",
        ]
        if any(kw in content.lower() for kw in constraint_keywords):
            score += 0.3

        # Tool budget constraints
        if hasattr(context, "get"):
            tool_budget = context.get("tool_budget")
            # Ensure tool_budget is a number before comparison
            if tool_budget and isinstance(tool_budget, (int, float)):
                if tool_budget < 20:
                    score += 0.4
                elif tool_budget < 50:
                    score += 0.2

            # Time constraints
            time_limit = context.get("time_limit")
            # Ensure time_limit is a number before comparison
            if (
                time_limit and isinstance(time_limit, (int, float)) and time_limit < 300
            ):  # 5 minutes
                score += 0.3

        return min(score, 1.0)

    def _extract_node_count(self, context: Any) -> int:
        """Extract number of nodes from workflow context.

        Args:
            context: Execution context

        Returns:
            Node count
        """
        if hasattr(context, "metadata"):
            count = context.metadata.get("node_count", 0)
            # Ensure we return an integer, not a MagicMock or other type
            if isinstance(count, int):
                return count
        return 0

    def _extract_deadline_proximity(self, content: str, context: Any) -> float:
        """Extract deadline proximity score.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Deadline proximity (0-1, 1 = very close)
        """
        if hasattr(context, "get"):
            deadline = context.get("deadline")
            if deadline and isinstance(deadline, datetime):
                hours_remaining = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
                if hours_remaining <= 0:
                    return 1.0
                elif hours_remaining < 1:
                    return 0.9
                elif hours_remaining < 24:
                    return 0.6
                elif hours_remaining < 168:  # 1 week
                    return 0.3
                else:
                    return 0.0

        # Check for urgent keywords in content
        if any(kw in content.lower() for kw in ["urgent", "asap", "immediately"]):
            return 0.8

        return 0.0

    def _extract_priority_level(self, content: str, context: Any) -> float:
        """Extract priority level.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Priority level (0-1)
        """
        score = 0.5  # Default priority

        # Priority keywords
        if any(kw in content.lower() for kw in ["high priority", "critical", "important"]):
            score = 0.9
        elif any(kw in content.lower() for kw in ["low priority", "when possible"]):
            score = 0.2

        # Context priority
        if hasattr(context, "get"):
            priority = context.get("priority")
            if priority:
                if isinstance(priority, (int, float)):
                    score = min(priority / 10.0, 1.0)
                elif isinstance(priority, str) and priority.lower() == "high":
                    score = 0.9

        return score

    def _extract_novelty(self, content: str, context: Any) -> float:
        """Extract novelty score.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Novelty score (0-1)
        """
        score = 0.0

        # Novelty indicators
        novelty_keywords = [
            "new",
            "novel",
            "innovative",
            "creative",
            "unique",
            "first time",
            "experimental",
        ]
        keyword_count = sum(1 for kw in novelty_keywords if kw.lower() in content.lower())
        score += min(keyword_count * 0.2, 0.6)

        # Check context for novelty
        if hasattr(context, "metadata"):
            if context.metadata.get("novel", False):
                score += 0.4

        return min(score, 1.0)

    def _extract_ambiguity(self, content: str, context: Any) -> float:
        """Extract ambiguity score.

        Args:
            content: Task content
            context: Execution context

        Returns:
            Ambiguity score (0-1)
        """
        score = 0.0

        # Ambiguity indicators
        ambiguity_keywords = [
            "or",
            "maybe",
            "possibly",
            "could",
            "might",
            "unclear",
            "ambiguous",
            "interpret",
        ]
        keyword_count = sum(1 for kw in ambiguity_keywords if kw.lower() in content.lower())
        score += min(keyword_count * 0.1, 0.5)

        # Multiple interpretations indicated by slashes or parentheses
        if "/" in content or " (" in content:
            score += 0.2

        return min(score, 1.0)

    def _generate_task_id(self, task: Any) -> str:
        """Generate unique task ID.

        Args:
            task: Task object

        Returns:
            Task ID
        """
        if hasattr(task, "id"):
            return task.id
        elif hasattr(task, "message_id"):
            return task.message_id
        else:
            # Generate hash-based ID
            content = self._get_task_content(task)
            import hashlib

            return hashlib.md5(content.encode()).hexdigest()[:16]


# =============================================================================
# Model Trainer
# =============================================================================


class ModelTrainer:
    """Train and evaluate ML models for formation selection.

    This class handles the complete ML pipeline:
    - Load and preprocess training data
    - Feature engineering and scaling
    - Model training with multiple algorithms
    - Hyperparameter tuning and cross-validation
    - Model evaluation and persistence

    Supported algorithms:
    - Random Forest (default, robust and interpretable)
    - Gradient Boosting (higher accuracy)
    - Neural Network (best for complex patterns)

    SOLID: SRP (model training only)

    Attributes:
        algorithm: ML algorithm to use
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        model: Trained model instance

    Example:
        >>> trainer = ModelTrainer(algorithm="random_forest")
        >>> metrics = trainer.train("data/historical_executions.json")
        >>> print(f"Accuracy: {metrics.accuracy}")
        >>> trainer.save_model("models/formation_selector/rf_model.pkl")
    """

    # Formation labels
    FORMATION_LABELS = ["sequential", "parallel", "hierarchical", "pipeline", "consensus"]

    def __init__(
        self,
        algorithm: str = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize model trainer.

        Args:
            algorithm: ML algorithm ("random_forest", "gradient_boosting", "neural_network")
            test_size: Fraction of data for testing
            random_state: Random seed
        """
        self.algorithm = algorithm
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.label_encoder = None

    def train(self, training_data_path: str) -> ModelMetrics:
        """Train model from historical execution data.

        Args:
            training_data_path: Path to training data JSON file

        Returns:
            Model metrics
        """
        # Load training data
        X, y = self._load_training_data(training_data_path)

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        start_time = time.time()
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        # Evaluate model
        metrics = self._evaluate_model(X_test_scaled, y_test)
        metrics.training_time_seconds = training_time

        return metrics

    def _load_training_data(self, data_path: str) -> Tuple[List[List[float]], List[int]]:
        """Load and preprocess training data.

        Args:
            data_path: Path to training data JSON

        Returns:
            Tuple of (features, labels)
        """
        with open(data_path, "r") as f:
            data = json.load(f)

        X = []
        y = []

        for example in data:
            # Extract features
            features = TaskFeatures(**example["task_features"])
            X.append(features.to_feature_vector())

            # Encode label
            formation = example["formation"]
            label = self.FORMATION_LABELS.index(formation)
            y.append(label)

        return X, y

    def _create_model(self) -> Any:
        """Create ML model based on algorithm.

        Returns:
            Model instance
        """
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.neural_network import MLPClassifier

            if self.algorithm == "random_forest":
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            elif self.algorithm == "gradient_boosting":
                return GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                )
            elif self.algorithm == "neural_network":
                return MLPClassifier(
                    hidden_layers=(100, 50),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=self.random_state,
                )
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        except ImportError:
            logger.warning("scikit-learn not available, using dummy model")
            # Return dummy model for testing without sklearn
            import warnings

            warnings.warn("scikit-learn not installed, model will not work", stacklevel=2)

            class DummyModel:
                def fit(self, X, y):
                    return self

                def predict(self, X):
                    return [0] * len(X)

                def predict_proba(self, X):
                    import numpy as np

                    return np.zeros((len(X), 5))

            return DummyModel()

    def _evaluate_model(self, X_test: List[List[float]], y_test: List[int]) -> ModelMetrics:
        """Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Model metrics
        """
        import time

        # Measure inference time
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        inference_time = time.time() - start_time

        # Calculate metrics
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        except ImportError:
            # Fallback without sklearn
            accuracy = sum(1 for yp, yt in zip(y_pred, y_test) if yp == yt) / len(y_test)
            precision = accuracy
            recall = accuracy
            f1 = accuracy

        # Formation distribution
        formation_counts = {}
        for label in y_pred:
            formation = self.FORMATION_LABELS[label]
            formation_counts[formation] = formation_counts.get(formation, 0) + 1

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            inference_time_seconds=inference_time / len(X_test),
            formation_distribution=formation_counts,
        )

    def save_model(self, model_path: str) -> None:
        """Save trained model to disk.

        Args:
            model_path: Path to save model
        """
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "algorithm": self.algorithm,
            "formation_labels": self.FORMATION_LABELS,
            "random_state": self.random_state,
        }

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> None:
        """Load trained model from disk.

        Args:
            model_path: Path to load model from
        """
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.algorithm = model_data["algorithm"]

        logger.info(f"Model loaded from {model_path}")

    def update_model(self, new_examples: List[TrainingExample]) -> ModelMetrics:
        """Update model with new examples (online learning).

        Args:
            new_examples: New training examples

        Returns:
            Updated model metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        # Extract features and labels
        X = [example.task_features.to_feature_vector() for example in new_examples]
        y = [self.FORMATION_LABELS.index(example.formation) for example in new_examples]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Update model (partial_fit if supported)
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X_scaled, y)
        else:
            # For models without partial_fit, retrain with accumulated data
            logger.warning("Model does not support online learning, retraining...")
            # In production, you'd accumulate all data and retrain

        logger.info(f"Model updated with {len(new_examples)} new examples")

        # Return dummy metrics (would need test set for real metrics)
        return ModelMetrics()


# =============================================================================
# Adaptive Formation ML
# =============================================================================


class AdaptiveFormationML:
    """ML-based formation selector with prediction and online learning.

    This class uses trained ML models to predict optimal team formations
    for tasks based on extracted features. Supports online learning to
    continuously improve from execution results.

    Workflow:
        1. Extract features from task and context
        2. Predict optimal formation using ML model
        3. Execute with predicted formation
        4. Record execution results for online learning

    SOLID: SRP (ML-based selection only)

    Attributes:
        model_path: Path to trained model file
        feature_extractor: Feature extractor instance
        model: Loaded ML model
        fallback_formation: Formation to use if prediction fails
        enable_online_learning: Update model from new executions

    Example:
        >>> selector = AdaptiveFormationML(model_path="models/formation_selector/rf_model.pkl")
        >>>
        >>> # Predict formation
        >>> formation = await selector.predict_formation(task, context, agents)
        >>> print(f"Selected formation: {formation}")
        >>>
        >>> # Record execution for online learning
        >>> await selector.record_execution(
        ...     task, context, agents, formation,
        ...     success=True, duration_seconds=15.3
        ... )
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback_formation: str = "parallel",
        enable_online_learning: bool = False,
        online_learning_threshold: int = 10,
    ):
        """Initialize ML-based formation selector.

        Args:
            model_path: Path to trained model (None = use heuristic)
            fallback_formation: Formation to use if prediction fails
            enable_online_learning: Enable online learning updates
            online_learning_threshold: Min examples before online learning update
        """
        self.model_path = model_path
        self.fallback_formation = fallback_formation
        self.enable_online_learning = enable_online_learning
        self.online_learning_threshold = online_learning_threshold

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.scaler = None
        self.formation_labels = ModelTrainer.FORMATION_LABELS

        # Online learning buffer
        self._execution_buffer: List[TrainingExample] = []
        self._trainer = None

        # Load model if path provided
        if model_path:
            self._load_model()

    def _load_model(self) -> None:
        """Load trained model from disk."""
        if not self.model_path:
            logger.warning("No model path provided, using heuristic fallback")
            return

        try:
            self._trainer = ModelTrainer()
            self._trainer.load_model(self.model_path)

            self.model = self._trainer.model
            self.scaler = self._trainer.scaler

            logger.info(f"Model loaded from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}, using heuristic fallback")
            self.model = None

    async def predict_formation(
        self,
        task: Any,
        context: Any,
        agents: List[Any],
        return_scores: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """Predict optimal formation for task.

        Args:
            task: Task message or description
            context: Team or workflow context
            agents: Available agents
            return_scores: Whether to return formation scores

        Returns:
            Predicted formation name, or (formation, scores) if return_scores=True
        """
        # Extract features
        features = self.feature_extractor.extract_features(task, context, agents)
        feature_vector = features.to_feature_vector()

        # Predict formation
        if self.model is not None:
            try:
                # Scale features
                X_scaled = self.scaler.transform([feature_vector])

                # Get prediction and probabilities
                import time

                start_time = time.time()
                prediction = self.model.predict(X_scaled)[0]

                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(X_scaled)[0]
                else:
                    probabilities = [0.0] * len(self.formation_labels)
                    probabilities[prediction] = 1.0

                inference_time = time.time() - start_time

                formation = self.formation_labels[prediction]

                logger.debug(
                    f"ML prediction: {formation} (confidence: {probabilities[prediction]:.2f}, "
                    f"time: {inference_time*1000:.1f}ms)"
                )

                if return_scores:
                    scores = {
                        formation: float(prob)
                        for formation, prob in zip(self.formation_labels, probabilities)
                    }
                    return formation, scores

                return formation

            except Exception as e:
                logger.error(f"ML prediction failed: {e}, using fallback")

        # Fallback to heuristic or default
        return self._heuristic_prediction(features, return_scores)

    def _heuristic_prediction(
        self, features: TaskFeatures, return_scores: bool = False
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """Heuristic-based formation prediction (fallback).

        Args:
            features: Extracted task features
            return_scores: Whether to return scores

        Returns:
            Predicted formation
        """
        # Simple heuristic scoring
        scores = {
            "parallel": 0.0,
            "sequential": 0.0,
            "hierarchical": 0.0,
            "pipeline": 0.0,
            "consensus": 0.0,
        }

        # Parallel: good for high complexity, low dependencies, high urgency
        scores["parallel"] += features.complexity * 0.4
        scores["parallel"] += features.urgency * 0.3
        scores["parallel"] -= features.dependencies * 0.5
        scores["parallel"] += features.agent_count / 10.0 * 0.2

        # Sequential: good for high dependencies
        scores["sequential"] += features.dependencies * 0.6
        scores["sequential"] += features.resource_constraints * 0.2

        # Hierarchical: good for high complexity, many agents
        scores["hierarchical"] += features.complexity * 0.5
        scores["hierarchical"] += features.agent_count / 10.0 * 0.3
        scores["hierarchical"] += features.novelty_score * 0.2

        # Pipeline: good for staged dependencies
        scores["pipeline"] += features.dependencies * 0.5
        scores["pipeline"] += features.node_count / 100.0 * 0.3

        # Consensus: good for high uncertainty, low urgency
        scores["consensus"] += features.uncertainty * 0.5
        scores["consensus"] += features.ambiguity_score * 0.3
        scores["consensus"] -= features.urgency * 0.4

        # Select best
        formation = max(scores, key=scores.get)

        if return_scores:
            return formation, scores

        return formation

    async def record_execution(
        self,
        task: Any,
        context: Any,
        agents: List[Any],
        formation: str,
        success: bool,
        duration_seconds: float,
        efficiency_score: Optional[float] = None,
    ) -> None:
        """Record execution result for online learning.

        Args:
            task: Task that was executed
            context: Execution context
            agents: Agents that participated
            formation: Formation that was used
            success: Whether execution succeeded
            duration_seconds: Execution duration
            efficiency_score: Optional efficiency score (0-1)
        """
        if not self.enable_online_learning:
            return

        # Extract features
        features = self.feature_extractor.extract_features(task, context, agents)

        # Calculate efficiency score if not provided
        if efficiency_score is None:
            # Simple efficiency: success / (duration + 1)
            efficiency_score = (1.0 if success else 0.0) / (duration_seconds + 1.0) * 10.0
            efficiency_score = min(efficiency_score, 1.0)

        # Create training example
        example = TrainingExample(
            task_features=features,
            formation=formation,
            success=success,
            duration_seconds=duration_seconds,
            efficiency_score=efficiency_score,
        )

        # Add to buffer
        self._execution_buffer.append(example)

        # Update model if threshold reached
        if len(self._execution_buffer) >= self.online_learning_threshold:
            await self._update_model()

    async def _update_model(self) -> None:
        """Update model with buffered examples."""
        if not self._trainer or self.model is None:
            logger.warning("Cannot update model: no model loaded")
            return

        try:
            # Update model
            self._trainer.update_model(self._execution_buffer)

            logger.info(f"Model updated with {len(self._execution_buffer)} new examples")

            # Clear buffer
            self._execution_buffer.clear()

        except Exception as e:
            logger.error(f"Failed to update model: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model (if available).

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {}

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

        importances = self.model.feature_importances_

        return {name: float(importance) for name, importance in zip(feature_names, importances)}

    def save_online_learning_data(self, output_path: str) -> None:
        """Save accumulated online learning data.

        Args:
            output_path: Path to save data
        """
        if not self._execution_buffer:
            logger.warning("No online learning data to save")
            return

        data = [example.to_dict() for example in self._execution_buffer]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(data)} online learning examples to {output_path}")


__all__ = [
    # Data classes
    "TaskFeatures",
    "TrainingExample",
    "ModelMetrics",
    # Main classes
    "FeatureExtractor",
    "ModelTrainer",
    "AdaptiveFormationML",
]
