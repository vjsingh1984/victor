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

"""Utility functions for integration test optimization.

This module provides:
1. Model complexity detection from model names
2. Timing instrumentation for debugging performance issues
3. Task complexity classification
4. Performance assertions based on model size
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class TaskComplexity:
    """Task complexity levels for model selection.

    Use these to select appropriate models:
    - SIMPLE: Ultra-fast models (1B-3B) - qwen2.5:0.5b, phi3:mini
    - LOW: Fast models (7B-8B) - qwen2.5-coder:7b, llama3.1:8b
    - MEDIUM: Balanced models (14B-20B) - qwen2.5-coder:14b, deepseek-coder-v2:16b
    - HIGH: Capable models (30B+) - qwen3-coder:30b, deepseek-coder:33b
    - MAXIMUM: Large models (70B+) - deepseek-r1:70b, llama3.3:70b
    """

    SIMPLE = "simple"  # Basic queries, error detection
    LOW = "low"  # Single tool calls, simple file reads
    MEDIUM = "medium"  # Multi-step tasks, tool orchestration
    HIGH = "high"  # Complex reasoning, multi-file operations
    MAXIMUM = "maximum"  # Maximum capability required


class ModelSize:
    """Model size categories from model names."""

    ULTRA_FAST = "ultra-fast"  # 0.5B-3B: 3-10s
    FAST = "fast"  # 7B-8B: 5-20s
    BALANCED = "balanced"  # 14B-20B: 10-45s
    CAPABLE = "capable"  # 30B+: 20-90s
    LARGE = "large"  # 70B+: 40-180s


def get_model_size(model_name: str) -> str:
    """Determine model size category from model name.

    Args:
        model_name: Model name (e.g., "qwen2.5-coder:7b", "deepseek-r1:70b")

    Returns:
        Model size category (ModelSize constant)
    """
    model_lower = model_name.lower()

    # Check for size indicators in model name
    if any(x in model_lower for x in ["0.5b", "1.5b", ":2b", ":3b", "mini", "tiny"]):
        return ModelSize.ULTRA_FAST
    elif any(x in model_lower for x in [":7b", ":8b", "12b"]):
        return ModelSize.FAST
    elif any(x in model_lower for x in [":14b", ":16b", ":20b", "19b", "27b"]):
        return ModelSize.BALANCED
    elif any(x in model_lower for x in [":30b", ":32b", ":33b", ":34b", ":262k", ":128k"]):
        return ModelSize.CAPABLE
    elif any(x in model_lower for x in [":70b", ":72b"]):
        return ModelSize.LARGE
    else:
        # Default to balanced if unknown
        return ModelSize.BALANCED


def get_max_expected_time(model_size: str, task_complexity: str = TaskComplexity.MEDIUM) -> float:
    """Get maximum expected execution time for a model size and task complexity.

    Args:
        model_size: Model size category (from get_model_size())
        task_complexity: Task complexity (from TaskComplexity)

    Returns:
        Maximum expected time in seconds
    """
    # Base times for each model size (simple task)
    base_times = {
        ModelSize.ULTRA_FAST: 10,  # 10s for 0.5B-3B
        ModelSize.FAST: 20,  # 20s for 7B-8B
        ModelSize.BALANCED: 45,  # 45s for 14B-20B
        ModelSize.CAPABLE: 90,  # 90s for 30B+
        ModelSize.LARGE: 180,  # 180s for 70B+
    }

    # Complexity multipliers
    complexity_multipliers = {
        TaskComplexity.SIMPLE: 0.5,  # Simple tasks are 2x faster
        TaskComplexity.LOW: 0.75,  # Low complexity is 1.33x faster
        TaskComplexity.MEDIUM: 1.0,  # Baseline
        TaskComplexity.HIGH: 1.5,  # High complexity takes 1.5x longer
        TaskComplexity.MAXIMUM: 2.0,  # Maximum complexity takes 2x longer
    }

    base_time = base_times.get(model_size, 60)
    multiplier = complexity_multipliers.get(task_complexity, 1.0)

    return base_time * multiplier


def get_recommended_model(task_complexity: str, available_models: list) -> Optional[str]:
    """Get recommended model for a task complexity from available models.

    Args:
        task_complexity: Task complexity level
        available_models: List of available model names

    Returns:
        Recommended model name or None if no suitable model found
    """
    if not available_models:
        return None

    # Map task complexity to preferred model sizes
    preferred_sizes = {
        TaskComplexity.SIMPLE: [ModelSize.ULTRA_FAST],
        TaskComplexity.LOW: [ModelSize.ULTRA_FAST, ModelSize.FAST],
        TaskComplexity.MEDIUM: [ModelSize.FAST, ModelSize.BALANCED],
        TaskComplexity.HIGH: [ModelSize.BALANCED, ModelSize.CAPABLE],
        TaskComplexity.MAXIMUM: [ModelSize.CAPABLE, ModelSize.LARGE],
    }

    # Get preferred sizes for this task
    preferred = preferred_sizes.get(task_complexity, [ModelSize.BALANCED])

    # Find first available model in preferred size range
    for model in available_models:
        model_size = get_model_size(model)
        if model_size in preferred:
            return model

    # Fallback: return first available model
    return available_models[0]


class TimingContext:
    """Context manager for tracking detailed timing information."""

    def __init__(self, operation_name: str, log_level: int = logging.INFO):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.checkpoints: dict[str, float] = {}

    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.log(self.log_level, f"⏱️  [START] {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        logger.log(self.log_level, f"⏱️  [END] {self.operation_name} - Total: {elapsed:.2f}s")

        # Log checkpoints if any
        if self.checkpoints:
            logger.log(self.log_level, f"⏱️  [CHECKPOINTS] {self.operation_name}:")
            for name, checkpoint_time in self.checkpoints.items():
                logger.log(self.log_level, f"   - {name}: {checkpoint_time:.2f}s")

    def checkpoint(self, name: str):
        """Record a checkpoint time.

        Args:
            name: Checkpoint name
        """
        if self.start_time is None:
            return
        checkpoint_time = time.perf_counter() - self.start_time
        self.checkpoints[name] = checkpoint_time
        logger.log(self.log_level, f"⏱️  [CHECKPOINT] {name}: {checkpoint_time:.2f}s")

    def elapsed(self) -> float:
        """Get elapsed time so far.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.perf_counter()
        return end - self.start_time


def log_performance_info(
    operation_name: str,
    model_name: str,
    elapsed_time: float,
    task_complexity: str = TaskComplexity.MEDIUM,
):
    """Log performance information with model size and expected time.

    Args:
        operation_name: Name of the operation
        model_name: Model used
        elapsed_time: Actual elapsed time
        task_complexity: Task complexity level
    """
    model_size = get_model_size(model_name)
    max_expected = get_max_expected_time(model_size, task_complexity)

    logger.info(f"⏱️  [PERF] {operation_name}:")
    logger.info(f"   - Model: {model_name}")
    logger.info(f"   - Size: {model_size}")
    logger.info(f"   - Task complexity: {task_complexity}")
    logger.info(f"   - Actual time: {elapsed_time:.2f}s")
    logger.info(f"   - Expected max: {max_expected:.2f}s")
    logger.info(f"   - Status: {'✓ PASS' if elapsed_time < max_expected else '⚠ SLOW'}")


def assert_reasonable_time(
    elapsed_time: float,
    model_name: str,
    task_complexity: str = TaskComplexity.MEDIUM,
    custom_max: Optional[float] = None,
):
    """Assert that execution time is reasonable for the model and task.

    Args:
        elapsed_time: Actual elapsed time
        model_name: Model used
        task_complexity: Task complexity level
        custom_max: Optional custom maximum time

    Raises:
        AssertionError: If time exceeds expected maximum
    """
    if custom_max is not None:
        max_expected = custom_max
    else:
        model_size = get_model_size(model_name)
        max_expected = get_max_expected_time(model_size, task_complexity)

    assert elapsed_time < max_expected, (
        f"Execution time {elapsed_time:.2f}s exceeds expected maximum {max_expected:.2f}s "
        f"for model '{model_name}' ({get_model_size(model_name)}) "
        f"and task complexity '{task_complexity}'"
    )


@contextmanager
def assert_time_context(
    model_name: str,
    task_complexity: str = TaskComplexity.MEDIUM,
    custom_max: Optional[float] = None,
    operation_name: Optional[str] = None,
):
    """Context manager that asserts execution time is reasonable.

    Usage:
        with assert_time_context(model_name, TaskComplexity.MEDIUM, "test_operation"):
            # Do work here
            pass

    Args:
        model_name: Model being used
        task_complexity: Task complexity level
        custom_max: Optional custom maximum time
        operation_name: Optional operation name for logging
    """
    start = time.perf_counter()
    op_name = operation_name or f"operation_with_{model_name}"

    logger.info(f"⏱️  [START] {op_name} (model: {model_name}, complexity: {task_complexity})")

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log_performance_info(op_name, model_name, elapsed, task_complexity)

        if custom_max is not None:
            max_expected = custom_max
        else:
            model_size = get_model_size(model_name)
            max_expected = get_max_expected_time(model_size, task_complexity)

        if elapsed > max_expected:
            logger.warning(
                f"⚠️  {op_name} took {elapsed:.2f}s, expected < {max_expected:.2f}s "
                f"(model: {model_name}, size: {model_size}, complexity: {task_complexity})"
            )
        else:
            logger.info(f"✓ {op_name} completed in reasonable time: {elapsed:.2f}s")
