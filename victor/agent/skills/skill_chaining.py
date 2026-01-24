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

"""Skill chaining system for dynamic tool composition.

This module provides the SkillChainer which enables:
- Planning skill chains for complex goals
- Executing skill chains with dependency resolution
- Validating chain structure and dependencies
- Optimizing chains (removing redundancy, parallelizing)
- Getting detailed chain statistics and insights

Key Features:
- Chain planning based on goal and available skills
- Dependency-aware execution with StateGraph integration
- Chain validation and optimization
- Alternative chain suggestions
- Comprehensive chain statistics

Architecture:
    SkillChainer
    ├── plan_chain()              # Plan execution chain
    ├── execute_chain()           # Execute chain
    ├── validate_chain()          # Validate chain
    ├── optimize_chain()          # Optimize chain
    └── get_chain_statistics()    # Get chain stats
    └── suggest_chain_alternatives()  # Suggest alternatives

Design Principles:
    - SRP: Single responsibility for chain planning and execution
    - OCP: Open for extension (new optimization strategies)
    - DIP: Depend on Skill abstractions, not concrete tools
    - ISP: Focused interfaces for planning, execution, validation

Usage:
    from victor.agent.skills.skill_chaining import SkillChainer

    chainer = SkillChainer(event_bus=event_bus, tool_pipeline=tool_pipeline)

    # Plan a chain
    chain = await chainer.plan_chain(
        goal="Analyze and fix Python code",
        available_skills=[code_analyzer, test_runner, code_fixer]
    )

    # Validate chain
    validation = await chainer.validate_chain(chain)
    if not validation.valid:
        print(f"Chain errors: {validation.errors}")

    # Execute chain
    result = await chainer.execute_chain(chain, context={"repo": "/path/to/repo"})

    # Get chain statistics
    stats = await chainer.get_chain_statistics(chain)
    print(f"Complexity score: {stats.complexity_score}")

    # Optimize chain
    optimized = await chainer.optimize_chain(chain)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from victor.core.events import UnifiedEventType
from victor.framework.graph import StateGraph, CompiledGraph, END

from .skill_discovery import Skill

logger = logging.getLogger(__name__)


class ChainExecutionStatus(Enum):
    """Status of chain execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class RetryPolicy(Enum):
    """Retry policy for chain steps."""

    NONE = "none"  # No retries
    FIXED = "fixed"  # Fixed number of retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff


@dataclass
class DataFlow:
    """Data flow between chain steps.

    Attributes:
        source_step_id: Source step ID
        target_step_id: Target step ID
        data_mapping: Mapping of source outputs to target inputs
        required: Whether this data flow is required
    """

    source_step_id: str
    target_step_id: str
    data_mapping: Dict[str, str] = field(default_factory=dict)
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_step_id": self.source_step_id,
            "target_step_id": self.target_step_id,
            "data_mapping": self.data_mapping,
            "required": self.required,
        }


@dataclass
class ChainStep:
    """Single step in a skill chain.

    Attributes:
        id: Unique step identifier
        skill_name: Name of skill to execute
        skill_id: ID of skill to execute
        description: Step description
        dependencies: List of step IDs this step depends on
        expected_outcome: Expected outcome description
        timeout_seconds: Timeout for step execution
        retry_policy: Retry policy for failures
        retry_count: Number of retries on failure
        retry_delay_seconds: Delay between retries
        inputs: Input data for this step
        outputs: Expected output data structure
        metadata: Additional metadata
        parallelizable: Whether this step can run in parallel with others
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    skill_name: str = ""
    skill_id: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    timeout_seconds: int = 300
    retry_policy: RetryPolicy = RetryPolicy.NONE
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parallelizable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "skill_name": self.skill_name,
            "skill_id": self.skill_id,
            "description": self.description,
            "dependencies": self.dependencies,
            "expected_outcome": self.expected_outcome,
            "timeout_seconds": self.timeout_seconds,
            "retry_policy": self.retry_policy.value,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
            "parallelizable": self.parallelizable,
        }


@dataclass
class SkillChain:
    """Chain of skills to execute in sequence or parallel.

    Attributes:
        id: Unique chain identifier
        name: Chain name
        description: Chain description
        goal: High-level goal this chain achieves
        steps: List of steps in the chain
        data_flows: Data flows between steps
        status: Current execution status
        created_at: Chain creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
        max_parallel_steps: Maximum parallel steps to execute
        execution_strategy: Execution strategy (sequential, parallel, adaptive)
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    goal: str = ""
    steps: List[ChainStep] = field(default_factory=list)
    data_flows: List[DataFlow] = field(default_factory=list)
    status: ChainExecutionStatus = ChainExecutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_parallel_steps: int = 3
    execution_strategy: str = "sequential"  # sequential, parallel, adaptive

    def add_step(self, step: ChainStep) -> None:
        """Add a step to the chain.

        Args:
            step: Step to add
        """
        if step not in self.steps:
            self.steps.append(step)
            self.updated_at = datetime.utcnow()

    def remove_step(self, step_id: str) -> bool:
        """Remove a step from the chain.

        Args:
            step_id: ID of step to remove

        Returns:
            True if step was removed, False if not found
        """
        original_length = len(self.steps)
        self.steps = [s for s in self.steps if s.id != step_id]
        # Also remove associated data flows
        self.data_flows = [
            df
            for df in self.data_flows
            if df.source_step_id != step_id and df.target_step_id != step_id
        ]
        if len(self.steps) < original_length:
            self.updated_at = datetime.utcnow()
            return True
        return False

    def get_step_by_id(self, step_id: str) -> Optional[ChainStep]:
        """Get step by ID.

        Args:
            step_id: Step ID

        Returns:
            Step if found, None otherwise
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_dependent_steps(self, step_id: str) -> List[ChainStep]:
        """Get steps that depend on the given step.

        Args:
            step_id: Step ID

        Returns:
            List of dependent steps
        """
        return [s for s in self.steps if step_id in s.dependencies]

    def get_parallelizable_steps(self) -> List[List[ChainStep]]:
        """Get groups of steps that can execute in parallel.

        Returns:
            List of step groups, where steps in each group can run in parallel
        """
        if self.execution_strategy == "sequential":
            return [[step] for step in self.steps]

        # Group by dependency level
        step_levels: List[List[ChainStep]] = []
        step_map = {step.id: step for step in self.steps}
        visited = set()

        def get_level(step: ChainStep) -> int:
            """Get dependency level for a step."""
            if not step.dependencies:
                return 0
            max_dep_level = 0
            for dep_id in step.dependencies:
                dep_step = step_map.get(dep_id)
                if dep_step:
                    dep_level = get_level(dep_step)
                    max_dep_level = max(max_dep_level, dep_level + 1)
            return max_dep_level

        level_map: Dict[int, List[ChainStep]] = {}
        for step in self.steps:
            if step.id not in visited:
                level = get_level(step)
                if level not in level_map:
                    level_map[level] = []
                level_map[level].append(step)
                visited.add(step.id)

        # Sort levels and convert to list
        for level in sorted(level_map.keys()):
            step_levels.append(level_map[level])

        return step_levels

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "data_flows": [df.to_dict() for df in self.data_flows],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "max_parallel_steps": self.max_parallel_steps,
            "execution_strategy": self.execution_strategy,
        }


@dataclass
class StepResult:
    """Result of executing a single chain step.

    Attributes:
        step_id: Step ID
        step_name: Step name
        success: Whether step succeeded
        output: Step output data
        error: Error message if failed
        duration_seconds: Execution duration
        retry_count: Number of retries performed
        cached: Whether result was from cache
        metadata: Additional metadata
    """

    step_id: str
    step_name: str = ""
    success: bool = False
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    retry_count: int = 0
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainResult:
    """Result of executing a skill chain.

    Attributes:
        chain_id: Chain ID
        chain_name: Chain name
        status: Final execution status
        step_results: Map of step ID to step results
        intermediate_results: Intermediate results from each step
        final_output: Final output from last step
        outcomes: High-level outcomes achieved
        metrics: Execution metrics
        failures: List of failed step IDs
        execution_time: Total execution time
        metadata: Additional metadata
    """

    chain_id: str
    chain_name: str = ""
    status: ChainExecutionStatus = ChainExecutionStatus.PENDING
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_output: Any = None
    outcomes: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if chain execution was successful."""
        return self.status == ChainExecutionStatus.COMPLETED

    @property
    def total_duration(self) -> float:
        """Get total execution duration."""
        return self.execution_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chain_id": self.chain_id,
            "chain_name": self.chain_name,
            "status": self.status.value,
            "step_results": {
                step_id: {
                    "step_name": result.step_name,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "duration_seconds": result.duration_seconds,
                    "retry_count": result.retry_count,
                    "cached": result.cached,
                }
                for step_id, result in self.step_results.items()
            },
            "intermediate_results": self.intermediate_results,
            "final_output": self.final_output,
            "outcomes": self.outcomes,
            "metrics": self.metrics,
            "failures": self.failures,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


@dataclass
class ValidationResult:
    """Result of chain validation.

    Attributes:
        valid: Whether chain is valid
        errors: List of validation errors
        warnings: List of validation warnings
        data_flow_errors: Data flow validation errors
        metadata: Additional metadata
    """

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_flow_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainStats:
    """Statistics about a skill chain.

    Attributes:
        total_steps: Total number of steps
        sequential_steps: Number of sequential steps
        parallel_steps: Number of parallelizable steps
        complexity_score: Complexity score (0-100)
        estimated_duration_seconds: Estimated execution duration
        data_flow_complexity: Data flow complexity score
        retry_overhead: Estimated retry overhead
        parallel_potential: Potential parallelization speedup
        critical_path: List of step IDs on critical path
        bottlenecks: List of potential bottleneck step IDs
    """

    total_steps: int = 0
    sequential_steps: int = 0
    parallel_steps: int = 0
    complexity_score: float = 0.0
    estimated_duration_seconds: float = 0.0
    data_flow_complexity: float = 0.0
    retry_overhead: float = 0.0
    parallel_potential: float = 1.0
    critical_path: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_steps": self.total_steps,
            "sequential_steps": self.sequential_steps,
            "parallel_steps": self.parallel_steps,
            "complexity_score": self.complexity_score,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "data_flow_complexity": self.data_flow_complexity,
            "retry_overhead": self.retry_overhead,
            "parallel_potential": self.parallel_potential,
            "critical_path": self.critical_path,
            "bottlenecks": self.bottlenecks,
        }


class SkillChainer:
    """Engine for planning and executing skill chains.

    This class provides:
    - Chain planning based on goals
    - Dependency-aware execution
    - Chain validation and optimization
    - Alternative chain suggestions
    - Comprehensive chain statistics

    Design Principles:
        - SRP: Single responsibility for chain planning and execution
        - OCP: Open for extension (new optimization strategies)
        - DIP: Depend on Skill abstractions, not concrete tools
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        skill_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        tool_pipeline: Optional[Any] = None,
    ):
        """Initialize SkillChainer.

        Args:
            event_bus: Optional event bus for publishing events
            skill_executor: Optional skill executor function
            tool_pipeline: Optional tool pipeline for execution
        """
        self._event_bus = event_bus
        self._skill_executor = skill_executor
        self._tool_pipeline = tool_pipeline
        self._executing_chains: Set[str] = set()
        self._chain_cache: Dict[str, SkillChain] = {}

    async def plan_chain(
        self,
        goal: str,
        available_skills: List[Skill],
        context: Optional[Dict[str, Any]] = None,
        max_steps: int = 10,
    ) -> SkillChain:
        """Plan a skill chain to achieve the given goal.

        Uses semantic matching and dependency analysis to create an optimal
        chain of skills for achieving the specified goal.

        Args:
            goal: High-level goal to achieve
            available_skills: List of available skills
            context: Optional context for planning
            max_steps: Maximum number of steps in chain

        Returns:
            Planned skill chain
        """
        logger.info(f"Planning chain for goal: {goal[:100]}...")

        import time

        start_time = time.time()

        # Analyze goal to identify required skill categories
        goal_lower = goal.lower()
        goal_words = set(goal_lower.split())

        # Score and rank skills by relevance
        scored_skills = []
        for skill in available_skills:
            score = self._calculate_skill_relevance(goal, goal_words, skill)
            if score > 0.0:
                scored_skills.append((skill, score))

        # Sort by relevance score
        scored_skills.sort(key=lambda x: x[1], reverse=True)

        # Select top skills
        selected_skills = [skill for skill, score in scored_skills[:max_steps]]

        if not selected_skills:
            logger.warning(f"No skills matched goal: {goal}")
            # Create empty chain
            return SkillChain(
                name="empty_chain",
                description="No skills matched the goal",
                goal=goal,
            )

        # Analyze dependencies between skills
        skill_deps = self._analyze_skill_dependencies(selected_skills, goal_words)

        # Build chain steps with dependencies
        steps = []
        step_map: Dict[str, ChainStep] = {}

        for i, skill in enumerate(selected_skills):
            # Determine dependencies based on skill dependency analysis
            dependencies = []
            dep_skill_ids = skill_deps.get(skill.id, [])
            for dep_id in dep_skill_ids:
                if dep_id in step_map:
                    dependencies.append(step_map[dep_id].id)

            # Determine if parallelizable
            parallelizable = skill.name.lower() in ("read", "search", "analyze", "fetch")

            step = ChainStep(
                skill_name=skill.name,
                skill_id=skill.id,
                description=f"Execute {skill.name}",
                dependencies=dependencies,
                expected_outcome=f"{skill.description} completed successfully",
                parallelizable=parallelizable,
                timeout_seconds=self._estimate_step_timeout(skill),
                metadata={
                    "skill_version": skill.version,
                    "relevance_score": next(
                        (score for s, score in scored_skills if s.id == skill.id), 0.0
                    ),
                },
            )
            steps.append(step)
            step_map[skill.id] = step

        # Determine optimal execution strategy
        execution_strategy = self._determine_execution_strategy(steps)

        # Create data flows
        data_flows = self._create_data_flows(steps)

        # Create chain
        chain = SkillChain(
            name=f"chain_for_{goal[:30].replace(' ', '_')}",
            description=f"Chain to achieve: {goal}",
            goal=goal,
            steps=steps,
            data_flows=data_flows,
            execution_strategy=execution_strategy,
            metadata={
                "skill_count": len(selected_skills),
                "skill_names": [s.name for s in selected_skills],
                "context": context or {},
                "planning_time_seconds": time.time() - start_time,
            },
        )

        logger.info(f"Planned chain '{chain.name}' with {len(steps)} steps")

        # Cache the chain
        self._chain_cache[chain.id] = chain

        # Publish event
        await self._publish_event(
            UnifiedEventType.WORKFLOW_COMPLETE,
            {
                "chain_id": chain.id,
                "chain_name": chain.name,
                "step_count": len(steps),
                "goal": goal,
                "execution_strategy": execution_strategy,
                "event_type": "plan_complete",
            },
        )

        return chain

    async def execute_chain(
        self,
        chain: SkillChain,
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = False,
        use_stategraph: bool = True,
    ) -> ChainResult:
        """Execute a skill chain.

        Executes the chain using either StateGraph (recommended) or direct execution.
        Handles dependency resolution, retries, and error recovery.

        Args:
            chain: Chain to execute
            context: Optional execution context
            parallel: Whether to execute independent steps in parallel
            use_stategraph: Whether to use StateGraph for execution

        Returns:
            Chain execution result
        """
        if chain.id in self._executing_chains:
            raise ValueError(f"Chain {chain.id} is already executing")

        logger.info(f"Executing chain '{chain.name}' with {len(chain.steps)} steps")

        import time

        start_time = time.time()

        self._executing_chains.add(chain.id)
        chain.status = ChainExecutionStatus.RUNNING
        chain.updated_at = datetime.utcnow()

        try:
            # Validate chain before execution
            validation = await self.validate_chain(chain)
            if not validation.valid:
                logger.error(f"Chain validation failed: {validation.errors}")
                return ChainResult(
                    chain_id=chain.id,
                    chain_name=chain.name,
                    status=ChainExecutionStatus.FAILED,
                    failures=[],
                    execution_time=time.time() - start_time,
                )

            # Choose execution method
            if use_stategraph and len(chain.steps) > 1:
                result = await self._execute_with_stategraph(chain, context, parallel)
            else:
                result = await self._execute_direct(chain, context, parallel)

            result.execution_time = time.time() - start_time
            result.chain_name = chain.name

            # Update chain status
            chain.status = result.status
            chain.updated_at = datetime.utcnow()

            logger.info(f"Chain execution complete: {result.status.value}")

            # Publish event
            await self._publish_event(
                UnifiedEventType.WORKFLOW_COMPLETE,
                {
                    "chain_id": chain.id,
                    "chain_name": chain.name,
                    "status": result.status.value,
                    "total_steps": len(chain.steps),
                    "failed_steps": len(result.failures),
                    "execution_time": result.execution_time,
                    "event_type": "execution_complete",
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error executing chain: {e}", exc_info=True)
            chain.status = ChainExecutionStatus.FAILED
            return ChainResult(
                chain_id=chain.id,
                chain_name=chain.name,
                status=ChainExecutionStatus.FAILED,
                failures=[s.id for s in chain.steps],
                execution_time=time.time() - start_time,
            )

        finally:
            self._executing_chains.discard(chain.id)

    async def validate_chain(self, chain: SkillChain) -> ValidationResult:
        """Validate a skill chain.

        Performs comprehensive validation including:
        - Cycle detection
        - Dependency validation
        - Data flow validation
        - Orphan detection

        Args:
            chain: Chain to validate

        Returns:
            Validation result
        """
        logger.info(f"Validating chain '{chain.name}'")

        errors = []
        warnings = []
        data_flow_errors = []

        # Check for cycles
        if self._has_cycles(chain):
            errors.append("Chain contains cyclic dependencies")

        # Check for missing dependencies
        all_step_ids = {step.id for step in chain.steps}
        for step in chain.steps:
            for dep_id in step.dependencies:
                if dep_id not in all_step_ids:
                    errors.append(f"Step '{step.skill_name}' depends on missing step: {dep_id}")

        # Check for orphaned steps
        if len(chain.steps) > 1:
            has_orphans = any(
                not step.dependencies and step.id != chain.steps[0].id for step in chain.steps
            )
            if has_orphans:
                warnings.append("Chain has orphaned steps (no dependencies but not first step)")

        # Check for empty chain
        if not chain.steps:
            errors.append("Chain has no steps")

        # Validate data flows
        for data_flow in chain.data_flows:
            source_exists = data_flow.source_step_id in all_step_ids
            target_exists = data_flow.target_step_id in all_step_ids

            if not source_exists:
                data_flow_errors.append(
                    f"Data flow source step not found: {data_flow.source_step_id}"
                )
            if not target_exists:
                data_flow_errors.append(
                    f"Data flow target step not found: {data_flow.target_step_id}"
                )

            if source_exists and target_exists:
                # Check for circular data flows
                source_step = chain.get_step_by_id(data_flow.source_step_id)
                if source_step and data_flow.target_step_id in source_step.dependencies:
                    data_flow_errors.append(
                        f"Circular data flow detected: {data_flow.source_step_id} -> {data_flow.target_step_id}"
                    )

        valid = len(errors) == 0 and len(data_flow_errors) == 0

        logger.info(f"Chain validation result: {'VALID' if valid else 'INVALID'}")

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            data_flow_errors=data_flow_errors,
        )

    async def optimize_chain(
        self,
        chain: SkillChain,
        strategy: str = "both",
    ) -> SkillChain:
        """Optimize a skill chain.

        Optimizes the chain by:
        - Removing redundant steps
        - Parallelizing independent steps
        - Optimizing data flows

        Args:
            chain: Chain to optimize
            strategy: Optimization strategy ('remove_redundant', 'parallelize', 'both')

        Returns:
            Optimized chain
        """
        logger.info(f"Optimizing chain '{chain.name}' with strategy '{strategy}'")

        # Create a copy to avoid modifying original
        optimized = SkillChain(
            id=str(uuid4()),  # New ID for optimized chain
            name=chain.name + "_optimized",
            description=chain.description,
            goal=chain.goal,
            steps=[self._copy_step(step) for step in chain.steps],
            data_flows=[],  # Clear data_flows - will be rebuilt after optimization
            metadata=chain.metadata.copy(),
            max_parallel_steps=chain.max_parallel_steps,
            execution_strategy=chain.execution_strategy,
        )

        if strategy in ["remove_redundant", "both"]:
            # Remove redundant steps (same skill executed multiple times)
            seen_skills = set()
            unique_steps = []
            removed_step_ids = set()

            for step in optimized.steps:
                if step.skill_name not in seen_skills:
                    seen_skills.add(step.skill_name)
                    unique_steps.append(step)
                else:
                    logger.info(f"Removing redundant step: {step.skill_name}")
                    removed_step_ids.add(step.id)

            optimized.steps = unique_steps

            # Rebuild dependencies
            for i, step in enumerate(optimized.steps):
                step.dependencies = []
                if i > 0:
                    step.dependencies.append(optimized.steps[i - 1].id)

        if strategy in ["parallelize", "both"]:
            # Mark parallelizable steps
            for step in optimized.steps:
                if step.parallelizable:
                    # Check if dependencies allow parallelization
                    can_parallelize = True
                    for dep_id in step.dependencies:
                        dep_step = optimized.get_step_by_id(dep_id)
                        if dep_step and not dep_step.parallelizable:
                            can_parallelize = False
                            break
                    step.parallelizable = can_parallelize

            # Update execution strategy
            parallel_count = sum(1 for s in optimized.steps if s.parallelizable)
            if parallel_count >= 2:
                optimized.execution_strategy = "parallel"

        optimized.updated_at = datetime.utcnow()

        logger.info(f"Optimized chain: {len(chain.steps)} -> {len(optimized.steps)} steps")

        # Publish event
        await self._publish_event(
            UnifiedEventType.WORKFLOW_COMPLETE,
            {
                "chain_id": optimized.id,
                "original_steps": len(chain.steps),
                "optimized_steps": len(optimized.steps),
                "strategy": strategy,
                "execution_strategy": optimized.execution_strategy,
                "event_type": "optimization_complete",
            },
        )

        return optimized

    async def get_chain_statistics(
        self,
        chain: SkillChain,
    ) -> ChainStats:
        """Get comprehensive statistics about a skill chain.

        Calculates:
        - Complexity score
        - Estimated execution duration
        - Parallelization potential
        - Critical path analysis
        - Bottleneck identification

        Args:
            chain: Chain to analyze

        Returns:
            Chain statistics
        """
        logger.debug(f"Calculating statistics for chain '{chain.name}'")

        stats = ChainStats()

        # Basic counts
        stats.total_steps = len(chain.steps)
        stats.sequential_steps = sum(1 for s in chain.steps if not s.parallelizable)
        stats.parallel_steps = sum(1 for s in chain.steps if s.parallelizable)

        # Calculate complexity score
        complexity_factors = {
            "step_count": min(stats.total_steps / 20.0, 1.0) * 30,
            "dependency_depth": self._calculate_max_depth(chain) / 10.0 * 25,
            "data_flow_complexity": len(chain.data_flows) / max(stats.total_steps, 1) * 20,
            "retry_complexity": sum(s.retry_count for s in chain.steps)
            / max(stats.total_steps, 1)
            * 15,
            "timeout_variance": self._calculate_timeout_variance(chain) * 10,
        }
        stats.complexity_score = sum(complexity_factors.values())

        # Estimate duration
        base_duration = sum(s.timeout_seconds for s in chain.steps)
        parallel_savings = min(stats.parallel_steps / max(stats.total_steps, 1), 0.8)
        estimated_duration = base_duration * (1.0 - parallel_savings * 0.5)
        stats.estimated_duration_seconds = estimated_duration

        # Data flow complexity
        stats.data_flow_complexity = len(chain.data_flows) / max(stats.total_steps, 1)

        # Retry overhead
        stats.retry_overhead = sum(s.retry_delay_seconds * s.retry_count for s in chain.steps)

        # Parallel potential
        if stats.total_steps > 1:
            stats.parallel_potential = 1.0 + (stats.parallel_steps / stats.total_steps) * 0.8

        # Critical path analysis
        stats.critical_path = self._find_critical_path(chain)

        # Bottleneck identification
        stats.bottlenecks = self._identify_bottlenecks(chain)

        return stats

    async def suggest_chain_alternatives(
        self,
        chain: SkillChain,
        available_skills: List[Skill],
    ) -> List[SkillChain]:
        """Suggest alternative chains.

        Generates alternative chains by:
        - Parallelizing independent steps
        - Using different skills
        - Reordering steps

        Args:
            chain: Original chain
            available_skills: Available skills to use

        Returns:
            List of alternative chains
        """
        logger.info(f"Suggesting alternatives for chain '{chain.name}'")

        alternatives = []

        # Alternative 1: Parallelize independent steps
        parallel_chain = await self.optimize_chain(chain, strategy="parallelize")
        parallel_chain.name = f"{chain.name}_parallel"
        parallel_chain.id = str(uuid4())
        alternatives.append(parallel_chain)

        # Alternative 2: Use different skills
        skill_map = {s.name: s for s in available_skills}
        alternative_steps = []

        for step in chain.steps:
            # Find alternative skills with similar tags
            current_skill = skill_map.get(step.skill_name)
            if not current_skill:
                alternative_steps.append(self._copy_step(step))
                continue

            alternatives_for_step = [
                s
                for s in available_skills
                if s.name != step.skill_name and any(tag in s.tags for tag in current_skill.tags)
            ]

            if alternatives_for_step:
                # Use first alternative
                alt_skill = alternatives_for_step[0]
                new_step = self._copy_step(step)
                new_step.id = str(uuid4())
                new_step.skill_name = alt_skill.name
                new_step.skill_id = alt_skill.id
                new_step.description = (
                    f"Execute {alt_skill.name} (alternative to {step.skill_name})"
                )
                alternative_steps.append(new_step)
            else:
                alternative_steps.append(self._copy_step(step))

        if alternative_steps != chain.steps:
            alt_chain = SkillChain(
                id=str(uuid4()),
                name=f"{chain.name}_alternative_skills",
                description=chain.description,
                goal=chain.goal,
                steps=alternative_steps,
                data_flows=[
                    DataFlow(
                        source_step_id=df.source_step_id,
                        target_step_id=df.target_step_id,
                        data_mapping=df.data_mapping.copy(),
                        required=df.required,
                    )
                    for df in chain.data_flows
                ],
                metadata=chain.metadata.copy(),
            )
            alternatives.append(alt_chain)

        logger.info(f"Generated {len(alternatives)} alternative chains")

        return alternatives

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _calculate_skill_relevance(
        self,
        goal: str,
        goal_words: Set[str],
        skill: Skill,
    ) -> float:
        """Calculate relevance score of a skill to the goal.

        Args:
            goal: Goal description
            goal_words: Set of words in goal
            skill: Skill to score

        Returns:
            Relevance score from 0.0 to 1.0
        """
        score = 0.0

        # Convert goal to lowercase once
        goal_lower = goal.lower()

        # Keyword matching in description
        skill_desc_lower = skill.description.lower()
        desc_words = set(skill_desc_lower.split())
        keyword_overlap = len(goal_words & desc_words)
        score += (keyword_overlap / max(len(goal_words), 1)) * 0.5

        # Tag matching
        skill_tags_lower = " ".join(skill.tags).lower()
        tag_words = set(skill_tags_lower.split())
        tag_overlap = len(goal_words & tag_words)
        score += (tag_overlap / max(len(goal_words), 1)) * 0.3

        # Exact phrase matching
        if goal_lower in skill_desc_lower:
            score += 0.2

        return min(score, 1.0)

    def _analyze_skill_dependencies(
        self,
        skills: List[Skill],
        goal_words: Set[str],
    ) -> Dict[str, List[str]]:
        """Analyze dependencies between skills.

        Args:
            skills: List of skills
            goal_words: Words from the goal

        Returns:
            Dict mapping skill ID to list of dependency skill IDs
        """
        # Simplified dependency analysis
        # In production, this would use more sophisticated NLP

        # Define common dependency patterns
        read_skills = {"read", "load", "fetch", "get", "search"}
        analyze_skills = {"analyze", "check", "validate", "verify", "inspect"}
        write_skills = {"write", "save", "update", "modify", "create"}

        dependencies: Dict[str, List[str]] = {}

        for i, skill in enumerate(skills):
            skill_lower = skill.name.lower()
            skill_deps = []

            # Check if this is a write/analyze skill that depends on read skills
            if any(word in skill_lower for word in write_skills | analyze_skills):
                # Add dependencies on earlier read skills
                for j in range(i):
                    prev_skill = skills[j]
                    prev_lower = prev_skill.name.lower()
                    if any(word in prev_lower for word in read_skills):
                        skill_deps.append(prev_skill.id)

            # Check if this is a write skill that depends on analyze skills
            if any(word in skill_lower for word in write_skills):
                for j in range(i):
                    prev_skill = skills[j]
                    prev_lower = prev_skill.name.lower()
                    if any(word in prev_lower for word in analyze_skills):
                        skill_deps.append(prev_skill.id)

            dependencies[skill.id] = skill_deps

        return dependencies

    def _determine_execution_strategy(self, steps: List[ChainStep]) -> str:
        """Determine optimal execution strategy for the chain.

        Args:
            steps: List of chain steps

        Returns:
            Execution strategy: 'sequential', 'parallel', or 'adaptive'
        """
        parallel_count = sum(1 for s in steps if s.parallelizable)
        parallel_ratio = parallel_count / max(len(steps), 1)

        if parallel_ratio >= 0.6:
            return "parallel"
        elif parallel_ratio >= 0.3:
            return "adaptive"
        else:
            return "sequential"

    def _create_data_flows(self, steps: List[ChainStep]) -> List[DataFlow]:
        """Create data flows between chain steps.

        Args:
            steps: List of chain steps

        Returns:
            List of data flows
        """
        data_flows = []

        for i, step in enumerate(steps):
            if i > 0:
                # Create data flow from previous step
                data_flow = DataFlow(
                    source_step_id=steps[i - 1].id,
                    target_step_id=step.id,
                    data_mapping={
                        "output": "input",  # Simple mapping
                    },
                    required=True,
                )
                data_flows.append(data_flow)

        return data_flows

    def _estimate_step_timeout(self, skill: Skill) -> int:
        """Estimate timeout for a skill based on its characteristics.

        Args:
            skill: Skill to estimate timeout for

        Returns:
            Timeout in seconds
        """
        # Base timeout
        base_timeout = 60

        # Adjust based on skill name
        skill_lower = skill.name.lower()

        if any(word in skill_lower for word in ["read", "fetch", "get"]):
            return base_timeout
        elif any(word in skill_lower for word in ["analyze", "process", "compute"]):
            return base_timeout * 3
        elif any(word in skill_lower for word in ["search", "index", "scan"]):
            return base_timeout * 5
        else:
            return base_timeout * 2

    def _copy_step(self, step: ChainStep) -> ChainStep:
        """Create a copy of a chain step.

        Args:
            step: Step to copy

        Returns:
            New step with same attributes
        """
        return ChainStep(
            id=str(uuid4()),
            skill_name=step.skill_name,
            skill_id=step.skill_id,
            description=step.description,
            dependencies=step.dependencies.copy(),
            expected_outcome=step.expected_outcome,
            timeout_seconds=step.timeout_seconds,
            retry_policy=step.retry_policy,
            retry_count=step.retry_count,
            retry_delay_seconds=step.retry_delay_seconds,
            inputs=step.inputs.copy(),
            outputs=step.outputs.copy(),
            metadata=step.metadata.copy(),
            parallelizable=step.parallelizable,
        )

    def _resolve_execution_order(self, chain: SkillChain) -> List[List[ChainStep]]:
        """Resolve execution order based on dependencies.

        Args:
            chain: Chain to resolve

        Returns:
            List of levels, each level is a list of steps that can execute in parallel
        """
        # Build dependency graph
        step_map = {step.id: step for step in chain.steps}

        # Topological sort with level detection
        in_degree = {step.id: len(step.dependencies) for step in chain.steps}
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]

        levels = []
        visited = set()

        while queue:
            current_level = []
            next_queue = []

            for step_id in queue:
                if step_id in visited:
                    continue

                visited.add(step_id)
                current_level.append(step_map[step_id])

                # Reduce in-degree for dependent steps
                for dependent_step in chain.get_dependent_steps(step_id):
                    if dependent_step.id not in visited:
                        in_degree[dependent_step.id] -= 1
                        if in_degree[dependent_step.id] == 0:
                            next_queue.append(dependent_step.id)

            if current_level:
                levels.append(current_level)

            queue = next_queue

        return levels

    async def _execute_with_stategraph(
        self,
        chain: SkillChain,
        context: Optional[Dict[str, Any]],
        parallel: bool,
    ) -> ChainResult:
        """Execute chain using StateGraph.

        Args:
            chain: Chain to execute
            context: Execution context
            parallel: Whether to use parallel execution

        Returns:
            Chain execution result
        """
        from victor.framework.graph import GraphExecutionResult

        # Define state schema
        from typing import TypedDict

        class ChainState(TypedDict):
            step_results: Dict[str, Any]
            current_step: Optional[str]
            context: Dict[str, Any]
            errors: List[str]

        # Create graph
        graph = StateGraph(ChainState)

        # Add nodes for each step
        for step in chain.steps:

            async def execute_step_node(state: ChainState, s=step) -> ChainState:
                result = await self._execute_step(s, context or {})
                state["step_results"][s.id] = result
                if not result.success:
                    state["errors"].append(f"Step {s.skill_name} failed: {result.error}")
                return state

            graph.add_node(step.id, execute_step_node)

        # Add edges based on dependencies
        for i, step in enumerate(chain.steps):
            if i == 0:
                # First step - entry point
                if i == len(chain.steps) - 1:
                    # Only one step
                    graph.set_entry_point(step.id)
                    graph.set_finish_point(step.id)
                else:
                    graph.set_entry_point(step.id)
                    # Add edge to next step(s)
                    next_steps = [s for s in chain.steps if s.id in step.dependencies]
                    if not next_steps and i + 1 < len(chain.steps):
                        # Sequential - link to next step
                        graph.add_edge(step.id, chain.steps[i + 1].id)
                    else:
                        # Link to dependencies
                        for next_step in next_steps:
                            graph.add_edge(step.id, next_step.id)
            else:
                # Add edge from dependencies to this step
                if not step.dependencies:
                    # No dependencies - link from previous step
                    if i > 0:
                        graph.add_edge(chain.steps[i - 1].id, step.id)
                else:
                    # Add edges from all dependencies
                    for dep_id in step.dependencies:
                        graph.add_edge(dep_id, step.id)

                # Last step - add edge to END
                if i == len(chain.steps) - 1:
                    graph.add_edge(step.id, END)

        # Compile and execute
        compiled_graph = graph.compile()

        initial_state: ChainState = {
            "step_results": {},
            "current_step": None,
            "context": context or {},
            "errors": [],
        }

        graph_result: GraphExecutionResult = await compiled_graph.invoke(initial_state)

        # Convert graph result to chain result
        result = ChainResult(
            chain_id=chain.id,
            status=(
                ChainExecutionStatus.COMPLETED
                if graph_result.success
                else ChainExecutionStatus.FAILED
            ),
            step_results={},
            intermediate_results=graph_result.state.get("step_results", {}),
        )

        # Extract step results
        for step_id, step_result_data in graph_result.state.get("step_results", {}).items():
            if isinstance(step_result_data, StepResult):
                result.step_results[step_id] = step_result_data

        # Collect failures
        for step_id, step_result in result.step_results.items():
            if not step_result.success:
                result.failures.append(step_id)

        # Set final output
        if chain.steps:
            last_step_id = chain.steps[-1].id
            if last_step_id in result.step_results:
                result.final_output = result.step_results[last_step_id].output

        result.metrics = {
            "total_steps": len(chain.steps),
            "successful_steps": len(result.step_results) - len(result.failures),
            "failed_steps": len(result.failures),
            "graph_iterations": graph_result.iterations,
        }

        return result

    async def _execute_direct(
        self,
        chain: SkillChain,
        context: Optional[Dict[str, Any]],
        parallel: bool,
    ) -> ChainResult:
        """Execute chain directly (without StateGraph).

        Args:
            chain: Chain to execute
            context: Execution context
            parallel: Whether to use parallel execution

        Returns:
            Chain execution result
        """
        import time

        result = ChainResult(
            chain_id=chain.id,
            status=ChainExecutionStatus.PENDING,
        )

        # Build dependency graph
        execution_order = self._resolve_execution_order(chain)

        # Execute steps in order
        if parallel:
            # Execute independent levels in parallel
            for level in execution_order:
                level_results = await asyncio.gather(
                    *[self._execute_step(step, context or {}) for step in level],
                    return_exceptions=True,
                )

                for i, step_result in enumerate(level_results):
                    if isinstance(step_result, Exception):
                        # Handle exception
                        step_id = level[i].id
                        result.failures.append(step_id)
                        error_result = StepResult(
                            step_id=step_id,
                            step_name=level[i].skill_name,
                            success=False,
                            error=str(step_result),
                        )
                        result.step_results[step_id] = error_result
                    else:
                        result.step_results[step_result.step_id] = step_result
                        result.intermediate_results[step_result.step_id] = step_result.output
        else:
            # Execute sequentially
            for level in execution_order:
                for step in level:
                    step_result = await self._execute_step(step, context or {})
                    result.step_results[step_result.step_id] = step_result
                    result.intermediate_results[step_result.step_id] = step_result.output

                    if not step_result.success:
                        result.failures.append(step_result.step_id)

        # Determine final status
        failed_count = len(result.failures)
        total_count = len(chain.steps)

        if failed_count == 0:
            result.status = ChainExecutionStatus.COMPLETED
        elif failed_count < total_count:
            result.status = ChainExecutionStatus.PARTIAL
        else:
            result.status = ChainExecutionStatus.FAILED

        # Calculate metrics
        result.metrics = {
            "total_steps": total_count,
            "successful_steps": total_count - failed_count,
            "failed_steps": failed_count,
            "total_duration": result.total_duration,
        }

        # Extract outcomes
        result.outcomes = [
            f"Step {result.step_results[step_id].step_name}: {result.step_results[step_id].output}"
            for step_id in result.step_results
            if result.step_results[step_id].success
        ]

        # Set final output
        if chain.steps:
            last_step_id = chain.steps[-1].id
            if last_step_id in result.step_results:
                result.final_output = result.step_results[last_step_id].output

        return result

    async def _execute_step(
        self,
        step: ChainStep,
        context: Dict[str, Any],
    ) -> StepResult:
        """Execute a single chain step.

        Args:
            step: Step to execute
            context: Execution context

        Returns:
            Step result
        """
        import time

        logger.info(f"Executing step '{step.skill_name}'")

        start_time = time.time()
        retry_count = 0

        # Execute with retry logic
        while retry_count <= step.retry_count:
            try:
                # Execute skill
                if self._skill_executor:
                    output = await self._execute_with_timeout(
                        self._skill_executor,
                        step.skill_name,
                        {**context, **step.metadata},
                        step.timeout_seconds,
                    )

                    return StepResult(
                        step_id=step.id,
                        step_name=step.skill_name,
                        success=True,
                        output=output,
                        duration_seconds=time.time() - start_time,
                        retry_count=retry_count,
                    )
                else:
                    # Use tool pipeline if available
                    if self._tool_pipeline:
                        tool_result = await self._tool_pipeline.executor.execute(
                            tool_name=step.skill_name,
                            arguments={**context, **step.inputs, **step.metadata},
                            context=context,
                        )

                        return StepResult(
                            step_id=step.id,
                            step_name=step.skill_name,
                            success=tool_result.success,
                            output=tool_result.result,
                            error=tool_result.error,
                            duration_seconds=time.time() - start_time,
                            retry_count=retry_count,
                        )
                    else:
                        # Mock execution for testing
                        await asyncio.sleep(0.1)
                        return StepResult(
                            step_id=step.id,
                            step_name=step.skill_name,
                            success=True,
                            output=f"Executed {step.skill_name}",
                            duration_seconds=time.time() - start_time,
                            retry_count=retry_count,
                        )

            except asyncio.TimeoutError:
                logger.warning(f"Step '{step.skill_name}' timed out")
                retry_count += 1
                if retry_count > step.retry_count:
                    return StepResult(
                        step_id=step.id,
                        step_name=step.skill_name,
                        success=False,
                        error=f"Step timed out after {step.retry_count + 1} attempts",
                        duration_seconds=time.time() - start_time,
                        retry_count=retry_count,
                    )

                # Apply retry delay
                if step.retry_policy == RetryPolicy.EXPONENTIAL:
                    delay = step.retry_delay_seconds * (2**retry_count)
                elif step.retry_policy == RetryPolicy.LINEAR:
                    delay = step.retry_delay_seconds * retry_count
                else:
                    delay = step.retry_delay_seconds

                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Error executing step '{step.skill_name}': {e}")
                return StepResult(
                    step_id=step.id,
                    step_name=step.skill_name,
                    success=False,
                    error=str(e),
                    duration_seconds=time.time() - start_time,
                    retry_count=retry_count,
                )

        # Should not reach here
        return StepResult(
            step_id=step.id,
            step_name=step.skill_name,
            success=False,
            error="Max retries exceeded",
            duration_seconds=time.time() - start_time,
            retry_count=retry_count,
        )

    async def _execute_with_timeout(
        self,
        func: Callable,
        skill_name: str,
        context: Dict[str, Any],
        timeout: int,
    ) -> Any:
        """Execute function with timeout.

        Args:
            func: Function to execute
            skill_name: Skill name
            context: Execution context
            timeout: Timeout in seconds

        Returns:
            Function output

        Raises:
            asyncio.TimeoutError: If execution times out
        """
        return await asyncio.wait_for(func(skill_name, context), timeout=timeout)

    def _has_cycles(self, chain: SkillChain) -> bool:
        """Check if chain has cyclic dependencies.

        Args:
            chain: Chain to check

        Returns:
            True if cycles detected, False otherwise
        """
        visited = set()
        rec_stack = set()

        def dfs(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)

            step = chain.get_step_by_id(step_id)
            if step:
                for dep_id in step.dependencies:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True

            rec_stack.remove(step_id)
            return False

        for step in chain.steps:
            if step.id not in visited:
                if dfs(step.id):
                    return True

        return False

    def _calculate_max_depth(self, chain: SkillChain) -> int:
        """Calculate maximum dependency depth.

        Args:
            chain: Chain to analyze

        Returns:
            Maximum depth
        """
        step_map = {step.id: step for step in chain.steps}

        def get_depth(step_id: str, visited: Set[str]) -> int:
            if step_id in visited:
                return 0  # Cycle detected

            step = step_map.get(step_id)
            if not step or not step.dependencies:
                return 0

            visited.add(step_id)
            max_dep_depth = 0
            for dep_id in step.dependencies:
                dep_depth = get_depth(dep_id, visited.copy())
                max_dep_depth = max(max_dep_depth, dep_depth + 1)

            return max_dep_depth

        return max((get_depth(step.id, set()) for step in chain.steps), default=0)

    def _calculate_timeout_variance(self, chain: SkillChain) -> float:
        """Calculate timeout variance (normalized).

        Args:
            chain: Chain to analyze

        Returns:
            Timeout variance from 0.0 to 1.0
        """
        if not chain.steps:
            return 0.0

        timeouts = [s.timeout_seconds for s in chain.steps]
        avg_timeout = sum(timeouts) / len(timeouts)

        if avg_timeout == 0:
            return 0.0

        variance = sum((t - avg_timeout) ** 2 for t in timeouts) / len(timeouts)
        return min(variance / (avg_timeout**2), 1.0)

    def _find_critical_path(self, chain: SkillChain) -> List[str]:
        """Find critical path (longest dependency chain).

        Args:
            chain: Chain to analyze

        Returns:
            List of step IDs on critical path
        """
        step_map = {step.id: step for step in chain.steps}

        def get_path_length(step_id: str, visited: Set[str]) -> int:
            if step_id in visited:
                return 0  # Cycle detected

            step = step_map.get(step_id)
            if not step:
                return 0

            visited.add(step_id)
            max_dep_length = 0
            for dep_id in step.dependencies:
                dep_length = get_path_length(dep_id, visited.copy())
                max_dep_length = max(max_dep_length, dep_length)

            return max_dep_length + step.timeout_seconds

        # Find step with longest path
        max_length = 0
        critical_end = None

        for step in chain.steps:
            path_length = get_path_length(step.id, set())
            if path_length > max_length:
                max_length = path_length
                critical_end = step.id

        # Reconstruct path
        if not critical_end:
            return []

        path = []
        current = critical_end
        while current:
            path.append(current)
            step = step_map.get(current)
            if not step or not step.dependencies:
                break
            current = step.dependencies[0]  # Take first dependency

        return list(reversed(path))

    def _identify_bottlenecks(self, chain: SkillChain) -> List[str]:
        """Identify potential bottleneck steps.

        Args:
            chain: Chain to analyze

        Returns:
            List of bottleneck step IDs
        """
        bottlenecks = []

        avg_timeout = sum(s.timeout_seconds for s in chain.steps) / max(len(chain.steps), 1)

        for step in chain.steps:
            # Steps with high timeout are potential bottlenecks
            if step.timeout_seconds > avg_timeout * 1.5:
                bottlenecks.append(step.id)

            # Steps with many dependents are bottlenecks
            dependents = chain.get_dependent_steps(step.id)
            if len(dependents) >= 3:
                if step.id not in bottlenecks:
                    bottlenecks.append(step.id)

        return bottlenecks

    async def _publish_event(self, event_type: UnifiedEventType, data: Dict[str, Any]) -> None:
        """Publish event to event bus.

        Args:
            event_type: Event type
            data: Event data
        """
        if self._event_bus:
            try:
                if hasattr(self._event_bus, "publish"):
                    await self._event_bus.publish(event_type, data)
                elif hasattr(self._event_bus, "emit"):
                    await self._event_bus.emit(event_type.value, data)
            except Exception as e:
                logger.warning(f"Failed to publish event {event_type}: {e}")


__all__ = [
    # Enums
    "ChainExecutionStatus",
    "RetryPolicy",
    # Data classes
    "DataFlow",
    "ChainStep",
    "SkillChain",
    "StepResult",
    "ChainResult",
    "ValidationResult",
    "ChainStats",
    # Main class
    "SkillChainer",
]
