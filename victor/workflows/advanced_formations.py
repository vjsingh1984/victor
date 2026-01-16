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

"""Advanced team formation strategies beyond the basic 5 patterns.

This module provides advanced formation strategies that build upon the basic
formations (sequential, parallel, hierarchical, pipeline, consensus) with
dynamic switching, AI-powered selection, and hybrid combinations.

Advanced Formations:
    - DynamicFormation: Automatically switches formation based on progress
    - AdaptiveFormation: AI-powered formation selection based on task analysis
    - HybridFormation: Combines multiple formations in sophisticated ways

These formations enable teams to:
    - Adapt their coordination strategy mid-execution
    - Select optimal formations based on task characteristics
    - Combine patterns for complex multi-phase workflows

SOLID Principles:
    - SRP: Each formation handles one specific strategy
    - OCP: Extensible via custom switching and selection logic
    - LSP: Substitutable with other BaseFormationStrategy implementations
    - DIP: Depends on TeamContext and BaseFormationStrategy abstractions

Usage:
    from victor.workflows.advanced_formations import (
        DynamicFormation,
        AdaptiveFormation,
        HybridFormation,
    )

    # Dynamic: switches based on progress
    dynamic = DynamicFormation(
        initial_formation="parallel",
        switching_rules={
            "dependencies_emerge": "sequential",
            "consensus_needed": "consensus",
        }
    )

    # Adaptive: AI-powered selection
    adaptive = AdaptiveFormation(
        criteria=["complexity", "deadline", "resource_availability"],
        default_formation="parallel",
        fallback_formation="sequential"
    )

    # Hybrid: combines patterns
    hybrid = HybridFormation(
        phases=[
            {"formation": "parallel", "goal": "explore"},
            {"formation": "sequential", "goal": "synthesize"},
        ]
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Dynamic Formation
# =============================================================================


class FormationPhase(str, Enum):
    """Phases of dynamic formation execution."""

    EXPLORATION = "exploration"
    EXECUTION = "execution"
    RESOLUTION = "resolution"


class DynamicFormation(BaseFormationStrategy):
    """Automatically switches formation based on execution progress.

    This formation monitors team execution and switches to different
    coordination patterns as the task evolves through phases.

    Typical Flow:
        1. EXPLORATION phase: Parallel formation for rapid discovery
        2. EXECUTION phase: Sequential when dependencies emerge
        3. RESOLUTION phase: Consensus for final decision making

    Switching Triggers:
        - Dependencies detected between agents → Switch to sequential
        - Conflicts or disagreements → Switch to consensus
        - Time pressure → Switch to parallel for speed
        - Quality concerns → Switch to consensus for validation

    SOLID: SRP (dynamic switching logic only), OCP (extensible triggers)

    Attributes:
        initial_formation: Starting formation (default: "parallel")
        switching_rules: Rules for when to switch formations
        max_switches: Maximum number of formation switches
        enable_auto_detection: Auto-detect dependencies and conflicts

    Example:
        >>> formation = DynamicFormation(
        ...     initial_formation="parallel",
        ...     switching_rules={
        ...         "dependencies_emerge": "sequential",
        ...         "consensus_needed": "consensus",
        ...     }
        ... )
        >>>
        >>> results = await formation.execute(agents, context, task)
        >>>
        >>> # Check formation history
        >>> print(f"Switches: {results[0].metadata['switches_made']}")
        >>> print(f"Phases: {results[0].metadata['phases']}")
    """

    def __init__(
        self,
        initial_formation: str = "parallel",
        switching_rules: Optional[Dict[str, str]] = None,
        max_switches: int = 5,
        enable_auto_detection: bool = True,
    ):
        """Initialize the dynamic formation.

        Args:
            initial_formation: Starting formation (default: "parallel")
            switching_rules: Rules mapping triggers to target formations
            max_switches: Maximum number of formation switches (default: 5)
            enable_auto_detection: Enable automatic dependency/conflict detection
        """
        self.initial_formation = initial_formation
        self.switching_rules = switching_rules or self._default_switching_rules()
        self.max_switches = max_switches
        self.enable_auto_detection = enable_auto_detection

        # State tracking
        self._current_formation: Optional[BaseFormationStrategy] = None
        self._current_phase = FormationPhase.EXPLORATION
        self._switches_made = 0
        self._formation_history: List[Dict[str, Any]] = []
        self._phase_transitions: List[Dict[str, str]] = []

    def _default_switching_rules(self) -> Dict[str, str]:
        """Get default switching rules.

        Returns:
            Default mapping of triggers to formations
        """
        return {
            # Dependency-based switching
            "dependencies_emerge": "sequential",
            "sequential_dependency": "sequential",
            # Conflict resolution
            "conflict_detected": "consensus",
            "consensus_needed": "consensus",
            "disagreement": "consensus",
            # Performance-based
            "time_pressure": "parallel",
            "slow_progress": "parallel",
            # Quality-based
            "quality_concerns": "consensus",
            "validation_needed": "consensus",
        }

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute task with dynamic formation switching.

        Args:
            agents: List of agents to execute
            context: Team context
            task: Task message to process

        Returns:
            List of MemberResult with metadata about switches and phases
        """
        # Initialize formation if needed
        if self._current_formation is None:
            self._current_formation = self._create_formation(self.initial_formation)
            self._formation_history.append({
                "timestamp": time.time(),
                "formation": self.initial_formation,
                "reason": "initial"
            })

        # Execute with current formation
        start_time = time.time()

        try:
            results = await self._current_formation.execute(agents, context, task)

            # Analyze results for switching triggers
            triggers = self._detect_triggers(results, context)

            # Check if we should switch
            if triggers and self._switches_made < self.max_switches:
                for trigger in triggers:
                    if trigger in self.switching_rules:
                        target_formation = self.switching_rules[trigger]
                        self._switch_formation(target_formation, trigger)

                        # Re-execute with new formation if needed
                        if self._should_reexecute(trigger):
                            return await self.execute(agents, context, task)

            # Update phase based on progress
            self._update_phase(results, context)

            # Enhance results with metadata
            duration = time.time() - start_time
            return self._enhance_results(results, duration)

        except Exception as e:
            logger.error(f"DynamicFormation execution failed: {e}")

            # Update phase even on error
            self._update_phase([], context)

            # Try switching on error
            if self._switches_made < self.max_switches:
                self._switch_formation("sequential", "error_recovery")
                return await self.execute(agents, context, task)

            # Return error with full metadata including history
            return [
                MemberResult(
                    member_id="dynamic_formation",
                    success=False,
                    output="",
                    error=str(e),
                    metadata={
                        "formation": "dynamic",
                        "current_formation": self._current_formation.__class__.__name__,
                        "switches_made": self._switches_made,
                        "current_phase": self._current_phase.value,
                        "formation_history": list(self._formation_history),
                        "phase_transitions": list(self._phase_transitions),
                    },
                )
            ]

    def _create_formation(self, formation_name: str) -> BaseFormationStrategy:
        """Create a formation instance by name.

        Args:
            formation_name: Name of formation to create

        Returns:
            Formation instance
        """
        # Lazy import to avoid circular dependencies
        formations = {
            "sequential": lambda: self._import_and_create("victor.coordination.formations.sequential", "SequentialFormation"),
            "parallel": lambda: self._import_and_create("victor.coordination.formations.parallel", "ParallelFormation"),
            "hierarchical": lambda: self._import_and_create("victor.coordination.formations.hierarchical", "HierarchicalFormation"),
            "pipeline": lambda: self._import_and_create("victor.coordination.formations.pipeline", "PipelineFormation"),
            "consensus": lambda: self._import_and_create("victor.coordination.formations.consensus", "ConsensusFormation"),
        }

        creator = formations.get(formation_name)
        if creator:
            return creator()

        # Fallback to sequential
        logger.warning(f"Unknown formation '{formation_name}', using sequential")
        return formations["sequential"]()

    def _import_and_create(self, module_path: str, class_name: str) -> BaseFormationStrategy:
        """Import and create a formation instance.

        Args:
            module_path: Python module path
            class_name: Class name to instantiate

        Returns:
            Formation instance
        """
        import importlib

        module = importlib.import_module(module_path)
        formation_class = getattr(module, class_name)
        return formation_class()

    def _detect_triggers(self, results: List[MemberResult], context: TeamContext) -> List[str]:
        """Detect triggers for formation switching.

        Args:
            results: Results from formation execution
            context: Team context

        Returns:
            List of trigger names detected
        """
        triggers = []

        if not self.enable_auto_detection:
            return triggers

        # Check for dependencies
        if self._detect_dependencies(results, context):
            triggers.append("dependencies_emerge")

        # Check for conflicts
        if self._detect_conflicts(results):
            triggers.append("conflict_detected")

        # Check for consensus needs
        if self._detect_consensus_need(results, context):
            triggers.append("consensus_needed")

        # Check time pressure
        if context.get("time_pressure", False):
            triggers.append("time_pressure")

        # Check quality concerns
        if context.get("quality_concerns", False):
            triggers.append("quality_concerns")

        return triggers

    def _detect_dependencies(self, results: List[MemberResult], context: TeamContext) -> bool:
        """Detect if agents have dependencies.

        Args:
            results: Member results
            context: Team context

        Returns:
            True if dependencies detected
        """
        # Check context for dependency indicators
        if "dependencies" in context.metadata:
            return True

        # Check results for dependency keywords
        for result in results:
            if result.output:
                output_lower = result.output.lower()
                if any(
                    kw in output_lower
                    for kw in ["depends on", "requires", "wait for", "blocked by", "prerequisite"]
                ):
                    return True

        return False

    def _detect_conflicts(self, results: List[MemberResult]) -> bool:
        """Detect if agents have conflicts.

        Args:
            results: Member results

        Returns:
            True if conflicts detected
        """
        # Check for disagreement keywords
        for result in results:
            if result.output:
                output_lower = result.output.lower()
                if any(
                    kw in output_lower
                    for kw in ["disagree", "conflict", "contradict", "oppose", "however"]
                ):
                    return True

        return False

    def _detect_consensus_need(self, results: List[MemberResult], context: TeamContext) -> bool:
        """Detect if consensus is needed.

        Args:
            results: Member results
            context: Team context

        Returns:
            True if consensus needed
        """
        # Check if explicitly requested
        if context.get("require_consensus", False):
            return True

        # Check for uncertainty in results
        uncertain_count = 0
        for result in results:
            if result.output:
                output_lower = result.output.lower()
                if any(
                    kw in output_lower
                    for kw in ["uncertain", "unsure", "maybe", "possibly", "need input"]
                ):
                    uncertain_count += 1

        # If multiple agents are uncertain, need consensus
        return uncertain_count >= 2

    def _should_reexecute(self, trigger: str) -> bool:
        """Determine if task should re-execute after switch.

        Args:
            trigger: Trigger that caused switch

        Returns:
            True if should re-execute
        """
        # Re-execute for error recovery and consensus
        return trigger in ["error_recovery", "consensus_needed", "conflict_detected"]

    def _switch_formation(self, target_formation: str, trigger: str) -> None:
        """Switch to a new formation.

        Args:
            target_formation: Formation to switch to
            trigger: Trigger that caused the switch
        """
        old_formation = self._current_formation.__class__.__name__
        self._current_formation = self._create_formation(target_formation)
        self._switches_made += 1

        # Record transition
        transition = {
            "timestamp": time.time(),
            "from_formation": old_formation,
            "to_formation": target_formation,
            "trigger": trigger,
            "phase": self._current_phase.value,
        }
        self._formation_history.append(transition)

        logger.info(
            f"DynamicFormation switched from {old_formation} to {target_formation} "
            f"(trigger: {trigger}, switches: {self._switches_made}/{self.max_switches})"
        )

    def _update_phase(self, results: List[MemberResult], context: TeamContext) -> None:
        """Update execution phase based on progress.

        Args:
            results: Member results
            context: Team context
        """
        # Simple heuristic: move through phases based on switches
        if self._switches_made == 0:
            self._current_phase = FormationPhase.EXPLORATION
        elif self._switches_made < self.max_switches // 2:
            self._current_phase = FormationPhase.EXECUTION
        else:
            self._current_phase = FormationPhase.RESOLUTION

        self._phase_transitions.append(
            {"timestamp": time.time(), "phase": self._current_phase.value}
        )

    def _enhance_results(self, results: List[MemberResult], duration: float) -> List[MemberResult]:
        """Enhance results with dynamic formation metadata.

        Args:
            results: Original results
            duration: Execution duration

        Returns:
            Enhanced results with metadata
        """
        enhanced = []

        for result in results:
            enhanced_metadata = {
                **result.metadata,
                "formation": "dynamic",
                "current_formation": self._current_formation.__class__.__name__,
                "current_phase": self._current_phase.value,
                "switches_made": self._switches_made,
                "formation_history": list(self._formation_history),
                "phase_transitions": list(self._phase_transitions),
                "duration_seconds": duration,
            }

            enhanced.append(
                MemberResult(
                    member_id=result.member_id,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    metadata=enhanced_metadata,
                    tool_calls_used=result.tool_calls_used,
                    duration_seconds=result.duration_seconds,
                    discoveries=result.discoveries,
                )
            )

        return enhanced

    def validate_context(self, context: TeamContext) -> bool:
        """Validate team context.

        Args:
            context: Team context to validate

        Returns:
            True if context is valid
        """
        # Dynamic formation works with any context that has agents
        return len(context.shared_state) > 0

    def supports_early_termination(self) -> bool:
        """Check if formation supports early termination.

        Returns:
            True (may terminate based on phase completion)
        """
        return True


# =============================================================================
# Adaptive Formation
# =============================================================================


@dataclass
class TaskCharacteristics:
    """Characteristics of a task for adaptive formation selection.

    Attributes:
        complexity: Task complexity (0-1)
        urgency: Time urgency (0-1)
        uncertainty: Uncertainty level (0-1)
        dependency_level: How interdependent subtasks are (0-1)
        collaboration_needed: How much collaboration is needed (0-1)
    """

    complexity: float = 0.5
    urgency: float = 0.5
    uncertainty: float = 0.5
    dependency_level: float = 0.5
    collaboration_needed: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "complexity": self.complexity,
            "urgency": self.urgency,
            "uncertainty": self.uncertainty,
            "dependency_level": self.dependency_level,
            "collaboration_needed": self.collaboration_needed,
        }


class AdaptiveFormation(BaseFormationStrategy):
    """AI-powered formation selection based on task analysis.

    Analyzes task characteristics and selects the optimal formation
    using a scoring model that considers complexity, urgency, uncertainty,
    dependencies, and collaboration needs.

    Selection Logic:
        - High complexity + low dependencies → Parallel (divide and conquer)
        - High dependencies → Sequential (coordinate dependencies)
        - High uncertainty + high stakes → Consensus (validate decisions)
        - Manager expertise available → Hierarchical (expert coordination)
        - Clear stages → Pipeline (flow through stages)

    SOLID: SRP (adaptive selection logic only), OCP (custom scoring)

    Attributes:
        criteria: List of criteria to consider for selection
        default_formation: Fallback formation if scoring fails
        fallback_formation: Alternative fallback
        scoring_weights: Custom weights for criteria (optional)
        use_ml: Use ML model if available (default: heuristic)

    Example:
        >>> formation = AdaptiveFormation(
        ...     criteria=["complexity", "deadline", "resource_availability"],
        ...     default_formation="parallel",
        ...     fallback_formation="sequential"
        ... )
        >>>
        >>> results = await formation.execute(agents, context, task)
        >>>
        >>> # Check what was selected
        >>> print(f"Selected: {results[0].metadata['selected_formation']}")
        >>> print(f"Characteristics: {results[0].metadata['task_characteristics']}")
        >>> print(f"Scores: {results[0].metadata['formation_scores']}")
    """

    def __init__(
        self,
        criteria: List[str] = None,
        default_formation: str = "parallel",
        fallback_formation: str = "sequential",
        scoring_weights: Optional[Dict[str, Dict[str, float]]] = None,
        use_ml: bool = False,
        model_path: Optional[str] = None,
        enable_online_learning: bool = False,
    ):
        """Initialize the adaptive formation.

        Args:
            criteria: List of criteria to consider
            default_formation: Default formation if scoring is neutral
            fallback_formation: Fallback if scoring fails
            scoring_weights: Custom weights: {formation: {criterion: weight}}
            use_ml: Use ML model if available (default: heuristic)
            model_path: Path to trained ML model (for ML-based selection)
            enable_online_learning: Enable online learning from executions
        """
        self.criteria = criteria or ["complexity", "deadline", "resource_availability"]
        self.default_formation = default_formation
        self.fallback_formation = fallback_formation
        self.scoring_weights = scoring_weights or self._default_scoring_weights()
        self.use_ml = use_ml
        self.model_path = model_path
        self.enable_online_learning = enable_online_learning

        # Initialize ML selector if enabled
        self._ml_selector = None
        if use_ml and model_path:
            try:
                from victor.workflows.ml_formation_selector import AdaptiveFormationML

                self._ml_selector = AdaptiveFormationML(
                    model_path=model_path,
                    fallback_formation=fallback_formation,
                    enable_online_learning=enable_online_learning,
                )
                logger.info(f"ML formation selector loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}, using heuristic scoring")
                self.use_ml = False

    def _default_scoring_weights(self) -> Dict[str, Dict[str, float]]:
        """Get default scoring weights for formations.

        Returns:
            Dictionary mapping formations to criterion weights
        """
        return {
            "parallel": {
                "complexity": 0.8,  # Good for complex tasks
                "deadline": 0.7,  # Fast execution
                "resource_availability": 0.9,  # Needs multiple agents
                "dependency_level": -0.5,  # Bad for dependencies
                "uncertainty": 0.3,  # Not great for uncertainty
            },
            "sequential": {
                "complexity": 0.5,
                "deadline": 0.4,  # Slower
                "resource_availability": 0.6,
                "dependency_level": 0.9,  # Good for dependencies
                "uncertainty": 0.5,
            },
            "hierarchical": {
                "complexity": 0.9,  # Great for complex tasks
                "deadline": 0.6,
                "resource_availability": 0.8,
                "dependency_level": 0.8,
                "uncertainty": 0.7,
            },
            "pipeline": {
                "complexity": 0.6,
                "deadline": 0.5,
                "resource_availability": 0.7,
                "dependency_level": 0.9,  # Excellent for staged dependencies
                "uncertainty": 0.4,
            },
            "consensus": {
                "complexity": 0.4,
                "deadline": 0.2,  # Slowest
                "resource_availability": 0.7,
                "dependency_level": 0.3,
                "uncertainty": 0.9,  # Best for uncertainty
            },
        }

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute task with adaptively selected formation.

        Args:
            agents: List of agents to execute
            context: Team context
            task: Task message to process

        Returns:
            List of MemberResult with selection metadata
        """
        start_time = time.time()

        try:
            # Use ML selector if enabled and available
            fallback_used = False
            selection_error = None

            if self.use_ml and self._ml_selector:
                selected_formation, scores = await self._ml_selector.predict_formation(
                    task, context, agents, return_scores=True
                )
                # Extract characteristics for metadata
                characteristics = await self._analyze_task(task, context, agents)
                selection_method = "ml"
            else:
                # Use heuristic scoring
                # Analyze task characteristics
                characteristics = await self._analyze_task(task, context, agents)

                # Score each formation
                scores = self._score_formations(characteristics)

                # Select best formation
                selected_formation = max(scores, key=scores.get)
                selection_method = "heuristic"

            # Check if we're using fallback (selection failed to find good match)
            if scores.get(selected_formation, 0) < 0.3:
                selected_formation = self.default_formation
                fallback_used = True

            # Create and execute formation
            formation = self._create_formation(selected_formation)

            # Try to execute, fall back to fallback_formation on error
            try:
                results = await formation.execute(agents, context, task)
            except Exception as exec_error:
                logger.warning(f"Formation {selected_formation} failed, using fallback: {exec_error}")
                formation = self._create_formation(self.fallback_formation)
                selected_formation = self.fallback_formation
                fallback_used = True
                selection_error = str(exec_error)
                results = await formation.execute(agents, context, task)

            duration = time.time() - start_time

            # Record execution for online learning if enabled
            if self.enable_online_learning and self._ml_selector:
                success = all(r.success for r in results)
                await self._ml_selector.record_execution(
                    task, context, agents, selected_formation, success, duration
                )

            # Enhance results with selection metadata
            return self._enhance_results(
                results,
                selected_formation,
                characteristics,
                scores,
                duration,
                error=selection_error,
                fallback_used=fallback_used,
                selection_method=selection_method,
            )

        except Exception as e:
            logger.error(f"AdaptiveFormation failed: {e}")

            # Fallback to default formation
            try:
                formation = self._create_formation(self.fallback_formation)
                results = await formation.execute(agents, context, task)

                duration = time.time() - start_time
                return self._enhance_results(
                    results,
                    self.fallback_formation,
                    TaskCharacteristics(),
                    {},
                    duration,
                    error=str(e),
                    fallback_used=True,
                    selection_method="fallback",
                )
            except Exception as fallback_error:
                # Last resort: return minimal metadata
                return [
                    MemberResult(
                        member_id="adaptive_formation",
                        success=False,
                        output="",
                        error=f"Both adaptive and fallback failed: {str(e)}, {str(fallback_error)}",
                        metadata={
                            "formation": "adaptive",
                            "selected_formation": self.fallback_formation,
                            "fallback_used": True,
                            "selection_error": str(e),
                            "task_characteristics": TaskCharacteristics().to_dict(),
                            "formation_scores": {},
                            "selection_criteria": self.criteria,
                        },
                    )
                ]

    async def _analyze_task(
        self, task: AgentMessage, context: TeamContext, agents: List[Any]
    ) -> TaskCharacteristics:
        """Analyze task to extract characteristics.

        Args:
            task: Task message
            context: Team context
            agents: Available agents

        Returns:
            Task characteristics
        """
        # Start with heuristics
        characteristics = TaskCharacteristics()

        # Analyze task content
        content = task.content.lower()
        word_count = len(task.content.split())

        # Complexity based on length and keyword density
        # Check for complexity keywords first
        complexity_keywords = [
            "complex", "complicated", "difficult", "intricate", "sophisticated",
            "multi-faceted", "multi-facets", "multiple", "various", "several",
            "investigation", "analysis", "research", "comprehensive", "detailed"
        ]
        complexity_keyword_count = sum(1 for kw in complexity_keywords if kw in content)

        if word_count > 500 or complexity_keyword_count >= 3:
            characteristics.complexity = 0.9
        elif word_count > 200 or complexity_keyword_count >= 2:
            characteristics.complexity = 0.7
        elif word_count > 100 or complexity_keyword_count >= 1:
            characteristics.complexity = 0.5
        else:
            characteristics.complexity = 0.2

        # Urgency from context and keywords
        if context.get("urgent", False) or context.get("deadline"):
            characteristics.urgency = 0.9
        elif any(kw in content for kw in ["asap", "urgent", "immediately", "deadline"]):
            characteristics.urgency = 0.8
        else:
            characteristics.urgency = 0.3

        # Uncertainty from keywords
        uncertainty_keywords = ["maybe", "possibly", "uncertain", "explore", "investigate", "figure out"]
        if any(kw in content for kw in uncertainty_keywords):
            characteristics.uncertainty = 0.8
        else:
            characteristics.uncertainty = 0.3

        # Dependency level from task structure
        dependency_keywords = ["depends", "requires", "then", "after", "wait for", "sequence"]
        dependency_count = sum(1 for kw in dependency_keywords if kw in content)
        characteristics.dependency_level = min(0.9, dependency_count * 0.2)

        # Collaboration needed from agent count and keywords
        collaboration_keywords = ["together", "collaborate", "team", "consensus", "agree", "vote"]
        if len(agents) > 3 or any(kw in content for kw in collaboration_keywords):
            characteristics.collaboration_needed = 0.8
        else:
            characteristics.collaboration_needed = 0.4

        # Try to use ML model if enabled and available
        if self.use_ml:
            try:
                characteristics = await self._ml_analysis(task, context, agents, characteristics)
            except Exception as e:
                logger.debug(f"ML analysis failed, using heuristics: {e}")

        return characteristics

    async def _ml_analysis(
        self,
        task: AgentMessage,
        context: TeamContext,
        agents: List[Any],
        heuristic_char: TaskCharacteristics,
    ) -> TaskCharacteristics:
        """Use ML model for task analysis if available.

        Args:
            task: Task message
            context: Team context
            agents: Available agents
            heuristic_char: Heuristic-based characteristics (fallback)

        Returns:
            ML-enhanced characteristics
        """
        # Check if ML model is available
        # This is a placeholder for actual ML integration
        # For now, return heuristic characteristics

        # Future: Integrate with task classifier or external ML service
        # Example:
        #   ml_service = context.get("ml_service")
        #   if ml_service:
        #       return await ml_service.analyze_task(task, agents)

        return heuristic_char

    def _score_formations(self, characteristics: TaskCharacteristics) -> Dict[str, float]:
        """Score each formation based on task characteristics.

        Args:
            characteristics: Task characteristics

        Returns:
            Dictionary mapping formation names to scores
        """
        scores = {}

        char_dict = characteristics.to_dict()

        for formation, weights in self.scoring_weights.items():
            score = 0.0
            for criterion, weight in weights.items():
                if criterion in char_dict:
                    score += weight * char_dict[criterion]

            # Normalize score to 0-1 range
            scores[formation] = max(0, min(1, (score + 2) / 4))

        return scores

    def _create_formation(self, formation_name: str) -> BaseFormationStrategy:
        """Create a formation instance by name.

        Args:
            formation_name: Name of formation to create

        Returns:
            Formation instance
        """
        import importlib

        formations = {
            "sequential": ("victor.coordination.formations.sequential", "SequentialFormation"),
            "parallel": ("victor.coordination.formations.parallel", "ParallelFormation"),
            "hierarchical": ("victor.coordination.formations.hierarchical", "HierarchicalFormation"),
            "pipeline": ("victor.coordination.formations.pipeline", "PipelineFormation"),
            "consensus": ("victor.coordination.formations.consensus", "ConsensusFormation"),
        }

        if formation_name in formations:
            module_path, class_name = formations[formation_name]
            module = importlib.import_module(module_path)
            formation_class = getattr(module, class_name)
            return formation_class()

        # Fallback
        logger.warning(f"Unknown formation '{formation_name}', using sequential")
        module = importlib.import_module("victor.coordination.formations.sequential")
        return module.SequentialFormation()

    def _enhance_results(
        self,
        results: List[MemberResult],
        selected_formation: str,
        characteristics: TaskCharacteristics,
        scores: Dict[str, float],
        duration: float,
        error: Optional[str] = None,
        fallback_used: bool = False,
        selection_method: str = "heuristic",
    ) -> List[MemberResult]:
        """Enhance results with adaptive selection metadata.

        Args:
            results: Original results
            selected_formation: Formation that was selected
            characteristics: Task characteristics
            scores: Formation scores
            duration: Execution duration
            error: Error if selection failed
            fallback_used: Whether fallback formation was used
            selection_method: Method used for selection

        Returns:
            Enhanced results with metadata
        """
        enhanced = []

        for result in results:
            enhanced_metadata = {
                **result.metadata,
                "formation": "adaptive",
                "selected_formation": selected_formation,
                "task_characteristics": characteristics.to_dict(),
                "formation_scores": scores,
                "selection_criteria": self.criteria,
                "selection_method": selection_method,
                "fallback_used": fallback_used,
                "duration_seconds": duration,
            }

            if error:
                enhanced_metadata["selection_error"] = error

            enhanced.append(
                MemberResult(
                    member_id=result.member_id,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    metadata=enhanced_metadata,
                    tool_calls_used=result.tool_calls_used,
                    duration_seconds=result.duration_seconds,
                    discoveries=result.discoveries,
                )
            )

        return enhanced

    def validate_context(self, context: TeamContext) -> bool:
        """Validate team context.

        Args:
            context: Team context to validate

        Returns:
            True if context is valid
        """
        return len(context.shared_state) > 0


# =============================================================================
# Hybrid Formation
# =============================================================================


@dataclass
class HybridPhase:
    """A single phase in a hybrid formation.

    Attributes:
        formation: Formation to use in this phase
        goal: Goal of this phase
        duration_budget: Time budget for this phase (None = no limit)
        iteration_limit: Max iterations for this phase (None = no limit)
        completion_criteria: When to move to next phase
    """

    formation: str
    goal: str
    duration_budget: Optional[float] = None
    iteration_limit: Optional[int] = None
    completion_criteria: Optional[Callable[[List[MemberResult]], bool]] = None


class HybridFormation(BaseFormationStrategy):
    """Combines multiple formations in sophisticated multi-phase workflows.

    Executes tasks through multiple phases, each using a different formation.
    Enables complex workflows like:
        - Parallel research → Sequential synthesis → Consensus validation
        - Hierarchical planning → Parallel execution → Pipeline review
        - Any custom combination of formations

    Phase Transitions:
        - Automatic: Based on duration/iteration limits
        - Manual: Based on completion criteria callback
        - Conditional: Based on result analysis

    SOLID: SRP (hybrid coordination logic only), OCP (custom phases)

    Attributes:
        phases: List of phases to execute
        enable_phase_logging: Log phase transitions
        stop_on_first_failure: Stop execution if any phase fails

    Example:
        >>> formation = HybridFormation(
        ...     phases=[
        ...         HybridPhase(
        ...             formation="parallel",
        ...             goal="Explore the problem space",
        ...             duration_budget=30.0
        ...         ),
        ...         HybridPhase(
        ...             formation="sequential",
        ...             goal="Synthesize findings",
        ...             iteration_limit=3
        ...         ),
        ...         HybridPhase(
        ...             formation="consensus",
        ...             goal="Validate final solution",
        ...             completion_criteria=lambda results: all(r.success for r in results)
        ...         ),
        ...     ]
        ... )
        >>>
        >>> results = await formation.execute(agents, context, task)
        >>>
        >>> # Check phase progression
        >>> print(f"Phases completed: {results[0].metadata['phases_completed']}")
        >>> print(f"Phase results: {results[0].metadata['phase_results']}")
    """

    def __init__(
        self,
        phases: List[HybridPhase],
        enable_phase_logging: bool = True,
        stop_on_first_failure: bool = False,
    ):
        """Initialize the hybrid formation.

        Args:
            phases: List of phases to execute in order
            enable_phase_logging: Log phase transitions
            stop_on_first_failure: Stop if any phase fails
        """
        self.phases = phases
        self.enable_phase_logging = enable_phase_logging
        self.stop_on_first_failure = stop_on_first_failure

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute task through multiple phases.

        Args:
            agents: List of agents to execute
            context: Team context
            task: Task message to process

        Returns:
            List of MemberResult with phase progression metadata
        """
        start_time = time.time()
        phase_results = []
        phases_completed = 0

        try:
            for phase_index, phase in enumerate(self.phases):
                if self.enable_phase_logging:
                    logger.info(
                        f"HybridFormation starting phase {phase_index + 1}/{len(self.phases)}: "
                        f"{phase.formation} - {phase.goal}"
                    )

                # Create formation for this phase
                formation = self._create_formation(phase.formation)

                # Create phase-specific task
                phase_task = self._create_phase_task(task, phase, phase_index)

                # Execute phase
                phase_start = time.time()
                results = await self._execute_phase(
                    formation, agents, context, phase_task, phase
                )
                phase_duration = time.time() - phase_start

                # Store phase results
                phase_result_entry = {
                    "phase": phase_index + 1,
                    "formation": phase.formation,
                    "goal": phase.goal,
                    "duration_seconds": phase_duration,
                    "results": results,
                }
                phase_results.append(phase_result_entry)

                # Check if phase failed
                phase_success = all(r.success for r in results)
                if not phase_success and self.stop_on_first_failure:
                    logger.warning(f"Phase {phase_index + 1} failed, stopping execution")
                    break

                # Update context with phase results in shared_state
                context.shared_state["last_phase_results"] = results

                phases_completed += 1

                # Check completion criteria
                if phase.completion_criteria and not phase.completion_criteria(results):
                    if self.enable_phase_logging:
                        logger.info(f"Phase {phase_index + 1} completion criteria not met, continuing")

            total_duration = time.time() - start_time

            # Return final results with metadata
            final_results = phase_results[-1]["results"] if phase_results else []
            return self._enhance_results(
                final_results,
                phases_completed,
                phase_results,
                total_duration,
            )

        except Exception as e:
            logger.error(f"HybridFormation execution failed: {e}")
            return [
                MemberResult(
                    member_id="hybrid_formation",
                    success=False,
                    output="",
                    error=str(e),
                    metadata={
                        "formation": "hybrid",
                        "phases_completed": phases_completed,
                        "phase_results": phase_results,
                    },
                )
            ]

    def _create_formation(self, formation_name: str) -> BaseFormationStrategy:
        """Create a formation instance by name.

        Args:
            formation_name: Name of formation to create

        Returns:
            Formation instance
        """
        import importlib

        formations = {
            "sequential": ("victor.coordination.formations.sequential", "SequentialFormation"),
            "parallel": ("victor.coordination.formations.parallel", "ParallelFormation"),
            "hierarchical": ("victor.coordination.formations.hierarchical", "HierarchicalFormation"),
            "pipeline": ("victor.coordination.formations.pipeline", "PipelineFormation"),
            "consensus": ("victor.coordination.formations.consensus", "ConsensusFormation"),
        }

        if formation_name in formations:
            module_path, class_name = formations[formation_name]
            module = importlib.import_module(module_path)
            formation_class = getattr(module, class_name)
            return formation_class()

        # Fallback
        logger.warning(f"Unknown formation '{formation_name}', using sequential")
        module = importlib.import_module("victor.coordination.formations.sequential")
        return module.SequentialFormation()

    def _create_phase_task(self, original_task: AgentMessage, phase: HybridPhase, phase_index: int) -> AgentMessage:
        """Create a phase-specific task message.

        Args:
            original_task: Original task message
            phase: Phase configuration
            phase_index: Phase index

        Returns:
            Phase-specific task message
        """
        # Add phase context to task
        phase_content = f"""Phase {phase_index + 1}: {phase.goal}

Original task: {original_task.content}

Focus on achieving this phase's goal. The results will be used in subsequent phases.
"""

        return AgentMessage(
            sender_id=original_task.sender_id,
            content=phase_content,
            message_type=original_task.message_type,
            recipient_id=original_task.recipient_id,
            data={
                **original_task.data,
                "phase": phase_index + 1,
                "phase_goal": phase.goal,
                "phase_formation": phase.formation,
            },
            timestamp=original_task.timestamp,
            reply_to=original_task.reply_to,
            priority=original_task.priority,
            id=original_task.id,
        )

    async def _execute_phase(
        self,
        formation: BaseFormationStrategy,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
        phase: HybridPhase,
    ) -> List[MemberResult]:
        """Execute a single phase.

        Args:
            formation: Formation to use
            agents: Agents to execute
            context: Team context
            task: Phase task
            phase: Phase configuration

        Returns:
            Phase results
        """
        # Execute with time/iteration limits if specified
        if phase.duration_budget:
            # Add timeout handling
            import asyncio

            try:
                results = await asyncio.wait_for(
                    formation.execute(agents, context, task),
                    timeout=phase.duration_budget,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Phase exceeded duration budget of {phase.duration_budget}s")
                # Return partial results or timeout indicator
                return [
                    MemberResult(
                        member_id="timeout",
                        success=False,
                        output="",
                        error=f"Phase exceeded duration budget of {phase.duration_budget}s",
                    )
                ]
        else:
            results = await formation.execute(agents, context, task)

        # Check iteration limit
        if phase.iteration_limit:
            # This is a simplified check - actual iteration tracking would
            # need to be implemented in each formation strategy
            pass

        return results

    def _enhance_results(
        self,
        results: List[MemberResult],
        phases_completed: int,
        phase_results: List[Dict[str, Any]],
        total_duration: float,
    ) -> List[MemberResult]:
        """Enhance results with hybrid formation metadata.

        Args:
            results: Final phase results
            phases_completed: Number of phases completed
            phase_results: Results from all phases
            total_duration: Total execution duration

        Returns:
            Enhanced results with metadata
        """
        enhanced = []

        for result in results:
            enhanced_metadata = {
                **result.metadata,
                "formation": "hybrid",
                "phases_completed": phases_completed,
                "total_phases": len(self.phases),
                "phase_results": phase_results,
                "duration_seconds": total_duration,
            }

            enhanced.append(
                MemberResult(
                    member_id=result.member_id,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    metadata=enhanced_metadata,
                    tool_calls_used=result.tool_calls_used,
                    duration_seconds=result.duration_seconds,
                    discoveries=result.discoveries,
                )
            )

        return enhanced

    def validate_context(self, context: TeamContext) -> bool:
        """Validate team context.

        Args:
            context: Team context to validate

        Returns:
            True if context is valid
        """
        return len(context.shared_state) > 0


__all__ = [
    "DynamicFormation",
    "AdaptiveFormation",
    "HybridFormation",
    "HybridPhase",
    "TaskCharacteristics",
    "FormationPhase",
]
