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

"""Team configuration optimization system.

This module provides optimization algorithms for finding optimal team
configurations based on task requirements, historical performance, and
resource constraints.

Example:
    from victor.teams.team_optimizer import TeamOptimizer

    optimizer = TeamOptimizer()

    # Optimize team for task
    optimal_config = optimizer.optimize_team(
        task="Implement authentication system",
        available_members=pool_members,
        constraints={"max_members": 5, "max_budget": 200}
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from victor.teams.team_predictor import TaskFeatures, TeamPredictor
    from victor.teams.types import TeamConfig, TeamFormation, TeamMember

logger = logging.getLogger(__name__)


# =============================================================================
# Optimization Objectives
# =============================================================================


class OptimizationObjective(str, Enum):
    """Objectives for team optimization."""

    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_SUCCESS = "maximize_success"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCED = "balanced"  # Balance all objectives


@dataclass
class OptimizationConstraints:
    """Constraints for team optimization.

    Attributes:
        max_members: Maximum team size
        min_members: Minimum team size
        max_budget: Maximum tool budget
        min_budget: Minimum tool budget
        required_expertise: Required expertise areas
        allowed_formations: Allowed formation patterns
        max_execution_time: Maximum acceptable execution time
        min_success_probability: Minimum success probability threshold
    """

    max_members: int = 10
    min_members: int = 1
    max_budget: int = 500
    min_budget: int = 10
    required_expertise: List[str] = field(default_factory=list)
    allowed_formations: List[str] = field(default_factory=list)
    max_execution_time: Optional[float] = None
    min_success_probability: Optional[float] = None

    def validate(self, config: "TeamConfig") -> bool:
        """Validate that team config meets constraints.

        Args:
            config: Team configuration to validate

        Returns:
            True if valid, False otherwise
        """
        # Check member count
        if not (self.min_members <= len(config.members) <= self.max_members):
            return False

        # Check budget
        if not (self.min_budget <= config.total_tool_budget <= self.max_budget):
            return False

        # Check formation
        if self.allowed_formations and config.formation.value not in self.allowed_formations:
            return False

        # Check required expertise coverage
        if self.required_expertise:
            team_expertise = set()
            for member in config.members:
                team_expertise.update(member.expertise)

            covered = set(self.required_expertise) & team_expertise
            if len(covered) < len(self.required_expertise):
                return False

        return True


@dataclass
class OptimizationResult:
    """Result of team optimization.

    Attributes:
        optimal_config: Optimal team configuration
        score: Optimization score
        predicted_metrics: Predicted performance metrics
        alternative_configs: Alternative configurations with scores
        optimization_time: Time taken to optimize
        metadata: Additional metadata
    """

    optimal_config: "TeamConfig"
    score: float
    predicted_metrics: Dict[str, Any]
    alternative_configs: List[Tuple["TeamConfig", float]] = field(default_factory=list)
    optimization_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Team Optimizer
# =============================================================================


class TeamOptimizer:
    """Team configuration optimizer using various algorithms.

    Supports multiple optimization strategies:
    - Greedy: Fast, good enough for small search spaces
    - Genetic Algorithm: Global optimization for complex spaces
    - Bayesian Optimization: Sample-efficient optimization
    - Random Search: Baseline comparison

    Example:
        optimizer = TeamOptimizer(
            optimizer_type="genetic",
            population_size=50,
            generations=20
        )

        result = optimizer.optimize_team(
            task="Implement authentication",
            available_members=member_pool,
            objective=OptimizationObjective.BALANCED,
            constraints=constraints
        )
    """

    def __init__(
        self,
        optimizer_type: str = "greedy",
        predictor: Optional["TeamPredictor"] = None,
        max_iterations: int = 100,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ):
        """Initialize team optimizer.

        Args:
            optimizer_type: Type of optimizer ("greedy", "genetic", "bayesian", "random")
            predictor: Team predictor for evaluating configs
            max_iterations: Maximum iterations for greedy/random search
            population_size: Population size for genetic algorithm
            generations: Number of generations for genetic algorithm
            mutation_rate: Mutation rate for genetic algorithm
            crossover_rate: Crossover rate for genetic algorithm
        """
        self.optimizer_type = optimizer_type
        self.predictor = predictor
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimize_team(
        self,
        task: str,
        available_members: List["TeamMember"],
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
        constraints: Optional[OptimizationConstraints] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Optimize team configuration for task.

        Args:
            task: Task description
            available_members: Pool of available members
            objective: Optimization objective
            constraints: Optimization constraints
            context: Additional context

        Returns:
            Optimization result with optimal configuration
        """
        import time

        start = time.time()
        context = context or {}
        constraints = constraints or OptimizationConstraints()

        # Route to appropriate optimizer
        if self.optimizer_type == "greedy":
            result = self._greedy_optimize(task, available_members, objective, constraints, context)
        elif self.optimizer_type == "genetic":
            result = self._genetic_optimize(
                task, available_members, objective, constraints, context
            )
        elif self.optimizer_type == "bayesian":
            result = self._bayesian_optimize(
                task, available_members, objective, constraints, context
            )
        elif self.optimizer_type == "random":
            result = self._random_optimize(task, available_members, objective, constraints, context)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        result.optimization_time = time.time() - start
        return result

    def _greedy_optimize(
        self,
        task: str,
        available_members: List["TeamMember"],
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
        context: Dict[str, Any],
    ) -> OptimizationResult:
        """Greedy optimization strategy.

        Args:
            task: Task description
            available_members: Available members
            objective: Optimization objective
            constraints: Constraints
            context: Context

        Returns:
            Optimization result
        """
        best_config = None
        best_score = float("-inf")
        best_metrics = {}
        alternatives = []

        # Greedy approach: iteratively add members
        current_members = []

        for _ in range(min(len(available_members), constraints.max_members)):
            # Try adding each remaining member
            best_addition = None
            best_addition_score = float("-inf")

            for member in available_members:
                if member in current_members:
                    continue

                test_members = current_members + [member]

                # Test each allowed formation
                for formation in self._get_formations_to_test(constraints):
                    config = self._create_config(test_members, formation, constraints, context)

                    if not constraints.validate(config):
                        continue

                    score, metrics = self._evaluate_config(config, task, objective, context)

                    if score > best_addition_score:
                        best_addition_score = score
                        best_addition = member
                        best_config = config
                        best_metrics = metrics

            if best_addition:
                current_members.append(best_addition)
                alternatives.append((best_config, best_addition_score))
            else:
                break

        return OptimizationResult(
            optimal_config=best_config
            or self._create_default_config(available_members, constraints),
            score=best_score if best_score > float("-inf") else 0.0,
            predicted_metrics=best_metrics,
            alternative_configs=alternatives[:5],  # Top 5 alternatives
        )

    def _genetic_optimize(
        self,
        task: str,
        available_members: List["TeamMember"],
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
        context: Dict[str, Any],
    ) -> OptimizationResult:
        """Genetic algorithm optimization.

        Args:
            task: Task description
            available_members: Available members
            objective: Optimization objective
            constraints: Constraints
            context: Context

        Returns:
            Optimization result
        """
        # Initialize population
        population = self._initialize_population(available_members, constraints)

        best_config = None
        best_score = float("-inf")
        best_metrics = {}
        alternatives = []

        # Evolve population
        for generation in range(self.generations):
            # Evaluate fitness
            scored_population = []
            for individual in population:
                if not constraints.validate(individual):
                    score = -1.0
                    metrics = {}
                else:
                    score, metrics = self._evaluate_config(individual, task, objective, context)

                scored_population.append((individual, score, metrics))

                # Track best
                if score > best_score:
                    best_score = score
                    best_config = individual
                    best_metrics = metrics

            # Sort by fitness
            scored_population.sort(key=lambda x: x[1], reverse=True)

            # Selection (elitism)
            elite_size = int(self.population_size * 0.1)
            new_population = [x[0] for x in scored_population[:elite_size]]

            # Crossover
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(scored_population)
                parent2 = self._tournament_selection(scored_population)

                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2, available_members)
                else:
                    child = parent1

                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child, available_members, constraints)

                new_population.append(child)

            population = new_population

            # Track alternatives from final generation
            if generation == self.generations - 1:
                alternatives = [(x[0], x[1]) for x in scored_population[:6]]

        return OptimizationResult(
            optimal_config=best_config
            or self._create_default_config(available_members, constraints),
            score=best_score if best_score > float("-inf") else 0.0,
            predicted_metrics=best_metrics,
            alternative_configs=alternatives[1:6],  # Exclude best itself
        )

    def _bayesian_optimize(
        self,
        task: str,
        available_members: List["TeamMember"],
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
        context: Dict[str, Any],
    ) -> OptimizationResult:
        """Bayesian optimization strategy (simplified placeholder).

        In production, this would use a library like scikit-optimize or GPyOpt.

        Args:
            task: Task description
            available_members: Available members
            objective: Optimization objective
            constraints: Constraints
            context: Context

        Returns:
            Optimization result
        """
        # Placeholder: use random sampling with intelligent selection
        return self._random_optimize(task, available_members, objective, constraints, context)

    def _random_optimize(
        self,
        task: str,
        available_members: List["TeamMember"],
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
        context: Dict[str, Any],
    ) -> OptimizationResult:
        """Random search optimization.

        Args:
            task: Task description
            available_members: Available members
            objective: Optimization objective
            constraints: Constraints
            context: Context

        Returns:
            Optimization result
        """
        best_config = None
        best_score = float("-inf")
        best_metrics = {}
        alternatives = []

        for _ in range(self.max_iterations):
            # Random sample
            config = self._random_config(available_members, constraints)

            if not constraints.validate(config):
                continue

            score, metrics = self._evaluate_config(config, task, objective, context)

            if score > best_score:
                best_score = score
                best_config = config
                best_metrics = metrics
                alternatives.append((config, score))

        return OptimizationResult(
            optimal_config=best_config
            or self._create_default_config(available_members, constraints),
            score=best_score if best_score > float("-inf") else 0.0,
            predicted_metrics=best_metrics,
            alternative_configs=sorted(alternatives[1:6], key=lambda x: x[1], reverse=True),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _evaluate_config(
        self,
        config: "TeamConfig",
        task: str,
        objective: OptimizationObjective,
        context: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate team configuration.

        Args:
            config: Team configuration
            task: Task description
            objective: Optimization objective
            context: Context

        Returns:
            Tuple of (score, metrics)
        """
        metrics = {}

        # Use predictor if available
        if self.predictor:
            from victor.teams.team_predictor import PredictionMetric

            # Predict key metrics
            time_result = self.predictor.predict(
                PredictionMetric.EXECUTION_TIME, config, task, context
            )
            success_result = self.predictor.predict(
                PredictionMetric.SUCCESS_PROBABILITY, config, task, context
            )
            quality_result = self.predictor.predict(
                PredictionMetric.QUALITY_SCORE, config, task, context
            )

            metrics["execution_time"] = time_result.predicted_value
            metrics["success_probability"] = success_result.predicted_value
            metrics["quality_score"] = quality_result.predicted_value
            metrics["confidence"] = (
                time_result.confidence + success_result.confidence + quality_result.confidence
            ) / 3

        # Calculate score based on objective
        if objective == OptimizationObjective.MINIMIZE_TIME:
            score = 1.0 / max(1.0, metrics.get("execution_time", 100))
        elif objective == OptimizationObjective.MAXIMIZE_SUCCESS:
            score = metrics.get("success_probability", 0.5)
        elif objective == OptimizationObjective.MAXIMIZE_QUALITY:
            score = metrics.get("quality_score", 0.5)
        elif objective == OptimizationObjective.MINIMIZE_COST:
            score = 1.0 / max(1.0, config.total_tool_budget)
        elif objective == OptimizationObjective.BALANCED:
            # Weighted combination
            time_score = 1.0 / max(1.0, metrics.get("execution_time", 100))
            success_score = metrics.get("success_probability", 0.5)
            quality_score = metrics.get("quality_score", 0.5)
            cost_score = 1.0 / max(1.0, config.total_tool_budget)

            score = (
                time_score * 0.25 + success_score * 0.35 + quality_score * 0.25 + cost_score * 0.15
            )
        else:
            score = 0.5

        # Apply confidence weighting
        confidence = metrics.get("confidence", 0.5)
        adjusted_score = score * (0.5 + confidence * 0.5)

        return adjusted_score, metrics

    def _create_config(
        self,
        members: List["TeamMember"],
        formation: "TeamFormation",
        constraints: OptimizationConstraints,
        context: Dict[str, Any],
    ) -> "TeamConfig":
        """Create team configuration.

        Args:
            members: Team members
            formation: Formation pattern
            constraints: Constraints
            context: Context

        Returns:
            TeamConfig
        """
        from victor.teams.types import TeamConfig

        return TeamConfig(
            name="Optimized Team",
            goal=context.get("goal", "Execute task"),
            members=members,
            formation=formation,
            total_tool_budget=min(constraints.max_budget, len(members) * 50),
        )

    def _create_default_config(
        self, available_members: List["TeamMember"], constraints: OptimizationConstraints
    ) -> "TeamConfig":
        """Create default configuration.

        Args:
            available_members: Available members
            constraints: Constraints

        Returns:
            Default TeamConfig
        """
        from victor.teams.types import TeamConfig, TeamFormation

        members = available_members[: constraints.min_members]
        return TeamConfig(
            name="Default Team",
            goal="Execute task",
            members=members,
            formation=TeamFormation.SEQUENTIAL,
            total_tool_budget=constraints.min_budget,
        )

    def _get_formations_to_test(
        self, constraints: OptimizationConstraints
    ) -> List["TeamFormation"]:
        """Get formations to test.

        Args:
            constraints: Constraints

        Returns:
            List of formations
        """
        from victor.teams.types import TeamFormation

        if constraints.allowed_formations:
            return [TeamFormation(f) for f in constraints.allowed_formations]

        return [
            TeamFormation.SEQUENTIAL,
            TeamFormation.PARALLEL,
            TeamFormation.HIERARCHICAL,
            TeamFormation.PIPELINE,
        ]

    def _initialize_population(
        self, available_members: List["TeamMember"], constraints: OptimizationConstraints
    ) -> List["TeamConfig"]:
        """Initialize population for genetic algorithm.

        Args:
            available_members: Available members
            constraints: Constraints

        Returns:
            Initial population
        """
        population = []

        while len(population) < self.population_size:
            config = self._random_config(available_members, constraints)
            population.append(config)

        return population

    def _random_config(
        self, available_members: List["TeamMember"], constraints: OptimizationConstraints
    ) -> "TeamConfig":
        """Generate random configuration.

        Args:
            available_members: Available members
            constraints: Constraints

        Returns:
            Random TeamConfig
        """
        import random

        from victor.teams.types import TeamConfig, TeamFormation

        # Random team size
        team_size = random.randint(
            constraints.min_members, min(len(available_members), constraints.max_members)
        )

        # Random members
        members = random.sample(available_members, team_size)

        # Random formation
        formations = self._get_formations_to_test(constraints)
        formation = random.choice(formations)

        # Random budget
        budget = random.randint(constraints.min_budget, constraints.max_budget)

        return TeamConfig(
            name="Random Team",
            goal="Execute task",
            members=members,
            formation=formation,
            total_tool_budget=budget,
        )

    def _tournament_selection(
        self, scored_population: List[Tuple["TeamConfig", float, Dict[str, Any]]]
    ) -> "TeamConfig":
        """Tournament selection for genetic algorithm.

        Args:
            scored_population: Population with fitness scores

        Returns:
            Selected individual
        """
        import random

        tournament_size = 3
        tournament = random.sample(scored_population, min(tournament_size, len(scored_population)))
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(
        self, parent1: "TeamConfig", parent2: "TeamConfig", available_members: List["TeamMember"]
    ) -> "TeamConfig":
        """Crossover two parent configurations.

        Args:
            parent1: First parent
            parent2: Second parent
            available_members: Available members

        Returns:
            Child configuration
        """
        import random

        from victor.teams.types import TeamConfig

        # Combine members from both parents
        all_members = list({m.id: m for m in parent1.members + parent2.members}.values())
        child_members = random.sample(
            all_members,
            random.randint(min(len(parent1.members), len(parent2.members)), len(all_members)),
        )

        # Randomly inherit formation or budget
        child_formation = random.choice([parent1.formation, parent2.formation])
        child_budget = random.choice([parent1.total_tool_budget, parent2.total_tool_budget])

        return TeamConfig(
            name="Child Team",
            goal="Execute task",
            members=child_members,
            formation=child_formation,
            total_tool_budget=child_budget,
        )

    def _mutate(
        self,
        individual: "TeamConfig",
        available_members: List["TeamMember"],
        constraints: OptimizationConstraints,
    ) -> "TeamConfig":
        """Mutate individual configuration.

        Args:
            individual: Individual to mutate
            available_members: Available members
            constraints: Constraints

        Returns:
            Mutated configuration
        """
        import random

        from victor.teams.types import TeamFormation

        # Random mutation type
        mutation_type = random.choice(
            ["add_member", "remove_member", "change_formation", "change_budget"]
        )

        if mutation_type == "add_member" and len(individual.members) < constraints.max_members:
            # Add random member
            available = [m for m in available_members if m not in individual.members]
            if available:
                individual.members.append(random.choice(available))

        elif mutation_type == "remove_member" and len(individual.members) > constraints.min_members:
            # Remove random member
            if individual.members:
                individual.members.pop(random.randint(0, len(individual.members) - 1))

        elif mutation_type == "change_formation":
            # Change to random formation
            formations = self._get_formations_to_test(constraints)
            individual.formation = random.choice(formations)

        elif mutation_type == "change_budget":
            # Change budget
            individual.total_tool_budget = random.randint(
                constraints.min_budget, constraints.max_budget
            )

        return individual


__all__ = [
    "OptimizationObjective",
    "OptimizationConstraints",
    "OptimizationResult",
    "TeamOptimizer",
]
