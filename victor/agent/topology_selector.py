# Copyright 2025 Vijaykumar Singh <singhv@gmail.com>
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

"""Heuristic topology selector for runtime structure decisions.

This module implements the first executable version of topology-aware routing.
The selector intentionally uses transparent heuristics so the runtime can emit
inspectable rationale and telemetry before moving to learned policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecision,
    TopologyDecisionInput,
    TopologyGroundingRequirements,
    TopologyKind,
)


@dataclass
class TopologySelectorConfig:
    """Configuration for heuristic topology selection."""

    fallback_score_gap: float = 0.12
    low_confidence_threshold: float = 0.65
    min_parallel_workers: int = 2
    max_parallel_workers: int = 4


class TopologySelector:
    """Choose a runtime topology from request-time execution hints."""

    LOCAL_PROVIDER_HINTS = {
        "local",
        "ollama",
        "vllm",
        "lmstudio",
        "llamacpp",
        "llama.cpp",
    }

    def __init__(self, config: Optional[TopologySelectorConfig] = None):
        self.config = config or TopologySelectorConfig()

    def select(self, decision_input: TopologyDecisionInput) -> TopologyDecision:
        """Return a heuristic topology decision for the given runtime hints."""
        score_breakdown = {
            TopologyAction.DIRECT_RESPONSE: self._score_direct_response(decision_input),
            TopologyAction.SINGLE_AGENT: self._score_single_agent(decision_input),
            TopologyAction.PARALLEL_EXPLORATION: self._score_parallel_exploration(decision_input),
            TopologyAction.TEAM_PLAN: self._score_team_plan(decision_input),
            TopologyAction.ESCALATE_MODEL: self._score_escalate_model(decision_input),
            TopologyAction.SAFE_STOP: self._score_safe_stop(decision_input),
        }

        ordered = sorted(score_breakdown.items(), key=lambda item: item[1], reverse=True)
        best_action, best_score = ordered[0]
        second_action, second_score = ordered[1]

        confidence = self._normalize_confidence(best_score)
        fallback_action = self._select_fallback_action(
            confidence=confidence,
            best_action=best_action,
            best_score=best_score,
            second_action=second_action,
            second_score=second_score,
        )

        provider = self._select_provider_hint(decision_input, best_action)
        formation = self._select_formation_hint(decision_input, best_action)
        grounding = self._build_grounding(
            decision_input=decision_input,
            action=best_action,
            provider=provider,
            formation=formation,
        )

        return TopologyDecision(
            action=best_action,
            topology=self._topology_kind_for_action(best_action),
            confidence=confidence,
            rationale=self._build_rationale(best_action, decision_input),
            score_breakdown={action.value: round(score, 4) for action, score in ordered},
            grounding_requirements=grounding,
            fallback_action=fallback_action,
            telemetry_tags={
                "task_type": decision_input.task_type,
                "task_complexity": decision_input.task_complexity,
                "privacy_sensitivity": decision_input.privacy_sensitivity,
                "latency_sensitivity": decision_input.latency_sensitivity,
            },
            provider=provider,
            formation=formation,
        )

    def _score_direct_response(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.0
        if self._is_low(decision_input.task_complexity):
            score += 0.25
        if self._is_low(decision_input.expected_depth):
            score += 0.2
        if self._is_low(decision_input.expected_breadth):
            score += 0.1
        if decision_input.tool_budget <= 1:
            score += 0.15
        if self._is_high(decision_input.latency_sensitivity):
            score += 0.1
        if decision_input.confidence_hint >= 0.75:
            score += 0.1
        if decision_input.prior_failures > 0:
            score -= 0.2
        if self._is_low(decision_input.observability_level):
            score -= 0.1
        return score

    def _score_single_agent(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.2
        if self._is_low(decision_input.task_complexity):
            score += 0.25
        elif self._is_medium(decision_input.task_complexity):
            score += 0.15
        else:
            score -= 0.05

        if self._is_low(decision_input.expected_depth):
            score += 0.15
        elif self._is_medium(decision_input.expected_depth):
            score += 0.05
        else:
            score -= 0.15

        if self._is_low(decision_input.expected_breadth):
            score += 0.1
        elif self._is_high(decision_input.expected_breadth):
            score -= 0.15

        if self._is_high(decision_input.observability_level):
            score += 0.15
        elif self._is_medium(decision_input.observability_level):
            score += 0.05
        else:
            score -= 0.1

        if decision_input.prior_failures == 0:
            score += 0.05
        else:
            score -= min(0.08 * decision_input.prior_failures, 0.24)

        if self._is_high(decision_input.latency_sensitivity):
            score += 0.05
        if self._is_high(decision_input.privacy_sensitivity):
            score += 0.05
        if decision_input.confidence_hint >= 0.6:
            score += 0.05
        elif decision_input.confidence_hint < 0.4:
            score -= 0.05
        return score

    def _score_parallel_exploration(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.1
        if self._is_high(decision_input.expected_breadth):
            score += 0.35
        elif self._is_medium(decision_input.expected_breadth):
            score += 0.1
        else:
            score -= 0.05

        if self._is_high(decision_input.task_complexity):
            score += 0.1
        elif self._is_medium(decision_input.task_complexity):
            score += 0.05

        if self._is_low(decision_input.observability_level):
            score += 0.15
        elif self._is_medium(decision_input.observability_level):
            score += 0.05

        if decision_input.prior_failures > 0:
            score += 0.1

        if self._is_high(decision_input.latency_sensitivity):
            score -= 0.2
        elif self._is_medium(decision_input.latency_sensitivity):
            score -= 0.05

        if self._is_high(decision_input.token_cost_pressure):
            score -= 0.1
        if decision_input.tool_budget < 4:
            score -= 0.15
        return score

    def _score_team_plan(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.05
        if self._is_high(decision_input.expected_depth):
            score += 0.35
        elif self._is_medium(decision_input.expected_depth):
            score += 0.1
        else:
            score -= 0.05

        if self._is_high(decision_input.task_complexity):
            score += 0.25
        elif self._is_medium(decision_input.task_complexity):
            score += 0.1

        if decision_input.available_team_formations:
            score += 0.1
        else:
            score -= 0.2

        if self._is_low(decision_input.observability_level):
            score += 0.1

        if self._is_high(decision_input.latency_sensitivity):
            score -= 0.2
        elif self._is_medium(decision_input.latency_sensitivity):
            score -= 0.05

        if self._is_high(decision_input.token_cost_pressure):
            score -= 0.15
        if decision_input.prior_failures > 1:
            score += 0.05
        return score

    def _score_escalate_model(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.05
        if decision_input.confidence_hint < 0.4:
            score += 0.25
        elif decision_input.confidence_hint < 0.6:
            score += 0.1

        if self._is_low(decision_input.observability_level):
            score += 0.2
        elif self._is_medium(decision_input.observability_level):
            score += 0.05

        if self._is_high(decision_input.task_complexity):
            score += 0.15
        elif self._is_medium(decision_input.task_complexity):
            score += 0.05

        if decision_input.prior_failures > 0:
            score += 0.15

        if self._is_high(decision_input.token_cost_pressure):
            score -= 0.1
        if self._is_high(decision_input.privacy_sensitivity) and self._is_remote_only(
            decision_input.provider_candidates
        ):
            score -= 0.4
        if not decision_input.provider_candidates:
            score -= 0.1
        return score

    def _score_safe_stop(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.0
        remote_only = self._is_remote_only(decision_input.provider_candidates)
        if self._is_high(decision_input.privacy_sensitivity) and remote_only:
            score += 0.5
        if self._is_high(decision_input.privacy_sensitivity) and self._is_low(
            decision_input.observability_level
        ):
            score += 0.2
        if decision_input.confidence_hint < 0.3 and decision_input.prior_failures > 1:
            score += 0.15
        if not decision_input.provider_candidates:
            score += 0.1
        if (
            self._is_high(decision_input.task_complexity)
            and self._is_low(decision_input.observability_level)
            and self._is_high(decision_input.privacy_sensitivity)
            and remote_only
        ):
            score += 0.1
        return score

    def _select_fallback_action(
        self,
        *,
        confidence: float,
        best_action: TopologyAction,
        best_score: float,
        second_action: TopologyAction,
        second_score: float,
    ) -> Optional[TopologyAction]:
        gap = best_score - second_score
        if (
            confidence < self.config.low_confidence_threshold
            or gap <= self.config.fallback_score_gap
        ):
            if second_action is not best_action:
                return second_action
        return None

    def _select_provider_hint(
        self,
        decision_input: TopologyDecisionInput,
        action: TopologyAction,
    ) -> Optional[str]:
        candidates = decision_input.provider_candidates
        if not candidates or action == TopologyAction.SAFE_STOP:
            return None

        if action == TopologyAction.ESCALATE_MODEL:
            remote_candidates = [
                provider for provider in candidates if not self._is_local_provider(provider)
            ]
            if remote_candidates:
                return remote_candidates[0]
            return candidates[0]

        if self._is_high(decision_input.privacy_sensitivity):
            local_candidates = [
                provider for provider in candidates if self._is_local_provider(provider)
            ]
            if local_candidates:
                return local_candidates[0]

        return candidates[0]

    def _select_formation_hint(
        self,
        decision_input: TopologyDecisionInput,
        action: TopologyAction,
    ) -> Optional[str]:
        if action != TopologyAction.TEAM_PLAN:
            return None

        formations = decision_input.available_team_formations
        if not formations:
            return "adaptive"
        if self._is_high(decision_input.expected_depth) and "hierarchical" in formations:
            return "hierarchical"
        if "adaptive" in formations:
            return "adaptive"
        return formations[0]

    def _build_grounding(
        self,
        *,
        decision_input: TopologyDecisionInput,
        action: TopologyAction,
        provider: Optional[str],
        formation: Optional[str],
    ) -> TopologyGroundingRequirements:
        if action == TopologyAction.DIRECT_RESPONSE:
            return TopologyGroundingRequirements(
                provider=provider,
                tool_budget=0,
                iteration_budget=1,
                metadata={"topology": "direct"},
            )

        if action == TopologyAction.SINGLE_AGENT:
            return TopologyGroundingRequirements(
                provider=provider,
                tool_budget=decision_input.tool_budget,
                iteration_budget=decision_input.iteration_budget,
                metadata={"topology": "single_agent"},
            )

        if action == TopologyAction.PARALLEL_EXPLORATION:
            return TopologyGroundingRequirements(
                provider=provider,
                max_workers=self._parallel_worker_count(decision_input),
                tool_budget=decision_input.tool_budget,
                iteration_budget=min(decision_input.iteration_budget, 2),
                metadata={"topology": "parallel_exploration"},
            )

        if action == TopologyAction.TEAM_PLAN:
            return TopologyGroundingRequirements(
                provider=provider,
                formation=formation,
                max_workers=self._parallel_worker_count(decision_input),
                tool_budget=decision_input.tool_budget,
                iteration_budget=decision_input.iteration_budget,
                metadata={"topology": "team_plan"},
            )

        if action == TopologyAction.ESCALATE_MODEL:
            return TopologyGroundingRequirements(
                provider=provider,
                tool_budget=max(decision_input.tool_budget, 4),
                iteration_budget=decision_input.iteration_budget,
                escalation_target=provider,
                metadata={"topology": "escalated_single_agent"},
            )

        return TopologyGroundingRequirements(
            safety_mode="privacy_block",
            metadata={"topology": "safe_stop"},
        )

    def _parallel_worker_count(self, decision_input: TopologyDecisionInput) -> int:
        if self._is_high(decision_input.expected_breadth):
            return min(
                self.config.max_parallel_workers,
                max(self.config.min_parallel_workers, decision_input.tool_budget // 2),
            )
        return self.config.min_parallel_workers

    def _topology_kind_for_action(self, action: TopologyAction) -> TopologyKind:
        return {
            TopologyAction.DIRECT_RESPONSE: TopologyKind.DIRECT,
            TopologyAction.SINGLE_AGENT: TopologyKind.SINGLE_AGENT,
            TopologyAction.PARALLEL_EXPLORATION: TopologyKind.PARALLEL_EXPLORATION,
            TopologyAction.TEAM_PLAN: TopologyKind.TEAM,
            TopologyAction.ESCALATE_MODEL: TopologyKind.ESCALATED_SINGLE_AGENT,
            TopologyAction.SAFE_STOP: TopologyKind.SAFE_STOP,
        }[action]

    def _build_rationale(
        self,
        action: TopologyAction,
        decision_input: TopologyDecisionInput,
    ) -> str:
        if action == TopologyAction.DIRECT_RESPONSE:
            return "Low-complexity, low-depth request favors a direct response path."
        if action == TopologyAction.SINGLE_AGENT:
            return "Manageable complexity and good observability favor a single-agent loop."
        if action == TopologyAction.PARALLEL_EXPLORATION:
            return "Breadth-heavy search with enough budget favors parallel exploration."
        if action == TopologyAction.TEAM_PLAN:
            return "High depth and coordination needs favor an explicit team plan."
        if action == TopologyAction.ESCALATE_MODEL:
            return (
                "Low confidence or weak observability favors escalating to a stronger model path."
            )
        return "High privacy sensitivity combined with remote-only execution pressure favors a safe stop."

    def _normalize_confidence(self, score: float) -> float:
        return max(0.0, min(score, 1.0))

    def _is_remote_only(self, provider_candidates: List[str]) -> bool:
        if not provider_candidates:
            return False
        return all(not self._is_local_provider(provider) for provider in provider_candidates)

    def _is_local_provider(self, provider_name: str) -> bool:
        normalized = provider_name.strip().lower()
        return any(hint in normalized for hint in self.LOCAL_PROVIDER_HINTS)

    @staticmethod
    def _is_low(value: str) -> bool:
        return value.strip().lower() == "low"

    @staticmethod
    def _is_medium(value: str) -> bool:
        return value.strip().lower() == "medium"

    @staticmethod
    def _is_high(value: str) -> bool:
        return value.strip().lower() == "high"


__all__ = [
    "TopologySelector",
    "TopologySelectorConfig",
]
