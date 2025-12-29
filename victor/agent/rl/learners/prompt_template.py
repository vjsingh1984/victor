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

"""RL learner for prompt template optimization.

This learner uses multi-armed bandit (Thompson Sampling) to learn
optimal prompt templates and prompt engineering patterns based on
task type, provider, and outcome feedback.

Problem:
- Different tasks may benefit from different prompt structures
- Different providers may respond better to different prompt styles
- Optimal prompt patterns should be learned from outcomes

Strategy:
- State: (task_type, provider, model)
- Arms: Different prompt template variants
- Reward: Task success, quality score, grounding verification

Algorithm: Thompson Sampling with Beta priors

Sprint 5: Advanced RL Patterns
"""

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.agent.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


class PromptStyle(str, Enum):
    """Available prompt styles/templates."""

    CONCISE = "concise"  # Minimal, direct prompts
    DETAILED = "detailed"  # Comprehensive with context
    STRUCTURED = "structured"  # Step-by-step with headers
    COT = "cot"  # Chain-of-thought prompting
    FEW_SHOT = "few_shot"  # With examples
    ROLE_BASED = "role_based"  # "You are an expert..." prefix


class PromptElement(str, Enum):
    """Optional prompt elements that can be included."""

    TASK_CONTEXT = "task_context"  # Explain the broader context
    CONSTRAINTS = "constraints"  # List constraints/requirements
    EXAMPLES = "examples"  # Include examples
    OUTPUT_FORMAT = "output_format"  # Specify expected output format
    THINKING_PROMPT = "thinking_prompt"  # Encourage step-by-step thinking
    VERIFICATION = "verification"  # Ask to verify/validate response
    CONFIDENCE = "confidence"  # Request confidence level


@dataclass
class PromptTemplate:
    """Represents a prompt template configuration.

    Attributes:
        style: The overall prompt style
        elements: Which optional elements to include
        confidence: Confidence in this template (from learning)
        sample_count: Number of times this template was used
    """

    style: PromptStyle = PromptStyle.STRUCTURED
    elements: List[PromptElement] = field(default_factory=list)
    confidence: float = 0.5
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "style": self.style.value,
            "elements": [e.value for e in self.elements],
            "confidence": self.confidence,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            style=PromptStyle(data.get("style", "structured")),
            elements=[PromptElement(e) for e in data.get("elements", [])],
            confidence=data.get("confidence", 0.5),
            sample_count=data.get("sample_count", 0),
        )


@dataclass
class BetaDistribution:
    """Beta distribution for Thompson Sampling.

    Attributes:
        alpha: Number of successes + 1 (prior)
        beta: Number of failures + 1 (prior)
    """

    alpha: float = 1.0  # Prior
    beta: float = 1.0  # Prior

    def sample(self) -> float:
        """Sample from the beta distribution."""
        return random.betavariate(self.alpha, self.beta)

    def mean(self) -> float:
        """Get the mean of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Get the variance of the distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    def update(self, success: bool) -> None:
        """Update distribution with outcome."""
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0


class PromptTemplateLearner(BaseLearner):
    """Learn optimal prompt templates per context.

    Uses Thompson Sampling to balance exploration and exploitation
    when selecting prompt templates for different task/provider contexts.

    Attributes:
        name: Always "prompt_template"
        db: SQLite database connection
        exploration_rate: Additional exploration probability
    """

    # Default templates for different task types
    DEFAULT_TEMPLATES: Dict[str, PromptTemplate] = {
        "analysis": PromptTemplate(
            style=PromptStyle.STRUCTURED,
            elements=[PromptElement.TASK_CONTEXT, PromptElement.OUTPUT_FORMAT],
        ),
        "code_generation": PromptTemplate(
            style=PromptStyle.DETAILED,
            elements=[
                PromptElement.CONSTRAINTS,
                PromptElement.OUTPUT_FORMAT,
                PromptElement.VERIFICATION,
            ],
        ),
        "debugging": PromptTemplate(
            style=PromptStyle.COT,
            elements=[PromptElement.THINKING_PROMPT, PromptElement.VERIFICATION],
        ),
        "explanation": PromptTemplate(
            style=PromptStyle.STRUCTURED,
            elements=[PromptElement.TASK_CONTEXT, PromptElement.EXAMPLES],
        ),
        "refactoring": PromptTemplate(
            style=PromptStyle.DETAILED,
            elements=[
                PromptElement.CONSTRAINTS,
                PromptElement.THINKING_PROMPT,
                PromptElement.VERIFICATION,
            ],
        ),
        "default": PromptTemplate(
            style=PromptStyle.STRUCTURED,
            elements=[PromptElement.TASK_CONTEXT, PromptElement.OUTPUT_FORMAT],
        ),
    }

    # All available arms (template configurations)
    ALL_STYLES = list(PromptStyle)
    ALL_ELEMENTS = list(PromptElement)

    # Minimum samples before confident recommendation
    MIN_SAMPLES_FOR_CONFIDENCE = 10

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
        exploration_rate: float = 0.1,
    ):
        """Initialize prompt template learner.

        Args:
            name: Learner name (should be "prompt_template")
            db_connection: SQLite database connection
            learning_rate: Not used (Thompson Sampling is Bayesian)
            provider_adapter: Optional provider adapter
            exploration_rate: Additional pure exploration probability
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.exploration_rate = exploration_rate

        # Beta distributions for each (context, style) pair
        # Key: (task_type, provider, style) -> BetaDistribution
        self._style_posteriors: Dict[Tuple[str, str, str], BetaDistribution] = {}

        # Beta distributions for each (context, element) pair
        # Key: (task_type, provider, element) -> BetaDistribution
        self._element_posteriors: Dict[Tuple[str, str, str], BetaDistribution] = {}

        # Track recent selections for feedback attribution
        self._recent_selections: Dict[str, PromptTemplate] = {}
        self._max_tracked_selections = 500

        # Sample counts per context
        self._sample_counts: Dict[Tuple[str, str], int] = {}

        # Load state from database
        self._load_state()

    def _ensure_tables(self) -> None:
        """Create tables for prompt template learning."""
        cursor = self.db.cursor()

        # Style posteriors table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.AGENT_PROMPT_STYLE} (
                task_type TEXT NOT NULL,
                provider TEXT NOT NULL,
                style TEXT NOT NULL,
                alpha REAL NOT NULL DEFAULT 1.0,
                beta REAL NOT NULL DEFAULT 1.0,
                sample_count INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (task_type, provider, style)
            )
            """
        )

        # Element posteriors table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.AGENT_PROMPT_ELEMENT} (
                task_type TEXT NOT NULL,
                provider TEXT NOT NULL,
                element TEXT NOT NULL,
                alpha REAL NOT NULL DEFAULT 1.0,
                beta REAL NOT NULL DEFAULT 1.0,
                sample_count INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (task_type, provider, element)
            )
            """
        )

        # Learning history
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.AGENT_PROMPT_HISTORY} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT,
                template_used TEXT NOT NULL,
                success REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )

        # Indexes
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_agent_prompt_style_context
            ON {Tables.AGENT_PROMPT_STYLE}(task_type, provider)
            """
        )
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_agent_prompt_element_context
            ON {Tables.AGENT_PROMPT_ELEMENT}(task_type, provider)
            """
        )

        self.db.commit()
        logger.debug("RL: prompt_template tables ensured")

    def _load_state(self) -> None:
        """Load state from database."""
        cursor = self.db.cursor()

        try:
            # Load style posteriors
            cursor.execute(f"SELECT * FROM {Tables.AGENT_PROMPT_STYLE}")
            for row in cursor.fetchall():
                row_dict = dict(row)
                key = (row_dict["task_type"], row_dict["provider"], row_dict["style"])
                self._style_posteriors[key] = BetaDistribution(
                    alpha=row_dict["alpha"],
                    beta=row_dict["beta"],
                )
                context_key = (row_dict["task_type"], row_dict["provider"])
                self._sample_counts[context_key] = max(
                    self._sample_counts.get(context_key, 0),
                    row_dict["sample_count"],
                )

            # Load element posteriors
            cursor.execute(f"SELECT * FROM {Tables.AGENT_PROMPT_ELEMENT}")
            for row in cursor.fetchall():
                row_dict = dict(row)
                key = (row_dict["task_type"], row_dict["provider"], row_dict["element"])
                self._element_posteriors[key] = BetaDistribution(
                    alpha=row_dict["alpha"],
                    beta=row_dict["beta"],
                )

        except Exception as e:
            logger.debug(f"RL: Could not load prompt template state: {e}")

        if self._style_posteriors:
            logger.info(
                f"RL: Loaded prompt template posteriors for "
                f"{len(self._style_posteriors)} style contexts"
            )

    def _get_context_key(self, task_type: str, provider: str) -> Tuple[str, str]:
        """Get context key for lookup."""
        return (task_type.lower(), provider.lower())

    def _get_style_posterior(
        self, task_type: str, provider: str, style: PromptStyle
    ) -> BetaDistribution:
        """Get or create posterior for a style."""
        key = (task_type.lower(), provider.lower(), style.value)
        if key not in self._style_posteriors:
            self._style_posteriors[key] = BetaDistribution()
        return self._style_posteriors[key]

    def _get_element_posterior(
        self, task_type: str, provider: str, element: PromptElement
    ) -> BetaDistribution:
        """Get or create posterior for an element."""
        key = (task_type.lower(), provider.lower(), element.value)
        if key not in self._element_posteriors:
            self._element_posteriors[key] = BetaDistribution()
        return self._element_posteriors[key]

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record prompt template outcome and update posteriors.

        Expected metadata:
        - session_id: For matching to selection
        - template_used: Optional explicit template that was used
        - grounding_score: Optional grounding verification score

        Args:
            outcome: Outcome with prompt usage data
        """
        task_type = outcome.task_type
        provider = outcome.provider
        model = outcome.model

        # Get template that was used
        session_id = outcome.metadata.get("session_id", "")
        template_dict = outcome.metadata.get("template_used")

        if template_dict:
            template = PromptTemplate.from_dict(template_dict)
        elif session_id in self._recent_selections:
            template = self._recent_selections[session_id]
        else:
            # No template tracking, use default
            default_key = task_type if task_type in self.DEFAULT_TEMPLATES else "default"
            template = self.DEFAULT_TEMPLATES.get(default_key, self.DEFAULT_TEMPLATES["default"])

        # Compute success signal
        success_score = self._compute_success_score(outcome)
        success = success_score >= 0.5

        # Update style posterior
        style_posterior = self._get_style_posterior(task_type, provider, template.style)
        style_posterior.update(success)

        # Update element posteriors
        for element in template.elements:
            element_posterior = self._get_element_posterior(task_type, provider, element)
            element_posterior.update(success)

        # Update sample count
        context_key = self._get_context_key(task_type, provider)
        self._sample_counts[context_key] = self._sample_counts.get(context_key, 0) + 1

        # Save to database
        self._save_to_db(task_type, provider, model, template, success_score)

        logger.debug(
            f"RL: Prompt template updated for {task_type}/{provider}, "
            f"style={template.style.value}, success={success}"
        )

    def _compute_success_score(self, outcome: RLOutcome) -> float:
        """Compute success score from outcome.

        Args:
            outcome: RLOutcome to evaluate

        Returns:
            Success score between 0 and 1
        """
        # Combine multiple signals
        task_success = 1.0 if outcome.success else 0.0
        quality_score = outcome.quality_score

        grounding_score = outcome.metadata.get("grounding_score", 0.5)
        if isinstance(grounding_score, dict):
            grounding_score = grounding_score.get("overall", 0.5)

        # Weighted combination
        return 0.4 * task_success + 0.4 * quality_score + 0.2 * grounding_score

    def _save_to_db(
        self,
        task_type: str,
        provider: str,
        model: Optional[str],
        template: PromptTemplate,
        success: float,
    ) -> None:
        """Save posteriors and history to database."""
        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()
        context_key = self._get_context_key(task_type, provider)
        sample_count = self._sample_counts.get(context_key, 0)

        # Save style posterior
        style_posterior = self._get_style_posterior(task_type, provider, template.style)
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.AGENT_PROMPT_STYLE}
            (task_type, provider, style, alpha, beta, sample_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_type.lower(),
                provider.lower(),
                template.style.value,
                style_posterior.alpha,
                style_posterior.beta,
                sample_count,
                timestamp,
            ),
        )

        # Save element posteriors
        for element in template.elements:
            element_posterior = self._get_element_posterior(task_type, provider, element)
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {Tables.AGENT_PROMPT_ELEMENT}
                (task_type, provider, element, alpha, beta, sample_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_type.lower(),
                    provider.lower(),
                    element.value,
                    element_posterior.alpha,
                    element_posterior.beta,
                    sample_count,
                    timestamp,
                ),
            )

        # Save history
        cursor.execute(
            f"""
            INSERT INTO {Tables.AGENT_PROMPT_HISTORY}
            (task_type, provider, model, template_used, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task_type.lower(),
                provider.lower(),
                model,
                json.dumps(template.to_dict()),
                success,
                timestamp,
            ),
        )

        self.db.commit()

    def get_recommendation(
        self,
        provider: str,
        model: str,
        task_type: str,
        session_id: Optional[str] = None,
    ) -> Optional[RLRecommendation]:
        """Get recommended prompt template for context.

        Uses Thompson Sampling to select style and elements.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type
            session_id: Optional session ID for tracking

        Returns:
            Recommendation with PromptTemplate dictionary
        """
        context_key = self._get_context_key(task_type, provider)
        sample_count = self._sample_counts.get(context_key, 0)

        # Check for pure exploration
        if random.random() < self.exploration_rate:
            template = self._random_template()
            confidence = 0.3
            reason = "Exploration: random template"
        else:
            # Thompson Sampling for style
            style_samples = {}
            for style in self.ALL_STYLES:
                posterior = self._get_style_posterior(task_type, provider, style)
                style_samples[style] = posterior.sample()

            best_style = max(style_samples, key=style_samples.get)

            # Thompson Sampling for elements
            selected_elements = []
            for element in self.ALL_ELEMENTS:
                posterior = self._get_element_posterior(task_type, provider, element)
                if posterior.sample() > 0.5:  # Include if sampled value > threshold
                    selected_elements.append(element)

            # Limit elements to avoid overwhelming prompts
            if len(selected_elements) > 4:
                # Keep highest-sampled elements
                element_samples = {
                    e: self._get_element_posterior(task_type, provider, e).sample()
                    for e in selected_elements
                }
                selected_elements = sorted(element_samples, key=element_samples.get, reverse=True)[
                    :4
                ]

            template = PromptTemplate(
                style=best_style,
                elements=selected_elements,
                sample_count=sample_count,
            )

            # Compute confidence from posterior variance
            best_posterior = self._get_style_posterior(task_type, provider, best_style)
            confidence = self._compute_confidence(best_posterior, sample_count)
            reason = f"Thompson Sampling: {best_style.value} with {len(selected_elements)} elements"

        # Track selection for feedback
        if session_id:
            self._track_selection(session_id, template)

        # Determine if baseline
        is_baseline = sample_count < self.MIN_SAMPLES_FOR_CONFIDENCE

        return RLRecommendation(
            value=template.to_dict(),
            confidence=confidence,
            reason=reason,
            sample_size=sample_count,
            is_baseline=is_baseline,
        )

    def _random_template(self) -> PromptTemplate:
        """Generate a random template for exploration."""
        style = random.choice(self.ALL_STYLES)
        # Random subset of elements (0-4)
        num_elements = random.randint(0, 4)
        elements = random.sample(self.ALL_ELEMENTS, k=num_elements)
        return PromptTemplate(style=style, elements=elements)

    def _compute_confidence(self, posterior: BetaDistribution, sample_count: int) -> float:
        """Compute confidence from posterior and samples.

        Args:
            posterior: Beta posterior for best arm
            sample_count: Total samples for this context

        Returns:
            Confidence score between 0.3 and 0.95
        """
        # Base confidence from sample count
        sample_confidence = 1 - math.exp(-sample_count / 20)

        # Posterior concentration (inverse of variance)
        variance = posterior.variance()
        posterior_confidence = 1 - min(1.0, variance * 4)

        # Combine
        confidence = 0.3 + 0.65 * (0.6 * sample_confidence + 0.4 * posterior_confidence)
        return min(0.95, max(0.3, confidence))

    def _track_selection(self, session_id: str, template: PromptTemplate) -> None:
        """Track template selection for later feedback.

        Args:
            session_id: Session identifier
            template: Selected template
        """
        self._recent_selections[session_id] = template

        # Limit tracked selections
        if len(self._recent_selections) > self._max_tracked_selections:
            # Remove oldest entries
            oldest = list(self._recent_selections.keys())[: self._max_tracked_selections // 2]
            for key in oldest:
                del self._recent_selections[key]

    def get_template(
        self, task_type: str, provider: str, session_id: Optional[str] = None
    ) -> PromptTemplate:
        """Get template for context (convenience method).

        Args:
            task_type: Task type
            provider: Provider name
            session_id: Optional session ID for tracking

        Returns:
            PromptTemplate instance
        """
        rec = self.get_recommendation(provider, "", task_type, session_id)
        if rec and rec.value:
            return PromptTemplate.from_dict(rec.value)
        return self.DEFAULT_TEMPLATES.get(task_type, self.DEFAULT_TEMPLATES["default"])

    def get_style_probabilities(self, task_type: str, provider: str) -> Dict[str, float]:
        """Get posterior mean probabilities for each style.

        Args:
            task_type: Task type
            provider: Provider name

        Returns:
            Dictionary of style -> probability
        """
        return {
            style.value: self._get_style_posterior(task_type, provider, style).mean()
            for style in self.ALL_STYLES
        }

    def get_element_probabilities(self, task_type: str, provider: str) -> Dict[str, float]:
        """Get posterior mean probabilities for each element.

        Args:
            task_type: Task type
            provider: Provider name

        Returns:
            Dictionary of element -> probability
        """
        return {
            element.value: self._get_element_posterior(task_type, provider, element).mean()
            for element in self.ALL_ELEMENTS
        }

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (0 to 1)
        """
        return self._compute_success_score(outcome)

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics for monitoring.

        Returns:
            Dictionary with learner stats
        """
        total_samples = sum(self._sample_counts.values())

        # Get top styles per task type
        top_styles = {}
        seen_contexts = set()

        for key, posterior in self._style_posteriors.items():
            task_type, provider, style = key
            context = f"{task_type}:{provider}"

            if context not in seen_contexts:
                seen_contexts.add(context)
                context_styles = {
                    k[2]: v.mean()
                    for k, v in self._style_posteriors.items()
                    if k[0] == task_type and k[1] == provider
                }
                if context_styles:
                    best = max(context_styles, key=context_styles.get)
                    top_styles[context] = {
                        "best_style": best,
                        "probability": context_styles[best],
                    }

        return {
            "learner": self.name,
            "contexts_learned": len(self._sample_counts),
            "total_samples": total_samples,
            "exploration_rate": self.exploration_rate,
            "style_posteriors_count": len(self._style_posteriors),
            "element_posteriors_count": len(self._element_posteriors),
            "tracked_selections": len(self._recent_selections),
            "top_styles_by_context": top_styles,
            "samples_per_context": {f"{k[0]}:{k[1]}": v for k, v in self._sample_counts.items()},
        }
