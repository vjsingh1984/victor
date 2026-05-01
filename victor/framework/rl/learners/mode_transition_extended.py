"""Extended ModeTransitionLearner with PhaseDetector integration.

Priority 2 Feature Integration:
- Detects task phase (EXPLORATION, PLANNING, EXECUTION, REVIEW)
- Manages phase transitions with cooldown
- Phase-aware context scoring
"""

from typing import Any, Dict, List, Optional

from victor.agent.context_phase_detector import PhaseDetector, PhaseTransitionDetector
from victor.agent.conversation.state_machine import ConversationStage
from victor.core.shared_types import TaskPhase
from victor.framework.rl.base import RLOutcome, RLRecommendation
from victor.framework.rl.learners.mode_transition import ModeTransitionLearner


class ExtendedModeTransitionLearner(ModeTransitionLearner):
    """Extend ModeTransitionLearner with PhaseDetector integration.

    Integrates Priority 2's PhaseDetector and PhaseTransitionDetector with
    the existing ModeTransitionLearner to provide phase-aware conversation
    management.

    Features:
        - Phase detection (EXPLORATION, PLANNING, EXECUTION, REVIEW)
        - Phase transition management with cooldown
        - Phase-aware context scoring
        - Learning from phase transitions
    """

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize extended learner with phase detector.

        Args:
            name: Learner name
            db_connection: Database connection
            learning_rate: Learning rate for Q-learning
            provider_adapter: Optional provider adapter
            **kwargs: Additional parameters passed to base class
        """
        # Initialize base class
        super().__init__(
            name=name,
            db_connection=db_connection,
            learning_rate=learning_rate,
            provider_adapter=provider_adapter,
            **kwargs,
        )

        # Integrate PhaseDetector from Priority 2
        self.phase_detector = PhaseDetector()
        self.transition_detector = PhaseTransitionDetector()

    @staticmethod
    def _make_recommendation(
        *,
        key: str,
        value: str,
        confidence: float,
        reason: str,
        sample_size: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RLRecommendation:
        payload = dict(metadata or {})
        payload.update(
            {
                "learner_name": "mode_transition",
                "recommendation_type": "phase_transition",
                "key": key,
            }
        )
        return RLRecommendation(
            value=value,
            confidence=confidence,
            reason=reason,
            sample_size=sample_size,
            metadata=payload,
        )

    def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn from mode transitions using phase detection.

        Tracks:
        - Phase transition patterns
        - Transition success rates
        - Optimal transition timing

        Args:
            outcomes: List of mode transition outcomes

        Returns:
            List of recommendations for transition optimization
        """
        recommendations = []

        for outcome in outcomes:
            # Get phase information
            detected_phase = outcome.metadata.get("detected_phase")
            transition_successful = outcome.metadata.get("transition_successful", True)
            from_phase = outcome.metadata.get("from_phase")
            to_phase = outcome.metadata.get("to_phase")

            # Learn from successful transitions
            if detected_phase and transition_successful:
                # Reinforce successful phase pattern
                recommendations.append(
                    self._make_recommendation(
                        key=f"to_{detected_phase}",
                        value="allow",
                        confidence=0.9,
                        reason=f"Transition to {detected_phase} succeeded",
                        sample_size=1,
                        metadata={
                            "detected_phase": detected_phase,
                            "from_phase": from_phase,
                            "to_phase": to_phase,
                            "transition_successful": transition_successful,
                        },
                    )
                )

            # Learn from failed transitions
            elif detected_phase and not transition_successful:
                # Discourage problematic transition
                recommendations.append(
                    self._make_recommendation(
                        key=f"to_{detected_phase}",
                        value="avoid",
                        confidence=0.7,
                        reason=f"Transition to {detected_phase} failed",
                        sample_size=1,
                        metadata={
                            "reason": "transition_failed",
                            "detected_phase": detected_phase,
                            "from_phase": from_phase,
                        },
                    )
                )

        return recommendations

    def detect_phase(
        self, current_stage: ConversationStage, recent_tools: List[str], message_content: str
    ) -> TaskPhase:
        """Detect current task phase using PhaseDetector.

        Args:
            current_stage: Current conversation stage
            recent_tools: Recently used tools
            message_content: Current message content

        Returns:
            Detected task phase
        """
        return self.phase_detector.detect_phase(
            current_stage=current_stage, recent_tools=recent_tools, message_content=message_content
        )

    def should_transition(self, current_phase: TaskPhase, new_phase: TaskPhase) -> bool:
        """Check if phase transition should be allowed.

        Uses PhaseTransitionDetector to enforce:
        - Cooldown between transitions
        - Thrashing prevention
        - Valid transition paths

        Args:
            current_phase: Current task phase
            new_phase: Desired new phase

        Returns:
            True if transition should be allowed
        """
        if self.transition_detector.get_current_phase() != current_phase:
            self.transition_detector._current_phase = current_phase
            self.transition_detector._last_transition_time = 0.0

        return self.transition_detector.should_transition(new_phase)

    def get_phase_statistics(self) -> Dict[str, Any]:
        """Get phase detection statistics.

        Returns:
            Statistics about phase transitions and detections
        """
        return {
            "transition_detector": {
                "last_transition_time": self.transition_detector._last_transition_time,
                "transition_count": self.transition_detector._transition_count,
            },
            "phase_detector": {
                # Add phase detector stats if available
            },
        }
