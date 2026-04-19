"""Extended ToolSelectorLearner with ToolPredictor and UsageAnalytics integration.

Priority 3 Feature Integration:
- Ensemble prediction (keyword, semantic, co-occurrence)
- Usage analytics for tool insights
- Preloads tool schemas for performance
"""

from typing import Any, Dict, List, Optional, Set

from victor.agent.planning.tool_predictor import ToolPredictor
from victor.agent.usage_analytics import UsageAnalytics
from victor.framework.rl.base import RLOutcome, RLRecommendation
from victor.framework.rl.learners.tool_selector import ToolSelectorLearner


class ExtendedToolSelectorLearner(ToolSelectorLearner):
    """Extend ToolSelectorLearner with ToolPredictor and UsageAnalytics integration.

    Integrates Priority 3's ToolPredictor and UsageAnalytics with the existing
    ToolSelectorLearner to provide predictive tool selection with learning.

    Features:
        - Ensemble tool prediction (keyword, semantic, co-occurrence)
        - Usage analytics for tool insights
        - Learning from tool execution outcomes
        - Tool success rate tracking
    """

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.05,
        provider_adapter: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize extended learner with predictor and analytics.

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

        # Integrate ToolPredictor from Priority 3
        self.predictor = ToolPredictor()

        # Integrate UsageAnalytics (singleton)
        self.analytics = UsageAnalytics.get_instance()

    def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn from tool execution outcomes using predictor and analytics.

        Training process:
        1. Update co-occurrence tracker with tool sequences
        2. Generate recommendations based on tool insights
        3. Learn optimal tool usage patterns

        Args:
            outcomes: List of tool execution outcomes

        Returns:
            List of recommendations for tool usage
        """
        recommendations = []

        # Train predictor with new outcomes
        for outcome in outcomes:
            if outcome.task_type == "tool_execution":
                # Update co-occurrence tracker
                tools_used = outcome.metadata.get("tools_used", [])
                if tools_used:
                    self.predictor.cooccurrence_tracker.record_tool_sequence(
                        tools=tools_used,
                        task_type=outcome.metadata.get("task_type", "unknown"),
                        success=outcome.success,
                    )

        # Generate recommendations using analytics
        tool_names = self._get_tool_names(outcomes)

        for tool_name in tool_names:
            # Get insights from UsageAnalytics
            insights = self.analytics.get_tool_insights(tool_name)

            # Create recommendation based on success rate
            if insights["success_rate"] > 0.7:
                # High success rate - recommend using
                recommendations.append(
                    RLRecommendation(
                        learner_name="tool_selector",
                        recommendation_type="tool_usage",
                        key=tool_name,
                        value="use",
                        confidence=insights["success_rate"],
                        metadata={
                            "avg_execution_ms": insights["avg_execution_ms"],
                            "sample_size": insights["execution_count"],
                            "reason": "high_success_rate",
                        },
                    )
                )
            elif insights["success_rate"] < 0.3:
                # Low success rate - recommend avoiding
                recommendations.append(
                    RLRecommendation(
                        learner_name="tool_selector",
                        recommendation_type="tool_usage",
                        key=tool_name,
                        value="avoid",
                        confidence=1.0 - insights["success_rate"],
                        metadata={
                            "avg_execution_ms": insights["avg_execution_ms"],
                            "sample_size": insights["execution_count"],
                            "reason": "low_success_rate",
                        },
                    )
                )

        return recommendations

    def predict_next_tool(
        self, task_description: str, current_step: str, recent_tools: List[str], task_type: str
    ) -> Optional[str]:
        """Predict next tool using ToolPredictor.

        Uses ensemble prediction combining:
        - Keyword matching (30% weight)
        - Semantic similarity (40% weight)
        - Co-occurrence patterns (20% weight)
        - Success rate multiplier (10% weight)

        Args:
            task_description: Description of the task
            current_step: Current step (e.g., "exploration", "planning")
            recent_tools: Recently used tools
            task_type: Type of task (e.g., "bugfix", "feature")

        Returns:
            Predicted tool name, or None if no prediction
        """
        predictions = self.predictor.predict_tools(
            task_description=task_description,
            current_step=current_step,
            recent_tools=recent_tools,
            task_type=task_type,
        )

        # Return top prediction
        return predictions[0].tool_name if predictions else None

    def get_tool_insights(self, tool_name: str) -> Dict[str, Any]:
        """Get insights for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool insights including success rate, execution time, etc.
        """
        return self.analytics.get_tool_insights(tool_name)

    def get_predictor_statistics(self) -> Dict[str, Any]:
        """Get predictor and analytics statistics.

        Returns:
            Statistics from predictor and analytics
        """
        return {
            "predictor": self.predictor.get_statistics(),
            "analytics": self.analytics.get_session_summary(),
        }

    def _get_tool_names(self, outcomes: List[RLOutcome]) -> Set[str]:
        """Extract unique tool names from outcomes.

        Args:
            outcomes: List of outcomes

        Returns:
            Set of unique tool names
        """
        tool_names = set()

        for outcome in outcomes:
            if outcome.task_type == "tool_execution":
                tool_name = outcome.metadata.get("tool_name")
                if tool_name:
                    tool_names.add(tool_name)

                # Also check tools_used list
                tools_used = outcome.metadata.get("tools_used", [])
                tool_names.update(tools_used)

        return tool_names
