#!/usr/bin/env python
"""Event-Driven Architecture Usage Example.

This example shows how to:
1. Define domain events
2. Publish events to an event bus
3. Subscribe to events with handlers
4. Filter events by topic
5. Correlate events across the system

Before Refactoring (Tight Coupling):
    class Orchestrator:
        def complete_evaluation(self, result):
            evaluation_coordinator.process_result(result)  # Direct call
            rl_coordinator.update(result)  # Direct call
            analytics_coordinator.record(result)  # Direct call

    # Tight coupling, hard to extend, circular dependencies

After Refactoring (Event-Driven):
    class Orchestrator:
        def complete_evaluation(self, result):
            event = EvaluationCompletedEvent(...)
            bus.emit(event)  # Publish and forget

    # Loose coupling, easy to extend, no circular dependencies
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: Define Event Bus (Simplified Implementation)
# =============================================================================


class EventBus:
    """Simple in-memory event bus for demonstration.

    In production, Victor uses a more sophisticated event bus with:
    - Async event handling
    - Event persistence
    - Dead letter queues
    - Circuit breakers
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Dict[str, Any]] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Subscribe to events on a topic.

        Args:
            topic: Event topic to subscribe to (e.g., "evaluation.completed")
            handler: Function to call when event is published
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)
        logger.info(f"Subscribed to topic: {topic}")

    def subscribe_with_filter(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], None],
        filter_fn: Callable[[Dict[str, Any]], bool],
    ) -> None:
        """Subscribe with event filtering.

        Args:
            topic: Event topic to subscribe to
            handler: Function to call when event is published
            filter_fn: Function to filter events (return True to handle)
        """
        def filtered_handler(event_data: Dict[str, Any]) -> None:
            if filter_fn(event_data):
                handler(event_data)

        self.subscribe(topic, filtered_handler)

    def emit(self, topic: str, data: Dict[str, Any]) -> None:
        """Publish an event to a topic.

        Args:
            topic: Event topic
            data: Event data
        """
        event = {
            "topic": topic,
            "data": data,
            "timestamp": time.time(),
        }
        self._event_history.append(event)

        logger.info(f"Event published: {topic}")

        # Notify subscribers
        if topic in self._subscribers:
            for handler in self._subscribers[topic]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")

    def get_history(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get event history.

        Args:
            topic: Optional topic filter

        Returns:
            List of events
        """
        if topic:
            return [e for e in self._event_history if e["topic"] == topic]
        return self._event_history


# =============================================================================
# PART 2: Define Domain Events
# =============================================================================


class EvaluationEventType(str, Enum):
    """Types of evaluation events."""

    EVALUATION_COMPLETED = "evaluation.completed"
    EVALUATION_FAILED = "evaluation.failed"
    RL_FEEDBACK_GENERATED = "evaluation.rl_feedback"
    ANALYTICS_FLUSHED = "evaluation.analytics_flushed"


@dataclass
class EvaluationCompletedEvent:
    """Event published when evaluation cycle completes.

    This event breaks the circular dependency between orchestrator and
    evaluation coordinator by providing evaluation results via events
    instead of direct method calls.
    """

    success: bool
    quality_score: float
    tool_calls_used: int
    tokens_used: Dict[str, int]
    session_id: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for event bus."""
        return {
            "success": self.success,
            "quality_score": self.quality_score,
            "tool_calls_used": self.tool_calls_used,
            "tokens_used": self.tokens_used,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }


@dataclass
class RLFeedbackEvent:
    """Event published when RL feedback signal is generated.

    This event breaks circular dependency by providing RL feedback via
    events instead of direct method calls to RL coordinator.
    """

    session_id: str
    reward: float
    outcome_type: str
    provider: str
    model: str
    latency_ms: int
    tool_calls: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for event bus."""
        return {
            "session_id": self.session_id,
            "reward": self.reward,
            "outcome_type": self.outcome_type,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp,
        }


# =============================================================================
# PART 3: Define Event Handlers
# =============================================================================


class AnalyticsListener:
    """Listens to evaluation events and records analytics.

    This class demonstrates how to decouple components using events.
    Instead of the orchestrator calling analytics directly, the
    analytics system listens to evaluation events.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._records: List[Dict[str, Any]] = []

        # Subscribe to evaluation events
        self.event_bus.subscribe(
            EvaluationEventType.EVALUATION_COMPLETED.value,
            self._on_evaluation_completed,
        )

        # Subscribe to RL feedback events
        self.event_bus.subscribe(
            EvaluationEventType.RL_FEEDBACK_GENERATED.value,
            self._on_rl_feedback,
        )

    def _on_evaluation_completed(self, event: Dict[str, Any]) -> None:
        """Handle evaluation completed event."""
        data = event["data"]
        logger.info(f"[Analytics] Recording evaluation: {data['session_id']}")

        record = {
            "type": "evaluation",
            "session_id": data["session_id"],
            "success": data["success"],
            "quality_score": data["quality_score"],
            "tool_calls": data["tool_calls_used"],
            "timestamp": data["timestamp"],
        }
        self._records.append(record)

    def _on_rl_feedback(self, event: Dict[str, Any]) -> None:
        """Handle RL feedback event."""
        data = event["data"]
        logger.info(f"[Analytics] Recording RL feedback: {data['session_id']}")

        record = {
            "type": "rl_feedback",
            "session_id": data["session_id"],
            "reward": data["reward"],
            "outcome": data["outcome_type"],
            "timestamp": data["timestamp"],
        }
        self._records.append(record)

    def get_records(self) -> List[Dict[str, Any]]:
        """Get all recorded analytics."""
        return self._records

    def get_average_quality(self) -> float:
        """Calculate average quality score."""
        eval_records = [r for r in self._records if r["type"] == "evaluation"]
        if not eval_records:
            return 0.0
        total = sum(r["quality_score"] for r in eval_records)
        return total / len(eval_records)


class RLListener:
    """Listens to events and updates reinforcement learning model.

    Demonstrates how RL system can observe system events without
    tight coupling to the orchestrator or evaluation coordinator.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._outcomes: List[Dict[str, Any]] = []

        # Subscribe to RL feedback events
        self.event_bus.subscribe(
            EvaluationEventType.RL_FEEDBACK_GENERATED.value,
            self._on_rl_feedback,
        )

    def _on_rl_feedback(self, event: Dict[str, Any]) -> None:
        """Handle RL feedback event."""
        data = event["data"]
        logger.info(f"[RL] Updating model with feedback: reward={data['reward']}")

        outcome = {
            "session_id": data["session_id"],
            "reward": data["reward"],
            "outcome_type": data["outcome_type"],
            "provider": data["provider"],
            "model": data["model"],
            "timestamp": data["timestamp"],
        }
        self._outcomes.append(outcome)

        # Simulate model update
        self._update_model(outcome)

    def _update_model(self, outcome: Dict[str, Any]) -> None:
        """Simulate updating RL model."""
        logger.info(f"[RL] Model updated with outcome: {outcome['outcome_type']}")

    def get_total_reward(self) -> float:
        """Calculate total reward across all outcomes."""
        return sum(o["reward"] for o in self._outcomes)


class QualityMonitor:
    """Monitors evaluation quality and alerts on issues.

    Demonstrates event filtering - only alerts on low-quality evaluations.
    """

    def __init__(self, event_bus: EventBus, min_quality: float = 0.5):
        self.event_bus = event_bus
        self.min_quality = min_quality
        self._alerts: List[Dict[str, Any]] = []

        # Subscribe with filter - only alert on low quality
        self.event_bus.subscribe_with_filter(
            EvaluationEventType.EVALUATION_COMPLETED.value,
            self._on_low_quality,
            filter_fn=lambda e: e["data"]["quality_score"] < self.min_quality,
        )

    def _on_low_quality(self, event: Dict[str, Any]) -> None:
        """Handle low quality evaluation."""
        data = event["data"]
        logger.warning(
            f"[QualityMonitor] Low quality detected: {data['quality_score']:.2f}"
        )

        alert = {
            "session_id": data["session_id"],
            "quality_score": data["quality_score"],
            "threshold": self.min_quality,
            "timestamp": data["timestamp"],
        }
        self._alerts.append(alert)

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get all quality alerts."""
        return self._alerts


class SessionTracker:
    """Tracks evaluation events by session ID.

    Demonstrates event correlation - tracking related events.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

        # Subscribe to all evaluation events
        for event_type in EvaluationEventType:
            self.event_bus.subscribe(
                event_type.value,
                self._on_evaluation_event,
            )

    def _on_evaluation_event(self, event: Dict[str, Any]) -> None:
        """Track evaluation events by session."""
        data = event["data"]
        session_id = data.get("session_id", "unknown")

        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append({
            "topic": event["topic"],
            "data": data,
            "timestamp": event["timestamp"],
        })

        logger.info(f"[SessionTracker] Tracking event for session: {session_id}")

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all events for a specific session."""
        return self._sessions.get(session_id, [])


# =============================================================================
# PART 4: Publisher (Orchestrator-like Component)
# =============================================================================


class EvaluationOrchestrator:
    """Orchestrator that publishes evaluation events.

    This is a simplified orchestrator that demonstrates publishing
    events instead of making direct calls to other components.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._session_counter = 0

    def run_evaluation(
        self,
        task: str,
        provider: str,
        model: str,
    ) -> Dict[str, Any]:
        """Run an evaluation and publish events.

        Instead of calling analytics, RL, and other systems directly,
        it publishes events that those systems can listen to.
        """
        self._session_counter += 1
        session_id = f"session_{self._session_counter}"

        logger.info(f"\n[Evaluation] Running evaluation for task: {task}")

        # Simulate evaluation
        start_time = time.time()
        time.sleep(0.1)  # Simulate work

        # Generate random results
        import random
        success = random.choice([True, True, False])  # 66% success rate
        quality_score = random.uniform(0.3, 0.95)
        tool_calls_used = random.randint(3, 15)
        tokens_used = {"input": random.randint(500, 2000), "output": random.randint(200, 1000)}

        latency_ms = int((time.time() - start_time) * 1000)

        # Publish evaluation completed event
        eval_event = EvaluationCompletedEvent(
            success=success,
            quality_score=quality_score,
            tool_calls_used=tool_calls_used,
            tokens_used=tokens_used,
            session_id=session_id,
        )

        self.event_bus.emit(
            EvaluationEventType.EVALUATION_COMPLETED.value,
            eval_event.to_dict(),
        )

        # Publish RL feedback event
        reward = 1.0 if success and quality_score > 0.7 else -0.5
        outcome_type = "success" if success else "failure"

        rl_event = RLFeedbackEvent(
            session_id=session_id,
            reward=reward,
            outcome_type=outcome_type,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            tool_calls=tool_calls_used,
        )

        self.event_bus.emit(
            EvaluationEventType.RL_FEEDBACK_GENERATED.value,
            rl_event.to_dict(),
        )

        return {
            "session_id": session_id,
            "success": success,
            "quality_score": quality_score,
            "tool_calls": tool_calls_used,
        }


# =============================================================================
# PART 5: Demonstration Functions
# =============================================================================


def demonstrate_event_publishing():
    """Demonstrate basic event publishing."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Event Publishing")
    logger.info("=" * 70)

    bus = EventBus()

    # Create listener
    analytics = AnalyticsListener(bus)

    # Create orchestrator
    orchestrator = EvaluationOrchestrator(bus)

    # Run evaluations
    logger.info("\n--- Running evaluations ---")
    orchestrator.run_evaluation(
        task="Refactor authentication code",
        provider="anthropic",
        model="claude-sonnet-4-5",
    )

    orchestrator.run_evaluation(
        task="Generate unit tests",
        provider="openai",
        model="gpt-4",
    )

    # Show analytics
    records = analytics.get_records()
    logger.info(f"\n--- Analytics captured {len(records)} records ---")
    for record in records:
        logger.info(f"  {record['type']}: {record['session_id']}")

    avg_quality = analytics.get_average_quality()
    logger.info(f"\nAverage quality score: {avg_quality:.2f}")


def demonstrate_event_filtering():
    """Demonstrate event filtering."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Event Filtering")
    logger.info("=" * 70)

    bus = EventBus()

    # Create quality monitor with filter
    monitor = QualityMonitor(bus, min_quality=0.6)

    # Create orchestrator
    orchestrator = EvaluationOrchestrator(bus)

    # Run multiple evaluations
    logger.info("\n--- Running evaluations ---")
    for i in range(5):
        orchestrator.run_evaluation(
            task=f"Test task {i}",
            provider="anthropic",
            model="claude-sonnet-4-5",
        )

    # Show alerts
    alerts = monitor.get_alerts()
    logger.info(f"\n--- Quality alerts: {len(alerts)} ---")
    for alert in alerts:
        logger.info(
            f"  Session {alert['session_id']}: "
            f"quality={alert['quality_score']:.2f} "
            f"(threshold={alert['threshold']:.2f})"
        )


def demonstrate_event_correlation():
    """Demonstrate event correlation by session."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Event Correlation")
    logger.info("=" * 70)

    bus = EventBus()

    # Create session tracker
    tracker = SessionTracker(bus)

    # Create orchestrator
    orchestrator = EvaluationOrchestrator(bus)

    # Run evaluation
    result = orchestrator.run_evaluation(
        task="Debug error in authentication",
        provider="anthropic",
        model="claude-sonnet-4-5",
    )

    session_id = result["session_id"]

    # Show correlated events
    events = tracker.get_session_events(session_id)
    logger.info(f"\n--- Events for session {session_id} ---")
    for event in events:
        logger.info(f"  {event['topic']}: {event['data']}")


def demonstrate_decoupling():
    """Demonstrate how events enable loose coupling."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Loose Coupling with Events")
    logger.info("=" * 70)

    bus = EventBus()

    # Create all listeners (independent components)
    analytics = AnalyticsListener(bus)
    rl_listener = RLListener(bus)
    monitor = QualityMonitor(bus, min_quality=0.5)
    tracker = SessionTracker(bus)

    # Create orchestrator
    orchestrator = EvaluationOrchestrator(bus)

    logger.info("\n--- Orchestrator publishes events ---")
    logger.info("--- Components react independently ---\n")

    # Run evaluations
    orchestrator.run_evaluation(
        task="Refactor code",
        provider="anthropic",
        model="claude-sonnet-4-5",
    )

    # Show how each component reacted
    logger.info("\n--- Component Reactions ---")
    logger.info(f"Analytics recorded: {len(analytics.get_records())} events")
    logger.info(f"RL system updated: {len(rl_listener._outcomes)} outcomes")
    logger.info(f"Quality monitor alerts: {len(monitor.get_alerts())} alerts")
    logger.info(f"Session tracker tracking: {len(tracker._sessions)} sessions")

    logger.info("\n✓ Orchestrator doesn't know about these components!")
    logger.info("✓ Components can be added/removed without changing orchestrator!")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 70)
    logger.info("EVENT-DRIVEN ARCHITECTURE USAGE")
    logger.info("=" * 70)

    # Basic publishing
    demonstrate_event_publishing()

    # Event filtering
    demonstrate_event_filtering()

    # Event correlation
    demonstrate_event_correlation()

    # Loose coupling
    demonstrate_decoupling()

    logger.info("\n" + "=" * 70)
    logger.info("KEY TAKEAWAYS")
    logger.info("=" * 70)
    logger.info("1. Events break circular dependencies")
    logger.info("2. Components subscribe to topics they care about")
    logger.info("3. Event filtering enables selective handling")
    logger.info("4. Event correlation tracks related events")
    logger.info("5. New components can be added without changes")
    logger.info("6. Easy to test by substituting event handlers")
    logger.info("\nRun with: python -m examples.architecture.event_usage")


if __name__ == "__main__":
    main()
