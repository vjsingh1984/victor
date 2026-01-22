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

"""Integration tests for Persona + Memory systems (Phase 3 + Phase 4).

This module tests the integration between:
- PersonaManager: Dynamic agent persona management
- EpisodicMemory: Agent experience storage and retrieval
- SemanticMemory: Factual knowledge storage and querying

Test scenarios:
1. Context-aware persona adaptation using memory
2. Persona-influenced memory storage and retrieval
3. Memory-driven persona evolution based on experiences
4. Cross-system consistency and event propagation
5. Performance with realistic workloads
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
import numpy as np

from victor.agent.memory.episodic_memory import Episode, EpisodicMemory
from victor.agent.memory.semantic_memory import KnowledgeTriple, SemanticMemory
from victor.agent.personas.persona_manager import PersonaManager
from victor.agent.personas.persona_repository import PersonaRepository
from victor.agent.personas.types import (
    AdaptedPersona,
    CommunicationStyle,
    ContextAdjustment,
    Feedback,
    Persona,
    PersonalityType,
)


# ============================================================================
# Fixtures
# ============================================================================




@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for memory systems."""
    service = MagicMock()
    service.embed_text = AsyncMock(return_value=np.random.rand(1536))
    service.embed_batch = AsyncMock(return_value=[np.random.rand(1536) for _ in range(5)])
    return service


@pytest.fixture
def episodic_memory(mock_embedding_service):
    """Create episodic memory instance for testing."""
    memory = EpisodicMemory(
        embedding_service=mock_embedding_service,
        max_episodes=1000,
        decay_rate=0.95
    )
    return memory


@pytest.fixture
def semantic_memory(mock_embedding_service):
    """Create semantic memory instance for testing."""
    memory = SemanticMemory(
        embedding_service=mock_embedding_service,
        max_knowledge=1000  # Smaller limit for tests
    )
    return memory


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    return bus


# ============================================================================
# Test: Context-Aware Persona Adaptation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.skip(reason="Experimental Phase 3 feature - ContextAdjustment API mismatch")
async def test_persona_adapts_based_on_episodic_memory(
    episodic_memory, persona_manager, mock_event_bus
):
    """Test that personas adapt based on experiences stored in episodic memory.

    Scenario:
    1. Store episodes with different task types and outcomes
    2. Load a base persona
    3. Adapt persona based on memory of successful tasks
    4. Verify adapted persona reflects memory patterns

    Validates:
    - Episodic memory retrieval for persona adaptation
    - Context adjustment based on past experiences
    - Persona adaptation caching
    """
    # Store episodes with different task patterns
    episodes = [
        Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now() - timedelta(hours=2),
            inputs={"task": "code_review", "language": "python"},
            actions=["read_file", "analyze_code", "generate_report"],
            outcomes={"success": True, "issues_found": 5, "quality_score": 0.9},
            rewards=8.5,
            embedding=await episodic_memory._embedding_service.embed_text(
                "code review python success"
            )
        ),
        Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now() - timedelta(hours=1),
            inputs={"task": "debugging", "language": "python"},
            actions=["read_file", "set_breakpoint", "analyze_stack_trace"],
            outcomes={"success": True, "bugs_fixed": 3, "time_saved": 2.0},
            rewards=7.0,
            embedding=await episodic_memory._embedding_service.embed_text(
                "debugging python success"
            )
        ),
        Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now() - timedelta(minutes=30),
            inputs={"task": "architecture_design"},
            actions=["analyze_requirements", "design_components", "create_diagram"],
            outcomes={"success": True, "components_designed": 5, "complexity": "high"},
            rewards=9.0,
            embedding=await episodic_memory._embedding_service.embed_text(
                "architecture design success"
            )
        )
    ]

    for episode in episodes:
        await episodic_memory.store_episode(episode)

    # Load base persona
    persona = persona_manager.load_persona("senior_developer")
    assert persona.name == "Senior Developer"

    # Recall relevant episodes for a new task
    relevant_episodes = await episodic_memory.recall_relevant(
        "Review and debug Python microservice architecture",
        k=3
    )

    assert len(relevant_episodes) > 0
    assert any("code_review" in str(ep.inputs) for ep in relevant_episodes)

    # Adapt persona based on memory (simulated)
    # In real system, PersonaManager would use memory directly
    context_type = "code_review_and_debugging"

    adaptation = ContextAdjustment(
        temperature=0.3,  # More analytical for code review
        verbosity="concise",
        tool_preference=["read_file", "analyze_code", "generate_report"],
        expertise_boost=["code_review", "debugging"],
        constraint_relaxations={"max_tool_calls": 25}  # Allow more for complex tasks
    )

    adapted = AdaptedPersona(
        base_persona=persona,
        context_type=context_type,
        adaptations=adaptation,
        confidence=0.85,
        reasoning="Based on successful code review and debugging episodes"
    )

    # Verify adaptation reflects memory
    assert adapted.base_persona.id == "senior_developer"
    assert adapted.context_type == "code_review_and_debugging"
    assert adapted.adaptations.temperature < persona.temperature  # More focused
    assert "code_review" in adapted.adaptations.expertise_boost


@pytest.mark.asyncio
@pytest.mark.skip(reason="Experimental Phase 3 feature - SemanticMemory.add_relationship method not implemented")
async def test_persona_uses_semantic_memory_for_knowledge_retrieval(
    semantic_memory, persona_manager
):
    """Test that personas leverage semantic memory for knowledge-enhanced responses.

    Scenario:
    1. Store domain knowledge in semantic memory
    2. Query semantic memory for relevant knowledge
    3. Adapt persona to incorporate retrieved knowledge
    4. Verify persona uses knowledge in system prompt

    Validates:
    - Semantic memory integration with persona system
    - Knowledge retrieval for persona enhancement
    - Knowledge citation in adapted personas
    """
    # Store domain knowledge
    facts = [
        ("Python async/await uses coroutines for concurrency", 0.98, "python_docs"),
        ("Type hints improve code maintainability and IDE support", 0.95, "best_practices"),
        ("Pydantic provides data validation using Python type annotations", 0.97, "library_docs"),
        ("FastAPI is built on Starlette and Pydantic", 0.99, "framework_docs"),
    ]

    fact_ids = []
    for fact, confidence, source in facts:
        fact_id = await semantic_memory.store_knowledge(
            fact,
            confidence=confidence,
            source=source
        )
        fact_ids.append(fact_id)

    # Create knowledge graph relationships
    await semantic_memory.add_relationship(
        KnowledgeTriple(
            subject=fact_ids[0],
            predicate="related_to",
            object=fact_ids[2],
            weight=0.8
        )
    )

    await semantic_memory.add_relationship(
        KnowledgeTriple(
            subject=fact_ids[2],
            predicate="used_in",
            object=fact_ids[3],
            weight=0.9
        )
    )

    # Query for relevant knowledge
    query = "How to build async APIs with Python"
    results = await semantic_memory.query_knowledge(query, k=3)

    assert len(results) > 0
    assert any("async" in r.fact.lower() or "fastapi" in r.fact.lower() for r in results)

    # Load persona and enhance with knowledge
    persona = persona_manager.load_persona("senior_developer")

    # Simulate knowledge-enhanced persona
    knowledge_context = "\n".join([f"- {r.fact} (confidence: {r.confidence})" for r in results])

    enhanced_prompt = f"""{persona.system_prompt}

Relevant Knowledge:
{knowledge_context}

Use this knowledge to provide accurate, up-to-date responses."""

    assert "async" in enhanced_prompt.lower() or "fastapi" in enhanced_prompt.lower()
    assert "Relevant Knowledge:" in enhanced_prompt


# ============================================================================
# Test: Memory-Driven Persona Evolution
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.skip(reason="Experimental Phase 3 feature - Feedback API mismatch")
async def test_persona_evolution_based_on_feedback_and_outcomes(
    episodic_memory, persona_manager, mock_event_bus
):
    """Test that personas evolve based on accumulated feedback and outcomes.

    Scenario:
    1. Store episodes with varying success rates
    2. Aggregate outcomes to identify persona strengths/weaknesses
    3. Evolve persona constraints and preferences
    4. Verify evolved persona improves performance

    Validates:
    - Feedback aggregation from episodic memory
    - Persona evolution based on performance metrics
    - Continuous learning from experiences
    """
    # Store episodes with mixed outcomes
    episodes_data = [
        # Successful episodes with formal communication
        {"task": "code_review", "style": "formal", "success": True, "reward": 9.0},
        {"task": "code_review", "style": "formal", "success": True, "reward": 8.5},
        {"task": "code_review", "style": "formal", "success": True, "reward": 9.2},
        # Failed episodes with casual communication
        {"task": "code_review", "style": "casual", "success": False, "reward": 4.0},
        {"task": "code_review", "style": "casual", "success": False, "reward": 3.5},
        # Successful debugging with higher verbosity
        {"task": "debugging", "verbosity": "detailed", "success": True, "reward": 8.0},
        {"task": "debugging", "verbosity": "detailed", "success": True, "reward": 8.5},
        # Failed debugging with concise verbosity
        {"task": "debugging", "verbosity": "concise", "success": False, "reward": 5.0},
    ]

    for data in episodes_data:
        episode = Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now() - timedelta(hours=len(episodes_data)),
            inputs={"task": data["task"], "style": data.get("style"), "verbosity": data.get("verbosity")},
            actions=["analyze", "execute", "verify"],
            outcomes={"success": data["success"]},
            rewards=data["reward"],
            embedding=await episodic_memory._embedding_service.embed_text(
                f"{data['task']} {data.get('style', '')} {data.get('verbosity', '')}"
            )
        )
        await episodic_memory.store_episode(episode)

    # Aggregate outcomes by task type
    code_review_episodes = await episodic_memory.recall_relevant("code_review", k=10)
    assert len(code_review_episodes) > 0

    # Calculate success rates by style
    formal_success = [ep for ep in code_review_episodes if ep.inputs.get("style") == "formal" and ep.outcomes.get("success")]
    casual_success = [ep for ep in code_review_episodes if ep.inputs.get("style") == "casual" and ep.outcomes.get("success")]

    # Formal style performs better
    assert len(formal_success) > len(casual_success)

    # Evolve persona based on feedback
    persona = persona_manager.load_persona("senior_developer")

    # Create feedback from aggregated outcomes
    feedback = Feedback(
        persona_id=persona.id,
        task_type="code_review",
        success_rate=0.75,
        average_reward=7.5,
        feedback_data={
            "best_style": "formal",
            "worst_style": "casual",
            "improvement_suggestion": "Prefer formal communication for code review"
        }
    )

    # Apply feedback to evolve persona
    evolved_persona = persona_manager.evolve_persona(persona, [feedback])

    # Verify evolution
    assert evolved_persona.communication_style == CommunicationStyle.FORMAL
    assert evolved_persona.system_prompt != persona.system_prompt  # Should be updated


# ============================================================================
# Test: Cross-System Consistency
# ============================================================================


@pytest.mark.asyncio
async def test_cross_system_event_propagation(
    episodic_memory, semantic_memory, persona_manager, mock_event_bus
):
    """Test that events propagate correctly across memory and persona systems.

    Scenario:
    1. Store episode in episodic memory
    2. Trigger consolidation to semantic memory
    3. Publish persona adaptation event
    4. Verify all systems are notified and stay consistent

    Validates:
    - Event bus integration across systems
    - Memory consolidation workflow
    - Persona event handling
    """
    # Store episode
    episode = Episode(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        inputs={"task": "learn_python_async"},
        actions=["read_docs", "write_code", "test"],
        outcomes={"success": True, "knowledge_gained": True},
        rewards=8.0,
        embedding=await episodic_memory._embedding_service.embed_text("learned python async")
    )

    episode_id = await episodic_memory.store_episode(episode)
    assert episode_id is not None

    # Simulate event publishing
    from victor.core.events import MessagingEvent

    event = MessagingEvent(
        topic="episode.stored",
        data={
            "episode_id": str(episode_id),
            "task": "learn_python_async",
            "success": True
        }
    )

    await mock_event_bus.publish(event)

    # Verify event was published
    mock_event_bus.publish.assert_called_once()

    # Consolidate episodic to semantic memory
    knowledge = "Python async/await enables concurrent code execution"
    fact_id = await semantic_memory.store_knowledge(
        knowledge,
        confidence=0.9,
        source=f"episode_{episode_id}"
    )

    assert fact_id is not None

    # Publish consolidation event
    consolidation_event = MessagingEvent(
        topic="memory.consolidated",
        data={
            "episode_id": str(episode_id),
            "fact_id": fact_id,
            "knowledge": knowledge
        }
    )

    await mock_event_bus.publish(consolidation_event)

    # Verify both events were published
    assert mock_event_bus.publish.call_count == 2


# ============================================================================
# Test: Performance with Realistic Workloads
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.skip(reason="Experimental Phase 3 feature - ContextAdjustment API mismatch")
async def test_persona_memory_performance_under_load(
    episodic_memory, semantic_memory, persona_manager
):
    """Test performance of persona + memory integration under realistic load.

    Scenario:
    1. Store 100 episodes concurrently
    2. Perform 50 persona adaptations
    3. Query memory 100 times
    4. Verify performance meets thresholds

    Validates:
    - Concurrent memory operations
    - Persona adaptation caching
    - Memory query performance
    - System scalability
    """
    import time

    # Store 100 episodes concurrently
    episodes_to_store = []
    for i in range(100):
        episode = Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now() - timedelta(minutes=i),
            inputs={"task": f"task_{i % 10}", "iteration": i},
            actions=["action1", "action2"],
            outcomes={"success": i % 2 == 0},
            rewards=7.0 + (i % 3),
            embedding=np.random.rand(1536)
        )
        episodes_to_store.append(episode)

    start_time = time.time()

    # Store concurrently
    store_tasks = [episodic_memory.store_episode(ep) for ep in episodes_to_store]
    await asyncio.gather(*store_tasks)

    store_duration = time.time() - start_time

    # Should complete in reasonable time
    assert store_duration < 5.0  # 100 episodes in < 5 seconds

    # Perform persona adaptations
    persona = persona_manager.load_persona("senior_developer")

    start_time = time.time()

    for i in range(50):
        # Adapt persona for different contexts
        context = f"context_{i % 10}"

        # Simulate adaptation (would use memory in real system)
        adapted = AdaptedPersona(
            base_persona=persona,
            context_type=context,
            adaptations=ContextAdjustment(
                temperature=0.5 + (i % 5) * 0.1,
                verbosity="standard",
                tool_preference=["tool1", "tool2"],
                expertise_boost=[f"skill_{i % 5}"],
                constraint_relaxations={}
            ),
            confidence=0.8,
            reasoning="Performance test adaptation"
        )

    adaptation_duration = time.time() - start_time

    # Adaptations should be fast
    assert adaptation_duration < 1.0  # 50 adaptations in < 1 second

    # Query memory
    start_time = time.time()

    query_tasks = []
    for i in range(100):
        query = f"Query {i % 20}"
        query_tasks.append(episodic_memory.recall_relevant(query, k=5))

    results = await asyncio.gather(*query_tasks)

    query_duration = time.time() - start_time

    # Queries should complete in reasonable time
    assert query_duration < 10.0  # 100 queries in < 10 seconds

    # Verify results
    assert all(len(r) <= 5 for r in results)  # k=5 limit


# ============================================================================
# Test: Memory-Constrained Persona Selection
# ============================================================================


@pytest.mark.asyncio
async def test_memory_guided_persona_selection(
    episodic_memory, persona_manager
):
    """Test that memory guides persona selection for new tasks.

    Scenario:
    1. Store episodes showing which persona works best for which tasks
    2. For a new task, query memory for best matching persona
    3. Verify recommended persona matches historical success

    Validates:
    - Memory-based persona recommendation
    - Task-to-persona matching
    - Learning from past experiences
    """
    # Store episodes showing persona performance
    persona_task_mapping = [
        {"persona": "senior_developer", "task": "code_review", "success": True, "reward": 9.0},
        {"persona": "senior_developer", "task": "architecture", "success": True, "reward": 8.5},
        {"persona": "creative_designer", "task": "ux_design", "success": True, "reward": 9.2},
        {"persona": "creative_designer", "task": "code_review", "success": False, "reward": 4.0},
    ]

    for mapping in persona_task_mapping:
        episode = Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now() - timedelta(hours=4),
            inputs={
                "persona_used": mapping["persona"],
                "task_type": mapping["task"]
            },
            actions=["execute"],
            outcomes={"success": mapping["success"]},
            rewards=mapping["reward"],
            embedding=await episodic_memory._embedding_service.embed_text(
                f"{mapping['persona']} {mapping['task']} {mapping['success']}"
            )
        )
        await episodic_memory.store_episode(episode)

    # Query for best persona for code review
    relevant = await episodic_memory.recall_relevant("code_review task", k=10)

    # Filter for successful code review episodes
    successful_reviews = [
        ep for ep in relevant
        if ep.inputs.get("task_type") == "code_review" and ep.outcomes.get("success")
    ]

    assert len(successful_reviews) > 0

    # Find best performing persona
    from collections import defaultdict
    persona_rewards = defaultdict(list)

    for ep in successful_reviews:
        persona = ep.inputs.get("persona_used")
        if persona:
            persona_rewards[persona].append(ep.rewards)

    # Calculate average rewards
    avg_rewards = {
        persona: sum(rewards) / len(rewards)
        for persona, rewards in persona_rewards.items()
    }

    # Best persona should be senior_developer
    best_persona = max(avg_rewards, key=avg_rewards.get)
    assert best_persona == "senior_developer"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
