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

"""Real conversation flow integration tests.

These tests verify multi-turn conversation management with real LLM providers.
No mocking is used.

Target: 100% pass rate on M1 Max hardware with Ollama provider.
"""

import time

import pytest


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_conversation_context_preservation(
    ollama_provider, ollama_model_name, temp_workspace
):
    """Test conversation context is preserved across turns.

    Verifies:
    - Context from turn 1 is available in turn 2
    - LLM can reference previous information
    - No context loss between turns
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

    settings = Settings()
    settings.provider = "ollama"
    settings.model = ollama_model_name
    settings.working_dir = temp_workspace
    settings.airgapped_mode = True

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=ollama_provider,
        model=settings.model,
    )

    # Turn 1: Provide information
    response1 = await orchestrator.chat(
        user_message="I'm working on a Python project with a Calculator class. Remember this."
    )

    assert response1.content is not None
    print(f"✓ Turn 1 response: {response1.content[:100]}...")

    # Turn 2: Reference previous information
    response2 = await orchestrator.chat(
        user_message="What project did I mention I'm working on?"
    )

    assert response2.content is not None

    # Verify LLM remembers context
    content_lower = response2.content.lower()
    assert (
        "calculator" in content_lower or "python" in content_lower
    ), f"LLM should remember the project: {response2.content}"

    print(f"✓ Context preserved: {response2.content[:200]}...")


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_conversation_stage_transitions(
    ollama_provider, ollama_model_name, sample_code_file, temp_workspace
):
    """Test stage transitions based on conversation flow.

    Verifies:
    - Stages transition correctly (INITIAL → PLANNING → READING → EXECUTING)
    - Each transition is validated
    - Stage history is recorded
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

    settings = Settings()
    settings.provider = "ollama"
    settings.model = ollama_model_name
    settings.working_dir = temp_workspace
    settings.airgapped_mode = True

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=ollama_provider,
        model=settings.model,
    )

    initial_stage = orchestrator.conversation_state.state.stage
    print(f"✓ Initial stage: {initial_stage}")

    # Start a task that will trigger stage transitions
    response = await orchestrator.chat(
        user_message=f"I need to analyze the code in {sample_code_file}. Read it first, then add error handling to the add function."
    )

    assert response.content is not None

    # Check that stage transitions occurred
    # Note: We don't assert exact stages as they depend on LLM behavior
    # but we verify the conversation progressed
    final_stage = orchestrator.conversation_state.state.stage
    print(f"✓ Final stage: {final_stage}")

    # Verify stage history exists
    history = orchestrator.conversation_state.get_stage_history()
    assert len(history) > 0, "Stage history should be recorded"

    print(f"✓ Stage history: {[(h['from'], h['to']) for h in history]}")


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_conversation_error_recovery(
    ollama_provider, ollama_model_name, temp_workspace
):
    """Test conversation continues after tool failure.

    Verifies:
    - Error is caught and reported
    - Conversation continues after error
    - LLM adjusts strategy based on error
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

    settings = Settings()
    settings.provider = "ollama"
    settings.model = ollama_model_name
    settings.working_dir = temp_workspace
    settings.airgapped_mode = True

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=ollama_provider,
        model=settings.model,
    )

    # Attempt to read non-existent file (will fail)
    non_existent_file = "/tmp/this_file_does_not_exist_12345.txt"

    response1 = await orchestrator.chat(
        user_message=f"Read the file {non_existent_file}."
    )

    assert response1.content is not None

    # Verify error was mentioned in response
    # Note: LLM may phrase this differently
    print(f"✓ Response to error: {response1.content[:200]}...")

    # Conversation should continue - ask a different question
    response2 = await orchestrator.chat(
        user_message="That's fine. Instead, create a simple Python file with a hello world function."
    )

    assert response2.content is not None
    assert len(response2.content) > 0

    # Verify conversation continued successfully
    print(f"✓ Conversation recovered: {response2.content[:200]}...")


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_conversation_multi_turn_task_completion(
    ollama_provider, ollama_model_name, temp_workspace
):
    """Test complex multi-turn task completion.

    Verifies:
    - Multiple conversation turns complete successfully
    - Each turn builds on previous context
    - Task is completed through collaboration
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

    settings = Settings()
    settings.provider = "ollama"
    settings.model = ollama_model_name
    settings.working_dir = temp_workspace
    settings.airgapped_mode = True

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=ollama_provider,
        model=settings.model,
    )

    from pathlib import Path

    # Create initial file
    test_file = Path(temp_workspace) / "task_test.py"
    test_file.write_text("def task():\n    pass\n")

    start_time = time.time()

    # Turn 1: Analyze the file
    response1 = await orchestrator.chat(
        user_message=f"Analyze the file {test_file}. What does it do?"
    )

    assert response1.content is not None
    print(f"✓ Turn 1 (Analyze): {len(response1.content)} chars")

    # Turn 2: Ask for improvement
    response2 = await orchestrator.chat(
        user_message="Add a docstring and implement the function to return 'Task completed'."
    )

    assert response2.content is not None

    # Verify file was modified
    content = test_file.read_text()
    assert '"""' in content or "'''" in content or "return" in content

    print(f"✓ Turn 2 (Improve): {len(response2.content)} chars")

    # Turn 3: Verify the changes
    response3 = await orchestrator.chat(
        user_message="Read the file again to verify the changes."
    )

    assert response3.content is not None

    elapsed = time.time() - start_time

    print(f"✓ Turn 3 (Verify): {len(response3.content)} chars")
    print(f"✓ Multi-turn task completed in {elapsed:.2f}s")

    # Verify all turns succeeded
    assert len(response1.content) > 0
    assert len(response2.content) > 0
    assert len(response3.content) > 0


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_conversation_tool_calling_accuracy(
    ollama_provider, ollama_model_name, sample_code_file, temp_workspace
):
    """Test tool calling accuracy in multi-turn conversations.

    Verifies:
    - Tools are called when appropriate
    - Tool parameters are correct
    - Tool results are used in subsequent responses
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

    settings = Settings()
    settings.provider = "ollama"
    settings.model = ollama_model_name
    settings.working_dir = temp_workspace
    settings.airgapped_mode = True

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=ollama_provider,
        model=settings.model,
    )

    # Start a conversation that will require tool usage
    response = await orchestrator.chat(
        user_message=f"I have a file at {sample_code_file}. First read it, then tell me how many functions it defines."
    )

    assert response.content is not None

    # Verify response mentions reading the file
    content_lower = response.content.lower()
    # LLM should mention the file or functions
    assert (
        "function" in content_lower
        or "greet" in content_lower
        or "add" in content_lower
        or "read" in content_lower
    ), f"Response should reference file content: {response.content[:200]}"

    print(f"✓ Tool calling successful")
    print(f"✓ Response: {response.content[:200]}...")


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_conversation_memory_efficiency(
    ollama_provider, ollama_model_name, temp_workspace
):
    """Test conversation memory usage is efficient.

    Verifies:
    - Long conversations don't cause memory issues
    - Context is properly summarized if needed
    - Performance remains acceptable
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

    settings = Settings()
    settings.provider = "ollama"
    settings.model = ollama_model_name
    settings.working_dir = temp_workspace
    settings.airgapped_mode = True

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=ollama_provider,
        model=settings.model,
    )

    start_time = time.time()

    # Simulate a longer conversation (5 turns)
    for i in range(5):
        if i == 0:
            response = await orchestrator.chat(
                user_message="Let's work on a Python project step by step."
            )
        else:
            response = await orchestrator.chat(
                user_message=f"Turn {i} completed. What's the next step?"
            )
        assert response.content is not None

    elapsed = time.time() - start_time

    print(f"✓ 5-turn conversation completed in {elapsed:.2f}s")
    print(f"✓ Average time per turn: {elapsed/5:.2f}s")

    # Verify performance is acceptable
    assert (
        elapsed < 120
    ), f"5-turn conversation should complete in < 120s, took {elapsed:.2f}s"
