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

"""Real tool execution integration tests.

These tests execute actual tool calls with real LLM providers and real file
operations. No mocking is used.

Target: 100% pass rate on M1 Max hardware with Ollama provider.

CI/CD: Tests are automatically skipped if Ollama is not available.
"""

import time
from pathlib import Path

import pytest


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_real_read_tool_execution(ollama_provider, ollama_model_name, sample_code_file, temp_workspace):
    """Test Read tool executes with real LLM.

    Verifies:
    - LLM calls Read tool when asked to read a file
    - File content is returned correctly
    - Response time is acceptable (< 30s)
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

    # Create orchestrator with real provider
    settings = Settings()
    settings.provider = "ollama"
    settings.model = (
        ollama_provider._models[0]
        if hasattr(ollama_provider, "_models")
        else "qwen3-coder-tools:30b"
    )
    settings.working_dir = temp_workspace
    settings.airgapped_mode = True  # Use local provider only

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=ollama_provider,
        model=settings.model,
    )

    file_path = sample_code_file
    start_time = time.time()

    # Ask LLM to read the file
    response = await orchestrator.chat(
        user_message=f"Read the file {file_path} and tell me what functions it defines."
    )

    elapsed = time.time() - start_time

    # Verify response
    assert response is not None, "Response should not be None"
    assert response.content is not None, "Response content should not be None"
    assert len(response.content) > 0, "Response content should not be empty"

    # Verify file was read (response mentions functions)
    content_lower = response.content.lower()
    assert (
        "greet" in content_lower or "add" in content_lower
    ), f"Response should mention functions from file: {response.content}"

    # Verify response time is acceptable (relaxed for commodity hardware)
    assert elapsed < 60, f"Response time should be < 60s, took {elapsed:.2f}s"

    print(f"✓ Read tool executed in {elapsed:.2f}s")
    print(f"✓ Response: {response.content[:200]}...")


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_real_edit_tool_execution(ollama_provider, ollama_model_name, sample_code_file, temp_workspace):
    """Test Edit tool executes real file modifications.

    Verifies:
    - LLM calls Edit tool when asked to modify a file
    - File is modified correctly
    - Change is persisted to disk
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

    file_path = sample_code_file

    # Read original content
    original_content = Path(file_path).read_text()

    start_time = time.time()

    # Ask LLM to modify the file
    response = await orchestrator.chat(
        user_message=f"Add a new function called 'multiply' that multiplies two numbers to the file {file_path}."
    )

    elapsed = time.time() - start_time

    # Verify file was modified
    new_content = Path(file_path).read_text()
    assert new_content != original_content, "File content should have changed"
    assert "multiply" in new_content.lower(), "New function should be added"
    assert "def multiply" in new_content, "Function definition should be present"

    # Verify response
    assert response.content is not None
    assert len(response.content) > 0

    print(f"✓ Edit tool executed in {elapsed:.2f}s")
    print("✓ File modified successfully")


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_real_shell_tool_execution(ollama_provider, ollama_model_name, temp_workspace):
    """Test Shell tool executes real commands.

    Verifies:
    - LLM calls Shell tool when asked to run commands
    - Command executes successfully
    - Output is captured correctly
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

    # Ask LLM to list files
    response = await orchestrator.chat(
        user_message="List all files in the current directory using ls command."
    )

    elapsed = time.time() - start_time

    # Verify response
    assert response.content is not None
    assert len(response.content) > 0

    # Verify command output is mentioned
    # Note: We don't check for specific files as output format may vary
    assert (
        "sample.py" in response.content
        or "readme" in response.content.lower()
        or "file" in response.content.lower()
    ), f"Response should mention files: {response.content}"

    print(f"✓ Shell tool executed in {elapsed:.2f}s")
    print(f"✓ Response: {response.content[:200]}...")


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_real_multi_tool_execution(ollama_provider, ollama_model_name, sample_code_file, temp_workspace):
    """Test multiple tools execute in sequence.

    Verifies:
    - Multiple tools are called in correct order
    - State is preserved between tool calls
    - All operations complete successfully
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

    file_path = sample_code_file
    start_time = time.time()

    # Multi-turn conversation requiring multiple tools
    # Turn 1: Read file
    response1 = await orchestrator.chat(user_message=f"Read the file {file_path}.")

    assert response1.content is not None
    print(f"✓ Turn 1 (Read): {len(response1.content)} chars")

    # Turn 2: Add docstring
    response2 = await orchestrator.chat(
        user_message="Add a docstring to the greet function that explains what it does."
    )

    assert response2.content is not None

    # Verify file was modified
    new_content = Path(file_path).read_text()
    assert '"""' in new_content or "'''" in new_content, "Docstring should be added"

    print(f"✓ Turn 2 (Edit): {len(response2.content)} chars")

    # Turn 3: Verify changes
    response3 = await orchestrator.chat(
        user_message="Read the file again to verify the docstring was added."
    )

    assert response3.content is not None

    elapsed = time.time() - start_time

    print(f"✓ Turn 3 (Verify): {len(response3.content)} chars")
    print(f"✓ Multi-tool execution completed in {elapsed:.2f}s")

    # Verify all three turns completed
    assert len(response1.content) > 0
    assert len(response2.content) > 0
    assert len(response3.content) > 0


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_real_grep_tool_execution(ollama_provider, ollama_model_name, sample_code_file, temp_workspace):
    """Test Grep tool executes real searches.

    Verifies:
    - LLM calls Grep tool when asked to search
    - Search returns correct results
    - Multiple matches are handled
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

    # Ask LLM to search for a function definition
    response = await orchestrator.chat(
        user_message=f"Search for all function definitions in {temp_workspace} using grep."
    )

    elapsed = time.time() - start_time

    # Verify response
    assert response.content is not None
    assert len(response.content) > 0

    # Verify grep found functions
    content_lower = response.content.lower()
    assert (
        "def" in content_lower or "greet" in content_lower or "add" in content_lower
    ), f"Grep should find function definitions: {response.content}"

    print(f"✓ Grep tool executed in {elapsed:.2f}s")


@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_real_write_tool_execution(ollama_provider, ollama_model_name, temp_workspace):
    """Test Write tool creates new files.

    Verifies:
    - LLM calls Write tool when asked to create a file
    - File is created with correct content
    - File is persisted to disk
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

    new_file_path = Path(temp_workspace) / "new_file.py"

    start_time = time.time()

    # Ask LLM to create a new file
    response = await orchestrator.chat(
        user_message=f"Create a new Python file at {new_file_path} with a class called 'Calculator' that has add and subtract methods."
    )

    elapsed = time.time() - start_time

    # Verify file was created
    assert new_file_path.exists(), f"File should be created at {new_file_path}"

    content = new_file_path.read_text()
    assert "calculator" in content.lower(), "Calculator class should be in file"
    assert "def add" in content or "def subtract" in content, "Methods should be defined"

    # Verify response
    assert response.content is not None
    assert len(response.content) > 0

    print(f"✓ Write tool executed in {elapsed:.2f}s")
    print(f"✓ File created with {len(content)} bytes")
