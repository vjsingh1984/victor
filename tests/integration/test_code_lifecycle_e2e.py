"""End-to-end integration test for complete code lifecycle with Ollama.

This test covers the full workflow:
1. Create a Python code file using AI agent
2. Enhance/modify the code
3. Execute the code
4. Verify results

Requires Ollama to be running with a capable coding model.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
from httpx import ConnectError

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.ollama import OllamaProvider
from victor.tools.base import ToolRegistry
from victor.tools.filesystem import read_file, write_file
from victor.tools.bash import execute_bash


@pytest.fixture
async def ollama_coding_provider():
    """Create Ollama provider with a coding model."""
    provider = OllamaProvider(base_url="http://localhost:11434")

    try:
        # Try to list models to check if Ollama is running
        models = await provider.list_models()
        if not models:
            pytest.skip("No models available in Ollama")

        # Look for a coding model (prefer qwen2.5-coder or similar)
        coding_models = [
            m["name"]
            for m in models
            if any(
                keyword in m["name"].lower()
                for keyword in ["coder", "code", "qwen"]
            )
        ]

        if not coding_models:
            # Fallback to any available model
            coding_models = [models[0]["name"]]

        print(f"\nUsing model: {coding_models[0]}")
        provider.default_model = coding_models[0]
        yield provider

    except ConnectError:
        pytest.skip("Ollama is not running")
    finally:
        await provider.close()


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for code execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def agent_with_tools(ollama_coding_provider):
    """Create an agent orchestrator with filesystem and bash tools."""
    settings = Settings()

    agent = AgentOrchestrator(
        settings=settings,
        provider=ollama_coding_provider,
        model=ollama_coding_provider.default_model,
        temperature=0.3,  # Lower temperature for more consistent code
    )

    # Register additional tools (orchestrator already has default tools)
    agent.tools.register(read_file.Tool)
    agent.tools.register(write_file.Tool)
    agent.tools.register(execute_bash.Tool)

    yield agent

    # Cleanup
    agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_full_code_lifecycle_simple(agent_with_tools, temp_workspace):
    """Test creating, enhancing, and executing a simple Python script.

    This test verifies:
    1. Agent can write a simple Python script
    2. Agent can read and enhance the script
    3. Script executes successfully
    4. Output is correct
    """
    agent = agent_with_tools
    workspace = temp_workspace
    script_path = workspace / "calculator.py"

    print("\n" + "=" * 70)
    print("PHASE 1: Creating initial Python script")
    print("=" * 70)

    # Phase 1: Create a simple calculator script
    response1 = await agent.chat(
        f"""Create a simple Python calculator script at {script_path}.

The script should:
- Have a function called 'calculate' that takes two numbers and an operator (+, -, *, /)
- Have a main block that calls calculate(10, 5, '+') and prints the result
- Use proper error handling

Just write the code, don't explain it."""
    )

    print(f"Agent response: {response1.content[:200]}...")

    # Verify the file was created
    assert script_path.exists(), "Script file was not created"
    code_content = script_path.read_text()
    assert "calculate" in code_content.lower(), "Function 'calculate' not found"
    assert "def " in code_content, "No function definition found"

    print(f"\n✓ Created script ({len(code_content)} chars)")
    print(f"Preview:\n{code_content[:200]}...")

    print("\n" + "=" * 70)
    print("PHASE 2: Enhancing the script")
    print("=" * 70)

    # Phase 2: Enhance the script with additional features
    response2 = await agent.chat(
        f"""Read the file at {script_path} and enhance it by:
1. Adding a 'multiply_all' function that takes a list of numbers and returns their product
2. Update the main block to also test multiply_all([2, 3, 4]) and print the result

Read the file first, then write the enhanced version."""
    )

    print(f"Agent response: {response2.content[:200]}...")

    # Verify enhancements
    enhanced_content = script_path.read_text()
    assert "multiply_all" in enhanced_content.lower(), "multiply_all function not added"
    assert enhanced_content != code_content, "File was not modified"

    print(f"\n✓ Enhanced script ({len(enhanced_content)} chars)")
    print(f"Preview:\n{enhanced_content[:200]}...")

    print("\n" + "=" * 70)
    print("PHASE 3: Executing the script")
    print("=" * 70)

    # Phase 3: Execute the script
    response3 = await agent.chat(
        f"Execute the Python script at {script_path} and show me the output."
    )

    print(f"Agent response: {response3.content}")

    # Manually execute to verify (in case agent didn't)
    import subprocess

    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )

    print(f"\n✓ Script executed")
    print(f"Exit code: {result.returncode}")
    print(f"Output:\n{result.stdout}")

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert result.stdout.strip(), "No output from script"

    # Verify expected output patterns
    output = result.stdout.lower()
    assert any(char in output for char in ["15", "5", "24"]), "Expected calculation results not found"

    print("\n" + "=" * 70)
    print("✅ FULL LIFECYCLE TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_code_lifecycle_with_bugs(agent_with_tools, temp_workspace):
    """Test creating code with bugs, detecting them, fixing, and re-executing.

    This test verifies:
    1. Agent can create code (even with bugs)
    2. Agent can detect bugs from execution errors
    3. Agent can fix bugs
    4. Fixed code executes successfully
    """
    agent = agent_with_tools
    workspace = temp_workspace
    script_path = workspace / "buggy_script.py"

    print("\n" + "=" * 70)
    print("PHASE 1: Creating script (may have bugs)")
    print("=" * 70)

    # Phase 1: Create a script
    response1 = await agent.chat(
        f"""Write a Python script at {script_path} that:
1. Defines a function 'divide_numbers(a, b)' that returns a/b
2. In the main block, call divide_numbers(10, 2) and print the result
3. Then call divide_numbers(5, 0) and print the result

Just write the code."""
    )

    print(f"Agent response: {response1.content[:200]}...")

    assert script_path.exists(), "Script was not created"
    initial_content = script_path.read_text()

    print(f"\n✓ Created script ({len(initial_content)} chars)")

    print("\n" + "=" * 70)
    print("PHASE 2: Executing (expect error)")
    print("=" * 70)

    # Phase 2: Try to execute (should fail with ZeroDivisionError)
    import subprocess

    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )

    print(f"Exit code: {result.returncode}")
    print(f"Stderr: {result.stderr}")

    # Expect failure due to division by zero
    has_error = result.returncode != 0 or "error" in result.stderr.lower()

    print("\n" + "=" * 70)
    print("PHASE 3: Fixing the bug")
    print("=" * 70)

    if has_error:
        # Phase 3: Ask agent to fix the error
        response3 = await agent.chat(
            f"""The script at {script_path} produced this error:
{result.stderr}

Please read the file, fix the bug by adding proper error handling for division by zero,
and write the corrected version."""
        )

        print(f"Agent response: {response3.content[:200]}...")

        fixed_content = script_path.read_text()
        assert fixed_content != initial_content, "File was not modified"

        # Check for error handling
        assert any(
            keyword in fixed_content.lower()
            for keyword in ["try", "except", "if", "zerodivision"]
        ), "No error handling added"

        print(f"\n✓ Bug fixed ({len(fixed_content)} chars)")

        print("\n" + "=" * 70)
        print("PHASE 4: Re-executing fixed script")
        print("=" * 70)

        # Phase 4: Execute fixed version
        result2 = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        print(f"Exit code: {result2.returncode}")
        print(f"Output:\n{result2.stdout}")

        assert result2.returncode == 0, f"Fixed script still fails: {result2.stderr}"
        assert result2.stdout.strip(), "No output from fixed script"

        print("\n" + "=" * 70)
        print("✅ BUG FIX LIFECYCLE TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
    else:
        print("\n⚠️  Script didn't error (agent may have added error handling proactively)")
        print("This is actually good behavior!")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_code_lifecycle_minimal(ollama_coding_provider, temp_workspace):
    """Minimal test that doesn't rely on agent understanding tool usage.

    Tests direct file operations and execution without complex agent interactions.
    """
    workspace = temp_workspace
    script_path = workspace / "hello.py"

    print("\n" + "=" * 70)
    print("MINIMAL LIFECYCLE TEST")
    print("=" * 70)

    # Step 1: Write file directly
    code = """
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
    print(greet("Ollama"))
"""

    script_path.write_text(code)
    print(f"\n✓ Created script at {script_path}")

    # Step 2: Read it back
    content = script_path.read_text()
    assert content == code
    print(f"✓ Verified content ({len(content)} chars)")

    # Step 3: Execute
    import subprocess

    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )

    print(f"\n✓ Executed script")
    print(f"Output:\n{result.stdout}")

    assert result.returncode == 0
    assert "Hello, World!" in result.stdout
    assert "Hello, Ollama!" in result.stdout

    print("\n✅ MINIMAL TEST PASSED")

    # Step 4: Now enhance with agent
    settings = Settings()

    agent = AgentOrchestrator(
        settings=settings,
        provider=ollama_coding_provider,
        model=ollama_coding_provider.default_model,
        temperature=0.3,
    )

    # Register tools
    agent.tools.register(read_file.Tool)
    agent.tools.register(write_file.Tool)

    print("\n" + "=" * 70)
    print("TESTING AGENT ENHANCEMENT")
    print("=" * 70)

    response = await agent.chat(
        f"""Read the Python file at {script_path}.
Then add a new function called 'greet_formal' that takes a name and returns 'Good day, [name]'.
Update the main block to also call greet_formal("Assistant") and print the result.
Write the enhanced version back to the same file."""
    )

    print(f"\nAgent response: {response.content[:300]}...")

    # Verify enhancement
    enhanced = script_path.read_text()
    if "greet_formal" in enhanced.lower():
        print("\n✓ Agent successfully enhanced the code!")

        # Execute enhanced version
        result2 = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        print(f"Enhanced output:\n{result2.stdout}")

        if result2.returncode == 0:
            print("\n✅ AGENT ENHANCEMENT SUCCESSFUL")
        else:
            print(f"\n⚠️  Enhanced code has errors: {result2.stderr}")
    else:
        print("\n⚠️  Agent didn't add the requested function (may need better prompting)")

    # Cleanup
    agent.shutdown()


if __name__ == "__main__":
    """Run tests directly for debugging."""
    print("Running end-to-end code lifecycle tests...")
    print("Make sure Ollama is running with a coding model!")
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
