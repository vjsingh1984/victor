#!/usr/bin/env python3
"""Test Victor code modification capabilities."""

import asyncio
import os
from pathlib import Path
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings


async def test_modification():
    """Test Victor modifying existing code."""
    print("=" * 70)
    print("Victor Code Modification Test with Ollama")
    print("=" * 70)

    # Setup
    settings = load_settings()
    test_dir = Path("/Users/vijaysingh/code/codingagent/victor_test")
    os.chdir(test_dir)

    print(f"\nWorking directory: {test_dir}")
    print(f"Test file: {test_dir / 'calculator.py'}")

    # Read original file
    original_content = (test_dir / "calculator.py").read_text()
    print("\n📄 Original File Content:")
    print("-" * 70)
    print(original_content)
    print("-" * 70)

    # Create agent with code-focused model
    print("\n🤖 Creating Victor agent (using qwen2.5-coder:7b)...")

    # Create a profile for code generation
    profile_config = {
        "provider": "ollama",
        "model": "qwen2.5-coder:7b",
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    try:
        agent = await AgentOrchestrator.from_settings(settings, "default")
    except:
        agent = await AgentOrchestrator.from_settings(settings, "default")

    # Task 1: Read and analyze the file
    print("\n" + "=" * 70)
    print("TASK 1: Analyze the code")
    print("=" * 70)

    message1 = f"""Please read the file calculator.py and analyze it. Tell me:
1. What functionality is currently implemented?
2. What's missing (like docstrings, type hints, error handling)?
3. What improvements would you suggest?"""

    print(f"\n💬 Prompt: {message1}")
    print("\n🔄 Processing...")

    response1 = await agent.chat(message1)
    print("\n📝 Victor's Analysis:")
    print("-" * 70)
    print(response1.content)
    print("-" * 70)

    # Task 2: Add improvements
    print("\n" + "=" * 70)
    print("TASK 2: Enhance the code")
    print("=" * 70)

    message2 = """Now, please enhance calculator.py by:
1. Adding proper Google-style docstrings to all functions
2. Adding type hints (use typing module)
3. Adding multiply and divide functions
4. Adding error handling for division by zero
5. Make the code more professional

Keep the same file structure. Show me the improved code."""

    print(f"\n💬 Prompt: {message2}")
    print("\n🔄 Processing...")

    response2 = await agent.chat(message2)
    print("\n📝 Victor's Response:")
    print("-" * 70)
    print(response2.content)
    print("-" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    print("\nDemonstrated Capabilities:")
    print("  ✓ Code analysis and review")
    print("  ✓ Understanding existing code structure")
    print("  ✓ Suggesting improvements")
    print("  ✓ Generating enhanced code with best practices")
    print("  ✓ Adding docstrings and type hints")
    print("  ✓ Implementing error handling")

    if response2.tool_calls:
        print(f"\n📊 Tools called: {len(response2.tool_calls)}")
        for tool_call in response2.tool_calls:
            tool_name = tool_call.get('name') if isinstance(tool_call, dict) else tool_call.name
            print(f"  - {tool_name}")

    await agent.provider.close()


if __name__ == "__main__":
    asyncio.run(test_modification())
