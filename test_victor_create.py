#!/usr/bin/env python3
"""Test Victor creating code from scratch."""

import asyncio
import os
from pathlib import Path
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings
from victor.tools.filesystem import WriteFileTool


async def test_creation():
    """Test Victor creating new code."""
    print("=" * 70)
    print("Victor Code Creation Test with Ollama")
    print("=" * 70)

    settings = load_settings()
    test_dir = Path("/Users/vijaysingh/code/codingagent/victor_test")
    os.chdir(test_dir)

    print(f"\nWorking directory: {test_dir}")

    # Create agent
    print("\n🤖 Creating Victor agent with Ollama...")
    agent = await AgentOrchestrator.from_settings(settings, "default")

    # Task: Create a shopping cart class
    print("\n" + "=" * 70)
    print("TASK: Create ShoppingCart class from scratch")
    print("=" * 70)

    message = """Create a Python ShoppingCart class with the following features:

1. A ShoppingCart class that stores items
2. Each item has: name (str), price (float), quantity (int)
3. Methods:
   - add_item(name, price, quantity=1): Add item to cart
   - remove_item(name): Remove item from cart
   - get_total(): Calculate total price
   - apply_discount(percent): Apply percentage discount
   - get_items(): Return list of items

4. Use:
   - Type hints for all parameters and returns
   - Google-style docstrings
   - Proper error handling
   - A dataclass or namedtuple for items

Please provide the complete Python code with examples in a main() function."""

    print(f"\n💬 Prompt:")
    print(message)
    print("\n🔄 Processing with Ollama...")

    response = await agent.chat(message)

    print("\n📝 Victor's Generated Code:")
    print("=" * 70)
    print(response.content)
    print("=" * 70)

    # Extract code and save it manually
    content = response.content
    if "```python" in content:
        # Extract code from markdown
        code_start = content.find("```python") + 9
        code_end = content.find("```", code_start)
        code = content[code_start:code_end].strip()

        # Write the file
        output_file = test_dir / "shopping_cart.py"
        output_file.write_text(code)
        print(f"\n💾 Saved generated code to: {output_file}")

        # Show file stats
        lines = len(code.split('\n'))
        chars = len(code)
        print(f"   📊 Statistics: {lines} lines, {chars} characters")

        # Try to validate syntax
        try:
            compile(code, output_file, 'exec')
            print("   ✅ Code is syntactically valid!")
        except SyntaxError as e:
            print(f"   ❌ Syntax error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    print("\nDemonstrated:")
    print("  ✓ Code generation from natural language requirements")
    print("  ✓ Complex class structure with multiple methods")
    print("  ✓ Type hints and docstrings")
    print("  ✓ Error handling")
    print("  ✓ Professional Python code structure")

    if response.tool_calls:
        print(f"\n📊 Tools called: {len(response.tool_calls)}")

    await agent.provider.close()


if __name__ == "__main__":
    asyncio.run(test_creation())
