"""
Code Analysis Example

Agent analyzes code quality, finds bugs, and suggests improvements.
"""

import asyncio
from victor import Agent


async def analyze_code_quality(file_path: str):
    """Analyze code quality and provide feedback."""
    agent = Agent.create(
        vertical="coding",
        tools=["read", "grep"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Analyze the code in {file_path}.

        Focus on:
        1. Code quality and readability
        2. Potential bugs or issues
        3. Security vulnerabilities
        4. Performance concerns
        5. Best practices violations

        Provide specific, actionable feedback with line numbers where applicable."""
    )

    return result.content


async def suggest_improvements(file_path: str):
    """Suggest specific code improvements."""
    agent = Agent.create(
        vertical="coding",
        tools=["read"],
        temperature=0.5
    )

    result = await agent.run(
        f"""Review the code in {file_path} and suggest improvements.

        For each suggestion:
        - What to change
        - Why it should be changed
        - How to change it (show code)

        Prioritize by impact."""
    )

    return result.content


async def main():
    """Run code analysis examples."""
    # Example usage
    file_to_analyze = "example.py"

    print("=== Code Analysis ===\n")

    # Analyze code quality
    print("Analyzing code quality...")
    quality_report = await analyze_code_quality(file_to_analyze)
    print(quality_report)

    print("\n" + "="*60 + "\n")

    # Suggest improvements
    print("Suggesting improvements...")
    improvements = await suggest_improvements(file_to_analyze)
    print(improvements)


if __name__ == "__main__":
    asyncio.run(main())
