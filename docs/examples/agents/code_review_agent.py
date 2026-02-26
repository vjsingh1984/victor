"""
Victor Code Review Agent Example

This example shows how to create an agent specialized for code review.
"""

import asyncio
from victor import Agent


async def review_code(file_path: str) -> str:
    """
    Create a code review agent and analyze the specified file.

    Args:
        file_path: Path to the Python file to review

    Returns:
        The review content
    """
    # Create an agent optimized for code review
    agent = Agent.create(
        provider="openai",
        model="gpt-4",
        vertical="coding",
        tools=["read", "grep"],
        temperature=0.3,  # Lower temperature for more focused analysis
        system_prompt="""You are a senior code reviewer. Your role is to:

1. Review code for correctness and bugs
2. Identify potential security issues
3. Suggest improvements for code quality
4. Check adherence to best practices (PEP 8, design patterns)
5. Provide specific, actionable feedback

Be constructive and respectful in your feedback."""
    )

    # Run the code review
    result = await agent.run(
        f"Please review the code in {file_path}. "
        "Focus on correctness, security, and best practices. "
        "Provide specific suggestions for improvement."
    )

    return result.content


async def main():
    """Run a code review example."""
    # Example: Review a file
    file_to_review = "main.py"

    print(f"Reviewing {file_to_review}...\n")

    try:
        review = await review_code(file_to_review)
        print(review)
    except Exception as e:
        print(f"Error during review: {e}")
        print("Make sure the file exists and is readable.")


if __name__ == "__main__":
    asyncio.run(main())
