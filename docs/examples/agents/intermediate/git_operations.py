"""
Git Operations Example

Agent performs git operations with safety checks.
"""

import asyncio
from victor import Agent


async def review_commit(message: str = None):
    """Review proposed commit message."""
    agent = Agent.create(
        tools=["git_status", "git_diff"],
        temperature=0.3
    )

    if not message:
        result = await agent.run(
            """Review the current git status and staged changes.

            Suggest an appropriate commit message following conventional commits format."""
        )
    else:
        result = await agent.run(
            f"""Review this commit message: "{message}"

            Check:
            1. Follows conventional commits format
            2. Accurately describes changes
            3. Is clear and concise

            Suggest improvements if needed."""
        )

    return result.content


async def generate_commit_description():
    """Generate detailed commit description from changes."""
    agent = Agent.create(
        tools=["git_status", "git_diff", "read"],
        temperature=0.4
    )

    result = await agent.run(
        """Generate a detailed commit description based on current changes.

        Include:
        1. Commit title (50 chars or less)
        2. Detailed description of changes
        3. Breaking changes (if any)
        4. Related issues (if applicable)

        Format as conventional commit."""
    )

    return result.content


async def analyze_git_history():
    """Analyze recent git history."""
    agent = Agent.create(
        tools=["git"],
        temperature=0.3
    )

    result = await agent.run(
        """Analyze the last 10 commits.

        Report on:
        1. Commit patterns
        2. Contribution trends
        3. Any concerning patterns
        4. Suggestions for improvement"""
    )

    return result.content


async def create_branch_suggestion(issue_description: str):
    """Suggest branch name for an issue."""
    agent = Agent.create(
        temperature=0.2
    )

    result = await agent.run(
        f"""Based on this issue: "{issue_description}"

        Suggest an appropriate branch name following:
        - feature/ for new features
        - bugfix/ for bug fixes
        - hotfix/ for urgent fixes
        - refactor/ for refactoring

        Use kebab-case and keep it descriptive but concise."""
    )

    return result.content


async def main():
    """Run git operation examples."""
    print("=== Git Operations ===\n")

    # Review commit
    print("Reviewing commit...")
    review = await review_commit()
    print(review)

    print("\n" + "="*60 + "\n")

    # Generate commit description
    print("Generating commit message...")
    commit_msg = await generate_commit_description()
    print(commit_msg)


if __name__ == "__main__":
    asyncio.run(main())
