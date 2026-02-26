"""
Web Research Example

Agent uses web search and fetch to research topics.
"""

import asyncio
from victor import Agent


async def research_topic(topic: str, depth: str = "quick"):
    """Research a topic using web search."""
    agent = Agent.create(
        vertical="research",
        tools=["web_search", "web_fetch"],
        temperature=0.4
    )

    if depth == "quick":
        prompt = f"""Quick research on: {topic}

        Provide:
        1. Brief overview (2-3 sentences)
        2. Key points (bullet list)
        3. Current state/developments"""
    else:  # deep research
        prompt = f"""Comprehensive research on: {topic}

        Provide:
        1. Detailed background
        2. Historical context
        3. Current state of the art
        4. Key players/companies
        5. Recent developments
        6. Future trends
        7. References and further reading"""

    result = await agent.run(prompt)
    return result.content


async def compare_options(options: list[str]):
    """Compare multiple options using web research."""
    agent = Agent.create(
        vertical="research",
        tools=["web_search", "web_fetch"],
        temperature=0.3
    )

    options_str = ", ".join(options)
    result = await agent.run(
        f"""Research and compare these options: {options_str}

        For each option, provide:
        1. Description
        2. Pros
        3. Cons
        4. Best use cases
        5. Comparison to others

        End with a recommendation table."""
    )

    return result.content


async def fact_check(statement: str):
    """Fact-check a statement using web search."""
    agent = Agent.create(
        vertical="research",
        tools=["web_search"],
        temperature=0.2
    )

    result = await agent.run(
        f"""Fact-check this statement: "{statement}"

        Search for:
        1. Supporting evidence
        2. Contradicting evidence
        3. Reliable sources

        Provide:
        - Verdict (True/False/Mixed/Unverifiable)
        - Evidence summary
        - Confidence level
        - Sources"""
    )

    return result.content


async def summarize_article(url: str):
    """Fetch and summarize an article."""
    agent = Agent.create(
        tools=["web_fetch"],
        temperature=0.4
    )

    result = await agent.run(
        f"""Fetch and summarize the article at: {url}

        Provide:
        1. Main points (3-5 bullets)
        2. Key takeaways
        3. Target audience
        4. Verdict (worth reading or not)"""
    )

    return result.content


async def main():
    """Run web research examples."""
    print("=== Web Research ===\n")

    # Quick research
    print("Researching quantum computing...")
    research = await research_topic("quantum computing applications")
    print(research)

    print("\n" + "="*60 + "\n")

    # Compare options
    print("Comparing Python frameworks...")
    comparison = await compare_options([
        "Django",
        "Flask",
        "FastAPI"
    ])
    print(comparison)


if __name__ == "__main__":
    asyncio.run(main())
