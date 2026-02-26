"""
Advanced Multi-Agent Example

Multiple agents working together on a complex task.
"""

import asyncio
from victor import Agent


async def software_development_team(task: str):
    """Simulate a software development team."""
    # Create specialized agents
    product_owner = Agent.create(
        system_prompt="You are a Product Owner focused on user value and requirements.",
        temperature=0.5
    )

    architect = Agent.create(
        system_prompt="You are a Software Architect focused on system design and best practices.",
        vertical="coding",
        temperature=0.4
    )

    developer = Agent.create(
        system_prompt="You are a Senior Developer focused on clean, maintainable code.",
        vertical="coding",
        tools=["read", "write", "edit"],
        temperature=0.3
    )

    qa_engineer = Agent.create(
        system_prompt="You are a QA Engineer focused on testing and quality assurance.",
        vertical="coding",
        temperature=0.3
    )

    # Step 1: Product Owner defines requirements
    print("1. Product Owner gathering requirements...")
    requirements = await product_owner.run(
        f"""Define user stories and acceptance criteria for: {task}

        Include:
        - User story format
        - Acceptance criteria
        - Priority
        - Dependencies"""
    )

    print(f"Requirements:\n{requirements.content}\n")

    # Step 2: Architect designs solution
    print("2. Architect designing system...")
    design = await architect.run(
        f"""Design a system architecture for these requirements:
        {requirements.content}

        Include:
        - Component design
        - Data flow
        - Technology choices
        - Security considerations
        - Scalability approach"""
    )

    print(f"Architecture:\n{design.content}\n")

    # Step 3: Developer implements
    print("3. Developer implementing...")
    implementation = await developer.run(
        f"""Write implementation code based on:
        Requirements: {requirements.content}
        Architecture: {design.content}

        Provide:
        - Clean, well-documented code
        - Error handling
        - Type hints
        - Examples"""
    )

    print(f"Implementation:\n{implementation.content[:500]}...\n")

    # Step 4: QA Engineer reviews
    print("4. QA Engineer reviewing...")
    review = await qa_engineer.run(
        f"""Review this implementation for quality:
        {implementation.content}

        Check:
        - Code quality
        - Edge cases
        - Error handling
        - Test coverage needs
        - Security issues"""
    )

    print(f"QA Review:\n{review.content}\n")

    # Summary
    summary = {
        "requirements": requirements.content,
        "design": design.content,
        "implementation": implementation.content,
        "review": review.content
    }

    return summary


async def research_consensus(topic: str):
    """Multiple research agents reach consensus on a topic."""
    # Create research agents with different perspectives
    technical_researcher = Agent.create(
        system_prompt="You research from a technical/engineering perspective.",
        tools=["web_search"],
        temperature=0.4
    )

    business_researcher = Agent.create(
        system_prompt="You research from a business/market perspective.",
        tools=["web_search"],
        temperature=0.4
    )

    academic_researcher = Agent.create(
        system_prompt="You research from an academic/scientific perspective.",
        tools=["web_search"],
        temperature=0.4
    )

    # Parallel research
    print(f"Researching: {topic}\n")

    tasks = [
        technical_researcher.run(f"Technical research on: {topic}"),
        business_researcher.run(f"Business research on: {topic}"),
        academic_researcher.run(f"Academic research on: {topic}")
    ]

    results = await asyncio.gather(*tasks)

    print("=== Technical Perspective ===")
    print(results[0].content)

    print("\n=== Business Perspective ===")
    print(results[1].content)

    print("\n=== Academic Perspective ===")
    print(results[2].content)

    # Synthesize
    synthesizer = Agent.create(temperature=0.3)

    synthesis = await synthesizer.run(
        f"""Synthesize these three research perspectives on {topic}:

        Technical: {results[0].content}

        Business: {results[1].content}

        Academic: {results[2].content}

        Provide:
        1. Consensus points
        2. Divergent views
        3. Key insights
        4. Recommendations"""
    )

    print("\n=== Synthesis ===")
    print(synthesis.content)

    return synthesis.content


async def main():
    """Run multi-agent examples."""
    print("=== Multi-Agent Examples ===\n")

    # Example 1: Software development team
    print("Software Development Team:")
    print("-" * 50)
    await software_development_team("Build a REST API for task management")

    print("\n\n")

    # Example 2: Research consensus
    print("Research Consensus:")
    print("-" * 50)
    await research_consensus("The future of AI in healthcare")


if __name__ == "__main__":
    asyncio.run(main())
