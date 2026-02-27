"""Basic agent recipes for common tasks.

These recipes are simple, ready-to-use agent configurations
for everyday tasks.
"""

RECIPE_CATEGORY = "agents/basic"
RECIPE_DIFFICULTY = "beginner"
RECIPE_TIME = "5 minutes"


async def simple_qa():
    """Answer a simple question."""
    from victor import Agent

    agent = Agent.create()
    result = await agent.run("What is the capital of France?")
    return result


async def text_generation(prompt: str = "Write a story about AI"):
    """Generate text on a given topic."""
    from victor import Agent

    agent = Agent.create()
    result = await agent.run(prompt)
    return result


async def text_summarization(text: str, max_words: int = 50):
    """Summarize text to specified word count."""
    from victor import Agent

    agent = Agent.create(temperature=0.3)
    result = await agent.run(
        f"Summarize this in {max_words} words or less:\n\n{text}"
    )
    return result


async def text_translation(text: str, target_language: str = "Spanish"):
    """Translate text to target language."""
    from victor import Agent

    agent = Agent.create(temperature=0.2)
    result = await agent.run(
        f"Translate this to {target_language}:\n\n{text}"
    )
    return result


async def creative_writing(topic: str = "a futuristic city"):
    """Generate creative writing on a topic."""
    from victor import Agent

    agent = Agent.create(temperature=0.9)
    result = await agent.run(
        f"Write a creative short story about {topic}. "
        "Make it vivid and imaginative."
    )
    return result


async def brainstorming(topic: str, num_ideas: int = 5):
    """Brainstorm ideas on a topic."""
    from victor import Agent

    agent = Agent.create(temperature=0.7)
    result = await agent.run(
        f"Brainstorm {num_ideas} creative ideas for: {topic}\n\n"
        "Present as a numbered list with brief descriptions."
    )
    return result


async def code_explanation(code: str, language: str = "Python"):
    """Explain how code works."""
    from victor import Agent

    agent = Agent.create(temperature=0.3)
    result = await agent.run(
        f"Explain this {language} code line by line:\n\n{code}\n\n"
        "Be clear and beginner-friendly."
    )
    return result


async def concept_explanation(concept: str, audience: str = "beginner"):
    """Explain a concept for a specific audience."""
    from victor import Agent

    agent = Agent.create(temperature=0.4)
    result = await agent.run(
        f"Explain '{concept}' to a {audience}.\n\n"
        "Use simple language and relatable examples."
    )
    return result


async def email_drafting(topic: str, tone: str = "professional"):
    """Draft an email on a topic."""
    from victor import Agent

    agent = Agent.create(temperature=0.5)
    result = await agent.run(
        f"Write a {tone} email about: {topic}\n\n"
        "Include a clear subject line and proper structure."
    )
    return result


async def social_media_post(topic: str, platform: str = "Twitter"):
    """Create social media post content."""
    from victor import Agent

    agent = Agent.create(temperature=0.6)
    result = await agent.run(
        f"Write an engaging {platform} post about: {topic}\n\n"
        "Include relevant hashtags and keep it concise."
    )
    return result


async def list_generation(items: list, title: str = "List"):
    """Generate a formatted list from items."""
    from victor import Agent

    agent = Agent.create(temperature=0.2)
    items_str = ", ".join(str(item) for item in items)
    result = await agent.run(
        f"Create a formatted {title} from these items: {items_str}\n\n"
        "Include descriptions for each item."
    )
    return result


async def comparison(item_a: str, item_b: str):
    """Compare two items in detail."""
    from victor import Agent

    agent = Agent.create(temperature=0.4)
    result = await agent.run(
        f"Compare {item_a} vs {item_b}.\n\n"
        "Include similarities, differences, and use cases."
    )
    return result


async def proofreading(text: str):
    """Proofread and correct text."""
    from victor import Agent

    agent = Agent.create(temperature=0.2)
    result = await agent.run(
        f"Proofread this text and correct any errors:\n\n{text}\n\n"
        "Provide the corrected version and list of changes."
    )
    return result


async def title_generation(content: str):
    """Generate title for content."""
    from victor import Agent

    agent = Agent.create(temperature=0.5)
    result = await agent.run(
        f"Generate 5 catchy titles for:\n\n{content}\n\n"
        "Make them engaging and relevant."
    )
    return result


async def meeting_agenda(topics: list[str]):
    """Generate meeting agenda from topics."""
    from victor import Agent

    agent = Agent.create(temperature=0.3)
    topics_str = "\n".join(f"- {topic}" for topic in topics)
    result = await agent.run(
        f"Create a structured meeting agenda for these topics:\n{topics_str}\n\n"
        "Include time allocations and facilitator notes."
    )
    return result


async def feedback_provision(content: str, feedback_type: str = "constructive"):
    """Provide feedback on content."""
    from victor import Agent

    agent = Agent.create(temperature=0.4)
    result = await agent.run(
        f"Provide {feedback_type} feedback on:\n\n{content}\n\n"
        "Be specific, actionable, and supportive."
    )
    return result


async def conversation_practice(language: str = "Spanish"):
    """Practice conversation in a language."""
    from victor import Agent

    agent = Agent.create()
    result = await agent.run(
        f"Let's practice conversation in {language}. "
        "Start with a simple greeting and ask me questions."
    )
    return result


async def learning_plan(topic: str, duration: str = "1 week", level: str = "beginner"):
    """Generate learning plan for a topic."""
    from victor import Agent

    agent = Agent.create(temperature=0.4)
    result = await agent.run(
        f"Create a {level} learning plan for {topic}.\n\n"
        f"Duration: {duration}\n\n"
        "Include:\n"
        "- Learning objectives\n"
        "- Daily schedule\n"
        "- Resources needed\n"
        "- Milestones\n"
        "- Assessment methods"
    )
    return result


async def recipe_generation(ingredients: list[str], dietary: str = "none"):
    """Generate recipe from ingredients."""
    from victor import Agent

    agent = Agent.create(temperature=0.7)
    ingredients_str = ", ".join(ingredients)
    result = await agent.run(
        f"Create a recipe using: {ingredients_str}\n\n"
        f"Dietary restrictions: {dietary}\n\n"
        "Include ingredients list, steps, and cooking time."
    )
    return result


async def travel_itinerary(destination: str, days: int = 5):
    """Generate travel itinerary."""
    from victor import Agent

    agent = Agent.create(temperature=0.6)
    result = await agent.run(
        f"Create a {days}-day travel itinerary for {destination}.\n\n"
        "Include:\n"
        "- Day-by-day breakdown\n"
        "- Must-see attractions\n"
        "- Restaurant recommendations\n"
        "- Travel tips"
    )
    return result


async def book_summary(book: str):
    """Generate book summary."""
    from victor import Agent

    agent = Agent.create(temperature=0.4)
    result = await agent.run(
        f"Generate a summary for the book: {book}\n\n"
        "Include:\n"
        "- Main themes\n"
        "- Key characters\n"
        "- Plot summary\n"
        "- Key takeaways\n"
        "- Target audience"
    )
    return result


async def main():
    """Run all basic agent recipes."""
    recipes = [
        ("simple_qa", "Simple Q&A"),
        ("text_generation", "Text generation"),
        ("text_summarization", "Text summarization"),
        ("code_explanation", "Code explanation"),
    ]

    for recipe_func, recipe_name in recipes:
        print(f"\n{'='*60}")
        print(f"Recipe: {recipe_name}")
        print('='*60)

        try:
            result = await recipe_func()
            if hasattr(result, 'content'):
                print(result.content[:200] + "...")
            else:
                print(result)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
