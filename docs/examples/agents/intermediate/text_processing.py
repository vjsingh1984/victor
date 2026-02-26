"""Text processing examples."""

import asyncio
from victor import Agent


async def summarize_text(text: str, max_length: int = 100):
    """Summarize text to max length."""
    agent = Agent.create(temperature=0.3)
    result = await agent.run(
        f"Summarize this in {max_length} words or less:\n\n{text}"
    )
    return result.content


async def extract_entities(text: str):
    """Extract named entities from text."""
    agent = Agent.create(temperature=0.2)
    result = await agent.run(
        f"""Extract named entities from this text:
        {text}

        Identify:
        - People
        - Organizations
        - Locations
        - Dates
        - Numbers"""
    )
    return result.content


async def sentiment_analysis(text: str):
    """Analyze sentiment of text."""
    agent = Agent.create(temperature=0.2)
    result = await agent.run(
        f"""Analyze the sentiment of this text:
        "{text}"

        Provide:
        1. Overall sentiment (positive/negative/neutral)
        2. Confidence score (0-1)
        3. Key phrases that indicate sentiment"""
    )
    return result.content


async def translate_text(text: str, target_language: str = "Spanish"):
    """Translate text to target language."""
    agent = Agent.create(temperature=0.3)
    result = await agent.run(
        f"Translate this to {target_language}:\n\n{text}"
    )
    return result.content


async def main():
    """Run text processing examples."""
    text = "Victor is an amazing AI framework that helps developers build agents."

    print("=== Text Processing Examples ===\n")

    print("1. Summarization:")
    summary = await summarize_text(text, 10)
    print(f"   {summary}\n")

    print("2. Entity Extraction:")
    entities = await extract_entities(text)
    print(f"   {entities}\n")

    print("3. Sentiment Analysis:")
    sentiment = await sentiment_analysis(text)
    print(f"   {sentiment}\n")

    print("4. Translation:")
    translation = await translate_text(text, "French")
    print(f"   {translation}\n")


if __name__ == "__main__":
    asyncio.run(main())
