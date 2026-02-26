"""Basic agent examples."""

import asyncio
from victor import Agent


async def basic_qa():
    """Simple question and answer."""
    agent = Agent.create()
    result = await agent.run("What is the capital of France?")
    return result


async def with_custom_prompt():
    """Agent with custom system prompt."""
    agent = Agent.create(
        system_prompt="You are a helpful assistant who speaks in pirate speak."
    )
    result = await agent.run("Hello!")
    return result


async def multi_turn():
    """Multi-turn conversation."""
    agent = Agent.create()
    await agent.chat("My name is Alice")
    result = await agent.chat("What's my name?")
    return result


async def streaming_response():
    """Stream response in real-time."""
    agent = Agent.create()
    chunks = []
    async for event in agent.stream("Tell me a short joke"):
        if event.type == "content":
            chunks.append(event.content)
    return "".join(chunks)


async def max_tokens():
    """Limit response length."""
    agent = Agent.create(max_tokens=50)
    result = await agent.run("Tell me a long story")
    return result
