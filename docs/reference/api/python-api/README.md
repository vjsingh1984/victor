# Python API Reference

Python package API for integrating Victor into Python applications.

---

## Parts

- **[Part 1: Core & Advanced API](part-1-core-advanced-api.md)** - Installation, core Agent API, task management, tools, state management, multi-agent, events, configuration, and error handling
- **[Part 2: Testing, Examples & Best Practices](part-2-testing-examples.md)** - Unit testing, mocking, integration tests, code examples, and best practices

---

## Quick Start

```python
import asyncio
from victor import Agent

async def main():
    # Create agent
    agent = await Agent.create(provider="anthropic")

    # Run task
    result = await agent.run("Write a REST API with FastAPI")
    print(result)

asyncio.run(main())
```text

---

## Installation

```bash
# Install Victor
pip install victor-ai

# With dev dependencies
pip install "victor-ai[dev]"

# From source
pip install -e .
```text

---

## See Also

- [Providers API](../internals/providers-api.md)
- [Tools API](../internals/tools-api.md)
- [API Reference Home](../README.md)
- [Documentation Home](../../README.md)


**Reading Time:** 22 min (both parts)
**Last Updated:** February 08, 2026**
