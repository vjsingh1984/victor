"""
Production Patterns Example

Shows production-ready patterns for using Victor agents.
"""

import asyncio
import logging
from typing import Optional
from victor import Agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionAgent:
    """Production-ready agent wrapper with error handling, retries, and monitoring."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.agent: Optional[Agent] = None
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0
        }

    async def initialize(self):
        """Initialize the agent."""
        self.agent = Agent.create(
            provider=self.provider,
            model=self.model,
            enable_observability=True
        )
        logger.info(f"Agent initialized: {self.provider}/{self.model}")

    async def run_with_retry(
        self,
        prompt: str,
        temperature: float = 0.7
    ) -> str:
        """Run agent with automatic retry on failure."""
        if not self.agent:
            await self.initialize()

        self.metrics["total_calls"] += 1
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}")
                result = await asyncio.wait_for(
                    self.agent.run(prompt),
                    timeout=self.timeout
                )

                self.metrics["successful_calls"] += 1
                logger.info("Call successful")
                return result.content

            except asyncio.TimeoutError:
                last_error = "Timeout"
                logger.warning(f"Attempt {attempt + 1} timed out")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

        self.metrics["failed_calls"] += 1
        logger.error(f"All attempts failed. Last error: {last_error}")
        raise Exception(f"Agent failed after {self.max_retries} attempts: {last_error}")

    async def run_streaming(self, prompt: str):
        """Run agent with streaming output."""
        if not self.agent:
            await self.initialize()

        logger.info("Starting streaming call")
        chunks = []

        try:
            async for event in self.agent.stream(prompt):
                if event.type == "content":
                    chunks.append(event.content)
                    # Process chunk in real-time
                    # (e.g., send to client, update UI, etc.)

            self.metrics["successful_calls"] += 1
            return "".join(chunks)

        except Exception as e:
            self.metrics["failed_calls"] += 1
            logger.error(f"Streaming call failed: {e}")
            raise

    def get_metrics(self) -> dict:
        """Get agent metrics."""
        if self.metrics["total_calls"] > 0:
            self.metrics["success_rate"] = (
                self.metrics["successful_calls"] / self.metrics["total_calls"]
            )
        return self.metrics

    async def cleanup(self):
        """Cleanup resources."""
        # Close connections, etc.
        logger.info("Agent cleaned up")


class AgentPool:
    """Pool of agents for concurrent processing."""

    def __init__(self, pool_size: int = 3):
        self.pool_size = pool_size
        self.agents = []
        self.semaphore = asyncio.Semaphore(pool_size)

    async def initialize(self):
        """Initialize agent pool."""
        for i in range(self.pool_size):
            agent = ProductionAgent(model="gpt-3.5-turbo")  # Faster model
            await agent.initialize()
            self.agents.append(agent)
        logger.info(f"Agent pool initialized with {self.pool_size} agents")

    async def process_batch(self, prompts: list[str]) -> list[str]:
        """Process multiple prompts concurrently."""
        async def process_with_agent(agent: ProductionAgent, prompt: str):
            async with self.semaphore:
                return await agent.run_with_retry(prompt)

        tasks = []
        for i, prompt in enumerate(prompts):
            agent = self.agents[i % len(self.agents)]
            tasks.append(process_with_agent(agent, prompt))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Prompt {i} failed: {result}")
                final_results.append(f"Error: {str(result)}")
            else:
                final_results.append(result)

        return final_results

    async def cleanup(self):
        """Cleanup all agents."""
        for agent in self.agents:
            await agent.cleanup()


class CircuitBreaker:
    """Circuit breaker for failing agents."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, agent_func, *args, **kwargs):
        """Execute agent function with circuit breaker."""
        if self.state == "open":
            if await self._should_attempt_reset():
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await agent_func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failures = 0
        if self.state == "half-open":
            self.state = "closed"
            logger.info("Circuit breaker reset to CLOSED")

    def _on_failure(self):
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = asyncio.get_event_loop().time()

        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failures} failures")

    async def _should_attempt_reset(self) -> bool:
        """Check if circuit should be reset."""
        if self.last_failure_time is None:
            return True
        return (asyncio.get_event_loop().time() - self.last_failure_time) > self.timeout


async def main():
    """Run production pattern examples."""
    print("=== Production Patterns ===\n")

    # Example 1: Production agent with retry
    print("1. Production Agent with Retry:")
    print("-" * 50)

    agent = ProductionAgent(max_retries=3)
    await agent.initialize()

    try:
        result = await agent.run_with_retry(
            "Explain asyncio in Python in one paragraph"
        )
        print(f"Result: {result}\n")
    finally:
        await agent.cleanup()

    # Example 2: Agent pool
    print("\n2. Agent Pool for Batch Processing:")
    print("-" * 50)

    pool = AgentPool(pool_size=3)
    await pool.initialize()

    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
        "What is Java?"
    ]

    try:
        results = await pool.process_batch(prompts)
        for i, result in enumerate(results):
            print(f"\nPrompt {i+1}: {prompts[i]}")
            print(f"Result: {result[:100]}...")
    finally:
        await pool.cleanup()

    # Example 3: Circuit breaker
    print("\n\n3. Circuit Breaker Pattern:")
    print("-" * 50)

    circuit_breaker = CircuitBreaker(failure_threshold=2)
    agent = ProductionAgent()

    async def failing_call():
        raise Exception("Simulated failure")

    # Test circuit breaker
    for i in range(4):
        try:
            await circuit_breaker.call(failing_call)
        except Exception as e:
            print(f"Call {i+1}: {e}")
            print(f"Circuit breaker state: {circuit_breaker.state}")


if __name__ == "__main__":
    asyncio.run(main())
