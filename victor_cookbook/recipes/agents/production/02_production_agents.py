"""Production-ready agent recipes.

These recipes include enterprise patterns like retry logic,
error handling, monitoring, and observability.
"""

RECIPE_CATEGORY = "agents/production"
RECIPE_DIFFICULTY = "advanced"
RECIPE_TIME = "15 minutes"


class ProductionAgent:
    """Production-ready agent wrapper with error handling and monitoring."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        max_retries: int = 3,
        timeout: int = 30,
        enable_metrics: bool = True
    ):
        """Initialize production agent.

        Args:
            provider: LLM provider
            model: Model name
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            enable_metrics: Enable metrics collection
        """
        from victor import Agent

        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_metrics = enable_metrics
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration_ms": 0
        }
        self._agent = None

    async def initialize(self):
        """Initialize the agent."""
        from victor import Agent

        self._agent = Agent.create(
            provider=self.provider,
            model=self.model,
            enable_observability=self.enable_metrics
        )

    async def run_with_retry(self, prompt: str, **kwargs) -> str:
        """Run agent with automatic retry on failure.

        Args:
            prompt: Prompt to send
            **kwargs: Additional arguments for agent.run()

        Returns:
            Agent response content

        Raises:
            Exception: If all retries are exhausted
        """
        if not self._agent:
            await self.initialize()

        import asyncio
        import time

        last_error = None
        self.metrics["total_calls"] += 1

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                result = await asyncio.wait_for(
                    self._agent.run(prompt, **kwargs),
                    timeout=self.timeout
                )

                duration_ms = (time.time() - start_time) * 1000
                self.metrics["total_duration_ms"] += duration_ms
                self.metrics["successful_calls"] += 1

                return result.content

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
            except Exception as e:
                last_error = str(e)

            if attempt < self.max_retries - 1:
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

        self.metrics["failed_calls"] += 1
        raise Exception(f"Agent failed after {self.max_retries} attempts: {last_error}")

    async def get_metrics(self) -> dict:
        """Get current metrics.

        Returns:
            Metrics dictionary
        """
        if self.metrics["total_calls"] > 0:
            self.metrics["success_rate"] = (
                self.metrics["successful_calls"] / self.metrics["total_calls"]
            )
            self.metrics["avg_duration_ms"] = (
                self.metrics["total_duration_ms"] / self.metrics["total_calls"]
            )
        return self.metrics.copy()

    async def health_check(self) -> bool:
        """Check if agent is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            result = await self.run_with_retry("Health check", timeout=5)
            return bool(result)
        except Exception:
            return False


async def main():
    """Demonstrate production agent."""
    agent = ProductionAgent(max_retries=3, timeout=10)
    await agent.initialize()

    try:
        result = await agent.run_with_retry(
            "What is async/await in Python?",
            temperature=0.3
        )
        print(f"Result: {result}")

        metrics = await agent.get_metrics()
        print(f"Metrics: {metrics}")
    finally:
        print(f"Final metrics: {agent.metrics}")


class StreamingAgent:
    """Agent with streaming support and real-time updates."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        """Initialize streaming agent.

        Args:
            provider: LLM provider
            model: Model name
        """
        from victor import Agent

        self._agent = Agent.create(provider=provider, model=model)
        self.chunks = []

    async def stream_and_collect(self, prompt: str) -> str:
        """Stream response and collect chunks.

        Args:
            prompt: Prompt to send

        Returns:
            Complete response
        """
        self.chunks = []

        async for event in self._agent.stream(prompt):
            if event.type == "content":
                self.chunks.append(event.content)
                # Could emit to websocket, write to file, etc.

        return "".join(self.chunks)

    async def stream_to_file(self, prompt: str, output_path: str):
        """Stream response to file.

        Args:
            prompt: Prompt to send
            output_path: Path to output file
        """
        import aiofiles

        async with aiofiles.open(output_path, "w") as f:
            async for event in self._agent.stream(prompt):
                if event.type == "content":
                    await f.write(event.content)

    async def stream_with_progress(self, prompt: str):
        """Stream with progress updates.

        Args:
            prompt: Prompt to send

        Yields:
            Tuples of (chunk_number, chunk_content)
        """
        chunk_num = 0
        async for event in self._agent.stream(prompt):
            if event.type == "content":
                chunk_num += 1
                yield (chunk_num, event.content)
            elif event.type == "thinking":
                print(f"[Thinking...]")


async def error_handling_wrapper(agent_func, *args, fallback_value=None):
    """Wrap agent call with error handling.

    Args:
        agent_func: Agent function to call
        *args: Arguments for function
        fallback_value: Value to return on error

    Returns:
        Function result or fallback value
    """
    try:
        return await agent_func(*args)
    except Exception as e:
        import logging
        logging.error(f"Agent call failed: {e}")
        return fallback_value


async def fallback_chain(agent_factory, prompts: list[str]):
    """Try multiple agents/providers as fallback.

    Args:
        agent_factory: Function that creates agents
        prompts: List of prompts to try

    Returns:
        First successful result
    """
    import logging

    for i, prompt in enumerate(prompts):
        try:
            agent = agent_factory()
            result = await agent.run(prompt, timeout=10)
            return result
        except Exception as e:
            logging.warning(f"Attempt {i+1} failed: {e}")

    raise Exception("All fallback attempts failed")


class CircuitBreaker:
    """Circuit breaker for failing agents."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            timeout: Seconds before attempting reset
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, agent_func, *args, **kwargs):
        """Execute function through circuit breaker.

        Args:
            agent_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
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

    def _on_failure(self):
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = asyncio.get_event_loop().time()

        if self.failures >= self.failure_threshold:
            self.state = "open"

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should be reset."""
        if self.last_failure_time is None:
            return True
        import time
        return (time.time() - self.last_failure_time) > self.timeout


class MultiAgentSystem:
    """Multi-agent system with specialized agents."""

    def __init__(self, agents: dict):
        """Initialize multi-agent system.

        Args:
            agents: Dictionary of agent name to agent instance
        """
        self.agents = agents

    async def parallel_consensus(self, prompt: str) -> dict:
        """Get consensus from all agents in parallel.

        Args:
            prompt: Prompt to send to all agents

        Returns:
            Dictionary of agent responses
        """
        import asyncio

        tasks = []
        for name, agent in self.agents.items():
            tasks.append(self._run_agent(name, agent, prompt))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            name: result if not isinstance(result, Exception) else str(result)
            for name, result in zip(self.agents.keys(), results)
        }

    async def _run_agent(self, name: str, agent, prompt: str):
        """Run a single agent with error handling."""
        try:
            result = await agent.run(prompt)
            return result.content
        except Exception as e:
            return f"Error in {name}: {str(e)}"

    async def sequential_pipeline(self, stages: list[tuple]):
        """Run agents in sequence, passing output between stages.

        Args:
            stages: List of (agent_name, prompt_func) tuples

        Returns:
            Final result
        """
        current_context = {}

        for agent_name, prompt_func in stages:
            agent = self.agents[agent_name]
            prompt = prompt_func(current_context)
            result = await agent.run(prompt)
            current_context[agent_name] = result.content

        return current_context


async def demo_production_agent():
    """Demonstrate production agent."""
    print("=== Production Agent Demo ===\n")

    agent = ProductionAgent(max_retries=2, timeout=15)
    await agent.initialize()

    try:
        result = await agent.run_with_retry("Explain microservices in 3 sentences.")
        print(f"Result: {result}\n")

        metrics = await agent.get_metrics()
        print(f"Metrics: {metrics}")
    finally:
        print(f"Final metrics: {agent.metrics}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_production_agent())
