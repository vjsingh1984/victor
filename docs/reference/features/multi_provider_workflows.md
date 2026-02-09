# Multi-Provider Workflows

## Overview

Victor's `OrchestratorPool` enables sophisticated multi-provider workflows, allowing you to strategically use different LLM providers for different tasks. This approach optimizes for cost, quality, and speed by leveraging each provider's strengths.

## Key Benefits

- **Cost Optimization**: Use free/cheap models for simple tasks, premium models for complex ones
- **Quality Optimization**: Use the best model for each specific task type
- **Performance**: Balance speed vs. quality based on task requirements
- **Flexibility**: Mix and match providers in a single workflow
- **Caching**: Automatic orchestrator caching avoids redundant initialization

## OrchestratorPool

The `OrchestratorPool` manages multiple orchestrators for different provider profiles:

```python
from victor.workflows.orchestrator_pool import OrchestratorPool
from victor.config.settings import Settings

# Initialize settings and pool
settings = Settings()
pool = OrchestratorPool(settings)

# Get orchestrator for specific profile
orchestrator = pool.get_orchestrator("my-profile")

# Get default orchestrator
default_orch = pool.get_default_orchestrator()
```text

## Profile Configuration

Configure multiple provider profiles in your settings:

```yaml
# profiles.yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 0.7

  fast:
    provider: openai
    model: gpt-4o-mini
    temperature: 0.5

  local:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.8

  premium:
    provider: anthropic
    model: claude-opus-4-5
    temperature: 0.7
```

## Common Use Cases

### 1. Cost-Optimized Workflow

Use free/cheap models for ideation and testing, premium models for critical reviews:

```python
import asyncio
from victor import Agent

async def cost_optimized_workflow():
    # Step 1: Brainstorm with Ollama (FREE)
    brainstormer = await Agent.create(
        provider="ollama",
        model="qwen2.5-coder:7b"
    )
    ideas = await brainstormer.run("Brainstorm 5 feature ideas")

    # Step 2: Implement with GPT-4o mini (CHEAP)
    implementer = await Agent.create(
        provider="openai",
        model="gpt-4o-mini"
    )
    implementation = await implementer.run(
        f"Implement this idea: {ideas.content[0]}"
    )

    # Step 3: Review with Claude (QUALITY)
    reviewer = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-5"
    )
    review = await reviewer.run(
        f"Review this code: {implementation.content}"
    )

    return review

asyncio.run(cost_optimized_workflow())
```text

### 2. Quality-Optimized Workflow

Use the best model for each task type:

```python
async def quality_optimized_workflow():
    # Research: Use model with large context
    researcher = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-5"
    )

    # Coding: Use model optimized for code
    coder = await Agent.create(
        provider="openai",
        model="gpt-4o"
    )

    # Testing: Use fast model for test generation
    tester = await Agent.create(
        provider="openai",
        model="gpt-4o-mini"
    )

    # Execute workflow
    research = await researcher.run("Research best practices for API design")
    code = await coder.run(f"Implement API based on: {research.content}")
    tests = await tester.run(f"Generate tests for: {code.content}")

    return tests
```

### 3. Performance-Optimized Workflow

Use fast models for rapid iteration:

```python
async def performance_workflow():
    # Generate multiple options in parallel
    fast_model = await Agent.create(
        provider="openai",
        model="gpt-4o-mini"
    )

    # Generate options
    tasks = [
        fast_model.run(f"Generate approach {i} for solving X")
        for i in range(5)
    ]
    options = await asyncio.gather(*tasks)

    # Evaluate with better model
    evaluator = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-5"
    )
    best = await evaluator.run(
        f"Evaluate these approaches and pick the best: {options}"
    )

    return best
```text

## OrchestratorPool API Reference

### Initialization

```python
from victor.workflows.orchestrator_pool import OrchestratorPool
from victor.config.settings import Settings

settings = Settings()
pool = OrchestratorPool(settings, container=None)
```

**Parameters:**
- `settings` (Settings): Application settings
- `container` (Optional[Any]): DI container for coordinator-based orchestrator

### Methods

#### get_orchestrator()

Get orchestrator for a specific profile:

```python
orchestrator = pool.get_orchestrator(profile="my-profile")
```text

**Parameters:**
- `profile` (Optional[str]): Profile name. Defaults to "default"

**Returns:**
- AgentOrchestrator instance

**Raises:**
- `ValueError`: If profile is not found
- `RuntimeError`: If orchestrator creation fails

**Note:** Orchestrators are cached. Subsequent calls return the same instance.

#### get_default_orchestrator()

Get orchestrator for the default profile:

```python
orchestrator = pool.get_default_orchestrator()
```

**Returns:**
- AgentOrchestrator instance for default profile

#### clear_cache()

Clear cached orchestrators and providers:

```python
# Clear specific profile
pool.clear_cache(profile="my-profile")

# Clear all profiles
pool.clear_cache()
```text

**Parameters:**
- `profile` (Optional[str]): Specific profile to clear, or None to clear all

#### get_cached_profiles()

Get list of profiles with cached orchestrators:

```python
cached = pool.get_cached_profiles()
print(f"Cached profiles: {cached}")
```

**Returns:**
- `list[str]`: List of profile names

#### shutdown()

Shutdown all orchestrators and providers:

```python
pool.shutdown()
```text

**Note:** Cleans up resources for all cached instances.

## Advanced Usage

### Custom Profile Configurations

```python
from pathlib import Path
from victor.config.settings import Settings

# Load settings from custom config
settings = Settings(
    config_file=Path("custom_profiles.yaml")
)

pool = OrchestratorPool(settings)
```

### Integration with Workflows

```python
from victor.workflows.orchestrator_pool import OrchestratorPool

class MultiProviderWorkflow:
    def __init__(self, pool: OrchestratorPool):
        self.pool = pool

    async def execute(self, task: str):
        # Use different profiles for different stages
        researcher = self.pool.get_orchestrator("research")
        coder = self.pool.get_orchestrator("coding")
        tester = self.pool.get_orchestrator("testing")

        # Execute workflow
        research = await researcher.run(task)
        code = await coder.run(f"Implement: {research}")
        tests = await tester.run(f"Test: {code}")

        return tests
```text

### Resource Management

```python
import asyncio

async def workflow_with_cleanup():
    pool = OrchestratorPool(settings)

    try:
        # Use orchestrators
        orchestrator = pool.get_orchestrator("profile1")
        result = await orchestrator.run("Do something")
        return result
    finally:
        # Clean up resources
        pool.shutdown()
```

## Best Practices

### 1. Profile Naming

Use descriptive profile names:

```yaml
profiles:
  brainstorming:    # Clear purpose
    provider: ollama
    model: qwen2.5-coder:7b

  implementation:   # Clear purpose
    provider: openai
    model: gpt-4o-mini

  review:           # Clear purpose
    provider: anthropic
    model: claude-sonnet-4-5
```text

### 2. Cost Monitoring

Track costs per profile:

```python
import asyncio
from victor import Agent

async def tracked_workflow():
    agents = {
        "brainstorm": await Agent.create(provider="ollama", model="qwen2.5"),
        "implement": await Agent.create(provider="openai", model="gpt-4o-mini"),
        "review": await Agent.create(provider="anthropic", model="claude-sonnet-4-5")
    }

    results = {}
    for stage, agent in agents.items():
        result = await agent.run(f"Execute {stage}")
        results[stage] = {
            "result": result,
            "tokens": result.usage.total_tokens,
            "cost": estimate_cost(stage, result.usage)
        }

    return results
```

### 3. Error Handling

Handle provider failures gracefully:

```python
async def resilient_workflow():
    profiles = ["ollama", "openai", "anthropic"]

    for profile in profiles:
        try:
            agent = await Agent.create(profile=profile)
            result = await agent.run("Do something")
            return result
        except Exception as e:
            print(f"{profile} failed: {e}")
            continue

    raise RuntimeError("All providers failed")
```text

### 4. Parallel Execution

Execute tasks in parallel across providers:

```python
async def parallel_workflow():
    providers = {
        "anthropic": ("claude-sonnet-4-5", "Review for security"),
        "openai": ("gpt-4o", "Review for performance"),
        "google": ("gemini-pro", "Review for accessibility")
    }

    agents = [
        await Agent.create(provider=p, model=m)
        for p, (m, _) in providers.items()
    ]

    tasks = [
        agent.run(task)
        for agent, (_, task) in zip(agents, providers.values())
    ]

    results = await asyncio.gather(*tasks)
    return results
```

## Cost Optimization Strategies

### Strategy 1: Funnel Approach

```text
Wide exploration (cheap) → Narrow implementation (medium) → Final review (premium)
```

- **Exploration**: Ollama (FREE) - Generate 10 ideas
- **Selection**: GPT-4o mini ($0.15/1M) - Pick top 3
- **Implementation**: GPT-4o ($2.50/1M) - Implement 1
- **Review**: Claude Sonnet ($3.00/1M) - Final review

**Cost**: ~$0.02 vs. $0.10 for all-premium approach (80% savings)

### Strategy 2: Tiered Quality

```text
Quick prototype (fast) → Iterative refinement (better) → Polish (best)
```

- **Prototype**: GPT-4o mini - Quick implementation
- **Refinement**: GPT-4o - Improve based on feedback
- **Polish**: Claude Sonnet - Final quality check

**Cost**: Balances speed and quality effectively

### Strategy 3: Specialized Models

```text
Research (large context) → Coding (code-optimized) → Testing (fast)
```

- **Research**: Claude (200K context) - Large document analysis
- **Coding**: GPT-4o - Optimized for code generation
- **Testing**: GPT-4o mini - Fast test generation

**Cost**: Uses each model's strengths

## Performance Considerations

### Cold Start vs. Cached

```python
import time

pool = OrchestratorPool(settings)

# First call (cold start)
start = time.time()
pool.get_orchestrator("profile1")
print(f"Cold start: {time.time() - start:.2f}s")

# Second call (cached)
start = time.time()
pool.get_orchestrator("profile1")
print(f"Cached: {time.time() - start:.2f}s")
```text

**Expected Output:**
```
Cold start: 1.23s
Cached: 0.001s
```text

### Memory Management

Monitor memory usage with multiple orchestrators:

```python
import psutil

def monitor_memory():
    process = psutil.Process()
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")

monitor_memory()
pool = OrchestratorPool(settings)

# Create multiple orchestrators
for profile in ["profile1", "profile2", "profile3"]:
    pool.get_orchestrator(profile)
    monitor_memory()

# Clear cache when done
pool.clear_cache()
monitor_memory()
```

## Troubleshooting

### Profile Not Found

**Error:** `ValueError: Profile 'my-profile' not found`

**Solution:** Ensure profile is defined in your settings YAML:

```yaml
profiles:
  my-profile:
    provider: openai
    model: gpt-4o-mini
```text

### Provider Initialization Failed

**Error:** `RuntimeError: Failed to create provider for profile 'my-profile'`

**Solution:** Check provider credentials and configuration:

```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### High Memory Usage

**Issue:** Memory grows with multiple orchestrators

**Solution:** Clear cache when profiles are no longer needed:

```python
# Clear specific profile
pool.clear_cache(profile="temp-profile")

# Clear all profiles
pool.clear_cache()
```text

## Examples

See the complete example:
- `/Users/vijaysingh/code/codingagent/examples/multi_provider_workflow.py`

Run it:
```bash
cd /Users/vijaysingh/code/codingagent
python examples/multi_provider_workflow.py
```

## Related Documentation

- [Provider Configuration](../user-guide/providers.md)
- [Workflow System](../user-guide/workflows.md)
- [Settings Reference](../reference/configuration/index.md)
- [Agent API](../reference/api/index.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
