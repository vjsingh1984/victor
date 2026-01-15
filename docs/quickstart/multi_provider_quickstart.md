# Quick Start: Multi-Provider Workflows

## What You'll Learn

In this guide, you'll learn how to:
- Set up multiple provider profiles
- Use OrchestratorPool to manage multiple orchestrators
- Create cost-optimized workflows
- Mix and match providers for different tasks

## Prerequisites

- Victor installed: `pip install victor-ai`
- API keys configured for providers you want to use
- Basic understanding of Victor workflows

## Step 1: Configure Multiple Profiles

Create a `profiles.yaml` file:

```yaml
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
```

Set environment variables:

```bash
# Anthropic (for default profile)
export ANTHROPIC_API_KEY="your-anthropic-key"

# OpenAI (for fast profile)
export OPENAI_API_KEY="your-openai-key"

# Ollama (for local profile - no API key needed)
# Just make sure Ollama is running: ollama serve
```

## Step 2: Basic Multi-Provider Workflow

Create `multi_provider_example.py`:

```python
import asyncio
from victor import Agent

async def cost_optimized_workflow():
    """Use different providers for different tasks."""

    # Task 1: Brainstorm with Ollama (FREE)
    print("📝 Brainstorming with Ollama...")
    brainstormer = await Agent.create(
        provider="ollama",
        model="qwen2.5-coder:7b"
    )
    ideas = await brainstormer.run(
        "Brainstorm 3 Python function names for email validation. "
        "Just list the names, one per line."
    )
    print(f"Ideas: {ideas.content}")

    # Task 2: Implement with GPT-4o mini (CHEAP)
    print("\n💻 Implementing with GPT-4o mini...")
    implementer = await Agent.create(
        provider="openai",
        model="gpt-4o-mini"
    )
    code = await implementer.run(
        "Write a Python function to validate email addresses. "
        "Use regex and include a docstring."
    )
    print(f"Code: {code.content[:200]}...")

    # Task 3: Review with Claude (QUALITY)
    print("\n🔍 Reviewing with Claude...")
    reviewer = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-5"
    )
    review = await reviewer.run(
        f"Review this code for security and best practices:\n\n{code.content}"
    )
    print(f"Review: {review.content[:200]}...")

    # Cleanup
    await brainstormer.close()
    await implementer.close()
    await reviewer.close()

asyncio.run(cost_optimized_workflow())
```

Run it:

```bash
python multi_provider_example.py
```

## Step 3: Using OrchestratorPool

For more control, use the OrchestratorPool directly:

```python
from victor.workflows.orchestrator_pool import OrchestratorPool
from victor.config.settings import Settings

# Initialize pool
settings = Settings()
pool = OrchestratorPool(settings)

# Get orchestrators for different profiles
ollama_orch = pool.get_orchestrator("local")
openai_orch = pool.get_orchestrator("fast")
anthropic_orch = pool.get_orchestrator("default")

# Use orchestrators
print("Cached profiles:", pool.get_cached_profiles())

# Cleanup when done
pool.shutdown()
```

## Step 4: Advanced Workflow Pattern

Create a sophisticated multi-stage workflow:

```python
async def advanced_workflow():
    """Advanced workflow with parallel execution."""

    # Stage 1: Generate options in parallel
    print("🎯 Stage 1: Generate options...")
    fast_agent = await Agent.create(
        provider="openai",
        model="gpt-4o-mini"
    )

    tasks = [
        fast_agent.run(f"Generate approach {i} for X")
        for i in range(3)
    ]
    options = await asyncio.gather(*tasks)
    print(f"Generated {len(options)} approaches")

    # Stage 2: Evaluate with better model
    print("\n📊 Stage 2: Evaluate approaches...")
    evaluator = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-5"
    )

    options_text = "\n".join([
        f"Approach {i}: {opt.content}"
        for i, opt in enumerate(options)
    ])

    evaluation = await evaluator.run(
        f"Evaluate these approaches and pick the best:\n\n{options_text}"
    )
    print(f"Best approach: {evaluation.content[:200]}...")

    await fast_agent.close()
    await evaluator.close()

asyncio.run(advanced_workflow())
```

## Cost Comparison

See the cost savings:

```
Scenario: Build a feature with review

All-Premium Approach:
- Brainstorm: Claude (200 tokens) = $0.0006
- Implement: Claude (1000 tokens) = $0.003
- Review: Claude (500 tokens) = $0.0015
- Total: $0.0051

Multi-Provider Approach:
- Brainstorm: Ollama (200 tokens) = FREE
- Implement: GPT-4o mini (1000 tokens) = $0.00015
- Review: Claude (500 tokens) = $0.0015
- Total: $0.00165

Savings: 68% reduction in cost!
```

## Common Patterns

### Pattern 1: Funnel Approach

```
Many cheap options → Few medium options → One premium choice
```

```python
# Generate 10 ideas with Ollama (FREE)
# Select top 3 with GPT-4o mini (CHEAP)
# Implement 1 with GPT-4o (MEDIUM)
# Review with Claude (PREMIUM)
```

### Pattern 2: Specialized Models

```
Use each model for its strength
```

```python
# Research: Claude (200K context for large docs)
# Coding: GPT-4o (optimized for code)
# Testing: GPT-4o mini (fast test generation)
```

### Pattern 3: Iterative Refinement

```
Quick prototype → Better implementation → Final polish
```

```python
# Prototype: GPT-4o mini (quick draft)
# Refine: GPT-4o (improve based on feedback)
# Polish: Claude (final quality check)
```

## Troubleshooting

### Issue: Provider not available

```python
try:
    agent = await Agent.create(provider="ollama")
except Exception as e:
    print(f"Ollama not available: {e}")
    # Fallback to another provider
    agent = await Agent.create(provider="openai", model="gpt-4o-mini")
```

### Issue: High memory usage

```python
# Clear orchestrator cache when done
pool.clear_cache(profile="temp-profile")

# Or clear all
pool.clear_cache()
```

### Issue: Slow initialization

```python
# Orchestrators are cached after first use
pool = OrchestratorPool(settings)

# First call (slow)
orch1 = pool.get_orchestrator("profile1")

# Second call (fast - cached)
orch2 = pool.get_orchestrator("profile1")
```

## Next Steps

1. **Read full documentation**: `docs/features/multi_provider_workflows.md`
2. **Run complete example**: `examples/multi_provider_workflow.py`
3. **Explore profiles**: Configure custom profiles for your use case
4. **Optimize costs**: Track usage and optimize provider selection

## Summary

✅ You learned:
- How to configure multiple provider profiles
- How to use OrchestratorPool for managing orchestrators
- Common multi-provider workflow patterns
- Cost optimization strategies
- Error handling and troubleshooting

You're now ready to build sophisticated multi-provider workflows!
