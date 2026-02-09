# Dynamic Skills Guide - Part 2

**Part 2 of 2:** Best Practices, Troubleshooting, and Examples

---

## Navigation

- [Part 1: Discovery, Composition, Chaining](part-1-discovery-composition-chaining-adaptation.md)
- **[Part 2: Best Practices, Troubleshooting, Examples](#)** (Current)
- [**Complete Guide**](../DYNAMIC_SKILLS.md)

---

## Best Practices

### 1. Use Semantic Matching

Leverage semantic similarity for tool selection:

```python
# Good: Semantic matching
skill = await discover_skill(
    task="analyze python code",
    strategy="semantic"
)

# Avoid: Exact keyword matching
skill = await discover_skill(
    task="analyze code",
    strategy="keyword"
)
```text

### 2. Chain Intelligently

Plan before chaining skills:

```python
# Plan first
plan = await plan_skill_chain(task="deploy to production")

# Review plan
print(plan.steps)

# Execute if acceptable
if confirm_plan(plan):
    result = await execute_skill_chain(plan)
```

### 3. Handle Failures Gracefully

Implement fallback logic:

```python
try:
    result = await execute_skill_chain(plan)
except SkillExecutionError as e:
    # Try fallback
    fallback = get_fallback_skill(e.failed_skill)
    result = await fallback.execute(**e.params)
```text

### 4. Cache Discovered Skills

Cache skill discovery results:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def get_skill_for_task(task: str):
    return await discover_skill(task)
```

---

## Troubleshooting

### Skills Not Discovered

**Problem**: No skills found for task.

**Solutions:**
1. **Check skill registry**: Ensure skills are registered
2. **Adjust similarity threshold**: Lower threshold for more matches
3. **Improve skill descriptions**: Better descriptions improve matching
4. **Use different strategy**: Try keyword or hybrid matching

```python
# Lower threshold
skill = await discover_skill(
    task="my task",
    threshold=0.3  # Lower from 0.5
)

# Try hybrid strategy
skill = await discover_skill(
    task="my task",
    strategy="hybrid"
)
```text

### Chain Failures

**Problem**: Skill chain fails midway.

**Solutions:**
1. **Check dependencies**: Ensure skills have required dependencies
2. **Validate parameters**: Check if parameters are correct
3. **Implement retries**: Retry failed skills
4. **Use checkpoints**: Save state for recovery

```python
# Retry with backoff
result = await execute_skill_chain(
    plan,
    max_retries=3,
    backoff=ExponentialBackoff()
)

# Use checkpoints
result = await execute_skill_chain(
    plan,
    checkpoints=True
)
```

### Performance Issues

**Problem**: Skill discovery is slow.

**Solutions:**
1. **Cache results**: Cache discovery results
2. **Limit search scope**: Reduce number of skills to search
3. **Use faster embeddings**: Use smaller/faster embedding models
4. **Parallel discovery**: Discover skills in parallel

```python
# Cache results
@lru_cache(maxsize=100)
async def discover_skill_cached(task):
    return await discover_skill(task)

# Limit scope
skill = await discover_skill(
    task="my task",
    categories=["analysis", "processing"]
)
```text

---

## Examples

### Example 1: Discover and Execute Skill

```python
from victor.skills import discover_skill, execute_skill

# Discover skill for task
skill = await discover_skill(
    task="analyze python code for bugs"
)

# Execute skill
result = await execute_skill(
    skill=skill,
    code="my_script.py",
    severity="medium"
)

print(result.analysis)
```

### Example 2: Chain Multiple Skills

```python
from victor.skills import plan_skill_chain, execute_skill_chain

# Plan skill chain
plan = await plan_skill_chain(
    task="test, lint, and format python code"
)

# Review plan
for step in plan.steps:
    print(f"{step.skill_name}: {step.description}")

# Execute chain
result = await execute_skill_chain(plan)

print(f"Success: {result.success}")
print(f"Output: {result.output}")
```text

### Example 3: Compose Custom Skill

```python
from victor.skills import compose_skill

# Compose skill from tools
skill = compose_skill(
    name="code_review",
    description="Review code for quality issues",
    tools=["read", "pylint", "mypy"],
    workflow=[
        ("read", "source_code"),
        ("pylint", "linting"),
        ("mypy", "type_checking")
    ]
)

# Register skill
await register_skill(skill)

# Use skill
result = await execute_skill(
    skill="code_review",
    source_code="my_script.py"
)
```

---

## Additional Resources

- [Skill API Reference](../../reference/skills/README.md)
- [Tool Registry](../../reference/tools/README.md)
- [MCP Integration](../../integrations/MCP_INTEGRATION.md)

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 2 min
**Last Updated:** February 01, 2026
