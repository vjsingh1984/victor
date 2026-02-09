# Dynamic Personas Guide - Part 2

**Part 2 of 2:** Best Practices, Troubleshooting, Examples, and Additional Resources

---

## Navigation

- [Part 1: Understanding & Creating](part-1-understanding-creating.md)
- **[Part 2: Best Practices](#)** (Current)
- [**Complete Guide](../DYNAMIC_PERSONAS.md)**

---
## Best Practices

### 1. Choose Appropriate Personas

```python
# Good: Match persona to task
await personas.set_persona("security_auditor")
review = await orchestrator.chat("Review authentication code")

# Bad: Mismatched persona
await personas.set_persona("writer")
review = await orchestrator.chat("Review authentication code")
```

### 2. Define Clear System Prompts

```python
# Good: Clear, specific system prompt
system_prompt = """You are a security auditor specializing in web applications.
You check for:
- OWASP Top 10 vulnerabilities
- Authentication and authorization issues
- Input validation problems
- Data exposure risks

Provide specific, actionable recommendations."""

# Bad: Vague system prompt
system_prompt = "You are a security expert. Be helpful."
```

### 3. Use Triggers Effectively

```python
# Good: Specific triggers
triggers=[
    {
        "type": "keyword",
        "keywords": ["SQL injection", "XSS", "CSRF"],
        "confidence": 0.9
    }
]

# Bad: Too general triggers
triggers=[
    {
        "type": "keyword",
        "keywords": ["code", "review"],
        "confidence": 0.5
    }
]
```

### 4. Combine Personas with Features

```python
# Persona + Hierarchical Planning
await personas.set_persona("architect")
plan = await orchestrator.planning.plan_for_goal("Design microservices")

# Persona + Memory
await personas.set_persona("researcher")
await orchestrator.memory.store_episode(...)

# Persona + Skills
await personas.set_persona("debugger")
tools = await orchestrator.skills.discover_tools("debugging")
```

### 5. Test Personas

```python
# Test persona behavior
await personas.set_persona("my_custom_persona")

test_queries = [
    "What can you help me with?",
    "Review this code",
    "Explain this concept"
]

for query in test_queries:
    response = await orchestrator.chat(query)
    print(f"Query: {query}")
    print(f"Response: {response}\n")

# Evaluate and refine
```

## Troubleshooting

### Persona Not Applied

**Problem**: Persona doesn't seem to affect responses.

**Solutions**:
1. **Check persona is set**: Verify current persona
2. **Review system prompt**: Ensure it's specific enough
3. **Test with simple query**: Verify behavior change
4. **Check conflicts**: Ensure no conflicting settings

```python
# Verify persona
current = await personas.get_current_persona()
print(f"Current persona: {current.name}")

# Test behavior
await personas.set_persona("teacher")
response = await orchestrator.chat("Explain recursion")
# Should include educational elements
```

### Poor Persona Suggestions

**Problem**: Wrong persona suggested for context.

**Solutions**:
1. **Improve triggers**: Add specific keywords
2. **Adjust confidence thresholds**: Tune suggestion logic
3. **Provide more context**: Give more detailed context
4. **Manually override**: Set persona explicitly

```python
# Better triggers
triggers=[
    {
        "type": "keyword",
        "keywords": ["authentication", "authorization", "JWT", "OAuth"],
        "confidence": 0.9
    }
]

# Provide more context
suggested = await personas.suggest_persona(
    context="I need to implement JWT authentication for a REST API",
    task_type="implementation",
    domain="security"
)
```

### Persona Conflicts

**Problem**: Multiple personas suggested for same context.

**Solutions**:
1. **Review triggers**: Check for overlapping keywords
2. **Adjust confidence**: Use higher confidence thresholds
3. **Create composite persona**: Combine aspects
4. **Manual selection**: Choose explicitly

```python
# Create composite persona
await personas.create_persona(
    name="fullstack_reviewer",
    description="Reviews both frontend and backend",
    system_prompt="...",
    triggers=[
        {"keywords": ["fullstack", "frontend", "backend"], "confidence": 0.9}
    ]
)
```

## Examples

### Example 1: Task-Specific Personas

```python
async def task_specific_execution(task):
    # Analyze task type
    if "security" in task.lower():
        await personas.set_persona("security_auditor")
    elif "performance" in task.lower():
        await personas.set_persona("performance_optimizer")
    elif "design" in task.lower():
        await personas.set_persona("architect")
    else:
        await personas.set_persona("assistant")

    # Execute with appropriate persona
    result = await orchestrator.chat(task)
    return result
```

### Example 2: Progressive Persona Workflow

```python
async def progressive_workflow(initial_task):
    # Start with researcher
    await personas.set_persona("researcher")

    research = await orchestrator.chat(
        f"Research best practices for: {initial_task}"
    )

    # Switch to architect
    await personas.set_persona("architect")

    design = await orchestrator.chat(
        f"Design solution based on research: {research}",
        context={"research": research}
    )

    # Switch to debugger
    await personas.set_persona("debugger")

    implementation = await orchestrator.chat(
        f"Implement the design: {design}",
        context={"design": design}
    )

    # Switch to code_reviewer
    await personas.set_persona("code_reviewer")

    review = await orchestrator.chat(
        f"Review the implementation: {implementation}"
    )

    return {
        "research": research,
        "design": design,
        "implementation": implementation,
        "review": review
    }
```

### Example 3: Adaptive Learning Persona

```python
# Create adaptive learning persona
await personas.create_persona(
    name="adaptive_tutor",
    description="Adapts teaching style based on learner responses",
    system_prompt="You are an adaptive tutor...",
    behavior_fn=adaptive_tutoring_behavior
)

async def adaptive_tutoring_behavior(context):
    # Get user's current level
    user_level = context.get("user_level", "beginner")

    # Adjust explanation complexity
    if user_level == "beginner":
        context["explanation_style"] = "simple with examples"
    elif user_level == "intermediate":
        context["explanation_style"] = "balanced theory and practice"
    else:
        context["explanation_style"] = "advanced with edge cases"

    # Generate response
    response = await orchestrator.generate_response(context)

    # Ask follow-up question to assess understanding
    response += "\n\nDid you understand this? (yes/no)"

    return response
```

### Example 4: Specialized Domain Persona

```python
# Create specialized persona
await personas.create_persona(
    name="devops_engineer",
    description="Expert in DevOps and infrastructure",
    system_prompt="""You are a DevOps engineer specializing in:
- CI/CD pipelines
- Container orchestration (Kubernetes, Docker)
- Infrastructure as Code (Terraform, CloudFormation)
- Monitoring and observability
- Cloud platforms (AWS, GCP, Azure)

You provide practical, production-ready guidance.""",
    capabilities=[
        "ci_cd",
        "containers",
        "infrastructure",
        "monitoring",
        "cloud"
    ],
    metadata={"certifications": ["AWS", "CKA"]},
    triggers=[
        {"keywords": ["deploy", "pipeline", "kubernetes", "docker"], "confidence": 0.9}
    ]
)

# Use for DevOps tasks
await personas.set_persona("devops_engineer")

pipeline = await orchestrator.chat(
    "Create a CI/CD pipeline for a Python application"
)
```

## Additional Resources

- [API Reference](../api/NEW_CAPABILITIES_API.md)
- [User Guide](../user-guide/index.md)
- Persona Management (not yet published)
- Persona Templates (not yet published)

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
