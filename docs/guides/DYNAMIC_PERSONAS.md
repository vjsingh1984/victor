# Dynamic Personas Guide

## Overview

Victor AI's dynamic personas enable context-adaptive agent personalities for specialized tasks. This guide explains how to use and create personas effectively.

## Table of Contents

- [What are Dynamic Personas?](#what-are-dynamic-personas)
- [Using Built-in Personas](#using-built-in-personas)
- [Creating Custom Personas](#creating-custom-personas)
- [Context Adaptation](#context-adaptation)
- [Persona Management](#persona-management)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## What are Dynamic Personas?

Personas define Victor's personality, communication style, and expertise areas.

### Key Features

- **Pre-built Personas**: Ready-to-use personas for common tasks
- **Dynamic Switching**: Change personas based on context
- **Custom Creation**: Define your own personas
- **Context-Aware**: Automatic persona suggestions
- **Persistence**: Save and reuse personas

### Available Built-in Personas

| Persona | Description | Best For |
|---------|-------------|----------|
| `assistant` | General-purpose helper | Everyday tasks |
| `code_reviewer` | Code quality and best practices | Reviewing code |
| `security_auditor` | Security vulnerability specialist | Security reviews |
| `architect` | System design and architecture | Design decisions |
| `teacher` | Educational and explanatory | Learning and tutoring |
| `researcher` | Information gathering and analysis | Research tasks |
| `writer` | Content creation and editing | Documentation |
| `debugger` | Troubleshooting and debugging | Bug fixing |

## Using Built-in Personas

### Basic Usage

```python
from victor.agent import AgentOrchestrator
from victor.agent.personas import PersonaManager

orchestrator = AgentOrchestrator(...)

# Get persona manager
personas = orchestrator.personas

# List available personas
available = personas.list_personas()
for p in available:
    print(f"{p.name}: {p.description}")

# Set a persona
await personas.set_persona("code_reviewer")

# Now chat with code reviewer persona
response = await orchestrator.chat(
    "Review this code for quality and best practices",
    context={"code": open("main.py").read()}
)

print(response)
```

### Persona-Specific Capabilities

```python
# Code reviewer persona
await personas.set_persona("code_reviewer")

review = await orchestrator.chat(
    "Review this code",
    context={"file": "app.py"}
)

# Reviewer will focus on:
# - Code quality
# - Best practices
# - Maintainability
# - Performance

# Security auditor persona
await personas.set_persona("security_auditor")

security_review = await orchestrator.chat(
    "Review this code",
    context={"file": "app.py"}
)

# Auditor will focus on:
# - Security vulnerabilities
# - Input validation
# - Authentication/authorization
# - Data exposure
```

### Switching Personas

```python
# Start with general assistant
await personas.set_persona("assistant")

plan = await orchestrator.chat(
    "Create a plan for user authentication"
)

# Switch to architect for design
await personas.set_persona("architect")

design = await orchestrator.chat(
    "Design the authentication system architecture",
    context={"plan": plan}
)

# Switch to debugger for implementation
await personas.set_persona("debugger")

implementation = await orchestrator.chat(
    "Implement the authentication system",
    context={"design": design}
)

# Switch to code_reviewer for review
await personas.set_persona("code_reviewer")

review = await orchestrator.chat(
    "Review the authentication implementation",
    context={"code": implementation}
)
```

## Creating Custom Personas

### Basic Custom Persona

```python
# Create a custom persona
await personas.create_persona(
    name="api_specialist",
    description="Expert in RESTful API design and implementation",
    system_prompt="""You are an API specialist with deep expertise in:
- RESTful API design principles
- API versioning strategies
- Rate limiting and throttling
- Authentication and authorization for APIs
- API documentation (OpenAPI/Swagger)

You provide clear, practical advice for building production-ready APIs.""",
    capabilities=[
        "api_design",
        "api_security",
        "api_documentation",
        "rate_limiting"
    ],
    metadata={
        "expertise_level": "senior",
        "specializations": ["REST", "GraphQL", "gRPC"]
    }
)

# Use the custom persona
await personas.set_persona("api_specialist")

response = await orchestrator.chat(
    "Design a RESTful API for a todo application"
)
```

### Advanced Persona with Triggers

```python
# Create persona with automatic triggers
await personas.create_persona(
    name="performance_optimizer",
    description="Specializes in optimizing code performance",
    system_prompt="You are a performance optimization expert...",
    capabilities=["profiling", "optimization", "benchmarking"],
    triggers=[
        {
            "type": "keyword",
            "keywords": ["slow", "performance", "optimize", "latency"],
            "confidence": 0.8
        },
        {
            "type": "context",
            "context_key": "task_type",
            "context_value": "optimization",
            "confidence": 0.9
        }
    ]
)

# Persona will be suggested automatically when keywords detected
```

### Persona from Template

```python
# Define persona template
template = {
    "system_prompt_template": """You are a {domain} expert specializing in {specialization}.
Your communication style is {style}.
You focus on {focus_areas}.""",
    "capabilities_template": ["{specialization}_analysis", "{specialization}_design"]
}

# Create persona from template
await personas.create_persona_from_template(
    name="ml_engineer",
    template=template,
    variables={
        "domain": "machine learning",
        "specialization": "deep learning",
        "style": "technical but accessible",
        "focus_areas": "model architecture, training strategies, optimization"
    }
)
```

### Persona with Custom Behavior

```python
# Define custom behavior
async def custom_behavior(context):
    """Custom behavior for persona."""

    # Pre-processing
    if "code" in context:
        context["code_analysis"] = analyze_code_quality(context["code"])

    # Get response
    response = await orchestrator.generate_response(context)

    # Post-processing
    if persona.name == "teacher":
        response = add_educational_notes(response)

    return response

# Create persona with custom behavior
await personas.create_persona(
    name="interactive_teacher",
    description="Interactive teacher with quizzes",
    system_prompt="You are an interactive teacher...",
    behavior_fn=custom_behavior
)
```

## Context Adaptation

Personas automatically adapt based on context.

### Automatic Persona Suggestion

```python
# Get persona suggestion for context
suggested = await personas.suggest_persona(
    context="I need to review this code for security issues",
    task_type="security_review"
)

print(f"Suggested persona: {suggested.name}")
print(f"Confidence: {suggested.confidence:.2f}")
print(f"Reason: {suggested.reason}")

# Apply suggested persona
if suggested.confidence > 0.7:
    await personas.set_persona(suggested.name)
```

### Context-Based Switching

```python
# Automatic switching based on context
async def adaptive_execution(task):
    # Analyze task
    analysis = await orchestrator.analyze_task(task)

    # Get persona suggestion
    persona = await personas.suggest_persona(
        context=task,
        task_type=analysis.task_type
    )

    # Switch persona
    await personas.set_persona(persona.name)

    # Execute task
    result = await orchestrator.chat(task)

    return result

# Usage
result = await adaptive_execution("Review this API for security issues")
# Automatically switches to security_auditor persona
```

### Multi-Persona Collaboration

```python
# Use multiple personas for different aspects
async def multi_persona_review(code):
    results = {}

    # Security review
    await personas.set_persona("security_auditor")
    results["security"] = await orchestrator.chat(
        f"Security review:\n{code}"
    )

    # Quality review
    await personas.set_persona("code_reviewer")
    results["quality"] = await orchestrator.chat(
        f"Quality review:\n{code}"
    )

    # Performance review
    await personas.set_persona("performance_optimizer")
    results["performance"] = await orchestrator.chat(
        f"Performance review:\n{code}"
    )

    # Combine results
    combined = await orchestrator.chat(
        f"Combine these reviews:\n{results}"
    )

    return combined
```

## Persona Management

### Saving and Loading Personas

```python
# Export persona
persona_data = await personas.export_persona("code_reviewer")

# Save to file
import json
with open("code_reviewer_persona.json", "w") as f:
    json.dump(persona_data, f)

# Import persona
with open("custom_persona.json", "r") as f:
    persona_data = json.load(f)

await personas.import_persona(persona_data)
```

### Updating Personas

```python
# Update existing persona
await personas.update_persona(
    name="code_reviewer",
    description="Updated description",
    system_prompt="Updated system prompt...",
    add_capabilities=["static_analysis"],
    remove_capabilities=[]
)
```

### Deleting Personas

```python
# Delete custom persona
await personas.delete_persona("my_custom_persona")

# Reset to default
await personas.reset_persona()
```

### Persona Analytics

```python
# Get usage statistics
stats = await personas.get_usage_stats()

print(f"Most used persona: {stats['most_used']}")
print(f"Usage counts: {stats['usage_counts']}")
print(f"Average session duration: {stats['avg_duration']}")

# Get effectiveness metrics
effectiveness = await personas.get_effectiveness_metrics()

for persona_name, metrics in effectiveness.items():
    print(f"{persona_name}:")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  User satisfaction: {metrics['satisfaction']:.1%}")
```

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
- [User Guide](../USER_GUIDE.md)
- [Persona Management](../personas/README.md)
- [Persona Templates](../personas/templates.md)
