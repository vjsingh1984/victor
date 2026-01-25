# Dynamic Skills Guide

## Overview

Victor AI's dynamic skills system enables runtime tool discovery, composition, and chaining for adaptive behavior. This guide explains how to use skill discovery and chaining effectively.

## Table of Contents

- [What are Dynamic Skills?](#what-are-dynamic-skills)
- [Skill Discovery](#skill-discovery)
- [Skill Composition](#skill-composition)
- [Skill Chaining](#skill-chaining)
- [Adaptation and Learning](#adaptation-and-learning)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## What are Dynamic Skills?

Dynamic skills extend Victor's tool system with:

### Skill Discovery
- **Runtime tool discovery**: Find available tools dynamically
- **MCP tool integration**: Discover tools from MCP servers
- **Semantic matching**: Match tools to tasks using similarity
- **Tool compatibility analysis**: Check if tools work well together

### Skill Composition
- **Multi-tool skills**: Combine multiple tools into cohesive skills
- **Skill patterns**: Reusable skill templates
- **Dynamic registration**: Register new skills at runtime
- **Skill metadata**: Rich descriptions and capabilities

### Skill Chaining
- **Automatic planning**: Plan multi-step workflows
- **Dependency management**: Handle tool dependencies
- **Parallel execution**: Execute independent skills in parallel
- **Error handling**: Graceful failure recovery

### When to Use Dynamic Skills

**Ideal for:**
- Uncertain which tools to use for a task
- Need to combine multiple tools
- Dynamic workflow composition
- MCP tool integration
- Adaptive agent behavior

**Not ideal for:**
- Simple, single-tool tasks
- Fixed, well-known workflows
- Performance-critical paths (use direct tool calls)

## Skill Discovery

Skill discovery finds and matches tools to tasks dynamically.

### Basic Discovery

```python
from victor.agent import AgentOrchestrator
from victor.config.settings import Settings

# Enable skill discovery
settings = Settings()
settings.enable_skill_discovery = True

orchestrator = AgentOrchestrator(settings=settings, ...)

# Discover tools for a task
tools = await orchestrator.skills.discover_tools(
    context="code analysis and refactoring",
    max_tools=10
)

for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Compatibility: {tool.compatibility:.2f}")
    print(f"Category: {tool.category}")
    print(f"Cost Tier: {tool.cost_tier}")
```

### Task-to-Tool Matching

```python
# Match specific tools to a task description
task_description = "Analyze Python code for security vulnerabilities"

matched_tools = await orchestrator.skills.match_tools_to_task(
    task=task_description,
    tools=tools,  # Optional: use all available tools
    min_compatibility=0.5,  # Minimum similarity threshold
    max_tools=5
)

for tool in matched_tools:
    print(f"{tool.name}: {tool.compatibility:.2f}")
    print(f"  {tool.description}")
```

### MCP Tool Discovery

```python
# Discover tools from MCP servers
mcp_tools = await orchestrator.skills.discover_mcp_tools(
    server_name="filesystem",  # Specific MCP server
    context="file operations"
)

# Discover from all MCP servers
all_mcp_tools = await orchestrator.skills.discover_mcp_tools(
    server_name=None,  # All servers
    context="development"
)

for tool in all_mcp_tools:
    print(f"MCP Tool: {tool.name}")
    print(f"Server: {tool.source}")
    print(f"Description: {tool.description}")
```

### Tool Compatibility Analysis

```python
# Analyze compatibility between tools
compatibility = await orchestrator.skills.analyze_compatibility(
    tool_1=tools[0],
    tool_2=tools[1]
)

print(f"Compatibility score: {compatibility['score']:.2f}")
print(f"Reason: {compatibility['reason']}")
print(f"Can chain: {compatibility['can_chain']}")
```

## Skill Composition

Skill composition combines multiple tools into cohesive skills.

### Basic Composition

```python
# Compose a skill from multiple tools
skill = await orchestrator.skills.compose_skill(
    name="code_analyzer",
    tools=[tools[0], tools[1], tools[2]],
    description="Analyzes code quality, security, and performance",
    metadata={
        "category": "analysis",
        "languages": ["python", "javascript"],
        "complexity": "medium"
    }
)

print(f"Skill: {skill.name}")
print(f"Tools: {[t.name for t in skill.tools]}")
print(f"Description: {skill.description}")
```

### Skill with Execution Logic

```python
# Define custom execution logic
async def execute_code_analyzer(context):
    """Execute multi-tool code analysis."""

    # Step 1: Read files
    files = await context.tool_executor.execute_tool(
        tool_name="read_multiple_files",
        parameters={"file_paths": context["files"]}
    )

    # Step 2: Analyze quality
    quality = await context.tool_executor.execute_tool(
        tool_name="analyze_code_quality",
        parameters={"code": files}
    )

    # Step 3: Check security
    security = await context.tool_executor.execute_tool(
        tool_name="security_scan",
        parameters={"code": files}
    )

    # Step 4: Combine results
    return {
        "quality": quality,
        "security": security,
        "overall_score": (quality['score'] + security['score']) / 2
    }

# Create skill with custom execution
skill = await orchestrator.skills.compose_skill(
    name="comprehensive_analyzer",
    tools=tools,
    description="Comprehensive code analysis",
    execution_fn=execute_code_analyzer
)
```

### Skill Registration

```python
# Register a custom skill
await orchestrator.skills.register_skill(
    skill=skill,
    overwrite=False  # Don't overwrite if exists
)

# Skill is now available for discovery
registered = await orchestrator.skills.discover_tools(
    context="code analysis"
)

# Find registered skill
my_skill = [s for s in registered if s.name == "comprehensive_analyzer"]
print(f"Found skill: {my_skill}")
```

### Skill Patterns

```python
# Define reusable skill patterns
patterns = {
    "read_analyze_report": {
        "tools": ["read_file", "analyze_code", "generate_report"],
        "description": "Read, analyze, and report on code",
        "execution": "sequential"
    },
    "parallel_analysis": {
        "tools": ["security_scan", "quality_check", "performance_test"],
        "description": "Run multiple analyses in parallel",
        "execution": "parallel"
    }
}

# Apply pattern
async def apply_pattern(pattern_name, context):
    pattern = patterns[pattern_name]

    skill = await orchestrator.skills.compose_skill(
        name=pattern_name,
        tools=[pattern["tools"]],
        description=pattern["description"],
        execution_strategy=pattern["execution"]
    )

    return skill
```

## Skill Chaining

Skill chaining plans and executes multi-step workflows.

### Basic Chaining

```python
# Create skills
skill1 = await orchestrator.skills.compose_skill(
    name="reader",
    tools=[read_tool],
    description="Reads files"
)

skill2 = await orchestrator.skills.compose_skill(
    name="analyzer",
    tools=[analyze_tool],
    description="Analyzes code"
)

skill3 = await orchestrator.skills.compose_skill(
    name="fixer",
    tools=[fix_tool],
    description="Fixes issues"
)

# Plan a skill chain
chain = await orchestrator.skills.plan_chain(
    goal="Analyze and fix code quality issues",
    skills=[skill1, skill2, skill3],
    max_length=5
)

print(f"Chain goal: {chain.goal}")
print(f"Steps: {len(chain.steps)}")

for step in chain.steps:
    print(f"  {step.order}. {step.skill.name} - {step.description}")
```

### Executing Chains

```python
# Execute the chain
result = await orchestrator.skills.execute_chain(
    chain=chain,
    context={
        "file_path": "src/main.py",
        "max_issues": 10
    }
)

print(f"Status: {result.status}")
print(f"Completed: {result.completed_steps}/{result.total_steps}")
print(f"Outputs: {result.outputs}")

if result.failed:
    print(f"Failed step: {result.failed_step}")
    print(f"Error: {result.error}")
```

### Conditional Chaining

```python
# Plan chain with conditions
chain = await orchestrator.skills.plan_chain(
    goal="Fix and test code",
    skills=[skill1, skill2, skill3],
    conditions=[
        {
            "step": 2,  # After analyzer
            "condition": lambda ctx: ctx.get("issues_found", 0) > 0,
            "action": "continue"  # Only continue if issues found
        }
    ]
)
```

### Parallel Execution

```python
# Execute independent skills in parallel
chain = await orchestrator.skills.plan_chain(
    goal="Comprehensive code analysis",
    skills=[security_skill, quality_skill, performance_skill],
    execution_strategy="parallel"  # Execute in parallel
)

result = await orchestrator.skills.execute_chain(
    chain=chain,
    context={"file_path": "src/main.py"}
)

# All skills executed in parallel
print(f"Parallel execution completed in {result.duration}s")
```

### Chain Validation

```python
# Validate chain before execution
validation = orchestrator.skills.validate_chain(chain)

if not validation.is_valid:
    print(f"Chain validation failed:")
    for error in validation.errors:
        print(f"  - {error}")
else:
    print("Chain is valid, executing...")
    result = await orchestrator.skills.execute_chain(chain)
```

## Adaptation and Learning

Dynamic skills adapt and learn from usage.

### Performance Tracking

```python
# Track skill performance
await orchestrator.skills.track_performance(
    skill_name="code_analyzer",
    success=True,
    duration=2.5,
    context={"files_analyzed": 10}
)

# Get performance statistics
stats = await orchestrator.skills.get_performance_stats(
    skill_name="code_analyzer"
)

print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average duration: {stats['avg_duration']:.2f}s")
print(f"Total executions: {stats['total_executions']}")
```

### Adaptive Tool Selection

```python
# Select tools based on performance
best_tools = await orchestrator.skills.select_best_tools(
    task="code analysis",
    max_tools=5,
    selection_criteria="performance"  # or "success_rate", "recent"
)

for tool in best_tools:
    print(f"{tool.name}: {tool.performance_score:.2f}")
```

### Skill Optimization

```python
# Optimize skill composition based on usage
optimization = await orchestrator.skills.optimize_skill(
    skill_name="code_analyzer",
    objective="minimize_duration"  # or "maximize_success_rate"
)

print(f"Optimized skill:")
print(f"  Original tools: {optimization['original_tools']}")
print(f"  Optimized tools: {optimization['optimized_tools']}")
print(f"  Improvement: {optimization['improvement']:.1%}")
```

## Best Practices

### 1. Use Descriptive Skill Names

```python
# Good: Clear, descriptive
skill = await orchestrator.skills.compose_skill(
    name="python_security_analyzer",
    description="Analyzes Python code for security vulnerabilities"
)

# Bad: Vague
skill = await orchestrator.skills.compose_skill(
    name="analyzer",
    description="Analyzes stuff"
)
```

### 2. Check Tool Compatibility

```python
# Check compatibility before composing
compatibility = await orchestrator.skills.analyze_compatibility(
    tool_1=tool1,
    tool_2=tool2
)

if compatibility['can_chain']:
    skill = await orchestrator.skills.compose_skill(
        name="combined_skill",
        tools=[tool1, tool2]
    )
else:
    print(f"Cannot chain: {compatibility['reason']}")
```

### 3. Validate Chains

```python
# Always validate before execution
chain = await orchestrator.skills.plan_chain(goal, skills)

validation = orchestrator.skills.validate_chain(chain)
if not validation.is_valid:
    print(f"Validation errors: {validation.errors}")
    # Fix or abort
else:
    result = await orchestrator.skills.execute_chain(chain)
```

### 4. Handle Failures Gracefully

```python
# Execute with error handling
try:
    result = await orchestrator.skills.execute_chain(
        chain=chain,
        context=context,
        on_failure="continue"  # Continue on failure
    )
except Exception as e:
    print(f"Chain execution failed: {e}")
    # Fallback logic
```

### 5. Use Parallel Execution When Possible

```python
# Parallel execution for independent skills
chain = await orchestrator.skills.plan_chain(
    goal="Run multiple analyses",
    skills=[security_skill, quality_skill, performance_skill],
    execution_strategy="parallel"  # Faster
)
```

### 6. Monitor Performance

```python
# Regular performance monitoring
stats = await orchestrator.skills.get_performance_stats(
    skill_name="my_skill"
)

if stats['success_rate'] < 0.8:
    print(f"Skill {skill_name} has low success rate")
    # Investigate or reoptimize
```

### 7. Document Skills

```python
# Rich metadata for documentation
skill = await orchestrator.skills.compose_skill(
    name="data_pipeline",
    tools=[extract_tool, transform_tool, load_tool],
    description="ETL pipeline for data processing",
    metadata={
        "author": "data-team",
        "version": "1.0",
        "tags": ["etl", "data", "pipeline"],
        "dependencies": ["pandas", "sqlalchemy"],
        "input_format": "csv",
        "output_format": "sql",
        "examples": [
            "Process user data",
            "Transform sales data"
        ]
    }
)
```

## Troubleshooting

### Tool Discovery Fails

**Problem**: No tools found for a task.

**Solutions**:
1. **Lower threshold**: Reduce min_compatibility
2. **Improve query**: Use more specific task description
3. **Check tools**: Verify tools are registered
4. **Use keywords**: Add relevant keywords

```python
# Lower threshold
tools = await orchestrator.skills.discover_tools(
    context="code analysis",
    min_compatibility=0.3  # Lower from 0.5
)

# More specific query
tools = await orchestrator.skills.discover_tools(
    context="Python static analysis for security vulnerabilities"
)
```

### Chain Execution Fails

**Problem**: Chain execution stops or fails.

**Solutions**:
1. **Validate chain**: Check validation errors
2. **Check dependencies**: Verify tool dependencies
3. **Handle errors**: Use error handling strategies
4. **Debug step**: Execute steps individually

```python
# Validate first
validation = orchestrator.skills.validate_chain(chain)
if not validation.is_valid:
    print(f"Errors: {validation.errors}")

# Handle errors
result = await orchestrator.skills.execute_chain(
    chain=chain,
    on_failure="skip"  # Skip failed steps
)
```

### Poor Performance

**Problem**: Skills execute slowly.

**Solutions**:
1. **Use parallel execution**: Execute independent skills in parallel
2. **Optimize tool selection**: Choose best-performing tools
3. **Cache results**: Cache expensive operations
4. **Reduce tool count**: Use fewer, more effective tools

```python
# Use parallel execution
chain = await orchestrator.skills.plan_chain(
    goal="analysis",
    skills=skills,
    execution_strategy="parallel"
)

# Select best tools
best_tools = await orchestrator.skills.select_best_tools(
    task="analysis",
    selection_criteria="performance"
)
```

## Examples

### Example 1: Code Review Workflow

```python
async def code_review_workflow(file_path):
    # Discover relevant tools
    tools = await orchestrator.skills.discover_tools(
        context="code review quality security",
        max_tools=10
    )

    # Compose skills
    reader = await orchestrator.skills.compose_skill(
        name="code_reader",
        tools=[tools[0]],  # read_file
        description="Read code files"
    )

    analyzer = await orchestrator.skills.compose_skill(
        name="code_analyzer",
        tools=[tools[1], tools[2]],  # analyze_code, security_scan
        description="Analyze code quality and security"
    )

    reporter = await orchestrator.skills.compose_skill(
        name="reporter",
        tools=[tools[3]],  # generate_report
        description="Generate review report"
    )

    # Plan and execute chain
    chain = await orchestrator.skills.plan_chain(
        goal="Comprehensive code review",
        skills=[reader, analyzer, reporter]
    )

    result = await orchestrator.skills.execute_chain(
        chain=chain,
        context={"file_path": file_path}
    )

    return result.outputs
```

### Example 2: Data Pipeline

```python
async def data_pipeline_workflow(source, destination):
    # Discover data tools
    tools = await orchestrator.skills.discover_tools(
        context="data extraction transformation loading",
        max_tools=10
    )

    # Compose ETL skills
    extract = await orchestrator.skills.compose_skill(
        name="extractor",
        tools=[tools[0]],
        description="Extract data from source"
    )

    transform = await orchestrator.skills.compose_skill(
        name="transformer",
        tools=[tools[1], tools[2]],
        description="Transform and clean data"
    )

    load = await orchestrator.skills.compose_skill(
        name="loader",
        tools=[tools[3]],
        description="Load data to destination"
    )

    # Execute chain
    chain = await orchestrator.skills.plan_chain(
        goal="ETL pipeline",
        skills=[extract, transform, load]
    )

    result = await orchestrator.skills.execute_chain(
        chain=chain,
        context={
            "source": source,
            "destination": destination
        }
    )

    return result
```

### Example 3: Parallel Testing

```python
async def parallel_testing_workflow(project_path):
    # Discover testing tools
    tools = await orchestrator.skills.discover_tools(
        context="testing unit integration e2e",
        max_tools=10
    )

    # Compose testing skills
    unit_tests = await orchestrator.skills.compose_skill(
        name="unit_tester",
        tools=[tools[0]],
        description="Run unit tests"
    )

    integration_tests = await orchestrator.skills.compose_skill(
        name="integration_tester",
        tools=[tools[1]],
        description="Run integration tests"
    )

    e2e_tests = await orchestrator.skills.compose_skill(
        name="e2e_tester",
        tools=[tools[2]],
        description="Run end-to-end tests"
    )

    # Execute in parallel
    chain = await orchestrator.skills.plan_chain(
        goal="Run all tests",
        skills=[unit_tests, integration_tests, e2e_tests],
        execution_strategy="parallel"
    )

    result = await orchestrator.skills.execute_chain(
        chain=chain,
        context={"project_path": project_path}
    )

    return result
```

### Example 4: Adaptive Workflow

```python
async def adaptive_workflow(task):
    # Discover tools dynamically
    tools = await orchestrator.skills.discover_tools(
        context=task,
        max_tools=15
    )

    # Select best performing tools
    best_tools = await orchestrator.skills.select_best_tools(
        task=task,
        max_tools=5,
        selection_criteria="performance"
    )

    # Compose skills
    skills = [
        await orchestrator.skills.compose_skill(
            name=f"skill_{i}",
            tools=[tool],
            description=tool.description
        )
        for i, tool in enumerate(best_tools)
    ]

    # Plan optimal chain
    chain = await orchestrator.skills.plan_chain(
        goal=task,
        skills=skills,
        optimization="duration"  # Minimize execution time
    )

    # Execute with monitoring
    result = await orchestrator.skills.execute_chain(
        chain=chain,
        context={},
        on_progress=lambda step: print(f"Completed: {step.skill.name}")
    )

    # Track performance
    await orchestrator.skills.track_performance(
        skill_name=chain.goal,
        success=(result.status == "completed"),
        duration=result.duration
    )

    return result
```

## Additional Resources

- [API Reference](../api/NEW_CAPABILITIES_API.md)
- [User Guide](../user-guide/index.md)
- [Hierarchical Planning Guide](HIERARCHICAL_PLANNING.md)
- [Enhanced Memory Guide](ENHANCED_MEMORY.md)
- [Tool System Documentation](../reference/tools/catalog.md)
