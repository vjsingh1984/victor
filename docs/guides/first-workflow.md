# Creating Your First Workflow

Learn how to build multi-step workflows with Victor's StateGraph engine and YAML workflow compiler.

## What is a Workflow?

A **workflow** is a multi-step process that:
- Defines multiple stages of execution
- Can include conditional logic
- Supports parallel execution
- Maintains state across steps
- Can include human-in-the-loop approvals

## Two Ways to Create Workflows

Victor supports two approaches:

1. **YAML Workflows** - Declarative, easy to read and modify
2. **Python StateGraphs** - Programmatic, full flexibility

This guide covers both, starting with YAML (recommended for beginners).

## YAML Workflows

### Basic Workflow Structure

Create a file called `my_workflow.yaml`:

```yaml
name: "My First Workflow"
description: "A simple greeting workflow"

nodes:
  - id: "greet"
    type: "agent"
    config:
      prompt: "Hello, how can I help you today?"

edges:
  - from: "start"
    to: "greet"
  - from: "greet"
    to: "complete"
```

Run the workflow:

```python
import asyncio
from victor import Agent

async def main():
    agent = Agent.create()

    # Run the workflow
    result = await agent.run_workflow("my_workflow.yaml")

    print(result.content)

asyncio.run(main())
```

### Multi-Step Workflow

Create a workflow that processes user input through multiple steps:

```yaml
name: "Content Processor"
description: "Analyze and summarize content"

nodes:
  # Step 1: Analyze input
  - id: "analyze"
    type: "agent"
    config:
      prompt: |
        Analyze the following content:
        {{input}}

        Identify:
        1. Main topic
        2. Key points
        3. Sentiment

  # Step 2: Create summary
  - id: "summarize"
    type: "agent"
    config:
      prompt: |
        Based on this analysis:
        {{analyze.output}}

        Create a concise summary in 2-3 sentences.

  # Step 3: Extract keywords
    type: "agent"
    config:
      prompt: |
        From the content:
        {{input}}

        Extract the top 5 keywords.

edges:
  - from: "start"
    to: "analyze"
  - from: "analyze"
    to: "summarize"
  - from: "summarize"
    to: "keywords"
  - from: "keywords"
    to: "complete"
```

### Conditional Workflows

Add conditional logic based on analysis:

```yaml
name: "Content Router"
description: "Route content based on type"

nodes:
  - id: "classify"
    type: "agent"
    config:
      prompt: |
        Classify this content as one of: [technical, creative, business]
        Content: {{input}}
        Respond with only the category name.

  - id: "technical"
    type: "agent"
    config:
      prompt: "As a technical expert, analyze: {{input}}"

  - id: "creative"
    type: "agent"
    config:
      prompt: "As a creative writer, analyze: {{input}}"

  - id: "business"
    type: "agent"
    config:
      prompt: "As a business analyst, analyze: {{input}}"

edges:
  - from: "start"
    to: "classify"

  # Conditional routing
  - from: "classify"
    to: "technical"
    condition: "{{classify.output}} == 'technical'"

  - from: "classify"
    to: "creative"
    condition: "{{classify.output}} == 'creative'"

  - from: "classify"
    to: "business"
    condition: "{{classify.output}} == 'business'"

  # All paths lead to completion
  - from: "technical"
    to: "complete"
  - from: "creative"
    to: "complete"
  - from: "business"
    to: "complete"
```

### Workflows with Tools

Create a workflow that analyzes code:

```yaml
name: "Code Review Workflow"
description: "Review and analyze code"

nodes:
  - id: "read_code"
    type: "handler"
    config:
      tool: "read"
      arguments:
        file_path: "{{file_path}}"

  - id: "analyze"
    type: "agent"
    config:
      tools: ["grep"]
      prompt: |
        Review this code:
        {{read_code.result}}

        Check for:
        - Potential bugs
        - Security issues
        - Code style violations
        - Performance problems

  - id: "suggest"
    type: "agent"
    config:
      prompt: |
        Based on the analysis:
        {{analyze.output}}

        Provide specific improvement suggestions.

edges:
  - from: "start"
    to: "read_code"
  - from: "read_code"
    to: "analyze"
  - from: "analyze"
    to: "suggest"
  - from: "suggest"
    to: "complete"
```

Run with input:

```python
result = await agent.run_workflow(
    "code_review.yaml",
    input={"file_path": "src/main.py"}
)
```

### Parallel Execution

Run multiple steps in parallel:

```yaml
name: "Multi-Perspective Analysis"
description: "Analyze from multiple angles simultaneously"

nodes:
  - id: "technical_review"
    type: "agent"
    config:
      prompt: "Technical analysis of: {{input}}"

  - id: "business_review"
    type: "agent"
    config:
      prompt: "Business analysis of: {{input}}"

  - id: "user_review"
    type: "agent"
    config:
      prompt: "User experience analysis of: {{input}}"

  - id: "synthesize"
    type: "agent"
    config:
      prompt: |
        Synthesize these three perspectives:
        Technical: {{technical_review.output}}
        Business: {{business_review.output}}
        User: {{user_review.output}}

        Provide a comprehensive analysis.

edges:
  # Start all reviews in parallel
  - from: "start"
    to: "technical_review"
  - from: "start"
    to: "business_review"
  - from: "start"
    to: "user_review"

  # Wait for all to complete, then synthesize
  - from: "technical_review"
    to: "synthesize"
  - from: "business_review"
    to: "synthesize"
  - from: "user_review"
    to: "synthesize"

  - from: "synthesize"
    to: "complete"
```

## Python StateGraphs

For more complex workflows, use Python directly:

### Basic StateGraph

```python
import asyncio
from victor.framework import StateGraph

async def analyze_node(state):
    """Analyze input."""
    prompt = f"Analyze this: {state['input']}"
    agent = Agent.create()
    result = await agent.run(prompt)
    return {"analysis": result.content}

async def summarize_node(state):
    """Summarize analysis."""
    prompt = f"Summarize: {state['analysis']}"
    agent = Agent.create()
    result = await agent.run(prompt)
    return {"summary": result.content}

# Create the workflow
workflow = StateGraph()

# Add nodes
workflow.add_node("analyze", analyze_node)
workflow.add_node("summarize", summarize_node)

# Add edges
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "summarize")
workflow.set_finish_point("summarize")

# Compile and run
compiled = workflow.compile()
result = await compiled.ainvoke({"input": "Your content here"})
print(result["summary"])
```

### Conditional Routing

```python
async def classify_node(state):
    """Classify content."""
    agent = Agent.create()
    result = await agent.run(
        f"Classify as [technical, creative, business]: {state['input']}"
    )
    category = result.content.strip().lower()
    return {"category": category}

async def route_function(state):
    """Route based on classification."""
    category = state.get("category", "business")
    return category

# Create workflow
workflow = StateGraph()
workflow.add_node("classify", classify_node)
workflow.add_node("technical", technical_node)
workflow.add_node("creative", creative_node)
workflow.add_node("business", business_node)

# Set entry point
workflow.set_entry_point("classify")

# Add conditional edges
workflow.add_conditional_edges(
    "classify",
    route_function,
    {
        "technical": "technical",
        "creative": "creative",
        "business": "business",
    }
)

# All paths finish
workflow.set_finish_point("technical")
workflow.set_finish_point("creative")
workflow.set_finish_point("business")
```

### Human-in-the-Loop

Add human approval steps:

```yaml
name: "Document Approval"
description: "Document requires approval before publishing"

nodes:
  - id: "draft"
    type: "agent"
    config:
      prompt: "Draft a document about: {{topic}}"

  - id: "human_review"
    type: "human"
    config:
      prompt: |
        Please review the following draft:
        {{draft.output}}

        Approve? (yes/no)

  - id: "publish"
    type: "handler"
    config:
      action: "publish"
      arguments:
        content: "{{draft.output}}"

  - id: "revise"
    type: "agent"
    config:
      prompt: |
        The draft was not approved.
        Original: {{draft.output}}
        Please revise based on feedback.

edges:
  - from: "start"
    to: "draft"
  - from: "draft"
    to: "human_review"
  - from: "human_review"
    to: "publish"
    condition: "{{human_review.approved}} == true"
  - from: "human_review"
    to: "revise"
    condition: "{{human_review.approved}} == false"
  - from: "revise"
    to: "human_review"
  - from: "publish"
    to: "complete"
```

## Streaming Workflow Execution

Stream workflow progress in real-time:

```python
async def run_workflow_with_streaming():
    agent = Agent.create()

    # Stream workflow execution
    async for node_id, state in agent.stream_workflow("my_workflow.yaml", input={"topic": "AI"}):
        print(f"✓ Completed: {node_id}")
        if "output" in state:
            print(f"  Output: {state['output'][:100]}...")
```

## Workflow Best Practices

### 1. Keep Nodes Focused

```yaml
# ❌ Bad: Too much responsibility
- id: "do_everything"
  type: "agent"
  config:
    prompt: "Analyze, summarize, translate, and format..."

# ✅ Good: Single responsibility
- id: "analyze"
  type: "agent"
  config:
    prompt: "Analyze the input"
- id: "summarize"
  type: "agent"
  config:
    prompt: "Summarize the analysis"
```

### 2. Use Descriptive Names

```yaml
# ❌ Bad: Vague names
- id: "step1"
- id: "step2"

# ✅ Good: Descriptive names
- id: "analyze_requirements"
- id: "design_solution"
```

### 3. Handle Errors Gracefully

```yaml
- id: "risky_operation"
  type: "handler"
  config:
    tool: "shell"
    arguments:
      command: "{{user_command}}"
    on_error: "continue"  # or "retry" or "fail"
    max_retries: 3
```

### 4. Document Your Workflow

```yaml
name: "Data Processing Pipeline"
description: |
  This workflow processes raw data through three stages:
  1. Validation - Check data quality
  2. Transformation - Convert to standard format
  3. Analysis - Generate insights

  Input: Raw data file path
  Output: Analysis report
```

## Common Workflow Patterns

### Pattern 1: Map-Reduce

```yaml
# Process items in parallel, then combine results
nodes:
  - id: "map"
    type: "agent"
    config:
      parallel: true
      items: "{{items}}"
      prompt: "Process: {{item}}"

  - id: "reduce"
    type: "agent"
    config:
      prompt: "Combine these results: {{map.outputs}}"
```

### Pattern 2: Sequential Approval

```yaml
# Multiple approval stages
nodes:
  - id: "manager_approval"
    type: "human"
  - id: "director_approval"
    type: "human"
  - id: "final_approval"
    type: "human"
```

### Pattern 3: Retry with Backoff

```yaml
nodes:
  - id: "attempt"
    type: "handler"
    config:
      max_retries: 3
      backoff: "exponential"
```

## Putting It All Together

### Example: Blog Post Pipeline

```yaml
name: "Blog Post Creator"
description: "Create and review a blog post"

nodes:
  # Step 1: Generate outline
  - id: "outline"
    type: "agent"
    config:
      prompt: |
        Create a detailed outline for a blog post about: {{topic}}
        Include:
        - Catchy title
        - Section headings
        - Key points for each section

  # Step 2: Write content
  - id: "write"
    type: "agent"
    config:
      prompt: |
        Write the full blog post based on this outline:
        {{outline.output}}

        Make it engaging and informative.

  # Step 3: SEO optimization
  - id: "seo"
    type: "agent"
    config:
      prompt: |
        Optimize this blog post for SEO:
        {{write.output}}

        Add:
        - Meta description
        - Keywords
        - Suggested title tags

  # Step 4: Human review
    type: "human"
    config:
      instructions: |
        Review the blog post:
        {{write.output}}

        SEO suggestions:
        {{seo.output}}

        Approve for publication?

  # Step 5: Finalize
  - id: "finalize"
    type: "agent"
    config:
      prompt: |
        Create the final HTML version:
        Content: {{write.output}}
        SEO: {{seo.output}}

edges:
  - from: "start"
    to: "outline"
  - from: "outline"
    to: "write"
  - from: "write"
    to: "seo"
  - from: "seo"
    to: "human_review"
  - from: "human_review"
    to: "finalize"
    condition: "{{human_review.approved}} == true"
  - from: "human_review"
    to: "write"
    condition: "{{human_review.approved}} == false"
  - from: "finalize"
    to: "complete"
```

## Troubleshooting

### Workflow Not Executing

**Check**: YAML syntax is valid
```bash
python -c "import yaml; yaml.safe_load(open('my_workflow.yaml'))"
```

### Nodes Not Connecting

**Check**: Edge node IDs match node IDs
```yaml
nodes:
  - id: "step1"  # Must match
edges:
  - from: "start"
    to: "step1"  # Must match
```

### Conditional Edges Not Working

**Check**: Condition syntax is correct
```yaml
condition: "{{previous_node.output}} == 'expected_value'"
```

### State Not Passing Between Nodes

**Check**: Referencing state correctly
```yaml
prompt: "Use this data: {{previous_node.output}}"
```

## Next Steps

- 📖 [Workflow API Reference](../api/workflows.md) - Full workflow API
- 🔄 [Advanced Workflows](../examples/workflows/advanced.md) - Complex patterns
- 🤖 [Building Agents](first-agent.md) - Create agents for workflows
- 📚 [StateGraph Reference](../api/stategraph.md) - Low-level API

## Quick Reference

```yaml
# Basic workflow structure
name: "Workflow Name"
nodes:
  - id: "node1"
    type: "agent"  # or "handler" or "human"
    config:
      prompt: "What to do"
edges:
  - from: "start"
    to: "node1"
  - from: "node1"
    to: "complete"

# Run workflow
result = await agent.run_workflow("workflow.yaml", input={...})
```
