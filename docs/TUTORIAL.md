# Victor AI Tutorial

**From zero to productive with Victor AI - Complete step-by-step lessons**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Lesson 1: Basic Chat Interaction](#lesson-1-basic-chat-interaction)
3. [Lesson 2: Using Tools](#lesson-2-using-tools)
4. [Lesson 3: Workflows](#lesson-3-workflows)
5. [Lesson 4: Provider Switching](#lesson-4-provider-switching)
6. [Lesson 5: Session Management](#lesson-5-session-management)
7. [Lesson 6: Multi-Agent Teams](#lesson-6-multi-agent-teams)
8. [Lesson 7: Python API](#lesson-7-python-api)
9. [Lesson 8: Advanced Features](#lesson-8-advanced-features)
10. [Practice Exercises](#practice-exercises)

---

## Introduction

### What You'll Learn

This tutorial takes you from beginner to proficient user through 8 hands-on lessons. Each lesson builds on the previous one, with practical exercises you can run immediately.

**By the end, you'll be able to:**
- Have productive conversations with Victor AI
- Use tools for code analysis and manipulation
- Automate tasks with workflows
- Switch providers mid-conversation
- Manage sessions effectively
- Coordinate multi-agent teams
- Use Victor programmatically

### Prerequisites

- Completed [Quick Start Guide](QUICKSTART.md)
- Victor AI installed (`victor --version` works)
- At least one provider configured (local or cloud)
- Basic Python knowledge helpful but not required

### Setup for Tutorial

```bash
# Create practice directory
mkdir victor-tutorial
cd victor-tutorial

# Create sample Python file
cat > calculator.py << 'EOF'
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b
EOF
```

---

## Lesson 1: Basic Chat Interaction

**Goal**: Learn how to have effective conversations with Victor AI.

### 1.1 Start Your First Conversation

```bash
# Start Victor with default provider
victor chat
```

**Try these prompts:**

```
Hello! Can you explain what you can help me with?
```

```
What's the difference between list and tuple in Python?
```

```
Write a function to calculate fibonacci numbers
```

### 1.2 Understanding Victor's Response

Victor responds with:
- **Direct answers** to your questions
- **Code examples** when relevant
- **Explanations** of concepts
- **Follow-up suggestions** for deeper exploration

### 1.3 Asking Better Questions

**Vague:**
```
Help me with this code
```

**Specific:**
```
Review this Python function for potential bugs and performance issues:
[paste code here]
```

### 1.4 Multi-Turn Conversations

Victor remembers context within a session:

```
You: How do I read a file in Python?
Victor: [Explains file reading methods]

You: Which one is the most Pythonic?
Victor: [Recommends 'with open()' based on context]

You: Show me an example
Victor: [Provides specific code example]
```

### 1.5 Exercise: Your First Task

**Task**: Ask Victor to explain a concept you're learning.

```bash
victor chat "Explain async/await in Python with a practical example"
```

**What to notice:**
- Did Victor provide both explanation and code?
- Was the example practical and runnable?
- Did Victor explain when to use async/await?

---

## Lesson 2: Using Tools

**Goal**: Understand how Victor uses tools to help with coding tasks.

### 2.1 What Are Tools?

Victor has **55 specialized tools** organized into categories:
- **File Operations**: Read, write, search files
- **Git Operations**: Commit, branch, status
- **Code Analysis**: AST parsing, complexity analysis
- **Shell Execution**: Run commands, tests
- **Web**: Search, fetch URLs

### 2.2 File Operations

**Read a file:**
```bash
victor chat "Read calculator.py and tell me what it does"
```

Victor will:
1. Use the `read_file` tool
2. Analyze the content
3. Explain the code structure

**Write a file:**
```bash
victor chat "Create a new file called utils.py with a function to validate email addresses"
```

### 2.3 Code Analysis

```bash
victor chat "Analyze calculator.py for code quality and suggest improvements"
```

Victor will use tools to:
- Parse the AST (Abstract Syntax Tree)
- Calculate complexity metrics
- Check for anti-patterns
- Suggest improvements

### 2.4 Shell Operations

```bash
victor chat "Run pytest on calculator.py and show me the results"
```

Victor can:
- Execute shell commands
- Parse the output
- Interpret results
- Suggest fixes for failures

### 2.5 Tool Selection

Victor automatically selects relevant tools based on your request:

```
User: "Test this file"
→ Victor selects: pytest tool

User: "Review this code"
→ Victor selects: AST analysis tool

User: "Commit these changes"
→ Victor selects: Git tool
```

### 2.6 Exercise: Tool Practice

**Task**: Use Victor to analyze and improve the calculator.

```bash
victor chat "
Analyze calculator.py and:
1. Identify any bugs
2. Check code quality
3. Add error handling for divide by zero
4. Add type hints
5. Improve documentation
"
```

**Expected outcome:**
- Victor reads the file
- Identifies the divide-by-zero issue
- Suggests improvements
- Can even make the changes if you approve

---

## Lesson 3: Workflows

**Goal**: Automate repetitive tasks with YAML workflows.

### 3.1 What Are Workflows?

Workflows are **reusable, multi-step automation** defined in YAML. They chain together tools and LLM calls.

**Benefits:**
- Save time on repetitive tasks
- Consistent processes
- Shareable with team
- Version controllable

### 3.2 Simple Workflow Example

Create `code_review.yaml`:

```yaml
name: code_review
description: "Comprehensive code review workflow"

steps:
  - name: read_file
    tool: read_file
    description: "Read the file to review"

  - name: analyze_quality
    agent: true
    prompt: |
      Analyze the code for:
      - Bugs and errors
      - Security issues
      - Performance problems
      - Code style violations

  - name: check_complexity
    tool: complexity_analysis
    description: "Calculate cyclomatic complexity"

  - name: generate_report
    agent: true
    prompt: |
      Generate a comprehensive review report with:
      - Summary of findings
      - Prioritized issues
      - Specific recommendations
```

Run the workflow:

```bash
victor workflow run code_review --file calculator.py
```

### 3.3 Listing Available Workflows

```bash
victor workflow list
```

### 3.4 Exercise: Create Your First Workflow

**Task**: Create a workflow that tests and documents code.

Create `test_and_docs.yaml`:

```yaml
name: test_and_docs
description: "Run tests and generate documentation"

steps:
  - name: run_tests
    tool: pytest
    description: "Run all tests"

  - name: analyze_coverage
    tool: coverage_analysis

  - name: generate_docs
    agent: true
    prompt: |
      Generate documentation for this code:
      - Module docstring
      - Function docstrings
      - Usage examples
      - Type information
```

Run it:

```bash
victor workflow run test_and_docs --file calculator.py
```

---

## Lesson 4: Provider Switching

**Goal**: Switch between LLM providers without losing context.

### 4.1 Why Switch Providers?

Different providers excel at different tasks:

| Task | Best Provider | Why |
|------|---------------|-----|
| Brainstorming | Local (Ollama) | Free, fast |
| Implementation | GPT-4o | Fast, capable |
| Review | Claude Sonnet | Thorough |
| Complex analysis | Claude Opus | Most capable |
| Quick tasks | Groq/Llama | Ultra-fast |

### 4.2 Basic Provider Switching

**Start with one provider:**
```bash
victor chat --provider ollama
```

**Switch mid-conversation (in TUI mode):**
```
/provider openai --model gpt-4o
```

**Example session:**
```
# Start with free local model
You: Brainstorm 5 API endpoint ideas for a todo app
[Ollama generates ideas]

# Switch to Claude for implementation
/provider anthropic --model claude-sonnet-4-5
You: Implement the first idea using FastAPI
[Claude implements]

# Switch to GPT-4 for review
/provider openai --model gpt-4o
You: Review this implementation for security issues
[GPT-4 reviews]
```

### 4.3 Cost-Optimized Workflow

```bash
# 1. Brainstorm with local model (FREE)
victor chat --provider ollama "
Brainstorm approaches to implement user authentication
"

# 2. Draft with cheaper model ($)
/provider openai --model gpt-4o-mini
"Draft a basic implementation"

# 3. Polish with best model ($$)
/provider anthropic --model claude-sonnet-4-5
"Refine this implementation with best practices"
```

### 4.4 Exercise: Multi-Provider Task

**Task**: Use 3 different providers for one task.

```bash
# Start with Ollama (free)
victor chat --provider ollama "
Brainstorm ways to improve calculator.py
"

# Switch to Claude for implementation
/provider anthropic --model claude-sonnet-4-5
"Implement the top 3 improvements"

# Switch to GPT-4 for testing
/provider openai --model gpt-4o
"Write comprehensive tests for the improved code"
```

**Notice how context is preserved across providers!**

---

## Lesson 5: Session Management

**Goal**: Save, restore, and manage conversation sessions.

### 5.1 Why Session Management?

- **Pick up where you left off**
- **Reference previous conversations**
- **Build knowledge over time**
- **Share conversations with team**

### 5.2 Saving Sessions

**In TUI mode:**
```
/save "Project Architecture Discussion"
```

**From CLI:**
```bash
victor chat --session my-project
```

Sessions are saved to `~/.victor/sessions/`

### 5.3 Listing Sessions

```bash
victor sessions
```

Output:
```
Available Sessions:
====================
20250120_153045  Project Architecture Discussion
20250120_143022  Code Review - calculator.py
20250119_120000  Debugging Session
```

### 5.4 Resuming Sessions

**Interactive restore:**
```bash
victor resume
```

**Restore specific session:**
```bash
victor resume 20250120_153045
```

**Latest session:**
```bash
victor resume --latest
```

### 5.5 Exercise: Session Workflow

**Task**: Create and resume a session.

```bash
# Start a named session
victor chat --session learning-async

# Have a conversation
"You: Explain async/await in Python"
"Victor: [explains]"
"You: Show me a practical example"
"Victor: [shows example]"

# Save and exit
/save "Learning Async Programming"
Ctrl+D

# Later, resume the session
victor resume learning-async

# Continue the conversation
"Now explain when NOT to use async"
```

---

## Lesson 6: Multi-Agent Teams

**Goal**: Coordinate multiple specialized agents for complex tasks.

### 6.1 What Are Multi-Agent Teams?

Instead of one agent doing everything, **specialized agents** collaborate:
- **Security Expert** - Focuses on security
- **Performance Expert** - Focuses on optimization
- **Test Expert** - Focuses on testing
- **Documentation Expert** - Focuses on docs

### 6.2 Team Formations

Victor supports 5 team formations:

#### 1. Pipeline
Sequential processing through stages.

#### 2. Parallel
Multiple agents work simultaneously, results aggregated.

#### 3. Sequential
Step-by-step execution.

#### 4. Hierarchical
Manager coordinates worker agents.

#### 5. Consensus
All agents vote on decision.

### 6.3 Creating a Team

Create `code_review_team.yaml`:

```yaml
name: comprehensive_review
description: "Multi-agent code review team"
formation: parallel

agents:
  - name: security_reviewer
    role: "Security Expert"
    persona: |
      You are a security expert specializing in:
      - OWASP Top 10 vulnerabilities
      - Authentication and authorization
      - Input validation
      - Cryptography best practices
    tools: [security_audit, vulnerability_scan]

  - name: performance_reviewer
    role: "Performance Expert"
    persona: |
      You are a performance optimization expert specializing in:
      - Algorithm efficiency
      - Database query optimization
      - Caching strategies
      - Memory management
    tools: [profiler, benchmark]

  - name: quality_reviewer
    role: "Code Quality Expert"
    persona: |
      You are a code quality expert specializing in:
      - SOLID principles
      - Design patterns
      - Code maintainability
      - Test coverage
    tools: [complexity_analysis, lint]

aggregation_strategy:
  type: consensus
  threshold: majority
```

### 6.4 Using a Team

```bash
# Create team from YAML
victor team create code_review_team.yaml

# Run team task
victor team run comprehensive_review --file calculator.py
```

### 6.5 Exercise: Team Creation

**Task**: Create a team for feature development.

```yaml
name: feature_dev_team
description: "Full-stack feature development team"
formation: hierarchical

manager:
  name: tech_lead
  role: "Technical Lead"
  persona: |
    You coordinate the development process:
    1. Break down requirements
    2. Assign tasks to specialists
    3. Review and integrate work
    4. Ensure quality standards

workers:
  - name: backend_dev
    role: "Backend Developer"
    persona: "Implement server-side logic and APIs"
    tools: [fastapi, sqlalchemy, pytest]

  - name: frontend_dev
    role: "Frontend Developer"
    persona: "Implement user interface and interactions"
    tools: [react, typescript, jest]

  - name: test_engineer
    role: "Test Engineer"
    persona: "Ensure comprehensive test coverage"
    tools: [pytest, selenium, coverage]
```

Run the team:

```bash
victor team run feature_dev_team "
Implement a user authentication feature with:
- Registration endpoint
- Login endpoint
- Password reset
- Session management
"
```

---

## Lesson 7: Python API

**Goal**: Use Victor programmatically in Python code.

### 7.1 Basic Usage

```python
import asyncio
from victor import Agent

async def main():
    # Create agent
    agent = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-5"
    )

    # Simple query
    result = await agent.run("Explain recursion in Python")
    print(result.content)

    # Clean up
    await agent.close()

asyncio.run(main())
```

### 7.2 Streaming Responses

```python
async def stream_example():
    agent = await Agent.create(provider="openai", model="gpt-4o")

    # Stream response in real-time
    async for chunk in agent.stream("Write a haiku about coding"):
        print(chunk.content, end="", flush=True)
    print()

    await agent.close()

asyncio.run(stream_example())
```

### 7.3 Multi-Turn Conversation

```python
async def conversation():
    agent = await Agent.create(provider="anthropic")

    # First message
    response1 = await agent.run(
        "I'm building a REST API. What's the best framework?"
    )
    print(response1.content)

    # Follow-up (context preserved)
    response2 = await agent.run(
        "Why did you recommend that one?"
    )
    print(response2.content)

    await agent.close()

asyncio.run(conversation())
```

### 7.4 Exercise: Python Integration

**Task**: Create a Python script that uses Victor for code review.

Create `auto_review.py`:

```python
#!/usr/bin/env python3
"""Automatic code review using Victor AI."""

import asyncio
import sys
from pathlib import Path
from victor import Agent, ToolSet

async def review_file(filepath: str):
    """Review a Python file with Victor AI."""

    # Create agent with tools
    agent = await Agent.create(
        provider="anthropic",
        model="claude-sonnet-4-5",
        tools=ToolSet.default()
    )

    # Read file content
    content = Path(filepath).read_text()

    # Review with Victor
    result = await agent.run(f"""
    Review this Python file comprehensively:

    File: {filepath}

    Content:
    {content}

    Analyze:
    1. Security vulnerabilities
    2. Performance issues
    3. Code quality
    4. Best practices
    5. Potential bugs

    Provide specific recommendations with code examples.
    """)

    print(result.content)

    await agent.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python auto_review.py <file>")
        sys.exit(1)

    asyncio.run(review_file(sys.argv[1]))
```

Run it:

```bash
python auto_review.py calculator.py
```

---

## Lesson 8: Advanced Features

**Goal**: Learn advanced capabilities for power users.

### 8.1 Agent Modes

Three modes for different use cases:

#### Build Mode (Default)
Full capabilities, can edit files.

```bash
victor chat --mode build "Implement user authentication"
```

#### Plan Mode
Analysis only, no edits. Great for architecture.

```bash
victor chat --mode plan "Analyze this system design"
```

#### Explore Mode
Deep exploration, no edits. Great for learning.

```bash
victor chat --mode explore "How does this codebase work?"
```

### 8.2 Configuration Profiles

Create profiles in `~/.victor/profiles/`:

`development.yaml`:
```yaml
provider: anthropic
model: claude-sonnet-4-5
mode: build
temperature: 0.7
max_tokens: 4096
```

`production.yaml`:
```yaml
provider: openai
model: gpt-4o
mode: plan
temperature: 0.3
max_tokens: 8192
```

Use profile:

```bash
victor chat --profile development
```

### 8.3 Air-Gapped Mode

For secure, offline environments:

```bash
# Enable air-gapped mode
export VICTOR_AIRGAPPED_MODE=true

# Only local providers available
victor chat --provider ollama
```

Air-gapped mode:
- Disables web tools
- Only allows local providers
- No external API calls
- Full offline functionality

---

## Practice Exercises

### Exercise 1: Code Refactoring (Beginner)

```bash
victor chat "
Refactor calculator.py to:
1. Add proper error handling
2. Include type hints
3. Add docstrings
4. Follow PEP 8 style guide
5. Add input validation
"
```

### Exercise 2: Test Generation (Intermediate)

```bash
victor chat "
Generate comprehensive tests for calculator.py:
1. Test all operations
2. Include edge cases (divide by zero, etc.)
3. Use pytest fixtures
4. Aim for 100% coverage
5. Include integration tests
"
```

### Exercise 3: Multi-Provider Workflow (Intermediate)

```bash
# Start with local model
victor chat --provider ollama "
Brainstorm ways to extend calculator.py with advanced features
"

# Switch to Claude for design
/provider anthropic --model claude-sonnet-4-5
"Design the architecture for the top 3 features"

# Switch to GPT-4 for implementation
/provider openai --model gpt-4o
"Implement the designed features"

# Back to Claude for review
/provider anthropic
"Review the implementation and suggest improvements"
```

### Exercise 4: Workflow Automation (Advanced)

Create a workflow that:
1. Analyzes code quality
2. Runs tests
3. Generates documentation
4. Creates a summary report

### Exercise 5: Multi-Agent Code Review (Advanced)

Create a team with:
- Security specialist
- Performance specialist
- Code quality specialist
- Test coverage specialist

Run comprehensive review on a project.

---

## Next Steps

### Continue Learning

- **[Recipe Book](RECIPES.md)** - Copy-paste solutions for common tasks
- **[FAQ](docs/user-guide/faq.md)** - Answers to common questions
- **[User Guide](docs/user-guide/index.md)** - Comprehensive documentation
- **[API Reference](docs/api/README.md)** - Complete API documentation

### Practice Projects

1. **Build a CLI Tool**: Use Victor to build a command-line tool
2. **Create a Web API**: Full-stack development with Victor
3. **Automate Testing**: Set up automated testing workflow
4. **Documentation Generator**: Auto-generate docs for a project
5. **Code Refactoring**: Improve a legacy codebase

### Join the Community

- **GitHub**: https://github.com/vjsingh1984/victor
- **Discussions**: https://github.com/vjsingh1984/victor/discussions
- **Issues**: https://github.com/vjsingh1984/victor/issues

---

## Summary

You've completed the Victor AI tutorial! You now know how to:

- Have effective conversations with Victor
- Use tools for code analysis and manipulation
- Automate tasks with workflows
- Switch providers for optimal cost/quality
- Manage sessions effectively
- Coordinate multi-agent teams
- Use Victor programmatically
- Configure advanced features

**What's next?** Practice, experiment, and build something amazing!

---

**Version**: 0.5.1
**Last Updated**: January 20, 2026
**Tutorial Version**: 1.0
