# Coding Vertical

The Coding vertical provides comprehensive code analysis, review, refactoring,
  and generation capabilities. It is designed to compete with GitHub Copilot Workspace, Cursor, and Sourcegraph Cody.

## Overview

The Coding vertical (`victor/coding/`) is Victor's core vertical for software development tasks. It provides intelligent code understanding through Tree-sitter AST parsing, LSP integration for real-time code intelligence, and semantic code search capabilities.

### Key Use Cases

- **Code Review**: Automated analysis of code changes with quality, security, and performance feedback
- **Feature Implementation**: End-to-end feature development with planning, implementation, and testing
- **Test-Driven Development (TDD)**: Red-green-refactor workflow with test generation
- **Refactoring**: Safe code restructuring with AST-aware transformations
- **Code Analysis**: Static analysis, dependency graphing, and codebase exploration

## Available Tools

The Coding vertical uses the following tools from `victor.tools.tool_names`:

| Tool | Description |
|------|-------------|
| `read` | Read file contents for analysis |
| `write` | Write new files or overwrite existing ones |
| `edit` | Make targeted edits to existing files |
| `ls` | List directory contents |
| `grep` | Keyword-based code search |
| `code_search` | Semantic code search using embeddings |
| `overview` | Generate codebase overview and structure |
| `graph` | Code graph analysis (PageRank, dependencies) |
| `shell` | Execute shell commands for builds, tests, etc. |

## Available Workflows

### 1. Code Review (`code_review.yaml`)

Systematic code review workflow with multiple analysis phases:

```yaml
workflows:
  code_review:
    nodes:
      - understand_changes    # Analyze what changed
      - review_correctness    # Verify logic and behavior
      - review_quality        # Code style and best practices
      - review_security       # Security vulnerability check
      - review_performance    # Performance considerations
      - synthesize_review     # Combine all feedback
      - generate_report       # Create review document
```

**Key Features**:
- Parallel review of correctness, quality, security, and performance
- Configurable review depth (quick, standard, thorough)
- HITL approval gates for critical issues
- Structured output with actionable suggestions

### 2. Feature Implementation (`feature.yaml`)

End-to-end feature development workflow:

```yaml
workflows:
  feature:
    nodes:
      - understand_requirements  # Parse feature request
      - analyze_codebase        # Understand existing code
      - create_plan             # Design implementation approach
      - implement_changes       # Write the code
      - write_tests             # Create test coverage
      - verify_implementation   # Run tests and validate
      - finalize                # Clean up and document
```

**Key Features**:
- Requirement analysis with clarification requests
- Codebase exploration before implementation
- Incremental implementation with verification
- Automatic test generation

### 3. Test-Driven Development (`tdd.yaml`)

Red-green-refactor TDD workflow:

```yaml
workflows:
  tdd:
    nodes:
      - understand_feature     # What to build
      - write_failing_test     # RED: Create test first
      - run_tests_red          # Verify test fails
      - implement_minimum      # GREEN: Make it pass
      - run_tests_green        # Verify test passes
      - refactor               # REFACTOR: Improve code
      - run_tests_final        # Verify still passes
```

**Key Features**:
- Strict TDD cycle enforcement
- Automatic test execution at each phase
- Refactoring suggestions based on code smells
- Coverage tracking

### 4. Refactoring (`refactor.yaml`)

Safe code restructuring workflow:

```yaml
workflows:
  refactor:
    nodes:
      - analyze_code           # Understand current structure
      - identify_improvements  # Find refactoring opportunities
      - plan_refactoring       # Design safe transformation steps
      - apply_changes          # Execute refactoring
      - verify_behavior        # Ensure no regressions
      - update_tests           # Adjust tests if needed
```

**Key Features**:
- AST-aware transformations
- Behavior preservation verification
- Incremental changes with rollback capability
- Test suite validation

## Stage Definitions

The Coding vertical progresses through these stages:

| Stage | Description | Primary Tools |
|-------|-------------|---------------|
| `INITIAL` | Understanding the coding request | `read`, `ls`, `overview` |
| `UNDERSTANDING` | Analyzing existing code | `read`, `grep`, `code_search`, `graph` |
| `PLANNING` | Creating implementation plan | `read`, `overview` |
| `IMPLEMENTING` | Writing and modifying code | `write`, `edit`, `shell` |
| `TESTING` | Running and writing tests | `shell`, `write`, `edit` |
| `REVIEWING` | Verifying changes | `read`, `grep`, `shell` |
| `COMPLETION` | Finalizing deliverables | `write`, `read` |

## Key Features

### Tree-sitter Integration

The Coding vertical uses Tree-sitter for fast, accurate AST parsing across 20+ programming languages:

```python
from victor.coding.ast.tree_sitter_manager import TreeSitterManager

manager = TreeSitterManager()
tree = manager.parse_file("example.py")
# Access nodes, find definitions, analyze structure
```

**Supported Languages**: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, and more.

### LSP Integration

Language Server Protocol support provides:
- Go-to-definition
- Find references
- Symbol search
- Diagnostics
- Hover information

```python
from victor.coding.lsp.manager import LSPManager

lsp = LSPManager()
await lsp.start_server("python")
definitions = await lsp.get_definitions(file_path, position)
```

### Code Validation (Pre-Write Enforcement)

The Coding vertical integrates with the **Unified Language Capability System** to validate code before writing to disk.
  This prevents syntax errors from being committed:

```python
from victor.core.language_capabilities.hooks import validate_code_before_write

# Validate before writing
should_proceed, result = validate_code_before_write(
    content="def foo():\n    pass",
    file_path=Path("main.py"),
    strict=False
)

if not should_proceed:
    print(f"Validation failed: {result.errors}")
```

**Supported Languages** (40+):
- **Tier 1** (Full Support): Python, JavaScript, TypeScript - Native AST + Tree-sitter + LSP
- **Tier 2** (Good Support): Go, Java, Rust, C, C++ - Native/Tree-sitter + LSP
- **Tier 3** (Basic Support): Ruby, PHP, Scala, Kotlin, Swift, etc. - Tree-sitter + optional LSP
- **Config Files**: JSON, YAML, TOML, XML, HOCON, Markdown - Native Python validators

**CLI Usage**:
```bash
victor validate files main.py config.json    # Validate files
victor validate files src/*.py --strict      # Exit 1 on any error
victor validate languages                    # List supported languages
victor validate check app.ts                 # Check validation support
```

**Environment Variables**:
```bash
VICTOR_VALIDATION_ENABLED=false    # Disable validation globally
VICTOR_STRICT_VALIDATION=true      # Block writes on any error
```

### Semantic Code Search

Beyond keyword search, the Coding vertical supports semantic code search using embeddings:

```python
# Find code semantically similar to a query
results = await code_search.search("function that validates user input")
```

### Code Graph Analysis

Understand code relationships through graph analysis:
- Dependency graphs
- Call graphs
- Import relationships
- PageRank-based importance scoring

## Configuration Options

### Vertical Configuration

```python
from victor.coding.assistant import CodingAssistant

# Get system prompt for coding tasks
prompt = CodingAssistant.get_system_prompt()

# Get tiered tool configuration
tiered_tools = CodingAssistant.get_tiered_tools()

# Access capability provider
capabilities = CodingAssistant.get_capability_provider()
```

### Mode Configuration

The Coding vertical supports three agent modes:

| Mode | Description | Tool Access |
|------|-------------|-------------|
| `BUILD` | Full implementation (default) | All tools, full edits |
| `PLAN` | Planning and exploration | 2.5x exploration, sandbox only |
| `EXPLORE` | Analysis only | 3.0x exploration, no edits |

### Workflow Parameters

Common workflow configuration options:

```yaml
# In workflow YAML
llm_config:
  temperature: 0.2      # Lower for precise code generation
  model_hint: claude-sonnet  # Preferred model

tool_budget: 30         # Max tool calls per node
timeout: 300            # Node timeout in seconds
```

## Example Usage

### Basic Code Review

```python
from victor.coding.workflows import CodingWorkflowProvider

provider = CodingWorkflowProvider()
workflow = provider.compile_workflow("code_review")

result = await workflow.invoke({
    "diff": git_diff_content,
    "context": {
        "repository": "my-project",
        "branch": "feature/new-api"
    }
})

print(result["review_report"])
```

### Feature Implementation

```python
result = await workflow.invoke({
    "feature_request": "Add user authentication with JWT tokens",
    "codebase_path": "/path/to/project",
    "test_framework": "pytest"
})
```

### Using the Coding Assistant Directly

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(
    vertical="coding",
    provider="anthropic",
    model="claude-sonnet-4-5"
)

response = await orchestrator.chat(
    "Review the changes in the last commit for security issues"
)
```

## Integration with Other Verticals

The Coding vertical integrates with:

- **DevOps**: Deployment pipelines after code changes
- **RAG**: Documentation search for context
- **Research**: Technical documentation lookup

## File Structure

```
victor/coding/
├── assistant.py          # CodingAssistant vertical definition
├── capabilities.py       # Capability providers
├── mode_config.py        # Mode configurations
├── prompts.py            # Prompt templates
├── safety.py             # Safety checks for code operations
├── tool_dependencies.py  # Tool dependency configuration
├── ast/
│   ├── tree_sitter_manager.py  # Tree-sitter integration
│   └── ...
├── lsp/
│   ├── manager.py        # LSP client management
│   └── ...
├── codebase/
│   ├── analyzer.py       # Codebase analysis
│   └── ...
├── review.py             # Code review logic
├── test_generation.py    # Test generation utilities
├── workflows/
│   ├── code_review.yaml
│   ├── feature.yaml
│   ├── tdd.yaml
│   └── refactor.yaml
├── handlers.py           # Compute handlers for workflows
├── escape_hatches.py     # Complex condition logic
├── rl.py                 # Reinforcement learning config
└── teams.py              # Multi-agent team specs
```

## Best Practices

1. **Start with exploration**: Use `EXPLORE` mode to understand the codebase before making changes
2. **Use semantic search**: Leverage `code_search` for finding relevant code beyond keyword matching
3. **Validate incrementally**: Run tests after each significant change
4. **Review before commit**: Use the code review workflow for self-review
5. **Document changes**: Update documentation as part of the workflow

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
