# Intelligent Tool Selection

> Status: This document includes planned behavior; verify current capabilities in README.md (Reality Check) and CODEBASE_ANALYSIS_REPORT.md.

Victor implements smart, context-aware tool selection to optimize performance and prevent overwhelming smaller models.

## How It Works

Instead of sending all 25+ enterprise tools with every request, Victor:

1. **Analyzes your prompt** for keywords and intent
2. **Selects relevant tools** from categorized toolsets
3. **Limits tools for small models** (max 10 for <7B parameters)
4. **Always includes core tools** (filesystem, bash, editor)

## Tool Categories

Victor organizes tools into semantic categories:

| Category | Tools | Triggered By Keywords |
|----------|-------|----------------------|
| **Core** | read_file, write_file, list_directory, execute_bash, file_editor_* | Always included |
| **Git** | git_status, git_diff, git_commit, git_branch | git, commit, branch, merge, repository |
| **Testing** | testing_generate, testing_run, testing_coverage | test, pytest, unittest, coverage |
| **Security** | security_scan_secrets, security_scan_dependencies | security, vulnerability, secret, scan |
| **Docs** | docs_generate_docstrings, docs_generate_api | document, docstring, readme, api doc |
| **Review** | code_review_file, code_review_security | review, analyze code, code quality |
| **Refactor** | refactor_rename_symbol, refactor_extract_function | refactor, rename, extract, reorganize |
| **Web** | web_search, web_fetch, web_summarize | search web, look up, find online |
| **Docker** | docker_ps, docker_images, docker_logs | docker, container, image |
| **Metrics** | metrics_complexity, metrics_maintainability | complexity, metrics, technical debt |

## Examples

### Simple Code Generation
**Prompt**: "Write a Python function to calculate fibonacci numbers"

**Tools Selected**: 8 (core only)
- read_file, write_file, list_directory, execute_bash
- file_editor_start_transaction, file_editor_add_create, file_editor_add_modify, file_editor_commit

**Result**: Fast response, focused tools

### Security Review
**Prompt**: "Review code for security issues and generate tests"

**Tools Selected**: 10 (core + security + testing, capped at 10 for small models)
- 8 core tools
- security_scan_secrets, security_scan_dependencies

**Result**: Model correctly uses security scanning tools

### Complex Multi-Category Task
**Prompt**: "Refactor the authentication module, add tests, update docs, and commit changes"

**Large Model (7B+)**: 20+ relevant tools across categories
- Core (8) + Testing (3) + Docs (3) + Git (6) + Refactor (3)

**Small Model (<7B)**: Limited to 10 most relevant
- Core tools prioritized, then most relevant from other categories

## Performance Benefits

### Before (All Tools)
- **Request payload**: ~50KB+ (25+ tool definitions)
- **Small models**: Hang/timeout, overwhelmed
- **Response time**: 60s+ timeout or failure

### After (Intelligent Selection)
- **Request payload**: ~15KB (8-10 relevant tools)
- **Small models**: Fast, responsive
- **Response time**: 12-40s for qwen2.5-coder:1.5b

## Model-Specific Behavior

### Small Models (<7B parameters)
Examples: qwen2.5-coder:0.5b, qwen2.5-coder:1.5b, llama3:3b

- **Max tools**: 10 (hard limit)
- **Priority**: Core tools first, then category-specific
- **Rationale**: Prevent context overwhelm, reduce inference time

### Medium/Large Models (7B+ parameters)
Examples: qwen2.5-coder:7b, llama3:8b, codellama:13b

- **Max tools**: No limit (send all relevant)
- **Selection**: All tools matching prompt categories
- **Rationale**: Can handle complex toolsets effectively

## Tool Discovery Pattern

Victor supports progressive tool discovery:

1. **Initial request**: Send focused toolset based on prompt
2. **Model response**: May request additional tools if needed
3. **Follow-up**: Victor can provide more tools dynamically

This mirrors MCP (Model Context Protocol) approach where tools are discovered on-demand rather than pre-loaded.

## Configuration

### Enable/Disable Categories

You can configure which tool categories are available:

```python
# In your profile or configuration
tool_categories = [
    "core",      # Always enabled
    "git",
    "testing",
    "security"   # Disable others for restricted environments
]
```

### Adjust Small Model Threshold

```python
# Consider models up to 3B as "small"
small_model_threshold = "3b"

# Or increase limit for newer efficient small models
max_tools_small_model = 15
```

## Implementation Details

The `AgentOrchestrator._select_relevant_tools()` method:

1. Tokenizes user message to lowercase
2. Matches keywords against category triggers
3. Builds selected tool set (core + matched categories)
4. Checks if model is "small" (< 7B)
5. Limits to 10 tools if small model detected
6. Prioritizes core tools in limiting

```python
def _select_relevant_tools(self, user_message: str) -> List[ToolDefinition]:
    # Keyword matching
    if any(kw in message_lower for kw in ["test", "pytest", "coverage"]):
        selected_categories.add("testing")

    # Small model detection
    is_small_model = any(":0.5b", ":1.5b", ":3b" in self.model.lower())

    # Limit for small models
    if is_small_model and len(selected_tools) > 10:
        selected_tools = prioritize_core_tools(selected_tools, max=10)
```

## Future Enhancements

1. **LLM-based tool selection**: Use a small model to analyze prompts and select tools
2. **User feedback loop**: Learn which tools are most effective for prompt patterns
3. **Dynamic tool limits**: Adjust based on model context window and inference speed
4. **MCP server integration**: Full tool discovery protocol support
5. **Tool usage analytics**: Track which tools are used most/least effectively

## Benefits Summary

- **Faster responses**: Smaller payloads, less processing
- **Better accuracy**: Models focus on relevant tools
- **Scale to small models**: Enable tool use on 1.5B models
- **Cost efficiency**: Less tokens, lower API costs
- **Improved UX**: Transparent, logged tool selection

---

**Example Log Output**:
```
2025-11-26 19:51:54,795 - victor.agent.orchestrator - INFO - Selected 10 tools for prompt (small_model=True): read_file, write_file, list_directory, execute_bash, file_editor_start_transaction, file_editor_add_create, file_editor_add_modify, file_editor_commit, security_scan_secrets, security_scan_dependencies
```

This shows exactly which tools were selected and why, providing transparency and debuggability.
