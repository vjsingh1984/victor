# Victor Coding

**AI-powered coding assistant vertical for the Victor framework.**

Victor Coding adds intelligent code analysis and assistance capabilities:

- **Tree-sitter Parsing**: AST-aware code understanding for 10+ languages
- **LSP Integration**: Language Server Protocol for IDE-like features
- **25 Coding Tools**: Code search, review, refactoring, test generation, and more
- **Semantic Code Search**: Vector-based code similarity search
- **Security Analysis**: Vulnerability detection and secret scanning

## Installation

```bash
pip install victor-coding
```

This will automatically install `victor-core` as a dependency.

For the full experience with all providers and tools:

```bash
pip install victor-ai
```

## Included Tools

| Tool | Description |
|------|-------------|
| `code_search` | Fast keyword-based code search |
| `semantic_code_search` | Vector similarity code search |
| `code_review` | AI-powered code review |
| `refactor` | Intelligent refactoring suggestions |
| `test_generator` | Automatic test case generation |
| `symbol_lookup` | Find symbol definitions and references |
| `ast_query` | Query code AST with patterns |
| `dependency_analyzer` | Analyze project dependencies |
| `complexity_analyzer` | Cyclomatic complexity analysis |
| `dead_code_finder` | Detect unused code |
| ... and 15 more |

## Quick Start

```python
from victor_coding import CodingVertical
from victor.config.settings import Settings

# Initialize the coding vertical
settings = Settings(provider="anthropic")
coding = CodingVertical(settings)

# Get available tools
tools = coding.get_tools()
print(f"Available tools: {[t.name for t in tools]}")
```

## Supported Languages

- Python
- JavaScript/TypeScript
- Rust
- Go
- Java
- C/C++
- Ruby
- PHP
- More via Tree-sitter

## Documentation

- [Full Documentation](https://github.com/vijayksingh/victor)
- [Tool Catalog](https://github.com/vijayksingh/victor/blob/main/docs/TOOL_CATALOG.md)
- [Migration Guide](https://github.com/vijayksingh/victor/blob/main/docs/PACKAGE_MIGRATION.md)

## License

Apache-2.0
