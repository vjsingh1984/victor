# Tool Dependency Graph (Scaffold)

This codebase includes a minimal dependency graph in `victor/tools/dependency_graph.py` to help plan chained tool calls. Current defaults cover:

- `code_search` / `semantic_code_search` / `plan_files` → outputs `file_candidates`
- `read_file` → outputs `file_contents`
- `analyze_docs` / `code_review` → outputs `summary`

The orchestrator registers these specs on startup (`_register_default_tool_dependencies`). The planner today is deterministic and returns a simple order for requested goals based on available inputs; wiring into automatic multi-step planning is a future step.
