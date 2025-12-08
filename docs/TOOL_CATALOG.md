# Tool Catalog (auto-generated)

| Tool | Description |
| --- | --- |
| `audit` | Access audit logs and compliance reports. Supported frameworks: - SOC 2 Type II - GDPR - HIPAA - PCI DSS - ISO 27001 Actions: - summary: Get audit activity summary - report: Generate compliance report - query: Query audit events - compliance: Check compliance status for a framework - export: Export audit logs to file |
| `batch` | Unified batch processing tool for multi-file operations. Performs parallel batch operations on multiple files including search, replace, analyze, and list operations. Consolidates all batch processing functionality into a single interface. |
| `cache` | Unified cache management tool for Victor's tiered caching system. Actions: - stats: Get cache statistics and performance metrics - clear: Clear cache entries (optionally by namespace) - info: Get cache configuration information |
| `cicd` | Unified CI/CD tool for pipeline management. Performs CI/CD operations including generating configurations, validating pipelines, and listing available templates. Consolidates all CI/CD functionality into a single interface. |
| `commit_msg` | Generate an AI-powered commit message from staged changes. Analyzes the staged diff and generates a conventional commit message using the configured LLM provider. The generated message follows the conventional commit format: type(scope): subject Types used: feat, fix, docs, style, refactor, test, chore |
| `conflicts` | Analyze merge conflicts and provide resolution guidance. Detects files with merge conflicts (marked as UU in git status) and provides detailed information about each conflict, including previews of conflict markers and step-by-step resolution instructions. |
| `db` | Unified database tool for SQL operations. Supports SQLite, PostgreSQL, MySQL, SQL Server. Actions: - connect: Connect to a database - query: Execute SQL queries - tables: List all tables - describe: Describe a table's structure - schema: Get complete database schema - disconnect: Close connection |
| `deps` | Python dependency management: list, outdated, security, generate, update, tree, check. Actions: list (packages), outdated, security (vulns), generate (requirements.txt), update (packages, dry_run), tree (package), check (requirements_file). |
| `docker` | Unified Docker operations for container and image management. Provides a single interface for all Docker operations, mirroring the Docker CLI structure. Consolidates 15 separate Docker tools into one. |
| `docs` | Unified documentation generation tool. Generate multiple types of documentation (docstrings, API docs, README sections, type hints) in a single unified interface. Consolidates all documentation generation functionality. |
| `docs_coverage` | Analyze documentation coverage and quality. Checks documentation coverage (percentage of functions/classes with docstrings) and optionally analyzes documentation quality. |
| `edit` | [TEXT-BASED] Edit files atomically with undo. NOT code-aware. Performs literal string replacement. Does NOT understand code structure. WARNING: May cause false positives in code (e.g., 'foo' matches 'foobar'). For Python symbol renaming, use rename() from refactor_tool instead. Ops: replace/create/modify/delete/rename. |
| `extract` | Extract code block into a new function. Extracts selected lines of code into a new function, analyzing variables to determine parameters and return values. |
| `fetch` | Fetch and extract main text content from a URL. |
| `git` | Unified git operations: status, diff, stage, commit, log, branch. Operations: status, diff (staged=True for staged), stage (files or all), commit (message required), log (limit), branch (list/create/switch). Supports custom author_name/author_email for commits. |
| `http` | Unified HTTP operations for requests and API testing. Modes: - "request": Standard HTTP request (default) - "test": API testing with validation |
| `iac_scanner` | Scan IaC files (Terraform, Docker, K8s) for security issues. Actions: scan, scan_file, summary, detect. Detects: secrets, IAM misconfig, missing encryption, network exposure. |
| `inline` | Inline a variable by replacing usages with its assigned value. |
| `ls` | List directory contents. |
| `lsp` | Language Server Protocol operations for code intelligence. Actions: status, start, stop, completions, hover, definition, references, diagnostics. Position-based actions require: file_path, line, character. |
| `mcp` | Call an MCP tool by name (prefixed with the MCP namespace). |
| `merge_conflicts` | Detect and resolve git merge conflicts. Actions: detect, analyze, resolve (auto), apply (ours/theirs), abort. Smart strategies: trivial (whitespace), import (sort/combine), union. |
| `metrics` | Comprehensive code metrics and quality analysis. Analyzes code quality metrics including complexity, maintainability, technical debt, and code structure. Consolidates multiple metric types into a single unified interface. |
| `organize_imports` | Organize imports: sort into groups (stdlib/third-party/local), remove duplicates. |
| `overview` | Get a curated project overview for initial exploration. Provides: 1. Directory structure at max_depth (default: 2 levels) 2. Top documentation files (README*, ARCHITECTURE*, INDEX*, etc.) 3. Largest source files by size (helps identify core modules) |
| `patch` | Unified patch operations: create diffs or apply patches. Operations: - "apply": Apply a unified diff patch to files (default) - "create": Create a unified diff from file and new content |
| `pipeline_analyzer` | Analyze CI/CD pipelines (GitHub Actions, GitLab) and coverage. Actions: analyze, coverage, compare_coverage, summary, detect. Coverage formats: Cobertura, LCOV, JaCoCo. |
| `pr` | Create a GitHub pull request with auto-generated or custom content. Creates a pull request using GitHub CLI (gh). Automatically pushes the current branch to origin if needed. If title or description are not provided and an AI provider is configured, generates them from the commit history and diff. |
| `read` | Read text/code file. Binary files rejected. |
| `refs` | [AST-AWARE] Find all references to a symbol using tree-sitter parsing. Scans Python files and identifies exact identifier matches using AST analysis. More accurate than grep for finding symbol usages (won't match substrings). |
| `rename` | [AST-AWARE] Rename symbols safely using word-boundary matching. Uses AST parsing + word boundaries to rename symbols without false positives. SAFE: Won't rename 'get_user' to 'fetch_user' inside 'get_username'. Use this for Python symbol refactoring. Use edit() for non-code text changes. |
| `review` | Comprehensive code review for automated quality analysis. Performs code review including security checks, complexity analysis, best practices validation, and documentation coverage. Consolidates multiple review aspects into a single unified interface. |
| `sandbox` | Unified sandbox operations for code execution in isolated Docker container. Operations: - "execute": Run Python code in the sandbox - "upload": Upload local files to the sandbox |
| `scaffold` | Unified project scaffolding tool. Performs scaffolding operations including creating projects from templates, listing available templates, adding files, and initializing git repositories. Consolidates all scaffolding functionality into a single interface. |
| `scan` | Comprehensive security scanning for code analysis. Performs security scans including secret detection, dependency vulnerability checks, and configuration security analysis. Consolidates multiple scan types into a single unified interface. |
| `search` | Unified code search with multiple modes. Modes: - "semantic": Embedding-based search. Best for concepts, patterns, inheritance. - "literal": Keyword matching (like grep). Best for exact text/identifiers. |
| `shell` | Run shell command with safety checks. Returns stdout/stderr/return_code. |
| `symbol` | [AST-AWARE] Find function/class definition using tree-sitter parsing. Uses AST analysis for accurate symbol lookup. For Python code analysis only. Use this instead of grep/text search when you need precise symbol definitions. |
| `test` | Runs tests using pytest and returns a structured summary of the results. This tool runs pytest on the specified path and captures the output in JSON format, providing a clean summary of test outcomes. |
| `web` | Search the web using DuckDuckGo. Optionally summarize with AI. Purpose: - Find links, docs, references on the public web. - Return titles, URLs, and snippets for relevance checking. - Optionally use AI to summarize search results. - Ideal when the user says "search the web", "find online", "lookup", "docs", "articles". |
| `workflow` | Runs a pre-defined, multi-step workflow to automate a complex task. |
| `write` | Write file. Creates parent dirs. Use edit_files for partial edits. |