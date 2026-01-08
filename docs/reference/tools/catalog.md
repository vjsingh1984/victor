# Tool Catalog (auto-generated)

Descriptions are truncated for brevity. Use `victor tools list` for full details.

| Tool | Description |
| --- | --- |
| `arch_summary` | Return a structured architecture snapshot (modules + symbol hotspots). Uses module-level PageRank and centrality plus symbol-level... |
| `audit` | Access audit logs and compliance reports. Supported frameworks: - SOC 2 Type II - GDPR - HIPAA - PCI DSS - ISO 27001 Actions: - summary:... |
| `batch` | Unified batch processing tool for multi-file operations. Performs parallel batch operations on multiple files including search, replace,... |
| `cache` | Unified cache management tool for Victor's tiered caching system. Actions: - stats: Get cache statistics and performance metrics -... |
| `cicd` | Unified CI/CD tool for pipeline management. Performs CI/CD operations including generating configurations, validating pipelines, and... |
| `commit_msg` | Generate an AI-powered commit message from staged changes. Analyzes the staged diff and generates a conventional commit message using... |
| `conflicts` | Analyze merge conflicts and provide resolution guidance. Detects files with merge conflicts (marked as UU in git status) and provides... |
| `db` | Unified database tool for SQL operations. Supports SQLite, PostgreSQL, MySQL, SQL Server. Actions: - connect: Connect to a database -... |
| `deps` | Python dependency management: list, outdated, security, generate, update, tree, check. Actions: list (packages), outdated, security... |
| `docker` | Unified Docker operations for container and image management. Provides a single interface for all Docker operations, mirroring the... |
| `docs` | Unified documentation generation tool. Generate multiple types of documentation (docstrings, API docs, README sections, type hints) in a... |
| `docs_coverage` | Analyze documentation coverage and quality. Checks documentation coverage (percentage of functions/classes with docstrings) and... |
| `edit` | Edit files atomically with undo. REQUIRED: 'ops' parameter with operation list. IMPORTANT: You MUST provide the 'ops' parameter.... |
| `extract` | Extract code block into a new function. Extracts selected lines of code into a new function, analyzing variables to determine parameters... |
| `find` | Find files by name pattern (like Unix find -name). Searches recursively through the directory tree to locate files matching the given... |
| `git` | Unified git operations: status, diff, stage, commit, log, branch. Operations: status, diff (staged=True for staged), stage (files or... |
| `graph` | [GRAPH] Query codebase STRUCTURE for relationships, impact, and importance. Uses the code graph built from AST analysis for structural... |
| `grep` | Find code by CONCEPT or TEXT when you DON'T know exact location/name. Use this tool for exploration when you need to discover where... |
| `http` | Unified HTTP operations for requests and API testing. Modes: - "request": Standard HTTP request (default) - "test": API testing with... |
| `iac` | Scan IaC files (Terraform, Docker, K8s) for security issues. Actions: scan, scan_file, summary, detect. Detects: secrets, IAM misconfig,... |
| `inline` | Inline a variable by replacing usages with its assigned value. |
| `jira` | Perform operations on Jira issues. |
| `ls` | List directory contents with file sizes. |
| `lsp` | Language Server Protocol operations for code intelligence. Actions: status, start, stop, completions, hover, definition, references,... |
| `mcp` | Call an MCP tool by name (prefixed with the MCP namespace). |
| `merge` | Detect and resolve git merge conflicts. Actions: detect, analyze, resolve (auto), apply (ours/theirs), abort. Smart strategies: trivial... |
| `metrics` | Comprehensive code metrics and quality analysis. Analyzes code quality metrics including complexity, maintainability, technical debt,... |
| `organize_imports` | Organize imports: sort into groups (stdlib/third-party/local), remove duplicates. |
| `overview` | Get a curated project overview for initial exploration. Provides: 1. Directory structure at max_depth (default: 2 levels) 2. Top... |
| `patch` | Unified patch operations: create diffs or apply patches. Operations: - "apply": Apply a unified diff patch to files (default) -... |
| `pipeline` | Analyze CI/CD pipelines (GitHub Actions, GitLab) and coverage. Actions: analyze, coverage, compare_coverage, summary, detect. Coverage... |
| `pr` | Create a GitHub pull request with auto-generated or custom content. Creates a pull request using GitHub CLI (gh). Automatically pushes... |
| `rag_delete` | Delete a document from the RAG knowledge base by ID |
| `rag_ingest` | Ingest documents into the RAG knowledge base. Supports local files, URLs, and directories. |
| `rag_list` | List all documents in the RAG knowledge base |
| `rag_query` | Query the RAG knowledge base and synthesize an answer using an LLM. Returns an answer with source citations grounded in retrieved documents. |
| `rag_search` | Search the RAG knowledge base for relevant document chunks |
| `rag_stats` | Get statistics about the RAG knowledge base |
| `read` | Read text/code file. Binary files rejected. TRUNCATION LIMITS: - Cloud models: Maximum 750 lines OR 25KB (whichever is reached first) -... |
| `refs` | [AST-AWARE] Find all USAGES of a symbol across the project. Project-wide scan using AST parsing. More accurate than grep (exact... |
| `rename` | [AST-AWARE] Rename symbols safely using word-boundary matching. Uses AST parsing + word boundaries to rename symbols without false... |
| `review` | Comprehensive code review for automated quality analysis. Performs code review including security checks, complexity analysis, best... |
| `sandbox` | Unified sandbox operations for code execution in isolated Docker container. Operations: - "execute": Run Python code in the sandbox -... |
| `scaffold` | Unified project scaffolding tool. Performs scaffolding operations including creating projects from templates, listing available... |
| `scan` | Comprehensive security scanning for code analysis. Performs security scans including secret detection, dependency vulnerability checks,... |
| `shell` | Run shell command with safety checks. Returns stdout/stderr/return_code. |
| `shell_readonly` | Execute readonly shell commands for exploration (pwd, ls, cat, grep, git status, etc). This tool only allows safe, readonly commands... |
| `slack` | Perform operations on Slack. |
| `symbol` | [AST-AWARE] Get FULL CODE of a function/class definition in a specific file. Returns the complete code block (not just a reference). Use... |
| `teams` | Perform operations on Microsoft Teams. |
| `test` | Runs tests using pytest and returns a structured summary of the results. This tool runs pytest on the specified path and captures the... |
| `workflow` | Runs a pre-defined, multi-step workflow to automate a complex task. |
| `write` | Write file. Creates parent dirs. Use edit_files for partial edits. |