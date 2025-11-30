# Tool Catalog (auto-generated)

| Tool | Description |
| --- | --- |
| `analyze_docs` | Analyze documentation coverage and quality. Checks documentation coverage (percentage of functions/classes with docstrings) and optionally analyzes documentation quality. |
| `analyze_metrics` | Comprehensive code metrics and quality analysis. Analyzes code quality metrics including complexity, maintainability, technical debt, and code structure. Consolidates multiple metric types into a single unified interface. |
| `batch` | Unified batch processing tool for multi-file operations. Performs parallel batch operations on multiple files including search, replace, analyze, and list operations. Consolidates all batch processing functionality into a single interface. |
| `cache_clear` | Clear cache entries. Can clear all cache or just a specific namespace. This affects both memory and disk caches. |
| `cache_info` | Get cache configuration information. Shows the current cache settings including memory/disk configuration, TTL settings, and enabled features. |
| `cache_stats` | Get cache statistics and performance metrics. Shows hit rates, counts, and cache sizes for Victor's tiered caching system (memory + disk). |
| `cicd` | Unified CI/CD tool for pipeline management. Performs CI/CD operations including generating configurations, validating pipelines, and listing available templates. Consolidates all CI/CD functionality into a single interface. |
| `code_search` | Lightweight code search: find relevant files/chunks for a query. Scans files under `root`, scores them by keyword match, and returns top-k with snippets. Use this before read_file to pick real targets. Caps files and snippet lengths for speed. |
| `database_connect` | Connect to a database. Supports SQLite, PostgreSQL, MySQL, and SQL Server. Returns a connection_id that must be used for subsequent operations. |
| `database_describe` | Describe a table's structure. Returns column information including names, types, nullable status, and primary key information. |
| `database_disconnect` | Disconnect from database. Closes the connection and removes it from the connection pool. |
| `database_query` | Execute a SQL query. Supports SELECT queries (returns results) and modification queries (INSERT/UPDATE/DELETE - if allowed). Read-only by default for safety. |
| `database_schema` | Get complete database schema. Returns information about all tables and their columns. This is a convenience function that combines database_tables and database_describe for all tables. |
| `database_tables` | List all tables in the database. |
| `dependency_check` | Check if installed packages match requirements file. Verifies that all packages specified in the requirements file are installed with the correct versions. |
| `dependency_generate` | Generate a requirements file from installed packages. Creates a requirements.txt file listing all installed packages with their versions. |
| `dependency_list` | List all installed Python packages. Returns a formatted list of all installed packages with their versions, grouped by first letter for easy browsing. |
| `dependency_outdated` | Check for outdated Python packages. Identifies packages that have newer versions available and categorizes them by update severity (major, minor, patch). |
| `dependency_security` | Check for security vulnerabilities in dependencies. Scans installed packages against known vulnerability databases to identify security issues. |
| `dependency_tree` | Show dependency tree. Displays the dependency tree for a specific package or all packages. Requires 'pipdeptree' to be installed. |
| `dependency_update` | Update Python packages. Updates specified packages to their latest versions. Runs in dry-run mode by default for safety. |
| `docker` | Unified Docker operations for container and image management. Provides a single interface for all Docker operations, mirroring the Docker CLI structure. Consolidates 15 separate Docker tools into one. |
| `edit_files` | Unified file editing with transaction support. Perform multiple file operations (create, modify, delete, rename) in a single transaction with built-in preview and rollback capability. Consolidates all file editing functionality into one unified interface. |
| `execute_bash` | Execute a bash command and return its output. This tool allows executing shell commands with safety checks to prevent dangerous operations. Commands are executed with a configurable timeout and working directory. |
| `execute_python_in_sandbox` | Executes a block of Python code in a stateful, sandboxed environment. Files can be uploaded to the sandbox using `upload_files_to_sandbox`. |
| `find_references` | Finds all references to a symbol in a directory using AST parsing. |
| `find_symbol` | Finds the definition of a class or function in a Python file using AST parsing. |
| `generate_docs` | Unified documentation generation tool. Generate multiple types of documentation (docstrings, API docs, README sections, type hints) in a single unified interface. Consolidates all documentation generation functionality. |
| `git` | Unified git operations tool. Performs common git operations (status, diff, stage, commit, log, branch) through a single unified interface. Consolidates basic git functionality. |
| `git_analyze_conflicts` | Analyze merge conflicts and provide resolution guidance. Detects conflicted files and provides information about the conflicts, including conflict markers and resolution steps. |
| `git_create_pr` | Create a pull request with auto-generated content. Creates a pull request using GitHub CLI. If title or description are not provided and AI is available, generates them from the commits. |
| `git_suggest_commit` | Generate AI commit message from staged changes. Analyzes the staged diff and generates a conventional commit message using the configured LLM provider. |
| `http_request` | Make an HTTP request to a URL. Supports all HTTP methods with headers, authentication, query parameters, JSON body, form data, and performance metrics. |
| `http_test` | Test an API endpoint with validation. Makes an HTTP request and validates the response against expected values. Currently validates status code, can be extended for more validations. |
| `list_directory` | List the contents of a directory. |
| `mcp_call` | Call an MCP tool by name (prefixed with the MCP namespace). |
| `plan_files` | Plan which files to inspect (evidence-focused). Finds up to `limit` existing files under `root` matching optional substrings in `patterns`. Use this to pick a small set of real files before calling read_file. |
| `read_file` | Read the contents of a file from the filesystem. |
| `refactor_extract_function` | Extract code block into a new function. Extracts selected lines of code into a new function, analyzing variables to determine parameters and return values. |
| `refactor_inline_variable` | Inline a simple variable assignment. Replaces all usages of a variable with its assigned value and removes the assignment statement. |
| `refactor_organize_imports` | Organize and optimize import statements. Sorts imports into groups (stdlib, third-party, local), removes duplicates, and follows PEP 8 conventions. |
| `refactor_rename_symbol` | Rename a symbol (variable, function, class). Safely renames symbols across a file using AST-based analysis to avoid false matches. |
| `rename_symbol` | Safely renames a symbol across all Python files in the specified search path. This tool finds all references to `symbol_name` and replaces them with `new_symbol_name` within a transactional file editing process. Users should review and commit the changes using the `file_editor` tool. |
| `run_tests` | Runs tests using pytest and returns a structured summary of the results. This tool runs pytest on the specified path and captures the output in JSON format, providing a clean summary of test outcomes. |
| `run_workflow` | Runs a pre-defined, multi-step workflow to automate a complex task. |
| `scaffold` | Unified project scaffolding tool. Performs scaffolding operations including creating projects from templates, listing available templates, adding files, and initializing git repositories. Consolidates all scaffolding functionality into a single interface. |
| `security_scan` | Comprehensive security scanning for code analysis. Performs security scans including secret detection, dependency vulnerability checks, and configuration security analysis. Consolidates multiple scan types into a single unified interface. |
| `semantic_code_search` | Semantic code search using the embedding-backed indexer. Builds (or reuses) an embedding index for the codebase and returns the top-k matches with file paths, scores, and line numbers. Reindexes automatically when files change; use force_reindex to rebuild on demand. |
| `upload_files_to_sandbox` | Uploads one or more local files to the code execution sandbox. The files will be placed in the root of the execution environment. |
| `web_fetch` | Fetch and extract content from a URL. Downloads a web page and extracts the main text content, removing scripts, styles, navigation, and other non-content elements. |
| `web_search` | Web search / online lookup using DuckDuckGo (internet search, find online, lookup docs). Purpose: - Find links, docs, references on the public web. - Return titles, URLs, and snippets for relevance checking. - Ideal when the user says "search the web", "find online", "lookup", "docs", "articles". |
| `web_summarize` | Search the web AND summarize results with AI (web search + summarization). Purpose: - Perform a web search, then produce a concise summary with key findings and cited links. - Use when the user asks to "summarize from the web", "give me top X with pros/cons", or needs synthesized web info. - Returns original results plus AI-written summary. |
| `write_file` | Write content to a file, creating it if it doesn't exist. |