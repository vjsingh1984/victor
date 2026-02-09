# Victor CLI Reference - Part 2

**Part 2 of 2:** RAG Operations, Benchmarks, Workflows, Sessions, Verticals, Global Options, and Environment Variables

---

## Navigation

- [Part 1: Core Commands & Config](part-1-core-config.md)
- **[Part 2: Advanced Operations](#)** (Current)
- [**Complete Guide](../cli-reference.md)**

---
## RAG Operations

### victor rag

Retrieval-Augmented Generation operations.

```bash
# Ingest documents
victor rag ingest ./docs --recursive --pattern "*.md"
victor rag ingest https://example.com/docs
victor rag ingest ./README.md --type markdown

# Search and query
victor rag search "authentication"
victor rag search "error handling" --top-k 10
victor rag query "How do I add a provider?" --synthesize

# Management
victor rag list                     # List documents
victor rag stats                    # Show statistics
victor rag delete doc_abc123        # Delete document

# Demos
victor rag demo docs                # Ingest project docs
victor rag demo-sec --preset faang  # SEC filings demo
```text

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `ingest <source>` | Ingest file, directory, or URL |
| `search <query>` | Search knowledge base |
| `query <question>` | Query with optional synthesis |
| `list` | List all documents |
| `stats` | Show knowledge base stats |
| `delete <id>` | Delete a document |
| `demo` | Run documentation demo |
| `demo-sec` | SEC filings demo |

**Ingest Options:**

| Option | Description |
|--------|-------------|
| `--recursive, -r` | Recursively ingest directory |
| `--pattern, -p` | Glob pattern (e.g., "*.md") |
| `--type, -t` | Document type: auto, text, markdown, code, pdf, html |
| `--id` | Custom document ID |

**Query Options:**

| Option | Description |
|--------|-------------|
| `--synthesize, -S` | Use LLM to synthesize answer |
| `--provider, -p` | LLM provider for synthesis |
| `--model, -m` | Model for synthesis |
| `--top-k, -k` | Number of results |

---

## Benchmark Operations

### victor benchmark

Run AI coding benchmarks.

```bash
# List and setup
victor benchmark list               # List available benchmarks
victor benchmark setup swe-bench --max-tasks 10

# Run benchmarks
victor benchmark run swe-bench --max-tasks 10 --profile default
victor benchmark run humaneval --model claude-3-sonnet
victor benchmark run mbpp --output results.json

# Compare and leaderboard
victor benchmark compare --benchmark swe-bench
victor benchmark leaderboard --benchmark swe-bench
victor benchmark capabilities       # Framework comparison
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List available benchmarks |
| `setup <bench>` | Setup benchmark repos |
| `run <bench>` | Run a benchmark |
| `compare` | Compare with other frameworks |
| `leaderboard` | Show benchmark leaderboard |
| `capabilities` | Compare framework capabilities |

**Available Benchmarks:**

| Benchmark | Description |
|-----------|-------------|
| `swe-bench` | Real-world GitHub issue resolution |
| `swe-bench-lite` | Curated SWE-bench subset |
| `humaneval` | Code generation from docstrings |
| `mbpp` | Mostly Basic Python Problems |

**Run Options:**

| Option | Description |
|--------|-------------|
| `--max-tasks, -n` | Maximum tasks to run |
| `--model, -m` | Model to use |
| `--profile, -p` | Victor profile |
| `--output, -o` | Output file (JSON) |
| `--timeout, -t` | Timeout per task (seconds) |
| `--max-turns` | Max conversation turns |
| `--parallel` | Parallel tasks |
| `--resume, -r` | Resume from checkpoint |

---

## Workflow Management

### victor workflow

Manage YAML workflows.

```bash
victor workflow validate path/to/workflow.yaml
victor workflow render workflow.yaml --format ascii
victor workflow list                # List workflows
victor workflow run my_workflow     # Execute workflow
```text

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `validate <path>` | Validate workflow YAML |
| `render <path>` | Render workflow graph |
| `list` | List available workflows |
| `run <name>` | Execute a workflow |

**Render Formats:**

- `ascii` - ASCII art diagram
- `mermaid` - Mermaid diagram
- `json` - JSON structure

---

### victor scheduler

Manage workflow scheduling.

```bash
# Service management
victor scheduler start              # Start in foreground
victor scheduler start --daemon     # Start as background daemon
victor scheduler stop               # Stop daemon
victor scheduler status             # Show status

# Schedule management
victor scheduler list               # List schedules
victor scheduler add daily_check --cron "0 9 * * *" --yaml workflow.yaml
victor scheduler remove schedule_id
victor scheduler history            # Show execution history

# Installation
victor scheduler install            # Generate systemd service
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `start` | Start scheduler service |
| `stop` | Stop scheduler daemon |
| `status` | Show scheduler status |
| `list` | List scheduled workflows |
| `add` | Schedule a workflow |
| `remove` | Remove a schedule |
| `history` | Show execution history |
| `install` | Generate systemd service file |

**Cron Expression Examples:**

| Expression | Description |
|------------|-------------|
| `0 9 * * *` | Daily at 9 AM |
| `0 */2 * * *` | Every 2 hours |
| `0 0 * * MON` | Every Monday at midnight |
| `@hourly` | Every hour |
| `@daily` | Every day at midnight |

---

## Session Management

### victor sessions

Manage conversation sessions.

```bash
victor sessions list                # List sessions
victor sessions list --all          # List all sessions
victor sessions show session_id     # Show session details
victor sessions search "keyword"    # Search sessions
victor sessions delete session_id   # Delete session
victor sessions export              # Export to JSON
victor sessions clear               # Clear all sessions
```text

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List saved sessions |
| `show <id>` | Show session details |
| `search <query>` | Search sessions |
| `delete <id>` | Delete a session |
| `export` | Export sessions to JSON |
| `clear` | Clear all sessions |

**Options:**

| Option | Description |
|--------|-------------|
| `--limit, -n` | Maximum results |
| `--all` | Show all (no limit) |
| `--json` | Output as JSON |

---

## Vertical Management

### victor vertical

Manage vertical packages (plugins).

```bash
# Installation
victor vertical install victor-security
victor vertical install "victor-security>=0.5.0"
victor vertical install git+https://github.com/user/victor-security.git
victor vertical install ./local/path

# Management
victor vertical list                # List all verticals
victor vertical list --source installed
victor vertical search security     # Search verticals
victor vertical info security       # Show details
victor vertical uninstall victor-security

# Creation
victor vertical create security --description "Security analysis"
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `install <pkg>` | Install a vertical |
| `uninstall <name>` | Uninstall a vertical |
| `list` | List verticals |
| `search <query>` | Search verticals |
| `info <name>` | Show vertical info |
| `create <name>` | Create new vertical |

**List Filters:**

| Option | Description |
|--------|-------------|
| `--source` | Filter: all, installed, builtin, available |
| `--category, -c` | Filter by category |
| `--tags, -t` | Filter by tags |
| `--verbose, -v` | Show detailed info |

---

## Global Options

These options are available for most commands:

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version |
| `--log-level` | Set logging level (DEBUG, INFO, WARNING, ERROR) |
| `--debug` | Enable debug mode |
| `--quiet, -q` | Minimal output |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VICTOR_PROFILE` | Default profile name |
| `VICTOR_PROVIDER` | Default provider |
| `VICTOR_MODEL` | Default model |
| `VICTOR_LOG_LEVEL` | Logging level |
| `VICTOR_CONFIG_DIR` | Config directory override |
| `VICTOR_AIRGAPPED` | Enable air-gapped mode |
| `VICTOR_SKIP_ENV_FILE` | Skip .env file loading |

**Provider API Keys:**

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `OPENAI_API_KEY` | OpenAI |
| `GOOGLE_API_KEY` | Google (Gemini) |
| `XAI_API_KEY` | xAI (Grok) |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `GROQ_API_KEY` | Groq |
| `TOGETHER_API_KEY` | Together AI |
| `OPENROUTER_API_KEY` | OpenRouter |

---

## See Also

- [TUI Mode Guide](./tui-mode.md) - Interactive TUI interface
- [Tool Catalog](../reference/tools/catalog.md) - Available tools
- [Provider Reference](../reference/providers/index.md) - Provider details
- [Workflow DSL](../guides/workflow-development/dsl.md) - Workflow syntax

---

**Last Updated:** February 01, 2026
**Reading Time:** 9 minutes
