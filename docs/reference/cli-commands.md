# CLI Command Reference

Complete reference for Victor CLI commands.

## Global Options

```bash
victor [OPTIONS] COMMAND [ARGS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-v` | Show version and exit |
| `--help` | | Show help message |

## Default Behavior

Running `victor` without a subcommand launches the interactive TUI mode.

---

## Core Commands

### `victor chat`

Start an interactive chat session with the AI assistant.

```bash
victor chat [OPTIONS] [PROMPT]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--profile` | `-p` | `default` | Profile name from profiles.yaml |
| `--provider` | | From profile | Override provider (ollama, anthropic, openai, google, etc.) |
| `--model` | `-m` | From profile | Override model identifier |
| `--no-tui` | | `False` | Use CLI mode instead of TUI |
| `--vertical` | | `coding` | Domain vertical (coding, research, devops, dataanalysis) |
| `--mode` | | `build` | Agent mode (build, plan, explore) |
| `--headless` | | `False` | Run without prompts, auto-approve safe actions |
| `--dry-run` | | `False` | Preview changes without applying |
| `--one-shot` | | `False` | Exit after completing single request |
| `--log-events` | | `False` | Enable observability logging to JSONL |
| `--thread-id` | | | Resume from checkpoint thread ID |

**Examples:**

```bash
# Interactive TUI mode with default profile
victor chat

# CLI mode with specific provider
victor chat --no-tui --provider anthropic --model claude-sonnet-4-20250514

# One-shot mode for automation
victor chat --one-shot --headless "Fix the failing test in test_utils.py"

# Resume from checkpoint
victor chat --thread-id abc123
```

### `victor init`

Initialize Victor configuration for a project.

```bash
victor init [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--force` | `-f` | `False` | Overwrite existing configuration |
| `--provider` | `-p` | `ollama` | Default provider to configure |
| `--model` | `-m` | | Default model to configure |
| `--analyze` | | `True` | Analyze codebase and generate context file |
| `--minimal` | | `False` | Create minimal configuration without analysis |

**What it creates:**

- `~/.victor/profiles.yaml` - Global profile configuration
- `.victor/init.md` - Project-specific context file

### `victor serve`

Start Victor as an API server.

```bash
victor serve [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | | `127.0.0.1` | Host to bind to |
| `--port` | | `8000` | Port to listen on |
| `--reload` | | `False` | Enable auto-reload for development |
| `--workers` | | `1` | Number of worker processes |
| `--api-key` | | | Require API key for requests |

---

## Provider & Model Management

### `victor providers`

List available LLM providers.

```bash
victor providers list
```

Displays all supported providers with their status (configured/not configured).

### `victor models`

List available models for a provider.

```bash
victor models list [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | `ollama` | Provider to list models from |
| `--endpoint` | `-e` | | Custom endpoint URL |

**Supported providers:** ollama, lmstudio, llamacpp, vllm, anthropic, openai, google, cerebras, groqcloud

**Examples:**

```bash
victor models list -p ollama
victor models list -p lmstudio -e http://192.168.1.20:1234
victor models list -p anthropic
```

### `victor test-provider`

Test provider connectivity and capabilities.

```bash
victor test-provider [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--provider` | `-p` | Provider to test |
| `--model` | `-m` | Model to test |
| `--verbose` | `-v` | Show detailed output |

---

## Profile Management

### `victor profiles`

Manage model profiles.

```bash
victor profiles COMMAND [OPTIONS]
```

#### `victor profiles list`

List all configured profiles.

#### `victor profiles create`

Create a new profile.

```bash
victor profiles create NAME [OPTIONS]
```

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--provider` | `-p` | Yes | Provider name |
| `--model` | `-m` | Yes | Model identifier |
| `--temperature` | `-t` | No | Temperature (0.0-2.0), default 0.7 |
| `--max-tokens` | | No | Maximum output tokens, default 4096 |
| `--description` | `-d` | No | Profile description |

#### `victor profiles edit`

Edit an existing profile.

```bash
victor profiles edit NAME [OPTIONS]
```

#### `victor profiles delete`

Delete a profile.

```bash
victor profiles delete NAME [--force]
```

#### `victor profiles show`

Show details of a specific profile.

```bash
victor profiles show NAME
```

#### `victor profiles set-default`

Set a profile as the default.

```bash
victor profiles set-default NAME
```

---

## API Key Management

### `victor keys`

Manage API keys for cloud providers and external services.

```bash
victor keys [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--setup` | `-s` | Create API keys template file |
| `--list` | `-l` | List configured providers (default) |
| `--set PROVIDER` | | Set API key for a provider |
| `--set-service SERVICE` | | Set API key for a service (finnhub, fred, etc.) |
| `--keyring` | `-k` | Store key in system keyring (secure) |
| `--migrate` | | Migrate keys from file to keyring |
| `--delete-keyring PROVIDER` | | Delete provider key from keyring |
| `--delete-service-keyring SERVICE` | | Delete service key from keyring |
| `--services` | | List configured services |

**Examples:**

```bash
# List configured API keys
victor keys

# Set Anthropic API key in system keyring (recommended)
victor keys --set anthropic --keyring

# Create template file for manual editing
victor keys --setup

# Migrate all keys from file to secure keyring
victor keys --migrate

# List configured external services
victor keys --services
```

---

## Codebase Tools

### `victor index`

Manage codebase index for semantic search.

```bash
victor index COMMAND [OPTIONS]
```

#### `victor index build`

Build or rebuild the codebase index.

```bash
victor index build [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--path` | `-p` | Directory to index (default: current) |
| `--force` | `-f` | Force full rebuild |
| `--verbose` | `-v` | Show detailed output |

#### `victor index status`

Show index status and statistics.

#### `victor index clear`

Clear the codebase index.

### `victor tools`

List available tools and their capabilities.

```bash
victor tools list [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--category` | `-c` | Filter by category |
| `--cost-tier` | | Filter by cost tier (free, low, medium, high) |
| `--verbose` | `-v` | Show detailed descriptions |

---

## Workflow Management

### `victor workflow`

Manage and execute workflows.

```bash
victor workflow COMMAND [OPTIONS]
```

#### `victor workflow validate`

Validate a workflow YAML file.

```bash
victor workflow validate PATH
```

#### `victor workflow list`

List available workflows.

#### `victor workflow run`

Execute a workflow.

```bash
victor workflow run WORKFLOW_NAME [OPTIONS]
```

### `victor scheduler`

Manage the workflow scheduler daemon.

```bash
victor scheduler COMMAND [OPTIONS]
```

#### `victor scheduler start`

Start the scheduler daemon.

```bash
victor scheduler start [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--daemonize` | `-d` | Run as background daemon |
| `--pid-file` | | PID file path |

#### `victor scheduler stop`

Stop the scheduler daemon.

#### `victor scheduler status`

Show scheduler status.

---

## Vertical Management

### `victor vertical`

Manage vertical packages (domain-specific extensions).

```bash
victor vertical COMMAND [OPTIONS]
```

#### `victor vertical list`

List available verticals.

```bash
victor vertical list [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--source` | `-s` | Filter: all, installed, builtin, available |
| `--category` | `-c` | Filter by category |
| `--tags` | `-t` | Filter by tags (comma-separated) |
| `--verbose` | `-v` | Show detailed information |

#### `victor vertical install`

Install a vertical package.

```bash
victor vertical install PACKAGE [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--no-validate` | | Skip validation checks |
| `--dry-run` | `-n` | Show what would be installed |
| `--verbose` | `-v` | Show detailed output |

**Examples:**

```bash
victor vertical install victor-security
victor vertical install "git+https://github.com/user/victor-security.git"
victor vertical install ./path/to/package
```

#### `victor vertical uninstall`

Uninstall a vertical package.

```bash
victor vertical uninstall NAME [OPTIONS]
```

#### `victor vertical info`

Show detailed information about a vertical.

```bash
victor vertical info NAME
```

#### `victor vertical search`

Search for vertical packages.

```bash
victor vertical search QUERY
```

#### `victor vertical create`

Generate a new vertical structure from templates.

```bash
victor vertical create NAME [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--description` | `-d` | Description of the vertical's purpose |
| `--service-provider` | `-s` | Include service_provider.py for DI registration |
| `--force` | `-f` | Overwrite existing files |
| `--dry-run` | `-n` | Show what would be created |

---

## Session Management

### `victor sessions`

Manage conversation sessions.

```bash
victor sessions COMMAND [OPTIONS]
```

#### `victor sessions list`

List saved sessions.

#### `victor sessions show`

Show session details and conversation history.

```bash
victor sessions show SESSION_ID
```

#### `victor sessions delete`

Delete a session.

```bash
victor sessions delete SESSION_ID [--force]
```

#### `victor sessions export`

Export session to file.

```bash
victor sessions export SESSION_ID [--format json|markdown]
```

---

## Benchmarking

### `victor benchmark`

Run benchmarks against standard datasets.

```bash
victor benchmark run BENCHMARK_NAME [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--max-tasks` | | Maximum tasks to run |
| `--profile` | | Profile to use |
| `--output` | `-o` | Output directory for results |

**Available benchmarks:** swe-bench, humaneval

**Examples:**

```bash
victor benchmark run swe-bench --max-tasks 10 --profile deepseek
victor benchmark run humaneval --max-tasks 5 --profile default
```

---

## RAG (Retrieval-Augmented Generation)

### `victor rag`

RAG operations for document ingestion and search.

```bash
victor rag COMMAND [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--log-level` | | Set logging level (DEBUG, INFO, WARN, ERROR) |

#### `victor rag ingest`

Ingest documents into the RAG system.

#### `victor rag search`

Search ingested documents.

#### `victor rag status`

Show RAG system status.

---

## MCP (Model Context Protocol)

### `victor mcp`

Manage MCP server connections.

```bash
victor mcp COMMAND [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--log-level` | | Set logging level |

#### `victor mcp list`

List configured MCP servers.

#### `victor mcp add`

Add an MCP server configuration.

#### `victor mcp remove`

Remove an MCP server configuration.

#### `victor mcp test`

Test MCP server connectivity.

---

## Utility Commands

### `victor config`

View and manage configuration.

```bash
victor config COMMAND
```

#### `victor config show`

Show current configuration.

#### `victor config path`

Show configuration file paths.

### `victor docs`

Open documentation.

```bash
victor docs [--local]
```

### `victor examples`

Show usage examples.

```bash
victor examples [TOPIC]
```

### `victor capabilities`

Show system capabilities and installed features.

```bash
victor capabilities
```

### `victor embeddings`

Manage embedding models and caches.

```bash
victor embeddings COMMAND [OPTIONS]
```

#### `victor embeddings status`

Show embedding system status.

#### `victor embeddings clear`

Clear embedding cache.

### `victor security`

Security scanning and auditing.

```bash
victor security COMMAND [OPTIONS]
```

#### `victor security scan`

Run security scan on codebase.

#### `victor security audit`

Generate security audit report.

### `victor dashboard`

Launch the observability dashboard.

```bash
victor dashboard [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--port` | `-p` | Dashboard port (default: 8501) |
| `--host` | | Dashboard host |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 130 | Interrupted (Ctrl+C) |

---

## Environment Variable Overrides

CLI options can be overridden via environment variables. See [Environment Variables Reference](environment-variables.md) for the complete list.

Common overrides:

| Variable | CLI Equivalent |
|----------|---------------|
| `VICTOR_PROFILE` | `--profile` |
| `VICTOR_PROVIDER` | `--provider` |
| `VICTOR_MODEL` | `--model` |
| `VICTOR_HEADLESS_MODE` | `--headless` |
| `VICTOR_DRY_RUN_MODE` | `--dry-run` |
