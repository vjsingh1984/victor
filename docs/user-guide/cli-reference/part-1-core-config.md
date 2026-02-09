# Victor CLI Reference - Part 1

**Part 1 of 2:** Quick Reference, Core Commands, Modes, API Keys, Providers, Models, Tools, and Configuration

---

## Navigation

- **[Part 1: Core Commands & Config](#)** (Current)
- [Part 2: Advanced Operations](part-2-advanced-operations.md)
- [**Complete Guide](../cli-reference.md)**

---
# CLI Command Reference

Complete reference for all Victor CLI commands.

## Quick Reference

| Command | Description |
|---------|-------------|
| `victor` | Start TUI mode (interactive) |
| `victor chat` | Start CLI chat mode |
| `victor init` | Initialize Victor in current directory |
| `victor serve` | Start HTTP API server |
| `victor mcp` | Run as MCP server |
| `victor keys` | Manage API keys |
| `victor providers list` | List available providers |
| `victor profiles` | Manage provider profiles |
| `victor tools` | Manage and list tools |
| `victor models` | Manage Ollama models |
| `victor config` | View/edit configuration |
| `victor rag` | RAG operations |
| `victor benchmark` | Run benchmarks |
| `victor workflow` | Workflow management |
| `victor scheduler` | Workflow scheduling |
| `victor sessions` | Session management |
| `victor vertical` | Vertical package management |

---

## Core Commands

### victor (default)

Start Victor in TUI (Terminal User Interface) mode.

```bash
victor                              # TUI mode with default profile
victor --profile local              # TUI with specific profile
victor --provider anthropic         # TUI with specific provider
```

**Options:**

| Option | Description |
|--------|-------------|
| `--profile, -p` | Use a saved profile |
| `--provider` | Override provider |
| `--model, -m` | Override model |
| `--airgapped` | Enable air-gapped mode (local only) |
| `--version` | Show version |
| `--help` | Show help |

---

### victor chat

Start an interactive chat session in CLI mode.

```bash
victor chat                         # Default provider/model
victor chat --no-tui                # CLI mode (non-interactive)
victor chat --provider anthropic --model claude-sonnet-4-5
victor chat --stream                # Enable streaming (default)
victor chat --mode plan             # Start in planning mode
```

**Options:**

| Option | Description |
|--------|-------------|
| `--profile, -p` | Use a saved profile |
| `--provider` | LLM provider (anthropic, openai, ollama, etc.) |
| `--model, -m` | Model name |
| `--no-tui` | Disable TUI, use CLI mode |
| `--stream/--no-stream` | Enable/disable streaming |
| `--mode` | Agent mode: build, plan, explore |
| `--tool-budget` | Override tool call budget |
| `--resume, -r` | Resume session by ID |
| `--log-level` | Set logging level |

**Examples:**

```bash
# Start with Claude
victor chat --provider anthropic

# Local model with Ollama
victor chat --provider ollama --model qwen2.5-coder:7b

# Planning mode for research
victor chat --mode plan

# Resume a previous session
victor chat --resume myproj-9Kx7Z2
```

---

## Modes

Victor supports three execution modes that control whether edits are allowed:

| Mode | Description | File Edits |
|------|-------------|-----------|
| **build** | Full implementation mode | Yes |
| **plan** | Analysis and planning | No |
| **explore** | Understanding only | No |

Use `--mode` with `victor chat`:

```bash
victor chat --mode plan "Analyze this module"
```

---

### victor init

Initialize Victor in the current directory. Creates `.victor/` directory with configuration files.

```bash
victor init                         # Interactive setup
victor init --force                 # Overwrite existing config
victor init --provider anthropic    # Pre-configure provider
victor init --model claude-3-5-sonnet
victor init --index                 # Build codebase index
```

**Options:**

| Option | Description |
|--------|-------------|
| `--force, -f` | Overwrite existing configuration |
| `--provider` | Pre-configure default provider |
| `--model` | Pre-configure default model |
| `--index/--no-index` | Build/skip codebase index |
| `--skip-git` | Skip git repository checks |
| `--quiet, -q` | Minimal output |

**Created Files:**

- `.victor/` - Configuration directory
- `.victor/init.md` - Project context file
- `.victor/config.yaml` - Local configuration

---

### victor serve

Start Victor as an HTTP API server.

```bash
victor serve                        # Default port 8765
victor serve --port 8080            # Custom port
victor serve --host 0.0.0.0         # Listen on all interfaces
victor serve --reload               # Auto-reload on changes
```

**Options:**

| Option | Description |
|--------|-------------|
| `--port` | Port number (default: 8765) |
| `--host` | Host address (default: 127.0.0.1) |
| `--reload` | Enable auto-reload for development |
| `--workers` | Number of worker processes |
| `--log-level` | Logging level |

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message |
| `/capabilities` | GET | List available tools |
| `/health` | GET | Health check |
| `/providers` | GET | List providers |

---

### victor mcp

Run Victor as an MCP (Model Context Protocol) server.

```bash
victor mcp                          # Run in stdio mode
victor mcp --stdio                  # Explicit stdio mode
victor mcp --log-level DEBUG        # Verbose logging
```

**Options:**

| Option | Description |
|--------|-------------|
| `--stdio/--no-stdio` | Run in stdio mode (default: true) |
| `--log-level, -l` | Set logging level |

**Usage with MCP Clients:**

Add to Claude Desktop `config.json`:
```json
{
  "mcpServers": {
    "victor": {
      "command": "victor",
      "args": ["mcp"]
    }
  }
}
```

---

## API Key Management

### victor keys

Manage API keys for cloud providers and services.

```bash
victor keys                         # List configured providers
victor keys --setup                 # Create template file
victor keys --set anthropic         # Set key (file storage)
victor keys --set anthropic --keyring  # Set key (secure keyring)
victor keys --migrate               # Migrate file to keyring
victor keys --services              # List external services
victor keys --set-service finnhub --keyring
```

**Options:**

| Option | Description |
|--------|-------------|
| `--setup, -s` | Create API keys template file |
| `--list, -l` | List configured providers |
| `--set <provider>` | Set API key for provider |
| `--set-service <svc>` | Set key for external service |
| `--keyring, -k` | Store in system keyring |
| `--migrate` | Migrate file keys to keyring |
| `--delete-keyring <prov>` | Remove provider from keyring |
| `--services` | List configured services |

**Supported Providers:**

- `anthropic` - Claude models
- `openai` - GPT models
- `google` - Gemini models
- `xai` - Grok models
- `deepseek` - DeepSeek models
- `mistral` - Mistral models
- `groqcloud` - Groq LPU
- `together` - Together AI
- `openrouter` - OpenRouter gateway
- `fireworks` - Fireworks AI
- And more...

**External Services:**

- `finnhub` - Stock data
- `fred` - Federal Reserve data
- `alphavantage` - Market data
- `newsapi` - News aggregation

---

## Provider Management

### victor providers list

List all available LLM providers.

```bash
victor providers list
```

Displays providers with status, features, and aliases.

---

### victor profiles

Manage saved provider/model profiles.

```bash
victor profiles                     # List profiles
victor profiles list                # Same as above
victor profiles show default        # Show profile details
victor profiles create fast --provider groqcloud --model llama3-70b
victor profiles delete old-profile
victor profiles set-default local
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List all profiles |
| `show <name>` | Show profile details |
| `create <name>` | Create new profile |
| `delete <name>` | Delete profile |
| `set-default <name>` | Set default profile |

**Profile Configuration:**

Profiles are stored in `~/.victor/profiles.yaml`:

```yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5
  local:
    provider: ollama
    model: qwen2.5-coder:7b
  fast:
    provider: groqcloud
    model: llama-3.1-70b-versatile
```

---

## Model Management

### victor models

Manage Ollama models.

```bash
victor models                       # List local models
victor models list                  # Same as above
victor models pull qwen2.5-coder:7b # Download model
victor models remove old-model      # Remove model
victor models info qwen2.5-coder:7b # Show model details
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List downloaded models |
| `pull <model>` | Download a model |
| `remove <model>` | Delete a model |
| `info <model>` | Show model information |

---

## Tool Management

### victor tools

Manage and inspect available tools.

```bash
victor tools                        # List all tools
victor tools list                   # Same as above
victor tools list --enabled         # Only enabled tools
victor tools info read_file         # Tool details
victor tools enable tool-name       # Enable a tool
victor tools disable tool-name      # Disable a tool
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List available tools |
| `info <tool>` | Show tool details |
| `enable <tool>` | Enable a disabled tool |
| `disable <tool>` | Disable a tool |

---

## Configuration

### victor config

View and edit configuration.

```bash
victor config                       # Show all config
victor config show                  # Same as above
victor config get default_provider  # Get specific value
victor config set default_provider anthropic
victor config edit                  # Open in editor
victor config path                  # Show config file path
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `show` | Display current configuration |
| `get <key>` | Get a config value |
| `set <key> <value>` | Set a config value |
| `edit` | Open config in editor |
| `path` | Show config file path |
| `reset` | Reset to defaults |

**Configuration Options:**

| Key | Description |
|-----|-------------|
| `default_provider` | Default LLM provider |
| `default_model` | Default model |
| `tool_call_budget` | Max tool calls per turn |
| `airgapped_mode` | Enable air-gapped mode |
| `use_semantic_tool_selection` | Enable semantic tool selection |
| `unified_embedding_model` | Embedding model for RAG |

---

