# Victor AI - VS Code Extension

AI-powered coding assistant with multi-provider support, semantic code search, and 46 enterprise tools.

[![GitHub](https://img.shields.io/badge/GitHub-vjsingh1984%2Fvictor-blue)](https://github.com/vjsingh1984/victor)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vjsingh1984/victor/blob/main/LICENSE)

## Features

### Automatic Server Management
- **Auto-start**: Server starts automatically when VS Code opens
- **Health monitoring**: Extension monitors server health
- **Status indicator**: See server status in status bar
- **Easy control**: Start/stop/restart via command palette

### Chat Panel
- Interactive chat with AI assistant
- Code explanation, refactoring, and generation
- Tool call visualization
- Streaming responses

### Inline Completions
- AI-powered code completions
- Context-aware suggestions
- Multi-language support

### Semantic Code Search
- Natural language code search
- Find code by describing what it does
- Jump to results instantly

### Context Management
- `@file:path` - Include file contents
- `@folder:path` - Include folder structure
- `@problems` - Include workspace diagnostics
- `@git` - Include git status

### Agent Modes
- **Build** - Full implementation mode with all tools
- **Plan** - Read-only analysis mode
- **Explore** - Code navigation and understanding mode

### Multi-Provider Support
- Anthropic (Claude Opus 4.5, Sonnet 4)
- OpenAI (GPT-4 Turbo, GPT-4o)
- Google (Gemini 2.0 Flash)
- Ollama (Qwen 2.5, Llama 3.1)
- LMStudio (Local models)
- vLLM (Self-hosted)
- xAI (Grok)

## Installation

### Prerequisites

1. **Python 3.10+** installed
2. **Victor** installed:
   ```bash
   pip install -e ".[dev]"
   victor init
   ```

### Install Extension

1. Install this extension from VS Code Marketplace
2. The server will start automatically on first activation

Or manually from source:
```bash
cd vscode-victor
npm install
npm run compile
```

## Commands

| Command | Keybinding | Description |
|---------|------------|-------------|
| Victor: Open Chat | `Ctrl+Shift+V` | Open chat panel |
| Victor: Explain | `Ctrl+Shift+E` | Explain selected code |
| Victor: Search | `Ctrl+Shift+S` | Semantic code search |
| Victor: Refactor | - | Refactor selected code |
| Victor: Fix | - | Fix issues in selection |
| Victor: Test | - | Generate tests |
| Victor: Document | - | Add documentation |
| Victor: Switch Model | - | Change AI model |
| Victor: Switch Mode | - | Change agent mode |
| Victor: Undo | - | Undo last change |
| Victor: Redo | - | Redo last change |
| Victor: Start Server | - | Manually start server |
| Victor: Stop Server | - | Stop server |
| Victor: Restart Server | - | Restart server |
| Victor: Server Status | - | View server status and options |
| Victor: Show Server Logs | - | View server output |

## Configuration

Open Settings (`Ctrl+,`) and search for "Victor":

```json
{
  "victor.serverPort": 8765,
  "victor.autoStart": true,
  "victor.provider": "anthropic",
  "victor.model": "claude-sonnet-4-20250514",
  "victor.mode": "build",
  "victor.showInlineCompletions": true,
  "victor.pythonPath": "",
  "victor.victorPath": "",
  "victor.semanticSearch.enabled": true,
  "victor.semanticSearch.maxResults": 10
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `serverPort` | Port for Victor server | `8765` |
| `autoStart` | Auto-start server on activation | `true` |
| `provider` | Default LLM provider | `anthropic` |
| `model` | Default model | `claude-sonnet-4-20250514` |
| `mode` | Agent mode (build/plan/explore) | `build` |
| `showInlineCompletions` | Enable inline suggestions | `true` |
| `pythonPath` | Custom Python path | auto-detect |
| `victorPath` | Path to bundled binary | none |

## Server Management

### Automatic Mode (Default)

When `victor.autoStart` is `true`:
1. Extension checks if server is already running
2. If not, spawns `victor serve` automatically
3. Waits for server to be ready
4. Shows status in status bar

### Manual Mode

Set `victor.autoStart` to `false`, then:
- Use Command Palette → "Victor: Start Server"
- Or run in terminal: `victor serve --port 8765`

### Status Bar

The server status bar shows:
- 🟢 **Running** - Server is healthy
- ⚪ **Stopped** - Server not running
- 🔄 **Starting** - Server is starting up
- 🔴 **Error** - Server has issues

Click to see options (start/stop/restart/logs).

## Troubleshooting

### Server won't start automatically

1. Verify Victor is installed:
   ```bash
   victor --version
   ```

2. Check Python is available:
   ```bash
   which python3
   ```

3. View server logs:
   - Command Palette → "Victor: Show Server Logs"

4. Try manual start:
   ```bash
   victor serve --port 8765 --log-level DEBUG
   ```

### Connection errors

1. Check server status in status bar
2. Verify port isn't in use: `lsof -i :8765`
3. Restart server: Command Palette → "Victor: Restart Server"

### Model errors

1. **Cloud providers**: Set API keys in `~/.victor/profiles.yaml`
2. **Ollama**: Ensure Ollama is running and model is pulled:
   ```bash
   ollama serve
   ollama pull qwen2.5-coder:7b
   ```

## Development

```bash
cd vscode-victor
npm install
npm run compile
```

Press F5 to launch the extension in debug mode.

## License

Apache 2.0
