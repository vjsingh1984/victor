# TUI Mode Guide

Victor's Terminal User Interface (TUI) provides a modern, interactive chat experience with rich formatting, keyboard navigation, and real-time streaming.

## Overview

The TUI is built using Textual and provides:

- Input box at the bottom (like Claude Code, Gemini CLI)
- Scrollable conversation history in the middle
- Status bar showing provider/model information
- Real-time streaming with thinking and tool call visualization
- 40+ slash commands for session management

## Layout

```
+-------------------------------------------------------+
| Victor | anthropic / claude-sonnet-4-5                | <- StatusBar
+-------------------------------------------------------+
|                                                       |
|  You                                                  | <- ConversationLog
|  Hello, can you help me with my code?                 |    (scrollable)
|                                                       |
|  Victor                                               |
|  Of course! I'd be happy to help. What would you      |
|  like to work on?                                     |
|                                                       |
+-------------------------------------------------------+
| > Type your message...                                | <- InputWidget
|   Enter send | Shift+Enter newline                    |
+-------------------------------------------------------+
| Ctrl+C Exit | Ctrl+L Clear | Ctrl+S Save              | <- Footer
+-------------------------------------------------------+
```

## Starting TUI Mode

```bash
# Default TUI mode
victor

# TUI with specific provider
victor --provider anthropic

# TUI with saved profile
victor --profile local

# TUI with specific model
victor --provider ollama --model qwen2.5-coder:7b
```

To use CLI mode instead of TUI:

```bash
victor chat --no-tui
```

---

## Keyboard Shortcuts

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Exit Victor |
| `Ctrl+L` | Clear conversation |
| `Ctrl+T` | Toggle thinking panel |
| `Ctrl+S` | Save current session |
| `Ctrl+/` | Show help overlay |
| `Escape` | Focus input widget |

### Input Navigation

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift+Enter` | Add newline |
| `Up Arrow` | Previous message in history |
| `Down Arrow` | Next message in history |

### Conversation Scrolling

| Shortcut | Action |
|----------|--------|
| `Ctrl+Up` | Scroll conversation up |
| `Ctrl+Down` | Scroll conversation down |
| `Ctrl+Home` | Scroll to top |
| `Ctrl+End` | Scroll to bottom |

---

## Slash Commands

Slash commands provide quick access to Victor's features without leaving the TUI. Type `/` followed by the command name.

### Quick Reference

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/clear` | Clear conversation |
| `/exit` | Exit Victor |
| `/model` | Switch model |
| `/provider` | Switch provider |
| `/mode` | Switch agent mode |
| `/save` | Save session |
| `/resume` | Resume session |
| `/status` | Show session status |

### System Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help [cmd]` | `/?, /commands` | Show help for commands |
| `/config` | `/settings` | Show current configuration |
| `/status` | `/info` | Show session status |
| `/clear` | `/reset` | Clear conversation history |
| `/exit` | `/quit, /bye` | Exit Victor |
| `/theme [dark\|light]` | `/dark, /light` | Toggle theme |
| `/bug` | `/issue, /feedback` | Report an issue |
| `/approvals [mode]` | `/safety` | Configure approval mode |

### Session Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/save [--new] [name]` | | Save conversation to session |
| `/load <id>` | | Load a saved session |
| `/sessions [limit]` | `/history` | List saved sessions |
| `/resume [id]` | | Resume a session |
| `/compact [--smart]` | `/summarize` | Compress conversation history |

**Examples:**

```
/save                       # Update current session
/save --new                 # Create new session
/save My Important Work     # Save with title
/sessions 20                # List last 20 sessions
/resume abc123              # Resume specific session
/compact --smart            # AI-powered summarization
```

### Model Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/model [name]` | `/models` | List models or switch model |
| `/profile [name]` | `/profiles` | Show/switch profile |
| `/provider [name]` | `/providers` | Show/switch provider |

**Examples:**

```
/model                      # List available models
/model qwen2.5:7b           # Switch to model
/profile                    # Show profiles with RL rankings
/profile local              # Switch to profile
/provider anthropic         # Switch provider
/provider deepseek:deepseek-coder  # Provider with model
```

### Mode Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/mode [build\|plan\|explore]` | `/m` | Show/switch agent mode |
| `/build` | | Switch to build mode |
| `/explore` | | Switch to explore mode |
| `/plan [task]` | | Enter planning mode |
| `/plan save [name]` | | Save current plan |
| `/plan load <id>` | | Load a saved plan |
| `/plan list` | | List saved plans |
| `/plan show` | | Show current plan |

**Agent Modes:**

| Mode | Description |
|------|-------------|
| `build` | Implementation mode (default) - full edits |
| `plan` | Planning mode - sandbox edits, 2.5x exploration |
| `explore` | Navigation mode - no edits, 3x exploration |

**Examples:**

```
/mode                       # Show current mode
/mode plan                  # Switch to planning mode
/plan Add authentication    # Start planning a task
/plan save auth-plan        # Save the plan
/plan list                  # List saved plans
```

### Tool Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/tools [pattern]` | | List available tools |
| `/context` | `/ctx, /memory` | Show project context |
| `/lmstudio` | `/lm` | Probe LMStudio endpoints |
| `/mcp` | `/servers` | Show MCP server status |
| `/search [on\|off]` | `/web` | Toggle web search |
| `/review [path]` | | Request code review |

**Examples:**

```
/tools                      # List all tools
/tools file                 # Search tools by pattern
/context                    # Show .victor/init.md
/search off                 # Disable web search
/review src/api.py          # Review specific file
```

### Navigation Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/directory [path]` | `/dir, /cd, /pwd` | Show/change directory |
| `/changes [show\|revert\|stash]` | `/diff, /rollback` | View/manage file changes |
| `/undo` | | Undo last file change |
| `/redo` | | Redo undone change |
| `/filehistory [limit]` | `/timeline` | Show file change history |
| `/snapshots [cmd]` | `/snap` | Manage workspace snapshots |
| `/commit [message]` | `/ci` | Commit with AI message |
| `/copy` | | Copy last response to clipboard |

**Snapshot Subcommands:**

```
/snapshots list             # List snapshots
/snapshots create [desc]    # Create snapshot
/snapshots restore <id>     # Restore snapshot
/snapshots diff <id>        # Show diff from snapshot
/snapshots clear            # Clear all snapshots
```

### Metrics Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/cost` | `/spending` | Show token costs |
| `/metrics` | | Show session metrics |
| `/mlstats` | | Show ML/RL statistics |

### Codebase Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/reindex` | `/index` | Rebuild codebase index |
| `/init` | | Initialize Victor in directory |

### Checkpoint Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/checkpoint save [name]` | `/cp` | Save checkpoint |
| `/checkpoint list` | | List checkpoints |
| `/checkpoint restore <id>` | | Restore checkpoint |
| `/checkpoint diff <id>` | | Show checkpoint diff |
| `/checkpoint timeline` | | Show checkpoint timeline |

### Entity Commands (RAG)

| Command | Aliases | Description |
|---------|---------|-------------|
| `/entities list` | `/ent` | List indexed entities |
| `/entities search <q>` | | Search entities |
| `/entities show <name>` | | Show entity details |
| `/entities related <name>` | | Show related entities |
| `/entities stats` | | Show entity statistics |
| `/entities clear` | | Clear entity cache |

### Debug Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/debug break [cond]` | | Set breakpoint |
| `/debug clear [id]` | | Clear breakpoint(s) |
| `/debug list` | | List breakpoints |
| `/debug enable <id>` | | Enable breakpoint |
| `/debug disable <id>` | | Disable breakpoint |
| `/debug state` | | Show debug state |
| `/debug continue` | | Continue execution |
| `/debug step` | | Step to next |

---

## Visual Features

### Streaming Display

When streaming is enabled (default), responses appear token by token with:

- Real-time content updates
- Thinking panel for reasoning models
- Tool call visualization with status indicators

### Thinking Panel

Models with thinking/reasoning capabilities show their thought process:

- Toggle with `Ctrl+T`
- Shows chain-of-thought reasoning
- Collapses when response completes

### Tool Call Display

Tool executions appear with:

- Tool name and arguments
- Status indicator (pending, success, error)
- Execution time

### Message Types

| Type | Style |
|------|-------|
| User | Highlighted panel with cyan header |
| Assistant | Standard panel with blue header |
| System | Italic text with muted styling |
| Error | Red border with error styling |

---

## Themes

Victor supports dark and light themes:

```
/theme dark                 # Switch to dark theme
/theme light                # Switch to light theme
/theme                      # Toggle theme
```

---

## Session Persistence

### Automatic Saving

Sessions are automatically saved to SQLite when:

- You use `/save` command
- Session has significant content

### Session Storage

Sessions are stored in `~/.victor/sessions.db` with:

- Full conversation history
- Conversation state machine state
- Provider and model information
- Timestamps

### Resuming Sessions

```bash
# From CLI
victor chat --resume session_id

# From TUI
/resume session_id
```

---

## Input History

The TUI maintains input history:

- Navigate with Up/Down arrows
- History persists across sessions
- Press `Escape` to focus input widget

---

## Configuration

### TUI-Specific Settings

Configure in `~/.victor/config.yaml`:

```yaml
# TUI settings
theme: dark
streaming: true
show_thinking: true
show_tool_calls: true
```

### Project Context

The TUI loads project context from:

1. `.victor/init.md`
2. `.victor.md`
3. `CLAUDE.md`

View with `/context` command.

---

## Tips and Best Practices

### Efficient Workflow

1. **Use profiles** - Create profiles for common setups
2. **Save often** - Use `/save` to checkpoint important work
3. **Use planning mode** - Start complex tasks with `/mode plan`
4. **Compact history** - Use `/compact --smart` to reduce context size

### Keyboard-First Navigation

1. `Ctrl+L` to clear and start fresh
2. `Ctrl+S` to save progress
3. Arrow keys for input history
4. `Ctrl+/` for quick help

### Managing Context

1. Use `/compact` when context gets too large
2. Check `/status` for message count
3. Use `/clear` to reset for new tasks
4. Save sessions before clearing

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| TUI not displaying | Ensure terminal supports ANSI |
| Input not working | Press `Escape` to focus input |
| Slow streaming | Check network connection |
| Theme not applying | Restart TUI |

### Debug Mode

For debugging TUI issues:

```bash
victor --log-level DEBUG
```

Or within TUI:

```
/debug state
```

---

## See Also

- [CLI Reference](./cli-reference.md) - Full CLI command reference
- [Session Management](./session-management.md) - Detailed session guide
- [Tool Catalog](../reference/tools/catalog.md) - Available tools
