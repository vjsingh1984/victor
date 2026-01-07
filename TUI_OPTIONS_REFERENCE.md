# Victor TUI - Complete Options and Features Reference

## 📋 Table of Contents

1. [Overview](#overview)
2. [Keyboard Shortcuts](#keyboard-shortcuts)
3. [Slash Commands](#slash-commands)
4. [TUI Components](#tui-components)
5. [Usage Examples](#usage-examples)
6. [Configuration](#configuration)
7. [Advanced Features](#advanced-features)

---

## Overview

Victor's Terminal User Interface (TUI) provides a modern, feature-rich chat experience with:

- **Real-time streaming responses** from AI models
- **Interactive tool execution** with live status updates
- **Multi-line input** support for complex prompts
- **Conversation persistence** with session management
- **43+ slash commands** for various operations
- **Keyboard-first design** for power users

---

## Keyboard Shortcuts

### Navigation Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+C` | Exit | Quit Victor TUI |
| `Ctrl+L` | Clear | Clear conversation history |
| `Escape` | Focus Input | Return focus to input field |
| `Ctrl+↑` | Scroll Up | Scroll conversation up |
| `Ctrl+↓` | Scroll Down | Scroll conversation down |
| `Ctrl+Home` | Scroll Top | Jump to top of conversation |
| `Ctrl+End` | Scroll Bottom | Jump to bottom of conversation |

### Feature Toggles

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+T` | Toggle Thinking | Show/hide thinking panel |
| `Ctrl+S` | Save Session | Save current conversation |
| `Ctrl+/` | Help | Show all keyboard shortcuts |

### Input Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Enter` | Send Message | Send current input |
| `Shift+Enter` | Newline | Add line break (multi-line input) |
| `↑` | Previous | Navigate to previous input in history |
| `↓` | Next | Navigate to next input in history |

---

## Slash Commands

### General Commands

#### `/help`
Show all available slash commands

**Usage:** `/help`

**Aliases:** None

**Example:**
```
> /help
Displays all commands with descriptions
```

#### `/exit`
Exit the TUI

**Usage:** `/exit`

**Aliases:** None

**Example:**
```
> /exit
Exits Victor TUI
```

#### `/clear`
Clear conversation history

**Usage:** `/clear`

**Aliases:** None

**Example:**
```
> /clear
Clears all messages from current conversation
```

---

### Model & Provider Commands

#### `/model`
List available models or switch model

**Usage:** `/model [model_name]`

**Aliases:** `models`

**Examples:**
```
> /model
Lists all available models

> /model claude-3-opus
Switch to Claude 3 Opus

> /model llama3:70b
Switch to Llama 3 70B
```

#### `/provider`
Show current provider info or switch provider

**Usage:** `/provider [provider_name]`

**Aliases:** `providers`

**Examples:**
```
> /provider
Shows current provider and model

> /provider ollama
Switch to Ollama provider

> /provider anthropic
Switch to Anthropic provider
```

#### `/profile`
Show or switch profile

**Usage:** `/profile [profile_name]`

**Aliases:** `profiles`

**Examples:**
```
> /profile
Lists all available profiles

> /profile fast
Switch to 'fast' profile (local model)

> /profile smart
Switch to 'smart' profile (cloud model)
```

---

### Mode Commands

#### `/mode`
Switch agent mode

**Usage:** `/mode [mode_name]`

**Aliases:** `m`

**Available Modes:**
- `build` - Implementation mode
- `plan` - Planning mode
- `explore` - Exploration mode

**Examples:**
```
> /mode build
Switch to build mode for implementation

> /mode plan
Enter planning mode

> /mode explore
Switch to exploration mode
```

#### `/build`
Switch to build mode for implementation

**Usage:** `/build`

**Aliases:** None

#### `/explore`
Switch to explore mode for code navigation

**Usage:** `/explore`

**Aliases:** None

#### `/plan`
Enter planning mode and manage plans

**Usage:** `/plan [subcommand]`

**Aliases:** None

**Subcommands:**
- `/plan` - Show current plan
- `/plan new` - Create new plan
- `/plan status` - Show plan status

---

### Session Management Commands

#### `/session`
Session management operations

**Usage:** `/session [subcommand]`

**Aliases:** `sessions`

**Subcommands:**
- `/session` - Show current session info
- `/session new` - Create new session
- `/session save [name]` - Save current session
- `/session load [id]` - Load session by ID
- `/session list` - List all sessions
- `/session delete [id]` - Delete a session

**Examples:**
```
> /session
Shows current session details

> /session save my-work
Saves conversation as 'my-work'

> /session list
Lists all saved sessions

> /session load abc123
Loads session with ID abc123
```

#### `/save` ⚡
Quick save current session

**Usage:** `/save [name]`

**Aliases:** None

**Example:**
```
> /save conversation-1
Saves current session
```

#### `/load` ⚡
Load a session

**Usage:** `/load [session_id]`

**Aliases:** `resume`

**Example:**
```
> /load abc123
Loads session with ID abc123
```

#### `/compact`
Compact sessions (remove old/duplicate sessions)

**Usage:** `/compact`

**Aliases:** None

---

### System & Tools Commands

#### `/system`
Show system information

**Usage:** `/system`

**Aliases:** `status`

**Shows:**
- Provider and model
- Available tools
- Session statistics
- System configuration

**Example:**
```
> /system
Displays comprehensive system information
```

#### `/tools`
List available tools

**Usage:** `/tools [filter]`

**Aliases:** None

**Examples:**
```
> /tools
Lists all available tools

> /tools read
Lists all read-related tools

> /tools git
Lists all git-related tools
```

#### `/config`
Show or edit configuration

**Usage:** `/config [key] [value]`

**Aliases:** None

**Example:**
```
> /config
Shows all configuration

> /config provider anthropic
Changes default provider

> /config temperature 0.7
Changes default temperature
```

#### `/theme`
Change TUI theme

**Usage:** `/theme [theme_name]`

**Aliases:** None

**Available Themes:**
- `default` - Standard dark theme
- `light` - Light theme
- `monokai` - Monokai color scheme
- `nord` - Nord color scheme

**Example:**
```
> /theme monokai
Switches to Monokai theme
```

---

### Codebase Commands

#### `/init`
Initialize or update .victor/init.md with codebase analysis

**Usage:** `/init`

**Aliases:** None

**Example:**
```
> /init
Analyzes codebase and creates init.md
```

#### `/reindex`
Reindex codebase for semantic search

**Usage:** `/reindex`

**Aliases:** `index`

**Example:**
```
> /reindex
Rebuilds semantic search index
```

#### `/directory`
Show or change working directory

**Usage:** `/directory [path]`

**Aliases:** `dir`, `cd`, `pwd`

**Examples:**
```
> /directory
Shows current directory

> /directory /path/to/project
Changes working directory

> /cd src
Changes to src directory
```

#### `/changes`
View, diff, or revert file changes

**Usage:** `/changes [subcommand] [file]`

**Aliases:** `diff`, `rollback`

**Subcommands:**
- `/changes` - List all changed files
- `/changes [file]` - Show diff for file
- `/changes revert [file]` - Revert file changes

**Examples:**
```
> /changes
Lists all modified files

> /changes src/main.py
Shows diff for main.py

> /changes revert src/main.py
Reverts changes to main.py
```

#### `/copy`
Copy file content to clipboard

**Usage:** `/copy [file]`

**Aliases:** None

**Example:**
```
> /copy src/utils.py
Copies file content to clipboard
```

#### `/search`
Search codebase

**Usage:** `/search [query]`

**Aliases:** None

**Example:**
```
> /search "async def"
Searches for async function definitions
```

#### `/filehistory`
Show file edit history

**Usage:** `/filehistory [file]`

**Aliases:** None

**Example:**
```
> /filehistory src/main.py
Shows edit history for main.py
```

---

### Metrics & Debug Commands

#### `/metrics`
Show streaming performance metrics and provider stats

**Usage:** `/metrics`

**Aliases:** `perf`, `performance`

**Shows:**
- Token usage
- Response times
- Tool call statistics
- Provider performance

**Example:**
```
> /metrics
Displays performance statistics
```

#### `/cost`
Show estimated token usage and cost for this session

**Usage:** `/cost`

**Aliases:** `usage`, `tokens`, `stats`

**Shows:**
- Total tokens used
- Estimated cost
- Cost by provider/model
- Token breakdown (input/output)

**Example:**
```
> /cost
Shows cost breakdown
```

#### `/mlstats`
Show ML-friendly aggregated session statistics for RL training

**Usage:** `/mlstats`

**Aliases:** `ml`, `analytics`

**Shows:**
- Aggregated metrics
- Training-friendly statistics
- Session summaries

**Example:**
```
> /mlstats
Shows ML statistics
```

#### `/learning`
Show Q-learning stats, adjust exploration, or reset session

**Usage:** `/learning [subcommand]`

**Aliases:** `qlearn`, `rl`

**Subcommands:**
- `/learning` - Show learning stats
- `/learning reset` - Reset learning session
- `/learning explore [rate]` - Adjust exploration rate

**Examples:**
```
> /learning
Shows Q-learning statistics

> /learning reset
Resets learning session

> /learning explore 0.3
Sets exploration rate to 30%
```

#### `/serialization`
Show token-optimized serialization statistics and savings

**Usage:** `/serialization`

**Aliases:** `serialize`, `ser`

**Shows:**
- Serialization statistics
- Token savings
- Compression metrics

**Example:**
```
> /serialization
Shows serialization stats
```

---

### Navigation Commands

#### `/undo`
Undo the last action

**Usage:** `/undo`

**Aliases:** None

#### `/redo`
Redo the last undone action

**Usage:** `/redo`

**Aliases:** None

#### `/snapshots`
Manage conversation snapshots

**Usage:** `/snapshots [subcommand]`

**Aliases:** None

**Subcommands:**
- `/snapshots` - List snapshots
- `/snapshots create` - Create snapshot
- `/snapshots restore [id]` - Restore snapshot

---

### Checkpoint Commands

#### `/checkpoint`
Manage conversation checkpoints for time-travel debugging

**Usage:** `/checkpoint [subcommand]`

**Aliases:** None

**Subcommands:**
- `/checkpoint` - Show current checkpoint
- `/checkpoint create` - Create checkpoint
- `/checkpoint restore [id]` - Restore checkpoint
- `/checkpoint list` - List all checkpoints

**Examples:**
```
> /checkpoint create
Creates a checkpoint

> /checkpoint restore abc123
Restores to checkpoint abc123

> /checkpoint list
Lists all checkpoints
```

---

### Entity Commands

#### `/entities`
Query and manage extracted entities

**Usage:** `/entities [entity_type]`

**Aliases:** None

**Entity Types:**
- `class` - Classes
- `function` - Functions
- `variable` - Variables
- `module` - Modules

**Examples:**
```
> /entities
Lists all entities

> /entities class
Lists all classes

> /entities MyClass
Shows details for MyClass
```

---

### Review Commands

#### `/review`
Start code review session

**Usage:** `/review [file]`

**Aliases:** None

**Example:**
```
> /review src/main.py
Reviews main.py file
```

#### `/bug`
Start bug investigation session

**Usage:** `/bug [description]`

**Aliases:** None

**Example:**
```
> /bug "Null pointer exception in user auth"
Starts bug investigation
```

---

### MCP Commands

#### `/mcp`
Manage Model Context Protocol servers

**Usage:** `/mcp [subcommand]`

**Aliases:** None

**Subcommands:**
- `/mcp` - List MCP servers
- `/mcp start [name]` - Start MCP server
- `/mcp stop [name]` - Stop MCP server
- `/mcp status` - Show MCP status

**Examples:**
```
> /mcp
Lists all MCP servers

> /mcp start filesystem
Starts filesystem MCP server
```

#### `/context`
Manage context settings

**Usage:** `/context [setting] [value]`

**Aliases:** None

**Example:**
```
> /context max_tokens 100000
Sets context window size
```

#### `/approvals`
Manage tool approval settings

**Usage:** `/approvals [on|off]`

**Aliases:** None

**Example:**
```
> /approvals on
Enable tool approval prompts
```

---

### Provider-Specific Commands

#### `/lmstudio`
LM Studio provider commands

**Usage:** `/lmstudio [subcommand]`

**Aliases:** None

**Subcommands:**
- `/lmstudio` - Show LM Studio status
- `/lmstudio connect` - Connect to LM Studio
- `/lmstudio disconnect` - Disconnect from LM Studio

---

## TUI Components

### 1. StatusBar
**Location:** Top of screen

**Displays:**
- Victor branding
- Current provider (e.g., "anthropic")
- Current model (e.g., "claude-3-5-sonnet")

### 2. ConversationLog
**Location:** Middle of screen (scrollable)

**Features:**
- Displays conversation history
- User messages (green/primary color)
- Assistant messages (blue/secondary color)
- System messages (gray/muted)
- Error messages (red/error color)
- Code blocks with syntax highlighting
- Auto-scrolling to latest message
- Manual scroll navigation

### 3. InputWidget
**Location:** Bottom of screen

**Features:**
- Multi-line input support
- Text wrapping
- Input history navigation (↑/↓)
- Character/word count
- Hint text showing shortcuts
- Auto-focus on startup

### 4. ThinkingWidget (Optional)
**Location:** Between conversation and input (toggleable)

**Displays:**
- Model's internal reasoning
- Thought process
- Decision making
- Toggle with `Ctrl+T`

### 5. ToolCallWidget (Optional)
**Location:** Between conversation and input

**Displays:**
- Tool execution status
- Pending tools (yellow border)
- Successful tools (green border)
- Failed tools (red border)
- Tool results

### 6. ToolProgressPanel
**Features:**
- Real-time tool execution updates
- Progress bars for long-running tools
- Tool output streaming
- Error reporting

---

## Usage Examples

### Example 1: Basic Chat Session

```bash
# Start TUI
$ victor chat

# In TUI:
> Hello, can you help me understand this codebase?
[Victor responds with analysis]

> /model claude-3-opus
[Switches to Opus model]

> What are the main components?
[Victor Opus provides detailed analysis]

> /session save architecture-review
[Session saved]
```

### Example 2: Code Exploration

```bash
$ victor chat --provider ollama --model llama3:8b

> /mode explore
[Switches to exploration mode]

> /directory
[Shows current directory]

> /init
[Initializes codebase analysis]

> /entities class
[Lists all classes]

> /search "async def"
[Searches for async functions]
```

### Example 3: Development Workflow

```bash
$ victor chat

> /mode build
[Switches to build mode]

> Implement a user authentication system
[Victor provides implementation plan]

> /checkpoint create dev-start
[Creates checkpoint]

> [Implementation continues...]

> /changes
[Shows all file changes]

> /test run user_auth.py
[Runs tests]

> /checkpoint restore dev-start
[Restores to earlier state if needed]
```

### Example 4: Multi-line Input

```bash
$ victor chat

> Here's a complex question with multiple parts: [Shift+Enter]
> Part 1: How does the authentication work? [Shift+Enter]
> Part 2: What are the security implications? [Shift+Enter]
> Part 3: How can we improve it? [Enter]
[Sends complete multi-line question]
```

---

## Configuration

### Settings File: `~/.victor/config.yaml`

```yaml
# Provider settings
provider: anthropic
model: claude-3-5-sonnet

# TUI settings
tui:
  theme: default
  show_thinking: true
  show_tool_calls: true
  max_history: 100

# Session settings
sessions:
  auto_save: true
  save_interval: 300  # seconds
  storage_path: ~/.victor/sessions

# Semantic search
semantic_search:
  enabled: true
  index_path: ~/.victor/index

# Tool settings
tools:
  approval_mode: auto
  timeout: 30
```

### Profiles File: `~/.victor/profiles.yaml`

```yaml
profiles:
  fast:
    provider: ollama
    model: llama3:8b
    temperature: 0.5
    description: "Fast local model for testing"

  smart:
    provider: anthropic
    model: claude-3-opus
    temperature: 0.8
    description: "Smart cloud model for complex tasks"

  coding:
    provider: anthropic
    model: claude-3-5-sonnet
    temperature: 0.3
    description: "Balanced model for coding"
```

### Environment Variables

```bash
# Provider selection
export VICTOR_PROVIDER=anthropic
export VICTOR_MODEL=claude-3-5-sonnet

# TUI customization
export VICTOR_THEME=monokai
export VICTOR_SHOW_THINKING=true

# Session management
export VICTOR_SESSION_PATH=~/.victor/sessions
export VICTOR_AUTO_SAVE=true

# API keys
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

---

## Advanced Features

### 1. Streaming Responses
The TUI shows AI responses streaming in real-time, character by character. This provides:
- Immediate feedback
- Perceived faster response
- Ability to stop generation (Ctrl+C)

### 2. Tool Execution Visualization
When the AI uses tools:
- Tool call is displayed immediately
- Status updates in real-time (pending → success/failure)
- Tool output shown inline
- Progress bars for long-running operations

### 3. Multi-line Input
Support for complex prompts:
- `Shift+Enter` for newlines
- Auto-expanding input box
- Syntax highlighting for code blocks
- Proper indentation preservation

### 4. Session Persistence
- Auto-save every 5 minutes (configurable)
- Manual save with `/session save` or `Ctrl+S`
- Session history with `/session list`
- Load previous sessions with `/session load [id]`

### 5. Semantic Search
- Codebase indexing with `/init`
- Natural language search with `/search`
- Entity extraction with `/entities`
- Reindex with `/reindex`

### 6. Checkpoints
- Create save points with `/checkpoint create`
- Restore to any checkpoint with `/checkpoint restore [id]`
- Time-travel debugging
- Experiment safely with rollbacks

### 7. Multiple Modes
- **Build Mode**: For implementation tasks
- **Plan Mode**: For planning and design
- **Explore Mode**: For code navigation and understanding

### 8. Metrics & Analytics
- Token usage tracking (`/cost`)
- Performance metrics (`/metrics`)
- ML statistics (`/mlstats`)
- Learning analytics (`/learning`)

### 9. Custom Themes
Multiple color schemes:
- Default (dark)
- Light
- Monokai
- Nord
- Custom via CSS

### 10. Keyboard-First Design
All features accessible via keyboard:
- No mouse required
- Fast power-user workflows
- Muscle memory friendly

---

## Tips and Tricks

### Productivity Tips

1. **Quick Model Switching**
   ```
   Use /profile to switch between pre-configured model setups
   ```

2. **Session Management**
   ```
   Create checkpoints before major changes
   Use /save frequently during long sessions
   ```

3. **Code Exploration**
   ```
   Start with /mode explore
   Use /init to analyze codebase
   Use /entities to navigate structure
   ```

4. **Debugging**
   ```
   Use /checkpoint create before experiments
   Use /changes to track modifications
   Use /checkpoint restore to rollback
   ```

5. **Cost Management**
   ```
   Use /cost to monitor token usage
   Switch to local models (/profile fast) for testing
   Use smart models (/profile smart) for complex tasks
   ```

### Keyboard Shortcuts Cheat Sheet

```
Ctrl+C    → Exit
Ctrl+L    → Clear
Ctrl+T    → Toggle thinking
Ctrl+S    → Save session
Ctrl+/    → Help
Ctrl+↑↓   → Scroll
Escape    → Focus input
```

---

## Troubleshooting

### TUI Not Starting

**Problem:** TUI fails to start

**Solutions:**
1. Check terminal compatibility (use iTerm2, Terminal.app, or Linux terminal)
2. Ensure Python 3.10+
3. Install dependencies: `pip install -e ".[tui]"`

### Display Issues

**Problem:** TUI looks garbled

**Solutions:**
1. Check terminal size (minimum 80x24)
2. Try different theme: `/theme default`
3. Ensure UTF-8 encoding

### Performance Issues

**Problem:** TUI is slow

**Solutions:**
1. Disable thinking panel: `Ctrl+T`
2. Use faster model: `/profile fast`
3. Reduce history in config

---

## Command-Line Options

```bash
# Basic usage
victor chat

# With specific provider/model
victor chat --provider ollama --model llama3:8b

# With profile
victor chat --profile fast

# One-shot message (non-interactive)
victor chat "Explain this code"

# With vertical
victor chat --vertical coding

# Disable streaming
victor chat --no-stream

# With custom config
victor chat --config /path/to/config.yaml
```

---

## Summary

Victor's TUI provides a powerful, feature-rich terminal interface with:

- **43+ slash commands** for all operations
- **11 keyboard shortcuts** for power users
- **6 main UI components** for a complete experience
- **Multiple modes** for different workflows
- **Full session management** with persistence
- **Real-time streaming** of responses
- **Tool execution** visualization
- **Customizable themes** and appearance

The TUI is designed for keyboard-first operation, making it efficient for developers who prefer terminal workflows over GUI applications.
