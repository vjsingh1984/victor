# Change Log

All notable changes to the Victor AI VS Code extension will be documented in this file.

## [0.2.0] - 2025-12-24

### Added
- **Background Agents**: Run AI agents asynchronously with real-time progress tracking
  - Start agents with `Cmd+Shift+A` / `Ctrl+Shift+A`
  - Agent sidebar shows active tasks with tool call details
  - WebSocket-based real-time updates
  - Cancel running agents, view output
  - Status bar indicator shows active agent count
- **Terminal Agent**: AI-assisted terminal command execution (like Cursor/Copilot)
  - Suggest commands based on natural language intent
  - Dangerous command detection with approval workflow
  - Command history and approval tracking
  - Terminal Agent panel in sidebar
- **@workspace Context**: GitHub Copilot-style repo-aware context
  - `@workspace`: Project structure, configs, open files
  - `@selection`: Current editor selection
  - `@terminal`: Active terminal context
  - Enhanced `@git`, `@problems`, `@file:`, `@folder:`
- **Inline Edit with Diff Preview**: Edit selected code with AI suggestions
  - Ghost text preview of changes
  - Show diff before applying
  - Accept/reject with Tab/Escape
- **FIM Completions**: Fill-in-the-Middle support for better inline suggestions
  - Includes code after cursor for context-aware completions
- **Autonomy Controls**: Control how much freedom Victor has
  - Autonomy levels: Manual, Semi-auto, Auto
  - Configurable approval thresholds for terminal/file operations
  - Dangerous command pattern detection
- **Smart Paste**: Adapt pasted code to current context (like Windsurf)
  - Automatically adjusts indentation, naming conventions
  - Adds missing imports based on file context
  - Preview adaptations before applying
  - Keybinding: `Cmd+Shift+P` / `Ctrl+Shift+P`
- **Multi-file Composer**: Complex multi-file edits in one prompt (like Cursor)
  - Add multiple files to composer context
  - Describe changes in natural language
  - Preview all changes before applying
  - Selective application of changes
  - Keybinding: `Cmd+Shift+C` / `Ctrl+Shift+C`

### Changed
- **Security**: API keys now sent via WebSocket message instead of URL query string
- **Security**: Fixed CORS to allow only localhost and VS Code webview origins
- **Security**: Added input validation for code search file patterns
- **Security**: Added path traversal prevention for git operations
- Improved inline completion with suffix context and request cancellation
- Better error handling with user-friendly messages in chat

### Configuration
- `victor.agents.maxConcurrent`: Max concurrent background agents (default: 4)
- `victor.agents.showNotifications`: Show notifications on agent completion (default: true)
- `victor.autonomy.level`: Autonomy level - manual/semi-auto/auto (default: semi-auto)
- `victor.autonomy.autoApproveReadOnly`: Auto-approve read operations (default: true)
- `victor.autonomy.requireApprovalForTerminal`: Require approval for terminal (default: true)
- `victor.autonomy.requireApprovalForFileWrites`: Require approval for writes (default: false)
- `victor.autonomy.dangerousCommandPatterns`: Patterns that always need approval

### Keyboard Shortcuts
- `Cmd+Shift+A` / `Ctrl+Shift+A`: Start Background Agent
- `Cmd+Shift+T` / `Ctrl+Shift+T`: Suggest Terminal Command
- `Cmd+Shift+P` / `Ctrl+Shift+P`: Smart Paste
- `Cmd+Shift+C` / `Ctrl+Shift+C`: Open Composer

## [0.1.0] - 2025-01-01

### Added
- Initial release of Victor AI VS Code extension
- **Chat Interface**: Interactive AI chat panel in sidebar
- **Auto Server Management**: Automatic server start/stop with health monitoring
- **Code Actions**: Explain, refactor, fix, test, and document selected code
- **Semantic Search**: Natural language code search
- **Inline Completions**: AI-powered code suggestions
- **Multi-Provider Support**: Claude, GPT-4, Gemini, Ollama, LMStudio
- **Agent Modes**: Build, Plan, and Explore modes
- **Undo/Redo**: Revert or reapply AI-made changes
- **Context Providers**: @file, @folder, @problems, @git mentions

### Configuration
- `victor.serverPort`: Port for Victor server (default: 8765)
- `victor.autoStart`: Auto-start server on activation (default: true)
- `victor.provider`: Default LLM provider (default: anthropic)
- `victor.model`: Default model (default: claude-sonnet-4-20250514)
- `victor.mode`: Agent mode (default: build)
- `victor.showInlineCompletions`: Enable inline suggestions (default: true)

### Keyboard Shortcuts
- `Ctrl+Shift+V` / `Cmd+Shift+V`: Open Chat
- `Ctrl+Shift+E` / `Cmd+Shift+E`: Explain Selection
- `Ctrl+Shift+S` / `Cmd+Shift+S`: Semantic Search
