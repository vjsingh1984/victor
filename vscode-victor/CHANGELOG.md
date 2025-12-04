# Change Log

All notable changes to the Victor AI VS Code extension will be documented in this file.

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
