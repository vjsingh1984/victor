# UX Redesign — Comprehensive Analysis & Implementation Plan

## Executive Summary

This document captures the systematic analysis, design decisions, and implementation
roadmap for rethinking the Victor AI framework's user experience across four key
dimensions: **Tool Calls**, **Preambles/Prompts**, **Conversation Flow**, and
**Packaging/Deployment**. The work is isolated in the `.worktrees/ux-redesign`
worktree based on `origin/develop`.

---

## Phase 1: Tool Call UX (COMPLETED)

### Analysis

The existing tool display had:
- **Flat invocation lines** with minimal visual hierarchy
- **No progress indicators** for long-running tools
- **Generic previews** that didn't leverage tool-specific semantics
- **Inconsistent categorization** with no visual grouping
- **No duration tracking** or caching heuristics

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Visual hierarchy** with section headers + badges | Users need to quickly scan tool activity; hierarchical display reduces cognitive load |
| **Rich progress bars** for long-running tools | Shell, code_search, web tools can take 10+ seconds; users need feedback |
| **Smart previews** per tool category | Different tools produce different output shapes; generic first-N-lines is wasteful |
| **Status icons** (success/failure/cached/pruned) | Instant visual recognition of tool outcome without reading text |
| **Caching heuristic** (elapsed < 50ms) | Fast results from cache should show cached icon to build user trust |

### Implementation

The `ToolDisplayManager` class was enhanced with:

- **Tool categories with emoji prefixes**: `_TOOL_CATEGORIES` dict maps tools to visual groups
- **Status icon class**: `_ToolStatusIcon` with PENDING, RUNNING, SUCCESS, FAILURE, WARNING, PRUNED, CACHED states
- **Structured `on_tool_start()`**: Section header on first call, category badge, invocation line with metadata
- **Live `on_tool_progress()`**: Terminal block with `Progress` bar, elapsed time, last N lines of output
- **Smart `on_tool_result()`**: Status icon + duration + context-aware preview per tool category
- **`_extract_result_summary()`**: 10+ tool-specific summary extractors (search counts, file lines, exit codes, test results)
- **`_was_cached()` heuristic**: Results under 50ms for non-trivial tools get cached icon

---

## Phase 2: Thinking Display UX (COMPLETED)

### Analysis

The existing thinking display had:
- **No duration tracking** for thinking sessions
- **No progressive disclosure** for long reasoning chains
- **No summary** at the end of long thinking sessions
- **No character count** or progress metrics

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Duration tracking** | Users need to know how long the model is thinking |
| **Progressive disclosure** at 500+ chars | Long reasoning chains overwhelm the display; summarization reduces noise |
| **Periodic status updates** every 5 chunks | Keeps user informed without flooding the terminal |
| **End-of-thinking summary** | Reasoning complete (N chars, X.Xs) provides closure |

### Implementation

The `ThinkingDisplayManager` class was enhanced with:

- `_thinking_start_time`: Tracks when thinking began for duration display
- `_thinking_char_count`: Running total of thinking content received
- `_LONG_THINKING_THRESHOLD_CHARS = 500`: Threshold for progressive disclosure
- `on_thinking_content()`: Shows inline text for short reasoning, periodic summaries for long chains
- `on_thinking_end()`: Displays summary with total chars and duration for long sessions

---

## Phase 3: Event Dispatcher Refinement (COMPLETED)

### Analysis

The existing event dispatcher had:
- **Flat if-elif chain** that was hard to extend
- **No early returns** — every condition was checked even after match
- **Inconsistent error handling** across event types

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Early return pattern** | Once an event is matched and handled, skip remaining checks |
| **Structured routing order** | Tool events -> Errors -> Progress -> Status -> Previews -> Reasoning -> Content |
| **Consistent handler signatures** | All handlers follow same pattern for testability |
| **Graceful fallback** | Unknown events with content are treated as content events |

---

## Phase 4: Preambles & Prompt UX (PLANNED)

### Analysis Points to Address

1. **System prompt visibility**: Users should be able to see what system prompt is active
2. **Prompt section management**: Current 5 sections should be toggleable and visible
3. **Prompt optimization feedback**: GEPA/MIPROv2/CoT distillation should show what changed
4. **Preamble customization**: Users should be able to inject custom preambles per session

### Proposed Implementation

```python
class PreambleManager:
    """Manages system prompt sections and user preambles."""
    
    def get_active_sections(self) -> dict[str, str]:
        """Return currently active prompt sections."""
        
    def set_preamble(self, text: str, position: PreamblePosition) -> None:
        """Inject custom preamble at specified position."""
        
    def toggle_section(self, section_name: str, enabled: bool) -> None:
        """Enable/disable a specific prompt section."""
```

**Slash commands to add:**
- `/preamble show` — Display current system prompt structure
- `/preamble set <text>` — Inject custom preamble
- `/preamble reset` — Reset to defaults
- `/prompt optimize status` — Show GEPA/MIPROv2 optimization state

---

## Phase 5: Conversation Flow UX (PLANNED)

### Analysis Points to Address

1. **Turn boundaries**: Enhanced with turn numbers and metadata
2. **Context window awareness**: Show when context is approaching limits
3. **Message threading**: Visual threading of related tool calls and responses
4. **Cost tracking per turn**: Show token/cost estimates per conversation turn

### Proposed Implementation

```python
class TurnTracker:
    """Tracks conversation turns with metadata."""
    
    def start_turn(self) -> int:
        """Begin a new turn, return turn number."""
        
    def end_turn(self, metrics: TurnMetrics) -> None:
        """Record turn completion with metrics."""
        
    def get_context_usage(self) -> ContextUsage:
        """Return current context window usage."""
```

---

## Phase 6: Packaging & Distribution (PLANNED)

### Analysis

Current packaging:
- Single `victor-ai` package with all dependencies
- No platform-specific wheels (universal wheel only)
- Heavy dependency chain (~20 core + optional extras)
- No binary distribution for non-Python environments

### Proposed Distribution Strategy

| Distribution | Format | Use Case | Priority |
|-------------|--------|----------|----------|
| **PyPI package** | sdist + wheel | pip install | P0 (existing) |
| **Platform wheels** | platform-specific | pip with native deps | P1 |
| **Standalone binary** | PyInstaller bundle | No-Python environments | P1 |
| **Docker image** | container | CI/CD, server | P0 (existing) |
| **Homebrew tap** | formula | macOS developers | P2 |
| **npm package** | npm | Node.js ecosystem | P3 |

### Proposed pyproject.toml Enhancements

```toml
[project.optional-dependencies]
minimal = ["rich>=13.0", "typer>=0.12", "httpx>=0.27"]
providers-openai = ["openai>=1.0"]
providers-anthropic = ["anthropic>=0.30"]
providers-ollama = ["httpx>=0.27"]
providers-all = ["victor-ai[providers-openai]", "victor-ai[providers-anthropic]", "victor-ai[providers-ollama]"]
```

---

## Phase 7: Cross-Platform UX (PLANNED)

### Terminal Detection & Adaptation

```python
class TerminalCapabilities:
    """Detect and adapt to terminal capabilities."""
    
    @staticmethod
    def supports_emoji() -> bool:
        """Check if terminal supports emoji display."""
        
    @staticmethod
    def supports_unicode() -> bool:
        """Check if terminal supports Unicode."""
        
    @staticmethod
    def supports_color() -> int:
        """Return color depth (8, 256, 16M, or 0)."""
        
    @staticmethod
    def get_terminal_width() -> int:
        """Return terminal width in characters."""
```

**Adaptation strategy:**
- **Full-featured**: iTerm2, Kitty, Alacritty, Windows Terminal - all features enabled
- **Basic**: CMD, xterm - ASCII-only, no colors
- **Restricted**: CI/CD, headless - plain text output
- **Web UI**: FastAPI/React - structured JSON events

---

## Implementation Roadmap

| Phase | Area | Status | Effort | Dependencies |
|-------|------|--------|--------|--------------|
| 1 | Tool Call UX | COMPLETED | 3 days | None |
| 2 | Thinking Display UX | COMPLETED | 1 day | Phase 1 |
| 3 | Event Dispatcher | COMPLETED | 1 day | Phases 1-2 |
| 4 | Preambles & Prompts | PLANNED | 3 days | None |
| 5 | Conversation Flow | PLANNED | 4 days | Phases 1-4 |
| 6 | Packaging & Distribution | PLANNED | 5 days | None |
| 7 | Cross-Platform UX | PLANNED | 3 days | Phases 1-6 |

---

## Files Modified (Phases 1-3)

| File | Change | Lines |
|------|--------|-------|
| `victor/ui/rendering/tool_display.py` | Enhanced tool display with visual hierarchy, progress bars, smart previews | ~500 |
| `victor/ui/rendering/thinking_display.py` | Enhanced thinking display with progressive disclosure, duration tracking | ~107 |
| `victor/ui/rendering/event_dispatcher.py` | Refined event routing with early returns, structured handlers | ~303 |

---

## Testing Strategy

Each phase includes:
1. **Unit tests** for new classes and methods
2. **Integration tests** for end-to-end rendering flows
3. **Performance benchmarks** for rendering latency

### Test Files to Create

```
tests/unit/ui/rendering/test_tool_display.py       # Phase 1
tests/unit/ui/rendering/test_thinking_display.py    # Phase 2
tests/unit/ui/rendering/test_event_dispatcher.py    # Phase 3
tests/unit/framework/test_preamble_manager.py       # Phase 4
tests/unit/ui/rendering/test_turn_tracker.py        # Phase 5
tests/integration/packaging/test_wheel_build.py     # Phase 6
tests/unit/ui/rendering/test_terminal_caps.py       # Phase 7
```

---

## Performance Considerations

| Component | Optimization | Expected Improvement |
|-----------|-------------|---------------------|
| Tool preview rendering | Lazy evaluation | 2-5x faster for large outputs |
| Thinking display | Progressive disclosure threshold | 10x less terminal output |
| Event dispatch | Early return pattern | ~15% fewer condition checks |
| Content buffer | HEAD/TAIL incremental rendering | ~5x faster live updates |
| Progress bars | Throttled updates (every 100ms) | Prevents terminal flooding |

---

*Last updated: 2026-07-08*
*Worktree: `.worktrees/ux-redesign`*
*Base branch: `origin/develop`*