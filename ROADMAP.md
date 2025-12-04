# Victor Feature Roadmap

Prioritized feature roadmap based on comparison with Claude Code, Aider, Continue.dev, Cline, OpenHands, Gemini CLI, and Codex CLI.

## Current Status

### Implemented Features
- Multi-provider support (6 providers: Anthropic, OpenAI, Ollama, Google, xAI, LMStudio/vLLM)
- 54 enterprise tools
- MCP Protocol (client + server)
- Semantic tool selection (embedding-based)
- Air-gapped mode with local LLMs
- Session persistence (save/load conversations)
- Safety checker with confirmation dialogs
- Workspace snapshots (create/restore/diff)
- Auto-commit with conventional commits
- Browser automation (Playwright-based)
- 28 slash commands (Claude Code parity)

---

## Priority 1: Critical for Developer Trust

### 1.1 Headless/CI Mode
Non-interactive mode for automation pipelines.

```bash
victor --headless "task description"
victor --headless --json "review code" > result.json
victor --headless --dry-run "refactor module"
```

**Flags:**
- `--headless` - Run without interactive UI
- `--json` - Output structured JSON
- `--quiet` - Only output errors
- `--dry-run` - Preview changes without applying
- `--max-changes N` - Limit file modifications
- `--timeout M` - Maximum runtime in minutes

**Exit codes:** 0=success, 1=error, 2=partial

**Reference:** Continue.dev, Gemini CLI, OpenHands

### 1.2 Repository Map
Build semantic map of codebase structure for better LLM context.

```python
class RepositoryMap:
    def build_map(self, root: Path) -> CodebaseMap
    def get_context_for_file(self, file: Path) -> str
    def find_related_files(self, file: Path) -> List[Path]
```

Extracts: functions, classes, imports, dependencies, call graphs.

**Reference:** Aider (repo map feature)

### 1.3 Auto Lint/Test Integration
Run linters and tests after each edit, auto-fix issues.

```python
# After successful edit:
1. Create snapshot
2. Apply edit
3. Run linters (black, ruff, mypy)
4. If lint fails: auto-fix or restore snapshot
5. Run tests if configured
6. If tests fail: restore snapshot, report error
```

**Reference:** Aider, Continue.dev

### 1.4 JSON Output Mode
Structured output for scripting and CI integration.

```json
{
  "success": true,
  "files_modified": ["src/api.py"],
  "changes": [...],
  "tokens_used": 1500,
  "duration_ms": 3200
}
```

---

## Priority 2: Enhanced Developer Experience

### 2.1 Chat Modes
Distinct operational modes for different tasks.

| Mode | Description | Tools Enabled |
|------|-------------|---------------|
| `/mode plan` | Research and planning | Read-only |
| `/mode ask` | Q&A about code | Read-only |
| `/mode code` | Full coding (default) | All |
| `/mode review` | Suggestions only | Read + suggest |

**Reference:** Aider (architect/ask/code), Continue.dev (4 modes)

### 2.2 Integrate Snapshots into Edit Flow
Auto-snapshot before each AI edit operation.

```
User: "refactor auth module"
Victor: [auto-snapshot] → [edit] → [verify] → [optional commit]
```

### 2.3 Integrate Auto-Commit into Edit Flow
Optional auto-commit after successful edits.

```yaml
# .victor.md or config
auto_commit: true
commit_style: conventional  # feat:, fix:, etc.
```

### 2.4 Image/Screenshot Context
Add images to conversation for visual debugging.

```
/image screenshot.png
Victor: "I see a React component with a broken layout..."
```

**Reference:** Aider, Cline

### 2.5 Voice Input
Speech-to-text for prompts.

**Reference:** Aider

---

## Priority 3: Enterprise Features

### 3.1 Custom Slash Commands
User-defined commands from markdown files.

```
~/.victor/commands/
├── review-pr.md      → /review-pr [number]
├── add-tests.md      → /add-tests [file]
└── deploy.md         → /deploy [env]
```

**Reference:** Claude Code (`.claude/commands/`)

### 3.2 Batch Processing
Process multiple files/tasks in parallel.

```bash
victor --batch tasks.json
victor --headless "add docstrings" --files "src/**/*.py"
```

**Reference:** Continue.dev

### 3.3 Pull Request Integration
Automated PR workflows.

- Summarize PR changes
- Apply reviewer feedback
- Fix failing tests
- Generate PR descriptions

**Reference:** OpenHands

### 3.4 Multi-Agent Coordination
Multiple specialized agents working on subtasks.

```python
class AgentCoordinator:
    async def delegate(self, task: str, agent_type: str)
    async def merge_results(self, results: List[AgentResult])

# Agent types: research, implement, test, review, docs
```

**Reference:** OpenHands

### 3.5 Dependency/Security Scanning
CVE detection, outdated dependencies, license compliance.

---

## Priority 4: Advanced Capabilities

### 4.1 IDE Plugin
VS Code and JetBrains integration.

**Reference:** Continue.dev, Cline

### 4.2 Metrics Dashboard
Usage analytics, cost tracking, success rates.

### 4.3 Team Collaboration
Shared sessions, handoffs between team members.

### 4.4 Fine-tuning Support
Use fine-tuned models for specific codebases.

---

## Feature Comparison Matrix

| Feature | Victor | Claude Code | Aider | Continue.dev | Cline |
|---------|--------|-------------|-------|--------------|-------|
| Multi-provider | Yes | No | Yes | Yes | Yes |
| MCP Protocol | Yes | Yes | No | Yes | Yes |
| Air-gapped | Yes | No | Yes | Yes | No |
| Semantic tools | Yes | No | No | No | No |
| Repo map | No | No | Yes | No | No |
| Headless/CI | No | Yes | No | Yes | No |
| Auto-commit | Yes | Yes | Yes | No | No |
| Snapshots | Yes | No | No | No | Yes |
| Browser | Yes | No | No | No | Yes |
| Chat modes | Partial | No | Yes | Yes | No |
| Voice input | No | No | Yes | No | No |
| Multi-agent | No | No | No | No | No |

---

## Implementation Order

```
Week 1-2:   Headless mode + JSON output
Week 3-4:   Repository map
Week 5-6:   Auto lint/test integration
Week 7-8:   Chat modes
Week 9-10:  Custom slash commands
Week 11-12: Batch processing
Month 4:    PR integration, multi-agent
Month 5+:   IDE plugin, advanced features
```

---

## Contributing

1. Pick a feature from this roadmap
2. Create feature branch: `git checkout -b feature/headless-mode`
3. Implement with tests
4. Run checks: `black . && ruff check . && mypy victor && pytest`
5. Submit PR referencing this roadmap

---

## Benchmarks & Metrics Integration

### Industry-Standard Benchmarks

| Benchmark | Type | Description | Priority |
|-----------|------|-------------|----------|
| HumanEval | Code Generation | 164 Python programming problems | High |
| MBPP | Code Generation | 974 Python problems from crowd-sourcing | Medium |
| SWE-Bench | Issue Resolution | Real GitHub issues with test verification | High |
| Aider Polyglot | Multi-file Editing | 225 exercises across 7 languages | Medium |
| LiveCodeBench | Continuous | Monthly updated coding challenges | Low |

### Current Metrics Tracking

**Implemented:**
- `victor/agent/observability.py` - Span-based tracing with hooks
- `victor/analytics/logger.py` - JSONL event logging (UsageLogger)
- `victor/context/manager.py` - Token counting and context management
- Provider-level token tracking (prompt_tokens, completion_tokens, total_tokens)
- `/usage` command for session token estimates

### Metrics Roadmap Items

**Phase A: Enhanced Token Tracking**
- Persist token usage across sessions to SQLite
- Add cost calculation per provider/model
- Cumulative session and lifetime metrics

**Phase B: Context Management Commands**
- `/compact` - Summarize conversation to reduce tokens
- `/context` - Show context window usage with visual indicator
- `/sessions` - List and resume previous sessions

**Phase C: Benchmark Runner**
- Create `victor/benchmarks/` module
- Implement HumanEval and SWE-Bench runners
- Auto-report pass@k metrics
- Track tool selection accuracy

**Phase D: Analytics Dashboard**
- `victor analytics --summary` for aggregate stats
- Export to CSV/JSON for external analysis
- Prometheus metrics endpoint

### Tool Selection Efficacy (Post-Optimization)

| Tool | Score | Keywords |
|------|-------|----------|
| database | 10/10 | SQL, schema, query, tables, columns, db |
| lsp | 10/10 | completions, hover, definition, references, diagnostics |
| dependency | 10/10 | package, pip, npm, requirements, outdated |
| cache | 10/10 | cache, stats, clear, hit rate |

---

## References

- [Claude Code](https://claude.ai/code)
- [Aider](https://aider.chat/)
- [Continue.dev](https://continue.dev/)
- [Cline](https://cline.bot/)
- [OpenHands](https://openhands.dev/)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- [HumanEval](https://github.com/openai/human-eval)
- [SWE-Bench](https://www.swebench.com/)
