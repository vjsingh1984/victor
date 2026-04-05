# Victor Verticals: 10-Minute Demo Script

> **Audience:** Developers evaluating Victor, community meetups, conference lightning talks
>
> **Goal:** Show how verticals turn a general-purpose agent framework into domain-specific tools — and how anyone can build one.
>
> **Prerequisites:** `pip install victor-ai` (security vertical ships as an example; benchmark is built-in)

---

## Act 1: Discovery (2 min) — "What can Victor do?"

### 1a. The Vertical Ecosystem

```bash
victor vertical list
```

```
                               Vertical Ecosystem
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name      ┃ Version ┃ Source   ┃ Description                                ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ benchmark │ builtin │ Builtin  │ Benchmark vertical for SWE-bench,          │
│           │         │          │ HumanEval, and code generation evaluation  │
│ security  │ 0.2.0   │ External │ Security auditing, vulnerability analysis, │
│           │         │          │ and compliance review                      │
└───────────┴─────────┴──────────┴────────────────────────────────────────────┘
```

**Talking point:** Victor discovers verticals automatically — built-in ones ship with the framework, external ones are found via Python entry points when you `pip install` them. No config files, no manual registration.

### 1b. Drill into a Vertical

```bash
victor vertical info security
```

```
Vertical: security
Version: 0.2.0
Source: External
Location: /path/to/victor-security

Description: Security auditing, vulnerability analysis, and compliance review
Authors: Victor Community
Category: security
Tags: security, audit, compliance, vulnerability

Capabilities:
  - file_ops
  - git
  - web_access

Provides Tools:
  - read, ls, code_search, shell, web_search, write

Provides Workflows:
  - vulnerability_scan
  - dependency_audit
  - incident_review
```

**Talking point:** Each vertical declares exactly what it needs — tools, capabilities, workflows. The framework provisions only those resources. A security vertical doesn't get database tools; a benchmark vertical doesn't get web search.

---

## Act 2: Security Vertical (4 min) — "Find real vulnerabilities"

### 2a. What the Security Vertical Brings

The security vertical is a complete security review system:

| Component | What It Does |
|-----------|-------------|
| **System prompt** | Instructs the LLM to confirm findings with evidence before reporting, explain severity in practical terms, and provide specific remediation |
| **3-stage workflow** | Reconnaissance (map the codebase) -> Analysis (scan for vulnerabilities) -> Reporting (summarize findings + fixes) |
| **2-agent team** | Security Analyst finds issues; Validation Reviewer confirms them and calibrates severity |
| **Task types** | `vulnerability_scan` (18 tool calls), `dependency_audit` (14 calls), `incident_review` (20 calls) |

### 2b. Live: Run a Security Scan

```bash
victor chat --vertical security
```

> **Prompt:** "Scan this repository for hardcoded secrets, SQL injection risks, and dependency vulnerabilities"

The agent will:
1. **Reconnaissance** — `ls` and `read` to map the project structure
2. **Analysis** — `code_search` for patterns like `password=`, `eval(`, SQL string concatenation; optionally `shell` to run `bandit` or `pip audit`
3. **Reporting** — Structured findings with severity, evidence, and remediation

Example output (paraphrased):

```
FINDINGS:

[HIGH] SQL Injection in src/db.py:45
  Evidence: f"SELECT * FROM users WHERE id={user_id}"
  Fix: Use parameterized queries — cursor.execute("SELECT ... WHERE id=?", (user_id,))

[MEDIUM] Hardcoded API key in config.py:12
  Evidence: API_KEY = "sk-live-abc123..."
  Fix: Move to environment variable or secrets manager

[LOW] Outdated dependency: requests==2.25.1
  CVE: CVE-2023-32681 (SSRF via redirect)
  Fix: pip install --upgrade requests>=2.31.0

3 issues found | 12 tool calls | Pipeline: analyst -> reviewer
```

### 2c. Key Architecture Points

```
SecurityAssistant (SDK-only, no framework imports)
    |
    +-- get_system_prompt()     -> domain expertise
    +-- get_tools()             -> [read, ls, code_search, shell, web_search, write]
    +-- get_stages()            -> reconnaissance -> analysis -> reporting
    +-- get_team_declarations() -> analyst + reviewer pipeline
    +-- get_task_type_hints()   -> tool budgets per workflow type
```

The security vertical imports only from `victor-sdk` — it has zero coupling to Victor internals. This is what makes external verticals truly pluggable.

---

## Act 3: Benchmark Vertical (3 min) — "Measure your agent"

### 3a. Available Benchmarks

```bash
victor benchmark list
```

```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Benchmark      ┃ Description                       ┃ Tasks ┃ Source          ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ swe-bench      │ Real-world GitHub issue resolution │ ~2300 │ Princeton NLP   │
│ swe-bench-lite │ Curated subset of SWE-bench       │ ~300  │ Princeton NLP   │
│ humaneval      │ Code generation from docstrings   │ 164   │ OpenAI          │
│ mbpp           │ Mostly Basic Python Problems      │ 974   │ Google Research │
│ mbpp-test      │ MBPP test split                   │ 500   │ Google Research │
└────────────────┴───────────────────────────────────┴───────┴─────────────────┘
```

### 3b. The Benchmark 4-Stage Pipeline

```
UNDERSTANDING ──> ANALYSIS ──> IMPLEMENTATION ──> VERIFICATION
  "What's the      "Where's       "Fix it"         "Prove it
   problem?"        the code?"                       works"
```

Each stage unlocks specific tools:
- **Understanding:** read, grep, ls (read-only exploration)
- **Analysis:** code_search, symbol, refs (semantic search)
- **Implementation:** edit, write (code modification)
- **Verification:** test, shell, git (validation)

This staged approach prevents the agent from jumping straight to editing before understanding the problem — a common failure mode in LLM coding agents.

### 3c. Live: Run a Benchmark

```bash
# Run HumanEval with 10 tasks
victor benchmark run humaneval --max-tasks 10 --output results.json

# Run SWE-bench lite
victor benchmark run swe-bench-lite --max-tasks 5

# Compare against other frameworks
victor benchmark compare --benchmark swe-bench

# View leaderboard
victor benchmark leaderboard
```

### 3d. Safety Built In

The benchmark vertical includes a safety extension that blocks dangerous operations during evaluation — because agents running benchmarks will try creative solutions:

- Blocks `rm -rf`, `format`, `fdisk` commands
- Prevents modifications to `/prod/` or `/production/` paths
- Blocks `git push --force`

---

## Act 4: Build Your Own (1 min) — "It's just a Python class"

### Scaffold a New Vertical

```bash
victor vertical new my-vertical --description "My custom domain vertical"
```

This generates a complete package structure with `pyproject.toml`, entry points, and a starter `victor-vertical.toml`.

### The Minimal Vertical

```python
# my_vertical/assistant.py
from victor_sdk import VerticalBase

class MyAssistant(VerticalBase):
    name = "my-vertical"
    version = "0.1.0"

    def get_system_prompt(self):
        return "You are a specialist in ..."

    def get_tools(self):
        return ["read", "write", "shell"]

    def get_stages(self):
        return [
            {"name": "research", "tools": ["read", "code_search"]},
            {"name": "execute",  "tools": ["write", "shell"]},
        ]
```

```toml
# pyproject.toml
[project.entry-points."victor.verticals"]
my-vertical = "my_vertical.assistant:MyAssistant"
```

```bash
pip install -e .
victor vertical list   # your vertical appears automatically
```

**Talking point:** No framework registration, no plugin config, no decorators. Just implement the protocol, declare the entry point, and `pip install`. Victor discovers it at runtime.

---

## Quick Reference Card

| Command | What It Does |
|---------|-------------|
| `victor vertical list` | Show all discovered verticals |
| `victor vertical list --verbose` | Include tools and capabilities |
| `victor vertical list --category security` | Filter by category |
| `victor vertical info <name>` | Detailed vertical metadata |
| `victor vertical install <package>` | Install from PyPI/git/local |
| `victor vertical new <name>` | Scaffold a new vertical package |
| `victor chat --vertical <name>` | Chat using a specific vertical |
| `victor benchmark list` | Show available benchmarks |
| `victor benchmark run <name>` | Run a benchmark evaluation |
| `victor benchmark compare` | Compare against other frameworks |

---

## Key Takeaways

1. **Verticals are domain specializations** — not just prompts, but tools + stages + teams + safety, all declared in one class
2. **External verticals are first-class** — same discovery, same capabilities, zero coupling to framework internals
3. **Benchmark vertical ships built-in** — evaluate your agent against SWE-bench, HumanEval, MBPP out of the box
4. **Security vertical shows the SDK model** — a complete security review system in ~200 lines of Python, using only `victor-sdk`
5. **The plugin system is pip-native** — entry points, editable installs, PyPI distribution all work
