# Skills System

Skills are composable units of agent expertise that sit between verticals (domain apps) and tools (atomic operations).

```
Vertical (domain app) → Skills (expertise) → Tools (operations)
```

## Quick Start

```python
from victor_sdk.skills import SkillDefinition

# Define a skill
my_skill = SkillDefinition(
    name="debug_test_failure",
    description="Diagnose and fix a failing test",
    category="coding",
    prompt_fragment=(
        "1. Read the failing test\n"
        "2. Run it to see the error\n"
        "3. Trace the code path\n"
        "4. Apply a fix\n"
        "5. Re-run to verify"
    ),
    required_tools=["read", "shell", "edit"],
    optional_tools=["grep", "code_search"],
    tags=frozenset({"debug", "test", "fix"}),
    max_tool_calls=25,
)
```

## Architecture

### SkillDefinition (SDK)

A frozen dataclass in `victor_sdk.skills.definition`:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique identifier |
| `description` | `str` | Human-readable description |
| `category` | `str` | Skill category (e.g., "coding") |
| `prompt_fragment` | `str` | Injected into agent prompt when active |
| `required_tools` | `List[str]` | Tools that MUST be available |
| `optional_tools` | `List[str]` | Tools that MAY be used |
| `tags` | `FrozenSet[str]` | Searchable tags |
| `max_tool_calls` | `int` | Max tool calls when active (default: 20) |
| `version` | `str` | Skill version (default: "1.0.0") |

### SkillProvider (SDK)

Protocol for classes that provide skills:

```python
from victor_sdk.skills import SkillProvider

class MyProvider:
    def get_skills(self) -> List[SkillDefinition]:
        return [my_skill]
```

### SkillRegistry (Framework)

Discovers and manages skills from verticals, plugins, and entry points:

```python
from victor.framework.skills import SkillRegistry

registry = SkillRegistry()

# Load from a vertical class
registry.from_vertical(CodingAssistant)

# Load from entry points
registry.from_entry_points()

# Search skills
results = registry.search("debug", category="coding")
```

## Adding Skills to a Vertical

Override `get_skills()` in your vertical:

```python
from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.skills import SkillDefinition

class MyVertical(VerticalBase):
    @classmethod
    def get_skills(cls) -> list:
        return [
            SkillDefinition(
                name="my_skill",
                description="Does something useful",
                category="my_vertical",
                prompt_fragment="Step-by-step instructions...",
                required_tools=["read", "write"],
            ),
        ]
```

## Entry Point Registration

External packages can register skills via the `victor.skills` entry point:

```toml
# pyproject.toml
[project.entry-points."victor.skills"]
my_skill = "my_package.skills:my_skill_definition"
```

## CLI

```bash
victor skill list                    # List all discovered skills
victor skill list --category coding  # Filter by category
victor skill info debug_test_failure # Show skill details
victor skill search "test debug"     # Search by keyword
```

## Built-in Skills (victor-coding)

| Skill | Description | Required Tools |
|-------|-------------|---------------|
| `debug_test_failure` | Diagnose and fix failing tests | read, shell, edit |
| `code_review` | Review code for correctness and security | read, grep |
| `implement_feature` | Implement a feature with tests | read, edit, write, shell |
| `refactor_code` | Refactor without changing behavior | read, edit, shell |
| `explore_codebase` | Understand codebase architecture | read, ls, grep |
