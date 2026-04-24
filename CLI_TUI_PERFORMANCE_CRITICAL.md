# CLI/TUI Performance Issues - CRITICAL

## Performance Metrics

**Current State:**
- ❌ Startup time: **7.46 seconds** (target: <1 second)
- ❌ Modules imported: **685 Victor modules** (target: <50)
- ❌ User experience: **Very sluggish**

**Impact:**
- CLI feels unresponsive
- Poor user experience on macOS
- Wastes 6+ seconds on every startup
- Makes Victor feel heavy and slow

---

## Root Causes

### 1. **Eager Import of Entire Framework** 🔴 CRITICAL

The `chat.py` command imports heavy modules at the top level:

```python
# victor/ui/commands/chat.py (lines 17-51)
from victor.agent.orchestrator import AgentOrchestrator  # ← 2,000+ lines
from victor.config.settings import load_settings  # ← Loads all config
from victor.framework.shim import FrameworkShim  # ← Loads framework
from victor.core.verticals import get_vertical, list_verticals  # ← Loads verticals
from victor.ui.commands.utils import (...)  # ← Imports AgentOrchestrator AGAIN
from victor.workflows import (...)  # ← Loads workflows
from victor.workflows.visualization import (...)  # ← Loads visualization
```

**Impact:** These imports trigger a cascade that loads:
- All 34 tool modules
- All 24 provider modules
- All 23 coordinators
- All 9 vertical packages
- Entire framework (agent, state, workflows, etc.)

**Result:** 685 modules imported at startup

---

### 2. **Duplicate Import Chain** 🔴 HIGH

```python
# chat.py imports:
from victor.agent.orchestrator import AgentOrchestrator

# utils.py imports:
from victor.agent.orchestrator import AgentOrchestrator

# chat.py imports utils:
from victor.ui.commands.utils import (...)
```

**Impact:** `AgentOrchestrator` is loaded multiple times

---

### 3. **No Lazy Loading** 🔴 HIGH

Everything is imported immediately, even if not needed:
- Workflow modules (only needed for `--workflow` flag)
- Visualization (only needed for `--visualize` flag)
- Code search index (only needed when using code search)
- RL profile suggestion (only needed for RL)

**Impact:** Features the user might not use are loaded at startup

---

## Fixes (Ordered by Impact)

### Fix 1: Move Heavy Imports Inside Functions ✅ CRITICAL

**Effort:** 1-2 hours  
**Impact:** 4-5 seconds improvement

**Strategy:** Move heavy imports inside functions that use them

```python
# BEFORE (victor/ui/commands/chat.py)
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings
from victor.framework.shim import FrameworkShim

@chat_app.command()
def chat(...):
    settings = load_settings()  # ← Already loaded at import
    orchestrator = AgentOrchestrator(...)  # ← Already loaded at import
```

```python
# AFTER (victor/ui/commands/chat.py)
# Keep lightweight imports at top level
from victor.core.errors import (
    ConfigurationError,
    ProviderError,
)

# Lazy import helpers
def _load_orchestrator():
    from victor.agent.orchestrator import AgentOrchestrator
    return AgentOrchestrator

def _load_settings():
    from victor.config.settings import load_settings
    return load_settings()

@chat_app.command()
def chat(...):
    # Import only when command is run
    settings = _load_settings()  # ← Loaded now, not at import
    AgentOrchestrator = _load_orchestrator()
    orchestrator = AgentOrchestrator(...)
```

---

### Fix 2: Use TYPE_CHECKING for Type Hints ✅ HIGH

**Effort:** 30 minutes  
**Impact:** 0.5-1 second improvement

**Strategy:** Use `TYPE_CHECKING` to avoid imports for type hints

```python
# BEFORE
from victor.agent.orchestrator import AgentOrchestrator

def chat(agent: AgentOrchestrator):  # ← Needs AgentOrchestrator at import
    pass
```

```python
# AFTER
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

def chat(agent: "AgentOrchestrator"):  # ← String, no import needed
    pass
```

---

### Fix 3: Remove Duplicate Imports ✅ HIGH

**Effort:** 15 minutes  
**Impact:** 0.5-1 second improvement

**Strategy:** Use TYPE_CHECKING in utils.py

```python
# BEFORE (victor/ui/commands/utils.py)
from victor.agent.orchestrator import AgentOrchestrator  # ← Line 16

_current_agent: Optional[AgentOrchestrator] = None  # ← Line 38
```

```python
# AFTER (victor/ui/commands/utils.py)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

_current_agent: Optional["AgentOrchestrator"] = None  # ← No import needed
```

---

### Fix 4: Lazy Load Feature-Specific Modules ✅ MEDIUM

**Effort:** 1 hour  
**Impact:** 0.5-1 second improvement

**Strategy:** Import feature modules only when needed

```python
# BEFORE
from victor.ui.commands.utils import (
    preload_semantic_index,  # ← Only for code search
    check_codebase_index,    # ← Only for code search
    get_rl_profile_suggestion,  # ← Only for RL
)

@chat_app.command()
def chat(...):
    if use_code_search:
        preload_semantic_index(...)  # ← Loaded at import even if not used
```

```python
# AFTER
@chat_app.command()
def chat(...):
    if use_code_search:
        from victor.ui.commands.utils import preload_semantic_index
        preload_semantic_index(...)  # ← Loaded only when needed
```

---

### Fix 5: Use __getattr__ for Module Exports ✅ MEDIUM

**Effort:** 2 hours  
**Impact:** 0.3-0.5 second improvement

**Strategy:** Use `__getattr__` for lazy module loading

```python
# BEFORE (victor/agent/__init__.py)
from .orchestrator import AgentOrchestrator
from .coordinators.chat_coordinator import ChatCoordinator
# ... 20 more imports
```

```python
# AFTER (victor/agent/__init__.py)
def __getattr__(name):
    if name == "AgentOrchestrator":
        from .orchestrator import AgentOrchestrator
        return AgentOrchestrator
    if name == "ChatCoordinator":
        from .coordinators.chat_coordinator import ChatCoordinator
        return ChatCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours, 4-5 seconds improvement)

1. **Move heavy imports inside functions** (Fix #1)
   - Edit `victor/ui/commands/chat.py`
   - Create lazy import helper functions
   - Move heavy imports inside command handlers

2. **Use TYPE_CHECKING for type hints** (Fix #2)
   - Edit `victor/ui/commands/chat.py`
   - Edit `victor/ui/commands/utils.py`
   - Change type annotations to strings

3. **Remove duplicate imports** (Fix #3)
   - Edit `victor/ui/commands/utils.py`
   - Use TYPE_CHECKING for AgentOrchestrator type hint

### Phase 2: Additional Optimizations (1-2 hours, 0.5-1 second improvement)

4. **Lazy load feature-specific modules** (Fix #4)
   - Move workflow imports inside workflow commands
   - Move visualization imports inside visualize commands
   - Move code search imports inside code search functions

5. **Use __getattr__ for exports** (Fix #5)
   - Edit `victor/agent/__init__.py`
   - Edit `victor/framework/__init__.py`
   - Edit other heavy __init__.py files

---

## Expected Results

**After Phase 1 (Quick Wins):**
- Startup time: 7.46s → **2-3 seconds** (4-5 second improvement)
- Modules imported: 685 → **200-300** (60% reduction)
- User experience: Much better

**After Phase 2 (Additional):**
- Startup time: 2-3s → **<1 second** (2-3 second improvement)
- Modules imported: 200-300 → **<50** (80% reduction)
- User experience: Excellent!

**Total Improvement:**
- Startup time: **7.46s → <1s** (87% improvement)
- Modules imported: **685 → <50** (93% reduction)

---

## Testing

After fixes, verify performance:

```bash
# Test startup time
time python -c "from victor.ui.commands import chat"
# Should be <1 second, not 7.46 seconds

# Test module count
python -c "
from victor.ui.commands import chat
import sys
victor_modules = [m for m in sys.modules if m.startswith('victor')]
print(f'Victor modules: {len(victor_modules)}')
# Should be <50, not 685
"

# Test CLI startup
time victor chat --help
# Should feel snappy, not sluggish
```

---

## Success Criteria

✅ Startup time <1 second  
✅ Modules imported <50  
✅ CLI feels responsive  
✅ No functional changes  
✅ All tests passing  
✅ No breaking changes

---

## Priority

**🔴 CRITICAL** - This is a major user-facing performance issue

**Recommendation:** Implement Phase 1 fixes immediately (1-2 hours work for 4-5 second improvement)

**User Impact:** High - Every user experiences slow CLI startup

**Risk:** Low - Lazy imports are a well-established pattern with minimal risk

---

**Status:** Ready to implement  
**Estimated Effort:** 2-4 hours for full optimization  
**Expected Impact:** 87% startup time improvement (7.46s → <1s)
