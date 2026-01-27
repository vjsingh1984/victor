# Victor API Server Comparison and Consolidation Plan

## Executive Summary

**Current State:**
- **3 API server implementations** exist in the codebase
- **`victor serve`** uses `VictorFastAPIServer` (port 8765)
- **Unified server** is a wrapper that composes `VictorFastAPIServer` + HITL + Workflows
- **Legacy server** (aiohttp) is still available but not recommended

**Recommendation:** Keep `VictorFastAPIServer` as the primary server, deprecate the others.

---

## Server Comparison Matrix

| Feature | FastAPI Server | Unified Server | AIOHTTP Server |
|---------|---------------|----------------|---------------|
| **File** | `fastapi_server.py` | `unified_server.py` | `server.py` |
| **Lines of Code** | 3,621 | 764 | 2,584 |
| **Status** | ✅ Active (default) | ⚠️ Wrapper/Orchestrator | ❌ Legacy |
| **Endpoints** | ~80+ comprehensive | 80+ (via mount) | ~60 basic |
| **Default Port** | 8765 | 8000 | 8765 |
| **Used By** | `victor serve` | Not used directly | Legacy option |
| **OpenAPI Docs** | ✅ Yes (/docs, /redoc) | ✅ Yes (inherits) | ❌ No |
| **HITL Support** | ✅ Built-in | ✅ Separate module | ❌ No |
| **WebSocket** | ✅ Streaming chat | ✅ (inherits) | ❌ No |
| **Maintenance** | ✅ Active | ⚠️ Confusing | ❌ Deprecated |

---

## Detailed Analysis

### 1. VictorFastAPIServer (`fastapi_server.py`)

**Purpose:** Comprehensive FastAPI-based HTTP API server for IDE integrations

**Size:** 3,621 lines (largest, most feature-complete)

**Key Features:**
- **Chat/Completions:** `/chat`, `/chat/stream`, `/completions`
- **Search:** `/search/semantic`, `/search/code`
- **LSP Services:** `/lsp/completions`, `/lsp/hover`, `/lsp/definition`, `/lsp/references`, `/lsp/diagnostics`
- **Git Integration:** `/git/status`, `/git/commit`, `/git/log`, `/git/diff`
- **Terminal:** `/terminal/suggest`, `/terminal/execute`
- **Workspace:** `/workspace/overview`, `/workspace/metrics`, `/workspace/security`, `/workspace/dependencies`
- **Configuration:** `/model/switch`, `/mode/switch`, `/models`, `/providers`, `/capabilities`
- **Tools:** `/tools`, `/tools/approve`, `/tools/pending`
- **Conversation:** `/conversation/reset`, `/conversation/export`, `/history`, `/undo`, `/redo`
- **Patching:** `/patch/apply`, `/patch/create`
- **Health/Status:** `/health`, `/status`
- **HITL:** Built-in support with optional auth token
- **WebSocket:** Real-time streaming chat
- **OpenAPI:** Auto-generated docs at `/docs` and `/redoc`

**Status:** ✅ **Primary server** (used by `victor serve`)

---

### 2. Unified Server (`unified_server.py`)

**Purpose:** Orchestrator that composes multiple services into one FastAPI app

**Size:** 764 lines (smallest, but just a wrapper)

**Architecture:**
```python
# The unified server DOES NOT implement endpoints directly
# Instead, it MOUNTS other servers:

FastAPI app
├── /api/v1 (mount) → VictorFastAPIServer (all main API)
├── /api/v1/hitl → HITL router (workflow approvals)
├── /api/v1/workflows → Workflow editor router
├── /ui → Landing page
├── /ui/hitl → HITL approval UI
└── /ui/workflow-editor → Workflow editor frontend
```

**Key Points:**
- **Not a standalone server** - it's a composition wrapper
- **Uses `VictorFastAPIServer` internally** for main API
- **Adds HITL and Workflow Editor routes separately**
- **Provides UI routes** (landing page, HITL UI, workflow editor)
- **Port confusion:** Defaults to 8000 (should be 8765)

**Status:** ⚠️ **Confusing wrapper** - should be removed or clearly labeled as orchestrator

---

### 3. VictorAPIServer (AIOHTTP, `server.py`)

**Purpose:** Legacy aiohttp-based server

**Size:** 2,584 lines

**Key Features:**
- Basic chat/completions endpoints
- Search endpoints
- Limited tool support
- No OpenAPI docs
- No WebSocket support
- No HITL support

**Status:** ❌ **Legacy/Deprecated** - kept for backward compatibility

---

## Port Confusion

### Current Issues:
1. **FastAPI Server:** Port 8765 (correct)
2. **Unified Server:** Port 8000 (wrong, causes test failures)
3. **AIOHTTP Server:** Port 8765 (legacy)

### Fix Applied:
- ✅ Updated all load tests to use port 8765
- ⚠️ Unified server still defaults to 8000 (should be 8765)

---

## Consolidation Recommendations

### Option 1: Keep Status Quo (Recommended ✅)

**Keep `VictorFastAPIServer` as primary, deprecate others.**

**Rationale:**
1. `VictorFastAPIServer` is already the default in `victor serve`
2. It has 3,621 lines of well-tested code with 80+ endpoints
3. It includes HITL support with optional auth
4. It provides OpenAPI docs automatically
5. It's actively maintained

**Changes Needed:**
- ✅ Keep `VictorFastAPIServer` as-is
- ⚠️ Rename `unified_server.py` → `unified_orchestrator.py` (clarify it's a wrapper)
- ⚠️ Update unified server to use port 8765
- ❌ Mark `server.py` (aiohttp) as deprecated
- ✅ Add documentation explaining the architecture

**Migration Path:**
```python
# Current (confusing)
from victor.integrations.api.unified_server import create_unified_server
app = create_unified_server(port=8000)  # Wrong port!

# Recommended (clear)
from victor.integrations.api.fastapi_server import VictorFastAPIServer
server = VictorFastAPIServer(port=8765)  # Correct port, direct usage
```

---

### Option 2: Full Consolidation (Not Recommended ❌)

**Merge everything into one monolithic server.**

**Why Not:**
- ❌ Would create a 5,000+ line monolith
- ❌ Lose separation of concerns (HITL, workflows, main API)
- ❌ Harder to maintain and test
- ❌ Breaking change for all users
- ✅ No actual benefit (composition is already working)

---

## Architecture Decision

### Recommended Architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     victor serve                           │
│                    (port 8765)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────┐
        │   VictorFastAPIServer        │
        │   (3,621 lines, 80+ endpoints)│
        └────────────────────────────────┘
                       │
        ┌──────────────┴───────────────┐
        │                              │
        ▼                              ▼
┌──────────────┐            ┌─────────────────┐
│ Main API     │            │ HITL Module    │
│ /api/v1/*    │            │ /api/v1/hitl/* │
│              │            │ (optional)      │
│ - Chat       │            │ - Approvals     │
│ - Search     │            │ - History       │
│ - Git        │            │ - Notifications │
│ - LSP        │            │ - UI            │
│ - Tools      │            │                 │
│ - Workspaces │            │                 │
│ - ...        │            │                 │
└──────────────┘            └─────────────────┘
```

### For Future: Workflow Editor (Optional)

```
                    ┌─────────────────┐
                    │  Workflow Editor │
                    │  /api/v1/workflow │
                    └─────────────────┘
```

---

## Action Plan

### Phase 1: Clarify (Immediate ✅)

1. **Rename unified_server.py** → **unified_orchestrator.py**
   - Clarifies it's a composition wrapper, not a standalone server
   - Prevents confusion with "main" server

2. **Add deprecation notice** to `server.py` (aiohttp):
   ```python
   """Legacy AIOHTTP server.

   .. deprecated::
       Use VictorFastAPIServer instead. This server is maintained
       only for backward compatibility and will be removed in v1.0.
   """
   ```

3. **Update documentation** in unified_orchestrator.py:
   ```python
   """Unified Server Orchestrator.

   This module composes multiple Victor API servers into a single
   FastAPI application for specific use cases (e.g., deploying HITL
   + Workflow Editor as one service).

   For most use cases, use VictorFastAPIServer directly instead:
       from victor.integrations.api.fastapi_server import VictorFastAPIServer
       server = VictorFastAPIServer(port=8765)

   Use this orchestrator only if you need:
   - HITL approval endpoints + Workflow Editor in one service
   - Custom UI routes (landing pages, workflow editor UI)
   """
   ```

### Phase 2: Fix Port (Immediate ✅)

1. **Update unified_server.py default port:** 8000 → 8765
   ```python
   def create_unified_server(
       port: int = 8765,  # Changed from 8000
       ...
   )
   ```

### Phase 3: Documentation (Next Sprint)

1. Add `SERVER_ARCHITECTURE.md` explaining:
   - VictorFastAPIServer is the main API server
   - Unified orchestrator is for special deployment scenarios
   - AIOHTTP server is legacy

2. Update `README.md` with:
   - How to start the server: `victor serve`
   - Default port: 8765
   - Available endpoints: http://localhost:8765/docs

### Phase 4: Deprecation (Future v1.0)

1. **Add deprecation warnings** to server.py (aiohttp):
   ```python
   warnings.warn(
       "VictorAPIServer (aiohttp) is deprecated. "
       "Use VictorFastAPIServer instead. "
       "This will be removed in Victor v1.0.",
       DeprecationWarning,
       stacklevel=2
   )
   ```

2. **Remove aiohttp server** in Victor v1.0

---

## Summary

### What to Keep:
- ✅ **VictorFastAPIServer** (3,621 lines, comprehensive)
- ✅ **Unified orchestrator** (rename to avoid confusion)
- ✅ **HITL module** (separate, composable)
- ✅ **Workflow editor** (separate, composable)

### What to Fix:
- ⚠️ **Rename:** `unified_server.py` → `unified_orchestrator.py`
- ⚠️ **Port:** Update unified server default: 8000 → 8765
- ⚠️ **Docs:** Clarify unified orchestrator is for special cases

### What to Deprecate:
- ❌ **VictorAPIServer** (aiohttp) - Legacy, remove in v1.0

### What NOT to Do:
- ❌ Don't merge everything into one file (5,000+ line monolith)
- ❌ Don't remove unified orchestrator (useful for composition)
- ❌ Don't change the default port (8765 is correct)

---

## Conclusion

**VictorFastAPIServer is the clear winner:**
- ✅ Most comprehensive (80+ endpoints)
- ✅ Actively maintained
- ✅ OpenAPI docs included
- ✅ HITL support built-in
- ✅ Default in `victor serve`
- ✅ Port 8765 (consistent)

**Unified orchestrator** should stay but be:
- Renamed to `unified_orchestrator.py`
- Clearly documented as a composition wrapper
- Used only for special deployment scenarios
- Port changed to 8765

**AIOHTTP server** should be:
- Marked as deprecated
- Removed in Victor v1.0
