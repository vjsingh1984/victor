# FastAPI Server Decomposition Plan

**Target**: `victor/integrations/api/fastapi_server.py` (3,587 LOC, 30+ endpoints)
**Goal**: Split into APIRouter modules under `victor/integrations/api/routes/`
**Priority**: D-02 (Tier 3 Design)

## Proposed Split

| Module | Tag | Endpoints | Est. LOC |
|--------|-----|-----------|----------|
| `system_routes.py` | System | /health, /status, /events/recent | ~100 |
| `chat_routes.py` | Chat | /chat, /chat/stream, /completions | ~200 |
| `search_routes.py` | Search | /search/semantic, /search/code | ~100 |
| `config_routes.py` | Configuration | /model/switch, /mode/switch, /models, /providers, /capabilities | ~200 |
| `tool_routes.py` | Tools | /tools, /tools/approve, /tools/pending | ~150 |
| `conversation_routes.py` | Conversation | /conversation/reset, /conversation/export, /undo, /redo, /history | ~150 |
| `git_routes.py` | Git | /git/status, /git/commit, /git/log, /git/diff, /patch/* | ~300 |
| `terminal_routes.py` | Terminal | /terminal/suggest, /terminal/execute | ~200 |
| `workspace_routes.py` | Workspace | /workspace/overview, /workspace/* | ~200 |
| `rl_routes.py` | RL | /rl/*, reinforcement learning endpoints | ~400 |
| `workflow_routes.py` | Workflow | /workflow/*, template execution | ~300 |
| `credential_routes.py` | Security | /credentials/* | ~100 |

## Migration Strategy

1. Create `victor/integrations/api/routes/` package
2. Extract each route group into an `APIRouter`
3. Keep `fastapi_server.py` as the app factory that includes all routers:
   ```python
   from victor.integrations.api.routes.chat_routes import router as chat_router
   app.include_router(chat_router)
   ```
4. Shared state (orchestrator, container) passed via FastAPI dependency injection

## Risk

- **Low**: FastAPI APIRouter is designed for this exact pattern
- **Testing**: API integration tests should continue to work unchanged
- **Shared state**: Orchestrator/container access via `Depends()` or app.state
