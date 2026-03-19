# Streaming Chat Pipeline Architecture

> Project-graph analysis (Feb 2025) showed that `_stream_chat_impl` inside
> `victor/agent/coordinators/chat_coordinator.py` still orchestrated every phase
> of streaming chat. Even after intent classification, continuation, and tool
> execution logic were extracted into dedicated modules, this monolithic method
> remained the hottest fan-out node in the agent layer. The canonical
> implementation now lives in `victor/agent/streaming/pipeline.py`, which
> coordinates all streaming phases via dedicated helpers.

## 1. Responsibilities

| Phase | Responsibilities formerly inside `_stream_chat_impl` | Extracted component |
| --- | --- | --- |
| Session bootstrap | Create `StreamingChatContext`, emit requirement events, reset dedup state | `victor.agent.streaming.context` |
| Iteration loop | Enforce limits, request provider responses, feed tool calls | `StreamingChatHandler`, `ToolExecutionHandler` |
| Intent handling | Run classifier, apply continuation overrides | `IntentClassificationHandler`, `ContinuationHandler` |
| Recovery & fallbacks | Delegate to `RecoveryCoordinator`, `response_completer` | `victor.agent.recovery_coordinator`, `response_completer` |
| Metrics & observability | Emit streaming metrics, cumulative token usage | `StreamingController`, `StreamingCoordinator` |

By moving `_stream_chat_impl` into `StreamingChatPipeline`, the fan-out now lives
inside a module purpose-built for streaming orchestration rather than inside the
coordinator façade.

## 2. Canonical Architecture

```
ChatCoordinator.stream_chat()
└── StreamingChatPipeline(coordinator)
    ├── setup() -> StreamingChatContext + requirement extraction
    ├── iterate() async generator
    │     • delegates provider streaming + tool execution
    │     • consults continuation + intent handlers
    └── finalize() -> completion fallback + metrics aggregation
```

`StreamingChatPipeline.run(user_message)` is the single entry point for streaming
sessions. It interacts with the coordinator through helper methods (context prep,
recovery, tool execution, etc.) so the coordinator keeps a slim façade.

## 3. Implemented Changes

1. **Pipeline landing** – `victor/agent/streaming/pipeline.py` now contains the
   migrated implementation. It reuses the existing helper modules but keeps the
   orchestration outside of `ChatCoordinator`.
2. **Coordinator delegation** – `ChatCoordinator.stream_chat` lazily instantiates
   the pipeline via `create_streaming_chat_pipeline(self)` and streams chunks by
   awaiting `pipeline.run(...)`. The legacy `_stream_chat_impl` method was
   deleted.
3. **Factory exposure** – `victor.agent.streaming.__all__` exports
   `create_streaming_chat_pipeline`, enabling orchestrator factories (or other
   components) to wire the canonical pipeline without importing internal files.
4. **Runtime injection** – the interaction runtime now asks
   `AgentOrchestratorFactory` to build the pipeline up front so `ChatCoordinator`
   receives a DI-managed pipeline rather than instantiating one ad hoc.

## 4. Testing Strategy

1. **Pipeline unit tests**
   - Mock/spy coordinator helpers to prove the pipeline exercises the expected
     phases (pre-checks, continuation, tool execution, recovery).
   - Simulate error and cancellation paths to guard the retry logic.
2. **Integration coverage**
   - Extend `tests/unit/agent/test_orchestrator_core.py` streaming suites to run
     through the pipeline (the legacy code path is gone, so assertions should now
     target the pipeline behaviour).
   - Maintain the streaming CLI/TUI integration tests to ensure real providers
     and tool executions still behave identically.

## 5. Follow-up Checklist

- [x] Create `StreamingChatPipeline` module.
- [x] Wire `ChatCoordinator.stream_chat` to the pipeline (no feature flag).
- [x] Remove `_stream_chat_impl`.
- [ ] Add dedicated unit tests for `StreamingChatPipeline` behaviours.
- [ ] Add CI checks that fail when streaming fan-out exceeds agreed thresholds.
