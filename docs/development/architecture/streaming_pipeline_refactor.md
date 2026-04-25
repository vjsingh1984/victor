# Streaming Chat Pipeline Architecture

> Project-graph analysis (Feb 2025) showed that `_stream_chat_impl` inside the
> old coordinator path still orchestrated every phase of streaming chat. The
> orchestration has since moved into `victor/agent/streaming/pipeline.py`, but
> the live runtime entry point is now the service-owned streaming runtime rather
> than `ChatCoordinator.stream_chat()`.

## 1. Responsibilities

| Phase | Responsibilities formerly inside `_stream_chat_impl` | Extracted component |
| --- | --- | --- |
| Session bootstrap | Create `StreamingChatContext`, emit requirement events, reset dedup state | `victor.agent.streaming.context` |
| Iteration loop | Enforce limits, request provider responses, feed tool calls | `StreamingChatHandler`, `ToolExecutionHandler` |
| Intent handling | Run classifier, apply continuation overrides | `IntentClassificationHandler`, `ContinuationHandler` |
| Recovery & fallbacks | Delegate to `RecoveryCoordinator`, `response_completer` | `victor.agent.recovery_coordinator`, `response_completer` |
| Metrics & observability | Emit streaming metrics, cumulative token usage | `StreamingController`, `StreamingCoordinator` |

By moving `_stream_chat_impl` into `StreamingChatPipeline`, the fan-out now
lives inside a module purpose-built for streaming orchestration rather than
inside a coordinator facade.

## 2. Canonical Architecture

```text
ChatService.stream_chat()
└── ServiceStreamingRuntime.stream_chat()
    └── StreamingChatPipeline(runtime_owner=ServiceStreamingRuntime)
        ├── setup() -> StreamingChatContext + requirement extraction
        ├── iterate() async generator
        │     • delegates provider streaming + tool execution
        │     • consults continuation + intent handlers
        └── finalize() -> completion fallback + metrics aggregation
```

`StreamingChatPipeline.run(user_message)` remains the single entry point for
streaming sessions, but the canonical owner is now
`victor.agent.services.chat_stream_runtime.ServiceStreamingRuntime`.
`ChatCoordinator.stream_chat()` only survives as a compatibility shim that
forwards to:
1. bound `ChatService.stream_chat()`
2. `_get_service_streaming_runtime()`
3. legacy `_stream_chat_runtime` hook

## 3. Implemented Changes

1. **Pipeline landing** – `victor/agent/streaming/pipeline.py` contains the
   migrated implementation and reuses dedicated helper modules for intent,
   continuation, tool execution, and recovery.
2. **Service-owned runtime** –
   `victor.agent.services.chat_stream_runtime.ServiceStreamingRuntime` is the
   canonical owner of pipeline construction and invocation.
3. **Factory exposure** – orchestrator/runtime builders expose
   `create_streaming_chat_pipeline(...)` and
   `create_service_streaming_runtime(...)` so canonical wiring stays out of the
   deprecated coordinator shims.
4. **Compatibility shim retention** – `ChatCoordinator.stream_chat()` no longer
   owns pipeline creation. It forwards to the service/runtime surfaces and only
   falls back to the legacy hook for older integrations.

## 4. Testing Strategy

1. **Pipeline unit tests**
   - Mock/spy runtime-owner helpers to prove the pipeline exercises the
     expected phases (pre-checks, continuation, tool execution, recovery).
   - Simulate error and cancellation paths to guard the retry logic.
2. **Compatibility tests**
   - Keep explicit coverage that deprecated shims prefer `ChatService`, then
     `ServiceStreamingRuntime`, and only then the legacy hook.
3. **Integration coverage**
   - Maintain `tests/unit/agent/test_orchestrator_core.py` and service
     delegation suites so the service-owned runtime remains the canonical path.
   - Maintain the streaming CLI/TUI integration tests to ensure real providers
     and tool executions still behave identically.

## 5. Follow-up Checklist

- [x] Create `StreamingChatPipeline` module.
- [x] Move canonical streaming ownership to `ServiceStreamingRuntime`.
- [x] Reduce `ChatCoordinator.stream_chat` to a compatibility forwarding layer.
- [x] Remove `_stream_chat_impl`.
- [ ] Add dedicated unit tests for `StreamingChatPipeline` behaviours.
- [ ] Add CI checks that fail when streaming fan-out exceeds agreed thresholds.
