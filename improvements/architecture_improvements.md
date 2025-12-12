# Victor Architecture Improvements

## Current Architecture Issues

### 1. **God Object Anti-Pattern**
- `AgentOrchestrator` has 2000+ lines and 15+ responsibilities
- Violates Single Responsibility Principle
- Hard to test, debug, and maintain

### 2. **Complex Inheritance Hierarchies**
- Multiple layers of abstraction without clear benefit
- `BaseToolCallingAdapter` → `OllamaAdapter` → `OllamaNativeAdapter`
- Makes debugging difficult

### 3. **Circular Dependencies**
- Components depend on each other in complex ways
- Hard to unit test individual components
- Tight coupling throughout

### 4. **Configuration Complexity**
- 8+ different config files for overlapping concerns
- No clear precedence rules
- Hard to understand which setting applies when

## Proposed Architecture

### 1. **Hexagonal Architecture (Ports & Adapters)**

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Core                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Chat Engine   │  │  Tool Executor  │  │ Loop Detector│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Task Classifier │  │ Progress Tracker│  │ Error Handler│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │   Provider  │     │    Tools    │     │   Storage   │
    │   Adapter   │     │   Adapter   │     │   Adapter   │
    └─────────────┘     └─────────────┘     └─────────────┘
           │                    │                    │
    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │   Ollama    │     │ File System │     │   SQLite    │
    │  Anthropic  │     │    Git      │     │  LanceDB    │
    │   OpenAI    │     │    Web      │     │   Memory    │
    └─────────────┘     └─────────────┘     └─────────────┘
```

### 2. **Clean Component Separation**

#### Core Components (Application Layer)
```python
class ChatEngine:
    """Orchestrates conversation flow without tool/provider details."""
    def chat(self, message: str) -> Response: pass

class ToolExecutor:
    """Executes tools with retry, caching, error handling."""
    def execute(self, tool: ToolCall) -> ToolResult: pass

class LoopDetector:
    """Detects and prevents infinite loops."""
    def should_stop(self, history: List[Action]) -> bool: pass

class TaskClassifier:
    """Classifies user intent and sets appropriate limits."""
    def classify(self, message: str) -> TaskType: pass
```

#### Adapter Layer (Infrastructure)
```python
class ProviderAdapter:
    """Adapts external LLM providers to internal interface."""
    def chat(self, messages: List[Message]) -> Response: pass

class ToolAdapter:
    """Adapts external tools to internal interface."""
    def execute(self, name: str, args: dict) -> ToolResult: pass

class StorageAdapter:
    """Adapts external storage to internal interface."""
    def store(self, key: str, value: Any) -> None: pass
```

### 3. **Dependency Injection**

```python
class VictorContainer:
    """Dependency injection container."""
    
    def __init__(self):
        self._services = {}
        self._configure_services()
    
    def _configure_services(self):
        # Core services
        self.register(LoopDetector, SimpleLoopDetector())
        self.register(TaskClassifier, KeywordTaskClassifier())
        self.register(ToolExecutor, RetryingToolExecutor())
        
        # Adapters
        self.register(ProviderAdapter, self._create_provider_adapter())
        self.register(ToolAdapter, FileSystemToolAdapter())
        
        # Main engine
        self.register(ChatEngine, ChatEngine(
            tool_executor=self.get(ToolExecutor),
            loop_detector=self.get(LoopDetector),
            task_classifier=self.get(TaskClassifier)
        ))
    
    def get(self, service_type: type) -> Any:
        return self._services[service_type]
```

### 4. **Event-Driven Architecture**

```python
class EventBus:
    """Decouples components via events."""
    
    def __init__(self):
        self._handlers = defaultdict(list)
    
    def subscribe(self, event_type: type, handler: Callable):
        self._handlers[event_type].append(handler)
    
    def publish(self, event: Any):
        for handler in self._handlers[type(event)]:
            handler(event)

# Events
@dataclass
class ToolExecuted:
    tool_name: str
    args: dict
    result: ToolResult
    timestamp: float

@dataclass
class IterationCompleted:
    iteration: int
    tools_used: int
    content_length: int

# Handlers
class ProgressTracker:
    def handle_tool_executed(self, event: ToolExecuted):
        self.update_progress(event.tool_name, event.result)
    
    def handle_iteration_completed(self, event: IterationCompleted):
        self.check_for_loops(event)
```

## Implementation Strategy

### Phase 1: Extract Core Components (1 week)

1. **Extract ChatEngine**
   ```python
   class SimpleChatEngine:
       def __init__(self, provider: ProviderAdapter, tools: ToolAdapter):
           self.provider = provider
           self.tools = tools
           self.conversation = []
       
       async def chat(self, message: str) -> str:
           self.conversation.append({"role": "user", "content": message})
           
           for _ in range(20):  # Max iterations
               response = await self.provider.chat(self.conversation)
               
               if response.tool_calls:
                   results = await self.tools.execute_batch(response.tool_calls)
                   self.conversation.extend(self._format_tool_results(results))
                   continue
               
               return response.content
           
           return "Max iterations reached"
   ```

2. **Extract ToolExecutor**
   ```python
   class SimpleToolExecutor:
       def __init__(self, tool_registry: ToolRegistry):
           self.tools = tool_registry
       
       async def execute(self, tool_call: ToolCall) -> ToolResult:
           try:
               tool = self.tools.get(tool_call.name)
               result = await tool.execute(**tool_call.arguments)
               return ToolResult(success=True, data=result)
           except Exception as e:
               return ToolResult(success=False, error=str(e))
   ```

3. **Extract LoopDetector**
   ```python
   class SimpleLoopDetector:
       def __init__(self):
           self.history = deque(maxlen=10)
           self.iteration_count = 0
       
       def record_action(self, action: str):
           self.iteration_count += 1
           self.history.append(action)
       
       def should_stop(self) -> tuple[bool, str]:
           if self.iteration_count > 20:
               return True, "Max iterations"
           
           if len(self.history) >= 3:
               recent = list(self.history)[-3:]
               if len(set(recent)) == 1:
                   return True, f"Loop detected: {recent[0]}"
           
           return False, ""
   ```

### Phase 2: Create Adapters (3-5 days)

1. **Provider Adapters**
   ```python
   class OllamaProviderAdapter(ProviderAdapter):
       def __init__(self, base_url: str, model: str):
           self.client = OllamaClient(base_url)
           self.model = model
       
       async def chat(self, messages: List[Message]) -> Response:
           # Implement Ollama-specific logic
           pass
   ```

2. **Tool Adapters**
   ```python
   class FileSystemToolAdapter(ToolAdapter):
       async def execute(self, name: str, args: dict) -> ToolResult:
           if name == "read_file":
               return await self._read_file(args["path"])
           elif name == "write_file":
               return await self._write_file(args["path"], args["content"])
           # etc.
   ```

### Phase 3: Dependency Injection (2-3 days)

1. **Create Container**
2. **Wire Dependencies**
3. **Replace Orchestrator**

### Phase 4: Event System (2-3 days)

1. **Add EventBus**
2. **Convert to Event-Driven**
3. **Decouple Components**

## Benefits

### 1. **Testability**
- Each component can be unit tested in isolation
- Mock dependencies easily
- Clear interfaces

### 2. **Maintainability**
- Single responsibility per component
- Clear separation of concerns
- Easy to understand and modify

### 3. **Reliability**
- Simpler components = fewer bugs
- Clear error boundaries
- Easier debugging

### 4. **Performance**
- Remove unnecessary abstractions
- Direct component communication
- Optimized for common cases

## Migration Strategy

### 1. **Parallel Implementation**
- Keep existing orchestrator
- Build new architecture alongside
- Gradual migration

### 2. **Feature Flags**
- Use flags to switch between old/new
- Test new architecture thoroughly
- Rollback if issues

### 3. **Backward Compatibility**
- Maintain existing APIs
- Adapter pattern for migration
- Deprecate gradually

## Success Metrics

1. **Code Quality**
   - Reduce orchestrator from 2000 → 500 lines
   - Increase test coverage 30% → 80%
   - Reduce cyclomatic complexity

2. **Reliability**
   - Zero infinite loops
   - < 1% error rate
   - Consistent performance

3. **Developer Experience**
   - Faster debugging
   - Easier feature addition
   - Clear documentation