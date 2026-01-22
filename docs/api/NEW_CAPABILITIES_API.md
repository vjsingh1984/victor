# New Capabilities API Reference

Complete API reference for Victor AI's new agentic capabilities.

## Table of Contents

- [Hierarchical Planning APIs](#hierarchical-planning-apis)
- [Memory APIs](#memory-apis)
- [Skill APIs](#skill-apis)
- [Multimodal APIs](#multimodal-apis)
- [Persona APIs](#persona-apis)
- [Performance APIs](#performance-apis)
- [Configuration APIs](#configuration-apis)
- [Type Definitions](#type-definitions)

## Hierarchical Planning APIs

### AutonomousPlanner

```python
class AutonomousPlanner:
    """Autonomous planning for goal-oriented execution."""

    def __init__(self, orchestrator: AgentOrchestrator):
        """Initialize planner.

        Args:
            orchestrator: Agent orchestrator instance
        """
        ...

    async def plan_for_goal(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_steps: int = 10,
        max_iterations: int = 3
    ) -> ExecutionPlan:
        """Generate execution plan for goal.

        Args:
            goal: High-level goal to achieve
            context: Additional context for planning
            max_steps: Maximum number of steps in plan
            max_iterations: Maximum planning iterations

        Returns:
            ExecutionPlan: Generated execution plan

        Raises:
            PlanningError: If planning fails
        """
        ...

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        auto_approve: bool = False,
        approval_callback: Optional[Callable[[PlanStep], bool]] = None,
        on_progress: Optional[Callable[[PlanStep], None]] = None
    ) -> PlanResult:
        """Execute execution plan.

        Args:
            plan: Execution plan to execute
            auto_approve: Auto-approve all steps
            approval_callback: Optional callback for step approval
            on_progress: Optional callback for progress updates

        Returns:
            PlanResult: Execution result

        Raises:
            ExecutionError: If execution fails
        """
        ...

    def validate_plan(self, plan: ExecutionPlan) -> ValidationResult:
        """Validate execution plan.

        Args:
            plan: Execution plan to validate

        Returns:
            ValidationResult: Validation result
        """
        ...
```

### ExecutionPlan

```python
@dataclass
class ExecutionPlan:
    """Execution plan with steps and dependencies."""

    goal: str
    """High-level goal of the plan"""

    steps: List[PlanStep]
    """Ordered list of plan steps"""

    created_at: datetime
    """When plan was created"""

    def to_markdown(self) -> str:
        """Convert plan to markdown format.

        Returns:
            Markdown representation of plan
        """
        ...

    def to_json(self) -> str:
        """Convert plan to JSON.

        Returns:
            JSON representation of plan
        """
        ...

    @property
    def total_steps(self) -> int:
        """Total number of steps."""
        ...

    @property
    def estimated_tool_calls(self) -> int:
        """Estimated total tool calls."""
        ...
```

### PlanStep

```python
@dataclass
class PlanStep:
    """Single step in execution plan."""

    id: str
    """Unique step identifier"""

    description: str
    """Step description"""

    step_type: StepType
    """Type of work (RESEARCH, PLANNING, IMPLEMENTATION, TESTING, REVIEW, DEPLOYMENT)"""

    depends_on: List[str]
    """List of step IDs this step depends on"""

    estimated_tool_calls: int
    """Estimated number of tool calls"""

    requires_approval: bool
    """Whether step requires user approval"""

    sub_agent_role: Optional[str]
    """Optional sub-agent role for delegation"""

    context: Dict[str, Any]
    """Additional context for the step"""

    status: StepStatus
    """Current status (PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED, BLOCKED)"""

    result: Optional[StepResult]
    """Result after execution (if completed)"""

    def is_ready(self, completed_steps: Set[str]) -> bool:
        """Check if step is ready to execute.

        Args:
            completed_steps: Set of completed step IDs

        Returns:
            True if ready to execute
        """
        ...
```

## Memory APIs

### EpisodicMemory

```python
class EpisodicMemory:
    """Episodic memory for storing and retrieving experiences."""

    def __init__(
        self,
        max_episodes: int = 1000,
        recall_threshold: float = 0.3,
        consolidation_interval: int = 100
    ):
        """Initialize episodic memory.

        Args:
            max_episodes: Maximum number of episodes to store
            recall_threshold: Minimum similarity for recall
            consolidation_interval: Episodes between consolidations
        """
        ...

    async def store_episode(
        self,
        inputs: Dict[str, Any],
        actions: List[str],
        outcomes: Dict[str, Any],
        rewards: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an episode.

        Args:
            inputs: Initial inputs/context
            actions: List of actions taken
            outcomes: Results and outcomes
            rewards: Reward signal
            context: Additional context

        Returns:
            Episode ID
        """
        ...

    async def recall_relevant(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.3
    ) -> List[Episode]:
        """Recall relevant episodes by semantic similarity.

        Args:
            query: Query string
            k: Number of episodes to recall
            threshold: Minimum similarity threshold

        Returns:
            List of relevant episodes
        """
        ...

    async def recall_recent(
        self,
        n: int = 10,
        task_type: Optional[str] = None
    ) -> List[Episode]:
        """Recall recent episodes.

        Args:
            n: Number of recent episodes
            task_type: Optional task type filter

        Returns:
            List of recent episodes
        """
        ...

    async def recall_by_outcome(
        self,
        outcome_key: str,
        outcome_value: Any,
        n: int = 10
    ) -> List[Episode]:
        """Recall episodes by outcome.

        Args:
            outcome_key: Outcome key to filter on
            outcome_value: Value to match
            n: Number of episodes to recall

        Returns:
            List of matching episodes
        """
        ...

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Statistics dictionary
        """
        ...

    async def clear_old_episodes(self, days: int = 30) -> int:
        """Clear old episodes.

        Args:
            days: Delete episodes older than this many days

        Returns:
            Number of episodes cleared
        """
        ...

    async def consolidate_into_knowledge(
        self,
        min_episodes: int = 5,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Consolidate episodes into semantic memory.

        Args:
            min_episodes: Minimum similar episodes for consolidation
            similarity_threshold: Similarity threshold for consolidation

        Returns:
            Consolidation result with counts
        """
        ...
```

### SemanticMemory

```python
class SemanticMemory:
    """Semantic memory for storing and querying knowledge."""

    def __init__(
        self,
        max_facts: int = 5000,
        query_threshold: float = 0.25,
        link_threshold: float = 0.4
    ):
        """Initialize semantic memory.

        Args:
            max_facts: Maximum number of facts to store
            query_threshold: Minimum similarity for queries
            link_threshold: Minimum similarity for linking facts
        """
        ...

    async def store_knowledge(
        self,
        fact: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        source: Optional[str] = None
    ) -> str:
        """Store knowledge fact.

        Args:
            fact: Fact to store
            metadata: Optional metadata
            confidence: Confidence level (0.0 - 1.0)
            source: Optional source

        Returns:
            Fact ID
        """
        ...

    async def query_knowledge(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.25
    ) -> List[Fact]:
        """Query knowledge by semantic similarity.

        Args:
            query: Query string
            k: Number of facts to return
            threshold: Minimum similarity threshold

        Returns:
            List of relevant facts
        """
        ...

    async def link_facts(
        self,
        fact_id_1: str,
        fact_id_2: str,
        link_type: str = "related",
        strength: float = 0.9,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Link two facts.

        Args:
            fact_id_1: First fact ID
            fact_id_2: Second fact ID
            link_type: Type of link
            strength: Link strength (0.0 - 1.0)
            metadata: Optional metadata

        Returns:
            Link ID
        """
        ...

    async def get_related_facts(
        self,
        fact_id: str,
        max_depth: int = 2,
        min_strength: float = 0.5
    ) -> List[Fact]:
        """Get facts related to given fact.

        Args:
            fact_id: Starting fact ID
            max_depth: Maximum traversal depth
            min_strength: Minimum link strength

        Returns:
            List of related facts
        """
        ...

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Statistics dictionary
        """
        ...
```

## Skill APIs

### SkillDiscoveryEngine

```python
class SkillDiscoveryEngine:
    """Dynamic skill discovery and composition."""

    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        event_bus: Optional[IEventBackend] = None,
        max_tools: int = 20,
        min_compatibility: float = 0.5
    ):
        """Initialize skill discovery engine.

        Args:
            tool_registry: Tool registry
            event_bus: Optional event bus
            max_tools: Maximum tools to discover
            min_compatibility: Minimum compatibility score
        """
        ...

    async def discover_tools(
        self,
        context: str,
        max_tools: int = 10,
        include_mcp: bool = True
    ) -> List[AvailableTool]:
        """Discover tools for context.

        Args:
            context: Task context description
            max_tools: Maximum tools to return
            include_mcp: Include MCP tools

        Returns:
            List of discovered tools
        """
        ...

    async def match_tools_to_task(
        self,
        task: str,
        tools: Optional[List[AvailableTool]] = None,
        min_compatibility: float = 0.5,
        max_tools: int = 5
    ) -> List[AvailableTool]:
        """Match tools to task.

        Args:
            task: Task description
            tools: Optional list of tools (default: all tools)
            min_compatibility: Minimum compatibility threshold
            max_tools: Maximum tools to return

        Returns:
            List of matched tools
        """
        ...

    async def compose_skill(
        self,
        name: str,
        tools: List[AvailableTool],
        description: str,
        execution_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Skill:
        """Compose skill from tools.

        Args:
            name: Skill name
            tools: List of tools
            description: Skill description
            execution_fn: Optional custom execution function
            metadata: Optional metadata

        Returns:
            Composed skill
        """
        ...

    async def register_skill(
        self,
        skill: Skill,
        overwrite: bool = False
    ) -> None:
        """Register skill.

        Args:
            skill: Skill to register
            overwrite: Overwrite if exists

        Raises:
            RegistrationError: If skill exists and overwrite=False
        """
        ...

    async def analyze_compatibility(
        self,
        tool_1: AvailableTool,
        tool_2: AvailableTool
    ) -> Dict[str, Any]:
        """Analyze compatibility between tools.

        Args:
            tool_1: First tool
            tool_2: Second tool

        Returns:
            Compatibility analysis with score and reasoning
        """
        ...
```

### SkillChainer

```python
class SkillChainer:
    """Skill chaining for multi-step workflows."""

    def __init__(
        self,
        max_chain_length: int = 10,
        validation_enabled: bool = True,
        parallel_enabled: bool = True
    ):
        """Initialize skill chainer.

        Args:
            max_chain_length: Maximum chain length
            validation_enabled: Enable chain validation
            parallel_enabled: Enable parallel execution
        """
        ...

    async def plan_chain(
        self,
        goal: str,
        skills: List[Skill],
        max_length: int = 5,
        execution_strategy: str = "sequential"
    ) -> SkillChain:
        """Plan skill chain.

        Args:
            goal: Goal to achieve
            skills: Available skills
            max_length: Maximum chain length
            execution_strategy: Strategy ("sequential", "parallel", "adaptive")

        Returns:
            Planned skill chain
        """
        ...

    async def execute_chain(
        self,
        chain: SkillChain,
        context: Dict[str, Any],
        on_failure: str = "stop",
        on_progress: Optional[Callable] = None
    ) -> ChainResult:
        """Execute skill chain.

        Args:
            chain: Skill chain to execute
            context: Execution context
            on_failure: Failure strategy ("stop", "skip", "continue")
            on_progress: Optional progress callback

        Returns:
            Chain execution result
        """
        ...

    def validate_chain(self, chain: SkillChain) -> ValidationResult:
        """Validate skill chain.

        Args:
            chain: Skill chain to validate

        Returns:
            Validation result
        """
        ...
```

## Multimodal APIs

### VisionAgent

```python
class VisionAgent:
    """Vision processing agent."""

    def __init__(self, orchestrator: AgentOrchestrator):
        """Initialize vision agent.

        Args:
            orchestrator: Agent orchestrator
        """
        ...

    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        detail_level: str = "auto"
    ) -> ImageAnalysis:
        """Analyze image.

        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            detail_level: Detail level ("low", "auto", "high")

        Returns:
            Image analysis result
        """
        ...

    async def extract_chart_data(
        self,
        image_path: str,
        chart_type: str = "auto"
    ) -> ChartData:
        """Extract data from chart.

        Args:
            image_path: Path to chart image
            chart_type: Chart type ("bar", "line", "pie", "scatter", "auto")

        Returns:
            Extracted chart data
        """
        ...

    async def extract_text(
        self,
        image_path: str,
        language: str = "auto"
    ) -> TextExtraction:
        """Extract text from image (OCR).

        Args:
            image_path: Path to image
            language: Language code ("auto", "en", "es", etc.)

        Returns:
            Extracted text with confidence
        """
        ...

    async def analyze_diagram(
        self,
        image_path: str,
        diagram_type: str = "auto"
    ) -> DiagramAnalysis:
        """Analyze diagram.

        Args:
            image_path: Path to diagram
            diagram_type: Diagram type ("flowchart", "sequence", "architecture", "auto")

        Returns:
            Diagram analysis
        """
        ...
```

### AudioAgent

```python
class AudioAgent:
    """Audio processing agent."""

    def __init__(self, orchestrator: AgentOrchestrator):
        """Initialize audio agent.

        Args:
            orchestrator: Agent orchestrator
        """
        ...

    async def transcribe_audio(
        self,
        audio_path: str,
        language: str = "auto",
        quality: str = "standard"
    ) -> Transcript:
        """Transcribe audio.

        Args:
            audio_path: Path to audio file
            language: Language code ("auto", "en", "es", etc.)
            quality: Quality level ("standard", "high")

        Returns:
            Transcript with timing and confidence
        """
        ...

    async def identify_speakers(
        self,
        audio_path: str,
        num_speakers: Union[int, str] = "auto"
    ) -> Diarization:
        """Identify speakers in audio.

        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers or "auto"

        Returns:
            Speaker diarization
        """
        ...

    async def transcribe_meeting(
        self,
        audio_path: str,
        include_summary: bool = True,
        include_action_items: bool = True
    ) -> MeetingTranscript:
        """Transcribe meeting with analysis.

        Args:
            audio_path: Path to audio file
            include_summary: Include meeting summary
            include_action_items: Extract action items

        Returns:
            Meeting transcript with summary and action items
        """
        ...
```

## Persona APIs

### PersonaManager

```python
class PersonaManager:
    """Persona management."""

    def __init__(self, orchestrator: AgentOrchestrator):
        """Initialize persona manager.

        Args:
            orchestrator: Agent orchestrator
        """
        ...

    async def set_persona(
        self,
        name: str
    ) -> None:
        """Set active persona.

        Args:
            name: Persona name

        Raises:
            PersonaNotFoundError: If persona doesn't exist
        """
        ...

    async def create_persona(
        self,
        name: str,
        description: str,
        system_prompt: str,
        capabilities: List[str],
        triggers: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create custom persona.

        Args:
            name: Persona name
            description: Persona description
            system_prompt: System prompt for persona
            capabilities: List of capabilities
            triggers: Optional automatic triggers
            metadata: Optional metadata

        Raises:
            PersonaExistsError: If persona already exists
        """
        ...

    async def suggest_persona(
        self,
        context: str,
        task_type: Optional[str] = None
    ) -> PersonaSuggestion:
        """Suggest persona for context.

        Args:
            context: Task context
            task_type: Optional task type

        Returns:
            Persona suggestion with confidence
        """
        ...

    def list_personas(self) -> List[PersonaInfo]:
        """List available personas.

        Returns:
            List of persona information
        """
        ...

    async def delete_persona(self, name: str) -> None:
        """Delete persona.

        Args:
            name: Persona name

        Raises:
            PersonaNotFoundError: If persona doesn't exist
        """
        ...

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get persona usage statistics.

        Returns:
            Usage statistics
        """
        ...
```

## Performance APIs

### MetricsRegistry

```python
class MetricsRegistry:
    """Metrics collection and registry."""

    def create_counter(
        self,
        name: str,
        description: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Counter:
        """Create counter metric.

        Args:
            name: Metric name
            description: Metric description
            tags: Optional tags

        Returns:
            Counter instance
        """
        ...

    def create_gauge(
        self,
        name: str,
        description: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Gauge:
        """Create gauge metric.

        Args:
            name: Metric name
            description: Metric description
            tags: Optional tags

        Returns:
            Gauge instance
        """
        ...

    def create_histogram(
        self,
        name: str,
        description: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Histogram:
        """Create histogram metric.

        Args:
            name: Metric name
            description: Metric description
            tags: Optional tags

        Returns:
            Histogram instance
        """
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics.

        Returns:
            Dictionary of all metrics
        """
        ...
```

## Configuration APIs

### Settings

```python
class Settings(BaseSettings):
    """Victor AI configuration settings."""

    # Feature Flags
    enable_hierarchical_planning: bool = False
    enable_episodic_memory: bool = False
    enable_semantic_memory: bool = False
    enable_skill_discovery: bool = False
    enable_skill_chaining: bool = False
    enable_self_improvement: bool = False
    enable_rl_coordinator: bool = True

    # Hierarchical Planning
    hierarchical_planning_max_depth: int = 5
    hierarchical_planning_min_subtasks: int = 2
    hierarchical_planning_max_subtasks: int = 10

    # Episodic Memory
    episodic_memory_max_episodes: int = 1000
    episodic_memory_recall_threshold: float = 0.3
    episodic_memory_consolidation_interval: int = 100

    # Semantic Memory
    semantic_memory_max_facts: int = 5000
    semantic_memory_query_threshold: float = 0.25
    semantic_memory_link_threshold: float = 0.4

    # Skill Discovery
    skill_discovery_max_tools: int = 20
    skill_discovery_min_compatibility: float = 0.5
    skill_discovery_auto_composition: bool = True

    # Skill Chaining
    skill_chaining_max_chain_length: int = 10
    skill_chaining_validation_enabled: bool = True
    skill_chaining_parallel_enabled: bool = True

    # Proficiency Tracking
    proficiency_tracker_window_size: int = 100
    proficiency_tracker_decay_rate: float = 0.95
    proficiency_tracker_min_samples: int = 5

    # RL Coordinator
    rl_reward_shaping: str = "sparse"
    rl_policy_update_interval: int = 50
    rl_exploration_rate: float = 0.3

    # Performance
    lazy_loading_enabled: bool = False
    parallel_execution_enabled: bool = True
    max_parallel_workers: int = 4

    # Caching
    tool_selection_cache_enabled: bool = True
    tool_selection_cache_size: int = 500
    tool_selection_cache_ttl: int = 3600

    # Memory
    memory_limit_mb: int = 2000
    memory_warning_threshold: float = 0.8

    # Provider
    vision_provider: str = "anthropic"
    vision_model: str = "claude-3-opus"
    audio_provider: str = "openai"
    audio_model: str = "whisper-1"

    class Config:
        env_prefix = "VICTOR_"
        env_file = ".env"
```

## Type Definitions

### StepType

```python
class StepType(Enum):
    """Type of plan step."""

    RESEARCH = "research"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    REVIEW = "review"
    DEPLOYMENT = "deployment"
```

### StepStatus

```python
class StepStatus(Enum):
    """Status of plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
```

### Episode

```python
@dataclass
class Episode:
    """Episodic memory entry."""

    id: str
    timestamp: datetime
    inputs: Dict[str, Any]
    actions: List[str]
    outcomes: Dict[str, Any]
    rewards: float
    context: Dict[str, Any]
    embedding: Optional[np.ndarray]
```

### Fact

```python
@dataclass
class Fact:
    """Semantic memory entry."""

    id: str
    fact: str
    metadata: Dict[str, Any]
    confidence: float
    source: Optional[str]
    created_at: datetime
    embedding: Optional[np.ndarray]
```

### Skill

```python
@dataclass
class Skill:
    """Composed skill."""

    name: str
    tools: List[AvailableTool]
    description: str
    execution_fn: Optional[Callable]
    metadata: Dict[str, Any]
```

### Persona

```python
@dataclass
class Persona:
    """Agent persona."""

    name: str
    description: str
    system_prompt: str
    capabilities: List[str]
    triggers: List[Dict]
    metadata: Dict[str, Any]
```

## Migration Notes

### Deprecated APIs

The following APIs are deprecated and should be replaced:

| Deprecated | Replacement |
|------------|-------------|
| `planner.plan()` | `planner.plan_for_goal()` |
| `memory.store()` | `memory.store_episode()` |
| `tool_composer` | `skill_discovery` |

### Breaking Changes

- All planning methods are now async
- Memory system split into episodic and semantic
- Skills renamed from tool composition
- Personas require explicit enabling

## Additional Resources

- [User Guide](../USER_GUIDE.md)
- [Hierarchical Planning Guide](../guides/HIERARCHICAL_PLANNING.md)
- [Enhanced Memory Guide](../guides/ENHANCED_MEMORY.md)
- [Dynamic Skills Guide](../guides/DYNAMIC_SKILLS.md)
- [Multimodal Capabilities Guide](../guides/MULTIMODAL_CAPABILITIES.md)
- [Dynamic Personas Guide](../guides/DYNAMIC_PERSONAS.md)
- [Performance Tuning Guide](../guides/PERFORMANCE_TUNING.md)
