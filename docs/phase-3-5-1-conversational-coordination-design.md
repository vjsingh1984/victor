# Phase 3.5.1: Conversational Agent Coordination Design

**Author:** Claude (AI Assistant)
**Date:** 2026-01-10
**Status:** Design Draft
**Related:** Phase 3.1 (TeamNode), Phase 3.0-3.3 (Orchestration Prerequisites)
**Estimated LOC:** ~800 production, ~400 tests (69% reduction through code reuse)

**Note:** This design has been refined for SOLID compliance and maximum code reuse. See [SOLID Compliance](#solid-compliance) and [Code Reuse Strategy](#code-reuse-strategy) sections below.

---

## Table of Contents

1. [Overview](#overview)
2. [SOLID Compliance](#solid-compliance)
3. [Code Reuse Strategy](#code-reuse-strategy)
4. [Architecture Overview](#architecture-overview)
5. [Core Components](#core-components)
6. [Data Structures](#data-structures)
7. [Integration with StateGraph](#integration-with-stategraph)
8. [Conversation Protocols](#conversation-protocols)
9. [Implementation Strategy](#implementation-strategy)
10. [Code Examples](#code-examples)
11. [Testing Strategy](#testing-strategy)
12. [Performance Considerations](#performance-considerations)
13. [Future Enhancements](#future-enhancements)

---

## Overview

### Problem Statement

Victor's current multi-agent coordination is **highly structured and predictable**:
- TeamNode executes agents in predefined formations (SEQUENTIAL, PARALLEL, etc.)
- Communication follows predetermined patterns (manager→worker, pipeline stages)
- Agents cannot dynamically converse during execution
- No support for ad-hoc collaboration, debate, or consensus-building

While this works for many workflows, it lacks the **flexibility of conversational AI systems** like AutoGen, where agents can:
- Engage in multi-turn conversations
- Ask each other questions
- Debate alternatives
- Negotiate solutions
- Build consensus through dialogue

### Solution: Conversational Agent Coordination

This design introduces **dynamic conversational capabilities** while maintaining Victor's architectural principles:

1. **ConversationalNode**: New workflow node type that hosts multi-turn conversations
2. **Conversation Protocols**: Structured patterns (request-response, debate, consensus, negotiation)
3. **Message Router**: Intelligent routing based on agent roles, capabilities, and context
4. **Conversation History**: Full tracking and visualization of agent interactions
5. **Integration**: Seamless integration with existing StateGraph and TeamNode infrastructure

### Key Design Principles

- **Backward Compatible**: Enhances existing systems without breaking changes
- **SOLID Compliant**: Protocol-based interfaces, single-responsibility components
- **Observable**: Full EventBus integration for debugging and monitoring
- **Testable**: Mock-friendly design with comprehensive test coverage
- **Async-First**: All operations are async for performance
- **Type-Safe**: Rich dataclasses with proper type hints

### Comparison: Structured vs Conversational

| Aspect | Current (Structured) | New (Conversational) |
|--------|---------------------|---------------------|
| Communication | Predefined edges | Dynamic message passing |
| Agent Interaction | One-shot execution | Multi-turn dialogue |
| Decision Making | Single agent or formation | Collaborative deliberation |
| Flexibility | Fixed formations | Ad-hoc conversations |
| Use Cases | Deterministic workflows | Complex problem-solving |

---

## SOLID Compliance

This design follows Victor's established SOLID patterns, particularly the protocol-based architecture used throughout the framework.

### Interface Segregation Principle (ISP)

**Key Refinement**: Split large monolithic protocols into focused, single-purpose protocols following `victor/framework/protocols.py` pattern.

#### Original Design (Anti-Pattern):
```python
class ConversationProtocol(Protocol):
    """Too many responsibilities - violates ISP."""
    def initialize(...) -> None: ...
    def get_next_speaker(...) -> Optional[str]: ...
    def should_continue(...) -> tuple[bool, Optional[str]]: ...
    def format_message(...) -> str: ...
    def parse_response(...) -> ConversationalMessage: ...
    def get_result(...) -> Dict[str, Any]: ...
```

#### Refined Design (ISP Compliant):
```python
from typing import Protocol
from victor.framework.protocols import OrchestratorProtocol

@runtime_checkable
class ConversationProtocol(Protocol):
    """Core conversation lifecycle - turn management only.

    Single responsibility: Manage conversation flow and turns.
    """
    async def initialize(
        self,
        participants: List[ConversationParticipant],
        context: ConversationContext,
    ) -> None: ...

    async def get_next_speaker(
        self,
        context: ConversationContext,
    ) -> Optional[str]: ...

    async def should_continue(
        self,
        context: ConversationContext,
    ) -> tuple[bool, Optional[str]]: ...

@runtime_checkable
class MessageFormatterProtocol(Protocol):
    """Message formatting - separate concern.

    Single responsibility: Format messages for agents.
    """
    async def format_message(
        self,
        message: ConversationalMessage,
        recipient: str,
        context: ConversationContext,
    ) -> str: ...

    async def parse_response(
        self,
        response: str,
        sender: str,
        context: ConversationContext,
    ) -> ConversationalMessage: ...

@runtime_checkable
class ConversationResultProtocol(Protocol):
    """Result extraction - separate concern.

    Single responsibility: Extract conversation results.
    """
    async def get_result(
        self,
        context: ConversationContext,
    ) -> Dict[str, Any]: ...
```

**Benefits**:
- Each protocol has single responsibility (SRP)
- Clients depend only on what they use (ISP)
- Easier to test and mock
- Follows existing Victor patterns from `victor/framework/protocols.py`

### Dependency Inversion Principle (DIP)

**Key Refinement**: Depend on abstractions (protocols), not concrete implementations.

#### Before (Concrete dependency):
```python
class ConversationalNode:
    def __init__(
        self,
        orchestrator: AgentOrchestrator,  # Concrete class!
        ...
    ) -> None: ...
```

#### After (Protocol dependency):
```python
class ConversationalNode:
    def __init__(
        self,
        orchestrator: OrchestratorProtocol,  # Protocol!
        ...
    ) -> None: ...
```

**Benefits**:
- Can substitute different orchestrator implementations
- Easier to test with mocks
- Follows DIP (high-level doesn't depend on low-level)
- Consistent with rest of Victor codebase

### Open/Closed Principle (OCP)

**Key Refinement**: Use strategy pattern for extensibility without modification.

#### Strategy-Based Protocol Implementations:
```python
@runtime_checkable
class ConversationStrategy(Protocol):
    """Strategy for conversation orchestration."""
    async def orchestrate(
        self,
        context: ConversationContext,
    ) -> AsyncIterator[ConversationTurn]: ...

class RequestResponseStrategy:
    """Request-response implementation."""
    async def orchestrate(
        self,
        context: ConversationContext,
    ) -> AsyncIterator[ConversationTurn]:
        # Implementation
        ...

class DebateStrategy:
    """Debate implementation."""
    async def orchestrate(
        self,
        context: ConversationContext,
    ) -> AsyncIterator[ConversationTurn]:
        # Implementation
        ...
```

**Benefits**:
- Add new conversation strategies without modifying existing code (OCP)
- Each strategy independently testable
- Follows Victor's strategy pattern conventions

### Protocol Compliance Checklist

- [x] All protocols are focused (≤7 methods each)
- [x] No concrete class dependencies in high-level modules
- [x] All extensions through protocols/strategies
- [x] All interfaces substitutable (LSP compliant)
- [x] Single responsibility for all classes

---

## Code Reuse Strategy

This design reuses **~80-90% of existing Victor infrastructure**, dramatically reducing the implementation effort from ~2,500 LOC to ~800 LOC (68% reduction).

### Messaging Infrastructure Reuse

**Original Design**: Proposed new `ConversationalMessage` class duplicating existing message structure.

**Refined Design**: Extend existing `AgentMessage` from `victor/teams/types.py`.

```python
# CORRECT: Extend canonical message types
from victor.teams.types import AgentMessage, MessageType

@dataclass
class ConversationalMessage(AgentMessage):
    """Extend AgentMessage with conversation-specific fields.

    Reuses canonical message structure from victor/teams/types.py:80-150
    """
    conversation_id: str = ""
    turn_number: int = 0
    reply_to: Optional[str] = None

    def to_agent_message(self) -> AgentMessage:
        """Convert to canonical AgentMessage for compatibility."""
        return AgentMessage(
            sender_id=self.sender,
            content=self.content,
            message_type=self.message_type,
            recipient_id=self.recipients[0] if self.recipients else None,
            data=self.data,
            timestamp=self.timestamp,
        )
```

**Benefits**:
- Reuses proven, tested message infrastructure
- Maintains compatibility with existing systems
- Follows canonical import rules
- Reduces code duplication by ~200 LOC

**Related Files**:
- `victor/teams/types.py:80-150` - canonical AgentMessage
- `victor/agent/teams/communication.py:131-350` - TeamMessageBus for routing

### Event System Reuse

**Refined Design**: Use existing `UnifiedEventType` taxonomy and `ObservabilityBus`.

```python
# CORRECT: Extend existing event taxonomy
from victor.core.events.taxonomy import UnifiedEventType

class ConversationEventType(UnifiedEventType):
    """Add conversation events to existing taxonomy."""
    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_TURN = "conversation.turn"
    CONVERSATION_COMPLETED = "conversation.completed"
    CONVERSATION_MESSAGE = "conversation.message"

# Emit through existing backends
from victor.core.events.backends import ObservabilityBus

bus = ObservabilityBus()
await bus.emit(ConversationEventType.CONVERSATION_STARTED, {
    "conversation_id": conversation_id,
    "protocol": protocol.name,
})
```

**Benefits**:
- Consistent event handling across system
- Reuses existing event filtering, subscription
- Single observability pipeline
- Works with existing debugging/visualization tools

**Related Files**:
- `victor/core/events/taxonomy.py:63-626` - event taxonomy
- `victor/core/events/backends.py:81-995` - event backends

### Team Coordination Reuse

**Refined Design**: Leverage existing `UnifiedTeamCoordinator` for multi-agent conversations.

```python
# CORRECT: Reuse existing team coordination
from victor.teams import create_coordinator, TeamFormation

class ConversationalNode:
    """Reuse existing team infrastructure."""

    async def _create_team_from_protocol(
        self,
        protocol: ConversationProtocol,
    ) -> ITeamCoordinator:
        """Create team using existing coordinator."""
        return create_coordinator(
            self._orchestrator,
            formation=TeamFormation.SEQUENTIAL,  # Or protocol-specific
            participants=[
                TeamMember(
                    id=p.id,
                    role=p.role,
                    goal=p.goal,
                )
                for p in self._participants
            ],
        )
```

**Benefits**:
- Reuses production-tested team coordination (~1500 LOC saved)
- Consistent behavior across system
- No need to re-implement formations
- Single point of maintenance

**Related Files**:
- `victor/teams/__init__.py:137-186` - create_coordinator factory
- `victor/teams/unified_coordinator.py` - 5 formation implementations

### Code Reuse Summary

| Component | Existing | Reuse Strategy | LOC Saved |
|-----------|----------|----------------|-----------|
| Messaging | TeamMessageBus | Direct reuse | ~200 |
| Message Types | AgentMessage | Extend, not duplicate | ~150 |
| Event System | UnifiedEventType | Add new types | ~100 |
| Event Backend | ObservabilityBus | Direct reuse | ~250 |
| Team Coordination | UnifiedTeamCoordinator | Direct reuse | ~500 |
| State Merging | MergeStrategy | Direct reuse | ~100 |
| **Total** | | | **~1,300** |

**Net New LOC After Reuse**: ~800 (vs. ~2,500 without reuse)
**Reduction**: 68%

### Canonical Import Paths

**Messaging**:
```python
# CORRECT - Canonical import
from victor.teams.types import AgentMessage, MessageType

# INCORRECT - Duplicate
from victor.framework.conversations.types import ConversationalMessage
```

**Events**:
```python
# CORRECT - Extend existing taxonomy
from victor.core.events.taxonomy import UnifiedEventType

# INCORRECT - Separate event system
from victor.framework.conversations.events import ConversationEvent
```

**Team Coordination**:
```python
# CORRECT - Use existing team infrastructure
from victor.teams import create_coordinator, TeamFormation

# INCORRECT - Duplicate team logic
from victor.framework.conversations.team import ConversationalTeam
```

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WORKFLOW GRAPH                              │
│                                                                  │
│  ┌─────────┐    ┌──────────────┐    ┌────────────────┐         │
│  │ Agent   │───▶│ Conversational│───▶│    Team       │         │
│  │ Node    │    │    Node      │    │    Node       │         │
│  └─────────┘    └──────────────┘    └────────────────┘         │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONVERSATIONAL COORDINATION LAYER                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ConversationalNode                                      │  │
│  │  - Orchestrates multi-turn conversations                 │  │
│  │  - Enforces conversation protocols                       │  │
│  │  - Manages turn-taking and termination                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                    ▲                             │
│  ┌───────────────────────────────┼──────────────────────────┐  │
│  │                               │                          │  │
│  │  ┌─────────────────┐    ┌─────┴─────┐    ┌────────────┐ │  │
│  │  │ MessageRouter   │    │ Conversation│    │ Conversation│ │  │
│  │  │ - Role-based    │    │  Protocol   │    │  History   │ │  │
│  │  │ - Capability    │    │  - Rules    │    │  - Tracker │ │  │
│  │  │ - Context       │    │  - Turn     │    │  - Storage │ │  │
│  │  └─────────────────┘    └─────────────┘    └────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXISTING VICTOR INFRASTRUCTURE                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌────────────┐  │
│  │ StateGraph│  │ TeamNode │  │EventBus   │  │MessageBus  │  │
│  └───────────┘  └───────────┘  └───────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Points

1. **StateGraph**: ConversationalNode as a new node type
2. **TeamNode**: Can host ConversationalNodes within teams
3. **EventBus**: Emit conversation events for observability
4. **MessageBus**: Reuse existing TeamMessageBus for transport
5. **Teams**: Conversational participants can be TeamMembers or SubAgents

---

## Core Components

### 1. ConversationalNode

**Location**: `victor/framework/conversations/node.py`

**Responsibility**: Orchestrates multi-turn conversations between agents

**Key Features**:
- Configurable conversation protocol
- Turn management and ordering
- Termination conditions
- Timeout handling
- State management
- Error recovery

**Interface**:
```python
class ConversationalNode:
    """Node that hosts a multi-turn conversation between agents.

    Attributes:
        id: Unique node identifier
        name: Human-readable name
        participants: List of conversation participants
        protocol: Conversation protocol to follow
        max_turns: Maximum conversation turns (prevents infinite loops)
        timeout_seconds: Maximum conversation duration
        termination_conditions: Conditions that end the conversation
        output_key: Where to store conversation result in graph state
    """

    def __init__(
        self,
        id: str,
        name: str,
        participants: List[ConversationParticipant],
        protocol: ConversationProtocol,
        max_turns: int = 20,
        timeout_seconds: int = 300,
        termination_conditions: Optional[List[ConversationTerminationCondition]] = None,
        output_key: str = "conversation_result",
    ) -> None: ...

    async def execute(
        self,
        orchestrator: AgentOrchestrator,
        graph_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the conversation.

        Args:
            orchestrator: For spawning SubAgents
            graph_state: Current workflow state

        Returns:
            Updated graph state with conversation results
        """
        ...
```

### 2. Conversation Protocol

**Location**: `victor/framework/conversations/protocols.py`

**Responsibility**: Defines conversation patterns and rules

**Protocols**:

#### A. Request-Response Protocol
Simple Q&A pattern with optional follow-ups.

#### B. Round-Robin Protocol
Sequential turn-taking among participants.

#### C. Debate Protocol
Structured debate between opposing positions with synthesis.

#### D. Consensus Protocol
Agents work toward agreement with configurable thresholds.

#### E. Negotiation Protocol
Agents trade proposals to reach agreement.

### 3. Message Router

**Location**: `victor/framework/conversations/router.py`

**Responsibility**: Intelligent message routing based on context

**Routing Strategies**:
- Role-based routing
- Capability-based routing
- Context-aware routing

### 4. Conversation History

**Location**: `victor/framework/conversations/history.py`

**Responsibility**: Track and visualize conversation flow

**Features**:
- Full message log with metadata
- Turn tracking
- Agent participation metrics
- Export to JSON, Markdown, Mermaid

---

## Data Structures

### ConversationalMessage

```python
@dataclass
class ConversationalMessage:
    """A message in a conversation.

    Attributes:
        id: Unique message identifier
        conversation_id: Conversation this belongs to
        sender: Who sent the message
        recipients: Who receives (None = broadcast)
        content: Message content
        message_type: Type of message
        timestamp: When sent
        reply_to: ID of message being replied to
        data: Structured data payload
        metadata: Additional metadata
        turn_number: Which turn this is
        requires_response: Whether response is expected
    """
```

### ConversationContext

```python
@dataclass
class ConversationContext:
    """Context shared across conversation.

    Attributes:
        conversation_id: Unique identifier
        participants: Dictionary of participants
        protocol: Conversation protocol being used
        current_turn: Current turn number
        started_at: Start timestamp
        shared_state: Shared state for conversation
        history: Conversation history
        is_terminated: Whether conversation has ended
        termination_reason: Why conversation ended
    """
```

---

## Integration with StateGraph

### ConversationalNode as StateGraph Node

```python
from victor.framework.graph import StateGraph
from victor.framework.conversations import (
    ConversationalNode,
    ConsensusProtocol,
    ConversationParticipant,
)

# Create conversation node
consensus_node = ConversationalNode(
    id="design_consensus",
    name="Architecture Consensus",
    participants=[...],
    protocol=ConsensusProtocol(...),
    max_turns=15,
    timeout_seconds=300,
    output_key="architecture_decision",
)

# Add to StateGraph
graph = StateGraph(ArchitectureState)
graph.add_node("reach_consensus", consensus_node.execute)
```

---

## Implementation Strategy

### Phase 3.5.1: Core Infrastructure (MVP)

**Estimated LOC**: ~1,200 production, ~600 tests
**Duration**: 2-3 weeks

**Components**:
1. Data structures (~300 LOC)
2. Conversation history (~200 LOC)
3. Message router (~250 LOC)
4. Base protocol (~200 LOC)
5. Request-Response protocol (~150 LOC)
6. Observability integration (~100 LOC)

### Phase 3.5.2: ConversationalNode Integration

**Estimated LOC**: ~800 production, ~400 tests
**Duration**: 2 weeks

**Components**:
1. ConversationalNode (~400 LOC)
2. Workflow definition support (~200 LOC)
3. State merging (~100 LOC)
4. Example workflows (~100 LOC)

### Phase 3.5.3: Advanced Protocols

**Estimated LOC**: ~1,000 production, ~500 tests
**Duration**: 3 weeks

**Components**:
1. Round-Robin protocol (~150 LOC)
2. Debate protocol (~250 LOC)
3. Consensus protocol (~300 LOC)
4. Negotiation protocol (~300 LOC)

---

## Testing Strategy

### Unit Tests

**Location**: `tests/unit/framework/conversations/`

**Test Coverage**:
1. Data structures (message, context, participant)
2. History tracking
3. Protocols (turn ordering, termination)
4. Router (rule evaluation, priority)
5. ConversationalNode (execution, state)

### Integration Tests

**Location**: `tests/integration/framework/conversations/`

**Test Scenarios**:
1. StateGraph integration
2. TeamNode integration
3. YAML workflows

---

## Performance Considerations

### Optimization Strategies

1. **Lazy History Loading**
2. **Async Message Processing**
3. **Caching**
4. **Resource Management**

### Performance Targets

- **Message latency**: < 100ms (excluding agent execution)
- **Routing decision**: < 10ms
- **History export**: < 500ms for 100 turns
- **Protocol overhead**: < 5% of total execution time
- **Memory per conversation**: < 10MB (100 turns)

---

## Future Enhancements

### Phase 3.5.5: Advanced Features

1. **Hierarchical Conversations**
2. **External Agent Integration**
3. **Semantic Routing**
4. **Learning and Adaptation**
5. **Visualization**

---

## Conclusion

This design introduces **conversational agent coordination** to Victor while maintaining backward compatibility, SOLID principles, observability, and testability. The phased implementation allows incremental delivery starting with an MVP focused on core infrastructure and Request-Response protocol.
