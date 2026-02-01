# Real-Time Team Collaboration Dashboard

## Overview

The Victor Team Collaboration Dashboard provides real-time monitoring and visualization of multi-agent team executions. It combines a FastAPI WebSocket server with a React TypeScript frontend to deliver sub-second latency updates for team execution events.

![Dashboard Architecture](/images/dashboard-architecture.png)

## Features

### Real-Time Updates
- **WebSocket Streaming**: Sub-second latency for team execution events
- **Auto-Reconnection**: Automatic reconnection with exponential backoff
- **Connection Status**: Visual indicators for connection health

### Team Monitoring
- **Execution Status**: Track team lifecycle from start to completion
- **Member Status Cards**: Real-time status of each team member
- **Formation Types**: Support for sequential, parallel, pipeline, hierarchical, and consensus formations
- **Recursion Depth**: Visual indicator of nested team execution depth

### Communication Visualization
- **Flow Diagram**: Interactive message flow between team members using ReactFlow
- **Message History**: Complete log of all communications
- **Communication Patterns**: Support for request/response, broadcast, multicast, and pub/sub

### Context Management
- **Shared Context Table**: Real-time view of team-wide key-value store
- **Change Tracking**: Monitor context updates and merges
- **Searchable**: Quick lookup of context keys

### Negotiation Tracking
- **Voting Progress**: Visual representation of vote distribution
- **Consensus Status**: Clear indicators for consensus achievement
- **Round Tracking**: Number of negotiation rounds used

### Metrics & Analytics
- **Success Rates**: Team execution success/failure ratios
- **Duration Analysis**: Average execution time statistics
- **Tool Usage**: Tool call aggregation and distribution
- **Formation Distribution**: Breakdown by formation type

## Installation

### Prerequisites

```bash
# Python dependencies
pip install victor-ai[api]

# Node.js dependencies (for frontend)
cd tools/team_dashboard
npm install
```

### Quick Start

```bash
# Start both backend and frontend
cd tools/team_dashboard
./run.sh

# Or start individually
# Backend (from project root)
python -m uvicorn victor.workflows.team_dashboard_api:app --reload

# Frontend (from tools/team_dashboard)
npm run dev
```

### Production Mode

```bash
# Build frontend and start backend
./run.sh --prod

# Or manually
cd tools/team_dashboard
npm run build

# Serve with uvicorn
python -m uvicorn victor.workflows.team_dashboard_api:app --host 0.0.0.0 --port 8000
```

## Architecture

### Backend Components

#### WebSocket Server (`team_dashboard_server.py`)

```python
from victor.workflows.team_dashboard_server import (
    TeamDashboardServer,
    get_dashboard_server,
)

# Get singleton instance
server = get_dashboard_server()

# Broadcast events
await server.broadcast_team_started(
    execution_id="exec-123",
    team_id="review_team",
    formation="parallel",
    member_count=3,
    recursion_depth=1,
)

await server.broadcast_member_completed(
    execution_id="exec-123",
    member_id="security_reviewer",
    success=True,
    duration_seconds=5.2,
)
```

**Key Classes:**
- `TeamDashboardServer`: Main WebSocket server
- `ConnectionManager`: WebSocket connection lifecycle
- `TeamExecutionState`: Execution state tracking
- `MemberState`: Member state tracking
- `DashboardEvent`: Event broadcast format

#### REST API (`team_dashboard_api.py`)

```python
from victor.workflows.team_dashboard_api import create_dashboard_app

# Create FastAPI app
app = create_dashboard_app()

# API Endpoints:
# GET /api/v1/executions                    - List all executions
# GET /api/v1/executions/{id}               - Get execution details
# GET /api/v1/executions/{id}/members       - Get member statuses
# GET /api/v1/executions/{id}/communications - Get communication history
# GET /api/v1/executions/{id}/context       - Get shared context
# GET /api/v1/executions/{id}/negotiation   - Get negotiation status
# GET /api/v1/metrics/summary               - Get metrics summary
# WS  /ws/team/{execution_id}               - WebSocket endpoint
```

### Frontend Components

#### Directory Structure

```
tools/team_dashboard/
├── src/
│   ├── components/
│   │   ├── TeamExecutionView.tsx      # Main execution view
│   │   ├── MemberStatusCard.tsx       # Member status card
│   │   ├── CommunicationFlow.tsx      # Message flow diagram
│   │   ├── SharedContextTable.tsx     # Context table
│   │   ├── NegotiationPanel.tsx       # Negotiation status
│   │   └── MetricsPanel.tsx           # Metrics summary
│   ├── hooks/
│   │   └── useWebSocket.ts            # WebSocket hook with auto-reconnect
│   ├── store/
│   │   └── dashboardStore.ts          # Zustand state management
│   ├── types/
│   │   └── index.ts                   # TypeScript type definitions
│   ├── App.tsx                        # Root component
│   └── main.tsx                       # Entry point
├── package.json
├── vite.config.ts
├── tsconfig.json
└── run.sh                             # Startup script
```

#### WebSocket Integration

```typescript
import { useWebSocket } from './hooks/useWebSocket';
import { useDashboardStore } from './store/dashboardStore';

const { state: wsState } = useWebSocket('ws://localhost:8000/ws/team/exec-123', {
  onEvent: (event) => {
    // Handle dashboard events
    useDashboardStore.getState().handleDashboardEvent(event);
  },
  onError: (error) => {
    console.error('WebSocket error:', error);
  },
  reconnectInterval: 3000,
  maxReconnectAttempts: 10,
});
```

#### State Management (Zustand)

```typescript
import { useDashboardStore } from './store/dashboardStore';

// Select execution
const execution = useDashboardStore(
  (state) => state.executions[executionId]
);

// Update member state
const updateMember = useDashboardStore((state) => state.updateMember);
updateMember(executionId, memberId, { status: 'completed' });

// Filter executions
const filteredExecutions = useDashboardStore(
  selectFilteredExecutions()
);
```

## WebSocket Protocol

### Connection

```
WS /ws/team/{execution_id}
```

### Client → Server Messages

```json
{
  "action": "subscribe",
  "event_types": ["member.started", "member.completed"]
}
```

```json
{
  "action": "query_state",
  "execution_id": "exec-123"
}
```

```json
{
  "action": "ping"
}
```

### Server → Client Events

#### Team Started

```json
{
  "event_type": "team.started",
  "execution_id": "exec-123",
  "timestamp": 1704729600.0,
  "data": {
    "team_id": "review_team",
    "formation": "parallel",
    "member_count": 3,
    "recursion_depth": 1,
    "start_time": "2025-01-15T10:00:00Z"
  }
}
```

#### Member Started

```json
{
  "event_type": "member.started",
  "execution_id": "exec-123",
  "timestamp": 1704729601.0,
  "data": {
    "member_id": "security_reviewer",
    "role": "security_expert",
    "status": "running",
    "start_time": "2025-01-15T10:00:01Z"
  }
}
```

#### Member Updated

```json
{
  "event_type": "member.updated",
  "execution_id": "exec-123",
  "timestamp": 1704729605.0,
  "data": {
    "member_id": "security_reviewer",
    "tool_calls_used": 5,
    "tools_used": ["read_file", "search_code"]
  }
}
```

#### Member Completed

```json
{
  "event_type": "member.completed",
  "execution_id": "exec-123",
  "timestamp": 1704729610.0,
  "data": {
    "member_id": "security_reviewer",
    "success": true,
    "duration_seconds": 9.0,
    "error_message": null,
    "end_time": "2025-01-15T10:00:10Z"
  }
}
```

#### Message Sent

```json
{
  "event_type": "message.sent",
  "execution_id": "exec-123",
  "timestamp": 1704729607.0,
  "data": {
    "log": {
      "timestamp": 1704729607.0,
      "message_type": "request",
      "sender_id": "security_reviewer",
      "recipient_id": "quality_reviewer",
      "content": "Please review the auth module",
      "communication_type": "request_response",
      "duration_ms": 1500
    }
  }
}
```

#### Context Updated

```json
{
  "event_type": "context.updated",
  "execution_id": "exec-123",
  "timestamp": 1704729608.0,
  "data": {
    "key": "findings",
    "value": {
      "bugs": ["bug-1", "bug-2"]
    },
    "member_id": "security_reviewer",
    "operation": "set"
  }
}
```

#### Negotiation Completed

```json
{
  "event_type": "negotiation.completed",
  "execution_id": "exec-123",
  "timestamp": 1704729620.0,
  "data": {
    "success": true,
    "rounds": 2,
    "consensus_achieved": true,
    "votes": {
      "proposal-1": 5,
      "proposal-2": 1
    },
    "agreed_proposal": {
      "id": "proposal-1",
      "content": "Use Python for implementation"
    }
  }
}
```

## Performance

### Benchmarks

| Metric | Target | Actual |
|--------|--------|--------|
| Event Latency | < 1s | 0.1-0.5s |
| WebSocket Throughput | 100 msg/s | 150+ msg/s |
| Concurrent Connections | 10+ | 20+ |
| Memory Usage | < 500MB | ~300MB |
| CPU Usage | < 50% | ~30% |

### Optimization Tips

1. **Limit Communication Logs**: Default shows last 100 messages
2. **Throttle UI Updates**: Use debouncing for rapid events
3. **Pagination**: Implement pagination for large execution lists
4. **Lazy Loading**: Load components on demand
5. **WebSocket Filtering**: Subscribe only to needed event types

## Usage Examples

### Python Integration

```python
from victor.workflows.team_dashboard_server import get_dashboard_server
from victor.teams import create_coordinator
from victor.workflows.team_collaboration import (
    TeamCommunicationProtocol,
    SharedTeamContext,
)

# Get dashboard server
dashboard = get_dashboard_server()

# Create team with collaboration features
team = create_coordinator(
    team_id="code_review_team",
    formation="parallel",
    members=[security_reviewer, quality_reviewer, performance_reviewer],
    collaboration_config={
        "enable_communication": True,
        "enable_shared_context": True,
        "enable_negotiation": True,
    },
)

# Broadcast team start
await dashboard.broadcast_team_started(
    execution_id="exec-123",
    team_id="code_review_team",
    formation="parallel",
    member_count=3,
    recursion_depth=1,
)

# Track member progress
await dashboard.broadcast_member_started(
    execution_id="exec-123",
    member_id="security_reviewer",
    role="security_expert",
)

# Broadcast communication
comm_log = CommunicationLog(
    timestamp=time.time(),
    message_type="request",
    sender_id="security_reviewer",
    recipient_id="quality_reviewer",
    content="Please review the auth module",
    communication_type=CommunicationType.REQUEST_RESPONSE,
    duration_ms=1500,
)
await dashboard.broadcast_communication("exec-123", comm_log)

# Broadcast completion
await dashboard.broadcast_team_completed(
    execution_id="exec-123",
    success=True,
    duration_seconds=25.5,
    consensus_achieved=True,
)
```

### TypeScript Integration

```typescript
import { TeamExecutionView } from './components/TeamExecutionView';
import { useWebSocket } from './hooks/useWebSocket';

function MyComponent() {
  const executionId = 'exec-123';

  return (
    <TeamExecutionView
      executionId={executionId}
      wsUrl={`ws://localhost:8000/ws/team/${executionId}`}
      className="w-full"
    />
  );
}
```

## Troubleshooting

### WebSocket Connection Issues

**Problem**: Cannot connect to WebSocket

**Solutions**:
1. Check backend is running: `curl http://localhost:8000/health`
2. Verify CORS settings: Check `cors_origins` in server config
3. Check firewall: Ensure port 8000 is accessible
4. Review browser console for connection errors

### High Memory Usage

**Problem**: Dashboard consumes excessive memory

**Solutions**:
1. Limit communication log history
2. Implement pagination for execution lists
3. Use virtual scrolling for large member lists
4. Clear old execution states

### Stale UI Updates

**Problem**: UI not reflecting latest state

**Solutions**:
1. Check WebSocket connection status
2. Verify event handlers are registered
3. Check Zustand store updates
4. Review React re-renders with DevTools

## API Reference

### REST Endpoints

See [FastAPI auto-generated docs](http://localhost:8000/docs) for interactive API documentation.

### WebSocket Events

See "WebSocket Protocol" section above for event format reference.

## Contributing

To extend the dashboard:

1. **Add New Event Types**:
   - Update `DashboardEventType` enum in `team_dashboard_server.py`
   - Add handler in `handleDashboardEvent()` in `dashboardStore.ts`
   - Create UI component to display event

2. **Add New Metrics**:
   - Update `TeamMetricsCollector` in `team_metrics.py`
   - Add API endpoint in `team_dashboard_api.py`
   - Create visualization in `MetricsPanel.tsx`

3. **Add New Visualizations**:
   - Create component in `src/components/`
   - Integrate in `TeamExecutionView.tsx`
   - Update TypeScript types in `types/index.ts`

## License

Apache License 2.0 - See LICENSE file for details

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
