# Phase 3.2: Workflow Visualization - Design Document

**Status:** Draft
**Created:** 2025-01-09
**Author:** Claude (Sonnet 4.5)
**Target:** Victor v0.5.1+

---

## Executive Summary

This document designs a **real-time workflow visualization enhancement** for Victor's existing observability infrastructure. The enhancement will add **interactive graph rendering** capabilities to the FastAPI server and TUI dashboard, leveraging the existing `WorkflowVisualizer`, `WorkflowEventType` system, and `EventBus`.

**Key Insight:** Victor already has comprehensive workflow infrastructure:
- `victor/workflows/visualization.py` - Static rendering (ASCII, Mermaid, DOT, SVG, PNG)
- `victor/workflows/streaming.py` - Real-time execution events (`WorkflowEventType`)
- `victor/integrations/api/fastapi_server.py` - FastAPI with WebSocket support
- `victor/observability/dashboard/app.py` - Textual TUI dashboard
- `victor/framework/graph.py` - StateGraph with compiled graph schema

**Our Task:** Add **dynamic, interactive visualization** that shows execution progress in real-time.

---

## 1. Graph Rendering Library Evaluation

### Candidates Analyzed

| Library | Type | Pros | Cons | Python Integration | Performance | Suitability |
|---------|------|------|------|-------------------|-------------|-------------|
| **React Flow** | React-based | • Excellent interactivity<br>• Huge ecosystem<br>• Built-in controls | • Requires Node.js build<br>• Heavy for simple graphs | Poor (requires Node) | Excellent | ⭐⭐⭐⭐ |
| **Cytoscape.js** | Pure JS | • Graph theory features<br>• Excellent performance<br>• Many layouts | • Steeper learning curve<br>• More complex API | Medium (via Jinja2) | Excellent | ⭐⭐⭐⭐⭐ |
| **vis.js Network** | Pure JS | • Simple API<br>• Good for large graphs<br>• Physics-based layout | • Less modern UI<br>• Limited customization | Medium (via Jinja2) | Good | ⭐⭐⭐ |
| **D3.js** | Pure JS | • Maximum flexibility<br>• Custom visualizations | • Lots of boilerplate<br>• No built-in graph components | Medium (via Jinja2) | Good | ⭐⭐⭐ |
| **Mermaid.js** | Pure JS | • Markdown-friendly<br>• Already used for static rendering<br>• Simple text format | • Limited interactivity<br>• Basic layouts only | Good | Medium | ⭐⭐⭐ |
| **Plotly** | Python-first | • Native Python support<br>• Built-in interactivity<br>• Easy integration | • Not graph-focused<br>• Limited DAG features | Excellent | Medium | ⭐⭐⭐ |

### Recommendation: **Cytoscape.js** (Primary) + **Mermaid.js** (Fallback)

**Why Cytoscape.js?**

1. **Performance Excellence**: Handles 1000+ nodes smoothly (important for complex workflows)
2. **Graph Theory Features**: Built-in DAG layout algorithms (dagre, cose, breadthfirst)
3. **Python Integration**: Simple Jinja2 template integration with FastAPI
4. **Event System**: Native support for click/hover events on nodes and edges
5. **Visual Styling**: Granular control over node colors, borders, labels based on state
6. **Maturity**: Production-ready with excellent docs (since 2002)

**Why Mermaid.js as Fallback?**

- Victor already generates Mermaid format (`WorkflowVisualizer.to_mermaid()`)
- Zero additional backend work
- Good enough for simple visualizations
- Can be embedded in markdown/docs

**Architecture Decision:**
- **Primary**: Cytoscape.js for interactive web UI (FastAPI templates)
- **Fallback**: Mermaid.js for static/embedded views
- **TUI**: ASCII art enhancements (existing `WorkflowVisualizer.to_ascii()`)

---

## 2. Architecture Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│  Browser (Cytoscape.js)    │    TUI (Textual Dashboard)          │
│  - Interactive graph         │    - Enhanced ASCII rendering     │
│  - Real-time updates         │    - Color-coded node states      │
│  - Zoom/pan controls         │    - Progress bars                │
└────────────────┬────────────────────────────┬───────────────────┘
                 │                            │
                 ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI SERVER                                │
├─────────────────────────────────────────────────────────────────┤
│  /workflows/{id}/graph      GET  → Static graph structure        │
│  /workflows/{id}/execution  GET  → Execution path/status         │
│  /workflows/{id}/stream     WS   → Real-time event stream        │
│  /workflows/visualize/{id}  HTML → Cytoscape.js UI page         │
└────────────────┬──────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVENT BRIDGE LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  EventBridge (victor/integrations/api/event_bridge.py)          │
│  - Subscribes to EventBus for WorkflowEventType events          │
│  - Broadcasts to WebSocket clients                              │
│  - Filters by workflow_id                                       │
└────────────────┬──────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW EXECUTION LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  StreamingWorkflowExecutor                                       │
│  WorkflowEventEmitter                                             │
│  - Emits NODE_START, NODE_COMPLETE, NODE_ERROR                  │
│  - Updates progress, timing, metadata                            │
└────────────────┬──────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STATEGRAPH / WORKFLOW DEFINITION               │
├─────────────────────────────────────────────────────────────────┤
│  CompiledGraph.get_graph_schema() → nodes + edges               │
│  WorkflowDefinition → WorkflowVisualizer                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 API Endpoint Specifications

#### A. GET `/workflows/{workflow_id}/graph` - Static Graph Structure

**Purpose:** Return workflow DAG structure (nodes + edges) for initial render.

**Response Format:**
```json
{
  "workflow_id": "wf_abc123",
  "name": "Deep Research Workflow",
  "description": "Multi-stage research with fact-checking",
  "nodes": [
    {
      "id": "research",
      "name": "Literature Review",
      "type": "AgentNode",
      "description": "Gather academic sources",
      "metadata": {
        "role": "researcher",
        "tool_budget": 15,
        "max_retries": 2
      }
    },
    {
      "id": "analyze",
      "name": "Analyze Findings",
      "type": "AgentNode",
      "description": "Synthesize research",
      "metadata": {
        "role": "analyst",
        "tool_budget": 10
      }
    }
  ],
  "edges": [
    {
      "source": "research",
      "target": "analyze",
      "label": null,
      "conditional": false
    },
    {
      "source": "analyze",
      "target": "fact_check",
      "label": "needs_verification",
      "conditional": true
    }
  ],
  "start_node": "research",
  "entry_point": "research",
  "total_nodes": 3
}
```

**Implementation:**
```python
@app.get("/workflows/{workflow_id}/graph", tags=["Workflows"])
async def get_workflow_graph(workflow_id: str) -> JSONResponse:
    """Get static workflow graph structure."""
    # 1. Load workflow from registry or execution history
    # 2. Extract nodes and edges from WorkflowDefinition or StateGraph
    # 3. Return JSON format compatible with Cytoscape.js
    pass
```

---

#### B. GET `/workflows/{workflow_id}/execution` - Execution Status

**Purpose:** Get current execution state (which nodes executed, in what order, status).

**Response Format:**
```json
{
  "workflow_id": "wf_abc123",
  "status": "running",
  "progress": 45.5,
  "started_at": "2025-01-09T10:30:00Z",
  "current_node": "fact_check",
  "completed_nodes": ["research", "analyze"],
  "failed_nodes": [],
  "skipped_nodes": [],
  "node_execution_path": [
    {
      "node_id": "research",
      "status": "completed",
      "started_at": "2025-01-09T10:30:05Z",
      "completed_at": "2025-01-09T10:32:15Z",
      "duration_seconds": 130.0,
      "tool_calls": 12,
      "tokens_used": 4500
    },
    {
      "node_id": "analyze",
      "status": "completed",
      "started_at": "2025-01-09T10:32:20Z",
      "completed_at": "2025-01-09T10:35:00Z",
      "duration_seconds": 160.0,
      "tool_calls": 8,
      "tokens_used": 3200
    },
    {
      "node_id": "fact_check",
      "status": "running",
      "started_at": "2025-01-09T10:35:05Z",
      "completed_at": null,
      "duration_seconds": 30.0,
      "tool_calls": 3,
      "tokens_used": 1200
    }
  ],
  "total_duration_seconds": 320.0,
  "total_tool_calls": 23,
  "total_tokens": 8900
}
```

**Implementation:**
```python
@app.get("/workflows/{workflow_id}/execution", tags=["Workflows"])
async def get_workflow_execution(workflow_id: str) -> JSONResponse:
    """Get current workflow execution status."""
    # 1. Query in-memory execution store (_workflow_executions in fastapi_server.py)
    # 2. Extract completed_nodes, current_node, node_execution_path
    # 3. Return structured JSON with timing metrics
    pass
```

---

#### C. WS `/workflows/{workflow_id}/stream` - Real-Time Event Stream

**Purpose:** WebSocket endpoint for real-time execution updates.

**Connection Flow:**
```
Client → Server: {"type": "subscribe", "workflow_id": "wf_abc123"}
Server → Client: {"type": "subscribed", "workflow_id": "wf_abc123"}
[Event stream...]
Server → Client: {
  "type": "event",
  "event_type": "node_start",
  "workflow_id": "wf_abc123",
  "node_id": "fact_check",
  "node_name": "Fact Checker",
  "timestamp": "2025-01-09T10:35:05Z",
  "progress": 45.5
}
Server → Client: {
  "type": "event",
  "event_type": "node_complete",
  "workflow_id": "wf_abc123",
  "node_id": "fact_check",
  "duration_seconds": 45.0,
  "progress": 66.6
}
Server → Client: {
  "type": "event",
  "event_type": "workflow_complete",
  "workflow_id": "wf_abc123",
  "progress": 100.0,
  "is_final": true
}
```

**Implementation:**
```python
@app.websocket("/workflows/{workflow_id}/stream")
async def workflow_stream(websocket: WebSocket, workflow_id: str):
    """Real-time workflow execution event stream."""
    await websocket.accept()

    # Subscribe to EventBus workflow events via EventBridge
    event_handler = lambda event: asyncio.create_task(
        websocket.send_json(event_to_dict(event))
    )

    # Register subscription with EventBridge
    subscription = event_bridge.subscribe_workflow(workflow_id, event_handler)

    try:
        while True:
            # Keep connection alive, handle ping/pong
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        subscription.unsubscribe()
```

---

#### D. GET `/workflows/visualize/{workflow_id}` - Web UI Page

**Purpose:** Serve HTML page with embedded Cytoscape.js visualization.

**Response:** HTML page with:
- Cytoscape.js library loaded (CDN)
- Graph structure from `/workflows/{id}/graph`
- Execution status from `/workflows/{id}/execution`
- WebSocket connection to `/workflows/{id}/stream`

**Implementation (Jinja2 Template):**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Victor Workflow Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape.js/3.28.1/cytoscape.min.js"></script>
    <style>
        #cy { width: 100%; height: 600px; border: 1px solid #ccc; }
        .node-running { background: #FFF3E0; border-color: #FF9800; }
        .node-completed { background: #E8F5E9; border-color: #4CAF50; }
        .node-failed { background: #FFEBEE; border-color: #F44336; }
        .node-pending { background: #F5F5F5; border-color: #9E9E9E; }
    </style>
</head>
<body>
    <h1>{{ workflow.name }}</h1>
    <div id="cy"></div>
    <div id="status">
        <p>Status: <span id="status-text">{{ status }}</span></p>
        <p>Progress: <span id="progress-text">{{ progress }}%</span></p>
        <p>Current Node: <span id="current-node">{{ current_node }}</span></p>
    </div>

    <script>
        // Fetch initial graph structure
        fetch('/workflows/{{ workflow_id }}/graph')
            .then(r => r.json())
            .then(graph => {
                const elements = {
                    nodes: graph.nodes.map(n => ({
                        data: { id: n.id, label: n.name, type: n.type }
                    })),
                    edges: graph.edges.map(e => ({
                        data: { source: e.source, target: e.target, label: e.label }
                    }))
                };

                window.cy = cytoscape({
                    container: document.getElementById('cy'),
                    elements: elements,
                    layout: { name: 'dagre', rankDir: 'TB' },
                    style: [
                        {
                            selector: 'node',
                            style: {
                                'label': 'data(label)',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'background-color': '#E3F2FD',
                                'border-width': 2,
                                'border-color': '#1976D2'
                            }
                        },
                        {
                            selector: '.running',
                            style: { 'background-color': '#FFF3E0', 'border-color': '#FF9800' }
                        },
                        {
                            selector: '.completed',
                            style: { 'background-color': '#E8F5E9', 'border-color': '#4CAF50' }
                        },
                        {
                            selector: '.failed',
                            style: { 'background-color': '#FFEBEE', 'border-color': '#F44336' }
                        }
                    ]
                });
            });

        // Connect WebSocket for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/workflows/{{ workflow_id }}/stream`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'event') {
                if (data.event_type === 'node_start') {
                    // Highlight node as running
                    window.cy.$(`#${data.node_id}`).addClass('running');
                    document.getElementById('current-node').textContent = data.node_name;
                }
                else if (data.event_type === 'node_complete') {
                    // Mark node as completed
                    window.cy.$(`#${data.node_id}`).removeClass('running').addClass('completed');
                    document.getElementById('progress-text').textContent = `${data.progress}%`;
                }
                else if (data.event_type === 'node_error') {
                    // Mark node as failed
                    window.cy.$(`#${data.node_id}`).removeClass('running').addClass('failed');
                }
                else if (data.event_type === 'workflow_complete') {
                    document.getElementById('status-text').textContent = 'Completed';
                }
            }
        };
    </script>
</body>
</html>
```

---

### 2.3 Graph JSON Format Specification

#### Cytoscape.js Element Format

**Nodes:**
```json
{
  "data": {
    "id": "node_id",
    "label": "Node Name",
    "type": "AgentNode | ComputeNode | ConditionNode | ParallelNode | HITLNode",
    "description": "Node description",
    "status": "pending | running | completed | failed | skipped",
    "metadata": {
      "role": "researcher",
      "tool_budget": 15,
      "duration_seconds": 130.0,
      "tool_calls": 12
    }
  },
  "classes": "node-completed"
}
```

**Edges:**
```json
{
  "data": {
    "source": "source_node_id",
    "target": "target_node_id",
    "label": "branch_name",
    "conditional": true
  }
}
```

**Status Classes:**
- `.node-pending` - Not yet executed
- `.node-running` - Currently executing
- `.node-completed` - Finished successfully
- `.node-failed` - Failed with error
- `.node-skipped` - Skipped in execution

---

## 3. Frontend Design

### 3.1 UI Components

#### A. Graph Canvas (Main Visualization)

**Features:**
- Full-width interactive Cytoscape.js canvas
- Pan (drag background) and zoom (mouse wheel)
- Click node to show details panel
- Hover node to show tooltip
- Mini-map for large graphs

**Styling:**
- 1000px width × 600px height (configurable)
- Light theme with color-coded node states
- Smooth transitions between states
- Directional arrows on edges

---

#### B. Node Details Panel (Sidebar)

**Purpose:** Show detailed information about selected/clicked node.

**UI:**
```html
<div id="node-details" class="sidebar">
    <h3 id="node-name">Node Name</h3>
    <p><strong>Type:</strong> <span id="node-type">AgentNode</span></p>
    <p><strong>Status:</strong> <span id="node-status">Completed</span></p>
    <p><strong>Description:</strong> <span id="node-description">...</span></p>

    <h4>Execution Metrics</h4>
    <ul>
        <li>Duration: <span id="node-duration">130s</span></li>
        <li>Tool Calls: <span id="node-tools">12</span></li>
        <li>Tokens: <span id="node-tokens">4500</span></li>
    </ul>

    <h4>Metadata</h4>
    <pre id="node-metadata">{...}</pre>
</div>
```

**Events:**
- Update on node click
- Auto-refresh every 1s if node is `.node-running`

---

#### C. Legend (Color Guide)

**UI:**
```html
<div id="legend">
    <h4>Node Types</h4>
    <div class="legend-item">
        <span class="legend-icon agent">🤖</span>
        <span>Agent Node (LLM-powered)</span>
    </div>
    <div class="legend-item">
        <span class="legend-icon compute">⚙️</span>
        <span>Compute Node (direct execution)</span>
    </div>
    <div class="legend-item">
        <span class="legend-icon condition">❓</span>
        <span>Condition Node (branching)</span>
    </div>

    <h4>Execution Status</h4>
    <div class="legend-item">
        <span class="legend-box pending"></span>
        <span>Pending</span>
    </div>
    <div class="legend-item">
        <span class="legend-box running"></span>
        <span>Running</span>
    </div>
    <div class="legend-item">
        <span class="legend-box completed"></span>
        <span>Completed</span>
    </div>
    <div class="legend-item">
        <span class="legend-box failed"></span>
        <span>Failed</span>
    </div>
</div>
```

---

#### D. Controls Toolbar

**Features:**
- **Fit to Screen**: Auto-layout and zoom to fit all nodes
- **Zoom In/Out**: Manual zoom control
- **Layout**: Switch between layouts (dagre, breadthfirst, circle)
- **Refresh**: Reload graph structure and execution status
- **Download SVG/PNG**: Export current visualization

**UI:**
```html
<div id="controls" class="toolbar">
    <button id="btn-fit">Fit</button>
    <button id="btn-zoom-in">+</button>
    <button id="btn-zoom-out">-</button>
    <select id="layout-select">
        <option value="dagre">Dagre (Hierarchical)</option>
        <option value="breadthfirst">Breadth-First</option>
        <option value="circle">Circle</option>
    </select>
    <button id="btn-refresh">Refresh</button>
    <button id="btn-download">Download SVG</button>
</div>
```

---

### 3.2 Page Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Victor Workflow Visualization                     [Fit] [+] [-] │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │                     Graph Canvas                           │   │
│  │                 (Cytoscape.js)                            │   │
│  │                                                           │   │
│  │   [🤖 Research] ──→ [❓ Check Quality]                    │   │
│  │         │                    │                            │   │
│  │         ↓                    ↓                            │   │
│  │   [⚙️ Analyze] ──→ [👤 Approval]                         │   │
│  │                                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Status: Running │ Progress: 45% │ Current: fact_check         │
│                                                                   │
├──────────────────────────┬────────────────────────────────────┤
│  Node Details             │  Legend                              │
│  ┌────────────────────┐  │  ┌────────────────────────────────┐ │
│  │ Name: Research      │  │  │ Node Types:                    │ │
│  │ Type: AgentNode    │  │  │   🤖 Agent  ⚙️ Compute         │ │
│  │ Status: Completed  │  │  │   ❓ Condition 👤 HITL          │ │
│  │ Duration: 130s     │  │  │                                │ │
│  │ Tools: 12          │  │  │ Status:                        │ │
│  │ Tokens: 4500       │  │  │   ⬜ Pending  🟡 Running        │ │
│  └────────────────────┘  │  │   🟢 Complete  🔴 Failed        │ │
│                          │  └────────────────────────────────┘ │
└──────────────────────────┴────────────────────────────────────┘
```

---

## 4. Integration with Existing Events

### 4.1 Event → Visual Update Mapping

| `WorkflowEventType` | Visual Action | Cytoscape.js Operation |
|---------------------|---------------|----------------------|
| `WORKFLOW_START` | Initialize graph | Create nodes + edges with `.node-pending` |
| `NODE_START` | Highlight node as running | `node.addClass('running')` |
| `NODE_COMPLETE` | Mark node as completed | `node.removeClass('running').addClass('completed')` |
| `NODE_ERROR` | Mark node as failed | `node.removeClass('running').addClass('failed')` |
| `WORKFLOW_COMPLETE` | Show completion message | Update status text, disable updates |
| `WORKFLOW_ERROR` | Show error message | Update status text, show error in sidebar |
| `PROGRESS_UPDATE` | Update progress bar | Update progress text/value |
| `AGENT_CONTENT` | Show streaming content (optional) | Update sidebar content preview |
| `AGENT_TOOL_CALL` | Show tool call indicator | Increment tool call counter in sidebar |

### 4.2 Event Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  WorkflowExecutor                                                │
│  - node.execute()                                                │
│  - emitter.emit_node_start()                                     │
│  - emitter.emit_node_complete()                                  │
└────────────────┬──────────────────────────────────────────────────┘
                 │ WorkflowStreamChunk
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  EventBus (ObservabilityBus)                                      │
│  - emit_lifecycle_event("node_start", {...})                      │
│  - emit_lifecycle_event("node_end", {...})                        │
└────────────────┬──────────────────────────────────────────────────┘
                 │ Event(topic="lifecycle.node_start")
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  EventBridge                                                     │
│  - Subscribes to EventBus "lifecycle.*"                          │
│  - Forwards to WebSocket clients                                 │
│  - Filters by workflow_id                                        │
└────────────────┬──────────────────────────────────────────────────┘
                 │ JSON message via WebSocket
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Browser (Cytoscape.js)                                           │
│  - ws.onmessage()                                                 │
│  - Parse event_type                                               │
│  - Update graph visualization                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 State Management

**Client-Side State:**
```javascript
const workflowState = {
    workflowId: "wf_abc123",
    status: "running",
    progress: 45.5,
    currentNode: "fact_check",
    nodes: {
        "research": {
            status: "completed",
            startedAt: "2025-01-09T10:30:05Z",
            completedAt: "2025-01-09T10:32:15Z",
            duration: 130.0,
            toolCalls: 12,
            tokens: 4500
        },
        "fact_check": {
            status: "running",
            startedAt: "2025-01-09T10:35:05Z",
            completedAt: null,
            duration: 30.0,
            toolCalls: 3,
            tokens: 1200
        }
    },
    executionPath: ["research", "analyze", "fact_check"]
};

function handleEvent(event) {
    if (event.event_type === "node_complete") {
        workflowState.nodes[event.node_id] = {
            ...workflowState.nodes[event.node_id],
            status: "completed",
            completedAt: event.timestamp,
            duration: event.metadata.duration_seconds
        };
        updateGraph();
    }
}
```

**Server-Side State (FastAPI):**
```python
# In-memory store (enhanced from fastapi_server.py)
_workflow_executions: Dict[str, Dict[str, Any]] = {
    "wf_abc123": {
        "id": "wf_abc123",
        "status": "running",
        "progress": 45.5,
        "current_node": "fact_check",
        "completed_nodes": ["research", "analyze"],
        "node_execution_path": [...],
        "nodes": {
            "research": {...},
            "analyze": {...},
            "fact_check": {...}
        }
    }
}
```

---

## 5. Data Flow

### 5.1 End-to-End Flow

```
1. User: GET /workflows/visualize/deep_research
   ↓
2. FastAPI: Render HTML template with workflow_id
   ↓
3. Browser (onload): Fetch graph structure
   GET /workflows/deep_research/graph
   ↓
4. FastAPI: Call WorkflowVisualizer → JSON
   {
     "nodes": [{"id": "research", ...}],
     "edges": [{"source": "research", "target": "analyze"}]
   }
   ↓
5. Browser: Initialize Cytoscape.js with elements
   - Create nodes with .node-pending class
   - Layout with dagre (top-down)
   ↓
6. Browser: Fetch execution status
   GET /workflows/deep_research/execution
   ↓
7. FastAPI: Query _workflow_executions store → JSON
   {
     "status": "running",
     "current_node": "fact_check",
     "completed_nodes": ["research"]
   }
   ↓
8. Browser: Update node styles
   - cy.$("#research").addClass("completed")
   - cy.$("#fact_check").addClass("running")
   ↓
9. Browser: Open WebSocket
   WS /workflows/deep_research/stream
   ↓
10. FastAPI: Subscribe EventBridge to workflow events
   - event_bridge.subscribe_workflow("deep_research", handler)
   ↓
11. WorkflowExecutor: Emit node events
   - emitter.emit_node_complete("research", ...)
   ↓
12. EventBus: Publish lifecycle.node_end event
   ↓
13. EventBridge: Forward to WebSocket clients
   ws.send_json({"event_type": "node_complete", "node_id": "research", ...})
   ↓
14. Browser: ws.onmessage → update graph
   - cy.$("#research").removeClass("running").addClass("completed")
   - Update progress: 66.6%
```

### 5.2 Graph Structure Extraction

**From StateGraph:**
```python
# In GET /workflows/{id}/graph endpoint
from victor.framework.graph import CompiledGraph

compiled_graph: CompiledGraph = load_compiled_graph(workflow_id)
schema = compiled_graph.get_graph_schema()

# Convert schema to Cytoscape.js format
nodes = [
    {
        "data": {
            "id": node_id,
            "label": node_id,
            "type": "agent"  # From metadata
        }
    }
    for node_id in schema["nodes"]
]

edges = [
    {
        "data": {
            "source": src,
            "target": tgt,
            "label": edge_data.get("label")
        }
    }
    for src, edge_list in schema["edges"].items()
    for edge_data in edge_list
]
```

**From WorkflowDefinition:**
```python
# Alternative: Use existing WorkflowVisualizer
from victor.workflows.visualization import WorkflowVisualizer
from victor.workflows.definition import WorkflowDefinition

workflow: WorkflowDefinition = load_workflow(workflow_id)
viz = WorkflowVisualizer(workflow)

# Convert internal DAGNode/DAGEdge to Cytoscape.js format
nodes = [
    {
        "data": {
            "id": n.id,
            "label": n.name,
            "type": n.node_type,
            "description": n.description
        }
    }
    for n in viz._nodes
]

edges = [
    {
        "data": {
            "source": e.source,
            "target": e.target,
            "label": e.label
        }
    }
    for e in viz._edges
]
```

---

## 6. Implementation Plan

### 6.1 File Structure

```
victor/
├── integrations/api/
│   ├── fastapi_server.py          # MODIFY: Add new endpoints
│   └── event_bridge.py            # EXISTING: Add workflow subscription
│
├── workflows/
│   ├── visualization.py           # EXISTING: Static rendering
│   └── visualization_api.py       # NEW: Dynamic API helpers
│       ├── graph_export.py        # Export StateGraph/WorkflowDefinition to JSON
│       ├── execution_tracker.py   # Track execution state per workflow
│       └── event_adapter.py       # Convert WorkflowStreamChunk to WS events
│
├── integrations/api/templates/    # NEW: Jinja2 templates
│   └── workflow_visualize.html   # Cytoscape.js UI page
│
├── observability/dashboard/
│   ├── app.py                     # MODIFY: Add workflow tab
│   └── workflow_view.py           # NEW: TUI workflow visualization widget
│
└── docs/
    └── phase-3-2-workflow-visualization-design.md  # THIS FILE
```

### 6.2 Estimated LOC

| Component | File | LOC | Complexity |
|-----------|------|-----|------------|
| **Backend** ||||
| Graph export logic | `workflows/visualization_api/graph_export.py` | 150 | Medium |
| Execution tracker | `workflows/visualization_api/execution_tracker.py` | 200 | Medium |
| Event adapter | `workflows/visualization_api/event_adapter.py` | 100 | Low |
| FastAPI endpoints | `integrations/api/fastapi_server.py` (modify) | +150 | Medium |
| EventBridge enhancement | `integrations/api/event_bridge.py` (modify) | +80 | Low |
| **Frontend** ||||
| HTML template | `integrations/api/templates/workflow_visualize.html` | 250 | Medium |
| JavaScript (inline) | Embedded in HTML | +200 | Medium |
| CSS styling | Embedded in HTML | +100 | Low |
| **TUI Enhancement** ||||
| Workflow tab widget | `observability/dashboard/workflow_view.py` | 300 | Medium |
| ASCII rendering enhance | `observability/dashboard/app.py` (modify) | +100 | Low |
| **Tests** ||||
| Unit tests | `tests/integrations/api/test_workflow_visualization.py` | 400 | High |
| Integration tests | `tests/workflows/test_visualization_api.py` | 250 | Medium |
| **TOTAL** | | **~2,280 LOC** | |

### 6.3 Dependencies

**Backend (Python):**
```toml
[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.115.0"
websockets = "^13.0"
jinja2 = "^3.1.0"  # For HTML templates
aiofiles = "^24.0"  # For static file serving

[tool.poetry.dev-dependencies]
pytest-asyncio = "^0.23.0"
```

**Frontend (CDN - no installation needed):**
- Cytoscape.js 3.28.1 (https://cdnjs.cloudflare.com/ajax/libs/cytoscape.js/3.28.1/cytoscape.min.js)
- Dagre layout (https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js)

**Optional (for enhanced layouts):**
- Cytoscape.js COSE-Bilkent (better force-directed layout)
- Cytoscape.js Euler (hierarchical layout)

---

## 7. MVP Feature List (Minimum Viable Visualization)

### Phase 1: Core Graph Rendering (Week 1)
- [x] Graph export: StateGraph → Cytoscape.js JSON format
- [x] FastAPI endpoint: `GET /workflows/{id}/graph`
- [x] HTML template with embedded Cytoscape.js
- [x] Basic rendering: Nodes + edges with dagre layout
- [x] Node styling: Color-coded by type (Agent, Compute, Condition)
- [x] Pan/zoom controls

### Phase 2: Real-Time Updates (Week 2)
- [x] Execution tracker: In-memory store of workflow state
- [x] FastAPI endpoint: `GET /workflows/{id}/execution`
- [x] WebSocket endpoint: `WS /workflows/{id}/stream`
- [x] EventBridge: Subscribe to EventBus workflow events
- [x] Client-side: WebSocket message handling
- [x] Visual updates: `.node-running`, `.node-completed`, `.node-failed`

### Phase 3: Enhanced UI (Week 3)
- [x] Node details panel: Click to show metadata
- [x] Legend: Node types and status colors
- [x] Progress bar: Real-time progress updates
- [x] Status indicators: Current node, elapsed time
- [x] Layout selector: Switch between dagre, breadthfirst, circle
- [x] Export: Download graph as SVG/PNG

### Phase 4: TUI Integration (Week 4)
- [x] Enhanced ASCII rendering: Color-coded node states
- [x] TUI workflow tab: Real-time progress view
- [x] Keyboard controls: Navigate graph, view node details
- [x] Progress bars: Visual completion indicators

### Phase 5: Polish & Performance (Week 5)
- [x] Optimizations: Lazy loading for large graphs
- [x] Error handling: Graceful degradation on failures
- [x] Documentation: API docs, user guide
- [x] Tests: Unit + integration coverage
- [x] Performance: Handle 100+ node graphs smoothly

---

## 8. Risk Mitigation

### 8.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Cytoscape.js performance** with large graphs (100+ nodes) | High | • Implement lazy loading<br>• Use hierarchical layouts<br>• Add node filtering/zoom thresholds |
| **WebSocket connection drops** during long workflows | Medium | • Auto-reconnect logic<br>• Periodic polling fallback<br>• Connection health ping/pong |
| **EventBridge bottleneck** with high event throughput | Medium | • Event batching/throttling<br>• Client-side event deduplication<br>• Per-workflow rate limiting |
| **Memory leaks** in browser with long-running visualizations | Medium | • Cleanup event listeners<br>• Limit history buffer size<br>• Periodic forced refresh |
| **TUI rendering** too slow for complex graphs | Low | • ASCII simplification for large graphs<br>• Scroll-based navigation<br>• Limit displayed depth |

### 8.2 Mitigation Strategies

**Performance Optimization:**
```javascript
// Lazy load nodes outside viewport
cy.nodes().forEach(node => {
    if (isOutsideViewport(node)) {
        node.style('display', 'none');
    }
});

// Reveal on zoom
cy.on('zoom', (event) => {
    const viewport = cy.extent();
    cy.nodes().style('display', 'element');
    cy.nodes().outside(viewport).style('display', 'none');
});
```

**WebSocket Reconnection:**
```javascript
let ws;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

function connect() {
    ws = new WebSocket(`ws://${host}/workflows/${id}/stream`);

    ws.onclose = () => {
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            setTimeout(connect, 1000 * reconnectAttempts);
        } else {
            // Fallback to polling
            setInterval(pollExecutionStatus, 2000);
        }
    };

    ws.onopen = () => {
        reconnectAttempts = 0; // Reset on successful connection
    };
}
```

---

## 9. Success Metrics

### 9.1 Performance Metrics

- **Graph Render Time:** < 500ms for 50-node graphs
- **WebSocket Latency:** < 100ms from event emission to client update
- **Memory Usage:** < 50MB browser heap for 100-node visualization
- **FPS:** ≥ 30 FPS during animations/transitions

### 9.2 User Experience Metrics

- **Setup Time:** < 1 minute from `victor chat` to seeing graph
- **Ease of Use:** No additional dependencies (uses existing FastAPI server)
- **Visual Clarity:** Node status distinguishable at a glance
- **Responsiveness:** Real-time updates feel instant (< 200ms)

### 9.3 Code Quality Metrics

- **Test Coverage:** ≥ 80% for new visualization code
- **Documentation:** 100% API endpoint coverage in OpenAPI docs
- **Code Review:** All code passes `make lint` and `make format`
- **Backward Compatibility:** No breaking changes to existing APIs

---

## 10. Future Enhancements (Beyond MVP)

### 10.1 Advanced Features

1. **Historical Playback:** Replay past workflow executions
2. **Comparative Visualization:** Side-by-side comparison of multiple runs
3. **Debug Mode:** Show detailed execution logs per node
4. **Graph Editing:** Interactive node/edge manipulation
5. **Export Formats:** PDF, DOT, Mermaid, PlantUML
6. **Collaboration:** Multi-user viewing with shared cursors

### 10.2 Integration Opportunities

- **Grafana:** Embed workflow graphs in dashboards
- **VS Code:** Add workflow visualization panel
- **CLI:** `victor workflow visualize --watch` for local dev
- **API:** Public API for external tools to query workflow state

---

## 11. References

### 11.1 Internal Code

- `victor/workflows/visualization.py` - Static rendering (1200 LOC)
- `victor/workflows/streaming.py` - Event types and chunks (270 LOC)
- `victor/integrations/api/fastapi_server.py` - FastAPI server (3262 LOC)
- `victor/observability/dashboard/app.py` - TUI dashboard (1759 LOC)
- `victor/framework/graph.py` - StateGraph DSL (2089 LOC)
- `victor/core/events/backends.py` - EventBus implementation (700 LOC)

### 11.2 External Libraries

- **Cytoscape.js Documentation:** https://js.cytoscape.org/
- **Cytoscape.js Layouts:** https://github.com/cytoscape/cytoscape.js#layouts
- **Dagre Layout:** https://github.com/dagrejs/dagre
- **FastAPI WebSocket:** https://fastapi.tiangolo.com/advanced/websockets/
- **Jinja2 Templates:** https://jinja.palletsprojects.com/

### 11.3 Design Inspiration

- **Airflow DAG Viewer:** Proven workflow visualization UI
- **GitHub Actions:** Clean, minimal graph interface
- **Prefect UI:** Excellent real-time execution tracking
- **Temporal UI:** Complex workflow state management

---

## Appendix A: API Endpoint Specifications

### A.1 GET `/workflows/{workflow_id}/graph`

**Request:**
```
GET /workflows/deep_research/graph
```

**Response (200 OK):**
```json
{
  "workflow_id": "deep_research",
  "name": "Deep Research Workflow",
  "description": "Multi-stage research with fact-checking",
  "nodes": [
    {
      "id": "research",
      "name": "Literature Review",
      "type": "AgentNode",
      "description": "Gather academic sources",
      "metadata": {
        "role": "researcher",
        "tool_budget": 15,
        "max_retries": 2
      }
    }
  ],
  "edges": [
    {
      "source": "research",
      "target": "analyze",
      "label": null,
      "conditional": false
    }
  ],
  "start_node": "research",
  "entry_point": "research",
  "total_nodes": 3
}
```

**Error Responses:**
- `404 Not Found`: Workflow ID not found
- `500 Internal Server Error`: Failed to extract graph structure

---

### A.2 GET `/workflows/{workflow_id}/execution`

**Request:**
```
GET /workflows/deep_research/execution
```

**Response (200 OK):**
```json
{
  "workflow_id": "deep_research",
  "status": "running",
  "progress": 45.5,
  "started_at": "2025-01-09T10:30:00Z",
  "current_node": "fact_check",
  "completed_nodes": ["research", "analyze"],
  "failed_nodes": [],
  "skipped_nodes": [],
  "node_execution_path": [
    {
      "node_id": "research",
      "status": "completed",
      "started_at": "2025-01-09T10:30:05Z",
      "completed_at": "2025-01-09T10:32:15Z",
      "duration_seconds": 130.0,
      "tool_calls": 12,
      "tokens_used": 4500
    }
  ],
  "total_duration_seconds": 320.0,
  "total_tool_calls": 23,
  "total_tokens": 8900
}
```

**Error Responses:**
- `404 Not Found`: Workflow execution not found
- `500 Internal Server Error`: Failed to query execution state

---

### A.3 WS `/workflows/{workflow_id}/stream`

**Connection:**
```
WS /workflows/deep_research/stream
```

**Client → Server Messages:**
```json
{"type": "subscribe", "workflow_id": "deep_research"}
{"type": "ping"}
```

**Server → Client Messages:**
```json
{"type": "subscribed", "workflow_id": "deep_research"}
{"type": "pong"}
{"type": "event", "event_type": "node_start", "node_id": "fact_check", ...}
{"type": "event", "event_type": "node_complete", "node_id": "fact_check", "progress": 66.6}
{"type": "event", "event_type": "workflow_complete", "is_final": true}
```

**Error Responses:**
- `1000 WebSocket Close`: Invalid workflow_id
- `1011 WebSocket Close`: Internal server error

---

## Appendix B: Graph JSON Schema

### B.1 Cytoscape.js Element Schema

```json
{
  "type": "object",
  "properties": {
    "elements": {
      "type": "object",
      "properties": {
        "nodes": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "data": {
                "type": "object",
                "properties": {
                  "id": {"type": "string"},
                  "label": {"type": "string"},
                  "type": {"type": "string", "enum": ["AgentNode", "ComputeNode", "ConditionNode", "ParallelNode", "HITLNode"]},
                  "status": {"type": "string", "enum": ["pending", "running", "completed", "failed", "skipped"]},
                  "description": {"type": "string"},
                  "metadata": {"type": "object"}
                },
                "required": ["id", "label", "type"]
              },
              "classes": {"type": "string"}
            }
          }
        },
        "edges": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "data": {
                "type": "object",
                "properties": {
                  "source": {"type": "string"},
                  "target": {"type": "string"},
                  "label": {"type": "string"},
                  "conditional": {"type": "boolean"}
                },
                "required": ["source", "target"]
              }
            }
          }
        }
      }
    }
  }
}
```

---

## Appendix C: Implementation Checklist

### Phase 1: Setup
- [ ] Create `victor/workflows/visualization_api/` directory
- [ ] Add `victor/integrations/api/templates/` directory
- [ ] Install Jinja2 dependency
- [ ] Create base HTML template skeleton

### Phase 2: Backend
- [ ] Implement `graph_export.py` (StateGraph → Cytoscape.js JSON)
- [ ] Implement `execution_tracker.py` (in-memory workflow state)
- [ ] Implement `event_adapter.py` (WorkflowStreamChunk → WS event)
- [ ] Add FastAPI endpoints to `fastapi_server.py`
- [ ] Enhance `event_bridge.py` with workflow subscription

### Phase 3: Frontend
- [ ] Create `workflow_visualize.html` template
- [ ] Implement Cytoscape.js graph initialization
- [ ] Implement WebSocket connection and message handling
- [ ] Implement node styling and state transitions
- [ ] Implement controls (fit, zoom, layout, refresh)

### Phase 4: TUI
- [ ] Create `workflow_view.py` TUI widget
- [ ] Add workflow tab to `app.py` dashboard
- [ ] Implement ASCII rendering enhancements
- [ ] Add keyboard navigation

### Phase 5: Testing
- [ ] Write unit tests for graph export logic
- [ ] Write integration tests for WebSocket streaming
- [ ] Write E2E tests with test workflow
- [ ] Performance testing with 100-node graphs

### Phase 6: Documentation
- [ ] Update API documentation with OpenAPI specs
- [ ] Write user guide for workflow visualization
- [ ] Document architecture and data flow
- [ ] Add screenshots and examples

---

**Document Version:** 1.0
**Last Updated:** 2025-01-09
**Status:** Ready for Review
