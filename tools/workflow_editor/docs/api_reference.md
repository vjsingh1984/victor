# API Reference - Workflow Editor Backend

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health & Info

#### GET /

Root endpoint - API information

**Response:**
```json
{
  "name": "Victor Workflow Editor API",
  "version": "0.1.0",
  "status": "running"
}
```

#### GET /health

Health check endpoint

**Response:**
```json
{
  "status": "healthy"
}
```

---

### Workflow Operations

#### POST /api/workflows/validate

Validate a workflow graph

**Request Body:**
```json
{
  "nodes": [
    {
      "id": "agent_1",
      "type": "agent",
      "name": "Research Agent",
      "config": {
        "role": "researcher",
        "goal": "Find information",
        "tool_budget": 25
      },
      "position": { "x": 100, "y": 100 }
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": "agent_1",
      "target": "agent_2"
    }
  ],
  "metadata": {}
}
```

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": []
}
```

#### POST /api/workflows/compile

Compile a workflow from YAML content

**Request Body:**
```json
{
  "yaml_content": "workflows:\n  example:\n    nodes: []",
  "workflow_name": "my_workflow"
}
```

**Response:**
```json
{
  "success": true,
  "workflow_id": "my_workflow",
  "graph_schema": {
    "nodes": [],
    "edges": []
  },
  "errors": []
}
```

#### POST /api/workflows/export/yaml

Export workflow graph to YAML format

**Request Body:**
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {}
}
```

**Response:**
```json
{
  "yaml_content": "workflows:\n  example:\n    ...",
  "success": true
}
```

#### POST /api/workflows/import/yaml

Import workflow from YAML file

**Request:** Multipart form data with YAML file

**Response:**
```json
{
  "graph": {
    "nodes": [...],
    "edges": [...]
  },
  "success": true
}
```

---

### Node Types

#### GET /api/nodes/types

Get available node types and their configuration schemas

**Response:**
```json
{
  "agent": {
    "name": "Agent Node",
    "description": "LLM-powered agent with role and goal",
    "color": "#E3F2FD",
    "config_schema": {
      "role": { "type": "string", "required": true },
      "goal": { "type": "string", "required": true },
      "tool_budget": { "type": "integer", "default": 25 }
    }
  },
  "compute": { ... },
  "team": { ... }
}
```

---

### Team Formations

#### GET /api/formations

Get available team formation types

**Response:**
```json
{
  "parallel": {
    "name": "Parallel Formation",
    "description": "All members work simultaneously",
    "icon": "||",
    "best_for": ["Independent analysis", "Multi-perspective review"],
    "communication_style": "structured"
  },
  "sequential": { ... },
  "pipeline": { ... },
  "hierarchical": { ... },
  "consensus": { ... }
}
```

---

## Data Models

### WorkflowNode

```typescript
interface WorkflowNode {
  id: string;
  type: NodeType;
  name: string;
  config: Record<string, unknown>;
  position: { x: number; y: number };
}
```

### WorkflowEdge

```typescript
interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}
```

### WorkflowGraph

```typescript
interface WorkflowGraph {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  metadata?: Record<string, unknown>;
}
```

### TeamMemberConfig

```typescript
interface TeamMemberConfig {
  id: string;
  role: string;
  goal: string;
  tool_budget: number;
  tools?: string[];
  backstory?: string;
  expertise?: string[];
  personality?: string;
}
```

### TeamNodeConfig

```typescript
interface TeamNodeConfig {
  id: string;
  name: string;
  goal: string;
  formation: 'parallel' | 'sequential' | 'pipeline' | 'hierarchical' | 'consensus';
  max_iterations: number;
  timeout_seconds?: number;
  members: TeamMemberConfig[];
}
```

---

## Error Responses

All endpoints may return error responses:

```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid input)
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

---

## Examples

### Validate Workflow

```bash
curl -X POST http://localhost:8000/api/workflows/validate \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [{"id": "1", "type": "agent", "name": "Test", "config": {}, "position": {"x": 0, "y": 0}}],
    "edges": [],
    "metadata": {}
  }'
```

### Compile Workflow

```bash
curl -X POST http://localhost:8000/api/workflows/compile \
  -H "Content-Type: application/json" \
  -d '{
    "yaml_content": "workflows:\n  test:\n    nodes: []",
    "workflow_name": "test"
  }'
```

### Get Node Types

```bash
curl http://localhost:8000/api/nodes/types
```

### Get Formations

```bash
curl http://localhost:8000/api/formations
```
