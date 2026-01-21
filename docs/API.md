# Victor AI API Documentation

**Version:** 0.5.0
**Last Updated:** 2025-01-20
**Base URL:** `http://localhost:8000`

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core APIs](#core-apis)
4. [Persona Management API](#persona-management-api)
5. [Memory Systems API](#memory-systems-api)
6. [Multimodal API](#multimodal-api)
7. [Security API](#security-api)
8. [Workflow API](#workflow-api)
9. [Provider API](#provider-api)
10. [Tool API](#tool-api)
11. [Examples](#examples)
12. [Error Handling](#error-handling)

---

## Overview

Victor AI provides a comprehensive REST API for all system functionality, including agent orchestration, vertical operations, multimodal processing, security, and workflow management.

### API Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                              │
│                  (FastAPI + Uvicorn)                        │
├─────────────────────────────────────────────────────────────┤
│  Core APIs     │  Advanced Features  │  Vertical APIs        │
│  - Chat        │  - Personas         │  - Coding             │
│  - Completions │  - Memory           │  - DevOps             │
│  - Streaming  │  - Multimodal       │  - RAG                │
│                │  - Security         │  - Research           │
│                │  - Workflows        │  - DataAnalysis       │
└─────────────────────────────────────────────────────────────┘
```

### Response Format

All API responses follow this structure:

```json
{
  "success": true,
  "data": { },
  "error": null,
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2025-01-20T10:00:00Z",
    "duration_ms": 125
  }
}
```

### Pagination

List endpoints support pagination:

```json
{
  "items": [ ],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "total_items": 150,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false
  }
}
```

---

## Authentication

### API Key Authentication

All API requests require authentication via API key.

**Header:**

```http
Authorization: Bearer YOUR_API_KEY
```

**Generate API Key:**

```bash
victor api-keys create --user-id=alice --description="Dev key"
# Output: sk-vict-abc123...
```

### Session Authentication

For interactive sessions:

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "alice",
  "password": "secure_password"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "session_id": "sess_xyz789",
    "expires_at": "2025-01-20T11:00:00Z",
    "user": {
      "id": "alice",
      "roles": ["developer"]
    }
  }
}
```

---

## Core APIs

### Chat Completions

**Endpoint:** `POST /api/v1/chat/completions`

Create a chat completion request.

**Request:**

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-5",
  "messages": [
    {"role": "user", "content": "Hello, Victor!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "tools": ["read_file", "write_file"],
  "stream": false
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | string | Yes | LLM provider name |
| `model` | string | Yes | Model identifier |
| `messages` | array | Yes | Conversation messages |
| `temperature` | float | No | Randomness (0.0-1.0) |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `tools` | array | No | Allowed tools |
| `stream` | boolean | No | Enable streaming |

**Response:**

```json
{
  "success": true,
  "data": {
    "id": "chatcmpl_abc123",
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 20,
      "total_tokens": 30
    },
    "model": "claude-sonnet-4-5",
    "provider": "anthropic"
  }
}
```

### Streaming Chat

**Endpoint:** `POST /api/v1/chat/completions` with `stream: true`

**Request:** Same as chat completions with `"stream": true`

**Response (Server-Sent Events):**

```
data: {"type":"token","token":"Hello","index":0}

data: {"type":"token","token":",","index":1}

data: {"type":"token","token":" Victor","index":2}

data: {"type":"done","usage":{"prompt_tokens":10,"completion_tokens":3}}
```

**Python Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True,
    },
    stream=True,
)

for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
```

### Code Completion

**Endpoint:** `POST /api/v1/code/completions`

Get code completion suggestions.

**Request:**

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-5",
  "code": "def hello_world():\n    ",
  "language": "python",
  "cursor_position": 28,
  "max_suggestions": 3
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "suggestions": [
      {
        "text": "print('Hello, World!')",
        "confidence": 0.95
      },
      {
        "text": "return 'Hello, World!'",
        "confidence": 0.85
      }
    ]
  }
}
```

---

## Persona Management API

### List Personas

**Endpoint:** `GET /api/v1/personas`

List all available personas.

**Response:**

```json
{
  "success": true,
  "data": {
    "personas": [
      {
        "id": "senior-developer",
        "name": "Senior Developer",
        "description": "Expert coding mentor",
        "traits": {
          "experience": "senior",
          "style": "mentoring",
          "communication": "detailed"
        },
        "expertise": ["python", "architecture", "best-practices"]
      },
      {
        "id": "security-specialist",
        "name": "Security Specialist",
        "description": "Security-focused reviewer",
        "traits": {
          "focus": "security",
          "style": "detailed"
        },
        "expertise": ["owasp", "cryptography", "penetration-testing"]
      }
    ]
  }
}
```

### Load Persona

**Endpoint:** `GET /api/v1/personas/{persona_id}`

Load a specific persona.

**Response:**

```json
{
  "success": true,
  "data": {
    "id": "senior-developer",
    "name": "Senior Developer",
    "system_prompt": "You are a senior developer with 10+ years...",
    "traits": { },
    "expertise": [ ]
  }
}
```

### Create Custom Persona

**Endpoint:** `POST /api/v1/personas`

Create a custom persona.

**Request:**

```json
{
  "name": "ML Engineer",
  "description": "Machine learning specialist",
  "traits": {
    "focus": "ml",
    "style": "analytical"
  },
  "expertise": [
    "tensorflow",
    "pytorch",
    "mlops",
    "data-engineering"
  ],
  "system_prompt": "You are an ML engineer specializing in..."
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "id": "ml-engineer-custom",
    "name": "ML Engineer",
    "created_at": "2025-01-20T10:00:00Z"
  }
}
```

### Adapt Persona

**Endpoint:** `POST /api/v1/personas/{persona_id}/adapt`

Adapt persona to specific context.

**Request:**

```json
{
  "context": {
    "task_type": "code_review",
    "urgency": "high",
    "user_preference": "concise"
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "persona": { },
    "adaptations": [
      "Increased directness for high urgency",
      "Simplified communication for concise preference"
    ]
  }
}
```

### Merge Personas

**Endpoint:** `POST /api/v1/personas/merge`

Merge multiple personas.

**Request:**

```json
{
  "persona_ids": [
    "senior-developer",
    "security-specialist"
  ],
  "strategy": "weighted",
  "weights": {
    "senior-developer": 0.6,
    "security-specialist": 0.4
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "merged_persona": {
      "id": "merged-senior-dev-security",
      "name": "Senior Developer + Security",
      "traits": { },
      "expertise": [
        "python",
        "architecture",
        "owasp",
        "cryptography"
      ]
    },
    "conflicts": [
      {
        "attribute": "communication_style",
        "resolution": "weighted_average",
        "values": ["detailed", "technical"]
      }
    ]
  }
}
```

---

## Memory Systems API

### Episodic Memory

#### Store Memory

**Endpoint:** `POST /api/v1/memory/episodic`

Store an episodic memory.

**Request:**

```json
{
  "task": "Fix authentication bug in login flow",
  "context": {
    "error": "JWT validation failing",
    "location": "src/auth.py:42"
  },
  "outcome": "Fixed by updating JWT secret validation",
  "success": true,
  "duration_seconds": 120,
  "metadata": {
    "project": "myapp",
    "user": "alice"
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "memory_id": "episodic_abc123",
    "created_at": "2025-01-20T10:00:00Z",
    "embedding": [0.1, 0.2, ...]
  }
}
```

#### Retrieve Similar Memories

**Endpoint:** `POST /api/v1/memory/episodic/search`

Find similar past experiences.

**Request:**

```json
{
  "query": "authentication error JWT validation",
  "top_k": 5,
  "similarity_threshold": 0.7
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "memories": [
      {
        "memory_id": "episodic_xyz789",
        "task": "Fix authentication bug",
        "outcome": "Fixed JWT validation",
        "similarity": 0.85,
        "created_at": "2025-01-15T10:00:00Z"
      }
    ]
  }
}
```

#### Consolidate Memories

**Endpoint:** `POST /api/v1/memory/episodic/consolidate`

Extract patterns from memories.

**Response:**

```json
{
  "success": true,
  "data": {
    "patterns": [
      {
        "pattern": "Authentication errors often require JWT validation update",
        "frequency": 5,
        "confidence": 0.92
      }
    ]
  }
}
```

### Semantic Memory

#### Learn Fact

**Endpoint:** `POST /api/v1/memory/semantic`

Learn a new fact.

**Request:**

```json
{
  "fact": "Python 3.12 released on October 2, 2023",
  "confidence": 0.95,
  "source": "official_blog",
  "metadata": {
    "category": "release",
    "importance": "high"
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "fact_id": "semantic_def456",
    "created_at": "2025-01-20T10:00:00Z"
  }
}
```

#### Query Facts

**Endpoint:** `POST /api/v1/memory/semantic/query`

Query semantic knowledge base.

**Request:**

```json
{
  "query": "Python 3.12 release date",
  "top_k": 5
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "facts": [
      {
        "fact": "October 2, 2023",
        "confidence": 0.95,
        "source": "official_blog",
        "metadata": { }
      }
    ]
  }
}
```

#### Build Knowledge Graph

**Endpoint:** `GET /api/v1/memory/semantic/graph`

Build and return knowledge graph.

**Response:**

```json
{
  "success": true,
  "data": {
    "nodes": [
      {"id": "python-3.12", "type": "concept"},
      {"id": "october-2-2023", "type": "date"}
    ],
    "edges": [
      {
        "source": "python-3.12",
        "target": "october-2-2023",
        "relation": "released_on",
        "confidence": 0.95
      }
    ]
  }
}
```

---

## Multimodal API

### Vision Agent

#### Analyze Image

**Endpoint:** `POST /api/v1/multimodal/vision/analyze`

Analyze image with vision LLM.

**Request (multipart/form-data):**

```http
POST /api/v1/multimodal/vision/analyze
Content-Type: multipart/form-data

provider: anthropic
image: [file]
query: Describe this UI layout and identify any usability issues
```

**Response:**

```json
{
  "success": true,
  "data": {
    "analysis": "The image shows a login form with email and password fields...",
    "provider": "anthropic",
    "model": "claude-3-opus",
    "image_format": "png",
    "image_size": 102400
  }
}
```

#### Extract Data from Plot

**Endpoint:** `POST /api/v1/multimodal/vision/extract-plot`

Extract data from chart/graph.

**Request (multipart/form-data):**

```http
POST /api/v1/multimodal/vision/extract-plot
Content-Type: multipart/form-data

image: [file]
```

**Response:**

```json
{
  "success": true,
  "data": {
    "chart_type": "line_chart",
    "data": [
      {"x": "Jan", "y": 100},
      {"x": "Feb", "y": 150}
    ],
    "labels": ["Revenue", "Month"],
    "confidence": 0.92
  }
}
```

#### OCR Extraction

**Endpoint:** `POST /api/v1/multimodal/vision/ocr`

Extract text from image.

**Request (multipart/form-data):**

```http
POST /api/v1/multimodal/vision/ocr
Content-Type: multipart/form-data

image: [file]
language: en
```

**Response:**

```json
{
  "success": true,
  "data": {
    "text": "Invoice #1234\nDate: 2025-01-20\nTotal: $500.00",
    "confidence": 0.98,
    "language": "en"
  }
}
```

#### Object Detection

**Endpoint:** `POST /api/v1/multimodal/vision/detect-objects`

Detect objects in image.

**Request (multipart/form-data):**

```http
POST /api/v1/multimodal/vision/detect-objects
Content-Type: multipart/form-data

image: [file]
```

**Response:**

```json
{
  "success": true,
  "data": {
    "objects": [
      {
        "label": "person",
        "bounding_box": [10, 20, 100, 200],
        "confidence": 0.95
      },
      {
        "label": "car",
        "bounding_box": [200, 50, 400, 300],
        "confidence": 0.88
      }
    ]
  }
}
```

### Audio Agent

#### Transcribe Audio

**Endpoint:** `POST /api/v1/multimodal/audio/transcribe`

Transcribe audio to text.

**Request (multipart/form-data):**

```http
POST /api/v1/multimodal/audio/transcribe
Content-Type: multipart/form-data

audio: [file]
language: en
```

**Response:**

```json
{
  "success": true,
  "data": {
    "transcript": "Hello, this is a test transcription.",
    "language": "en",
    "confidence": 0.98,
    "duration": 5.2,
    "provider": "whisper"
  }
}
```

#### Speaker Diarization

**Endpoint:** `POST /api/v1/multimodal/audio/diarize`

Identify speakers in audio.

**Request (multipart/form-data):**

```http
POST /api/v1/multimodal/audio/diarize
Content-Type: multipart/form-data

audio: [file]
min_speakers: 2
max_speakers: 5
```

**Response:**

```json
{
  "success": true,
  "data": {
    "segments": [
      {
        "speaker": "SPEAKER_1",
        "start": 0.0,
        "end": 5.2,
        "text": "Hello, how are you?",
        "confidence": 0.95
      },
      {
        "speaker": "SPEAKER_2",
        "start": 5.5,
        "end": 10.0,
        "text": "I'm doing well, thanks!",
        "confidence": 0.92
      }
    ],
    "num_speakers": 2
  }
}
```

---

## Security API

### Authorization

#### Check Permission

**Endpoint:** `POST /api/v1/security/authorize`

Check if user has permission.

**Request:**

```json
{
  "user_id": "alice",
  "resource": "tools",
  "action": "execute",
  "context": {
    "tool_name": "read_file"
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "allowed": true,
    "reason": "User has 'developer' role with 'tools:*' permission",
    "matched_policy": "developer_role"
  }
}
```

### User Management

#### Create User

**Endpoint:** `POST /api/v1/security/users`

Create a new user.

**Request:**

```json
{
  "username": "bob",
  "email": "bob@example.com",
  "attributes": {
    "department": "engineering",
    "clearance": "confidential"
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "user_id": "bob",
    "created_at": "2025-01-20T10:00:00Z"
  }
}
```

#### Grant Role

**Endpoint:** `POST /api/v1/security/users/{user_id}/roles`

Grant role to user.

**Request:**

```json
{
  "role": "developer"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "user_id": "bob",
    "roles": ["developer"],
    "granted_at": "2025-01-20T10:00:00Z"
  }
}
```

### Penetration Testing

#### Run Security Audit

**Endpoint:** `POST /api/v1/security/audit`

Run penetration testing.

**Request:**

```json
{
  "categories": [
    "prompt_injection",
    "authorization_bypass",
    "code_injection"
  ],
  "safe_mode": true
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "audit_id": "audit_abc123",
    "risk_score": 2.5,
    "findings": [
      {
        "category": "prompt_injection",
        "severity": "medium",
        "description": "System vulnerable to role manipulation",
        "recommendation": "Implement stricter input validation"
      }
    ],
    "tested_payloads": 35,
    "vulnerabilities_found": 1
  }
}
```

---

## Workflow API

### Create Workflow

**Endpoint:** `POST /api/v1/workflows`

Create a new workflow.

**Request:**

```json
{
  "name": "code_review_workflow",
  "description": "Comprehensive code review",
  "nodes": [
    {
      "id": "analyze",
      "type": "agent",
      "goal": "Analyze code structure",
      "tools": ["ast_analysis"]
    },
    {
      "id": "review",
      "type": "agent",
      "goal": "Review code quality",
      "tools": ["code_review"]
    }
  ],
  "edges": [
    {"from": "START", "to": "analyze"},
    {"from": "analyze", "to": "review"},
    {"from": "review", "to": "END"}
  ]
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "workflow_id": "wf_xyz789",
    "created_at": "2025-01-20T10:00:00Z"
  }
}
```

### Execute Workflow

**Endpoint:** `POST /api/v1/workflows/{workflow_id}/execute`

Execute a workflow.

**Request:**

```json
{
  "input": {
    "file_path": "/path/to/file.py"
  },
  "options": {
    "stream": false,
    "checkpoint": true
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "execution_id": "exec_abc123",
    "status": "running",
    "result": null
  }
}
```

### Get Execution Status

**Endpoint:** `GET /api/v1/workflows/executions/{execution_id}`

Get workflow execution status.

**Response:**

```json
{
  "success": true,
  "data": {
    "execution_id": "exec_abc123",
    "status": "completed",
    "result": {
      "analyze": { },
      "review": { }
    },
    "duration_seconds": 15.5
  }
}
```

---

## Provider API

### List Providers

**Endpoint:** `GET /api/v1/providers`

List all available providers.

**Response:**

```json
{
  "success": true,
  "data": {
    "providers": [
      {
        "name": "anthropic",
        "display_name": "Anthropic",
        "models": [
          "claude-sonnet-4-5",
          "claude-3-opus",
          "claude-3-haiku"
        ],
        "features": ["chat", "streaming", "tools"]
      },
      {
        "name": "openai",
        "display_name": "OpenAI",
        "models": [
          "gpt-4-turbo",
          "gpt-3.5-turbo"
        ],
        "features": ["chat", "streaming", "tools"]
      }
    ]
  }
}
```

### Test Provider Connectivity

**Endpoint:** `POST /api/v1/providers/{provider_name}/test`

Test provider connection.

**Response:**

```json
{
  "success": true,
  "data": {
    "provider": "anthropic",
    "status": "healthy",
    "latency_ms": 125,
    "models_available": 3
  }
}
```

---

## Tool API

### List Tools

**Endpoint:** `GET /api/v1/tools`

List all available tools.

**Query Parameters:**
- `vertical`: Filter by vertical (coding, devops, rag, research, dataanalysis)
- `cost_tier`: Filter by cost tier (free, low, medium, high)

**Response:**

```json
{
  "success": true,
  "data": {
    "tools": [
      {
        "name": "read_file",
        "description": "Read file contents",
        "vertical": "coding",
        "cost_tier": "free",
        "parameters": {
          "file_path": {
            "type": "string",
            "required": true
          }
        }
      },
      {
        "name": "docker_build",
        "description": "Build Docker image",
        "vertical": "devops",
        "cost_tier": "medium",
        "parameters": { }
      }
    ],
    "total": 55
  }
}
```

### Execute Tool

**Endpoint:** `POST /api/v1/tools/execute`

Execute a tool.

**Request:**

```json
{
  "tool_name": "read_file",
  "parameters": {
    "file_path": "/path/to/file.py"
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "result": {
      "content": "def hello():\n    print('Hello, World!')",
      "language": "python",
      "line_count": 2
    },
    "execution_time_ms": 15,
    "tool_name": "read_file"
  }
}
```

---

## Examples

### Complete Agentic Workflow

```python
import requests

API_KEY = "sk-vict-abc123"
BASE_URL = "http://localhost:8000"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 1. Create hierarchical plan
plan_response = requests.post(
    f"{BASE_URL}/api/v1/planning/decompose",
    headers=headers,
    json={
        "task": "Add OAuth2 authentication",
        "context": {"project": "myapp"}
    }
)
plan = plan_response.json()["data"]

# 2. Execute with memory
execute_response = requests.post(
    f"{BASE_URL}/api/v1/planning/execute",
    headers=headers,
    json={
        "plan_id": plan["plan_id"],
        "use_memory": True,
        "use_skills": True
    }
)

# 3. Store experience
memory_response = requests.post(
    f"{BASE_URL}/api/v1/memory/episodic",
    headers=headers,
    json={
        "task": "Add OAuth2",
        "outcome": execute_response.json()["data"],
        "success": True
    }
)

print(f"Execution complete: {execute_response.json()}")
```

### Multimodal Code Review

```python
# 1. Load UI designer persona
persona_response = requests.get(
    f"{BASE_URL}/api/v1/personas/ui-designer",
    headers=headers
)
persona = persona_response.json()["data"]

# 2. Analyze UI mockup
with open("mockup.png", "rb") as f:
    vision_response = requests.post(
        f"{BASE_URL}/api/v1/multimodal/vision/analyze",
        headers={"Authorization": f"Bearer {API_KEY}"},
        data={"image": f, "query": "Evaluate this UI for accessibility"}
    )
    analysis = vision_response.json()["data"]

# 3. Code review with persona
review_response = requests.post(
    f"{BASE_URL}/api/v1/coding/review",
    headers=headers,
    json={
        "files": ["app.py", "styles.css"],
        "persona_id": persona["id"],
        "context": {"ui_analysis": analysis}
    }
)

print(f"Review complete: {review_response.json()}")
```

### Security Audit

```python
# Run security audit
audit_response = requests.post(
    f"{BASE_URL}/api/v1/security/audit",
    headers=headers,
    json={
        "categories": "all",
        "safe_mode": True
    }
)

audit = audit_response.json()["data"]

print(f"Risk Score: {audit['risk_score']}/10")
print(f"Vulnerabilities Found: {len(audit['findings'])}")

# Generate report
report_response = requests.post(
    f"{BASE_URL}/api/v1/security/audit/{audit['audit_id']}/report",
    headers=headers,
    json={"format": "markdown"}
)

with open("security_report.md", "w") as f:
    f.write(report_response.json()["data"]["report"])
```

---

## Error Handling

### Error Response Format

All errors follow this structure:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameter",
    "details": {
      "field": "temperature",
      "issue": "Must be between 0.0 and 1.0"
    }
  },
  "request_id": "req_abc123",
  "timestamp": "2025-01-20T10:00:00Z"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid input parameters |
| `AUTHENTICATION_ERROR` | Invalid API key or session |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `PROVIDER_ERROR` | LLM provider error |
| `TOOL_EXECUTION_ERROR` | Tool execution failed |
| `RATE_LIMIT_ERROR` | Rate limit exceeded |
| `RESOURCE_NOT_FOUND` | Requested resource not found |

---

## Additional Resources

- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Operations Guide**: [OPERATIONS.md](OPERATIONS.md)
- **Features Guide**: [FEATURES.md](FEATURES.md)
- **Security Guide**: [SECURITY.md](SECURITY.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

**Document Version:** 1.0.0
**Last Reviewed:** 2025-01-20
**Next Review:** 2025-02-20
