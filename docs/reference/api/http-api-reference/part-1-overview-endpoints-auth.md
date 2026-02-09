# HTTP API Reference - Part 1

**Part 1 of 2:** Overview, Quick Start, Endpoints, Authentication, Configuration, Error Handling, and Rate Limiting

---

## Navigation

- **[Part 1: Core API](#)** (Current)
- [Part 2: Examples, Testing, Deployment, Monitoring](part-2-examples-testing-deployment.md)
- [**Complete Reference](../http-api.md)**

---

REST API for Victor's AI coding assistant capabilities.

## Overview

The HTTP API provides language-agnostic access to Victor's features through a REST interface.

**Start server**:
```bash
victor serve --port 8080
```text

**Base URL**: `http://localhost:8080/api/v1`

**Authentication**: API key in `X-API-Key` header (optional, configurable)

---

## Quick Start

### Basic Chat Request

```bash
curl http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, Victor!",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514"
  }'
```

**Response**:
```json
{
  "success": true,
  "response": "Hello! How can I help you today?",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "tokens_used": 25,
  "duration_ms": 1250
}
```text

### Streaming Chat

```bash
curl http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
```

[Content continues through Rate Limiting...]


**Reading Time:** 1 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 2: Examples, Testing, Deployment, Monitoring](part-2-examples-testing-deployment.md)**
