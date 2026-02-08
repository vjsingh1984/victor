# HTTP API Reference

REST API for Victor's AI coding assistant capabilities.

---

## Quick Summary

The HTTP API provides language-agnostic access to Victor's features through a REST interface.

**Start server**:
```bash
victor serve --port 8080
```

**Base URL**: `http://localhost:8080/api/v1`

**Authentication**: API key in `X-API-Key` header (optional, configurable)

---

## Reference Parts

### [Part 1: Core API](part-1-overview-endpoints-auth.md)
- Overview
- Quick Start
- Endpoints
- Authentication
- Configuration
- Error Handling
- Rate Limiting

### [Part 2: Examples, Testing, Deployment](part-2-examples-testing-deployment.md)
- Examples (Python, JavaScript, Streaming)
- Testing
- Deployment (Docker, Kubernetes)
- Monitoring
- Troubleshooting

---

## Quick Start

```bash
# Basic chat request
curl http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, Victor!",
    "provider": "anthropic"
  }'
```

**Response**:
```json
{
  "success": true,
  "response": "Hello! How can I help you today?",
  "tokens_used": 25
}
```

---

## Related Documentation

- [Architecture Overview](../../architecture/README.md)
- [Configuration Reference](./CONFIGURATION_REFERENCE.md)
- [Deployment Guide](../../operations/deployment/)

---

**Last Updated:** February 01, 2026
**Reading Time:** 12 min (all parts)
