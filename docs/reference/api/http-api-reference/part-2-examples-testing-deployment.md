# HTTP API Reference - Part 2

**Part 2 of 2:** Examples, Testing, Deployment, Monitoring, and Troubleshooting

---

## Navigation

- [Part 1: Core API](part-1-overview-endpoints-auth.md)
- **[Part 2: Examples, Testing, Deployment](#)** (Current)
- [**Complete Reference](../http-api.md)**

---

## Examples

### Python Client

```python
import requests

API_BASE = "http://localhost:8080/api/v1"

def chat(message: str, provider: str = "anthropic"):
    """Send chat message."""
    response = requests.post(
        f"{API_BASE}/chat",
        json={"message": message, "provider": provider},
        headers={"X-API-Key": "your-api-key"}
    )
    return response.json()

# Use
result = chat("Help me debug this code")
print(result["response"])
```

### JavaScript Client

```javascript
const API_BASE = "http://localhost:8080/api/v1";

async function chat(message, provider = "anthropic") {
    const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-API-Key": "your-api-key"
        },
        body: JSON.stringify({ message, provider })
    });
    return await response.json();
}

// Use
const result = await chat("Help me debug this code");
console.log(result.response);
```

### Streaming Response

```python
import requests
import json

def stream_chat(message: str):
    """Stream chat response."""
    response = requests.post(
        f"{API_BASE}/chat/stream",
        json={"message": message},
        stream=True
    )

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk["content"], end="")

# Use
stream_chat("Explain async/await in Python")
```

---

## Testing

### Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from victor.api.server import app

client = TestClient(app)

def test_chat_endpoint():
    """Test chat endpoint."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete API workflow."""
    # Start server
    # Make requests
    # Verify responses
    pass
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["victor", "serve", "--port", "8080"]
```

### Kubernetes

```yaml
apiVersion: v1
kind: Service
metadata:
  name: victor-api
spec:
  selector:
    app: victor
  ports:
  - port: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: victor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: victor
  template:
    metadata:
      labels:
        app: victor
    spec:
      containers:
      - name: victor
        image: victor:latest
        ports:
        - containerPort: 8080
```

---

## Monitoring

### Metrics

The API exposes metrics at `/metrics`:

```bash
curl http://localhost:8080/metrics
```

**Key metrics**:
- `api_requests_total` - Total API requests
- `api_request_duration_ms` - Request duration
- `api_errors_total` - Total errors
- `api_active_connections` - Active connections

### Health Check

```bash
curl http://localhost:8080/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "0.5.0",
  "uptime": 3600
}
```

---

## Troubleshooting

### Connection Refused

**Problem**: Cannot connect to API server.

**Solutions**:
1. **Check server is running**:
   ```bash
   ps aux | grep victor
   ```

2. **Check port**:
   ```bash
   lsof -i :8080
   ```

3. **Check firewall**:
   ```bash
   sudo ufw allow 8080
   ```

### Authentication Errors

**Problem**: 401 Unauthorized errors.

**Solutions**:
1. **Check API key**:
   ```bash
   curl -H "X-API-Key: your-key" http://localhost:8080/api/v1/chat
   ```

2. **Verify API key format**:
   - Should be in header `X-API-Key`
   - No extra whitespace

### Rate Limiting

**Problem**: 429 Too Many Requests.

**Solutions**:
1. **Reduce request frequency**
2. **Implement exponential backoff**:
   ```python
   import time

   for attempt in range(3):
       response = requests.post(url, json=data)
       if response.status_code == 429:
           time.sleep(2 ** attempt)  # Exponential backoff
       else:
           break
   ```

---

## Additional Resources

- [Architecture Overview](../../../architecture/README.md)
- [Configuration Reference](../CONFIGURATION_REFERENCE.md)
- [Deployment Guide](../../../operations/deployment/)

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 2 min
**Last Updated:** February 01, 2026
