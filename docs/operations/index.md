# Operations Guide

Complete guide for deploying, monitoring, and maintaining Victor in production.

## Overview

This section covers everything you need to run Victor in production environments, from deployment to monitoring, security, and performance optimization.

**New to Victor?** Start with [User Guide](../user-guide/) or [Development Guide](../development/).

## Quick Links

| Topic | Documentation | Description |
|-------|--------------|-------------|
| **Deployment** | [Docker](deployment/docker.md) | Container deployment |
| **Deployment** | [Kubernetes](deployment/kubernetes.md) | K8s deployment |
| **Deployment** | [Cloud](deployment/cloud.md) | AWS, GCP, Azure |
| **Deployment** | [Air-Gapped](deployment/air-gapped.md) | Offline deployment |
| **Monitoring** | [Metrics](monitoring/metrics.md) | Metrics & alerting |
| **Monitoring** | [Logging](monitoring/logging.md) | Logging config |
| **Monitoring** | [Health Checks](monitoring/health-checks.md) | Health monitoring |
| **Security** | [Overview](security/overview.md) | Security architecture |
| **Security** | [API Keys](security/api-keys.md) | Key management |
| **Security** | [Compliance](security/compliance.md) | SOC2, GDPR |
| **Security** | [Auditing](security/auditing.md) | Audit logs |
| **Performance** | [Optimization](performance/optimization.md) | Performance tuning |
| **Performance** | [Scaling](performance/scaling.md) | Scaling strategies |
| **Performance** | [Benchmarks](performance/benchmarks.md) | Benchmark results |

## Quick Start

### Docker Deployment (Recommended)

```bash
# Pull image
docker pull ghcr.io/vjsingh1984/victor:latest

# Run Victor
docker run -it \
  -v ~/.victor:/root/.victor \
  -p 8080:8080 \
  ghcr.io/vjsingh1984/victor:latest

# With API server
docker run -it \
  -e ANTHROPIC_API_KEY=sk-... \
  -v ~/.victor:/root/.victor \
  -p 8080:8080 \
  ghcr.io/vjsingh1984/victor:latest \
  victor serve --host 0.0.0.0 --port 8080
```

[Full Docker Guide →](deployment/docker.md)

### Kubernetes Deployment

```yaml
# deployment.yaml
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
        image: ghcr.io/vjsingh1984/victor:latest
        ports:
        - containerPort: 8080
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: victor-secrets
              key: anthropic-api-key
        volumeMounts:
        - name: config
          mountPath: /root/.victor
      volumes:
      - name: config
        persistentVolumeClaim:
          claimName: victor-config
```

```bash
kubectl apply -f deployment.yaml
```

[Full Kubernetes Guide →](deployment/kubernetes.md)

## Deployment

### Deployment Options

| Platform | Complexity | Scalability | Cost | Guide |
|----------|------------|-------------|------|-------|
| **Docker** | Low | Low | Low | [Docker →](deployment/docker.md) |
| **Kubernetes** | High | High | Medium | [K8s →](deployment/kubernetes.md) |
| **AWS** | Medium | High | High | [AWS →](deployment/cloud.md) |
| **GCP** | Medium | High | High | [GCP →](deployment/cloud.md) |
| **Azure** | Medium | High | High | [Azure →](deployment/cloud.md) |
| **Air-Gapped** | Medium | Low | Low | [Air-Gapped →](deployment/air-gapped.md) |

### Configuration Management

**Environment Variables**:
```bash
# Provider configuration
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...

# Victor configuration
export VICTOR_LOG_LEVEL=info
export VICTOR_CACHE_ENABLED=true
export VICTOR_MAX_WORKERS=4
```

**Config Files** (`~/.victor/`):
```yaml
# profiles.yaml
profiles:
  production:
    provider: anthropic
    model: claude-sonnet-4-20250514
    max_tokens: 4096

# config.yaml
logging:
  level: info
  file: /var/log/victor.log

cache:
  enabled: true
  ttl: 3600
```

### High Availability

**Load Balancing** (multiple instances):
```yaml
# docker-compose.yml
version: '3'
services:
  victor:
    image: ghcr.io/vjsingh1984/victor:latest
    deploy:
      replicas: 3
    environment:
      - VICTOR_REDIS_HOST=redis
      - VICTOR_REDIS_PORT=6379
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

**Session Affinity**:
- For stateful conversations, configure load balancer with session affinity
- Use shared storage (Redis, PostgreSQL) for conversation state
- Implement health checks for failover

[Full Deployment Guide →](deployment/)

## Monitoring

### Metrics Collection

Victor exposes metrics for monitoring:

**Built-in Metrics**:
- `victor_requests_total`: Total requests
- `victor_request_duration_seconds`: Request latency
- `victor_tool_executions_total`: Tool executions
- `victor_provider_errors_total`: Provider errors
- `victor_workflow_executions_total`: Workflow runs

**Prometheus Integration**:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'victor'
    static_configs:
      - targets: ['victor:8080']
    metrics_path: '/metrics'
```

**Grafana Dashboard**:
- Request rate and latency
- Tool execution metrics
- Provider error rates
- Workflow success rates
- Resource utilization

[Full Metrics Guide →](monitoring/metrics.md)

### Logging

**Log Levels**:
- `DEBUG`: Detailed debugging information
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical failures

**Log Configuration**:
```yaml
# config.yaml
logging:
  level: INFO
  format: json  # or text
  outputs:
    - type: file
      path: /var/log/victor.log
      rotation: daily
      retention: 30
    - type: syslog
      facility: local0
```

**Structured Logging**:
```json
{
  "timestamp": "2025-01-07T10:30:00Z",
  "level": "INFO",
  "message": "Tool execution completed",
  "context": {
    "tool_name": "read_file",
    "duration_ms": 45,
    "success": true
  }
}
```

[Full Logging Guide →](monitoring/logging.md)

### Health Checks

**HTTP Health Endpoint**:
```bash
curl http://localhost:8080/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.2.3",
  "uptime_seconds": 3600,
  "checks": {
    "database": "healthy",
    "providers": "healthy",
    "cache": "healthy"
  }
}
```

**Kubernetes Probes**:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

[Full Health Check Guide →](monitoring/health-checks.md)

### Alerting

**Prometheus Alerting**:
```yaml
# alerts.yml
groups:
  - name: victor
    rules:
      - alert: HighErrorRate
        expr: rate(victor_provider_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: SlowResponseTime
        expr: victor_request_duration_seconds > 5
        for: 10m
        labels:
          severity: warning
```

**Recommended Alerts**:
- High error rate (>5%)
- Slow response times (>5s)
- High memory usage (>80%)
- Disk space low (<20%)
- Provider API failures

## Security

### Security Architecture

**Security Layers**:
1. **Authentication**: API key validation
2. **Authorization**: Access control
3. **Encryption**: TLS for data in transit
4. **Secrets Management**: Secure key storage
5. **Audit Logging**: Track all operations

[Full Security Overview →](security/overview.md)

### API Key Management

**Best Practices**:
- **Never commit** API keys to git
- Use environment variables or secret managers
- Rotate keys regularly
- Use separate keys for different environments
- Monitor key usage

**Environment Variables**:
```bash
# Production
export ANTHROPIC_API_KEY=sk-prod-...
export VICTOR_ENV=production

# Development
export ANTHROPIC_API_KEY=sk-dev-...
export VICTOR_ENV=development
```

**Secret Managers**:
- **AWS**: AWS Secrets Manager, Parameter Store
- **GCP**: Secret Manager
- **Azure**: Key Vault
- **Kubernetes**: Secrets

**Example: Kubernetes Secrets**:
```bash
# Create secret
kubectl create secret generic victor-secrets \
  --from-literal=anthropic-api-key=sk-... \
  --from-literal=openai-api-key=sk-...

# Use in deployment
env:
  - name: ANTHROPIC_API_KEY
    valueFrom:
      secretKeyRef:
        name: victor-secrets
        key: anthropic-api-key
```

[Full Key Management Guide →](security/api-keys.md)

### Compliance

**SOC2 Compliance**:
- Access controls
- Audit logging
- Change management
- Incident response
- Data encryption

**GDPR Compliance**:
- Data minimization
- Right to erasure
- Data portability
- Consent management
- Breach notification

**HIPAA Compliance** (if handling PHI):
- PHI identification
- Access controls
- Audit trails
- Business associate agreements
- Risk assessments

[Full Compliance Guide →](security/compliance.md)

### Auditing

**Audit Events**:
- User authentication
- Tool execution
- Provider usage
- File operations
- Workflow execution
- Configuration changes

**Audit Log Format**:
```json
{
  "timestamp": "2025-01-07T10:30:00Z",
  "event_type": "tool.execution",
  "user_id": "user@example.com",
  "session_id": "sess_123",
  "details": {
    "tool_name": "read_file",
    "file_path": "/path/to/file.py",
    "success": true,
    "duration_ms": 45
  }
}
```

**Audit Trail Storage**:
- **Database**: PostgreSQL, MongoDB
- **Object Storage**: S3, GCS
- **Log Aggregation**: ELK, Splunk
- **SIEM**: Datadog, Sumo Logic

[Full Auditing Guide →](security/auditing.md)

## Performance

### Optimization

**Startup Time** (<500ms):
- Lazy loading of tools and providers
- Async initialization
- Connection pooling
- Cache preloading

**Tool Execution** (<100ms):
- Tool result caching
- Parallel execution
- Connection reuse
- Batch operations

**Memory Usage** (~200MB base):
- Efficient data structures
- Prompt cleanup
- Connection pooling
- Memory profiling

**Optimization Tips**:
```yaml
# config.yaml
cache:
  enabled: true
  ttl: 3600
  max_size: 1000

providers:
  pool_size: 10
  timeout: 30

tools:
  lazy_load: true
  parallel_execution: true
```

[Full Optimization Guide →](performance/optimization.md)

### Scaling

**Horizontal Scaling**:
```yaml
# docker-compose.yml
version: '3'
services:
  victor:
    image: ghcr.io/vjsingh1984/victor:latest
    deploy:
      replicas: 5
    environment:
      - VICTOR_REDIS_HOST=redis
      - VICTOR_REDIS_PORT=6379
  redis:
    image: redis:alpine
    volumes:
      - redis-data:/data
```

**Vertical Scaling**:
- Increase CPU cores for parallel tool execution
- More memory for large contexts
- Faster storage for cache
- Network bandwidth for streaming

**Caching Strategy**:
```yaml
# Multi-layer caching
cache:
  layers:
    - type: memory
      size: 1000
      ttl: 300
    - type: redis
      url: redis://localhost:6379
      ttl: 3600
    - type: disk
      path: /var/cache/victor
      ttl: 86400
```

[Full Scaling Guide →](performance/scaling.md)

### Benchmarks

**Performance Metrics**:

| Operation | P50 | P95 | P99 | Notes |
|-----------|-----|-----|-----|-------|
| Startup Time | 400ms | 600ms | 1s | Lazy loading |
| Tool Execution | 50ms | 150ms | 500ms | Local tools |
| Provider Request | 1s | 3s | 5s | Anthropic |
| Context Switch | <10ms | <50ms | <100ms | Provider switch |
| Workflow Step | 2s | 5s | 10s | Agent node |

**Resource Usage**:

| Metric | Value | Notes |
|--------|-------|-------|
| Base Memory | 200MB | Without model |
| CPU Usage | 5-10% | Idle |
| Network | 100KB/s | Average |
| Disk I/O | 10MB/s | With caching |

[Full Benchmarks →](performance/benchmarks.md)

## Troubleshooting

### Common Issues

**1. High Memory Usage**
```bash
# Check memory usage
docker stats victor

# Solution: Tune cache size
# config.yaml
cache:
  max_size: 500  # Reduce from 1000
```

**2. Slow Response Times**
```bash
# Check provider latency
victor --profile

# Solution: Use caching
cache:
  enabled: true
  ttl: 3600
```

**3. Provider Errors**
```bash
# Check provider health
victor provider health

# Solution: Add fallback providers
profiles:
  production:
    providers:
      - anthropic
      - openai
      - ollama  # Fallback
```

**4. Database Connection Issues**
```bash
# Check database connectivity
victor db check

# Solution: Increase pool size
database:
  pool_size: 20
  timeout: 30
```

### Debug Mode

```bash
# Enable debug logging
export VICTOR_LOG_LEVEL=DEBUG
victor serve

# Enable profiling
victor --profile serve

# Enable debug endpoints
victor serve --debug
```

### Log Analysis

```bash
# View recent errors
grep ERROR /var/log/victor.log | tail -100

# Analyze tool execution
grep "tool.execution" /var/log/victor.log | jq '.details.duration_ms'

# Provider error rate
grep "provider.error" /var/log/victor.log | wc -l
```

## Maintenance

### Regular Tasks

**Daily**:
- Monitor health checks
- Review error logs
- Check disk space
- Verify backups

**Weekly**:
- Rotate logs
- Review performance metrics
- Update dependencies
- Test failover

**Monthly**:
- Security updates
- Capacity planning
- Cost optimization
- Archive old logs

### Updates

**Update Victor**:
```bash
# Pull latest image
docker pull ghcr.io/vjsingh1984/victor:latest

# Recreate containers
docker-compose up -d

# Or with pip
pip install --upgrade victor-ai
```

**Database Migrations**:
```bash
# Run migrations
victor db migrate

# Rollback if needed
victor db migrate --rollback
```

### Backup and Recovery

**Backup Config**:
```bash
# Backup configuration
tar -czf victor-config-backup.tar.gz ~/.victor/

# Backup database
victor db backup > backup.sql

# Backup to S3
aws s3 cp ~/.victor/ s3://my-bucket/victor/ --recursive
```

**Restore**:
```bash
# Restore configuration
tar -xzf victor-config-backup.tar.gz -C ~/

# Restore database
victor db restore < backup.sql
```

## Additional Resources

- **Deployment**: [Deployment Guide →](deployment/)
- **User Guide**: [User Guide →](../user-guide/)
- **Development**: [Development Guide →](../development/)
- **Reference**: [Configuration →](../reference/configuration/)
- **Troubleshooting**: [User Troubleshooting →](../user-guide/troubleshooting.md)

---

**Next**: [Docker Deployment →](deployment/docker.md)
