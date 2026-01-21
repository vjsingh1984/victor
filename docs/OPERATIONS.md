# Victor AI Operations Guide

**Version:** 0.5.0
**Last Updated:** 2025-01-20
**Target Audience:** System Administrators, DevOps Engineers, SREs

---

## Table of Contents

1. [Overview](#overview)
2. [Daily Operations Checklist](#daily-operations-checklist)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Performance Tuning](#performance-tuning)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Backup and Recovery](#backup-and-recovery)
7. [Scaling Guidelines](#scaling-guidelines)
8. [Maintenance Procedures](#maintenance-procedures)
9. [Incident Management](#incident-management)

---

## Overview

This guide provides operational procedures for managing Victor AI in production environments. It covers monitoring, maintenance, troubleshooting, and scaling strategies to ensure reliable operation.

### Operational Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **API Response Time** | < 500ms | > 2s |
| **Tool Selection Latency** | < 200ms | > 1s |
| **Error Rate** | < 0.1% | > 1% |
| **CPU Usage** | < 70% | > 90% |
| **Memory Usage** | < 80% | > 95% |
| **Database Connections** | < 80% | > 90% |
| **Cache Hit Rate** | > 60% | < 40% |

---

## Daily Operations Checklist

### Morning Checks (Daily)

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== Victor AI Daily Health Check ==="
echo "Timestamp: $(date)"
echo ""

# 1. Service Status
echo "[1/8] Checking service status..."
systemctl status victor-api || docker ps | grep victor-ai

# 2. Health Endpoint
echo "[2/8] Checking health endpoint..."
curl -f http://localhost:8000/health || echo "❌ Health check failed"

# 3. Provider Connectivity
echo "[3/8] Checking provider connectivity..."
victor doctor --providers-only

# 4. Database Connectivity
echo "[4/8] Checking database..."
victor db check || echo "❌ Database check failed"

# 5. Disk Space
echo "[5/8] Checking disk space..."
df -h | grep -E "(Filesystem|/$|/.victor)"

# 6. Memory Usage
echo "[6/8] Checking memory usage..."
free -h || vm_stat

# 7. Recent Errors
echo "[7/8] Checking for recent errors..."
tail -n 50 ~/.victor/logs/victor.log | grep -i error || echo "✓ No recent errors"

# 8. Backup Verification
echo "[8/8] Verifying recent backups..."
victor backup list | tail -n 5

echo ""
echo "=== Health Check Complete ==="
```

### Weekly Tasks

```bash
#!/bin/bash
# weekly_maintenance.sh

echo "=== Victor AI Weekly Maintenance ==="

# 1. Review logs for anomalies
echo "[1/5] Analyzing logs..."
victor logs analyze --period=7d --severity=warning

# 2. Performance review
echo "[2/5] Generating performance report..."
victor metrics report --period=7d --output=weekly_report.md

# 3. Database maintenance
echo "[3/5] Running database maintenance..."
victor db vacuum
victor db analyze

# 4. Cache statistics
echo "[4/5] Reviewing cache performance..."
victor cache stats

# 5. Security audit
echo "[5/5] Running security audit..."
victor security audit --safe-mode

echo "=== Weekly Maintenance Complete ==="
```

### Monthly Tasks

```bash
#!/bin/bash
# monthly_maintenance.sh

echo "=== Victor AI Monthly Maintenance ==="

# 1. Full backup
echo "[1/6] Creating full backup..."
victor backup create --full --compress

# 2. Update dependencies
echo "[2/6] Checking for updates..."
pip list --outdated

# 3. Storage cleanup
echo "[3/6] Cleaning old logs and caches..."
victor cleanup --logs --older-than=90d
victor cleanup --cache --older-than=30d

# 4. Performance baseline
echo "[4/6] Running performance benchmarks..."
python scripts/benchmark_tool_selection.py run --group all
python scripts/benchmark_tool_selection.py report --format markdown

# 5. Capacity planning
echo "[5/6] Generating capacity report..."
victor metrics capacity --output=capacity_report.md

# 6. Documentation review
echo "[6/6] Checking documentation currency..."
victor docs check --outdated

echo "=== Monthly Maintenance Complete ==="
```

---

## Monitoring and Alerting

### Prometheus Metrics Integration

Victor AI exposes metrics at `/metrics` endpoint:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'victor-ai'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
    metrics_path: /metrics
```

**Key Metrics:**

```promql
# API Response Time
rate(victor_api_response_time_seconds_sum[5m]) / rate(victor_api_response_time_seconds_count[5m])

# Tool Selection Latency
rate(victor_tool_selection_duration_seconds_sum[5m]) / rate(victor_tool_selection_duration_seconds_count[5m])

# Error Rate
rate(victor_errors_total[5m])

# Cache Hit Rate
rate(victor_cache_hits_total[5m]) / (rate(victor_cache_hits_total[5m]) + rate(victor_cache_misses_total[5m]))

# Provider Requests
sum by(provider) (rate(victor_provider_requests_total[5m]))

# Memory Usage
victor_memory_usage_bytes

# Database Connections
victor_db_connections_active
```

### Grafana Dashboard

**Recommended Panels:**

1. **Overview Panel**
   - Requests per second (5min rate)
   - Error rate (%)
   - P50/P95/P99 latency
   - Active users

2. **Provider Panel**
   - Requests by provider (stacked)
   - Provider error rate
   - Provider latency comparison
   - Rate limit utilization

3. **Tool Panel**
   - Most used tools (top 10)
   - Tool execution time (box plot)
   - Tool failure rate

4. **System Panel**
   - CPU usage (%)
   - Memory usage (%)
   - Disk I/O
   - Network I/O

5. **Database Panel**
   - Connection pool usage
   - Query duration (P95)
   - Transaction rate

6. **Cache Panel**
   - Hit rate (%)
   - Eviction rate
   - Memory usage

### Alert Rules

**Prometheus Alert Rules:**

```yaml
groups:
  - name: victor-ai-alerts
    interval: 30s
    rules:
      # High Error Rate
      - alert: HighErrorRate
        expr: rate(victor_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      # Slow API Response
      - alert: SlowAPIResponse
        expr: |
          histogram_quantile(0.95,
            rate(victor_api_response_time_seconds_bucket[5m])
          ) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "API response time is slow"
          description: "P95 latency is {{ $value }}s"

      # Provider Down
      - alert: ProviderDown
        expr: victor_provider_up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Provider is down"
          description: "Provider {{ $labels.provider }} is unreachable"

      # Low Cache Hit Rate
      - alert: LowCacheHitRate
        expr: |
          rate(victor_cache_hits_total[5m]) /
          (rate(victor_cache_hits_total[5m]) + rate(victor_cache_misses_total[5m]))
          < 0.4
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value }}"

      # High Memory Usage
      - alert: HighMemoryUsage
        expr: victor_memory_usage_bytes / victor_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}"

      # Database Connection Pool Exhausted
      - alert: DBPoolExhausted
        expr: victor_db_connections_active / victor_db_connections_max > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "Pool usage is {{ $value }}"
```

### Log Aggregation

**ELK Stack Integration:**

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - ~/.victor/logs/victor.log
    fields:
      service: victor-ai
      environment: production
    fields_under_root: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "victor-ai-%{+yyyy.MM.dd}"

setup.ilm.enabled: false
setup.template.name: "victor-ai"
setup.template.pattern: "victor-ai-*"
```

**Log Queries (Kibana):**

```json
// Errors in last hour
{
  "query": {
    "bool": {
      "must": [
        { "range": { "@timestamp": { "gte": "now-1h" } } },
        { "match": { "level": "ERROR" } }
      ]
    }
  }
}

// Slow tool selection (> 1s)
{
  "query": {
    "bool": {
      "must": [
        { "range": { "tool_selection.duration": { "gte": 1000 } } }
      ]
    }
  }
}

// Provider errors by type
{
  "size": 0,
  "aggs": {
    "providers": {
      "terms": { "field": "provider.name" },
      "aggs": {
        "error_types": {
          "terms": { "field": "error.type" }
        }
      }
    }
  }
}
```

---

## Performance Tuning

### Tool Selection Optimization

**Enable Caching:**

```bash
# Enable all caching layers
export VICTOR_CACHE_ENABLED=true
export VICTOR_CACHE_BACKEND=redis
export VICTOR_CACHE_REDIS_URL=redis://localhost:6379/0

# Configure TTL for different cache types
export VICTOR_TOOL_SELECTION_CACHE_TTL=3600  # 1 hour
export VICTOR_CONTEXT_CACHE_TTL=300  # 5 minutes
export VICTOR_RL_CACHE_TTL=3600  # 1 hour

# Configure cache size
export VICTOR_CACHE_MAX_SIZE=1000
```

**Selection Strategy Tuning:**

```bash
# For fast, simple selection
export VICTOR_TOOL_SELECTION_STRATEGY=keyword

# For accurate, semantic selection
export VICTOR_TOOL_SELECTION_STRATEGY=semantic

# For balanced approach (recommended)
export VICTOR_TOOL_SELECTION_STRATEGY=hybrid

# Hybrid strategy tuning
export VICTOR_HYBRID_SEMANTIC_WEIGHT=0.7
export VICTOR_HYBRID_KEYWORD_WEIGHT=0.3
```

### Provider Optimization

**Request Batching:**

```bash
# Enable request batching
export VICTOR_BATCH_REQUESTS=true
export VICTOR_BATCH_SIZE=10
export VICTOR_BATCH_TIMEOUT=0.5  # seconds
```

**Connection Pooling:**

```bash
# Configure connection pools
export VICTOR_HTTP_MAX_CONNECTIONS=100
export VICTOR_HTTP_MAX_KEEPALIVE=50
export VICTOR_HTTP_KEEPALIVE_EXPIRY=300
```

**Retry Configuration:**

```bash
# Exponential backoff
export VICTOR_RETRY_MAX_ATTEMPTS=3
export VICTOR_RETRY_INITIAL_DELAY=1.0
export VICTOR_RETRY_MAX_DELAY=10.0
export VICTOR_RETRY_BACKOFF_MULTIPLIER=2.0
```

### Memory Optimization

**Lazy Loading:**

```bash
# Enable lazy component loading
export VICTOR_LAZY_LOADING=true
export VICTOR_LOADING_STRATEGY=adaptive  # lazy, eager, adaptive

# Memory optimization
export VICTOR_OPTIMIZE_MEMORY=true
export VICTOR_CACHE_MAX_MEMORY_MB=512
```

**Garbage Collection Tuning:**

```python
# In ~/.victor/config.yaml
performance:
  gc:
    enabled: true
    threshold: 1000  # objects
    generation: 2  # 0, 1, or 2
```

### Parallel Execution

**Worker Pool Sizing:**

```bash
# Auto-scale workers
export VICTOR_AUTO_SCALE_WORKERS=true
export VICTOR_MIN_WORKERS=1
export VICTOR_MAX_WORKERS=4
export VICTOR_WORKER_SCALE_UP_THRESHOLD=0.8  # CPU usage
export VICTOR_WORKER_SCALE_DOWN_THRESHOLD=0.3
```

**Load Balancing:**

```bash
# Enable work stealing
export VICTOR_LOAD_BALANCING=work_stealing

# Configure parallel execution
export VICTOR_MAX_PARALLEL_TOOLS=5
export VICTOR_PARALLEL_TIMEOUT=300  # seconds
```

---

## Troubleshooting Guide

### Diagnostic Commands

```bash
# System health check
victor doctor

# Detailed status
victor status --verbose

# Check provider connectivity
victor providers check --all

# Database diagnostics
victor db check
victor db stats

# Cache diagnostics
victor cache stats
victor cache clear --pattern=*

# Log analysis
victor logs analyze --severity=error --period=1h

# Performance profiling
victor profile --output=profile.json
```

### Common Issues

#### Issue: High Memory Usage

**Diagnosis:**

```bash
# Check memory usage
victor metrics memory

# Profile memory
python -m memory_profiler victor/cli/main.py

# Check for memory leaks
victor metrics memory --leak-detect
```

**Resolution:**

```bash
# 1. Enable memory optimization
export VICTOR_OPTIMIZE_MEMORY=true

# 2. Reduce cache size
export VICTOR_CACHE_MAX_SIZE=500

# 3. Enable lazy loading
export VICTOR_LAZY_LOADING=true

# 4. Reduce workers
export VICTOR_MAX_WORKERS=2

# 5. Clear old data
victor cleanup --cache --older-than=7d
victor cleanup --logs --older-than=30d
```

#### Issue: Slow Tool Selection

**Diagnosis:**

```bash
# Benchmark tool selection
python scripts/benchmark_tool_selection.py run --group all

# Check cache hit rate
victor cache stats --tool-selection

# Profile tool selection
victor profile --component=tool_selection
```

**Resolution:**

```bash
# 1. Enable caching
export VICTOR_CACHE_ENABLED=true

# 2. Use simpler strategy
export VICTOR_TOOL_SELECTION_STRATEGY=keyword

# 3. Reduce tool budget
export VICTOR_TOOL_BUDGET=50

# 4. Enable semantic caching
export VICTOR_SEMANTIC_CACHE_ENABLED=true

# 5. Warm up cache
victor cache warm --tools
```

#### Issue: Provider Rate Limiting

**Diagnosis:**

```bash
# Check provider stats
victor providers stats --all

# Check rate limit status
victor providers rate-limit --provider=anthropic
```

**Resolution:**

```bash
# 1. Configure rate limit handling
export VICTOR_RATE_LIMIT_RETRY=true
export VICTOR_RATE_LIMIT_BACKOFF=exponential

# 2. Enable provider fallback
export VICTOR_PROVIDER_FALLBACK=true
export VICTOR_FALLBACK_PROVIDERS=openai,ollama

# 3. Reduce concurrent requests
export VICTOR_MAX_CONCURRENT_REQUESTS=5

# 4. Enable request queuing
export VICTOR_REQUEST_QUEUE_ENABLED=true
export VICTOR_REQUEST_QUEUE_SIZE=100
```

#### Issue: Database Lock Contention

**Diagnosis:**

```bash
# Check database locks
victor db locks

# Check connection pool
victor db pool-stats

# Analyze slow queries
victor db slow-queries --threshold=1000
```

**Resolution:**

```bash
# 1. Increase pool size
export VICTOR_DB_POOL_SIZE=20

# 2. Increase timeout
export VICTOR_DB_TIMEOUT=30

# 3. Enable connection recycling
export VICTOR_DB_POOL_RECYCLE=3600

# 4. Switch to PostgreSQL (production)
export VICTOR_CHECKPOINT_DB_URL=postgresql://user:pass@localhost/victor

# 5. Optimize queries
victor db analyze
victor db vacuum
```

### Emergency Procedures

**Emergency Restart:**

```bash
# Graceful shutdown
victor api stop --graceful --timeout=30

# Force kill (if graceful fails)
pkill -9 -f victor-api

# Clear locks
victor db clear-locks

# Restart
victor api start

# Verify
victor doctor
```

**Emergency Rollback:**

```bash
# 1. Stop service
victor api stop

# 2. Rollback code
pip install victor-ai==<previous-version>

# 3. Restore database
victor db restore --backup=<backup-file>

# 4. Restore configuration
cp ~/.victor.backup/config.yaml ~/.victor/config.yaml

# 5. Restart
victor api start

# 6. Verify
victor doctor
```

---

## Backup and Recovery

### Backup Strategy

**Automated Backups (cron):**

```bash
# /etc/cron.d/victor-backups

# Daily incremental backup at 2 AM
0 2 * * * root victor backup create --incremental --compress

# Weekly full backup on Sunday at 3 AM
0 3 * * 0 root victor backup create --full --compress

# Hourly transaction log backup
0 * * * * root victor backup transaction-logs
```

**Backup Configuration:**

```yaml
# ~/.victor/backup_config.yaml
backups:
  enabled: true
  schedule:
    incremental: "0 2 * * *"  # Daily at 2 AM
    full: "0 3 * * 0"  # Sunday at 3 AM

  storage:
    type: s3  # local, s3, gcs, azure
    path: /backups/victor
    s3:
      bucket: victor-backups
      region: us-east-1
      access_key: ${AWS_ACCESS_KEY_ID}
      secret_key: ${AWS_SECRET_ACCESS_KEY}

  retention:
    daily: 7  # Keep 7 daily backups
    weekly: 4  # Keep 4 weekly backups
    monthly: 12  # Keep 12 monthly backups

  compression: true
  encryption: true
```

### Backup Commands

**Create Backup:**

```bash
# Full backup
victor backup create --full --compress --encrypt

# Incremental backup
victor backup create --incremental

# Database only
victor backup create --database-only

# Configuration only
victor backup create --config-only
```

**List Backups:**

```bash
# All backups
victor backup list

# Filter by type
victor backup list --type=full

# Filter by date
victor backup list --since="2025-01-01"
```

**Restore Backup:**

```bash
# Restore full backup
victor backup restore <backup-id>

# Restore database only
victor backup restore <backup-id> --database-only

# Restore to specific point in time
victor backup restore --point-in-time="2025-01-20 10:00:00"
```

### Disaster Recovery

**Recovery Procedure:**

```bash
#!/bin/bash
# disaster_recovery.sh

BACKUP_ID=$1

echo "=== Victor AI Disaster Recovery ==="
echo "Backup ID: $BACKUP_ID"

# 1. Stop all services
echo "[1/7] Stopping services..."
victor api stop
systemctl stop victor-scheduler

# 2. Backup current state (rollback safety)
echo "[2/7] Backing current state..."
cp -r ~/.victor ~/.victor.before_recovery

# 3. Restore from backup
echo "[3/7] Restoring from backup..."
victor backup restore $BACKUP_ID --verify

# 4. Verify database integrity
echo "[4/7] Verifying database..."
victor db verify

# 5. Verify configuration
echo "[5/7] Verifying configuration..."
victor config validate

# 6. Start services
echo "[6/7] Starting services..."
victor api start
systemctl start victor-scheduler

# 7. Health check
echo "[7/7] Running health check..."
sleep 30
victor doctor

echo "=== Recovery Complete ==="
```

**Recovery Time Objective (RTO):** 30 minutes
**Recovery Point Objective (RPO):** 1 hour (incremental backups)

---

## Scaling Guidelines

### Vertical Scaling

**CPU Bound:**

```bash
# Add more CPU cores
# Update deployment: 4 cores → 8 cores

# Increase workers
export VICTOR_MAX_WORKERS=8

# Increase parallelism
export VICTOR_MAX_PARALLEL_TOOLS=10
```

**Memory Bound:**

```bash
# Add more RAM
# Update deployment: 8GB → 16GB

# Increase cache sizes
export VICTOR_CACHE_MAX_MEMORY_MB=2048
export VICTOR_TOOL_SELECTION_CACHE_SIZE=2000
```

### Horizontal Scaling

**Load Balancer Setup:**

```nginx
# nginx.conf
upstream victor_backend {
    least_conn;
    server victor-1:8000 weight=3;
    server victor-2:8000 weight=3;
    server victor-3:8000 weight=2;
    server victor-4:8000 weight=2;

    # Health check
    check interval=3000 rise=2 fall=3 timeout=1000;
}

server {
    listen 80;
    server_name victor.example.com;

    location / {
        proxy_pass http://victor_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://victor_backend/health;
    }
}
```

**Session Affinity:**

For long-running conversations:

```nginx
# Sticky sessions
upstream victor_backend {
    ip_hash;  # Hash by client IP
    server victor-1:8000;
    server victor-2:8000;
    server victor-3:8000;
}
```

**Shared State:**

```bash
# Use PostgreSQL for checkpointing (all instances share state)
export VICTOR_CHECKPOINT_DB_URL=postgresql://user:pass@postgres:5432/victor

# Use Redis for caching (shared cache)
export VICTOR_CACHE_BACKEND=redis
export VICTOR_CACHE_REDIS_URL=redis://redis:6379/0

# Use Kafka for events (shared event bus)
export VICTOR_EVENT_BACKEND=kafka
export VICTOR_EVENT_KAFKA_BOOTSTRAP_SERVERS=kafka:9092
```

### Auto-Scaling

**Kubernetes HPA:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: victor-ai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: victor-ai
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

---

## Maintenance Procedures

### Database Maintenance

**PostgreSQL:**

```bash
# Daily vacuum (autovacuum usually handles this)
victor db vacuum --analyze

# Reindex if needed
victor db reindex

# Update statistics
victor db analyze

# Check fragmentation
victor db fragmentation

# Optimize tables
victor db optimize
```

**SQLite:**

```bash
# Vacuum database
victor db vacuum

# Rebuild database
victor db rebuild

# Check integrity
victor db integrity-check
```

### Cache Maintenance

```bash
# View cache statistics
victor cache stats

# Clear expired entries
victor cache clear --expired

# Clear specific cache
victor cache clear --pattern=tool_selection

# Warm up cache
victor cache warm --tools --workflows

# Compact cache (remove fragmentation)
victor cache compact
```

### Log Rotation

```bash
# /etc/logrotate.d/victor-ai

~/.victor/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 victor victor
    sharedscripts
    postrotate
        victor api reload
    endscript
}
```

### Dependency Updates

```bash
# Check for updates
pip list --outdated

# Update Victor AI
pip install --upgrade victor-ai

# Update all dependencies
pip install --upgrade --refresh-dependencies victor-ai

# Check for security issues
pip check
pip-audit  # Requires: pip install pip-audit
```

---

## Incident Management

### Severity Levels

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| **P1 - Critical** | System down, total outage | 15 minutes | API server down, all providers failing |
| **P2 - High** | Major functionality broken | 1 hour | Database down, single provider failure |
| **P3 - Medium** | Partial degradation | 4 hours | Slow responses, intermittent errors |
| **P4 - Low** | Minor issues | 1 day | UI glitches, non-critical bugs |

### Incident Response Flow

**1. Detection**

```bash
# Automated monitoring alerts
# User reports
# Health check failures
```

**2. Triage**

```bash
# Assess severity
victor incident assess --severity=p1

# Gather diagnostics
victor doctor --output=diagnostics.json

# Identify affected systems
victor incident scope --affected=all
```

**3. Response**

```bash
# Declare incident
victor incident declare --severity=p1 --title="API Server Down"

# Assign owner
victor incident assign --owner=on-call-engineer

# Create communication channel
victor incident slack --create-channel
```

**4. Resolution**

```bash
# Apply fix
# See Troubleshooting Guide above

# Verify fix
victor doctor

# Monitor for recurrence
victor monitor --watch
```

**5. Post-Incident**

```bash
# Create postmortem
victor incident postmortem --template

# Extract lessons learned
victor incident learnings --generate

# Update runbooks
victor docs update --runbook
```

### Runbook Templates

**Runbook: API Server Down**

```bash
#!/bin/bash
# runbooks/api_server_down.sh

echo "=== API Server Down - Runbook ==="

# 1. Check service status
echo "[1/5] Checking service status..."
systemctl status victor-api

# 2. Check logs
echo "[2/5] Checking logs for errors..."
journalctl -u victor-api -n 100 --no-pager | grep -i error

# 3. Check port availability
echo "[3/5] Checking port 8000..."
netstat -tuln | grep 8000 || lsof -i :8000

# 4. Attempt restart
echo "[4/5] Attempting restart..."
systemctl restart victor-api

# 5. Verify health
echo "[5/5] Verifying health..."
sleep 10
curl -f http://localhost:8000/health || echo "❌ Health check failed"

echo "=== Runbook Complete ==="
```

**Runbook: Database Connection Failed**

```bash
#!/bin/bash
# runbooks/database_connection_failed.sh

echo "=== Database Connection Failed - Runbook ==="

# 1. Check database status
echo "[1/6] Checking database..."
systemctl status postgresql

# 2. Test connectivity
echo "[2/6] Testing connectivity..."
psql -U victor -d victor -c "SELECT 1" || echo "❌ Cannot connect"

# 3. Check connection pool
echo "[3/6] Checking connection pool..."
victor db pool-stats

# 4. Check for locks
echo "[4/6] Checking locks..."
victor db locks

# 5. Clear locks if needed
echo "[5/6] Clearing locks..."
victor db clear-locks

# 6. Restart service
echo "[6/6] Restarting service..."
systemctl restart victor-api

echo "=== Runbook Complete ==="
```

---

## Additional Resources

- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Features Guide**: [FEATURES.md](FEATURES.md)
- **Security Guide**: [SECURITY.md](SECURITY.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

**Document Version:** 1.0.0
**Last Reviewed:** 2025-01-20
**Next Review:** 2025-02-20
