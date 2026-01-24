# Victor Coordinator Orchestrator - Production Runbook

This runbook provides operational guidance for troubleshooting and maintaining the coordinator-based orchestrator in production.

**Last Updated**: 2025-01-14
**Version**: 0.5.0
**Environment**: Production

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Common Issues and Resolutions](#common-issues-and-resolutions)
3. [Performance Tuning](#performance-tuning)
4. [Capacity Planning](#capacity-planning)
5. [Emergency Procedures](#emergency-procedures)
6. [Maintenance Tasks](#maintenance-tasks)
7. [Monitoring Setup](#monitoring-setup)
8. [Contact and Escalation](#contact-and-escalation)

---

## Quick Reference

### Key Metrics

| Metric | Healthy | Warning | Critical | Action |
|--------|---------|---------|----------|--------|
| Error Rate | < 1% | 1-5% | > 5% | [Error Rate Issues](#error-rate-issues) |
| Latency (p95) | < 2s | 2-5s | > 5s | [High Latency](#high-latency) |
| Cache Hit Rate | > 80% | 50-80% | < 50% | [Cache Performance](#cache-performance) |
| Memory | < 2GB | 2-4GB | > 4GB | [High Memory Usage](#high-memory-usage) |
| CPU | < 50% | 50-80% | > 80% | [High CPU Usage](#high-cpu-usage) |

### Important Commands

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/live

# View metrics
curl http://localhost:9090/metrics

# Check logs
tail -f /var/log/victor/coordinators.log

# Restart service
systemctl restart victor-coordinators

# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=victor_coordinator_error_rate
```

---

## Common Issues and Resolutions

### Error Rate Issues

#### High Error Rate (5-15%)

**Symptoms:**
- Prometheus alert: `VictorCoordinatorHighErrorRate`
- Error rate > 5% for sustained period
- User reports of failed requests

**Diagnosis:**

1. Check which coordinator is failing:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=rate(victor_coordinator_errors_total[5m])/rate(victor_coordinator_executions_total[5m])'
   ```

2. Check logs for errors:
   ```bash
   jq 'select(.level == "ERROR") | select(.coordinator == "ChatCoordinator")' \
     /var/log/victor/coordinators.log | tail -50
   ```

3. Check health status:
   ```bash
   curl http://localhost:8000/health/detailed | jq '.components'
   ```

**Resolutions:**

1. **Provider Issues**: If provider-related errors:
   ```bash
   # Check provider health
   curl http://localhost:8000/health/detailed | jq '.components.provider*'

   # Restart provider connection
   victor admin restart-provider --provider anthropic
   ```

2. **Timeout Errors**: Increase timeout in configuration:
   ```yaml
   # config/victor.yml
   providers:
     anthropic:
       timeout: 60  # Increase from default 30
   ```

3. **Rate Limiting**: If API rate limits hit:
   - Implement backoff strategy
   - Reduce concurrent requests
   - Upgrade API tier

4. **Resource Exhaustion**: Check memory/CPU usage:
   ```bash
   free -h
   top -p $(pgrep -f victor)
   ```

#### Critical Error Rate (> 15%)

**Symptoms:**
- Prometheus alert: `VictorCoordinatorCriticalErrorRate`
- Error rate > 15%
- System mostly unavailable

**Immediate Actions:**

1. Check if service is running:
   ```bash
   systemctl status victor-coordinators
   ```

2. If service is down, restart:
   ```bash
   systemctl restart victor-coordinators
   ```

3. Check for deployment issues:
   ```bash
   git log -1 --oneline
   git diff HEAD~1 HEAD
   ```

4. Rollback if recent deployment:
   ```bash
   victor admin rollback --version <previous-version>
   ```

5. Escalate to on-call engineer

---

### High Latency

#### Moderate Latency (2-5 seconds)

**Symptoms:**
- Prometheus alert: `VictorCoordinatorHighLatency`
- p95 latency > 2 seconds
- Slow user experience

**Diagnosis:**

1. Check coordinator latency breakdown:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=rate(victor_coordinator_duration_seconds_total[5m])/rate(victor_coordinator_executions_total[5m])'
   ```

2. Check which coordinator is slow:
   ```bash
   jq 'select(.duration_ms > 2000)' /var/log/victor/coordinators.log | tail -20
   ```

**Resolutions:**

1. **Database Queries**: Optimize slow queries
   ```bash
   # Enable query logging
   victor admin config set database.log_queries=true
   ```

2. **Cache Misses**: Improve cache hit rate
   ```bash
   # Check cache stats
   curl http://localhost:8000/health/detailed | jq '.coordinators[].cache_hit_rate'
   ```

3. **Network Latency**: Check network to provider
   ```bash
   ping api.anthropic.com
   traceroute api.anthropic.com
   ```

4. **Resource Contention**: Check CPU/memory
   ```bash
   top -p $(pgrep -f victor)
   ```

#### Critical Latency (> 15 seconds)

**Symptoms:**
- Prometheus alert: `VictorCoordinatorCriticalLatency`
- Requests timing out
- System effectively unavailable

**Immediate Actions:**

1. Check for deadlock or hung process:
   ```bash
   kill -USR1 $(pgrep -f victor)  # Trigger thread dump
   ```

2. Check if external service is down:
   ```bash
   curl -I https://api.anthropic.com/v1/messages
   ```

3. Restart service if hung:
   ```bash
   systemctl restart victor-coordinators
   ```

4. Scale horizontally if overloaded:
   ```bash
   kubectl scale deployment victor-coordinators --replicas=4
   ```

---

### Cache Performance

#### Low Cache Hit Rate (< 50%)

**Symptoms:**
- Prometheus alert: `VictorCoordinatorLowCacheHitRate`
- Cache hit rate < 50%
- Increased latency and cost

**Diagnosis:**

1. Check cache metrics:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=victor_coordinator_cache_hit_rate'
   ```

2. Check cache configuration:
   ```bash
   cat config/victor.yml | grep -A 10 cache
   ```

**Resolutions:**

1. **Increase Cache Size**:
   ```yaml
   # config/victor.yml
   cache:
     max_size_mb: 512  # Increase from 256
   ```

2. **Adjust TTL**:
   ```yaml
   cache:
     default_ttl: 3600  # Increase from 1800
   ```

3. **Enable Cache Warming**:
   ```yaml
   cache:
     warm_on_startup: true
     warm_items: 1000
   ```

4. **Check Cache Eviction**:
   ```bash
   jq 'select(.message | contains("cache eviction"))' /var/log/victor/coordinators.log
   ```

#### Cache Miss Burst

**Symptoms:**
- Prometheus alert: `VictorCoordinatorCacheMissBurst`
- Sudden spike in cache misses
- Performance degradation

**Resolutions:**

1. **Invalidation Storm**: Check for cache invalidation loops
   ```bash
   jq 'select(.message | contains("cache invalidation"))' /var/log/victor/coordinators.log
   ```

2. **Cold Cache**: Warm up cache after restart
   ```bash
   victor admin warm-cache --coordinator ChatCoordinator --items=1000
   ```

3. **Cache Stampede**: Implement request coalescing
   ```yaml
   cache:
     enable_coalescing: true
     coalescing_window_ms: 100
   ```

---

### Coordinator Unavailable

**Symptoms:**
- Prometheus alert: `VictorCoordinatorUnavailable`
- No executions in last 10 minutes
- Health checks failing

**Diagnosis:**

1. Check if coordinator is registered:
   ```bash
   curl http://localhost:8000/health/detailed | jq '.monitored_coordinators'
   ```

2. Check if service is running:
   ```bash
   systemctl status victor-coordinators
   ```

3. Check for crashes:
   ```bash
   journalctl -u victor-coordinators -n 100 --no-pager
   ```

**Resolutions:**

1. **Service Not Running**: Start service
   ```bash
   systemctl start victor-coordinators
   ```

2. **Coordinator Not Initialized**: Manually initialize
   ```bash
   victor admin init-coordinator --name ChatCoordinator
   ```

3. **Configuration Error**: Validate config
   ```bash
   victor admin validate-config
   ```

4. **Dependency Missing**: Check required services
   ```bash
   victor admin check-dependencies
   ```

---

### High Memory Usage

**Symptoms:**
- Prometheus alert: `VictorCoordinatorHighMemory`
- Memory usage > 2GB
- Risk of OOM kill

**Diagnosis:**

1. Check memory breakdown:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=victor_coordinator_memory_bytes'
   ```

2. Check for memory leaks:
   ```bash
   ps aux | grep victor
   ```

3. Enable memory profiling:
   ```bash
   victor admin profile-memory --duration=60
   ```

**Resolutions:**

1. **Clear Caches**: Reduce memory footprint
   ```bash
   victor admin clear-cache --all
   ```

2. **Reduce History**: Limit execution history
   ```yaml
   coordinators:
     history_limit: 1000  # Reduce from 10000
   ```

3. **Adjust GC Settings**: Tune garbage collection
   ```yaml
   python:
     gc_threshold: (700, 10, 10)  # More aggressive GC
   ```

4. **Restart Service**: Free memory
   ```bash
   systemctl restart victor-coordinators
   ```

#### Critical Memory Usage (> 4GB)

**Immediate Actions:**

1. Check if OOM is imminent:
   ```bash
   free -h
   cat /proc/meminfo | grep -i oom
   ```

2. Restart service immediately:
   ```bash
   systemctl restart victor-coordinators
   ```

3. Increase memory limits:
   ```yaml
   # Kubernetes
   resources:
     limits:
       memory: "8Gi"
   ```

4. Enable memory profiling for long-term fix
   ```bash
   victor admin profile-memory --continuous
   ```

---

### High CPU Usage

**Symptoms:**
- Prometheus alert: `VictorHighCPUUsage`
- CPU usage > 80%
- Slow response times

**Diagnosis:**

1. Check CPU by coordinator:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=victor_coordinator_cpu_percent'
   ```

2. Profile CPU usage:
   ```bash
   victor admin profile-cpu --duration=30
   ```

**Resolutions:**

1. **Reduce Polling**: Check for busy loops
   ```bash
   strace -p $(pgrep -f victor) -c
   ```

2. **Optimize Algorithms**: Profile hot paths
   ```bash
   python -m cProfile -o profile.stats your_script.py
   ```

3. **Load Balancing**: Distribute load
   ```bash
   kubectl scale deployment victor-coordinators --replicas=4
   ```

---

## Performance Tuning

### Configuration Optimization

#### Cache Tuning

```yaml
# config/victor.yml
cache:
  # Cache size
  max_size_mb: 512

  # Time-to-live
  default_ttl: 3600

  # Cache warming
  warm_on_startup: true
  warm_items: 1000

  # Eviction policy
  eviction_policy: "lru"  # or "lfu", "fifo"

  # Request coalescing
  enable_coalescing: true
  coalescing_window_ms: 100
```

#### Concurrency Tuning

```yaml
coordinators:
  # Max concurrent operations
  max_concurrent: 10

  # Queue size
  queue_size: 100

  # Timeout
  timeout_seconds: 30

  # Retry settings
  retry:
    max_attempts: 3
    backoff_ms: 1000
    exponential_backoff: true
```

#### Provider Tuning

```yaml
providers:
  anthropic:
    # Timeout
    timeout: 60

    # Connection pooling
    max_connections: 10
    keepalive: true

    # Rate limiting
    rate_limit:
      requests_per_second: 50
      burst: 100

    # Batching
    enable_batching: true
    batch_size: 10
    batch_timeout_ms: 100
```

### Monitoring Optimization

1. **Reduce Metrics Scraping Interval**:
   ```yaml
   # prometheus.yml
   scrape_configs:
     - job_name: 'victor-coordinators'
       scrape_interval: 30s  # Increase from 15s
   ```

2. **Use Metrics Aggregation**:
   ```python
   # Aggregate metrics before export
   exporter = PrometheusMetricsExporter(
       collector,
       aggregate_histograms=True,
   )
   ```

3. **Enable Metrics Sampling**:
   ```yaml
   metrics:
     sample_rate: 0.1  # Sample 10% of executions
   ```

---

## Capacity Planning

### Scaling Guidelines

#### Vertical Scaling (More Resources)

| Metric | Current | Recommended Scaling |
|--------|---------|---------------------|
| CPU | > 80% | Increase to 2x current cores |
| Memory | > 4GB | Increase to 2x current memory |
| Throughput | > 100 req/sec | Scale horizontally |

#### Horizontal Scaling (More Instances)

**When to Scale:**
- CPU > 80% across all instances
- Queue depth consistently high
- Latency increasing despite optimization

**Scaling Steps:**

1. Check current capacity:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=victor_coordinator_throughput'
   ```

2. Add instances:
   ```bash
   # Kubernetes
   kubectl scale deployment victor-coordinators --replicas=4

   # Docker
   docker-compose up -d --scale victor-coordinators=4
   ```

3. Verify load balancing:
   ```bash
   curl http://loadbalancer/health
   ```

### Capacity Planning Checklist

- [ ] Monitor daily/weekly peak usage
- [ ] Project growth for next 3/6/12 months
- [ ] Plan for 3x headroom
- [ ] Test load capacity quarterly
- [ ] Update infrastructure based on projections
- [ ] Document scaling procedures

---

## Emergency Procedures

### Service Restart

**Graceful Restart:**

```bash
# 1. Check for active requests
curl http://localhost:8000/health

# 2. Drain traffic (if using load balancer)
kubectl drain node-1 --ignore-daemonsets

# 3. Graceful shutdown
systemctl stop victor-coordinators

# 4. Wait for shutdown
sleep 30

# 5. Start service
systemctl start victor-coordinators

# 6. Verify health
curl http://localhost:8000/health

# 7. Re-enable traffic
kubectl uncordon node-1
```

### Rollback Procedure

**Rollback to Previous Version:**

```bash
# 1. Check current version
victor admin version

# 2. List available versions
victor admin list-versions

# 3. Rollback
victor admin rollback --version 0.5.0

# 4. Verify
curl http://localhost:8000/health
```

### Incident Response

**Severity Levels:**

- **P1 - Critical**: System down, complete outage
  - Immediate action required
  - Page on-call
  - Update incident channel

- **P2 - High**: Degraded performance, partial outage
  - Action within 15 minutes
  - Notify team
  - Create incident ticket

- **P3 - Medium**: Minor issues, workarounds available
  - Action within 1 hour
  - Create ticket

- **P4 - Low**: Cosmetic issues, enhancements
  - Action within 1 week
  - Add to backlog

---

## Maintenance Tasks

### Daily

- [ ] Check error rates
- [ ] Review alerts
- [ ] Verify health endpoints
- [ ] Check disk space

### Weekly

- [ ] Review performance metrics
- [ ] Check for anomalies
- [ ] Review recent logs
- [ ] Update runbook if needed

### Monthly

- [ ] Full system audit
- [ ] Capacity planning review
- [ ] Security updates
- [ ] Performance tuning
- [ ] Backup verification

### Quarterly

- [ ] Load testing
- [ ] Disaster recovery drill
- [ ] Architecture review
- [ ] Cost optimization

---

## Monitoring Setup

### Prometheus Configuration

```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 30s
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'victor-coordinators'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 30s
    metrics_path: /metrics

rule_files:
  - "/etc/prometheus/rules/victor-alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

### Grafana Dashboard

1. Import dashboard:
   ```bash
   curl -X POST http://localhost:3000/api/dashboards/import \
     -H "Content-Type: application/json" \
     -d @docs/production/grafana_dashboard.json
   ```

2. Configure datasource: Prometheus on `http://localhost:9090`

3. Set up alerts and notifications

### AlertManager Configuration

```yaml
# /etc/alertmanager/alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'

    - match:
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '<PAGERDUTY_KEY>'

  - name: 'slack'
    slack_configs:
      - api_url: '<SLACK_WEBHOOK>'
        channel: '#victor-alerts'
```

---

## Contact and Escalation

### On-Call Rotation

| Role | Name | Contact | Hours |
|------|------|---------|-------|
| Primary | On-Call Engineer | +1-555-0100 | 24/7 |
| Secondary | Site Reliability | +1-555-0101 | 24/7 |
| Management | Engineering Lead | +1-555-0102 | Business hours |

### Escalation Path

1. **Level 1**: On-Call Engineer
   - Initial triage
   - Implement fixes
   - Document incident

2. **Level 2**: Engineering Lead (1 hour)
   - Complex issues
   - Architectural decisions
   - Coordination

3. **Level 3**: CTO (2 hours)
   - Critical incidents
   - Business impact
   - Major decisions

### Communication Channels

- **Incidents**: `#victor-incidents`
- **Alerts**: `#victor-alerts`
- **General**: `#victor-ops`
- **Email**: `victor-ops@example.com`

---

## Additional Resources

- [Victor Documentation](https://docs.victor.ai)
- [Architecture Guide](./architecture.md)
- [API Reference](./api-reference.md)
- [Deployment Guide](./deployment.md)
- [Internal Wiki](https://wiki.example.com/victor)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-14
**Maintained By**: Victor Operations Team
**Next Review**: 2025-04-14
