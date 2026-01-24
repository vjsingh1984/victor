# Victor AI Monitoring Runbook

**Version**: 0.5.0
**Last Updated**: 2026-01-21
**Purpose**: Operational guide for Victor AI production monitoring stack

## Table of Contents

1. [Overview](#overview)
2. [Monitoring Architecture](#monitoring-architecture)
3. [Deployment Procedures](#deployment-procedures)
4. [Daily Operations](#daily-operations)
5. [Alert Response Procedures](#alert-response-procedures)
6. [Common Issues and Resolutions](#common-issues-and-resolutions)
7. [Maintenance Procedures](#maintenance-procedures)
8. [Escalation Procedures](#escalation-procedures)

---

## Overview

### Monitoring Stack Components

The Victor AI monitoring stack consists of:

- **Prometheus** (v2.45.0): Metrics collection and storage
  - 30-day data retention
  - 50GB storage
  - Service account with cluster-wide read access

- **Grafana** (v10.0.0): Visualization and dashboards
  - 17 pre-configured panels
  - Auto-refresh every 30 seconds
  - Custom Victor AI performance dashboard

- **AlertManager** (v0.25.0): Alert routing and notification
  - 30+ alert rules
  - Email and Slack integration
  - Alert grouping and silencing

- **OpenTelemetry Collector**: Metrics aggregation
  - Receives metrics from Victor AI pods
  - Forwards to Prometheus

### Key Metrics

The monitoring system tracks **46 metrics** across 6 categories:

1. **Cache Metrics** (8 metrics)
   - Hit rate, miss rate, utilization, eviction rate
   - Operation latency (P50, P95, P99)

2. **Tool Selection Metrics** (10 metrics)
   - Selection latency, cache effectiveness
   - Query vs. context vs. RL cache performance

3. **Provider Pool Metrics** (12 metrics)
   - Health status, request rate, error rate
   - Latency per provider (P50, P95, P99)

4. **Tool Execution Metrics** (8 metrics)
   - Execution latency (P50, P95, P99)
   - Success rate, error rate
   - Concurrent execution count

5. **System Metrics** (4 metrics)
   - Memory usage, CPU usage, thread count
   - Uptime

6. **Bootstrap Metrics** (4 metrics)
   - Startup time, component initialization time
   - Service registration time

---

## Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Victor AI Pods                           │
│  (victor-api, victor-worker, victor-scheduler)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Metrics (Prometheus format)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenTelemetry Collector                         │
│  - Receives metrics on OTLP port (4317)                     │
│  - Batches and processes metrics                            │
│  - Exports to Prometheus                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP /metrics
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Prometheus                                │
│  - Scrapes OTEL collector every 15s                         │
│  - Evaluates alert rules every 30s                          │
│  - Stores data for 30 days                                  │
└──────┬──────────────────────────────────────────┬───────────┘
       │                                          │
       │ HTTP /api/v1/alerts                      │ HTTP /api/v1/query
       │                                          │
       ▼                                          ▼
┌──────────────────┐                  ┌──────────────────────┐
│  AlertManager    │                  │      Grafana         │
│  - Routes alerts │                  │  - Dashboards        │
│  - Sends emails  │                  │  - Queries           │
│  - Slack webhooks│                  │  - Visualizations    │
└──────────────────┘                  └──────────────────────┘
       │                                          │
       ▼                                          ▼
┌──────────────────┐                  ┌──────────────────────┐
│  Email/Slack     │                  │   Web Browser        │
│  Notifications   │                  │   (User Access)      │
└──────────────────┘                  └──────────────────────┘
```

### Data Flow

1. **Victor AI pods** expose metrics endpoint at `/metrics` (Prometheus format)
2. **OpenTelemetry Collector** scrapes Victor AI pods every 15s
3. **Prometheus** scrapes OTEL collector and stores metrics
4. **Prometheus** evaluates alert rules and fires alerts to AlertManager
5. **AlertManager** routes alerts to appropriate receivers (email, Slack)
6. **Grafana** queries Prometheus for dashboard visualization

---

## Deployment Procedures

### Initial Deployment

#### Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured with cluster access
- helm installed (optional, for Helm deployment)
- Storage class configured for PVs (default: `standard`)

#### Deploy with Script (Recommended)

```bash
# Deploy using kubectl (default)
bash deployment/scripts/deploy_monitoring.sh

# Deploy using Helm
bash deployment/scripts/deploy_monitoring.sh --helm

# Skip waiting for pods to be ready
bash deployment/scripts/deploy_monitoring.sh --skip-wait
```

#### Manual Deployment

```bash
# 1. Create namespace
kubectl apply -f deployment/kubernetes/monitoring/namespace.yaml

# 2. Deploy Prometheus
kubectl apply -f deployment/kubernetes/monitoring/prometheus-configmap.yaml
kubectl apply -f deployment/kubernetes/monitoring/prometheus-deployment.yaml

# 3. Deploy Grafana
kubectl apply -f deployment/kubernetes/monitoring/grafana-configs.yaml
kubectl apply -f deployment/kubernetes/monitoring/grafana-deployment.yaml

# 4. Deploy AlertManager
kubectl apply -f deployment/kubernetes/monitoring/alertmanager-deployment.yaml

# 5. Deploy OTEL Collector
kubectl apply -f deployment/kubernetes/monitoring/otel-collector-deployment.yaml

# 6. Deploy Ingress (optional)
kubectl apply -f deployment/kubernetes/monitoring/ingress.yaml

# 7. Deploy alert rules
kubectl apply -f deployment/kubernetes/monitoring/performance-alerts.yaml
```

#### Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n victor-monitoring

# Expected output:
# NAME                           READY   STATUS    RESTARTS   AGE
# prometheus-xxxxxxxxxx-xxxxx    1/1     Running   0          2m
# grafana-xxxxxxxxxx-xxxxx       1/1     Running   0          2m
# alertmanager-xxxxxxxxxx-xxxxx  1/1     Running   0          2m
# otel-collector-xxxxxxxxxx-xxxx 1/1     Running   0          2m

# Verify services
kubectl get svc -n victor-monitoring

# Check persistent volumes are bound
kubectl get pvc -n victor-monitoring
```

---

## Daily Operations

### Access Monitoring Interfaces

#### Grafana Dashboard

```bash
# Port forward to local machine
kubectl port-forward -n victor-monitoring svc/grafana 3000:3000

# Access in browser
open http://localhost:3000

# Default credentials
# Username: admin
# Password: changeme123
```

#### Prometheus UI

```bash
# Port forward to local machine
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090

# Access in browser
open http://localhost:9090
```

#### AlertManager UI

```bash
# Port forward to local machine
kubectl port-forward -n victor-monitoring svc/alertmanager 9093:9093

# Access in browser
open http://localhost:9093
```

### Daily Health Checks

#### Morning Checklist (Daily)

- [ ] Check Grafana dashboard for overnight anomalies
- [ ] Review active alerts in AlertManager
- [ ] Verify all monitoring pods are healthy
- [ ] Check storage utilization (Prometheus: 50GB limit)
- [ ] Review error rates and latencies

#### Commands

```bash
# 1. Check pod health
kubectl get pods -n victor-monitoring

# 2. Check resource usage
kubectl top pods -n victor-monitoring

# 3. Check storage
kubectl get pvc -n victor-monitoring
df -h /var/lib/kubelet/pods/*/volumes/kubernetes.io~*/

# 4. Check Prometheus targets
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'

# 5. Check alert status
kubectl port-forward -n victor-monitoring svc/alertmanager 9093:9093 &
curl http://localhost:9093/api/v2/alerts | jq '.[] | select(.status.state == "firing")'

# 6. Check Grafana health
kubectl exec -n victor-monitoring deployment/grafana -- curl -s http://localhost:3000/api/health
```

### Key Metrics to Monitor Daily

1. **Performance Health Score** (victor_performance_health_score)
   - Target: > 80
   - Warning: < 70
   - Critical: < 50

2. **Cache Hit Rate** (victor_cache_hit_rate)
   - Target: > 60%
   - Warning: < 40%
   - Critical: < 20%

3. **Tool Error Rate** (victor_tool_error_rate)
   - Target: < 1%
   - Warning: > 5%
   - Critical: > 15%

4. **P95 Tool Execution Latency** (victor_tool_duration_ms)
   - Target: < 500ms
   - Warning: > 1s
   - Critical: > 5s

5. **Memory Usage** (victor_system_memory_bytes)
   - Target: < 1GB
   - Warning: > 1GB
   - Critical: > 2GB

6. **CPU Usage** (victor_system_cpu_percent)
   - Target: < 60%
   - Warning: > 80%
   - Critical: > 95%

---

## Alert Response Procedures

### Alert Severity Levels

| Severity | Response Time | Notification Channel    |
|----------|--------------|-------------------------|
| **Critical** | Immediate | Email + Slack + Pager   |
| **Warning** | 1 hour      | Email + Slack           |
| **Info**     | Next day    | Slack only              |

### Alert Response Workflow

```
1. Alert fires
   ↓
2. Receive notification (email/Slack)
   ↓
3. Access Grafana dashboard
   ↓
4. Investigate root cause
   ↓
5. Implement fix or workaround
   ↓
6. Verify resolution
   ↓
7. Document incident
   ↓
8. Silence alert if temporary
```

### Common Alert Scenarios

#### Scenario 1: HighCacheMissRate (Warning)

**Alert**: Cache hit rate below 40% for 5 minutes

**Impact**: Increased latency, higher load on backend

**Response Steps**:
1. Check Grafana "Cache Performance" panel
2. Identify which cache namespace is affected
3. Verify cache TTL and size configuration
4. Check if cache eviction rate is elevated
5. Consider increasing cache size or TTL

**Resolution Commands**:
```bash
# Check cache metrics in Prometheus
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
open "http://localhost:9090/graph?g0.expr=victor_cache_hit_rate&g0.tab=0"

# View cache utilization
open "http://localhost:9090/graph?g0.expr=victor_cache_utilization&g0.tab=0"
```

**Prevention**:
- Monitor cache size vs. utilization
- Adjust TTL based on access patterns
- Consider LRU eviction tuning

---

#### Scenario 2: CriticalToolExecutionLatency (Critical)

**Alert**: P95 tool execution latency > 5s for 5 minutes

**Impact**: Tools are timing out, user experience severely degraded

**Response Steps**:
1. Check Grafana "Tool Execution" panel
2. Identify which tools have high latency
3. Check provider health (are external APIs slow?)
4. Verify network connectivity
5. Check for resource exhaustion (CPU/memory)

**Resolution Commands**:
```bash
# Check tool execution latency by tool
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
open "http://localhost:9090/graph?g0.expr=histogram_quantile(0.95,sum(rate(victor_tool_duration_ms_bucket[5m]))+by+(le,tool))&g0.tab=0"

# Check provider latency
open "http://localhost:9090/graph?g0.expr=histogram_quantile(0.95,sum(rate(victor_provider_latency_ms_bucket[5m]))+by+(le,provider))&g0.tab=0"

# Check resource usage
kubectl top pods -n victor-ai
```

**Immediate Actions**:
- If provider issue: Switch to backup provider
- If resource issue: Scale up pods or increase resource limits
- If network issue: Check network policies and service mesh

---

#### Scenario 3: CriticalMemoryUsage (Critical)

**Alert**: Memory usage > 2GB for 5 minutes

**Impact**: Risk of OOM kill, service disruption

**Response Steps**:
1. Check Grafana "System Resources" panel
2. Identify memory leak (is memory growing over time?)
3. Check pod memory usage
4. Review memory profile (if profiling enabled)
5. Consider restarting pod if memory leak confirmed

**Resolution Commands**:
```bash
# Check memory usage by pod
kubectl top pods -n victor-ai --sort-by=memory

# Check memory limits
kubectl get pods -n victor-ai -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].resources.limits.memory}{"\n"}{end}'

# Restart pod if needed
kubectl rollout restart deployment/victor-api -n victor-ai
```

**Prevention**:
- Configure appropriate memory limits
- Enable memory profiling
- Regular pod restarts (weekly) to prevent leaks
- Monitor memory growth trends

---

#### Scenario 4: CriticalProviderFailureRate (Critical)

**Alert**: Provider failure rate > 30% for 2 minutes

**Impact**: External LLM providers failing, service degraded

**Response Steps**:
1. Check Grafana "Provider Pool" panel
2. Identify which providers are failing
3. Check provider API status (e.g., Anthropic, OpenAI status pages)
4. Verify API keys and credentials
5. Check rate limits (are we hitting limits?)
6. Switch to healthy provider if available

**Resolution Commands**:
```bash
# Check provider error rate
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
open "http://localhost:9090/graph?g0.expr=sum(rate(victor_provider_errors_total[5m]))+by+(provider)%2F+sum(rate(victor_provider_requests_total[5m]))+by+(provider)&g0.tab=0"

# Check provider health
open "http://localhost:9090/graph?g0.expr=victor_provider_health&g0.tab=0"

# Check provider latency
open "http://localhost:9090/graph?g0.expr=histogram_quantile(0.95,sum(rate(victor_provider_latency_ms_bucket[5m]))+by+(le,provider))&g0.tab=0"
```

**Immediate Actions**:
- If rate limit: Switch provider or reduce request rate
- If API down: Switch to backup provider
- If credential issue: Rotate API keys
- If network issue: Check network policies

---

#### Scenario 5: PerformanceCritical (Critical)

**Alert**: Overall performance health score < 50 for 5 minutes

**Impact**: Multiple metrics degraded, service severely impacted

**Response Steps**:
1. This is a composite alert indicating multiple issues
2. Check Grafana "Performance Health Score" panel
3. Review all firing alerts (not just performance)
4. Identify which components are failing
5. Prioritize critical alerts first
6. Address underlying issues systematically

**Resolution Commands**:
```bash
# List all firing alerts
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing") | {alert: .labels.alertname, severity: .labels.severity}'

# Check health score components
open "http://localhost:9090/graph?g0.expr=victor_performance_health_score&g0.tab=0"

# Check all performance metrics
open "http://localhost:9090/graph?g0.expr=%7B__name__=~%22victor_.*%22%7D&g0.tab=0"
```

**Immediate Actions**:
- Address critical alerts first (memory, CPU, providers)
- Scale up resources if needed
- Switch to backup providers
- Restart failing components

---

## Common Issues and Resolutions

### Issue 1: Prometheus Not Scraping Targets

**Symptoms**:
- Grafana dashboards show "No data"
- Prometheus UI shows targets as "DOWN"

**Diagnosis**:
```bash
# Check Prometheus targets
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health, error: .lastError}'
```

**Common Causes**:
1. Service monitor not configured
2. Network policies blocking scrape
3. Service endpoint not exposed
4. Invalid metrics endpoint

**Resolution**:
```bash
# 1. Verify OTEL collector is running
kubectl get pods -n victor-monitoring -l app=otel-collector

# 2. Check OTEL collector metrics endpoint
kubectl port-forward -n victor-monitoring svc/otel-collector 8889:8889 &
curl http://localhost:8889/metrics

# 3. Check Prometheus configuration
kubectl get configmap -n victor-monitoring prometheus-config -o yaml

# 4. Verify service endpoints
kubectl get endpoints -n victor-monitoring

# 5. Check network policies
kubectl get networkpolicies -n victor-monitoring
kubectl get networkpolicies -n victor-ai
```

---

### Issue 2: Grafana Dashboard Shows "N/A"

**Symptoms**:
- Dashboard panels show "No data" or "N/A"
- Data sources are configured but not returning data

**Diagnosis**:
```bash
# Check Grafana data source configuration
kubectl exec -n victor-monitoring deployment/grafana -- curl -s http://localhost:3000/api/datasources | jq '.'
```

**Common Causes**:
1. Prometheus data source misconfigured
2. Query time range has no data
3. Metrics not being collected
4. Dashboard queries incorrect

**Resolution**:
```bash
# 1. Verify Prometheus is accessible from Grafana
kubectl exec -n victor-monitoring deployment/grafana -- curl -s http://prometheus:9090/api/v1/query?query=up

# 2. Check Grafana data source
kubectl exec -n victor-monitoring deployment/grafana -- \
  curl -s http://localhost:3000/api/datasources/proxies/1/api/v1/query?query=up

# 3. Test Prometheus query directly
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s "http://localhost:9090/api/v1/query?query=victor_cache_hit_rate" | jq '.'

# 4. Check dashboard JSON configuration
kubectl get configmap -n victor-monitoring grafana-dashboards -o jsonpath='{.data.victor-performance\.json}' | jq '.panels[] | select(.title=="Cache Hit Rate")'
```

---

### Issue 3: Alerts Not Firing

**Symptoms**:
- Metrics exceed thresholds but no alerts fired
- AlertManager UI shows no alerts

**Diagnosis**:
```bash
# Check Prometheus alert rules
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.type=="alerting") | {name: .name, state: .state}'
```

**Common Causes**:
1. Alert rules not loaded in Prometheus
2. `for` duration not met
3. Alert rules disabled
4. AlertManager not receiving alerts from Prometheus

**Resolution**:
```bash
# 1. Verify alert rules are loaded
kubectl get configmap -n victor-monitoring prometheus-rules -o yaml

# 2. Reload Prometheus configuration
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -X POST http://localhost:9090/-/reload

# 3. Check Prometheus -> AlertManager connectivity
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s http://alertmanager:9093/-/healthy

# 4. Check AlertManager configuration
kubectl get configmap -n victor-monitoring alertmanager-config -o yaml

# 5. Test alert manually (trigger alert expression)
curl -s "http://localhost:9090/api/v1/query?query=victor_cache_hit_rate+%3C+0.2" | jq '.data.result'
```

---

### Issue 4: High Storage Utilization

**Symptoms**:
- Prometheus PVC near capacity
- Warning: "Disk space low"

**Diagnosis**:
```bash
# Check PVC usage
kubectl get pvc -n victor-monitoring
kubectl exec -n victor-monitoring deployment/prometheus -- df -h /prometheus
```

**Common Causes**:
1. High cardinality metrics (too many label combinations)
2. Long retention period (30 days)
3. High-frequency scraping
4. Large metric samples

**Resolution**:
```bash
# 1. Check Prometheus stats
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s http://localhost:9090/api/v1/status/tsdb | jq '.'

# 2. Identify high-cardinality metrics
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s http://localhost:9090/api/v1/label/__name__/values | jq '.data[]' | \
  while read metric; do
    count=$(kubectl exec -n victor-monitoring deployment/prometheus -- \
      curl -s "http://localhost:9090/api/v1/label/${metric}/values" | jq '.data | length')
    echo "${metric}: ${count}"
  done | sort -t: -k2 -rn | head -20

# 3. Reduce retention time (temporary fix)
kubectl patch configmap prometheus-config -n victor-monitoring --type=json \
  -p='[{"op": "replace", "path": "/data/prometheus.yml", "value": "..."}]'

# 4. Clean old data
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -X POST http://localhost:9090/api/v1/admin/tsdb/delete_series?match[]=[]

# 5. Increase PVC size (long-term fix)
kubectl patch pvc prometheus-pvc -n victor-monitoring -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'
```

---

### Issue 5: OTEL Collector CrashLoopBackOff

**Symptoms**:
- OTEL collector pod keeps restarting
- Logs show connection errors

**Diagnosis**:
```bash
# Check pod status
kubectl get pods -n victor-monitoring -l app=otel-collector

# Check logs
kubectl logs -n victor-monitoring deployment/otel-collector --tail=50

# Describe pod
kubectl describe pod -n victor-monitoring -l app=otel-collector
```

**Common Causes**:
1. Configuration error
2. Cannot connect to Victor AI pods
3. Cannot connect to Prometheus
4. Resource limits too low

**Resolution**:
```bash
# 1. Check OTEL collector configuration
kubectl get configmap -n victor-monitoring otel-collector-config -o yaml

# 2. Test connectivity to Victor AI pods
kubectl run -n victor-monitoring test-pod --image=busybox --rm -it --restart=Never -- \
  wget -O- http://victor-api.victor-ai.svc.cluster.local:8000/metrics

# 3. Test connectivity to Prometheus
kubectl run -n victor-monitoring test-pod --image=busybox --rm -it --restart=Never -- \
  wget -O- http://prometheus:9090/-/healthy

# 4. Increase resource limits
kubectl set resources deployment/otel-collector -n victor-monitoring \
  --limits=cpu=500m,memory=512Mi --requests=cpu=250m,memory=256Mi

# 5. Validate configuration
kubectl exec -n victor-monitoring deployment/otel-collector -- \
  /otelcol --config=/etc/otelcol/config.yaml validate
```

---

## Maintenance Procedures

### Weekly Maintenance

#### 1. Review Alert Rules

```bash
# Review alert firing patterns
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
open "http://localhost:9090/alerts"

# Check for false positives
# Document any adjustments needed
```

#### 2. Check Storage Growth

```bash
# Monitor Prometheus storage growth
kubectl exec -n victor-monitoring deployment/prometheus -- df -h /prometheus

# Compare week-over-week
# If growth rate > 10GB/week, investigate high-cardinality metrics
```

#### 3. Backup Grafana Dashboards

```bash
# Export all dashboards
kubectl exec -n victor-monitoring deployment/grafana -- \
  curl -s http://localhost:3000/api/search | jq '.[] | .uid' | \
  while read uid; do
    kubectl exec -n victor-monitoring deployment/grafana -- \
      curl -s "http://localhost:3000/api/dashboards/uid/${uid}" | \
      jq '.dashboard' > "backup-dashboard-${uid}.json"
  done
```

#### 4. Review and Update Documentation

- Update this runbook with new procedures
- Document any new issues and resolutions
- Share learnings with team

### Monthly Maintenance

#### 1. Performance Review

```bash
# Generate monthly performance report
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &

# Cache performance (average hit rate)
curl -s "http://localhost:9090/api/v1/query_range?query=avg(victor_cache_hit_rate)&start=$(date -d '30 days ago' +%s)&end=$(date +%s)&step=1h" | \
  jq '.data.result[0].values[-1][1]'

# Tool execution latency (P95)
curl -s "http://localhost:9090/api/v1/query_range?query=histogram_quantile(0.95,sum(rate(victor_tool_duration_ms_bucket[5m]))+by+(le))&start=$(date -d '30 days ago' +%s)&end=$(date +%s)&step=1h" | \
  jq '.data.result[0].values[-1][1]'

# Error rate
curl -s "http://localhost:9090/api/v1/query_range?query=victor_tool_error_rate&start=$(date -d '30 days ago' +%s)&end=$(date +%s)&step=1h" | \
  jq '.data.result[0].values[-1][1]'
```

#### 2. Capacity Planning

```bash
# Review resource usage trends
kubectl top pods -n victor-monitoring --sort-by=cpu
kubectl top pods -n victor-monitoring --sort-by=memory

# Project future needs
# Plan for scale-out if needed
```

#### 3. Security Updates

```bash
# Check for image updates
kubectl get pods -n victor-monitoring -o jsonpath='{range .items[*]}{.spec.containers[*].image}{"\n"}{end}' | \
  sort -u

# Update images to latest stable versions
# Test in staging first
```

#### 4. Alert Rule Tuning

```bash
# Review alert firing history
# Adjust thresholds based on observed patterns
# Add new rules for emerging patterns
```

### Quarterly Maintenance

#### 1. Architecture Review

- Review monitoring architecture
- Evaluate new tools/technologies
- Consider consolidating or expanding stack

#### 2. Disaster Recovery Test

```bash
# Test backup and restore procedures
# Document recovery time objectives
# Update runbook with lessons learned
```

#### 3. Cost Optimization

- Review resource allocation
- Right-size instances based on actual usage
- Consider reserved instances for predictable workloads

---

## Escalation Procedures

### Escalation Matrix

| Severity | Level 1 (On-Call) | Level 2 (Team Lead) | Level 3 (Management) |
|----------|------------------|---------------------|---------------------|
| Critical | 15 minutes       | 30 minutes          | 1 hour              |
| Warning  | 1 hour           | 4 hours             | Next business day   |
| Info     | Next day         | Weekly review       | Monthly review      |

### Escalation Triggers

**Escalate to Level 2** if:
- Issue not resolved within 30 minutes (critical)
- Multiple critical alerts firing simultaneously
- Service disruption affecting users
- Unclear root cause

**Escalate to Level 3** if:
- Issue not resolved within 1 hour (critical)
- Complete service outage
- Data loss or corruption
- Security incident

### Contact Information

**Level 1 (On-Call Engineer)**:
- Primary: oncall@victor-ai.com
- Slack: #victor-oncall
- Pager: +1-555-0100

**Level 2 (Team Lead)**:
- Primary: team-lead@victor-ai.com
- Slack: #victor-leads

**Level 3 (Management)**:
- Director: director@victor-ai.com
- Slack: #victor-management

### Incident Communication Template

```markdown
## Victor AI Incident Report

**Severity**: [Critical/Warning/Info]
**Start Time**: YYYY-MM-DD HH:MM:SS UTC
**Duration**: X hours
**Status**: [Active/Resolved/Monitoring]

### Summary
[Brief description of the incident]

### Impact
- [ ] Service disruption
- [ ] Data loss
- [ ] User-facing error
- [ ] Performance degradation

### Root Cause
[What caused the incident]

### Resolution
[What was done to fix it]

### Timeline
- HH:MM: Incident detected
- HH:MM: Investigation started
- HH:MM: Root cause identified
- HH:MM: Fix implemented
- HH:MM: Service restored

### Action Items
- [ ] Prevent recurrence
- [ ] Update documentation
- [ ] Improve monitoring
- [ ] Post-mortem scheduled
```

---

## Appendix

### Useful Prometheus Queries

```promql
# Cache hit rate over time
rate(victor_cache_hits_total[5m]) / rate(victor_cache_operations_total[5m])

# P95 tool execution latency by tool
histogram_quantile(0.95, sum(rate(victor_tool_duration_ms_bucket[5m])) by (le, tool))

# Provider error rate
sum(rate(victor_provider_errors_total[5m])) by (provider) /
sum(rate(victor_provider_requests_total[5m])) by (provider)

# Overall health score
victor_performance_health_score

# Top 10 slowest tools
topk(10, histogram_quantile(0.95, sum(rate(victor_tool_duration_ms_bucket[5m])) by (le, tool)))

# Memory usage trend
victor_system_memory_bytes

# CPU usage trend
victor_system_cpu_percent

# Thread count
victor_system_threads

# Provider health by provider
victor_provider_health

# Cache utilization by namespace
victor_cache_utilization
```

### Grafana Dashboard Panels

The Victor AI performance dashboard includes 17 panels:

1. **Performance Health Score** (Gauge)
2. **Cache Hit Rate** (Time series graph)
3. **Cache Utilization** (Time series graph)
4. **Tool Selection Latency P95** (Time series graph)
5. **Tool Execution Latency P95** (Time series graph)
6. **Tool Error Rate** (Time series graph)
7. **Provider Health** (Stat)
8. **Provider Request Rate** (Time series graph)
9. **Provider Latency P95** (Time series graph)
10. **System Memory Usage** (Time series graph)
11. **System CPU Usage** (Time series graph)
12. **Thread Count** (Time series graph)
13. **Tool Execution Count** (Time series graph)
14. **Provider Error Rate** (Time series graph)
15. **Cache Eviction Rate** (Time series graph)
16. **Active Alerts** (Stat)
17. **Uptime** (Stat)

### Alert Rules Summary

| Alert Name | Severity | Threshold | Duration |
|------------|----------|-----------|----------|
| HighCacheMissRate | Warning | Hit rate < 40% | 5m |
| CriticalCacheMissRate | Critical | Hit rate < 20% | 10m |
| HighCacheUtilization | Warning | Utilization > 80% | 10m |
| CriticalCacheUtilization | Critical | Utilization > 95% | 5m |
| HighCacheEvictionRate | Warning | > 10 evictions/s | 5m |
| HighToolSelectionLatency | Warning | P95 > 1ms | 10m |
| CriticalToolSelectionLatency | Critical | P95 > 5ms | 5m |
| HighToolSelectionMissRate | Warning | Miss rate > 70% | 5m |
| HighMemoryUsage | Warning | Memory > 1GB | 10m |
| CriticalMemoryUsage | Critical | Memory > 2GB | 5m |
| HighCPUUsage | Warning | CPU > 80% | 10m |
| CriticalCPUUsage | Critical | CPU > 95% | 5m |
| HighThreadCount | Warning | Threads > 100 | 10m |
| HighToolExecutionLatency | Warning | P95 > 1s | 10m |
| CriticalToolExecutionLatency | Critical | P95 > 5s | 5m |
| HighToolErrorRate | Warning | Error rate > 5% | 10m |
| CriticalToolErrorRate | Critical | Error rate > 15% | 5m |
| UnhealthyProviders | Warning | Any unhealthy | 5m |
| HighProviderFailureRate | Warning | Failure rate > 10% | 5m |
| CriticalProviderFailureRate | Critical | Failure rate > 30% | 2m |
| HighProviderLatency | Warning | P95 > 10s | 10m |
| SlowStartup | Info | Startup in progress | - |
| PerformanceDegraded | Warning | Health score < 70 | 10m |
| PerformanceCritical | Critical | Health score < 50 | 5m |

---

**Document Owner**: Victor AI Operations Team
**Review Cycle**: Monthly
**Next Review**: 2026-02-21
