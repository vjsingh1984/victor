# Victor AI Monitoring Troubleshooting Guide

**Version**: 0.5.0
**Last Updated**: 2026-01-21
**Purpose**: Diagnostic procedures and solutions for common monitoring issues

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Component-Specific Issues](#component-specific-issues)
3. [Data Flow Issues](#data-flow-issues)
4. [Alerting Issues](#alerting-issues)
5. [Performance Issues](#performance-issues)
6. [Network Issues](#network-issues)
7. [Storage Issues](#storage-issues)
8. [Advanced Diagnostics](#advanced-diagnostics)

---

## Quick Diagnostics

### Health Check Script

```bash
#!/bin/bash
# Quick health check for Victor AI monitoring stack

echo "=== Victor AI Monitoring Health Check ==="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check pods
echo "1. Checking pod status..."
pods=$(kubectl get pods -n victor-monitoring -o json)
pod_count=$(echo "$pods" | jq '.items | length')
ready_count=$(echo "$pods" | jq '[.items[] | select(.status.containerStatuses[].ready == true)] | length')

if [ "$pod_count" -eq "$ready_count" ]; then
    echo -e "${GREEN}✓ All $pod_count pods are running${NC}"
else
    echo -e "${RED}✗ $ready_count/$pod_count pods are ready${NC}"
    echo "$pods" | jq -r '.items[] | select(.status.containerStatuses[].ready == false) | "  - \(.metadata.name): \(.status.containerStatuses[].state.waiting.reason)"'
fi
echo ""

# Check services
echo "2. Checking services..."
services=$(kubectl get svc -n victor-monitoring -o json)
service_count=$(echo "$services" | jq '.items | length')
echo -e "${GREEN}✓ $service_count services defined${NC}"
echo ""

# Check PVCs
echo "3. Checking persistent volumes..."
pvcs=$(kubectl get pvc -n victor-monitoring -o json)
pvc_count=$(echo "$pvcs" | jq '.items | length')
bound_count=$(echo "$pvcs" | jq '[.items[] | select(.status.phase == "Bound")] | length')

if [ "$pvc_count" -eq "$bound_count" ]; then
    echo -e "${GREEN}✓ All $pvc_count PVCs are bound${NC}"
else
    echo -e "${RED}✗ $bound_count/$pvc_count PVCs are bound${NC}"
fi
echo ""

# Check Prometheus targets
echo "4. Checking Prometheus targets..."
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 > /dev/null 2>&1 &
PF_PID=$!
sleep 2

targets=$(curl -s http://localhost:9090/api/v1/targets)
up_targets=$(echo "$targets" | jq '[.data.activeTargets[] | select(.health == "up")] | length')
total_targets=$(echo "$targets" | jq '.data.activeTargets | length')

kill $PF_PID 2>/dev/null

if [ "$up_targets" -eq "$total_targets" ]; then
    echo -e "${GREEN}✓ All $total_targets Prometheus targets are up${NC}"
else
    echo -e "${YELLOW}⚠ $up_targets/$total_targets Prometheus targets are up${NC}"
    echo "$targets" | jq -r '.data.activeTargets[] | select(.health != "up") | "  - \(.labels.job): \(.health)"'
fi
echo ""

# Check firing alerts
echo "5. Checking active alerts..."
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 > /dev/null 2>&1 &
PF_PID=$!
sleep 2

alerts=$(curl -s http://localhost:9090/api/v1/alerts)
firing_count=$(echo "$alerts" | jq '[.data.alerts[] | select(.state == "firing")] | length')

kill $PF_PID 2>/dev/null

if [ "$firing_count" -eq 0 ]; then
    echo -e "${GREEN}✓ No firing alerts${NC}"
else
    echo -e "${YELLOW}⚠ $firing_count alerts firing${NC}"
    echo "$alerts" | jq -r '.data.alerts[] | select(.state == "firing") | "  - \(.labels.alertname) (\(.labels.severity))"'
fi
echo ""

# Check disk usage
echo "6. Checking disk usage..."
prometheus_usage=$(kubectl exec -n victor-monitoring deployment/prometheus -- df -h /prometheus | tail -1 | awk '{print $5}' | sed 's/%//')
grafana_usage=$(kubectl exec -n victor-monitoring deployment/grafana -- df -h /var/lib/grafana | tail -1 | awk '{print $5}' | sed 's/%//')

if [ "$prometheus_usage" -lt 80 ]; then
    echo -e "${GREEN}✓ Prometheus disk usage: ${prometheus_usage}%${NC}"
else
    echo -e "${YELLOW}⚠ Prometheus disk usage: ${prometheus_usage}% (warning threshold: 80%)${NC}"
fi

if [ "$grafana_usage" -lt 80 ]; then
    echo -e "${GREEN}✓ Grafana disk usage: ${grafana_usage}%${NC}"
else
    echo -e "${YELLOW}⚠ Grafana disk usage: ${grafana_usage}% (warning threshold: 80%)${NC}"
fi
echo ""

echo "=== Health Check Complete ==="
```

### Common Symptoms and Quick Fixes

| Symptom | Quick Check | Quick Fix |
|---------|-------------|-----------|
| Grafana shows "No data" | Prometheus targets up? | Restart OTEL collector |
| Alerts not firing | Alert rules loaded? | Reload Prometheus config |
| High memory | Check pod memory | Restart pod, scale up |
| Disk full | Check Prometheus storage | Clean old data, expand PVC |
| Dashboard slow | Check query latency | Reduce time range, simplify queries |

---

## Component-Specific Issues

### Prometheus Issues

#### Issue: Prometheus Failed to Start

**Symptoms**:
- Pod in CrashLoopBackOff
- Logs show "error loading config"

**Diagnosis**:
```bash
# Check logs
kubectl logs -n victor-monitoring deployment/prometheus --tail=50

# Validate config
kubectl exec -n victor-monitoring deployment/prometheus -- \
  promtool check config /etc/prometheus/prometheus.yml
```

**Common Causes**:
1. YAML syntax error in ConfigMap
2. Invalid alert rule expression
3. Invalid scrape configuration

**Resolution**:
```bash
# 1. View current config
kubectl get configmap -n victor-monitoring prometheus-config -o yaml

# 2. Fix syntax errors (use YAML linter)
yamllint deployment/kubernetes/monitoring/prometheus-configmap.yaml

# 3. Validate alert rules
kubectl get configmap -n victor-monitoring prometheus-rules -o yaml
promtool check rules /path/to/alerts.yaml

# 4. Reload config (if Prometheus is running)
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -X POST http://localhost:9090/-/reload

# 5. Restart Prometheus (if reload fails)
kubectl rollout restart deployment/prometheus -n victor-monitoring
```

---

#### Issue: Prometheus Not Storing Data

**Symptoms**:
- Queries return no data
- UI shows "no data" for all metrics

**Diagnosis**:
```bash
# Check if metrics are being received
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/query?query=up | jq '.'

# Check TSDB stats
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s http://localhost:9090/api/v1/status/tsdb | jq '.'
```

**Common Causes**:
1. No targets configured
2. Targets not exposing `/metrics` endpoint
3. Network policies blocking scrapes
4. TSDB corruption

**Resolution**:
```bash
# 1. Verify targets are configured
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# 2. Check if Victor AI pods expose metrics
kubectl exec -n victor-ai deployment/victor-api -- curl -s http://localhost:8000/metrics | head -20

# 3. Verify network policies
kubectl get networkpolicies -A -o json | jq '.items[] | select(.metadata.namespace=="victor-ai" or .metadata.namespace=="victor-monitoring")'

# 4. Check OTEL collector
kubectl logs -n victor-monitoring deployment/otel-collector --tail=50

# 5. If TSDB corrupted, delete and recreate (CAUTION: data loss)
kubectl exec -n victor-monitoring deployment/prometheus -- rm -rf /prometheus/*
kubectl rollout restart deployment/prometheus -n victor-monitoring
```

---

#### Issue: High Memory Usage

**Symptoms**:
- Prometheus pod OOMKilled
- Memory usage constantly increasing

**Diagnosis**:
```bash
# Check memory usage
kubectl top pod -n victor-monitoring -l app=prometheus

# Check memory limits
kubectl get deployment prometheus -n victor-monitoring -o jsonpath='{.spec.template.spec.containers[0].resources}'

# Check TSDB memory stats
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s http://localhost:9090/api/v1/status/tsdb | jq '.data.heapSizeBytes'
```

**Common Causes**:
1. High cardinality metrics (too many label combinations)
2. Long retention period
3. High scrape frequency
4. Insufficient memory limits

**Resolution**:
```bash
# 1. Identify high-cardinality metrics
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s http://localhost:9090/api/v1/label/__name__/values | \
  jq -r '.data[]' | \
  while read metric; do
    count=$(kubectl exec -n victor-monitoring deployment/prometheus -- \
      curl -s "http://localhost:9090/api/v1/label/${metric}/values" | jq '.data | length')
    echo "${metric}: ${count}"
  done | sort -t: -k2 -rn | head -20

# 2. Reduce cardinality by removing high-cardinality labels
# Edit Victor AI code to reduce label values

# 3. Increase memory limits
kubectl set resources deployment/prometheus -n victor-monitoring \
  --limits=memory=4Gi --requests=memory=2Gi

# 4. Reduce retention time
# Edit prometheus-config ConfigMap, change retention.time from 30d to 15d
kubectl edit configmap prometheus-config -n victor-monitoring

# 5. Restart Prometheus
kubectl rollout restart deployment/prometheus -n victor-monitoring
```

---

### Grafana Issues

#### Issue: Grafana Login Failed

**Symptoms**:
- Cannot login to Grafana
- "Invalid username or password" error

**Diagnosis**:
```bash
# Check current credentials
kubectl get secret grafana-credentials -n victor-monitoring -o jsonpath='{.data}'
```

**Resolution**:
```bash
# 1. Reset admin password
kubectl exec -n victor-monitoring deployment/grafana -- \
  grafana-cli admin reset-admin-password newpassword123

# 2. Or update secret
kubectl create secret generic grafana-credentials -n victor-monitoring \
  --from-literal=admin-user=admin \
  --from-literal=admin-password=newpassword123 \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Restart Grafana
kubectl rollout restart deployment/grafana -n victor-monitoring
```

---

#### Issue: Dashboard Not Loading

**Symptoms**:
- Dashboard shows "Dashboard not found"
- Panels show "N/A"

**Diagnosis**:
```bash
# Check if dashboard configmap exists
kubectl get configmap -n victor-monitoring grafana-dashboards

# Check dashboard JSON
kubectl get configmap -n victor-monitoring grafana-dashboards -o jsonpath='{.data}' | jq 'keys'

# Check Grafana logs
kubectl logs -n victor-monitoring deployment/grafana --tail=50
```

**Common Causes**:
1. Dashboard JSON corrupted
2. ConfigMap not mounted correctly
3. Grafana provisioning misconfigured

**Resolution**:
```bash
# 1. Verify dashboard JSON is valid
kubectl get configmap -n victor-monitoring grafana-dashboards -o jsonpath='{.data.victor-performance\.json}' | jq '.'

# 2. Check provisioning config
kubectl get configmap -n victor-monitoring grafana-dashboard-providers -o yaml

# 3. Restart Grafana to reload dashboards
kubectl rollout restart deployment/grafana -n victor-monitoring

# 4. Manually import dashboard
# - Access Grafana UI
# - Go to Dashboards -> Import
# - Paste dashboard JSON from deployment/kubernetes/monitoring/dashboards/victor-performance.json
```

---

#### Issue: Data Source Connection Failed

**Symptoms**:
- Grafana shows "Data source connection failed"
- Panels show "Network Error"

**Diagnosis**:
```bash
# Check data source configuration
kubectl exec -n victor-monitoring deployment/grafana -- \
  curl -s http://localhost:3000/api/datasources | jq '.'

# Test connectivity from Grafana to Prometheus
kubectl exec -n victor-monitoring deployment/grafana -- \
  curl -s http://prometheus:9090/api/v1/query?query=up
```

**Common Causes**:
1. Prometheus URL incorrect
2. Network policies blocking connection
3. Prometheus service not accessible

**Resolution**:
```bash
# 1. Verify Prometheus service exists
kubectl get svc -n victor-monitoring prometheus

# 2. Check DNS resolution
kubectl exec -n victor-monitoring deployment/grafana -- nslookup prometheus

# 3. Check network policies
kubectl get networkpolicies -n victor-monitoring

# 4. Update data source configuration
kubectl exec -n victor-monitoring deployment/grafana -- \
  curl -X POST http://localhost:3000/api/datasources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy",
    "isDefault": true
  }'

# 5. Restart Grafana
kubectl rollout restart deployment/grafana -n victor-monitoring
```

---

### AlertManager Issues

#### Issue: Alerts Not Received

**Symptoms**:
- AlertManager UI shows alerts
- No email/Slack notifications received

**Diagnosis**:
```bash
# Check AlertManager configuration
kubectl get configmap -n victor-monitoring alertmanager-config -o yaml

# Check AlertManager logs
kubectl logs -n victor-monitoring deployment/alertmanager --tail=50

# Test alert notification
kubectl port-forward -n victor-monitoring svc/alertmanager 9093:9093 &
curl -X POST http://localhost:9093/api/v1/alerts -d '[{
  "labels": {"alertname": "TestAlert", "severity": "warning"},
  "annotations": {"description": "This is a test alert"}
}]'
```

**Common Causes**:
1. SMTP/Slack configuration incorrect
2. Webhook URL invalid
3. Receiver not configured correctly

**Resolution**:
```bash
# 1. Verify SMTP settings
kubectl get configmap -n victor-monitoring alertmanager-config -o yaml | grep -A 10 smtp_

# 2. Update SMTP configuration
kubectl edit configmap alertmanager-config -n victor-monitoring
# Update smtp_smarthost, smtp_from, smtp_auth_username, smtp_auth_password

# 3. Update Slack webhook URL
kubectl edit configmap alertmanager-config -n victor-monitoring
# Update api_url under slack_configs

# 4. Restart AlertManager
kubectl rollout restart deployment/alertmanager -n victor-monitoring

# 5. Test with simple webhook (use webhook.site for testing)
# Update alertmanager-config to use webhook.site URL
# Send test alert and check if received
```

---

### OpenTelemetry Collector Issues

#### Issue: OTEL Collector CrashLoopBackOff

**Symptoms**:
- Pod keeps restarting
- Logs show configuration errors

**Diagnosis**:
```bash
# Check logs
kubectl logs -n victor-monitoring deployment/otel-collector --tail=100

# Validate configuration
kubectl exec -n victor-monitoring deployment/otel-collector -- \
  /otelcol --config=/etc/otelcol/config.yaml validate
```

**Common Causes**:
1. Invalid YAML configuration
2. Unsupported receivers/exporters
3. Resource limits too low

**Resolution**:
```bash
# 1. Check OTEL collector configuration
kubectl get configmap -n victor-monitoring otel-collector-config -o yaml

# 2. Validate configuration syntax
# Download config, validate locally
kubectl get configmap -n victor-monitoring otel-collector-config -o yaml > otel-config.yaml
# Install otelcol and validate: otelcol --config otel-config.yaml validate

# 3. Increase resource limits
kubectl set resources deployment/otel-collector -n victor-monitoring \
  --limits=cpu=500m,memory=512Mi --requests=cpu=250m,memory=256Mi

# 4. Restart OTEL collector
kubectl rollout restart deployment/otel-collector -n victor-monitoring
```

---

#### Issue: Metrics Not Received by Prometheus

**Symptoms**:
- Prometheus targets up but no data
- OTEL collector logs show errors

**Diagnosis**:
```bash
# Check if OTEL collector is receiving metrics
kubectl logs -n victor-monitoring deployment/otel-collector --tail=50 | grep "received"

# Check if OTEL collector is exporting to Prometheus
kubectl logs -n victor-monitoring deployment/otel-collector --tail=50 | grep "export"

# Check OTEL collector metrics endpoint
kubectl port-forward -n victor-monitoring svc/otel-collector 8889:8889 &
curl -s http://localhost:8889/metrics | grep otelcol_
```

**Common Causes**:
1. Victor AI pods not exposing metrics
2. Network policies blocking scrapes
3. OTEL collector not scraping Victor AI pods

**Resolution**:
```bash
# 1. Check if Victor AI pods expose /metrics endpoint
kubectl exec -n victor-ai deployment/victor-api -- curl -s http://localhost:8000/metrics | head -20

# 2. Check network policies
kubectl get networkpolicies -A -o json | jq '.items[] | select(.metadata.namespace=="victor-ai")'

# 3. Create network policy to allow OTEL collector to scrape Victor AI pods
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-otel-scrape
  namespace: victor-ai
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: victor-monitoring
    ports:
    - protocol: TCP
      port: 8000
EOF

# 4. Restart OTEL collector
kubectl rollout restart deployment/otel-collector -n victor-monitoring
```

---

## Data Flow Issues

### Issue: End-to-End Data Flow Broken

**Symptoms**:
- Victor AI metrics not appearing in Grafana
- Data missing from dashboards

**Diagnostic Flowchart**:

```
1. Victor AI Pods
   ↓ Are pods running and exposing /metrics?
   ├─ No: Fix Victor AI pods
   └─ Yes: ↓

2. OTEL Collector
   ↓ Is OTEL collector scraping Victor AI pods?
   ├─ No: Fix OTEL collector configuration
   └─ Yes: ↓

3. Prometheus
   ↓ Is Prometheus scraping OTEL collector?
   ├─ No: Fix Prometheus scrape config
   └─ Yes: ↓

4. AlertManager
   ↓ Are alerts being fired?
   └─ Yes: ↓

5. Grafana
   ↓ Is Grafana querying Prometheus?
   └─ No: Fix Grafana data source
```

**Diagnostic Commands**:

```bash
#!/bin/bash
# End-to-end data flow check

echo "=== Victor AI Monitoring Data Flow Check ==="
echo ""

# 1. Check Victor AI pods
echo "1. Checking Victor AI pods..."
victor_pods=$(kubectl get pods -n victor-ai -l app=victor-api -o json)
victor_count=$(echo "$victor_pods" | jq '.items | length')
victor_ready=$(echo "$victor_pods" | jq '[.items[] | select(.status.containerStatuses[].ready == true)] | length')

if [ "$victor_count" -gt 0 ] && [ "$victor_count" -eq "$victor_ready" ]; then
    echo -e "✓ Victor AI pods running"

    # Check if metrics endpoint is accessible
    first_pod=$(echo "$victor_pods" | jq -r '.items[0].metadata.name')
    metrics=$(kubectl exec -n victor-ai "$first_pod" -- curl -s http://localhost:8000/metrics | head -20)
    if [ -n "$metrics" ]; then
        echo -e "✓ Metrics endpoint accessible"
        echo "   Sample metrics:"
        echo "$metrics" | head -5 | sed 's/^/   /'
    else
        echo -e "✗ Metrics endpoint not accessible"
    fi
else
    echo -e "✗ Victor AI pods not running"
fi
echo ""

# 2. Check OTEL collector
echo "2. Checking OTEL collector..."
otel_pods=$(kubectl get pods -n victor-monitoring -l app=otel-collector -o json)
otel_count=$(echo "$otel_pods" | jq '.items | length')
otel_ready=$(echo "$otel_pods" | jq '[.items[] | select(.status.containerStatuses[].ready == true)] | length')

if [ "$otel_count" -gt 0 ] && [ "$otel_count" -eq "$otel_ready" ]; then
    echo -e "✓ OTEL collector running"

    # Check if OTEL collector metrics endpoint is accessible
    kubectl port-forward -n victor-monitoring svc/otel-collector 8889:8889 > /dev/null 2>&1 &
    PF_PID=$!
    sleep 2

    otel_metrics=$(curl -s http://localhost:8889/metrics | grep "otelcol_receiver_" | head -5)
    if [ -n "$otel_metrics" ]; then
        echo -e "✓ OTEL collector metrics endpoint accessible"
        echo "   Sample receiver metrics:"
        echo "$otel_metrics" | sed 's/^/   /'
    else
        echo -e "✗ OTEL collector metrics endpoint not accessible"
    fi

    kill $PF_PID 2>/dev/null
else
    echo -e "✗ OTEL collector not running"
fi
echo ""

# 3. Check Prometheus
echo "3. Checking Prometheus..."
prometheus_pods=$(kubectl get pods -n victor-monitoring -l app=prometheus -o json)
prometheus_count=$(echo "$prometheus_pods" | jq '.items | length')
prometheus_ready=$(echo "$prometheus_pods" | jq '[.items[] | select(.status.containerStatuses[].ready == true)] | length')

if [ "$prometheus_count" -gt 0 ] && [ "$prometheus_count" -eq "$prometheus_ready" ]; then
    echo -e "✓ Prometheus running"

    # Check if Prometheus has data
    kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 > /dev/null 2>&1 &
    PF_PID=$!
    sleep 2

    prometheus_metrics=$(curl -s "http://localhost:9090/api/v1/query?query=up" | jq '.data.result | length')
    if [ "$prometheus_metrics" -gt 0 ]; then
        echo -e "✓ Prometheus has metrics ($prometheus_metrics series)"
    else
        echo -e "✗ Prometheus has no metrics"
    fi

    kill $PF_PID 2>/dev/null
else
    echo -e "✗ Prometheus not running"
fi
echo ""

# 4. Check Grafana
echo "4. Checking Grafana..."
grafana_pods=$(kubectl get pods -n victor-monitoring -l app=grafana -o json)
grafana_count=$(echo "$grafana_pods" | jq '.items | length')
grafana_ready=$(echo "$grafana_pods" | jq '[.items[] | select(.status.containerStatuses[].ready == true)] | length')

if [ "$grafana_count" -gt 0 ] && [ "$grafana_count" -eq "$grafana_ready" ]; then
    echo -e "✓ Grafana running"

    # Check if Grafana can query Prometheus
    grafana_query=$(kubectl exec -n victor-monitoring deployment/grafana -- \
        curl -s "http://prometheus:9090/api/v1/query?query=up" | jq '.data.result | length')
    if [ "$grafana_query" -gt 0 ]; then
        echo -e "✓ Grafana can query Prometheus"
    else
        echo -e "✗ Grafana cannot query Prometheus"
    fi
else
    echo -e "✗ Grafana not running"
fi
echo ""

echo "=== Data Flow Check Complete ==="
```

---

## Alerting Issues

### Issue: False Positive Alerts

**Symptoms**:
- Alerts firing when conditions are normal
- Alert threshold too sensitive

**Diagnosis**:
```bash
# Check current alert status
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'

# Check alert expression
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
open "http://localhost:9090/graph?g0.expr=victor_cache_hit_rate&g0.tab=0"
```

**Resolution**:
```bash
# 1. Analyze alert pattern
# - Check if alert fires consistently or intermittently
# - Check if threshold is appropriate for your workload

# 2. Adjust alert threshold
# Edit performance-alerts.yaml
vim deployment/kubernetes/monitoring/performance-alerts.yaml

# Example: Increase HighCacheMissRate threshold from 40% to 30%
# - alert: HighCacheMissRate
#   expr: |
#     victor_cache_hit_rate{namespace="overall"} < 0.3  # Changed from 0.4

# 3. Apply updated rules
kubectl apply -f deployment/kubernetes/monitoring/performance-alerts.yaml

# 4. Reload Prometheus
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -X POST http://localhost:9090/-/reload
```

---

### Issue: Alert Not Firing When Expected

**Symptoms**:
- Metrics exceed threshold but no alert
- Alert state shows "inactive"

**Diagnosis**:
```bash
# Check alert rule evaluation
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.name=="HighCacheMissRate")'

# Check alert expression manually
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s "http://localhost:9090/api/v1/query?query=victor_cache_hit_rate+%3C+0.4" | jq '.data.result'
```

**Common Causes**:
1. `for` duration not met
2. Alert expression incorrect
3. Label mismatch
4. Alert rule not loaded

**Resolution**:
```bash
# 1. Check alert rule syntax
kubectl get configmap -n victor-monitoring prometheus-rules -o yaml

# 2. Validate alert expression
# Test in Prometheus UI: http://localhost:9090/graph
# Paste alert expression and check if it returns data

# 3. Check `for` duration
# If `for: 5m`, alert only fires after 5 minutes of sustained breach

# 4. Reduce `for` duration for testing
vim deployment/kubernetes/monitoring/performance-alerts.yaml
# Change "for: 5m" to "for: 1m"

# 5. Apply and reload
kubectl apply -f deployment/kubernetes/monitoring/performance-alerts.yaml
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -X POST http://localhost:9090/-/reload
```

---

## Performance Issues

### Issue: Slow Grafana Dashboard

**Symptoms**:
- Dashboard takes > 10 seconds to load
- Panels timeout
- Browser becomes unresponsive

**Diagnosis**:
```bash
# Check query latency
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
time curl -s "http://localhost:9090/api/v1/query_range?query=victor_cache_hit_rate&start=$(date -d '7 days ago' +%s)&end=$(date +%s)&step=300" | jq '.data.result | length'

# Check Prometheus query stats
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/status/tsdb | jq '.data.'
```

**Common Causes**:
1. Querying too much data (long time ranges)
2. Complex queries (many joins, aggregations)
3. High cardinality metrics
4. Insufficient Prometheus resources

**Resolution**:
```bash
# 1. Reduce query time range
# - Instead of 7 days, use 24 hours
# - Use "Last 6 hours" instead of "Last 7 days"

# 2. Simplify queries
# - Use recording rules for complex queries
# - Pre-aggregate data

# 3. Increase Prometheus resources
kubectl set resources deployment/prometheus -n victor-monitoring \
  --limits=cpu=2000m,memory=4Gi --requests=cpu=1000m,memory=2Gi

# 4. Add recording rules for expensive queries
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-recording-rules
  namespace: victor-monitoring
data:
  recording.rules.yaml: |
    groups:
      - name: victor_recording
        interval: 30s
        rules:
          - record: victor_cache_hit_rate:5m
            expr: avg(victor_cache_hit_rate[5m])
EOF

# 5. Restart Prometheus
kubectl rollout restart deployment/prometheus -n victor-monitoring
```

---

### Issue: High CPU Usage by Prometheus

**Symptoms**:
- Prometheus pod using 100% CPU
- Other pods throttled
- Node CPU saturated

**Diagnosis**:
```bash
# Check CPU usage
kubectl top pod -n victor-monitoring -l app=prometheus

# Check scrape rate
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s http://localhost:9090/api/v1/status/tsdb | jq '.data.sampleCount'

# Check number of series
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s http://localhost:9090/api/v1/status/tsdb | jq '.data.numSeries'
```

**Common Causes**:
1. Too many scrape targets
2. High scrape frequency
3. Too many metric series
4. Complex recording rules

**Resolution**:
```bash
# 1. Reduce scrape frequency
# Edit prometheus-config, change scrape interval from 15s to 30s
kubectl edit configmap prometheus-config -n victor-monitoring

# 2. Reduce number of series
# - Identify high-cardinality metrics
# - Drop unnecessary labels
# - Use metric_relabel_configs to drop series

# 3. Add metric relabeling to reduce series
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: victor-monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 30s
      evaluation_interval: 30s
    scrape_configs:
      - job_name: 'otel-collector'
        static_configs:
          - targets: ['otel-collector:8889']
        metric_relabel_configs:
          # Drop high-cardinality labels
          - source_labels: [pod]
            regex: '.*'
            action: labeldrop
EOF

# 4. Increase CPU limits
kubectl set resources deployment/prometheus -n victor-monitoring \
  --limits=cpu=2000m --requests=cpu=1000m

# 5. Restart Prometheus
kubectl rollout restart deployment/prometheus -n victor-monitoring
```

---

## Network Issues

### Issue: Cannot Access Monitoring Services

**Symptoms**:
- Port-forward fails
- Services timeout
- "Connection refused" errors

**Diagnosis**:
```bash
# Check service endpoints
kubectl get endpoints -n victor-monitoring

# Check pod IPs
kubectl get pods -n victor-monitoring -o wide

# Test network connectivity
kubectl run -n victor-monitoring test-pod --image=busybox --rm -it --restart=Never -- \
  wget -O- http://prometheus:9090/-/healthy
```

**Common Causes**:
1. Service not exposed
2. No pod endpoints
3. Network policies blocking
4. DNS resolution failed

**Resolution**:
```bash
# 1. Check service exists
kubectl get svc -n victor-monitoring

# 2. Check if pods are ready
kubectl get pods -n victor-monitoring

# 3. Check network policies
kubectl get networkpolicies -n victor-monitoring

# 4. Create allow-all network policy for testing
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-all
  namespace: victor-monitoring
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - {}
  egress:
  - {}
EOF

# 5. Test DNS resolution
kubectl run -n victor-monitoring test-dns --image=busybox --rm -it --restart=Never -- \
  nslookup prometheus
```

---

## Storage Issues

### Issue: Prometheus PVC Full

**Symptoms**:
- Prometheus logs "no space left on device"
- Metrics not being written
- Pod in CrashLoopBackOff

**Diagnosis**:
```bash
# Check disk usage
kubectl exec -n victor-monitoring deployment/prometheus -- df -h /prometheus

# Check PVC size
kubectl get pvc -n victor-monitoring prometheus-pvc

# Check TSDB size
kubectl exec -n victor-monitoring deployment/prometheus -- \
  du -sh /prometheus
```

**Resolution**:
```bash
# 1. Clean old data (delete data older than 15 days)
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -X POST 'http://localhost:9090/api/v1/admin/tsdb/delete_series?match[]=&start='"$(date -d '15 days ago' +%s)"'&end='"$(date +%s)

# 2. Compact TSDB to reclaim space
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -X POST http://localhost:9090/api/v1/admin/tsdb/compact

# 3. Reduce retention time (long-term fix)
kubectl edit configmap prometheus-config -n victor-monitoring
# Change storage.tsdb.retention.time from 30d to 15d

# 4. Expand PVC (if storage class supports expansion)
kubectl patch pvc prometheus-pvc -n victor-monitoring -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'

# 5. Restart Prometheus
kubectl rollout restart deployment/prometheus -n victor-monitoring
```

---

## Advanced Diagnostics

### Prometheus Query Performance Analysis

```bash
# Enable query logging
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -X PUT -H 'Content-Type: application/json' -d '{"log_level": "debug"}' http://localhost:9090/api/v1/status/runtimeinfo

# Analyze slow queries
kubectl logs -n victor-monitoring deployment/prometheus --tail=1000 | grep "query=" | \
  awk -F'time=' '{print $2}' | awk '{print $1}' | \
  sort -t's' -k1 -rn | head -20

# Check query statistics
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/status/tsdb | jq '.data.'
```

### Metrics Cardinality Analysis

```bash
# Identify high-cardinality metrics
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s http://localhost:9090/api/v1/label/__name__/values | \
  jq -r '.data[]' | \
  while read metric; do
    count=$(kubectl exec -n victor-monitoring deployment/prometheus -- \
      curl -s "http://localhost:9090/api/v1/label/${metric}/values" | jq '.data | length')
    echo "${metric}: ${count}"
  done | sort -t: -k2 -rn | head -20

# Analyze label cardinality for specific metric
kubectl exec -n victor-monitoring deployment/prometheus -- \
  curl -s "http://localhost:9090/api/v1/label/__name__/values" | \
  jq -r '.data[]' | \
  while read metric; do
    echo "=== $metric ==="
    kubectl exec -n victor-monitoring deployment/prometheus -- \
      curl -s "http://localhost:9090/api/v1/labels" | \
      jq -r '.data[]' | \
      while read label; do
        count=$(kubectl exec -n victor-monitoring deployment/prometheus -- \
          curl -s "http://localhost:9090/api/v1/label/${label}/values" | jq '.data | length')
        echo "  ${label}: ${count}"
      done | sort -t: -k2 -rn | head -5
  done | head -100
```

### Alert Performance Analysis

```bash
# Check alert evaluation duration
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/rules | \
  jq '.data.groups[].rules[] | select(.type=="alerting") | {name: .name, evaluationTime: .evaluationTime}' | \
  jq -s 'sort_by(.evaluationTime) | reverse | .[:10]'

# Identify slow alerts
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
curl -s http://localhost:9090/api/v1/rules | \
  jq '.data.groups[].rules[] | select(.type=="alerting" and .evaluationTime > 1) | {name: .name, evaluationTime: .evaluationTime}'
```

---

## Emergency Procedures

### Emergency Scale-Down

```bash
# Scale down monitoring to free resources
kubectl scale deployment/prometheus -n victor-monitoring --replicas=0
kubectl scale deployment/grafana -n victor-monitoring --replicas=0
kubectl scale deployment/alertmanager -n victor-monitoring --replicas=0

# Scale back up
kubectl scale deployment/prometheus -n victor-monitoring --replicas=1
kubectl scale deployment/grafana -n victor-monitoring --replicas=1
kubectl scale deployment/alertmanager -n victor-monitoring --replicas=1
```

### Emergency Data Backup

```bash
# Backup Prometheus data
kubectl exec -n victor-monitoring deployment/prometheus -- tar czf /tmp/prometheus-backup.tar.gz /prometheus
kubectl cp victor-monitoring/$(kubectl get pod -n victor-monitoring -l app=prometheus -o jsonpath='{.items[0].metadata.name}'):/tmp/prometheus-backup.tar.gz ./prometheus-backup-$(date +%Y%m%d).tar.gz

# Backup Grafana dashboards
kubectl exec -n victor-monitoring deployment/grafana -- tar czf /tmp/grafana-backup.tar.gz /var/lib/grafana
kubectl cp victor-monitoring/$(kubectl get pod -n victor-monitoring -l app=grafana -o jsonpath='{.items[0].metadata.name}'):/tmp/grafana-backup.tar.gz ./grafana-backup-$(date +%Y%m%d).tar.gz
```

### Complete Monitoring Reset

```bash
#!/bin/bash
# Emergency monitoring reset - CAUTION: Deletes all data

set -e

echo "WARNING: This will delete all monitoring data!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted"
    exit 1
fi

# Delete all monitoring resources
kubectl delete namespace victor-monitoring

# Wait for deletion to complete
echo "Waiting for namespace deletion..."
kubectl wait --for=delete namespace/victor-monitoring --timeout=60s

# Redeploy
bash deployment/scripts/deploy_monitoring.sh

echo "Monitoring reset complete"
```

---

**Document Owner**: Victor AI Operations Team
**Review Cycle**: Monthly
**Next Review**: 2026-02-21
