# Victor AI - Complete Monitoring Stack Deployment

This directory contains comprehensive monitoring stack deployment scripts for Victor AI, including Prometheus, Grafana, and AlertManager with pre-configured dashboards and alerting rules.

## Quick Start

### 1. Deploy Complete Monitoring Stack

```bash
# Basic deployment
./deployment/scripts/deploy_monitoring_complete.sh

# With email notifications
./deployment/scripts/deploy_monitoring_complete.sh --email admin@example.com

# With Slack notifications
./deployment/scripts/deploy_monitoring_complete.sh --slack-webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Custom namespace and storage
./deployment/scripts/deploy_monitoring_complete.sh \
  --namespace monitoring-prod \
  --storage-class fast-ssd
```

### 2. Setup Secure Credentials

```bash
# Generate secure password
./deployment/scripts/setup_monitoring_credentials.sh --generate-password

# Set specific password
./deployment/scripts/setup_monitoring_credentials.sh \
  --password "your-secure-password"

# Setup email notifications
./deployment/scripts/setup_monitoring_credentials.sh \
  --email email-credentials.json

# Setup Slack notifications
./deployment/scripts/setup_monitoring_credentials.sh \
  --slack slack-webhook.txt
```

### 3. Verify Deployment

```bash
# Basic verification
./deployment/scripts/verify_monitoring_complete.sh

# Detailed verification
./deployment/scripts/verify_monitoring_complete.sh --detailed

# Auto-fix common issues
./deployment/scripts/verify_monitoring_complete.sh --fix

# Export report
./deployment/scripts/verify_monitoring_complete.sh \
  --export-report monitoring-report.txt
```

## Components

### Deployed Components

1. **Prometheus** - Metrics collection and storage
   - Persistent storage: 10Gi (configurable)
   - Retention: 15 days (configurable)
   - Port: 9090

2. **Grafana** - Visualization dashboards
   - 7 pre-configured dashboards
   - Persistent storage: 5Gi (configurable)
   - Port: 3000
   - Default credentials: admin / changeme123 (change immediately!)

3. **AlertManager** - Alert routing and notifications
   - Persistent storage: 2Gi (configurable)
   - Port: 9093
   - Supports: Email, Slack

### Dashboards (7 Total)

All dashboards are imported from `observability/dashboards/`:

1. **Overview** (`overview.json`) - System-wide overview
2. **Features** (`features.json`) - Feature flag metrics
3. **Performance** (`performance.json`) - Performance metrics
4. **Team Overview** (`team_overview.json`) - Multi-agent team overview
5. **Team Members** (`team_members.json`) - Individual team member metrics
6. **Team Performance** (`team_performance.json`) - Team performance analysis
7. **Team Recursion** (`team_recursion.json`) - Recursion depth tracking

### Alerting Rules (50+ Rules)

All rules are loaded from `observability/alerts/`:

- **rules.yml** - Core system alerts
- **team_alerts.yml** - Team coordination alerts

Categories:
- Feature performance alerts
- Resource exhaustion alerts
- Model and provider alerts
- Tool execution alerts
- System health alerts
- Feature adoption alerts
- Dependency health alerts
- Performance degradation alerts

## Accessing the Monitoring Stack

### Port Forwarding

```bash
# Grafana
kubectl port-forward -n victor-monitoring svc/grafana 3000:3000
# Open: http://localhost:3000

# Prometheus
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090
# Open: http://localhost:9090

# AlertManager
kubectl port-forward -n victor-monitoring svc/alertmanager 9093:9093
# Open: http://localhost:9093
```

### Ingress (Production)

Configure ingress for external access:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: monitoring-ingress
  namespace: victor-monitoring
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - grafana.victor-ai.example.com
    - prometheus.victor-ai.example.com
    - alertmanager.victor-ai.example.com
    secretName: monitoring-tls
  rules:
  - host: grafana.victor-ai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000
  - host: prometheus.victor-ai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
  - host: alertmanager.victor-ai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: alertmanager
            port:
              number: 9093
```

## Configuration

### Storage Configuration

Default storage requirements:
- Prometheus: 10Gi
- Grafana: 5Gi
- AlertManager: 2Gi

Customize storage:
```bash
./deploy_monitoring_complete.sh \
  --storage-class fast-ssd \
  --retention-days 30
```

Environment variables:
- `PROMETHEUS_RETENTION_SIZE` - Prometheus storage size (default: 10Gi)
- `GRAFANA_RETENTION_SIZE` - Grafana storage size (default: 5Gi)
- `ALERTMANAGER_RETENTION_SIZE` - AlertManager storage size (default: 2Gi)
- `RETENTION_DAYS` - Prometheus data retention (default: 15 days)

### Email Notification Setup

Create `email-credentials.json`:
```json
{
  "smtp_host": "smtp.gmail.com",
  "smtp_port": "587",
  "smtp_user": "your-email@gmail.com",
  "smtp_password": "your-app-password",
  "from_address": "alertmanager@victor-ai.com"
}
```

Apply credentials:
```bash
./setup_monitoring_credentials.sh --email email-credentials.json
```

### Slack Notification Setup

Create `slack-webhook.txt`:
```
https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

Apply credentials:
```bash
./setup_monitoring_credentials.sh --slack slack-webhook.txt
```

## Health Checks

The verification script performs comprehensive checks:

1. ✅ Kubernetes cluster connectivity
2. ✅ Namespace existence
3. ✅ Pod health and status
4. ✅ Service availability
5. ✅ Persistent volume claims
6. ✅ ConfigMaps and secrets
7. ✅ Prometheus configuration and targets
8. ✅ AlertManager configuration
9. ✅ Grafana access and dashboards
10. ✅ Alerting rules (50+ rules)
11. ✅ Metrics collection
12. ✅ Resource usage
13. ✅ Notification channels

Run verification:
```bash
./verify_monitoring_complete.sh --detailed
```

## Security Best Practices

### 1. Change Default Credentials

```bash
# Generate secure password
./setup_monitoring_credentials.sh --generate-password

# Or set specific password
./setup_monitoring_credentials.sh --password "your-secure-password"
```

### 2. Enable TLS/SSL

Use cert-manager for automatic certificate management:
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### 3. Restrict Secret Access

Configure RBAC:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: victor-monitoring
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get"]
  resourceNames: ["grafana-admin-credentials"]
```

### 4. Enable Secrets Encryption

Configure etcd encryption for Kubernetes secrets.

### 5. Network Policies

Restrict network access:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: monitoring-deny-ingress
  namespace: victor-monitoring
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n victor-monitoring

# Describe pod
kubectl describe pod <pod-name> -n victor-monitoring

# Check logs
kubectl logs <pod-name> -n victor-monitoring --previous
```

### Persistent Volume Issues

```bash
# Check PVC status
kubectl get pvc -n victor-monitoring

# Check storage class
kubectl get storageclass

# Describe PVC
kubectl describe pvc <pvc-name> -n victor-monitoring
```

### Grafana Login Issues

Reset admin password:
```bash
kubectl -n victor-monitoring create secret generic grafana-admin-credentials \
  --from-literal=admin-user=admin \
  --from-literal=admin-password=new-password \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n victor-monitoring rollout restart deployment/grafana
```

### Alerts Not Firing

```bash
# Check Prometheus rules
kubectl exec -n victor-monitoring \
  $(kubectl get pods -n victor-monitoring -l app.kubernetes.io/component=prometheus -o jsonpath='{.items[0].metadata.name}') \
  -- wget -q -O- http://localhost:9090/api/v1/rules | jq .

# Check AlertManager alerts
kubectl exec -n victor-monitoring \
  $(kubectl get pods -n victor-monitoring -l app.kubernetes.io/component=alertmanager -o jsonpath='{.items[0].metadata.name}') \
  -- wget -q -O- http://localhost:9093/api/v1/alerts | jq .
```

## Maintenance

### Backup Configuration

```bash
# Backup ConfigMaps
kubectl get configmap -n victor-monitoring -o yaml > monitoring-configs-backup.yaml

# Backup Secrets
kubectl get secrets -n victor-monitoring -o yaml > monitoring-secrets-backup.yaml
```

### Update Dashboards

Edit dashboards in `observability/dashboards/` and re-run:
```bash
./deploy_monitoring_complete.sh --skip-alerts
```

### Update Alerting Rules

Edit rules in `observability/alerts/` and re-run:
```bash
./deploy_monitoring_complete.sh --skip-dashboards
```

### Rotate Credentials

```bash
# Rotate all credentials
./setup_monitoring_credentials.sh --rotate

# Or rotate specific credentials
./setup_monitoring_credentials.sh --generate-password
```

### Scaling

Increase Prometheus storage:
```bash
kubectl edit pvc prometheus-data -n victor-monitoring
# Edit spec.resources.requests.storage
```

Scale resources:
```bash
kubectl edit deployment prometheus -n victor-monitoring
# Edit resources.requests and resources.limits
```

## Monitoring the Monitoring Stack

The monitoring stack monitors itself:

- **Prometheus**: Scrapes its own metrics on `http://localhost:9090/metrics`
- **Grafana**: Has built-in metrics dashboard
- **AlertManager**: Exposes metrics on `http://localhost:9093/metrics`

View internal metrics:
```bash
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
open http://localhost:9090/metrics
```

## Advanced Configuration

### Custom Prometheus Scraping Configs

Edit the ConfigMap:
```bash
kubectl edit configmap prometheus-config -n victor-monitoring
```

Then reload Prometheus:
```bash
kubectl exec -n victor-monitoring \
  $(kubectl get pods -n victor-monitoring -l app.kubernetes.io/component=prometheus -o jsonpath='{.items[0].metadata.name}') \
  -- wget -q --post-data="" http://localhost:9090/-/reload
```

### Custom Grafana Dashboards

1. Create dashboard JSON
2. Create ConfigMap:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-custom
  labels:
    grafana_dashboard: "1"
data:
  custom-dashboard.json: |
    { ...dashboard JSON... }
```

### Alert Silences

Create silence via AlertManager API:
```bash
kubectl exec -n victor-monitoring \
  $(kubectl get pods -n victor-monitoring -l app.kubernetes.io/component=alertmanager -o jsonpath='{.items[0].metadata.name}') \
  -- wget -q --post-data='{
    "matchers": [{"name": "alertname", "value": "TestAlert"}],
    "startsAt": "2024-01-01T00:00:00Z",
    "endsAt": "2024-12-31T23:59:59Z",
    "createdBy": "admin",
    "comment": "Maintenance silence"
  }' http://localhost:9093/api/v2/silences
```

## Integration with Victor AI

The monitoring stack automatically discovers Victor AI pods with label `app.kubernetes.io/name: victor`:

```yaml
metadata:
  labels:
    app.kubernetes.io/name: victor
```

Metrics exposed by Victor AI:
- `/metrics` - Prometheus metrics endpoint
- Port: 9090 (default)

View targets in Prometheus:
```bash
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
open http://localhost:9090/targets
```

## Support and Documentation

- **Victor AI Documentation**: `docs/` directory
- **Monitoring Deployment Guide**: `deployment/monitoring/DEPLOYMENT_GUIDE.md`
- **Runbook**: `deployment/monitoring/runbook.md`
- **Troubleshooting**: `deployment/monitoring/troubleshooting.md`

For issues and questions:
- Check troubleshooting guides
- Review pod logs
- Run verification script with `--detailed` flag
- Check AlertManager for active alerts

## License

Part of the Victor AI project. See main project LICENSE file.
