# Victor AI - Helm Chart

Enterprise-Ready AI Coding Assistant deployment via Helm.

## Installation

### Add Helm Repository (if applicable)

```bash
helm repo add victor https://vijayksingh.github.io/victor-charts
helm repo update
```

### Install Chart

```bash
# Install with default values
helm install victor ./config/helm/victor

# Install with custom values
helm install victor ./config/helm/victor -f custom-values.yaml

# Install into specific namespace
helm install victor ./config/helm/victor --namespace victor --create-namespace
```

### Upgrade

```bash
helm upgrade victor ./config/helm/victor -f custom-values.yaml
```

### Uninstall

```bash
helm uninstall victor
```

## Configuration

### Required Values

No required values - chart will work with defaults.

### Common Configuration

```yaml
# Replica count
replicaCount: 3

# Image configuration
image:
  repository: vijayksingh/victor
  tag: "0.5.0"
  pullPolicy: Always

# Resource limits
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

# Ingress configuration
ingress:
  enabled: true
  className: nginx
  hosts:
  - host: victor.example.com
    paths:
    - path: /
      pathType: Prefix
```

### API Keys Configuration

Create a secret file `secrets.yaml`:

```yaml
secrets:
  anthropicApiKey: "your-anthropic-key"
  openaiApiKey: "your-openai-key"
  googleApiKey: "your-google-key"
```

Install with secrets:

```bash
helm install victor ./config/helm/victor -f secrets.yaml
```

Or use Kubernetes secrets:

```bash
kubectl create secret generic victor-secrets \
  --from-literal=anthropicApiKey='your-key' \
  --from-literal=openaiApiKey='your-key' \
  --namespace=victor
```

### Environment-Specific Values

#### Development

```yaml
replicaCount: 1

resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"

autoscaling:
  enabled: false

config:
  environment: "development"
  logLevel: "DEBUG"
```

#### Production

```yaml
replicaCount: 3

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10

podDisruptionBudget:
  enabled: true
  minAvailable: 2

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"

config:
  environment: "production"
  logLevel: "INFO"
```

## Monitoring

The chart includes ServiceMonitor and PrometheusRule resources for Prometheus Operator.

```yaml
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
  prometheusRules:
    enabled: true
```

## Persistence

Configure persistent storage for Victor data:

```yaml
persistence:
  enabled: true
  storageClass: fast-ssd
  accessModes:
  - ReadWriteOnce
  size: 10Gi
```

## Values Reference

See `values.yaml` for all available configuration options.

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n victor
kubectl describe pod <pod-name> -n victor
```

### View Logs

```bash
kubectl logs -f <pod-name> -n victor
```

### Port Forward

```bash
kubectl port-forward svc/victor 8000:80 -n victor
```

### Debug Installation

```bash
helm template victor ./config/helm/victor --debug
```

## Support

- Documentation: https://github.com/vijayksingh/victor#readme
- Issues: https://github.com/vijayksingh/victor/issues
