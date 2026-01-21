# Helm Chart Guide for Victor AI

This directory contains the Helm chart for deploying Victor AI on Kubernetes.

## Quick Start

### Prerequisites

- Helm 3.x installed
- Kubernetes cluster configured
- kubectl configured

### Installation

```bash
# Add Helm repository (if using remote chart)
helm repo add victor-ai https://charts.victor.ai
helm repo update

# Install from local chart
helm install victor-ai ./deployment/helm

# Install from remote repository
helm install victor-ai victor-ai/victor-ai

# Install with custom values
helm install victor-ai ./deployment/helm -f deployment/helm/values-prod.yaml

# Install in specific namespace
helm install victor-ai ./deployment/helm --namespace victor-ai-prod --create-namespace
```

### Upgrade

```bash
# Upgrade release
helm upgrade victor-ai ./deployment/helm

# Upgrade with new values
helm upgrade victor-ai ./deployment/helm -f deployment/helm/values-prod.yaml

# Upgrade with specific image version
helm upgrade victor-ai ./deployment/helm --set image.tag=0.5.2

# Upgrade and reuse existing values
helm upgrade victor-ai ./deployment/helm --reuse-values
```

### Uninstall

```bash
# Uninstall release
helm uninstall victor-ai

# Uninstall with namespace cleanup
helm uninstall victor-ai --namespace victor-ai-prod

# Uninstall and keep history
helm uninstall victor-ai --keep-history
```

## Configuration

### Values Overview

The Helm chart uses the following default values (see `values.yaml`):

```yaml
# Deployment
replicaCount: 3
image:
  repository: victorai/victor
  tag: "0.5.1"
  pullPolicy: Always

# Service
service:
  type: ClusterIP
  port: 80
  metricsPort: 9090

# Ingress
ingress:
  enabled: false
  className: "nginx"

# Resources
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

# Autoscaling
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

# Configuration
config:
  profile: production
  logLevel: INFO
  maxWorkers: 4
```

### Environment-Specific Values

#### Development

```bash
helm install victor-ai-dev ./deployment/helm \
  -f deployment/helm/values.yaml \
  --set replicaCount=1 \
  --set config.profile=development \
  --set config.logLevel=DEBUG \
  --namespace victor-ai-dev
```

#### Staging

```bash
helm install victor-ai-staging ./deployment/helm \
  -f deployment/helm/values.yaml \
  --set replicaCount=2 \
  --set config.profile=staging \
  --set resources.requests.cpu=500m \
  --namespace victor-ai-staging
```

#### Production

```bash
helm install victor-ai-prod ./deployment/helm \
  -f deployment/helm/values-prod.yaml \
  --namespace victor-ai-prod
```

## Advanced Configuration

### Using Existing Secrets

```bash
# Create secret manually
kubectl create secret generic victor-ai-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=anthropic-api-key="..." \
  --namespace victor-ai-prod

# Use existing secret
helm install victor-ai ./deployment/helm \
  --set secrets.existingSecret=victor-ai-secrets \
  --namespace victor-ai-prod
```

### Enabling Ingress

```bash
helm install victor-ai ./deployment/helm \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set ingress.hosts[0].host=victor-ai.example.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.tls[0].secretName=victor-ai-tls \
  --set ingress.tls[0].hosts[0]=victor-ai.example.com \
  --namespace victor-ai-prod
```

### Using PostgreSQL Dependency

```bash
helm install victor-ai ./deployment/helm \
  --set postgresql.enabled=true \
  --set postgresql.auth.password=secretpassword \
  --set postgresql.auth.database=victor \
  --set secrets.database.url="postgresql://victor:secretpassword@victor-ai-postgresql:5432/victor" \
  --namespace victor-ai-prod
```

### Using Redis Dependency

```bash
helm install victor-ai ./deployment/helm \
  --set redis.enabled=true \
  --set cache.backend=redis \
  --set secrets.redis.url="redis://victor-ai-redis-master:6379/0" \
  --namespace victor-ai-prod
```

### Custom Resources

```bash
helm install victor-ai ./deployment/helm \
  --set resources.requests.cpu=2000m \
  --set resources.requests.memory=2Gi \
  --set resources.limits.cpu=8000m \
  --set resources.limits.memory=8Gi \
  --namespace victor-ai-prod
```

### Node Selector and Tolerations

```bash
helm install victor-ai ./deployment/helm \
  --set nodeSelector.nodepool=production \
  --set tolerations[0].key=dedicated \
  --set tolerations[0].operator=Equal \
  --set tolerations[0].value=victor-ai \
  --set tolerations[0].effect=NoSchedule \
  --namespace victor-ai-prod
```

### Affinity Rules

```bash
helm install victor-ai ./deployment/helm \
  --set affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchLabels.app=victor-ai \
  --set affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey=kubernetes.io/hostname \
  --namespace victor-ai-prod
```

## Operations

### Check Release Status

```bash
# List releases
helm list --namespace victor-ai-prod

# Get release status
helm status victor-ai --namespace victor-ai-prod

# Get release values
helm get values victor-ai --namespace victor-ai-prod

# Get all manifests
helm get manifest victor-ai --namespace victor-ai-prod

# Get release history
helm history victor-ai --namespace victor-ai-prod
```

### Debugging

```bash
# Dry-run installation
helm install victor-ai ./deployment/helm --dry-run --debug

# Template rendering
helm template victor-ai ./deployment/helm

# Validate chart
helm lint ./deployment/helm

# Show diff for upgrade
helm diff upgrade victor-ai ./deployment/helm --namespace victor-ai-prod

# Test release
helm test victor-ai --namespace victor-ai-prod
```

### Rollback

```bash
# Rollback to previous release
helm rollback victor-ai --namespace victor-ai-prod

# Rollback to specific revision
helm rollback victor-ai 2 --namespace victor-ai-prod

# View history before rollback
helm history victor-ai --namespace victor-ai-prod
```

### Scaling

```bash
# Scale using HPA (automatic)
kubectl scale hpa victor-ai --replicas=10 -n victor-ai-prod

# Disable HPA and manually scale
helm upgrade victor-ai ./deployment/helm \
  --set autoscaling.enabled=false \
  --set replicaCount=10 \
  --namespace victor-ai-prod
```

## Monitoring

### Using Prometheus Operator

```bash
# Enable ServiceMonitor
helm install victor-ai ./deployment/helm \
  --set monitoring.enabled=true \
  --set monitoring.serviceMonitor.enabled=true \
  --namespace victor-ai-prod
```

### Custom Metrics

```yaml
# In custom-values.yaml
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
    scrapeTimeout: 10s
    relabelings:
      - sourceLabels: [__meta_kubernetes_pod_node_name]
        targetLabel: node
```

## Secrets Management

### Using External Secrets

```yaml
# External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: victor-ai-secrets
  namespace: victor-ai-prod
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: victor-ai
  data:
    - secretKey: database-url
      remoteRef:
        key: victor/prod/database-url
```

### Using Sealed Secrets

```bash
# Create sealed secret
kubectl create secret generic victor-ai-secrets \
  --from-literal=database-url="..." \
  --dry-run=client -o yaml | kubeseal -f - \
  > deployment/kubernetes/base/sealed-secret.yaml

# Apply sealed secret
kubectl apply -f deployment/kubernetes/base/sealed-secret.yaml
```

## Upgrading Chart

### Major Version Upgrade

```bash
# Check upgrade notes
helm get notes victor-ai --namespace victor-ai-prod

# Test upgrade in staging
helm upgrade victor-ai-staging victor-ai/victor-ai \
  --version 0.6.0 \
  --namespace victor-ai-staging --dry-run

# Production upgrade
helm upgrade victor-ai victor-ai/victor-ai \
  --version 0.6.0 \
  --namespace victor-ai-prod \
  --timeout 10m \
  --wait \
  --atomic
```

### Zero-Downtime Upgrade

```bash
# Use rolling update with surge
helm upgrade victor-ai ./deployment/helm \
  --set replicaCount=3 \
  --set strategy.rollingUpdate.maxSurge=1 \
  --set strategy.rollingUpdate.maxUnavailable=0 \
  --namespace victor-ai-prod
```

## Backup and Restore

### Backup Release

```bash
# Backup release values
helm get values victor-ai --namespace victor-ai-prod > victor-ai-values-backup.yaml

# Backup release manifest
helm get manifest victor-ai --namespace victor-ai-prod > victor-ai-manifest-backup.yaml

# Backup all releases
helm list --all-namespaces -o json > helm-releases-backup.json
```

### Restore Release

```bash
# Restore from backup
helm install victor-ai-restored ./deployment/helm \
  -f victor-ai-values-backup.yaml \
  --namespace victor-ai-prod-restore
```

## Troubleshooting

### Helm Installation Fails

```bash
# Check helm version
helm version

# Check connection to cluster
kubectl cluster-info

# Check if release exists
helm status victor-ai --namespace victor-ai-prod

# Check helm logs
helm ls --all-namespaces

# Force delete release (if stuck)
helm delete victor-ai --namespace victor-ai-prod --no-hooks
```

### Pod Fails to Start

```bash
# Check pod status
kubectl get pods -n victor-ai-prod -l app.kubernetes.io/name=victor-ai

# Describe pod
kubectl describe pod <pod-name> -n victor-ai-prod

# Check pod logs
kubectl logs <pod-name> -n victor-ai-prod

# Check events
kubectl get events -n victor-ai-prod --sort-by='.lastTimestamp'
```

### Image Pull Errors

```bash
# Check if secret exists
kubectl get secret regcred -n victor-ai-prod

# Create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=<username> \
  --docker-password=<password> \
  -n victor-ai-prod

# Update chart to use secret
helm upgrade victor-ai ./deployment/helm \
  --set imagePullSecrets[0].name=regcred \
  --namespace victor-ai-prod
```

## Best Practices

1. **Always test in staging** before production deployments
2. **Use version pinning** for chart versions
3. **Enable HPA** for automatic scaling
4. **Use resource limits** to prevent resource exhaustion
5. **Configure ingress** with TLS certificates
6. **Use secrets management** (SealedSecrets, External Secrets)
7. **Enable monitoring** with Prometheus/Grafana
8. **Configure PDB** for high availability
9. **Use node selectors** for workload isolation
10. **Keep values files in Git** for version control

## Chart Testing

### Lint Chart

```bash
# Lint chart
helm lint ./deployment/helm

# Lint with specific values
helm lint ./deployment/helm --values deployment/helm/values-prod.yaml
```

### Unit Tests

```bash
# Install helm-unittest plugin
helm plugin install https://github.com/quintush/helm-unittest

# Run tests
helm unittest ./deployment/helm
```

### Integration Tests

```bash
# Install chart in test namespace
helm install victor-ai-test ./deployment/helm \
  --namespace victor-ai-test \
  --set replicaCount=1

# Run tests
helm test victor-ai-test --namespace victor-ai-test

# Cleanup
helm uninstall victor-ai-test --namespace victor-ai-test
```

## Additional Resources

- [Helm Documentation](https://helm.sh/docs/)
- [Helm Best Practices](https://helm.sh/docs/chart_best_practices/)
- [Helm Charts Guide](https://helm.sh/docs/topics/charts/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
