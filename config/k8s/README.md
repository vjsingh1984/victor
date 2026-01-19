# Victor AI - Kubernetes Deployment

Production-ready Kubernetes manifests for Victor AI.

## Quick Start

```bash
# Apply all manifests
kubectl apply -f config/k8s/

# Check status
kubectl get all -n victor
```

## Manifests

### Core Components

- **namespace.yaml** - Namespace for Victor resources
- **configmap.yaml** - Configuration settings
- **secret.yaml** - API keys and sensitive data (template)
- **serviceaccount.yaml** - Service account and RBAC
- **pvc.yaml** - Persistent volume claim for data storage

### Application

- **deployment.yaml** - Main application deployment
- **service.yaml** - ClusterIP and headless services
- **ingress.yaml** - Ingress for external access

### Scaling and Resilience

- **hpa.yaml** - Horizontal Pod Autoscaler
- **pdb.yaml** - Pod Disruption Budget

## Configuration

### Secrets

Create secrets before applying manifests:

```bash
kubectl create secret generic victor-secrets \
  --from-literal=anthropic-api-key=your-key \
  --from-literal=openai-api-key=your-key \
  --namespace=victor
```

### ConfigMap

Edit `configmap.yaml` to customize:

```yaml
VICTOR_ENV: "production"
VICTOR_LOG_LEVEL: "INFO"
VICTOR_MAX_CONCURRENT_TOOLS: "5"
```

### Resources

Edit `deployment.yaml` to adjust resources:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

## Deployment Options

### Option 1: All at Once

```bash
kubectl apply -f config/k8s/
```

### Option 2: Step by Step

```bash
# 1. Namespace
kubectl apply -f config/k8s/namespace.yaml

# 2. Configuration
kubectl apply -f config/k8s/configmap.yaml
kubectl apply -f config/k8s/secret.yaml

# 3. Storage
kubectl apply -f config/k8s/pvc.yaml

# 4. Application
kubectl apply -f config/k8s/deployment.yaml
kubectl apply -f config/k8s/service.yaml

# 5. Networking
kubectl apply -f config/k8s/ingress.yaml

# 6. Scaling
kubectl apply -f config/k8s/hpa.yaml
kubectl apply -f config/k8s/pdb.yaml
```

### Option 3: Using Helm

```bash
helm install victor ./config/helm/victor \
  --namespace victor \
  --create-namespace
```

## Verification

```bash
# Check pods
kubectl get pods -n victor

# Check services
kubectl get svc -n victor

# Check ingress
kubectl get ingress -n victor

# Check HPA
kubectl get hpa -n victor

# Describe deployment
kubectl describe deployment victor-api -n victor

# View logs
kubectl logs -f deployment/victor-api -n victor

# Port forward for testing
kubectl port-forward svc/victor-api 8000:80 -n victor
```

## Scaling

### Manual Scaling

```bash
kubectl scale deployment victor-api --replicas=5 -n victor
```

### Auto-scaling

HPA is configured by default:

```yaml
minReplicas: 3
maxReplicas: 10
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

Check HPA status:

```bash
kubectl get hpa -n victor
kubectl describe hpa victor-api-hpa -n victor
```

## Updates

### Update Image

```bash
kubectl set image deployment/victor-api \
  victor-api=vijayksingh/victor:v0.5.1 \
  -n victor

# Watch rollout
kubectl rollout status deployment/victor-api -n victor
```

### Update Configuration

```bash
kubectl apply -f config/k8s/configmap.yaml
kubectl rollout restart deployment/victor-api -n victor
```

### Rollback

```bash
# View history
kubectl rollout history deployment/victor-api -n victor

# Rollback
kubectl rollout undo deployment/victor-api -n victor

# Rollback to specific revision
kubectl rollout undo deployment/victor-api --to-revision=2 -n victor
```

## Monitoring

### Metrics

Victor exposes Prometheus metrics on port 9090:

```bash
# Port forward
kubectl port-forward svc/victor-api 9090:9090 -n victor

# Access metrics
curl http://localhost:9090/metrics
```

### Grafana Dashboards

Import dashboards from `config/dashboards/`:

```bash
kubectl apply -f config/dashboards/
```

### Logs

```bash
# Stream logs
kubectl logs -f deployment/victor-api -n victor

# All pods
kubectl logs -f -l app=victor-api -n victor

# Previous container
kubectl logs -p deployment/victor-api -n victor
```

## Troubleshooting

### Pod Not Ready

```bash
kubectl describe pod <pod-name> -n victor
kubectl logs <pod-name> -n victor
```

### Service Issues

```bash
kubectl get endpoints victor-api -n victor
kubectl describe svc victor-api -n victor
```

### High Resource Usage

```bash
kubectl top pods -n victor
kubectl top nodes
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f config/k8s/

# Or using Helm
helm uninstall victor -n victor

# Delete namespace
kubectl delete namespace victor
```

## Support

- Documentation: https://github.com/vijayksingh/victor#readme
- Issues: https://github.com/vijayksingh/victor/issues
- Deployment Guide: [DEPLOYMENT_GUIDE.md](../../docs/DEPLOYMENT_GUIDE.md)
- Kubernetes Setup: [KUBERNETES_SETUP.md](../../docs/KUBERNETES_SETUP.md)
