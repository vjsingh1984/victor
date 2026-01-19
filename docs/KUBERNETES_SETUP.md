# Kubernetes Setup Guide for Victor AI

Complete guide for deploying Victor AI on Kubernetes clusters.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Cluster Setup](#cluster-setup)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Scaling and Performance](#scaling-and-performance)
- [Monitoring](#monitoring)
- [Maintenance](#maintenance)

## Prerequisites

### Required Tools

- Kubernetes cluster (1.25+)
- kubectl (1.25+)
- helm (3.13+)
- Access to container registry (Docker Hub, ECR, GCR, etc.)

### Cluster Requirements

- Minimum 3 nodes
- 4 cores per node
- 16GB RAM per node
- 50GB disk per node

## Cluster Setup

### Create Cluster

#### Using kubectl

```bash
# Create namespace
kubectl create namespace victor

# Verify
kubectl get namespace victor
```

#### Using Helm

```bash
# Namespace is created automatically
helm install victor ./config/helm/victor \
  --namespace victor \
  --create-namespace
```

### Configure kubectl

```bash
# Set default namespace
kubectl config set-context --current --namespace=victor

# Verify
kubectl config view --minify
```

## Deployment Options

### Option 1: Raw Manifests

#### Quick Deploy

```bash
# Apply all manifests
kubectl apply -f config/k8s/ -n victor

# Verify
kubectl get all -n victor
```

#### Step-by-Step Deploy

```bash
# 1. Create namespace
kubectl apply -f config/k8s/namespace.yaml

# 2. Create configmap
kubectl apply -f config/k8s/configmap.yaml

# 3. Create secrets
kubectl apply -f config/k8s/secret.yaml

# 4. Create service account
kubectl apply -f config/k8s/serviceaccount.yaml

# 5. Create PVC
kubectl apply -f config/k8s/pvc.yaml

# 6. Create deployment
kubectl apply -f config/k8s/deployment.yaml

# 7. Create service
kubectl apply -f config/k8s/service.yaml

# 8. Create ingress
kubectl apply -f config/k8s/ingress.yaml

# 9. Create HPA
kubectl apply -f config/k8s/hpa.yaml

# 10. Create PDB
kubectl apply -f config/k8s/pdb.yaml
```

### Option 2: Helm Chart

#### Install from Local

```bash
# Install
helm install victor ./config/helm/victor \
  --namespace victor \
  --create-namespace

# Upgrade
helm upgrade victor ./config/helm/victor \
  --namespace victor

# Uninstall
helm uninstall victor --namespace victor
```

#### Install with Custom Values

```bash
# Create values file
cat > victor-values.yaml << EOF
replicaCount: 3

image:
  repository: vijayksingh/victor
  tag: "v0.5.0"

secrets:
  anthropicApiKey: "your-key"
  openaiApiKey: "your-key"

resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

ingress:
  enabled: true
  hosts:
  - host: victor.example.com
    paths:
    - path: /
      pathType: Prefix
EOF

# Install with values
helm install victor ./config/helm/victor \
  -f victor-values.yaml \
  --namespace victor
```

## Configuration

### Environment Variables

Edit `config/k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: victor-config
  namespace: victor
data:
  VICTOR_ENV: "production"
  VICTOR_LOG_LEVEL: "INFO"
  VICTOR_GRAPH_STORE: "sqlite"
  VICTOR_CACHE_DIR: "/home/victor/.victor/cache"
  VICTOR_MAX_CONCURRENT_TOOLS: "5"
  VICTOR_ENABLE_PARALLEL_EXECUTION: "true"
  VICTOR_ENABLE_OBSERVABILITY: "true"
  VICTOR_ENABLE_PROMETHEUS_EXPORT: "true"
  VICTOR_PROMETHEUS_PORT: "9090"
```

Apply changes:

```bash
kubectl apply -f config/k8s/configmap.yaml
kubectl rollout restart deployment/victor-api -n victor
```

### Secrets

#### Create from Command Line

```bash
kubectl create secret generic victor-secrets \
  --from-literal=anthropic-api-key='your-key' \
  --from-literal=openai-api-key='your-key' \
  --namespace=victor
```

#### Create from File

```bash
# Create secrets file
cat > secrets.txt << EOF
anthropic-api-key=your-anthropic-key
openai-api-key=your-openai-key
google-api-key=your-google-key
EOF

# Create secret
kubectl create secret generic victor-secrets \
  --from-env-file=secrets.txt \
  --namespace=victor

# Cleanup
rm secrets.txt
```

#### Update Existing Secret

```bash
kubectl patch secret victor-secrets \
  -n victor \
  --type=json \
  -p='[{"op": "replace", "path": "/data/anthropic-api-key", "value":"'$(echo -n 'new-key' | base64)'"}]'

# Restart pods to pick up changes
kubectl rollout restart deployment/victor-api -n victor
```

### Resource Limits

Edit `config/k8s/deployment.yaml`:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "4000m"
```

Apply:

```bash
kubectl apply -f config/k8s/deployment.yaml
```

## Scaling and Performance

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment victor-api --replicas=5 -n victor

# Verify
kubectl get deployment victor-api -n victor
```

### Horizontal Pod Autoscaler

The HPA is configured in `config/k8s/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: victor-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: victor-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

Check HPA status:

```bash
kubectl get hpa -n victor
kubectl describe hpa victor-api-hpa -n victor
```

### Vertical Pod Autoscaler

Enable VPA for automatic resource tuning:

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: victor-api-vpa
  namespace: victor
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: victor-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: victor-api
      minAllowed:
        cpu: "500m"
        memory: "512Mi"
      maxAllowed:
        cpu: "4000m"
        memory: "4Gi"
```

### Performance Tuning

#### Increase Replicas

```bash
kubectl scale deployment victor-api --replicas=10 -n victor
```

#### Adjust Resource Limits

```bash
kubectl set resources deployment victor-api \
  --limits=memory=4Gi,cpu=4000m \
  --requests=memory=1Gi,cpu=1000m \
  -n victor
```

#### Enable Caching

```yaml
# ConfigMap
VICTOR_ENABLE_CACHE: "true"
VICTOR_CACHE_TTL: "3600"
VICTOR_EMBEDDING_CACHE_ENABLED: "true"
```

## Monitoring

### Prometheus Integration

The ServiceMonitor is configured in the Helm chart:

```bash
# Verify ServiceMonitor
kubectl get servicemonitor -n victor

# Check targets in Prometheus
# Access Prometheus UI and check victor-api targets
```

### Grafana Dashboards

Import dashboards from `config/dashboards/`:

```bash
# Apply dashboard ConfigMaps
kubectl apply -f config/dashboards/ -n victor

# Access Grafana
kubectl port-forward svc/grafana 3000:80 -n victor
```

### Logs

```bash
# Stream logs
kubectl logs -f deployment/victor-api -n victor

# Logs from all pods
kubectl logs -f -l app=victor-api -n victor --max-log-requests=10

# Previous container logs
kubectl logs -p deployment/victor-api -n victor

# Logs with stern (if installed)
stern victor-api -n victor
```

### Metrics

Access Prometheus metrics:

```bash
# Port forward
kubectl port-forward svc/victor-api 9090:9090 -n victor

# Access metrics
curl http://localhost:9090/metrics
```

Key metrics to monitor:
- `http_requests_total`: Request count
- `http_request_duration_seconds`: Latency
- `tool_execution_duration_seconds`: Tool execution time
- `llm_api_duration_seconds`: LLM API time
- `cache_hits_total`: Cache effectiveness

## Maintenance

### Updates

#### Rolling Update

```bash
# Update image
kubectl set image deployment/victor-api \
  victor-api=vijayksingh/victor:v0.5.1 \
  -n victor

# Watch rollout
kubectl rollout status deployment/victor-api -n victor

# Check history
kubectl rollout history deployment/victor-api -n victor
```

#### Canary Deployment

```bash
# Create canary deployment
kubectl apply -f config/k8s/canary-deployment.yaml

# Split traffic (requires service mesh like Istio)
# Update VirtualService to route 10% to canary
```

### Rollback

```bash
# Rollback to previous
kubectl rollout undo deployment/victor-api -n victor

# Rollback to specific revision
kubectl rollout history deployment/victor-api -n victor
kubectl rollout undo deployment/victor-api --to-revision=2 -n victor
```

### Backup

```bash
# Backup resources
kubectl get all -n victor -o yaml > victor-backup.yaml

# Backup secrets
kubectl get secrets -n victor -o yaml > victor-secrets-backup.yaml

# Backup PVC data
kubectl cp victor/victor-api-pod:/home/victor/.victor ./victor-data-backup
```

### Disaster Recovery

```bash
# Restore from backup
kubectl apply -f victor-backup.yaml
kubectl apply -f victor-secrets-backup.yaml

# Scale to zero
kubectl scale deployment victor-api --replicas=0 -n victor

# Scale back up
kubectl scale deployment victor-api --replicas=3 -n victor
```

## Troubleshooting

### Pod Not Ready

```bash
# Describe pod
kubectl describe pod <pod-name> -n victor

# Common issues:
# - Image pull errors: Check image name and registry access
# - Resource limits: Check node capacity
# - Missing secrets: Verify secrets exist
# - ConfigMap errors: Check config syntax
```

### Service Not Working

```bash
# Check service
kubectl get svc victor-api -n victor

# Check endpoints
kubectl get endpoints victor-api -n victor

# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -n victor -- wget -O- http://victor-api:80
```

### High Memory/CPU

```bash
# Check resource usage
kubectl top pods -n victor
kubectl top nodes

# Check limits
kubectl get deployment victor-api -n victor -o yaml | grep -A 5 resources

# Adjust resources
kubectl set resources deployment victor-api \
  --limits=memory=4Gi,cpu=4000m \
  -n victor
```

### CrashLoopBackOff

```bash
# Check logs
kubectl logs <pod-name> -n victor
kubectl logs <pod-name> -n victor --previous

# Common causes:
# - Application errors: Check logs for stack traces
# - Missing dependencies: Check if all services are available
# - Configuration errors: Check ConfigMaps and secrets
```

## Advanced Topics

### Multi-AZ Deployment

```yaml
# Pod anti-affinity for multi-AZ
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - victor-api
        topologyKey: topology.kubernetes.io/zone
```

### Node Selectors

```yaml
# Deploy to specific nodes
nodeSelector:
  nodepool: applications

# Or taints and tolerations
tolerations:
- key: "dedicated"
  operator: "Equal"
  value: "victor"
  effect: "NoSchedule"
```

### Custom Certificate

```yaml
# Use existing TLS certificate
ingress:
  tls:
  - hosts:
    - victor.example.com
    secretName: victor-custom-tls
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/vijayksingh/victor/issues
- Documentation: https://github.com/vijayksingh/victor#readme
