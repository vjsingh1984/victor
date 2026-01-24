# Kubernetes Deployment Guide for Victor AI

This directory contains Kubernetes manifests and configurations for deploying Victor AI in production environments.

## Overview

The Kubernetes deployment uses Kustomize for environment-specific configurations across three environments:
- **Development** - Single replica, minimal resources, debugging enabled
- **Staging** - 2 replicas, moderate resources, testing environment
- **Production** - 6+ replicas with HPA, high resources, full observability

## Directory Structure

```
deployment/kubernetes/
├── base/                    # Base manifests (common across all environments)
│   ├── deployment.yaml      # Main deployment specification
│   ├── service.yaml         # Service and headless service
│   ├── configmap.yaml       # Application configuration
│   ├── secret.yaml          # Secrets template (use SealedSecrets or External Secrets)
│   ├── hpa.yaml             # Horizontal Pod Autoscaler
│   ├── pdb.yaml             # Pod Disruption Budget
│   ├── networkpolicy.yaml   # Network policies
│   ├── serviceaccount.yaml  # Service account and RBAC
│   └── priorityclass.yaml   # Priority classes for scheduling
├── overlays/
│   ├── production/          # Production overrides
│   │   ├── kustomization.yaml
│   │   ├── deployment-patch.yaml
│   │   └── hpa-patch.yaml
│   ├── staging/             # Staging overrides
│   │   ├── kustomization.yaml
│   │   └── deployment-patch.yaml
│   └── development/         # Development overrides
│       ├── kustomization.yaml
│       └── deployment-patch.yaml
└── README.md
```

## Prerequisites

1. **Kubernetes cluster** (version 1.24+)
2. **kubectl** configured to talk to your cluster
3. **kustomize** (version 4.0+) or kubectl with kustomize support
4. **Ingress controller** (nginx, traefik, or AWS ALB)
5. **Certificate manager** (cert-manager) for TLS certificates
6. **Monitoring stack** (Prometheus, Grafana) - optional

## Quick Start

### 1. Create namespaces

```bash
kubectl create namespace victor-ai-prod
kubectl create namespace victor-ai-staging
kubectl create namespace victor-ai-dev
```

### 2. Configure secrets

**Option A: Using kubectl (not recommended for production)**

```bash
kubectl create secret generic victor-ai-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=redis-url="redis://host:6379/0" \
  --from-literal=anthropic-api-key="your-key" \
  -n victor-ai-prod
```

**Option B: Using SealedSecrets (recommended for GitOps)**

```bash
# Install SealedSecrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Create secret
kubectl create secret generic victor-ai-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=redis-url="redis://host:6379/0" \
  --dry-run=client -o yaml > secret.yaml

# Seal the secret
kubeseal -f secret.yaml -w sealed-secret.yaml

# Apply sealed secret (can be committed to Git)
kubectl apply -f sealed-secret.yaml -n victor-ai-prod
```

**Option C: Using External Secrets Operator**

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: victor-ai-prod
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
---
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
    name: victor-ai-secrets
    creationPolicy: Owner
  data:
    - secretKey: database-url
      remoteRef:
        key: victor-ai/prod/database-url
```

### 3. Deploy to environment

```bash
# Development
kubectl apply -k deployment/kubernetes/overlays/development

# Staging
kubectl apply -k deployment/kubernetes/overlays/staging

# Production
kubectl apply -k deployment/kubernetes/overlays/production
```

### 4. Verify deployment

```bash
# Check pods
kubectl get pods -n victor-ai-prod

# Check deployment status
kubectl rollout status deployment/victor-ai -n victor-ai-prod

# Check services
kubectl get svc -n victor-ai-prod

# Check logs
kubectl logs -l app=victor-ai -n victor-ai-prod --tail=100 -f
```

## Configuration

### ConfigMap Parameters

Edit `base/configmap.yaml` or use Kustomize overlays to customize:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `event-bus-backend` | Event bus backend (memory, kafka, sqs, rabbitmq, redis) | memory |
| `checkpoint-backend` | Checkpoint backend (memory, sqlite, postgres, redis) | memory |
| `cache-backend` | Cache backend (memory, redis) | memory |
| `tool-selection-strategy` | Tool selection strategy (keyword, semantic, hybrid) | hybrid |
| `default-provider` | Default LLM provider | anthropic |
| `default-model` | Default model | claude-sonnet-4-5 |
| `enable-metrics` | Enable Prometheus metrics | true |
| `enable-tracing` | Enable distributed tracing | false |
| `enable-rate-limiting` | Enable rate limiting | true |

### Resource Management

Production deployment uses the following resource limits:

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| Victor AI (Production) | 1000m | 4000m | 1Gi | 4Gi |
| Victor AI (Staging) | 500m | 2000m | 512Mi | 2Gi |
| Victor AI (Dev) | 250m | 1000m | 256Mi | 1Gi |

### Autoscaling

Production deployment includes HPA with the following configuration:

- **Min replicas**: 6
- **Max replicas**: 50
- **Target CPU utilization**: 70%
- **Target memory utilization**: 80%
- **Scale down stabilization**: 10 minutes
- **Scale up**: Immediate

## High Availability

The production deployment is configured for high availability:

1. **Pod Anti-Affinity**: Pods are distributed across nodes and availability zones
2. **Pod Disruption Budget**: Minimum 2 pods must be available during disruptions
3. **Horizontal Pod Autoscaler**: Automatically scales based on CPU/memory
4. **Priority Classes**: Critical pods get scheduling priority
5. **Resource Requests/Limits**: Prevent resource starvation
6. **Liveness/Readiness Probes**: Ensure pod health
7. **Rolling Updates**: Zero-downtime deployments

## Monitoring and Observability

### Metrics

Victor AI exposes Prometheus metrics on port 9090:

```
http://victor-ai-service:9090/metrics
```

Configure ServiceMonitor (if using Prometheus Operator):

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: victor-ai
  namespace: victor-ai-prod
spec:
  selector:
    matchLabels:
      app: victor-ai
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Logging

Logs are output to stdout in JSON format:

```bash
# View logs
kubectl logs -l app=victor-ai -n victor-ai-prod --tail=100 -f

# Stream all pods
kubectl logs -l app=victor-ai -n victor-ai-prod --all-containers=true -f
```

### Health Checks

- **Liveness probe**: `http://localhost:8000/health` every 10 seconds
- **Readiness probe**: `http://localhost:8000/ready` every 5 seconds
- **Startup probe**: `http://localhost:8000/health` for 30 seconds on startup

## Networking

### Service Types

- **ClusterIP** (default): Internal cluster access
- **Headless Service**: For direct pod-to-pod communication (statefulsets, mesh)

### Ingress

Configure ingress for external access:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: victor-ai-ingress
  namespace: victor-ai-prod
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - victor-ai.example.com
    secretName: victor-ai-tls
  rules:
  - host: victor-ai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: victor-ai
            port:
              number: 80
```

### Network Policies

Network policies restrict network access:

- Allow ingress from ingress controller
- Allow monitoring from Prometheus
- Allow egress to external APIs (LLM providers)
- Allow database/Redis access
- Allow DNS resolution

## Scaling

### Manual Scaling

```bash
# Scale to 10 replicas
kubectl scale deployment/victor-ai --replicas=10 -n victor-ai-prod

# Scale HPA min/max replicas
kubectl edit hpa victor-ai -n victor-ai-prod
```

### Autoscaling

HPA automatically scales based on metrics:

```bash
# View HPA status
kubectl get hpa -n victor-ai-prod

# View HPA details
kubectl describe hpa victor-ai -n victor-ai-prod
```

## Updating

### Rolling Update

```bash
# Update image tag
kubectl set image deployment/victor-ai \
  victor-ai=victorai/victor:0.5.2 \
  -n victor-ai-prod

# Watch rollout status
kubectl rollout status deployment/victor-ai -n victor-ai-prod
```

### Rollback

```bash
# View rollout history
kubectl rollout history deployment/victor-ai -n victor-ai-prod

# Rollback to previous version
kubectl rollout undo deployment/victor-ai -n victor-ai-prod

# Rollback to specific revision
kubectl rollout undo deployment/victor-ai --to-revision=3 -n victor-ai-prod
```

## Troubleshooting

### Pod not starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n victor-ai-prod

# Check pod logs
kubectl logs <pod-name> -n victor-ai-prod

# Check events
kubectl get events -n victor-ai-prod --sort-by='.lastTimestamp'
```

### CrashLoopBackOff

```bash
# Check logs for all restarts
kubectl logs <pod-name> -n victor-ai-prod --previous

# Check resource limits
kubectl describe pod <pod-name> -n victor-ai-prod | grep -A 5 Limits
```

### Image pull errors

```bash
# Check if image exists
docker pull victorai/victor:0.5.0

# Check image pull secrets
kubectl get secrets -n victor-ai-prod

# Create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=<username> \
  --docker-password=<password> \
  -n victor-ai-prod
```

### Performance issues

```bash
# Check resource usage
kubectl top pods -n victor-ai-prod
kubectl top nodes

# Check HPA metrics
kubectl describe hpa victor-ai -n victor-ai-prod

# Check pod resource requests/limits
kubectl get pod <pod-name> -n victor-ai-prod -o jsonpath='{.spec.containers[*].resources}'
```

## Security

### RBAC

The deployment includes a service account with minimal permissions:

```bash
# View service account
kubectl get serviceaccount victor-ai -n victor-ai-prod

# View role bindings
kubectl get rolebinding -n victor-ai-prod
```

### Security Context

Pods run with non-root user and hardened security:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault
```

### Pod Security Standards

Configure namespace for PSA:

```bash
kubectl label namespace victor-ai-prod \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

## Backup and Recovery

### Etcd Backup

```bash
# Backup etcd
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  snapshot save /backup/etcd-snapshot.db
```

### Resource Backup

```bash
# Backup all resources
kubectl get all -n victor-ai-prod -o yaml > victor-ai-backup.yaml

# Backup specific resources
kubectl get deployment,service,configmap,secret -n victor-ai-prod -o yaml > backup.yaml
```

## Best Practices

1. **Use separate namespaces** for each environment
2. **Use SealedSecrets or External Secrets** for secrets management
3. **Enable resource limits** on all pods
4. **Configure HPA** for automatic scaling
5. **Use PDB** to maintain availability during updates
6. **Enable network policies** for security
7. **Monitor metrics and logs** continuously
8. **Test in staging** before deploying to production
9. **Keep manifests in Git** for version control
10. **Use GitOps** (ArgoCD, FluxCD) for deployment automation

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kustomize Documentation](https://kustomize.io/)
- [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets)
- [External Secrets Operator](https://external-secrets.io/)
- [Prometheus Operator](https://prometheus-operator.dev/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
