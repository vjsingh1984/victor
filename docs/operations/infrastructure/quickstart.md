# Victor AI - Infrastructure Quick Start

Get Victor AI running in production in minutes.

## Prerequisites

- Docker installed (for Docker deployment)
- kubectl installed (for Kubernetes deployment)
- Helm installed (optional, for Kubernetes deployment)
- API keys for your LLM provider(s)

## Option 1: Docker (Quickest)

### 1. Build Image

```bash
git clone https://github.com/vjsingh1984/victor.git
cd victor

docker build -f Dockerfile.production -t victor:latest .
```text

### 2. Run Container

```bash
# Interactive mode
docker run -it --rm \
  -v $(pwd)/workspace:/workspace \
  -e ANTHROPIC_API_KEY=your_key_here \
  victor:latest

# API server mode
docker run -d --name victor-api \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key_here \
  victor:latest victor serve --port 8000
```

### 3. Test

```bash
curl http://localhost:8000/health/live
```text

## Option 2: Docker Compose (Full Stack)

### 1. Configure Environment

```bash
cat > .env << EOF
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
VICTOR_PROFILE=production
EOF
```

### 2. Deploy

```bash
docker-compose -f docker-compose.production.yml up -d
```text

### 3. Access Services

- Victor API: http://localhost:8000
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (admin/admin)

## Option 3: Kubernetes (Production)

### 1. Configure kubectl

```bash
export KUBECONFIG=~/.kube/config
kubectl cluster-info
```

### 2. Create Namespace

```bash
kubectl create namespace victor
```text

### 3. Create Secrets

```bash
kubectl create secret generic victor-secrets \
  --from-literal=anthropic-api-key=your_key_here \
  --from-literal=openai-api-key=your_key_here \
  -n victor
```

### 4. Deploy (Kubectl)

```bash
kubectl apply -f config/k8s/namespace.yaml
kubectl apply -f config/k8s/configmap.yaml
kubectl apply -f config/k8s/secret.yaml
kubectl apply -f config/k8s/serviceaccount.yaml
kubectl apply -f config/k8s/pvc.yaml
kubectl apply -f config/k8s/deployment.yaml
kubectl apply -f config/k8s/service.yaml
kubectl apply -f config/k8s/hpa.yaml
```text

### 5. Deploy (Helm)

```bash
helm upgrade --install victor ./config/helm/victor \
  --namespace victor \
  --create-namespace \
  --set image.tag=latest \
  --set replicaCount=3
```

### 6. Verify

```bash
kubectl get pods -n victor
kubectl logs -f deployment/victor-api -n victor
```text

## CI/CD Deployment

### Automatic Deployments

Push to `main` branch triggers:

1. ✅ Run tests
2. 🔨 Build Docker image
3. 🔍 Security scan
4. 🚀 Deploy to staging
5. ✅ Run smoke tests

### Manual Production Deploy

```bash
# Using script
./scripts/ci/deploy_production.sh production --version v0.5.0

# Using kubectl
kubectl set image deployment/victor-api \
  victor-api=victor:v0.5.0 \
  -n victor

# Using Helm
helm upgrade victor ./config/helm/victor \
  --namespace victor \
  --set image.tag=v0.5.0
```

## Next Steps

1. Read the [full infrastructure guide](README.md)
2. Configure monitoring and alerting
3. Set up backup procedures
4. Review security best practices
5. Configure autoscaling

## Troubleshooting

### Pod Not Starting

```bash
kubectl describe pod victor-api-xxx -n victor
kubectl logs victor-api-xxx -n victor
```text

### Health Checks Failing

```bash
kubectl get endpoints victor-api -n victor
kubectl port-forward svc/victor-api 8000:80 -n victor
curl http://localhost:8000/health/live
```

### Rollback

```bash
kubectl rollout undo deployment/victor-api -n victor
```text

## Support

- [Documentation](README.md)
- [GitHub Issues](https://github.com/vjsingh1984/victor/issues)
- [Discussions](https://github.com/vjsingh1984/victor/discussions)

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
