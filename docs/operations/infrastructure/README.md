# Victor AI - Production Infrastructure

Complete production infrastructure for automated deployment, monitoring, and scaling.

## 📁 Infrastructure Files

### Docker
- **Dockerfile.production** - Multi-stage production Dockerfile (< 1GB)
- **docker-compose.production.yml** - Full stack with monitoring
- **.dockerignore** - Optimized build context

### Kubernetes
- **config/k8s/namespace.yaml** - Namespace definition
- **config/k8s/configmap.yaml** - Configuration management
- **config/k8s/secret.yaml** - Secret management template
- **config/k8s/serviceaccount.yaml** - Service account with RBAC
- **config/k8s/pvc.yaml** - Persistent volume claims
- **config/k8s/deployment.yaml** - Main deployment with HPA
- **config/k8s/service.yaml** - Service exposure
- **config/k8s/ingress.yaml** - Ingress configuration
- **config/k8s/hpa.yaml** - Horizontal Pod Autoscaler
- **config/k8s/pdb.yaml** - Pod Disruption Budget

### Helm
- **config/helm/victor/** - Complete Helm chart
  - Chart.yaml
  - values.yaml
  - templates/*.yaml

### CI/CD Workflows
- **.github/workflows/infrastructure-ci.yml** - Infrastructure validation
- **.github/workflows/cd-production.yml** - Production deployment
- **.github/workflows/test-performance.yml** - Performance regression tests
- **.github/workflows/deploy-k8s.yml** - Kubernetes deployment
- **.github/workflows/deploy.yml** - General deployment

### Scripts
- **scripts/ci/deploy_production.sh** - Production deployment script
- **scripts/ci/smoke_test.sh** - Smoke test automation
- **scripts/ci/build_docker.sh** - Docker build automation
- **scripts/ci/run_tests.sh** - Test execution
- **scripts/ci/validate_workflows.sh** - Workflow validation

### Monitoring Config
- **config/prometheus/prometheus.yml** - Prometheus configuration
- **config/prometheus/alerts.yml** - Alerting rules
- **config/grafana/dashboards/** - Grafana dashboards
- **config/grafana/provisioning/** - Grafana provisioning
- **config/nginx/nginx.conf** - Reverse proxy configuration

### Documentation
- **docs/INFRASTRUCTURE.md** - Complete infrastructure guide
- **docs/infrastructure/quickstart.md** - Quick start guide
- **config/k8s/README.md** - Kubernetes documentation

## 🚀 Quick Start

### Docker Deployment

```bash
# Build
docker build -f Dockerfile.production -t victor:latest .

# Run
docker run -d --name victor-api \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=xxx \
  victor:latest victor serve --port 8000

# Test
curl http://localhost:8000/health/live
```text

### Docker Compose Deployment

```bash
# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=xxx
OPENAI_API_KEY=xxx
EOF

# Deploy
docker-compose -f docker-compose.production.yml up -d

# Access
# - API: http://localhost:8000
# - Prometheus: http://localhost:9091
# - Grafana: http://localhost:3000
```

### Kubernetes Deployment

```bash
# Deploy with kubectl
kubectl apply -f config/k8s/

# Deploy with Helm
helm upgrade --install victor ./config/helm/victor \
  --namespace victor \
  --create-namespace

# Verify
kubectl get pods -n victor
```text

## 🔧 CI/CD Pipeline

### Automatic Deployments

1. **Push to main** → Deploy to staging
2. **Tests pass** → Build Docker image
3. **Security scan** → Push to registry
4. **Staging tests** → Deploy to production (manual approval)

### Manual Deploy

```bash
# Using script
./scripts/ci/deploy_production.sh production --version v0.5.0

# Using kubectl
kubectl set image deployment/victor-api \
  victor-api=victor:v0.5.0 -n victor

# Using Helm
helm upgrade victor ./config/helm/victor \
  --set image.tag=v0.5.0
```

## 📊 Monitoring

### Metrics

- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)
- **Metrics endpoint**: http://localhost:9090/metrics

### Key Metrics

- API request rate and latency
- Tool execution time
- LLM token usage and cost
- Cache hit ratios
- System resources (CPU, memory)

### Distributed Tracing

- **Jaeger UI**: http://localhost:16686
- Trace API requests, tool calls, LLM calls

## 🔒 Security

### Features

- Non-root user (UID 1000)
- Multi-stage builds
- Vulnerability scanning (Trivy)
- Network policies
- Secrets management
- RBAC

### Scanning

```bash
# Scan image
docker scan victor:latest

# Trivy scan
trivy image victor:latest
```text

## 📈 Scaling

### Horizontal Pod Autoscaler

```yaml
minReplicas: 3
maxReplicas: 10
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment victor-api --replicas=5 -n victor

# Check HPA status
kubectl get hpa -n victor
```text

## 🔄 Rollback

```bash
# Rollback deployment
kubectl rollout undo deployment/victor-api -n victor

# Rollback to specific revision
kubectl rollout undo deployment/victor-api --to-revision=2 -n victor

# Helm rollback
helm rollback victor -n victor
```

## 🧪 Testing

### Smoke Tests

```bash
./scripts/ci/smoke_test.sh staging
./scripts/ci/smoke_test.sh production
```text

### Performance Tests

```bash
# Run locally
python scripts/benchmark_tool_selection.py run --group all

# Run in CI (automatic on PR)
```

## 📝 Documentation

- **[Complete Guide](quickstart.md)** - Full documentation
- **[Quick Start](quickstart.md)** - Get started in 5 minutes
- **K8s Manifests**: `config/k8s/` - Kubernetes documentation
- **Helm Chart**: `config/helm/victor/` - Helm documentation

## 🎯 Checklist

### Pre-Deployment

- [ ] Configure API keys in secrets
- [ ] Set up monitoring and alerting
- [ ] Configure resource limits
- [ ] Set up backup procedures
- [ ] Test in staging environment

### Post-Deployment

- [ ] Verify pod status
- [ ] Check health endpoints
- [ ] Run smoke tests
- [ ] Monitor metrics
- [ ] Check error logs
- [ ] Verify autoscaling

### Monitoring

- [ ] Set up Prometheus alerts
- [ ] Configure Grafana dashboards
- [ ] Enable distributed tracing
- [ ] Set up log aggregation
- [ ] Configure error tracking

## 🔗 Links

- [Victor AI Repository](https://github.com/vjsingh1984/victor)
- [Documentation](../index.md)
- [Issues](https://github.com/vjsingh1984/victor/issues)
- [Discussions](https://github.com/vjsingh1984/victor/discussions)

## 📞 Support

For help:
1. Check the troubleshooting section in [quickstart.md](quickstart.md)
2. Search [existing issues](https://github.com/vjsingh1984/victor/issues)
3. Start a [discussion](https://github.com/vjsingh1984/victor/discussions)
4. Open a new [issue](https://github.com/vjsingh1984/victor/issues/new)

## 📜 License

Apache License 2.0 - see `LICENSE` in the repo root for details.

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
