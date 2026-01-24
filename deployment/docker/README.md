# Docker Deployment Guide for Victor AI

This directory contains Docker configurations for deploying Victor AI in containerized environments.

## Quick Start

### Using Docker Compose (Recommended for testing)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
vim .env

# Start all services
docker-compose -f deployment/docker/docker-compose.prod.yml up -d

# View logs
docker-compose -f deployment/docker/docker-compose.prod.yml logs -f

# Stop services
docker-compose -f deployment/docker/docker-compose.prod.yml down
```

### Using Docker directly

```bash
# Pull the image
docker pull victorai/victor:0.5.0

# Run Victor AI
docker run -d \
  --name victor-ai \
  -p 8000:8000 \
  -e VICTOR_PROFILE=production \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  -e ANTHROPIC_API_KEY="your-key" \
  victorai/victor:0.5.0

# View logs
docker logs -f victor-ai

# Stop container
docker stop victor-ai
```

## Multi-Stage Build

The Dockerfile uses a multi-stage build for optimization:

### Stage 1: Builder
- Installs build dependencies
- Creates virtual environment
- Installs Python packages
- Copies application code

### Stage 2: Runtime
- Installs runtime dependencies only
- Copies virtual environment from builder
- Creates non-root user
- Sets up health checks

### Stage 3: Slim (Production)
- Removes unnecessary files
- Reduces image size by ~40%
- Optimized for production

### Stage 4: Development
- Includes development tools
- Installs pytest, black, mypy
- Configured for debugging

## Building Images

### Build for production

```bash
# Build slim image (recommended)
docker build -f deployment/docker/Dockerfile --target slim -t victorai/victor:0.5.0 .

# Build with build args
docker build \
  -f deployment/docker/Dockerfile \
  --target slim \
  --build-arg VERSION=0.5.0 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t victorai/victor:0.5.0 .
```

### Build for development

```bash
# Build development image
docker build -f deployment/docker/Dockerfile --target development -t victorai/victor:dev .

# Run tests in container
docker run --rm victorai/victor:dev pytest tests/unit -v
```

### Build for multiple platforms

```bash
# Build for AMD64 and ARM64
docker buildx create --use
docker buildx build \
  -f deployment/docker/Dockerfile \
  --platform linux/amd64,linux/arm64 \
  --target slim \
  -t victorai/victor:0.5.0 \
  --push .
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VICTOR_PROFILE` | Profile (production, development) | production | No |
| `VICTOR_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING) | INFO | No |
| `VICTOR_MAX_WORKERS` | Number of workers | 4 | No |
| `DATABASE_URL` | PostgreSQL connection string | - | Yes* |
| `REDIS_URL` | Redis connection string | - | No |
| `ANTHROPIC_API_KEY` | Anthropic API key | - | Yes** |
| `OPENAI_API_KEY` | OpenAI API key | - | No |
| `GOOGLE_API_KEY` | Google API key | - | No |

*Required if using PostgreSQL checkpoint backend
**Required if using Anthropic as default provider

### Volume Mounts

```bash
# Cache directory
-v victor-cache:/app/.cache

# Data directory
-v victor-data:/app/data

# Logs directory
-v victor-logs:/app/logs

# Configuration (read-only)
-v ./config:/app/config:ro
```

## Production Deployment

### Using Docker Compose

The production compose file includes:

- **Victor AI**: Main application (3 replicas)
- **PostgreSQL**: Database for checkpoints
- **Redis**: Cache and message broker
- **Kafka**: Event bus (optional)
- **Nginx**: Reverse proxy and load balancer
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboard

```bash
# Start production stack
docker-compose -f deployment/docker/docker-compose.prod.yml up -d

# Scale Victor AI
docker-compose -f deployment/docker/docker-compose.prod.yml up -d --scale victor-ai=5

# View status
docker-compose -f deployment/docker/docker-compose.prod.yml ps

# Restart service
docker-compose -f deployment/docker/docker-compose.prod.yml restart victor-ai
```

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c deployment/docker/docker-compose.prod.yml victor-ai

# Scale services
docker service scale victor-ai_victor-ai=5

# View services
docker stack services victor-ai

# Remove stack
docker stack rm victor-ai
```

## Health Checks

The container includes health checks:

```bash
# Docker health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Manual health check
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

View health status:

```bash
docker inspect --format='{{json .State.Health}}' victor-ai | jq
```

## Logging

### View logs

```bash
# View all logs
docker logs victor-ai

# Follow logs
docker logs -f victor-ai

# View last 100 lines
docker logs --tail 100 victor-ai

# View logs with timestamps
docker logs -t victor-ai

# Compose logs
docker-compose -f deployment/docker/docker-compose.prod.yml logs -f victor-ai
```

### Log configuration

Logs are output to stdout in JSON format:

```json
{
  "timestamp": "2024-01-20T10:30:45Z",
  "level": "INFO",
  "message": "Starting Victor AI",
  "context": {
    "version": "0.5.0",
    "profile": "production"
  }
}
```

## Monitoring

### Metrics

Victor AI exposes Prometheus metrics on port 9090:

```bash
# Access metrics
curl http://localhost:9090/metrics
```

Example metrics:

```
# HELP victor_requests_total Total number of requests
# TYPE victor_requests_total counter
victor_requests_total{method="GET",endpoint="/health"} 1234

# HELP victor_request_duration_seconds Request duration
# TYPE victor_request_duration_seconds histogram
victor_request_duration_seconds_bucket{le="0.1"} 500
victor_request_duration_seconds_bucket{le="0.5"} 950
```

### Resource Usage

```bash
# View container stats
docker stats victor-ai

# View stats for all containers
docker stats $(docker ps -q)
```

## Security

### Running as Non-Root User

The container runs as a non-root user (UID 1000) for security:

```dockerfile
USER victor
```

### Read-Only Root Filesystem

For enhanced security, run with read-only root:

```bash
docker run -d \
  --name victor-ai \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /app/.cache \
  -p 8000:8000 \
  victorai/victor:0.5.0
```

### Secrets Management

**Option 1: Environment variables**

```bash
docker run -d \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e DATABASE_URL="$DATABASE_URL" \
  victorai/victor:0.5.0
```

**Option 2: Docker secrets**

```bash
# Create secret
echo "your-api-key" | docker secret create anthropic_api_key -

# Use secret in compose
docker-compose -f deployment/docker/docker-compose.prod.yml up -d
```

**Option 3: External secrets (HashiCorp Vault)**

```bash
docker run -d \
  --cap-add=IPC_LOCK \
  -e 'VAULT_ADDR=http://vault:8200' \
  -e 'VAULT_TOKEN=your-token' \
  victorai/victor:0.5.0
```

## Performance Tuning

### Resource Limits

```bash
docker run -d \
  --name victor-ai \
  --memory="2g" \
  --memory-swap="2g" \
  --cpus="2.0" \
  -p 8000:8000 \
  victorai/victor:0.5.0
```

### Docker Daemon Optimization

Edit `/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 10
}
```

Restart daemon:

```bash
sudo systemctl restart docker
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs victor-ai

# Check container status
docker inspect victor-ai

# Check if port is already in use
netstat -tuln | grep 8000
```

### Out of memory errors

```bash
# Check container memory usage
docker stats victor-ai --no-stream

# Increase memory limit
docker update victor-ai --memory="4g"

# Check Docker daemon memory
docker system df
```

### Slow performance

```bash
# Check resource usage
docker stats victor-ai

# Increase CPU limit
docker update victor-ai --cpus="4.0"

# Enable multi-threading
docker run -e VICTOR_MAX_WORKERS=8 victorai/victor:0.5.0
```

## Best Practices

1. **Use specific version tags** instead of `latest`
2. **Run as non-root user** for security
3. **Set resource limits** to prevent resource exhaustion
4. **Use health checks** for automatic recovery
5. **Mount volumes** for persistent data
6. **Use secrets** for sensitive data
7. **Scan images** for vulnerabilities
8. **Use multi-stage builds** to reduce image size
9. **Enable logging** for debugging and monitoring
10. **Test in development** before deploying to production

## Image Variants

| Variant | Description | Size | Use Case |
|---------|-------------|------|----------|
| `slim` | Production-optimized | ~500MB | Production deployments |
| `runtime` | Full runtime | ~800MB | Standard deployments |
| `development` | Dev tools included | ~1.2GB | Development and testing |

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Swarm Documentation](https://docs.docker.com/engine/swarm/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
