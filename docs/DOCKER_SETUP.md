# Docker Setup Guide for Victor AI

Complete guide for deploying Victor AI using Docker and Docker Compose.

## Table of Contents

- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Configuration](#configuration)
- [Persistence](#persistence)
- [Networking](#networking)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Pull and Run

```bash
# Pull latest image
docker pull vijayksingh/victor:latest

# Run Victor
docker run -it --rm \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest

# Run in background
docker run -d \
  --name victor \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest
```

### With Volume Mount

```bash
# Create named volume
docker volume create victor-data

# Run with volume
docker run -it \
  --name victor \
  -e ANTHROPIC_API_KEY=your-key \
  -v victor-data:/home/victor/.victor \
  vijayksingh/victor:latest
```

## Docker Deployment

### Building the Image

```bash
# Clone repository
git clone https://github.com/vijayksingh/victor.git
cd victor

# Build image
docker build -t victor:latest .

# Build with no cache
docker build --no-cache -t victor:latest .

# Build with custom tag
docker build -t victor:v0.5.0 .

# Build for specific platform
docker buildx build --platform linux/amd64 -t victor:latest .
```

### Running the Container

#### Basic Usage

```bash
docker run -it --rm \
  -e ANTHROPIC_API_KEY=your-key \
  -e OPENAI_API_KEY=your-key \
  vijayksingh/victor:latest
```

#### With Custom Profile

```bash
docker run -it \
  -e ANTHROPIC_API_KEY=your-key \
  -e VICTOR_PROFILE=airgapped \
  vijayksingh/victor:latest
```

#### With Custom Configuration

```bash
# Mount custom config
docker run -it \
  -e ANTHROPIC_API_KEY=your-key \
  -v $(pwd)/custom-config.yaml:/home/victor/.victor/config.yaml \
  vijayksingh/victor:latest
```

#### Detached Mode

```bash
# Run in background
docker run -d \
  --name victor \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest

# View logs
docker logs -f victor

# Attach to container
docker attach victor

# Stop container
docker stop victor

# Remove container
docker rm victor
```

### Resource Limits

```bash
# With resource constraints
docker run -it \
  --name victor \
  --memory="2g" \
  --cpus="2.0" \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest
```

### Port Forwarding

```bash
# Forward port for API server
docker run -d \
  --name victor \
  -p 8000:8000 \
  -p 9090:9090 \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest \
  victor-api --host 0.0.0.0

# Access API
curl http://localhost:8000/health/live
```

## Docker Compose Deployment

### Basic Setup

```bash
# Clone repository
git clone https://github.com/vijayksingh/victor.git
cd victor

# Create environment file
cat > .env << EOF
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
EOF

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Development Mode

```bash
# Start with hot reload
docker-compose -f docker-compose.yml up

# Start specific services
docker-compose up victor ollama

# Start with profiles
docker-compose --profile demo up
docker-compose --profile notebook up
```

### Production Mode

```bash
# Create production env file
cat > .env.production << EOF
VICTOR_ENV=production
VICTOR_LOG_LEVEL=INFO
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
PROMETHEUS_PORT=9091
GRAFANA_PORT=3000
EOF

# Start production stack
docker-compose -f docker-compose.production.yml --env-file .env.production up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f victor
```

### Multiple Services

The production compose file includes:

```yaml
services:
  victor:          # Main application
  ollama:          # Local LLM server
  prometheus:      # Metrics collection
  grafana:         # Visualization
  alertmanager:    # Alert routing
  node-exporter:   # System metrics
```

### Scaling

```bash
# Scale victor service
docker-compose up -d --scale victor=3

# Check scaled instances
docker-compose ps
```

## Configuration

### Environment Variables

#### Required Variables

```bash
ANTHROPIC_API_KEY=your-key    # Anthropic Claude API key
OPENAI_API_KEY=your-key       # OpenAI GPT API key
GOOGLE_API_KEY=your-key       # Google Gemini API key
```

#### Optional Variables

```bash
# Environment
VICTOR_ENV=production
VICTOR_LOG_LEVEL=INFO
VICTOR_LOG_FORMAT=json

# API Configuration
VICTOR_API_HOST=0.0.0.0
VICTOR_API_PORT=8000

# Cache Configuration
VICTOR_ENABLE_CACHE=true
VICTOR_CACHE_TTL=3600
VICTOR_CACHE_DIR=/home/victor/.victor/cache

# Tool Execution
VICTOR_MAX_CONCURRENT_TOOLS=5
VICTOR_TOOL_TIMEOUT=300
VICTOR_ENABLE_PARALLEL_EXECUTION=true

# Observability
VICTOR_ENABLE_OBSERVABILITY=true
VICTOR_ENABLE_PROMETHEUS_EXPORT=true
VICTOR_PROMETHEUS_PORT=9090

# Semantic Search
VICTOR_ENABLE_SEMANTIC_SEARCH=true
VICTOR_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
VICTOR_EMBEDDING_CACHE_ENABLED=true

# Vector Store
VICTOR_VECTOR_STORE=lancedb
VICTOR_VECTOR_STORE_PATH=/home/victor/.victor/data/vectors

# Graph Store
VICTOR_GRAPH_STORE=sqlite
VICTOR_GRAPH_STORE_PATH=/home/victor/.victor/data/graph.db
```

### Custom Compose File

Create `docker-compose.custom.yml`:

```yaml
version: '3.8'

services:
  victor:
    image: vijayksingh/victor:latest
    container_name: victor-custom
    environment:
      - VICTOR_ENV=production
      - VICTOR_LOG_LEVEL=INFO
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - victor-data:/home/victor/.victor
      - ./custom-config.yaml:/home/victor/.victor/config.yaml:ro
    ports:
      - "8000:8000"
      - "9090:9090"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '4'
        reservations:
          memory: 1G
          cpus: '1'

volumes:
  victor-data:
    driver: local
```

Run with custom configuration:

```bash
docker-compose -f docker-compose.custom.yml up -d
```

## Persistence

### Named Volumes

```bash
# Create volume
docker volume create victor-data

# Use volume
docker run -it \
  -v victor-data:/home/victor/.victor \
  vijayksingh/victor:latest

# List volumes
docker volume ls

# Inspect volume
docker volume inspect victor-data

# Remove volume
docker volume rm victor-data
```

### Bind Mounts

```bash
# Mount local directory
docker run -it \
  -v $(pwd)/victor-data:/home/victor/.victor \
  vijayksingh/victor:latest

# Mount with read-only
docker run -it \
  -v $(pwd)/config:/home/victor/.victor/config:ro \
  vijayksingh/victor:latest
```

### Backup and Restore

```bash
# Backup volume
docker run --rm \
  -v victor-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/victor-backup.tar.gz -C /data .

# Restore volume
docker run --rm \
  -v victor-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/victor-backup.tar.gz -C /data
```

## Networking

### Container Networking

```bash
# Create network
docker network create victor-network

# Run with custom network
docker run -d \
  --network victor-network \
  --name victor \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest

# Connect existing container
docker network connect victor-network victor
```

### Port Mapping

```bash
# Map multiple ports
docker run -d \
  -p 8000:8000 \    # API
  -p 9090:9090 \    # Metrics
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest

# Map to specific interface
docker run -d \
  -p 127.0.0.1:8000:8000 \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest

# Map random port
docker run -d \
  -p 8000 \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest
```

### Container Communication

```bash
# Start database
docker run -d \
  --name postgres \
  -e POSTGRES_DB=victor \
  -e POSTGRES_PASSWORD=secret \
  postgres:16

# Start victor with link
docker run -d \
  --name victor \
  --link postgres \
  -e POSTGRES_URL=postgres://postgres:5432/victor \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest
```

## Monitoring

### Viewing Logs

```bash
# Follow logs
docker logs -f victor

# Last 100 lines
docker logs --tail 100 victor

# Logs with timestamps
docker logs -t victor

# Logs from last hour
docker logs --since 1h victor
```

### Health Checks

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' victor

# Health check details
docker inspect --format='{{json .State.Health}}' victor | jq

# Manual health check
docker exec victor curl http://localhost:8000/health/live
```

### Metrics

```bash
# Access Prometheus metrics
docker exec victor curl http://localhost:9090/metrics

# Port forward for external access
docker run -d \
  --name victor \
  -p 9090:9090 \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest

# Access metrics
curl http://localhost:9090/metrics
```

### Resource Monitoring

```bash
# Container stats
docker stats victor

# All containers
docker stats

# Detailed inspection
docker inspect victor

# Resource usage
docker exec victor top
docker exec victor htop
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs victor

# Check if container exists
docker ps -a | grep victor

# Inspect container
docker inspect victor

# Common issues:
# - Missing API keys: Check environment variables
# - Port conflicts: Change port mapping
# - Resource limits: Check available memory/CPU
```

### High Memory Usage

```bash
# Check resource usage
docker stats victor

# Limit memory
docker run -d \
  --memory="2g" \
  --memory-swap="2g" \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest

# Check memory usage in container
docker exec victor free -h
docker exec victor du -sh /home/victor/.victor/cache
```

### Network Issues

```bash
# Check container network
docker network inspect bridge

# Test DNS
docker exec victor nslookup google.com

# Test connectivity
docker exec victor ping -c 3 google.com

# Check port bindings
docker port victor
```

### Permission Issues

```bash
# Fix volume permissions
docker run --rm \
  -v victor-data:/data \
  alpine chown -R 1000:1000 /data

# Run with specific user
docker run -it \
  -u 1000:1000 \
  -v victor-data:/home/victor/.victor \
  vijayksingh/victor:latest
```

### Debug Mode

```bash
# Run with debug logging
docker run -it \
  -e VICTOR_LOG_LEVEL=DEBUG \
  -e ANTHROPIC_API_KEY=your-key \
  vijayksingh/victor:latest

# Run with shell access
docker run -it \
  --entrypoint /bin/bash \
  vijayksingh/victor:latest

# Execute commands in running container
docker exec -it victor /bin/bash
```

### Cleanup

```bash
# Stop all containers
docker stop $(docker ps -aq)

# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Complete cleanup
docker system prune -a --volumes
```

## Best Practices

### Security

```bash
# Run as non-root user (already configured in Dockerfile)
docker run -it \
  -u 1000:1000 \
  vijayksingh/victor:latest

# Use read-only root filesystem
docker run -it \
  --read-only \
  --tmpfs /tmp \
  vijayksingh/victor:latest

# Drop capabilities
docker run -it \
  --cap-drop=ALL \
  vijayksingh/victor:latest
```

### Performance

```bash
# Use multi-stage build (already in Dockerfile)
# Enable caching
docker run -it \
  -e VICTOR_ENABLE_CACHE=true \
  vijayksingh/victor:latest

# Limit resources
docker run -it \
  --memory="2g" \
  --cpus="2.0" \
  vijayksingh/victor:latest
```

### Maintainability

```bash
# Use specific version tags
docker pull vijayksingh/victor:v0.5.0

# Tag custom builds
docker build -t victor:custom .
docker tag victor:custom victor:v0.5.0-custom

# Document with labels
docker build \
  --label "com.example.version=0.5.0" \
  --label "com.example.maintainer=singhvjd@gmail.com" \
  -t victor:latest .
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/vijayksingh/victor/issues
- Documentation: https://github.com/vijayksingh/victor#readme
