# Victor AI Deployment Guide

**Version:** 0.5.0
**Last Updated:** 2025-01-20
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Methods](#installation-methods)
4. [Configuration Guide](#configuration-guide)
5. [Database Setup](#database-setup)
6. [Deployment Strategies](#deployment-strategies)
7. [Verification Steps](#verification-steps)
8. [Troubleshooting](#troubleshooting)
9. [Upgrade Procedures](#upgrade-procedures)

---

## Overview

Victor AI is an open-source AI coding assistant supporting 21 LLM providers with 55 specialized tools across 5 domain verticals (Coding, DevOps, RAG, Data Analysis, Research). This guide provides comprehensive deployment instructions for production environments.

### Key Components

- **Agent Orchestrator**: Facade pattern coordinating all system components
- **Provider System**: 21 LLM providers (Anthropic, OpenAI, Ollama, etc.)
- **Tool Pipeline**: 55 tools across 5 verticals with cost tier optimization
- **Workflow Engine**: StateGraph DSL with YAML-first workflow definitions
- **Event Bus**: Pluggable backends for pub/sub messaging
- **Security**: RBAC/ABAC authorization with comprehensive audit logging

---

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **Memory** | 8 GB RAM | 16+ GB RAM |
| **Storage** | 10 GB SSD | 50+ GB SSD |
| **Python** | 3.10 | 3.11 or 3.12 |
| **Network** | 1 Mbps | 10+ Mbps |

### Operating Systems

- **Linux**: Ubuntu 20.04+, Debian 11+, RHEL 8+
- **macOS**: 11+ (Big Sur or later)
- **Windows**: WSL2 only (native Windows not supported)

### External Dependencies

| Dependency | Purpose | Required |
|------------|---------|----------|
| **LLM Provider API** | AI model access | Yes (one of: Anthropic, OpenAI, etc.) |
| **Git** | Code analysis | Yes |
| **Docker** (optional) | Container execution | For DevOps vertical |
| **Vector DB** (optional) | RAG operations | For RAG vertical |

---

## Installation Methods

### Method 1: pip install (Recommended)

```bash
# Basic installation
pip install victor-ai

# With all optional dependencies
pip install "victor-ai[all]"

# With specific verticals
pip install "victor-ai[coding,devops,rag,research,dataanalysis]"

# With API server support
pip install "victor-ai[api]"

# Development installation
pip install -e ".[dev]"
```

### Method 2: Docker Deployment

```bash
# Build image
docker build -f docker/Dockerfile -t victor-ai:0.5.0 .

# Run container
docker run -d \
  --name victor-ai \
  -p 8000:8000 \
  -v ~/.victor:/app/.victor \
  -e ANTHROPIC_API_KEY=your_key_here \
  victor-ai:0.5.0
```

### Method 3: From Source

```bash
# Clone repository
git clone https://github.com/your-org/victor-ai.git
cd victor-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run initialization
victor init
```

### Method 4: Air-Gapped Installation

For air-gapped environments without internet access:

```bash
# 1. Download wheel files and dependencies on internet-connected machine
pip download -d packages "victor-ai[all]"

# 2. Transfer packages directory to air-gapped system

# 3. Install from local packages
pip install --no-index --find-links packages victor-ai

# 4. Configure for local-only providers
export VICTOR_AIRGAPPED_MODE=true
```

---

## Configuration Guide

### Environment Variables

Create a `.env` file in your home directory or project root:

```bash
# Provider Configuration
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
GOOGLE_API_KEY=xxxxx

# Provider Selection
VICTOR_DEFAULT_PROVIDER=anthropic
VICTOR_DEFAULT_MODEL=claude-sonnet-4-5

# Operation Modes
VICTOR_PROFILE=production  # development, production, airgapped
VICTOR_AIRGAPPED_MODE=false

# Vertical Configuration
VICTOR_ENABLE_CODING=true
VICTOR_ENABLE_DEVOPS=true
VICTOR_ENABLE_RAG=true
VICTOR_ENABLE_RESEARCH=true
VICTOR_ENABLE_DATAANALYSIS=true

# Security
VICTOR_AUTH_ENABLED=true
VICTOR_AUTH_MODE=rbac  # rbac, abac, policy
VICTOR_AUDIT_LOG_ENABLED=true

# Performance
VICTOR_MAX_WORKERS=4
VICTOR_TOOL_BUDGET=100
VICTOR_CACHE_ENABLED=true
VICTOR_CACHE_TTL=3600

# Workflow System
VICTOR_WORKFLOW_CACHE_ENABLED=true
VICTOR_WORKFLOW_SCHEDULER_ENABLED=false

# Event Bus
VICTOR_EVENT_BACKEND=memory  # memory, kafka, sqs, rabbitmq, redis
VICTOR_EVENT_KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### Configuration File (YAML)

Create `~/.victor/config.yaml`:

```yaml
# Provider Settings
providers:
  default: anthropic
  model: claude-sonnet-4-5
  fallback:
    - openai
    - ollama

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    max_retries: 3
    timeout: 60

  openai:
    api_key: ${OPENAI_API_KEY}
    max_retries: 3
    timeout: 60

# Tool Settings
tools:
  budget: 100
  selection_strategy: hybrid  # keyword, semantic, hybrid
  cache_enabled: true
  max_parallel_tools: 5

# Security Settings
security:
  enabled: true
  mode: rbac
  default_role: viewer
  audit_log: true
  safe_mode: true

  roles:
    admin:
      permissions:
        - "*:*"
    developer:
      permissions:
        - "tools:*"
        - "workflows:*"
        - "coding:*"
        - "devops:read"

# Workflow Settings
workflows:
  cache_enabled: true
  cache_size: 100
  checkpoint_backend: memory  # memory, sqlite, postgres
  auto_validation: true

# Performance Settings
performance:
  lazy_loading: true
  parallel_workers: 4
  cache_ttl: 3600
  optimize_memory: true

# Memory Systems
memory:
  episodic:
    enabled: true
    max_memories: 1000
    ttl: 604800  # 7 days

  semantic:
    enabled: true
    consolidation_threshold: 100
```

### Project-Specific Configuration

Create `.victor.md` in your project directory:

```markdown
# My Project Configuration

## Vertical Settings
vertical: coding
mode: build

## Persona
persona: senior-developer

## Custom Instructions
- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Add type hints to all functions
- Test coverage must exceed 80%

## Tool Restrictions
allowed_tools:
  - read_file
  - write_file
  - bash
  - git_operations

blocked_tools:
  - docker_exec  # Safety restriction
```

---

## Database Setup

### SQLite (Default - No Configuration Required)

SQLite is used by default for lightweight deployments:

```bash
# Database location: ~/.victor/victor.db
# No setup required - automatically created on first run
```

### PostgreSQL (Recommended for Production)

For production deployments with high concurrency:

```bash
# 1. Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# 2. Create database and user
sudo -u postgres psql
CREATE DATABASE victor;
CREATE USER victor_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE victor TO victor_user;
\q

# 3. Install dependencies
pip install "victor-ai[checkpoints]"

# 4. Configure environment
export VICTOR_CHECKPOINT_DB_URL=postgresql://victor_user:secure_password@localhost/victor

# 5. Run migrations
victor db migrate
```

### Redis (Optional - for Caching and Event Bus)

```bash
# 1. Install Redis
sudo apt-get install redis-server

# 2. Configure environment
export VICTOR_CACHE_BACKEND=redis
export VICTOR_CACHE_REDIS_URL=redis://localhost:6379/0
export VICTOR_EVENT_BACKEND=redis
export VICTOR_EVENT_REDIS_URL=redis://localhost:6379/1
```

### Vector Database (Optional - for RAG)

```bash
# ChromaDB (default, embedded)
pip install chromadb

# Qdrant (recommended for production)
docker run -d -p 6333:6333 qdrant/qdrant

export VICTOR_RAG_VECTOR_DB=qdrant
export VICTOR_RAG_QDRANT_URL=http://localhost:6333
```

---

## Deployment Strategies

### Strategy 1: Single-Server Deployment

**Best for:** Small teams, development environments

```bash
# 1. Install Victor
pip install "victor-ai[all]"

# 2. Configure environment
cat > ~/.victor/.env << EOF
ANTHROPIC_API_KEY=your_key
VICTOR_PROFILE=production
VICTOR_DEFAULT_PROVIDER=anthropic
EOF

# 3. Initialize
victor init

# 4. Start API server (optional)
victor api start --port 8000

# 5. Verify
victor doctor
```

### Strategy 2: Docker Compose Deployment

**Best for:** Production, scalable deployments

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  victor-api:
    image: victor-ai:0.5.0
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - VICTOR_PROFILE=production
      - VICTOR_EVENT_BACKEND=redis
      - VICTOR_EVENT_REDIS_URL=redis://redis:6379/1
      - VICTOR_CHECKPOINT_DB_URL=postgresql://victor:victor@postgres:5432/victor
    volumes:
      - victor-data:/app/.victor
      - ./projects:/app/projects
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_DB=victor
      - POSTGRES_USER=victor
      - POSTGRES_PASSWORD=victor
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage
    restart: unless-stopped

volumes:
  victor-data:
  postgres-data:
  qdrant-data:
```

Deploy:

```bash
# 1. Create environment file
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-xxxxx
EOF

# 2. Start services
docker-compose up -d

# 3. Check status
docker-compose ps

# 4. View logs
docker-compose logs -f victor-api
```

### Strategy 3: Kubernetes Deployment

**Best for:** Enterprise, high-availability deployments

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: victor-ai
  labels:
    app: victor-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: victor-ai
  template:
    metadata:
      labels:
        app: victor-ai
    spec:
      containers:
      - name: victor-ai
        image: victor-ai:0.5.0
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: victor-secrets
              key: anthropic-api-key
        - name: VICTOR_PROFILE
          value: "production"
        - name: VICTOR_EVENT_BACKEND
          value: "kafka"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: victor-ai-service
spec:
  selector:
    app: victor-ai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:

```bash
# 1. Create secrets
kubectl create secret generic victor-secrets \
  --from-literal=anthropic-api-key=sk-ant-xxxxx

# 2. Apply deployment
kubectl apply -f k8s/deployment.yaml

# 3. Check status
kubectl get pods -l app=victor-ai

# 4. Get service URL
kubectl get service victor-ai-service
```

### Strategy 4: Air-Gapped Deployment

**Best for:** Offline environments, high-security installations

```bash
# 1. Download all dependencies on internet-connected machine
pip download -d victor-packages "victor-ai[all]"

# 2. Create local provider configuration
cat > airgapped_config.yaml << EOF
providers:
  default: ollama
  ollama:
    base_url: http://localhost:11434
    models:
      - llama3
      - codellama

airgapped_mode: true
disable_network_tools: true
EOF

# 3. Transfer to air-gapped system and install
pip install --no-index --find-links victor-packages victor-ai

# 4. Start Ollama with local models
ollama serve
ollama pull llama3
ollama pull codellama

# 5. Configure Victor
export VICTOR_AIRGAPPED_MODE=true
export VICTOR_CONFIG_FILE=airgapped_config.yaml

# 6. Verify
victor --provider ollama --model llama3 chat
```

---

## Verification Steps

### Health Check

```bash
# Run comprehensive health check
victor doctor

# Expected output:
# ✓ Provider connectivity: OK
# ✓ Tool registry: OK (55 tools)
# ✓ Verticals loaded: OK (5/5)
# ✓ Workflow engine: OK
# ✓ Event bus: OK
# ✓ Database: OK
# ✓ Cache: OK
```

### Quick Start Test

```bash
# Test basic functionality
victor chat --no-tui

# Try: "List Python files in current directory"
# Should work without errors
```

### API Server Test

```bash
# Start API server
victor api start --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, Victor!",
    "provider": "anthropic",
    "model": "claude-sonnet-4-5"
  }'
```

### Integration Tests

```bash
# Run smoke tests
pytest tests/smoke -v

# Run integration tests
pytest tests/integration -v -m integration

# Expected: All tests passing
```

### Performance Baseline

```bash
# Run performance benchmarks
python scripts/benchmark_tool_selection.py run --group all

# Expected: Tool selection latency < 200ms
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Provider Connection Failed

**Symptoms:**
```
Error: Failed to connect to Anthropic API
```

**Solutions:**
```bash
# 1. Verify API key
echo $ANTHROPIC_API_KEY

# 2. Test connectivity
curl https://api.anthropic.com/v1/messages

# 3. Check rate limits
# Visit: https://console.anthropic.com/settings/limits

# 4. Configure fallback provider
export VICTOR_DEFAULT_PROVIDER=openai
```

#### Issue 2: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'tree_sitter_python'
```

**Solutions:**
```bash
# 1. Reinstall with all dependencies
pip install --force-reinstall "victor-ai[all]"

# 2. Install specific language parser
pip install tree-sitter-python

# 3. Verify installation
python -c "import tree_sitter_python; print('OK')"
```

#### Issue 3: Database Lock Errors

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**
```bash
# 1. Switch to PostgreSQL for production
export VICTOR_CHECKPOINT_DB_URL=postgresql://user:pass@localhost/victor

# 2. Or use SQLite with timeout
export VICTOR_SQLITE_TIMEOUT=30

# 3. Check for other running instances
ps aux | grep victor
```

#### Issue 4: Out of Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate memory
```

**Solutions:**
```bash
# 1. Enable lazy loading
export VICTOR_LAZY_LOADING=true

# 2. Reduce worker count
export VICTOR_MAX_WORKERS=2

# 3. Enable memory optimization
export VICTOR_OPTIMIZE_MEMORY=true

# 4. Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Issue 5: Slow Tool Selection

**Symptoms:**
```
Tool selection takes > 1 second
```

**Solutions:**
```bash
# 1. Enable caching
export VICTOR_CACHE_ENABLED=true
export VICTOR_CACHE_TTL=3600

# 2. Use simpler selection strategy
export VICTOR_TOOL_SELECTION_STRATEGY=keyword

# 3. Reduce tool budget
export VICTOR_TOOL_BUDGET=50

# 4. Run benchmark to identify bottlenecks
python scripts/benchmark_tool_selection.py run --group all
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Enable debug logging
export VICTOR_LOG_LEVEL=DEBUG
export VICTOR_LOG_FORMAT=detailed

# Run with verbose output
victor --verbose chat

# Check logs
tail -f ~/.victor/logs/victor.log
```

### Getting Help

```bash
# Get help on any command
victor --help
victor chat --help
victor api --help

# Check system status
victor status

# View configuration
victor config show

# Validate configuration
victor config validate
```

---

## Upgrade Procedures

### Standard Upgrade

```bash
# 1. Backup configuration
cp -r ~/.victor ~/.victor.backup

# 2. Backup database (if using PostgreSQL)
pg_dump victor > victor_backup.sql

# 3. Upgrade package
pip install --upgrade victor-ai

# 4. Run migrations (if needed)
victor db migrate

# 5. Verify upgrade
victor doctor

# 6. Run tests
pytest tests/smoke -v
```

### Rolling Upgrade (Zero Downtime)

```bash
# 1. Deploy new version to one node
kubectl set image deployment/victor-ai \
  victor-ai=victor-ai:0.5.1 \
  --rolling-update=true

# 2. Monitor new pods
kubectl get pods -w

# 3. Verify new version is working
kubectl logs -f deployment/victor-ai

# 4. Continue rolling update if healthy
# Kubernetes will automatically update remaining pods
```

### Rollback Procedure

```bash
# 1. Stop current version
victor api stop

# 2. Restore previous version
pip install victor-ai==0.4.9

# 3. Restore configuration
cp -r ~/.victor.backup ~/.victor

# 4. Restore database (if needed)
psql victor < victor_backup.sql

# 5. Restart service
victor api start

# 6. Verify rollback
victor doctor
```

---

## Additional Resources

- **Architecture Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Features Guide**: [FEATURES.md](FEATURES.md)
- **Operations Guide**: [OPERATIONS.md](OPERATIONS.md)
- **Security Guide**: [SECURITY.md](SECURITY.md)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## Support

- **GitHub Issues**: https://github.com/your-org/victor-ai/issues
- **Discord Community**: https://discord.gg/victor-ai
- **Documentation**: https://docs.victor-ai.com
- **Email**: support@victor-ai.com

---

**Document Version:** 1.0.0
**Last Reviewed:** 2025-01-20
**Next Review:** 2025-02-20
