# victor-devops

**DevOps Vertical for Victor AI**

A complete DevOps implementation showcasing infrastructure automation, CI/CD, and cloud operations with Victor AI.

## Features

- 🚀 **Infrastructure Automation**
  - Docker and Kubernetes operations
  - Terraform and CloudFormation support
  - Infrastructure-as-code generation and validation

- 🔧 **CI/CD Pipeline Management**
  - GitHub Actions workflow generation
  - Jenkins pipeline configuration
  - Build and deployment automation

- ☁️ **Cloud Operations**
  - AWS, Azure, GCP integration
  - Service deployment and scaling
  - Monitoring and alerting setup

- 🔍 **Configuration Analysis**
  - Dockerfile optimization
  - Kubernetes manifest validation
  - Infrastructure security scanning

## Installation

```bash
# Install with Victor core
pip install victor-ai

# Install DevOps vertical
pip install victor-devops
```

## Quick Start

```python
from victor.framework import Agent

# Create agent with DevOps vertical
agent = await Agent.create(
    provider="anthropic",
    model="claude-3-opus-20240229",
    vertical="devops"
)

# Deploy a Docker container
await agent.chat("Deploy a nginx container with port 8080 exposed")

# Create a Kubernetes deployment
await agent.chat("Create a Kubernetes deployment for a Node.js app")
```

## Available Tools

Once installed, the DevOps vertical provides these tools:

- **docker_build** - Build Docker images
- **docker_run** - Run Docker containers
- **k8s_apply** - Apply Kubernetes manifests
- **terraform_plan** - Plan Terraform changes
- **terraform_apply** - Apply Terraform configuration
- **github_workflow** - Generate GitHub Actions workflows

## System Prompt

The DevOps vertical includes specialized guidance for:

- Infrastructure best practices
- Container security and optimization
- CI/CD pipeline patterns
- Cloud provider-specific recommendations
- Configuration management

## Configuration

The DevOps vertical can be configured via environment variables:

```bash
# Default cloud provider
export VICTOR_DEVOPS_CLOUD_PROVIDER=aws

# Default Kubernetes namespace
export VICTOR_DEVOPS_K8S_NAMESPACE=default

# Docker registry
export VICTOR_DEVOPS_DOCKER_REGISTRY=docker.io
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black victor_devops/

# Type check
mypy victor_devops/
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Links

- **Victor AI**: https://github.com/vijay-singh/codingagent
- **Documentation**: https://docs.victor.dev/verticals/devops
- **Victor Registry**: https://github.com/vjsingh1984/victor-registry
