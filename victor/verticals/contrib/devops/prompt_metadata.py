# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serializable prompt metadata for the DevOps vertical."""

from __future__ import annotations

from typing import Any, Dict

DEVOPS_TASK_TYPE_HINTS: Dict[str, Dict[str, Any]] = {
    "infrastructure": {
        "hint": """[INFRASTRUCTURE] Deploy infrastructure (Kubernetes, Terraform, Docker, cloud):
1. Use Infrastructure as Code (Terraform, CloudFormation, Pulumi)
2. Implement multi-stage Docker builds for smaller images
3. Define resource limits and requests for Kubernetes
4. Use ConfigMaps/Secrets for configuration management
5. Tag all resources for cost tracking and organization""",
        "tool_budget": 15,
        "priority_tools": ["shell", "read", "write", "edit"],
    },
    "ci_cd": {
        "hint": """[CI/CD] Configure continuous integration/deployment:
1. Define clear stages: lint, test, build, deploy
2. Cache dependencies for faster builds
3. Use matrix builds for cross-platform testing
4. Implement proper secret management (GitHub Secrets, Vault)
5. Add manual approval for production deployments""",
        "tool_budget": 12,
        "priority_tools": ["read", "write", "edit", "shell"],
    },
    "dockerfile": {
        "hint": """[DOCKER] Create or optimize Dockerfile:
1. Use official base images with specific tags
2. Implement multi-stage builds for smaller images
3. Order layers for cache optimization
4. Add health checks and proper signal handling
5. Run as non-root user when possible""",
        "tool_budget": 8,
        "priority_tools": ["read", "write", "edit", "shell"],
    },
    "docker_compose": {
        "hint": """[COMPOSE] Create Docker Compose configuration:
1. Define all services with explicit dependencies
2. Use named volumes for persistent data
3. Configure proper network isolation
4. Add health checks for service readiness
5. Use environment files for secrets (never hardcode)""",
        "tool_budget": 10,
        "priority_tools": ["read", "write", "edit", "ls"],
    },
    "kubernetes": {
        "hint": """[K8S] Create Kubernetes manifests:
1. Use Deployments for stateless apps, StatefulSets for stateful
2. Define resource requests and limits
3. Add liveness and readiness probes
4. Use ConfigMaps for config, Secrets for sensitive data
5. Implement NetworkPolicies for security""",
        "tool_budget": 12,
        "priority_tools": ["read", "write", "edit", "shell"],
    },
    "terraform": {
        "hint": """[TERRAFORM] Write Infrastructure as Code:
1. Organize into modules for reusability
2. Use remote state with locking
3. Implement proper variable typing and validation
4. Tag all resources for cost tracking
5. Use data sources instead of hardcoded IDs""",
        "tool_budget": 15,
        "priority_tools": ["read", "write", "edit", "ls", "shell"],
    },
    "monitoring": {
        "hint": """[MONITOR] Set up observability:
1. Define key metrics and SLIs/SLOs
2. Configure alerting with appropriate thresholds
3. Set up distributed tracing for microservices
4. Implement structured logging
5. Create dashboards for visibility""",
        "tool_budget": 12,
        "priority_tools": ["read", "write", "edit", "shell"],
    },
    "general": {
        "hint": """[GENERAL DEVOPS] For general infrastructure queries:
1. Read existing configuration files first (ls, read)
2. Check for Dockerfiles, compose files, k8s manifests
3. Use shell for quick inspections
4. Follow security best practices
5. Document any changes made""",
        "tool_budget": 10,
        "priority_tools": ["read", "ls", "shell", "grep"],
    },
}

DEVOPS_SYSTEM_PROMPT_SECTION = """
## Security Checklist

Before finalizing any infrastructure configuration:
- [ ] No hardcoded secrets, passwords, or API keys
- [ ] Using least-privilege IAM/RBAC policies
- [ ] Network traffic encrypted in transit
- [ ] Data encrypted at rest
- [ ] Container running as non-root
- [ ] Resource limits defined
- [ ] Logging and audit trails enabled

## Common Pitfalls to Avoid

1. **Docker**: Using `latest` tag, running as root, missing health checks
2. **Kubernetes**: No resource limits, missing probes, using default namespace
3. **Terraform**: Local state, no locking, hardcoded values
4. **CI/CD**: Secrets in logs, no artifact versioning, missing rollback
5. **Monitoring**: Alert fatigue, missing business metrics, no runbooks
""".strip()

DEVOPS_GROUNDING_RULES = """GROUNDING: Base ALL responses on tool output only. Never invent file paths or content.
Verify configuration syntax before suggesting. Always check existing resources first.""".strip()

DEVOPS_PROMPT_PRIORITY = 5

DEVOPS_PROMPT_TEMPLATES: Dict[str, str] = {
    "devops_operations": DEVOPS_SYSTEM_PROMPT_SECTION,
}

__all__ = [
    "DEVOPS_GROUNDING_RULES",
    "DEVOPS_PROMPT_PRIORITY",
    "DEVOPS_PROMPT_TEMPLATES",
    "DEVOPS_SYSTEM_PROMPT_SECTION",
    "DEVOPS_TASK_TYPE_HINTS",
]
