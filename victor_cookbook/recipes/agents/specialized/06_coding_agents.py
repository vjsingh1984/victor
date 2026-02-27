"""Specialized coding agent recipes.

These recipes focus on software development tasks like
code review, refactoring, debugging, testing, and documentation.
"""

RECIPE_CATEGORY = "agents/specialized"
RECIPE_DIFFICULTY = "intermediate"
RECIPE_TIME = "10 minutes"


async def code_review_agent(code: str, language: str = "Python"):
    """Review code for best practices and potential issues."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        temperature=0.3
    )

    result = await agent.run(
        f"""Review this {language} code:

{code}

Provide:
1. Code quality assessment
2. Bug identification
3. Security vulnerabilities
4. Performance issues
5. Best practices violations
6. Suggested improvements with examples"""
    )

    return result.content


async def refactoring_agent(code: str, goals: list[str]):
    """Refactor code according to specified goals."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        tools=["read", "write"],
        temperature=0.4
    )

    goals_str = "\n".join(f"- {g}" for g in goals)

    result = await agent.run(
        f"""Refactor this code:

{code}

Refactoring goals:
{goals_str}

Provide:
1. Refactored code
2. Explanation of changes
3. Benefits of refactoring
4. Any trade-offs"""
    )

    return result.content


async def debugging_agent(
    code: str,
    error_message: str,
    context: str = ""
):
    """Debug code with error information."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        tools=["read", "grep", "python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Debug this code:

CODE:
{code}

ERROR:
{error_message}

CONTEXT:
{context}

Provide:
1. Root cause analysis
2. Explanation of the bug
3. Fixed code
4. How to prevent similar bugs"""
    )

    return result.content


async def test_generation_agent(code: str, test_framework: str = "pytest"):
    """Generate unit tests for code."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        temperature=0.3
    )

    result = await agent.run(
        f"""Generate comprehensive unit tests for this code using {test_framework}:

{code}

Provide:
1. Test file with multiple test cases
2. Edge cases covered
3. Mock setup if needed
4. Test documentation"""
    )

    return result.content


async def documentation_agent(code_path: str, doc_style: str = "google"):
    """Generate documentation for code."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        tools=["read"],
        temperature=0.4
    )

    result = await agent.run(
        f"""Generate documentation for code at {code_path}.

Docstring style: {doc_style}

Provide:
1. Module docstring
2. Function/class docstrings
3. Inline comments for complex logic
4. Usage examples
5. Type hints where missing"""
    )

    return result.content


async def code_explanation_agent(code: str, audience: str = "intermediate"):
    """Explain code at appropriate level."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        temperature=0.4
    )

    result = await agent.run(
        f"""Explain this code for a {audience} developer:

{code}

Provide:
1. High-level summary
2. Line-by-line explanation
3. Design patterns used
4. Why it works this way
5. Common pitfalls"""
    )

    return result.content


async def api_design_agent(requirements: str):
    """Design API from requirements."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        temperature=0.5
    )

    result = await agent.run(
        f"""Design a REST API based on these requirements:

{requirements}

Provide:
1. Endpoint specifications
2. Request/response schemas
3. Authentication strategy
4. Error handling
5. Rate limiting approach
6. Versioning strategy
7. OpenAPI/Swagger spec"""
    )

    return result.content


async def database_schema_agent(requirements: str):
    """Design database schema from requirements."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        temperature=0.4
    )

    result = await agent.run(
        f"""Design a database schema:

REQUIREMENTS:
{requirements}

Provide:
1. Entity-Relationship diagram (text)
2. Table definitions with SQL
3. Indexes for performance
4. Foreign key relationships
5. Constraints and validations"""
    )

    return result.content


async def microservice_design_agent(feature: str):
    """Design microservice architecture for a feature."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        temperature=0.5
    )

    result = await agent.run(
        f"""Design microservice architecture for: {feature}

Provide:
1. Service boundaries
2. API contracts between services
3. Data storage strategy
4. Inter-service communication
5. Service discovery
6. Observability approach
7. Deployment strategy"""
    )

    return result.content


async def code_migration_agent(
    code: str,
    source_lang: str,
    target_lang: str
):
    """Migrate code from one language to another."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        temperature=0.3
    )

    result = await agent.run(
        f"""Migrate this code from {source_lang} to {target_lang}:

{code}

Provide:
1. Migrated code
2. Language-specific idioms used
3. Library equivalents
4. Any behavioral differences
5. Testing recommendations"""
    )

    return result.content


async def performance_optimization_agent(code: str, bottleneck: str = ""):
    """Optimize code for performance."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        tools=["python"],
        temperature=0.4
    )

    result = await agent.run(
        f"""Optimize this code for performance:

{code}

BOTTLENECK: {bottleneck if bottleneck else 'Unknown - identify it'}

Provide:
1. Performance analysis
2. Optimized code
3. Performance improvement estimate
4. Trade-offs made
5. Profiling recommendations"""
    )

    return result.content


async def security_audit_agent(code: str):
    """Audit code for security vulnerabilities."""
    from victor import Agent

    agent = Agent.create(
        vertical="security",
        temperature=0.2
    )

    result = await agent.run(
        f"""Perform security audit of this code:

{code}

Check for:
1. SQL injection
2. XSS vulnerabilities
3. CSRF issues
4. Authentication/authorization flaws
5. Sensitive data exposure
6. Insecure dependencies
7. Cryptographic issues

Provide severity ratings (Critical/High/Medium/Low) for each finding."""
    )

    return result.content


async def dependency_upgrade_agent(
    requirements_file: str,
    constraints: str = ""
):
    """Plan dependency upgrades."""
    from victor import Agent

    agent = Agent.create(
        vertical="coding",
        tools=["read"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Plan dependency upgrades for {requirements_file}.

CONSTRAINTS:
{constraints}

Provide:
1. Current dependency analysis
2. Available upgrades
3. Breaking changes to watch for
4. Upgrade order
5. Testing strategy
6. Rollback plan"""
    )

    return result.content


async def ci_cd_pipeline_agent(project_type: str):
    """Generate CI/CD pipeline configuration."""
    from victor import Agent

    agent = Agent.create(
        vertical="devops",
        temperature=0.3
    )

    result = await agent.run(
        f"""Generate CI/CD pipeline for {project_type} project.

Provide:
1. GitHub Actions workflow
2. Build steps
3. Test automation
4. Deployment strategy
5. Environment variable management
6. Secrets handling
7. Notification setup"""
    )

    return result.content


async def docker_compose_agent(services: list[str]):
    """Generate Docker Compose configuration."""
    from victor import Agent

    agent = Agent.create(
        vertical="devops",
        temperature=0.3
    )

    services_str = ", ".join(services)

    result = await agent.run(
        f"""Generate Docker Compose configuration for: {services_str}

Provide:
1. docker-compose.yml
2. Service definitions
3. Volume mounts
4. Network configuration
5. Environment variables
6. Health checks
7. Deployment instructions"""
    )

    return result.content


async def kubernetes_manifest_agent(service_spec: str):
    """Generate Kubernetes manifests."""
    from victor import Agent

    agent = Agent.create(
        vertical="devops",
        temperature=0.3
    )

    result = await agent.run(
        f"""Generate Kubernetes manifests for:

{service_spec}

Provide:
1. Deployment YAML
2. Service YAML
3. ConfigMap YAML
4. Secret YAML
5. Ingress YAML (if needed)
6. HPA YAML (if needed)
7. Namespace and RBAC if needed"""
    )

    return result.content


async def terraform_agent(infrastructure: str):
    """Generate Terraform configuration."""
    from victor import Agent

    agent = Agent.create(
        vertical="devops",
        temperature=0.3
    )

    result = await agent.run(
        f"""Generate Terraform configuration for:

{infrastructure}

Provider: AWS

Provide:
1. Main Terraform files
2. Resource definitions
3. Variables
4. Outputs
5. Module structure
6. State management recommendations"""
    )

    return result.content


async def demo_coding_agents():
    """Demonstrate coding agent recipes."""
    print("=== Coding Agent Recipes ===\n")

    print("1. Code Review:")
    result = await code_review_agent(
        "def add(a, b): return a + b",
        "Python"
    )
    print(result[:300] + "...\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_coding_agents())
