# Coordinator-Based Architecture: Recipes

**Version**: 1.0
**Date**: 2025-01-13
**Audience**: Developers, Advanced Users

---

## Table of Contents

1. [Introduction](#introduction)
2. [Recipe 1: Add File-Based Configuration Provider](#recipe-1-add-file-based-configuration-provider)
3. [Recipe 2: Add Project-Specific Prompt Prompts](#recipe-2-add-project-specific-prompt-contributors)
4. [Recipe 3: Export Analytics to Database](#recipe-3-export-analytics-to-database)
5. [Recipe 4: Implement Smart Context Compaction](#recipe-4-implement-smart-context-compaction)
6. [Recipe 5: Add Custom Middleware Integration](#recipe-5-add-custom-middleware-integration)
7. [Recipe 6: Multi-Tenant Configuration](#recipe-6-multi-tenant-configuration)
8. [Recipe 7: Real-Time Analytics Dashboard](#recipe-7-real-time-analytics-dashboard)
9. [Recipe 8: Custom Tool Selection Strategy](#recipe-8-custom-tool-selection-strategy)
10. [Recipe 9: A/B Testing Coordinator](#recipe-9-ab-testing-coordinator)

---

## Introduction

This document provides step-by-step recipes for common coordinator customization tasks. Each recipe includes:

- **Problem Statement**: What problem does this recipe solve?
- **Solution Overview**: High-level approach
- **Step-by-Step Instructions**: Detailed implementation
- **Code Example**: Complete, runnable code
- **Testing**: How to test the implementation
- **Production Considerations**: Things to consider for production use

### Prerequisites

- Victor installed (`pip install victor-ai`)
- Basic knowledge of Python and async/await
- Understanding of coordinator architecture (see [Quick Start](coordinator_quickstart.md))

---

## Recipe 1: Add File-Based Configuration Provider

### Problem Statement

You want to load orchestrator configuration from YAML/JSON files instead of (or in addition to) environment variables and settings objects.

### Solution Overview

Create a custom `IConfigProvider` that reads configuration from files, and register it with `ConfigCoordinator`.

### Step-by-Step Instructions

#### Step 1: Create the File Config Provider

```python
# file_config_provider.py
from pathlib import Path
from typing import Dict
import yaml
import json
from victor.protocols import IConfigProvider

class FileConfigProvider(IConfigProvider):
    """Load configuration from YAML or JSON files."""

    def __init__(self, config_path: Path, priority: int = 75):
        """
        Args:
            config_path: Path to config file (YAML or JSON)
            priority: Provider priority (higher = checked first)
        """
        self.config_path = config_path
        self._priority = priority

    def priority(self) -> int:
        return self._priority

    async def get_config(self, session_id: str) -> Dict:
        """Load config from file."""
        if not self.config_path.exists():
            return {}  # Let next provider try

        try:
            if self.config_path.suffix in ['.yml', '.yaml']:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            elif self.config_path.suffix == '.json':
                with open(self.config_path, 'r') as f:
                    return json.load(f) or {}
            else:
                return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
```

#### Step 2: Use with Orchestrator

```python
# main.py
from pathlib import Path
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.coordinators import ConfigCoordinator
from victor.config.settings import Settings

# Create file config provider
config_path = Path("config/orchestrator.yml")
file_provider = FileConfigProvider(config_path, priority=100)

# Create config coordinator with file provider
config_coordinator = ConfigCoordinator(providers=[
    file_provider,              # Try file first
    EnvironmentConfigProvider(),  # Fallback to environment
])

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(),
    provider=provider,
    model="claude-sonnet-4-5",
    _config_coordinator=config_coordinator
)
```

#### Step 3: Create Configuration File

```yaml
# config/orchestrator.yml
model: claude-sonnet-4-5
temperature: 0.7
max_tokens: 4096
thinking: false

# Tool selection
tool_selection:
  strategy: hybrid
  hybrid_alpha: 0.7

# Context management
context_compaction_strategy: semantic
context_compaction_threshold: 0.8

# Analytics
enable_analytics: true
analytics_export_interval: 60
```

### Testing

```python
import pytest
import tempfile
from pathlib import Path

@pytest.mark.asyncio
async def test_file_config_provider():
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("model: test-model\ntemperature: 0.5\n")
        temp_path = Path(f.name)

    try:
        provider = FileConfigProvider(temp_path)
        config = await provider.get_config("test-session")

        assert config['model'] == 'test-model'
        assert config['temperature'] == 0.5
    finally:
        temp_path.unlink()
```

### Production Considerations

- **File permissions**: Ensure config files have appropriate permissions (e.g., 0600)
- **Secrets management**: Don't store API keys in config files; use environment variables or secret managers
- **Validation**: Add schema validation for configuration files
- **Hot reload**: Consider implementing config file watching for dynamic updates
- **Fallbacks**: Always have fallback providers (e.g., environment variables)

---

## Recipe 2: Add Project-Specific Prompt Contributors

### Problem Statement

You want to add project-specific instructions to prompts (e.g., coding standards, compliance requirements, team conventions).

### Solution Overview

Create custom `IPromptContributor` implementations that inject project-specific prompts.

### Step-by-Step Instructions

#### Step 1: Create Custom Prompt Contributors

```python
# prompt_contributors.py
from victor.agent.coordinators.prompt_coordinator import BasePromptContributor
from victor.protocols import PromptContext

class ProjectStandardsContributor(BasePromptContributor):
    """Add project coding standards to prompts."""

    def __init__(self, project_path: str):
        self.project_path = project_path

    def priority(self) -> int:
        return 60  # Medium priority

    async def get_contribution(self, context: PromptContext) -> str:
        """Read project standards from files."""

        # Check for project standards file
        standards_file = Path(self.project_path) / ".victor" / "standards.md"
        if not standards_file.exists():
            return ""

        with open(standards_file, 'r') as f:
            standards = f.read()

        return f"""

## Project-Specific Standards
{standards}
"""

class ComplianceContributor(BasePromptContributor):
    """Add compliance requirements based on industry."""

    def __init__(self, industry: str):
        self.industry = industry

    def priority(self) -> int:
        return 70  # High priority

    async def get_contribution(self, context: PromptContext) -> str:
        """Add compliance instructions."""

        requirements = {
            "healthcare": """
## Healthcare Compliance
- Follow HIPAA guidelines for patient data
- No personal health information (PHI) in code
- Ensure audit logging for data access
""",
            "finance": """
## Financial Compliance
- Follow PCI-DSS standards for payment data
- Implement SOC2 controls
- No hardcoding of credentials
""",
            "general": """
## General Compliance
- Follow GDPR data protection guidelines
- Implement proper error handling
- Use secure communication channels
"""
        }

        return requirements.get(self.industry, requirements["general"])

class TechStackContributor(BasePromptContributor):
    """Add technology-specific instructions."""

    def __init__(self, tech_stack: list):
        self.tech_stack = tech_stack

    def priority(self) -> int:
        return 50  # Medium-low priority

    async def get_contribution(self, context: PromptContext) -> str:
        """Add tech stack specific guidance."""

        guidelines = {
            "python": "- Use type hints for all functions\n- Follow PEP 8 style guide\n- Use async/await for I/O operations",
            "typescript": "- Use strict mode\n- Prefer interfaces over types\n- Use async/await over promises",
            "rust": "- Use Result types for error handling\n- Prefer iterators over collections\n- Use clap for CLI argument parsing",
        }

        instructions = [
            f"\n## {tech.title()} Guidelines\n{guidelines.get(tech, '')}"
            for tech in self.tech_stack
            if tech in guidelines
        ]

        return "\n".join(instructions)
```

#### Step 2: Register with Orchestrator

```python
# main.py
from victor.agent.coordinators import PromptCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create custom contributors
project_contributor = ProjectStandardsContributor(project_path="/path/to/project")
compliance_contributor = ComplianceContributor(industry="healthcare")
tech_stack_contributor = TechStackContributor(tech_stack=["python", "typescript"])

# Create prompt coordinator
prompt_coordinator = PromptCoordinator(contributors=[
    compliance_contributor,    # High priority, applied first
    project_contributor,       # Medium priority
    tech_stack_contributor,    # Lower priority
])

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(),
    provider=provider,
    model=model,
    _prompt_coordinator=prompt_coordinator
)
```

#### Step 3: Create Project Standards File

```markdown
# .victor/standards.md

## Code Style

- Max line length: 100 characters
- Use 4 spaces for indentation
- Docstrings required for all public functions

## Testing

- Minimum 80% code coverage
- All functions must have unit tests
- Integration tests for API endpoints

## Security

- No secrets in code
- Use environment variables for configuration
- Run security scans before commits
```

### Testing

```python
@pytest.mark.asyncio
async def test_project_standards_contributor():
    contributor = ProjectStandardsContributor(project_path="/tmp/test_project")

    # Create test standards file
    standards_dir = Path("/tmp/test_project/.victor")
    standards_dir.mkdir(parents=True, exist_ok=True)

    standards_file = standards_dir / "standards.md"
    standards_file.write_text("# Test Standards\n- Follow PEP 8")

    context = PromptContext({"task": "code_generation"})
    contribution = await contributor.get_contribution(context)

    assert "Test Standards" in contribution
    assert "Follow PEP 8" in contribution
```

### Production Considerations

- **File caching**: Cache prompt contributions to avoid repeated file reads
- **Validation**: Validate prompt contribution size to avoid token limit issues
- **Version control**: Track prompt contributors in version control
- **Dynamic updates**: Consider hot-reloading prompt files when they change
- **Fallbacks**: Provide default prompts if files are missing

---

## Recipe 3: Export Analytics to Database

### Problem Statement

You want to store usage analytics in a database for reporting, billing, or analysis.

### Solution Overview

Create a custom `IAnalyticsExporter` that writes analytics events to a database.

### Step-by-Step Instructions

#### Step 1: Create Database Exporter

```python
# analytics_exporters.py
from typing import List
from victor.protocols import IAnalyticsExporter, ExportResult, AnalyticsEvent
import asyncpg

class PostgreSQLAnalyticsExporter(BaseAnalyticsExporter):
    """Export analytics to PostgreSQL database."""

    def __init__(self, connection_string: str, batch_size: int = 100):
        """
        Args:
            connection_string: PostgreSQL connection string
            batch_size: Number of events to batch per insert
        """
        self.connection_string = connection_string
        self.batch_size = batch_size
        self._pool = None

    async def _get_pool(self):
        """Lazy-load connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.connection_string)
        return self._pool

    async def export(self, events: List[AnalyticsEvent]) -> ExportResult:
        """Export events to PostgreSQL."""
        if not events:
            return ExportResult(success=True, exported_count=0)

        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                # Batch insert events
                await conn.executemany(
                    """
                    INSERT INTO analytics_events
                    (session_id, event_type, event_data, timestamp, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (session_id, timestamp) DO NOTHING
                    """,
                    [
                        (
                            e.session_id,
                            e.type,
                            json.dumps(e.data),
                            e.timestamp
                        )
                        for e in events
                    ]
                )

            return ExportResult(
                success=True,
                exported_count=len(events)
            )
        except Exception as e:
            return ExportResult(
                success=False,
                error=str(e),
                exported_count=0
            )

    async def shutdown(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
```

#### Step 2: Create Database Schema

```sql
-- migrations/001_create_analytics_table.sql

CREATE TABLE IF NOT EXISTS analytics_events (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_event UNIQUE (session_id, timestamp)
);

-- Indexes for common queries
CREATE INDEX idx_session_id ON analytics_events(session_id);
CREATE INDEX idx_event_type ON analytics_events(event_type);
CREATE INDEX idx_timestamp ON analytics_events(timestamp);

-- Partitioning for large datasets (optional)
-- CREATE TABLE analytics_events_2025_01 PARTITION OF analytics_events
-- FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

#### Step 3: Use with Orchestrator

```python
# main.py
import os
from victor.agent.coordinators import AnalyticsCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create database exporter
db_exporter = PostgreSQLAnalyticsExporter(
    connection_string=os.getenv("DATABASE_URL"),
    batch_size=100
)

# Create analytics coordinator
analytics_coordinator = AnalyticsCoordinator(exporters=[
    db_exporter,
    ConsoleAnalyticsExporter(),  # Also log to console
])

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(enable_analytics=True),
    provider=provider,
    model=model,
    _analytics_coordinator=analytics_coordinator
)
```

### Testing

```python
import pytest
import asyncpg
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_postgresql_analytics_exporter():
    exporter = PostgreSQLAnalyticsExporter(
        connection_string="postgresql://localhost/test"
    )

    events = [
        AnalyticsEvent(
            session_id="test-session",
            type="tool_call",
            data={"tool": "read"},
            timestamp="2025-01-13T00:00:00Z"
        )
    ]

    # Mock database connection
    with patch.object(exporter, '_get_pool') as mock_pool:
        mock_conn = AsyncMock()
        mock_pool.return_value.__aenter__.return_value = mock_conn

        result = await exporter.export(events)

        assert result.success
        assert result.exported_count == 1
        mock_conn.executemany.assert_called_once()
```

### Production Considerations

- **Connection pooling**: Use connection pools for performance
- **Batch inserts**: Insert events in batches for efficiency
- **Retry logic**: Implement retry logic for transient failures
- **Dead letter queue**: Store failed events for later retry
- **Partitioning**: Partition tables by date for large datasets
- **Retention policy**: Implement data retention policies
- **Monitoring**: Monitor database performance and connection pool health

---

## Recipe 4: Implement Smart Context Compaction

### Problem Statement

You want to intelligently compact conversation context to maintain the most important information while reducing token usage.

### Solution Overview

Create a custom `BaseCompactionStrategy` that uses semantic analysis to preserve important messages.

### Step-by-Step Instructions

#### Step 1: Create Semantic Compaction Strategy

```python
# compaction_strategies.py
from typing import List
from victor.protocols import Context
from victor.agent.coordinators.context_coordinator import BaseCompactionStrategy

class SemanticCompactionStrategy(BaseCompactionStrategy):
    """Use semantic similarity to preserve important messages."""

    def __init__(self, embedding_service, target_ratio: float = 0.5):
        """
        Args:
            embedding_service: Service for computing embeddings
            target_ratio: Target size ratio (0.5 = keep 50% of messages)
        """
        self.embedding_service = embedding_service
        self.target_ratio = target_ratio

    async def compact(self, context: Context) -> Context:
        """Compaction using semantic clustering."""
        messages = context.messages

        if len(messages) <= 10:
            return context  # No compaction needed

        # Always keep system messages
        system_messages = [m for m in messages if m.role == "system"]
        user_assistant_messages = [m for m in messages if m.role != "system"]

        # Calculate target count
        target_count = max(
            10,  # Keep at least 10 messages
            int(len(user_assistant_messages) * self.target_ratio)
        )

        # Compute embeddings
        embeddings = await self._compute_embeddings(user_assistant_messages)

        # Select diverse messages
        selected_indices = await self._select_diverse_messages(
            embeddings,
            target_count
        )

        selected_messages = [user_assistant_messages[i] for i in selected_indices]

        # Combine system + selected messages
        return Context(messages=system_messages + selected_messages)

    async def _compute_embeddings(self, messages: List) -> List[List[float]]:
        """Compute embeddings for messages."""
        texts = [m.content for m in messages]
        return await self.embedding_service.embed_batch(texts)

    async def _select_diverse_messages(
        self,
        embeddings: List[List[float]],
        target_count: int
    ) -> List[int]:
        """Select diverse messages using k-means clustering."""
        from sklearn.cluster import KMeans
        import numpy as np

        # Convert to numpy array
        embedding_array = np.array(embeddings)

        # Perform k-means clustering
        n_clusters = min(target_count, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embedding_array)

        # Select message closest to centroid for each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_embeddings = embedding_array[cluster_mask]

            # Find closest to centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = np.argmin(distances)

            # Map back to original indices
            cluster_indices = np.where(cluster_mask)[0]
            selected_indices.append(cluster_indices[closest_idx])

        return sorted(selected_indices)

class RecentImportantCompactionStrategy(BaseCompactionStrategy):
    """Keep recent messages and important keywords."""

    def __init__(self, keep_last_n: int = 10, keywords: List[str] = None):
        self.keep_last_n = keep_last_n
        self.keywords = keywords or ["error", "bug", "issue", "fix", "important"]

    async def compact(self, context: Context) -> Context:
        """Keep recent messages + keyword matches."""
        messages = context.messages

        if len(messages) <= self.keep_last_n:
            return context

        # Always keep system messages
        system_messages = [m for m in messages if m.role == "system"]
        user_assistant_messages = [m for m in messages if m.role != "system"]

        # Keep messages with keywords
        keyword_messages = [
            m for m in user_assistant_messages
            if any(kw in m.content.lower() for kw in self.keywords)
        ]

        # Keep recent messages
        recent_messages = user_assistant_messages[-self.keep_last_n:]

        # Combine and deduplicate
        all_messages = keyword_messages + recent_messages
        unique_messages = list({id(m): m for m in all_messages}.values())

        # Sort by original order
        message_order = {m: i for i, m in enumerate(user_assistant_messages)}
        unique_messages.sort(key=lambda m: message_order.get(m, float('inf')))

        return Context(messages=system_messages + unique_messages)
```

#### Step 2: Use with Orchestrator

```python
# main.py
from victor.agent.coordinators import ContextCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create compaction strategy
compaction_strategy = SemanticCompactionStrategy(
    embedding_service=your_embedding_service,
    target_ratio=0.6  # Keep 60% of messages
)

# Create context coordinator
context_coordinator = ContextCoordinator(
    compaction_strategy=compaction_strategy
)

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(
        context_compaction_threshold=0.8,  # Trigger at 80% of context limit
    ),
    provider=provider,
    model=model,
    _context_coordinator=context_coordinator
)
```

### Testing

```python
@pytest.mark.asyncio
async def test_semantic_compaction():
    from unittest.mock import AsyncMock

    strategy = SemanticCompactionStrategy(
        embedding_service=AsyncMock(),
        target_ratio=0.5
    )

    # Create test context
    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content=f"Message {i}"),
        Message(role="assistant", content=f"Response {i}"),
        for i in range(20)
    ]
    context = Context(messages=messages)

    # Mock embeddings
    strategy.embedding_service.embed_batch = AsyncMock(
        return_value=[[0.1] * 1536 for _ in messages]
    )

    compacted = await strategy.compact(context)

    # Should have fewer messages
    assert len(compacted.messages) < len(messages)
    # Should still have system message
    assert any(m.role == "system" for m in compacted.messages)
```

### Production Considerations

- **Embedding cost**: Semantic compaction requires embedding computation (API costs)
- **Latency**: Clustering adds latency; consider caching
- **Fallback**: Have fallback to simpler strategies if semantic compaction fails
- **Tuning**: Tune `target_ratio` based on your use case
- **Performance**: For high-volume scenarios, consider approximate nearest neighbor algorithms

---

## Recipe 5: Add Custom Middleware Integration

### Problem Statement

You want to add custom middleware logic (e.g., logging, authentication, rate limiting) to all orchestrator operations.

### Solution Overview

Create a middleware coordinator that wraps all orchestrator calls.

### Step-by-Step Instructions

#### Step 1: Create Middleware Coordinator

```python
# middleware_coordinator.py
from typing import Callable, Any
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

class MiddlewareCoordinator:
    """Apply middleware to orchestrator operations."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.middlewares = []

    def add_middleware(self, middleware: Callable):
        """Add middleware function."""
        self.middlewares.append(middleware)

    def apply_middleware(self, func: Callable) -> Callable:
        """Apply all middleware to a function."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute middleware in order
            for middleware in self.middlewares:
                result = await middleware(self.orchestrator, func, *args, **kwargs)
                if result is not None:  # Middleware can return early
                    return result

            # Execute original function
            return await func(*args, **kwargs)

        return wrapper

# Middleware functions
async def logging_middleware(orchestrator, func, *args, **kwargs):
    """Log all orchestrator calls."""
    logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
    start = time.time()

    try:
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} completed in {duration:.2f}s")
        return result
    except Exception as e:
        duration = time.time() - start
        logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
        raise

async def rate_limit_middleware(orchestrator, func, *args, **kwargs):
    """Apply rate limiting to orchestrator calls."""
    # Implement rate limiting logic here
    # For example, using Redis or in-memory rate limiter
    return None  # Return None to continue to next middleware

async def authentication_middleware(orchestrator, func, *args, **kwargs):
    """Validate authentication tokens."""
    # Extract session_id from kwargs
    session_id = kwargs.get('session_id')

    if not session_id:
        raise AuthenticationError("No session_id provided")

    # Validate session
    is_valid = await orchestrator._session_coordinator.validate_session(session_id)

    if not is_valid:
        raise AuthenticationError(f"Invalid session: {session_id}")

    return None  # Continue to next middleware
```

#### Step 2: Apply Middleware

```python
# main.py
from victor.agent.orchestrator import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Create middleware coordinator
middleware_coordinator = MiddlewareCoordinator(orchestrator)

# Add middleware
middleware_coordinator.add_middleware(logging_middleware)
middleware_coordinator.add_middleware(authentication_middleware)
middleware_coordinator.add_middleware(rate_limit_middleware)

# Apply middleware to chat function
orchestrator.chat = middleware_coordinator.apply_middleware(orchestrator.chat)

# Use orchestrator (middleware is automatically applied)
response = await orchestrator.chat("Hello!", session_id="valid-session")
```

### Production Considerations

- **Middleware order**: Order matters (e.g., logging before auth for audit trail)
- **Performance**: Each middleware adds overhead
- **Error handling**: Ensure middleware handles errors gracefully
- **Testing**: Test middleware in isolation
- **Configuration**: Make middleware configurable via settings

---

## Recipe 6: Multi-Tenant Configuration

### Problem Statement

You want to serve multiple tenants with different configurations, prompts, and analytics.

### Solution Overview

Create tenant-specific coordinators that load configuration and behavior per tenant.

### Step-by-Step Instructions

#### Step 1: Create Tenant Config Provider

```python
# tenant_providers.py
from victor.protocols import IConfigProvider

class TenantConfigProvider(IConfigProvider):
    """Load configuration per tenant."""

    def __init__(self, db_connection):
        self.db = db_connection

    def priority(self) -> int:
        return 100  # High priority

    async def get_config(self, session_id: str) -> dict:
        """Load tenant config from database."""
        # Extract tenant_id from session_id
        tenant_id = session_id.split(':')[0]  # Format: "tenant:session"

        query = """
            SELECT config FROM tenant_configs
            WHERE tenant_id = $1 AND active = true
        """

        result = await self.db.fetchrow(query, tenant_id)

        if result:
            return result['config']
        return {}
```

#### Step 2: Create Tenant Prompt Contributor

```python
# tenant_contributors.py
from victor.agent.coordinators.prompt_coordinator import BasePromptContributor

class TenantPromptContributor(BasePromptContributor):
    """Add tenant-specific prompts."""

    def __init__(self, db_connection):
        self.db = db_connection

    def priority(self) -> int:
        return 80

    async def get_contribution(self, context: PromptContext) -> str:
        """Load tenant prompt template."""
        tenant_id = context.get('tenant_id')

        if not tenant_id:
            return ""

        query = """
            SELECT prompt_template FROM tenant_prompt_templates
            WHERE tenant_id = $1
        """

        result = await self.db.fetchrow(query, tenant_id)

        if result:
            return f"\n{result['prompt_template']}"
        return ""
```

#### Step 3: Create Multi-Tenant Orchestrator

```python
# multi_tenant_orchestrator.py
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.coordinators import ConfigCoordinator, PromptCoordinator

class MultiTenantOrchestrator:
    """Manage orchestrators per tenant."""

    def __init__(self, db_connection, base_settings, base_provider):
        self.db = db_connection
        self.base_settings = base_settings
        self.base_provider = base_provider
        self.orchestrators = {}  # tenant_id -> orchestrator

    async def get_orchestrator(self, tenant_id: str) -> AgentOrchestrator:
        """Get or create orchestrator for tenant."""
        if tenant_id not in self.orchestrators:
            # Create tenant-specific coordinators
            config_coordinator = ConfigCoordinator(providers=[
                TenantConfigProvider(self.db),
            ])

            prompt_coordinator = PromptCoordinator(contributors=[
                TenantPromptContributor(self.db),
            ])

            # Create orchestrator
            orchestrator = AgentOrchestrator(
                settings=self.base_settings,
                provider=self.base_provider,
                model="claude-sonnet-4-5",
                _config_coordinator=config_coordinator,
                _prompt_coordinator=prompt_coordinator,
            )

            self.orchestrators[tenant_id] = orchestrator

        return self.orchestrators[tenant_id]

# Usage
multi_tenant = MultiTenantOrchestrator(db, settings, provider)

tenant_orchestrator = await multi_tenant.get_orchestrator("tenant-abc")
response = await tenant_orchestrator.chat("Hello!", session_id="tenant-abc:session-123")
```

### Production Considerations

- **Isolation**: Ensure tenant data is isolated at database level
- **Resource limits**: Implement per-tenant resource limits
- **Caching**: Cache tenant configurations
- **Fallbacks**: Provide default configuration for tenants without custom config
- **Monitoring**: Monitor per-tenant usage and performance

---

## Recipe 7: Real-Time Analytics Dashboard

### Problem Statement

You want to display real-time analytics (tokens, tool usage, costs) in a dashboard.

### Solution Overview

Create a WebSocket-based analytics exporter that pushes events to connected clients.

### Step-by-Step Instructions

#### Step 1: Create WebSocket Analytics Exporter

```python
# websocket_analytics.py
from typing import Set, List
from victor.protocols import IAnalyticsExporter, ExportResult, AnalyticsEvent
import websockets
import json

class WebSocketAnalyticsExporter(BaseAnalyticsExporter):
    """Export analytics to WebSocket clients."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    async def export(self, events: List[AnalyticsEvent]) -> ExportResult:
        """Broadcast events to all connected clients."""
        if not self.clients or not events:
            return ExportResult(success=True, exported_count=0)

        message = json.dumps([e.model_dump() for e in events])

        # Broadcast to all clients
        for client in list(self.clients):  # Copy to avoid modification during iteration
            try:
                await client.send(message)
            except Exception as e:
                print(f"Error sending to client: {e}")
                self.clients.discard(client)

        return ExportResult(success=True, exported_count=len(events))

    async def start_server(self):
        """Start WebSocket server."""
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port
        ):
            print(f"WebSocket analytics server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def _handle_client(self, websocket, path):
        """Handle new WebSocket client connection."""
        print(f"New client connected: {websocket.remote_address}")
        self.clients.add(websocket)

        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print(f"Client disconnected: {websocket.remote_address}")
```

#### Step 2: Create Dashboard Client

```html
<!-- dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Victor Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Real-Time Analytics</h1>
    <canvas id="tokenChart"></canvas>
    <canvas id="toolChart"></canvas>

    <script>
        const ws = new WebSocket('ws://localhost:8765');

        const tokenData = { labels: [], datasets: [{ label: 'Tokens', data: [] }] };
        const toolData = { labels: [], datasets: [{ label: 'Tool Calls', data: [] }] };

        const tokenChart = new Chart(document.getElementById('tokenChart'), {
            type: 'line',
            data: tokenData,
        });

        const toolChart = new Chart(document.getElementById('toolChart'), {
            type: 'bar',
            data: toolData,
        });

        ws.onmessage = (event) => {
            const analyticsEvents = JSON.parse(event.data);

            analyticsEvents.forEach(e => {
                if (e.type === 'token_usage') {
                    // Update token chart
                    tokenData.labels.push(new Date(e.timestamp).toLocaleTimeString());
                    tokenData.datasets[0].data.push(e.data.total_tokens);
                    tokenChart.update();
                } else if (e.type === 'tool_call') {
                    // Update tool chart
                    const tool = e.data.tool;
                    const existingIndex = toolData.labels.indexOf(tool);

                    if (existingIndex >= 0) {
                        toolData.datasets[0].data[existingIndex]++;
                    } else {
                        toolData.labels.push(tool);
                        toolData.datasets[0].data.push(1);
                    }
                    toolChart.update();
                }
            });
        };
    </script>
</body>
</html>
```

#### Step 3: Integrate with Orchestrator

```python
# main.py
import asyncio
from victor.agent.coordinators import AnalyticsCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create WebSocket exporter
ws_exporter = WebSocketAnalyticsExporter(host="0.0.0.0", port=8765)

# Create analytics coordinator
analytics_coordinator = AnalyticsCoordinator(exporters=[
    ws_exporter,
])

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(enable_analytics=True),
    provider=provider,
    model=model,
    _analytics_coordinator=analytics_coordinator,
)

# Start WebSocket server in background
async def run_server():
    await ws_exporter.start_server()

asyncio.create_task(run_server())

# Use orchestrator (analytics are broadcast to dashboard)
await orchestrator.chat("Hello!")
```

### Production Considerations

- **Authentication**: Add authentication to WebSocket connections
- **Rate limiting**: Limit client message frequency
- **Scalability**: Use Redis Pub/Sub for multiple dashboard instances
- **Data retention**: Aggregate data to reduce chart data points
- **Error handling**: Handle client disconnections gracefully

---

## Recipe 8: Custom Tool Selection Strategy

### Problem Statement

You want to customize how tools are selected based on your specific use case.

### Solution Overview

Create a custom `IToolSelectionStrategy` that implements your selection logic.

### Step-by-Step Instructions

#### Step 1: Create Custom Selection Strategy

```python
# tool_selection.py
from typing import List
from victor.protocols import IToolSelectionStrategy, Tool
from victor.agent.coordinators.tool_selection_coordinator import IToolSelectionCoordinator

class CostAwareToolSelectionStrategy(IToolSelectionStrategy):
    """Select tools based on cost constraints."""

    def __init__(self, max_cost_per_call: float = 0.01):
        self.max_cost_per_call = max_cost_per_call

    async def select_tools(
        self,
        query: str,
        available_tools: List[Tool],
        selection_coordinator: IToolSelectionCoordinator
    ) -> List[Tool]:
        """Select tools within cost budget."""
        # Filter tools by cost
        affordable_tools = [
            tool for tool in available_tools
            if tool.cost_tier in ["free", "low"]
        ]

        # Use semantic selection on affordable tools
        return await selection_coordinator._semantic_selection(
            query,
            affordable_tools
        )

class DomainSpecificToolSelectionStrategy(IToolSelectionStrategy):
    """Select tools based on domain classification."""

    def __init__(self, domain_classifier):
        self.domain_classifier = domain_classifier

        self.domain_tool_mapping = {
            "coding": ["read", "write", "search", "bash"],
            "data_analysis": ["pandas_query", "visualization", "statistics"],
            "web": ["web_search", "web_scrape", "browse"],
        }

    async def select_tools(
        self,
        query: str,
        available_tools: List[Tool],
        selection_coordinator: IToolSelectionCoordinator
    ) -> List[Tool]:
        """Select tools based on domain."""
        # Classify query domain
        domain = await self.domain_classifier.classify(query)

        # Get allowed tools for domain
        allowed_tool_names = self.domain_tool_mapping.get(domain, [])

        # Filter available tools
        domain_tools = [
            tool for tool in available_tools
            if tool.name in allowed_tool_names
        ]

        return domain_tools
```

#### Step 2: Register Strategy

```python
# main.py
from victor.agent.coordinators import ToolSelectionCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create custom strategy
strategy = CostAwareToolSelectionStrategy(max_cost_per_call=0.01)

# Create tool selection coordinator
tool_selection_coordinator = ToolSelectionCoordinator(
    default_strategy=strategy
)

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(),
    provider=provider,
    model=model,
    _tool_selection_coordinator=tool_selection_coordinator
)
```

### Production Considerations

- **Caching**: Cache classification results
- **Fallback**: Provide fallback to default strategy
- **Monitoring**: Track strategy performance
- **A/B testing**: Test multiple strategies

---

## Recipe 9: A/B Testing Coordinator

### Problem Statement

You want to A/B test different configurations or prompts.

### Solution Overview

Create a coordinator that randomly assigns sessions to different configurations.

### Step-by-Step Instructions

```python
# ab_testing_coordinator.py
import random
from typing import Dict
from victor.agent.coordinators import ConfigCoordinator

class ABTestingConfigCoordinator(ConfigCoordinator):
    """A/B test different configurations."""

    def __init__(self, variants: Dict[str, dict], weights: Dict[str, float] = None):
        """
        Args:
            variants: Dict of variant_name -> config
            weights: Dict of variant_name -> weight (for weighted sampling)
        """
        self.variants = variants
        self.weights = weights or {v: 1.0 for v in variants}
        self.session_assignments = {}

    async def load_config(self, session_id: str, config_override: dict = None) -> dict:
        """Load config for session (assigning to A/B variant)."""
        # Assign session to variant
        if session_id not in self.session_assignments:
            variant = self._assign_variant(session_id)
        else:
            variant = self.session_assignments[session_id]

        # Load variant config
        variant_config = self.variants[variant].copy()

        # Apply overrides
        if config_override:
            variant_config.update(config_override)

        # Track assignment in analytics
        await self._track_assignment(session_id, variant)

        return variant_config

    def _assign_variant(self, session_id: str) -> str:
        """Assign session to variant."""
        # Weighted random selection
        variants = list(self.variants.keys())
        weights = [self.weights.get(v, 1.0) for v in variants]

        variant = random.choices(variants, weights=weights, k=1)[0]
        self.session_assignments[session_id] = variant

        return variant

    async def _track_assignment(self, session_id: str, variant: str):
        """Track A/B assignment in analytics."""
        # Emit analytics event
        pass
```

### Usage

```python
# main.py
coordinator = ABTestingConfigCoordinator(
    variants={
        "control": {"temperature": 0.7, "model": "claude-sonnet-4-5"},
        "variant_a": {"temperature": 0.5, "model": "claude-sonnet-4-5"},
        "variant_b": {"temperature": 0.9, "model": "claude-opus-4-5"},
    },
    weights={
        "control": 0.5,
        "variant_a": 0.25,
        "variant_b": 0.25,
    }
)

orchestrator = AgentOrchestrator(
    settings=Settings(),
    provider=provider,
    model="claude-sonnet-4-5",
    _config_coordinator=coordinator
)

# Sessions are automatically assigned to variants
response = await orchestrator.chat("Hello!", session_id="session-123")
```

---

## Summary

This recipes guide provided 9 practical solutions:

1. **File-based configuration** - Load config from YAML/JSON files
2. **Project-specific prompts** - Add coding standards, compliance
3. **Database analytics** - Export analytics to PostgreSQL
4. **Smart context compaction** - Semantic compaction strategies
5. **Custom middleware** - Add logging, auth, rate limiting
6. **Multi-tenant configuration** - Serve multiple tenants
7. **Real-time dashboard** - WebSocket-based analytics
8. **Custom tool selection** - Cost-aware, domain-specific
9. **A/B testing** - Test different configurations

### Next Steps

- [Quick Start Guide](coordinator_quickstart.md) - Get started
- [Usage Examples](../examples/coordinator_examples.md) - More examples
- [Migration Guide](../migration/orchestrator_refactoring_guide.md) - Migrate from legacy

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Next Review**: 2025-04-13

---

**End of Recipes**
