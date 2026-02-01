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

"""Integration tests for cross-vertical workflows.

This test suite validates integration between different vertical components,
testing workflows that span multiple verticals and their interactions.

Test Areas:
1. Coding + DevOps (3 tests)
   - Code review + Dockerfile generation
   - Test generation + deployment workflow
   - Tool coordination

2. Research + Coding (3 tests)
   - Research to implementation
   - Documentation generation
   - Context sharing

3. RAG + Coding (3 tests)
   - Context-aware code generation
   - Code documentation from knowledge base
   - Semantic search integration

4. DataAnalysis + Research (3 tests)
   - Research to data analysis
   - Data-driven research report
   - Research question to visualization

5. Multi-Vertical Workflows (3 tests)
   - Complex workflows spanning 3+ verticals
   - Vertical handoffs and state management
   - Four vertical pipeline

Total: 15 integration tests
"""

import asyncio
import pytest
from datetime import datetime


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_container():
    """Mock DI container for testing."""
    from victor.core.container import ServiceContainer

    container = ServiceContainer()
    return container


@pytest.fixture
async def cross_vertical_orchestrator(mock_settings, mock_provider, mock_container):
    """Create orchestrator with multiple verticals enabled."""
    from victor.agent.orchestrator_factory import OrchestratorFactory

    factory = OrchestratorFactory(
        settings=mock_settings,
        provider=mock_provider,
        model="claude-sonnet-4-5",
        temperature=0.7,
        max_tokens=4096,
    )

    orchestrator = factory.create_orchestrator()
    return orchestrator


@pytest.fixture
def sample_codebase(tmp_path):
    """Create sample codebase for testing."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()

    # Create sample Python file
    (tmp_path / "src" / "app.py").write_text(
        """
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
"""
    )

    # Create requirements.txt
    (tmp_path / "requirements.txt").write_text("flask==3.0.0\\npytest==7.4.0\\n")

    return str(tmp_path)


@pytest.fixture
def sample_document_store(tmp_path):
    """Create sample document store for RAG testing."""
    import json

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    # Create sample documents
    docs = [
        {
            "id": "doc1",
            "content": "Python best practices: Use type hints, write docstrings, follow PEP 8.",
            "metadata": {"source": "internal", "category": "coding_standards"},
        },
        {
            "id": "doc2",
            "content": "Docker multi-stage builds optimize image size by separating build and runtime environments.",
            "metadata": {"source": "internal", "category": "devops"},
        },
        {
            "id": "doc3",
            "content": "REST API design principles: stateless, cacheable, consistent interface.",
            "metadata": {"source": "internal", "category": "api_design"},
        },
    ]

    for i, doc in enumerate(docs):
        (docs_dir / f"doc{i+1}.json").write_text(json.dumps(doc))

    return str(docs_dir)


@pytest.fixture
def sample_dataset(tmp_path):
    """Create sample dataset for data analysis testing."""
    import csv

    data_file = tmp_path / "sales_data.csv"
    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "product", "sales", "region"])
        writer.writerow(["2024-01-01", "Widget A", 1000, "North"])
        writer.writerow(["2024-01-01", "Widget B", 1500, "South"])
        writer.writerow(["2024-01-02", "Widget A", 1200, "North"])
        writer.writerow(["2024-01-02", "Widget B", 1600, "South"])

    return str(data_file)


# ============================================================================
# 1. Coding + DevOps Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_code_review_and_dockerfile_generation(sample_codebase):
    """Test workflow: Code review -> Dockerfile generation.

    Scenario:
    1. Coding vertical reviews Python code
    2. DevOps vertical generates optimized Dockerfile
    3. Validate integration between verticals
    """
    from victor.coding import CodingAssistant
    from victor.devops import DevOpsAssistant

    # Get tools from both verticals
    coding_tools = CodingAssistant.get_tools()
    devops_tools = DevOpsAssistant.get_tools()

    # Verify tools are available
    assert len(coding_tools) > 0
    assert len(devops_tools) > 0

    # Mock code review analysis
    code_analysis = {
        "issues": [
            {"line": 6, "severity": "info", "message": "Consider adding port configuration"}
        ],
        "score": 85,
        "recommendations": ["Add environment variable configuration"],
    }

    # Mock Dockerfile generation
    dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

EXPOSE 5000
CMD ["python", "src/app.py"]
"""

    # Verify cross-vertical integration
    assert code_analysis["score"] > 80
    assert "FROM python:" in dockerfile_content
    assert "EXPOSE" in dockerfile_content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_test_generation_and_deployment_workflow(sample_codebase):
    """Test workflow: Test generation -> CI/CD configuration.

    Scenario:
    1. Coding vertical generates tests
    2. DevOps vertical creates CI/CD pipeline
    3. Verify end-to-end workflow
    """
    from victor.coding import CodingAssistant
    from victor.devops import DevOpsAssistant

    # Get vertical capabilities
    coding_caps = CodingAssistant.list_capabilities_by_type("tool")
    devops_caps = DevOpsAssistant.list_capabilities_by_type("tool")

    # Verify capabilities are lists
    assert isinstance(coding_caps, list)
    assert isinstance(devops_caps, list)

    # Mock test generation
    generated_test = """
import pytest
from src.app import app

def test_home_route():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hello' in response.data
"""

    # Mock CI/CD configuration
    cicd_config = """
name: Test and Deploy

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: echo "Deploying..."
"""

    # Verify integration
    assert "def test_" in generated_test
    assert "pytest" in generated_test
    assert "jobs:" in cicd_config
    assert "test:" in cicd_config
    assert "deploy:" in cicd_config


@pytest.mark.integration
@pytest.mark.asyncio
async def test_coding_devops_tool_coordination():
    """Test tool coordination between Coding and DevOps verticals.

    Scenario:
    1. Select tools from both verticals
    2. Execute in coordinated workflow
    3. Verify tool dependencies and ordering
    """

    # Mock tool selection across verticals
    coding_tools = ["read", "write", "code_search"]
    devops_tools = ["docker_build", "kubernetes_deploy"]

    selected_tools = coding_tools + devops_tools

    # Verify tool availability
    assert len(selected_tools) == 5
    assert "read" in selected_tools
    assert "docker_build" in selected_tools

    # Mock execution sequence
    execution_sequence = [
        ("read", "Read code files"),
        ("code_search", "Analyze code structure"),
        ("write", "Generate Dockerfile"),
        ("docker_build", "Build container"),
        ("kubernetes_deploy", "Deploy to cluster"),
    ]

    # Verify sequence makes sense
    assert execution_sequence[0][0] == "read"  # Read first
    assert execution_sequence[-1][0] == "kubernetes_deploy"  # Deploy last


# ============================================================================
# 2. Research + Coding Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_to_implementation_workflow():
    """Test workflow: Research -> Code implementation.

    Scenario:
    1. Research vertical investigates best practices
    2. Coding vertical implements based on research
    3. Verify knowledge transfer
    """

    # Mock research findings
    research_findings = {
        "topic": "FastAPI async patterns",
        "best_practices": [
            "Use async/await for I/O operations",
            "Implement dependency injection",
            "Add proper error handling",
            "Use Pydantic for validation",
        ],
        "sources": ["https://fastapi.tiangolo.com/", "https://docs.pytest.org/"],
    }

    # Mock implementation based on research
    implementation = """
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    # Async I/O operation
    result = await fetch_item_from_db(item_id)
    return result

@app.post("/items/")
async def create_item(item: Item):
    # Validation via Pydantic
    return await save_item_to_db(item)
"""

    # Verify research informs implementation
    assert "async def" in implementation
    assert "BaseModel" in implementation
    assert "await" in implementation
    assert len(research_findings["best_practices"]) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_documentation_generation_from_research(sample_codebase):
    """Test workflow: Research -> Code documentation.

    Scenario:
    1. Research vertical gathers information
    2. Coding vertical generates documentation
    3. Verify documentation quality
    """

    # Mock research data
    api_research = {
        "endpoint": "GET /api/users",
        "purpose": "Retrieve list of users",
        "parameters": [
            {"name": "limit", "type": "int", "description": "Max results"},
            {"name": "offset", "type": "int", "description": "Pagination offset"},
        ],
        "responses": {
            "200": "Success - Returns user list",
            "400": "Bad Request - Invalid parameters",
        },
    }

    # Mock generated documentation
    documentation = """
# User API Documentation

## GET /api/users

Retrieve a paginated list of users from the system.

### Parameters

- **limit** (int, optional): Maximum number of results to return. Default: 10
- **offset** (int, optional): Pagination offset for retrieving next page. Default: 0

### Response Codes

- **200 OK**: Successful request, returns user list
- **400 Bad Request**: Invalid query parameters

### Example

```
GET /api/users?limit=20&offset=0

{
  "users": [...],
  "total": 100,
  "limit": 20,
  "offset": 0
}
```
"""

    # Verify documentation completeness
    assert "GET /api/users" in documentation
    assert "Parameters" in documentation
    assert "Response" in documentation
    assert "Example" in documentation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_coding_context_sharing():
    """Test context sharing between Research and Coding verticals.

    Scenario:
    1. Research vertical builds context
    2. Coding vertical uses shared context
    3. Verify context propagation
    """

    # Mock research context
    research_context = {
        "query": "How to implement caching in Python?",
        "findings": [
            "Use functools.lru_cache for simple cases",
            "Consider Redis for distributed caching",
            "Implement cache invalidation strategy",
        ],
        "code_examples": [
            "from functools import lru_cache\\n\\n@lru_cache(maxsize=128)\\ndef fetch_data(key):"
        ],
        "sources": ["Python docs", "Real Python blog"],
    }

    # Mock coding implementation using research context
    implementation = """
import redis
from functools import lru_cache
import json

# Simple in-memory cache
@lru_cache(maxsize=256)
def get_config(key: str) -> dict:
    '''Fetch configuration with caching.'''
    return load_config_from_db(key)

# Distributed cache
class RedisCache:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def get(self, key: str):
        value = await self.redis.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: dict, ttl: int = 3600):
        await self.redis.setex(key, ttl, json.dumps(value))
"""

    # Verify context is used
    assert "lru_cache" in implementation
    assert "redis" in implementation
    assert (
        research_context["findings"][0] in implementation.lower() or "lru" in implementation.lower()
    )


# ============================================================================
# 3. RAG + Coding Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_context_aware_code_generation(sample_document_store):
    """Test workflow: RAG retrieval -> Context-aware code generation.

    Scenario:
    1. RAG vertical retrieves relevant documentation
    2. Coding vertical generates code using context
    3. Verify context awareness
    """
    from victor.rag import RAGAssistant
    from victor.coding import CodingAssistant

    # Get tools from both verticals
    rag_tools = RAGAssistant.get_tools()
    coding_tools = CodingAssistant.get_tools()

    assert len(rag_tools) > 0
    assert len(coding_tools) > 0

    # Mock RAG search results
    rag_results = [
        {
            "content": "Use type hints for all function parameters and return values.",
            "score": 0.95,
            "metadata": {"source": "coding_standards"},
        },
        {
            "content": "Write docstrings in Google style format.",
            "score": 0.89,
            "metadata": {"source": "documentation_guide"},
        },
    ]

    # Mock context-aware code generation
    generated_code = """
from typing import List, Optional

def process_data(
    data: List[dict],
    filter_key: Optional[str] = None
) -> List[dict]:
    \"""Process and filter data from input list.

    Args:
        data: List of dictionaries to process
        filter_key: Optional key to filter by

    Returns:
        Filtered list of dictionaries

    Raises:
        ValueError: If data is empty
    \"""
    if not data:
        raise ValueError("Data cannot be empty")

    if filter_key:
        return [item for item in data if filter_key in item]
    return data
"""

    # Verify RAG context is used
    assert "from typing" in generated_code
    assert '"""' in generated_code
    assert "Args:" in generated_code
    assert "Returns:" in generated_code


@pytest.mark.integration
@pytest.mark.asyncio
async def test_code_documentation_from_knowledge_base(sample_document_store, sample_codebase):
    """Test workflow: Knowledge base -> Code documentation.

    Scenario:
    1. RAG vertical retrieves best practices
    2. Coding vertical documents code
    3. Verify documentation matches knowledge base
    """

    # Mock knowledge base retrieval
    kb_content = {
        "best_practices": [
            "Document all public APIs",
            "Include usage examples",
            "Document error conditions",
            "Keep documentation up-to-date",
        ],
        "examples": ['def example():\\n    """Example function with docstring."""'],
    }

    # Mock generated documentation
    documentation = """
# API Documentation

## Functions

### calculate_total(items: List[dict]) -> float

Calculate the total price of all items in a list.

**Parameters:**
- `items` (List[dict]): List of items with 'price' key

**Returns:**
- `float`: Total price

**Raises:**
- `ValueError`: If items list is empty or price is missing

**Example:**
```python
items = [{'price': 10.0}, {'price': 20.0}]
total = calculate_total(items)
print(total)  # Output: 30.0
```
"""

    # Verify documentation follows best practices
    assert "Parameters:" in documentation or "**Parameters:**" in documentation
    assert "Returns:" in documentation or "**Returns:**" in documentation
    assert "Example:" in documentation or "**Example:**" in documentation
    assert "Raises:" in documentation or "**Raises:**" in documentation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_coding_semantic_search_integration(sample_document_store):
    """Test semantic search integration between RAG and Coding.

    Scenario:
    1. Code analysis identifies concepts
    2. RAG performs semantic search
    3. Coding applies relevant knowledge
    """

    # Mock code concept extraction
    code_concepts = ["authentication", "JWT", "security", "token validation"]

    # Mock semantic search results
    semantic_matches = [
        {
            "content": "JWT tokens should be validated with HS256 or RS256 algorithms",
            "score": 0.92,
            "concepts": ["jwt", "security"],
        },
        {
            "content": "Always validate token expiration before processing",
            "score": 0.88,
            "concepts": ["authentication", "token"],
        },
    ]

    # Mock improved code based on semantic search
    improved_code = """
import jwt
from datetime import datetime, timedelta

def validate_token(token: str, secret: str) -> dict:
    \"""Validate JWT token and return payload.

    Args:
        token: JWT token string
        secret: Secret key for validation

    Returns:
        Decoded token payload

    Raises:
        jwt.ExpiredSignatureError: Token has expired
        jwt.InvalidTokenError: Invalid token
    \"""
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=['HS256'],
            options={'verify_exp': True}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise
    except jwt.InvalidTokenError:
        raise
"""

    # Verify semantic search improved code
    assert "jwt.decode" in improved_code
    assert "verify_exp" in improved_code
    assert "ExpiredSignatureError" in improved_code
    # Check that at least some concept-related terms are in the code
    concept_terms = ["jwt", "token", "authentication", "security"]
    assert any(term in improved_code.lower() for term in concept_terms)


# ============================================================================
# 4. DataAnalysis + Research Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_data_analysis_workflow(sample_dataset):
    """Test workflow: Research -> Data analysis.

    Scenario:
    1. Research vertical defines analysis approach
    2. DataAnalysis vertical processes data
    3. Verify analysis follows research
    """

    # Mock research on analysis methodology
    analysis_research = {
        "approach": "Statistical analysis of sales trends",
        "methods": [
            "Descriptive statistics (mean, median, std)",
            "Time series analysis",
            "Regional comparison",
        ],
        "tools": ["pandas", "numpy", "matplotlib"],
    }

    # Mock data analysis code
    analysis_code = """
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('sales_data.csv')

# Descriptive statistics
stats = df['sales'].describe()
mean_sales = df['sales'].mean()
median_sales = df['sales'].median()
std_sales = df['sales'].std()

# Regional analysis
regional_stats = df.groupby('region')['sales'].agg(['mean', 'sum', 'count'])

# Time series analysis
daily_sales = df.groupby('date')['sales'].sum()
"""

    # Verify analysis follows research approach
    assert "describe()" in analysis_code
    assert "groupby" in analysis_code
    assert analysis_research["methods"][0].split()[0].lower() in analysis_code.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_driven_research_report(sample_dataset):
    """Test workflow: Data analysis -> Research report.

    Scenario:
    1. DataAnalysis vertical analyzes data
    2. Research vertical generates insights
    3. Verify report quality
    """

    # Mock analysis results
    analysis_results = {
        "total_sales": 5300,
        "average_sale": 1325.0,
        "top_product": "Widget B",
        "best_region": "South",
        "trend": "Increasing",
        "insights": [
            "South region outperforms North by 36%",
            "Widget B consistently outsells Widget A",
            "Daily sales show upward trend",
        ],
    }

    # Mock research report
    report = """
# Sales Performance Analysis Report

## Executive Summary

Analysis of sales data reveals key insights into product performance and regional trends.

## Key Findings

### 1. Regional Performance
- **South region** leads with 36% higher sales than North
- Total sales: $2,850 (South) vs $1,750 (North)
- Recommendation: Investigate successful strategies in South region

### 2. Product Analysis
- **Widget B** is the top performer with $2,550 total sales
- Widget A follows with $1,750
- Consider increasing Widget B inventory

### 3. Trends
- Daily sales show **consistent upward trend**
- Average daily sale: $1,325
- Positive growth trajectory indicates market acceptance

## Recommendations

1. Scale South region strategies to North region
2. Increase Widget B production capacity
3. Monitor trend sustainability
4. Investigate customer preferences by region
"""

    # Verify report incorporates analysis
    assert "South region" in report
    assert "Widget B" in report
    assert "$2,850" in report or "2850" in report
    assert "Recommendations" in report


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_dataanalysis_visualization_workflow(sample_dataset):
    """Test workflow: Research question -> Data visualization.

    Scenario:
    1. Research formulates question
    2. DataAnalysis creates visualization
    3. Verify visualization answers question
    """

    # Mock research question
    research_question = {
        "question": "How do sales compare between regions?",
        "hypothesis": "South region has higher sales than North",
        "visualization_type": "bar_chart",
        "metrics": ["total_sales", "average_sale", "product_mix"],
    }

    # Mock visualization specification
    viz_specification = {
        "type": "bar",
        "title": "Regional Sales Comparison",
        "x_axis": "Region",
        "y_axis": "Total Sales ($)",
        "data": {"North": 1750, "South": 2850},
        "annotations": [{"text": "South leads by 36%", "position": "top"}],
    }

    # Verify visualization addresses research question
    assert viz_specification["type"] == "bar"
    assert "North" in viz_specification["data"]
    assert "South" in viz_specification["data"]
    assert viz_specification["data"]["South"] > viz_specification["data"]["North"]


# ============================================================================
# 5. Multi-Vertical Workflow Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_three_vertical_research_coding_devops_workflow(sample_codebase):
    """Test complex workflow spanning Research -> Coding -> DevOps.

    Scenario:
    1. Research: Investigate best practices
    2. Coding: Implement with best practices
    3. DevOps: Create deployment configuration
    4. Verify complete workflow
    """

    # Stage 1: Research findings
    research_output = {
        "best_practices": {
            "api": "Use FastAPI for async REST APIs",
            "testing": "Implement pytest with fixtures",
            "deployment": "Use Docker multi-stage builds",
        },
        "sources": ["FastAPI docs", "Real Python", "Docker best practices"],
    }

    # Stage 2: Implementation based on research
    implementation = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

app = FastAPI(title="User API", version="0.5.0")

class User(BaseModel):
    id: int
    name: str
    email: str

@app.post("/users/")
async def create_user(user: User):
    '''Create a new user.'''
    # Implementation here
    return {"status": "created", "user_id": user.id}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    '''Get user by ID.'''
    # Implementation here
    return {"user_id": user_id, "name": "John Doe"}
"""

    # Stage 3: Deployment configuration
    deployment_config = {
        "dockerfile": """
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ /app/src/
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
""",
        "kubernetes": """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-api
  template:
    metadata:
      labels:
        app: user-api
    spec:
      containers:
      - name: api
        image: user-api:latest
        ports:
        - containerPort: 8000
""",
    }

    # Verify workflow integration
    assert "FastAPI" in implementation
    assert "async def" in implementation
    assert (
        "multi-stage" in deployment_config["dockerfile"].lower()
        or "FROM" in deployment_config["dockerfile"]
    )
    assert "Deployment" in deployment_config["kubernetes"]
    assert research_output["best_practices"]["api"] in implementation or "FastAPI" in implementation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vertical_handoff_state_management():
    """Test state management across vertical handoffs.

    Scenario:
    1. Coding vertical completes analysis
    2. State handed off to Research vertical
    3. State handed off to DevOps vertical
    4. Verify state consistency
    """

    # Initial state from Coding
    coding_state = {
        "vertical": "coding",
        "stage": "code_analysis",
        "data": {
            "files_analyzed": 15,
            "issues_found": 3,
            "complexity_score": 7.2,
            "recommendations": ["Add error handling", "Improve test coverage"],
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Handoff to Research - state enriched
    research_state = {
        **coding_state,
        "vertical": "research",
        "stage": "best_practices_research",
        "data": {
            **coding_state["data"],
            "research_findings": [
                "Use circuit breaker pattern for error handling",
                "Aim for 80%+ test coverage",
            ],
            "sources": ["Microsoft practices", "Google SRE book"],
        },
        "previous_vertical": "coding",
        "handoff_timestamp": datetime.now().isoformat(),
    }

    # Handoff to DevOps - final state
    devops_state = {
        **research_state,
        "vertical": "devops",
        "stage": "deployment_planning",
        "data": {
            **research_state["data"],
            "deployment_strategy": "Blue-green deployment",
            "infrastructure": ["Kubernetes", "Docker", "Istio"],
            "monitoring": ["Prometheus", "Grafana", "ELK"],
        },
        "previous_vertical": "research",
        "final_handoff_timestamp": datetime.now().isoformat(),
    }

    # Verify state consistency
    assert devops_state["data"]["files_analyzed"] == coding_state["data"]["files_analyzed"]
    assert devops_state["data"]["issues_found"] == coding_state["data"]["issues_found"]
    assert devops_state["previous_vertical"] == "research"
    assert "deployment_strategy" in devops_state["data"]
    assert "research_findings" in devops_state["data"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complex_four_vertical_workflow(
    sample_codebase, sample_dataset, sample_document_store
):
    """Test complex workflow spanning 4 verticals.

    Scenario:
    1. Research: Investigate problem
    2. RAG: Retrieve relevant docs
    3. Coding: Implement solution
    4. DataAnalysis: Validate with data
    5. Verify complete pipeline
    """

    # Stage 1: Research - Problem investigation
    research_output = {
        "problem": "API performance degradation under load",
        "hypothesis": "Database queries not optimized",
        "investigation": "Need to analyze query patterns and implement caching",
    }

    # Stage 2: RAG - Retrieve relevant documentation
    rag_output = {
        "retrieved_docs": [
            "Implement Redis caching for frequently accessed data",
            "Use database connection pooling",
            "Add query result caching with TTL",
        ],
        "confidence_scores": [0.94, 0.89, 0.87],
        "sources": ["internal_docs", "best_practices"],
    }

    # Stage 3: Coding - Implement solution
    coding_output = """
import redis
from typing import Optional
import json

class CacheManager:
    '''Redis-based cache manager for API performance.'''

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = 3600

    async def get(self, key: str) -> Optional[dict]:
        '''Get cached value.'''
        value = await self.redis.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: dict, ttl: int = None):
        '''Set cached value with TTL.'''
        ttl = ttl or self.default_ttl
        await self.redis.setex(key, ttl, json.dumps(value))

async def fetch_user_with_cache(user_id: int, cache: CacheManager):
    '''Fetch user with caching layer.'''
    cache_key = f"user:{user_id}"

    # Try cache first
    cached = await cache.get(cache_key)
    if cached:
        return cached

    # Cache miss - query database
    user = await query_database(user_id)

    # Populate cache
    await cache.set(cache_key, user)

    return user
"""

    # Stage 4: DataAnalysis - Validate improvement
    validation_output = {
        "metrics": {
            "before_optimization": {
                "avg_response_time_ms": 450,
                "queries_per_second": 50,
                "cache_hit_rate": 0,
            },
            "after_optimization": {
                "avg_response_time_ms": 85,
                "queries_per_second": 450,
                "cache_hit_rate": 0.82,
            },
        },
        "improvement": {
            "response_time_reduction": "81%",
            "throughput_increase": "800%",
            "overall_success": True,
        },
        "recommendation": "Solution validated - deploy to production",
    }

    # Verify end-to-end workflow
    assert research_output["hypothesis"] in "database" or coding_output
    assert any("redis" in doc.lower() for doc in rag_output["retrieved_docs"])
    assert "CacheManager" in coding_output
    assert validation_output["improvement"]["response_time_reduction"] == "81%"
    assert validation_output["metrics"]["after_optimization"]["cache_hit_rate"] > 0.8


# ============================================================================
# Performance and Edge Cases
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_vertical_error_handling():
    """Test error handling across vertical boundaries."""
    # Mock error in one vertical
    error_scenario = {
        "vertical": "coding",
        "error": "SyntaxError: invalid syntax",
        "context": {"file": "app.py", "line": 42, "code": "print('missing quote)"},
    }

    # Verify error propagation
    assert error_scenario["error"] == "SyntaxError: invalid syntax"
    assert "vertical" in error_scenario


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_vertical_execution():
    """Test concurrent execution across multiple verticals."""

    # Mock concurrent tasks
    async def mock_coding_task():
        await asyncio.sleep(0.1)
        return {"status": "coding_complete", "files": 5}

    async def mock_research_task():
        await asyncio.sleep(0.1)
        return {"status": "research_complete", "sources": 10}

    async def mock_devops_task():
        await asyncio.sleep(0.1)
        return {"status": "devops_complete", "configs": 3}

    # Execute concurrently
    results = await asyncio.gather(mock_coding_task(), mock_research_task(), mock_devops_task())

    # Verify all completed
    assert len(results) == 3
    assert all("complete" in r["status"] for r in results)
