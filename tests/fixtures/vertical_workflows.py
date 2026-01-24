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

"""Fixtures and mock data for cross-vertical workflow tests.

This module provides:
- Sample cross-vertical workflow scenarios
- Mock data for multi-vertical operations
- Test scenario definitions
- Expected results for validation
"""

from typing import Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum


class VerticalType(str, Enum):
    """Available vertical types for testing."""

    CODING = "coding"
    RESEARCH = "research"
    DEVOPS = "devops"
    RAG = "rag"
    DATA_ANALYSIS = "dataanalysis"
    BENCHMARK = "benchmark"


@dataclass
class CrossVerticalScenario:
    """Definition of a cross-vertical workflow scenario."""

    name: str
    description: str
    verticals: List[VerticalType]
    input_data: Dict[str, Any]
    expected_tools: List[str]
    expected_outcome: str
    workflow_steps: List[Dict[str, Any]] = field(default_factory=list)
    performance_baseline_ms: float = 1000.0  # Expected execution time


@dataclass
class VerticalIsolationTestCase:
    """Test case for vertical isolation."""

    vertical: VerticalType
    test_operation: str
    dependencies: List[str]  # Expected dependencies
    forbidden_dependencies: List[str] = field(default_factory=list)


# ============================================================================
# Cross-Vertical Workflow Scenarios
# ============================================================================

CROSS_VERTICAL_SCENARIOS: Dict[str, CrossVerticalScenario] = {
    "coding_research_analysis": CrossVerticalScenario(
        name="Code Analysis with Web Research",
        description="Analyze code quality and research best practices online",
        verticals=[VerticalType.CODING, VerticalType.RESEARCH],
        input_data={
            "code_snippet": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
""",
            "query": "Analyze this Fibonacci implementation and research best practices",
        },
        expected_tools=[
            "read_file",
            "search_codebase",
            "web_search",
            "analyze_code",
            "code_review",
        ],
        expected_outcome="Code analysis with researched best practices and recommendations",
        workflow_steps=[
            {"vertical": "coding", "action": "analyze_code", "tools": ["analyze_code"]},
            {
                "vertical": "research",
                "action": "search_best_practices",
                "tools": ["web_search"],
            },
            {
                "vertical": "coding",
                "action": "generate_report",
                "tools": ["code_review"],
            },
        ],
        performance_baseline_ms=1500.0,
    ),
    "devops_coding_deployment": CrossVerticalScenario(
        name="Deployment with Code Review",
        description="Review code and prepare deployment configuration",
        verticals=[VerticalType.DEVOPS, VerticalType.CODING],
        input_data={
            "repository": "https://github.com/example/app",
            "branch": "main",
            "query": "Prepare deployment for this application",
        },
        expected_tools=[
            "read_file",
            "code_review",
            "docker_build",
            "generate_kubernetes_manifests",
            "terraform_generate",
        ],
        expected_outcome="Deployment configurations (Docker, Kubernetes, Terraform)",
        workflow_steps=[
            {"vertical": "coding", "action": "review_code", "tools": ["code_review"]},
            {
                "vertical": "devops",
                "action": "generate_dockerfile",
                "tools": ["docker_build"],
            },
            {
                "vertical": "devops",
                "action": "generate_k8s",
                "tools": ["generate_kubernetes_manifests"],
            },
            {
                "vertical": "devops",
                "action": "generate_terraform",
                "tools": ["terraform_generate"],
            },
        ],
        performance_baseline_ms=2000.0,
    ),
    "rag_coding_documentation": CrossVerticalScenario(
        name="Documentation Generation from Code",
        description="Generate comprehensive documentation from codebase",
        verticals=[VerticalType.RAG, VerticalType.CODING],
        input_data={
            "codebase_path": "/path/to/project",
            "query": "Generate API documentation for this codebase",
        },
        expected_tools=[
            "read_file",
            "search_codebase",
            "rag_ingest",
            "rag_search",
            "rag_query",
            "generate_docs",
        ],
        expected_outcome="Comprehensive API documentation with examples",
        workflow_steps=[
            {"vertical": "coding", "action": "scan_codebase", "tools": ["search_codebase"]},
            {"vertical": "rag", "action": "ingest_docs", "tools": ["rag_ingest"]},
            {
                "vertical": "rag",
                "action": "query_patterns",
                "tools": ["rag_query"],
            },
            {
                "vertical": "coding",
                "action": "generate_documentation",
                "tools": ["generate_docs"],
            },
        ],
        performance_baseline_ms=1800.0,
    ),
    "dataanalysis_research_insights": CrossVerticalScenario(
        name="Data-Backed Research Insights",
        description="Analyze data and research context to generate insights",
        verticals=[VerticalType.DATA_ANALYSIS, VerticalType.RESEARCH],
        input_data={
            "data_file": "sales_data.csv",
            "query": "Analyze sales trends and research market factors",
        },
        expected_tools=[
            "read_csv",
            "analyze_statistics",
            "web_search",
            "correlation_analysis",
            "generate_insights",
        ],
        expected_outcome="Statistical analysis with researched market context",
        workflow_steps=[
            {
                "vertical": "dataanalysis",
                "action": "load_data",
                "tools": ["read_csv"],
            },
            {
                "vertical": "dataanalysis",
                "action": "analyze_statistics",
                "tools": ["analyze_statistics"],
            },
            {
                "vertical": "research",
                "action": "research_market",
                "tools": ["web_search"],
            },
            {
                "vertical": "dataanalysis",
                "action": "correlate",
                "tools": ["correlation_analysis"],
            },
            {
                "vertical": "dataanalysis",
                "action": "generate_report",
                "tools": ["generate_insights"],
            },
        ],
        performance_baseline_ms=2200.0,
    ),
    "multi_vertical_debugging": CrossVerticalScenario(
        name="Multi-Vertical Debugging Workflow",
        description="Debug issue using code analysis, log research, and web search",
        verticals=[
            VerticalType.CODING,
            VerticalType.RESEARCH,
            VerticalType.DEVOPS,
        ],
        input_data={
            "error_log": "Error: Connection timeout after 30s",
            "query": "Debug this connection timeout issue",
        },
        expected_tools=[
            "read_file",
            "search_codebase",
            "web_search",
            "analyze_logs",
            "debug_code",
            "test_fix",
        ],
        expected_outcome="Root cause analysis and fix for connection timeout",
        workflow_steps=[
            {"vertical": "coding", "action": "analyze_error", "tools": ["analyze_code"]},
            {"vertical": "devops", "action": "check_logs", "tools": ["analyze_logs"]},
            {
                "vertical": "research",
                "action": "search_similar_issues",
                "tools": ["web_search"],
            },
            {
                "vertical": "coding",
                "action": "implement_fix",
                "tools": ["debug_code"],
            },
            {
                "vertical": "coding",
                "action": "verify_fix",
                "tools": ["test_fix"],
            },
        ],
        performance_baseline_ms=2500.0,
    ),
}


# ============================================================================
# Vertical Isolation Test Cases
# ============================================================================

VERTICAL_ISOLATION_TESTS: List[VerticalIsolationTestCase] = [
    VerticalIsolationTestCase(
        vertical=VerticalType.CODING,
        test_operation="Analyze Python code",
        dependencies=["ast_parser", "code_reader"],
        forbidden_dependencies=["docker_client", "kubernetes_api"],
    ),
    VerticalIsolationTestCase(
        vertical=VerticalType.RESEARCH,
        test_operation="Search web for information",
        dependencies=["web_search", "content_extractor"],
        forbidden_dependencies=["code_executor", "docker_client"],
    ),
    VerticalIsolationTestCase(
        vertical=VerticalType.DEVOPS,
        test_operation="Generate Kubernetes manifests",
        dependencies=["kubernetes_api", "yaml_generator"],
        forbidden_dependencies=["ast_parser", "code_executor"],
    ),
    VerticalIsolationTestCase(
        vertical=VerticalType.RAG,
        test_operation="Query document store",
        dependencies=["vector_store", "embedding_service"],
        forbidden_dependencies=["web_search", "code_executor"],
    ),
    VerticalIsolationTestCase(
        vertical=VerticalType.DATA_ANALYSIS,
        test_operation="Analyze CSV data",
        dependencies=["pandas", "statistics"],
        forbidden_dependencies=["docker_client", "kubernetes_api"],
    ),
]


# ============================================================================
# Mock Data for Testing
# ============================================================================

MOCK_CODE_SNIPPETS = {
    "python_fibonacci": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(fibonacci(i))
""",
    "python_api": """
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # In real implementation, fetch from database
    if user_id < 1:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user_id, "name": "Test User"}
""",
    "dockerfile": """
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
""",
}

MOCK_RESEARCH_RESULTS = {
    "fibonacci_best_practices": {
        "title": "Python Fibonacci Best Practices",
        "url": "https://example.com/fibonacci",
        "snippet": "Use memoization or iteration for better performance",
        "sources": ["GeeksforGeeks", "Stack Overflow", "Python Docs"],
    },
    "api_design_patterns": {
        "title": "REST API Design Patterns",
        "url": "https://example.com/api-design",
        "snippet": "Use proper HTTP status codes and error handling",
        "sources": ["REST API Tutorial", "MDN Web Docs"],
    },
}

MOCK_DATA_FILES = {
    "sales_data.csv": """date,product,sales,region
2024-01-01,Product A,1000,North
2024-01-02,Product B,1500,South
2024-01-03,Product A,1200,East
2024-01-04,Product C,900,West
2024-01-05,Product B,1800,North
""",
    "user_metrics.csv": """user_id,page_views,session_duration,bounce_rate
1,45,300,0.25
2,78,450,0.15
3,32,180,0.40
4,95,600,0.10
5,56,380,0.20
""",
}

MOCK_DEPLOYMENT_CONFIGS = {
    "kubernetes_deployment": """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: example-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: example
  template:
    metadata:
      labels:
        app: example
    spec:
      containers:
      - name: app
        image: example/app:latest
        ports:
        - containerPort: 8000
""",
    "terraform_config": """
resource "aws_instance" "app_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "ExampleAppServer"
  }
}
""",
}


# ============================================================================
# Helper Functions
# ============================================================================


def get_scenario(name: str) -> CrossVerticalScenario:
    """Get a cross-vertical scenario by name.

    Args:
        name: Scenario name

    Returns:
        CrossVerticalScenario object

    Raises:
        KeyError: If scenario not found
    """
    return CROSS_VERTICAL_SCENARIOS[name]


def list_all_scenarios() -> List[str]:
    """List all available cross-vertical scenarios.

    Returns:
        List of scenario names
    """
    return list(CROSS_VERTICAL_SCENARIOS.keys())


def get_vertical_combinations() -> List[tuple]:
    """Get all possible 2-vertical combinations for testing.

    Returns:
        List of (vertical1, vertical2) tuples
    """
    verticals = list(VerticalType)
    combinations = []
    for i, v1 in enumerate(verticals):
        for v2 in verticals[i + 1 :]:
            combinations.append((v1, v2))
    return combinations


def get_mock_code_snippet(name: str) -> str:
    """Get mock code snippet by name.

    Args:
        name: Snippet name

    Returns:
        Code string
    """
    return MOCK_CODE_SNIPPETS.get(name, "")


def get_mock_research_result(topic: str) -> Dict[str, Any]:
    """Get mock research result by topic.

    Args:
        topic: Research topic

    Returns:
        Research result dictionary
    """
    key = f"{topic}_best_practices" if "_best_practices" not in topic else topic
    return MOCK_RESEARCH_RESULTS.get(key, {})


def create_mock_workflow_result(
    scenario: CrossVerticalScenario, success: bool = True
) -> Dict[str, Any]:
    """Create a mock workflow execution result.

    Args:
        scenario: The scenario being executed
        success: Whether the workflow succeeded

    Returns:
        Mock result dictionary
    """
    return {
        "scenario": scenario.name,
        "success": success,
        "verticals_used": [v.value for v in scenario.verticals],
        "tools_used": scenario.expected_tools,
        "outcome": scenario.expected_outcome if success else "Execution failed",
        "execution_time_ms": scenario.performance_baseline_ms * 0.9,
        "steps_completed": len(scenario.workflow_steps) if success else 0,
    }
