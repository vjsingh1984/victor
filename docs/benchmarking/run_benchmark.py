#!/usr/bin/env python3
"""Competitive Benchmark Execution Script for Agentic AI Frameworks.

This script executes the competitive benchmark suite defined in
docs/benchmarking/competitive-benchmark-rubric.md.

Usage:
    # Run all tasks for Victor
    python docs/benchmarking/run_benchmark.py --framework victor

    # Run specific task
    python docs/benchmarking/run_benchmark.py --framework victor --task C1

    # Run with verbose output
    python docs/benchmarking/run_benchmark.py --framework victor --verbose

    # Dry run (show what would be executed)
    python docs/benchmarking/run_benchmark.py --framework victor --dry-run
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Task Definitions
# ============================================================================

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "C1": {
        "name": "Single-file generation",
        "category": "Code Generation",
        "prompt": """Generate a Python class named `DataProcessor` with the following:
- Constructor taking `data_path: str` and `batch_size: int = 100`
- Method `process()` that loads CSV, applies transformations, returns DataFrame
- Method `save()` that saves processed data to CSV
- Type hints and docstrings
- Error handling for missing files

Requirements:
1. Code compiles without syntax errors
2. Class has all required methods
3. Type hints present
4. Error handling included
5. Follows PEP 8 style""",
        "complexity": "Simple",
        "timeout_seconds": 60,
        "max_tokens": 2000,
    },
    "C2": {
        "name": "Multi-file refactoring",
        "category": "Code Generation",
        "prompt": """Refactor the following codebase to use dependency injection:

1. Create an interface for the data service
2. Modify UserService to accept the interface via constructor
3. Update the factory to inject the correct implementation
4. Ensure all tests still pass

Files to modify:
- src/services/user_service.py
- src/services/data_service.py
- src/factories.py

The refactoring should maintain backward compatibility.""",
        "complexity": "Medium",
        "timeout_seconds": 120,
        "max_tokens": 4000,
    },
    "C3": {
        "name": "Bug fix with context",
        "category": "Code Generation",
        "prompt": """Fix the bug in the following authentication module. Do NOT search for files — the code is provided below.

```python
import time
import threading

class SessionManager:
    def __init__(self, token_ttl=300):
        self.token_ttl = token_ttl  # 5 minutes
        self._tokens = {}
        self._lock = threading.Lock()

    def create_session(self, user_id: str) -> str:
        token = f"tok_{user_id}_{int(time.time())}"
        with self._lock:
            self._tokens[token] = {
                "user_id": user_id,
                "created_at": time.time(),
                "expires_at": time.time() + self.token_ttl,
            }
        return token

    def validate_token(self, token: str) -> bool:
        with self._lock:
            session = self._tokens.get(token)
            if not session:
                return False
            if time.time() > session["expires_at"]:
                del self._tokens[token]
                return False
            return True

    def _refresh_token(self, token: str) -> bool:
        # BUG: This method exists but is never called
        with self._lock:
            session = self._tokens.get(token)
            if session:
                session["expires_at"] = time.time() + self.token_ttl
                return True
        return False
```

Bug: Users are being logged out after 5 minutes because `_refresh_token` is never called.

Fix the code ensuring:
1. Tokens are refreshed automatically before expiry (call _refresh_token in validate_token)
2. No changes to the public API
3. Thread safety is maintained
4. Provide the complete fixed code""",
        "complexity": "Medium",
        "timeout_seconds": 120,
        "max_tokens": 3000,
    },
    "C4": {
        "name": "Code review",
        "category": "Code Generation",
        "prompt": """Review the following Python code and provide structured feedback:

```python
def process_data(data: List[Dict]) -> List[Dict]:
    result = []
    for item in data:
        if item.get('active'):
            new_item = {}
            new_item['id'] = item['id']
            new_item['value'] = item.get('value', 0) * 2
            if new_item['value'] > 100:
                new_item['flag'] = True
            result.append(new_item)
    return result
```

Provide feedback on:
1. Code style and PEP 8 compliance
2. Performance considerations
3. Error handling
4. Type safety
5. Suggested improvements""",
        "complexity": "Medium",
        "timeout_seconds": 90,
        "max_tokens": 2000,
    },
    "C5": {
        "name": "Documentation generation",
        "category": "Code Generation",
        "prompt": """Generate comprehensive documentation for this API class:

```python
class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None

    def connect(self) -> bool:
        \"\"\"Establish connection to API.\"\"\"
        pass

    def get_resource(self, resource_id: str) -> Optional[Dict]:
        \"\"\"Fetch a resource by ID.\"\"\"
        pass

    def create_resource(self, data: Dict) -> Optional[str]:
        \"\"\"Create a new resource.\"\"\"
        pass
```

Generate:
1. Module-level docstring
2. Class docstring with usage example
3. Method docstrings with parameters, returns, raises
4. Example usage in a docstring
5. Type hints for all parameters""",
        "complexity": "Simple",
        "timeout_seconds": 60,
        "max_tokens": 2000,
    },
    "R1": {
        "name": "Research synthesis",
        "category": "Multi-Step Reasoning",
        "prompt": """Synthesize findings from the following research sources on microservices architecture:

Source 1: Discusses benefits including scalability, fault isolation, independent deployment
Source 2: Covers challenges: network latency, distributed transactions, debugging complexity
Source 3: Presents best practices: API gateways, service mesh, observability

Provide a comprehensive synthesis covering:
1. Key benefits with real-world examples
2. Major challenges and mitigation strategies
3. When to use microservices vs monolith
4. Recommended technology stack components
5. Common pitfalls to avoid""",
        "complexity": "Complex",
        "timeout_seconds": 180,
        "max_tokens": 4000,
    },
    "R2": {
        "name": "Architecture design",
        "category": "Multi-Step Reasoning",
        "prompt": """Design a system architecture for a real-time collaborative document editing platform with the following requirements:

Functional Requirements:
- Multiple users edit documents simultaneously
- Real-time sync with conflict resolution
- Document history and version control
- Rich text formatting with images
- User comments and suggestions

Non-Functional Requirements:
- Support 10,000 concurrent users
- <100ms latency for edits to propagate
- 99.9% availability
- Data durability guarantees

Provide:
1. System architecture diagram (described in text)
2. Technology choices with justification
3. Data model and persistence strategy
4. Real-time sync mechanism
5. Conflict resolution strategy
6. Scalability approach
7. Security considerations""",
        "complexity": "Complex",
        "timeout_seconds": 300,
        "max_tokens": 5000,
    },
    "T1": {
        "name": "File operations",
        "category": "Tool Usage",
        "prompt": """Perform the following file operations in /tmp/benchmark_t1/:

1. Create a directory structure: /tmp/benchmark_t1/{docs,images,data}
2. Write a sample file /tmp/benchmark_t1/docs/hello.py with a simple Python function
3. Read the file back and verify its contents
4. List the directory structure you created
5. Create a summary file /tmp/benchmark_t1/summary.txt listing all created files

Use file system tools to complete these operations.""",
        "complexity": "Simple",
        "timeout_seconds": 90,
        "max_tokens": 2000,
    },
    "W1": {
        "name": "Sequential workflow",
        "category": "Workflow & Coordination",
        "prompt": """Execute a sequential data processing workflow:

Step 1: Load data from data.csv
Step 2: Validate the data (check for missing values, invalid types)
Step 3: If validation fails, clean the data (fill/remove)
Step 4: Transform the data (normalize numeric fields, encode categoricals)
Step 5: Save the processed data to processed.csv

Each step should only execute if the previous step succeeded.
If any step fails, provide an error message and stop.""",
        "complexity": "Medium",
        "timeout_seconds": 180,
        "max_tokens": 3000,
    },
    "R3": {
        "name": "Migration planning",
        "category": "Multi-Step Reasoning",
        "prompt": """Plan a migration from a monolithic Django application to a microservices architecture.

Current system:
- Django 3.2 monolith with 50+ models, 200+ views
- PostgreSQL database with 30+ tables
- Celery for async tasks
- Redis for caching
- Deployed on single EC2 instance

Target architecture:
- Microservices with domain-bounded contexts
- Container orchestration (Kubernetes)
- Event-driven communication between services
- Independent databases per service

Provide a comprehensive migration plan including:
1. Service boundary identification (which models/views go where)
2. Migration phases with dependencies
3. Data migration strategy (shared DB → independent DBs)
4. Risk assessment with mitigation strategies
5. Rollback plan for each phase
6. Timeline estimate with milestones
7. Testing strategy during migration""",
        "complexity": "Complex",
        "timeout_seconds": 240,
        "max_tokens": 5000,
    },
    "R4": {
        "name": "Debug investigation",
        "category": "Multi-Step Reasoning",
        "prompt": """Investigate and diagnose the following bug. This is an analytical exercise — reason through the problem using the information provided below. Do NOT search for or read files.

Symptom: Users report that search results are inconsistent—the same query
returns different results on repeated executions, and sometimes returns
stale data that was deleted hours ago.

System context:
- Python FastAPI application with Elasticsearch backend
- Elasticsearch cluster: 3 nodes, 2 replicas per index
- Application uses connection pooling with 10 connections
- Search results are cached in Redis with 5-minute TTL
- Recently deployed: index alias rotation for zero-downtime reindexing

Logs show:
- Occasional "ConnectionTimeout" errors from Elasticsearch
- Redis cache hit rate dropped from 85% to 40% after last deploy
- Some search requests take 50ms, others take 3000ms+

Investigate and provide:
1. Root cause analysis (identify the most likely cause)
2. Evidence chain (which symptoms point to which cause)
3. Additional diagnostics you would run to confirm
4. Fix recommendation with code changes
5. Prevention measures for the future""",
        "complexity": "Medium",
        "timeout_seconds": 180,
        "max_tokens": 4000,
    },
    "T2": {
        "name": "Git workflow",
        "category": "Tool Usage",
        "prompt": """Execute a complete Git workflow:

1. Initialize a new git repository in a temporary directory
2. Create an initial commit with a README.md file
3. Create a feature branch named 'feature/add-config'
4. On the feature branch, create a config.yaml file with database settings
5. Commit the config file with an appropriate message
6. Switch back to main branch
7. Create a hotfix branch named 'hotfix/fix-typo'
8. Make a small change to README.md on the hotfix branch
9. Commit the hotfix
10. Merge the hotfix into main
11. Merge the feature branch into main (resolve any conflicts)
12. Create a tagged release v1.0.0
13. Show the final git log with graph

Use git tools to complete all operations.""",
        "complexity": "Medium",
        "timeout_seconds": 120,
        "max_tokens": 3000,
    },
    "T3": {
        "name": "Web research",
        "category": "Tool Usage",
        "prompt": """Conduct web research on the current state of WebAssembly (Wasm) adoption:

1. Search for recent articles on WebAssembly usage trends
2. Find the top 5 languages that compile to WebAssembly
3. Identify 3 major companies using WebAssembly in production
4. Summarize the key advantages and limitations of WebAssembly
5. Compare WebAssembly performance vs JavaScript for compute tasks
6. List notable WebAssembly frameworks and toolchains

Provide a structured research report with:
- Executive summary (2-3 sentences)
- Findings organized by topic
- Sources cited for each claim
- Conclusion with future outlook

Use web search and scraping tools to gather information.""",
        "complexity": "Medium",
        "timeout_seconds": 120,
        "max_tokens": 4000,
    },
    "T4": {
        "name": "Database operations",
        "category": "Tool Usage",
        "prompt": """Perform database operations on a SQLite database:

1. Create a new SQLite database named 'benchmark.db'
2. Create tables:
   - users (id, name, email, created_at)
   - orders (id, user_id, amount, status, created_at)
   - products (id, name, price, stock)
   - order_items (id, order_id, product_id, quantity)
3. Insert sample data:
   - 10 users
   - 20 orders across various users
   - 15 products
   - 30 order items
4. Execute queries:
   - Find top 5 users by total order amount
   - Find products with low stock (< 5 units)
   - Calculate average order value per user
   - Find orders with status 'pending' older than 7 days
5. Create an index on orders(user_id) and explain query plan
6. Export results to a JSON summary file

Use database tools to complete all operations.""",
        "complexity": "Medium",
        "timeout_seconds": 120,
        "max_tokens": 3000,
    },
    "T5": {
        "name": "Command execution",
        "category": "Tool Usage",
        "prompt": """Execute and validate a series of shell commands:

1. Check system information (OS, Python version, available memory)
2. Create a temporary working directory
3. Download a small text file from the internet (or create a sample)
4. Count lines, words, and characters in the file
5. Sort the file contents and save to a new file
6. Calculate an MD5 checksum of both files
7. Compare the two files and report differences
8. Clean up temporary files

For each command:
- Show the command being executed
- Validate the output is correct
- Handle any errors gracefully

Use shell/command execution tools.""",
        "complexity": "Simple",
        "timeout_seconds": 90,
        "max_tokens": 2000,
    },
    "A1": {
        "name": "Security audit",
        "category": "Analysis",
        "prompt": """Perform a security audit on the following Python web application code:

```python
import sqlite3
from flask import Flask, request, session, redirect

app = Flask(__name__)
app.secret_key = "mysecretkey123"

def get_db():
    db = sqlite3.connect("app.db")
    return db

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    db = get_db()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    user = db.execute(query).fetchone()
    if user:
        session["user_id"] = user[0]
        session["role"] = user[3]
        return redirect("/dashboard")
    return "Invalid credentials", 401

@app.route("/admin")
def admin():
    if session.get("role") == "admin":
        return "Admin panel"
    return "Forbidden", 403

@app.route("/api/users/<user_id>")
def get_user(user_id):
    db = get_db()
    user = db.execute(f"SELECT * FROM users WHERE id={user_id}").fetchone()
    return {"id": user[0], "name": user[1], "email": user[2], "role": user[3]}

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    file.save(f"/uploads/{file.filename}")
    return "Uploaded", 200
```

Provide:
1. List of vulnerabilities found (with severity: Critical/High/Medium/Low)
2. OWASP Top 10 classification for each vulnerability
3. Exploit scenario for each critical/high vulnerability
4. Fixed code for each vulnerability
5. Additional security recommendations""",
        "complexity": "Complex",
        "timeout_seconds": 240,
        "max_tokens": 5000,
    },
    "A2": {
        "name": "Performance analysis",
        "category": "Analysis",
        "prompt": """Analyze the performance of the following Python code and identify bottlenecks:

```python
import json
import re

def process_log_file(filepath):
    results = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parsed = json.loads(line)
        if parsed.get('level') == 'ERROR':
            # Extract error details
            message = parsed['message']
            timestamp = parsed['timestamp']
            # Find all IP addresses in the message
            ips = re.findall(r'\\d+\\.\\d+\\.\\d+\\.\\d+', message)
            # Look up each IP in the full log
            for ip in ips:
                count = 0
                for other_line in lines:
                    if ip in other_line:
                        count += 1
                results.append({
                    'timestamp': timestamp,
                    'message': message,
                    'ip': ip,
                    'occurrences': count
                })

    # Remove duplicates
    unique_results = []
    for r in results:
        is_dup = False
        for u in unique_results:
            if r['ip'] == u['ip'] and r['timestamp'] == u['timestamp']:
                is_dup = True
                break
        if not is_dup:
            unique_results.append(r)

    # Sort by occurrences
    for i in range(len(unique_results)):
        for j in range(i + 1, len(unique_results)):
            if unique_results[j]['occurrences'] > unique_results[i]['occurrences']:
                unique_results[i], unique_results[j] = unique_results[j], unique_results[i]

    return unique_results

```

Provide:
1. Time complexity analysis for each section
2. Identification of specific bottlenecks (with Big-O notation)
3. Memory usage analysis
4. Optimized version of the code
5. Expected performance improvement estimates
6. Profiling recommendations for validation""",
        "complexity": "Complex",
        "timeout_seconds": 240,
        "max_tokens": 4000,
    },
    "A3": {
        "name": "Dependency analysis",
        "category": "Analysis",
        "prompt": """Analyze the following Python project's dependency configuration:

```toml
[project]
name = "myapp"
requires-python = ">=3.8"
dependencies = [
    "flask==2.0.1",
    "requests>=2.25.0,<3.0",
    "sqlalchemy==1.4.23",
    "celery[redis]>=5.0,<6.0",
    "pydantic>=1.8,<2.0",
    "boto3",
    "numpy==1.21.0",
    "pandas>=1.3",
    "cryptography>=3.0",
    "PyJWT==2.1.0",
    "pillow>=8.0",
    "redis>=3.5",
    "psycopg2-binary>=2.9",
    "gunicorn>=20.1",
    "python-dateutil",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "black",
    "mypy",
    "coverage",
]
```

Provide:
1. Dependency graph showing direct and transitive relationships
2. Version constraint analysis (too strict, too loose, unpinned)
3. Known security vulnerabilities in specified versions
4. Compatibility issues (Python version, inter-dependency conflicts)
5. Recommendations for version updates
6. Risk assessment for each unpinned dependency""",
        "complexity": "Medium",
        "timeout_seconds": 180,
        "max_tokens": 4000,
    },
    "A4": {
        "name": "Test coverage analysis",
        "category": "Analysis",
        "prompt": """Analyze the test coverage for the following Python module and its test file:

Module (src/calculator.py):
```python
class Calculator:
    def __init__(self, precision=2):
        self.precision = precision
        self.history = []

    def add(self, a, b):
        result = round(a + b, self.precision)
        self.history.append(('add', a, b, result))
        return result

    def subtract(self, a, b):
        result = round(a - b, self.precision)
        self.history.append(('subtract', a, b, result))
        return result

    def multiply(self, a, b):
        result = round(a * b, self.precision)
        self.history.append(('multiply', a, b, result))
        return result

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = round(a / b, self.precision)
        self.history.append(('divide', a, b, result))
        return result

    def get_history(self):
        return self.history.copy()

    def clear_history(self):
        self.history.clear()

    def undo(self):
        if not self.history:
            raise IndexError("No operations to undo")
        return self.history.pop()
```

Tests (tests/test_calculator.py):
```python
from src.calculator import Calculator

def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_subtract():
    calc = Calculator()
    assert calc.subtract(5, 3) == 2

def test_multiply():
    calc = Calculator()
    assert calc.multiply(3, 4) == 12
```

Provide:
1. Current coverage percentage (line and branch)
2. Uncovered lines and branches identified
3. Missing test cases categorized by priority
4. Edge cases that should be tested
5. Test quality assessment (assertions, isolation, naming)
6. Recommended test additions with example code""",
        "complexity": "Medium",
        "timeout_seconds": 180,
        "max_tokens": 4000,
    },
    "W2": {
        "name": "Parallel execution",
        "category": "Workflow & Coordination",
        "prompt": """Execute a parallel data processing workflow:

You have 4 independent data sources that need to be processed simultaneously:

Source 1: User activity logs (parse and count events per user)
Source 2: Transaction records (calculate daily totals)
Source 3: Error logs (categorize and count by severity)
Source 4: Performance metrics (compute p50, p95, p99 latencies)

Workflow:
1. Launch all 4 processing tasks in parallel
2. Each task should:
   - Load its data source
   - Process according to its specific logic
   - Return a structured result
3. Wait for all tasks to complete (with 60-second timeout per task)
4. Aggregate results into a unified dashboard summary
5. If any task fails, include partial results and error details
6. Generate a final report combining all source analyses

Requirements:
- Tasks must execute concurrently (not sequentially)
- Individual task failures should not block other tasks
- Results should include timing information per task
- Final aggregation must handle missing data gracefully""",
        "complexity": "Complex",
        "timeout_seconds": 240,
        "max_tokens": 4000,
    },
    "W3": {
        "name": "Human-in-the-loop",
        "category": "Workflow & Coordination",
        "prompt": """Execute a workflow that requires human approval at key decision points:

Scenario: Automated code deployment pipeline

Step 1: Analyze the proposed changes (automated)
- List files modified
- Categorize changes (feature, bugfix, refactor)
- Run static analysis checks

Step 2: **HUMAN APPROVAL REQUIRED**
- Present change summary to human reviewer
- Wait for approval/rejection/modification request
- If rejected, stop workflow with reason

Step 3: Run test suite (automated)
- Execute unit tests
- Execute integration tests
- Report results

Step 4: **HUMAN APPROVAL REQUIRED**
- Present test results
- If any tests failed, request human decision (deploy anyway / fix / abort)
- Wait for human input

Step 5: Deploy (automated, only if approved)
- Execute deployment steps
- Verify deployment health
- Report final status

Requirements:
- Workflow must pause and wait at human approval points
- Human responses must be validated
- Timeout after 5 minutes of no human response
- All decisions must be logged for audit trail""",
        "complexity": "Complex",
        "timeout_seconds": 300,
        "max_tokens": 4000,
    },
    "W4": {
        "name": "Error recovery",
        "category": "Workflow & Coordination",
        "prompt": """Design and implement a Python error recovery pipeline. Provide the complete code — do NOT search for files or execute commands.

Scenario: Multi-step data pipeline that must handle failures gracefully.

Pipeline steps:
1. Fetch data from external API (may timeout or return errors)
2. Validate and parse response (may have schema violations)
3. Transform data (may encounter unexpected formats)
4. Load into database (may fail on constraints or connectivity)
5. Send notification (may fail on service unavailability)

Error recovery requirements:
- Step 1: Retry up to 3 times with exponential backoff (1s, 2s, 4s)
- Step 2: Use fallback parser if primary fails; skip malformed records
- Step 3: Log transformation errors, continue with valid records
- Step 4: Batch insert with per-record error isolation; retry failed batch once
- Step 5: Queue notification for retry if service unavailable

Additional requirements:
- Implement circuit breaker pattern (open after 5 consecutive failures)
- Maintain a dead-letter queue for unrecoverable errors
- Generate error summary report at pipeline completion
- Ensure partial progress is preserved (checkpoint after each step)

Provide production-ready Python code with type hints and docstrings.""",
        "complexity": "Medium",
        "timeout_seconds": 180,
        "max_tokens": 4000,
    },
}


# ============================================================================
# Framework Adapters
# ============================================================================


class FrameworkAdapter:
    """Base class for framework benchmark adapters."""

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a benchmark task.

        Args:
            task_id: Task identifier (e.g., "C1")
            task_def: Task definition from TASK_REGISTRY

        Returns:
            Result dictionary with success, duration, output, error, etc.
        """
        raise NotImplementedError


class VictorAdapter(FrameworkAdapter):
    """Victor framework adapter for benchmark execution.

    Uses victor's keyring-based API key management to authenticate
    with the configured provider (default: anthropic/claude-sonnet-4-20250514).
    """

    def __init__(
        self,
        timeout: int = 300,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
    ):
        super().__init__(timeout)
        self.provider = provider
        self.model = model

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task using Victor with keyring-based API keys."""
        start_time = time.time()
        try:
            import psutil
            from victor.config.api_keys import get_api_key
            from victor.config.settings import Settings
            from victor.providers.registry import ProviderRegistry

            # Resolve API key from keyring/env
            api_key = get_api_key(self.provider)
            if not api_key:
                raise RuntimeError(
                    f"No API key for '{self.provider}'. "
                    f"Run: victor keys set {self.provider}"
                )

            # Start resource monitoring
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create provider from registry with keyring key and retry support
            provider = ProviderRegistry.create(
                self.provider,
                api_key=api_key,
                timeout=self.timeout,
                max_retries=3,  # Use BaseProvider's built-in retry with backoff
            )

            # Use the provider directly for chat completion
            # (avoids full orchestrator overhead for benchmarks)
            from victor.providers.base import Message

            messages = [
                Message(role="user", content=task_def["prompt"]),
            ]

            response = await provider.chat(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=task_def.get("max_tokens", 2000),
            )

            output = response.content or ""

            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_mb = end_memory - start_memory
            cpu_percent = process.cpu_percent()

            return {
                "task_id": task_id,
                "framework": "victor",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "success": True,
                "duration_ms": round(duration_ms, 2),
                "output": output,
                "output_quality": None,
                "memory_mb": round(memory_mb, 2),
                "cpu_percent": cpu_percent,
                "model": self.model,
                "provider": self.provider,
                "token_usage": {
                    "input": getattr(response, "input_tokens", 0),
                    "output": getattr(response, "output_tokens", 0),
                },
                "error": None,
                "notes": "Completed successfully",
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return {
                "task_id": task_id,
                "framework": "victor",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "success": False,
                "duration_ms": round(duration_ms, 2),
                "output": "",
                "output_quality": None,
                "memory_mb": 0,
                "cpu_percent": 0,
                "model": self.model,
                "provider": self.provider,
                "error": str(e),
                "notes": f"Failed: {type(e).__name__}",
            }


class MockAdapter(FrameworkAdapter):
    """Mock adapter for testing and dry-run mode."""

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simulate task execution without actually running."""
        await asyncio.sleep(0.1)  # Simulate minimal work

        return {
            "task_id": task_id,
            "framework": "mock",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": True,
            "duration_ms": 1000.0,
            "output": f"Mock output for {task_id}",
            "output_quality": 3,
            "memory_mb": 100.0,
            "cpu_percent": 25.0,
            "error": None,
            "notes": "Mock execution",
        }


class LangGraphAdapter(FrameworkAdapter):
    """LangGraph framework adapter for benchmark execution.

    Note: This is a stub for demonstration. Full implementation requires:
    1. LangGraph and LangChain dependencies
    2. State graph definition for each task
    3. Tool configuration matching Victor's tools
    """

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task using LangGraph.

        Stub implementation for demonstration.
        """
        # TODO: Implement LangGraph execution
        # 1. Create StateGraph with nodes for task execution
        # 2. Configure tools matching the task requirements
        # 3. Execute graph and capture results
        return {
            "task_id": task_id,
            "framework": "langgraph",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": False,
            "duration_ms": 0,
            "output": "",
            "output_quality": None,
            "memory_mb": 0,
            "cpu_percent": 0,
            "error": "Not yet implemented",
            "notes": "LangGraph adapter is a stub",
        }


class CrewAIAdapter(FrameworkAdapter):
    """CrewAI framework adapter for benchmark execution.

    Note: This is a stub for demonstration. Full implementation requires:
    1. CrewAI dependencies
    2. Agent definitions for each task type
    3. Tool configuration matching Victor's tools
    """

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task using CrewAI.

        Stub implementation for demonstration.
        """
        # TODO: Implement CrewAI execution
        # 1. Create Agent with appropriate role
        # 2. Configure tools matching the task requirements
        # 3. Execute crew and capture results
        return {
            "task_id": task_id,
            "framework": "crewai",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": False,
            "duration_ms": 0,
            "output": "",
            "output_quality": None,
            "memory_mb": 0,
            "cpu_percent": 0,
            "error": "Not yet implemented",
            "notes": "CrewAI adapter is a stub",
        }


# ============================================================================
# Benchmark Runner
# ============================================================================


async def run_benchmark(
    framework: str,
    task_ids: Optional[List[str]] = None,
    timeout: int = 300,
    verbose: bool = False,
    dry_run: bool = False,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
) -> Dict[str, Any]:
    """Run benchmark suite.

    Args:
        framework: Framework name (victor, langgraph, crewai, etc.)
        task_ids: List of task IDs to run (None = all tasks)
        timeout: Per-task timeout in seconds
        verbose: Enable verbose output
        dry_run: Show what would be executed without running
        provider: LLM provider name (for victor framework)
        model: Model name (for victor framework)

    Returns:
        Summary dictionary with all results
    """
    # Select tasks to run
    if task_ids is None:
        tasks_to_run = list(TASK_REGISTRY.keys())
    else:
        tasks_to_run = [t for t in task_ids if t in TASK_REGISTRY]

    if not tasks_to_run:
        return {"error": "No valid tasks to run"}

    # Select adapter
    if dry_run:
        adapter = MockAdapter()
    elif framework == "victor":
        adapter = VictorAdapter(timeout=timeout, provider=provider, model=model)
    elif framework == "langgraph":
        adapter = LangGraphAdapter(timeout=timeout)
    elif framework == "crewai":
        adapter = CrewAIAdapter(timeout=timeout)
    else:
        return {"error": f"Framework '{framework}' not yet implemented"}

    results = []
    successful = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"Benchmark: {framework.upper()} ({provider}/{model})")
    print(f"Tasks: {len(tasks_to_run)}")
    print(f"Timeout: {timeout}s per task")
    print(f"{'='*60}\n")

    for task_id in tasks_to_run:
        task_def = TASK_REGISTRY[task_id]

        if dry_run:
            print(f"[DRY RUN] {task_id}: {task_def['name']}")
            print(f"  Category: {task_def['category']}")
            print(f"  Complexity: {task_def['complexity']}")
            print(f"  Timeout: {task_def['timeout_seconds']}s")
            print()
            continue

        print(f"[{task_id}] {task_def['name']} ({task_def['complexity']})...")
        start_time = time.time()

        # Rate-limit retries are handled by BaseProvider's ProviderRetryStrategy
        # (exponential backoff with Retry-After header respect, controlled by max_retries)
        result = await adapter.execute_task(task_id, task_def)
        results.append(result)

        duration = time.time() - start_time

        if result["success"]:
            successful += 1
            print(f"  ✅ SUCCESS in {duration:.2f}s")
            if verbose:
                print(f"  Output length: {len(result['output'])} chars")
                print(f"  Memory: {result['memory_mb']:.1f} MB")
                print(f"  CPU: {result['cpu_percent']:.1f}%")
        else:
            failed += 1
            print(f"  ❌ FAILED: {result['error'][:100]}")

    # Calculate summary
    if dry_run:
        return {
            "framework": framework,
            "dry_run": True,
            "tasks_count": len(tasks_to_run),
            "tasks": tasks_to_run,
        }

    total_duration = sum(r["duration_ms"] for r in results) / 1000
    success_rate = successful / len(tasks_to_run) if tasks_to_run else 0

    summary = {
        "framework": framework,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tasks_total": len(tasks_to_run),
        "tasks_successful": successful,
        "tasks_failed": failed,
        "success_rate": round(success_rate * 100, 2),
        "total_duration_seconds": round(total_duration, 2),
        "avg_duration_seconds": (
            round(total_duration / len(tasks_to_run), 2) if tasks_to_run else 0
        ),
        "results": results,
    }

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total tasks: {len(tasks_to_run)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {summary['success_rate']}%")
    print(f"Total duration: {summary['total_duration_seconds']:.2f}s")
    print(f"{'='*60}\n")

    return summary


def save_results(results: Dict[str, Any], framework: str) -> str:
    """Save benchmark results to file.

    Args:
        results: Results dictionary from run_benchmark
        framework: Framework name

    Returns:
        Path to saved results file
    """
    results_dir = PROJECT_ROOT / "docs" / "benchmarking" / "results" / framework
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"benchmark_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")
    return str(results_file)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run competitive benchmarks for agentic AI frameworks"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="victor",
        choices=["victor", "langgraph", "crewai", "mock", "dry-run"],
        help="Framework to benchmark",
    )
    parser.add_argument(
        "--task",
        type=str,
        action="append",
        help="Task ID(s) to run (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-task timeout in seconds",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="anthropic",
        help="LLM provider (anthropic, openai, deepseek, google)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model name (default: cheapest for provider)",
    )
    parser.add_argument(
        "--output",
        "-o",
        action="store_true",
        help="Save results to file",
    )

    args = parser.parse_args()

    # Default models per provider (cheapest capable)
    DEFAULT_MODELS = {
        "anthropic": "claude-haiku-4-5-20251001",
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "google": "gemini-2.0-flash",
    }
    model = args.model or DEFAULT_MODELS.get(args.provider, "gpt-4o-mini")

    # Run benchmark
    results = asyncio.run(
        run_benchmark(
            framework=args.framework,
            task_ids=args.task,
            timeout=args.timeout,
            verbose=args.verbose,
            dry_run=args.dry_run,
            provider=args.provider,
            model=model,
        )
    )

    # Save results if requested
    if args.output and not args.dry_run and "error" not in results:
        save_results(results, args.framework)

    # Exit with error code if any tasks failed
    if results.get("tasks_failed", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
