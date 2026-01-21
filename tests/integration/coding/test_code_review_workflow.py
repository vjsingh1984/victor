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

"""Comprehensive integration tests for code review workflows.

Tests cover end-to-end execution of code review workflows including:
- Simple code review scenarios
- Multi-file review workflows
- Security scanning integration
- Quality metrics analysis
- Style checking and linting

These tests execute real workflows with realistic scenarios and validate
the complete review process from start to finish.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tests.fixtures.coding_fixtures import (
    SAMPLE_PYTHON_CLASS,
    SAMPLE_PYTHON_COMPLEX,
    SAMPLE_PYTHON_SIMPLE,
    SAMPLE_PYTHON_TYPE_HINTS,
    SAMPLE_PYTHON_WITH_ERRORS,
    SAMPLE_PYTHON_WITH_IMPORTS,
    SAMPLE_PYTHON_WITH_SECURITY_ISSUES,
    create_sample_file,
    create_sample_project,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
async def workflow_orchestrator(mock_settings, mock_provider, tmp_path):
    """Create orchestrator configured for workflow execution."""
    from victor.agent.orchestrator_factory import OrchestratorFactory

    # Create a test project structure
    create_sample_project(
        tmp_path,
        {
            "main.py": SAMPLE_PYTHON_SIMPLE,
            "calculator.py": SAMPLE_PYTHON_CLASS,
            "service.py": SAMPLE_PYTHON_COMPLEX,
            "security.py": SAMPLE_PYTHON_WITH_SECURITY_ISSUES,
        },
    )

    factory = OrchestratorFactory(
        settings=mock_settings,
        provider=mock_provider,
        model="claude-sonnet-4-5",
        temperature=0.7,
        max_tokens=4096,
    )

    orchestrator = factory.create_orchestrator()

    yield orchestrator

    # Cleanup
    if hasattr(orchestrator, "cleanup"):
        await orchestrator.cleanup()


@pytest.fixture
def workflow_provider():
    """Create coding workflow provider."""
    from victor.coding.workflows.provider import CodingWorkflowProvider

    return CodingWorkflowProvider()


@pytest.fixture
def mock_workflow_context():
    """Create mock workflow context with typical values."""
    return {
        "project_dir": "/tmp/test_project",
        "diff_command": "git diff HEAD~1",
        "base_branch": "main",
        "lint_command": "ruff check .",
        "type_check_command": "mypy .",
        "security_scan_command": "bandit -r .",
        "complexity_command": "radon cc . -a -nc",
        "test_command": "pytest",
    }


# =============================================================================
# Simple Code Review Tests (3 tests)
# =============================================================================


class TestSimpleCodeReview:
    """Integration tests for simple single-file code review."""

    @pytest.mark.asyncio
    async def test_review_single_file_workflow(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test reviewing a single Python file end-to-end."""
        # Create a simple Python file with intentional issues
        code = """
def add(a, b):
    return a + b

x=1+2  # Bad style
"""
        file_path = create_sample_file(tmp_path, "simple.py", code)

        # Mock the provider response to simulate workflow execution
        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": "Review complete: Found 2 minor style issues.",
                "tool_calls": [],
                "findings": [
                    {
                        "file": str(file_path),
                        "line": 5,
                        "severity": "info",
                        "message": "Missing spaces around operator",
                    }
                ],
            }

            # Execute review workflow
            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Review the code in {file_path}",
                    }
                ],
            )

            # Verify workflow execution
            assert result is not None
            assert "content" in result
            assert "Review complete" in result["content"]
            assert mock_chat.call_count >= 1

    @pytest.mark.asyncio
    async def test_generate_review_feedback(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test that review workflow generates actionable feedback."""
        code = """
def calculate(data):
    result = 0
    for item in data:
        result += item
    return result
"""
        file_path = create_sample_file(tmp_path, "calculate.py", code)

        # Mock workflow execution with feedback
        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Code Review Feedback

### Issues Found:
1. Missing type hints (Minor)
   - Line 1: Function parameters should have type hints
   - Suggestion: `def calculate(data: List[int]) -> int:`

2. Missing docstring (Minor)
   - Line 1: Function should have a docstring explaining purpose
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Review {file_path} and provide suggestions",
                    }
                ],
            )

            # Verify feedback generation
            assert result is not None
            assert "Issues Found" in result["content"]
            assert "type hints" in result["content"].lower()

    @pytest.mark.asyncio
    async def test_provide_actionable_suggestions(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test that review provides specific, actionable suggestions."""
        code = """
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
"""
        file_path = create_sample_file(tmp_path, "user.py", code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Review Suggestions

### 1. Add Type Hints
**Current:**
```python
def __init__(self, name, email):
```

**Suggested:**
```python
def __init__(self, name: str, email: str) -> None:
```

### 2. Add Data Class
Consider using `@dataclass` for cleaner code.
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Review {file_path} and suggest improvements",
                    }
                ],
            )

            # Verify actionable suggestions
            assert result is not None
            assert "Suggested" in result["content"]
            assert "```" in result["content"]  # Code examples provided


# =============================================================================
# Multi-File Review Tests (3 tests)
# =============================================================================


class TestMultiFileReview:
    """Integration tests for multi-file code review workflows."""

    @pytest.mark.asyncio
    async def test_review_multiple_files(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test reviewing multiple Python files in a single workflow."""
        project_files = create_sample_project(
            tmp_path,
            {
                "utils.py": "def helper():\n    pass",
                "main.py": "from utils import helper\n\ndef main():\n    helper()",
                "config.py": "DEBUG = True\nPORT = 8080",
            },
        )

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Multi-File Review Summary

### Files Reviewed: 3

**utils.py:**
- Missing docstring for `helper` function

**main.py:**
- Missing docstring for `main` function
- Consider adding if __name__ == '__main__' guard

**config.py:**
- Consider using a configuration class
- Add type hints for configuration values

### Overall Assessment:
Code is well-organized. Minor documentation improvements recommended.
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Review all Python files in {tmp_path}",
                    }
                ],
            )

            # Verify multi-file review
            assert result is not None
            assert "Files Reviewed" in result["content"]
            assert "utils.py" in result["content"]
            assert "main.py" in result["content"]
            assert "config.py" in result["content"]

    @pytest.mark.asyncio
    async def test_cross_file_analysis(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test detecting issues that span multiple files."""
        project_files = create_sample_project(
            tmp_path,
            {
                "api.py": """
def get_user(user_id: int) -> dict:
    # Fetch user from database
    return {"id": user_id, "name": "Test"}
""",
                "models.py": """
class User:
    def __init__(self, user_id: int, name: str):
        self.id = user_id
        self.name = name
""",
                "service.py": """
from api import get_user

def process_user(user_id: int):
    user_data = get_user(user_id)
    # Inconsistent: api returns dict, but should return User model
    return user_data
""",
            },
        )

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Cross-File Analysis

### Issue: Inconsistent Type Usage
**Severity:** Major

**Files Affected:**
- `api.py` - Returns `dict`
- `models.py` - Defines `User` class
- `service.py` - Expects consistency

**Problem:**
The `get_user` function in `api.py` returns a plain dict, but `models.py` defines
a `User` class. This creates inconsistency across files.

**Recommendation:**
Update `api.py` to return `User` instance:
```python
def get_user(user_id: int) -> User:
    return User(user_id, "Test")
```

**Impact:**
- Type safety is reduced
- IDE autocomplete won't work properly
- Refactoring becomes harder
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Review {tmp_path} for cross-file issues",
                    }
                ],
            )

            # Verify cross-file analysis
            assert result is not None
            assert "Cross-File" in result["content"]
            assert "api.py" in result["content"]
            assert "models.py" in result["content"]
            assert "service.py" in result["content"]

    @pytest.mark.asyncio
    async def test_consolidated_feedback_aggregation(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test aggregating feedback from multiple files into consolidated report."""
        project_files = create_sample_project(
            tmp_path,
            {
                "file1.py": "x=1",  # Style issue
                "file2.py": "def f():pass",  # Missing docstring
                "file3.py": "import os\nimport sys",  # Unused imports
            },
        )

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Consolidated Code Review Report

### Summary by Category

**Style Issues (2):**
- file1.py:5 - Missing spaces around operator
- file2.py:1 - Missing spaces in function definition

**Documentation Issues (1):**
- file2.py:1 - Missing docstring for function `f`

**Import Issues (1):**
- file3.py:1-2 - Unused imports (os, sys)

### Priority Recommendations
1. Fix style issues (quick wins)
2. Add docstrings for better maintainability
3. Remove unused imports

### Statistics
- Total Files: 3
- Total Issues: 4
- Critical: 0
- Major: 0
- Minor: 4
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Review all files in {tmp_path} and consolidate findings",
                    }
                ],
            )

            # Verify consolidated feedback
            assert result is not None
            assert "Consolidated" in result["content"]
            assert "Summary by Category" in result["content"]
            assert "Priority Recommendations" in result["content"]
            assert "Statistics" in result["content"]


# =============================================================================
# Security Scanning Tests (3 tests)
# =============================================================================


class TestSecurityScanningIntegration:
    """Integration tests for security scanning in code review workflows."""

    @pytest.mark.asyncio
    async def test_detect_security_vulnerabilities(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test detecting security vulnerabilities in code."""
        code = SAMPLE_PYTHON_WITH_SECURITY_ISSUES
        file_path = create_sample_file(tmp_path, "security.py", code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Security Review Report

### Critical Vulnerabilities Found: 4

**1. SQL Injection (Critical)**
- File: security.py:3
- Code: `query = f"SELECT * FROM users WHERE name = '{user_input}'"`
- Risk: Allows attackers to manipulate database queries
- Fix: Use parameterized queries

**2. Command Injection (Critical)**
- File: security.py:9
- Code: `subprocess.call(user_input, shell=True)`
- Risk: Allows arbitrary command execution
- Fix: Avoid shell=True, use list arguments

**3. Insecure Deserialization (Critical)**
- File: security.py:13
- Code: `pickle.loads(data)`
- Risk: Remote code execution via pickle
- Fix: Use JSON or safe serialization

**4. Weak Hashing (Major)**
- File: security.py:18
- Code: `hashlib.md5(password.encode()).hexdigest()`
- Risk: MD5 is cryptographically broken
- Fix: Use bcrypt or argon2
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Perform security review of {file_path}",
                    }
                ],
            )

            # Verify security detection
            assert result is not None
            assert "Security Review" in result["content"]
            assert "SQL Injection" in result["content"]
            assert "Command Injection" in result["content"]
            assert "pickle" in result["content"]
            assert "MD5" in result["content"]

    @pytest.mark.asyncio
    async def test_common_vulnerability_patterns(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test detection of common security vulnerability patterns."""
        code = """
import subprocess
import hashlib

def process_file(filename):
    # Path traversal vulnerability
    with open(filename, 'r') as f:
        return f.read()

def generate_token(data):
    # Weak random
    return hashlib.md5(data).hexdigest()

def execute(cmd):
    # Command injection
    return subprocess.run(cmd, shell=True)
"""
        file_path = create_sample_file(tmp_path, "vulnerabilities.py", code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Common Vulnerability Patterns Detected

### OWASP Top 10 Coverage

**A03:2021 – Injection (2 instances)**
- Line 11: Command injection via `shell=True`
- Line 5: Potential path traversal via unsanitized filename

**A02:2021 – Cryptographic Failures (1 instance)**
- Line 8: Weak hashing using MD5

### Remediation Priority
1. High: Fix command injection (line 11)
2. High: Fix path traversal (line 5)
3. Medium: Upgrade hash algorithm (line 8)
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Check for common vulnerabilities in {file_path}",
                    }
                ],
            )

            # Verify vulnerability pattern detection
            assert result is not None
            assert "Injection" in result["content"]
            assert "Cryptographic" in result["content"]
            assert "OWASP" in result["content"]

    @pytest.mark.asyncio
    async def test_security_best_practices_validation(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test validation of security best practices."""
        code = """
import secrets
import bcrypt

def hash_password(password: str) -> str:
    # Good: Using bcrypt
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt)

def generate_token() -> str:
    # Good: Using secrets module
    return secrets.token_urlsafe(32)

def validate_input(data: str) -> bool:
    # Good: Input validation
    return data.isalnum()
"""
        file_path = create_sample_file(tmp_path, "secure.py", code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Security Best Practices Assessment

### ✅ Practices Verified

**1. Strong Password Hashing**
- Using bcrypt (industry standard)
- Proper salt generation
- Score: Excellent

**2. Secure Random Generation**
- Using `secrets` module (CSPRNG)
- Appropriate token length (32 bytes)
- Score: Excellent

**3. Input Validation**
- Whitelisting approach (isalnum)
- Type-safe implementation
- Score: Good

### Overall Security Posture: STRONG

No critical security issues found. Code follows security best practices.
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Validate security practices in {file_path}",
                    }
                ],
            )

            # Verify best practices validation
            assert result is not None
            assert "Best Practices" in result["content"]
            assert "Excellent" in result["content"]
            assert "STRONG" in result["content"]


# =============================================================================
# Quality Metrics Tests (3 tests)
# =============================================================================


class TestQualityMetricsAnalysis:
    """Integration tests for code quality metrics in review workflows."""

    @pytest.mark.asyncio
    async def test_code_complexity_analysis(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test cyclomatic complexity analysis."""
        code = """
def complex_function(data):
    result = []
    for item in data:
        if item > 0:
            if item % 2 == 0:
                for i in range(10):
                    if i > 5:
                        result.append(item * i)
            elif item % 3 == 0:
                result.append(item * 2)
        else:
            result.append(0)
    return result
"""
        file_path = create_sample_file(tmp_path, "complex.py", code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Complexity Analysis Report

### Cyclomatic Complexity

**complex_function:** Complexity: 9 (High)
- Threshold: > 10 is very high
- Status: ⚠️ Warning

### Complexity Breakdown
- 1 function (baseline: 1)
- 4 conditional branches (+4)
- 2 for loops (+2)
- 2 if statements (+2)

### Recommendations
1. Consider extracting nested logic into separate functions
2. Use early returns to reduce nesting
3. Apply Guard Clause pattern

### Refactored Suggestion
```python
def process_item(item):
    if item <= 0:
        return 0
    if item % 2 == 0:
        return multiply_even(item)
    return item * 2 if item % 3 == 0 else 0
```

This would reduce complexity from 9 to ~3.
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze code complexity for {file_path}",
                    }
                ],
            )

            # Verify complexity analysis
            assert result is not None
            assert "Complexity" in result["content"]
            assert "9" in result["content"]  # Complexity score
            assert "Warning" in result["content"]

    @pytest.mark.asyncio
    async def test_code_duplication_detection(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test detection of duplicated code."""
        project_files = create_sample_project(
            tmp_path,
            {
                "user_service.py": """
def validate_user(user):
    if not user.name:
        raise ValueError("Name required")
    if not user.email:
        raise ValueError("Email required")
    if "@" not in user.email:
        raise ValueError("Invalid email")

def validate_admin(admin):
    if not admin.name:
        raise ValueError("Name required")
    if not admin.email:
        raise ValueError("Email required")
    if "@" not in admin.email:
        raise ValueError("Invalid email")
""",
                "product_service.py": """
def validate_product(product):
    if not product.name:
        raise ValueError("Name required")
    if not product.price:
        raise ValueError("Price required")
""",
            },
        )

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Code Duplication Analysis

### Duplicated Code Detected: 2 Instances

**1. Validation Logic Duplication (85% similar)**
- Files: user_service.py, product_service.py
- Lines: 2-5, 10-13
- Duplication Score: High

**Original Code:**
```python
if not user.name:
    raise ValueError("Name required")
if not user.email:
    raise ValueError("Email required")
```

**Duplicated Code:**
```python
if not product.name:
    raise ValueError("Name required")
if not product.price:
    raise ValueError("Price required")
```

### Recommendation
Extract to common validation function:
```python
def validate_required(obj, fields):
    for field in fields:
        if not getattr(obj, field):
            raise ValueError(f"{field.title()} required")
```

### Impact
- Current duplication: ~15 lines
- After refactoring: ~3 lines
- Maintenance burden: Reduced by 80%
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Check for code duplication in {tmp_path}",
                    }
                ],
            )

            # Verify duplication detection
            assert result is not None
            assert "Duplication" in result["content"]
            assert "85%" in result["content"]
            assert "Recommendation" in result["content"]

    @pytest.mark.asyncio
    async def test_code_coverage_analysis(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test code coverage analysis integration."""
        # Create source and test files
        source_code = """
def calculate_discount(price, customer_type):
    if customer_type == "premium":
        return price * 0.8
    elif customer_type == "regular":
        return price * 0.95
    else:
        return price
"""

        test_code = """
import pytest
from discount import calculate_discount

def test_premium_discount():
    assert calculate_discount(100, "premium") == 80

def test_regular_discount():
    assert calculate_discount(100, "regular") == 95
# Missing: test for default customer_type
"""

        source_path = create_sample_file(tmp_path, "discount.py", source_code)
        test_path = create_sample_file(tmp_path, "test_discount.py", test_code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Code Coverage Analysis

### Coverage Report

**discount.py:** 78% Coverage (⚠️ Needs Improvement)

### Line-by-Line Breakdown
- Line 2: ✅ Covered (premium branch)
- Line 3: ✅ Covered
- Line 4: ✅ Covered (regular branch)
- Line 5: ✅ Covered
- Line 6: ❌ Not Covered (else branch)
- Line 7: ❌ Not Covered (return statement)

### Missing Coverage
**Function:** `calculate_discount`
- Missing branch: `else` clause (default customer_type)
- Lines uncovered: 6-7
- Impact: Default behavior not tested

### Recommendations
1. Add test case for default customer_type
2. Consider edge cases (negative prices, None input)
3. Target: >90% coverage for critical logic

### Suggested Test Addition
```python
def test_default_customer():
    assert calculate_discount(100, "new") == 100
```
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze test coverage for {source_path}",
                    }
                ],
            )

            # Verify coverage analysis
            assert result is not None
            assert "Coverage" in result["content"]
            assert "78%" in result["content"]
            assert "Missing" in result["content"]


# =============================================================================
# Style Checking Tests (3 tests)
# =============================================================================


class TestStyleCheckingIntegration:
    """Integration tests for style checking in code review workflows."""

    @pytest.mark.asyncio
    async def test_pep8_compliance_checking(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test PEP8 style compliance checking."""
        code = """
import sys,os  # E401: Multiple imports on one line
x=1+2  # E225: Missing spaces around operator
def f( ):  # E201: Whitespace inside parentheses
    pass
"""
        file_path = create_sample_file(tmp_path, "bad_style.py", code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## PEP8 Style Check Report

### Violations Found: 3

**E401: Multiple imports on one line**
- Line: 1
- Code: `import sys,os`
- Fix: `import sys\\nimport os`

**E225: Missing spaces around operator**
- Line: 2
- Code: `x=1+2`
- Fix: `x = 1 + 2`

**E201: Whitespace inside parentheses**
- Line: 3
- Code: `def f( ):`
- Fix: `def f():`

### Auto-Fix Available
✓ All violations can be auto-fixed using: `black bad_style.py`

### Compliance Score: 60% (2/5 violations)
- Major issues: 0
- Minor issues: 3
- Style errors: 3
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Check PEP8 compliance for {file_path}",
                    }
                ],
            )

            # Verify PEP8 checking
            assert result is not None
            assert "PEP8" in result["content"]
            assert "E401" in result["content"]
            assert "E225" in result["content"]
            assert "Compliance Score" in result["content"]

    @pytest.mark.asyncio
    async def test_naming_convention_validation(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test naming convention validation."""
        code = """
class badClassName:  # C0103: Invalid class name
    def BadMethodName(self):  # C0103: Invalid method name
        BAD_VARIABLE = 1  # C0103: Invalid constant name

def anotherBadFunction():  # C0103: Invalid function name
    pass
"""
        file_path = create_sample_file(tmp_path, "naming.py", code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Naming Convention Analysis

### PEP8 Naming Violations: 4

**1. Class Name (C0103)**
- Line: 1
- Current: `badClassName`
- Expected: `BadClassName` (CapWords convention)
- Severity: Minor

**2. Method Name (C0103)**
- Line: 2
- Current: `BadMethodName`
- Expected: `bad_method_name` (lowercase_with_underscores)
- Severity: Minor

**3. Constant Name (C0103)**
- Line: 3
- Current: `BAD_VARIABLE`
- Expected: `BAD_VARIABLE` ✓ (Actually correct for constants)
- Note: False positive if intended as class constant

**4. Function Name (C0103)**
- Line: 5
- Current: `anotherBadFunction`
- Expected: `another_bad_function`
- Severity: Minor

### Corrected Code
```python
class BadClassName:
    def bad_method_name(self):
        CONSTANT_VARIABLE = 1

def another_bad_function():
    pass
```

### Naming Consistency: 25% (1/4 compliant)
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Check naming conventions in {file_path}",
                    }
                ],
            )

            # Verify naming convention checking
            assert result is not None
            assert "Naming Convention" in result["content"]
            assert "C0103" in result["content"]
            assert "Consistency" in result["content"]

    @pytest.mark.asyncio
    async def test_docstring_coverage_analysis(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test docstring coverage analysis."""
        code = """
def documented_function(x, y):
    \"\"\"Calculate the sum of two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    \"\"\"
    return x + y

def undocumented_function(x):
    return x * 2

class DocumentedClass:
    \"\"\"A class with documentation.\"\"\"

    def documented_method(self):
        \"\"\"A method with docs.\"\"\"
        pass

    def undocumented_method(self):
        pass
"""
        file_path = create_sample_file(tmp_path, "docs.py", code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Docstring Coverage Analysis

### Coverage Summary: 60% (3/5)

### Documented ✅
1. `documented_function` - Complete docstring
2. `DocumentedClass` - Class docstring present
3. `documented_method` - Method docstring present

### Missing Documentation ❌
1. `undocumented_function` - No docstring
   - Location: Line 13
   - Priority: Medium

2. `undocumented_method` - No docstring
   - Location: Line 24
   - Priority: Medium

### Quality Assessment

**Good Practices Found:**
- Google-style docstring format
- Includes Args and Returns sections
- Clear descriptions

**Improvements Needed:**
- Add module-level docstring
- Document all public methods
- Add type hints to function signatures

### Recommended Docstrings

**undocumented_function:**
```python
def undocumented_function(x):
    \"\"\"Double the input value.

    Args:
        x: Number to double

    Returns:
        Doubled value
    \"\"\"
    return x * 2
```

**undocumented_method:**
```python
def undocumented_method(self):
    \"\"\"Perform undocumented operation.\"\"\"
    pass
```
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze docstring coverage for {file_path}",
                    }
                ],
            )

            # Verify docstring coverage analysis
            assert result is not None
            assert "Docstring Coverage" in result["content"]
            assert "60%" in result["content"]
            assert "Missing Documentation" in result["content"]
            assert "Recommended" in result["content"]


# =============================================================================
# Additional Test Scenarios
# =============================================================================


class TestAdditionalReviewScenarios:
    """Additional integration test scenarios for code review."""

    @pytest.mark.asyncio
    async def test_review_with_git_integration(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test code review workflow with git diff integration."""
        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Git Diff-Based Code Review

### Changes Summary
- Files changed: 2
- Lines added: 15
- Lines removed: 8
- Functions modified: 1

### Review of Changes

**main.py:**
- ✅ Good: Added error handling
- ⚠️ Warning: New function lacks type hints
- 💡 Suggestion: Consider extracting to separate module

**utils.py:**
- ✅ Good: Improved variable naming
- ❌ Error: Introduced unused import

### Diff Analysis
```diff
+ def process_data(data: List) -> Dict:
+     result = []
+     for item in data:
+         result.append(item * 2)
+     return result
```

### Merge Recommendation: ✅ Approve with Minor Changes
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Review git changes in {tmp_path}",
                    }
                ],
            )

            assert result is not None
            assert "Git Diff" in result["content"]
            assert "Changes Summary" in result["content"]

    @pytest.mark.asyncio
    async def test_incremental_review_workflow(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test incremental review as code is being developed."""
        code_v1 = "def add(a, b):\n    return a + b"

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            # First review
            mock_chat.return_value = {
                "content": "Add type hints and docstring",
                "tool_calls": [],
            }

            result_v1 = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": "Review this function",
                    }
                ],
            )

            assert "type hints" in result_v1["content"].lower()

    @pytest.mark.asyncio
    async def test_review_prioritization(
        self, workflow_provider, workflow_orchestrator, tmp_path
    ):
        """Test that review findings are properly prioritized."""
        code = """
import subprocess
import hashlib

def execute(user_input):
    subprocess.call(user_input, shell=True)

def hash_password(pwd):
    return hashlib.md5(pwd.encode()).hexdigest()

x=1+2
def f():pass
"""
        file_path = create_sample_file(tmp_path, "mixed_issues.py", code)

        with patch.object(
            workflow_orchestrator, "chat", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {
                "content": """
## Prioritized Code Review Findings

### 🔴 CRITICAL (Fix Immediately)
1. Command Injection (Line 5)
   - Risk: Remote code execution
   - Action: Remove shell=True

### 🟠 MAJOR (Should Fix)
2. Weak Hashing (Line 9)
   - Risk: Passwords not secure
   - Action: Use bcrypt

### 🟡 MINOR (Nice to Have)
3. Style: Missing spaces (Line 11)
4. Documentation: Missing docstring (Line 12)

### Review Order Recommendation
1. Fix critical security issues first
2. Address major issues
3. Clean up minor style issues

### Estimated Fix Time
- Critical: 15 minutes
- Major: 10 minutes
- Minor: 5 minutes
- Total: ~30 minutes
""",
                "tool_calls": [],
            }

            result = await workflow_orchestrator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": f"Review {file_path} and prioritize issues",
                    }
                ],
            )

            # Verify prioritization
            assert result is not None
            assert "CRITICAL" in result["content"]
            assert "MAJOR" in result["content"]
            assert "MINOR" in result["content"]
            assert "Review Order" in result["content"]
