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

"""Test fixtures for coding vertical tests.

Provides sample code snippets, mock AST trees, LSP responses, and
code review scenarios for comprehensive testing.
"""

from pathlib import Path
from typing import Any, Dict, List

from victor.protocols.lsp_types import (
    CompletionItem,
    CompletionItemKind,
    Diagnostic,
    DiagnosticSeverity,
    DocumentSymbol,
    Position,
    Range,
    SymbolKind,
)


# =============================================================================
# Sample Python Code Snippets
# =============================================================================

SAMPLE_PYTHON_SIMPLE = """def hello_world():
    print("Hello, World!")
"""

SAMPLE_PYTHON_CLASS = """class Calculator:
    \"\"\"A simple calculator class.\"\"\"

    def __init__(self):
        self.result = 0

    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b

    def multiply(self, a: int, b: int) -> int:
        return a * b
"""

SAMPLE_PYTHON_COMPLEX = """from typing import List, Optional
from dataclasses import dataclass

@dataclass
class User:
    \"\"\"User entity.\"\"\"
    id: int
    name: str
    email: str

class UserService:
    \"\"\"Service for managing users.\"\"\"

    def __init__(self, db_connection):
        self.db = db_connection
        self.cache = {}

    async def get_user(self, user_id: int) -> Optional[User]:
        \"\"\"Get user by ID.\"\"\"
        if user_id in self.cache:
            return self.cache[user_id]

        query = "SELECT * FROM users WHERE id = ?"
        result = await self.db.execute(query, (user_id,))
        return User(**result) if result else None

    def create_user(self, name: str, email: str) -> User:
        \"\"\"Create a new user.\"\"\"
        user = User(id=self._generate_id(), name=name, email=email)
        self.cache[user.id] = user
        return user

    def _generate_id(self) -> int:
        return hash(str(__name__)) % 1000000
"""

SAMPLE_PYTHON_WITH_IMPORTS = """import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from requests import get

def process_data(file_path: str) -> Dict[str, Any]:
    data = pd.read_csv(file_path)
    return data.to_dict()
"""

SAMPLE_PYTHON_WITH_ERRORS = """def calculate(x, y):
    result = x + y
    # Missing return statement

def undefined_function():
    var = undefined_variable  # Undefined variable

class BadClass:
    def method1(self):
        pass

    def method2(self):  # Inconsistent indentation
      pass
"""

SAMPLE_PYTHON_WITH_SECURITY_ISSUES = """import subprocess
import pickle

def dangerous_command(user_input):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return query

def run_command(user_input):
    # Command injection vulnerability
    subprocess.call(user_input, shell=True)

def deserialize(data):
    # Insecure deserialization
    return pickle.loads(data)

def weak_hash(password):
    # Weak hashing
    import hashlib
    return hashlib.md5(password.encode()).hexdigest()
"""

SAMPLE_PYTHON_ASYNC = """import asyncio
from typing import AsyncIterator

async def fetch_data(url: str) -> dict:
    await asyncio.sleep(0.1)
    return {"status": "ok", "data": [1, 2, 3]}

async def process_items(items: List[str]) -> AsyncIterator[str]:
    for item in items:
        result = await fetch_data(item)
        yield str(result)

async def main():
    items = ["item1", "item2", "item3"]
    async for result in process_items(items):
        print(result)
"""

SAMPLE_PYTHON_DECORATORS = """def retry(max_attempts: int = 3):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
            return None
        return wrapper
    return decorator

@retry(max_attempts=5)
async def external_api_call():
    return {"success": True}
"""

SAMPLE_PYTHON_CONTEXT_MANAGERS = """class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None

    async def __aenter__(self):
        self.connection = await self._connect()
        return self.connection

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._disconnect()

    async def _connect(self):
        return f"Connected to {self.connection_string}"

    async def _disconnect(self):
        self.connection = None

async with DatabaseConnection("sqlite:///db.sqlite") as db:
    print(db)
"""

SAMPLE_PYTHON_TYPE_HINTS = """from typing import Union, Optional, List, Dict, Callable, TypeVar, Generic

T = TypeVar('T')

def process(value: Union[int, str]) -> str:
    return str(value)

class Container(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    def get(self) -> Optional[T]:
        return self.value

def apply_func(
    data: List[int],
    func: Callable[[int], int]
) -> Dict[int, int]:
    return {x: func(x) for x in data}
"""


# =============================================================================
# Mock AST Trees
# =============================================================================


def create_mock_function_node(name: str, start_line: int = 1) -> Dict[str, Any]:
    """Create a mock function AST node."""
    return {
        "type": "function_definition",
        "name": name,
        "parameters": [],
        "return_type": None,
        "body": [],
        "decorators": [],
        "start_line": start_line,
        "end_line": start_line + 2,
        "is_async": False,
    }


def create_mock_class_node(
    name: str,
    bases: List[str] | None = None,
    start_line: int = 1,
) -> Dict[str, Any]:
    """Create a mock class AST node."""
    return {
        "type": "class_definition",
        "name": name,
        "bases": bases or [],
        "body": [],
        "decorators": [],
        "start_line": start_line,
        "end_line": start_line + 5,
        "methods": [],
    }


def create_mock_import_node(
    module: str,
    names: List[str] | None = None,
    start_line: int = 1,
) -> Dict[str, Any]:
    """Create a mock import AST node."""
    return {
        "type": "import_statement",
        "module": module,
        "names": names or [],
        "is_from_import": bool(names),
        "start_line": start_line,
    }


def create_mock_variable_node(
    name: str,
    value_type: str = "Any",
    start_line: int = 1,
) -> Dict[str, Any]:
    """Create a mock variable AST node."""
    return {
        "type": "assignment",
        "name": name,
        "value_type": value_type,
        "start_line": start_line,
    }


def create_mock_call_node(
    func: str,
    args: List[str] | None = None,
    start_line: int = 1,
) -> Dict[str, Any]:
    """Create a mock function call AST node."""
    return {
        "type": "call",
        "function": func,
        "arguments": args or [],
        "start_line": start_line,
    }


# =============================================================================
# Mock AST Trees for Various Constructs
# =============================================================================

MOCK_AST_SIMPLE_FUNCTION = {
    "type": "module",
    "body": [
        create_mock_function_node("hello_world", start_line=1),
    ],
}

MOCK_AST_CLASS_WITH_METHODS = {
    "type": "module",
    "body": [
        create_mock_class_node(
            "Calculator",
            start_line=1,
        ),
        create_mock_function_node("add", start_line=5),
        create_mock_function_node("subtract", start_line=8),
    ],
}

MOCK_AST_WITH_IMPORTS = {
    "type": "module",
    "body": [
        create_mock_import_node("os", start_line=1),
        create_mock_import_node("sys", start_line=2),
        create_mock_import_node("pathlib", names=["Path"], start_line=3),
        create_mock_import_node("typing", names=["Dict", "List"], start_line=4),
        create_mock_function_node("process_data", start_line=7),
    ],
}

MOCK_AST_COMPLEX_HIERARCHY = {
    "type": "module",
    "body": [
        create_mock_import_node("dataclasses", names=["dataclass"], start_line=1),
        create_mock_class_node("User", start_line=3),
        create_mock_class_node("UserService", start_line=10),
        create_mock_function_node("get_user", start_line=15),
        create_mock_function_node("create_user", start_line=22),
    ],
}


# =============================================================================
# Mock LSP Responses
# =============================================================================


def create_mock_completion(
    label: str,
    kind: CompletionItemKind,
    detail: str = "",
    documentation: str = "",
    insert_text: str | None = None,
) -> CompletionItem:
    """Create a mock completion item."""
    return CompletionItem(
        label=label,
        kind=kind,
        detail=detail,
        documentation=documentation,
        insert_text=insert_text or label,
    )


def create_mock_diagnostic(
    message: str,
    severity: DiagnosticSeverity,
    line: int = 1,
    column: int = 0,
    source: str = "pylint",
    code: str = "",
) -> Diagnostic:
    """Create a mock diagnostic."""
    return Diagnostic(
        range=Range(
            start=Position(line=line, character=column),
            end=Position(line=line, character=column + 10),
        ),
        message=message,
        severity=severity,
        source=source,
        code=code,
    )


def create_mock_symbol(
    name: str,
    kind: SymbolKind,
    detail: str = "",
    range_start_line: int = 1,
    range_end_line: int = 2,
    children: List["DocumentSymbol"] | None = None,
) -> DocumentSymbol:
    """Create a mock document symbol."""
    return DocumentSymbol(
        name=name,
        kind=kind,
        detail=detail,
        range=Range(
            start=Position(line=range_start_line, character=0),
            end=Position(line=range_end_line, character=0),
        ),
        selection_range=Range(
            start=Position(line=range_start_line, character=0),
            end=Position(line=range_start_line, character=len(name)),
        ),
        children=children or [],
    )


# =============================================================================
# Mock LSP Response Collections
# =============================================================================

MOCK_LSP_COMPLETIONS = [
    create_mock_completion(
        "print",
        CompletionItemKind.FUNCTION,
        detail="def print",
        documentation="Print to stdout",
    ),
    create_mock_completion(
        "Calculator",
        CompletionItemKind.CLASS,
        detail="class Calculator",
        documentation="A simple calculator class",
    ),
    create_mock_completion(
        "add",
        CompletionItemKind.METHOD,
        detail="def add",
        documentation="Add two numbers",
    ),
    create_mock_completion(
        "os",
        CompletionItemKind.MODULE,
        detail="module os",
        documentation="Operating system interfaces",
    ),
    create_mock_completion(
        "Path",
        CompletionItemKind.CLASS,
        detail="class Path",
        documentation="Filesystem path",
    ),
]

MOCK_LSP_DIAGNOSTICS = [
    create_mock_diagnostic(
        "Undefined variable 'undefined_variable'",
        DiagnosticSeverity.ERROR,
        line=6,
        code="E0602",
    ),
    create_mock_diagnostic(
        "Function 'calculate' does not return a value",
        DiagnosticSeverity.WARNING,
        line=1,
        code="R1710",
    ),
    create_mock_diagnostic(
        "Inconsistent indentation",
        DiagnosticSeverity.WARNING,
        line=11,
        code="W0312",
    ),
]

MOCK_LSP_SYMBOLS = [
    create_mock_symbol(
        "Calculator",
        SymbolKind.CLASS,
        detail="class Calculator",
        range_start_line=1,
        range_end_line=15,
        children=[
            create_mock_symbol(
                "__init__",
                SymbolKind.METHOD,
                detail="def __init__",
                range_start_line=5,
                range_end_line=7,
            ),
            create_mock_symbol(
                "add",
                SymbolKind.METHOD,
                detail="def add",
                range_start_line=8,
                range_end_line=10,
            ),
            create_mock_symbol(
                "subtract",
                SymbolKind.METHOD,
                detail="def subtract",
                range_start_line=11,
                range_end_line=13,
            ),
        ],
    ),
    create_mock_symbol(
        "hello_world",
        SymbolKind.FUNCTION,
        detail="def hello_world",
        range_start_line=18,
        range_end_line=20,
    ),
]


# =============================================================================
# Mock Code Review Scenarios
# =============================================================================


class CodeReviewScenario:
    """Container for code review test scenarios."""

    def __init__(
        self,
        name: str,
        code: str,
        file_path: str,
        expected_findings: int,
        categories: List[str] | None = None,
    ):
        self.name = name
        self.code = code
        self.file_path = Path(file_path)
        self.expected_findings = expected_findings
        self.categories = categories or []


CODE_REVIEW_SCENARIOS = [
    CodeReviewScenario(
        name="simple_function",
        code=SAMPLE_PYTHON_SIMPLE,
        file_path="/test/simple.py",
        expected_findings=2,  # Missing docstring, type hints
        categories=["readability"],
    ),
    CodeReviewScenario(
        name="class_with_methods",
        code=SAMPLE_PYTHON_CLASS,
        file_path="/test/calculator.py",
        expected_findings=3,  # Missing error handling, input validation, type hints
        categories=["readability", "maintainability"],
    ),
    CodeReviewScenario(
        name="complex_code",
        code=SAMPLE_PYTHON_COMPLEX,
        file_path="/test/user_service.py",
        expected_findings=5,  # SQL injection, no error handling, etc.
        categories=["security", "error_handling"],
    ),
    CodeReviewScenario(
        name="security_issues",
        code=SAMPLE_PYTHON_WITH_SECURITY_ISSUES,
        file_path="/test/dangerous.py",
        expected_findings=4,  # SQL injection, command injection, insecure deserialization, weak hash
        categories=["security"],
    ),
    CodeReviewScenario(
        name="code_with_errors",
        code=SAMPLE_PYTHON_WITH_ERRORS,
        file_path="/test/bad_code.py",
        expected_findings=3,  # Undefined variable, no return, inconsistent indentation
        categories=["error_handling", "style"],
    ),
]


# =============================================================================
# Fixture Factory Functions
# =============================================================================


def create_sample_file(
    tmp_path: Path,
    name: str = "sample.py",
    content: str = SAMPLE_PYTHON_SIMPLE,
) -> Path:
    """Create a sample Python file in a temporary directory."""
    file_path = tmp_path / name
    file_path.write_text(content)
    return file_path


def create_sample_project(
    tmp_path: Path,
    files: Dict[str, str] | None = None,
) -> Path:
    """Create a sample project structure with multiple files."""
    if files is None:
        files = {
            "main.py": SAMPLE_PYTHON_SIMPLE,
            "calculator.py": SAMPLE_PYTHON_CLASS,
            "service.py": SAMPLE_PYTHON_COMPLEX,
        }

    for file_name, content in files.items():
        file_path = tmp_path / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    return tmp_path


def create_mock_syntax_error_node() -> Dict[str, Any]:
    """Create a mock AST node representing a syntax error."""
    return {
        "type": "error",
        "message": "invalid syntax",
        "line": 5,
        "column": 10,
    }


def create_mock_dependency_graph() -> Dict[str, List[str]]:
    """Create a mock import dependency graph."""
    return {
        "main.py": ["utils.py", "config.py"],
        "utils.py": ["helpers.py"],
        "config.py": [],
        "helpers.py": [],
    }


def create_mock_call_graph() -> Dict[str, List[str]]:
    """Create a mock function call graph."""
    return {
        "main": ["process_data", "validate_input"],
        "process_data": ["fetch_data", "transform_data"],
        "validate_input": ["check_format"],
        "fetch_data": [],
        "transform_data": [],
        "check_format": [],
    }


def create_mock_code_metrics() -> Dict[str, Any]:
    """Create mock code complexity metrics."""
    return {
        "total_lines": 150,
        "code_lines": 120,
        "comment_lines": 15,
        "blank_lines": 15,
        "cyclomatic_complexity": 8,
        "function_count": 5,
        "class_count": 2,
        "max_nesting_depth": 3,
        "maintainability_index": 75,
    }


# =============================================================================
# JavaScript/TypeScript Sample Code
# =============================================================================

SAMPLE_JAVASCRIPT_SIMPLE = """function greet(name) {
    return `Hello, ${name}!`;
}

class Calculator {
    constructor() {
        this.result = 0;
    }

    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}

const calc = new Calculator();
console.log(calc.add(5, 3));
"""

SAMPLE_JAVASCRIPT_ASYNC = """async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

async function processItems(items) {
    const results = [];
    for (const item of items) {
        const result = await fetchData(item.url);
        results.push(result);
    }
    return results;
}

const items = [
    { url: '/api/1' },
    { url: '/api/2' }
];

processItems(items).then(console.log);
"""

SAMPLE_JAVASCRIPT_WITH_ERRORS = """function calculateSum(numbers) {
    let total = 0;
    for (let i = 0; i <= numbers.length; i++) {  // Off-by-one error
        total += numbers[i];
    }
    // Missing return statement
}

const undefinedVar = undefinedVariable;  // Undefined variable

class BadClass {
    constructor() {

    method1() {  // Incorrect indentation
        return 1;
    }
    }
}
"""

SAMPLE_TYPESCRIPT = """interface User {
    id: number;
    name: string;
    email: string;
}

type UserRole = 'admin' | 'user' | 'guest';

class UserService {
    private users: Map<number, User> = new Map();

    constructor(private db: DatabaseConnection) {}

    async getUser(id: number): Promise<User | null> {
        const cached = this.users.get(id);
        if (cached) {
            return cached;
        }

        const result = await this.db.query('SELECT * FROM users WHERE id = ?', [id]);
        if (result) {
            const user: User = {
                id: result.id,
                name: result.name,
                email: result.email
            };
            this.users.set(id, user);
            return user;
        }
        return null;
    }

    createUser(data: Omit<User, 'id'>): User {
        const user: User = {
            id: this.generateId(),
            ...data
        };
        this.users.set(user.id, user);
        return user;
    }

    private generateId(): number {
        return Math.floor(Math.random() * 1000000);
    }
}

interface DatabaseConnection {
    query(sql: string, params: any[]): Promise<any>;
}
"""


# =============================================================================
# Advanced Mock LSP Responses (Hover, CodeAction, CodeLens)
# =============================================================================


def create_mock_hover(
    contents: str,
    range: Range | None = None,
    kind: str = "markdown",
) -> Dict[str, Any]:
    """Create a mock hover response."""
    from victor.protocols.lsp_types import Hover

    return {
        "contents": {
            "kind": kind,
            "value": contents,
        },
        "range": range.to_dict() if range else None,
    }


def create_mock_code_action(
    title: str,
    kind: str,
    edit: Dict[str, Any] | None = None,
    command: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a mock code action."""
    return {
        "title": title,
        "kind": kind,
        "edit": edit,
        "command": command,
    }


def create_mock_code_lens(
    range: Range,
    command: Dict[str, Any] | None = None,
    data: Any | None = None,
) -> Dict[str, Any]:
    """Create a mock code lens."""
    return {
        "range": range.to_dict(),
        "command": command,
        "data": data,
    }


MOCK_LSP_HOVER_RESPONSES = [
    create_mock_hover(
        "```python\ndef print(*args, sep=' ', end='\\n')\n```\n\nPrint to stdout",
    ),
    create_mock_hover(
        "```python\nclass Calculator\n```\n\nA simple calculator class",
    ),
    create_mock_hover(
        "```python\nstr\n```\n\nBuilt-in string type",
    ),
]

MOCK_LSP_CODE_ACTIONS = [
    create_mock_code_action(
        title="Quick Fix: Add return statement",
        kind="quickfix",
        edit={
            "changes": {
                "file:///test.py": [
                    {
                        "range": {
                            "start": {"line": 2, "character": 4},
                            "end": {"line": 2, "character": 4},
                        },
                        "newText": "return ",
                    }
                ]
            }
        },
    ),
    create_mock_code_action(
        title="Refactor: Extract method",
        kind="refactor.extract",
    ),
    create_mock_code_action(
        title="Organize imports",
        kind="source.organizeImports",
    ),
]

MOCK_LSP_CODE_LENS = [
    create_mock_code_lens(
        range=Range(
            start=Position(line=0, character=0),
            end=Position(line=0, character=15),
        ),
        command={
            "title": "0 references",
            "command": "editor.action.showReferences",
        },
    ),
    create_mock_code_lens(
        range=Range(
            start=Position(line=5, character=0),
            end=Position(line=5, character=20),
        ),
        command={
            "title": "Run tests",
            "command": "python.runTests",
        },
    ),
]


# =============================================================================
# Tree-sitter Specific Node Mocks
# =============================================================================


def create_mock_tree_sitter_node(
    type_: str,
    text: str,
    start_point: tuple = (0, 0),
    end_point: tuple = (0, 10),
    children: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Create a mock tree-sitter node for testing.

    Args:
        type_: Node type (e.g., "function_definition", "class_definition")
        text: Node text content
        start_point: (row, column) tuple for start position
        end_point: (row, column) tuple for end position
        children: List of child nodes

    Returns:
        Mock tree-sitter node dictionary
    """
    return {
        "type": type_,
        "text": text,
        "start_point": {"row": start_point[0], "column": start_point[1]},
        "end_point": {"row": end_point[0], "column": end_point[1]},
        "children": children or [],
        "is_named": True,
        "is_missing": False,
        "has_changes": False,
        "is_error": False,
    }


MOCK_TREE_SITTER_FUNCTION_NODE = create_mock_tree_sitter_node(
    type_="function_definition",
    text="def hello_world():\n    print('Hello')",
    start_point=(0, 0),
    end_point=(1, 20),
    children=[
        create_mock_tree_sitter_node(
            type_="identifier",
            text="hello_world",
            start_point=(0, 4),
            end_point=(0, 15),
        ),
    ],
)

MOCK_TREE_SITTER_CLASS_NODE = create_mock_tree_sitter_node(
    type_="class_definition",
    text="class Calculator:\n    pass",
    start_point=(0, 0),
    end_point=(1, 8),
    children=[
        create_mock_tree_sitter_node(
            type_="identifier",
            text="Calculator",
            start_point=(0, 6),
            end_point=(0, 17),
        ),
    ],
)

MOCK_TREE_SITTER_IMPORT_NODE = create_mock_tree_sitter_node(
    type_="import_statement",
    text="import os",
    start_point=(0, 0),
    end_point=(0, 9),
    children=[
        create_mock_tree_sitter_node(
            type_="dotted_name",
            text="os",
            start_point=(0, 7),
            end_point=(0, 9),
        ),
    ],
)


# =============================================================================
# Multi-file Project Structure
# =============================================================================

SAMPLE_MULTI_FILE_PROJECT = {
    "src/__init__.py": "",
    "src/main.py": SAMPLE_PYTHON_SIMPLE,
    "src/calculator.py": SAMPLE_PYTHON_CLASS,
    "src/service.py": SAMPLE_PYTHON_COMPLEX,
    "tests/__init__.py": "",
    "tests/test_calculator.py": """import pytest
from src.calculator import Calculator

def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_subtract():
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
""",
    "README.md": """# Sample Project

This is a sample project for testing.
""",
    "requirements.txt": """pytest>=7.0.0
""",
}


# =============================================================================
# Edge Cases and Error Scenarios
# =============================================================================

SAMPLE_PYTHON_SYNTAX_ERROR = """def broken_function(
    # Missing closing parenthesis
    print("This won't work")
"""

SAMPLE_PYTHON_INDENTATION_ERROR = """class BadIndent:
    def method1(self):
        print("method1")
  def method2(self):  # Wrong indentation
        print("method2")
"""

SAMPLE_PYTHON_TYPE_ERROR = """def process(value: int) -> str:
    return value  # Should be str(value)

x: int = "string"  # Type mismatch
"""

SAMPLE_JAVASCRIPT_SYNTAX_ERROR = """function broken(
    console.log('missing paren')
}
"""


# =============================================================================
# Complex Scenarios
# =============================================================================

SAMPLE_PYTHON_INHERITANCE = """class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        raise NotImplementedError

class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name} says Meow!"

def make_animal_speak(animal: Animal) -> str:
    return animal.speak()
"""

SAMPLE_PYTHON_GENERATORS = """def fibonacci(n: int):
    \"\"\"Generate Fibonacci numbers.\"\"\"
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def squares(numbers):
    \"\"\"Generate squares of numbers.\"\"\"
    for num in numbers:
        yield num ** 2

# Usage
for fib in fibonacci(10):
    print(fib)
"""

SAMPLE_PYTHON_METACLASSES = '''class SingletonMeta(type):
    """Metaclass for implementing singleton pattern."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = None

    def connect(self):
        if not self.connection:
            self.connection = "Connected"
        return self.connection
'''


# =============================================================================
# Additional Helper Functions
# =============================================================================


def create_mock_position(line: int = 0, character: int = 0) -> Position:
    """Create a mock position."""
    return Position(line=line, character=character)


def create_mock_range(
    start_line: int = 0,
    start_char: int = 0,
    end_line: int = 0,
    end_char: int = 10,
) -> Range:
    """Create a mock range."""
    return Range(
        start=Position(line=start_line, character=start_char),
        end=Position(line=end_line, character=end_char),
    )


def create_mock_location(
    uri: str = "file:///test.py",
    start_line: int = 0,
    start_char: int = 0,
    end_line: int = 0,
    end_char: int = 10,
) -> Dict[str, Any]:
    """Create a mock location."""
    from victor.protocols.lsp_types import Location

    return Location(
        uri=uri,
        range=Range(
            start=Position(line=start_line, character=start_char),
            end=Position(line=end_line, character=end_char),
        ),
    ).to_dict()


def create_mock_file_content(
    language: str = "python",
    complexity: str = "simple",
) -> str:
    """Create mock file content based on language and complexity.

    Args:
        language: Programming language (python, javascript, typescript)
        complexity: Code complexity (simple, medium, complex)

    Returns:
        Sample code string
    """
    samples = {
        "python": {
            "simple": SAMPLE_PYTHON_SIMPLE,
            "medium": SAMPLE_PYTHON_CLASS,
            "complex": SAMPLE_PYTHON_COMPLEX,
        },
        "javascript": {
            "simple": SAMPLE_JAVASCRIPT_SIMPLE,
            "medium": SAMPLE_JAVASCRIPT_ASYNC,
            "complex": SAMPLE_JAVASCRIPT_WITH_ERRORS,
        },
        "typescript": {
            "simple": SAMPLE_TYPESCRIPT,
            "medium": SAMPLE_TYPESCRIPT,
            "complex": SAMPLE_TYPESCRIPT,
        },
    }

    return samples.get(language, {}).get(complexity, SAMPLE_PYTHON_SIMPLE)


def create_mock_test_scenario(
    name: str,
    file_path: str,
    content: str,
    expected_diagnostics: int = 0,
    expected_symbols: int = 0,
) -> Dict[str, Any]:
    """Create a complete test scenario with metadata.

    Args:
        name: Scenario name
        file_path: File path
        content: File content
        expected_diagnostics: Expected number of diagnostics
        expected_symbols: Expected number of symbols

    Returns:
        Test scenario dictionary
    """
    return {
        "name": name,
        "file_path": Path(file_path),
        "content": content,
        "expected_diagnostics": expected_diagnostics,
        "expected_symbols": expected_symbols,
    }


# =============================================================================
# Integration Test Scenarios
# =============================================================================

INTEGRATION_TEST_SCENARIOS = [
    create_mock_test_scenario(
        name="simple_python",
        file_path="/test/simple.py",
        content=SAMPLE_PYTHON_SIMPLE,
        expected_diagnostics=0,
        expected_symbols=1,
    ),
    create_mock_test_scenario(
        name="python_with_errors",
        file_path="/test/errors.py",
        content=SAMPLE_PYTHON_WITH_ERRORS,
        expected_diagnostics=3,
        expected_symbols=2,
    ),
    create_mock_test_scenario(
        name="javascript_simple",
        file_path="/test/script.js",
        content=SAMPLE_JAVASCRIPT_SIMPLE,
        expected_diagnostics=0,
        expected_symbols=2,
    ),
]


# =============================================================================
# Quick Access Exports
# =============================================================================

__all__ = [
    # Sample code - Python
    "SAMPLE_PYTHON_SIMPLE",
    "SAMPLE_PYTHON_CLASS",
    "SAMPLE_PYTHON_COMPLEX",
    "SAMPLE_PYTHON_WITH_IMPORTS",
    "SAMPLE_PYTHON_WITH_ERRORS",
    "SAMPLE_PYTHON_WITH_SECURITY_ISSUES",
    "SAMPLE_PYTHON_ASYNC",
    "SAMPLE_PYTHON_DECORATORS",
    "SAMPLE_PYTHON_CONTEXT_MANAGERS",
    "SAMPLE_PYTHON_TYPE_HINTS",
    "SAMPLE_PYTHON_INHERITANCE",
    "SAMPLE_PYTHON_GENERATORS",
    "SAMPLE_PYTHON_METACLASSES",
    "SAMPLE_PYTHON_SYNTAX_ERROR",
    "SAMPLE_PYTHON_INDENTATION_ERROR",
    "SAMPLE_PYTHON_TYPE_ERROR",
    # Sample code - JavaScript/TypeScript
    "SAMPLE_JAVASCRIPT_SIMPLE",
    "SAMPLE_JAVASCRIPT_ASYNC",
    "SAMPLE_JAVASCRIPT_WITH_ERRORS",
    "SAMPLE_JAVASCRIPT_SYNTAX_ERROR",
    "SAMPLE_TYPESCRIPT",
    # Mock nodes
    "create_mock_function_node",
    "create_mock_class_node",
    "create_mock_import_node",
    "create_mock_variable_node",
    "create_mock_call_node",
    # Mock ASTs
    "MOCK_AST_SIMPLE_FUNCTION",
    "MOCK_AST_CLASS_WITH_METHODS",
    "MOCK_AST_WITH_IMPORTS",
    "MOCK_AST_COMPLEX_HIERARCHY",
    # Mock LSP - Basic
    "create_mock_completion",
    "create_mock_diagnostic",
    "create_mock_symbol",
    "MOCK_LSP_COMPLETIONS",
    "MOCK_LSP_DIAGNOSTICS",
    "MOCK_LSP_SYMBOLS",
    # Mock LSP - Advanced
    "create_mock_hover",
    "create_mock_code_action",
    "create_mock_code_lens",
    "MOCK_LSP_HOVER_RESPONSES",
    "MOCK_LSP_CODE_ACTIONS",
    "MOCK_LSP_CODE_LENS",
    # Tree-sitter mocks
    "create_mock_tree_sitter_node",
    "MOCK_TREE_SITTER_FUNCTION_NODE",
    "MOCK_TREE_SITTER_CLASS_NODE",
    "MOCK_TREE_SITTER_IMPORT_NODE",
    # Project structures
    "SAMPLE_MULTI_FILE_PROJECT",
    # Code review scenarios
    "CodeReviewScenario",
    "CODE_REVIEW_SCENARIOS",
    # Factories
    "create_sample_file",
    "create_sample_project",
    "create_mock_syntax_error_node",
    "create_mock_dependency_graph",
    "create_mock_call_graph",
    "create_mock_code_metrics",
    "create_mock_position",
    "create_mock_range",
    "create_mock_location",
    "create_mock_file_content",
    "create_mock_test_scenario",
    # Integration scenarios
    "INTEGRATION_TEST_SCENARIOS",
]
