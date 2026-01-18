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

"""Multi-language code analysis using tree-sitter.

This module provides a unified interface for analyzing code across multiple
programming languages. It uses tree-sitter for AST parsing, enabling accurate
complexity analysis, documentation detection, and security scanning.

Architecture:
- LanguageAnalyzer: Protocol defining the analysis interface
- BaseLanguageAnalyzer: Common implementation with tree-sitter integration
- Language-specific analyzers for 20+ languages
- LanguageRegistry: Auto-detection and analyzer lookup

Supported Languages:
- Core: Python, JavaScript, TypeScript, Java, Go, Rust
- Systems: C, C++, C#
- Scripting: Ruby, PHP
- JVM: Kotlin, Scala
- Apple: Swift
- Shell/Data: Bash, SQL
- Other: Lua, Elixir, Haskell, R

Features:
- Security vulnerability detection (language-specific patterns)
- Code smell detection (anti-patterns, debug statements)
- Cyclomatic complexity calculation (tree-sitter AST)
- Documentation coverage analysis

Example:
    from victor.tools.language_analyzer import get_analyzer, analyze_file

    # Auto-detect language and analyze
    result = await analyze_file("src/main.py", aspects=["complexity", "security"])

    # Or use specific analyzer
    analyzer = get_analyzer("python")
    issues = analyzer.check_security(code)

    # List all supported languages
    from victor.tools.language_analyzer import supported_languages
    print(supported_languages())  # ['python', 'javascript', 'c', 'ruby', ...]
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Type, runtime_checkable

# Import Rust-accelerated regex engine for 10-20x faster pattern matching
try:
    from victor.native.accelerators import get_regex_engine_accelerator

    _REGEX_ACCELERATOR_AVAILABLE = True
except ImportError:
    _REGEX_ACCELERATOR_AVAILABLE = False
    get_regex_engine_accelerator = None  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AnalysisIssue:
    """Represents a code analysis issue."""

    type: str  # "security", "complexity", "smell", "documentation"
    severity: str  # "critical", "high", "medium", "low"
    issue: str  # Human-readable issue name
    file: str
    line: int
    code: str  # The problematic code snippet
    recommendation: str
    metric: Optional[int] = None  # For complexity issues


@dataclass
class FunctionMetrics:
    """Metrics for a function/method."""

    name: str
    line: int
    end_line: Optional[int]
    complexity: int
    has_docstring: bool
    parameter_count: int
    return_count: int


@dataclass
class AnalysisResult:
    """Result of analyzing a file."""

    file_path: str
    language: str
    issues: List[AnalysisIssue] = field(default_factory=list)
    functions: List[FunctionMetrics] = field(default_factory=list)
    lines_of_code: int = 0
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Security Patterns per Language
# =============================================================================


@dataclass
class SecurityPattern:
    """Security vulnerability pattern."""

    name: str
    pattern: str
    severity: str
    recommendation: str


# Language-specific security patterns
SECURITY_PATTERNS: Dict[str, List[SecurityPattern]] = {
    "python": [
        SecurityPattern(
            "hardcoded_password",
            r"password\s*=\s*['\"][\w]+['\"]",
            "high",
            "Use environment variables or secure vaults",
        ),
        SecurityPattern(
            "hardcoded_key",
            r"(api_key|secret_key|private_key|token)\s*=\s*['\"][\w-]+['\"]",
            "high",
            "Store keys in environment variables or secure key management",
        ),
        SecurityPattern(
            "sql_injection",
            r"(execute|cursor\.execute)\s*\(['\"].*(%s|%d|\{).*['\"]",
            "critical",
            "Use parameterized queries instead of string formatting",
        ),
        SecurityPattern(
            "command_injection",
            r"(os\.system|subprocess\.call|subprocess\.run|eval|exec)\s*\(",
            "critical",
            "Validate and sanitize inputs, use subprocess with list arguments",
        ),
        SecurityPattern(
            "insecure_random",
            r"random\.(random|randint|choice|shuffle)\s*\(",
            "medium",
            "Use secrets module for cryptographic randomness",
        ),
        SecurityPattern(
            "weak_crypto",
            r"(hashlib\.)?(md5|sha1)\s*\(",
            "medium",
            "Use SHA-256 or stronger algorithms",
        ),
        SecurityPattern(
            "pickle_load",
            r"pickle\.(load|loads)\s*\(",
            "high",
            "Avoid pickle for untrusted data; use JSON or safe alternatives",
        ),
    ],
    "javascript": [
        SecurityPattern(
            "eval_usage",
            r"\beval\s*\(",
            "critical",
            "Avoid eval(); use safer alternatives like JSON.parse()",
        ),
        SecurityPattern(
            "innerhtml_xss",
            r"\.innerHTML\s*=",
            "high",
            "Use textContent or sanitize HTML to prevent XSS",
        ),
        SecurityPattern(
            "document_write",
            r"document\.write\s*\(",
            "high",
            "Avoid document.write(); use DOM manipulation methods",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|api_key|token)\s*[=:]\s*['\"][^'\"]+['\"]",
            "high",
            "Use environment variables for secrets",
        ),
        SecurityPattern(
            "sql_concat",
            r"(query|sql)\s*[+=]\s*.*\+",
            "critical",
            "Use parameterized queries to prevent SQL injection",
        ),
        SecurityPattern(
            "shell_exec",
            r"(child_process\.|exec|spawn)\s*\(",
            "high",
            "Validate inputs before executing shell commands",
        ),
        SecurityPattern(
            "unsafe_regex",
            r"new\s+RegExp\s*\([^)]*\+",
            "medium",
            "Avoid dynamic regex from user input (ReDoS risk)",
        ),
    ],
    "typescript": [],  # Inherits from javascript
    "java": [
        SecurityPattern(
            "sql_injection",
            r"(executeQuery|executeUpdate|execute)\s*\([^)]*\+",
            "critical",
            "Use PreparedStatement with parameterized queries",
        ),
        SecurityPattern(
            "command_injection",
            r"Runtime\.getRuntime\(\)\.exec\s*\(",
            "critical",
            "Validate inputs; use ProcessBuilder with argument list",
        ),
        SecurityPattern(
            "hardcoded_password",
            r"(password|passwd|pwd)\s*=\s*\"[^\"]+\"",
            "high",
            "Use environment variables or secure credential storage",
        ),
        SecurityPattern(
            "weak_crypto",
            r"(MD5|SHA-1|DES|RC4)",
            "medium",
            "Use SHA-256 or AES for cryptographic operations",
        ),
        SecurityPattern(
            "xxe_vulnerability",
            r"DocumentBuilderFactory|SAXParserFactory|XMLInputFactory",
            "high",
            "Disable external entities in XML parsers",
        ),
        SecurityPattern(
            "deserialize_untrusted",
            r"ObjectInputStream\s*\(",
            "critical",
            "Avoid deserializing untrusted data; use JSON/allowlists",
        ),
    ],
    "go": [
        SecurityPattern(
            "command_injection",
            r"exec\.Command\s*\([^)]*\+",
            "critical",
            "Avoid string concatenation in exec.Command arguments",
        ),
        SecurityPattern(
            "sql_injection",
            r"(Query|Exec)\s*\([^)]*\+",
            "critical",
            "Use parameterized queries with $1, $2 placeholders",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|apiKey|token)\s*:?=\s*\"[^\"]+\"",
            "high",
            "Use environment variables or secret management",
        ),
        SecurityPattern(
            "weak_crypto",
            r"(md5|sha1)\.New\(\)",
            "medium",
            "Use sha256 or stronger hash algorithms",
        ),
        SecurityPattern(
            "tls_skip_verify",
            r"InsecureSkipVerify:\s*true",
            "high",
            "Never skip TLS verification in production",
        ),
    ],
    "rust": [
        SecurityPattern(
            "unsafe_block",
            r"\bunsafe\s*\{",
            "medium",
            "Minimize unsafe blocks; document safety invariants",
        ),
        SecurityPattern(
            "unwrap_usage",
            r"\.(unwrap|expect)\s*\(",
            "low",
            "Use proper error handling instead of unwrap in production",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|api_key|token)\s*=\s*\"[^\"]+\"",
            "high",
            "Use environment variables for secrets",
        ),
        SecurityPattern(
            "sql_format",
            r"format!\s*\([^)]*SELECT|INSERT|UPDATE|DELETE",
            "critical",
            "Use parameterized queries with sqlx or diesel",
        ),
    ],
    # C Language
    "c": [
        SecurityPattern(
            "buffer_overflow",
            r"\b(strcpy|strcat|gets|sprintf|scanf)\s*\(",
            "critical",
            "Use strncpy, strncat, fgets, snprintf, or sscanf with size limits",
        ),
        SecurityPattern(
            "format_string",
            r"(printf|fprintf|sprintf|snprintf)\s*\([^,]*\)",
            "critical",
            "Always use format string; never pass user input directly",
        ),
        SecurityPattern(
            "system_call",
            r"\bsystem\s*\(",
            "high",
            "Validate all inputs; prefer exec family with explicit arguments",
        ),
        SecurityPattern(
            "hardcoded_password",
            r"(password|passwd|pwd)\s*=\s*\"[^\"]+\"",
            "high",
            "Use environment variables or secure storage",
        ),
        SecurityPattern(
            "integer_overflow",
            r"malloc\s*\([^)]*\*[^)]*\)",
            "high",
            "Check for integer overflow before multiplication in malloc",
        ),
    ],
    # C++ Language
    "cpp": [
        SecurityPattern(
            "buffer_overflow",
            r"\b(strcpy|strcat|gets|sprintf|scanf)\s*\(",
            "critical",
            "Use std::string or safe C++ alternatives",
        ),
        SecurityPattern(
            "raw_pointer",
            r"\bnew\s+\w+(?!\s*\[)",
            "medium",
            "Prefer smart pointers (unique_ptr, shared_ptr) over raw new",
        ),
        SecurityPattern(
            "system_call",
            r"\bsystem\s*\(",
            "high",
            "Avoid system(); use safer alternatives",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|api_key|token)\s*=\s*\"[^\"]+\"",
            "high",
            "Use environment variables for secrets",
        ),
        SecurityPattern(
            "unchecked_cast",
            r"\breinterpret_cast\s*<",
            "medium",
            "Avoid reinterpret_cast; use static_cast or dynamic_cast",
        ),
    ],
    # C# Language
    "c_sharp": [
        SecurityPattern(
            "sql_injection",
            r"(SqlCommand|ExecuteReader|ExecuteNonQuery)\s*\([^)]*\+",
            "critical",
            "Use parameterized queries with SqlParameter",
        ),
        SecurityPattern(
            "command_injection",
            r"Process\.Start\s*\(",
            "high",
            "Validate inputs; avoid shell=true patterns",
        ),
        SecurityPattern(
            "hardcoded_password",
            r"(password|connectionString)\s*=\s*\"[^\"]+\"",
            "high",
            "Use configuration files or Azure Key Vault",
        ),
        SecurityPattern(
            "weak_crypto",
            r"(MD5|SHA1|DES)\.",
            "medium",
            "Use SHA256 or AES for cryptographic operations",
        ),
        SecurityPattern(
            "deserialize_untrusted",
            r"(BinaryFormatter|XmlSerializer)\.Deserialize",
            "critical",
            "Avoid deserializing untrusted data; use JSON with type checking",
        ),
    ],
    # Ruby Language
    "ruby": [
        SecurityPattern(
            "eval_usage",
            r"\b(eval|instance_eval|class_eval)\s*[\(\s]",
            "critical",
            "Avoid eval; use safer alternatives",
        ),
        SecurityPattern(
            "command_injection",
            r"(`[^`]*`|system\s*\(|exec\s*\(|%x\{)",
            "critical",
            "Validate inputs; use array form for system commands",
        ),
        SecurityPattern(
            "sql_injection",
            r"(find_by_sql|execute)\s*\([^)]*#\{",
            "critical",
            "Use parameterized queries with ActiveRecord",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]",
            "high",
            "Use environment variables or Rails credentials",
        ),
        SecurityPattern(
            "mass_assignment",
            r"\.create\s*\(params\[",
            "high",
            "Use strong parameters; whitelist permitted attributes",
        ),
    ],
    # PHP Language
    "php": [
        SecurityPattern(
            "sql_injection",
            r"(mysql_query|mysqli_query|->query)\s*\([^)]*\$",
            "critical",
            "Use prepared statements with PDO or mysqli",
        ),
        SecurityPattern(
            "command_injection",
            r"\b(exec|system|passthru|shell_exec|popen)\s*\(",
            "critical",
            "Validate inputs; use escapeshellarg/escapeshellcmd",
        ),
        SecurityPattern(
            "eval_usage",
            r"\b(eval|assert|create_function|preg_replace.*/e)\s*\(",
            "critical",
            "Avoid eval and related functions",
        ),
        SecurityPattern(
            "file_inclusion",
            r"\b(include|require|include_once|require_once)\s*\(\$",
            "critical",
            "Never include files based on user input; use whitelist",
        ),
        SecurityPattern(
            "xss_vulnerable",
            r"echo\s+\$_(GET|POST|REQUEST|COOKIE)",
            "high",
            "Always use htmlspecialchars() for output",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|api_key)\s*=\s*['\"][^'\"]+['\"]",
            "high",
            "Use environment variables or config files outside webroot",
        ),
    ],
    # Kotlin Language
    "kotlin": [
        SecurityPattern(
            "sql_injection",
            r"(rawQuery|execSQL)\s*\([^)]*\$",
            "critical",
            "Use parameterized queries",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|apiKey|token)\s*=\s*\"[^\"]+\"",
            "high",
            "Use BuildConfig or encrypted preferences",
        ),
        SecurityPattern(
            "insecure_http",
            r"http://[^\s\"']+",
            "medium",
            "Use HTTPS for all network requests",
        ),
        SecurityPattern(
            "webview_javascript",
            r"setJavaScriptEnabled\s*\(\s*true\s*\)",
            "high",
            "Be cautious with JavaScript in WebViews; validate URLs",
        ),
    ],
    # Swift Language
    "swift": [
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|apiKey|token)\s*=\s*\"[^\"]+\"",
            "high",
            "Use Keychain for secrets; never hardcode",
        ),
        SecurityPattern(
            "force_unwrap",
            r"[a-zA-Z_]\w*!(?!\s*=)",
            "low",
            "Avoid force unwrap; use optional binding or guard",
        ),
        SecurityPattern(
            "insecure_http",
            r"http://[^\s\"']+",
            "medium",
            "Use HTTPS; configure App Transport Security",
        ),
        SecurityPattern(
            "sql_injection",
            r"(executeQuery|executeUpdate)\s*\([^)]*\\",
            "critical",
            "Use parameterized queries with SQLite.swift",
        ),
    ],
    # Scala Language
    "scala": [
        SecurityPattern(
            "sql_injection",
            r"sql\"[^\"]*\$",
            "critical",
            "Use parameterized queries with Slick or Doobie",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|apiKey|token)\s*=\s*\"[^\"]+\"",
            "high",
            "Use Typesafe Config or environment variables",
        ),
        SecurityPattern(
            "command_injection",
            r"(Runtime\.exec|Process\.|sys\.process)",
            "high",
            "Validate all inputs to shell commands",
        ),
    ],
    # Bash/Shell Script
    "bash": [
        SecurityPattern(
            "command_injection",
            r"\$\([^)]*\$",
            "critical",
            "Quote variables; validate user input",
        ),
        SecurityPattern(
            "eval_usage",
            r"\beval\s+",
            "critical",
            "Avoid eval; use arrays and proper quoting",
        ),
        SecurityPattern(
            "unquoted_variable",
            r"\$\w+(?!['\"])",
            "medium",
            'Always quote variables: "$var" instead of $var',
        ),
        SecurityPattern(
            "hardcoded_password",
            r"(PASSWORD|SECRET|API_KEY|TOKEN)=['\"]?[a-zA-Z0-9]+",
            "high",
            "Use environment variables; never commit secrets",
        ),
        SecurityPattern(
            "temp_file_race",
            r"/tmp/[a-zA-Z]+",
            "medium",
            "Use mktemp for secure temporary files",
        ),
    ],
    # SQL (patterns are for detecting issues in SQL scripts)
    "sql": [
        SecurityPattern(
            "grant_all",
            r"GRANT\s+ALL",
            "high",
            "Grant only necessary permissions; follow least privilege",
        ),
        SecurityPattern(
            "no_password",
            r"IDENTIFIED\s+BY\s*''",
            "critical",
            "Always set strong passwords for database users",
        ),
        SecurityPattern(
            "drop_without_where",
            r"DELETE\s+FROM\s+\w+\s*;",
            "high",
            "Always use WHERE clause with DELETE statements",
        ),
        SecurityPattern(
            "public_schema",
            r"GRANT\s+.*\s+TO\s+public",
            "medium",
            "Avoid granting permissions to public; use specific roles",
        ),
    ],
    # Lua Language
    "lua": [
        SecurityPattern(
            "loadstring_usage",
            r"\b(loadstring|load)\s*\(",
            "critical",
            "Avoid loadstring; it enables code injection",
        ),
        SecurityPattern(
            "os_execute",
            r"os\.(execute|popen)\s*\(",
            "high",
            "Validate inputs to shell commands",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]",
            "high",
            "Use environment variables for secrets",
        ),
    ],
    # Elixir Language
    "elixir": [
        SecurityPattern(
            "code_eval",
            r"Code\.(eval_string|eval_quoted)\s*\(",
            "critical",
            "Avoid Code.eval_*; use pattern matching instead",
        ),
        SecurityPattern(
            "sql_injection",
            r"(from|where)\s*\([^)]*\#\{",
            "critical",
            "Use Ecto parameterized queries",
        ),
        SecurityPattern(
            "hardcoded_secret",
            r"(password|secret|api_key|token):\s*\"[^\"]+\"",
            "high",
            "Use environment variables or runtime config",
        ),
    ],
}

# TypeScript inherits JavaScript patterns
SECURITY_PATTERNS["typescript"] = SECURITY_PATTERNS["javascript"].copy()
# C++ inherits some C patterns
SECURITY_PATTERNS["cpp"] = SECURITY_PATTERNS["c"] + SECURITY_PATTERNS["cpp"]


# =============================================================================
# Code Smell Patterns per Language
# =============================================================================


CODE_SMELL_PATTERNS: Dict[str, List[SecurityPattern]] = {
    "python": [
        SecurityPattern(
            "print_debug",
            r"\bprint\s*\(",
            "low",
            "Use logging module instead of print statements",
        ),
        SecurityPattern(
            "bare_except",
            r"\bexcept\s*:",
            "medium",
            "Catch specific exceptions instead of bare except",
        ),
        SecurityPattern(
            "global_variable",
            r"^\s*global\s+\w+",
            "medium",
            "Avoid global variables; use function parameters",
        ),
        SecurityPattern(
            "star_import",
            r"from\s+\w+\s+import\s+\*",
            "low",
            "Avoid wildcard imports; import specific names",
        ),
        SecurityPattern(
            "mutable_default",
            r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\}|\set\(\))",
            "medium",
            "Avoid mutable default arguments; use None and initialize inside",
        ),
    ],
    "javascript": [
        SecurityPattern(
            "console_log",
            r"\bconsole\.(log|debug|info)\s*\(",
            "low",
            "Remove console.log in production; use proper logging",
        ),
        SecurityPattern(
            "var_usage",
            r"\bvar\s+\w+",
            "low",
            "Use const or let instead of var",
        ),
        SecurityPattern(
            "triple_equals",
            r"[^=!]==[^=]",
            "low",
            "Use === for strict equality comparison",
        ),
        SecurityPattern(
            "callback_hell",
            r"function\s*\([^)]*\)\s*\{[^}]*function\s*\(",
            "medium",
            "Consider async/await or Promise chains",
        ),
    ],
    "typescript": [],  # Inherits from javascript
    "java": [
        SecurityPattern(
            "system_out",
            r"System\.(out|err)\.print",
            "low",
            "Use logging framework instead of System.out",
        ),
        SecurityPattern(
            "empty_catch",
            r"catch\s*\([^)]+\)\s*\{\s*\}",
            "medium",
            "Don't swallow exceptions; log or rethrow",
        ),
        SecurityPattern(
            "string_concat_loop",
            r"for\s*\([^)]+\)[^{]*\{[^}]*\+=\s*\"",
            "medium",
            "Use StringBuilder in loops for string concatenation",
        ),
    ],
    "go": [
        SecurityPattern(
            "fmt_print",
            r"fmt\.Print(ln|f)?\s*\(",
            "low",
            "Use structured logging instead of fmt.Print",
        ),
        SecurityPattern(
            "ignored_error",
            r",\s*_\s*:?=\s*\w+\(",
            "medium",
            "Don't ignore errors; handle or propagate them",
        ),
        SecurityPattern(
            "panic_usage",
            r"\bpanic\s*\(",
            "medium",
            "Avoid panic; return errors instead",
        ),
    ],
    "rust": [
        SecurityPattern(
            "println_debug",
            r"\bprintln!\s*\(",
            "low",
            "Use tracing or log crate instead of println!",
        ),
        SecurityPattern(
            "clone_overuse",
            r"\.clone\(\)",
            "low",
            "Consider borrowing instead of cloning where possible",
        ),
    ],
}

# TypeScript inherits JavaScript patterns
CODE_SMELL_PATTERNS["typescript"] = CODE_SMELL_PATTERNS["javascript"].copy()

# C Language code smells
CODE_SMELL_PATTERNS["c"] = [
    SecurityPattern(
        "printf_debug",
        r"\bprintf\s*\(",
        "low",
        "Use proper logging or remove debug prints",
    ),
    SecurityPattern(
        "magic_number",
        r"\b(?:if|while|for)\s*\([^)]*\b\d{2,}\b",
        "low",
        "Define named constants for magic numbers",
    ),
    SecurityPattern(
        "goto_usage",
        r"\bgoto\s+\w+",
        "medium",
        "Avoid goto; use structured control flow",
    ),
]

# C++ Language code smells
CODE_SMELL_PATTERNS["cpp"] = CODE_SMELL_PATTERNS["c"] + [
    SecurityPattern(
        "cout_debug",
        r"\bstd::cout\s*<<",
        "low",
        "Use proper logging framework instead of cout",
    ),
    SecurityPattern(
        "using_namespace_std",
        r"using\s+namespace\s+std",
        "low",
        "Avoid 'using namespace std'; prefer explicit std::",
    ),
    SecurityPattern(
        "raw_array",
        r"\w+\s+\w+\s*\[\s*\d+\s*\]",
        "low",
        "Consider std::array or std::vector instead of raw arrays",
    ),
]

# C# Language code smells
CODE_SMELL_PATTERNS["c_sharp"] = [
    SecurityPattern(
        "console_write",
        r"Console\.(Write|WriteLine)\s*\(",
        "low",
        "Use logging framework instead of Console.Write",
    ),
    SecurityPattern(
        "empty_catch",
        r"catch\s*\([^)]*\)\s*\{\s*\}",
        "medium",
        "Don't swallow exceptions; log or handle appropriately",
    ),
    SecurityPattern(
        "public_field",
        r"public\s+\w+\s+\w+\s*;",
        "low",
        "Use properties instead of public fields",
    ),
    SecurityPattern(
        "magic_string",
        r"if\s*\([^)]*==\s*\"[^\"]+\"",
        "low",
        "Use constants for magic strings",
    ),
]

# Ruby Language code smells
CODE_SMELL_PATTERNS["ruby"] = [
    SecurityPattern(
        "puts_debug",
        r"\b(puts|p|pp)\s+",
        "low",
        "Use Logger or Rails.logger instead of puts",
    ),
    SecurityPattern(
        "rescue_all",
        r"rescue\s*$|rescue\s+Exception",
        "medium",
        "Rescue specific exceptions, not all",
    ),
    SecurityPattern(
        "class_variable",
        r"@@\w+",
        "medium",
        "Avoid class variables; use class instance variables",
    ),
    SecurityPattern(
        "unless_else",
        r"\bunless\s+.+\n\s*else",
        "low",
        "Avoid unless with else; use if instead",
    ),
]

# PHP Language code smells
CODE_SMELL_PATTERNS["php"] = [
    SecurityPattern(
        "echo_debug",
        r"\b(echo|print)\s+",
        "low",
        "Use proper logging instead of echo/print",
    ),
    SecurityPattern(
        "var_dump",
        r"\b(var_dump|print_r|die)\s*\(",
        "low",
        "Remove debug functions in production code",
    ),
    SecurityPattern(
        "global_variable",
        r"\bglobal\s+\$",
        "medium",
        "Avoid global variables; use dependency injection",
    ),
    SecurityPattern(
        "short_open_tag",
        r"<\?(?!php|xml)",
        "low",
        "Use full <?php tag for compatibility",
    ),
]

# Kotlin Language code smells
CODE_SMELL_PATTERNS["kotlin"] = [
    SecurityPattern(
        "println_debug",
        r"\bprintln\s*\(",
        "low",
        "Use logging framework instead of println",
    ),
    SecurityPattern(
        "force_cast",
        r"\bas\s+\w+(?!\?)",
        "medium",
        "Use safe cast (as?) instead of forced cast",
    ),
    SecurityPattern(
        "bang_operator",
        r"!!\s*$|!!\.",
        "low",
        "Avoid !! operator; use safe calls or elvis",
    ),
]

# Swift Language code smells
CODE_SMELL_PATTERNS["swift"] = [
    SecurityPattern(
        "print_debug",
        r"\bprint\s*\(",
        "low",
        "Use OSLog or proper logging instead of print",
    ),
    SecurityPattern(
        "force_try",
        r"\btry!\s+",
        "medium",
        "Use do-catch or try? instead of try!",
    ),
    SecurityPattern(
        "implicitly_unwrapped",
        r"var\s+\w+\s*:\s*\w+!",
        "low",
        "Avoid implicitly unwrapped optionals where possible",
    ),
]

# Scala Language code smells
CODE_SMELL_PATTERNS["scala"] = [
    SecurityPattern(
        "println_debug",
        r"\bprintln\s*\(",
        "low",
        "Use logging framework instead of println",
    ),
    SecurityPattern(
        "var_usage",
        r"\bvar\s+\w+",
        "low",
        "Prefer val (immutable) over var where possible",
    ),
    SecurityPattern(
        "null_usage",
        r"\bnull\b",
        "medium",
        "Use Option instead of null",
    ),
    SecurityPattern(
        "return_statement",
        r"\breturn\s+",
        "low",
        "Avoid explicit return in Scala; use expression results",
    ),
]

# Bash/Shell code smells
CODE_SMELL_PATTERNS["bash"] = [
    SecurityPattern(
        "echo_debug",
        r"^\s*echo\s+['\"]debug|^\s*echo\s+\$",
        "low",
        "Use proper logging or remove debug echoes",
    ),
    SecurityPattern(
        "no_set_e",
        r"^#!.*bash",
        "low",
        "Add 'set -euo pipefail' for safer scripts",
    ),
    SecurityPattern(
        "backtick_command",
        r"`[^`]+`",
        "low",
        "Use $(command) instead of backticks for clarity",
    ),
]

# SQL code smells
CODE_SMELL_PATTERNS["sql"] = [
    SecurityPattern(
        "select_star",
        r"SELECT\s+\*\s+FROM",
        "low",
        "Specify column names instead of SELECT *",
    ),
    SecurityPattern(
        "no_alias",
        r"JOIN\s+\w+\s+ON\s+\w+\.",
        "low",
        "Use table aliases for readability in JOINs",
    ),
    SecurityPattern(
        "implicit_join",
        r"FROM\s+\w+\s*,\s*\w+",
        "low",
        "Use explicit JOIN syntax instead of comma joins",
    ),
]

# Lua code smells
CODE_SMELL_PATTERNS["lua"] = [
    SecurityPattern(
        "print_debug",
        r"\bprint\s*\(",
        "low",
        "Use proper logging instead of print",
    ),
    SecurityPattern(
        "global_variable",
        r"^\s*\w+\s*=\s*[^=]",
        "medium",
        "Use local variables; globals are implicit",
    ),
]

# Elixir code smells
CODE_SMELL_PATTERNS["elixir"] = [
    SecurityPattern(
        "io_inspect",
        r"\bIO\.(inspect|puts)\s*\(",
        "low",
        "Use Logger instead of IO.inspect in production",
    ),
    SecurityPattern(
        "nested_case",
        r"case\s+[^d]+do\s*\n\s*\w+\s*->\s*\n\s*case",
        "medium",
        "Avoid deeply nested case; use with or pattern matching",
    ),
]

# Haskell code smells
CODE_SMELL_PATTERNS["haskell"] = [
    SecurityPattern(
        "putstrln_debug",
        r"\bputStrLn\s+",
        "low",
        "Use proper logging instead of putStrLn",
    ),
    SecurityPattern(
        "head_tail",
        r"\b(head|tail|init|last)\s+",
        "medium",
        "Use pattern matching instead of partial functions",
    ),
]

# R code smells
CODE_SMELL_PATTERNS["r"] = [
    SecurityPattern(
        "print_debug",
        r"\b(print|cat)\s*\(",
        "low",
        "Use proper logging in production code",
    ),
    SecurityPattern(
        "attach_usage",
        r"\battach\s*\(",
        "medium",
        "Avoid attach(); use explicit data frame references",
    ),
]


# =============================================================================
# File Extension to Language Mapping
# =============================================================================


EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    # Python
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    # JavaScript
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "typescript",
    # Java
    ".java": "java",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # C
    ".c": "c",
    ".h": "c",
    # C++
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    # C#
    ".cs": "c_sharp",
    # Ruby
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    # PHP
    ".php": "php",
    ".phtml": "php",
    ".php3": "php",
    ".php4": "php",
    ".php5": "php",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",
    # Swift
    ".swift": "swift",
    # Scala
    ".scala": "scala",
    ".sc": "scala",
    # Bash/Shell
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ksh": "bash",
    # SQL
    ".sql": "sql",
    ".ddl": "sql",
    ".dml": "sql",
    # Lua
    ".lua": "lua",
    # Elixir
    ".ex": "elixir",
    ".exs": "elixir",
    # Haskell
    ".hs": "haskell",
    ".lhs": "haskell",
    # R
    ".r": "r",
    ".R": "r",
    ".rmd": "r",
    ".Rmd": "r",
}

# Glob patterns per language
LANGUAGE_GLOB_PATTERNS: Dict[str, str] = {
    "python": "*.py",
    "javascript": "*.{js,mjs,cjs,jsx}",
    "typescript": "*.{ts,tsx}",
    "java": "*.java",
    "go": "*.go",
    "rust": "*.rs",
    "c": "*.{c,h}",
    "cpp": "*.{cpp,cc,cxx,hpp,hh,hxx}",
    "c_sharp": "*.cs",
    "ruby": "*.{rb,rake,gemspec}",
    "php": "*.{php,phtml}",
    "kotlin": "*.{kt,kts}",
    "swift": "*.swift",
    "scala": "*.{scala,sc}",
    "bash": "*.{sh,bash,zsh}",
    "sql": "*.sql",
    "lua": "*.lua",
    "elixir": "*.{ex,exs}",
    "haskell": "*.{hs,lhs}",
    "r": "*.{r,R,rmd,Rmd}",
    "all": "*",  # Special: all supported languages
}


# =============================================================================
# Tree-sitter Query Patterns per Language
# =============================================================================

# Queries to find functions/methods for complexity analysis
FUNCTION_QUERIES: Dict[str, str] = {
    "python": """
        (function_definition
            name: (identifier) @name
        ) @function
        (class_definition
            body: (block
                (function_definition
                    name: (identifier) @method_name
                ) @method
            )
        )
    """,
    "javascript": """
        (function_declaration
            name: (identifier) @name
        ) @function
        (arrow_function) @arrow
        (method_definition
            name: (property_identifier) @method_name
        ) @method
    """,
    "typescript": """
        (function_declaration
            name: (identifier) @name
        ) @function
        (arrow_function) @arrow
        (method_definition
            name: (property_identifier) @method_name
        ) @method
    """,
    "java": """
        (method_declaration
            name: (identifier) @name
        ) @method
        (constructor_declaration
            name: (identifier) @constructor_name
        ) @constructor
    """,
    "go": """
        (function_declaration
            name: (identifier) @name
        ) @function
        (method_declaration
            name: (field_identifier) @method_name
        ) @method
    """,
    "rust": """
        (function_item
            name: (identifier) @name
        ) @function
        (impl_item
            body: (declaration_list
                (function_item
                    name: (identifier) @method_name
                ) @method
            )
        )
    """,
}

# Control flow nodes that increase complexity (per language)
COMPLEXITY_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {
        "if_statement",
        "elif_clause",
        "while_statement",
        "for_statement",
        "except_clause",
        "with_statement",
        "assert_statement",
        "boolean_operator",
        "conditional_expression",
        "list_comprehension",
        "dictionary_comprehension",
        "set_comprehension",
        "generator_expression",
    },
    "javascript": {
        "if_statement",
        "else_clause",
        "while_statement",
        "for_statement",
        "for_in_statement",
        "do_statement",
        "switch_case",
        "catch_clause",
        "ternary_expression",
        "binary_expression",  # && and || operators
    },
    "typescript": {
        "if_statement",
        "else_clause",
        "while_statement",
        "for_statement",
        "for_in_statement",
        "do_statement",
        "switch_case",
        "catch_clause",
        "ternary_expression",
        "binary_expression",
    },
    "java": {
        "if_statement",
        "while_statement",
        "for_statement",
        "enhanced_for_statement",
        "do_statement",
        "switch_expression",
        "catch_clause",
        "ternary_expression",
        "binary_expression",
    },
    "go": {
        "if_statement",
        "for_statement",
        "expression_switch_statement",
        "type_switch_statement",
        "select_statement",
        "binary_expression",
    },
    "rust": {
        "if_expression",
        "while_expression",
        "for_expression",
        "loop_expression",
        "match_expression",
        "match_arm",
        "binary_expression",
    },
    "c": {
        "if_statement",
        "while_statement",
        "for_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "conditional_expression",
        "binary_expression",
    },
    "cpp": {
        "if_statement",
        "while_statement",
        "for_statement",
        "for_range_loop",
        "do_statement",
        "switch_statement",
        "case_statement",
        "catch_clause",
        "conditional_expression",
        "binary_expression",
    },
    "c_sharp": {
        "if_statement",
        "while_statement",
        "for_statement",
        "foreach_statement",
        "do_statement",
        "switch_statement",
        "switch_section",
        "catch_clause",
        "conditional_expression",
        "binary_expression",
    },
    "ruby": {
        "if",
        "unless",
        "while",
        "until",
        "for",
        "case",
        "when",
        "rescue",
        "conditional",
        "binary",
    },
    "php": {
        "if_statement",
        "while_statement",
        "for_statement",
        "foreach_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "catch_clause",
        "conditional_expression",
        "binary_expression",
    },
    "kotlin": {
        "if_expression",
        "while_statement",
        "for_statement",
        "do_while_statement",
        "when_expression",
        "when_entry",
        "catch_block",
        "elvis_expression",
        "conjunction",
        "disjunction",
    },
    "swift": {
        "if_statement",
        "while_statement",
        "for_in_statement",
        "repeat_while_statement",
        "switch_statement",
        "switch_case",
        "guard_statement",
        "catch_clause",
        "ternary_expression",
    },
    "scala": {
        "if_expression",
        "while_expression",
        "for_expression",
        "match_expression",
        "case_clause",
        "catch_clause",
        "infix_expression",
    },
    "bash": {
        "if_statement",
        "while_statement",
        "for_statement",
        "case_statement",
        "case_item",
        "elif_clause",
        "binary_expression",
    },
    "sql": set(),  # SQL doesn't have typical control flow
    "lua": {
        "if_statement",
        "while_statement",
        "for_statement",
        "for_in_statement",
        "repeat_statement",
        "binary_expression",
    },
    "elixir": {
        "if",
        "unless",
        "case",
        "cond",
        "with",
        "binary_operator",
    },
    "haskell": {
        "if",
        "case",
        "guard",
        "lambda",
        "infix_application",
    },
    "r": {
        "if",
        "while",
        "for",
        "repeat",
        "binary",
    },
}


# =============================================================================
# Language Analyzer Protocol
# =============================================================================


@runtime_checkable
class LanguageAnalyzer(Protocol):
    """Protocol defining the language analysis interface."""

    @property
    def language(self) -> str:
        """Return the language name."""
        ...

    @property
    def file_extensions(self) -> List[str]:
        """Return supported file extensions."""
        ...

    def check_security(self, content: str, file_path: Path) -> List[AnalysisIssue]:
        """Check for security vulnerabilities."""
        ...

    def check_code_smells(self, content: str, file_path: Path) -> List[AnalysisIssue]:
        """Check for code smells and anti-patterns."""
        ...

    def calculate_complexity(
        self, content: str, file_path: Path
    ) -> Tuple[List[AnalysisIssue], List[FunctionMetrics]]:
        """Calculate cyclomatic complexity for functions."""
        ...

    def check_documentation(self, content: str, file_path: Path) -> List[AnalysisIssue]:
        """Check documentation coverage."""
        ...

    def analyze(self, content: str, file_path: Path, aspects: List[str]) -> AnalysisResult:
        """Run full analysis on content."""
        ...


# =============================================================================
# Base Language Analyzer Implementation
# =============================================================================


class BaseLanguageAnalyzer(ABC):
    """Base implementation of language analyzer with tree-sitter support.

    Subclasses can override specific methods for language-specific behavior
    while inheriting common functionality.
    """

    def __init__(self, max_complexity: int = 10):
        """Initialize the analyzer.

        Args:
            max_complexity: Maximum allowed cyclomatic complexity (default: 10)
        """
        self.max_complexity = max_complexity
        self._parser = None
        self._language_obj = None

        # Initialize regex accelerator for 10-20x faster pattern matching
        if _REGEX_ACCELERATOR_AVAILABLE and get_regex_engine_accelerator is not None:
            self._regex_accelerator = get_regex_engine_accelerator()
            if self._regex_accelerator.rust_available:
                logger.info(
                    f"Language analyzer ({self.__class__.__name__}): Using Rust accelerator "
                    "(10-20x faster pattern matching)"
                )
        else:
            self._regex_accelerator = None

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language name."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Return supported file extensions."""
        pass

    def _get_parser(self):
        """Lazy-load tree-sitter parser for this language."""
        if self._parser is None:
            try:
                from victor.coding.codebase.tree_sitter_manager import get_parser

                self._parser = get_parser(self.language)
            except (ImportError, ValueError) as e:
                logger.warning(
                    f"Tree-sitter not available for {self.language}: {e}. "
                    "Falling back to regex-only analysis."
                )
                self._parser = None
        return self._parser

    def _get_security_patterns(self) -> List[SecurityPattern]:
        """Get security patterns for this language."""
        return SECURITY_PATTERNS.get(self.language, [])

    def _get_smell_patterns(self) -> List[SecurityPattern]:
        """Get code smell patterns for this language."""
        return CODE_SMELL_PATTERNS.get(self.language, [])

    def check_security(self, content: str, file_path: Path) -> List[AnalysisIssue]:
        """Check for security vulnerabilities using regex patterns."""
        issues = []
        lines = content.split("\n")
        patterns = self._get_security_patterns()

        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern.pattern, line, re.IGNORECASE):
                    issues.append(
                        AnalysisIssue(
                            type="security",
                            severity=pattern.severity,
                            issue=pattern.name.replace("_", " ").title(),
                            file=str(file_path),
                            line=line_num,
                            code=line.strip()[:100],
                            recommendation=pattern.recommendation,
                        )
                    )

        return issues

    def check_code_smells(self, content: str, file_path: Path) -> List[AnalysisIssue]:
        """Check for code smells using regex patterns."""
        issues = []
        lines = content.split("\n")
        patterns = self._get_smell_patterns()

        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern.pattern, line, re.IGNORECASE):
                    issues.append(
                        AnalysisIssue(
                            type="smell",
                            severity=pattern.severity,
                            issue=pattern.name.replace("_", " ").title(),
                            file=str(file_path),
                            line=line_num,
                            code=line.strip()[:100],
                            recommendation=pattern.recommendation,
                        )
                    )

        return issues

    def calculate_complexity(
        self, content: str, file_path: Path
    ) -> Tuple[List[AnalysisIssue], List[FunctionMetrics]]:
        """Calculate cyclomatic complexity using tree-sitter.

        Falls back to regex-based estimation if tree-sitter unavailable.
        """
        issues = []
        functions = []

        parser = self._get_parser()
        if parser is None:
            # Fallback: estimate complexity from regex
            return self._estimate_complexity_regex(content, file_path)

        try:
            tree = parser.parse(bytes(content, "utf8"))
            root = tree.root_node

            # Find all functions
            func_nodes = self._find_function_nodes(root)

            for func_name, func_node in func_nodes:
                complexity = self._calculate_node_complexity(func_node)
                has_doc = self._has_docstring(func_node, content)
                param_count = self._count_parameters(func_node)

                metrics = FunctionMetrics(
                    name=func_name,
                    line=func_node.start_point[0] + 1,
                    end_line=func_node.end_point[0] + 1,
                    complexity=complexity,
                    has_docstring=has_doc,
                    parameter_count=param_count,
                    return_count=self._count_returns(func_node),
                )
                functions.append(metrics)

                if complexity > self.max_complexity:
                    issues.append(
                        AnalysisIssue(
                            type="complexity",
                            severity="high" if complexity > 20 else "medium",
                            issue=f"High Complexity: {func_name}",
                            file=str(file_path),
                            line=metrics.line,
                            code=f"Function '{func_name}' has complexity {complexity}",
                            recommendation=f"Refactor to reduce complexity below {self.max_complexity}",
                            metric=complexity,
                        )
                    )

        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
            return self._estimate_complexity_regex(content, file_path)

        return issues, functions

    def _find_function_nodes(self, root) -> List[Tuple[str, Any]]:
        """Find all function/method nodes in the AST.

        Override in subclasses for language-specific traversal.
        """
        results = []

        def visit(node):
            # Check if this is a function node
            if self._is_function_node(node):
                name = self._get_function_name(node)
                if name:
                    results.append((name, node))

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(root)
        return results

    def _is_function_node(self, node) -> bool:
        """Check if node is a function definition.

        Override in subclasses for language-specific logic.
        """
        func_types = {
            "python": {"function_definition"},
            "javascript": {"function_declaration", "arrow_function", "method_definition"},
            "typescript": {"function_declaration", "arrow_function", "method_definition"},
            "java": {"method_declaration", "constructor_declaration"},
            "go": {"function_declaration", "method_declaration"},
            "rust": {"function_item"},
            "c": {"function_definition"},
            "cpp": {"function_definition"},
            "c_sharp": {"method_declaration", "constructor_declaration"},
            "ruby": {"method", "singleton_method"},
            "php": {"function_definition", "method_declaration"},
            "kotlin": {"function_declaration"},
            "swift": {"function_declaration", "subscript_declaration"},
            "scala": {"function_definition", "function_declaration"},
            "bash": {"function_definition"},
            "sql": {"create_function_statement", "create_procedure_statement"},
            "lua": {"function_declaration", "local_function_declaration"},
            "elixir": {"call"},  # def/defp calls
            "haskell": {"function"},
            "r": {"function_definition"},
        }
        return node.type in func_types.get(self.language, set())

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from node.

        Override in subclasses for language-specific logic.
        """
        for child in node.children:
            if child.type == "identifier" or child.type == "property_identifier":
                return child.text.decode("utf8")
            if child.type == "field_identifier":  # Go methods
                return child.text.decode("utf8")
        return None

    def _calculate_node_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a function node."""
        complexity = 1  # Base complexity
        complexity_types = COMPLEXITY_NODE_TYPES.get(self.language, set())

        def visit(n):
            nonlocal complexity
            if n.type in complexity_types:
                complexity += 1
                # Boolean operators add complexity
                if n.type == "binary_expression" or n.type == "boolean_operator":
                    # Check for && or || operators
                    for child in n.children:
                        if child.type in {"&&", "||", "and", "or"}:
                            complexity += 1
            for child in n.children:
                visit(child)

        visit(node)
        return complexity

    def _has_docstring(self, func_node, content: str) -> bool:
        """Check if function has a docstring.

        Override in subclasses for language-specific logic.
        """
        # Default: look for string literal as first statement
        for child in func_node.children:
            if child.type == "block" or child.type == "statement_block":
                for stmt in child.children:
                    if stmt.type in {"expression_statement", "string"}:
                        text = stmt.text.decode("utf8") if hasattr(stmt, "text") else ""
                        return text.startswith(('"""', "'''", '"', "'"))
                    elif stmt.type not in {"comment", "{", "}"}:
                        break
        return False

    def _count_parameters(self, func_node) -> int:
        """Count function parameters."""
        for child in func_node.children:
            if child.type in {"parameters", "formal_parameters", "parameter_list"}:
                return sum(1 for c in child.children if c.type not in {"(", ")", ",", "self"})
        return 0

    def _count_returns(self, func_node) -> int:
        """Count return statements in function."""
        count = 0

        def visit(n):
            nonlocal count
            if n.type in {"return_statement", "return"}:
                count += 1
            for child in n.children:
                visit(child)

        visit(func_node)
        return count

    def _estimate_complexity_regex(
        self, content: str, file_path: Path
    ) -> Tuple[List[AnalysisIssue], List[FunctionMetrics]]:
        """Fallback: estimate complexity using regex patterns."""
        # Simple heuristic: count control flow keywords
        keywords = {
            "python": ["if ", "elif ", "for ", "while ", "except ", "and ", "or "],
            "javascript": ["if ", "else ", "for ", "while ", "switch ", "case ", "&&", "||"],
            "typescript": ["if ", "else ", "for ", "while ", "switch ", "case ", "&&", "||"],
            "java": ["if ", "else ", "for ", "while ", "switch ", "case ", "&&", "||"],
            "go": ["if ", "for ", "switch ", "case ", "select ", "&&", "||"],
            "rust": ["if ", "else ", "for ", "while ", "loop ", "match ", "&&", "||"],
            "c": ["if ", "else ", "for ", "while ", "switch ", "case ", "&&", "||"],
            "cpp": ["if ", "else ", "for ", "while ", "switch ", "case ", "&&", "||", "catch "],
            "c_sharp": [
                "if ",
                "else ",
                "for ",
                "foreach ",
                "while ",
                "switch ",
                "case ",
                "&&",
                "||",
                "catch ",
            ],
            "ruby": [
                "if ",
                "else ",
                "unless ",
                "for ",
                "while ",
                "until ",
                "case ",
                "when ",
                "rescue ",
                "&&",
                "||",
            ],
            "php": [
                "if ",
                "else ",
                "for ",
                "foreach ",
                "while ",
                "switch ",
                "case ",
                "&&",
                "||",
                "catch ",
            ],
            "kotlin": ["if ", "else ", "for ", "while ", "when ", "->", "&&", "||"],
            "swift": ["if ", "else ", "for ", "while ", "switch ", "case ", "guard ", "&&", "||"],
            "scala": ["if ", "else ", "for ", "while ", "match ", "case ", "=>", "&&", "||"],
            "bash": ["if ", "elif ", "for ", "while ", "case ", ";;", "&&", "||"],
            "sql": [],  # SQL has no typical control flow
            "lua": ["if ", "else ", "for ", "while ", "repeat ", "and ", "or "],
            "elixir": ["if ", "unless ", "case ", "cond ", "with ", "->"],
            "haskell": ["if ", "case ", "| ", "->"],
            "r": ["if ", "else ", "for ", "while ", "repeat ", "&", "|"],
        }

        lang_keywords = keywords.get(self.language, [])
        complexity = 1
        for kw in lang_keywords:
            complexity += content.count(kw)

        # Rough estimate - no function-level breakdown
        return [], []

    def check_documentation(self, content: str, file_path: Path) -> List[AnalysisIssue]:
        """Check documentation coverage using tree-sitter."""
        issues = []
        parser = self._get_parser()

        if parser is None:
            return issues

        try:
            tree = parser.parse(bytes(content, "utf8"))
            func_nodes = self._find_function_nodes(tree.root_node)

            for func_name, func_node in func_nodes:
                if not self._has_docstring(func_node, content):
                    # Skip private/internal functions
                    if not func_name.startswith("_"):
                        issues.append(
                            AnalysisIssue(
                                type="documentation",
                                severity="low",
                                issue=f"Missing Docstring: {func_name}",
                                file=str(file_path),
                                line=func_node.start_point[0] + 1,
                                code=f"Function '{func_name}' lacks documentation",
                                recommendation="Add a docstring describing purpose, args, and returns",
                            )
                        )

        except Exception as e:
            logger.warning(f"Documentation check failed for {file_path}: {e}")

        return issues

    def analyze_code_accelerated(
        self,
        source_code: str,
        language: str,
        patterns: Optional[List[str]] = None,
    ) -> List["PatternMatch"]:
        """Analyze code using Rust-accelerated regex engine.

        Provides 10-20x faster pattern matching than Python re module.

        Args:
            source_code: Source code to analyze
            language: Programming language name
            patterns: Optional list of regex patterns (uses defaults if None)

        Returns:
            List of PatternMatch objects

        Example:
            >>> analyzer = get_analyzer("python")
            >>> matches = analyzer.analyze_code_accelerated(code, "python")
            >>> for match in matches:
            ...     print(f"Line {match.line}: {match.pattern}")
        """
        if self._regex_accelerator is None:
            # Fallback to Python implementation
            logger.debug("Regex accelerator unavailable, using Python re")
            return self._analyze_code_python(source_code, language, patterns)

        try:
            # Compile patterns for this language
            compiled_set = self._regex_accelerator.compile_patterns(language, patterns)

            # Match all patterns against source code
            matches = self._regex_accelerator.match_all(source_code, compiled_set)

            logger.debug(
                f"Accelerated analysis found {len(matches)} matches "
                f"(cache hit rate: {self._regex_accelerator.cache_stats.cache_hit_rate:.1f}%)"
            )

            return matches
        except Exception as e:
            logger.warning(f"Accelerated pattern matching failed: {e}, using Python fallback")
            return self._analyze_code_python(source_code, language, patterns)

    def _analyze_code_python(
        self,
        source_code: str,
        language: str,
        patterns: Optional[List[str]] = None,
    ) -> List["PatternMatch"]:
        """Python fallback for code analysis.

        Args:
            source_code: Source code to analyze
            language: Programming language name
            patterns: Optional list of regex patterns

        Returns:
            List of PatternMatch objects
        """
        # Import PatternMatch locally to avoid circular import
        from victor.native.accelerators.regex_engine import PatternMatch

        matches = []
        lines = source_code.split("\n")

        # Use language-specific patterns if none provided
        if patterns is None:
            security_patterns = self._get_security_patterns()
            smell_patterns = self._get_smell_patterns()
            patterns = [p.pattern for p in security_patterns + smell_patterns]

        # Compile and match patterns
        for pattern_str in patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                for line_num, line in enumerate(lines, 1):
                    match = pattern.search(line)
                    if match:
                        matches.append(
                            PatternMatch(
                                pattern=pattern_str,
                                line=line_num,
                                column=match.start() + 1,
                                matched_text=match.group(0),
                                context=line.strip(),
                            )
                        )
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern_str}': {e}")

        return matches

    def analyze(self, content: str, file_path: Path, aspects: List[str]) -> AnalysisResult:
        """Run full analysis on content.

        Args:
            content: File content to analyze
            file_path: Path to the file
            aspects: List of aspects to check: "security", "complexity",
                    "best_practices", "documentation", "all"

        Returns:
            AnalysisResult with all findings
        """
        result = AnalysisResult(
            file_path=str(file_path),
            language=self.language,
            lines_of_code=len(content.split("\n")),
        )

        # Expand "all" to all aspects
        if "all" in aspects:
            aspects = ["security", "complexity", "best_practices", "documentation"]

        try:
            if "security" in aspects:
                result.issues.extend(self.check_security(content, file_path))

            if "best_practices" in aspects:
                result.issues.extend(self.check_code_smells(content, file_path))

            if "complexity" in aspects:
                complexity_issues, functions = self.calculate_complexity(content, file_path)
                result.issues.extend(complexity_issues)
                result.functions = functions

            if "documentation" in aspects:
                result.issues.extend(self.check_documentation(content, file_path))

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Analysis failed for {file_path}: {e}")

        return result


# =============================================================================
# Language-Specific Analyzer Implementations
# =============================================================================


class PythonAnalyzer(BaseLanguageAnalyzer):
    """Python-specific code analyzer."""

    @property
    def language(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> List[str]:
        return [".py", ".pyw", ".pyi"]

    def _has_docstring(self, func_node, content: str) -> bool:
        """Python-specific docstring detection."""
        for child in func_node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for expr in stmt.children:
                            if expr.type == "string":
                                text = expr.text.decode("utf8")
                                return text.startswith(('"""', "'''"))
                    elif stmt.type not in {"comment", "newline"}:
                        break
        return False


class JavaScriptAnalyzer(BaseLanguageAnalyzer):
    """JavaScript-specific code analyzer."""

    @property
    def language(self) -> str:
        return "javascript"

    @property
    def file_extensions(self) -> List[str]:
        return [".js", ".mjs", ".cjs", ".jsx"]

    def _has_docstring(self, func_node, content: str) -> bool:
        """JavaScript: Check for JSDoc comment before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.endswith("*/") or prev_line.startswith("//")
        return False


class TypeScriptAnalyzer(JavaScriptAnalyzer):
    """TypeScript-specific code analyzer (extends JavaScript)."""

    @property
    def language(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> List[str]:
        return [".ts", ".tsx"]


class JavaAnalyzer(BaseLanguageAnalyzer):
    """Java-specific code analyzer."""

    @property
    def language(self) -> str:
        return "java"

    @property
    def file_extensions(self) -> List[str]:
        return [".java"]

    def _has_docstring(self, func_node, content: str) -> bool:
        """Java: Check for Javadoc comment before method."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.endswith("*/")
        return False


class GoAnalyzer(BaseLanguageAnalyzer):
    """Go-specific code analyzer."""

    @property
    def language(self) -> str:
        return "go"

    @property
    def file_extensions(self) -> List[str]:
        return [".go"]

    def _has_docstring(self, func_node, content: str) -> bool:
        """Go: Check for doc comment before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.startswith("//")
        return False


class RustAnalyzer(BaseLanguageAnalyzer):
    """Rust-specific code analyzer."""

    @property
    def language(self) -> str:
        return "rust"

    @property
    def file_extensions(self) -> List[str]:
        return [".rs"]

    def _has_docstring(self, func_node, content: str) -> bool:
        """Rust: Check for doc comment (///) before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.startswith("///") or prev_line.startswith("//!")
        return False


class CAnalyzer(BaseLanguageAnalyzer):
    """C-specific code analyzer."""

    @property
    def language(self) -> str:
        return "c"

    @property
    def file_extensions(self) -> List[str]:
        return [".c", ".h"]

    def _is_function_node(self, node) -> bool:
        return node.type == "function_definition"

    def _has_docstring(self, func_node, content: str) -> bool:
        """C: Check for comment block (/** or //) before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.endswith("*/") or prev_line.startswith("//")
        return False


class CppAnalyzer(CAnalyzer):
    """C++-specific code analyzer (extends C)."""

    @property
    def language(self) -> str:
        return "cpp"

    @property
    def file_extensions(self) -> List[str]:
        return [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"]


class CSharpAnalyzer(BaseLanguageAnalyzer):
    """C#-specific code analyzer."""

    @property
    def language(self) -> str:
        return "c_sharp"

    @property
    def file_extensions(self) -> List[str]:
        return [".cs"]

    def _is_function_node(self, node) -> bool:
        return node.type in {"method_declaration", "constructor_declaration"}

    def _has_docstring(self, func_node, content: str) -> bool:
        """C#: Check for XML doc comment (///) before method."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.startswith("///") or prev_line.endswith("*/")
        return False


class RubyAnalyzer(BaseLanguageAnalyzer):
    """Ruby-specific code analyzer."""

    @property
    def language(self) -> str:
        return "ruby"

    @property
    def file_extensions(self) -> List[str]:
        return [".rb", ".rake", ".gemspec"]

    def _is_function_node(self, node) -> bool:
        return node.type in {"method", "singleton_method"}

    def _has_docstring(self, func_node, content: str) -> bool:
        """Ruby: Check for RDoc/YARD comment before method."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.startswith("#")
        return False


class PHPAnalyzer(BaseLanguageAnalyzer):
    """PHP-specific code analyzer."""

    @property
    def language(self) -> str:
        return "php"

    @property
    def file_extensions(self) -> List[str]:
        return [".php", ".phtml"]

    def _is_function_node(self, node) -> bool:
        return node.type in {"function_definition", "method_declaration"}

    def _has_docstring(self, func_node, content: str) -> bool:
        """PHP: Check for PHPDoc (/** */) before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.endswith("*/")
        return False


class KotlinAnalyzer(BaseLanguageAnalyzer):
    """Kotlin-specific code analyzer."""

    @property
    def language(self) -> str:
        return "kotlin"

    @property
    def file_extensions(self) -> List[str]:
        return [".kt", ".kts"]

    def _is_function_node(self, node) -> bool:
        return node.type == "function_declaration"

    def _has_docstring(self, func_node, content: str) -> bool:
        """Kotlin: Check for KDoc (/** */) before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.endswith("*/")
        return False


class SwiftAnalyzer(BaseLanguageAnalyzer):
    """Swift-specific code analyzer."""

    @property
    def language(self) -> str:
        return "swift"

    @property
    def file_extensions(self) -> List[str]:
        return [".swift"]

    def _is_function_node(self, node) -> bool:
        return node.type in {"function_declaration", "subscript_declaration"}

    def _has_docstring(self, func_node, content: str) -> bool:
        """Swift: Check for doc comment (///) before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.startswith("///") or prev_line.endswith("*/")
        return False


class ScalaAnalyzer(BaseLanguageAnalyzer):
    """Scala-specific code analyzer."""

    @property
    def language(self) -> str:
        return "scala"

    @property
    def file_extensions(self) -> List[str]:
        return [".scala", ".sc"]

    def _is_function_node(self, node) -> bool:
        return node.type in {"function_definition", "function_declaration"}

    def _has_docstring(self, func_node, content: str) -> bool:
        """Scala: Check for Scaladoc (/** */) before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.endswith("*/")
        return False


class BashAnalyzer(BaseLanguageAnalyzer):
    """Bash/Shell-specific code analyzer."""

    @property
    def language(self) -> str:
        return "bash"

    @property
    def file_extensions(self) -> List[str]:
        return [".sh", ".bash", ".zsh", ".ksh"]

    def _is_function_node(self, node) -> bool:
        return node.type == "function_definition"

    def _has_docstring(self, func_node, content: str) -> bool:
        """Bash: Check for comment block before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.startswith("#")
        return False


class SQLAnalyzer(BaseLanguageAnalyzer):
    """SQL-specific code analyzer.

    Note: SQL doesn't have functions in the traditional sense,
    so complexity and documentation checks are minimal.
    """

    @property
    def language(self) -> str:
        return "sql"

    @property
    def file_extensions(self) -> List[str]:
        return [".sql", ".ddl", ".dml"]

    def _is_function_node(self, node) -> bool:
        # SQL stored procedures/functions
        return node.type in {"create_function_statement", "create_procedure_statement"}

    def calculate_complexity(
        self, content: str, file_path: Path
    ) -> Tuple[List[AnalysisIssue], List[FunctionMetrics]]:
        # SQL doesn't have typical control flow complexity
        return [], []


class LuaAnalyzer(BaseLanguageAnalyzer):
    """Lua-specific code analyzer."""

    @property
    def language(self) -> str:
        return "lua"

    @property
    def file_extensions(self) -> List[str]:
        return [".lua"]

    def _is_function_node(self, node) -> bool:
        return node.type in {"function_declaration", "local_function_declaration"}

    def _has_docstring(self, func_node, content: str) -> bool:
        """Lua: Check for comment before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.startswith("--")
        return False


class ElixirAnalyzer(BaseLanguageAnalyzer):
    """Elixir-specific code analyzer."""

    @property
    def language(self) -> str:
        return "elixir"

    @property
    def file_extensions(self) -> List[str]:
        return [".ex", ".exs"]

    def _is_function_node(self, node) -> bool:
        return node.type in {"call"}  # def/defp calls

    def _has_docstring(self, func_node, content: str) -> bool:
        """Elixir: Check for @doc or @moduledoc before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        # Check previous lines for @doc
        for i in range(max(0, start_line - 3), start_line):
            if "@doc" in lines[i] or "@moduledoc" in lines[i]:
                return True
        return False


class HaskellAnalyzer(BaseLanguageAnalyzer):
    """Haskell-specific code analyzer."""

    @property
    def language(self) -> str:
        return "haskell"

    @property
    def file_extensions(self) -> List[str]:
        return [".hs", ".lhs"]

    def _is_function_node(self, node) -> bool:
        return node.type == "function"

    def _has_docstring(self, func_node, content: str) -> bool:
        """Haskell: Check for Haddock comment (--|) before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.startswith("-- |") or prev_line.startswith("{-|")
        return False


class RAnalyzer(BaseLanguageAnalyzer):
    """R-specific code analyzer."""

    @property
    def language(self) -> str:
        return "r"

    @property
    def file_extensions(self) -> List[str]:
        return [".r", ".R", ".rmd", ".Rmd"]

    def _is_function_node(self, node) -> bool:
        # R functions are defined with <- or = assignment
        return node.type == "function_definition"

    def _has_docstring(self, func_node, content: str) -> bool:
        """R: Check for roxygen2 comment (#') before function."""
        start_line = func_node.start_point[0]
        lines = content.split("\n")
        if start_line > 0:
            prev_line = lines[start_line - 1].strip()
            return prev_line.startswith("#'") or prev_line.startswith("#")
        return False


# =============================================================================
# Language Registry
# =============================================================================


class LanguageRegistry:
    """Registry for language analyzers with auto-detection."""

    _analyzers: Dict[str, Type[BaseLanguageAnalyzer]] = {
        # Core languages
        "python": PythonAnalyzer,
        "javascript": JavaScriptAnalyzer,
        "typescript": TypeScriptAnalyzer,
        "java": JavaAnalyzer,
        "go": GoAnalyzer,
        "rust": RustAnalyzer,
        # Systems languages
        "c": CAnalyzer,
        "cpp": CppAnalyzer,
        "c_sharp": CSharpAnalyzer,
        # Scripting languages
        "ruby": RubyAnalyzer,
        "php": PHPAnalyzer,
        # JVM languages
        "kotlin": KotlinAnalyzer,
        "scala": ScalaAnalyzer,
        # Apple ecosystem
        "swift": SwiftAnalyzer,
        # Shell/SQL
        "bash": BashAnalyzer,
        "sql": SQLAnalyzer,
        # Other languages
        "lua": LuaAnalyzer,
        "elixir": ElixirAnalyzer,
        "haskell": HaskellAnalyzer,
        "r": RAnalyzer,
    }

    _instances: Dict[str, BaseLanguageAnalyzer] = {}

    @classmethod
    def get_analyzer(
        cls, language: str, max_complexity: int = 10
    ) -> Optional[BaseLanguageAnalyzer]:
        """Get analyzer for a language.

        Args:
            language: Language name (python, javascript, etc.)
            max_complexity: Maximum complexity threshold

        Returns:
            Language analyzer instance or None if unsupported
        """
        language = language.lower()

        # Check cache
        cache_key = f"{language}_{max_complexity}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Create new instance
        analyzer_class = cls._analyzers.get(language)
        if analyzer_class is None:
            return None

        analyzer = analyzer_class(max_complexity=max_complexity)
        cls._instances[cache_key] = analyzer
        return analyzer

    @classmethod
    def get_analyzer_for_file(
        cls, file_path: Path, max_complexity: int = 10
    ) -> Optional[BaseLanguageAnalyzer]:
        """Get analyzer based on file extension.

        Args:
            file_path: Path to the file
            max_complexity: Maximum complexity threshold

        Returns:
            Appropriate analyzer or None if unsupported
        """
        ext = file_path.suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(ext)
        if language:
            return cls.get_analyzer(language, max_complexity)
        return None

    @classmethod
    def supported_languages(cls) -> List[str]:
        """Return list of supported language names."""
        return list(cls._analyzers.keys())

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Return list of supported file extensions."""
        return list(EXTENSION_TO_LANGUAGE.keys())

    @classmethod
    def register_analyzer(cls, language: str, analyzer_class: Type[BaseLanguageAnalyzer]) -> None:
        """Register a custom language analyzer.

        Args:
            language: Language name
            analyzer_class: Analyzer class to register
        """
        cls._analyzers[language.lower()] = analyzer_class
        logger.info(f"Registered custom analyzer for {language}")


# =============================================================================
# Convenience Functions
# =============================================================================


def get_analyzer(language: str, max_complexity: int = 10) -> Optional[BaseLanguageAnalyzer]:
    """Get analyzer for a language.

    Args:
        language: Language name (python, javascript, etc.)
        max_complexity: Maximum complexity threshold

    Returns:
        Language analyzer instance or None if unsupported
    """
    return LanguageRegistry.get_analyzer(language, max_complexity)


def get_analyzer_for_file(
    file_path: Path, max_complexity: int = 10
) -> Optional[BaseLanguageAnalyzer]:
    """Get analyzer based on file extension.

    Args:
        file_path: Path to the file
        max_complexity: Maximum complexity threshold

    Returns:
        Appropriate analyzer or None if unsupported
    """
    return LanguageRegistry.get_analyzer_for_file(file_path, max_complexity)


def detect_language(file_path: Path) -> Optional[str]:
    """Detect language from file extension.

    Args:
        file_path: Path to the file

    Returns:
        Language name or None if unsupported
    """
    ext = file_path.suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext)


def get_glob_pattern(language: str) -> str:
    """Get glob pattern for a language.

    Args:
        language: Language name or "all"

    Returns:
        Glob pattern string
    """
    return LANGUAGE_GLOB_PATTERNS.get(language.lower(), f"*.{language}")


async def analyze_file(
    file_path: Path,
    aspects: Optional[List[str]] = None,
    max_complexity: int = 10,
) -> AnalysisResult:
    """Analyze a single file with auto-detected language.

    Args:
        file_path: Path to the file
        aspects: Aspects to check (default: all)
        max_complexity: Maximum complexity threshold

    Returns:
        AnalysisResult with findings
    """
    if aspects is None:
        aspects = ["all"]

    file_path = Path(file_path)
    analyzer = get_analyzer_for_file(file_path, max_complexity)

    if analyzer is None:
        return AnalysisResult(
            file_path=str(file_path),
            language="unknown",
            success=False,
            error=f"Unsupported file type: {file_path.suffix}",
        )

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return AnalysisResult(
            file_path=str(file_path),
            language=analyzer.language,
            success=False,
            error=f"Failed to read file: {e}",
        )

    return analyzer.analyze(content, file_path, aspects)


def supported_languages() -> List[str]:
    """Return list of supported language names."""
    return LanguageRegistry.supported_languages()


def supported_extensions() -> List[str]:
    """Return list of supported file extensions."""
    return LanguageRegistry.supported_extensions()
