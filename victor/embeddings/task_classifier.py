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

"""Task type classification using semantic embeddings.

This module provides task type classification for detecting:
- CREATE_SIMPLE: Simple standalone code generation (no exploration needed)
- CREATE: Create new code that integrates with existing codebase
- EDIT: Modify existing code/files
- SEARCH: Find/locate code or files
- ANALYZE: Count, measure, analyze code metrics
- DESIGN: Conceptual/planning tasks (no tools needed)
- GENERAL: Ambiguous or general help

Uses semantic similarity with embeddings instead of hardcoded regex patterns.
This handles variations and paraphrases more robustly.

Additionally provides:
- Preprocessing: Normalizes prompts by stripping file paths and identifiers
  before embedding comparison to prevent false semantic matches
- Nudging: Rule-based adjustments for known edge cases that semantic
  similarity struggles with (e.g., "read and explain" vs "edit")
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from victor.embeddings.collections import CollectionItem, StaticEmbeddingCollection
from victor.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be detected from prompts."""

    EDIT = "edit"  # Modify existing code/files
    SEARCH = "search"  # Find/locate code or files
    CREATE = "create"  # Write new files (with context/location specified)
    CREATE_SIMPLE = "create_simple"  # Generate standalone code (no exploration needed)
    ANALYZE = "analyze"  # Count, measure, analyze code
    DESIGN = "design"  # Conceptual/planning tasks (no tools needed)
    GENERAL = "general"  # Ambiguous or general help
    ACTION = "action"  # Git operations, test runs, script execution
    ANALYSIS_DEEP = "analysis_deep"  # Comprehensive codebase analysis


@dataclass
class TaskTypeResult:
    """Result of task type classification."""

    task_type: TaskType
    confidence: float  # 0-1 confidence score
    top_matches: List[Tuple[str, float]]  # Top matching phrases with scores
    has_file_context: bool  # Whether the prompt mentions specific files
    nudge_applied: Optional[str] = None  # Name of nudge rule applied, if any
    preprocessed_prompt: Optional[str] = None  # Preprocessed prompt used for classification


@dataclass
class NudgeRule:
    """A rule-based nudge for correcting edge case classifications.

    Nudge rules are applied AFTER embedding classification to correct
    known edge cases where semantic similarity fails.
    """

    name: str  # Rule identifier for debugging
    pattern: re.Pattern  # Regex pattern to match (on original prompt)
    target_type: TaskType  # Task type to nudge towards
    min_confidence_boost: float = 0.1  # Minimum boost to apply
    override: bool = False  # If True, always override regardless of scores


# Compiled nudge rules - applied in order, first match wins
# These handle edge cases that semantic similarity struggles with
NUDGE_RULES: List[NudgeRule] = [
    # "Read and explain" patterns → ANALYZE (not EDIT)
    NudgeRule(
        name="read_and_explain",
        pattern=re.compile(r"^read\s+(and\s+)?explain", re.IGNORECASE),
        target_type=TaskType.ANALYZE,
        override=True,
    ),
    # "Explain the code in" → ANALYZE (code understanding)
    NudgeRule(
        name="explain_code_in",
        pattern=re.compile(r"^explain\s+(the\s+)?code\s+in", re.IGNORECASE),
        target_type=TaskType.ANALYZE,
        override=True,
    ),
    # "Analyze the code in" → ANALYZE
    NudgeRule(
        name="analyze_code_in",
        pattern=re.compile(r"^analyze\s+(the\s+)?code\s+in", re.IGNORECASE),
        target_type=TaskType.ANALYZE,
        override=True,
    ),
    # "Review the code in" → ANALYZE
    NudgeRule(
        name="review_code_in",
        pattern=re.compile(r"^review\s+(the\s+)?code\s+in", re.IGNORECASE),
        target_type=TaskType.ANALYZE,
        override=True,
    ),
    # "Explain what X does" → DESIGN (understanding, no tools)
    NudgeRule(
        name="explain_what_does",
        pattern=re.compile(r"^explain\s+what\s+", re.IGNORECASE),
        target_type=TaskType.DESIGN,
        override=True,
    ),
    # "What does X do" → DESIGN
    NudgeRule(
        name="what_does",
        pattern=re.compile(r"^what\s+does\s+", re.IGNORECASE),
        target_type=TaskType.DESIGN,
        override=True,
    ),
    # "Without reading/looking at code" → DESIGN
    NudgeRule(
        name="without_code",
        pattern=re.compile(
            r"without\s+(reading|looking\s+at)\s+(any\s+)?(code|files)", re.IGNORECASE
        ),
        target_type=TaskType.DESIGN,
        override=True,
    ),
    # Explicit "edit <path>" at start → EDIT (strong signal)
    NudgeRule(
        name="edit_command",
        pattern=re.compile(
            r"^edit\s+[\w/\\.-]+\.(py|js|ts|go|rs|java|yaml|json|md)", re.IGNORECASE
        ),
        target_type=TaskType.EDIT,
        override=True,
    ),
    # "Find all classes that inherit" → SEARCH
    NudgeRule(
        name="find_inheritance",
        pattern=re.compile(r"find\s+(all\s+)?class(es)?\s+(that\s+)?inherit", re.IGNORECASE),
        target_type=TaskType.SEARCH,
        override=True,
    ),
    # "List all subclasses" → SEARCH
    NudgeRule(
        name="list_subclasses",
        pattern=re.compile(r"list\s+(all\s+)?(subclass|class)", re.IGNORECASE),
        target_type=TaskType.SEARCH,
        min_confidence_boost=0.15,
    ),
    # "Count/how many" → ANALYZE
    NudgeRule(
        name="count_query",
        pattern=re.compile(r"^(count|how\s+many)\s+", re.IGNORECASE),
        target_type=TaskType.ANALYZE,
        min_confidence_boost=0.15,
    ),
    # "Create a simple/just create" → CREATE_SIMPLE
    NudgeRule(
        name="simple_create",
        pattern=re.compile(r"(create|write|make)\s+a?\s*simple\s+", re.IGNORECASE),
        target_type=TaskType.CREATE_SIMPLE,
        min_confidence_boost=0.1,
    ),
    # "Just write a function" → CREATE_SIMPLE
    NudgeRule(
        name="just_function",
        pattern=re.compile(r"^just\s+(write|create|make)\s+", re.IGNORECASE),
        target_type=TaskType.CREATE_SIMPLE,
        min_confidence_boost=0.1,
    ),
    # Git read operations → SEARCH (not ACTION)
    NudgeRule(
        name="git_read",
        pattern=re.compile(r"\bgit\s+(status|log|branch|diff|show)\b", re.IGNORECASE),
        target_type=TaskType.SEARCH,
        override=True,
    ),
    # Git write operations → ACTION
    NudgeRule(
        name="git_commit",
        pattern=re.compile(
            r"\b(commit|push|pull|merge)\s+(all|the|these|my)?\s*(changes?|files?)?\b",
            re.IGNORECASE,
        ),
        target_type=TaskType.ACTION,
        override=True,
    ),
    NudgeRule(
        name="git_command",
        pattern=re.compile(r"\bgit\s+(add|commit|push|pull|merge|rebase|stash)\b", re.IGNORECASE),
        target_type=TaskType.ACTION,
        override=True,
    ),
    NudgeRule(
        name="create_pr",
        pattern=re.compile(r"\b(create|make|open)\s+(a\s+)?(pr|pull\s*request)\b", re.IGNORECASE),
        target_type=TaskType.ACTION,
        override=True,
    ),
    NudgeRule(
        name="grouped_commits",
        pattern=re.compile(r"\bgroup(ed)?\s+commits?\b", re.IGNORECASE),
        target_type=TaskType.ACTION,
        override=True,
    ),
    # Test execution → ACTION
    NudgeRule(
        name="run_tests",
        pattern=re.compile(
            r"\b(run|execute)\s+(the\s+)?(tests?|pytest|unittest|jest|mocha)\b", re.IGNORECASE
        ),
        target_type=TaskType.ACTION,
        override=True,
    ),
    NudgeRule(
        name="test_command",
        pattern=re.compile(r"\bpytest\b|\bnpm\s+test\b|\bcargo\s+test\b", re.IGNORECASE),
        target_type=TaskType.ACTION,
        override=True,
    ),
    # Build/deploy → ACTION
    NudgeRule(
        name="build_deploy",
        pattern=re.compile(
            r"\b(build|compile|deploy)\s+(the\s+)?(project|app|code)\b", re.IGNORECASE
        ),
        target_type=TaskType.ACTION,
        min_confidence_boost=0.15,
    ),
    # Comprehensive analysis → ANALYSIS_DEEP
    NudgeRule(
        name="full_codebase_analysis",
        pattern=re.compile(
            r"\b(analyze|review|audit)\s+(the\s+)?(entire\s+)?(codebase|project|repo)\b",
            re.IGNORECASE,
        ),
        target_type=TaskType.ANALYSIS_DEEP,
        override=True,
    ),
    NudgeRule(
        name="comprehensive_analysis",
        pattern=re.compile(
            r"\b(comprehensive|thorough|detailed|full)\s+(analysis|review|audit)\b", re.IGNORECASE
        ),
        target_type=TaskType.ANALYSIS_DEEP,
        override=True,
    ),
    NudgeRule(
        name="architecture_review",
        pattern=re.compile(r"\barchitecture\s+(review|analysis|overview)\b", re.IGNORECASE),
        target_type=TaskType.ANALYSIS_DEEP,
        override=True,
    ),
    NudgeRule(
        name="security_audit",
        pattern=re.compile(r"\b(security|vulnerability)\s+(audit|scan|review)\b", re.IGNORECASE),
        target_type=TaskType.ANALYSIS_DEEP,
        min_confidence_boost=0.15,
    ),
    # Web search and API research → ACTION (requires multiple tool calls)
    NudgeRule(
        name="web_search_action",
        pattern=re.compile(
            r"\b(perform|do|run)\s+(a\s+)?(web\s*search|websearch)\b", re.IGNORECASE
        ),
        target_type=TaskType.ACTION,
        override=True,
    ),
    NudgeRule(
        name="web_search",
        pattern=re.compile(r"\bweb\s*search\b", re.IGNORECASE),
        target_type=TaskType.ACTION,
        min_confidence_boost=0.2,
    ),
    NudgeRule(
        name="search_web",
        pattern=re.compile(r"\bsearch\s+(the\s+)?(web|internet|online)\b", re.IGNORECASE),
        target_type=TaskType.ACTION,
        min_confidence_boost=0.2,
    ),
    NudgeRule(
        name="fetch_api",
        pattern=re.compile(r"\bfetch\s+(for\s+)?(api|data|docs|documentation)\b", re.IGNORECASE),
        target_type=TaskType.ACTION,
        min_confidence_boost=0.15,
    ),
    NudgeRule(
        name="research_api",
        pattern=re.compile(
            r"\b(research|investigate|explore)\s+(the\s+)?\w+\s+(api|library|package)\b",
            re.IGNORECASE,
        ),
        target_type=TaskType.ACTION,
        min_confidence_boost=0.15,
    ),
]


# Canonical phrases for each task type
# These are used to build embedding collections for semantic matching
# More phrases = better semantic coverage for variations

CREATE_SIMPLE_PHRASES = [
    # "simple" + code type patterns
    "create a simple function",
    "write a simple script",
    "generate a simple class",
    "make a simple function",
    "write a simple Python function",
    "create a simple Python script",
    "generate a simple utility function",
    "create a simple helper function",
    "write a simple utility",
    "make a simple Python class",
    # "function that" patterns - common standalone requests
    "create a function that calculates",
    "write a function that validates",
    "generate a function that parses",
    "make a function that converts",
    "create a function to compute",
    "write a function to handle",
    "create a function that checks",
    "write a function that returns",
    "generate a function that processes",
    "make a function that formats",
    "implement a function that sorts",
    "create a function that filters",
    # Direct generation requests
    "write me a function",
    "generate me some code",
    "create code that",
    "implement a function",
    "just create a function",
    "just write a simple",
    "can you write a function",
    "please create a function",
    "I need a function that",
    "give me code for",
    # Standalone algorithm/utility requests
    "create a factorial function",
    "write a fibonacci function",
    "generate a sorting function",
    "implement an algorithm for",
    "create a utility to calculate",
    "write a helper for",
    "create a parser for",
    "implement a validator for",
    "write a converter function",
    "create a calculator function",
    # Common programming tasks (standalone)
    "write code to calculate",
    "create a function for prime numbers",
    "implement binary search",
    "write a recursive function",
    "create a lambda function",
    "implement a decorator",
    "write a generator function",
    "create an async function",
]

CREATE_PHRASES = [
    # Create with location context
    "create a new file in",
    "write a new module in",
    "add a new class to",
    "create a new component in",
    "write a new test file for",
    "create a file called",
    "make a new file in the",
    "add a new file to",
    # Create with integration
    "create a new feature that integrates",
    "write a new handler for",
    "add a new endpoint to",
    "create a new service for",
    "implement a new tool for",
    "create a provider for",
    "add a new command to",
    # New file creation with path
    "create a new Python file",
    "write a new module called",
    "add a new config file",
    "create a new script for",
    "create victor/tools/new_tool.py",
    "write tests/test_new.py",
    "add a new test file",
    # Build/implement with context
    "build a new feature",
    "implement a new system",
    "create the infrastructure for",
    "set up a new module",
    "scaffold a new component",
    "bootstrap a new service",
    # Integration patterns
    "create a tool that uses",
    "implement integration with",
    "add support for",
    "create a wrapper for",
]

EDIT_PHRASES = [
    # Direct edit patterns - HIGH PRIORITY for "Edit <file>" commands
    "edit the file",
    "edit this file",
    "edit base.py",
    "edit orchestrator.py",
    "edit the py file",
    "edit the file to add",
    "edit the file to fix",
    "edit the file to change",
    "modify this file to",
    "update this file to",
    "change the file to",
    # Edit <path> patterns (generic paths)
    "edit tools/base.py",
    "edit agent/orchestrator.py",
    "edit src/main.py",
    "edit tests/test.py",
    "edit the module",
    "edit the script",
    "edit the test file",
    # Modify patterns
    "modify the function",
    "update the code in",
    "change the implementation",
    "fix the bug in",
    "change the file",
    "update this file",
    "modify this function",
    # Add to existing patterns
    "add a method to",
    "add error handling to",
    "add logging to the file",
    "add a property to the class",
    "add validation to",
    "add a parameter to",
    "add a new field to",
    "add tests for",
    "add documentation to",
    "add type hints to",
    "add a docstring to",
    "insert code into",
    # Remove/delete patterns
    "remove the duplicate code",
    "delete the unused function",
    "remove the commented code",
    "delete this method",
    "remove the import",
    "clean up unused variables",
    "remove dead code",
    # Refactor patterns
    "refactor the function",
    "rename the variable",
    "extract the method",
    "move the code to",
    "clean up the code",
    "refactor this class",
    "simplify the logic",
    "optimize the function",
    "restructure the code",
    "split the function into",
    # Fix patterns - many variations
    "fix the issue in",
    "fix the error in",
    "fix this code",
    "correct the bug",
    "resolve the problem in",
    "fix the failing test",
    "fix the syntax error",
    "fix the type error",
    "fix the import error",
    "debug this function",
    "patch the vulnerability",
    "fix the bug in the parser",
    "fix the bug in the function",
    "fix the bug in this file",
    "fix the problem with",
    "repair the broken function",
    "fix the issue with the",
    "resolve the bug in",
    "debug the issue in",
    "troubleshoot the error",
    # Update patterns
    "update the imports",
    "update the configuration",
    "change the behavior of",
    "update the version",
    "change the default value",
    "update the docstring",
    "modify the return type",
    "change the parameter",
    # Improve patterns
    "improve the performance of",
    "make this function faster",
    "enhance the error handling",
    "improve the readability",
]

SEARCH_PHRASES = [
    # Find/locate patterns
    "find all occurrences of",
    "locate the file",
    "search for the function",
    "where is the class",
    "which file contains",
    "find the file that",
    "locate where",
    "search for references to",
    "find usages of",
    "where is this defined",
    "where is the configuration",
    "where is configured",
    "where is the connection",
    "where is the database",
    "where is the setting",
    # Show/list patterns
    "show me all the files",
    "list all functions that",
    "show the code that",
    "list the files in",
    "show me the classes",
    "list all imports",
    "show the methods in",
    "list all tests",
    "show me the directory structure",
    "list all the endpoints",
    "list all API endpoints",
    "list all the API endpoints",
    "list the endpoints in",
    "show me all endpoints",
    "list all routes",
    # Look for patterns
    "look for the implementation",
    "find the definition of",
    "search the codebase for",
    "find where this is used",
    "look for TODO comments",
    "find all FIXME markers",
    "search for deprecated code",
    # Explore patterns
    "explore the module",
    "what files are in",
    "show me what's in",
    "what classes exist",
    "what functions are available",
    "how is this implemented",
    "where does this come from",
    "what endpoints are there",
    "what routes exist",
    # Find all/classes patterns
    "find all classes that inherit",
    "find all classes inheriting from",
    "find all subclasses of",
    "list all classes that inherit",
    "list all subclasses",
    "find classes extending",
    "what classes inherit from",
    "which classes inherit from",
    "show me all classes that",
    "get all subclasses of",
]

ANALYZE_PHRASES = [
    # Count/measure patterns
    "count the number of",
    "how many files are",
    "measure the lines of code",
    "calculate the complexity",
    "count all functions",
    "how many classes",
    "count the tests",
    "how many lines",
    # Statistics patterns
    "show statistics for",
    "analyze the code coverage",
    "what is the total",
    "how much code is",
    "give me metrics for",
    "show code statistics",
    "calculate test coverage",
    "measure the size of",
    # Analysis patterns
    "analyze the dependencies",
    "review the code quality",
    "check the test coverage",
    "assess the performance",
    "analyze the imports",
    "review the architecture",
    "check for code smells",
    "assess the complexity",
    "analyze the structure",
    # Comparison patterns
    "compare the implementations",
    "difference between",
    "what changed in",
    "compare versions",
    # Read and explain patterns (code understanding)
    "read and explain",
    "read the code and explain",
    "analyze the code in",
    "analyze this file",
    "analyze the module",
    "review the code in",
    "examine the code in",
    "look at the code and explain",
    "read the file and tell me",
    "summarize the code in",
    "understand the code in",
]

DESIGN_PHRASES = [
    # Conceptual patterns
    "design a system for",
    "plan the architecture",
    "outline the approach",
    "describe how to implement",
    "design the structure",
    "architect a solution",
    "plan the implementation",
    "propose an approach",
    "design a rate limiting system",
    "design a caching system",
    "design authentication",
    # No-code patterns
    "without reading any code",
    "just explain the concept",
    "give me an overview",
    "conceptually describe",
    "explain without looking at code",
    "describe the theory",
    "explain the algorithm",
    "describe the pattern",
    "just list the components",
    "just outline",
    # Planning patterns
    "what would be the best way",
    "how should I approach",
    "what is the recommended",
    "explain the design for",
    "what's the best practice",
    "how would you implement",
    "suggest an approach",
    "recommend a solution",
    "what's the best way to implement",
    "how would you design",
    "what approach should I take",
    # High-level questions
    "what components do I need",
    "what modules should I create",
    "how should I structure",
    "what's the ideal architecture",
    "design a workflow for",
    "outline the steps to",
    "what are the key components for",
    "what's needed for",
    "explain the concepts behind",
    # Explain/compare conceptual patterns
    "explain the difference between",
    "what is the difference between",
    "compare lists and tuples",
    "explain lists vs tuples",
    "difference between lists and",
    "difference between arrays and",
    "explain how X works",
    "describe the difference",
    "what are the differences",
    # "What does X do" patterns (understanding)
    "what does the class do",
    "what does the function do",
    "what does the method do",
    "what does this code do",
    "what does the module do",
    "what does this module do",
    "how does the class work",
    "how does this function work",
    "how does the system work",
    "explain what this does",
    "tell me what this does",
    "describe what this does",
    "what is the purpose of",
    "what is the role of",
]

GENERAL_PHRASES = [
    # Ambiguous help requests - no specific action
    "help me with this",
    "help me with this code",
    "help with this",
    "can you help me",
    "I need help",
    "assist me with",
    "need help with",
    "could you help",
    "please help",
    "what should I do",
    "I'm stuck",
    "having trouble with",
    "not sure what to do",
    # Vague code references
    "look at this code",
    "check this code",
    "review this",
    "what do you think",
    "is this correct",
    "is this ok",
    "any suggestions",
    "what's wrong here",
    # Very short/vague
    "this code",
    "help please",
    "not working",
    "doesn't work",
    "broken",
    "something wrong",
]

# Action phrases: git operations, test runs, script execution
ACTION_PHRASES = [
    # Git commit operations
    "commit all changes",
    "commit these changes",
    "commit the changes",
    "commit my changes",
    "commit with message",
    "stage and commit",
    "git commit",
    "make a commit",
    "create a commit",
    "commit the files",
    "commit everything",
    # Git push/pull operations
    "push to remote",
    "push the changes",
    "git push",
    "pull from remote",
    "git pull",
    "sync with remote",
    "push to origin",
    "push to main",
    "push to master",
    # Git branch operations
    "create a branch",
    "switch to branch",
    "merge the branch",
    "git merge",
    "rebase onto main",
    "cherry pick commit",
    "git stash",
    "stash changes",
    # Pull request operations
    "create a pull request",
    "open a PR",
    "make a PR",
    "create PR",
    "submit pull request",
    "open pull request",
    "create a PR for",
    # Grouped commit operations
    "group commits by",
    "grouped commits",
    "separate commits for",
    "organize commits",
    "split into commits",
    "commit in groups",
    # Test execution
    "run the tests",
    "run all tests",
    "execute tests",
    "run pytest",
    "run unittest",
    "run the test suite",
    "npm test",
    "cargo test",
    "run unit tests",
    "run integration tests",
    "execute the test suite",
    "run e2e tests",
    # Script/command execution
    "run the script",
    "execute the script",
    "run this script",
    "execute this command",
    "run the command",
    "start the server",
    "run the build",
    "execute the build",
    # Build/deploy operations
    "build the project",
    "compile the code",
    "deploy the app",
    "build and deploy",
    "npm build",
    "pip install",
    "cargo build",
    "npm install",
    "install dependencies",
    # File operations
    "move all files",
    "rename the files",
    "copy these files",
    "delete old files",
    "clean up temp files",
    "remove unused files",
    # Web search and API research
    "perform websearch",
    "perform a websearch",
    "perform web search",
    "do a web search",
    "run a websearch",
    "search the web for",
    "search the internet for",
    "search online for",
    "websearch for API",
    "websearch for documentation",
    "fetch the API documentation",
    "fetch for API",
    "fetch API docs",
    "research the API",
    "investigate the library",
    "explore the package API",
    "find the API documentation",
    "look up the API",
]

# Deep analysis phrases: comprehensive codebase exploration
ANALYSIS_DEEP_PHRASES = [
    # Full codebase analysis
    "analyze the entire codebase",
    "analyze the codebase",
    "review the entire project",
    "audit the codebase",
    "analyze the whole project",
    "review the entire codebase",
    "full codebase analysis",
    "complete codebase review",
    "analyze all the code",
    "review all modules",
    # Comprehensive analysis patterns
    "comprehensive code analysis",
    "thorough code review",
    "detailed analysis of",
    "full analysis of",
    "comprehensive review",
    "thorough review of",
    "detailed code review",
    "in-depth analysis",
    "deep dive into the code",
    "extensive code review",
    # Architecture analysis
    "architecture review",
    "architecture analysis",
    "architecture overview",
    "analyze the architecture",
    "review the architecture",
    "understand the architecture",
    "map the architecture",
    "document the architecture",
    # System understanding
    "explain the entire system",
    "describe the whole codebase",
    "understand the full system",
    "map the entire codebase",
    "document the codebase",
    "document all modules",
    # Security/quality audits
    "security audit",
    "security review",
    "vulnerability scan",
    "code quality audit",
    "quality assessment",
    "technical debt analysis",
    "tech debt review",
    "performance analysis",
    "optimization review",
    # Find all patterns
    "identify all issues",
    "find all problems",
    "find every bug",
    "identify all bugs",
    "find all code smells",
    "identify technical debt",
]


class TaskTypeClassifier:
    """Semantic task type classifier using embeddings.

    Replaces hardcoded regex patterns with semantic similarity.
    Benefits:
    - Handles variations and paraphrases
    - More robust to prompt variations
    - Reduces false positives

    Usage:
        # Use singleton to avoid duplicate initialization
        classifier = TaskTypeClassifier.get_instance()
        classifier.initialize_sync()

        result = classifier.classify(
            "Create a simple Python function that calculates factorial"
        )
        print(result.task_type)  # TaskType.CREATE_SIMPLE
        print(result.confidence)  # 0.85
    """

    _instance: Optional["TaskTypeClassifier"] = None
    _lock = __import__("threading").Lock()

    @classmethod
    def get_instance(
        cls,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
        threshold: float = 0.35,
    ) -> "TaskTypeClassifier":
        """Get or create the singleton TaskTypeClassifier instance.

        Args:
            cache_dir: Directory for cache files (only used on first call)
            embedding_service: Shared embedding service (only used on first call)
            threshold: Minimum similarity threshold (only used on first call)

        Returns:
            The singleton TaskTypeClassifier instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        cache_dir=cache_dir,
                        embedding_service=embedding_service,
                        threshold=threshold,
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
            logger.debug("Reset TaskTypeClassifier singleton")

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
        threshold: float = 0.35,
    ):
        """Initialize task type classifier.

        Args:
            cache_dir: Directory for cache files
            embedding_service: Shared embedding service
            threshold: Minimum similarity threshold for classification
        """
        from victor.config.settings import get_project_paths

        self.cache_dir = cache_dir or get_project_paths().global_embeddings_dir
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.threshold = threshold

        # Phrase lists by task type
        self._phrase_lists: Dict[TaskType, List[str]] = {
            TaskType.CREATE_SIMPLE: CREATE_SIMPLE_PHRASES,
            TaskType.CREATE: CREATE_PHRASES,
            TaskType.EDIT: EDIT_PHRASES,
            TaskType.SEARCH: SEARCH_PHRASES,
            TaskType.ANALYZE: ANALYZE_PHRASES,
            TaskType.DESIGN: DESIGN_PHRASES,
            TaskType.GENERAL: GENERAL_PHRASES,
            TaskType.ACTION: ACTION_PHRASES,
            TaskType.ANALYSIS_DEEP: ANALYSIS_DEEP_PHRASES,
        }

        # Single unified collection for all task phrases (1 file instead of 9)
        self._collection = StaticEmbeddingCollection(
            name="task_classifier",
            cache_dir=self.cache_dir,
            embedding_service=self.embedding_service,
        )

        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if classifier is initialized."""
        return self._initialized

    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess prompt to normalize for better embedding matching.

        This strips or normalizes:
        - Full file paths (victor/agent/foo.py → foo.py)
        - Class/function names in PascalCase/snake_case
        - Specific identifiers that cause false semantic matches

        The goal is to preserve the ACTION words and structure while removing
        specific identifiers that can cause semantic similarity issues.

        Args:
            prompt: Original user prompt

        Returns:
            Preprocessed prompt suitable for embedding comparison
        """
        processed = prompt

        # Step 1: Extract and preserve the action prefix (first 1-3 words that matter)
        # This ensures "read and explain" stays intact
        action_patterns = [
            r"^(read\s+and\s+explain)\s+",
            r"^(explain\s+what)\s+",
            r"^(what\s+does)\s+",
            r"^(find\s+all)\s+",
            r"^(list\s+all)\s+",
            r"^(edit)\s+",
            r"^(create)\s+",
            r"^(write)\s+",
            r"^(add)\s+",
            r"^(fix)\s+",
            r"^(modify)\s+",
            r"^(update)\s+",
            r"^(remove)\s+",
            r"^(delete)\s+",
            r"^(search)\s+",
            r"^(count)\s+",
            r"^(how\s+many)\s+",
        ]

        action_prefix = ""
        for pattern in action_patterns:
            match = re.match(pattern, processed, re.IGNORECASE)
            if match:
                action_prefix = match.group(1).lower() + " "
                processed = processed[match.end() :]
                break

        # Step 2: Replace full file paths with just the filename
        # "victor/agent/orchestrator.py" → "orchestrator.py"
        processed = re.sub(
            r"[\w/\\]+/([\w.-]+\.(py|js|ts|go|rs|java|yaml|yml|json|md|txt))",
            r"\1",
            processed,
        )

        # Step 3: Replace PascalCase class names with generic "the class"
        # But preserve certain keywords
        preserve_words = {
            "tool",
            "base",
            "provider",
            "agent",
            "function",
            "method",
            "file",
            "class",
            "module",
            "endpoint",
            "api",
            "test",
        }

        def replace_pascal(match):
            word = match.group(0)
            if word.lower() in preserve_words:
                return word
            return "the class"

        # Match PascalCase words (2+ uppercase letters with lowercase between)
        processed = re.sub(
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",
            replace_pascal,
            processed,
        )

        # Step 4: Reconstruct with preserved action prefix
        result = action_prefix + processed

        # Clean up extra whitespace
        result = re.sub(r"\s+", " ", result).strip()

        logger.debug(f"Preprocessed prompt: '{prompt[:50]}...' → '{result[:50]}...'")
        return result

    def _apply_nudge_rules(
        self,
        prompt: str,
        embedding_result: TaskType,
        embedding_score: float,
        all_scores: Dict[TaskType, float],
    ) -> Tuple[TaskType, float, Optional[str]]:
        """Apply nudge rules to correct embedding classification.

        Nudge rules handle edge cases where semantic similarity fails,
        such as "read and explain" being confused with "edit".

        Args:
            prompt: Original user prompt (not preprocessed)
            embedding_result: Task type from embedding classification
            embedding_score: Confidence score from embedding
            all_scores: Scores for all task types

        Returns:
            Tuple of (final_type, final_score, nudge_name or None)
        """
        for rule in NUDGE_RULES:
            if rule.pattern.search(prompt):
                target_score = all_scores.get(rule.target_type, 0.0)

                if rule.override:
                    # Override: always use the nudged type
                    logger.debug(
                        f"Nudge '{rule.name}' OVERRIDE: {embedding_result.value} → "
                        f"{rule.target_type.value} (was {embedding_score:.2f})"
                    )
                    # Use max of existing score and boost
                    final_score = max(target_score, embedding_score, self.threshold + 0.1)
                    return rule.target_type, final_score, rule.name

                elif embedding_result != rule.target_type:
                    # Boost: add confidence to target type
                    boosted_score = target_score + rule.min_confidence_boost
                    if boosted_score > embedding_score:
                        logger.debug(
                            f"Nudge '{rule.name}' BOOST: {embedding_result.value} → "
                            f"{rule.target_type.value} ({target_score:.2f} + "
                            f"{rule.min_confidence_boost:.2f} = {boosted_score:.2f})"
                        )
                        return rule.target_type, boosted_score, rule.name

        # No nudge applied
        return embedding_result, embedding_score, None

    def initialize_sync(self) -> None:
        """Initialize unified task classifier collection with all phrases."""
        if self._initialized:
            return

        # Build all items with task_type in metadata
        all_items: List[CollectionItem] = []
        for task_type, phrases in self._phrase_lists.items():
            for i, phrase in enumerate(phrases):
                all_items.append(
                    CollectionItem(
                        id=f"{task_type.value}_{i}",
                        text=phrase,
                        metadata={"task_type": task_type.value},
                    )
                )

        # Initialize single unified collection
        self._collection.initialize_sync(all_items)

        self._initialized = True
        logger.info(f"TaskTypeClassifier initialized with {len(all_items)} canonical phrases")

    async def initialize(self) -> None:
        """Initialize unified task classifier collection (async version)."""
        if self._initialized:
            return

        # Build all items with task_type in metadata
        all_items: List[CollectionItem] = []
        for task_type, phrases in self._phrase_lists.items():
            for i, phrase in enumerate(phrases):
                all_items.append(
                    CollectionItem(
                        id=f"{task_type.value}_{i}",
                        text=phrase,
                        metadata={"task_type": task_type.value},
                    )
                )

        # Initialize single unified collection
        await self._collection.initialize(all_items)

        self._initialized = True
        logger.info(f"TaskTypeClassifier initialized with {len(all_items)} canonical phrases")

    def _detect_file_context(self, prompt: str) -> bool:
        """Detect if prompt mentions specific file paths.

        Args:
            prompt: User prompt

        Returns:
            True if prompt contains file path references
        """
        # Common file path patterns
        patterns = [
            r"[\w/\\.-]+\.(py|js|ts|go|rs|java|yaml|yml|json|md|txt)(?:\s|$|,)",
            r"in\s+\w+/\w+",  # "in victor/agent"
            r"to\s+\w+/\w+",  # "to victor/tools"
        ]

        prompt_lower = prompt.lower()
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                return True
        return False

    def classify_sync(self, prompt: str) -> TaskTypeResult:
        """Classify the task type of a user prompt.

        Uses a three-stage approach:
        1. Preprocess: Normalize prompt to strip file paths and identifiers
        2. Embed: Use semantic similarity with preprocessed prompt
        3. Nudge: Apply rule-based corrections for edge cases

        Args:
            prompt: User prompt text

        Returns:
            TaskTypeResult with classified task type
        """
        if not self._initialized:
            self.initialize_sync()

        # Detect file context (on original prompt)
        has_file_context = self._detect_file_context(prompt)

        # Step 1: Preprocess prompt for embedding comparison
        preprocessed = self._preprocess_prompt(prompt)

        # Step 2: Search unified collection and aggregate by task_type
        # Get top 20 results to ensure we see all task types
        results = self._collection.search_sync(preprocessed, top_k=20)

        all_scores: Dict[TaskType, float] = {}
        all_matches: List[Tuple[str, float]] = []

        for item, score in results:
            task_type_str = item.metadata.get("task_type", "general")
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.GENERAL

            # Keep best score per task type
            if task_type not in all_scores or score > all_scores[task_type]:
                all_scores[task_type] = score

            all_matches.append((f"{task_type.value}:{item.text[:40]}", score))

        # Sort matches by score
        all_matches.sort(key=lambda x: x[1], reverse=True)

        # Find best matching task type from embeddings
        if not all_scores:
            return TaskTypeResult(
                task_type=TaskType.GENERAL,
                confidence=0.0,
                top_matches=all_matches[:5],
                has_file_context=has_file_context,
                preprocessed_prompt=preprocessed,
            )

        # Get highest scoring task type from embeddings
        embedding_type = max(all_scores.keys(), key=lambda t: all_scores[t])
        embedding_score = all_scores[embedding_type]

        # Apply threshold check
        if embedding_score < self.threshold:
            # Even below threshold, try nudge rules (they may override)
            final_type, final_score, nudge_name = self._apply_nudge_rules(
                prompt, TaskType.GENERAL, embedding_score, all_scores
            )
            return TaskTypeResult(
                task_type=final_type,
                confidence=final_score,
                top_matches=all_matches[:5],
                has_file_context=has_file_context,
                nudge_applied=nudge_name,
                preprocessed_prompt=preprocessed,
            )

        # Step 3: Apply nudge rules on ORIGINAL prompt
        final_type, final_score, nudge_name = self._apply_nudge_rules(
            prompt, embedding_type, embedding_score, all_scores
        )

        # Special handling: CREATE_SIMPLE vs CREATE
        # If CREATE_SIMPLE has highest score but there's file context, use CREATE instead
        if final_type == TaskType.CREATE_SIMPLE and has_file_context:
            # Check if CREATE score is close enough to use instead
            create_score = all_scores.get(TaskType.CREATE, 0.0)
            if create_score >= self.threshold * 0.8:  # 80% of threshold
                final_type = TaskType.CREATE
                final_score = create_score

        return TaskTypeResult(
            task_type=final_type,
            confidence=final_score,
            top_matches=all_matches[:5],
            has_file_context=has_file_context,
            nudge_applied=nudge_name,
            preprocessed_prompt=preprocessed,
        )

    async def classify(self, prompt: str) -> TaskTypeResult:
        """Classify task type (async version)."""
        if not self._initialized:
            await self.initialize()

        # Use sync version internally (embedding search is fast)
        return self.classify_sync(prompt)

    def clear_cache(self) -> None:
        """Clear cached collections."""
        self._collection.clear_cache()
        self._initialized = False
        logger.info("TaskTypeClassifier cache cleared")
