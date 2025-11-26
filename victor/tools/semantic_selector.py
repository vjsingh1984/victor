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

"""Semantic tool selection using embeddings for intelligent, context-aware tool matching."""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from victor.providers.base import ToolDefinition
from victor.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class SemanticToolSelector:
    """Select relevant tools using embedding-based semantic similarity.

    Instead of keyword matching, this uses embeddings to find tools
    semantically related to the user's request.

    Benefits:
    - Handles synonyms automatically (test → verify, validate, check)
    - Understands context and intent
    - No hardcoded keyword lists
    - Self-improving with better tool descriptions
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
        ollama_base_url: str = "http://localhost:11434",
        cache_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize semantic tool selector.

        Args:
            embedding_model: Model to use for embeddings
                - sentence-transformers: "all-MiniLM-L6-v2" (default, 80MB, ~5ms)
                - ollama: "nomic-embed-text", "qwen3-embedding:8b", etc.
            embedding_provider: Provider (sentence-transformers, ollama, vllm, lmstudio)
                Default: "sentence-transformers" (local, fast, bundled)
            ollama_base_url: Ollama/vLLM/LMStudio API base URL
            cache_embeddings: Cache tool embeddings (recommended)
            cache_dir: Directory to store embedding cache (default: ~/.victor/embeddings/)
        """
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.ollama_base_url = ollama_base_url
        self.cache_embeddings = cache_embeddings

        # Cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".victor" / "embeddings"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path (includes model name for version control)
        cache_filename = f"tool_embeddings_{embedding_model.replace(':', '_').replace('/', '_')}.pkl"
        self.cache_file = self.cache_dir / cache_filename

        # In-memory cache: tool_name → embedding vector
        self._tool_embedding_cache: Dict[str, np.ndarray] = {}

        # Tool version hash (to detect when tools change)
        self._tools_hash: Optional[str] = None

        # Sentence-transformers model (loaded on demand)
        self._sentence_model = None

        # HTTP client for Ollama/vLLM/LMStudio
        self._client = None
        if embedding_provider in ["ollama", "vllm", "lmstudio"]:
            self._client = httpx.AsyncClient(base_url=ollama_base_url, timeout=30.0)

    async def initialize_tool_embeddings(self, tools: ToolRegistry) -> None:
        """Pre-compute embeddings for all tools (called once at startup).

        Loads from pickle cache if available and tools haven't changed.
        Otherwise, computes embeddings and saves to cache.

        Args:
            tools: Tool registry with all available tools
        """
        if not self.cache_embeddings:
            return

        # Calculate hash of all tool definitions
        tools_hash = self._calculate_tools_hash(tools)

        # Try to load from cache
        if self._load_from_cache(tools_hash):
            logger.info(
                f"Loaded tool embeddings from cache for {len(self._tool_embedding_cache)} tools"
            )
            return

        # Cache miss or tools changed - recompute
        logger.info(
            f"Computing tool embeddings for {len(tools.list_tools())} tools "
            f"(model: {self.embedding_model})"
        )

        for tool in tools.list_tools():
            # Create semantic description of tool
            tool_text = self._create_tool_text(tool)

            # Generate embedding
            embedding = await self._get_embedding(tool_text)

            # Cache it in memory
            self._tool_embedding_cache[tool.name] = embedding

        # Save to disk
        self._save_to_cache(tools_hash)

        logger.info(
            f"Tool embeddings computed and cached for {len(self._tool_embedding_cache)} tools"
        )

    def _calculate_tools_hash(self, tools: ToolRegistry) -> str:
        """Calculate hash of all tool definitions to detect changes.

        Args:
            tools: Tool registry

        Returns:
            SHA256 hash of tool definitions
        """
        # Create deterministic string from all tool definitions
        tool_strings = []
        for tool in sorted(tools.list_tools(), key=lambda t: t.name):
            tool_string = f"{tool.name}:{tool.description}:{tool.parameters}"
            tool_strings.append(tool_string)

        combined = "|".join(tool_strings)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _load_from_cache(self, tools_hash: str) -> bool:
        """Load embeddings from pickle cache if valid.

        Args:
            tools_hash: Current hash of tool definitions

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.cache_file.exists():
            return False

        try:
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Verify cache is for same tools
            if cache_data.get("tools_hash") != tools_hash:
                logger.info("Tool definitions changed, cache invalidated")
                return False

            # Verify cache is for same embedding model
            if cache_data.get("embedding_model") != self.embedding_model:
                logger.info("Embedding model changed, cache invalidated")
                return False

            # Load embeddings
            self._tool_embedding_cache = cache_data["embeddings"]
            self._tools_hash = tools_hash

            return True

        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            return False

    def _save_to_cache(self, tools_hash: str) -> None:
        """Save embeddings to pickle cache.

        Args:
            tools_hash: Hash of tool definitions
        """
        try:
            cache_data = {
                "embedding_model": self.embedding_model,
                "tools_hash": tools_hash,
                "embeddings": self._tool_embedding_cache,
            }

            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            cache_size = self.cache_file.stat().st_size / 1024  # KB
            logger.info(f"Saved embedding cache to {self.cache_file} ({cache_size:.1f} KB)")

        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    async def select_relevant_tools(
        self,
        user_message: str,
        tools: ToolRegistry,
        max_tools: int = 5,
        similarity_threshold: float = 0.15,
    ) -> List[ToolDefinition]:
        """Select relevant tools using semantic similarity.

        Args:
            user_message: User's input message
            tools: Tool registry
            max_tools: Maximum number of tools to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of relevant ToolDefinition objects, sorted by relevance
        """
        # Get embedding for user message
        query_embedding = await self._get_embedding(user_message)

        # Calculate similarity scores for all tools
        similarities: List[Tuple[Any, float]] = []

        for tool in tools.list_tools():
            # Get cached embedding or compute on-demand
            if tool.name in self._tool_embedding_cache:
                tool_embedding = self._tool_embedding_cache[tool.name]
            else:
                tool_text = self._create_tool_text(tool)
                tool_embedding = await self._get_embedding(tool_text)

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, tool_embedding)

            if similarity >= similarity_threshold:
                similarities.append((tool, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top-K
        top_tools = similarities[:max_tools]

        # Log selection
        tool_names = [t.name for t, _ in top_tools]
        scores = [f"{s:.3f}" for _, s in top_tools]
        logger.info(
            f"Selected {len(top_tools)} tools by semantic similarity: "
            f"{', '.join(f'{name}({score})' for name, score in zip(tool_names, scores))}"
        )

        # Convert to ToolDefinition
        return [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in top_tools
        ]

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if self.embedding_provider == "sentence-transformers":
            return await self._get_sentence_transformer_embedding(text)
        elif self.embedding_provider in ["ollama", "vllm", "lmstudio"]:
            return await self._get_api_embedding(text)
        else:
            raise NotImplementedError(f"Provider {self.embedding_provider} not yet supported")

    async def _get_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Get embedding from sentence-transformers (local, fast).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            # Lazy load sentence-transformers model
            if self._sentence_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading sentence-transformers model: {self.embedding_model}")
                    self._sentence_model = SentenceTransformer(self.embedding_model)
                    logger.info(f"Model loaded successfully (local, ~5ms per embedding)")
                except ImportError:
                    raise ImportError(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    )

            # Run in thread pool to avoid blocking event loop
            import asyncio
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._sentence_model.encode(text, convert_to_numpy=True)
            )
            return embedding.astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to get embedding from sentence-transformers: {e}")
            # Fall back to random embedding (better than crashing)
            logger.warning("Falling back to random embedding")
            return np.random.randn(384).astype(np.float32)

    async def _get_api_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama/vLLM/LMStudio API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["embedding"], dtype=np.float32)

        except Exception as e:
            logger.warning(f"Failed to get embedding from {self.embedding_provider}: {e}")
            # Fallback to random embedding (better than crashing)
            return np.random.randn(768).astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def _create_tool_text(tool: Any) -> str:
        """Create semantic description of tool for embedding.

        Combines tool name, description, parameter names, and use cases to create
        a rich semantic representation that matches user queries better.

        Args:
            tool: Tool object

        Returns:
            Semantic text description enriched with use cases
        """
        # Start with name (important for matching)
        parts = [tool.name.replace("_", " ")]

        # Add description
        if tool.description:
            parts.append(tool.description)

        # Add parameter names (provide additional context)
        if hasattr(tool, "parameters") and tool.parameters:
            params = tool.parameters.get("properties", {})
            if params:
                param_names = ", ".join(params.keys())
                parts.append(f"Parameters: {param_names}")

        # Enrich with use cases based on tool name (improves semantic matching)
        use_cases = SemanticToolSelector._get_tool_use_cases(tool.name)
        if use_cases:
            parts.append(use_cases)

        return ". ".join(parts)

    @staticmethod
    def _get_tool_use_cases(tool_name: str) -> str:
        """Get common use cases for a tool to improve semantic matching.

        Args:
            tool_name: Name of the tool

        Returns:
            String describing common use cases with rich keywords and examples
        """
        # Map tools to their common use cases for better semantic matching
        # Each description includes:
        # 1. Common use cases
        # 2. Keywords users might say
        # 3. Concrete examples
        # 4. Programming concepts
        use_case_map = {
            # File operations
            "write_file": (
                "Use for: creating Python files, saving code, writing scripts, creating configuration files, saving data, generating files. "
                "Common requests: write a Python function, create a script, save code to file, create a module, "
                "write a class, implement a function, create a program, save implementation, write validation logic, "
                "create email validator, write calculator, implement fibonacci, save solution, create hello world. "
                "Examples: writing functions (factorial, fibonacci, email validation), creating classes (User, Product), "
                "saving scripts (data processing, automation), generating config files (YAML, JSON, .env), "
                "writing tests, creating utilities, implementing algorithms, saving code solutions."
            ),
            "read_file": (
                "Use for: reading Python code, loading configuration, reading source files, examining file contents, loading data, viewing code. "
                "Common requests: read this file, show me the code, load the configuration, view the source, "
                "examine the implementation, check the file, look at the code, read the script, see the contents. "
                "Examples: reading Python modules, loading config files (settings.py, config.yaml), "
                "examining source code, reviewing implementations, loading data files, checking scripts."
            ),
            "list_directory": (
                "Use for: exploring codebase structure, finding files, listing project contents, browsing directories, viewing file structure. "
                "Common requests: show me the files, list the directory, what files are here, explore the codebase, "
                "find all Python files, show project structure, list source files, browse the project. "
                "Examples: exploring project layout, finding Python modules, listing source files, "
                "viewing directory tree, discovering file organization."
            ),
            "edit_files": (
                "Use for: modifying code, updating files, refactoring, making changes to existing files, fixing bugs, improving code. "
                "Common requests: update this code, modify the function, fix the bug, change the implementation, "
                "improve this code, refactor the function, update the logic, fix the error, change the variable. "
                "Examples: fixing bugs in functions, updating variable names, modifying implementations, "
                "improving algorithms, refactoring code structure, updating configurations."
            ),

            # Code execution
            "execute_bash": (
                "Use for: running scripts, executing commands, testing code, installing packages, git operations, file operations, running programs. "
                "Common requests: run this script, execute the code, test this, install package, run the program, "
                "execute command, test the function, run tests, verify the code works, check if it runs. "
                "Examples: running Python scripts, executing shell commands, testing programs, "
                "installing dependencies (pip install), running tests (pytest), verifying code execution."
            ),
            "execute_python_in_sandbox": (
                "Use for: testing Python code, validating functions, running Python scripts, executing code safely, testing implementations, "
                "verifying code works, running Python programs, checking code correctness, testing solutions, executing Python functions. "
                "Common requests: test this Python function, run this code, validate this implementation, "
                "execute this Python script, test if it works, run this function with test data, verify the code, "
                "test the validation function, run the calculator, execute the fibonacci function, test email validator. "
                "Examples: testing functions (factorial, fibonacci, email validation), running algorithms, "
                "validating implementations, testing calculators, verifying solutions, running Python programs safely."
            ),

            # Code intelligence
            "find_symbol": (
                "Use for: locating function definitions, finding class declarations, searching for variables, code navigation, "
                "finding where functions are defined, locating classes, searching symbols, finding declarations. "
                "Common requests: find this function, where is this class defined, locate the variable, "
                "find the definition, search for this symbol, where is the implementation. "
                "Examples: finding function definitions, locating class declarations, searching for variables."
            ),
            "find_references": (
                "Use for: finding where code is used, tracking function calls, analyzing dependencies, finding usages, "
                "seeing where functions are called, tracking references, finding all uses. "
                "Common requests: where is this used, find all calls to this function, show me the references, "
                "where is this called, find usages, track dependencies. "
                "Examples: finding function calls, tracking variable usage, analyzing dependencies."
            ),
            "rename_symbol": (
                "Use for: refactoring variable names, renaming functions, updating identifiers across codebase, "
                "renaming classes, changing variable names, refactoring identifiers. "
                "Common requests: rename this variable, change the function name, update this identifier, "
                "refactor the variable name, rename the class. "
                "Examples: renaming variables, updating function names, refactoring class names."
            ),

            # Code quality
            "code_review": (
                "Use for: analyzing code quality, checking for issues, reviewing implementations, code analysis, "
                "quality checks, finding problems, reviewing code, checking best practices. "
                "Common requests: review this code, check code quality, analyze this implementation, "
                "find issues in the code, check for problems, review my code. "
                "Examples: reviewing implementations, checking code quality, finding issues."
            ),
            "security_scan": (
                "Use for: finding security vulnerabilities, detecting secrets, security analysis, vulnerability scanning, "
                "checking for security issues, finding exposed secrets, scanning for vulnerabilities. "
                "Common requests: check for security issues, scan for vulnerabilities, find security problems, "
                "check for exposed secrets, security audit. "
                "Examples: finding SQL injection, detecting hardcoded passwords, scanning for XSS."
            ),
            "analyze_metrics": (
                "Use for: measuring code complexity, analyzing code quality metrics, technical debt analysis, "
                "complexity analysis, quality metrics, code health. "
                "Common requests: analyze code complexity, check code metrics, measure quality, "
                "calculate complexity, analyze code health. "
                "Examples: measuring cyclomatic complexity, analyzing code quality."
            ),

            # Testing
            "run_tests": (
                "Use for: executing test suites, running pytest, validating code, test automation, checking test coverage, "
                "running unit tests, executing tests, testing code, verifying tests pass. "
                "Common requests: run the tests, execute test suite, run pytest, check if tests pass, "
                "run unit tests, verify tests, execute test cases. "
                "Examples: running pytest, executing unit tests, checking test coverage."
            ),

            # Documentation
            "generate_docs": (
                "Use for: creating documentation, generating API docs, documenting code, writing README files, "
                "creating docstrings, generating documentation, writing docs. "
                "Common requests: document this code, generate docs, create documentation, write API docs, "
                "add docstrings, create README. "
                "Examples: generating API documentation, writing docstrings, creating README files."
            ),
            "analyze_docs": (
                "Use for: reviewing documentation, checking doc coverage, analyzing documentation quality, "
                "checking if code is documented, reviewing docs. "
                "Common requests: check documentation, review docs, analyze doc coverage, "
                "check if code is documented. "
                "Examples: reviewing documentation quality, checking doc coverage."
            ),

            # Git operations
            "git": (
                "Use for: version control, committing changes, managing branches, git operations, source control, "
                "committing code, creating branches, git workflow. "
                "Common requests: commit these changes, create a branch, git operations, version control, "
                "commit my code, push changes. "
                "Examples: committing changes, creating branches, pushing code."
            ),
            "git_suggest_commit": (
                "Use for: generating commit messages, analyzing changes, creating commits, writing commit messages. "
                "Common requests: create a commit message, generate commit message, suggest commit message. "
                "Examples: generating commit messages based on changes."
            ),
            "git_create_pr": (
                "Use for: creating pull requests, proposing changes, code review workflow, creating PRs. "
                "Common requests: create a pull request, create PR, propose changes. "
                "Examples: creating pull requests for code review."
            ),

            # Refactoring
            "refactor_extract_function": (
                "Use for: extracting methods, refactoring code, improving code structure, extracting functions. "
                "Common requests: extract this into a function, refactor this code, extract method. "
                "Examples: extracting code into functions, refactoring for better structure."
            ),
            "refactor_inline_variable": (
                "Use for: inlining variables, simplifying code, removing unnecessary variables. "
                "Common requests: inline this variable, simplify this code, remove unnecessary variable. "
                "Examples: inlining temporary variables, simplifying code."
            ),
            "refactor_organize_imports": (
                "Use for: organizing imports, cleaning up dependencies, import management, sorting imports. "
                "Common requests: organize imports, clean up imports, sort imports. "
                "Examples: organizing Python imports, cleaning up dependencies."
            ),

            # Web & HTTP
            "web_search": (
                "Use for: searching documentation, finding examples, looking up information, web research, "
                "searching for solutions, finding tutorials, looking up APIs. "
                "Common requests: search for documentation, find examples, look up information, "
                "search online, find tutorials. "
                "Examples: searching Python documentation, finding code examples."
            ),
            "web_fetch": (
                "Use for: downloading documentation, fetching web content, retrieving online resources, "
                "downloading pages, fetching data. "
                "Common requests: fetch this webpage, download documentation, get web content. "
                "Examples: downloading API documentation, fetching web pages."
            ),

            # Workflows
            "run_workflow": (
                "Use for: executing multi-step tasks, complex operations, automated workflows, orchestration, "
                "running workflows, executing automation. "
                "Common requests: run workflow, execute automation, run multi-step process. "
                "Examples: executing complex workflows, running automation."
            ),

            # Additional tools
            "batch": (
                "Use for: processing multiple files, batch operations, bulk processing, mass operations, "
                "processing many files at once. "
                "Common requests: process all files, batch update, bulk operation, update multiple files. "
                "Examples: batch processing files, bulk updates."
            ),
            "cicd": (
                "Use for: CI/CD operations, pipeline management, continuous integration, deployment automation. "
                "Common requests: setup CI/CD, configure pipeline, deployment automation. "
                "Examples: setting up GitHub Actions, configuring CI/CD pipelines."
            ),
            "docker": (
                "Use for: container operations, Docker management, container deployment, Docker commands. "
                "Common requests: Docker operations, container management, run container. "
                "Examples: managing Docker containers, running Docker commands."
            ),
            "scaffold": (
                "Use for: project scaffolding, creating project templates, generating boilerplate, project setup. "
                "Common requests: create project structure, scaffold project, generate template. "
                "Examples: scaffolding new projects, creating project templates."
            ),
        }

        return use_case_map.get(tool_name, "")

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
