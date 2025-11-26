# GEMINI.md

## Project Overview

This project, "Victor," is a universal terminal-based AI coding assistant built in Python. It provides a unified interface to interact with various Large Language Model (LLM) providers, including frontier models like Claude, GPT-4, and Gemini, as well as locally run open-source models through Ollama, LMStudio, and vLLM.

The primary goal of Victor is to offer a powerful and cost-effective tool for developers, enabling them to leverage AI for a wide range of coding tasks without being locked into a single provider. It includes a rich set of over 25 enterprise-grade tools for professional development workflows.

### Key Features:

*   **Universal Provider Support:** Switch between different LLM providers on the fly.
*   **Cost-Effective:** Supports free, local models for development and testing.
*   **Enterprise-Grade Tools:** Includes tools for batch processing, code refactoring, test generation, CI/CD automation, documentation, dependency management, code metrics, and more.
*   **Security-Focused:** Features for secret detection, vulnerability scanning, and dependency auditing.
*   **Extensible Architecture:** Designed to be easily extended with new providers, tools, and capabilities.

### Core Technologies:

*   **Python:** The project is written in Python and requires version 3.10 or higher.
*   **Typer/Rich:** The command-line interface is built using Typer for command parsing and Rich for a better terminal UI.
*   **Pydantic:** Used for data validation and settings management.
*   **HTTPX:** For asynchronous HTTP requests to LLM provider APIs.
*   **Tree-sitter:** For abstract syntax tree (AST) based code analysis and manipulation.

## Building and Running

### 1. Set Up the Environment

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vijaysingh/victor.git
    cd victor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:** The project uses `setuptools` for packaging. The development dependencies, including testing and linting tools, are listed in `pyproject.toml` under `[project.optional-dependencies]`. To install the project in editable mode with all development dependencies, use the following command:
    ```bash
    pip install -e ".[dev]"
    ```

### 2. Configure API Keys

Victor requires API keys for the LLM providers you intend to use. These can be set as environment variables.

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export XAI_API_KEY="your-key-here"
```

### 3. Running the Application

The `pyproject.toml` file defines a script entry point, allowing you to run the application using the `victor` command.

*   **Interactive REPL mode:**
    ```bash
    victor
    ```

*   **One-shot command:**
    ```bash
    victor "Write a Python function to calculate Fibonacci numbers"
    ```

### 4. Running Tests

The project uses `pytest` for testing.

*   **Run all tests:**
    ```bash
    pytest
    ```

*   **Run tests with coverage:**
    ```bash
    pytest --cov
    ```

## Development Conventions

### Code Style and Linting

*   **Formatter:** The project uses `black` for code formatting.
*   **Linter:** `ruff` is used for linting.
*   **Type Checking:** `mypy` is used for static type checking.

You can run these tools from the command line:

```bash
# Format code
black victor tests

# Lint
ruff check victor tests

# Type check
mypy victor
```

### Pre-commit Hooks

The project is set up to use pre-commit hooks to automatically run formatting and linting before each commit. To install the hooks, run:

```bash
pre-commit install
```

### Contribution Guidelines

The `CONTRIBUTING.md` file provides guidelines for contributing to the project. It is recommended to review this file before making any contributions.
