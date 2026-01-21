# Documentation Generator

An intelligent documentation generation tool powered by Victor AI that automatically creates comprehensive documentation from code using advanced research tools, RAG (Retrieval-Augmented Generation), and workflow orchestration.

## Features

- **Automatic Code Analysis**: Parses code structure and generates API documentation
- **Intelligent Summaries**: Uses AI to understand code purpose and functionality
- **Architecture Diagrams**: Generates visual diagrams showing code structure
- **Interactive Web Interface**: Flask-based web app for easy usage
- **Multiple Output Formats**: Markdown, HTML, PDF, and reStructuredText
- **Searchable Documentation**: RAG-powered search across generated docs
- **Version Tracking**: Tracks documentation changes over time
- **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Java

## Screenshots

The web interface provides:
- Project overview with statistics
- Interactive documentation viewer
- Search functionality
- Export options

## Installation

```bash
# Navigate to demo directory
cd examples/doc_generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface

Start the web server:

```bash
python app.py
```

Then open your browser to `http://localhost:5000`

### CLI Usage

Generate documentation for a project:

```bash
python generate_docs.py /path/to/project --output ./docs
```

With specific options:

```bash
# Generate HTML docs
python generate_docs.py /path/to/project --format html --output ./docs

# Generate with diagrams
python generate_docs.py /path/to/project --include-diagrams

# Search existing documentation
python search_docs.py "API authentication" --index ./docs
```

## Configuration

Create a `doc_config.yaml` in your project:

```yaml
# Documentation configuration
project:
  name: "My Project"
  version: "1.0.0"
  description: "Project description"

documentation:
  # Output format
  format: markdown

  # Sections to include
  sections:
    - overview
    - installation
    - api
    - examples
    - faq

  # Include diagrams
  include_diagrams: true

  # Include type hints
  include_type_hints: true

  # Include examples from docstrings
  include_examples: true

# RAG configuration
rag:
  enabled: true
  chunk_size: 1000
  chunk_overlap: 200

# Provider configuration
provider:
  name: anthropic
  model: claude-sonnet-4-5
```

## Features in Detail

### 1. Code Analysis

The tool analyzes your codebase using Victor AI's coding vertical:

```python
from victor.coding.ast import Parser
from victor.coding.codebase import CodebaseIndexer

parser = Parser()
codebase = CodebaseIndexer()

# Parse all Python files
ast_trees = parser.parse_directory("src/")

# Extract functions, classes, and modules
api_docs = codebase.extract_api_documentation(ast_trees)
```

### 2. AI-Powered Summaries

Uses Victor AI's research capabilities:

```python
from victor.agent.orchestrator_factory import create_orchestrator

orchestrator = create_orchestrator(settings)

# Generate intelligent summaries
summary = await orchestrator.process_request(
    "Generate a summary of this code's purpose and functionality",
    context={"code": code_snippet}
)
```

### 3. RAG-Powered Search

Index and search documentation:

```python
from victor.rag.tools import DocumentIndexer, SemanticSearch

# Index documents
indexer = DocumentIndexer()
indexer.index_documents(docs)

# Semantic search
search = SemanticSearch(indexer)
results = search.query("How do I authenticate users?")
```

### 4. Workflow Orchestration

Automated documentation generation workflow:

```python
from victor.workflows import YAMLWorkflowProvider

provider = YAMLWorkflowProvider("doc_generator")
workflow = provider.compile_workflow("generate_docs")

result = await workflow.invoke({
    "project_path": "/path/to/project",
    "output_format": "markdown"
})
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Web Interface (Flask)                   │
│         (app.py, templates/)                         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         Documentation Generator                      │
│  - Orchestrates analysis pipeline                    │
│  - Manages RAG indexing                              │
│  - Executes workflows                                │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Code      │ │   Research  │ │   RAG       │
│   Analysis  │ │   Tools     │ │   Search    │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌─────────────────┐
              │  Victor AI      │
              │  Orchestrator   │
              └─────────────────┘
```

## Example Output

The tool generates comprehensive documentation including:

### API Documentation

```markdown
# User Service API

## UserService

Service class for user operations with proper error handling.

### Methods

#### `__init__(logger: Optional[logging.Logger] = None)`

Initialize service with dependency injection.

**Parameters:**
- `logger` (Optional[logging.Logger]): Logger instance

#### `get_active_users(self, users: List[User]) -> List[User]`

Filter active users from the list.

**Parameters:**
- `users` (List[User]): List of user objects

**Returns:**
- List[User]: List of active users

**Example:**
```python
service = UserService()
active = service.get_active_users(users)
```
```

### Architecture Diagrams

Generated Mermaid diagrams showing:

- Module dependencies
- Class hierarchies
- Function call graphs
- Data flow diagrams

### Usage Examples

Automatically extracted from docstrings and test files.

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test specific functionality
pytest tests/test_generator.py::test_parse_code
```

## Integration with Victor AI

This demo showcases:

### Research Tools
- Web search for external documentation
- Citation management
- Source aggregation

### RAG Capabilities
- Document chunking and embedding
- Semantic search
- Context retrieval

### Workflow Engine
- YAML-based documentation generation workflow
- Multi-step processing pipeline
- Checkpointing for long-running tasks

## Contributing

This is a demo for Victor AI. Contributions welcome!

## License

MIT License

## Support

- **Documentation**: https://victor-ai.readthedocs.io
- **Issues**: https://github.com/your-org/victor-ai/issues
