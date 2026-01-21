# Victor AI Demo Applications - Complete Guide

Comprehensive guide to all 5 production-quality demo applications showcasing Victor AI capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Demo Applications](#demo-applications)
3. [Installation](#installation)
4. [Victor AI Integration Points](#victor-ai-integration-points)
5. [Architecture Patterns](#architecture-patterns)
6. [Learning Path](#learning-path)
7. [Extending Demos](#extending-demos)
8. [Best Practices](#best-practices)

## Overview

Victor AI demo applications are designed to:

- **Showcase Real-World Use Cases**: Each demo solves practical problems
- **Demonstrate Capabilities**: Highlight specific Victor AI features
- **Provide Starting Points**: Production-quality code you can build upon
- **Teach Best Practices**: Show proper integration patterns

### Demo Matrix

| Demo | Verticals | Complexity | Time to Complete |
|------|-----------|------------|------------------|
| Code Review Assistant | Coding | ⭐⭐ | 30 minutes |
| Documentation Generator | Research, RAG | ⭐⭐⭐ | 45 minutes |
| Data Analysis Dashboard | DataAnalysis | ⭐⭐ | 30 minutes |
| Multi-Agent Research Team | Teams, Research | ⭐⭐⭐⭐ | 60 minutes |
| Multimodal Assistant | Multimodal | ⭐⭐⭐ | 45 minutes |

## Demo Applications

### 1. Code Review Assistant

**Location**: `examples/code_review_assistant/`

**Purpose**: Automated code review using AST analysis, security scanning, and quality metrics.

**Key Features**:
- AST-based code parsing (Python, JavaScript, TypeScript, Go, Rust, Java)
- Security vulnerability detection (SQL injection, XSS, hardcoded secrets)
- Quality metrics (cyclomatic complexity, code duplication)
- Style checking (PEP 8, line length, formatting)
- Multi-agent team reviews with different formations
- Report generation (HTML, JSON, Markdown)

**Victor AI Components Used**:
```python
# Core orchestrator
from victor.agent.orchestrator_factory import create_orchestrator

# Coding vertical
from victor.coding.ast import Parser
from victor.coding.review import CodeReviewer
from victor.coding.lsp import LSPClient

# Multi-agent teams
from victor.teams import TeamFormation, create_coordinator
```

**Quick Start**:
```bash
cd examples/code_review_assistant
pip install -r requirements.txt

# Review code
python src/main.py review sample_code/example.py

# Generate HTML report
python src/main.py report sample_code/ --format html --output report.html

# Multi-agent review
python src/main.py team-review sample_code/ --formation parallel
```

**Learning Outcomes**:
- Using the coding vertical for AST analysis
- Implementing security scanners
- Coordinating multi-agent teams
- Generating formatted reports

---

### 2. Documentation Generator

**Location**: `examples/doc_generator/`

**Purpose**: AI-powered documentation generation with RAG-powered search.

**Key Features**:
- Automatic code structure analysis
- AI-generated summaries and descriptions
- Architecture diagram generation (Mermaid)
- RAG-powered documentation search
- Web interface for easy usage
- Multiple output formats (Markdown, HTML, PDF)

**Victor AI Components Used**:
```python
# Codebase analysis
from victor.coding.codebase import CodebaseIndexer

# Research tools
from victor.research import WebSearchTool, CitationManager

# RAG capabilities
from victor.rag.tools import DocumentIndexer, SemanticSearch

# Workflow orchestration
from victor.workflows import YAMLWorkflowProvider
```

**Quick Start**:
```bash
cd examples/doc_generator
pip install -r requirements.txt

# Start web interface
python app.py

# Or use CLI
python generate_docs.py /path/to/project --output ./docs
```

**Learning Outcomes**:
- Using RAG for document search
- Orchestrating complex workflows
- Building web interfaces
- Generating diagrams programmatically

---

### 3. Data Analysis Dashboard

**Location**: `examples/data_dashboard/`

**Purpose**: Interactive data analysis with AI-powered insights and visualizations.

**Key Features**:
- Smart data loading (CSV, Excel, JSON, Parquet)
- Automated data profiling and statistics
- Interactive visualizations (Plotly)
- Natural language querying
- Anomaly detection
- Statistical analysis (correlations, distributions)
- Export reports

**Victor AI Components Used**:
```python
# Data analysis tools
from victor.dataanalysis import DataProfiler, InsightGenerator

# Statistical analysis
from victor.dataanalysis import StatisticalAnalyzer, AutoVisualizer

# Natural language queries
from victor.dataanalysis import NLQueryEngine
```

**Quick Start**:
```bash
cd examples/data_dashboard
pip install -r requirements.txt

# Start Streamlit dashboard
streamlit run app.py
```

**Learning Outcomes**:
- Building interactive dashboards
- Natural language query processing
- Statistical analysis automation
- Visualization selection

---

### 4. Multi-Agent Research Team

**Location**: `examples/research_team/`

**Purpose**: Demonstrate advanced multi-agent coordination with 5 different team formations.

**Key Features**:
- 5 team formations (pipeline, parallel, hierarchical, sequential, consensus)
- Specialized agent personas (researcher, analyst, writer, reviewer)
- Dynamic task allocation based on capabilities
- Agent communication protocols
- Result aggregation and consensus
- Real-time monitoring
- Interactive control

**Victor AI Components Used**:
```python
# Multi-agent teams
from victor.teams import TeamFormation, create_coordinator

# Framework capabilities
from victor.framework import AgentBuilder, AgentSession

# Research tools
from victor.research import WebSearchTool, AcademicSearchTool
```

**Quick Start**:
```bash
cd examples/research_team
pip install -r requirements.txt

# Run research task
python main.py research "Analyze AI trends in 2024" --formation parallel

# List available agents
python main.py list-agents

# Interactive mode
python main.py interactive
```

**Learning Outcomes**:
- Multi-agent coordination patterns
- Team formation strategies
- Agent persona design
- Result aggregation

---

### 5. Multimodal Assistant

**Location**: `examples/multimodal_assistant/`

**Purpose**: Demonstrate vision, audio, and OCR processing capabilities.

**Key Features**:
- Image analysis and description
- Object detection in images
- Audio transcription (speech-to-text)
- OCR text extraction
- Video analysis
- Cross-modal queries (combining image + text + audio)
- Batch processing

**Victor AI Components Used**:
```python
# Vision processing
from victor.tools.vision import ImageAnalyzer, OCRProcessor

# Audio processing
from victor.tools.audio import AudioTranscriber

# Multimodal engine
from victor.multimodal import CrossModalEngine
```

**Quick Start**:
```bash
cd examples/multimodal_assistant
pip install -r requirements.txt

# Analyze image
python main.py analyze-image photo.jpg --question "What objects are visible?"

# Transcribe audio
python main.py transcribe recording.wav

# Extract text (OCR)
python main.py ocr document.png --output text.txt

# Cross-modal query
python main.py query --image chart.png "What data is shown?"
```

**Learning Outcomes**:
- Working with multimodal inputs
- Image and audio processing
- OCR text extraction
- Cross-modal understanding

## Installation

### Prerequisites

All demos require:

- Python 3.10 or higher
- Virtual environment (recommended)
- Victor AI installed from parent directory

### General Installation Pattern

```bash
# 1. Navigate to demo directory
cd examples/<demo_name>

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
python main.py --help
```

### System Dependencies

Some demos require additional system packages:

**Audio/Video** (Multimodal Assistant):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile
```

**OCR** (Multimodal Assistant):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

## Victor AI Integration Points

Each demo demonstrates specific integration points with Victor AI:

### 1. Agent Orchestrator

All demos use the agent orchestrator as the central coordination point:

```python
from victor.agent.orchestrator_factory import create_orchestrator
from victor.config.settings import Settings

settings = Settings()
orchestrator = create_orchestrator(settings)

# Process requests
result = await orchestrator.process_request(
    "Your task here",
    context={"key": "value"}
)
```

### 2. Vertical-Specific Tools

Each vertical provides specialized tools:

**Coding Vertical**:
```python
from victor.coding.ast import Parser
from victor.coding.review import CodeReviewer

parser = Parser()
ast = parser.parse(source_code, language="python")

reviewer = CodeReviewer(orchestrator)
issues = await reviewer.analyze(ast, file_path)
```

**Research Vertical**:
```python
from victor.research import WebSearchTool, CitationManager

search_tool = WebSearchTool()
results = await search_tool.search("AI trends 2024")

citation_mgr = CitationManager()
citations = citation_mgr.format_citations(results)
```

**RAG Capabilities**:
```python
from victor.rag.tools import DocumentIndexer, SemanticSearch

indexer = DocumentIndexer()
indexer.index_documents(documents)

search = SemanticSearch(indexer)
results = search.query("How does X work?")
```

### 3. Multi-Agent Teams

```python
from victor.teams import TeamFormation, create_coordinator

coordinator = create_coordinator(
    formation=TeamFormation.PARALLEL,
    roles=[role1, role2, role3]
)

result = await coordinator.execute_task(
    task="Research topic",
    context={"max_sources": 10}
)
```

### 4. Framework Capabilities

```python
from victor.framework import AgentBuilder, StateGraph

# Build custom agent
builder = AgentBuilder()
agent = builder \
    .with_name("my_agent") \
    .with_persona("Expert in...") \
    .with_tools([tool1, tool2]) \
    .build()

# Create workflow
graph = StateGraph()
graph.add_node("process", agent_node)
graph.add_edge("start", "process")
graph.add_edge("process", "end")
```

## Architecture Patterns

All demos follow consistent patterns:

### Pattern 1: Orchestrator as Facade

```python
class DemoApplication:
    def __init__(self):
        self.orchestrator = create_orchestrator(settings)
        self.tools = self._initialize_tools()
        self.coordinators = self._initialize_coordinators()

    async def process(self, input_data):
        # Use orchestrator for coordination
        result = await self.orchestrator.process_request(
            input_data,
            context=self._build_context(input_data)
        )
        return result
```

### Pattern 2: Protocol-Based Design

```python
from victor.protocols import ToolExecutorProtocol, ITeamCoordinator

class MyComponent:
    def __init__(self, tool_executor: ToolExecutorProtocol):
        self._tool_executor = tool_executor

    def execute(self, tool_name: str, **kwargs):
        return self._tool_executor.execute_tool(tool_name, kwargs)
```

### Pattern 3: Dependency Injection

```python
from victor.core.container import ServiceContainer

container = ServiceContainer()
container.register(
    ToolExecutorProtocol,
    lambda c: ToolExecutor(tool_registry=c.get(ToolRegistryProtocol)),
    ServiceLifetime.SINGLETON
)

tool_executor = container.get(ToolExecutorProtocol)
```

## Learning Path

We recommend exploring demos in this order:

### Level 1: Basics (Days 1-2)

1. **Code Review Assistant** - Learn Victor AI fundamentals
2. **Data Analysis Dashboard** - Understand data processing

### Level 2: Intermediate (Days 3-4)

3. **Documentation Generator** - Workflows and RAG
4. **Multimodal Assistant** - Cross-modal processing

### Level 3: Advanced (Days 5-7)

5. **Multi-Agent Research Team** - Advanced coordination

### Suggested Timeline

**Week 1**: Complete all demos
- Days 1-2: Code Review Assistant + Data Dashboard
- Days 3-4: Documentation Generator + Multimodal Assistant
- Days 5-7: Multi-Agent Research Team

**Week 2**: Extend and customize
- Choose one demo to extend
- Add new features
- Integrate with your own data/tools

## Extending Demos

### Adding New Features

1. **Create new module** in `src/` directory
2. **Import Victor AI components** from canonical locations
3. **Follow existing patterns** for consistency
4. **Add tests** in `tests/` directory
5. **Update README** with new capabilities

### Example: Adding a New Analyzer

```python
# src/my_analyzer.py
from victor.agent.orchestrator_factory import create_orchestrator
from victor.protocols import ToolExecutorProtocol

class MyAnalyzer:
    """Custom analyzer using Victor AI."""

    def __init__(self, tool_executor: ToolExecutorProtocol):
        self.orchestrator = create_orchestrator(settings)
        self.tool_executor = tool_executor

    async def analyze(self, data: str) -> dict:
        """Analyze data using Victor AI tools."""

        # Use tools
        result = await self.tool_executor.execute_tool(
            "my_tool",
            {"data": data}
        )

        # Use orchestrator for AI processing
        insights = await self.orchestrator.process_request(
            "Analyze this data",
            context={"result": result}
        )

        return {
            "result": result,
            "insights": insights
        }
```

### Integration Checklist

When extending demos, ensure:

- [ ] Import from canonical locations (see CLAUDE.md)
- [ ] Use protocols for loose coupling
- [ ] Add type hints to public APIs
- [ ] Write docstrings (Google-style)
- [ ] Include error handling
- [ ] Add tests for new functionality
- [ ] Update documentation

## Best Practices

### 1. Code Organization

```
demo_name/
├── README.md              # User-facing documentation
├── requirements.txt       # Dependencies
├── app.py                # Entry point (if web app)
├── main.py               # Entry point (if CLI)
├── src/                  # Source code
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── processor.py      # Core processing logic
│   └── utils.py          # Helper functions
├── tests/                # Tests
│   ├── test_processor.py
│   └── conftest.py
└── sample_data/          # Example data
```

### 2. Error Handling

```python
from victor.providers.error_handler import handle_provider_errors

@handle_provider_errors
async def process_data(data):
    try:
        result = await orchestrator.process_request(data)
        return result
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise
    except ProviderError as e:
        logger.error(f"Provider error: {e}")
        # Retry or fallback
        return await fallback_strategy(data)
```

### 3. Logging

```python
import logging

logger = logging.getLogger(__name__)

def process(data):
    logger.info(f"Processing data: {data[:50]}...")
    try:
        result = do_processing(data)
        logger.info(f"Successfully processed: {result}")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
```

### 4. Configuration

```python
# Use YAML for configuration
# config.yaml
provider:
  name: anthropic
  model: claude-sonnet-4-5

settings:
  timeout: 30
  max_retries: 3

# Load config
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

### 5. Testing

```python
import pytest
from unittest.mock import Mock, patch

def test_analyzer():
    # Arrange
    mock_orchestrator = Mock()
    mock_orchestrator.process_request.return_value = "result"

    analyzer = MyAnalyzer(mock_orchestrator)

    # Act
    result = analyzer.analyze("test data")

    # Assert
    assert result == "expected result"
    mock_orchestrator.process_request.assert_called_once()
```

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Solution: Ensure Victor AI is installed in development mode
cd /path/to/victor-ai
pip install -e .
```

**Missing Dependencies**:
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Provider Errors**:
```bash
# Solution: Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

**System Dependencies**:
```bash
# Solution: Install required system packages
# See installation section above
```

## Resources

- **Main Documentation**: https://victor-ai.readthedocs.io
- **Architecture Guide**: `docs/architecture/`
- **Best Practices**: `CLAUDE.md`
- **GitHub Issues**: https://github.com/your-org/victor-ai/issues

## Contributing

We welcome contributions to the demo applications!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Ensure all tests pass
6. Submit a pull request

## License

All demo applications are licensed under the MIT License.

## Acknowledgments

These demos showcase the power and flexibility of Victor AI:

- **Vertical Architecture** - Domain-specific capabilities
- **Provider Agnosticism** - 21+ LLM providers
- **Multi-Agent Coordination** - Advanced team formations
- **Framework Capabilities** - Reusable building blocks
- **Event-Driven Design** - Scalable and maintainable

Each demo is production-quality and ready for use as a starting point for your own applications.

---

**Happy Coding with Victor AI!** 🚀
