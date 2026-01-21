# Multi-Agent Research Team

A sophisticated demonstration of Victor AI's multi-agent coordination capabilities, featuring specialized research agents working together using different team formations.

## Features

- **Multiple Agent Personas**: Specialized agents with distinct roles and skills
- **Team Formations**: 5 different coordination strategies
- **Dynamic Task Allocation**: Automatic assignment based on agent capabilities
- **Communication Protocols**: Structured agent-to-agent communication
- **Result Aggregation**: Combines insights from multiple agents
- **Real-time Monitoring**: Watch agents collaborate in real-time
- **Interactive Control**: Direct the research team interactively

## Team Formations

### 1. Pipeline Formation
Sequential processing through specialized stages.

```
Researcher → Analyst → Writer → Reviewer → Final Output
```

**Use Case**: Content creation, report generation

### 2. Parallel Formation
Multiple agents work simultaneously on different aspects.

```
                ┌─ Analyst A ─┐
Researcher ────┼─ Analyst B ─┼─→ Aggregation → Output
                └─ Analyst C ─┘
```

**Use Case**: Comprehensive analysis, multiple perspectives

### 3. Hierarchical Formation
Manager-agent coordination with specialized sub-teams.

```
                    ┌─ Sub-team A ─┐
Team Lead ────┬─────┼─ Sub-team B ─┼─→ Coordination → Output
              │     └─ Sub-team C ─┘
              └─ Direct Analysis ──┘
```

**Use Case**: Complex projects, multi-faceted research

### 4. Sequential Formation
Step-by-step execution with handoffs between agents.

```
Agent 1 → Agent 2 → Agent 3 → Agent 4 → Output
   ↓         ↓         ↓         ↓
Stage 1   Stage 2   Stage 3   Stage 4
```

**Use Case**: Linear workflows, dependencies between steps

### 5. Consensus Formation
Multiple agents propose solutions, vote on best approach.

```
Agent 1 ──┐
Agent 2 ──┼─→ Discussion → Voting → Consensus → Output
Agent 3 ──┘
```

**Use Case**: Decision making, quality assurance

## Installation

```bash
# Navigate to demo directory
cd examples/research_team

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run a research task with default team:

```bash
python main.py research "Analyze the latest trends in AI for 2024"
```

Specify team formation:

```bash
python main.py research "Analyze renewable energy technologies" \
    --formation parallel \
    --roles researcher,analyst,writer
```

Interactive mode:

```bash
python main.py interactive
```

### Web Interface

Start web server:

```bash
python app.py
```

Open browser to `http://localhost:5000`

### Python API

```python
from research_team import ResearchTeam

# Create team
team = ResearchTeam(formation="parallel")

# Add agents
team.add_agent("researcher", persona="Expert in web research...")
team.add_agent("analyst", persona="Expert in data analysis...")
team.add_agent("writer", persona="Expert in technical writing...")

# Execute research
result = team.research("Investigate quantum computing applications")

# Access results
print(result.final_report)
print(result.agent_insights)
```

## Agent Personas

### Research Specialist
- **Skills**: Web search, source evaluation, citation management
- **Tools**: Web search tools, academic databases
- **Role**: Gather information from diverse sources

### Data Analyst
- **Skills**: Statistical analysis, pattern recognition, data visualization
- **Tools**: Analysis tools, charting libraries
- **Role**: Analyze data and identify trends

### Technical Writer
- **Skills**: Technical documentation, clear explanations, structure
- **Tools**: Text editing, markdown formatting
- **Role**: Synthesize findings into coherent reports

### Subject Matter Expert
- **Skills**: Domain expertise, critical analysis, validation
- **Tools**: Domain-specific tools
- **Role**: Provide expert insights and verify accuracy

### Quality Reviewer
- **Skills**: Proofreading, fact-checking, consistency checks
- **Tools**: Validation tools
- **Role**: Ensure quality and accuracy

## Example Workflows

### Example 1: Market Research

```bash
python main.py research \
    "Analyze the electric vehicle market in 2024" \
    --formation parallel \
    --roles "market_researcher,data_analyst,industry_expert,writer"
```

**Team Collaboration**:
1. Market Researcher gathers market data and trends
2. Data Analyst analyzes statistics and patterns
3. Industry Expert provides domain insights
4. Writer synthesizes findings into report
5. Results aggregated into final report

### Example 2: Technical Documentation

```bash
python main.py research \
    "Create documentation for a REST API" \
    --formation pipeline \
    --roles "technical_researcher,developer,writer,reviewer"
```

**Pipeline Execution**:
1. Technical Researcher investigates API specifications
2. Developer validates technical accuracy
3. Writer creates documentation
4. Reviewer checks for completeness

### Example 3: Competitive Analysis

```bash
python main.py research \
    "Compare top 5 cloud providers" \
    --formation hierarchical \
    --roles "team_lead,researcher_a,researcher_b,analyst,writer"
```

**Hierarchical Coordination**:
1. Team Lead coordinates research efforts
2. Researcher A investigates AWS and Azure
3. Researcher B investigates GCP and others
4. Analyst performs comparison
5. Writer creates comparison matrix

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              User Interface                          │
│          (CLI, Web, Python API)                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          Research Team Coordinator                   │
│  - Manages team formations                          │
│  - Coordinates agent communication                  │
│  - Aggregates results                               │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Agent 1   │ │   Agent 2   │ │   Agent 3   │
│  (Research) │ │  (Analysis) │ │  (Writing)  │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌─────────────────┐
              │  Victor AI      │
              │  Orchestrator   │
              └─────────────────┘
```

## Configuration

Create a `team_config.yaml`:

```yaml
# Team configuration
team:
  formation: parallel
  max_iterations: 3
  communication_style: structured

# Agent definitions
agents:
  - name: senior_researcher
    persona: |
      You are a senior researcher with 20 years of experience.
      You excel at finding high-quality sources and synthesizing information.
    tools: [web_search, academic_search, citation_manager]
    capabilities: [research, source_evaluation, citation]

  - name: data_analyst
    persona: |
      You are a data analyst specializing in statistics and trends.
      You identify patterns and extract insights from data.
    tools: [analysis_tools, visualization]
    capabilities: [analysis, statistics, visualization]

  - name: technical_writer
    persona: |
      You are a technical writer who creates clear, concise documentation.
      You excel at explaining complex topics simply.
    tools: [text_editor, markdown_formatter]
    capabilities: [writing, editing, formatting]

# Workflow settings
workflow:
  timeout_seconds: 300
  min_confidence: 0.8
  aggregation_method: weighted_vote
```

## Output Formats

Research teams generate:

### Final Report
```markdown
# Research Report: [Topic]

## Executive Summary
[Concise overview]

## Key Findings
1. Finding 1
2. Finding 2
3. Finding 3

## Detailed Analysis
[In-depth analysis]

## Conclusions
[Key conclusions]

## References
[Citations and sources]
```

### Agent Insights
Each agent provides their perspective:
- Raw findings
- Confidence scores
- Reasoning process
- Recommendations

### Metadata
- Execution time
- Agent participation
- Communication logs
- Tool usage

## Integration with Victor AI

This demo showcases:

### Multi-Agent Teams
```python
from victor.teams import TeamFormation, create_coordinator

coordinator = create_coordinator(
    formation=TeamFormation.PARALLEL,
    roles=[researcher, analyst, writer]
)

result = await coordinator.execute_task(
    task="Research quantum computing",
    context={"max_sources": 10}
)
```

### Agent Personas
```python
from victor.framework import AgentBuilder

builder = AgentBuilder()
agent = builder \
    .with_name("researcher") \
    .with_persona("Expert in...") \
    .with_tools([web_search, citation_manager]) \
    .with_capabilities([research, analysis]) \
    .build()
```

### Memory Systems
Agents share context through:
- Working memory: Current task information
- Long-term memory: Persistent knowledge
- Shared memory: Team communication

## Testing

```bash
# Run tests
pytest tests/

# Test specific formation
pytest tests/test_formations.py::test_parallel

# Integration test
pytest tests/integration/test_team_research.py
```

## Contributing

This is a demo for Victor AI. Contributions welcome!

## License

MIT License

## Support

- **Documentation**: https://victor-ai.readthedocs.io
- **Issues**: https://github.com/your-org/victor-ai/issues
