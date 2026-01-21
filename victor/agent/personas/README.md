# Dynamic Persona Manager for Victor AI

A comprehensive persona management system that enables adaptive agent behavior through dynamic personality traits, communication styles, and expertise areas.

## Features

### Core Capabilities

- **Dynamic Persona Loading**: Load predefined personas from YAML configuration
- **Context Adaptation**: Adapt personas based on task type, urgency, and user preferences
- **Custom Persona Creation**: Create fully customized personas with trait-based system
- **Automatic Persona Suggestion**: AI-powered persona matching for tasks
- **Persona Merging**: Combine multiple personas into hybrid specialists
- **Validation System**: Comprehensive validation of persona definitions
- **Feedback Integration**: Improve personas through user feedback
- **Export/Import**: Share personas across instances

## Architecture

### Key Components

```
victor/agent/personas/
├── __init__.py                 # Public API exports
├── types.py                    # Core data types (Persona, AdaptedPersona, etc.)
├── persona_manager.py          # PersonaManager class (968 lines)
├── persona_repository.py       # Storage and persistence layer
├── usage_examples.py           # Comprehensive usage examples
└── README.md                   # This file
```

### Data Structures

#### Persona
Base persona definition with:
- `id`: Unique identifier
- `name`: Display name
- `description`: Brief description
- `personality`: Core personality type (PersonalityType enum)
- `communication_style`: Communication preferences (CommunicationStyle enum)
- `expertise`: List of expertise areas
- `backstory`: Optional background context
- `constraints`: Behavioral constraints (PersonaConstraints)
- `prompt_templates`: Custom prompt templates (PromptTemplates)

#### AdaptedPersona
Context-aware persona with:
- `base_persona`: Original persona
- `context_adjustments`: Applied adjustments
- `dynamic_traits`: Calculated traits
- `adaptation_reason`: Explanation of changes

#### PersonaConstraints
Behavioral constraints:
- `max_tool_calls`: Tool call limit
- `preferred_tools`: Tools to prefer
- `forbidden_tools`: Tools to avoid
- `response_length`: Target response length (short/medium/long)
- `explanation_depth`: Explanation depth (brief/standard/detailed)

## Usage

### Basic Usage

```python
from victor.agent.personas import PersonaManager

# Initialize manager
manager = PersonaManager()

# Load predefined persona
persona = manager.load_persona("senior_developer")

# Access persona properties
print(persona.name)              # "Senior Developer"
print(persona.personality.value)  # "pragmatic"
print(persona.expertise)          # ["coding", "debugging", ...]
```

### Context Adaptation

```python
# Adapt persona to context
adapted = manager.adapt_persona(
    persona,
    context={
        "task_type": "security_review",
        "urgency": "high",
        "complexity": "high",
        "user_preference": "thorough",
    }
)

# Access adapted properties
print(adapted.personality)  # May differ from base
print(adapted.dynamic_traits)  # Calculated traits
```

### Custom Persona Creation

```python
from victor.agent.personas.types import PersonalityType, CommunicationStyle

# Create custom persona
custom = manager.create_custom_persona(
    name="API Design Specialist",
    traits={
        "personality": PersonalityType.SYSTEMATIC,
        "communication_style": CommunicationStyle.TECHNICAL,
        "description": "Expert in API design and microservices",
        "expertise": ["api_design", "rest", "graphql"],
        "backstory": "Years of experience building scalable APIs...",
        "constraints": {
            "max_tool_calls": 50,
            "response_length": "long",
        }
    }
)
```

### Automatic Persona Suggestion

```python
# Get suggested persona for task
task = "Review this code for security vulnerabilities"
suggested = manager.get_suggested_persona(task)

if suggested:
    print(f"Recommended: {suggested.name}")
    # Use suggested persona for the task
```

### Persona Merging

```python
# Load multiple personas
security = manager.load_persona("security_expert")
performance = manager.load_persona("performance_specialist")

# Merge into hybrid
hybrid = manager.merge_personas(
    personas=[security, performance],
    merged_name="Secure & Performant Architect",
    merged_id="secure_perf_architect"
)

# Hybrid combines expertise from both
print(hybrid.expertise)  # Combined expertise areas
```

### Validation

```python
# Validate persona
try:
    manager.validate_persona(persona)
    print("✓ Persona is valid")
except ValueError as e:
    print(f"✗ Invalid: {e}")
```

### Feedback Integration

```python
from victor.agent.personas.types import Feedback

# Create feedback
feedback = Feedback(
    persona_id=persona.id,
    success_rating=4.5,
    user_comments="Great, but add more testing expertise",
    suggested_improvements={
        "add_expertise": ["testing", "unit_tests"]
    }
)

# Apply feedback
manager.update_persona_from_feedback(persona.id, feedback)
```

## Predefined Personas

Victor AI includes 10 predefined personas:

| Persona | Personality | Best For |
|---------|------------|----------|
| `senior_developer` | Pragmatic | General coding, debugging |
| `security_expert` | Cautious | Security reviews, vulnerability assessment |
| `researcher` | Curious | Investigation, documentation |
| `architect` | Systematic | System design, architecture |
| `code_reviewer` | Critical | Code review, quality assurance |
| `mentor` | Supportive | Teaching, learning guidance |
| `performance_specialist` | Pragmatic | Performance optimization |
| `devops_engineer` | Systematic | Deployment, infrastructure |
| `data_scientist` | Methodical | Data analysis, ML |
| (Custom) | (Any) | Specialized use cases |

## Personality Types

- **CURIOUS**: Exploratory, inquisitive, research-oriented
- **CAUTIOUS**: Careful, security-conscious, risk-averse
- **CREATIVE**: Innovative, design-focused, imaginative
- **PRAGMATIC**: Practical, balanced, results-driven
- **SYSTEMATIC**: Organized, methodical, structured
- **CRITICAL**: Analytical, detail-oriented, quality-focused
- **SUPPORTIVE**: Helpful, educational, encouraging
- **METHODICAL**: Thorough, precise, step-by-step

## Communication Styles

- **CONCISE**: Brief, to-the-point responses
- **VERBOSE**: Detailed, comprehensive explanations
- **FORMAL**: Professional, structured communication
- **CASUAL**: Friendly, conversational tone
- **TECHNICAL**: Jargon-appropriate, technical depth
- **ACCESSIBLE**: Simple, clear, easy to understand
- **EDUCATIONAL**: Teaching-focused, explanatory
- **DIRECT**: Straightforward, no-nonsense
- **CONSTRUCTIVE**: Feedback-oriented, improvement-focused

## Integration with Victor AI

### With Orchestrator

```python
from victor.agent.orchestrator_factory import OrchestratorFactory
from victor.agent.personas import PersonaManager

# Create orchestrator
factory = OrchestratorFactory(settings, provider, model)
orchestrator = factory.create_orchestrator()

# Apply persona
manager = PersonaManager()
persona = manager.load_persona("senior_developer")

# Adapt persona to current context
adapted = manager.adapt_persona(
    persona,
    context={"task_type": "code_review", "urgency": "normal"}
)

# Generate and set system prompt
system_prompt = adapted.generate_system_prompt()
orchestrator.system_prompt = system_prompt
```

### With Framework State

```python
from victor.framework import State

# Access agent state
state = agent.state

# Adapt persona based on state
context = {
    "task_type": "debugging",
    "complexity": "high" if state.tool_calls_used > 30 else "medium",
    "urgency": "high" if state.tools_remaining < 10 else "normal",
}

adapted = manager.adapt_persona(persona, context)
```

## Configuration

Personas are defined in YAML:
```yaml
# victor/config/personas/agent_personas.yaml

personas:
  my_persona:
    id: my_persona
    name: My Custom Persona
    description: A specialized persona
    personality: pragmatic
    communication_style: technical
    expertise:
      - coding
      - debugging
    constraints:
      max_tool_calls: 50
      response_length: medium
```

## API Reference

### PersonaManager

#### Methods

- `load_persona(persona_id: str) -> Persona`
  Load a persona by ID

- `adapt_persona(persona: Persona, context: Dict) -> AdaptedPersona`
  Adapt a persona based on context

- `create_custom_persona(name: str, traits: Dict) -> Persona`
  Create a custom persona

- `get_suggested_persona(task: str) -> Optional[Persona]`
  Get best-matching persona for task

- `get_suitable_personas(task: str, min_score: float) -> List[Tuple[Persona, float]]`
  Get all suitable personas ranked by score

- `merge_personas(personas: List[Persona], merged_name: str) -> Persona`
  Merge multiple personas into hybrid

- `validate_persona(persona: Persona) -> None`
  Validate persona definition

- `update_persona_from_feedback(persona_id: str, feedback: Feedback) -> None`
  Update persona from user feedback

- `export_persona(persona_id: str) -> Dict`
  Export persona for sharing

- `import_persona(definition: Dict) -> Persona`
  Import persona from definition

## Examples

See `usage_examples.py` for comprehensive examples:
- Loading predefined personas
- Context adaptation
- Custom persona creation
- Automatic suggestion
- Persona merging
- Validation
- Feedback loops
- Export/import
- Complete workflow

Run examples:
```bash
python -m victor.agent.personas.usage_examples
```

## Testing

```bash
# Run persona tests
pytest tests/unit/personas/ -v

# Run specific test
pytest tests/unit/personas/test_persona_manager.py::test_merge_personas -v
```

## Best Practices

1. **Choose personas based on task type**: Use `get_suggested_persona()` for automatic selection
2. **Adapt to context**: Always adapt personas to current task context
3. **Validate custom personas**: Use `validate_persona()` before using custom personas
4. **Provide feedback**: Use feedback mechanism to improve personas over time
5. **Merge carefully**: Only merge compatible personalities (e.g., avoid CAUTIOUS + CREATIVE)
6. **Consider expertise**: Ensure persona expertise matches task requirements
7. **Respect constraints**: Persona constraints help maintain consistent behavior

## Advanced Features

### Dynamic Traits

Personas can calculate dynamic traits based on context:
- Task complexity handling
- Efficiency focus (urgency-driven)
- Activated expertise (relevant to task)
- Communication adjustments

### Context Adjustments

Context can trigger automatic adjustments:
- **Security review**: Switch to CAUTIOUS + FORMAL
- **Debugging**: Switch to METHODICAL + detailed explanations
- **High urgency**: Switch to CONCISE + short responses
- **User preference**: Adjust explanation depth

### Caching

Adapted personas are cached for performance:
```python
# Same context = cached result
adapted1 = manager.adapt_persona(persona, context)
adapted2 = manager.adapt_persona(persona, context)  # Returns cached version
```

## License

Apache License 2.0 - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

### Adding New Personas

1. Define in `victor/config/personas/agent_personas.yaml`
2. Follow existing persona structure
3. Include clear expertise areas
4. Add appropriate constraints
5. Test with `validate_persona()`
