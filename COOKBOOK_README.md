# Victor Cookbook

> 100+ production-ready recipes for the Victor AI Framework

Victor Cookbook provides copy-paste ready code recipes for common AI agent tasks, workflows, and integrations. Each recipe is designed to be immediately useful in production applications.

## Installation

```bash
pip install victor-cookbook
```

Or install with development dependencies:

```bash
pip install victor-cookbook[dev]
```

## Quick Start

### Using a Recipe

```python
from victor import Agent
from victor_cookbook.loader import RecipeLoader

# Load and use a recipe
loader = RecipeLoader()
recipe = loader.load_recipe("agents/basic", "text_generation")

# Recipe returns a callable function
agent = Agent.create()
result = await recipe["function"]("Write a story about AI")
print(result.content)
```

### List Available Recipes

```python
from victor_cookbook.loader import RecipeLoader

loader = RecipeLoader()

# List all recipes
all_recipes = loader.list_recipes()
for category, recipes in all_recipes.items():
    print(f"\n{category}:")
    for recipe in recipes:
        print(f"  - {recipe}")

# Search for recipes
matching = loader.search_recipes("slack")
for recipe in matching:
    print(f"{recipe['category']}/{recipe['name']}: {recipe['description']}")
```

## Recipe Categories

### Agents (`agents/`)

Basic and specialized agent configurations for common tasks.

**Basic Agents** (`agents/basic/`)
- `simple_qa` - Answer simple questions
- `text_generation` - Generate text on any topic
- `text_summarization` - Summarize long text
- `text_translation` - Translate between languages
- `creative_writing` - Generate creative content
- `brainstorming` - Brainstorm ideas
- `code_explanation` - Explain how code works
- `concept_explanation` - Explain concepts for any audience
- `email_drafting` - Draft professional emails
- `social_media_post` - Create social media content
- And 10+ more basic agent patterns

**Production Agents** (`agents/production/`)
- `ProductionAgent` - Agent with retry logic and monitoring
- `StreamingAgent` - Real-time streaming responses
- `CircuitBreaker` - Circuit breaker for failing LLM calls
- `MultiAgentSystem` - Coordinated multi-agent patterns
- `error_handling_wrapper` - Error handling patterns
- `fallback_chain` - Fallback to backup providers

**Specialized Agents** (`agents/specialized/`)
- **Coding Agents** (18 recipes) - Code review, refactoring, debugging, testing, documentation
- **Research Agents** (16 recipes) - Literature review, fact-checking, citation verification
- **Data Agents** (16 recipes) - Data exploration, cleaning, visualization, ML pipelines
- **Business Agents** (17 recipes) - Requirements, use cases, market research, financial analysis

### Workflows (`workflows/`)

Pre-built workflow templates for complex multi-step processes.

**Automation Workflows** (`workflows/automation/`)
- `document_processing_pipeline` - Multi-stage document analysis
- `content_scheduling` - Plan and schedule content production
- `automated_reporting` - Generate reports from data
- `batch_processing` - Process multiple items in parallel
- `email_automation` - Classify and route emails

**Decision-Making Workflows** (`workflows/decision_making/`)
- `swot_analysis` - Structured SWOT analysis
- `pros_cons` - Weighted pros/cons analysis
- `multi_criteria_decision` - MCDA with weighted scoring
- `risk_assessment` - Identify and mitigate risks
- `root_cause_analysis` - 5 Whys and fishbone diagrams

**Data Processing Workflows** (`workflows/data_processing/`)
- `data_pipeline` - End-to-end ETL pipeline
- `data_validation` - Validate against business rules
- `data_enrichment` - Join and enrich data
- `data_aggregation` - Group and aggregate
- `data_merging` - Merge multiple datasets

### Integrations (`integrations/`)

Integration recipes for connecting Victor agents with external services.

**API Integrations** (`integrations/apis/`)
- Slack, Discord, Telegram bots
- REST API wrappers (FastAPI)
- GraphQL APIs (Strawberry)
- Webhook handlers
- Email, SMS services
- Stripe payments
- AWS S3 storage

**Database Integrations** (`integrations/databases/`)
- PostgreSQL query execution
- SQLite schema analysis
- MongoDB query generation
- Redis caching strategies
- ETL pipeline design
- Data validation rules
- Backup strategies
- Full-text search setup

**Messaging Integrations** (`integrations/messaging/`)
- Slack webhooks and workflows
- Discord interactions
- Microsoft Teams notifications
- Telegram bots
- WhatsApp Business API
- Email response systems
- Twilio SMS
- Push notifications (FCM/APNS)

**Cloud Integrations** (`integrations/cloud/`)
- AWS Lambda, Azure Functions, GCP Cloud Functions deployment
- AWS Bedrock, Azure OpenAI, GCP Vertex AI provider adapters
- AWS SageMaker model deployment
- AWS SQS, Event Grid, Pub/Sub message processing
- DynamoDB, Cosmos DB, Firestore state storage

## Example Recipes

### Code Review Agent

```python
from victor_cookbook.recipes.agents.specialized.coding_agents import code_review_agent

result = await code_review_agent(
    code="""
def calculate_sum(numbers):
    total = 0
    for n in numbers:
        total += n
    return total
    """,
    language="Python"
)
print(result)
```

### Data Processing Pipeline

```python
from victor_cookbook.recipes.workflows.data_processing.data_workflows import (
    data_pipeline_workflow
)

result = await data_pipeline_workflow(
    source_path="data/raw/sales.csv",
    destination_path="data/processed/sales_clean.csv",
    transformations=[
        "Normalize text to lowercase",
        "Convert date strings to datetime",
        "Fill missing values with median"
    ]
)
```

### Slack Bot Integration

```python
from victor_cookbook.recipes.integrations.apis.api_integrations import (
    slack_bot_integration
)

integration_code = await slack_bot_integration()
# Returns complete Slack bot code with:
# - Slash command handling
# - Event handlers
# - Victor agent integration
# - Error handling
```

## Recipe Structure

Each recipe file contains:

1. **Metadata** - Category, difficulty, time estimate
2. **Recipe Functions** - Async functions that implement the pattern
3. **Demo Function** - Example usage of the recipe

```python
"""Recipe description."""

RECIPE_CATEGORY = "category/subcategory"
RECIPE_DIFFICULTY = "beginner|intermediate|advanced"
RECIPE_TIME = "5 minutes"

async def recipe_function(param1, param2):
    """Recipe function with docstring."""
    from victor import Agent

    agent = Agent.create()
    result = await agent.run("...")
    return result.content

async def demo_recipe():
    """Demonstrate recipe usage."""
    result = await recipe_function("example")
    print(result)
```

## Development

### Adding New Recipes

1. Create a new file in the appropriate category directory
2. Follow the recipe structure above
3. Add `RECIPE_*` metadata
4. Implement recipe function(s) with clear docstrings
5. Add a demo function
6. Run tests: `pytest tests/`

### Recipe Conventions

- **Async-first**: All recipes use async/await
- **Type hints**: Include parameter and return types
- **Docstrings**: Describe purpose, parameters, return values
- **Error handling**: Include appropriate error handling
- **Examples**: Provide usage examples in docstrings
- **Production-ready**: Consider security, performance, scalability

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

Areas where we'd love contributions:
- More specialized agent recipes
- Additional workflow patterns
- More cloud provider integrations
- Industry-specific recipes (healthcare, finance, legal)
- Performance optimization recipes
- Security and compliance recipes

## Documentation

Full documentation is available at [https://docs.victor.ai/cookbook](https://docs.victor.ai/cookbook)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [Victor AI Framework](https://github.com/victor-ai/victor) - Core AI agent framework
- [victor-coding](https://github.com/victor-ai/victor-coding) - Coding assistant vertical
- [victor-devops](https://github.com/victor-ai/victor-devops) - DevOps assistant vertical
- [victor-research](https://github.com/victor-ai/victor-research) - Research assistant vertical
- [victor-dataanalysis](https://github.com/victor-ai/victor-dataanalysis) - Data analysis vertical
- [victor-rag](https://github.com/victor-ai/victor-rag) - RAG assistant vertical

## Citation

If you use Victor Cookbook in your research or project:

```bibtex
@software{victor_cookbook,
  title = {Victor Cookbook: Production-Ready Recipes for AI Agents},
  author = {Victor AI Contributors},
  year = {2025},
  url = {https://github.com/victor-ai/victor-cookbook}
}
```
