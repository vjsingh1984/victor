# API Key Management Guide

Victor supports secure credential management through OS-level keyring integration. All 17 cloud providers can be used without explicitly passing API keys.

## Quick Start

### Setup (One-Time)

```bash
# Store API key in system keyring (recommended)
victor keys --set anthropic --keyring
# Enter your API key when prompted

# List configured providers
victor keys --list

# Verify a specific provider
victor keys --check anthropic
```

### Usage

**CLI** (automatic key retrieval):
```bash
victor chat --provider anthropic --model claude-sonnet-4-20250514
```

**Python API** (automatic key retrieval):
```python
from victor.providers.anthropic_provider import AnthropicProvider
from victor.config.settings import Settings
from victor.agent.orchestrator import AgentOrchestrator

# No api_key parameter needed - loads from keyring automatically
provider = AnthropicProvider()
settings = Settings()
orchestrator = AgentOrchestrator(
    settings=settings,
    provider=provider,
    model="claude-sonnet-4-20250514"
)
```

## Supported Providers

All 17 cloud providers support keyring:

| Provider | Environment Variable | Keyring Name |
|----------|---------------------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic` |
| OpenAI | `OPENAI_API_KEY` | `openai` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek` |
| Google | `GOOGLE_API_KEY` | `google` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `azure_openai`, `azure` |
| Vertex AI | `VERTEX_API_KEY` | `vertex`, `gcp` |
| xAI/Grok | `XAI_API_KEY` | `xai`, `grok` |
| ZhipuAI (ZAI) | `ZAI_API_KEY` | `zai`, `zhipuai`, `zhipu` |
| Cerebras | `CEREBRAS_API_KEY` | `cerebras` |
| Fireworks | `FIREWORKS_API_KEY` | `fireworks` |
| GroqCloud | `GROQCLOUD_API_KEY` | `groqcloud` |
| HuggingFace | `HF_TOKEN` | `huggingface` |
| Mistral | `MISTRAL_API_KEY` | `mistral` |
| Moonshot | `MOONSHOT_API_KEY` | `moonshot` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter` |
| Replicate | `REPLICATE_API_TOKEN` | `replicate` |
| Together | `TOGETHER_API_KEY` | `together` |

**Local Providers** (no API key needed): Ollama, LMStudio, vLLM, llama.cpp

**AWS Credential-based**: Bedrock (uses AWS credential chain)

## Credential Resolution Order

For each provider, credentials are loaded in this order:

1. **Parameter** (highest priority) - `provider = AnthropicProvider(api_key="sk-...")`
2. **Environment Variable** - `export ANTHROPIC_API_KEY="sk-..."`
3. **System Keyring** - `victor keys --set anthropic --keyring`
4. **Config File** - `~/.victor/api_keys.yaml` (fallback)
5. **None** - Warning logged (for local providers or missing keys)

## Advanced Usage

### Multiple Providers

```bash
# Configure multiple providers
victor keys --set anthropic --keyring
victor keys --set openai --keyring
victor keys --set deepseek --keyring

# Use any provider - credentials load automatically
victor chat --provider anthropic --model claude-haiku
victor chat --provider openai --model gpt-4o-mini
victor chat --provider deepseek --model deepseek-chat
```

### Environment Variables (CI/CD)

For automation and CI/CD, environment variables take priority over keyring:

```bash
# Set environment variable (overrides keyring)
export ANTHROPIC_API_KEY="sk-ant-..."
victor chat --provider anthropic
```

### Config File

Create `~/.victor/api_keys.yaml`:
```yaml
anthropic: sk-ant-...
openai: sk-openai-...
deepseek: sk-deepseek-...
```

Set permissions:
```bash
chmod 600 ~/.victor/api_keys.yaml
```

## Troubleshooting

### Check Configured Providers

```bash
victor keys --list
```

### Verify Specific Provider

```bash
victor keys --check anthropic
```

### Remove Provider Key

```bash
victor keys --remove anthropic
```

### Keyring Not Working?

1. **Check keyring installation**:
   ```bash
   python3 -c "import keyring; print(keyring.__version__)"
   ```

2. **Verify provider name**:
   ```bash
   victor keys --list  # Check if provider is listed
   ```

3. **Use environment variable fallback**:
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   victor chat --provider anthropic
   ```

## Security

- **Encrypted Storage**: Keyring uses OS-level encrypted storage (Keychain, Credential Manager, Secret Service)
- **Provider Isolation**: Each provider's credentials stored separately
- **Audit Logging**: All key access attempts logged
- **Memory Safety**: Credentials cleared from memory after use
- **No Exposure**: Keys never appear in process listings or logs

## Migration from Previous Versions

### Old Way (still works)

```python
# Passing api_key explicitly still works
provider = AnthropicProvider(api_key="sk-ant-...")
```

### New Way (recommended)

```python
# Let Victor handle credentials automatically
provider = AnthropicProvider()  # Loads from keyring
```

**No Breaking Changes**: Both methods work. The new method is more convenient and secure.
