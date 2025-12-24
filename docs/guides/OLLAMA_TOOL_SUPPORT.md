# Local Provider Tool Support Guide

This guide explains how tool calling support varies across different local LLM providers (Ollama, LMStudio, vLLM), and how Victor handles each provider differently.

## Overview: Model Capability vs Provider Support

**Critical insight**: A model's tool calling capability (from training) is different from whether a serving provider enables it. The same model can have:
- **Native tool support on LMStudio** (llama.cpp enables it for all models)
- **NO tool support on Ollama** (requires `{{ if .Tools }}` in Modelfile)

| Provider | Tool Support Mechanism | All Models Supported? |
|----------|----------------------|----------------------|
| **Ollama** | Modelfile template (`{{ if .Tools }}`) | No - template dependent |
| **LMStudio** | llama.cpp built-in (0.3.6+) | Yes - all models |
| **vLLM** | Guided decoding (Outlines) | Yes - all models |

## Ollama-Specific Tool Support

Ollama models may or may not support native tool calling. The actual support depends on whether the model's **Modelfile template** includes tool handling blocks.

**Key insight**: The authoritative source of truth for Ollama tool support is the presence of `{{ if .Tools }}` blocks in the model's template.

## Quick Reference: Models with Verified Tool Support

Based on template inspection of Ollama models, here are the verified results:

### Models WITH Native Tool Support

These models have `{{ if .Tools }}` blocks in their templates and can use Ollama's native tool calling:

| Model Family | Tool Format | Notes |
|--------------|-------------|-------|
| `llama3.1*` | JSON | All variants (8b, 70b, etc.) |
| `llama3.2*` | JSON | All variants |
| `llama3.3*` | JSON | All variants |
| `qwen2.5*` | XML | qwen2.5-coder variants work well |
| `qwen3*` (non-coder) | XML | qwen3:30b, qwen3:32b confirmed |
| `command-r*` | JSON | Cohere Command-R models |
| `hermes*` | JSON | Hermes function calling variants |
| `functionary*` | JSON | Designed for function calling |
| `firefunction*` | JSON | Fireworks function calling |
| `gpt-oss*` | JSON | GPT-OSS variants |

### Models WITHOUT Native Tool Support

These models lack `{{ if .Tools }}` in their templates. Victor will use fallback parsing (extracting JSON/XML tool calls from text content):

| Model Family | Workaround |
|--------------|------------|
| `qwen3-coder*` | Fallback parsing (despite base qwen3 having support) |
| `qwen2.5-coder:14b*`, `qwen2.5-coder:32b*` | Fallback parsing (see note below) |
| `mistral*` | Fallback parsing |
| `mixtral*` | Fallback parsing |
| `ministral*` | Fallback parsing |
| `devstral*` | Fallback parsing |
| `deepseek-coder*` | Fallback parsing |
| `deepseek-coder-v2*` | Fallback parsing |
| `deepseek-r1*` | Fallback parsing |
| `gemma3*` | Fallback parsing (despite model capability) |
| `codellama*` | Fallback parsing |

> **Note: qwen2.5-coder Size Inconsistency**
>
> The `qwen2.5-coder` model family has **inconsistent tool support** across different sizes in Ollama:
> - `qwen2.5-coder:1.5b`, `qwen2.5-coder:7b` → HAS `{{ if .Tools }}` in template (native support)
> - `qwen2.5-coder:14b`, `qwen2.5-coder:32b` → NO tool template (requires fallback parsing)
>
> Victor handles this automatically via granular pattern matching in `model_capabilities.yaml`:
> - Generic `qwen2.5*` pattern provides native support for smaller sizes
> - Specific `qwen2.5-coder:14b*` and `qwen2.5-coder:32b*` patterns override with fallback parsing
>
> **Workaround for larger sizes**: Create a `-tools` variant using `scripts/create_ollama_tool_models.py`:
> ```bash
> python scripts/create_ollama_tool_models.py --models qwen2.5-coder:14b
> victor chat --model qwen2.5-coder-tools:14b
> ```
| `phi*` | Fallback parsing |
| `yi*` | Fallback parsing |

## Diagnosing Tool Support

### Method 1: Use the Detection Script (Recommended)

Victor includes a script that authoritatively checks all your installed Ollama models:

```bash
# Check all models on localhost
python scripts/check_ollama_tool_support.py

# Check a specific Ollama server
python scripts/check_ollama_tool_support.py --host http://192.168.1.20:11434

# Check a specific model
python scripts/check_ollama_tool_support.py --model qwen2.5-coder:14b

# Output as JSON for programmatic use
python scripts/check_ollama_tool_support.py --json

# Update model_capabilities.yaml with detected patterns
python scripts/check_ollama_tool_support.py --update-config
```

The script will output a detailed report showing which models support tools and which don't.

### Method 2: Manual Template Inspection

You can manually check a model's template using the Ollama CLI:

```bash
ollama show qwen2.5-coder:14b
```

Look for the **TEMPLATE** section in the output. If you see patterns like:

```
{{ if .Tools }}
...
{{ end }}
```

or

```
{{- if .Tools }}
...
{{- end }}
```

Then the model supports native tool calling.

**Example of a model WITH tool support** (qwen2.5-coder:14b):
```
{{ if .Tools }}
# Tools

You may call one or more functions to assist with the user query.
...
{{ end }}
```

**Example of a model WITHOUT tool support** (qwen3-coder:30b):
The template will lack any `{{ if .Tools }}` block entirely.

### Method 3: Programmatic Detection

Use the `OllamaCapabilityDetector` in your code:

```python
from victor.providers.ollama_capability_detector import (
    OllamaCapabilityDetector,
    get_model_tool_support,
)

# Check single model
support = get_model_tool_support("http://localhost:11434", "qwen2.5-coder:14b")
print(f"Supports tools: {support.supports_tools}")
print(f"Tool format: {support.tool_response_format}")

# Use detector with caching for multiple checks
detector = OllamaCapabilityDetector("http://localhost:11434")
if detector.supports_tools("llama3.1:8b"):
    print("Model has native tool calling!")
else:
    print("Model requires fallback parsing")
```

## How Victor Handles Models Without Tool Support

When a model lacks native tool support, Victor automatically:

1. **Enables fallback parsing**: Extracts tool calls from the model's text content by looking for JSON or XML patterns
2. **Uses stricter prompting**: Adds explicit instructions about tool usage format in the system prompt
3. **Applies conservative tool budgets**: Uses lower `recommended_tool_budget` values

This is configured in `victor/config/model_capabilities.yaml`:

```yaml
# Model without native tool support
"qwen3-coder*":
  native_tool_calls: false
  json_fallback_parsing: true
  xml_fallback_parsing: true
  thinking_mode: true
  requires_strict_prompting: true
  recommended_tool_budget: 10
```

## Enabling Tool Support for Unsupported Models

### Method 1: Automated Script (Recommended)

Victor includes a script that automatically creates tool-enabled model variants for all models that need them:

```bash
# Check which models need tool-enabled variants (dry run)
python scripts/create_ollama_tool_models.py --check-only

# Create tool-enabled variants on localhost
python scripts/create_ollama_tool_models.py

# Create on a specific Ollama server
python scripts/create_ollama_tool_models.py --host http://192.168.1.20:11434

# Create variants for specific models only
python scripts/create_ollama_tool_models.py --models qwen3-coder:30b,mistral:7b

# Preview what would be created without actually creating
python scripts/create_ollama_tool_models.py --dry-run
```

The script:
- Analyzes installed Ollama models for tool capability
- Identifies models that have tool training but lack Ollama template support
- Creates `-tools` variants (e.g., `qwen3-coder-tools:30b`) with proper tool templates
- Shares underlying model weights - only the Modelfile template changes

**Supported model families** for automatic variant creation:
- Mistral family: `mistral`, `mixtral`, `ministral`, `devstral`
- DeepSeek family: `deepseek-coder`, `deepseek-coder-v2`, `deepseek-r1`
- Qwen3-coder: `qwen3-coder`
- Google Gemma: `gemma3`
- Microsoft Phi: `phi`
- Yi models: `yi`

After creating variants, use them like any other model:
```bash
victor chat --model qwen3-coder-tools:30b
victor chat --model mistral-tools:7b
```

### Method 2: Manual Modelfile Creation

If you want to manually create a tool-enabled variant:

1. Create a new Modelfile:

```dockerfile
FROM qwen3-coder:30b

TEMPLATE """
{{- if .System }}
<|im_start|>system
{{ .System }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

Available functions:
{{ range .Tools }}
{{ . }}
{{ end }}
{{- end }}
<|im_end|>
{{- end }}
{{- range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}
<|im_end|>
{{ end }}
<|im_start|>assistant
"""
```

2. Create the new model:

```bash
ollama create qwen3-coder-tools -f Modelfile
```

3. Use the new model which will have tool support.

See the [Ollama Tool Support Blog](https://ollama.com/blog/tool-support) for more details.

## Configuration Reference

### model_capabilities.yaml

The `victor/config/model_capabilities.yaml` file defines tool calling capabilities with **provider-specific overrides**:

```yaml
# Global defaults (apply to all providers)
defaults:
  native_tool_calls: false
  json_fallback_parsing: true
  recommended_tool_budget: 12

# Provider-specific configuration
providers:
  ollama:
    defaults:
      native_tool_calls: false  # Conservative default for Ollama
    models:
      "llama3.1*":
        native_tool_calls: true  # Template has {{ if .Tools }}
      "qwen3-coder-tools*":
        native_tool_calls: true  # Custom variant with tool support
      "mistral*":
        native_tool_calls: false  # No template support

  lmstudio:
    defaults:
      native_tool_calls: true  # llama.cpp enables tools for ALL models
      streaming_tool_calls: true
      parallel_tool_calls: true

  vllm:
    defaults:
      native_tool_calls: true  # Guided decoding via Outlines
      streaming_tool_calls: true
```

**Key insight**: The same model can have different capabilities on different providers:

| Model | Ollama | LMStudio | vLLM |
|-------|--------|----------|------|
| `mistral:7b` | `native_tool_calls: false` | `native_tool_calls: true` | `native_tool_calls: true` |
| `qwen3-coder:30b` | `native_tool_calls: false` | `native_tool_calls: true` | `native_tool_calls: true` |
| `qwen3-coder-tools:30b` | `native_tool_calls: true` | `native_tool_calls: true` | `native_tool_calls: true` |

### Capability Fields

```yaml
"model-pattern*":
  native_tool_calls: bool      # Model returns structured tool_calls
  streaming_tool_calls: bool   # Tool calls can be streamed
  parallel_tool_calls: bool    # Model can request multiple tools at once
  tool_choice_param: bool      # Supports tool_choice parameter
  json_fallback_parsing: bool  # Can parse JSON tool calls from content
  xml_fallback_parsing: bool   # Can parse XML tool calls from content
  thinking_mode: bool          # Supports /think /no_think (Qwen3)
  requires_strict_prompting: bool  # Needs strict system prompts
  recommended_max_tools: int   # Max tools to send
  recommended_tool_budget: int # Max tool calls per turn
```

### Capability Resolution Order

The `ModelCapabilityLoader` resolves capabilities in this order (later overrides earlier):

1. **Global defaults** (`defaults:` section)
2. **Legacy top-level model patterns** (`models:` section - for backwards compatibility)
3. **Provider defaults** (`providers.{provider}.defaults`)
4. **Provider-specific model patterns** (`providers.{provider}.models.{pattern}`)

More specific patterns take precedence over less specific ones.

### Runtime Override

You can also set tool support at runtime via the adapter:

```python
from victor.agent.tool_calling.adapters import OllamaToolCallingAdapter

adapter = OllamaToolCallingAdapter(model="custom-model")
# Force native tool calls
adapter._capabilities.native_tool_calls = True
```

## Troubleshooting

### "HTTP 400 Bad Request" when using tools

This typically means:
1. The model doesn't support native tool calling
2. Victor is trying to pass tools via the API when it shouldn't

**Solution**: Check `model_capabilities.yaml` and ensure the model pattern is configured correctly with `native_tool_calls: false`.

### Tools not being called reliably

For models using fallback parsing:
1. Ensure `requires_strict_prompting: true` is set
2. Consider lowering `recommended_tool_budget`
3. Check if the model is outputting valid JSON/XML tool call format

### Verifying actual behavior

Enable debug logging to see how Victor is handling tool calls:

```bash
victor chat --renderer text --log-level DEBUG
```

Look for log messages about:
- "Using native tool calling"
- "Using fallback parsing"
- "Parsed tool call from content"

## Related Files

- `victor/config/model_capabilities.yaml` - Model capability configuration
- `victor/providers/ollama_capability_detector.py` - Runtime detection module
- `victor/agent/tool_calling/adapters.py` - Tool calling adapter implementations
- `victor/agent/tool_calling/capabilities.py` - YAML config loader with provider-specific resolution
- `scripts/check_ollama_tool_support.py` - CLI detection script
- `scripts/create_ollama_tool_models.py` - Automated tool-enabled model variant creator
