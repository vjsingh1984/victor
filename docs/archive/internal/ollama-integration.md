# Ollama Integration for Demo Scripts

## Summary

All demo scripts have been updated to dynamically detect and use the simplest available ollama model instead of hardcoded paid Anthropic models.

## Key Changes

### 1. New Ollama Helper Module

**File**: `victor/observability/ollama_helper.py`

**Features**:
- `get_ollama_models()` - Fetches available models from localhost:11434 API
- `select_simplest_model()` - Selects the smallest/lightest model
- `get_default_ollama_model()` - Returns detected model or fallback
- `is_ollama_available()` - Checks if ollama is running

**Model Selection Priority**:
1. Models with "2b" or "1b" (smallest parameter count)
2. Models with "tiny" or "mini"
3. Phi models (usually small and fast)
4. Gemma models (efficient)
5. Qwen models (often have small versions)
6. First available model

**Embedding Model Filtering**:
Automatically filters out embedding models (e.g., nomic-embed-text, mxbai-embed) to ensure only text generation models are selected.

**Fallback Model**: `gpt-oss`

### 2. Updated Scripts

#### `scripts/demo_observability.py`
- Detects ollama models on startup
- Passes detected model to all demo functions
- Shows which model is being used
- Falls back to "gpt-oss" if ollama unavailable

#### `scripts/test_eventbus_integration.py`
- Detects and uses simplest ollama model
- Reports which model is being used in test output

#### `scripts/verify_dashboard.py`
- Dynamically detects model before emitting events
- Provides clear feedback on model selection

## Usage

### With Ollama Installed

```bash
# Install ollama (macOS)
brew install ollama

# Start ollama service
ollama serve

# Pull some models
ollama pull phi3
ollama pull llama3.2
ollama pull gemma2:2b

# Run demo - will automatically detect and use simplest model
python scripts/demo_observability.py
```

Output:
```
Detecting ollama models...
✓ Using ollama model: gemma2
```

### Without Ollama

```bash
# Run demo - will use fallback
python scripts/demo_observability.py
```

Output:
```
Detecting ollama models...
⚠️  Ollama not available, using fallback: gpt-oss
   Install ollama: brew install ollama && ollama pull gpt-oss
```

## Benefits

### 1. Cost Savings
- ✅ No API costs (ollama is free)
- ✅ No rate limits
- ✅ No API key management

### 2. Privacy
- ✅ All processing is local
- ✅ No data leaves your machine
- ✅ No network dependency (after model download)

### 3. Performance
- ✅ Fast inference on local hardware
- ✅ Automatically selects smallest/lightest model
- ✅ No network latency

### 4. Flexibility
- ✅ Works with whatever models you have installed
- ✅ Gracefully falls back if ollama unavailable
- ✅ Easy to test with different models

## Testing the Integration

### Test 1: Verify Model Detection

```bash
python -m victor.observability.ollama_helper
```

Expected output (if ollama running):
```
Testing Ollama model detection...

Found 3 available models:
  - phi3:latest
  - llama3.2:latest
  - gemma2:2b

Simplest model: gemma2
```

### Test 2: Run Demo with Dashboard

```bash
# Terminal 1: Start dashboard
victor dashboard

# Terminal 2: Run demo
python scripts/demo_observability.py
```

The dashboard will show events using the detected model.

### Test 3: EventBus Integration

```bash
python scripts/test_eventbus_integration.py
```

Will show all 16 events using the detected model.

## Model Recommendations

### For Testing (Fastest)
```bash
ollama pull phi3          # 3.8B parameters, very fast
ollama pull gemma2:2b     # 2B parameters, fastest
ollama pull qwen2:1.5b    # 1.5B parameters, extremely fast
```

### For Production (Better Quality)
```bash
ollama pull llama3.2      # Meta's Llama 3.2
ollama pull mistral       # Mistral 7B
ollama pull phi3          # Microsoft Phi 3
```

### Avoid for Testing (Too Large)
```bash
# These work but are slower:
ollama pull llama3.1:70b  # 70B parameters, very slow
ollama pull mixtral       # 47B parameters, slow
```

## Troubleshooting

### Ollama Not Detected

**Problem**: `⚠️ Ollama not available, using fallback: gpt-oss`

**Solution 1**: Check ollama is running
```bash
ps aux | grep ollama
# Or
curl http://localhost:11434/api/tags
```

**Solution 2**: Start ollama service
```bash
ollama serve
```

**Solution 3**: Install ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Then start it
ollama serve
```

### Wrong Model Selected

**Problem**: Selected model is too large or slow

**Solution**: Pull smaller models
```bash
ollama pull gemma2:2b
ollama pull phi3
```

The helper will automatically prioritize smaller models.

### Only Embedding Models Available

**Problem**: Only embedding models (like nomic-embed-text) are installed

**Solution**: Install text generation models
```bash
ollama pull phi3
ollama pull llama3.2
```

The helper automatically filters out embedding models.

## API Reference

### `get_ollama_models() -> List[str]`

Returns list of available LLM models (filters out embeddings).

```python
from victor.observability.ollama_helper import get_ollama_models

models = get_ollama_models()
print(models)  # ['phi3:latest', 'llama3.2:latest', 'gemma2:2b']
```

### `select_simplest_model(models: List[str]) -> Optional[str]`

Selects the smallest/lightest model from the list.

```python
from victor.observability.ollama_helper import select_simplest_model

models = ['llama3.2', 'phi3', 'gemma2:2b']
simplest = select_simplest_model(models)
print(simplest)  # 'gemma2'
```

### `get_default_ollama_model() -> str`

Returns the best model to use (detected or fallback).

```python
from victor.observability.ollama_helper import get_default_ollama_model

model = get_default_ollama_model()
print(f"Using: {model}")  # 'phi3' or 'gpt-oss' (fallback)
```

### `is_ollama_available() -> bool`

Checks if ollama API is accessible.

```python
from victor.observability.ollama_helper import is_ollama_available

if is_ollama_available():
    print("Ollama is ready!")
else:
    print("Using fallback")
```

## Files Modified

### New Files
- `victor/observability/ollama_helper.py` - Model detection and selection

### Updated Files
- `scripts/demo_observability.py` - Dynamic model detection
- `scripts/test_eventbus_integration.py` - Dynamic model detection
- `scripts/verify_dashboard.py` - Dynamic model detection

### Documentation
- `docs/OLLAMA_INTEGRATION.md` - This document
- `docs/EVENTBUS_TEST_RESULTS.md` - EventBus test results

## Summary

All demo scripts now:
- ✅ Dynamically detect available ollama models
- ✅ Select the simplest/smallest model automatically
- ✅ Filter out embedding models
- ✅ Gracefully fall back to "gpt-oss" if ollama unavailable
- ✅ Provide clear feedback on model selection
- ✅ Work with any ollama model you have installed
- ✅ No API costs, no rate limits, full privacy

This makes testing and development much easier and cost-effective!
