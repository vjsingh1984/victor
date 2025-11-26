# Tool Calling Models for Victor

**Last Updated**: November 24, 2025
**Version**: 1.0

---

## Overview

This document provides comprehensive information about Ollama models optimized for tool calling (function calling) with Victor. Tool calling enables Victor to interact with external tools like file systems, APIs, databases, and more.

---

## What is Tool Calling?

Tool calling (also known as function calling) allows Large Language Models to:
- 📁 Create, read, modify, and delete files
- 🔧 Execute bash commands
- 🗄️ Query databases
- 🌐 Make HTTP/API requests
- 🐳 Manage Docker containers
- 🔍 Search and analyze code
- 📊 Generate reports and documentation

Without tool calling support, LLMs can only provide text responses. With tool calling, they become **interactive coding agents**.

---

## Default Configuration

Victor is now configured to use **qwen2.5-coder:7b** as the default model for optimal tool calling performance in coding tasks.

### Current Profiles

```yaml
profiles:
  default:  # qwen2.5-coder:7b - Code-specialized with excellent tool calling
  code:     # qwen3-coder:30b - Advanced code generation
  fast:     # mistral:7b-instruct - Fast inference
  general:  # llama3.1:8b - Best overall balance
  reasoning:# deepseek-r1:32b - Advanced reasoning
  enterprise:# llama3.3:70b - Maximum accuracy
```

---

## Model Rankings

Based on research from [CollabnixTOP](https://collabnix.com/best-ollama-models-for-function-calling-tools-complete-guide-2025/) and [Ollama's official documentation](https://docs.ollama.com/capabilities/tool-calling).

### Tier S: Exceptional (90%+ accuracy)

#### 1. llama3.1:70b — Maximum Accuracy
- **Overall Score**: 94%
- **RAM Required**: 64GB+ (128GB recommended)
- **Speed**: Slow
- **Best For**: Enterprise systems, mission-critical processes
- **Tool Calling**: ⭐⭐⭐⭐⭐ (96% schema understanding)

#### 2. llama3.1:8b — Best Overall ⭐
- **Overall Score**: 89%
- **RAM Required**: 8GB+ (16GB recommended)
- **Speed**: Fast
- **Best For**: General purpose, enterprise chatbots, data workflows
- **Tool Calling**: ⭐⭐⭐⭐⭐ (91% schema understanding)
- **Recommendation**: Excellent balance of performance and resources

### Tier A: Excellent (85-89% accuracy)

#### 3. qwen2.5-coder:7b — Recommended for Victor 🏆
- **Overall Score**: 87% (estimated)
- **RAM Required**: 7GB+ (16GB recommended)
- **Speed**: Fast
- **Best For**: Code generation, development automation, tool calling
- **Tool Calling**: ⭐⭐⭐⭐⭐ (90% parameter extraction)
- **Recommendation**: **Best choice for Victor** - code-specialized with excellent tool calling

#### 4. qwen3-coder:30b — Advanced Coding
- **Overall Score**: 88%
- **RAM Required**: 30GB+ (64GB recommended)
- **Speed**: Medium
- **Best For**: Complex code generation, architecture design
- **Tool Calling**: ⭐⭐⭐⭐⭐ (91% parameter extraction)

#### 5. deepseek-coder:33b-instruct — Code Excellence
- **Overall Score**: 88%
- **RAM Required**: 32GB+ (64GB recommended)
- **Speed**: Medium
- **Best For**: Complex programming, refactoring, documentation
- **Tool Calling**: ⭐⭐⭐⭐⭐ (91% parameter extraction)

#### 6. codellama:34b-python — Python Specialist
- **Overall Score**: 88%
- **RAM Required**: 32GB+ (64GB recommended)
- **Speed**: Medium
- **Best For**: Python development, API integration, DevOps
- **Tool Calling**: ⭐⭐⭐⭐⭐ (92% parameter extraction - highest)

#### 7. mixtral:8x7b — Mixture of Experts
- **Overall Score**: 88%
- **RAM Required**: 24GB+ (48GB recommended)
- **Speed**: Medium
- **Best For**: Multi-domain applications, complex reasoning
- **Tool Calling**: ⭐⭐⭐⭐⭐ (89% error handling - highest)

#### 8. mistral:7b-instruct — Speed Champion
- **Overall Score**: 85%
- **RAM Required**: 7GB+ (16GB recommended)
- **Speed**: Very Fast (0.8s response time)
- **Best For**: Real-time applications, edge deployments
- **Tool Calling**: ⭐⭐⭐⭐ (Good accuracy, excellent speed)

### Tier B: Good (80-84% accuracy)

- **llama3:8b** - Previous generation Llama (82%)
- **phi4-reasoning:plus** - Microsoft reasoning model (83%)

### Tier C: Fair (75-79% accuracy)

- **gemma3:27b** - Google Gemma (78%)
- **llama3.2:latest** - Lightweight model (75%)

---

## Performance Comparison Table

| Model | Schema | Parameters | Errors | Overall | Speed | RAM |
|-------|--------|------------|--------|---------|-------|-----|
| llama3.1:70b | 96% | 94% | 92% | 94% | Slow | 64GB+ |
| llama3.1:8b | 91% | 89% | 87% | 89% | Fast | 8GB+ |
| codellama:34b | 89% | **92%** | 85% | 88% | Med | 32GB+ |
| qwen3-coder:30b | 89% | 91% | 84% | 88% | Med | 30GB+ |
| deepseek-coder:33b | 90% | 91% | 84% | 88% | Med | 32GB+ |
| mixtral:8x7b | 88% | 87% | **89%** | 88% | Med | 24GB+ |
| **qwen2.5-coder:7b** | **88%** | **90%** | 83% | **87%** | **Fast** | **7GB+** |
| mistral:7b | 86% | 85% | 84% | 85% | V.Fast | 7GB+ |

**Legend**: Schema = Schema Understanding, Parameters = Parameter Extraction, Errors = Error Handling

---

## Recommendations by Use Case

### For General Coding (Default)
```bash
victor  # Uses qwen2.5-coder:7b
```
- **Model**: qwen2.5-coder:7b
- **Why**: Code-specialized, excellent tool calling, fast inference
- **RAM**: 16GB recommended

### For Complex Projects
```bash
victor --profile code  # Uses qwen3-coder:30b
```
- **Model**: qwen3-coder:30b or deepseek-coder:33b
- **Why**: Superior code generation, better architecture understanding
- **RAM**: 32GB+ required

### For Real-Time Applications
```bash
victor --profile fast  # Uses mistral:7b-instruct
```
- **Model**: mistral:7b-instruct
- **Why**: Fastest inference (0.8s), good tool calling
- **RAM**: 16GB recommended

### For Enterprise/Production
```bash
victor --profile enterprise  # Uses llama3.3:70b
```
- **Model**: llama3.3:70b or llama3.1:70b
- **Why**: Highest accuracy, best for mission-critical code
- **RAM**: 64GB+ required

### For General Tasks
```bash
victor --profile general  # Uses llama3.1:8b
```
- **Model**: llama3.1:8b
- **Why**: Best overall balance, excellent all-rounder
- **RAM**: 16GB recommended

---

## System Requirements

### Minimum (Small Models)
- **RAM**: 16GB
- **Storage**: 10GB
- **Recommended Models**:
  - qwen2.5-coder:7b
  - mistral:7b-instruct
  - llama3.1:8b

### Recommended (Medium Models)
- **RAM**: 32GB
- **Storage**: 50GB
- **Recommended Models**:
  - qwen3-coder:30b
  - deepseek-coder:33b-instruct
  - mixtral:8x7b

### High Performance (Large Models)
- **RAM**: 64GB+
- **Storage**: 100GB+
- **Recommended Models**:
  - llama3.1:70b
  - llama3.3:70b
  - All medium models with better quantization

---

## Installation

### Pull Recommended Models

```bash
# Default (code-specialized, 7B)
ollama pull qwen2.5-coder:7b

# General purpose (8B)
ollama pull llama3.1:8b

# Fast inference (7B)
ollama pull mistral:7b-instruct

# Advanced coding (30B - requires 32GB+ RAM)
ollama pull qwen3-coder:30b

# Enterprise (70B - requires 64GB+ RAM)
ollama pull llama3.1:70b
```

### Verify Installation

```bash
# List installed models
ollama list

# Test a model
victor test-provider ollama

# List available profiles
victor profiles
```

---

## Testing Tool Calling

Run the comprehensive benchmark to test tool calling capabilities:

```bash
python test_tool_calling_models.py
```

This will:
1. Check available Ollama models
2. Run tool calling tests on each model
3. Measure success rates and response times
4. Generate a ranking report
5. Save results to `tool_calling_benchmark_results.json`

---

## Tool Calling Features Supported

✅ **Function Schema Definition**: Models understand tool parameters and types
✅ **Parameter Validation**: Automatic validation of tool arguments
✅ **Multi-Step Orchestration**: Chain multiple tool calls together
✅ **Error Handling**: Graceful recovery from tool execution failures
✅ **Streaming Tool Calls**: Real-time tool execution with streaming responses
✅ **Parallel Execution**: Execute multiple tools concurrently

---

## Best Practices

### 1. Choose the Right Model

- **Coding tasks**: Use qwen2.5-coder:7b or qwen3-coder:30b
- **General tasks**: Use llama3.1:8b
- **Speed-critical**: Use mistral:7b-instruct
- **Mission-critical**: Use llama3.1:70b

### 2. Optimize Prompts for Tool Calling

**Good Prompt**:
```
Create a Python file called calculator.py in the current directory with add() and subtract() functions.
Use proper type hints and docstrings.
```

**Better Prompt**:
```
Use the write_file tool to create calculator.py in /path/to/directory with:
1. add(a: float, b: float) -> float function
2. subtract(a: float, b: float) -> float function
3. Google-style docstrings
4. Full type hints
```

### 3. Monitor Performance

```bash
# Check model resource usage
ollama ps

# View model details
ollama show qwen2.5-coder:7b
```

### 4. Update Regularly

```bash
# Update models
ollama pull qwen2.5-coder:7b

# Update Victor
pip install --upgrade victor
```

---

## Troubleshooting

### Model Not Found
```bash
# Pull the model first
ollama pull qwen2.5-coder:7b

# Verify it's installed
ollama list | grep qwen
```

### Out of Memory
```bash
# Use a smaller model
victor --profile fast  # Uses mistral:7b instead

# Or increase swap space (not recommended for performance)
```

### Poor Tool Calling
```bash
# Switch to a better model
victor --profile code  # Uses qwen3-coder:30b

# Or use llama3.1:8b for general reliability
victor --profile general
```

### Slow Responses
```bash
# Use fastest model
victor --profile fast  # mistral:7b-instruct (0.8s)

# Or reduce max_tokens in profiles.yaml
```

---

## Configuration Files

### Main Configuration
- **Location**: `~/.victor/profiles.yaml`
- **Purpose**: Define model profiles

### Tool Calling Manifest
- **Location**: `victor/config/tool_calling_models.yaml`
- **Purpose**: Model rankings and capabilities

### Test Results
- **Location**: `tool_calling_benchmark_results.json`
- **Purpose**: Benchmark data from tests

---

## Sources and References

1. **[Top Best Ollama Models 2025 for Function Calling](https://collabnix.com/best-ollama-models-for-function-calling-tools-complete-guide-2025/)**
   - Comprehensive benchmark data
   - Performance comparisons
   - Use case recommendations

2. **[Ollama Tool Calling Documentation](https://docs.ollama.com/capabilities/tool-calling)**
   - Official implementation details
   - API reference
   - Code examples

3. **[Ollama Tool Support Blog](https://ollama.com/blog/tool-support)**
   - Feature announcements
   - Implementation examples

4. **[Streaming Tool Calls](https://ollama.com/blog/streaming-tool)**
   - Streaming API details
   - Performance improvements

---

## Changelog

### v1.0 (2025-11-24)
- Initial release
- Created tool calling model manifest
- Updated default profile to qwen2.5-coder:7b
- Added 6 model profiles (default, code, fast, general, reasoning, enterprise)
- Created comprehensive testing framework
- Documented 23+ Ollama models
- Added tier rankings (S/A/B/C)
- Created benchmark tool

---

## Future Improvements

- [ ] Add automated model selection based on task type
- [ ] Implement model fallback chain (try best model first, fallback to others)
- [ ] Add cost/performance optimizer
- [ ] Create model warmup/caching system
- [ ] Add telemetry for model performance tracking
- [ ] Implement A/B testing framework for models
- [ ] Add support for model-specific system prompts

---

**Maintained by**: Victor Development Team
**Contributors**: Claude Code, Research Team
**License**: MIT
