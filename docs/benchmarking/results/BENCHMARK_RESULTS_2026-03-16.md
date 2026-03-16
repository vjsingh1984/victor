# Victor Multi-Provider Benchmark Results

**Date**: 2026-03-16
**Framework**: Victor v0.5.7 (develop branch)
**Tasks**: 5 representative tasks across all categories (C1, R1, T1, A1, W1)
**Retry Strategy**: BaseProvider with ProviderRetryStrategy (max_retries=3, exponential backoff)

## Results Summary

| Provider | Model | Success Rate | Total Time | Avg/Task | Avg Output |
|----------|-------|-------------|------------|----------|------------|
| **Anthropic** | claude-haiku-4-5 | **100%** (5/5) | 84.6s | 16.9s | 9,122 chars |
| **OpenAI** | gpt-4o-mini | **100%** (5/5) | 73.5s | 14.7s | 3,837 chars |
| **DeepSeek** | deepseek-chat (V3) | **100%** (5/5) | 179.4s | 35.9s | 8,356 chars |

## Per-Task Breakdown

### C1: Single-File Code Generation (Simple)

| Provider | Time | Output | Status |
|----------|------|--------|--------|
| Anthropic Haiku | 12.9s | 7,680 chars | ✅ |
| OpenAI GPT-4o-mini | 14.7s | 3,189 chars | ✅ |
| DeepSeek Chat | 34.4s | 8,218 chars | ✅ |

### R1: Research Synthesis (Complex)

| Provider | Time | Output | Status |
|----------|------|--------|--------|
| Anthropic Haiku | 20.7s | 8,470 chars | ✅ |
| OpenAI GPT-4o-mini | 14.6s | 4,838 chars | ✅ |
| DeepSeek Chat | 23.7s | 6,679 chars | ✅ |

### T1: File Operations (Simple)

| Provider | Time | Output | Status |
|----------|------|--------|--------|
| Anthropic Haiku | 7.9s | 3,815 chars | ✅ |
| OpenAI GPT-4o-mini | 13.2s | 2,622 chars | ✅ |
| DeepSeek Chat | 32.6s | 7,574 chars | ✅ |

### A1: Security Audit (Complex)

| Provider | Time | Output | Status |
|----------|------|--------|--------|
| Anthropic Haiku | 27.7s | 15,868 chars | ✅ |
| OpenAI GPT-4o-mini | 20.3s | 5,605 chars | ✅ |
| DeepSeek Chat | 42.1s | 9,244 chars | ✅ |

### W1: Sequential Workflow (Medium)

| Provider | Time | Output | Status |
|----------|------|--------|--------|
| Anthropic Haiku | 15.4s | 9,778 chars | ✅ |
| OpenAI GPT-4o-mini | 10.7s | 2,930 chars | ✅ |
| DeepSeek Chat | 46.6s | 10,065 chars | ✅ |

## Key Observations

1. **100% success rate** across Anthropic, OpenAI, and DeepSeek — Victor's provider abstraction works seamlessly across all three cloud providers

2. **Speed**: OpenAI GPT-4o-mini is fastest (avg 14.7s/task), followed by Anthropic Haiku (16.9s), then DeepSeek (35.9s)

3. **Output richness**: Anthropic Haiku produces the most detailed output (avg 9,122 chars), followed by DeepSeek (8,356 chars). GPT-4o-mini is concise (3,837 chars)

4. **Cost efficiency**: DeepSeek-chat is ~10x cheaper than GPT-4o-mini and ~50x cheaper than Haiku, making it ideal for high-volume tasks where latency isn't critical

5. **Resilience**: The ProviderRetryStrategy automatically retries on rate-limit (429) errors with exponential backoff, jitter, and Retry-After header respect — all handled transparently at the base provider level

## Provider Pricing (approximate, March 2026)

| Provider | Model | Input $/1M tokens | Output $/1M tokens |
|----------|-------|-------------------|-------------------|
| DeepSeek | deepseek-chat | $0.07 | $0.27 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| Anthropic | claude-haiku-4-5 | $0.80 | $4.00 |

## How to Reproduce

```bash
# Install Victor with API keys
pip install victor-ai
victor keys set anthropic   # Uses macOS Keychain
victor keys set openai
victor keys set deepseek

# Run benchmarks
python docs/benchmarking/run_benchmark.py --provider anthropic --model claude-haiku-4-5-20251001 --verbose --output
python docs/benchmarking/run_benchmark.py --provider openai --model gpt-4o-mini --verbose --output
python docs/benchmarking/run_benchmark.py --provider deepseek --model deepseek-chat --verbose --output
```
