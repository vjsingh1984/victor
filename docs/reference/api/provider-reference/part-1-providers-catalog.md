# Victor AI 0.5.0 Provider Reference - Part 1

**Part 1 of 2:** Overview, Local Providers, Major Cloud Providers, AI Research Companies

---

## Navigation

- **[Part 1: Provider Catalog](#)** (Current)
- [Part 2: Free-Tier, Enterprise, Switching](part-2-free-tier-enterprise-switching.md)
- [**Complete Reference](../PROVIDER_REFERENCE.md)**

---

> **Note**: This legacy API documentation is retained for reference. For current docs, see `docs/reference/api/`.

Complete reference for all 21 supported LLM providers.

**Table of Contents**
- [Overview](#overview)
- [Local Providers](#local-providers)
  - [Ollama](#ollama)
  - [LMStudio](#lmstudio)
  - [vLLM](#vllm)
  - [Llama.cpp](#llamacpp)
- [Major Cloud Providers](#major-cloud-providers)
  - [Anthropic](#anthropic)
  - [OpenAI](#openai)
  - [Google](#google)
  - [Azure OpenAI](#azure-openai)
  - [AWS Bedrock](#aws-bedrock)
  - [Google Vertex AI](#google-vertex-ai)
- [AI Research Companies](#ai-research-companies)
  - [xAI](#xai)
  - [DeepSeek](#deepseek)
  - [Moonshot](#moonshot)
  - [Zhipu AI (ZAI)](#zhipu-ai-zai)
- [Free-Tier Providers (2025)](#free-tier-providers-2025) *(in Part 2)*
- [Enterprise/Other](#enterpriseother) *(in Part 2)*
- [Provider Switching](#provider-switching) *(in Part 2)*
- [Model Capabilities](#model-capabilities) *(in Part 2)*

---

## Overview

Victor AI supports 21 LLM providers through a unified `BaseProvider` interface. All providers implement:

- `chat()`: Non-streaming chat completion
- `stream_chat()`: Streaming chat completion
- `supports_tools()`: Tool calling capability query
- `name`: Provider identifier

[Content continues through AI Research Companies...]


**Reading Time:** 1 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 2: Free-Tier, Enterprise, Switching](part-2-free-tier-enterprise-switching.md)**
