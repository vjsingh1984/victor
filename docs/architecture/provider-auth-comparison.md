# Provider Auth: Victor vs OpenCode — Architecture Comparison

## Z.AI Configuration Reference

### Three Z.AI Endpoints

```
Endpoint                                    Use Case              SDK Compatibility
──────────────────────────────────────────────────────────────────────────────────
https://api.z.ai/api/paas/v4               Standard API          OpenAI-compatible
https://api.z.ai/api/coding/paas/v4        Coding Plan           OpenAI-compatible
https://api.z.ai/api/anthropic/v1          Anthropic adapter     Anthropic-compatible
https://open.bigmodel.cn/api/paas/v4       China mainland        OpenAI-compatible
```

### Z.AI in Claude Code

```json
// ~/.claude/settings.json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "<zai-api-key>",
    "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.5-air"
  }
}
```

### Z.AI in OpenCode

```json
// opencode.json — OpenAI-compatible (standard)
{
  "provider": {
    "zai": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "https://api.z.ai/api/paas/v4" },
      "models": { "glm-4.7": { "name": "GLM-4.7" } }
    }
  }
}

// opencode.json — Anthropic-compatible (alternative)
{
  "provider": {
    "zai-anthropic": {
      "npm": "@ai-sdk/anthropic",
      "options": { "baseURL": "https://api.z.ai/api/anthropic/v1" },
      "models": {
        "glm-4.7": { "name": "GLM-4.7" },
        "glm-4.5-air": { "name": "GLM-4.5-Air" }
      }
    }
  }
}

// opencode.json — Coding Plan
{
  "provider": {
    "zai-coding-plan": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "https://api.z.ai/api/coding/paas/v4" }
    }
  }
}
```

## Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OPENCODE (TypeScript)                           │
│                                                                        │
│  models.dev/api.json ──► Provider Registry ──► SDK Factory             │
│       (remote)              (runtime)           (@ai-sdk/*)            │
│                                                                        │
│  Auth Types:                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐                         │
│  │  "api"   │  │ "oauth"  │  │ "wellknown"  │                         │
│  │  { key } │  │ { access │  │ { key,       │                         │
│  │          │  │   refresh │  │   token }    │                         │
│  │          │  │   expires}│  │              │                         │
│  └──────────┘  └──────────┘  └──────────────┘                         │
│       │              │              │                                   │
│       └──────────────┴──────────────┘                                  │
│                      │                                                  │
│               auth.json (0600)                                         │
│               ~/.local/share/opencode/auth.json                        │
│                                                                        │
│  Provider init:                                                        │
│    baseURL = config.options.baseURL ?? model.api.url                   │
│    apiKey  = Auth.get(providerID) ?? env[ZHIPU_API_KEY]               │
│    sdk     = BUNDLED_PROVIDERS[npm](options)                           │
│                                                                        │
│  Plugin system: auth hooks can provide OAuth or API key                │
│  for any provider (OpenAI Codex OAuth is a plugin)                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        VICTOR (Python)                                  │
│                                                                        │
│  BaseProvider ──► ProviderConfigStrategy ──► UnifiedApiKeyResolver      │
│    (abstract)        (per-provider)            (key lookup)             │
│                                                                        │
│  Auth Types:                                                           │
│  ┌──────────────────┐  ┌──────────────────────┐                        │
│  │  API Key          │  │  OAuth (NEW)          │                       │
│  │  env / keyring /  │  │  SSOAuthenticator     │                       │
│  │  file / explicit  │  │  + OAuthTokenManager  │                       │
│  └──────────────────┘  └──────────────────────┘                        │
│       │                        │                                        │
│       │                 oauth_tokens.yaml (0600)                        │
│       │                 ~/.victor/oauth_tokens.yaml                     │
│       │                        │                                        │
│  api_keys.yaml (0600)         │                                        │
│  ~/.victor/api_keys.yaml       │                                        │
│       │                        │                                        │
│       └────────────┬───────────┘                                        │
│                    │                                                    │
│              Provider.__init__(api_key=resolved_key)                    │
│              AsyncOpenAI(api_key=..., base_url=...)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Feature Comparison

```
Feature                    OpenCode                    Victor (current)
─────────────────────────────────────────────────────────────────────────
Provider registry          models.dev (remote JSON)    Hardcoded classes
Auth storage               auth.json (single file)     api_keys.yaml + oauth_tokens.yaml
Auth types                 api | oauth | wellknown     api_key | oauth (FEP-0004)
OAuth flow                 Plugin-based (per-provider) SSOAuthenticator (shared)
Token refresh              Plugin auth.loader()        OAuthTokenManager.get_valid_token()
Base URL override          config → models.dev → env   base_url param / env var
SDK compatibility layer    @ai-sdk/* (many adapters)   openai + anthropic Python SDKs
Z.AI support               3 provider IDs              ZAIProvider + 4 endpoints + config strategies
Anthropic compat endpoint  baseURL override             endpoint="anthropic" param
Coding Plan support        Separate provider ID        coding_plan=True / ZAICodingPlanConfig
Qwen support               Not built-in                QwenProvider + OAuth + QwenConfig
Plugin extensibility       Full plugin system           Entry-point verticals
Credential isolation       Per-provider in auth.json   Per-provider in yaml
```

## What Victor Can Learn from OpenCode

### 1. Dynamic Provider Registry (models.dev pattern)

OpenCode fetches provider metadata from `models.dev/api.json` at runtime:
```
provider_id → { name, npm_package, base_url, env_vars, models[] }
```

Victor equivalent: a remote or bundled `providers.yaml` with:
```yaml
zai:
  name: "Z.AI"
  base_url: "https://api.z.ai/api/paas/v4"
  env_vars: ["ZHIPU_API_KEY"]
  sdk_compat: "openai"           # which SDK adapter to use
  models:
    glm-4.7: { context: 204800, cost_input: 0.60, cost_output: 2.20 }
    glm-4.5-air: { context: 131072, cost_input: 0.20, cost_output: 1.10 }

zai-coding-plan:
  name: "Z.AI Coding Plan"
  base_url: "https://api.z.ai/api/coding/paas/v4"
  env_vars: ["ZHIPU_API_KEY"]
  sdk_compat: "openai"
  models:
    glm-4.7: { context: 204800 }
    glm-5: { context: 204800 }
```

### 2. Multi-Endpoint Same-Provider Pattern

OpenCode supports the SAME provider (Z.AI) via DIFFERENT SDK adapters:
- `zai` → OpenAI-compatible SDK → `/api/paas/v4`
- `zai-anthropic` → Anthropic SDK → `/api/anthropic/v1`
- `zai-coding-plan` → OpenAI-compatible → `/api/coding/paas/v4`

Victor should support **base_url + sdk_compat** per provider instance.

### 3. Unified Auth Store

OpenCode uses ONE file (`auth.json`) with discriminated union:
```json
{ "type": "api", "key": "..." }
{ "type": "oauth", "access": "...", "refresh": "...", "expires": ... }
```

Victor currently splits: `api_keys.yaml` for keys, `oauth_tokens.yaml` for OAuth.
Consider unifying or at least sharing the credential lookup path.

### 4. Plugin-Based Auth Hooks

OpenCode's auth is extensible via plugins:
```typescript
auth: {
  provider: "openai",
  methods: [
    { type: "oauth", authorize() { /* PKCE flow */ } },
    { type: "api",   authorize() { /* key validation */ } }
  ],
  loader(auth) { /* transform stored cred → SDK options */ }
}
```

Victor's equivalent: the `ProviderConfigStrategy` pattern could gain an
`auth_hook` method that returns either an API key or OAuth token.

## Z.AI Provider Implementation Plan for Victor

### Approach: OpenAI-Compatible Provider with Base URL Override

Z.AI's API is **fully OpenAI-compatible**. Victor already has `OpenAIProvider`
which uses `AsyncOpenAI`. The simplest path:

```python
# Option A: Direct instantiation
provider = OpenAIProvider(
    api_key="<zhipu-api-key>",
    base_url="https://api.z.ai/api/paas/v4",
)

# Option B: Coding Plan
provider = OpenAIProvider(
    api_key="<zhipu-api-key>",
    base_url="https://api.z.ai/api/coding/paas/v4",
)

# Option C: Dedicated ZAIProvider (thin wrapper)
class ZAIProvider(OpenAIProvider):
    ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
    ZAI_CODING_URL = "https://api.z.ai/api/coding/paas/v4"

    def __init__(self, api_key=None, coding_plan=False, **kwargs):
        base = self.ZAI_CODING_URL if coding_plan else self.ZAI_BASE_URL
        super().__init__(
            api_key=api_key,
            base_url=base,
            **kwargs,
        )
```

### Environment Variables

```bash
ZHIPU_API_KEY=<key>                    # Standard
VICTOR_ZAI_BASE_URL=<url>             # Override
VICTOR_ZAI_CODING_PLAN=true           # Use coding endpoint
```

### Config File

```yaml
# ~/.victor/config.yaml
providers:
  zai:
    api_key_env: ZHIPU_API_KEY
    base_url: "https://api.z.ai/api/paas/v4"
    # OR for coding plan:
    # base_url: "https://api.z.ai/api/coding/paas/v4"
```

## References

- [Z.AI Developer Docs](https://docs.z.ai/api-reference/introduction)
- [Z.AI Claude Code Setup](https://docs.z.ai/scenario-example/develop-tools/claude)
- [Z.AI Coding Plan](https://z.ai/subscribe)
- [OpenCode Providers](https://opencode.ai/docs/providers/)
- [OpenCode Source — provider.ts](https://github.com/sst/opencode/blob/main/packages/opencode/src/provider/provider.ts)
- [OpenCode Source — auth/index.ts](https://github.com/sst/opencode/blob/main/packages/opencode/src/auth/index.ts)
- [models.dev API](https://models.dev/api.json)
