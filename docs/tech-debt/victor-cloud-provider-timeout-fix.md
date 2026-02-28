# Technical Debt: Victor Framework - Cloud Provider Integration Issues

## Issue Summary

Cloud provider scoring jobs (DeepSeek, OpenAI, Anthropic, etc.) are failing with `worker_timeout_exceeded` errors when called from the interviewer project. Local providers (Ollama) work correctly.

**New Finding:** Users are being prompted for keychain access during job execution, suggesting Victor IS available and attempting to access the keychain, but may require non-interactive approval.

## Current State

### Environment
- **Backend:** interviewer project
- **Victor Module:** Available (keychain access is being attempted)
- **Keychain Access:** Requires user approval (interactive prompt)
- **DeepSeek API Key:** ✓ Present in macOS keychain (service: `victor`, account: `deepseek_api_key`)
- **Environment Variables:** ✗ Provider API keys not set

### Problem Analysis

**Root Cause:** The backend process requests keychain access interactively, which blocks:
1. Job execution waits for keychain approval
2. If user is not monitoring the terminal, approval never happens
3. Job eventually times out (195s for DeepSeek profile)

**Expected Flow:**
```
Job → Victor Provider → Keychain Access Request → User Approval → API Key Retrieved → LLM Call
```

**Actual Flow:**
```
Job → Victor Provider → Keychain Access Request → [WAITING] → Timeout
```

## Solution Requirements

### Phase 1: Non-Interactive Keychain Access (HIGH Priority)

**FR-1.1:** Enable non-interactive keychain access
- Research macOS keychain ACLs for non-interactive access
- Consider using `security unlock` or proper keychain ACLs
- Alternative: Add environment variable fallback
- Priority: **HIGH**
- Effort: MEDIUM

**FR-1.2:** Add keychain pre-flight check
- Before submitting job, verify keychain access is available
- Warn user if keychain access requires interaction
- Offer to set environment variable as alternative
- Priority: **HIGH**
- Effort: LOW

### Phase 2: Debug Logging (HIGH Priority)

**FR-2.1:** Add provider initialization logging
- Log provider creation with config (excluding secrets)
- Log API key resolution: which source was used (env/keychain/file)
- Log whether keychain access succeeded or failed
- Priority: **HIGH**
- Effort: MEDIUM

**FR-2.2:** Add LLM API call tracing
- Log API call initiation with provider/model
- Log timing for each phase: key resolution → API call → response
- Log authentication success/failure
- Log API errors with details (without exposing secrets)
- Priority: **HIGH**
- Effort: MEDIUM

**FR-2.3:** Add job lifecycle logging
- Log job state transitions with timestamps
- Log timeout warnings with stack traces
- Log errors with actionable messages
- Priority: **MEDIUM**
- Effort: LOW

### Phase 3: Error Handling Improvements (MEDIUM Priority)

**FR-3.1:** Surface provider errors to user
- Catch and parse provider-specific errors (auth, rate limit, network)
- Return actionable error messages via API
- Examples:
  - "DeepSeek API authentication failed: API key not found. Set DEEPSEEK_API_KEY environment variable or ensure keychain access is permitted."
  - "DeepSeek API rate limit exceeded. Retry after 60s."
  - "Network timeout connecting to DeepSeek API."
- Priority: **MEDIUM**
- Effort: MEDIUM

**FR-3.2:** Add provider health check endpoint
- `/victor/providers/health` endpoint
- Test provider connectivity without full job
- Returns: provider status, API key status, test call result
- Priority: **MEDIUM**
- Effort: MEDIUM

### Phase 4: Keychain Integration Improvements (LOW Priority)

**FR-4.1:** Keychain access control documentation
- Document macOS keychain ACL setup for non-interactive access
- Provide alternative: environment variable setup script
- Priority: **LOW**
- Effort: LOW

**FR-4.2:** Consider alternative keychain backends
- Research cross-platform keychain solutions
- Consider `keyring` Python package as universal solution
- Priority: **LOW**
- Effort: MEDIUM

## Implementation Plan

### Immediate (this week)
1. **Add debug logging** (FR-2)
   - Log provider initialization
   - Log API key resolution attempts
   - Log LLM call timing
   - Log errors with details

2. **Surface provider errors** (FR-3.1)
   - Catch provider exceptions
   - Parse error types (auth, network, timeout)
   - Return actionable messages

### Short-term (next sprint)
1. **Keychain pre-flight** (FR-1.2)
   - Check keychain access availability
   - Warn user if interaction required
   - Offer env var setup

2. **Provider health check** (FR-3.2)
   - Test provider connectivity
   - Validate API key resolution
   - Return health status

### Medium-term
1. **Non-interactive keychain** (FR-1.1)
   - Research macOS ACLs
   - Implement proper access control

2. **Environment variable fallback**
   - Support env vars as primary
   - Keychain as secondary
   - Config file as tertiary

## Acceptance Criteria

### Phase 1 & 2 (Immediate)
- [ ] Debug logs show which API key source was attempted
- [ ] Logs show whether keychain access succeeded/failed
- [ ] Provider errors are surfaced (not just timeouts)
- [ ] Error messages are actionable

### Phase 3 (Short-term)
- [ ] Provider health check works
- [ ] Users are warned about keychain requirements
- [ ] Environment variable setup instructions provided

### Phase 4 (Medium-term)
- [ ] Non-interactive keychain access configured
- [ ] Cross-platform keychain solution evaluated

## Research Questions

1. Why does keychain access require interactive approval?
2. What are the macOS keychain ACL requirements for non-interactive access?
3. How does codingagent/victor-invest handle this in production?
4. Should we use environment variables as default instead of keychain?

## Related Files

- Victor API keys: `victor/config/api_keys.py`
- Victor provider factory: `victor/providers/factory.py`
- DeepSeek provider: `victor/providers/deepseek_provider.py`
- Interviewer integration: `interviewer/backend/scoring/local_llm_scorer.py`

## Testing Strategy

1. **Unit tests:** Mock keychain access, test provider initialization
2. **Integration tests:** Test full scoring job with mocked APIs
3. **Manual tests:** Test with real DeepSeek API key in keychain
4. **Load tests:** Test with multiple concurrent jobs
