# Victor Error Reference

Complete guide to understanding and resolving Victor errors.

## Quick Reference

| Error Code | Error Type | Common Cause | Quick Fix |
|------------|------------|--------------|-----------|
| `PROV-001` | ProviderNotFoundError | Provider not registered | Check provider name with `victor providers list` |
| `PROV-002` | ProviderInitializationError | Invalid API credentials | Set API key in environment variable |
| `PROV-003` | ProviderConnectionError | Network/connection failure | Check network connection and provider URL |
| `PROV-004` | ProviderAuthError | Authentication failed | Verify API key is correct and valid |
| `PROV-005` | ProviderRateLimitError | Rate limit exceeded | Wait before retrying or use different provider |
| `PROV-006` | ProviderTimeoutError | Request timeout | Increase timeout or check provider status |
| `PROV-007` | ProviderInvalidResponseError | Invalid response format | Try again or use different provider |
| `TOOL-001` | ToolNotFoundError | Tool not registered | Check tool name with `victor tools list` |
| `TOOL-002` | ToolExecutionError | Tool failed | Check tool arguments and permissions |
| `TOOL-003` | ToolValidationError | Invalid arguments | Verify required parameters |
| `TOOL-004` | ToolTimeoutError | Tool execution timeout | Increase timeout or simplify operation |
| `CFG-001` | ConfigurationError | Invalid configuration | Check YAML syntax and required fields |
| `CFG-002` | ValidationError | Input validation failed | Check input values and types |
| `SRCH-001` | SearchError | Search backend failed | Check backend configuration and connectivity |
| `WRK-001` | WorkflowExecutionError | Workflow execution failed | Check workflow logs and fix node |
| `FILE-001` | FileNotFoundError | File not found | Check file path exists |
| `FILE-002` | FileError | File operation failed | Check file permissions |
| `NET-001` | NetworkError | Network failure | Check network connection |
| `EXT-001` | ExtensionLoadError | Extension failed to load | Fix extension or mark as optional |

---

## Error Categories

### Provider Errors (PROV-XXX)

Provider errors occur when communicating with LLM providers.

#### PROV-001: ProviderNotFoundError

**Message**: `"Provider not found: {provider_name}. Available: {providers}"`

**Cause**: The specified provider is not registered in Victor.

**Severity**: ERROR

**Solutions**:
1. List available providers:
   ```bash
   victor providers list
   ```

2. Check for typos in provider name

3. Install provider dependencies if needed:
   ```bash
   pip install victor-ai[{provider}]
   ```

**Example**:
```bash
# Wrong
victor chat --provider anthproc

# Correct
victor chat --provider anthropic
```

**Recovery Hint**: "Check provider name spelling. Use 'victor providers list' to list available providers."

---

#### PROV-002: ProviderInitializationError

**Message**: `"Provider '{provider}' failed to initialize: {reason}"`

**Cause**: Invalid configuration or missing credentials.

**Severity**: ERROR

**Solutions**:
1. Check API key is set:
   ```bash
   echo $ANTHROPIC_API_KEY  # For Anthropic
   echo $OPENAI_API_KEY     # For OpenAI
   ```

2. Set missing API key:
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```

3. Verify API key format and validity

4. Check provider-specific requirements

**Recovery Hint**: "Set {PROVIDER}_API_KEY environment variable or check your configuration."

---

#### PROV-003: ProviderConnectionError

**Message**: `"Failed to connect to provider '{provider}': {reason}"`

**Cause**: Network issues or provider service unavailable.

**Severity**: ERROR

**Solutions**:
1. Check internet connection:
   ```bash
   ping api.anthropic.com
   ```

2. Verify provider URL is correct

3. Check if provider service is operational

4. Try alternative endpoint

**Recovery Hint**: "Check network connection and provider URL. Verify the provider service is running."

---

#### PROV-004: ProviderAuthError

**Message**: `"Authentication failed for provider '{provider}': {reason}"`

**Cause**: Invalid or expired API credentials.

**Severity**: ERROR

**Solutions**:
1. Verify API key is correct:
   ```bash
   echo $PROVIDER_API_KEY
   ```

2. Check API key hasn't expired

3. Regenerate API key if needed

4. Ensure key has required permissions

**Recovery Hint**: "Check your API key or credentials. Ensure they are correctly set in environment variables or configuration."

---

#### PROV-005: ProviderRateLimitError

**Message**: `"Rate limit exceeded for provider '{provider}'. Retry after {seconds} seconds"`

**Cause**: Too many requests in short time period.

**Severity**: WARNING

**Solutions**:
1. Wait specified time before retrying

2. Implement exponential backoff

3. Use a different provider:
   ```bash
   victor chat --provider openai
   ```

4. Reduce request frequency

**Recovery Hint**: "Wait {retry_after} seconds before retrying. Consider using a different model or provider."

---

#### PROV-006: ProviderTimeoutError

**Message**: `"Request to provider '{provider}' timed out after {seconds} seconds"`

**Cause**: Request took too long to complete.

**Severity**: ERROR

**Solutions**:
1. Increase timeout:
   ```bash
   victor chat --timeout 120
   ```

2. Check provider status page

3. Try with simpler prompt

4. Use alternative provider

**Recovery Hint**: "Request timed out after {timeout} seconds. Try increasing timeout or check provider status."

---

#### PROV-007: ProviderInvalidResponseError

**Message**: `"Provider '{provider}' returned invalid response: {reason}"`

**Cause**: Provider returned unexpected response format.

**Severity**: ERROR

**Solutions**:
1. Check provider API version compatibility

2. Try again (temporary issue)

3. Use different provider

4. Report bug if persistent

**Recovery Hint**: "The provider returned an unexpected response format. Try again or use a different provider."

---

### Tool Errors (TOOL-XXX)

Tool errors occur during tool execution.

#### TOOL-001: ToolNotFoundError

**Message**: `"Tool not found: {tool_name}"`

**Cause**: Tool is not registered in Victor.

**Severity**: ERROR

**Solutions**:
1. List available tools:
   ```bash
   victor tools list
   ```

2. Check tool name spelling

3. Install tool dependencies if needed

4. Check if tool is available in current mode

**Example**:
```python
# Wrong
agent.use_tool("file_read")

# Correct
agent.use_tool("read")
```

**Recovery Hint**: "Check tool name spelling. Use list_tools() to see available tools."

---

#### TOOL-002: ToolExecutionError

**Message**: `"Tool '{tool_name}' execution failed: {reason}"`

**Cause**: Tool execution failed (network, permissions, invalid args).

**Severity**: ERROR

**Solutions**:
1. Check tool arguments match expected schema:
   ```python
   help(tool)  # Show tool signature
   ```

2. Verify file permissions (for file operations)

3. Check network connectivity (for web tools)

4. Review correlation ID in logs

**Example**:
```python
# Wrong - missing required argument
tool.execute(path="/tmp/file")

# Correct - all required arguments
tool.execute(path="/tmp/file", mode="read")
```

**Recovery Hint**: "Check tool arguments or try with different parameters."

---

#### TOOL-003: ToolValidationError

**Message**: `"Tool '{tool_name}' validation failed: {reason}"`

**Cause**: Tool arguments don't match expected schema.

**Severity**: ERROR

**Solutions**:
1. Check required parameters

2. Verify parameter types

3. Check parameter constraints (min/max, enum values)

4. Review tool schema

**Example**:
```python
# Wrong - invalid type
tool.execute(max_tokens="100")  # Should be int

# Correct
tool.execute(max_tokens=100)
```

**Recovery Hint**: "Check the required arguments for this tool."

---

#### TOOL-004: ToolTimeoutError

**Message**: `"Tool '{tool_name}' timed out after {seconds} seconds"`

**Cause**: Tool execution took too long.

**Severity**: ERROR

**Solutions**:
1. Increase timeout setting

2. Simplify the operation

3. Break into smaller tasks

4. Check if resource is available

**Recovery Hint**: "Try with a longer timeout or simplify the operation."

---

### Configuration Errors (CFG-XXX)

Configuration errors occur when Victor is misconfigured.

#### CFG-001: ConfigurationError

**Message**: `"Configuration error: {message}"`

**Cause**: Invalid or incompatible configuration.

**Severity**: ERROR

**Solutions**:
1. Validate workflow YAML:
   ```bash
   victor workflow validate path/to/workflow.yaml
   ```

2. Check YAML syntax

3. Verify required fields

4. Check for circular dependencies

**Example**:
```yaml
# Wrong - missing required field
modes:
  build:
    exploration: standard
    # Missing: edit_permission

# Correct
modes:
  build:
    name: build
    exploration: standard
    edit_permission: full
```

**Recovery Hint**: "Fix validation errors in configuration file."

---

#### CFG-002: ValidationError

**Message**: `"Validation error for field '{field}': {message}"`

**Cause**: Input doesn't meet validation requirements.

**Severity**: ERROR

**Solutions**:
1. Check field requirements

2. Verify data type matches

3. Check value constraints

4. Review validation schema

**Example**:
```python
# Wrong - invalid value
config.set_mode("invalid_mode")

# Correct - valid mode
config.set_mode("build")
```

**Recovery Hint**: "Check input values and types."

---

### Search Errors (SRCH-XXX)

Search errors occur when search backends fail.

#### SRCH-001: SearchError

**Message**: `"All {count} search backends failed for '{search_type}': {details}"`

**Cause**: Search backends unavailable or misconfigured.

**Severity**: ERROR

**Solutions**:
1. Check backend configuration:
   ```bash
   victor config search.type=keyword
   ```

2. Verify API keys for search services

3. Check network connectivity

4. Try alternative search type

**Recovery Hint**: "Check backend configuration and connectivity. Try alternative search type."

---

### Workflow Errors (WRK-XXX)

Workflow errors occur during workflow execution.

#### WRK-001: WorkflowExecutionError

**Message**: `"Workflow execution failed at node '{node_id}': {reason}"`

**Cause**: Workflow node failed during execution.

**Severity**: ERROR

**Solutions**:
1. Check workflow logs for correlation ID

2. Fix failed node

3. Resume from checkpoint:
   ```bash
   victor workflow resume --checkpoint-id {checkpoint_id}
   ```

4. Validate workflow definition

**Recovery Hint**: "Fix node '{node_id}' and retry workflow execution. Use checkpoint to resume."

---

### File Errors (FILE-XXX)

File errors occur during file operations.

#### FILE-001: FileNotFoundError

**Message**: `"File not found: {path}"`

**Cause**: File doesn't exist or path is incorrect.

**Severity**: ERROR

**Solutions**:
1. Check file path is correct

2. Verify file exists:
   ```bash
   ls -la /path/to/file
   ```

3. Check working directory

4. Use absolute path

**Recovery Hint**: "Check the file path. The file may have been moved or deleted."

---

#### FILE-002: FileError

**Message**: `"File operation failed: {reason}"`

**Cause**: File permission or I/O error.

**Severity**: ERROR

**Solutions**:
1. Check file permissions:
   ```bash
   ls -l /path/to/file
   ```

2. Verify disk space available

3. Check file is not locked

4. Ensure directory exists

**Recovery Hint**: "Check file permissions and disk space."

---

### Network Errors (NET-XXX)

Network errors occur during network operations.

#### NET-001: NetworkError

**Message**: `"Network error: {reason}"`

**Cause**: Network connectivity or DNS resolution failure.

**Severity**: ERROR

**Solutions**:
1. Check internet connection:
   ```bash
   ping google.com
   ```

2. Verify DNS resolution

3. Check proxy settings

4. Try alternative endpoint

**Recovery Hint**: "Check your network connection. The service may be temporarily unavailable."

---

### Extension Errors (EXT-XXX)

Extension errors occur when loading vertical extensions.

#### EXT-001: ExtensionLoadError

**Message**: `"Failed to load '{extension_type}' extension for vertical '{vertical}': {reason}"`

**Cause**: Extension failed to initialize or missing dependencies.

**Severity**: WARNING or CRITICAL (depending if required)

**Solutions**:
1. Install extension dependencies

2. Check extension configuration

3. Verify extension is compatible

4. Mark extension as optional if not critical

**Recovery Hint**: "Fix the underlying error or mark the extension as optional."

---

## Debugging with Correlation IDs

Every error includes a correlation ID (8-character hex) for tracking:

```
[abc12345] Provider 'anthropic' failed to initialize
Recovery hint: Set ANTHROPIC_API_KEY environment variable
```

### Using Correlation IDs

1. **Find full error in logs**:
   ```bash
   grep "abc12345" ~/.victor/logs/victor.log
   ```

2. **Track error across services**:
   Use the same correlation ID in distributed systems

3. **Report bugs**:
   Include correlation ID when reporting issues:
   ```bash
   victor --version  # Get version
   victor logs show --correlation-id abc12345  # Get full context
   ```

---

## Common Error Patterns

### "ImportError" or "ModuleNotFoundError"

**Cause**: Missing Python dependencies.

**Solution**:
```bash
pip install -e ".[dev]"
# Or for specific provider
pip install victor-ai[anthropic]
```

---

### "PermissionError"

**Cause**: Insufficient file or directory permissions.

**Solution**:
```bash
# Make script executable
chmod +x script.sh

# Check directory permissions
ls -la /path/to/dir

# Run with appropriate permissions
sudo victor chat  # Use with caution
```

---

### "Connection timeout" / "Network unreachable"

**Cause**: Network connectivity issues.

**Solution**:
1. Check internet connection
2. Verify proxy settings:
   ```bash
   echo $HTTP_PROXY
   echo $HTTPS_PROXY
   ```
3. Check firewall rules
4. Try alternative endpoint

---

### "KeyError: 'API_KEY'"

**Cause**: Missing environment variable.

**Solution**:
```bash
# Set API key
export ANTHROPIC_API_KEY="your-key"

# Or add to .env file
echo "ANTHROPIC_API_KEY=your-key" >> .env

# Verify
echo $ANTHROPIC_API_KEY
```

---

### "ValidationError: Invalid mode"

**Cause**: Invalid agent mode specified.

**Solution**:
```bash
# List available modes
victor config list modes

# Use valid mode
victor chat --mode build  # Valid
victor chat --mode invalid  # Invalid
```

---

### "Workflow validation failed"

**Cause**: Invalid YAML workflow configuration.

**Solution**:
```bash
# Validate workflow
victor workflow validate path/to/workflow.yaml

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('workflow.yaml'))"

# Fix validation errors
# 1. Check indentation
# 2. Verify required fields
# 3. Check for circular dependencies
```

---

## Error Categories by Severity

### CRITICAL Errors
These prevent Victor from running:
- `ProviderInitializationError` (PROV-002)
- `ExtensionLoadError` (EXT-001) when required

**Action**: Fix immediately before continuing.

### ERROR Errors
These prevent the current operation but Victor can continue:
- All provider errors (except CRITICAL)
- Most tool errors
- Configuration errors
- Workflow execution errors

**Action**: Fix to proceed with operation.

### WARNING Errors
These indicate issues but don't prevent operation:
- `ProviderRateLimitError` (PROV-005)
- `ExtensionLoadError` (EXT-001) when optional

**Action**: Monitor and fix when convenient.

---

## Getting Help

If you can't resolve an error:

### 1. Check Documentation
- This error reference
- Troubleshooting guide: `docs/troubleshooting.md`
- API documentation

### 2. Search Issues
```bash
# Search GitHub issues
gh issue list --search "error message"

# Or search web
https://github.com/your-org/victor/issues
```

### 3. Create Bug Report

Include:
- Error message (full)
- Correlation ID
- Steps to reproduce
- Victor version:
  ```bash
  victor --version
  ```
- Environment:
  ```bash
  victor doctor  # System diagnostics
  ```
- Logs:
  ```bash
  victor logs show --correlation-id abc12345 > error.log
  ```

### 4. Community Support
- Discord/Slack community
- Stack Overflow (tag: `victor-ai`)

---

## Best Practices

### 1. Always Check Correlation IDs
Correlation IDs help track errors across services and in logs.

### 2. Read Recovery Hints
Every Victor error includes a recovery hint - read it first!

### 3. Validate Configuration
Before running:
```bash
victor workflow validate workflow.yaml
victor config check
```

### 4. Use Appropriate Timeouts
```bash
victor chat --timeout 120  # Increase for long operations
```

### 5. Monitor Error Rates
```bash
victor errors stats  # Check error patterns
```

### 6. Keep Dependencies Updated
```bash
pip install --upgrade victor-ai
```

---

## Error Prevention

### Provider Errors
- Keep API keys secure and valid
- Monitor rate limits
- Use alternative providers when needed
- Implement retry logic with backoff

### Tool Errors
- Validate tool arguments before execution
- Check file permissions beforehand
- Use appropriate timeouts
- Handle edge cases

### Configuration Errors
- Validate YAML files
- Use schema validation
- Test configuration changes
- Keep backups of working configs

### Workflow Errors
- Test workflows locally
- Use checkpoints for long workflows
- Monitor workflow execution
- Handle failures gracefully

---

## Appendix A: Error Code Cross-Reference

| Error Class | Error Code | Category |
|-------------|------------|----------|
| `ProviderNotFoundError` | PROV-001 | Provider |
| `ProviderInitializationError` | PROV-002 | Provider |
| `ProviderConnectionError` | PROV-003 | Provider |
| `ProviderAuthError` | PROV-004 | Provider |
| `ProviderRateLimitError` | PROV-005 | Provider |
| `ProviderTimeoutError` | PROV-006 | Provider |
| `ProviderInvalidResponseError` | PROV-007 | Provider |
| `ToolNotFoundError` | TOOL-001 | Tool |
| `ToolExecutionError` | TOOL-002 | Tool |
| `ToolValidationError` | TOOL-003 | Tool |
| `ToolTimeoutError` | TOOL-004 | Tool |
| `ConfigurationError` | CFG-001 | Configuration |
| `ValidationError` | CFG-002 | Configuration |
| `SearchError` | SRCH-001 | Search |
| `WorkflowExecutionError` | WRK-001 | Workflow |
| `FileNotFoundError` | FILE-001 | File |
| `FileError` | FILE-002 | File |
| `NetworkError` | NET-001 | Network |
| `ExtensionLoadError` | EXT-001 | Extension |

---

## Appendix B: CLI Commands for Error Handling

```bash
# List all errors
victor errors list

# Show error statistics
victor errors stats

# Export error metrics
victor errors export metrics.json

# Show logs for correlation ID
victor logs show --correlation-id abc12345

# Validate workflow
victor workflow validate workflow.yaml

# List available providers
victor providers list

# List available tools
victor tools list

# Check configuration
victor config check

# System diagnostics
victor doctor
```

---

**Last Updated**: 2025-01-14
**Version**: 0.5.1
