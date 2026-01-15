# Victor Troubleshooting Guide

Practical solutions to common issues and problems when using Victor.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Provider Problems](#provider-problems)
3. [Tool Issues](#tool-issues)
4. [Configuration Problems](#configuration-problems)
5. [Workflow Failures](#workflow-failures)
6. [Performance Issues](#performance-issues)
7. [Search and RAG Problems](#search-and-rag-problems)
8. [Network and Connectivity](#network-and-connectivity)
9. [File and Path Issues](#file-and-path-issues)
10. [Memory and Resources](#memory-and-resources)

---

## Installation Issues

### Problem: pip install fails with dependency errors

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement ...
ERROR: No matching distribution found for ...
```

**Solutions**:

1. **Upgrade pip and setuptools**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Install with extras for specific providers**:
   ```bash
   # Core only
   pip install victor-ai

   # With dev dependencies
   pip install -e ".[dev]"

   # With specific provider
   pip install -e ".[anthropic]"
   pip install -e ".[openai]"
   pip install -e ".[all]"  # All providers
   ```

3. **Use Python 3.10+**:
   ```bash
   python --version  # Should be 3.10 or higher
   ```

4. **Check for conflicting packages**:
   ```bash
   pip list | grep -i anthropic
   pip uninstall conflicting-package
   ```

---

### Problem: "Command not found: victor"

**Symptoms**:
```bash
victor --version
# zsh: command not found: victor
```

**Solutions**:

1. **Install with editable mode**:
   ```bash
   cd /path/to/victor
   pip install -e .
   ```

2. **Check PATH**:
   ```bash
   which python
   pip show victor-ai | grep Location
   ```

3. **Add to PATH if needed**:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export PATH="$HOME/.local/bin:$PATH"
   ```

4. **Use python module directly**:
   ```bash
   python -m victor.ui.cli --version
   ```

---

## Provider Problems

### Problem: "Provider not found" error

**Symptoms**:
```
[abc12345] Provider 'xyz' not found. Available: anthropic, openai, ollama
```

**Solutions**:

1. **List available providers**:
   ```bash
   victor providers list
   ```

2. **Check provider name spelling**:
   ```bash
   # Wrong
   victor chat --provider anthproc

   # Correct
   victor chat --provider anthropic
   ```

3. **Install provider dependencies**:
   ```bash
   pip install -e ".[anthropic]"
   ```

4. **Check provider is registered**:
   ```python
   from victor.providers.registry import ProviderRegistry
   registry = ProviderRegistry.get_instance()
   print(registry.list_providers())
   ```

---

### Problem: API key errors

**Symptoms**:
```
[abc12345] Provider 'anthropic' failed to initialize: API key not found
```

**Solutions**:

1. **Set environment variable**:
   ```bash
   # For Anthropic
   export ANTHROPIC_API_KEY="your-key-here"

   # For OpenAI
   export OPENAI_API_KEY="your-key-here"

   # Verify
   echo $ANTHROPIC_API_KEY
   ```

2. **Use .env file**:
   ```bash
   # Create .env in project directory
   echo "ANTHROPIC_API_KEY=your-key-here" > .env

   # Or use victor config
   victor config set anthropic.api_key "your-key-here"
   ```

3. **Check key format**:
   - API keys should not have quotes
   - No extra spaces
   - Correct provider key name

4. **Verify key is valid**:
   ```bash
   # Test with curl
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01"
   ```

---

### Problem: Rate limit errors

**Symptoms**:
```
[abc12345] Rate limit exceeded for provider 'anthropic'. Retry after 60 seconds
```

**Solutions**:

1. **Wait and retry**:
   - Implement exponential backoff
   - Wait specified time before retrying

2. **Use different provider**:
   ```bash
   victor chat --provider openai
   ```

3. **Reduce request frequency**:
   ```python
   # Add delays between requests
   import time
   time.sleep(1)  # Wait 1 second
   ```

4. **Upgrade API tier**:
   - Check provider pricing plans
   - Higher tiers have higher limits

5. **Use local provider**:
   ```bash
   # Use Ollama for local inference
   victor chat --provider ollama --model llama2
   ```

---

### Problem: Timeout errors

**Symptoms**:
```
[abc12345] Request to provider 'openai' timed out after 30 seconds
```

**Solutions**:

1. **Increase timeout**:
   ```bash
   victor chat --timeout 120
   ```

2. **Check provider status**:
   - Visit provider status page
   - Check for service outages

3. **Simplify prompt**:
   - Shorter prompts process faster
   - Break complex tasks into smaller steps

4. **Use alternative provider**:
   ```bash
   victor chat --provider anthropic
   ```

5. **Check network connection**:
   ```bash
   ping api.openai.com
   traceroute api.openai.com
   ```

---

## Tool Issues

### Problem: "Tool not found" error

**Symptoms**:
```
[abc12345] Tool not found: file_reader
```

**Solutions**:

1. **List available tools**:
   ```bash
   victor tools list
   ```

2. **Check tool name**:
   ```python
   # Wrong
   agent.use_tool("file_reader")

   # Correct
   agent.use_tool("read")
   ```

3. **Check tool is available in current mode**:
   ```bash
   # Some tools restricted by mode
   victor chat --mode explore  # Different tools available
   ```

4. **Register custom tool**:
   ```python
   from victor.tools.registry import SharedToolRegistry

   registry = SharedToolRegistry.get_instance()
   registry.register_tool(my_tool)
   ```

---

### Problem: Tool execution failed

**Symptoms**:
```
[abc12345] Tool 'read' execution failed: Permission denied
```

**Solutions**:

1. **Check tool arguments**:
   ```python
   # Show tool signature
   help(read_tool)

   # Check required parameters
   read_tool.execute(path="/tmp/file")
   ```

2. **Verify file permissions**:
   ```bash
   ls -la /path/to/file
   chmod +r file  # Make readable
   ```

3. **Check file exists**:
   ```bash
   ls -la /path/to/file
   pwd  # Check current directory
   ```

4. **Use absolute path**:
   ```python
   # Wrong
   tool.execute(path="file.txt")

   # Correct
   tool.execute(path="/full/path/to/file.txt")
   ```

5. **Check correlation ID in logs**:
   ```bash
   grep "abc12345" ~/.victor/logs/victor.log
   ```

---

### Problem: Tool validation errors

**Symptoms**:
```
[abc12345] Tool 'write' validation failed: Invalid parameter 'mode'
```

**Solutions**:

1. **Check tool schema**:
   ```python
   from victor.tools.base import BaseTool

   # Get tool schema
   schema = tool.get_parameters_schema()
   print(schema)
   ```

2. **Verify parameter types**:
   ```python
   # Wrong - type mismatch
   tool.execute(max_tokens="100")  # String

   # Correct
   tool.execute(max_tokens=100)  # Integer
   ```

3. **Check required parameters**:
   ```python
   # All required parameters
   tool.execute(
       path="/tmp/file",
       content="Hello",
       mode="write"
   )
   ```

4. **Review parameter constraints**:
   - Min/max values
   - Enum values
   - Pattern matching

---

## Configuration Problems

### Problem: Invalid YAML configuration

**Symptoms**:
```
[abc12345] Configuration error: Invalid YAML syntax
```

**Solutions**:

1. **Validate YAML**:
   ```bash
   victor workflow validate workflow.yaml
   ```

2. **Check YAML syntax**:
   ```python
   import yaml
   with open('workflow.yaml') as f:
       yaml.safe_load(f)  # Will raise if invalid
   ```

3. **Use online YAML validator**:
   - https://www.yamllint.com/
   - https://www.onlineyamltools.com/validate-yaml

4. **Check indentation**:
   ```yaml
   # Wrong - inconsistent indentation
   modes:
     build:
       exploration: standard
    edit_permission: full  # Wrong indent

   # Correct
   modes:
     build:
       exploration: standard
       edit_permission: full
   ```

5. **Check for special characters**:
   - Use quotes for strings with colons
   - Escape special characters
   - Use proper boolean values (true/false)

---

### Problem: Missing configuration fields

**Symptoms**:
```
[abc12345] Configuration error: Missing required field 'name'
```

**Solutions**:

1. **Check required fields in schema**:
   ```python
   from victor.config import get_config_schema
   schema = get_config_schema()
   print(schema['required_fields'])
   ```

2. **Compare with example config**:
   ```bash
   victor init --example  # Generate example config
   ```

3. **Use config validation**:
   ```bash
   victor config check
   ```

4. **Add missing fields**:
   ```yaml
   # Add missing required fields
   modes:
     build:
       name: build  # Required
       display_name: Build  # Required
       exploration: standard
       edit_permission: full
   ```

---

### Problem: Invalid mode or agent configuration

**Symptoms**:
```
[abc12345] ValidationError: Invalid mode 'invalid_mode'
```

**Solutions**:

1. **List available modes**:
   ```bash
   victor config list modes
   ```

2. **Check vertical-specific modes**:
   ```python
   from victor.coding import CodingAssistant
   modes = CodingAssistant.list_modes()
   print(modes)
   ```

3. **Use valid mode**:
   ```bash
   # Valid modes: build, plan, explore
   victor chat --mode build
   ```

4. **Check mode configuration**:
   ```yaml
   # victor/config/modes/coding_modes.yaml
   modes:
     build:  # Must be defined here
       name: build
       display_name: Build
   ```

---

## Workflow Failures

### Problem: Workflow validation fails

**Symptoms**:
```
[abc12345] Workflow validation failed: Circular dependency detected
```

**Solutions**:

1. **Validate workflow**:
   ```bash
   victor workflow validate workflow.yaml
   ```

2. **Check for circular dependencies**:
   ```yaml
   # Wrong - circular reference
   nodes:
     - id: node_a
       next: [node_b]
     - id: node_b
       next: [node_a]  # Circular!

   # Correct - linear or DAG
   nodes:
     - id: node_a
       next: [node_b]
     - id: node_b
       next: [node_c]
   ```

3. **Verify node references**:
   ```yaml
   # All 'next' references must exist
   nodes:
     - id: start
       next: [process]  # 'process' must exist

     - id: process
       next: [end]
   ```

4. **Check node types**:
   ```yaml
   # Valid node types: agent, compute, condition, parallel, transform
   nodes:
     - id: my_node
       type: agent  # Must be valid type
   ```

---

### Problem: Workflow execution fails at node

**Symptoms**:
```
[abc12345] Workflow execution failed at node 'data_processor': Division by zero
```

**Solutions**:

1. **Check workflow logs**:
   ```bash
   victor logs show --correlation-id abc12345
   ```

2. **Fix the failing node**:
   ```yaml
   # Add error handling
   nodes:
     - id: data_processor
       type: compute
       handler: safe_divide  # Use safe handler
   ```

3. **Resume from checkpoint**:
   ```bash
   victor workflow resume --checkpoint-id chk_abc123
   ```

4. **Test node independently**:
   ```python
   # Test handler function
   from my_workflow import safe_divide
   result = safe_divide(10, 0)  # Should handle gracefully
   ```

5. **Add validation**:
   ```yaml
   nodes:
     - id: validator
       type: condition
       condition: validate_input
       branches:
         "valid": process
         "invalid": error_handler
   ```

---

## Performance Issues

### Problem: Slow response times

**Symptoms**: Victor takes a long time to respond or execute tools.

**Solutions**:

1. **Check provider performance**:
   ```bash
   # Try different provider
   victor chat --provider anthropic  # May be faster
   ```

2. **Use caching**:
   ```python
   # Enable semantic cache
   victor chat --cache-enabled
   ```

3. **Optimize tool selection**:
   ```bash
   # Use hybrid strategy (default)
   victor config set tool_selection_strategy hybrid
   ```

4. **Reduce tool budget**:
   ```bash
   # Limit tool calls
   victor chat --tool-budget 10
   ```

5. **Use local provider**:
   ```bash
   # Ollama for local inference (faster, no network)
   victor chat --provider ollama --model llama2
   ```

6. **Profile performance**:
   ```bash
   victor benchmark run --profile
   ```

---

### Problem: High memory usage

**Symptoms**: Victor uses excessive memory or crashes with OOM.

**Solutions**:

1. **Check available memory**:
   ```bash
   free -h  # Linux
   vm_stat  # macOS
   ```

2. **Reduce context window**:
   ```bash
   victor chat --max-tokens 2000  # Smaller context
   ```

3. **Clear cache**:
   ```bash
   victor cache clear
   ```

4. **Use streaming**:
   ```bash
   victor chat --stream  # Process in chunks
   ```

5. **Limit concurrent operations**:
   ```python
   # Reduce parallelism
   config.set("max_concurrent_tools", 3)
   ```

---

## Search and RAG Problems

### Problem: Search backend fails

**Symptoms**:
```
[abc12345] All 2 search backends failed for 'semantic'
```

**Solutions**:

1. **Check backend configuration**:
   ```bash
   victor config show search.backends
   ```

2. **Try alternative search type**:
   ```bash
   victor config set search.type keyword
   ```

3. **Verify embeddings service**:
   ```bash
   victor doctor  # Check embedding service status
   ```

4. **Check API keys**:
   ```bash
   echo $OPENAI_API_KEY  # For embeddings
   ```

5. **Use local embeddings**:
   ```bash
   # Use Ollama for local embeddings
   victor config set embeddings.provider ollama
   ```

---

### Problem: Poor search results

**Symptoms**: Search returns irrelevant results.

**Solutions**:

1. **Improve query**:
   - Use specific terms
   - Include relevant keywords
   - Try different phrasing

2. **Adjust search strategy**:
   ```bash
   # Try hybrid search
   victor config set search.type hybrid

   # Adjust semantic weight
   victor config set search.semantic_weight 0.7
   ```

3. **Rebuild index**:
   ```bash
   victor index rebuild --force
   ```

4. **Check index quality**:
   ```bash
   victor index stats
   ```

5. **Update embeddings**:
   ```bash
   victor embeddings update --model text-embedding-3-small
   ```

---

## Network and Connectivity

### Problem: Connection refused errors

**Symptoms**:
```
[abc12345] Network error: Connection refused
```

**Solutions**:

1. **Check internet connection**:
   ```bash
   ping google.com
   ping api.anthropic.com
   ```

2. **Verify proxy settings**:
   ```bash
   echo $HTTP_PROXY
   echo $HTTPS_PROXY

   # Or unset proxy
   unset HTTP_PROXY
   unset HTTPS_PROXY
   ```

3. **Check firewall rules**:
   ```bash
   # Linux
   sudo iptables -L

   # macOS
   sudo pfctl -s rules
   ```

4. **Try different endpoint**:
   ```python
   # Configure alternative endpoint
   config.set("anthropic.base_url", "https://alternative-endpoint.com")
   ```

5. **Use VPN if needed**:
   - Some providers blocked in certain regions

---

### Problem: DNS resolution fails

**Symptoms**:
```
[abc12345] Network error: Name or service not known
```

**Solutions**:

1. **Check DNS configuration**:
   ```bash
   cat /etc/resolv.conf
   ```

2. **Use alternative DNS**:
   ```bash
   # Use Google DNS
   echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
   ```

3. **Flush DNS cache**:
   ```bash
   # macOS
   sudo dscacheutil -flushcache
   sudo killall -HUP mDNSResponder

   # Linux
   sudo systemd-resolve --flush-caches
   ```

4. **Test DNS**:
   ```bash
   nslookup api.anthropic.com
   dig api.anthropic.com
   ```

---

## File and Path Issues

### Problem: File not found errors

**Symptoms**:
```
[abc12345] File not found: /path/to/file.txt
```

**Solutions**:

1. **Verify file exists**:
   ```bash
   ls -la /path/to/file.txt
   ```

2. **Check current directory**:
   ```bash
   pwd
   ```

3. **Use absolute path**:
   ```python
   # Wrong - relative path
   tool.execute(path="file.txt")

   # Correct - absolute path
   tool.execute(path="/full/path/to/file.txt")
   ```

4. **Check working directory context**:
   ```python
   import os
   print(os.getcwd())  # Current working directory
   ```

5. **Use path expansion**:
   ```python
   import os
   path = os.path.expanduser("~/file.txt")  # Expands ~
   path = os.path.abspath("file.txt")  # Converts to absolute
   ```

---

### Problem: Permission denied errors

**Symptoms**:
```
[abc12345] File operation failed: Permission denied
```

**Solutions**:

1. **Check file permissions**:
   ```bash
   ls -la /path/to/file
   ```

2. **Change permissions**:
   ```bash
   # Make readable
   chmod +r file

   # Make writable
   chmod +w file

   # Make executable
   chmod +x script.sh
   ```

3. **Check ownership**:
   ```bash
   ls -l /path/to/file
   chown user:group file  # Change owner
   ```

4. **Run with appropriate permissions**:
   ```bash
   # Use sudo with caution
   sudo victor chat

   # Or fix permissions instead (recommended)
   ```

5. **Check directory permissions**:
   ```bash
   ls -ld /path/to/dir
   chmod +x /path/to/dir  # Execute permission needed for directory access
   ```

---

## Memory and Resources

### Problem: Out of memory errors

**Symptoms**:
```
MemoryError: Unable to allocate memory
```

**Solutions**:

1. **Check available memory**:
   ```bash
   free -h  # Linux
   vm_stat  # macOS
   ```

2. **Reduce context size**:
   ```bash
   victor chat --max-tokens 1000
   ```

3. **Limit concurrent operations**:
   ```python
   config.set("max_concurrent_tools", 2)
   ```

4. **Clear cache**:
   ```bash
   victor cache clear
   ```

5. **Use streaming**:
   ```bash
   victor chat --stream  # Process incrementally
   ```

6. **Close other applications**:
   - Free up system memory

---

### Problem: CPU usage too high

**Symptoms**: System becomes unresponsive when using Victor.

**Solutions**:

1. **Limit parallelism**:
   ```python
   config.set("max_workers", 2)
   ```

2. **Use resource limits**:
   ```bash
   # Use ulimit to restrict
   ulimit -u 100  # Limit processes
   ```

3. **Profile performance**:
   ```bash
   victor profile --cpu
   ```

4. **Use less intensive provider**:
   ```bash
   # Local providers use less CPU
   victor chat --provider ollama
   ```

---

## Getting Additional Help

If none of these solutions work:

### 1. Check Documentation
- Error Reference: `docs/errors.md`
- API Documentation
- Architecture Docs

### 2. Run Diagnostics
```bash
victor doctor  # System diagnostics
victor logs show --tail 100  # Recent logs
```

### 3. Export Debug Info
```bash
victor debug-export > debug-info.txt
```

### 4. Report Issue
Include:
- Error message and correlation ID
- Steps to reproduce
- Victor version: `victor --version`
- System info: `victor doctor`
- Logs: `victor logs show --correlation-id <id>`

### 5. Community Support
- GitHub Issues
- Discord/Slack
- Stack Overflow

---

**Last Updated**: 2025-01-14
**Version**: 0.5.1
