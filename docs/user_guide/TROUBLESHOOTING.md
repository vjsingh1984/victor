# Victor AI Troubleshooting Guide

**Solving common issues quickly**

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Provider Issues](#provider-issues)
4. [Performance Issues](#performance-issues)
5. [Team Coordination Issues](#team-coordination-issues)
6. [Tool Execution Issues](#tool-execution-issues)
7. [Memory and Context Issues](#memory-and-context-issues)
8. [Workflow Issues](#workflow-issues)
9. [Debug Mode](#debug-mode)
10. [Getting Help](#getting-help)

---

## Quick Diagnostics

### Health Check Command

```bash
# Run comprehensive health check
victor doctor

# Checks:
# - Installation integrity
# - Provider connectivity
# - Configuration validity
# - Tool availability
# - Memory and disk space
```

### Version Information

```bash
# Check Victor version
victor --version

# Check Python version
python --version  # Should be 3.10+

# Check provider versions
victor provider list
```

### Configuration Check

```bash
# View current configuration
victor config show

# Validate configuration
victor config validate

# Test provider connection
victor provider test
```

---

## Installation Issues

### Issue: "Command not found: victor"

**Symptoms**:
```bash
victor
# bash: victor: command not found
```

**Causes**:
1. Victor not installed
2. Installation path not in PATH
3. Installation in different Python environment

**Solutions**:

```bash
# Solution 1: Install Victor
pip install victor-ai

# Solution 2: Install with pipx (recommended)
pipx install victor-ai

# Solution 3: Check Python path
which python
pip install victor-ai
# Ensure ~/.local/bin is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Solution 4: Use python -m
python -m victor chat "Test"
```

### Issue: "ModuleNotFoundError: No module named 'victor'"

**Symptoms**:
```python
import victor
# ModuleNotFoundError: No module named 'victor'
```

**Solutions**:

```bash
# Solution 1: Install in correct environment
pip install victor-ai

# Solution 2: Check which Python you're using
which python
pip install victor-ai

# Solution 3: Install for specific Python
python3.10 -m pip install victor-ai

# Solution 4: Verify installation
pip show victor-ai
# Should show version and location
```

### Issue: "Permission denied" during installation

**Solutions**:

```bash
# Solution 1: Use pipx (recommended)
pipx install victor-ai

# Solution 2: Use user install
pip install --user victor-ai

# Solution 3: Use virtual environment
python -m venv venv
source venv/bin/activate
pip install victor-ai
```

### Issue: "SSL certificate verification failed"

**Symptoms**:
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

**Solutions**:

```bash
# Solution 1: Disable SSL verify (not recommended for production)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org victor-ai

# Solution 2: Update certificates (macOS)
/Applications/Python\ 3.10/Install\ Certificates.command

# Solution 3: Update pip
pip install --upgrade pip
pip install victor-ai
```

---

## Provider Issues

### Issue: "API key not found"

**Symptoms**:
```
ValueError: ANTHROPIC_API_KEY not found in environment or config
```

**Solutions**:

```bash
# Solution 1: Set environment variable
export ANTHROPIC_API_KEY=sk-ant-...
# Or add to ~/.bashrc or ~/.zshrc
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc

# Solution 2: Set in config file
cat > ~/.victor/config.yaml << EOF
provider:
  name: anthropic
  api_key: sk-ant-...
EOF

# Solution 3: Pass as argument (not recommended for security)
victor chat --api-key sk-ant-... "Test"
```

### Issue: "Provider not responding"

**Symptoms**:
```
TimeoutError: Provider did not respond within 30 seconds
```

**Solutions**:

```bash
# Solution 1: Check network connectivity
ping api.anthropic.com

# Solution 2: Increase timeout
victor chat --timeout 60 "Test"

# Solution 3: Check provider status
curl https://status.anthropic.com

# Solution 4: Switch provider
victor chat --provider ollama "Test"  # Use local provider

# Solution 5: Use proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### Issue: "Rate limit exceeded"

**Symptoms**:
```
RateLimitError: Rate limit exceeded, please retry later
```

**Solutions**:

```bash
# Solution 1: Wait and retry
# Rate limits typically reset per minute

# Solution 2: Switch provider
victor chat --provider openai "Test"

# Solution 3: Use local provider (no rate limits)
victor chat --provider ollama "Test"

# Solution 4: Reduce request frequency
# Add delay between requests
```

### Issue: "Model not found"

**Symptoms**:
```
ValueError: Model 'claude-sonnet-4-5' not found for provider 'anthropic'
```

**Solutions**:

```bash
# Solution 1: List available models
victor provider models

# Solution 2: Use correct model name
victor chat --model claude-sonnet-4-5 "Test"  # Correct
victor chat --model Claude-Sonnet-4-5 "Test"  # Wrong case

# Solution 3: Update model capabilities cache
victor provider update-models

# Solution 4: Check provider documentation
# https://docs.anthropic.com/claude/docs/models-overview
```

### Issue: Ollama connection failed

**Symptoms**:
```
ConnectionError: Cannot connect to Ollama at localhost:11434
```

**Solutions**:

```bash
# Solution 1: Start Ollama
ollama serve

# Solution 2: Check Ollama is running
curl http://localhost:11434/api/tags

# Solution 3: Pull a model
ollama pull qwen2.5-coder:7b

# Solution 4: Check Ollama logs
ollama logs

# Solution 5: Restart Ollama
killall ollama
ollama serve
```

---

## Performance Issues

### Issue: Slow response times

**Symptoms**:
- Responses take 10+ seconds
- High latency on simple queries

**Solutions**:

```bash
# Solution 1: Use faster provider
# Local: Ollama (fastest)
victor chat --provider ollama "Test"

# Cloud: Groq (ultra-fast)
victor chat --provider groq "Test"

# Solution 2: Use smaller model
victor chat --model haiku "Test"  # Faster than sonnet

# Solution 3: Reduce context
victor chat --context-size 1000 "Test"

# Solution 4: Disable observability (production)
export VICTOR_OBSERVABILITY_ENABLED=false

# Solution 5: Check network speed
ping api.anthropic.com
```

### Issue: High memory usage

**Symptoms**:
- Victor using 500MB+ memory
- System slows down

**Solutions**:

```bash
# Solution 1: Reduce context window
victor chat --max-context-size 50000 "Test"

# Solution 2: Clear conversation history
victor session clear

# Solution 3: Disable caching
export VICTOR_CACHE_ENABLED=false

# Solution 4: Use streaming (reduces memory)
victor chat --stream "Test"

# Solution 5: Monitor memory
victor metrics memory
```

### Issue: CPU usage high

**Symptoms**:
- Victor using 100% CPU
- Fan running constantly

**Solutions**:

```bash
# Solution 1: Use smaller model for local providers
ollama pull qwen2.5-coder:3b  # Smaller than 7b

# Solution 2: Reduce concurrent operations
export VICTOR_MAX_CONCURRENT_TOOLS=1

# Solution 3: Disable expensive features
export VICTOR_EMBEDDINGS_ENABLED=false
export VICTOR_SEMANTIC_SEARCH_ENABLED=false

# Solution 4: Use cloud provider (offloads CPU)
victor chat --provider anthropic "Test"
```

---

## Team Coordination Issues

### Issue: Team execution hangs

**Symptoms**:
- Team task never completes
- No progress after several minutes

**Solutions**:

```python
# Solution 1: Add timeout
from victor.teams import create_coordinator

coordinator = create_coordinator(orchestrator)
result = await coordinator.execute_task(
    "Review code",
    {},
    timeout=300  # 5 minutes
)

# Solution 2: Reduce iterations
coordinator.config.max_iterations = 25  # Instead of 50

# Solution 3: Use simpler formation
coordinator.set_formation(TeamFormation.SEQUENTIAL)  # Instead of CONSENSUS

# Solution 4: Check member budgets
for member in coordinator.members:
    member.tool_budget = 15  # Reduce from 25
```

### Issue: "No team members configured"

**Symptoms**:
```
ValueError: Cannot execute task: no team members configured
```

**Solutions**:

```python
# Solution 1: Add team members
from victor.teams import TeamMember
from victor.agent.subagents.base import SubAgentRole

coordinator.add_member(
    role=SubAgentRole.RESEARCHER,
    name="Researcher",
    goal="Research task"
)

# Solution 2: Use default team
coordinator = create_coordinator(orchestrator)
coordinator.use_default_team()

# Solution 3: Check team configuration
print(f"Members: {len(coordinator.members)}")
```

### Issue: Team formation not working

**Symptoms**:
- Team formation doesn't execute as expected
- Members execute in wrong order

**Solutions**:

```python
# Solution 1: Verify formation is set
coordinator.set_formation(TeamFormation.PIPELINE)
print(f"Formation: {coordinator.config.formation}")

# Solution 2: Check member priorities
for member in coordinator.members:
    print(f"{member.name}: priority={member.priority}")
# Adjust priorities for sequential/pipeline

# Solution 3: Use correct formation for task
# HIERARCHICAL requires one manager
if coordinator.config.formation == TeamFormation.HIERARCHICAL:
    managers = [m for m in coordinator.members if m.is_manager]
    assert len(managers) == 1, "Need exactly one manager"
```

---

## Tool Execution Issues

### Issue: "Tool not found"

**Symptoms**:
```
ValueError: Tool 'read_file' not found in registry
```

**Solutions**:

```bash
# Solution 1: List available tools
victor tools list

# Solution 2: Check tool name spelling
victor tools show read_file  # Correct
victor tools show ReadFile   # Wrong case

# Solution 3: Update tool registry
victor tools update

# Solution 4: Enable tool category
victor config set tools.enabled_categories "file_operations,code_analysis"
```

### Issue: "Tool execution failed"

**Symptoms**:
```
ToolExecutionError: Tool 'execute_command' failed: Permission denied
```

**Solutions**:

```bash
# Solution 1: Check tool permissions
# Some tools require explicit approval

# Solution 2: Use auto-approve (caution!)
victor chat --auto-approve "Run tests"

# Solution 3: Check tool availability
victor tools test execute_command

# Solution 4: Use safer alternative
# Instead of execute_command, use read_file
```

### Issue: Tool budget exceeded

**Symptoms**:
```
ToolBudgetExceededError: Tool budget (100) exceeded
```

**Solutions**:

```python
# Solution 1: Increase budget
victor chat --tool-budget 200 "Review code"

# Solution 2: Use more efficient tools
victor chat --tool-selection-strategy keyword "Test"

# Solution 3: Monitor budget usage
from victor.agent.tool_pipeline import ToolPipeline

pipeline = ToolPipeline(orchestrator)
stats = pipeline.get_tool_usage_stats()
print(f"Used: {stats['used']}, Budget: {stats['budget']}")

# Solution 4: Disable expensive tools
export VICTOR_TOOLS_EXCLUDE="execute_command,run_tests"
```

---

## Memory and Context Issues

### Issue: "Context too large"

**Symptoms**:
```
ValueError: Context size (150000 tokens) exceeds maximum (100000)
```

**Solutions**:

```bash
# Solution 1: Reduce context size
victor chat --max-context-size 50000 "Review code"

# Solution 2: Clear conversation history
victor session clear

# Solution 3: Disable context compaction
export VICTOR_CONTEXT_COMPACTION_ENABLED=false

# Solution 4: Use mode with less exploration
victor chat --mode build "Review code"  # Less than --mode plan
```

### Issue: Conversation memory lost

**Symptoms**:
- Victor doesn't remember previous messages
- Context resets unexpectedly

**Solutions**:

```bash
# Solution 1: Enable persistent sessions
victor config set sessions.persistent true

# Solution 2: Check session file
ls -la ~/.victor/sessions/

# Solution 3: Enable checkpoints
pip install victor-ai[checkpoints]
export VICTOR_CHECKPOINT_ENABLED=true

# Solution 4: Use explicit session ID
victor chat --session-id my-session "Test"
```

### Issue: Semantic search not working

**Symptoms**:
- Search doesn't find relevant code
- Search results are poor

**Solutions**:

```bash
# Solution 1: Enable embeddings
export VICTOR_EMBEDDINGS_ENABLED=true

# Solution 2: Check embedding service
victor embeddings test

# Solution 3: Rebuild index
victor index rebuild

# Solution 4: Use hybrid search
export VICTOR_SEARCH_STRATEGY=hybrid
```

---

## Workflow Issues

### Issue: "Workflow not found"

**Symptoms**:
```
ValueError: Workflow 'deep_research' not found
```

**Solutions**:

```bash
# Solution 1: List available workflows
victor workflow list

# Solution 2: Check workflow path
victor workflow validate path/to/workflow.yaml

# Solution 3: Use correct workflow name
# Workflows are in ~/.victor/workflows/
ls ~/.victor/workflows/

# Solution 4: Create workflow
victor workflow create my-workflow
```

### Issue: Workflow validation failed

**Symptoms**:
```
WorkflowValidationError: Invalid workflow YAML
```

**Solutions**:

```bash
# Solution 1: Validate workflow
victor workflow validate workflow.yaml

# Solution 2: Check YAML syntax
python -c "import yaml; yaml.safe_load(open('workflow.yaml'))"

# Solution 3: Check required fields
# Each node needs: id, type, and appropriate fields

# Solution 4: Use workflow template
victor workflow template --type pipeline > my-workflow.yaml
```

### Issue: Workflow execution stuck

**Symptoms**:
- Workflow node doesn't complete
- No progress in workflow execution

**Solutions**:

```python
# Solution 1: Add timeout to workflow
workflow = compiler.compile("my-workflow", timeout=300)

# Solution 2: Check node dependencies
# Ensure node.next references exist

# Solution 3: Use workflow debugger
victor workflow debug workflow.yaml

# Solution 4: Enable workflow logging
export VICTOR_WORKFLOW_LOG_LEVEL=DEBUG
```

---

## Debug Mode

### Enable Debug Logging

```bash
# Solution 1: Enable debug mode
victor --debug chat "Test"

# Solution 2: Set log level
export VICTOR_LOG_LEVEL=DEBUG
victor chat "Test"

# Solution 3: Enable verbose output
victor --verbose chat "Test"

# Solution 4: Log to file
export VICTOR_LOG_FILE=/tmp/victor.log
victor chat "Test"
```

### Debug Python API

```python
# Solution 1: Enable logging
import logging
logging.basicConfig(level=logging.DEBUG)

from victor import Victor
vic = Victor(debug=True)

# Solution 2: Use debugger
import pdb
pdb.set_trace()

# Solution 3: Profile performance
import cProfile
cProfile.run('vic.chat("Test")', 'victor_profile')
# Then analyze with:
# python -m pstats victor_profile
```

### Get Diagnostic Information

```bash
# Generate diagnostic report
victor doctor > diagnostics.txt

# Include in bug report
```

---

## Getting Help

### Before Asking for Help

1. **Check the docs**: https://github.com/vjsingh1984/victor/docs
2. **Search existing issues**: https://github.com/vjsingh1984/victor/issues
3. **Run diagnostics**: `victor doctor`
4. **Check the FAQ**: [FAQ.md](FAQ.md)

### Creating a Bug Report

Include the following information:

```markdown
## Victor Version
victor --version

## Python Version
python --version

## Operating System
uname -a

## Error Message
[Paste full error traceback]

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Configuration
victor config show

## Diagnostics
victor doctor
```

### Getting Community Help

- **GitHub Issues**: https://github.com/vjsingh1984/victor/issues
- **GitHub Discussions**: https://github.com/vjsingh1984/victor/discussions
- **Email**: singhvjd@gmail.com

### Enterprise Support

For enterprise support, contact: singhvjd@gmail.com

---

## Quick Reference

### Common Commands

```bash
# Health check
victor doctor

# Test provider
victor provider test

# Validate config
victor config validate

# Debug mode
victor --debug chat "Test"

# Verbose output
victor --verbose chat "Test"

# Check logs
tail -f ~/.victor/logs/victor.log
```

### Environment Variables

```bash
# Debug
export VICTOR_LOG_LEVEL=DEBUG
export VICTOR_DEBUG=true

# Provider
export VICTOR_PROVIDER=anthropic
export VICTOR_API_KEY=sk-ant-...

# Performance
export VICTOR_OBSERVABILITY_ENABLED=false
export VICTOR_CACHE_ENABLED=false

# Tools
export VICTOR_TOOL_BUDGET=200
export VICTOR_TOOL_SELECTION_STRATEGY=keyword
```

### Common Fixes

| Issue | Quick Fix |
|-------|-----------|
| Command not found | `pipx install victor-ai` |
| API key not found | `export ANTHROPIC_API_KEY=sk-...` |
| Provider timeout | `victor chat --timeout 60 "Test"` |
| Slow response | `victor chat --provider ollama "Test"` |
| High memory | `victor session clear` |
| Team hangs | Add timeout to `execute_task()` |
| Tool not found | `victor tools update` |
| Context too large | `victor chat --max-context-size 50000 "Test"` |

---

**Still stuck?** Check the [FAQ](FAQ.md) or [create an issue](https://github.com/vjsingh1984/victor/issues/new)
