# Victor AI Demo Recording Script

## Setup

```bash
cd ~/code/webui/investor_homelab
```

---

## Demo 1: Simple Prompt - Local Ollama (Default Profile)

**Mode:** Explore
**Provider:** Ollama (qwen3-coder:30b)
**Complexity:** Simple

```bash
# Start Victor with default profile (local Ollama)
victor chat --mode explore --tui
```

**Prompt:**
```
List all Python files in this project and give me a one-line description of each
```

**Expected:** Quick file discovery, ~3-5 tool calls, 20-30 seconds

---

## Demo 2: Medium Prompt - DeepSeek Cloud

**Mode:** Explore → Plan
**Provider:** DeepSeek
**Complexity:** Medium

```bash
victor chat --profile deepseek --mode explore --tui
```

**Prompt:**
```
Analyze the database_schema.py file. What design improvements would you recommend for better data integrity and query performance?
```

**Expected:** Deep analysis with thinking tokens visible, ~60-90 seconds, recommendations for:
- Foreign key constraints
- Index optimization
- ENUM constraints
- Unique constraints

---

## Demo 3: Complex Prompt - OpenAI GPT-4.1

**Mode:** Plan → Build
**Provider:** OpenAI
**Complexity:** Complex

```bash
victor chat --profile gpt-4.1 --mode plan --tui
```

**Prompt:**
```
Create a comprehensive pytest test suite for the WebSearchClient class in utils/web_search_client.py. Include tests for:
- Initialization and configuration
- Google Custom Search API calls (success and error cases)
- News API integration
- Database operations
- Edge cases and error handling

Use proper mocking with unittest.mock.patch for all external dependencies.
```

**Expected:**
1. Plan mode creates detailed implementation plan
2. Switch to build mode: `/mode build`
3. Generates ~200-300 lines of pytest code
4. ~2-3 minutes total

---

## Demo 4: Explore Mode Deep Dive - Local

**Mode:** Explore
**Provider:** Default (Ollama)
**Complexity:** Medium

```bash
victor chat --mode explore --tui
```

**Prompt:**
```
How does the protocols/ directory implement SOLID principles? Show me the interface definitions and explain the dependency injection pattern used.
```

**Expected:** Uses semantic search, reads multiple files, explains architecture

---

## Demo 5: Plan Mode Architecture - DeepSeek

**Mode:** Plan
**Provider:** DeepSeek
**Complexity:** Complex

```bash
victor chat --profile deepseek --mode plan --tui
```

**Prompt:**
```
I want to add a new data provider for Yahoo Finance that follows the existing protocol patterns. Create a detailed implementation plan including:
1. Interface compliance with IDataProvider
2. Rate limiting strategy
3. Error handling approach
4. Integration with existing NewsModel
```

**Expected:** Detailed plan with file locations, code snippets, architecture decisions

---

## Demo 6: Build Mode Implementation - GPT-4.1

**Mode:** Build
**Provider:** OpenAI GPT-4.1
**Complexity:** Medium

```bash
victor chat --profile gpt-4.1 --mode build --tui
```

**Prompt:**
```
Add a retry decorator with exponential backoff to the web_search_client.py file. It should:
- Retry up to 3 times
- Use exponential backoff (1s, 2s, 4s delays)
- Log retry attempts
- Re-raise the final exception if all retries fail
```

**Expected:** Creates/modifies code with proper implementation

---

## Demo 7: Multi-Provider Comparison

Run the same prompt with different providers to show speed/quality tradeoffs:

**Prompt:**
```
What are the main components of this codebase and how do they interact?
```

### Local (Fastest startup, moderate speed)
```bash
victor chat --mode explore --no-tui "What are the main components of this codebase and how do they interact?"
```

### DeepSeek (Thinking visible, thorough)
```bash
victor chat --profile deepseek --mode explore --no-tui "What are the main components of this codebase and how do they interact?"
```

### OpenAI (Fast, concise)
```bash
victor chat --profile gpt-4.1 --mode explore --no-tui "What are the main components of this codebase and how do they interact?"
```

---

## Slash Commands to Demonstrate

During any session, show these commands:

```
/help           # Show available commands
/mode explore   # Switch to explore mode
/mode plan      # Switch to plan mode
/mode build     # Switch to build mode
/tools          # List available tools
/profile        # Show current profile
/stats          # Show session statistics
/clear          # Clear conversation
```

---

## Key Talking Points

1. **25+ LLM Providers** - Switch seamlessly between local and cloud
2. **45 Enterprise Tools** - Code search, git, file editing, web fetch
3. **3 Modes** - Explore (research), Plan (design), Build (implement)
4. **Air-Gapped Mode** - Works entirely offline with local LLMs
5. **MCP Support** - Integrates with Claude Desktop
6. **VS Code Extension** - Full IDE integration

---

## Recording Tips

1. Use a clean terminal with good font size (14-16pt)
2. Pause briefly after each command for readability
3. Show the status bar updates during tool execution
4. Highlight thinking tokens when using DeepSeek
5. Show tool execution progress in real-time
6. End with `/stats` to show performance metrics
