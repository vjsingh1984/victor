"""
Critical fixes for Victor's orchestrator to resolve infinite loops and improve reliability.
"""

# Fix 1: Tool Execution Counter Bug
def fix_tool_execution_counter():
    """
    CRITICAL: The main issue is that tool_calls_used is never incremented.
    
    In orchestrator.py around line 2800+ in _stream_chat_impl:
    """
    
    # BEFORE (broken):
    """
    if tool_calls:
        tool_results = await self._handle_tool_calls(tool_calls)
        # Missing: self.tool_calls_used += len(tool_calls)
        continue  # Loop continues without updating counter
    """
    
    # AFTER (fixed):
    """
    if tool_calls:
        tool_results = await self._handle_tool_calls(tool_calls)
        self.tool_calls_used += len(tool_calls)  # CRITICAL FIX
        
        # Update unified tracker
        for tc in tool_calls:
            self.unified_tracker.record_tool_call(tc.get("name", ""), tc.get("arguments", {}))
        
        continue  # Now loop will eventually terminate
    """


# Fix 2: Simplify Tool Selection
def simplified_tool_selection():
    """
    Replace the complex 5-layer tool selection with a simple, reliable approach.
    """
    
    def select_tools_simple(self, message: str) -> List[ToolDefinition]:
        """Simplified tool selection that actually works."""
        from victor.providers.base import ToolDefinition
        
        # Always include core tools
        core_tools = ["read", "write", "ls", "search", "shell"]
        selected = set(core_tools)
        
        # Add context-specific tools based on keywords
        message_lower = message.lower()
        
        if any(kw in message_lower for kw in ["web", "search", "online", "lookup"]):
            selected.update(["web_search", "web_fetch"])
        
        if any(kw in message_lower for kw in ["git", "commit", "branch"]):
            selected.add("git")
        
        if any(kw in message_lower for kw in ["test", "run", "execute"]):
            selected.add("test")
        
        if any(kw in message_lower for kw in ["edit", "modify", "change", "update"]):
            selected.add("edit")
        
        # Convert to ToolDefinition objects
        tools = []
        for tool_name in selected:
            if self.tools.is_tool_enabled(tool_name):
                tool = self.tools.get_tool(tool_name)
                if tool:
                    tools.append(ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters
                    ))
        
        return tools[:12]  # Limit to 12 tools max


# Fix 3: Clear Loop Detection
def simplified_loop_detection():
    """
    Replace complex loop detection with simple, reliable checks.
    """
    
    class SimpleLoopDetector:
        def __init__(self):
            self.tool_history = []
            self.max_same_tool = 3
            self.max_iterations = 20
            self.iteration_count = 0
        
        def record_tool_call(self, tool_name: str, args: dict):
            self.iteration_count += 1
            signature = f"{tool_name}:{str(sorted(args.items()))}"
            self.tool_history.append(signature)
            
            # Keep only recent history
            if len(self.tool_history) > 10:
                self.tool_history = self.tool_history[-10:]
        
        def should_stop(self) -> tuple[bool, str]:
            # Check iteration limit
            if self.iteration_count >= self.max_iterations:
                return True, f"Max iterations reached ({self.iteration_count})"
            
            # Check for repeated tool calls
            if len(self.tool_history) >= self.max_same_tool:
                recent = self.tool_history[-self.max_same_tool:]
                if len(set(recent)) == 1:  # All same
                    return True, f"Same tool repeated {self.max_same_tool} times: {recent[0]}"
            
            return False, ""


# Fix 4: Timeout Protection
def add_timeout_protection():
    """
    Add timeout protection to prevent infinite loops.
    """
    
    import asyncio
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def timeout_protection(seconds: int = 300):  # 5 minute timeout
        """Context manager to prevent infinite loops."""
        try:
            async with asyncio.timeout(seconds):
                yield
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {seconds} seconds")
            raise RuntimeError(f"Victor timed out after {seconds} seconds - possible infinite loop")
    
    # Usage in stream_chat:
    """
    async def stream_chat(self, user_message: str):
        async with timeout_protection(300):  # 5 minute max
            async for chunk in self._stream_chat_impl(user_message):
                yield chunk
    """


# Fix 5: Error Recovery
def add_error_recovery():
    """
    Add proper error recovery to handle tool failures gracefully.
    """
    
    async def handle_tool_calls_with_recovery(self, tool_calls: List[dict]) -> List[dict]:
        """Execute tool calls with proper error recovery."""
        results = []
        
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("arguments", {})
            
            try:
                # Execute tool with timeout
                result = await asyncio.wait_for(
                    self.tool_executor.execute(tool_name, **tool_args),
                    timeout=60.0  # 1 minute per tool
                )
                
                results.append({
                    "name": tool_name,
                    "args": tool_args,
                    "success": True,
                    "result": result,
                    "elapsed": 0.0  # Add timing
                })
                
            except asyncio.TimeoutError:
                logger.warning(f"Tool {tool_name} timed out")
                results.append({
                    "name": tool_name,
                    "args": tool_args,
                    "success": False,
                    "error": "Tool execution timed out",
                    "elapsed": 60.0
                })
                
            except Exception as e:
                logger.warning(f"Tool {tool_name} failed: {e}")
                results.append({
                    "name": tool_name,
                    "args": tool_args,
                    "success": False,
                    "error": str(e),
                    "elapsed": 0.0
                })
        
        return results


# Fix 6: Minimal Orchestrator
def create_minimal_orchestrator():
    """
    Simplified orchestrator that focuses on core functionality.
    """
    
    class MinimalOrchestrator:
        def __init__(self, provider, model, tools):
            self.provider = provider
            self.model = model
            self.tools = tools
            self.conversation = []
            self.tool_calls_used = 0
            self.max_tool_calls = 50
            self.max_iterations = 20
            self.loop_detector = SimpleLoopDetector()
        
        async def chat(self, message: str) -> str:
            """Simple, reliable chat implementation."""
            self.conversation.append({"role": "user", "content": message})
            
            for iteration in range(self.max_iterations):
                # Check stopping conditions
                should_stop, reason = self.loop_detector.should_stop()
                if should_stop:
                    return f"Stopped: {reason}"
                
                if self.tool_calls_used >= self.max_tool_calls:
                    return "Tool budget exceeded"
                
                # Get tools for this iteration
                tools = self.select_tools_simple(message)
                
                # Get response from model
                response = await self.provider.chat(
                    messages=self.conversation,
                    model=self.model,
                    tools=tools
                )
                
                # Add response to conversation
                if response.content:
                    self.conversation.append({"role": "assistant", "content": response.content})
                
                # Execute tool calls if any
                if response.tool_calls:
                    tool_results = await self.execute_tools(response.tool_calls)
                    self.tool_calls_used += len(response.tool_calls)
                    
                    # Add tool results to conversation
                    for result in tool_results:
                        self.conversation.append({
                            "role": "tool",
                            "content": str(result.get("result", "")),
                            "tool_call_id": result.get("id", "")
                        })
                    
                    # Continue for follow-up response
                    continue
                
                # No tool calls - return final response
                return response.content or "No response generated"
            
            return "Max iterations reached"
        
        def select_tools_simple(self, message: str) -> List[dict]:
            """Simple tool selection."""
            # Implementation from simplified_tool_selection above
            pass
        
        async def execute_tools(self, tool_calls: List[dict]) -> List[dict]:
            """Execute tools with proper error handling."""
            # Implementation from add_error_recovery above
            pass


# Implementation Priority:
"""
1. IMMEDIATE (Fix the infinite loop):
   - Add tool_calls_used increment in _stream_chat_impl
   - Add timeout protection wrapper
   
2. SHORT TERM (1-2 days):
   - Replace complex tool selection with simplified version
   - Add simple loop detection
   - Improve error recovery
   
3. MEDIUM TERM (1 week):
   - Create minimal orchestrator alternative
   - Extract components properly
   - Add comprehensive testing
"""