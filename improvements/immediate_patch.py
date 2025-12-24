"""
IMMEDIATE PATCH: Fix the infinite loop in Victor's orchestrator.

Apply this patch to orchestrator.py to fix the critical tool execution bug.
"""

# PATCH 1: Fix tool execution counter in _stream_chat_impl
# Location: victor/agent/orchestrator.py, around line 2800-3000

def patch_tool_execution_counter():
    """
    Find this section in _stream_chat_impl and add the missing counter update:
    """
    
    # FIND THIS CODE (around line 2950):
    """
    if tool_calls:
        # Record tool calls in progress tracker for loop detection
        # Progress tracker handles unique resource tracking internally
        for tc in tool_calls or []:
            tool_name = tc.get("name", "")
            tool_args = tc.get("arguments", {})

            # Record tool call in unified tracker (single source of truth)
            self.unified_tracker.record_tool_call(tool_name, tool_args)
    """
    
    # ADD THIS AFTER THE ABOVE BLOCK:
    """
        # Execute tool calls and update counters
        tool_results = await self._handle_tool_calls(tool_calls)
        
        # CRITICAL FIX: Update tool calls counter
        self.tool_calls_used += len(tool_calls)
        
        # Yield tool results
        for result in tool_results:
            tool_name = result.get("name", "tool")
            elapsed = result.get("elapsed", 0.0)
            tool_args = result.get("args", {})
            if result.get("success"):
                yield StreamChunk(
                    content="",
                    metadata={
                        "tool_result": {
                            "name": tool_name,
                            "success": True,
                            "elapsed": elapsed,
                            "arguments": tool_args,
                        }
                    },
                )
            else:
                error_msg = result.get("error", "failed")
                yield StreamChunk(
                    content="",
                    metadata={
                        "tool_result": {
                            "name": tool_name,
                            "success": False,
                            "elapsed": elapsed,
                            "arguments": tool_args,
                            "error": error_msg,
                        }
                    },
                )
        
        # Continue loop for follow-up response
        continue
    """


# PATCH 2: Add timeout protection to stream_chat
# Location: victor/agent/orchestrator.py, around line 2400

def patch_timeout_protection():
    """
    Wrap the stream_chat method with timeout protection:
    """
    
    # FIND THIS METHOD:
    """
    async def stream_chat(self, user_message: str) -> AsyncIterator[StreamChunk]:
        async for chunk in self._stream_chat_impl(user_message):
            yield chunk
    """
    
    # REPLACE WITH:
    """
    async def stream_chat(self, user_message: str) -> AsyncIterator[StreamChunk]:
        import asyncio
        
        try:
            # 5 minute timeout to prevent infinite loops
            async with asyncio.timeout(300):
                async for chunk in self._stream_chat_impl(user_message):
                    yield chunk
        except asyncio.TimeoutError:
            logger.error("Victor timed out after 5 minutes - possible infinite loop")
            yield StreamChunk(
                content="\\n\\n⚠️ Operation timed out after 5 minutes. This may indicate an infinite loop.\\n",
                is_final=True
            )
    """


# PATCH 3: Simplify tool selection (optional but recommended)
# Location: victor/agent/tool_selection.py

def patch_tool_selection():
    """
    Add a simple fallback tool selection method:
    """
    
    # ADD THIS METHOD TO ToolSelector class:
    """
    def select_tools_fallback(self, user_message: str) -> List[ToolDefinition]:
        '''Fallback tool selection when complex selection fails.'''
        from victor.providers.base import ToolDefinition
        
        # Core tools always available
        core_tools = ["read", "write", "ls", "search", "shell"]
        selected = set(core_tools)
        
        # Add context-specific tools
        message_lower = user_message.lower()
        
        if any(kw in message_lower for kw in ["web", "online", "lookup"]):
            selected.update(["web_search", "web_fetch"])
        
        if "git" in message_lower:
            selected.add("git")
        
        if any(kw in message_lower for kw in ["test", "run"]):
            selected.add("test")
        
        if any(kw in message_lower for kw in ["edit", "modify", "change"]):
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
        
        return tools[:10]  # Limit to 10 tools
    """
    
    # MODIFY select_tools method to use fallback:
    """
    async def select_tools(self, user_message: str, **kwargs) -> List[ToolDefinition]:
        try:
            # Try complex selection first
            if self.use_semantic and self.semantic_selector:
                tools = await self.select_semantic(user_message, **kwargs)
            else:
                tools = self.select_keywords(user_message, **kwargs)
            
            # If no tools selected, use fallback
            if not tools:
                logger.warning("Complex tool selection returned 0 tools, using fallback")
                tools = self.select_tools_fallback(user_message)
            
            return tools
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}, using fallback")
            return self.select_tools_fallback(user_message)
    """


# PATCH 4: Add emergency stop mechanism
# Location: victor/agent/orchestrator.py

def patch_emergency_stop():
    """
    Add emergency stop mechanism to prevent runaway loops:
    """
    
    # ADD THIS TO __init__ method:
    """
    # Emergency stop mechanism
    self._emergency_stop_threshold = 100  # Max iterations before emergency stop
    self._iteration_count = 0
    """
    
    # ADD THIS CHECK IN _stream_chat_impl main loop:
    """
    # Emergency stop check
    self._iteration_count += 1
    if self._iteration_count > self._emergency_stop_threshold:
        logger.error(f"Emergency stop triggered after {self._iteration_count} iterations")
        yield StreamChunk(
            content=f"\\n\\n🛑 Emergency stop: Too many iterations ({self._iteration_count}). "
                   "This indicates a serious bug. Please report this issue.\\n",
            is_final=True
        )
        return
    """


# VERIFICATION SCRIPT
def verify_patches():
    """
    Script to verify the patches are working:
    """
    
    verification_code = '''
import asyncio
import logging
from victor.agent.orchestrator import AgentOrchestrator

async def test_no_infinite_loop():
    """Test that Victor doesn't get stuck in infinite loops."""
    
    # Set up logging to see what's happening
    logging.basicConfig(level=logging.INFO)
    
    # Create orchestrator (you'll need to adapt this to your setup)
    orchestrator = create_test_orchestrator()
    
    # Test with a timeout
    try:
        async with asyncio.timeout(60):  # 1 minute max
            response_chunks = []
            async for chunk in orchestrator.stream_chat("analyze the codebase"):
                response_chunks.append(chunk)
                print(f"Chunk: {chunk.content[:50]}...")
                
                # Stop if we get too many chunks (indicates loop)
                if len(response_chunks) > 100:
                    print("❌ Too many chunks - possible infinite loop")
                    break
            
            print(f"✅ Completed with {len(response_chunks)} chunks")
            print(f"Tool calls used: {orchestrator.tool_calls_used}")
            
    except asyncio.TimeoutError:
        print("❌ Test timed out - infinite loop detected")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_no_infinite_loop())
'''
    
    return verification_code


# SUMMARY OF PATCHES:
"""
1. CRITICAL: Add tool_calls_used increment in _stream_chat_impl
2. SAFETY: Add timeout protection to stream_chat
3. RELIABILITY: Add fallback tool selection
4. EMERGENCY: Add emergency stop mechanism

Apply patches in order:
1. Patch 1 (critical) - fixes the main bug
2. Patch 2 (safety) - prevents hangs
3. Patch 3 (optional) - improves reliability
4. Patch 4 (emergency) - last resort protection

Test with the verification script to ensure patches work.
"""