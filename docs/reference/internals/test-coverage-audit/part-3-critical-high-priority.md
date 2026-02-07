# Coordinator Test Coverage Audit - Part 3

**Part 3 of 4:** CRITICAL and HIGH Priority Test Recommendations

---

## Navigation

- [Part 1: Executive Summary & Modules 1-16](part-1-executive-summary-modules-1-16.md)
- [Part 2: Modules 17-28](part-2-modules-17-28.md)
- **[Part 3: CRITICAL & HIGH Priority](#)** (Current)
- [Part 4: MEDIUM/LOW Priority & Roadmap](part-4-medium-low-priority-roadmap.md)
- [**Complete Audit**](../TEST_COVERAGE_AUDIT.md)

---


### CRITICAL Priority (Extracted Methods - Must Test First)

These coordinators contain methods extracted from AgentOrchestrator. They have **ZERO** coverage and represent the highest risk.

#### 1. ChatCoordinator (4.77% coverage)
**Risk**: Core chat and streaming logic extracted from orchestrator
**Estimated Effort**: 40 hours
**Impact**: HIGH - Chat is the primary user interaction path

**Required Tests**:
```python
# test_chat_coordinator.py
class TestChatCoordinator:
    async def test_chat_single_turn(self):
        """Test basic non-streaming chat without tools"""

    async def test_chat_with_tool_calls(self):
        """Test chat with single tool execution"""

    async def test_chat_with_multiple_tools(self):
        """Test chat with sequential tool execution"""

    async def test_chat_max_iterations(self):
        """Test chat respects max iteration limit"""

    async def test_chat_error_recovery(self):
        """Test chat recovers from tool errors"""

    async def test_chat_with_rate_limit(self):
        """Test chat handles rate limiting"""

    async def test_stream_chat_basic(self):
        """Test basic streaming chat"""

    async def test_stream_chat_with_tools(self):
        """Test streaming chat with tool execution"""

    async def test_stream_chat_continuation(self):
        """Test streaming chat with automatic continuation"""

    async def test_stream_chat_intent_classification(self):
        """Test streaming chat with intent detection"""
```

#### 2. ConversationCoordinator (27.27% coverage)
**Risk**: Message management extracted from orchestrator
**Estimated Effort**: 12 hours
**Impact**: HIGH - All conversations go through this coordinator

**Required Tests**:
```python
# test_conversation_coordinator.py
class TestConversationCoordinator:
    def test_add_message_user(self):
        """Test adding user message"""

    def test_add_message_assistant(self):
        """Test adding assistant message"""

    def test_add_message_with_memory_manager(self):
        """Test persistence to memory manager"""

    def test_add_message_memory_manager_disabled(self):
        """Test behavior when memory manager disabled"""

    def test_add_message_with_usage_logger(self):
        """Test usage logging integration"""

    def test_reset_conversation_clears_history(self):
        """Test reset clears message history"""

    def test_reset_conversation_clears_tool_counters(self):
        """Test reset clears tool call counters"""

    def test_reset_conversation_resets_state_machine(self):
        """Test reset resets conversation state"""
```

#### 3. SearchCoordinator (60% coverage)
**Risk**: Search routing extracted from orchestrator
**Estimated Effort**: 8 hours
**Impact**: HIGH - Search is critical for codebase understanding

**Required Tests**:
```python
# test_search_coordinator.py
class TestSearchCoordinator:
    def test_route_search_query_web(self):
        """Test routing to web search"""

    def test_route_search_query_local(self):
        """Test routing to local search"""

    def test_route_search_query_empty_query(self):
        """Test handling of empty query"""

    def test_route_search_query_no_tools(self):
        """Test handling when no search tools available"""

    def test_route_search_query_invalid_type(self):
        """Test handling of invalid query type"""

    def test_get_recommended_search_tool_web(self):
        """Test web tool recommendation"""

    def test_get_recommended_search_tool_local(self):
        """Test local tool recommendation"""

    def test_get_recommended_search_tool_none(self):
        """Test recommendation when no tools available"""
```

#### 4. TeamCoordinator (44.44% coverage)
**Risk**: Multi-agent coordination extracted from orchestrator
**Estimated Effort**: 16 hours
**Impact**: HIGH - Multi-agent workflows are a key feature

**Required Tests**:
```python
# test_team_coordinator.py
class TestTeamCoordinator:
    def test_get_team_suggestions_coding(self):
        """Test team suggestions for coding tasks"""

    def test_get_team_suggestions_research(self):
        """Test team suggestions for research tasks"""

    def test_get_team_suggestions_empty(self):
        """Test team suggestions when no teams available"""

    def test_set_team_specs_valid(self):
        """Test setting valid team specs"""

    def test_set_team_specs_invalid(self):
        """Test setting invalid team specs"""

    def test_set_team_specs_overwrite(self):
        """Test overwriting existing team specs"""

    def test_create_team_pipeline(self):
        """Test creating pipeline team"""

    def test_create_team_parallel(self):
        """Test creating parallel team"""

    def test_create_team_failure(self):
        """Test handling of team creation failure"""
```

### HIGH Priority (Core Functionality)

#### 5. ToolCoordinator (21.65% coverage)
**Risk**: Core tool orchestration facade
**Estimated Effort**: 32 hours
**Impact**: CRITICAL - All tool operations go through this coordinator

**Required Tests**:
```python
# test_tool_coordinator.py
class TestToolCoordinator:
    async def test_select_and_execute_tools_success(self):
        """Test successful tool selection and execution"""

    async def test_select_and_execute_tools_no_tools(self):
        """Test handling when no tools available"""

    async def test_select_and_execute_tools_parallel(self):
        """Test parallel tool execution"""

    async def test_select_and_execute_tools_failure(self):
        """Test handling of tool execution failures"""

    async def test_select_and_execute_tools_budget_exceeded(self):
        """Test budget enforcement"""

    async def test_execute_tool_success(self):
        """Test single tool execution"""

    async def test_execute_tool_not_found(self):
        """Test handling of unknown tool"""

    async def test_execute_tool_disabled(self):
        """Test handling of disabled tool"""

    async def test_execute_tool_validation_error(self):
        """Test handling of validation errors"""

    async def test_execute_tool_calls_single(self):
        """Test single tool call execution"""

    async def test_execute_tool_calls_multiple(self):
        """Test multiple tool call execution"""

    async def test_execute_tool_calls_parallel(self):
        """Test parallel tool call execution"""

    async def test_execute_tool_calls_failure_handling(self):
        """Test error handling in tool calls"""
```

#### 6. ToolSelectionCoordinator (11.49% coverage)
**Risk**: Tool discovery and ranking
**Estimated Effort**: 24 hours
**Impact**: CRITICAL - Agent intelligence depends on this

**Required Tests**:
```python
# test_tool_selection_coordinator.py
class TestToolSelectionCoordinator:
    async def test_select_tools_keyword(self):
        """Test keyword-based selection"""

    async def test_select_tools_semantic(self):
        """Test semantic selection"""

    async def test_select_tools_hybrid(self):
        """Test hybrid selection"""

    async def test_select_tools_no_matches(self):
        """Test handling when no tools match"""

    async def test_select_tools_cached(self):
        """Test selection caching"""

    async def test_rank_tools_by_relevance(self):
        """Test ranking by relevance score"""

    async def test_rank_tools_by_priority(self):
        """Test ranking by tool priority"""

    async def test_rank_tools_with_budget(self):
        """Test ranking with budget constraints"""

    async def test_filter_tools_enabled(self):
        """Test filtering by enabled status"""

    async def test_filter_tools_budget(self):
        """Test filtering by budget"""

    async def test_filter_tools_mode(self):
        """Test filtering by agent mode"""
```

#### 7. ProviderCoordinator (26.42% coverage)
**Risk**: Provider switching and management
**Estimated Effort**: 16 hours
**Impact**: HIGH - Multi-provider support is a key feature

**Required Tests**:
```python
# test_provider_coordinator.py
class TestProviderCoordinator:
    def test_switch_provider_success(self):
        """Test successful provider switch"""

    def test_switch_provider_same_provider(self):
        """Test switching to same provider is no-op"""

    def test_switch_provider_invalid(self):
        """Test handling of invalid provider"""

    def test_switch_provider_preserves_context(self):
        """Test context preservation across switch"""

    def test_get_provider_current(self):
        """Test getting current provider"""

    def test_get_provider_by_name(self):
        """Test getting specific provider"""

    def test_get_provider_not_found(self):
        """Test handling when provider not found"""

    def test_list_providers(self):
        """Test listing all providers"""

    def test_list_providers_filtered(self):
        """Test listing providers with filters"""
```

### MEDIUM Priority (Important but Lower Risk)

#### 8. ToolBudgetCoordinator (36.36% coverage)
**Estimated Effort**: 12 hours

**Required Tests**:
```python
class TestToolBudgetCoordinator:
    def test_get_budget_status_within_budget(self):
        """Test budget status when within limits"""

    def test_get_budget_status_exceeded(self):
        """Test budget status when exceeded"""

    def test_get_budget_status_warning(self):
        """Test budget status at warning threshold"""

    def test_record_tool_usage_free(self):
        """Test recording free tool usage"""

    def test_record_tool_usage_low(self):
        """Test recording low cost tool usage"""

    def test_record_tool_usage_high(self):
        """Test recording high cost tool usage"""

    def test_can_execute_tool_true(self):
        """Test tool execution allowed"""

    def test_can_execute_tool_false_budget(self):
        """Test tool execution blocked by budget"""

    def test_can_execute_tool_false_disabled(self):
        """Test tool execution blocked by disabled status"""
```

#### 9. ToolAccessCoordinator (26.61% coverage)
**Estimated Effort**: 12 hours

**Required Tests**:
```python
class TestToolAccessCoordinator:
    def test_is_tool_enabled_true(self):
        """Test tool is enabled"""

    def test_is_tool_enabled_false(self):
        """Test tool is disabled"""

    def test_is_tool_enabled_disabled(self):
        """Test checking disabled tool"""

    def test_enable_tool_success(self):
        """Test enabling a tool"""

    def test_enable_tool_already_enabled(self):
        """Test enabling already enabled tool"""

    def test_enable_tool_disabled(self):
        """Test enabling disabled tool"""

    def test_disable_tool_success(self):
        """Test disabling a tool"""

    def test_disable_tool_already_disabled(self):
        """Test disabling already disabled tool"""

    def test_disable_tool_required(self):
        """Test disabling required tool"""
```

#### 10. ToolExecutionCoordinator (22.46% coverage)
**Estimated Effort**: 16 hours

**Required Tests**:
```python
class TestToolExecutionCoordinator:
    async def test_execute_tool_call_success(self):
        """Test successful tool execution"""

    async def test_execute_tool_call_not_found(self):
        """Test handling of unknown tool"""

    async def test_execute_tool_call_exception(self):
        """Test handling of tool exceptions"""

    async def test_execute_parallel_success(self):
        """Test parallel execution success"""

    async def test_execute_parallel_partial_failure(self):
        """Test parallel execution with partial failures"""

    async def test_execute_parallel_all_fail(self):
        """Test parallel execution with all failures"""

    async def test_handle_error_retry(self):
        """Test error handling with retry"""

    async def test_handle_error_fatal(self):
        """Test handling of fatal errors"""

    async def test_handle_error_user_cancel(self):
        """Test handling of user cancellation"""
```

#### 11. ToolAliasResolver (31.25% coverage)
**Estimated Effort**: 12 hours

**Required Tests**:
```python
class TestToolAliasResolver:
    def test_resolve_alias_canonical(self):
        """Test resolving canonical name"""

    def test_resolve_alias_valid(self):
        """Test resolving valid alias"""

    def test_resolve_alias_invalid(self):
        """Test resolving invalid alias"""

    def test_resolve_alias_recursive(self):
        """Test resolving recursive alias"""

    def test_add_alias_simple(self):
        """Test adding simple alias"""

    def test_add_alias_chain(self):
        """Test adding alias chain"""

    def test_add_alias_cycle(self):
        """Test handling of alias cycles"""

    def test_remove_alias_success(self):
        """Test removing alias"""

    def test_remove_alias_not_found(self):
        """Test removing non-existent alias"""
```

### LOW Priority (Nice to Have)
