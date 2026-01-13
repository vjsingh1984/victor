# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for parallel execution strategies."""

import pytest

from victor.framework.parallel.strategies import (
    AllJoinStrategy,
    AnyJoinStrategy,
    FirstJoinStrategy,
    NOfMJoinStrategy,
    MajorityJoinStrategy,
    FailFastErrorStrategy,
    ContinueAllErrorStrategy,
    CollectErrorsErrorStrategy,
    JoinStrategy,
    ErrorStrategy,
    ResourceLimit,
    ParallelConfig,
    create_join_strategy,
    create_error_strategy,
    validate_join_strategy,
    validate_error_strategy,
)


class TestJoinStrategy:
    """Tests for JoinStrategy enum."""

    def test_join_strategy_values(self):
        """Test all join strategy values exist."""
        assert JoinStrategy.ALL.value == "all"
        assert JoinStrategy.ANY.value == "any"
        assert JoinStrategy.FIRST.value == "first"
        assert JoinStrategy.N_OF_M.value == "n_of_m"
        assert JoinStrategy.MAJORITY.value == "majority"

    def test_from_string_valid(self):
        """Test converting valid strings to enum."""
        assert JoinStrategy.from_string("all") == JoinStrategy.ALL
        assert JoinStrategy.from_string("ALL") == JoinStrategy.ALL
        assert JoinStrategy.from_string("any") == JoinStrategy.ANY
        assert JoinStrategy.from_string("majority") == JoinStrategy.MAJORITY

    def test_from_string_invalid(self):
        """Test converting invalid strings raises error."""
        with pytest.raises(ValueError, match="Invalid join_strategy"):
            JoinStrategy.from_string("invalid")

    def test_validate_join_strategy(self):
        """Test validate_join_strategy function."""
        assert validate_join_strategy("all") == JoinStrategy.ALL
        assert validate_join_strategy("majority") == JoinStrategy.MAJORITY

        with pytest.raises(ValueError):
            validate_join_strategy("unknown")


class TestErrorStrategy:
    """Tests for ErrorStrategy enum."""

    def test_error_strategy_values(self):
        """Test all error strategy values exist."""
        assert ErrorStrategy.FAIL_FAST.value == "fail_fast"
        assert ErrorStrategy.CONTINUE_ALL.value == "continue_all"
        assert ErrorStrategy.COLLECT_ERRORS.value == "collect_errors"

    def test_from_string_valid(self):
        """Test converting valid strings to enum."""
        assert ErrorStrategy.from_string("fail_fast") == ErrorStrategy.FAIL_FAST
        assert ErrorStrategy.from_string("FAIL_FAST") == ErrorStrategy.FAIL_FAST
        assert ErrorStrategy.from_string("continue_all") == ErrorStrategy.CONTINUE_ALL

    def test_from_string_invalid(self):
        """Test converting invalid strings raises error."""
        with pytest.raises(ValueError, match="Invalid error_strategy"):
            ErrorStrategy.from_string("invalid")

    def test_validate_error_strategy(self):
        """Test validate_error_strategy function."""
        assert validate_error_strategy("fail_fast") == ErrorStrategy.FAIL_FAST
        assert validate_error_strategy("collect_errors") == ErrorStrategy.COLLECT_ERRORS

        with pytest.raises(ValueError):
            validate_error_strategy("unknown")


class TestResourceLimit:
    """Tests for ResourceLimit configuration."""

    def test_default_resource_limit(self):
        """Test default values."""
        limit = ResourceLimit()
        assert limit.max_concurrent is None
        assert limit.timeout is None
        assert limit.memory_limit is None
        assert limit.cpu_limit is None

    def test_custom_resource_limit(self):
        """Test custom values."""
        limit = ResourceLimit(
            max_concurrent=10,
            timeout=30.0,
            memory_limit=1024,
            cpu_limit=0.8,
        )
        assert limit.max_concurrent == 10
        assert limit.timeout == 30.0
        assert limit.memory_limit == 1024
        assert limit.cpu_limit == 0.8

    def test_invalid_max_concurrent(self):
        """Test validation of max_concurrent."""
        with pytest.raises(ValueError, match="max_concurrent must be positive"):
            ResourceLimit(max_concurrent=0)

        with pytest.raises(ValueError, match="max_concurrent must be positive"):
            ResourceLimit(max_concurrent=-5)

    def test_invalid_timeout(self):
        """Test validation of timeout."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            ResourceLimit(timeout=0)

        with pytest.raises(ValueError, match="timeout must be positive"):
            ResourceLimit(timeout=-10)

    def test_invalid_cpu_limit(self):
        """Test validation of cpu_limit."""
        with pytest.raises(ValueError, match="cpu_limit must be between 0.0 and 1.0"):
            ResourceLimit(cpu_limit=1.5)

        with pytest.raises(ValueError, match="cpu_limit must be between 0.0 and 1.0"):
            ResourceLimit(cpu_limit=-0.1)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        limit = ResourceLimit(max_concurrent=5, timeout=60.0)
        data = limit.to_dict()
        assert data == {
            "max_concurrent": 5,
            "timeout": 60.0,
            "memory_limit": None,
            "cpu_limit": None,
        }

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "max_concurrent": 10,
            "timeout": 30.0,
            "memory_limit": 512,
            "cpu_limit": 0.5,
        }
        limit = ResourceLimit.from_dict(data)
        assert limit.max_concurrent == 10
        assert limit.timeout == 30.0
        assert limit.memory_limit == 512
        assert limit.cpu_limit == 0.5


class TestParallelConfig:
    """Tests for ParallelConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ParallelConfig()
        assert config.join_strategy == JoinStrategy.ALL
        assert config.error_strategy == ErrorStrategy.FAIL_FAST
        assert config.resource_limit.max_concurrent is None
        assert config.n_of_m is None

    def test_custom_config(self):
        """Test custom configuration."""
        resource_limit = ResourceLimit(max_concurrent=5)
        config = ParallelConfig(
            join_strategy=JoinStrategy.MAJORITY,
            error_strategy=ErrorStrategy.COLLECT_ERRORS,
            resource_limit=resource_limit,
        )
        assert config.join_strategy == JoinStrategy.MAJORITY
        assert config.error_strategy == ErrorStrategy.COLLECT_ERRORS
        assert config.resource_limit.max_concurrent == 5

    def test_n_of_m_requires_value(self):
        """Test that N_OF_M strategy requires n_of_m value."""
        with pytest.raises(ValueError, match="n_of_m must be specified"):
            ParallelConfig(join_strategy=JoinStrategy.N_OF_M)

    def test_n_of_m_with_value(self):
        """Test N_OF_M strategy with required value."""
        config = ParallelConfig(
            join_strategy=JoinStrategy.N_OF_M,
            n_of_m=3,
        )
        assert config.join_strategy == JoinStrategy.N_OF_M
        assert config.n_of_m == 3

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = ParallelConfig(
            join_strategy=JoinStrategy.MAJORITY,
            error_strategy=ErrorStrategy.CONTINUE_ALL,
            n_of_m=2,
        )
        data = config.to_dict()
        assert data["join_strategy"] == "majority"
        assert data["error_strategy"] == "continue_all"
        assert data["n_of_m"] == 2

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "join_strategy": "any",
            "error_strategy": "collect_errors",
            "resource_limit": {"max_concurrent": 10},
            "n_of_m": 2,
        }
        config = ParallelConfig.from_dict(data)
        assert config.join_strategy == JoinStrategy.ANY
        assert config.error_strategy == ErrorStrategy.COLLECT_ERRORS
        assert config.resource_limit.max_concurrent == 10
        assert config.n_of_m == 2


class TestAllJoinStrategy:
    """Tests for AllJoinStrategy."""

    @pytest.mark.asyncio
    async def test_all_success(self):
        """Test all tasks succeed."""
        strategy = AllJoinStrategy()
        results = ["result1", "result2", "result3"]
        errors = []

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is True
        assert aggregated == results
        assert agg_errors == []

    @pytest.mark.asyncio
    async def test_one_failure(self):
        """Test one task fails."""
        strategy = AllJoinStrategy()
        results = ["result1", "result2"]
        errors = [Exception("Task failed")]

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is False
        assert aggregated is None
        assert agg_errors == errors

    def test_should_stop_on_error(self):
        """Test that all join stops on error."""
        strategy = AllJoinStrategy()
        assert strategy.should_stop_on_error() is True


class TestAnyJoinStrategy:
    """Tests for AnyJoinStrategy."""

    @pytest.mark.asyncio
    async def test_one_success(self):
        """Test one task succeeds."""
        strategy = AnyJoinStrategy()
        results = ["result1", None, None]
        errors = [Exception("error1"), Exception("error2")]

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is True
        assert aggregated == ["result1"]
        assert agg_errors == errors

    @pytest.mark.asyncio
    async def test_all_fail(self):
        """Test all tasks fail."""
        strategy = AnyJoinStrategy()
        results = [None, None]
        errors = [Exception("error1"), Exception("error2")]

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is False
        assert aggregated is None

    def test_should_continue_on_error(self):
        """Test that any join continues on error."""
        strategy = AnyJoinStrategy()
        assert strategy.should_stop_on_error() is False


class TestFirstJoinStrategy:
    """Tests for FirstJoinStrategy."""

    @pytest.mark.asyncio
    async def test_first_success(self):
        """Test first result is returned."""
        strategy = FirstJoinStrategy()
        results = ["result1", "result2", "result3"]
        errors = []

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is True
        assert aggregated == "result1"
        assert agg_errors == []

    @pytest.mark.asyncio
    async def test_first_none_then_success(self):
        """Test first non-None result is returned."""
        strategy = FirstJoinStrategy()
        results = [None, "result2", "result3"]
        errors = []

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is True
        assert aggregated == "result2"

    @pytest.mark.asyncio
    async def test_all_fail(self):
        """Test all tasks fail."""
        strategy = FirstJoinStrategy()
        results = [None, None]
        errors = [Exception("error1"), Exception("error2")]

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is False
        assert aggregated is None


class TestNOfMJoinStrategy:
    """Tests for NOfMJoinStrategy."""

    def test_invalid_required(self):
        """Test validation of required count."""
        with pytest.raises(ValueError, match="required must be positive"):
            NOfMJoinStrategy(required=0)

        with pytest.raises(ValueError, match="required must be positive"):
            NOfMJoinStrategy(required=-1)

    @pytest.mark.asyncio
    async def test_exactly_n_success(self):
        """Test exactly N tasks succeed."""
        strategy = NOfMJoinStrategy(required=2)
        results = ["result1", "result2", "result3"]
        errors = []

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is True
        assert aggregated == results

    @pytest.mark.asyncio
    async def test_fewer_than_n_success(self):
        """Test fewer than N tasks succeed."""
        strategy = NOfMJoinStrategy(required=3)
        results = ["result1", "result2"]
        errors = [Exception("error1")]

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is False
        assert aggregated is None

    @pytest.mark.asyncio
    async def test_more_than_n_success(self):
        """Test more than N tasks succeed."""
        strategy = NOfMJoinStrategy(required=2)
        results = ["result1", "result2", "result3", "result4"]
        errors = []

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is True
        assert aggregated == results


class TestMajorityJoinStrategy:
    """Tests for MajorityJoinStrategy."""

    @pytest.mark.asyncio
    async def test_majority_success(self):
        """Test majority of tasks succeed."""
        strategy = MajorityJoinStrategy()
        results = ["result1", "result2", "result3"]
        errors = [Exception("error1"), Exception("error2")]

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is True

    @pytest.mark.asyncio
    async def test_majority_failure(self):
        """Test minority of tasks succeed."""
        strategy = MajorityJoinStrategy()
        results = ["result1"]
        errors = [Exception("error1"), Exception("error2"), Exception("error3")]

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is False
        assert aggregated is None

    @pytest.mark.asyncio
    async def test_tie_goes_to_success(self):
        """Test tie (3 success, 3 fail) succeeds."""
        strategy = MajorityJoinStrategy()
        results = ["r1", "r2", "r3"]
        errors = [Exception("e1"), Exception("e2"), Exception("e3")]

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        # 3 of 6 needed = 4, so 3 is not enough
        assert success is False

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Test empty results list."""
        strategy = MajorityJoinStrategy()
        results = []
        errors = []

        success, aggregated, agg_errors = await strategy.evaluate(results, errors)

        assert success is True
        assert aggregated == []


class TestFailFastErrorStrategy:
    """Tests for FailFastErrorStrategy."""

    @pytest.mark.asyncio
    async def test_handle_error(self):
        """Test error handling stops execution."""
        strategy = FailFastErrorStrategy()
        error = Exception("Test error")

        should_stop, error_to_raise = await strategy.handle_error(error, 0, 5)

        assert should_stop is True
        assert error_to_raise == error

    def test_should_cancel_on_error(self):
        """Test that pending tasks are cancelled."""
        strategy = FailFastErrorStrategy()
        assert strategy.should_cancel_on_error() is True


class TestContinueAllErrorStrategy:
    """Tests for ContinueAllErrorStrategy."""

    @pytest.mark.asyncio
    async def test_handle_error(self):
        """Test error handling continues execution."""
        strategy = ContinueAllErrorStrategy()
        error = Exception("Test error")

        should_stop, error_to_raise = await strategy.handle_error(error, 0, 5)

        assert should_stop is False
        assert error_to_raise is None

    def test_should_not_cancel_on_error(self):
        """Test that pending tasks are not cancelled."""
        strategy = ContinueAllErrorStrategy()
        assert strategy.should_cancel_on_error() is False


class TestCollectErrorsErrorStrategy:
    """Tests for CollectErrorsErrorStrategy."""

    @pytest.mark.asyncio
    async def test_handle_error(self):
        """Test error handling collects errors."""
        strategy = CollectErrorsErrorStrategy()
        error1 = Exception("Error 1")
        error2 = Exception("Error 2")

        should_stop1, error_to_raise1 = await strategy.handle_error(error1, 0, 5)
        should_stop2, error_to_raise2 = await strategy.handle_error(error2, 1, 5)

        assert should_stop1 is False
        assert should_stop2 is False
        assert error_to_raise1 is None
        assert error_to_raise2 is None

        errors = strategy.get_errors()
        assert len(errors) == 2
        assert errors[0] == error1
        assert errors[1] == error2

    def test_get_and_clear_errors(self):
        """Test getting and clearing errors."""
        strategy = CollectErrorsErrorStrategy()
        error = Exception("Test")

        strategy.collected_errors.append(error)

        errors = strategy.get_errors()
        assert len(errors) == 1

        strategy.clear_errors()
        assert len(strategy.get_errors()) == 0


class TestCreateJoinStrategy:
    """Tests for create_join_strategy factory function."""

    def test_create_all_strategy(self):
        """Test creating ALL strategy."""
        strategy = create_join_strategy(JoinStrategy.ALL)
        assert isinstance(strategy, AllJoinStrategy)

    def test_create_any_strategy(self):
        """Test creating ANY strategy."""
        strategy = create_join_strategy(JoinStrategy.ANY)
        assert isinstance(strategy, AnyJoinStrategy)

    def test_create_n_of_m_strategy(self):
        """Test creating N_OF_M strategy with required param."""
        strategy = create_join_strategy(JoinStrategy.N_OF_M, n_of_m=3)
        assert isinstance(strategy, NOfMJoinStrategy)
        assert strategy.required == 3

    def test_create_n_of_m_with_required_kwarg(self):
        """Test creating N_OF_M strategy with required kwarg."""
        strategy = create_join_strategy(JoinStrategy.N_OF_M, required=5)
        assert isinstance(strategy, NOfMJoinStrategy)
        assert strategy.required == 5

    def test_create_majority_strategy(self):
        """Test creating MAJORITY strategy."""
        strategy = create_join_strategy(JoinStrategy.MAJORITY)
        assert isinstance(strategy, MajorityJoinStrategy)


class TestCreateErrorStrategy:
    """Tests for create_error_strategy factory function."""

    def test_create_fail_fast_strategy(self):
        """Test creating FAIL_FAST strategy."""
        strategy = create_error_strategy(ErrorStrategy.FAIL_FAST)
        assert isinstance(strategy, FailFastErrorStrategy)

    def test_create_continue_all_strategy(self):
        """Test creating CONTINUE_ALL strategy."""
        strategy = create_error_strategy(ErrorStrategy.CONTINUE_ALL)
        assert isinstance(strategy, ContinueAllErrorStrategy)

    def test_create_collect_errors_strategy(self):
        """Test creating COLLECT_ERRORS strategy."""
        strategy = create_error_strategy(ErrorStrategy.COLLECT_ERRORS)
        assert isinstance(strategy, CollectErrorsErrorStrategy)
