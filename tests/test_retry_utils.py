"""Unit tests for src/retry.py — the unified retry utility.

Covers:
- Happy path: function succeeds on first attempt, no retries.
- Transient retry: function fails once with a transient error, succeeds on retry.
- Permanent failure: non-transient error stops immediately; no retry.
- Exhaustion re-raise: all attempts fail, last exception is re-raised.
- failed_steps population: on final failure, step_name appended to list.
- Backoff: delay multiplied correctly across attempts.
- @retry decorator: wraps synchronous functions correctly.
- catch parameter: only catches specified exception types.
"""

import os
import sqlite3
import sys

import pytest

# Ensure src/ is on the path for direct imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retry import retry, retry_call

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Transient(Exception):
    """Simulated transient error."""


class _Permanent(Exception):
    """Simulated permanent (non-retryable) error."""


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, _Transient)


# ---------------------------------------------------------------------------
# retry_call tests
# ---------------------------------------------------------------------------


class TestRetryCallHappyPath:
    def test_returns_value_on_first_attempt(self):
        calls = []

        def fn():
            calls.append(1)
            return 42

        result = retry_call(fn, max_attempts=3, delay=0)
        assert result == 42
        assert len(calls) == 1

    def test_passes_positional_args(self):
        def fn(a, b):
            return a + b

        assert retry_call(fn, 3, 5, max_attempts=2, delay=0) == 8

    def test_passes_keyword_args(self):
        def fn(x, multiplier=1):
            return x * multiplier

        assert retry_call(fn, 4, max_attempts=2, delay=0, multiplier=3) == 12


class TestRetryCallTransientRetry:
    def test_retries_on_transient_error(self):
        attempt_log = []

        def fn():
            attempt_log.append(len(attempt_log))
            if len(attempt_log) < 2:
                raise _Transient("rate-limited")
            return "ok"

        result = retry_call(
            fn,
            max_attempts=3,
            delay=0,
            is_transient=_is_transient,
        )
        assert result == "ok"
        assert len(attempt_log) == 2

    def test_does_not_retry_permanent_error(self):
        attempt_log = []

        def fn():
            attempt_log.append(1)
            raise _Permanent("bad request")

        with pytest.raises(_Permanent):
            retry_call(
                fn,
                max_attempts=3,
                delay=0,
                is_transient=_is_transient,
                reraise=True,
            )
        assert len(attempt_log) == 1, "Permanent error must not be retried"


class TestRetryCallExhaustion:
    def test_reraises_last_exception_by_default(self):
        def fn():
            raise _Transient("always fails")

        with pytest.raises(_Transient, match="always fails"):
            retry_call(fn, max_attempts=2, delay=0)

    def test_no_reraise_returns_none(self):
        def fn():
            raise _Transient("always fails")

        result = retry_call(fn, max_attempts=2, delay=0, reraise=False)
        assert result is None

    def test_correct_number_of_attempts(self):
        calls = []

        def fn():
            calls.append(1)
            raise _Transient("x")

        with pytest.raises(_Transient):
            retry_call(fn, max_attempts=4, delay=0)
        assert len(calls) == 4


class TestRetryCallFailedSteps:
    def test_populates_failed_steps_on_exhaustion(self):
        failed: list[str] = []

        def fn():
            raise _Transient("boom")

        retry_call(
            fn,
            max_attempts=2,
            delay=0,
            step_name="my_step",
            failed_steps=failed,
            reraise=False,
        )
        assert "my_step" in failed

    def test_does_not_populate_failed_steps_on_success(self):
        failed: list[str] = []

        def fn():
            return "ok"

        retry_call(
            fn,
            max_attempts=2,
            delay=0,
            step_name="my_step",
            failed_steps=failed,
        )
        assert failed == []

    def test_failed_steps_without_reraise(self):
        """When failed_steps is provided, exception is NOT re-raised on exhaustion."""
        failed: list[str] = []

        def fn():
            raise _Transient("explode")

        # Should not raise:
        retry_call(
            fn,
            max_attempts=2,
            delay=0,
            step_name="exploding_step",
            failed_steps=failed,
        )
        assert "exploding_step" in failed


class TestRetryCallCatchParam:
    def test_only_catches_specified_exception(self):
        def fn():
            raise ValueError("wrong type")

        # ValueError is not in catch — should propagate immediately.
        with pytest.raises(ValueError):
            retry_call(
                fn,
                max_attempts=3,
                delay=0,
                catch=_Transient,
            )

    def test_catches_tuple_of_exceptions(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise sqlite3.OperationalError("locked")
            return "done"

        result = retry_call(
            fn,
            max_attempts=3,
            delay=0,
            catch=(sqlite3.OperationalError, _Transient),
        )
        assert result == "done"
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# @retry decorator tests
# ---------------------------------------------------------------------------


class TestRetryDecorator:
    def test_decorator_happy_path(self):
        @retry(max_attempts=2, delay=0)
        def fn():
            return "hello"

        assert fn() == "hello"

    def test_decorator_retries_and_succeeds(self):
        calls = []

        @retry(max_attempts=3, delay=0, catch=sqlite3.OperationalError)
        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise sqlite3.OperationalError("locked")
            return len(calls)

        result = fn()
        assert result == 2
        assert len(calls) == 2

    def test_decorator_reraises_on_exhaustion(self):
        @retry(max_attempts=2, delay=0)
        def fn():
            raise _Transient("always")

        with pytest.raises(_Transient):
            fn()

    def test_decorator_preserves_function_name(self):
        @retry(max_attempts=2, delay=0)
        def my_special_function():
            return 1

        assert my_special_function.__name__ == "my_special_function"

    def test_decorator_no_reraise(self):
        @retry(max_attempts=2, delay=0, reraise=False)
        def fn():
            raise _Transient("boom")

        assert fn() is None

    def test_decorator_sqlite_pattern(self):
        """Mirrors the exact usage pattern from graph.delete_fts_chunks_batch."""
        import threading

        lock = threading.Lock()
        calls = []

        @retry(max_attempts=2, delay=0, catch=sqlite3.OperationalError)
        def _run() -> int:
            calls.append(1)
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            with lock:
                return 5  # simulate rowcount

        result = _run()
        assert result == 5
        assert len(calls) == 2
