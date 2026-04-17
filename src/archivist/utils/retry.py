"""Reusable synchronous retry utility for Archivist.

Provides a single ``retry_call`` function (and a ``@retry`` decorator factory)
that handles both Qdrant and SQLite retry patterns through a unified interface.

Design notes
------------
A true Python decorator (applied at class/function-definition time) cannot
accept per-call mutable state such as a ``failed_steps`` list that is created
fresh for each request.  The canonical pattern here is therefore
``retry_call(fn, *args, **kwargs)`` — a plain function call — which receives
runtime arguments including the mutable ``failed_steps`` list.

The ``@retry`` factory is provided for cases where ``failed_steps`` is *not*
needed (e.g. the SQLite batch helpers in ``graph.py``).  For cascade helpers
that track ``failed_steps`` use ``retry_call`` directly.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger("archivist.retry")

T = TypeVar("T")


def retry_call(
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = 2,
    delay: float = 0.5,
    backoff: float = 1.0,
    catch: type[Exception] | tuple[type[Exception], ...] = Exception,
    is_transient: Callable[[Exception], bool] | None = None,
    step_name: str = "",
    failed_steps: list[str] | None = None,
    reraise: bool = True,
    **kwargs: Any,
) -> T:
    """Call *fn* with *args*/*kwargs*, retrying on eligible exceptions.

    Args:
        fn: The callable to invoke.
        *args: Positional arguments forwarded to *fn*.
        max_attempts: Maximum number of total attempts (default 2 → one retry).
        delay: Sleep duration (seconds) before the first retry.
        backoff: Multiplier applied to *delay* for each subsequent retry.
            ``1.0`` gives a constant delay; ``2.0`` gives exponential back-off.
        catch: Exception type(s) to intercept.  Other exceptions propagate
            immediately without counting as an attempt.
        is_transient: Optional predicate.  When provided, the exception is only
            retried when ``is_transient(exc)`` returns ``True``.  A permanent
            exception stops retrying immediately regardless of remaining
            attempts.
        step_name: When non-empty and *failed_steps* is provided, appended to
            *failed_steps* on final failure instead of re-raising.
        failed_steps: Mutable list to record step failures.  If ``None`` and
            *reraise* is ``True``, the last exception is re-raised on
            exhaustion.
        reraise: Whether to re-raise on exhaustion when *failed_steps* is
            ``None`` (default ``True``).
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn* on success.

    Raises:
        The last caught exception when *reraise* is ``True`` and *failed_steps*
        is ``None`` and all attempts are exhausted.
    """
    last_err: Exception | None = None
    current_delay = delay

    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except catch as exc:
            last_err = exc
            is_last = attempt >= max_attempts - 1
            if is_transient is not None and not is_transient(exc):
                # Permanent error — stop immediately.
                break
            if is_last:
                break
            logger.debug(
                "retry_call: %s attempt %d/%d failed (%s), retrying in %.2fs",
                step_name or fn.__name__,
                attempt + 1,
                max_attempts,
                exc,
                current_delay,
            )
            time.sleep(current_delay)
            current_delay *= backoff

    # All attempts exhausted (or permanent error hit).
    if step_name and failed_steps is not None:
        failed_steps.append(step_name)
        return None  # type: ignore[return-value]

    if reraise and last_err is not None:
        raise last_err

    return None  # type: ignore[return-value]


def retry(
    *,
    max_attempts: int = 2,
    delay: float = 0.5,
    backoff: float = 1.0,
    catch: type[Exception] | tuple[type[Exception], ...] = Exception,
    is_transient: Callable[[Exception], bool] | None = None,
    reraise: bool = True,
) -> Callable:
    """Decorator factory for retry logic without per-call mutable state.

    Use this decorator for functions where failure should re-raise (the default)
    or silently return ``None``.  For cascade helpers that need to populate a
    per-request ``failed_steps`` list, use ``retry_call`` directly.

    Example — constant-delay SQLite retry::

        @retry(max_attempts=2, delay=0.2, catch=sqlite3.OperationalError)
        def _run_batch():
            with GRAPH_WRITE_LOCK:
                ...

    Example — exponential back-off for an HTTP call::

        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def _fetch():
            return requests.get(url)
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return retry_call(
                fn,
                *args,
                max_attempts=max_attempts,
                delay=delay,
                backoff=backoff,
                catch=catch,
                is_transient=is_transient,
                reraise=reraise,
                **kwargs,
            )

        return wrapper

    return decorator
