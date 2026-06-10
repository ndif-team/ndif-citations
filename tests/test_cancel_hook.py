"""Tests for the per-thread cancellation hook (events.set_cancel_check) that
rate_limit_sleep honors — so a cancelled run aborts inside ANY rate-limited
loop (discover / enrich-papers / enrich-repos), not only in process_stage.
"""
import pytest

from ndif_citations import events, utils
from ndif_citations.events import RunCancelled


def teardown_function():
    # Never leak a cancel hook into other tests on this thread.
    events.clear_cancel_check()


def test_raise_if_cancelled_noop_when_unset():
    events.clear_cancel_check()
    events.raise_if_cancelled()  # must not raise


def test_rate_limit_sleep_aborts_when_cancelled():
    events.set_cancel_check(lambda: True)
    with pytest.raises(RunCancelled):
        utils.rate_limit_sleep(0.5, "GitHub API")  # raises BEFORE sleeping


def test_runcancelled_is_not_swallowed_by_except_exception():
    """REGRESSION: the enrich loops wrap rate_limit_sleep in `except Exception`.
    RunCancelled must subclass BaseException so it propagates through them
    instead of being silently swallowed (the 'cancel never cancels' bug)."""
    assert issubclass(RunCancelled, BaseException)
    assert not issubclass(RunCancelled, Exception)

    events.set_cancel_check(lambda: True)
    swallowed = False
    try:
        try:
            utils.rate_limit_sleep(0.3, "OpenReview")
        except Exception:          # mirrors extract.py's per-API guard
            swallowed = True
    except RunCancelled:
        pass                        # propagated past `except Exception` — correct
    assert swallowed is False, "RunCancelled was swallowed by `except Exception`"


def test_rate_limit_sleep_runs_when_not_cancelled():
    events.set_cancel_check(lambda: False)
    utils.rate_limit_sleep(0, "OpenAlex")  # 0s + not cancelled → no raise, no sleep


def test_cancel_hook_is_thread_local():
    """A cancel hook set on this thread must NOT leak into a worker thread."""
    import threading

    events.set_cancel_check(lambda: True)
    seen = {}

    def worker():
        try:
            events.raise_if_cancelled()   # unset on this fresh thread → no raise
            seen["raised"] = False
        except RunCancelled:
            seen["raised"] = True

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["raised"] is False
