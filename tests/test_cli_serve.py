"""Tests for the `ndif-citations serve` CLI command (Task 2.6).

Patching strategy
-----------------
- ``uvicorn.run``: the command does ``import uvicorn; uvicorn.run(...)``, so we
  patch the attribute on the already-imported module via
  ``monkeypatch.setattr("uvicorn.run", spy)``.  This works because Python caches
  module objects; the lazy ``import uvicorn`` inside the command will retrieve the
  same module object that monkeypatch already modified.
- ``webbrowser.open``: same approach — ``monkeypatch.setattr("webbrowser.open", spy)``.
- ``threading.Timer``: replaced with a no-op fake that records the interval and
  callback so we can assert the browser-open path without starting a real timer or
  a real browser.

No real server starts, no real browser opens, no live timer thread leaks.
"""
from __future__ import annotations

import uvicorn  # import first so monkeypatch can target the module attribute
import webbrowser  # same

from click.testing import CliRunner

from ndif_citations.cli import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeTimer:
    """Minimal threading.Timer stand-in: records args, never starts a thread."""

    _instances: list["_FakeTimer"] = []

    def __init__(self, interval: float, function, args=None, kwargs=None):
        self.interval = interval
        self.function = function
        self.daemon = False
        _FakeTimer._instances.append(self)

    def start(self):
        pass  # intentionally does nothing — no real thread

    @classmethod
    def reset(cls):
        cls._instances.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestServeCLI:
    def test_serve_invokes_uvicorn_with_localhost(self, monkeypatch):
        """serve --no-open --port 9999 → uvicorn.run called with the right args."""
        calls = []

        def fake_uvicorn_run(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

        result = CliRunner().invoke(cli, ["serve", "--no-open", "--port", "9999"])
        assert result.exit_code == 0, result.output

        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] == "ndif_citations.server.app:app"
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 9999

    def test_serve_custom_host_port(self, monkeypatch):
        """serve --no-open --host 127.0.0.1 --port 8001 → correct host/port forwarded."""
        calls = []

        def fake_uvicorn_run(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

        result = CliRunner().invoke(cli, ["serve", "--no-open", "--host", "127.0.0.1", "--port", "8001"])
        assert result.exit_code == 0, result.output

        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] == "ndif_citations.server.app:app"
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8001

    def test_serve_opens_browser_by_default(self, monkeypatch):
        """Without --no-open, a Timer is scheduled with the correct URL.

        We patch threading.Timer so no real timer/browser fires, then manually
        invoke the captured callback to confirm it calls webbrowser.open with
        the right URL.
        """
        import threading

        # Reset fake timer registry
        _FakeTimer.reset()

        browser_calls = []

        def fake_browser_open(url):
            browser_calls.append(url)

        def fake_uvicorn_run(*args, **kwargs):
            pass  # don't block

        monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
        monkeypatch.setattr(webbrowser, "open", fake_browser_open)
        monkeypatch.setattr(threading, "Timer", _FakeTimer)

        result = CliRunner().invoke(cli, ["serve", "--port", "8723"])
        assert result.exit_code == 0, result.output

        # A timer should have been created
        assert len(_FakeTimer._instances) == 1
        timer = _FakeTimer._instances[0]
        assert timer.interval == 1.0
        assert timer.daemon is True

        # Invoke the callback manually (the real timer would do this after 1 s)
        timer.function()

        # Confirm webbrowser.open was called with the correct URL
        assert len(browser_calls) == 1
        assert browser_calls[0] == "http://127.0.0.1:8723"

    def test_serve_no_open_skips_browser_timer(self, monkeypatch):
        """With --no-open, no Timer is created at all."""
        import threading

        _FakeTimer.reset()

        def fake_uvicorn_run(*args, **kwargs):
            pass

        monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
        monkeypatch.setattr(threading, "Timer", _FakeTimer)

        result = CliRunner().invoke(cli, ["serve", "--no-open"])
        assert result.exit_code == 0, result.output

        assert len(_FakeTimer._instances) == 0
