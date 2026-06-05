"""In-process, one-run-at-a-time background pipeline runner (Tasks 2.1-2.2).

This is a *local single-user* app, so there is no Celery/Redis: a single
``JobRunner`` instance (created in the server layer later) drives
``orchestrator.run_pipeline`` on a daemon ``threading.Thread`` and captures the
progress-event buffer for that run. At most one run is active at a time, which
is what makes touching the global config / Surya state safe without locking the
pipeline itself.

Scope (Task 2.1):
    * Start a run, track its state ("running" -> "done"|"error").
    * Capture the ProgressEvent buffer for the run.
    * Persist a run record to ``out/runs/<run_id>.json`` (the run history).

Scope (Task 2.2):
    * Cancel a run via ``JobRunner.cancel()``.
    * Each run gets a ``threading.Event`` cancel token forwarded to ``run_pipeline``
      as ``cancel_check``. When ``RunCancelled`` propagates out, the run ends in
      state "cancelled" — nothing is written to disk for the in-flight run.
    * Cancel is stop-and-discard: no partial-merge finalization (deferred to Phase 3).

Scope (Task 2.5):
    * Live SSE fan-out via ``JobRunner.subscribe(run_id)``. Each ``RunRecord`` now
      carries a list of subscriber ``queue.Queue`` objects in addition to the
      ``events`` buffer. The event callback appends to the buffer *and* pushes onto
      every subscriber queue (under the lock). On run end a module-level ``_DONE``
      sentinel is pushed onto each queue so live streams know to stop.
    * ``subscribe`` snapshots the buffer and registers a fresh queue in one
      critical section (so no event is lost or duplicated between snapshot and
      queue), then yields the snapshot followed by live events until the sentinel.
      For a terminal run it replays the buffer only (no queue wait). The queue is
      always unregistered in a ``finally`` so a disconnected client can't leak or
      block ``_append``.

Out of scope (later tasks):
    * The FastAPI streaming layer wires ``subscribe`` to ``StreamingResponse``
      (Task 2.5, server/routers/runs.py).

Threading correctness (critical):
    * ``events.set_sink`` uses ``threading.local`` storage, so it MUST be called
      *inside* the worker thread (``_run``), not in ``start()`` on the caller's
      thread — otherwise the sink would be invisible to the worker that actually
      emits events. It is cleared in the ``finally`` of ``_run``.
    * A single ``threading.Lock`` guards the quick critical sections only:
      the start/active check and each event append + state read/write. The lock
      is NOT held while ``run_pipeline`` executes (that would serialize nothing
      useful and would block ``status()`` reads).
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ndif_citations import events, orchestrator
from ndif_citations.events import ProgressEvent, RunCancelled

logger = logging.getLogger(__name__)

# Sentinel pushed onto every subscriber queue when a run ends, so a live SSE
# stream blocked on ``queue.Queue.get()`` knows to stop without polling state.
_DONE = object()


class RunActiveError(Exception):
    """Raised by ``JobRunner.start`` when a run is already active.

    Only one pipeline run may execute at a time (it owns the global config /
    Surya state). Callers should surface this as a 409-style conflict.
    """


# Fields lifted from ``FinalizeResult.run_stats`` (a ``PipelineRun``) onto
# ``RunRecord.counts`` on success. Kept small and meaningful: merge tallies plus
# the routing-bucket breakdown, which is what a run-history UI cares about.
_COUNT_FIELDS = (
    "total_unique",
    "new_papers",
    "updated_papers",
    "existing_papers",
    "github_dependents_found",
    "bucket_new",
    "bucket_reprocess",
    "bucket_fill_gaps",
    "bucket_skip",
    "bucket_protected",
)


@dataclass
class RunRecord:
    """The state of a single pipeline run.

    ``events`` is the captured progress buffer (the live subscriber/queue for SSE
    is a later task). ``counts`` is filled from ``FinalizeResult.run_stats`` on
    success.

    ``cancel_event`` is a ``threading.Event`` that the caller can set to request
    cancellation. It is NOT serialized to JSON (it's a threading primitive).

    ``_subscribers`` is the live SSE fan-out: a list of ``queue.Queue`` objects,
    one per active ``subscribe`` stream. The event callback pushes each new event
    onto every queue. Like ``cancel_event`` it is NOT serialized to JSON.
    """

    run_id: str
    state: str  # "running" | "done" | "error" | "cancelled"
    mode: str
    skip_papers: bool
    skip_github: bool
    started_at: str  # ISO-8601
    finished_at: str | None = None
    error: str | None = None
    traceback: str | None = None
    counts: dict = field(default_factory=dict)
    events: list[ProgressEvent] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _subscribers: list[queue.Queue] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for persistence; events go through ``ProgressEvent.to_dict``.

        ``cancel_event`` and ``_subscribers`` are intentionally excluded — they
        are threading primitives / live state and must not appear in the
        persisted JSON.
        """
        return {
            "run_id": self.run_id,
            "state": self.state,
            "mode": self.mode,
            "skip_papers": self.skip_papers,
            "skip_github": self.skip_github,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "traceback": self.traceback,
            "counts": self.counts,
            "events": [ev.to_dict() for ev in self.events],
        }


class JobRunner:
    """Drives at most one ``run_pipeline`` at a time on a daemon thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: RunRecord | None = None

    @property
    def active(self) -> bool:
        """True iff a run is currently in state ``"running"``."""
        with self._lock:
            return self._current is not None and self._current.state == "running"

    def start(
        self,
        out: Path,
        *,
        mode: str,
        skip_papers: bool = False,
        skip_github: bool = False,
    ) -> str:
        """Start a pipeline run on a daemon thread; return its ``run_id``.

        Raises ``RunActiveError`` if a run is already active. The active-check and
        the install of the new record happen under the lock so two concurrent
        ``start()`` calls cannot both win.
        """
        # Pre-import openpyxl on the caller's thread to avoid a first-ever lazy
        # import on the worker thread racing against test infrastructure.
        #
        # NOTE — this is a **test-stability aid**, not a production guarantee:
        #   • In the real FastAPI server the main thread is blocked in asyncio's
        #     event loop (select()), so it does not busy-poll and the GIL-release
        #     argument holds — this warm-up is unnecessary there.
        #   • It only covers openpyxl; heavier lazy imports (surya, openai, etc.)
        #     are not pre-warmed and the same reasoning applies to them.
        #   • Empirically, removing it caused test_run_completes_and_persists to
        #     exceed its 2 s timeout when openpyxl had not yet been imported into
        #     sys.modules (cold-start pytest run), even though _wait_until already
        #     uses time.sleep(0.01).  The worker's first-ever openpyxl import takes
        #     ~1 s on this host, pushing total run time past the test deadline.
        #     Restore it here as a best-effort timing guard for the test suite only.
        self._warm_imports()

        with self._lock:
            if self._current is not None and self._current.state == "running":
                raise RunActiveError("a pipeline run is already active")

            run_id = self._new_run_id()
            record = RunRecord(
                run_id=run_id,
                state="running",
                mode=mode,
                skip_papers=skip_papers,
                skip_github=skip_github,
                started_at=datetime.now().isoformat(),
            )
            self._current = record

        thread = threading.Thread(
            target=self._run,
            args=(record, out),
            kwargs={
                "mode": mode,
                "skip_papers": skip_papers,
                "skip_github": skip_github,
            },
            name=f"pipeline-run-{run_id}",
            daemon=True,
        )
        thread.start()
        return run_id

    def status(self, run_id: str | None = None) -> RunRecord:
        """Return the current/most-recent record, or look one up by id.

        Raises ``KeyError`` if no run matches (or none has ever started).
        """
        with self._lock:
            current = self._current
        if current is None:
            raise KeyError("no run has been started")
        if run_id is not None and current.run_id != run_id:
            raise KeyError(run_id)
        return current

    def cancel(self, run_id: str | None = None) -> None:
        """Request cancellation of the active run (or the run identified by *run_id*).

        Sets the run's ``cancel_event`` so that the next ``cancel_check`` call
        inside ``process_papers`` / ``process_repos`` fires and raises
        ``RunCancelled``.

        Safe no-op semantics:
          * If no run has ever started, does nothing.
          * If the identified run is not in state "running" (already done / error /
            cancelled), does nothing — it is not an error to cancel a finished run.
          * The lock is held only for the brief lookup, NOT while waiting for the
            pipeline to actually stop.
        """
        with self._lock:
            current = self._current
        if current is None:
            return
        if run_id is not None and current.run_id != run_id:
            return
        # Only set the event if the run is currently running.
        if current.state == "running":
            current.cancel_event.set()

    def subscribe(self, run_id: str) -> Iterator[ProgressEvent]:
        """Yield this run's events: the buffered prefix, then live events.

        Used by the SSE endpoint. Behaviour:

          * Unknown ``run_id`` → ``KeyError`` (the HTTP layer maps this to 404 or
            a persisted-file replay).
          * In a single critical section (under ``self._lock``) we look up the
            record, snapshot ``list(record.events)``, and — only if the run is
            still ``"running"`` — register a fresh queue in ``record._subscribers``.
            Taking the snapshot and registering atomically guarantees no event is
            lost or duplicated: any event appended *before* registration is in the
            snapshot; any appended *after* goes onto the queue (because ``_append``
            takes the same lock).
          * The generator first yields every snapshot event. If the run was
            already terminal at subscribe time, it stops there (buffer replay only;
            no queue is registered, so there is nothing to wait on).
          * Otherwise it loops on ``q.get()`` (NOT under the lock — never block
            while holding it), stopping on the ``_DONE`` sentinel and yielding any
            real event.
          * A ``finally`` unregisters the queue so a disconnected client neither
            leaks a queue nor blocks ``_append`` forever.
        """
        with self._lock:
            current = self._current
            if current is None or current.run_id != run_id:
                raise KeyError(run_id)
            record = current
            snapshot = list(record.events)
            is_running = record.state == "running"
            q: queue.Queue | None = None
            if is_running:
                q = queue.Queue()
                record._subscribers.append(q)

        try:
            yield from snapshot
            if q is None:
                # Run was already terminal — buffer replay only, no live wait.
                return
            while True:
                item = q.get()
                if item is _DONE:
                    break
                yield item
        finally:
            if q is not None:
                with self._lock:
                    try:
                        record._subscribers.remove(q)
                    except ValueError:
                        pass

    def history(self, out: Path) -> list[dict]:
        """Read every persisted run record from ``out/runs/*.json``.

        Returns the parsed dicts sorted by ``started_at`` (most recent first).
        Malformed/unreadable files are skipped rather than aborting the listing.
        """
        import json

        runs_dir = Path(out) / "runs"
        if not runs_dir.is_dir():
            return []
        records: list[dict] = []
        for path in runs_dir.glob("*.json"):
            try:
                records.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
        records.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return records

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _new_run_id() -> str:
        """``<timestamp>-<short-uuid>`` — sorts chronologically, stays unique."""
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{ts}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _warm_imports() -> None:
        """Pre-import openpyxl to improve test-suite timing stability.

        This is a **test-stability aid only** — not a production guarantee.
        See the comment in ``start()`` for full context.  Best-effort: a
        missing optional dependency must not block starting a run (the
        pipeline will surface the real error on the worker thread).
        """
        try:
            import openpyxl  # noqa: F401  (warm the import cache)
        except Exception:  # noqa: BLE001 — warming is best-effort
            pass

    def _run(
        self,
        record: RunRecord,
        out: Path,
        *,
        mode: str,
        skip_papers: bool,
        skip_github: bool,
    ) -> None:
        """Worker-thread body. Owns the events sink for THIS thread.

        The sink is installed here (not in ``start``) because ``events`` uses
        ``threading.local`` storage — a sink set on the caller's thread would be
        invisible to this worker. It is always cleared in ``finally``.
        """

        def _append(ev: ProgressEvent) -> None:
            # Under the lock: append to the persistent buffer AND fan out to every
            # live subscriber queue. Holding the lock here pairs with subscribe()'s
            # snapshot+register critical section so an event can't be lost (in
            # neither the snapshot nor a queue) or duplicated (in both).
            with self._lock:
                record.events.append(ev)
                for q in record._subscribers:
                    q.put(ev)

        events.set_sink(_append)
        # Compute the terminal outcome first, but DON'T flip the visible state
        # to a terminal value yet. The state is flipped to "done"/"error" only
        # in the `finally`, *after* the record is persisted — so an observer that
        # sees a terminal state via ``status()`` is guaranteed the run file on
        # disk already exists (no read-your-write race for callers).
        terminal_state = "done"
        try:
            result = orchestrator.run_pipeline(
                out,
                mode=mode,
                skip_papers=skip_papers,
                skip_github=skip_github,
                cancel_check=record.cancel_event.is_set,
            )
            counts = self._extract_counts(result)
            with self._lock:
                record.counts = counts
        except RunCancelled:
            # Cancel is stop-and-discard: nothing written to disk for the in-flight
            # run (RunCancelled fires before finalize_stage). Do NOT set error.
            terminal_state = "cancelled"
            logger.info("run %s was cancelled", record.run_id)
        except Exception as e:  # noqa: BLE001 — record any failure on the record
            terminal_state = "error"
            tb = traceback.format_exc()
            logger.exception("run %s failed", record.run_id)
            with self._lock:
                record.error = repr(e)
                record.traceback = tb
        finally:
            events.clear_sink()
            with self._lock:
                record.finished_at = datetime.now().isoformat()
            # Persist while still nominally "running", then flip the state so the
            # terminal state and the on-disk file become visible together.
            self._persist(record, out, state_override=terminal_state)
            with self._lock:
                record.state = terminal_state
                # Wake every live subscriber: push the sentinel so a stream
                # blocked on q.get() stops. Done under the lock and AFTER the
                # state flip so a subscriber that drains the queue then re-reads
                # state sees the terminal value.
                for q in record._subscribers:
                    q.put(_DONE)

    @staticmethod
    def _extract_counts(result: orchestrator.FinalizeResult) -> dict:
        """Pick a few meaningful fields from ``run_stats`` (a ``PipelineRun``)."""
        stats = result.run_stats.model_dump()
        return {k: stats[k] for k in _COUNT_FIELDS if k in stats}

    def _persist(
        self, record: RunRecord, out: Path, *, state_override: str | None = None
    ) -> None:
        """Write the record to ``out/runs/<run_id>.json`` (the run history).

        ``state_override`` lets the worker persist the terminal state ("done" /
        "error") before it flips ``record.state`` in memory — see ``_run`` for why
        the flip is deferred until after this write.
        """
        import json

        runs_dir = Path(out) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = record.to_dict()
        if state_override is not None:
            payload["state"] = state_override
        path = runs_dir / f"{record.run_id}.json"
        path.write_text(json.dumps(payload, indent=2))
