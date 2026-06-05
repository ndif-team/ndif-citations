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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ndif_citations import edit_schema, events, orchestrator
from ndif_citations.events import ProgressEvent, RunCancelled
from ndif_citations.models import Bucket, PaperReason
from ndif_citations.router import ProcessingBucket

logger = logging.getLogger(__name__)

# Sentinel pushed onto every subscriber queue when a run ends, so a live SSE
# stream blocked on ``queue.Queue.get()`` knows to stop without polling state.
_DONE = object()


class RunActiveError(Exception):
    """Raised by ``JobRunner.start`` when a run is already active.

    Only one pipeline run may execute at a time (it owns the global config /
    Surya state). Callers should surface this as a 409-style conflict. A run
    parked at the review gate (state ``"awaiting_review"``) still counts as
    active.
    """


class GateError(Exception):
    """Raised by ``JobRunner.submit_gate`` when no run is awaiting review.

    The gate selection can only be submitted while the worker is parked at the
    gate (state ``"awaiting_review"``). Submitting against a run in any other
    state (running / done / error / cancelled, or an unknown id) is an error.
    Callers should surface this as a 409/404-style conflict.
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

    Gate fields (Task 3.1 — incremental human-in-the-loop review):
      * ``gate_event`` — per-run ``threading.Event`` the worker blocks on while
        ``state == "awaiting_review"``; ``submit_gate``/``cancel`` set it to
        release the worker. Threading primitive — NOT serialized.
      * ``route_result`` — the live ``RouteResult`` captured at the gate so the
        worker can rebuild its ``paper_decisions`` from the curator selection.
        Holds live model objects — NOT serialized.
      * ``paper_candidates`` / ``repo_candidates`` — serializable display lists
        of the NEW/REPROCESS paper candidates (and new repos) presented at the
        gate. These ARE serialized so the HTTP layer / persisted record can
        show what was awaiting review.
      * ``gate_selection`` — the curator's stashed ``{process_ids, discard_ids,
        edits}`` selection. Live state — NOT serialized.
    """

    run_id: str
    state: str  # "running" | "awaiting_review" | "done" | "error" | "cancelled"
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
    gate_event: threading.Event = field(default_factory=threading.Event)
    route_result: "orchestrator.RouteResult | None" = None
    paper_candidates: list[dict] = field(default_factory=list)
    repo_candidates: list[dict] = field(default_factory=list)
    gate_selection: dict | None = None

    def to_dict(self) -> dict:
        """Serialize for persistence; events go through ``ProgressEvent.to_dict``.

        ``cancel_event``, ``gate_event``, ``_subscribers``, ``route_result`` and
        ``gate_selection`` are intentionally excluded — they are threading
        primitives / live model state and must not appear in the persisted JSON.
        ``paper_candidates`` / ``repo_candidates`` ARE serialized (plain dicts)
        so a persisted/awaiting-review record can show what is at the gate.
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
            "paper_candidates": self.paper_candidates,
            "repo_candidates": self.repo_candidates,
        }


class JobRunner:
    """Drives at most one ``run_pipeline`` at a time on a daemon thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: RunRecord | None = None

    _ACTIVE_STATES = ("running", "awaiting_review")

    @property
    def active(self) -> bool:
        """True iff a run is currently ``"running"`` or parked ``"awaiting_review"``.

        A run blocked at the review gate still owns the global config / Surya
        state, so it must keep blocking a new ``start()``.
        """
        with self._lock:
            return (
                self._current is not None
                and self._current.state in self._ACTIVE_STATES
            )

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
            if (
                self._current is not None
                and self._current.state in self._ACTIVE_STATES
            ):
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

    def start_job(
        self,
        out: Path,
        job_fn: "Callable[[Callable[[], bool]], dict]",
        *,
        kind: str,
    ) -> str:
        """Start an arbitrary ``job_fn`` on a daemon thread; return its ``run_id``.

        Mirrors :meth:`start` but runs ``job_fn(cancel_check) -> dict`` on the
        worker instead of the full pipeline. ``cancel_check`` is the run's
        ``cancel_event.is_set`` callable; the returned dict (if any) is stored on
        ``record.counts``. ``kind`` is reused as ``RunRecord.mode`` (e.g.
        ``"reprocess"``) so the run shows up in history / SSE exactly like a
        normal run.

        Used for heavy off-request work that must respect the single-run gate
        (e.g. targeted reprocess: LLM + Surya), so it shares the same Surya-safe
        serialization, cancellation, persistence, and SSE fan-out as a pipeline
        run.

        Raises ``RunActiveError`` if a run is already active (the active-check and
        record install happen under the lock so two ``start_job`` / ``start``
        calls cannot both win).
        """
        with self._lock:
            if (
                self._current is not None
                and self._current.state in self._ACTIVE_STATES
            ):
                raise RunActiveError("a pipeline run is already active")

            run_id = self._new_run_id()
            record = RunRecord(
                run_id=run_id,
                state="running",
                mode=kind,
                skip_papers=False,
                skip_github=False,
                started_at=datetime.now().isoformat(),
            )
            self._current = record

        thread = threading.Thread(
            target=self._run_job,
            args=(record, out, job_fn),
            name=f"job-{kind}-{run_id}",
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

        A run parked at the review gate (``"awaiting_review"``) can also be
        cancelled: we set BOTH the ``cancel_event`` and the ``gate_event`` so the
        blocked worker wakes up, sees the cancel, and abandons the run WITHOUT
        finalizing (no on-disk write — cancel-during-gate = abandon). Partial
        merge on cancel remains out of scope (deferred to a later phase).

        Safe no-op semantics:
          * If no run has ever started, does nothing.
          * If the identified run is not active (already done / error /
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
        # Only act on an active run. Set the gate_event too so a worker blocked
        # at the gate wakes up and observes the cancel.
        if current.state in self._ACTIVE_STATES:
            current.cancel_event.set()
            current.gate_event.set()

    def submit_gate(
        self,
        run_id: str,
        *,
        process_ids: list[str],
        discard_ids: list[str],
        edits: dict[str, dict],
    ) -> None:
        """Submit the curator's gate selection and release the parked worker.

        Only valid while the identified run is in state ``"awaiting_review"``
        (raises ``GateError`` otherwise — unknown id, or any non-gate state).

        The selection is *stashed* on the record and the worker is unblocked via
        ``gate_event``; the worker thread does the actual decisions-rebuild +
        ``process_stage`` + ``finalize_stage`` (so this method never holds the
        lock across heavy work). ``process_ids`` / ``discard_ids`` are paper
        candidate ids (``DiscoveredPaper.merge_key()``); ``edits`` maps a paper
        id to a ``{field: raw_value}`` dict of pre-processing fixes.
        """
        with self._lock:
            current = self._current
            if (
                current is None
                or current.run_id != run_id
                or current.state != "awaiting_review"
            ):
                raise GateError(
                    f"run {run_id!r} is not awaiting review "
                    f"(state={getattr(current, 'state', None)!r})"
                )

        # --- Validate and pre-parse all edits BEFORE touching any state ---
        # This must happen synchronously so bad edits raise GateError to the
        # caller, keeping the run in "awaiting_review" (not consumed).
        parsed_edits: dict[str, dict[str, object]] = {}
        for paper_id, field_edits in (edits or {}).items():
            parsed_field_edits: dict[str, object] = {}
            for name, raw in (field_edits or {}).items():
                field_def = edit_schema.get_field(name)
                if field_def is None:
                    raise GateError(
                        f"edit for paper {paper_id!r}: unknown or non-editable field {name!r}"
                    )
                try:
                    raw_str = raw if isinstance(raw, str) else str(raw)
                    parsed_field_edits[name] = field_def.parse(raw_str)
                except (ValueError, TypeError) as exc:
                    raise GateError(
                        f"edit for paper {paper_id!r}: failed to parse field "
                        f"{name!r}={raw!r}: {exc}"
                    ) from exc
            parsed_edits[paper_id] = parsed_field_edits

        # All edits are valid — now stash the selection and unblock the worker.
        with self._lock:
            current = self._current
            if (
                current is None
                or current.run_id != run_id
                or current.state != "awaiting_review"
            ):
                # Race: run moved out of awaiting_review between our two lock
                # acquisitions (e.g. concurrent cancel). Raise rather than silently
                # dropping the selection.
                raise GateError(
                    f"run {run_id!r} is not awaiting review "
                    f"(state={getattr(current, 'state', None)!r})"
                )
            current.gate_selection = {
                "process_ids": list(process_ids),
                "discard_ids": list(discard_ids),
                "edits": parsed_edits,  # store pre-parsed values
            }
            gate_event = current.gate_event
        # Release the worker OUTSIDE the lock — it re-acquires the lock to read
        # the stashed selection, so we must not still be holding it.
        gate_event.set()

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

        events.set_sink(self._make_sink(record))
        # Compute the terminal outcome first, but DON'T flip the visible state
        # to a terminal value yet. The state is flipped to "done"/"error" only
        # in the `finally`, *after* the record is persisted — so an observer that
        # sees a terminal state via ``status()`` is guaranteed the run file on
        # disk already exists (no read-your-write race for callers).
        terminal_state = "done"
        try:
            if mode == "incremental":
                # Stage-driven path with the human-in-the-loop gate (Task 3.1).
                # Returns None if the run was cancelled at the gate (no finalize).
                result = self._run_incremental_with_gate(
                    record, out, skip_papers=skip_papers, skip_github=skip_github
                )
                if result is None:
                    terminal_state = "cancelled"
                    logger.info("run %s cancelled at gate", record.run_id)
                else:
                    counts = self._extract_counts(result)
                    with self._lock:
                        record.counts = counts
            else:
                # Fresh mode: unchanged end-to-end driver, NO gate.
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
            self._finalize_record(record, out, terminal_state)

    def _run_job(
        self,
        record: RunRecord,
        out: Path,
        job_fn: "Callable[[Callable[[], bool]], dict]",
    ) -> None:
        """Worker-thread body for an arbitrary ``job_fn`` (see ``start_job``).

        Owns the events sink for THIS thread (installed here, not in
        ``start_job``, because ``events`` uses ``threading.local`` storage). The
        terminal-state / persist / sentinel logic is shared with ``_run`` via
        ``_finalize_record`` so a job record behaves exactly like a pipeline run
        for ``status`` / ``subscribe`` / ``history`` / ``cancel``.
        """
        events.set_sink(self._make_sink(record))
        terminal_state = "done"
        try:
            result = job_fn(record.cancel_event.is_set)
            with self._lock:
                record.counts = result if isinstance(result, dict) else {}
        except RunCancelled:
            terminal_state = "cancelled"
            logger.info("job %s was cancelled", record.run_id)
        except Exception as e:  # noqa: BLE001 — record any failure on the record
            terminal_state = "error"
            tb = traceback.format_exc()
            logger.exception("job %s failed", record.run_id)
            with self._lock:
                record.error = repr(e)
                record.traceback = tb
        finally:
            self._finalize_record(record, out, terminal_state)

    def _make_sink(self, record: RunRecord) -> "Callable[[ProgressEvent], None]":
        """Build the events sink for *record* (shared by ``_run`` / ``_run_job``).

        Under the lock: append to the persistent buffer AND fan out to every live
        subscriber queue. Holding the lock here pairs with ``subscribe``'s
        snapshot+register critical section so an event can't be lost (in neither
        the snapshot nor a queue) or duplicated (in both).
        """

        def _append(ev: ProgressEvent) -> None:
            with self._lock:
                record.events.append(ev)
                for q in record._subscribers:
                    q.put(ev)

        return _append

    def _finalize_record(
        self, record: RunRecord, out: Path, terminal_state: str
    ) -> None:
        """Clear the sink, stamp finished_at, persist, then flip the terminal state.

        Shared by ``_run`` and ``_run_job``. The state is flipped to its terminal
        value only AFTER the record is persisted, so an observer that sees a
        terminal state via ``status()`` is guaranteed the on-disk file already
        exists (no read-your-write race). The ``_DONE`` sentinel wakes every live
        subscriber after the flip.
        """
        events.clear_sink()
        with self._lock:
            record.finished_at = datetime.now().isoformat()
        # Persist while still nominally "running", then flip the state so the
        # terminal state and the on-disk file become visible together.
        self._persist(record, out, state_override=terminal_state)
        with self._lock:
            record.state = terminal_state
            for q in record._subscribers:
                q.put(_DONE)

    # -- incremental gate driver -------------------------------------------

    # Paper candidates that are *gated* (need a curator decision before the
    # expensive LLM pass). FILL_GAPS / SKIP / PROTECTED are existing-paper
    # maintenance and flow through automatically — they are NOT gated.
    _GATED_BUCKETS = (ProcessingBucket.NEW, ProcessingBucket.REPROCESS)

    def _run_incremental_with_gate(
        self,
        record: RunRecord,
        out: Path,
        *,
        skip_papers: bool,
        skip_github: bool,
    ) -> orchestrator.FinalizeResult | None:
        """Drive discover -> enrich -> route, PAUSE at the gate, then process.

        Returns the ``FinalizeResult`` on success, or ``None`` if the run was
        cancelled at the gate (in which case nothing is finalized / written).

        Steps:
          1. Run the three cheap stages individually.
          2. Partition ``route_result.paper_decisions`` into gated *candidates*
             (NEW/REPROCESS) and *auto-flow* decisions (FILL_GAPS/SKIP/PROTECTED).
          3. Build serializable candidate lists, stash them + the live
             RouteResult on the record, emit ``awaiting_review``, flip state, and
             BLOCK on ``gate_event`` (lock released while blocked).
          4. On unblock: if cancelled → return None. Otherwise read the stashed
             selection, rebuild ``paper_decisions`` (process / discard / drop),
             then run process + finalize.
        """
        fresh = False  # incremental mode always merges against existing state

        d = orchestrator.discover_stage(
            out, skip_papers=skip_papers, skip_github=skip_github, fresh=fresh
        )
        e = orchestrator.enrich_stage(
            out, d, skip_papers=skip_papers, skip_github=skip_github, fresh=fresh
        )
        route_result = orchestrator.route_stage(
            out, e, skip_papers=skip_papers, skip_github=skip_github, fresh=fresh
        )

        # Partition paper decisions into gated candidates vs auto-flow.
        candidates = [
            dec for dec in route_result.paper_decisions
            if dec.bucket in self._GATED_BUCKETS
        ]
        auto_flow = [
            dec for dec in route_result.paper_decisions
            if dec.bucket not in self._GATED_BUCKETS
        ]

        paper_candidates = [self._paper_candidate_dict(dec) for dec in candidates]
        # New repos are surfaced as light candidates for visibility only — repos
        # all auto-flow (cheap, no LLM) and are not actually gated.
        repo_candidates = [
            self._repo_candidate_dict(dec)
            for dec in route_result.repo_decisions
            if dec.bucket == ProcessingBucket.NEW
        ]

        # Publish gate state + candidates, then block the worker.
        with self._lock:
            record.route_result = route_result
            record.paper_candidates = paper_candidates
            record.repo_candidates = repo_candidates
            record.state = "awaiting_review"
        events.emit(
            "awaiting_review",
            paper_candidates=paper_candidates,
            repo_candidates=repo_candidates,
        )

        # BLOCK (not under the lock) until submit_gate or cancel sets the event.
        record.gate_event.wait()

        # Woke up. If cancelled, abandon WITHOUT finalizing (no on-disk write).
        if record.cancel_event.is_set():
            return None

        with self._lock:
            selection = record.gate_selection or {
                "process_ids": [],
                "discard_ids": [],
                "edits": {},
            }
        process_ids = set(selection.get("process_ids", []))
        discard_ids = set(selection.get("discard_ids", []))
        edits = selection.get("edits", {})

        # Rebuild the kept candidate decisions from the curator selection.
        kept_candidates = self._apply_gate_selection(
            candidates, process_ids=process_ids, discard_ids=discard_ids, edits=edits
        )

        # Auto-flow decisions are always kept; gated candidates only if selected.
        route_result.paper_decisions = auto_flow + kept_candidates

        completed = orchestrator.process_stage(
            out,
            route_result,
            skip_papers=skip_papers,
            skip_github=skip_github,
            cancel_check=record.cancel_event.is_set,
        )
        return orchestrator.finalize_stage(
            out,
            route_result,
            d.run_stats,
            skip_papers=skip_papers,
            skip_github=skip_github,
            fresh=fresh,
            completed=completed,
        )

    @staticmethod
    def _paper_candidate_dict(dec) -> dict:
        """Serializable display record for a single gated paper candidate."""
        paper = dec.paper
        abstract = paper.abstract or ""
        return {
            "id": paper.merge_key(),
            "title": paper.title,
            "authors": paper.authors,
            "venue": paper.venue,
            "year": paper.year,
            "abstract": abstract[:300],
            "processing_bucket": dec.bucket.value,
            "source": paper.source.value,
        }

    @staticmethod
    def _repo_candidate_dict(dec) -> dict:
        """Serializable display record for a single new repo (visibility only)."""
        repo = dec.repo
        return {
            "id": repo.merge_key(),
            "owner": repo.owner,
            "repo": repo.repo,
            "stars": repo.stars,
            "repo_type": repo.repo_type,
        }

    def _apply_gate_selection(
        self,
        candidates: list,
        *,
        process_ids: set[str],
        discard_ids: set[str],
        edits: dict[str, dict],
    ) -> list:
        """Turn gated candidate decisions into the kept decisions list.

        For each candidate (keyed by ``decision.paper.merge_key()``):
          * id in ``discard_ids`` → mark the paper DISCARDED (manual_override,
            MANUAL_DISCARD reason) and zero out ``processing_needed`` so
            ``process_stage`` won't LLM it, but KEEP it so finalize merges it as
            discarded.
          * id in ``process_ids`` → apply any ``edits[id]`` (pre-processing
            fixes; NO manual_override) and keep it for processing.
          * id in NEITHER → DROP it (not processed, not merged — it reappears on
            the next discovery).

        ``discard_ids`` wins if an id appears in both sets.
        """
        kept: list = []
        for dec in candidates:
            cid = dec.paper.merge_key()
            if cid in discard_ids:
                dec.paper.bucket = Bucket.DISCARDED
                dec.paper.reason = PaperReason.MANUAL_DISCARD
                dec.paper.manual_override = True
                dec.processing_needed = {
                    k: False for k in dec.processing_needed
                }
                kept.append(dec)
            elif cid in process_ids:
                self._apply_edits(dec.paper, edits.get(cid, {}))
                kept.append(dec)
            # else: dropped — neither processed nor merged.
        return kept

    @staticmethod
    def _apply_edits(paper, field_edits: dict) -> None:
        """Apply gate edits to ``paper``.

        Gate edits are pre-processing fixes — they do NOT set ``manual_override``
        (unlike a curator ``edit`` command).

        ``field_edits`` maps field name → **already-parsed** value (pre-validated
        and pre-parsed by ``submit_gate`` before the worker was released).  We
        therefore just look up the field (to get the canonical attribute name) and
        ``setattr`` directly — no re-parsing needed.

        Any unknown field name here is a programming error (submit_gate guarantees
        it cannot happen via normal usage), so we raise ``AssertionError`` rather
        than silently dropping the edit.
        """
        for name, value in (field_edits or {}).items():
            field_def = edit_schema.get_field(name)
            if field_def is None:
                # This should never happen: submit_gate validated every field
                # before stashing the selection.  If it does, it is a bug.
                raise AssertionError(
                    f"gate edit: unexpected non-editable field {name!r} reached worker "
                    "(should have been rejected by submit_gate)"
                )
            setattr(paper, field_def.name, value)

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
