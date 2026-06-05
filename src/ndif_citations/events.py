from __future__ import annotations
import threading, time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

EventType = str  # "stage_start"|"stage_done"|"source_count"|"dedup"|"route_summary"
                 # |"item_start"|"item_step"|"rate_limit_wait"|"awaiting_review"
                 # |"merge_result"|"report"|"error"|"cancelled"|"done"|"log"

@dataclass
class ProgressEvent:
    type: EventType
    stage: Optional[str] = None      # "discover"|"enrich"|"route"|"process"|"finalize"
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    def to_dict(self) -> dict: return asdict(self)

_local = threading.local()
def set_sink(fn: Callable[[ProgressEvent], None]) -> None: _local.fn = fn
def clear_sink() -> None: _local.fn = None
def emit(type: EventType, stage: str | None = None, **data: Any) -> None:
    fn = getattr(_local, "fn", None)
    if fn is not None:
        fn(ProgressEvent(type=type, stage=stage, data=data))
