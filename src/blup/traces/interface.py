from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, TYPE_CHECKING

from blup.traces.session import TraceSession
from blup.types import TraceID

if TYPE_CHECKING:
    from blup.data_model import TraceMeta


@dataclass(frozen=True)
class TraceRecord:
    trace_id:           TraceID
    label:              str
    session:            TraceSession

    @property
    def meta(self) -> TraceMeta:
        return self.session.meta


class TraceRegistryAccess(Protocol):
    """Application interface for reading loaded traces."""

    def all_trace_ids(self) -> tuple[TraceID, ...]:
        ...
    def contains(self, trace_id: TraceID | None) -> bool:
        ...

    def get_record(self, trace_id: TraceID) -> TraceRecord:
        ...
    def all_trace_records(self) -> tuple[TraceRecord, ...]:
        ...

    def trace_options(self) -> tuple[tuple[TraceID, str], ...]:
        ...

    def get_session(self, trace_id: TraceID) -> TraceSession:
        ...
    def get_sessions(
        self,
        trace_ids: tuple[TraceID, ...],
    ) -> tuple[TraceSession, ...]:
        ...

    def valid_trace_ids(
        self,
        trace_ids: tuple[TraceID, ...],
    ) -> tuple[TraceID, ...]:
        ...

    def thread_names_for(
        self,
        trace_ids: tuple[TraceID, ...],
    ) -> tuple[str, ...]:
        ...

    def time_bounds_for(
        self,
        trace_ids: tuple[TraceID, ...],
    ) -> tuple[int, int] | None:
        ...


