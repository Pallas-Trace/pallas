from __future__ import annotations

import re
from collections.abc import Iterable

from blup.traces.interface import (
    TraceRecord,
    TraceRegistryAccess,
)
from blup.traces.session import TraceSession
from blup.types import TraceID


_THREAD_RE = re.compile(r"^PT(\d+)(?:\.(\d+))?$")

def _thread_sort_key(name: str) -> tuple[object, ...]:
    match = _THREAD_RE.match(name)
    if match is not None:
        return (
            0,
            int(match.group(1)),
            int(match.group(2) or 0),
            name,
        )

    numbers = tuple(int(value) for value in re.findall(r"\d+", name))
    if numbers:
        return (1, numbers, name)

    return (2, name)


class TraceRegistry(TraceRegistryAccess):
    """Common registry of all currently loaded trace sessions."""

    def __init__(self, records: Iterable[TraceRecord]) -> None:
        records_unrolled = tuple(records)

        self._records_by_id = {
            record.trace_id: record
            for record in records_unrolled
        }
        self._trace_order = tuple(
            record.trace_id
            for record in records_unrolled
        )

        if len(self._records_by_id) != len(records_unrolled):
            raise ValueError("Trace IDs must be unique")

    def all_trace_ids(self) -> tuple[TraceID, ...]:
        return self._trace_order

    def contains(self, trace_id: TraceID | None) -> bool:
        return trace_id is not None and trace_id in self._records_by_id

    def get_record(self, trace_id: TraceID) -> TraceRecord:
        try:
            return self._records_by_id[trace_id]
        except KeyError as exc:
            raise KeyError(f"Unknown trace ID: {trace_id!r}") from exc

    def all_trace_records(self) -> tuple[TraceRecord, ...]:
        return tuple(
            self._records_by_id[trace_id]
            for trace_id in self._trace_order
        )

    def trace_options(self) -> tuple[tuple[TraceID, str], ...]:
        return tuple(
            (trace.trace_id, trace.label)
            for trace in self.all_trace_records()
        )

    def get_session(self, trace_id: TraceID) -> TraceSession:
        return self.get_record(trace_id).session

    def get_sessions(
        self,
        trace_ids: tuple[TraceID, ...],
    ) -> tuple[TraceSession, ...]:
        return tuple(
            self.get_session(trace_id)
            for trace_id in self.valid_trace_ids(trace_ids)
        )

    def valid_trace_ids(
        self,
        trace_ids: tuple[TraceID, ...],
    ) -> tuple[TraceID, ...]:
        seen: set[TraceID] = set()
        valid: list[TraceID] = []

        for trace_id in trace_ids:
            if trace_id in seen or not self.contains(trace_id):
                continue
            seen.add(trace_id)
            valid.append(trace_id)

        return tuple(valid)

    def thread_names_for(
        self,
        trace_ids: tuple[TraceID, ...],
    ) -> tuple[str, ...]:
        names: set[str] = set()

        for trace_id in self.valid_trace_ids(trace_ids):
            names.update(
                str(name)
                for name in self.get_record(trace_id).meta.thread_names
            )

        return tuple(sorted(names, key=_thread_sort_key))

    def time_bounds_for(
        self,
        trace_ids: tuple[TraceID, ...],
    ) -> tuple[int, int] | None:
        selected_ids = self.valid_trace_ids(trace_ids)
        if not selected_ids:
            return None

        trace_metas = tuple(
            self.get_record(trace_id).meta
            for trace_id in selected_ids
        )

        return (
            min(meta.start_ns for meta in trace_metas),
            max(meta.end_ns for meta in trace_metas)
        )


